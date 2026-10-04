//! §24.8 Swept transform — CCD broad-phase bounds and motion data.
//!
//! A fast-moving object can tunnel through a thin obstacle inside a single
//! fixed step: at the start of the step it is in front of the wall, at the end
//! it is behind it, and no discrete snapshot ever overlaps. Continuous
//! collision detection (CCD) fixes this by reasoning about the whole
//! `prev → curr` motion, not just the endpoints. This module provides the
//! **swept bounds** data model that a CCD broad-phase needs:
//!
//! - [`SweptMotion`] stores a proxy's previous and current world pose plus its
//!   object-space [`Aabb3`], reusing the prev/curr snapshots the render
//!   interpolation (§10) already keeps — no extra per-entity storage.
//! - [`SweptMotion::bounds`] returns a **conservative** world [`Aabb3`] that
//!   provably contains the proxy at every instant of the interpolated motion,
//!   suitable as a temporal broad-phase box.
//! - [`SweptMotion::bounds_sub`] bounds an arbitrary sub-interval `[t0, t1]`,
//!   and [`SweptMotion::subdivide`] walks a uniform partition of `[0, 1]`,
//!   giving a *conservative advancement* algorithm the time-subdivided boxes it
//!   needs to bracket a time of impact (TOI).
//!
//! ## Why the bound is conservative
//! The interpolated pose blends translation / scale with `lerp` and rotation
//! with shortest-path `slerp` (identical to [`GlobalTransform::interpolate`]).
//! The construction mirrors the classic temporal-AABB bound used in production
//! physics engines:
//!
//! 1. Take the union of the exact world boxes at the two endpoints. Translation
//!    moves the frame origin along a straight segment (covered by the union),
//!    and under scale `lerp` every corner stays on the segment between its two
//!    endpoint positions, whose norm never exceeds the endpoints' — so the
//!    linear/scale motion is already inside the union.
//! 2. Rotation makes each corner trace an arc of radius `r` (its distance from
//!    the frame origin) through the sweep angle `θ`. The arc bulges past its
//!    chord by at most the arc length `θ · r`, so growing the union outward by
//!    `θ · r` on every axis conservatively captures the rotational excursion.
//!
//! Smaller sub-intervals shrink both the endpoint span and `θ`, so
//! [`SweptMotion::subdivide`] produces progressively tighter boxes — the basis
//! for conservative advancement.
//!
//! ## Teleports
//! A discontinuous reposition (teleport) must **not** be treated as motion, or
//! the swept box would smear across empty space and the velocity used for
//! motion blur / TAA would be bogus. [`SweptMotion::teleported`] builds a
//! motion whose bounds collapse to the current box and whose reported velocity
//! is zero.
//!
//! This module is `no_std` + `alloc`, pure math over [`prism_math`] types, with
//! no threads and no clock. The authoritative simulation state is never
//! mutated; a [`SweptMotion`] only *reads* prev/curr poses. Design-doc §24.9
//! records the honest boundary: the narrow-phase sweep test and the actual
//! motion-vector pass live in `prism_physics` / `prism_render_scene`.

use alloc::vec::Vec;

use prism_math::{Aabb3, Mat3, Quat, Vec3};

use crate::spatial_sync::transform_aabb;
use crate::GlobalTransform;

/// The shortest-path rotation angle (radians, in `[0, π]`) between two
/// orientations extracted from the world poses.
#[inline]
#[must_use]
fn sweep_angle(a: Quat, b: Quat) -> f32 {
    // |dot| folds a quaternion and its negation together (same orientation),
    // giving the shortest arc; clamp guards `acos`'s domain against float drift.
    let d = a.dot(b);
    let d = libm::fabsf(d);
    let d = if d > 1.0 { 1.0 } else { d };
    2.0 * libm::acosf(d)
}

/// The rotation part of a world pose (shear-free decomposition).
#[inline]
#[must_use]
fn rotation_of(pose: &GlobalTransform) -> Quat {
    let (_, rotation, _) = pose.affine().to_scale_rotation_translation();
    rotation
}

/// The bounding radius of `local` under the linear map `linear`: the greatest
/// distance from the frame origin to any transformed corner. Rotation preserves
/// this radius, so it bounds how far a corner can swing while the frame rotates.
#[inline]
#[must_use]
fn linear_radius(linear: Mat3, local: Aabb3) -> f32 {
    let mut r2 = 0.0_f32;
    for corner in local.corners() {
        let v = linear.mul_vec3(corner);
        r2 = r2.max(v.length_squared());
    }
    libm::sqrtf(r2)
}

/// A proxy's `prev → curr` world motion over one step, plus the object-space
/// box being swept.
///
/// Construct with [`SweptMotion::new`] for ordinary motion or
/// [`SweptMotion::teleported`] when the pose jumped discontinuously. See the
/// [module docs](crate::sweep) for the conservative-bounds guarantee.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SweptMotion {
    prev: GlobalTransform,
    curr: GlobalTransform,
    local: Aabb3,
    teleport: bool,
}

impl SweptMotion {
    /// A swept motion from `prev` to `curr` for a proxy whose object-space
    /// bounds are `local`.
    #[inline]
    #[must_use]
    pub fn new(prev: GlobalTransform, curr: GlobalTransform, local: Aabb3) -> Self {
        Self {
            prev,
            curr,
            local,
            teleport: false,
        }
    }

    /// A non-moving proxy sitting at `pose` (prev == curr). Its swept bounds are
    /// just its static world box.
    #[inline]
    #[must_use]
    pub fn still(pose: GlobalTransform, local: Aabb3) -> Self {
        Self {
            prev: pose,
            curr: pose,
            local,
            teleport: false,
        }
    }

    /// A proxy that was teleported to `curr`. The previous pose is irrelevant:
    /// bounds collapse to the current box and the reported velocity is zero, so
    /// no false sweep volume or motion-blur streak is produced.
    #[inline]
    #[must_use]
    pub fn teleported(curr: GlobalTransform, local: Aabb3) -> Self {
        Self {
            prev: curr,
            curr,
            local,
            teleport: true,
        }
    }

    /// The previous world pose.
    #[inline]
    #[must_use]
    pub fn prev(&self) -> GlobalTransform {
        self.prev
    }

    /// The current world pose.
    #[inline]
    #[must_use]
    pub fn curr(&self) -> GlobalTransform {
        self.curr
    }

    /// The object-space bounds being swept.
    #[inline]
    #[must_use]
    pub fn local(&self) -> Aabb3 {
        self.local
    }

    /// Whether this motion was marked as a teleport.
    #[inline]
    #[must_use]
    pub fn is_teleport(&self) -> bool {
        self.teleport
    }

    /// The world-space translation delta `curr - prev` (zero for a teleport).
    ///
    /// This is the per-step linear motion a motion-vector / TAA pass divides by
    /// the step duration to get velocity.
    #[inline]
    #[must_use]
    pub fn translation_delta(&self) -> Vec3 {
        if self.teleport {
            Vec3::ZERO
        } else {
            self.curr.translation() - self.prev.translation()
        }
    }

    /// The shortest-path rotation angle (radians) swept from `prev` to `curr`
    /// (zero for a teleport).
    #[inline]
    #[must_use]
    pub fn angular_motion(&self) -> f32 {
        if self.teleport {
            0.0
        } else {
            sweep_angle(rotation_of(&self.prev), rotation_of(&self.curr))
        }
    }

    /// The interpolated world pose at `t ∈ [0, 1]` (`0` → prev, `1` → curr),
    /// using the same SRT blend as [`GlobalTransform::interpolate`]. A teleport
    /// reports `curr` for every `t`.
    #[inline]
    #[must_use]
    pub fn pose_at(&self, t: f32) -> GlobalTransform {
        if self.teleport {
            self.curr
        } else {
            self.prev.interpolate(&self.curr, t)
        }
    }

    /// The exact tight world [`Aabb3`] of the proxy at `t ∈ [0, 1]`.
    #[inline]
    #[must_use]
    pub fn box_at(&self, t: f32) -> Aabb3 {
        let pose = self.pose_at(t);
        let affine = pose.affine();
        transform_aabb(affine.matrix3, affine.translation, self.local)
    }

    /// A conservative world box over the sub-interval `[t0, t1]` of the motion
    /// (`t0`, `t1` clamped to `[0, 1]`, order-insensitive).
    ///
    /// The returned box provably contains the proxy at **every** instant in the
    /// sub-interval, not just the endpoints. A teleport returns the current box.
    #[must_use]
    pub fn bounds_sub(&self, t0: f32, t1: f32) -> Aabb3 {
        if self.teleport {
            return self.box_at(1.0);
        }
        let lo = clamp01(t0.min(t1));
        let hi = clamp01(t0.max(t1));

        let pose_lo = self.pose_at(lo);
        let pose_hi = self.pose_at(hi);
        let affine_lo = pose_lo.affine();
        let affine_hi = pose_hi.affine();

        let box_lo = transform_aabb(affine_lo.matrix3, affine_lo.translation, self.local);
        let box_hi = transform_aabb(affine_hi.matrix3, affine_hi.translation, self.local);
        let union = box_lo.merge(box_hi);

        // Rotational excursion bound: arc length θ·r over the sub-interval. The
        // radius uses the larger endpoint linear map; scale `lerp` keeps every
        // intermediate corner within that radius (norm is convex on the
        // segment between the two endpoint images).
        let angle = sweep_angle(rotation_of(&pose_lo), rotation_of(&pose_hi));
        let r = linear_radius(affine_lo.matrix3, self.local)
            .max(linear_radius(affine_hi.matrix3, self.local));
        let expansion = angle * r;

        if expansion > 0.0 {
            union.expand(expansion)
        } else {
            union
        }
    }

    /// The conservative swept world box over the whole motion `[0, 1]`.
    ///
    /// Equivalent to `self.bounds_sub(0.0, 1.0)`; this is the temporal AABB a
    /// CCD broad-phase inserts for the moving proxy.
    #[inline]
    #[must_use]
    pub fn bounds(&self) -> Aabb3 {
        self.bounds_sub(0.0, 1.0)
    }

    /// Partition `[0, 1]` into `segments` equal sub-intervals and return the
    /// conservative box of each, in time order.
    ///
    /// Each box bounds its own slice of the motion, so a conservative
    /// advancement algorithm can scan the slices, discard those whose box
    /// misses the obstacle, and recurse into the first that overlaps to
    /// bracket the TOI. `segments == 0` is treated as `1`.
    #[must_use]
    pub fn subdivide(&self, segments: usize) -> Vec<SweepSegment> {
        let n = segments.max(1);
        let mut out = Vec::with_capacity(n);
        let inv = 1.0 / n as f32;
        for i in 0..n {
            let t0 = i as f32 * inv;
            let t1 = (i + 1) as f32 * inv;
            out.push(SweepSegment {
                t0,
                t1,
                bounds: self.bounds_sub(t0, t1),
            });
        }
        out
    }
}

/// One slice of a subdivided sweep: a sub-interval of normalized time and the
/// conservative world box bounding the motion over it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SweepSegment {
    /// Start of the sub-interval (normalized time in `[0, 1]`).
    pub t0: f32,
    /// End of the sub-interval (normalized time in `[0, 1]`).
    pub t1: f32,
    /// Conservative world box over `[t0, t1]`.
    pub bounds: Aabb3,
}

impl SweepSegment {
    /// The sub-interval midpoint, a convenient sample time for the slice.
    #[inline]
    #[must_use]
    pub fn midpoint(&self) -> f32 {
        0.5 * (self.t0 + self.t1)
    }

    /// The normalized duration `t1 - t0` of the slice.
    #[inline]
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.t1 - self.t0
    }
}

#[inline]
fn clamp01(t: f32) -> f32 {
    let t = if t > 1.0 { 1.0 } else { t };
    if t > 0.0 {
        t
    } else {
        0.0
    }
}
