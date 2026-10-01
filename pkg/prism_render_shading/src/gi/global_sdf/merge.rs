//! Union of transformed object SDFs into a single global distance field — CPU
//! golden.
//!
//! Lumen's global distance field is the merge of every placed object's local
//! distance field.  Each object has a rigid placement (rotation + translation)
//! and, in practice, a uniform scale; the merged field at a world point `p` is
//! the signed distance to the *nearest* object surface.  For a set of objects
//! `{i}` with world-to-local transforms `T_i^{-1}` this is the constructive
//! solid-geometry **union**:
//!
//! ```text
//! d(p) = min_i scale_i * sdf_i( T_i^{-1} p )
//! ```
//!
//! where `sdf_i` is the object's local analytic field and the `scale_i` factor
//! converts the local distance back to world units.  Taking the minimum of two
//! signed distance fields is exactly the union of the two solids, so merging
//! many primitives this way yields the distance to their combined surface.
//! This module is the backend-neutral reference for that merge; the result is
//! what [`crate::gi::global_sdf::brick_grid::BrickGrid::from_distance_fn`] bakes
//! into a sparse brick grid.
//!
//! # Conventions
//! * **Primitives.** [`SdfPrimitive`] carries the handful of closed-form shapes
//!   whose distance has a cheap exact form (sphere, box, rounded box).  Their
//!   extents are clamped non-negative so a degenerate primitive collapses
//!   rather than inverting its sign.
//! * **Placement.** [`SdfObject`] stores a `rotation` quaternion, a
//!   `translation`, and a positive uniform `scale`.  World-to-local is
//!   `local = rotation^{-1} * (p - translation) / scale`; the local distance is
//!   multiplied by `scale` to return to world units.  A non-finite or
//!   near-zero quaternion falls back to the identity rotation and a
//!   non-positive scale is clamped to a tiny positive value, so the transform
//!   is always invertible and `NaN`-free.
//! * **Union.** [`merge_distance`] returns the minimum object distance; an
//!   empty object list reports [`FAR_DISTANCE`](super::brick_grid::FAR_DISTANCE)
//!   (empty space everywhere).
//! * **Determinism / safety.** Pure, deterministic functions — no RNG, I/O,
//!   GPU, or `unsafe`.  Every divisor is guarded and no path yields `NaN`.

use alloc::vec::Vec;
use bevy_math::{Quat, Vec3};

use super::brick_grid::{BrickGrid, FAR_DISTANCE};

/// Exact signed distance from `point` to a sphere surface.
///
/// `|point - center| - radius`, negative inside; `radius` is clamped
/// non-negative.  This is the analytic ground truth the baked field is tested
/// against.
#[inline]
pub fn sphere_sdf(point: Vec3, center: Vec3, radius: f32) -> f32 {
    (point - center).length() - radius.max(0.0)
}

/// Exact signed distance from `point` to an axis-aligned box surface.
///
/// Inigo Quilez's closed form: with `q = |point - center| - half_extents` the
/// distance is `|max(q, 0)| + min(max(q.x, q.y, q.z), 0)`, correct both outside
/// (positive) and inside (negative).  Negative half extents are clamped to
/// zero.
#[inline]
pub fn box_sdf(point: Vec3, center: Vec3, half_extents: Vec3) -> f32 {
    let he = half_extents.max(Vec3::ZERO);
    let q = (point - center).abs() - he;
    let outside = q.max(Vec3::ZERO).length();
    let inside = q.x.max(q.y).max(q.z).min(0.0);
    outside + inside
}

/// A closed-form SDF primitive evaluated in its own local frame (centred at the
/// local origin).
///
/// All extents are interpreted at face value here; [`SdfObject`] applies the
/// clamps described in the [module docs](self) when it evaluates them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SdfPrimitive {
    /// A sphere of the given radius centred at the local origin.
    Sphere {
        /// Sphere radius; clamped non-negative on evaluation.
        radius: f32,
    },
    /// An axis-aligned box of the given half extents centred at the local
    /// origin.
    Box {
        /// Per-axis half extents; clamped non-negative on evaluation.
        half_extents: Vec3,
    },
    /// A box of half extents `half_extents` whose edges are rounded by
    /// `radius`, i.e. the box field minus a constant.
    RoundBox {
        /// Per-axis half extents of the inner box; clamped non-negative.
        half_extents: Vec3,
        /// Rounding radius subtracted from the box distance; clamped
        /// non-negative.
        radius: f32,
    },
}

impl SdfPrimitive {
    /// Signed distance from a *local-space* point to this primitive.
    ///
    /// Extents and radii are clamped non-negative so a degenerate primitive
    /// collapses to a lower-dimensional shape (a point, line, or plane) rather
    /// than inverting its inside/outside sign.
    #[inline]
    pub fn distance_local(&self, local: Vec3) -> f32 {
        match *self {
            SdfPrimitive::Sphere { radius } => sphere_sdf(local, Vec3::ZERO, radius),
            SdfPrimitive::Box { half_extents } => box_sdf(local, Vec3::ZERO, half_extents),
            SdfPrimitive::RoundBox {
                half_extents,
                radius,
            } => {
                let r = radius.max(0.0);
                let he = (half_extents - Vec3::splat(r)).max(Vec3::ZERO);
                box_sdf(local, Vec3::ZERO, he) - r
            }
        }
    }
}

/// A primitive placed in the world by a rigid transform plus uniform scale.
///
/// See the [module docs](self) for the exact world-to-local convention and the
/// defensive clamps on the rotation and scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfObject {
    /// Local-to-world rotation.  Normalised on use, with an identity fallback
    /// for a degenerate (non-finite or near-zero) quaternion.
    pub rotation: Quat,
    /// Local-to-world translation (world position of the local origin).
    pub translation: Vec3,
    /// Uniform scale; clamped to a tiny positive value on use.
    pub scale: f32,
    /// The local primitive this object instances.
    pub primitive: SdfPrimitive,
}

impl SdfObject {
    /// Smallest permitted scale; clamps away a zero/negative scale so the
    /// world-to-local divide stays finite and sign-preserving.
    const MIN_SCALE: f32 = 1.0e-6;

    /// Builds an unrotated, unit-scale object at `translation`.
    #[inline]
    pub fn new(primitive: SdfPrimitive, translation: Vec3) -> Self {
        Self {
            rotation: Quat::IDENTITY,
            translation,
            scale: 1.0,
            primitive,
        }
    }

    /// Returns a copy with the rotation replaced.
    #[inline]
    pub fn with_rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    /// Returns a copy with the uniform scale replaced.
    #[inline]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Sanitised rotation: the normalised quaternion, or identity if the input
    /// is non-finite or too short to normalise.
    #[inline]
    fn safe_rotation(&self) -> Quat {
        if self.rotation.is_finite()
            && self.rotation.length_squared() > f32::MIN_POSITIVE
        {
            self.rotation.normalize()
        } else {
            Quat::IDENTITY
        }
    }

    /// Sanitised scale: the input clamped to a tiny positive floor.
    #[inline]
    fn safe_scale(&self) -> f32 {
        self.scale.max(Self::MIN_SCALE)
    }

    /// Signed distance from a *world* point to this object's surface.
    ///
    /// Applies the inverse transform `local = rotation^{-1} * (p - t) / scale`,
    /// evaluates the local primitive, and multiplies by `scale` to return the
    /// distance in world units.  All operands are sanitised (see the module
    /// docs), so the result is always finite.
    #[inline]
    pub fn distance(&self, world: Vec3) -> f32 {
        let scale = self.safe_scale();
        let rot = self.safe_rotation();
        let local = rot.inverse() * ((world - self.translation) / scale);
        self.primitive.distance_local(local) * scale
    }
}

/// Signed distance to the union of a set of objects: the per-point minimum.
///
/// Evaluates every object's [`distance`](SdfObject::distance) at `world` and
/// returns the smallest, which is the signed distance to the nearest surface
/// (the CSG union).  An empty slice reports
/// [`FAR_DISTANCE`](super::brick_grid::FAR_DISTANCE) — empty space everywhere.
#[inline]
pub fn merge_distance(objects: &[SdfObject], world: Vec3) -> f32 {
    let mut d = FAR_DISTANCE;
    for obj in objects {
        let di = obj.distance(world);
        if di < d {
            d = di;
        }
    }
    d
}

/// Bakes the merged union of `objects` into a sparse [`BrickGrid`].
///
/// The grid placement (`origin`, `voxel_size`, `brick_dim`) and the inclusive
/// brick range `[brick_min, brick_max]` are forwarded to
/// [`BrickGrid::from_distance_fn`], whose distance callback is
/// [`merge_distance`] over `objects`.  A positive `cull_band` drops bricks that
/// lie entirely farther than that band from every surface and records the band
/// as the grid's conservative empty-space distance; a non-positive band keeps
/// every brick.
pub fn bake_merged(
    objects: &[SdfObject],
    origin: Vec3,
    voxel_size: f32,
    brick_dim: u32,
    brick_min: bevy_math::IVec3,
    brick_max: bevy_math::IVec3,
    cull_band: f32,
) -> BrickGrid {
    // Copy into an owned buffer so the baker closure can be `move`-free of the
    // borrowed slice's lifetime while staying allocation-light.
    let objects: Vec<SdfObject> = objects.to_vec();
    BrickGrid::from_distance_fn(
        origin,
        voxel_size,
        brick_dim,
        brick_min,
        brick_max,
        cull_band,
        |p| merge_distance(&objects, p),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::IVec3;
    use core::f32::consts::FRAC_PI_2;

    #[test]
    fn identity_object_matches_analytic_sphere() {
        let obj = SdfObject::new(SdfPrimitive::Sphere { radius: 1.0 }, Vec3::ZERO);
        for p in [
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.3, -0.4, 0.2),
        ] {
            let got = obj.distance(p);
            let truth = sphere_sdf(p, Vec3::ZERO, 1.0);
            assert!((got - truth).abs() < 1.0e-5, "{p:?}: {got} vs {truth}");
        }
    }

    #[test]
    fn translation_shifts_field() {
        let center = Vec3::new(3.0, -1.0, 2.0);
        let obj = SdfObject::new(SdfPrimitive::Sphere { radius: 0.5 }, center);
        for p in [center, center + Vec3::X, Vec3::ZERO] {
            let got = obj.distance(p);
            let truth = sphere_sdf(p, center, 0.5);
            assert!((got - truth).abs() < 1.0e-5, "{p:?}: {got} vs {truth}");
        }
    }

    #[test]
    fn rotation_leaves_sphere_invariant() {
        // A sphere is rotation-symmetric: any rotation must leave its field
        // unchanged.
        let plain = SdfObject::new(SdfPrimitive::Sphere { radius: 1.0 }, Vec3::ZERO);
        let rotated = plain.with_rotation(Quat::from_rotation_z(FRAC_PI_2));
        for p in [Vec3::new(1.5, 0.3, -0.2), Vec3::new(-0.4, 0.9, 0.1)] {
            assert!((plain.distance(p) - rotated.distance(p)).abs() < 1.0e-5);
        }
    }

    #[test]
    fn rotated_box_distance_is_correct() {
        // A box rotated 90 deg about Z, queried along a world axis, equals the
        // unrotated box distance along the swapped local axis.
        let obj = SdfObject::new(
            SdfPrimitive::Box {
                half_extents: Vec3::new(2.0, 0.5, 0.5),
            },
            Vec3::ZERO,
        )
        .with_rotation(Quat::from_rotation_z(FRAC_PI_2));
        // World +Y maps to local +X (long axis), so the surface sits ~2 away.
        let d = obj.distance(Vec3::new(0.0, 2.0, 0.0));
        assert!(d.abs() < 1.0e-4, "expected on-surface, got {d}");
    }

    #[test]
    fn scale_grows_the_shape() {
        // A unit sphere scaled by 2 behaves like a radius-2 sphere.
        let obj = SdfObject::new(SdfPrimitive::Sphere { radius: 1.0 }, Vec3::ZERO)
            .with_scale(2.0);
        let p = Vec3::new(3.0, 0.0, 0.0);
        let got = obj.distance(p);
        let truth = sphere_sdf(p, Vec3::ZERO, 2.0);
        assert!((got - truth).abs() < 1.0e-4, "{got} vs {truth}");
    }

    #[test]
    fn union_takes_the_minimum() {
        let a = SdfObject::new(SdfPrimitive::Sphere { radius: 1.0 }, Vec3::new(-2.0, 0.0, 0.0));
        let b = SdfObject::new(SdfPrimitive::Sphere { radius: 1.0 }, Vec3::new(2.0, 0.0, 0.0));
        let objs = [a, b];
        // A point near sphere b should report b's (smaller) distance.
        let p = Vec3::new(2.5, 0.0, 0.0);
        let merged = merge_distance(&objs, p);
        assert!((merged - b.distance(p)).abs() < 1.0e-6);
        assert!(merged < a.distance(p));
        // Midpoint is equidistant and positive (outside both).
        let mid = merge_distance(&objs, Vec3::ZERO);
        assert!(mid > 0.0);
        assert!((a.distance(Vec3::ZERO) - mid).abs() < 1.0e-4);
    }

    #[test]
    fn empty_union_is_far() {
        assert_eq!(merge_distance(&[], Vec3::ZERO), FAR_DISTANCE);
    }

    #[test]
    fn degenerate_scale_and_rotation_stay_finite() {
        let obj = SdfObject {
            rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0), // zero quat -> identity
            translation: Vec3::ZERO,
            scale: 0.0, // -> tiny positive
            primitive: SdfPrimitive::Sphere { radius: 1.0 },
        };
        let d = obj.distance(Vec3::new(1.0, 2.0, 3.0));
        assert!(d.is_finite());
    }

    #[test]
    fn bake_merged_matches_merge_distance() {
        let objs = [
            SdfObject::new(SdfPrimitive::Sphere { radius: 0.8 }, Vec3::new(-0.6, 0.0, 0.0)),
            SdfObject::new(SdfPrimitive::Sphere { radius: 0.8 }, Vec3::new(0.6, 0.0, 0.0)),
        ];
        let grid = bake_merged(
            &objs,
            Vec3::ZERO,
            0.1,
            8,
            IVec3::splat(-3),
            IVec3::splat(2),
            0.0,
        );
        for p in [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.2, 0.0)] {
            let baked = grid.sample_distance(p);
            let truth = merge_distance(&objs, p);
            assert!((baked - truth).abs() < 0.03, "{p:?}: {baked} vs {truth}");
        }
    }

    #[test]
    fn merge_is_deterministic() {
        let objs = [
            SdfObject::new(SdfPrimitive::Box { half_extents: Vec3::splat(0.5) }, Vec3::ZERO),
        ];
        let p = Vec3::new(0.7, 0.2, -0.3);
        assert_eq!(merge_distance(&objs, p), merge_distance(&objs, p));
    }
}
