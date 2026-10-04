//! §24.6 Lightweight transform constraints (`look-at` / `aim` / `parent-blend`).
//!
//! These are the runtime-cheap constraints found in production engines
//! (Unity's `Constraint` components, Unreal's control-rig look-at) — **not** a
//! full animation rig, which belongs to `prism_anim_runtime`. A constraint is
//! evaluated **after** hierarchy propagation and rewrites a node's world
//! [`GlobalTransform`]:
//!
//! | Constraint | Behaviour |
//! |---|---|
//! | [`LookAt`] | Orient so the node's forward (`-Z`) points at a world target, with an up hint. |
//! | [`Aim`] | Rotate minimally so a chosen local axis points at a world target (turret / camera). |
//! | [`ParentBlend`] | Weighted blend of several candidate world poses (vehicle hand-off). |
//! | [`PositionLimit`] | Clamp the world translation into an axis-aligned box. |
//!
//! Every solver is a pure deterministic function of its inputs: the world
//! target points / parent poses are resolved by the caller (so the solver never
//! needs an entity lookup), and the arithmetic is fixed-order `f32`, so a given
//! input always yields the same output. Constraints compose through
//! [`solve_chain`], which folds a slice in order; the caller is responsible for
//! ordering constraints and avoiding cyclic dependencies (see the honest
//! boundary in the design doc §24.9).

use alloc::vec::Vec;

use prism_math::{Mat3, Quat, Vec3};

use crate::GlobalTransform;

/// Tolerance below which two directions are treated as parallel / a vector is
/// treated as degenerate (zero length). Chosen well above `f32` round-off yet
/// far below any meaningful geometric angle.
const EPS: f32 = 1.0e-6;

/// Build the rotation whose forward axis (`-Z`, matching
/// [`crate::Transform::forward`]) points along `forward_dir`, using `up` as the
/// hint for the roll about that axis.
///
/// Returns [`Quat::IDENTITY`] if `forward_dir` is degenerate. If `forward_dir`
/// is (anti)parallel to `up`, a deterministic fallback up axis is chosen so the
/// basis is always well formed.
pub fn look_at_rotation(forward_dir: Vec3, up: Vec3) -> Quat {
    let f = normalize_or(forward_dir, Vec3::ZERO);
    if f == Vec3::ZERO {
        return Quat::IDENTITY;
    }
    let mut up = normalize_or(up, Vec3::Y);
    if up == Vec3::ZERO {
        up = Vec3::Y;
    }
    // If forward is (anti)parallel to up, pick a different, deterministic up.
    if abs_f32(f.dot(up)) > 1.0 - EPS {
        up = if abs_f32(f.dot(Vec3::Y)) > 1.0 - EPS {
            Vec3::Z
        } else {
            Vec3::Y
        };
    }
    let right = normalize_or(f.cross(up), Vec3::X);
    // Recompute an exactly-orthogonal up.
    let true_up = right.cross(f);
    // Columns are the world images of local +X, +Y, +Z. Local +Z is backward,
    // so it maps to `-f`.
    let basis = Mat3::from_cols(right, true_up, -f);
    Quat::from_mat3(basis)
}

/// Shortest-arc rotation taking unit vector `from` onto unit vector `to`.
///
/// Returns [`Quat::IDENTITY`] when either input is degenerate or they already
/// coincide; for an exact `180` degree flip it rotates about a stable axis
/// orthogonal to `from`.
pub fn rotation_arc(from: Vec3, to: Vec3) -> Quat {
    let a = normalize_or(from, Vec3::ZERO);
    let b = normalize_or(to, Vec3::ZERO);
    if a == Vec3::ZERO || b == Vec3::ZERO {
        return Quat::IDENTITY;
    }
    let d = a.dot(b).clamp(-1.0, 1.0);
    if d >= 1.0 - EPS {
        return Quat::IDENTITY;
    }
    if d <= -1.0 + EPS {
        // Opposite vectors: rotate 180 degrees about any axis orthogonal to a.
        let axis = orthonormal(a);
        return Quat::from_axis_angle(axis, core::f32::consts::PI);
    }
    let c = a.cross(b);
    // Half-angle quaternion built directly from cross/dot (robust, no trig).
    let s = sqrt_f32((1.0 + d) * 2.0);
    let inv_s = 1.0 / s;
    Quat::from_xyzw(c.x * inv_s, c.y * inv_s, c.z * inv_s, s * 0.5).normalize()
}

/// Keep orienting a node so its forward (`-Z`) axis points at `target`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LookAt {
    /// World-space point to look at.
    pub target: Vec3,
    /// World-space up hint used to resolve roll about the view direction.
    pub up: Vec3,
}

impl LookAt {
    /// Build a look-at constraint toward `target` with up hint `up`.
    #[inline]
    pub const fn new(target: Vec3, up: Vec3) -> Self {
        Self { target, up }
    }

    /// Solve against the node's `current` world transform: keep its translation
    /// and scale, replace its rotation so forward points at [`LookAt::target`].
    ///
    /// If the node already sits on the target (zero view direction) the current
    /// orientation is kept.
    pub fn solve(&self, current: GlobalTransform) -> GlobalTransform {
        let (scale, rotation, translation) = current.0.to_scale_rotation_translation();
        let dir = self.target - translation;
        if dir.length_squared() <= EPS * EPS {
            return current;
        }
        let new_rotation = look_at_rotation(dir, self.up);
        let _ = rotation;
        GlobalTransform::from_transform(&crate::Transform {
            translation,
            rotation: new_rotation,
            scale,
        })
    }
}

/// Rotate a node minimally so a chosen **local** axis points at `target`.
///
/// Unlike [`LookAt`], an `Aim` leaves one rotational degree of freedom free
/// (the twist about the aim axis is not constrained): it applies the smallest
/// rotation that brings the current world image of [`Aim::aim_axis`] onto the
/// direction to the target. This is the usual turret / spotlight behaviour.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Aim {
    /// The local axis (in the node's own space) that should point at the
    /// target, e.g. [`Vec3::Z`] for a `+Z`-barrelled turret.
    pub aim_axis: Vec3,
    /// World-space point to aim at.
    pub target: Vec3,
}

impl Aim {
    /// Build an aim constraint pointing `aim_axis` (local) at `target` (world).
    #[inline]
    pub const fn new(aim_axis: Vec3, target: Vec3) -> Self {
        Self { aim_axis, target }
    }

    /// Solve against the node's `current` world transform.
    ///
    /// Keeps translation and scale; post-multiplies the smallest rotation that
    /// moves the current world aim axis onto the direction to the target. If
    /// the node sits on the target the current orientation is kept.
    pub fn solve(&self, current: GlobalTransform) -> GlobalTransform {
        let (scale, rotation, translation) = current.0.to_scale_rotation_translation();
        let dir = self.target - translation;
        if dir.length_squared() <= EPS * EPS {
            return current;
        }
        let current_axis_world = rotation * self.aim_axis;
        let delta = rotation_arc(current_axis_world, dir);
        let new_rotation = (delta * rotation).normalize();
        GlobalTransform::from_transform(&crate::Transform {
            translation,
            rotation: new_rotation,
            scale,
        })
    }
}

/// Blend several candidate world poses by weight (e.g. transitioning a rider
/// between two moving vehicles).
///
/// Translation and scale blend linearly; rotation blends by a sign-aligned
/// normalized quaternion average (an `nlerp`-style mean), which is stable and
/// deterministic for the small, nearby orientations typical of a hand-off.
#[derive(Clone, Debug, Default)]
pub struct ParentBlend {
    /// Candidate world poses paired with their (not necessarily normalized)
    /// weights.
    pub poses: Vec<(GlobalTransform, f32)>,
}

impl ParentBlend {
    /// Create an empty blend.
    #[inline]
    pub fn new() -> Self {
        Self { poses: Vec::new() }
    }

    /// Add a candidate pose with the given weight.
    #[inline]
    pub fn push(&mut self, pose: GlobalTransform, weight: f32) {
        self.poses.push((pose, weight));
    }

    /// Blend the candidate poses. Returns [`GlobalTransform::IDENTITY`] when
    /// there are no candidates or the weights sum to (near) zero.
    ///
    /// `current` is ignored; a `ParentBlend` fully determines the world pose
    /// from its candidates.
    pub fn solve(&self, current: GlobalTransform) -> GlobalTransform {
        let _ = current;
        self.blend()
    }

    /// Blend the candidate poses into a single world transform.
    pub fn blend(&self) -> GlobalTransform {
        if self.poses.is_empty() {
            return GlobalTransform::IDENTITY;
        }
        let mut total = 0.0f32;
        for &(_, w) in &self.poses {
            total += w;
        }
        if abs_f32(total) <= EPS {
            return GlobalTransform::IDENTITY;
        }
        let inv_total = 1.0 / total;

        // Reference orientation = the first candidate; align every other
        // quaternion's sign to it before averaging so antipodal
        // representations do not cancel.
        let (first_pose, _) = self.poses[0];
        let (_, ref_rot, _) = first_pose.0.to_scale_rotation_translation();

        let mut translation = Vec3::ZERO;
        let mut scale = Vec3::ZERO;
        let mut qx = 0.0f32;
        let mut qy = 0.0f32;
        let mut qz = 0.0f32;
        let mut qw = 0.0f32;

        for &(pose, weight) in &self.poses {
            let w = weight * inv_total;
            let (s, r, t) = pose.0.to_scale_rotation_translation();
            translation += t * w;
            scale += s * w;
            let aligned = if r.dot(ref_rot) < 0.0 { -r } else { r };
            qx += aligned.x * w;
            qy += aligned.y * w;
            qz += aligned.z * w;
            qw += aligned.w * w;
        }

        let blended = Quat::from_xyzw(qx, qy, qz, qw);
        let rotation = if blended.length_squared() <= EPS * EPS {
            ref_rot
        } else {
            blended.normalize()
        };

        GlobalTransform::from_transform(&crate::Transform {
            translation,
            rotation,
            scale,
        })
    }
}

/// Clamp a node's world translation into an axis-aligned box.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PositionLimit {
    /// Lower corner of the allowed box.
    pub min: Vec3,
    /// Upper corner of the allowed box.
    pub max: Vec3,
}

impl PositionLimit {
    /// Build a position limit over `[min, max]`.
    #[inline]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Solve against `current`: clamp its translation, keeping orientation and
    /// scale.
    pub fn solve(&self, current: GlobalTransform) -> GlobalTransform {
        let (scale, rotation, translation) = current.0.to_scale_rotation_translation();
        let clamped = translation.clamp(self.min, self.max);
        if clamped == translation {
            return current;
        }
        GlobalTransform::from_transform(&crate::Transform {
            translation: clamped,
            rotation,
            scale,
        })
    }
}

/// A runtime constraint. Each variant carries the already-resolved world data
/// its solver needs, so [`Constraint::solve`] is a pure function of the
/// constraint and the node's current world pose.
#[derive(Clone, Debug)]
pub enum Constraint {
    /// Orient forward (`-Z`) at a world target.
    LookAt(LookAt),
    /// Point a local axis at a world target (minimal rotation).
    Aim(Aim),
    /// Weighted blend of candidate world poses.
    ParentBlend(ParentBlend),
    /// Clamp world translation into a box.
    PositionLimit(PositionLimit),
}

impl Constraint {
    /// Solve this constraint against the node's `current` world transform.
    pub fn solve(&self, current: GlobalTransform) -> GlobalTransform {
        match self {
            Self::LookAt(c) => c.solve(current),
            Self::Aim(c) => c.solve(current),
            Self::ParentBlend(c) => c.solve(current),
            Self::PositionLimit(c) => c.solve(current),
        }
    }
}

/// Apply a chain of constraints in slice order, feeding each solver the output
/// of the previous one. Order is significant and fully determines the result;
/// the caller must order constraints and avoid cycles.
pub fn solve_chain(constraints: &[Constraint], current: GlobalTransform) -> GlobalTransform {
    let mut pose = current;
    for c in constraints {
        pose = c.solve(pose);
    }
    pose
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Normalize `v`, or return `fallback` if `v` is (near) zero length.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq <= EPS * EPS {
        fallback
    } else {
        v * (1.0 / sqrt_f32(len_sq))
    }
}

/// Produce a unit vector orthogonal to the unit vector `v`, chosen
/// deterministically from the least-aligned cardinal axis.
#[inline]
fn orthonormal(v: Vec3) -> Vec3 {
    let axis = if abs_f32(v.x) <= abs_f32(v.y) && abs_f32(v.x) <= abs_f32(v.z) {
        Vec3::X
    } else if abs_f32(v.y) <= abs_f32(v.z) {
        Vec3::Y
    } else {
        Vec3::Z
    };
    normalize_or(v.cross(axis), Vec3::X)
}

#[inline]
fn abs_f32(x: f32) -> f32 {
    libm::fabsf(x)
}

#[inline]
fn sqrt_f32(x: f32) -> f32 {
    libm::sqrtf(x)
}
