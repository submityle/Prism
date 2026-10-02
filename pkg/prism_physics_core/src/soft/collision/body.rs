//! Body-proxy collision and per-particle backstops for the soft-body kernel.
//!
//! A garment is simulated against a cheap *proxy* of the animated body: a small
//! set of analytic [`BodyCollider`] primitives (spheres, capsules, half-spaces)
//! that approximate limbs and torso, plus optional per-particle
//! [`Backstop`] planes that stop cloth from sinking into the skinned surface.
//! Both tiers are position-level projections run after the internal
//! constraints: a particle inside a collider is pushed to its surface, and a
//! particle that has sunk behind its backstop is pushed back onto the limiting
//! plane. With friction enabled, the tangential slide of each body contact is
//! rubbed with position-level Coulomb friction so cloth grips the body instead
//! of sliding freely.
//!
//! Like the self-collision pass, the resolvers are plain array-in / array-out
//! math over the particle store's raw columns
//! ([`crate::soft::particle::ParticleStorage`]): identical inputs produce
//! bit-identical outputs, pinned particles (`inverse_mass <= 0`) never move,
//! degenerate colliders fall back deterministically, and no path can produce a
//! [`f32::NAN`]. Only [`f32::sqrt`] is used.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! analytic projections and one-sided backstop are standard position-based
//! collision techniques; the tangential-friction projection is the one
//! published by Macklin et al. (2014), "Unified Particle Physics for Real-Time
//! Applications".

use glam::{Quat, Vec3};

use crate::math::scalar::Real;

use super::friction::{apply_coulomb_friction, sanitize_friction};
use super::EPS_LEN_SQ;

/// An analytic body-proxy collision primitive.
///
/// Each variant projects an interior point out to its surface; a point already
/// outside (or a degenerate primitive) is left untouched. The set is kept small
/// and analytic so a limb or torso proxy costs a handful of operations per
/// particle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BodyCollider {
    /// A solid sphere: interior points are pushed radially out to the surface.
    Sphere {
        /// World-space center.
        center: Vec3,
        /// Radius; a non-positive radius makes the collider inert.
        radius: Real,
    },
    /// A solid capsule (the segment `p0`..`p1` inflated by `radius`): points
    /// within `radius` of the segment are pushed out perpendicular to the
    /// nearest point on the segment axis.
    Capsule {
        /// One segment endpoint.
        p0: Vec3,
        /// The other segment endpoint.
        p1: Vec3,
        /// Inflation radius; a non-positive radius makes the collider inert.
        radius: Real,
    },
    /// A half-space with feasible region `normal.dot(x) >= offset` (a ground
    /// plane or backboard). Points on the infeasible side are pushed onto the
    /// plane along `normal`.
    HalfSpace {
        /// Plane normal pointing into the feasible region; need not be unit
        /// length. A (near) zero normal makes the collider inert.
        normal: Vec3,
        /// Plane offset measured against `normal`: the plane is the locus of
        /// points `x` with `normal.dot(x) == offset`.
        offset: Real,
    },
    /// A solid oriented box (OBB): interior points are pushed out along the
    /// local axis of least penetration to the nearest face. `orientation` maps
    /// box-local axes to world space (assumed unit); `half_extents` are the box
    /// half-sizes along its local x/y/z. A non-positive half extent collapses
    /// that axis and makes the box inert along it.
    Obb {
        /// World-space box center.
        center: Vec3,
        /// Orientation mapping box-local axes to world space (assumed unit).
        orientation: Quat,
        /// Half-extents along the box's local x/y/z axes.
        half_extents: Vec3,
    },
}

impl BodyCollider {
    /// Returns `pos` projected out of this collider, or `pos` unchanged when it
    /// is already outside (or the collider is degenerate).
    ///
    /// When the point was inside, the result lies exactly on the surface, which
    /// is what lets the solver settle cloth to rest against the body instead of
    /// jittering.
    #[must_use]
    pub fn project(self, pos: Vec3) -> Vec3 {
        match self {
            BodyCollider::Sphere { center, radius } => project_out_of_sphere(pos, center, radius),
            BodyCollider::Capsule { p0, p1, radius } => {
                let closest = closest_point_on_segment(p0, p1, pos);
                project_out_of_sphere(pos, closest, radius)
            }
            BodyCollider::HalfSpace { normal, offset } => {
                project_out_of_half_space(pos, normal, offset)
            }
            BodyCollider::Obb {
                center,
                orientation,
                half_extents,
            } => project_out_of_obb(pos, center, orientation, half_extents),
        }
    }
}

/// Projects `pos` out to the surface of the sphere `(center, radius)`.
///
/// A non-positive radius leaves the point untouched. When `pos` coincides with
/// `center` there is no defined radial direction, so the point is nudged out
/// along `+Y`: a fixed, deterministic fallback that avoids a [`f32::NAN`]
/// direction.
#[must_use]
pub fn project_out_of_sphere(pos: Vec3, center: Vec3, radius: Real) -> Vec3 {
    if radius <= 0.0 {
        return pos;
    }
    let delta = pos - center;
    let dist_sq = delta.length_squared();
    if dist_sq >= radius * radius {
        return pos;
    }
    if dist_sq <= EPS_LEN_SQ {
        // Coincident with the center: pick a fixed axis for a deterministic,
        // non-`NaN` result.
        return center + Vec3::new(0.0, radius, 0.0);
    }
    let dir = delta.normalize_or_zero();
    center + dir * radius
}

/// Projects `pos` onto the half-space plane `normal.dot(x) == offset` when it
/// lies on the infeasible side (`normal.dot(pos) < offset`), otherwise returns
/// `pos` unchanged.
///
/// The normal need not be unit length; the correction divides by
/// `normal.length_squared()` so the plane geometry is respected for any scale.
/// A (near) zero normal has no defined plane, so the point is returned
/// untouched rather than producing a [`f32::NAN`].
#[must_use]
pub fn project_out_of_half_space(pos: Vec3, normal: Vec3, offset: Real) -> Vec3 {
    let len_sq = normal.length_squared();
    if len_sq <= EPS_LEN_SQ {
        return pos;
    }
    let signed = normal.dot(pos) - offset;
    if signed >= 0.0 {
        return pos;
    }
    // Move along `normal` by `t` so that `normal.dot(pos + t*normal) == offset`.
    let t = -signed / len_sq;
    pos + normal * t
}

/// Projects `pos` out to the nearest face of the oriented box
/// `(center, orientation, half_extents)` when it lies inside, otherwise returns
/// `pos` unchanged.
///
/// The point is taken into the box's local frame (`orientation` maps local →
/// world, so its conjugate maps world → local for a unit quaternion). A point
/// strictly inside all three local slabs is pushed along the axis of *least*
/// penetration out to that face: the minimal translation that puts it on the
/// surface, matching the "push interior points to the surface" semantics of the
/// sphere and capsule arms. A point on or outside any slab is already outside
/// the solid and is returned untouched (the `>=` test mirrors the sphere arm's
/// `dist_sq >= r*r` early-out). A non-positive half extent collapses that axis
/// (its slab test is always "outside"), and an all-non-positive box is inert.
/// The transform is multiplies plus the quaternion rotation only, and the
/// center fallback picks fixed `+` faces, so no path yields a [`f32::NAN`].
#[must_use]
pub fn project_out_of_obb(pos: Vec3, center: Vec3, orientation: Quat, half_extents: Vec3) -> Vec3 {
    // No positive extent means there is no interior to project out of.
    if half_extents.x <= 0.0 && half_extents.y <= 0.0 && half_extents.z <= 0.0 {
        return pos;
    }
    // World -> local. For a unit quaternion the conjugate is the inverse
    // rotation; body orientations are unit, matching the capsule-axis path.
    let local = orientation.conjugate() * (pos - center);
    // Outside any slab => already outside the solid box (a collapsed axis with
    // a non-positive half extent is always "outside", making the box inert).
    if local.x.abs() >= half_extents.x
        || local.y.abs() >= half_extents.y
        || local.z.abs() >= half_extents.z
    {
        return pos;
    }
    // Interior: push to the face of least penetration (smallest `he - |local|`).
    let pen = half_extents - local.abs();
    let mut local_out = local;
    if pen.x <= pen.y && pen.x <= pen.z {
        local_out.x = if local.x >= 0.0 {
            half_extents.x
        } else {
            -half_extents.x
        };
    } else if pen.y <= pen.z {
        local_out.y = if local.y >= 0.0 {
            half_extents.y
        } else {
            -half_extents.y
        };
    } else {
        local_out.z = if local.z >= 0.0 {
            half_extents.z
        } else {
            -half_extents.z
        };
    }
    center + orientation * local_out
}

/// Returns the point on segment `p0`..`p1` closest to `pos`.
///
/// The projection parameter is clamped to `[0, 1]` so the result never leaves
/// the segment, which turns the capsule end-caps into hemispheres. A
/// zero-length segment (`p0 == p1`) degenerates gracefully to `p0`, so a
/// collapsed capsule behaves like a sphere.
#[must_use]
pub fn closest_point_on_segment(p0: Vec3, p1: Vec3, pos: Vec3) -> Vec3 {
    let axis = p1 - p0;
    let len_sq = axis.length_squared();
    if len_sq <= EPS_LEN_SQ {
        return p0;
    }
    let t = ((pos - p0).dot(axis) / len_sq).clamp(0.0, 1.0);
    p0 + axis * t
}

/// Projects every free particle out of every body collider, in place.
///
/// `positions` is the particle position column and `inverse_masses` the
/// index-aligned inverse-mass column (`0` marks a pinned particle). Particles
/// are visited in index order and, for each, every collider is applied in slice
/// order, so overlapping colliders resolve deterministically (the last collider
/// to push wins for that particle). The cost is `O(particles * colliders)`.
/// Pinned particles are never moved. An empty `colliders` slice, or an
/// `inverse_masses` slice whose length differs from `positions`, is a no-op, so
/// callers can pass `&[]` to disable body collision without a branch.
pub fn resolve_body_collisions(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    colliders: &[BodyCollider],
) {
    if colliders.is_empty() || inverse_masses.len() != positions.len() {
        return;
    }
    for (pos, &inv_mass) in positions.iter_mut().zip(inverse_masses.iter()) {
        if inv_mass <= 0.0 {
            continue;
        }
        for collider in colliders {
            *pos = collider.project(*pos);
        }
    }
}

/// Resolves body collision like [`resolve_body_collisions`], but rubs the
/// tangential slide of every body contact with position-level Coulomb friction
/// so cloth grips the body instead of sliding freely.
///
/// The projection, traversal order, and pinned-particle handling are identical
/// to [`resolve_body_collisions`]; friction is layered on per contact using
/// each particle's frame-start position from `prev_positions`. For each collider
/// the normal push `projected - before` yields both the contact normal and its
/// magnitude, which drive [`apply_coulomb_friction`]. A contact whose push is at
/// or below [`EPS_LEN_SQ`] (the particle was already outside) applies no
/// friction. `friction` is clamped to `0..=1`; a value of `0` delegates
/// straight to [`resolve_body_collisions`]. An empty `colliders` slice or a
/// mismatched `inverse_masses` length is a no-op, and a `prev_positions` slice
/// shorter than `positions` degrades to no tangential slide (hence no friction)
/// for the missing indices.
pub fn resolve_body_collisions_with_friction(
    positions: &mut [Vec3],
    prev_positions: &[Vec3],
    inverse_masses: &[Real],
    colliders: &[BodyCollider],
    friction: Real,
) {
    if colliders.is_empty() || inverse_masses.len() != positions.len() {
        return;
    }
    let mu = sanitize_friction(friction);
    if mu <= 0.0 {
        resolve_body_collisions(positions, inverse_masses, colliders);
        return;
    }
    for (index, (pos, &inv_mass)) in positions.iter_mut().zip(inverse_masses.iter()).enumerate() {
        if inv_mass <= 0.0 {
            continue;
        }
        // Frame-start position for this particle; the current position (no
        // tangential slide, hence no friction) is the safe fallback when the
        // snapshot is missing.
        let prev = prev_positions.get(index).copied().unwrap_or(*pos);
        for collider in colliders {
            let before = *pos;
            let projected = collider.project(before);
            let correction = projected - before;
            let push_sq = correction.length_squared();
            if push_sq <= EPS_LEN_SQ {
                // Already outside this collider: no contact, no friction.
                *pos = projected;
                continue;
            }
            let push = push_sq.sqrt();
            let normal = correction * (1.0 / push);
            *pos = apply_coulomb_friction(projected, prev, normal, push, mu);
        }
    }
}

/// A per-particle backstop plane anchored to the skinned pose.
///
/// A backstop is a one-sided constraint: the particle may move freely in front
/// of the plane (along `+normal`) but must not sink more than `distance` behind
/// `origin` along `-normal`. Concretely, with `s = normal.dot(pos - origin)`
/// the feasible region is `s >= -distance`; a particle at `s < -distance` is
/// pushed forward onto that limiting plane. This matches the painted-backstop
/// authoring in production cloth: it stops a garment from collapsing into the
/// body while still letting it billow outward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Backstop {
    /// Anchor point on the skinned surface (typically the skinned position of
    /// the particle's rest vertex).
    pub origin: Vec3,
    /// Outward plane normal; need not be unit length. A (near) zero normal
    /// makes the backstop inert.
    pub normal: Vec3,
    /// How far behind `origin` (along `-normal`) the particle may travel before
    /// the backstop pushes it back. A non-positive distance pins the particle
    /// to the front side of `origin`.
    pub distance: Real,
}

/// Returns `pos` clamped to the front side of `backstop`.
///
/// When the signed distance `normal.dot(pos - origin)` drops below `-distance`
/// the point is pushed forward along the (unit) normal onto the limiting plane;
/// otherwise `pos` is returned unchanged. A (near) zero normal has no defined
/// plane, so the point is returned untouched rather than producing a
/// [`f32::NAN`].
#[must_use]
pub fn apply_backstop(pos: Vec3, backstop: Backstop) -> Vec3 {
    let len_sq = backstop.normal.length_squared();
    if len_sq <= EPS_LEN_SQ {
        return pos;
    }
    let n = backstop.normal.normalize_or_zero();
    let s = n.dot(pos - backstop.origin);
    let min_s = -backstop.distance;
    if s < min_s {
        pos + n * (min_s - s)
    } else {
        pos
    }
}

/// Applies each backstop to its matching particle, in place.
///
/// `backstops[i]` constrains `positions[i]`; `inverse_masses` is the
/// index-aligned inverse-mass column (`0` marks a pinned particle). The pass
/// runs over the shorter of the position and backstop lengths, so a short
/// `backstops` slice simply leaves the trailing particles unconstrained (and
/// never panics). Pinned particles are skipped. An empty `backstops` slice, or
/// an `inverse_masses` slice whose length differs from `positions`, is a no-op.
pub fn resolve_backstops(positions: &mut [Vec3], inverse_masses: &[Real], backstops: &[Backstop]) {
    if inverse_masses.len() != positions.len() {
        return;
    }
    for ((pos, &inv_mass), backstop) in positions
        .iter_mut()
        .zip(inverse_masses.iter())
        .zip(backstops.iter())
    {
        if inv_mass <= 0.0 {
            continue;
        }
        *pos = apply_backstop(*pos, *backstop);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: Real = 1.0e-6;

    fn approx_eq(a: Vec3, b: Vec3) {
        assert!((a - b).length() < TOL, "{a:?} != {b:?}");
    }

    #[test]
    fn sphere_pushes_interior_point_to_surface() {
        let out = project_out_of_sphere(Vec3::new(0.5, 0.0, 0.0), Vec3::ZERO, 1.0);
        approx_eq(out, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn sphere_leaves_exterior_point_untouched() {
        let p = Vec3::new(2.0, 0.0, 0.0);
        approx_eq(project_out_of_sphere(p, Vec3::ZERO, 1.0), p);
    }

    #[test]
    fn sphere_center_point_escapes_along_up_axis() {
        let out = project_out_of_sphere(Vec3::ZERO, Vec3::ZERO, 1.0);
        approx_eq(out, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn zero_radius_sphere_is_inert() {
        let p = Vec3::new(0.1, 0.2, 0.3);
        approx_eq(project_out_of_sphere(p, Vec3::ZERO, 0.0), p);
    }

    #[test]
    fn capsule_pushes_point_out_perpendicular_to_axis() {
        let capsule = BodyCollider::Capsule {
            p0: Vec3::new(-1.0, 0.0, 0.0),
            p1: Vec3::new(1.0, 0.0, 0.0),
            radius: 1.0,
        };
        // A point above the axis mid-span is pushed straight up to the surface.
        let out = capsule.project(Vec3::new(0.0, 0.5, 0.0));
        approx_eq(out, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn capsule_uses_endpoint_cap_beyond_segment() {
        let capsule = BodyCollider::Capsule {
            p0: Vec3::new(-1.0, 0.0, 0.0),
            p1: Vec3::new(1.0, 0.0, 0.0),
            radius: 1.0,
        };
        // Beyond the `p1` endpoint: the nearest axis point is the endpoint, so
        // the cap behaves like a sphere centred at `p1`.
        let out = capsule.project(Vec3::new(1.5, 0.0, 0.0));
        approx_eq(out, Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn closest_point_clamps_to_endpoints() {
        let p0 = Vec3::new(-1.0, 0.0, 0.0);
        let p1 = Vec3::new(1.0, 0.0, 0.0);
        approx_eq(
            closest_point_on_segment(p0, p1, Vec3::new(-5.0, 1.0, 0.0)),
            p0,
        );
        approx_eq(
            closest_point_on_segment(p0, p1, Vec3::new(5.0, 1.0, 0.0)),
            p1,
        );
        approx_eq(
            closest_point_on_segment(p0, p1, Vec3::new(0.0, 1.0, 0.0)),
            Vec3::ZERO,
        );
    }

    #[test]
    fn zero_length_capsule_behaves_like_sphere() {
        let capsule = BodyCollider::Capsule {
            p0: Vec3::ZERO,
            p1: Vec3::ZERO,
            radius: 1.0,
        };
        let out = capsule.project(Vec3::new(0.5, 0.0, 0.0));
        approx_eq(out, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn half_space_pushes_infeasible_point_to_plane() {
        // Feasible region is y >= 0; a point below is pushed up to y = 0.
        let plane = BodyCollider::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        };
        approx_eq(
            plane.project(Vec3::new(0.3, -0.5, 0.2)),
            Vec3::new(0.3, 0.0, 0.2),
        );
    }

    #[test]
    fn half_space_leaves_feasible_point_untouched() {
        let plane = BodyCollider::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        };
        let p = Vec3::new(0.3, 0.5, 0.2);
        approx_eq(plane.project(p), p);
    }

    #[test]
    fn half_space_respects_non_unit_normal() {
        // Normal of length 2 still projects onto the same geometric plane y = 1.
        let plane = BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 2.0, 0.0),
            offset: 2.0,
        };
        approx_eq(
            plane.project(Vec3::new(0.0, 0.0, 0.0)),
            Vec3::new(0.0, 1.0, 0.0),
        );
    }

    #[test]
    fn zero_normal_half_space_is_inert() {
        let plane = BodyCollider::HalfSpace {
            normal: Vec3::ZERO,
            offset: 1.0,
        };
        let p = Vec3::new(0.1, 0.2, 0.3);
        approx_eq(plane.project(p), p);
    }

    fn unit_obb() -> BodyCollider {
        BodyCollider::Obb {
            center: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            half_extents: Vec3::new(1.0, 0.5, 2.0),
        }
    }

    #[test]
    fn obb_pushes_interior_point_to_nearest_face() {
        // Point just below the top face (local axis of least penetration is +Y,
        // the thinnest half-extent of 0.5). It should land on y = 0.5, keeping
        // x and z, because that face is nearer than the x or z faces.
        let obb = unit_obb();
        let out = obb.project(Vec3::new(0.2, 0.4, -0.3));
        approx_eq(out, Vec3::new(0.2, 0.5, -0.3));
    }

    #[test]
    fn obb_leaves_exterior_point_untouched() {
        let obb = unit_obb();
        // Outside along x (|x| = 1.5 > 1.0): untouched.
        let p = Vec3::new(1.5, 0.0, 0.0);
        approx_eq(obb.project(p), p);
    }

    #[test]
    fn obb_point_on_face_is_treated_as_outside() {
        let obb = unit_obb();
        // Exactly on the +Y face (y = 0.5): the `>=` slab test leaves it put.
        let p = Vec3::new(0.0, 0.5, 0.0);
        approx_eq(obb.project(p), p);
    }

    #[test]
    fn obb_center_point_escapes_along_a_fixed_face() {
        // At the exact center the penetration along each axis equals that
        // axis's half-extent, so the least-penetration axis is the thinnest
        // one — Y (0.5) for this box — and the point exits to the +Y face
        // deterministically (`local.y >= 0.0` picks the `+` side).
        let obb = unit_obb();
        let out = obb.project(Vec3::ZERO);
        approx_eq(out, Vec3::new(0.0, 0.5, 0.0));
    }

    #[test]
    fn obb_respects_orientation_true_face_not_inscribed_sphere() {
        // A box rotated 45° about Z. A particle sitting just inside the top
        // (local +Y) face at a world point must be pushed to the *rotated*
        // face, not to where an axis-aligned or inscribed-sphere proxy would
        // put it. Build a known interior local point, map it to world, project,
        // and confirm it lands on the oriented +Y face.
        let orientation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let half_extents = Vec3::new(1.0, 0.5, 2.0);
        let obb = BodyCollider::Obb {
            center: Vec3::new(3.0, 1.0, -2.0),
            orientation,
            half_extents,
        };
        // Interior local point near the +Y face (least-penetration axis).
        let local_in = Vec3::new(0.1, 0.45, 0.3);
        let world_in = Vec3::new(3.0, 1.0, -2.0) + orientation * local_in;
        let out = obb.project(world_in);
        // Expected: same local point snapped to y = +0.5, mapped back to world.
        let local_out = Vec3::new(0.1, 0.5, 0.3);
        let world_out = Vec3::new(3.0, 1.0, -2.0) + orientation * local_out;
        approx_eq(out, world_out);
        // Sanity: an inscribed-sphere proxy (radius = min half extent 0.5) would
        // have produced a different point, so the face really governs.
        let sphere_out = project_out_of_sphere(world_in, Vec3::new(3.0, 1.0, -2.0), 0.5);
        assert!(
            (sphere_out - out).length() > 1.0e-3,
            "box face must differ from inscribed-sphere projection"
        );
    }

    #[test]
    fn obb_degenerate_box_is_inert() {
        let obb = BodyCollider::Obb {
            center: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            half_extents: Vec3::ZERO,
        };
        let p = Vec3::new(0.1, -0.2, 0.3);
        approx_eq(obb.project(p), p);
    }

    #[test]
    fn resolve_body_skips_pinned_and_moves_free() {
        let mut positions = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [0.0, 1.0];
        let colliders = [BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }];
        resolve_body_collisions(&mut positions, &inverse_masses, &colliders);
        // Pinned particle at the center is untouched despite being interior.
        approx_eq(positions[0], Vec3::ZERO);
        // Free particle is pushed to the surface.
        approx_eq(positions[1], Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn resolve_body_empty_colliders_is_noop() {
        let mut positions = [Vec3::ZERO];
        let inverse_masses = [1.0];
        resolve_body_collisions(&mut positions, &inverse_masses, &[]);
        approx_eq(positions[0], Vec3::ZERO);
    }

    #[test]
    fn resolve_body_mismatched_inverse_mass_is_noop() {
        let mut positions = [Vec3::new(0.5, 0.0, 0.0)];
        let inverse_masses = [1.0, 1.0];
        let colliders = [BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }];
        resolve_body_collisions(&mut positions, &inverse_masses, &colliders);
        approx_eq(positions[0], Vec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn resolve_body_is_deterministic() {
        let run = || {
            let mut positions = [Vec3::new(0.2, 0.1, 0.0), Vec3::new(0.4, -0.3, 0.1)];
            let inverse_masses = [1.0, 1.0];
            let colliders = [
                BodyCollider::Sphere {
                    center: Vec3::ZERO,
                    radius: 1.0,
                },
                BodyCollider::HalfSpace {
                    normal: Vec3::Y,
                    offset: 0.0,
                },
            ];
            resolve_body_collisions(&mut positions, &inverse_masses, &colliders);
            positions
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn body_friction_zero_matches_frictionless() {
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        }];
        let mut with = [Vec3::new(0.3, -0.5, 0.0)];
        let mut plain = with;
        let prev = [Vec3::new(-0.2, -0.5, 0.0)];
        let inverse_masses = [1.0];
        resolve_body_collisions_with_friction(&mut with, &prev, &inverse_masses, &colliders, 0.0);
        resolve_body_collisions(&mut plain, &inverse_masses, &colliders);
        approx_eq(with[0], plain[0]);
    }

    #[test]
    fn body_static_friction_cancels_slide_inside_cone() {
        // Floor at y = 0; particle penetrates by 0.5 (large normal push) and
        // slides a tiny amount tangentially, well inside the cone, so a unit
        // friction coefficient cancels the whole tangential slide.
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        }];
        let mut positions = [Vec3::new(0.01, -0.5, 0.0)];
        let prev = [Vec3::new(0.0, -0.5, 0.0)];
        let inverse_masses = [1.0];
        resolve_body_collisions_with_friction(
            &mut positions,
            &prev,
            &inverse_masses,
            &colliders,
            1.0,
        );
        // Normal component is resolved onto the plane (y = 0).
        assert!(positions[0].y.abs() < TOL, "y: {}", positions[0].y);
        // Tangential slide (x) is fully removed back to the frame-start x.
        assert!(
            (positions[0].x - prev[0].x).abs() < TOL,
            "x: {}",
            positions[0].x
        );
    }

    #[test]
    fn body_dynamic_friction_shrinks_slide() {
        // Floor at y = 0; small normal push (0.1) but large tangential slide so
        // only `mu * push` is removed, leaving residual slide.
        let colliders = [BodyCollider::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        }];
        let mu = 0.25;
        let push = 0.1;
        let mut positions = [Vec3::new(1.0, -push, 0.0)];
        let prev = [Vec3::new(0.0, -push, 0.0)];
        let inverse_masses = [1.0];
        resolve_body_collisions_with_friction(
            &mut positions,
            &prev,
            &inverse_masses,
            &colliders,
            mu,
        );
        // y resolved to the plane.
        assert!(positions[0].y.abs() < TOL, "y: {}", positions[0].y);
        // x slide of 1.0 shrunk by exactly mu*push.
        assert!(
            (positions[0].x - (1.0 - mu * push)).abs() < TOL,
            "x: {}",
            positions[0].x
        );
    }

    #[test]
    fn backstop_pushes_point_behind_plane_forward() {
        // Backstop faces +Y, anchored at the origin, allowing 0.1 behind.
        let backstop = Backstop {
            origin: Vec3::ZERO,
            normal: Vec3::Y,
            distance: 0.1,
        };
        // A point 0.5 behind is pushed forward to the limiting plane y = -0.1.
        let out = apply_backstop(Vec3::new(0.2, -0.5, 0.3), backstop);
        approx_eq(out, Vec3::new(0.2, -0.1, 0.3));
    }

    #[test]
    fn backstop_leaves_point_in_front_untouched() {
        let backstop = Backstop {
            origin: Vec3::ZERO,
            normal: Vec3::Y,
            distance: 0.1,
        };
        let p = Vec3::new(0.2, 0.5, 0.3);
        approx_eq(apply_backstop(p, backstop), p);
    }

    #[test]
    fn zero_normal_backstop_is_inert() {
        let backstop = Backstop {
            origin: Vec3::ZERO,
            normal: Vec3::ZERO,
            distance: 0.1,
        };
        let p = Vec3::new(0.1, -5.0, 0.2);
        approx_eq(apply_backstop(p, backstop), p);
    }

    #[test]
    fn resolve_backstops_skips_pinned_and_respects_short_slice() {
        let mut positions = [
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
        ];
        let inverse_masses = [0.0, 1.0, 1.0];
        let backstops = [
            Backstop {
                origin: Vec3::ZERO,
                normal: Vec3::Y,
                distance: 0.0,
            },
            Backstop {
                origin: Vec3::ZERO,
                normal: Vec3::Y,
                distance: 0.0,
            },
        ];
        resolve_backstops(&mut positions, &inverse_masses, &backstops);
        // Pinned particle untouched even though it is behind the plane.
        approx_eq(positions[0], Vec3::new(0.0, -0.5, 0.0));
        // Free particle with a backstop is pushed to y = 0.
        approx_eq(positions[1], Vec3::ZERO);
        // Third particle has no backstop (short slice): untouched.
        approx_eq(positions[2], Vec3::new(0.0, -0.5, 0.0));
    }

    #[test]
    fn resolve_backstops_mismatched_inverse_mass_is_noop() {
        let mut positions = [Vec3::new(0.0, -0.5, 0.0)];
        let inverse_masses = [1.0, 1.0];
        let backstops = [Backstop {
            origin: Vec3::ZERO,
            normal: Vec3::Y,
            distance: 0.0,
        }];
        resolve_backstops(&mut positions, &inverse_masses, &backstops);
        approx_eq(positions[0], Vec3::new(0.0, -0.5, 0.0));
    }
}
