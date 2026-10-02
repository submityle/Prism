//! Body-proxy collision, self-collision, and backstop projection for cloth.
//!
//! Production cloth engines (`PhysX` Clothing, `Havok` Cloth, NVIDIA
//! `NvCloth`, UE5 `Chaos` Cloth, Houdini `Vellum`) keep a garment off the body
//! and out of itself with three cooperating tiers (design §4, §6.2):
//!
//! 1. **Body proxies** — a handful of analytic colliders (sphere, capsule,
//!    half-space) fitted to the skinned skeleton. Each cloth particle inside a
//!    proxy is projected to its surface. This is the cheap, always-on base that
//!    catches "cloth through the body" the way the strand solver does for hair.
//!    A particle pushed to a proxy's surface also has its tangential slip
//!    across the frame damped by the fabric's Coulomb friction, so silk and
//!    wool slide differently over the same body (see
//!    [`resolve_body_collisions_with_friction`]).
//! 2. **Backstops** — per-particle one-sided planes anchored to the skinned
//!    pose. A backstop keeps its particle from sinking more than an authored
//!    distance behind the body, which is how painted backstop constraints stop
//!    a skirt from collapsing into the legs without fully pinning it.
//! 3. **Self-collision** — a deterministic uniform spatial hash. Particles are
//!    bucketed by integer cell, and only the 27-cell neighborhood of each
//!    bucket is tested, so the pass is near `O(n)` instead of `O(n^2)`. Pairs
//!    closer than a cloth `thickness` are pushed symmetrically apart with
//!    inverse-mass weighting (a pinned partner does not move; its free partner
//!    takes the whole correction).
//!
//! Everything here is deterministic array-in/array-out math: the same particles
//! and colliders always produce bit-identical positions, so the stage is
//! CPU-golden-testable. Only [`f32::sqrt`] is used; no transcendental functions
//! are called. Pinned particles (`inverse_mass <= 0`) are never moved, an empty
//! collider/backstop set is a no-op, out-of-range accesses are skipped rather
//! than panicking, and degenerate inputs (zero radius, zero-length segment,
//! zero normal, coincident particles) fall back deterministically and never
//! produce `NaN`.
//!
//! The self-collision tier here is the discrete, position-level approximation:
//! it resolves interpenetration measured at the current positions, and the
//! `thickness` buffer provides the tunnelling margin (a particle moving less
//! than `thickness` per step cannot pass cleanly through another). A full
//! continuous-collision-detection (CCD) sweep against per-particle motion
//! segments is a heavier future slot layered on the same spatial hash; it is
//! deliberately not stubbed here so this module stays fully runnable.

use alloc::vec::Vec;

use super::{physics_bridge, ClothParticle, Vec3};

/// An analytic collision proxy fitted to part of the body.
///
/// The set is intentionally small: a garment's body-collision budget is spent
/// on a few of these fitted to the skinned skeleton rather than on the render
/// mesh, mirroring the collider sets in `PhysX` Clothing and `Havok` Cloth.
/// Each variant knows how to push a point to its surface via
/// [`BodyCollider::project`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BodyCollider {
    /// A solid sphere: interior points are pushed radially out to the surface.
    Sphere {
        /// World-space center.
        center: Vec3,
        /// Radius; a non-positive radius makes the collider inert.
        radius: f32,
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
        radius: f32,
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
        offset: f32,
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
        }
    }
}

/// Projects `pos` out to the surface of the sphere `(center, radius)`.
///
/// A non-positive radius leaves the point untouched. When `pos` coincides with
/// `center` there is no defined radial direction, so the point is nudged out
/// along `+Y`: a fixed, deterministic fallback that avoids a `NaN` direction.
#[must_use]
pub fn project_out_of_sphere(pos: Vec3, center: Vec3, radius: f32) -> Vec3 {
    physics_bridge::from_glam(prism_physics_core::soft::collision::project_out_of_sphere(
        physics_bridge::to_glam(pos),
        physics_bridge::to_glam(center),
        radius,
    ))
}

/// Projects `pos` onto the half-space plane `normal.dot(x) == offset` when it
/// lies on the infeasible side (`normal.dot(pos) < offset`), otherwise returns
/// `pos` unchanged.
///
/// The normal need not be unit length; the correction divides by
/// `normal.length_squared()` so the plane geometry is respected for any scale.
/// A (near) zero normal has no defined plane, so the point is returned
/// untouched rather than producing a `NaN`.
#[must_use]
pub fn project_out_of_half_space(pos: Vec3, normal: Vec3, offset: f32) -> Vec3 {
    physics_bridge::from_glam(prism_physics_core::soft::collision::project_out_of_half_space(
        physics_bridge::to_glam(pos),
        physics_bridge::to_glam(normal),
        offset,
    ))
}

/// Returns the point on segment `p0`..`p1` closest to `pos`.
///
/// The projection parameter is clamped to `[0, 1]` so the result never leaves
/// the segment, which turns the capsule end-caps into hemispheres. A
/// zero-length segment (`p0 == p1`) degenerates gracefully to `p0`, so a
/// collapsed capsule behaves like a sphere.
#[must_use]
pub fn closest_point_on_segment(p0: Vec3, p1: Vec3, pos: Vec3) -> Vec3 {
    physics_bridge::from_glam(prism_physics_core::soft::collision::closest_point_on_segment(
        physics_bridge::to_glam(p0),
        physics_bridge::to_glam(p1),
        physics_bridge::to_glam(pos),
    ))
}

/// Converts a render [`BodyCollider`] into the physics-engine collider the
/// single-source resolver consumes. The variants and fields line up exactly;
/// only the vector type differs, so this is a lossless component copy.
#[must_use]
pub(super) fn to_physics_collider(collider: BodyCollider) -> prism_physics_core::soft::collision::BodyCollider {
    use prism_physics_core::soft::collision::BodyCollider as Phys;
    match collider {
        BodyCollider::Sphere { center, radius } => Phys::Sphere {
            center: physics_bridge::to_glam(center),
            radius,
        },
        BodyCollider::Capsule { p0, p1, radius } => Phys::Capsule {
            p0: physics_bridge::to_glam(p0),
            p1: physics_bridge::to_glam(p1),
            radius,
        },
        BodyCollider::HalfSpace { normal, offset } => Phys::HalfSpace {
            normal: physics_bridge::to_glam(normal),
            offset,
        },
    }
}

/// Converts a physics-engine collider back into the render [`BodyCollider`],
/// the inverse of [`to_physics_collider`]. Used to write a two-way-coupling
/// body's translated pose back onto the render-side proxy after the
/// single-source resolver has moved it.
#[must_use]
pub(super) fn from_physics_collider(collider: prism_physics_core::soft::collision::BodyCollider) -> BodyCollider {
    use prism_physics_core::soft::collision::BodyCollider as Phys;
    match collider {
        Phys::Sphere { center, radius } => BodyCollider::Sphere {
            center: physics_bridge::from_glam(center),
            radius,
        },
        Phys::Capsule { p0, p1, radius } => BodyCollider::Capsule {
            p0: physics_bridge::from_glam(p0),
            p1: physics_bridge::from_glam(p1),
            radius,
        },
        Phys::HalfSpace { normal, offset } => BodyCollider::HalfSpace {
            normal: physics_bridge::from_glam(normal),
            offset,
        },
    }
}

/// Projects every free particle out of every body collider, in place.
///
/// Particles are visited in index order and, for each, every collider is
/// applied in slice order, so overlapping colliders resolve deterministically
/// (the last collider to push wins for that particle). The cost is
/// `O(particles * colliders)`. Pinned particles are never moved. An empty
/// `colliders` slice is a no-op, so callers can pass `&[]` to disable body
/// collision without a branch.
pub fn resolve_body_collisions(particles: &mut [ClothParticle], colliders: &[BodyCollider]) {
    if colliders.is_empty() {
        return;
    }
    let physics_colliders: Vec<_> = colliders.iter().copied().map(to_physics_collider).collect();
    let (mut positions, inverse_masses) = physics_bridge::to_soa(particles);
    prism_physics_core::soft::collision::resolve_body_collisions(
        &mut positions,
        &inverse_masses,
        &physics_colliders,
    );
    physics_bridge::write_positions_back(particles, &positions);
}

/// Projects every free particle out of every body collider like
/// [`resolve_body_collisions`], then applies position-level Coulomb friction
/// after each collider push so cloth grips the body instead of sliding
/// frictionlessly (design gap: `FabricMaterial::friction` was previously
/// unconsumed by any collision pass).
///
/// For each collider the outward unit normal and push-out depth come straight
/// from the projection displacement (`projected - before`); friction then rubs
/// the particle's tangential slide since its frame-start position
/// `prev_positions[i]` against that contact via the physics-engine Coulomb
/// friction projection (Macklin et al. 2014). Body
/// proxies are infinitely massive, so the whole tangential correction lands on
/// the particle. `friction` is the material coefficient, clamped to `0..=1`; a
/// value of `0` reproduces [`resolve_body_collisions`] exactly.
///
/// Particles are visited in index order and colliders in slice order, matching
/// [`resolve_body_collisions`], so the cost stays `O(particles * colliders)`
/// with no hidden inner loop. Pinned particles never move, an empty collider
/// slice is a no-op, and a `prev_positions` slice shorter than `particles`
/// falls back to no tangential slide (hence no friction) for the missing
/// indices rather than panicking.
pub fn resolve_body_collisions_with_friction(
    particles: &mut [ClothParticle],
    prev_positions: &[Vec3],
    colliders: &[BodyCollider],
    friction: f32,
) {
    if colliders.is_empty() {
        return;
    }
    let physics_colliders: Vec<_> = colliders.iter().copied().map(to_physics_collider).collect();
    let (mut positions, inverse_masses) = physics_bridge::to_soa(particles);
    let prev: Vec<_> = prev_positions
        .iter()
        .copied()
        .map(physics_bridge::to_glam)
        .collect();
    prism_physics_core::soft::collision::resolve_body_collisions_with_friction(
        &mut positions,
        &prev,
        &inverse_masses,
        &physics_colliders,
        friction,
    );
    physics_bridge::write_positions_back(particles, &positions);
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
    pub distance: f32,
}

/// Converts a render [`Backstop`] into the physics-engine backstop the
/// single-source resolver consumes (a lossless component copy).
#[must_use]
fn to_physics_backstop(backstop: Backstop) -> prism_physics_core::soft::collision::Backstop {
    prism_physics_core::soft::collision::Backstop {
        origin: physics_bridge::to_glam(backstop.origin),
        normal: physics_bridge::to_glam(backstop.normal),
        distance: backstop.distance,
    }
}

/// Returns `pos` clamped to the front side of `backstop`.
///
/// When the signed distance `normal.dot(pos - origin)` drops below `-distance`
/// the point is pushed forward along the (unit) normal onto the limiting plane;
/// otherwise `pos` is returned unchanged. A (near) zero normal has no defined
/// plane, so the point is returned untouched rather than producing a `NaN`.
#[must_use]
pub fn apply_backstop(pos: Vec3, backstop: Backstop) -> Vec3 {
    physics_bridge::from_glam(prism_physics_core::soft::collision::apply_backstop(
        physics_bridge::to_glam(pos),
        to_physics_backstop(backstop),
    ))
}

/// Applies each backstop to its matching particle, in place.
///
/// `backstops[i]` constrains `particles[i]`. The pass runs over the shorter of
/// the two lengths, so a short `backstops` slice simply leaves the trailing
/// particles unconstrained (and never panics). Pinned particles are skipped. An
/// empty `backstops` slice is a no-op.
pub fn resolve_backstops(particles: &mut [ClothParticle], backstops: &[Backstop]) {
    let physics_backstops: Vec<_> =
        backstops.iter().copied().map(to_physics_backstop).collect();
    let (mut positions, inverse_masses) = physics_bridge::to_soa(particles);
    prism_physics_core::soft::collision::resolve_backstops(
        &mut positions,
        &inverse_masses,
        &physics_backstops,
    );
    physics_bridge::write_positions_back(particles, &positions);
}

/// Resolves cloth self-collision with a deterministic uniform spatial hash.
///
/// The separation law lives in the physics engine
/// (`prism_physics_core::soft::collision::resolve_self_collision`): particles
/// are bucketed by integer cell, each particle tests only its 27-cell
/// neighborhood, and a pair closer than `thickness` is pushed symmetrically
/// apart split by inverse mass (a pinned partner never moves, so its free
/// partner takes the whole correction). Coincident particles separate along a
/// fixed `+X` axis so the result stays deterministic and free of `NaN`. The
/// render path keeps no second copy of this solver; it only marshals its
/// compact particle layout into the structure-of-arrays columns the engine
/// consumes and writes the solved positions back.
///
/// A non-positive `cell_size` or `thickness`, or fewer than two particles, is a
/// no-op.
pub fn resolve_self_collision(particles: &mut [ClothParticle], cell_size: f32, thickness: f32) {
    // `physics_bridge::to_soa` maps a pinned particle to a zero inverse mass,
    // which reproduces the former `inverse_mass.max(0.0)` weighting exactly
    // because `ClothParticle::is_pinned()` is defined as `inverse_mass <= 0`.
    let (mut positions, inverse_masses) = physics_bridge::to_soa(particles);
    prism_physics_core::soft::collision::resolve_self_collision(
        &mut positions,
        &inverse_masses,
        cell_size,
        thickness,
    );
    physics_bridge::write_positions_back(particles, &positions);
}

/// Resolves self-collision like [`resolve_self_collision`], but rubs the
/// tangential slide of every separated pair with position-level Coulomb
/// friction so stacked cloth layers grip instead of shearing freely.
///
/// The spatial hash, traversal order, and inverse-mass-weighted normal push are
/// identical to [`resolve_self_collision`]; the physics engine layers Coulomb
/// friction (Macklin et al. 2014) on each separated pair using its frame-start
/// position from `prev_positions`. `friction` is clamped to `0..=1`; a value of
/// `0` reproduces [`resolve_self_collision`] exactly. A non-positive
/// `cell_size` or `thickness`, or fewer than two particles, is a no-op, and a
/// short `prev_positions` slice degrades to no friction for the missing
/// indices.
pub fn resolve_self_collision_with_friction(
    particles: &mut [ClothParticle],
    prev_positions: &[Vec3],
    cell_size: f32,
    thickness: f32,
    friction: f32,
) {
    // The friction separation law lives in the physics engine
    // (`prism_physics_core::soft::collision::resolve_self_collision_with_friction`).
    // Marshal the render particle layout plus the frame-start snapshot into the
    // structure-of-arrays columns that solver consumes, then write the solved
    // positions back. A short `prev_positions` slice is preserved verbatim, so
    // the physics solver falls back to no tangential slide for the missing
    // indices exactly as the former in-line pass did.
    let (mut positions, inverse_masses) = physics_bridge::to_soa(particles);
    let prev: Vec<_> = prev_positions
        .iter()
        .copied()
        .map(physics_bridge::to_glam)
        .collect();
    prism_physics_core::soft::collision::resolve_self_collision_with_friction(
        &mut positions,
        &prev,
        &inverse_masses,
        cell_size,
        thickness,
        friction,
    );
    physics_bridge::write_positions_back(particles, &positions);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a movable particle with unit inverse mass at `(x, y, z)`.
    fn free_at(x: f32, y: f32, z: f32) -> ClothParticle {
        ClothParticle::new(Vec3::new(x, y, z), 1.0)
    }

    const TOL: f32 = 1.0e-6;

    /// Asserts two vectors are equal within [`TOL`].
    fn approx_eq(a: Vec3, b: Vec3) {
        assert!((a.x - b.x).abs() < TOL, "x: {} vs {}", a.x, b.x);
        assert!((a.y - b.y).abs() < TOL, "y: {} vs {}", a.y, b.y);
        assert!((a.z - b.z).abs() < TOL, "z: {} vs {}", a.z, b.z);
    }

    #[test]
    fn sphere_pushes_interior_point_to_surface() {
        let out = project_out_of_sphere(Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO, 2.0);
        approx_eq(out, Vec3::new(0.0, 2.0, 0.0));
    }

    #[test]
    fn sphere_leaves_exterior_point_untouched() {
        let p = Vec3::new(3.0, 0.0, 0.0);
        assert_eq!(project_out_of_sphere(p, Vec3::ZERO, 1.0), p);
    }

    #[test]
    fn sphere_center_point_escapes_along_up_axis() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let out = project_out_of_sphere(center, center, 0.5);
        approx_eq(out, Vec3::new(1.0, 2.5, 3.0));
    }

    #[test]
    fn zero_radius_sphere_is_inert() {
        let p = Vec3::new(0.0, 0.0, 0.0);
        assert_eq!(project_out_of_sphere(p, Vec3::ZERO, 0.0), p);
    }

    #[test]
    fn capsule_pushes_point_out_perpendicular_to_axis() {
        // Axis along X from -1..1, radius 1. A point above the middle at 0.25 is
        // inside and is pushed straight up to height 1.
        let capsule = BodyCollider::Capsule {
            p0: Vec3::new(-1.0, 0.0, 0.0),
            p1: Vec3::new(1.0, 0.0, 0.0),
            radius: 1.0,
        };
        approx_eq(
            capsule.project(Vec3::new(0.0, 0.25, 0.0)),
            Vec3::new(0.0, 1.0, 0.0),
        );
    }

    #[test]
    fn capsule_uses_endpoint_cap_beyond_segment() {
        // A point past `p1` is measured against the cap at `p1`, not the
        // infinite axis: closest axis point is (1,0,0), dist 0.5 < radius 1, so
        // it is pushed to 2.0.
        let capsule = BodyCollider::Capsule {
            p0: Vec3::new(-1.0, 0.0, 0.0),
            p1: Vec3::new(1.0, 0.0, 0.0),
            radius: 1.0,
        };
        approx_eq(
            capsule.project(Vec3::new(1.5, 0.0, 0.0)),
            Vec3::new(2.0, 0.0, 0.0),
        );
    }

    #[test]
    fn capsule_closest_point_clamps_to_endpoints() {
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let p1 = Vec3::new(4.0, 0.0, 0.0);
        // Behind p0 clamps to p0.
        approx_eq(
            closest_point_on_segment(p0, p1, Vec3::new(-2.0, 3.0, 0.0)),
            p0,
        );
        // Beyond p1 clamps to p1.
        approx_eq(
            closest_point_on_segment(p0, p1, Vec3::new(9.0, -1.0, 0.0)),
            p1,
        );
        // In the middle projects onto the axis.
        approx_eq(
            closest_point_on_segment(p0, p1, Vec3::new(2.0, 5.0, 0.0)),
            Vec3::new(2.0, 0.0, 0.0),
        );
    }

    #[test]
    fn zero_length_capsule_behaves_like_sphere() {
        let capsule = BodyCollider::Capsule {
            p0: Vec3::ZERO,
            p1: Vec3::ZERO,
            radius: 2.0,
        };
        approx_eq(
            capsule.project(Vec3::new(0.0, 1.0, 0.0)),
            Vec3::new(0.0, 2.0, 0.0),
        );
    }

    #[test]
    fn half_space_pushes_infeasible_point_to_plane() {
        // Feasible region y >= 0. A point at y = -0.5 lands on y = 0.
        let plane = BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        };
        approx_eq(
            plane.project(Vec3::new(2.0, -0.5, -3.0)),
            Vec3::new(2.0, 0.0, -3.0),
        );
    }

    #[test]
    fn half_space_leaves_feasible_point_untouched() {
        let plane = BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        };
        let p = Vec3::new(1.0, 4.0, 2.0);
        assert_eq!(plane.project(p), p);
    }

    #[test]
    fn half_space_respects_non_unit_normal() {
        // Normal (0,2,0), offset 4 => plane y = 2. A point at y = 1 is pushed
        // to y = 2 regardless of the normal's scale.
        let plane = BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 2.0, 0.0),
            offset: 4.0,
        };
        approx_eq(
            plane.project(Vec3::new(0.0, 1.0, 0.0)),
            Vec3::new(0.0, 2.0, 0.0),
        );
    }

    #[test]
    fn zero_normal_half_space_is_inert() {
        let plane = BodyCollider::HalfSpace {
            normal: Vec3::ZERO,
            offset: 5.0,
        };
        let p = Vec3::new(0.0, -10.0, 0.0);
        assert_eq!(plane.project(p), p);
    }

    #[test]
    fn resolve_body_skips_pinned_and_moves_free() {
        let mut particles = [ClothParticle::pinned(Vec3::ZERO), free_at(0.0, 0.5, 0.0)];
        let colliders = [BodyCollider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }];
        resolve_body_collisions(&mut particles, &colliders);
        // Pinned particle stays put even though it is inside the sphere.
        assert_eq!(particles[0].position, Vec3::ZERO);
        // Free particle is pushed to the surface.
        approx_eq(particles[1].position, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn resolve_body_empty_colliders_is_noop() {
        let mut particles = [free_at(0.0, 0.0, 0.0)];
        resolve_body_collisions(&mut particles, &[]);
        assert_eq!(particles[0].position, Vec3::ZERO);
    }

    #[test]
    fn resolve_body_is_deterministic() {
        let colliders = [
            BodyCollider::Sphere {
                center: Vec3::ZERO,
                radius: 1.0,
            },
            BodyCollider::Capsule {
                p0: Vec3::new(0.0, -1.0, 0.0),
                p1: Vec3::new(0.0, 1.0, 0.0),
                radius: 0.5,
            },
        ];
        let mut a = [free_at(0.1, 0.2, 0.05), free_at(-0.3, 0.0, 0.1)];
        let mut b = a;
        resolve_body_collisions(&mut a, &colliders);
        resolve_body_collisions(&mut b, &colliders);
        assert_eq!(a[0].position, b[0].position);
        assert_eq!(a[1].position, b[1].position);
    }

    /// A ground-plane collider: feasible region `y >= 0`, outward normal `+Y`.
    fn floor() -> BodyCollider {
        BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: 0.0,
        }
    }

    #[test]
    fn friction_zero_is_identical_to_frictionless() {
        // mu = 0 with a frame-start reference must match the plain frictionless
        // pass exactly: the particle lands on the floor with no tangential damp.
        let colliders = [floor()];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let mut with = [free_at(1.0, -0.2, 0.0)];
        resolve_body_collisions_with_friction(&mut with, &prev, &colliders, 0.0);
        approx_eq(with[0].position, Vec3::new(1.0, 0.0, 0.0));

        let mut without = [free_at(1.0, -0.2, 0.0)];
        resolve_body_collisions(&mut without, &colliders);
        approx_eq(without[0].position, with[0].position);
    }

    #[test]
    fn friction_static_sticks_when_slip_below_cone() {
        // Depth 0.5, tangential slip 0.1 < mu * depth = 0.5, so the slip is
        // removed entirely and the particle sticks at its tangential origin.
        let colliders = [floor()];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let mut particles = [free_at(0.1, -0.5, 0.0)];
        resolve_body_collisions_with_friction(&mut particles, &prev, &colliders, 1.0);
        approx_eq(particles[0].position, Vec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn friction_dynamic_decays_tangential_linearly() {
        // Depth 0.2, tangential slip 1.0 > mu * depth for these mu, so the
        // removed slip is exactly mu * 0.2 and x = 1 - mu * 0.2, linear in mu.
        let colliders = [floor()];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        for (mu, want_x) in [(0.25_f32, 0.95_f32), (0.5, 0.9), (1.0, 0.8)] {
            let mut particles = [free_at(1.0, -0.2, 0.0)];
            resolve_body_collisions_with_friction(&mut particles, &prev, &colliders, mu);
            approx_eq(particles[0].position, Vec3::new(want_x, 0.0, 0.0));
        }
    }

    #[test]
    fn friction_larger_mu_slides_less() {
        // Strictly monotone: more friction removes more tangential slip, so the
        // surviving x-slide strictly decreases as mu grows.
        let colliders = [floor()];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let mut last_x = f32::INFINITY;
        for mu in [0.0_f32, 0.1, 0.2, 0.3, 0.4] {
            let mut particles = [free_at(1.0, -0.2, 0.0)];
            resolve_body_collisions_with_friction(&mut particles, &prev, &colliders, mu);
            let x = particles[0].position.x;
            assert!(x < last_x, "mu {mu}: x {x} not < {last_x}");
            last_x = x;
        }
    }

    #[test]
    fn friction_skips_particles_without_penetration() {
        // A particle already above the floor is never projected, so there is no
        // contact and no tangential damp regardless of mu.
        let colliders = [floor()];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let mut particles = [free_at(2.0, 5.0, 0.0)];
        resolve_body_collisions_with_friction(&mut particles, &prev, &colliders, 1.0);
        approx_eq(particles[0].position, Vec3::new(2.0, 5.0, 0.0));
    }

    #[test]
    fn friction_handles_pinned_and_short_prev() {
        // Index 0 is pinned and never moves. Index 1 has no matching prev entry
        // (prev has length 1), so it takes the frictionless `None` branch.
        let colliders = [floor()];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let mut particles = [
            ClothParticle::pinned(Vec3::new(1.0, -0.2, 0.0)),
            free_at(1.0, -0.2, 0.0),
        ];
        resolve_body_collisions_with_friction(&mut particles, &prev, &colliders, 1.0);
        assert_eq!(particles[0].position, Vec3::new(1.0, -0.2, 0.0));
        approx_eq(particles[1].position, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn friction_nan_coefficient_is_treated_as_frictionless() {
        // A NaN mu clamps to 0 (frictionless) and never leaks a NaN into a
        // position.
        let colliders = [floor()];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let mut particles = [free_at(1.0, -0.2, 0.0)];
        resolve_body_collisions_with_friction(&mut particles, &prev, &colliders, f32::NAN);
        approx_eq(particles[0].position, Vec3::new(1.0, 0.0, 0.0));
        assert!(particles[0].position.x.is_finite());
    }

    #[test]
    fn friction_is_deterministic() {
        let colliders = [floor()];
        let prev = [Vec3::new(0.0, 1.0, 0.0)];
        let mut a = [free_at(1.0, -0.2, 0.0)];
        let mut b = [free_at(1.0, -0.2, 0.0)];
        resolve_body_collisions_with_friction(&mut a, &prev, &colliders, 0.4);
        resolve_body_collisions_with_friction(&mut b, &prev, &colliders, 0.4);
        assert_eq!(a[0].position, b[0].position);
    }

    #[test]
    fn backstop_pushes_particle_behind_limit_back() {
        // Outward normal +Y, origin at y = 0, may sink 1 unit behind => limit
        // plane y = -1. A particle at y = -3 is pushed forward to y = -1.
        let backstop = Backstop {
            origin: Vec3::ZERO,
            normal: Vec3::new(0.0, 1.0, 0.0),
            distance: 1.0,
        };
        approx_eq(
            apply_backstop(Vec3::new(2.0, -3.0, 1.0), backstop),
            Vec3::new(2.0, -1.0, 1.0),
        );
    }

    #[test]
    fn backstop_leaves_legal_particle_untouched() {
        let backstop = Backstop {
            origin: Vec3::ZERO,
            normal: Vec3::new(0.0, 1.0, 0.0),
            distance: 1.0,
        };
        // In front of the origin: legal, untouched.
        let front = Vec3::new(0.0, 5.0, 0.0);
        assert_eq!(apply_backstop(front, backstop), front);
        // Slightly behind but within the distance budget: legal, untouched.
        let within = Vec3::new(0.0, -0.5, 0.0);
        assert_eq!(apply_backstop(within, backstop), within);
    }

    #[test]
    fn backstop_zero_normal_is_inert() {
        let backstop = Backstop {
            origin: Vec3::ZERO,
            normal: Vec3::ZERO,
            distance: 1.0,
        };
        let p = Vec3::new(0.0, -10.0, 0.0);
        assert_eq!(apply_backstop(p, backstop), p);
    }

    #[test]
    fn resolve_backstops_matches_by_index_and_skips_pinned() {
        let backstop = Backstop {
            origin: Vec3::ZERO,
            normal: Vec3::new(0.0, 1.0, 0.0),
            distance: 0.0,
        };
        let mut particles = [
            ClothParticle::pinned(Vec3::new(0.0, -5.0, 0.0)),
            free_at(0.0, -5.0, 0.0),
        ];
        resolve_backstops(&mut particles, &[backstop, backstop]);
        // Pinned particle is never moved.
        assert_eq!(particles[0].position, Vec3::new(0.0, -5.0, 0.0));
        // Free particle is clamped onto the limit plane (y = 0).
        approx_eq(particles[1].position, Vec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn resolve_backstops_short_slice_does_not_panic() {
        let backstop = Backstop {
            origin: Vec3::ZERO,
            normal: Vec3::new(0.0, 1.0, 0.0),
            distance: 0.0,
        };
        let mut particles = [free_at(0.0, -1.0, 0.0), free_at(0.0, -1.0, 0.0)];
        // Only one backstop for two particles: the second is left untouched.
        resolve_backstops(&mut particles, &[backstop]);
        approx_eq(particles[0].position, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(particles[1].position, Vec3::new(0.0, -1.0, 0.0));
    }

    #[test]
    fn self_collision_separates_two_close_free_particles() {
        // Two free particles 0.4 apart with thickness 1.0 => each moves 0.3 so
        // they end up exactly 1.0 apart, symmetric about the midpoint.
        let mut particles = [free_at(0.0, 0.0, 0.0), free_at(0.4, 0.0, 0.0)];
        resolve_self_collision(&mut particles, 1.0, 1.0);
        let sep = particles[1].position.distance(particles[0].position);
        assert!((sep - 1.0).abs() < TOL, "separation {sep}");
        // Symmetric split about the original midpoint x = 0.2.
        approx_eq(particles[0].position, Vec3::new(-0.3, 0.0, 0.0));
        approx_eq(particles[1].position, Vec3::new(0.7, 0.0, 0.0));
    }

    #[test]
    fn self_collision_pinned_partner_takes_no_correction() {
        // Pinned + free at distance 0.4, thickness 1.0 => only the free one
        // moves, and it moves the whole 0.6 to reach separation 1.0.
        let mut particles = [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            free_at(0.4, 0.0, 0.0),
        ];
        resolve_self_collision(&mut particles, 1.0, 1.0);
        assert_eq!(particles[0].position, Vec3::new(0.0, 0.0, 0.0));
        approx_eq(particles[1].position, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn self_collision_leaves_distant_particles_untouched() {
        let mut particles = [free_at(0.0, 0.0, 0.0), free_at(5.0, 0.0, 0.0)];
        resolve_self_collision(&mut particles, 1.0, 1.0);
        assert_eq!(particles[0].position, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(particles[1].position, Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn self_collision_coincident_particles_separate_without_nan() {
        let mut particles = [free_at(1.0, 1.0, 1.0), free_at(1.0, 1.0, 1.0)];
        resolve_self_collision(&mut particles, 1.0, 1.0);
        let sep = particles[1].position.distance(particles[0].position);
        assert!(sep.is_finite());
        assert!((sep - 1.0).abs() < TOL, "separation {sep}");
        // Fixed +X separation axis, symmetric split of the full thickness.
        approx_eq(particles[0].position, Vec3::new(0.5, 1.0, 1.0));
        approx_eq(particles[1].position, Vec3::new(1.5, 1.0, 1.0));
    }

    #[test]
    fn self_collision_is_deterministic() {
        let mut a = [
            free_at(0.0, 0.0, 0.0),
            free_at(0.3, 0.1, 0.0),
            free_at(0.9, 0.0, 0.2),
            free_at(1.05, 0.05, 0.15),
        ];
        let mut b = a;
        resolve_self_collision(&mut a, 0.5, 0.5);
        resolve_self_collision(&mut b, 0.5, 0.5);
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.position, pb.position);
        }
    }

    #[test]
    fn self_collision_no_op_guards() {
        let mut particles = [free_at(0.0, 0.0, 0.0), free_at(0.1, 0.0, 0.0)];
        let snapshot = particles;
        // Non-positive cell size / thickness and a single particle are no-ops.
        resolve_self_collision(&mut particles, 0.0, 1.0);
        resolve_self_collision(&mut particles, 1.0, 0.0);
        let mut single = [free_at(0.0, 0.0, 0.0)];
        resolve_self_collision(&mut single, 1.0, 1.0);
        assert_eq!(particles[0].position, snapshot[0].position);
        assert_eq!(particles[1].position, snapshot[1].position);
        assert_eq!(single[0].position, Vec3::ZERO);
    }

    #[test]
    fn self_collision_neighbor_cells_are_tested_across_boundaries() {
        // Two particles straddling a cell boundary (cell_size 0.5): x = 0.45 is
        // in cell 0, x = 0.55 in cell 1. They must still be detected as a pair
        // and pushed to thickness 0.5 apart.
        let mut particles = [free_at(0.45, 0.0, 0.0), free_at(0.55, 0.0, 0.0)];
        resolve_self_collision(&mut particles, 0.5, 0.5);
        let sep = particles[1].position.distance(particles[0].position);
        assert!((sep - 0.5).abs() < TOL, "separation {sep}");
    }

    // ----- Coulomb friction (body collision) -----

    #[test]
    fn body_friction_zero_mu_leaves_tangential_slide_untouched() {
        // Penetrating particle that slid +X: with mu = 0 only the normal push
        // applies, so x is preserved (pure `resolve_body_collisions` behaviour).
        let mut particles = [free_at(1.0, -0.5, 0.0)];
        let prev = [Vec3::new(0.0, 0.0, 0.0)];
        resolve_body_collisions_with_friction(&mut particles, &prev, &[floor()], 0.0);
        approx_eq(particles[0].position, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn body_friction_static_cone_locks_small_slide() {
        // Slide 0.3, push-out depth 0.5, mu = 1 => 0.3 <= 1.0 * 0.5, so the
        // whole tangential slide is cancelled: the particle locks at x = 0.
        let mut particles = [free_at(0.3, -0.5, 0.0)];
        let prev = [Vec3::new(0.0, 0.0, 0.0)];
        resolve_body_collisions_with_friction(&mut particles, &prev, &[floor()], 1.0);
        approx_eq(particles[0].position, Vec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn body_friction_dynamic_shrinks_slide_by_mu_times_depth() {
        // Slide 1.0, depth 0.5, mu = 0.25 => remove mu*depth = 0.125, leaving
        // x = 0.875 with the slide direction unchanged.
        let mut particles = [free_at(1.0, -0.5, 0.0)];
        let prev = [Vec3::new(0.0, 0.0, 0.0)];
        resolve_body_collisions_with_friction(&mut particles, &prev, &[floor()], 0.25);
        approx_eq(particles[0].position, Vec3::new(0.875, 0.0, 0.0));
    }

    #[test]
    fn body_friction_mu_one_grips_harder_than_half() {
        let prev = [Vec3::new(0.0, 0.0, 0.0)];
        let mut half = [free_at(1.0, -0.5, 0.0)];
        let mut full = [free_at(1.0, -0.5, 0.0)];
        resolve_body_collisions_with_friction(&mut half, &prev, &[floor()], 0.5);
        resolve_body_collisions_with_friction(&mut full, &prev, &[floor()], 1.0);
        // Stronger friction leaves less residual tangential travel.
        assert!(full[0].position.x < half[0].position.x);
        assert!(half[0].position.x < 1.0);
    }

    #[test]
    fn body_friction_skips_pinned_particle() {
        let mut particles = [ClothParticle::pinned(Vec3::new(0.5, -0.5, 0.0))];
        let prev = [Vec3::new(0.0, 0.0, 0.0)];
        resolve_body_collisions_with_friction(&mut particles, &prev, &[floor()], 1.0);
        assert_eq!(particles[0].position, Vec3::new(0.5, -0.5, 0.0));
    }

    #[test]
    fn body_friction_no_contact_leaves_particle_free() {
        // Particle above the plane: no push-out, so no friction even at mu = 1.
        let mut particles = [free_at(2.0, 3.0, 0.0)];
        let prev = [Vec3::new(0.0, 3.0, 0.0)];
        resolve_body_collisions_with_friction(&mut particles, &prev, &[floor()], 1.0);
        approx_eq(particles[0].position, Vec3::new(2.0, 3.0, 0.0));
    }

    #[test]
    fn body_friction_missing_prev_slice_is_no_friction_not_panic() {
        // Empty prev slice => fallback prev = current position => zero slide, so
        // only the normal push applies and nothing panics.
        let mut particles = [free_at(1.0, -0.5, 0.0)];
        resolve_body_collisions_with_friction(&mut particles, &[], &[floor()], 1.0);
        approx_eq(particles[0].position, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn body_friction_is_finite_for_coincident_center_projection() {
        // Particle at a sphere centre escapes along +Y; friction must stay
        // finite even though the tangential slide degenerates.
        let sphere = BodyCollider::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 0.5,
        };
        let mut particles = [free_at(0.0, 0.0, 0.0)];
        let prev = [Vec3::new(0.1, 0.0, 0.0)];
        resolve_body_collisions_with_friction(&mut particles, &prev, &[sphere], 1.0);
        let p = particles[0].position;
        assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
    }

    // ----- Coulomb friction (self collision) -----

    #[test]
    fn self_friction_zero_mu_matches_plain_separation() {
        // Pinned + free at distance 0.4, thickness 1.0: mu = 0 reproduces the
        // plain separation (free partner reaches separation 1.0 in X).
        let mut particles = [ClothParticle::pinned(Vec3::ZERO), free_at(0.4, 0.0, 0.0)];
        let prev = [Vec3::ZERO, Vec3::new(0.4, -0.5, 0.0)];
        resolve_self_collision_with_friction(&mut particles, &prev, 1.0, 1.0, 0.0);
        approx_eq(particles[1].position, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn self_friction_static_cone_locks_relative_slide() {
        // Penetration 0.6, tangential slide 0.5 (in Y): mu = 1 => 0.5 <= 0.6,
        // so the free partner's slide is fully cancelled back to y = -0.5.
        let mut particles = [ClothParticle::pinned(Vec3::ZERO), free_at(0.4, 0.0, 0.0)];
        let prev = [Vec3::ZERO, Vec3::new(0.4, -0.5, 0.0)];
        resolve_self_collision_with_friction(&mut particles, &prev, 1.0, 1.0, 1.0);
        approx_eq(particles[0].position, Vec3::ZERO);
        approx_eq(particles[1].position, Vec3::new(1.0, -0.5, 0.0));
    }

    #[test]
    fn self_friction_dynamic_removes_mu_times_penetration() {
        // mu*penetration = 0.5 * 0.6 = 0.3 removed from the 0.5 slide, leaving
        // the free partner at y = -0.3.
        let mut particles = [ClothParticle::pinned(Vec3::ZERO), free_at(0.4, 0.0, 0.0)];
        let prev = [Vec3::ZERO, Vec3::new(0.4, -0.5, 0.0)];
        resolve_self_collision_with_friction(&mut particles, &prev, 1.0, 1.0, 0.5);
        approx_eq(particles[1].position, Vec3::new(1.0, -0.3, 0.0));
    }

    #[test]
    fn self_friction_is_deterministic() {
        let build = || {
            let mut p = [ClothParticle::pinned(Vec3::ZERO), free_at(0.4, 0.05, 0.0)];
            let prev = [Vec3::ZERO, Vec3::new(0.4, -0.5, 0.1)];
            resolve_self_collision_with_friction(&mut p, &prev, 1.0, 1.0, 0.7);
            p
        };
        let a = build();
        let b = build();
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.position, pb.position);
        }
    }
}
