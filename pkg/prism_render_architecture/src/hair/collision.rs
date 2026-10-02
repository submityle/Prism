//! Body-collider collision resolution for guide-strand dynamics.
//!
//! `TressFX`-class strand solvers keep hair off the body by projecting each
//! free particle out of a small set of analytic *collider* proxies (spheres and
//! capsules fitted to the head, neck, and shoulders) after every constraint
//! sweep (design §6.2). This module owns that projection as deterministic
//! array-in/array-out math: the same particles and colliders always produce
//! bit-identical positions, so it stays in the "compute-portable" bucket
//! (design §9) and can be golden-tested by hand.
//!
//! Only the analytic body proxies live here. Signed-distance-field (SDF) body
//! collision and strand self-collision are separate, heavier tiers on the same
//! roadmap item and are layered on later; the analytic proxies are the cheap,
//! always-on base that catches the common "hair through the face/shoulder"
//! artifact. A pinned particle (the skinned root) is never moved, and an empty
//! collider set is a no-op, so callers can pass `&[]` to disable collision
//! without a branch.

use prism_physics_core::soft::collision::{closest_point_on_segment, project_out_of_sphere};

use super::dynamics::{StrandParticle, Vec3};
use super::physics_bridge::{from_glam, to_glam};

/// An analytic collision proxy fitted to part of the body.
///
/// The proxies are intentionally minimal: a groom's collision budget is spent
/// on a handful of these fitted to the skinned skeleton (head sphere, neck and
/// shoulder capsules) rather than on the render mesh. Each variant knows how to
/// push a point out to its surface via [`Collider::push_out`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Collider {
    /// A solid sphere: points inside are pushed radially out to the surface.
    Sphere {
        /// World-space center.
        center: Vec3,
        /// Radius; a non-positive radius makes the collider inert.
        radius: f32,
    },
    /// A solid capsule (a line segment `a`..`b` inflated by `radius`): points
    /// within `radius` of the segment are pushed out along the shortest
    /// direction to the segment axis.
    Capsule {
        /// One segment endpoint.
        a: Vec3,
        /// The other segment endpoint.
        b: Vec3,
        /// Inflation radius; a non-positive radius makes the collider inert.
        radius: f32,
    },
}

impl Collider {
    /// Returns `point` pushed out of this collider, or `point` unchanged when
    /// it is already outside (or the collider is degenerate).
    ///
    /// The result lies exactly on the surface when the point was inside, which
    /// is what lets the solver settle hair to rest against the body instead of
    /// jittering.
    ///
    /// The analytic sphere/capsule projection is delegated to the authoritative
    /// physics engine ([`prism_physics_core::soft::collision`]) rather than
    /// re-implemented here, so the strand collider shares the single copy of the
    /// push-out math (the same delegation discipline as the edge-length solve in
    /// [`super::physics_bridge`]). The hair `Collider` enum stays strand-side as
    /// the compact, GPU-twin-facing proxy description; only the geometry call is
    /// routed through the engine.
    #[must_use]
    pub fn push_out(self, point: Vec3) -> Vec3 {
        let p = to_glam(point);
        let pushed = match self {
            Collider::Sphere { center, radius } => {
                project_out_of_sphere(p, to_glam(center), radius)
            }
            Collider::Capsule { a, b, radius } => {
                let closest = closest_point_on_segment(to_glam(a), to_glam(b), p);
                project_out_of_sphere(p, closest, radius)
            }
        };
        from_glam(pushed)
    }
}

/// Projects every free particle out of every collider, in place.
///
/// Particles are visited in order and, for each, every collider is applied in
/// order, so overlapping colliders resolve deterministically (the last one to
/// push wins for that particle). Pinned particles (the skinned root) are never
/// moved. An empty `colliders` slice is a no-op.
pub fn resolve_strand_collisions(particles: &mut [StrandParticle], colliders: &[Collider]) {
    if colliders.is_empty() {
        return;
    }
    for particle in particles.iter_mut() {
        if particle.is_pinned() {
            continue;
        }
        for collider in colliders {
            particle.position = collider.push_out(particle.position);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_at(x: f32, y: f32, z: f32) -> StrandParticle {
        StrandParticle::free(Vec3::new(x, y, z))
    }

    #[test]
    fn sphere_pushes_interior_point_to_surface() {
        let sphere = Collider::Sphere {
            center: Vec3::ZERO,
            radius: 2.0,
        };
        // A point one unit up from the center lands exactly on the top.
        let out = sphere.push_out(Vec3::new(0.0, 1.0, 0.0));
        assert!((out.x - 0.0).abs() < 1.0e-6);
        assert!((out.y - 2.0).abs() < 1.0e-6);
        assert!((out.z - 0.0).abs() < 1.0e-6);
    }

    #[test]
    fn sphere_leaves_exterior_point_untouched() {
        let sphere = Collider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let p = Vec3::new(3.0, 0.0, 0.0);
        assert_eq!(sphere.push_out(p), p);
    }

    #[test]
    fn sphere_center_point_escapes_along_up_axis() {
        let sphere = Collider::Sphere {
            center: Vec3::new(1.0, 2.0, 3.0),
            radius: 0.5,
        };
        let out = sphere.push_out(Vec3::new(1.0, 2.0, 3.0));
        assert!((out.x - 1.0).abs() < 1.0e-6);
        assert!((out.y - 2.5).abs() < 1.0e-6);
        assert!((out.z - 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn zero_radius_sphere_is_inert() {
        let sphere = Collider::Sphere {
            center: Vec3::ZERO,
            radius: 0.0,
        };
        let p = Vec3::new(0.0, 0.0, 0.0);
        assert_eq!(sphere.push_out(p), p);
    }

    #[test]
    fn capsule_pushes_point_out_perpendicular_to_axis() {
        // Axis along X from -1..1, radius 1. A point above the middle at height
        // 0.25 is inside and should be pushed straight up to height 1.
        let capsule = Collider::Capsule {
            a: Vec3::new(-1.0, 0.0, 0.0),
            b: Vec3::new(1.0, 0.0, 0.0),
            radius: 1.0,
        };
        let out = capsule.push_out(Vec3::new(0.0, 0.25, 0.0));
        assert!((out.x - 0.0).abs() < 1.0e-6);
        assert!((out.y - 1.0).abs() < 1.0e-6);
        assert!((out.z - 0.0).abs() < 1.0e-6);
    }

    #[test]
    fn capsule_uses_endpoint_cap_beyond_segment() {
        // A point past the `b` endpoint is measured against the cap, not the
        // infinite axis, so it is pushed out relative to `b`.
        let capsule = Collider::Capsule {
            a: Vec3::new(-1.0, 0.0, 0.0),
            b: Vec3::new(1.0, 0.0, 0.0),
            radius: 1.0,
        };
        let out = capsule.push_out(Vec3::new(1.5, 0.0, 0.0));
        // Closest axis point is `b`=(1,0,0); dist 0.5 < radius 1, pushed to 2.0.
        assert!((out.x - 2.0).abs() < 1.0e-6);
        assert!((out.y - 0.0).abs() < 1.0e-6);
        assert!((out.z - 0.0).abs() < 1.0e-6);
    }

    #[test]
    fn zero_length_capsule_behaves_like_sphere() {
        let capsule = Collider::Capsule {
            a: Vec3::ZERO,
            b: Vec3::ZERO,
            radius: 2.0,
        };
        let out = capsule.push_out(Vec3::new(0.0, 1.0, 0.0));
        assert!((out.y - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn resolve_skips_pinned_and_moves_free() {
        let mut particles = [StrandParticle::pinned(Vec3::ZERO), free_at(0.0, 0.5, 0.0)];
        let colliders = [Collider::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }];
        resolve_strand_collisions(&mut particles, &colliders);
        // Pinned root stays at the center even though it is inside the sphere.
        assert_eq!(particles[0].position, Vec3::ZERO);
        // Free particle is pushed to the surface.
        assert!((particles[1].position.y - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn empty_colliders_is_noop() {
        let mut particles = [free_at(0.0, 0.0, 0.0)];
        resolve_strand_collisions(&mut particles, &[]);
        assert_eq!(particles[0].position, Vec3::ZERO);
    }

    #[test]
    fn resolution_is_deterministic() {
        let colliders = [
            Collider::Sphere {
                center: Vec3::ZERO,
                radius: 1.0,
            },
            Collider::Capsule {
                a: Vec3::new(0.0, -1.0, 0.0),
                b: Vec3::new(0.0, 1.0, 0.0),
                radius: 0.5,
            },
        ];
        let mut a = [free_at(0.1, 0.2, 0.05), free_at(-0.3, 0.0, 0.1)];
        let mut b = a;
        resolve_strand_collisions(&mut a, &colliders);
        resolve_strand_collisions(&mut b, &colliders);
        assert_eq!(a[0].position, b[0].position);
        assert_eq!(a[1].position, b[1].position);
    }
}
