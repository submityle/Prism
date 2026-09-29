//! Signed-distance-field (SDF) body collision for guide-strand dynamics.
//!
//! Analytic sphere/capsule proxies (see [`super::collision`]) are the cheap,
//! always-on base that keeps hair off the head and shoulders, but a body is not
//! a union of a few round primitives: a jaw line, a collarbone, or a prop
//! against the scalp needs a tighter fit than a sphere gives. Production strand
//! solvers (`TressFX` 4) add a *signed distance field* collider for exactly
//! this: the body is sampled as a field whose value is negative inside the
//! surface and positive outside, and any particle that lands inside is pushed
//! back along the field gradient to the zero isosurface (design §6.2, the
//! heavier tier layered on top of the analytic proxies).
//!
//! This module owns a compact, deterministic analytic SDF built from a union of
//! primitives (half-space planes, axis-aligned boxes, spheres, and capsules).
//! The union distance is the minimum over the primitives, and a point is pushed
//! out along the numeric gradient of that union field until it is no longer
//! inside any primitive or an iteration cap is reached. Everything is
//! array-in/array-out math (design §9): the same particles and primitives
//! always produce bit-identical positions, so it stays in the compute-portable
//! bucket and can be golden-tested by hand. Pinned particles (skinned roots)
//! never move, an empty primitive set is a no-op, and no input panics — a
//! particle sitting exactly on a zero-gradient spot escapes along a fixed axis
//! instead of producing a NaN direction.

use alloc::vec::Vec;

use super::dynamics::{StrandParticle, Vec3};

/// Vectors shorter than the square root of this are treated as zero-length,
/// matching the epsilon used by the dynamics solver and the analytic colliders.
const EPS_LEN_SQ: f32 = 1.0e-24;

/// Step used by the central-difference gradient of the union field, in world
/// units. Small relative to a body-scale collider so the numeric gradient
/// tracks the true surface normal closely.
const GRAD_STEP: f32 = 1.0e-3;

/// Default number of push-out relaxation passes.
///
/// One pass moves a point to the zero isosurface of the *nearest* primitive;
/// extra passes let it settle when unions overlap so it ends up outside the
/// whole body rather than tunnelling from one primitive into another.
pub const DEFAULT_SDF_ITERATIONS: u32 = 4;

/// One primitive of a body SDF. The collider is the union of a slice of these;
/// each contributes a signed distance that is negative inside its solid region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SdfPrimitive {
    /// A solid sphere of the given `radius` about `center`.
    Sphere {
        /// World-space center.
        center: Vec3,
        /// Radius; a non-positive radius makes the primitive inert.
        radius: f32,
    },
    /// A solid capsule: the segment `a`..`b` inflated by `radius`.
    Capsule {
        /// One segment endpoint.
        a: Vec3,
        /// The other segment endpoint.
        b: Vec3,
        /// Inflation radius; a non-positive radius makes the primitive inert.
        radius: f32,
    },
    /// A solid half-space: the region on the negative side of the plane
    /// `dot(normal, p) = offset`. `normal` need not be unit length; a
    /// zero-length normal makes the primitive inert.
    HalfSpace {
        /// Plane normal, pointing *out* of the solid region.
        normal: Vec3,
        /// Signed plane offset along the (normalized) normal.
        offset: f32,
    },
    /// A solid axis-aligned box centered at `center` with the given positive
    /// half extents. A non-positive half extent on any axis makes the primitive
    /// inert.
    Box {
        /// World-space center.
        center: Vec3,
        /// Half the box size on each axis.
        half_extents: Vec3,
    },
}

impl SdfPrimitive {
    /// Signed distance from `point` to this primitive: negative inside the
    /// solid region, positive outside, zero on the surface. An inert primitive
    /// (degenerate radius/extent/normal) reports [`f32::INFINITY`] so it never
    /// counts as containing a point.
    #[must_use]
    pub fn signed_distance(self, point: Vec3) -> f32 {
        match self {
            SdfPrimitive::Sphere { center, radius } => {
                if radius <= 0.0 {
                    return f32::INFINITY;
                }
                point.sub(center).length() - radius
            }
            SdfPrimitive::Capsule { a, b, radius } => {
                if radius <= 0.0 {
                    return f32::INFINITY;
                }
                let closest = closest_point_on_segment(a, b, point);
                point.sub(closest).length() - radius
            }
            SdfPrimitive::HalfSpace { normal, offset } => {
                let len_sq = normal.length_squared();
                if len_sq <= EPS_LEN_SQ {
                    return f32::INFINITY;
                }
                let inv_len = 1.0 / len_sq.sqrt();
                point.dot(normal) * inv_len - offset
            }
            SdfPrimitive::Box {
                center,
                half_extents,
            } => box_signed_distance(center, half_extents, point),
        }
    }
}

/// Exact signed distance to an axis-aligned box (Quilez form): the length of
/// the positive part of `|p - center| - half_extents` (outside distance) plus
/// the largest interior component clamped to non-positive (inside distance).
fn box_signed_distance(center: Vec3, half_extents: Vec3, point: Vec3) -> f32 {
    if half_extents.x <= 0.0 || half_extents.y <= 0.0 || half_extents.z <= 0.0 {
        return f32::INFINITY;
    }
    let rel = point.sub(center);
    let dx = rel.x.abs() - half_extents.x;
    let dy = rel.y.abs() - half_extents.y;
    let dz = rel.z.abs() - half_extents.z;
    // Distance while outside: only positive components contribute.
    let ox = dx.max(0.0);
    let oy = dy.max(0.0);
    let oz = dz.max(0.0);
    let outside = (ox * ox + oy * oy + oz * oz).sqrt();
    // Distance while inside: negative, the least-penetrating face.
    let inside = dx.max(dy).max(dz).min(0.0);
    outside + inside
}

/// Returns the point on segment `a`..`b` closest to `point`; a zero-length
/// segment collapses to `a`, so a degenerate capsule behaves like a sphere.
fn closest_point_on_segment(a: Vec3, b: Vec3, point: Vec3) -> Vec3 {
    let ab = b.sub(a);
    let len_sq = ab.length_squared();
    if len_sq <= EPS_LEN_SQ {
        return a;
    }
    let t = (point.sub(a).dot(ab) / len_sq).clamp(0.0, 1.0);
    a.add(ab.scale(t))
}

/// Union signed distance: the minimum over all primitives (an empty union is
/// [`f32::INFINITY`], i.e. everywhere outside).
#[must_use]
pub fn union_signed_distance(primitives: &[SdfPrimitive], point: Vec3) -> f32 {
    let mut d = f32::INFINITY;
    for primitive in primitives {
        let di = primitive.signed_distance(point);
        if di < d {
            d = di;
        }
    }
    d
}

/// Outward unit gradient of the union field at `point`, by central differences.
///
/// Returns the zero vector when the field is locally flat (all six samples
/// equal, e.g. deep inside overlapping primitives), letting the caller fall
/// back to a fixed escape axis instead of dividing by zero.
fn union_gradient(primitives: &[SdfPrimitive], point: Vec3) -> Vec3 {
    let gx = union_signed_distance(primitives, Vec3::new(point.x + GRAD_STEP, point.y, point.z))
        - union_signed_distance(primitives, Vec3::new(point.x - GRAD_STEP, point.y, point.z));
    let gy = union_signed_distance(primitives, Vec3::new(point.x, point.y + GRAD_STEP, point.z))
        - union_signed_distance(primitives, Vec3::new(point.x, point.y - GRAD_STEP, point.z));
    let gz = union_signed_distance(primitives, Vec3::new(point.x, point.y, point.z + GRAD_STEP))
        - union_signed_distance(primitives, Vec3::new(point.x, point.y, point.z - GRAD_STEP));
    Vec3::new(gx, gy, gz).normalize_or_zero()
}

/// Pushes `point` out of the union field until it is outside every primitive or
/// `iterations` passes are spent, and returns the resolved position.
///
/// Each pass steps the point along the outward field gradient by the current
/// (negative) union distance, which lands it on the nearest zero isosurface;
/// repeating settles points caught in overlapping primitives. A point with no
/// defined gradient escapes straight up `+Y` by the penetration depth, a
/// deterministic fallback that never yields NaN.
#[must_use]
pub fn push_out_of_field(primitives: &[SdfPrimitive], point: Vec3, iterations: u32) -> Vec3 {
    let mut p = point;
    for _ in 0..iterations {
        let d = union_signed_distance(primitives, p);
        if d >= 0.0 || d.is_nan() {
            break;
        }
        let grad = union_gradient(primitives, p);
        if grad.length_squared() <= EPS_LEN_SQ {
            // Locally flat field: escape along a fixed axis by the depth.
            p = p.add(Vec3::new(0.0, -d, 0.0));
            break;
        }
        p = p.add(grad.scale(-d));
    }
    p
}

/// Projects every free particle out of the SDF union, in place.
///
/// Particles are visited in order; pinned particles (the skinned root) are
/// never moved. An empty `primitives` slice or zero `iterations` is a no-op, so
/// callers can disable SDF collision without a branch. Run this like the
/// analytic collision pass — after the constraint solve — as the tighter,
/// heavier tier on top of [`super::collision::resolve_strand_collisions`].
pub fn resolve_sdf_collisions(
    particles: &mut [StrandParticle],
    primitives: &[SdfPrimitive],
    iterations: u32,
) {
    if primitives.is_empty() || iterations == 0 {
        return;
    }
    for particle in particles.iter_mut() {
        if particle.is_pinned() {
            continue;
        }
        particle.position = push_out_of_field(primitives, particle.position, iterations);
    }
}

/// Convenience bucket for callers that want to carry an SDF collider alongside
/// its resolution budget as one owned value.
#[derive(Clone, Debug, Default)]
pub struct SdfCollider {
    /// The primitives whose union defines the body surface.
    pub primitives: Vec<SdfPrimitive>,
    /// Push-out relaxation passes per resolve; see [`DEFAULT_SDF_ITERATIONS`].
    pub iterations: u32,
}

impl SdfCollider {
    /// Builds a collider from a primitive vector with the default iteration
    /// count.
    #[must_use]
    pub fn new(primitives: Vec<SdfPrimitive>) -> Self {
        Self {
            primitives,
            iterations: DEFAULT_SDF_ITERATIONS,
        }
    }

    /// Resolves a particle array against this collider.
    pub fn resolve(&self, particles: &mut [StrandParticle]) {
        resolve_sdf_collisions(particles, &self.primitives, self.iterations);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn sphere_field_matches_analytic_distance() {
        let s = SdfPrimitive::Sphere {
            center: Vec3::ZERO,
            radius: 2.0,
        };
        // Inside: negative; on surface: zero; outside: positive.
        assert!((s.signed_distance(Vec3::ZERO) + 2.0).abs() < 1.0e-6);
        assert!(s.signed_distance(Vec3::new(2.0, 0.0, 0.0)).abs() < 1.0e-6);
        assert!((s.signed_distance(Vec3::new(3.0, 0.0, 0.0)) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn box_signed_distance_inside_and_outside() {
        let b = SdfPrimitive::Box {
            center: Vec3::ZERO,
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        };
        // Center is one unit deep on every face.
        assert!((b.signed_distance(Vec3::ZERO) + 1.0).abs() < 1.0e-6);
        // One unit outside the +X face.
        assert!((b.signed_distance(Vec3::new(2.0, 0.0, 0.0)) - 1.0).abs() < 1.0e-6);
        // Corner distance is the diagonal of the outside offset.
        let d = b.signed_distance(Vec3::new(2.0, 2.0, 0.0));
        assert!((d - 2.0_f32.sqrt()).abs() < 1.0e-5);
    }

    #[test]
    fn half_space_normalizes_normal() {
        // Non-unit normal along +Y; solid region is y < offset.
        let h = SdfPrimitive::HalfSpace {
            normal: Vec3::new(0.0, 5.0, 0.0),
            offset: 1.0,
        };
        assert!((h.signed_distance(Vec3::new(0.0, 0.0, 0.0)) + 1.0).abs() < 1.0e-6);
        assert!((h.signed_distance(Vec3::new(0.0, 1.0, 0.0))).abs() < 1.0e-6);
        assert!((h.signed_distance(Vec3::new(0.0, 4.0, 0.0)) - 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn degenerate_primitives_are_inert() {
        let cases = [
            SdfPrimitive::Sphere {
                center: Vec3::ZERO,
                radius: 0.0,
            },
            SdfPrimitive::Capsule {
                a: Vec3::ZERO,
                b: Vec3::new(1.0, 0.0, 0.0),
                radius: -1.0,
            },
            SdfPrimitive::HalfSpace {
                normal: Vec3::ZERO,
                offset: 0.0,
            },
            SdfPrimitive::Box {
                center: Vec3::ZERO,
                half_extents: Vec3::new(0.0, 1.0, 1.0),
            },
        ];
        for c in cases {
            assert!(c.signed_distance(Vec3::ZERO).is_infinite());
        }
    }

    #[test]
    fn push_out_sphere_lands_on_surface() {
        let prims = vec![SdfPrimitive::Sphere {
            center: Vec3::ZERO,
            radius: 2.0,
        }];
        let out = push_out_of_field(&prims, Vec3::new(0.0, 1.0, 0.0), DEFAULT_SDF_ITERATIONS);
        // Started inside on +Y; ends on the surface at radius 2.
        assert!((union_signed_distance(&prims, out)).abs() < 1.0e-3);
        assert!((out.y - 2.0).abs() < 1.0e-2);
    }

    #[test]
    fn push_out_box_leaves_field() {
        let prims = vec![SdfPrimitive::Box {
            center: Vec3::ZERO,
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        }];
        // A point just inside the +X face resolves to (about) the face.
        let out = push_out_of_field(&prims, Vec3::new(0.6, 0.0, 0.0), DEFAULT_SDF_ITERATIONS);
        assert!(union_signed_distance(&prims, out) >= -1.0e-3);
    }

    #[test]
    fn exterior_point_is_untouched() {
        let prims = vec![SdfPrimitive::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }];
        let p = Vec3::new(5.0, 0.0, 0.0);
        let out = push_out_of_field(&prims, p, DEFAULT_SDF_ITERATIONS);
        assert!(out.sub(p).length_squared() < 1.0e-12);
    }

    #[test]
    fn field_center_escapes_along_up_axis() {
        // A point at the exact center of a sphere has a symmetric field: the
        // numeric gradient cancels, so the fallback escapes straight up.
        let prims = vec![SdfPrimitive::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        }];
        let out = push_out_of_field(&prims, Vec3::ZERO, DEFAULT_SDF_ITERATIONS);
        assert!(out.x.abs() < 1.0e-6);
        assert!(out.z.abs() < 1.0e-6);
        assert!(out.y > 0.0);
    }

    #[test]
    fn pinned_particle_never_moves() {
        let prims = vec![SdfPrimitive::Sphere {
            center: Vec3::ZERO,
            radius: 2.0,
        }];
        let mut ps = [
            StrandParticle::pinned(Vec3::new(0.0, 0.5, 0.0)),
            StrandParticle::free(Vec3::new(0.0, 0.5, 0.0)),
        ];
        resolve_sdf_collisions(&mut ps, &prims, DEFAULT_SDF_ITERATIONS);
        // Pinned root stays inside; free particle is pushed toward the surface.
        assert!((ps[0].position.y - 0.5).abs() < 1.0e-6);
        assert!(ps[1].position.y > 0.5);
    }

    #[test]
    fn empty_or_zero_iteration_is_noop() {
        let prims = vec![SdfPrimitive::Sphere {
            center: Vec3::ZERO,
            radius: 2.0,
        }];
        let mut a = [StrandParticle::free(Vec3::new(0.0, 0.5, 0.0))];
        resolve_sdf_collisions(&mut a, &[], DEFAULT_SDF_ITERATIONS);
        assert!((a[0].position.y - 0.5).abs() < 1.0e-6);

        let mut b = [StrandParticle::free(Vec3::new(0.0, 0.5, 0.0))];
        resolve_sdf_collisions(&mut b, &prims, 0);
        assert!((b[0].position.y - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn union_takes_nearest_surface() {
        // Two spheres; union distance is the smaller (more-inside) of the two.
        let prims = vec![
            SdfPrimitive::Sphere {
                center: Vec3::new(-1.0, 0.0, 0.0),
                radius: 1.0,
            },
            SdfPrimitive::Sphere {
                center: Vec3::new(1.0, 0.0, 0.0),
                radius: 1.0,
            },
        ];
        // Midpoint sits on both surfaces (distance 0 to each) → union 0.
        assert!(union_signed_distance(&prims, Vec3::ZERO).abs() < 1.0e-6);
        // Deep in the left sphere: union distance is the left sphere's.
        let p = Vec3::new(-1.0, 0.0, 0.0);
        assert!((union_signed_distance(&prims, p) + 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn resolution_is_deterministic() {
        let prims = vec![
            SdfPrimitive::Sphere {
                center: Vec3::ZERO,
                radius: 1.0,
            },
            SdfPrimitive::Box {
                center: Vec3::new(0.5, 0.0, 0.0),
                half_extents: Vec3::new(1.0, 0.5, 0.5),
            },
        ];
        let mut a = [
            StrandParticle::free(Vec3::new(0.1, 0.2, 0.05)),
            StrandParticle::free(Vec3::new(-0.3, 0.0, 0.1)),
        ];
        let mut b = a;
        resolve_sdf_collisions(&mut a, &prims, DEFAULT_SDF_ITERATIONS);
        resolve_sdf_collisions(&mut b, &prims, DEFAULT_SDF_ITERATIONS);
        assert_eq!(a[0].position, b[0].position);
        assert_eq!(a[1].position, b[1].position);
    }

    #[test]
    fn collider_struct_resolves_like_free_function() {
        let prims = vec![SdfPrimitive::Sphere {
            center: Vec3::ZERO,
            radius: 2.0,
        }];
        let collider = SdfCollider::new(prims.clone());
        assert_eq!(collider.iterations, DEFAULT_SDF_ITERATIONS);
        let mut a = [StrandParticle::free(Vec3::new(0.0, 0.5, 0.0))];
        let mut b = a;
        collider.resolve(&mut a);
        resolve_sdf_collisions(&mut b, &prims, DEFAULT_SDF_ITERATIONS);
        assert_eq!(a[0].position, b[0].position);
    }
}
