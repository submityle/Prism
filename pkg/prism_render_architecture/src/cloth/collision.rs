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

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::{ClothParticle, Vec3, EPS_LEN_SQ};

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
    if radius <= 0.0 {
        return pos;
    }
    let delta = pos.sub(center);
    let dist_sq = delta.length_squared();
    if dist_sq >= radius * radius {
        return pos;
    }
    if dist_sq <= EPS_LEN_SQ {
        // Coincident with the center: pick a fixed axis for a deterministic,
        // non-`NaN` result.
        return center.add(Vec3::new(0.0, radius, 0.0));
    }
    let dir = delta.normalize_or_zero();
    center.add(dir.scale(radius))
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
    pos.add(normal.scale(t))
}

/// Returns the point on segment `p0`..`p1` closest to `pos`.
///
/// The projection parameter is clamped to `[0, 1]` so the result never leaves
/// the segment, which turns the capsule end-caps into hemispheres. A
/// zero-length segment (`p0 == p1`) degenerates gracefully to `p0`, so a
/// collapsed capsule behaves like a sphere.
#[must_use]
pub fn closest_point_on_segment(p0: Vec3, p1: Vec3, pos: Vec3) -> Vec3 {
    let axis = p1.sub(p0);
    let len_sq = axis.length_squared();
    if len_sq <= EPS_LEN_SQ {
        return p0;
    }
    let t = (pos.sub(p0).dot(axis) / len_sq).clamp(0.0, 1.0);
    p0.add(axis.scale(t))
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
    for particle in particles.iter_mut() {
        if particle.is_pinned() {
            continue;
        }
        for collider in colliders {
            particle.position = collider.project(particle.position);
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
    pub distance: f32,
}

/// Returns `pos` clamped to the front side of `backstop`.
///
/// When the signed distance `normal.dot(pos - origin)` drops below `-distance`
/// the point is pushed forward along the (unit) normal onto the limiting plane;
/// otherwise `pos` is returned unchanged. A (near) zero normal has no defined
/// plane, so the point is returned untouched rather than producing a `NaN`.
#[must_use]
pub fn apply_backstop(pos: Vec3, backstop: Backstop) -> Vec3 {
    let len_sq = backstop.normal.length_squared();
    if len_sq <= EPS_LEN_SQ {
        return pos;
    }
    let n = backstop.normal.normalize_or_zero();
    let s = n.dot(pos.sub(backstop.origin));
    let min_s = -backstop.distance;
    if s < min_s {
        pos.add(n.scale(min_s - s))
    } else {
        pos
    }
}

/// Applies each backstop to its matching particle, in place.
///
/// `backstops[i]` constrains `particles[i]`. The pass runs over the shorter of
/// the two lengths, so a short `backstops` slice simply leaves the trailing
/// particles unconstrained (and never panics). Pinned particles are skipped. An
/// empty `backstops` slice is a no-op.
pub fn resolve_backstops(particles: &mut [ClothParticle], backstops: &[Backstop]) {
    for (particle, backstop) in particles.iter_mut().zip(backstops.iter()) {
        if particle.is_pinned() {
            continue;
        }
        particle.position = apply_backstop(particle.position, *backstop);
    }
}

/// Maps a world-space position to its integer spatial-hash cell.
///
/// The float-to-int cast saturates, so a particle far from the origin yields a
/// saturated cell index rather than wrapping; it still buckets deterministically
/// and never panics. `cell_size` is assumed positive (the caller guards this).
#[must_use]
fn cell_of(pos: Vec3, cell_size: f32) -> (i32, i32, i32) {
    let inv = 1.0 / cell_size;
    let cx = (pos.x * inv).floor() as i32;
    let cy = (pos.y * inv).floor() as i32;
    let cz = (pos.z * inv).floor() as i32;
    (cx, cy, cz)
}

/// Resolves cloth self-collision with a deterministic uniform spatial hash.
///
/// Particles are bucketed into a [`BTreeMap`] keyed by integer cell so both the
/// cell traversal and (because indices are inserted in ascending order) the
/// per-bucket traversal are deterministic. For each particle only its 27-cell
/// neighborhood is examined, and each unordered pair is tested exactly once (by
/// requiring the neighbor index to exceed the current index), keeping the pass
/// near `O(n)` for well-distributed particles.
///
/// A pair closer than `thickness` is separated along the line joining them,
/// split by inverse mass: two equal free particles each move half the
/// penetration, while a free particle paired with a pinned one takes the whole
/// correction (the pinned particle never moves). Coincident particles are
/// separated along a fixed axis (`+X`) so the result stays deterministic and
/// free of `NaN`. Corrections are applied in place as they are found
/// (Gauss-Seidel style), which is deterministic given the fixed traversal
/// order.
///
/// A non-positive `cell_size` or `thickness`, or fewer than two particles, is a
/// no-op.
pub fn resolve_self_collision(particles: &mut [ClothParticle], cell_size: f32, thickness: f32) {
    if cell_size <= 0.0 || thickness <= 0.0 || particles.len() < 2 {
        return;
    }

    let mut grid: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for (index, particle) in particles.iter().enumerate() {
        let cell = cell_of(particle.position, cell_size);
        grid.entry(cell).or_default().push(index as u32);
    }

    let thickness_sq = thickness * thickness;
    for (&cell, bucket) in &grid {
        for &a in bucket {
            let ai = a as usize;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                        let Some(nbucket) = grid.get(&neighbor) else {
                            continue;
                        };
                        for &b in nbucket {
                            if b <= a {
                                continue;
                            }
                            let bi = b as usize;
                            resolve_pair(particles, ai, bi, thickness, thickness_sq);
                        }
                    }
                }
            }
        }
    }
}

/// Separates the particle pair `(ai, bi)` if they are closer than `thickness`.
///
/// The penetration is split by inverse mass so pinned partners stay put. When
/// the two positions coincide (no defined separating direction) a fixed `+X`
/// axis is used for determinism. Reads and writes go through distinct indices,
/// so there is no aliasing.
fn resolve_pair(
    particles: &mut [ClothParticle],
    ai: usize,
    bi: usize,
    thickness: f32,
    thickness_sq: f32,
) {
    let pa = particles[ai].position;
    let pb = particles[bi].position;
    let delta = pb.sub(pa);
    let dist_sq = delta.length_squared();
    if dist_sq >= thickness_sq {
        return;
    }

    let wa = particles[ai].inverse_mass.max(0.0);
    let wb = particles[bi].inverse_mass.max(0.0);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        // Both pinned: nothing can move.
        return;
    }

    let (dir, penetration) = if dist_sq <= EPS_LEN_SQ {
        // Coincident particles: separate along a fixed axis by the full
        // thickness so the result is deterministic and never `NaN`.
        (Vec3::new(1.0, 0.0, 0.0), thickness)
    } else {
        let dist = dist_sq.sqrt();
        (delta.scale(1.0 / dist), thickness - dist)
    };

    // `dir` points from `ai` toward `bi`; push them apart along it.
    let move_a = -penetration * (wa / w_sum);
    let move_b = penetration * (wb / w_sum);
    particles[ai].position = pa.add(dir.scale(move_a));
    particles[bi].position = pb.add(dir.scale(move_b));
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
}
