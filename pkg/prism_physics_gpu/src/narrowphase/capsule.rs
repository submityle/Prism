//! Sphere-capsule contact geometry, shared bit-for-bit with the `WGSL` kernel.
//!
//! [`sphere_capsule_contact`] is the single source of truth for the collision
//! test and manifold construction; the `CPU` twin ([`cpu_capsule_narrowphase`])
//! and the device kernel (`shaders/narrowphase_capsule.wgsl`) both run this
//! exact arithmetic so their contacts agree to within the floating-point
//! tolerance the parity test allows (only the square root and the reciprocal in
//! the normalisation differ in their low bits).
//!
//! # Geometry
//!
//! A capsule is a line segment `(p0, p1)` swept by a radius `rc`; a sphere is a
//! centre `c` with radius `rs`. The test collapses to a sphere-sphere test
//! against the closest point `q` on the segment to `c`:
//!
//! ```text
//! ab = p1 - p0
//! t  = clamp(dot(c - p0, ab) / dot(ab, ab), 0, 1)
//! q  = p0 + t * ab
//! ```
//!
//! When the segment is degenerate (`dot(ab, ab) <= SEG_EPS2`, a zero-length
//! capsule) the closest point is simply `p0`, so the division is never taken.
//! The sphere and capsule penetrate when `dist(c, q)` is strictly less than
//! `rs + rc`. The contact normal is the unit vector from the capsule toward the
//! sphere (`b` to `a`, the push-out direction), the penetration depth is
//! `(rs + rc) - dist`, and the contact point sits on the plane midway through
//! the overlap, `c - normal * (rs - depth / 2)`.
//!
//! # Degenerate directions
//!
//! When the sphere centre lies on the segment (or within [`COINCIDENT_EPS2`] of
//! it) the direction is undefined, so the test falls back to the `+x` axis and a
//! full `rs + rc` depth. This is deterministic and identical on both paths, so a
//! sphere sitting on the capsule axis never produces a `NaN` normal.
//!
//! Provenance: textbook sphere-capsule (segment-point) closest-feature
//! collision; no Unreal Engine source or derived code.

use glam::Vec3;

use super::contact::Contact;
use crate::broadphase::Particle;

/// Squared-length threshold below which a capsule segment is treated as a
/// single point (a degenerate zero-length capsule). Kept identical to the
/// `WGSL` constant so both paths take the point fallback on the same capsules.
pub(crate) const SEG_EPS2: f32 = 1.0e-12;

/// Squared-distance threshold below which the sphere centre is treated as lying
/// on the segment. Redefined here (rather than imported from `sphere.rs`) so it
/// stays lock-step with the identical `WGSL` constant in this feature's kernel.
pub(crate) const COINCIDENT_EPS2: f32 = 1.0e-12;

/// A capsule collision proxy: a line segment swept by a radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule {
    /// First segment endpoint of the capsule axis.
    pub p0: Vec3,
    /// Second segment endpoint of the capsule axis.
    pub p1: Vec3,
    /// Swept radius around the segment; must be non-negative.
    pub radius: f32,
}

impl Capsule {
    /// Creates a capsule from its two axis endpoints and swept radius.
    #[must_use]
    pub fn new(p0: Vec3, p1: Vec3, radius: f32) -> Capsule {
        Capsule { p0, p1, radius }
    }
}

/// A candidate sphere-capsule pair: the sphere index and the capsule index.
///
/// Unlike a sphere-sphere [`CandidatePair`](crate::broadphase::CandidatePair)
/// the two indices address different arrays, so the pair is intrinsically
/// ordered (`sphere` is always the `a` side, `capsule` the `b` side) and is not
/// canonicalised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SphereCapsulePair {
    /// Index into the sphere (particle) slice; becomes `Contact::a`.
    pub sphere: u32,
    /// Index into the capsule slice; becomes `Contact::b`.
    pub capsule: u32,
}

impl SphereCapsulePair {
    /// Creates a sphere-capsule candidate pair.
    #[must_use]
    pub fn new(sphere: u32, capsule: u32) -> SphereCapsulePair {
        SphereCapsulePair { sphere, capsule }
    }
}

/// Closest point on the capsule segment to the sphere centre `c`.
///
/// Returns `cap.p0` for a degenerate (`<= SEG_EPS2`) zero-length segment,
/// otherwise the point `p0 + t * ab` with `t` clamped to `[0, 1]`. Mirrors the
/// `WGSL` `closest_on_segment` helper operation for operation.
#[must_use]
fn closest_on_segment(c: Vec3, cap: &Capsule) -> Vec3 {
    let ab = cap.p1 - cap.p0;
    let ab_len2 = ab.dot(ab);
    if ab_len2 <= SEG_EPS2 {
        // Degenerate capsule: the "segment" is a single point, so the closest
        // point is that point and no division is taken.
        cap.p0
    } else {
        let t = ((c - cap.p0).dot(ab) / ab_len2).clamp(0.0, 1.0);
        cap.p0 + ab * t
    }
}

/// Tests whether the sphere and capsule penetrate and, if so, builds the contact.
///
/// Returns [`Some`] with the manifold for the pair `(sphere_id, capsule_id)`
/// when they overlap, or [`None`] when they are separated or exactly touching.
/// The normal points from the capsule (`b`) toward the sphere (`a`), the
/// push-out direction a solver applies. The arithmetic mirrors
/// `narrowphase_capsule.wgsl` operation for operation.
#[must_use]
pub(crate) fn sphere_capsule_contact(
    sphere_id: u32,
    capsule_id: u32,
    c: Vec3,
    rs: f32,
    cap: &Capsule,
) -> Option<Contact> {
    let q = closest_on_segment(c, cap);
    let delta = c - q;
    let dist2 = delta.dot(delta);
    let sum_r = rs + cap.radius;
    // Strict overlap: an exactly-touching pair shares only a boundary point and
    // carries no penetration, so it is not a contact.
    if dist2 >= sum_r * sum_r {
        return None;
    }

    let (normal, dist) = if dist2 <= COINCIDENT_EPS2 {
        // Sphere centre on the segment: choose a stable axis rather than
        // dividing by zero.
        (Vec3::X, 0.0)
    } else {
        let dist = dist2.sqrt();
        (delta / dist, dist)
    };

    let depth = sum_r - dist;
    // Contact point on the plane midway through the overlap, measured back from
    // the sphere centre along the (capsule -> sphere) normal.
    let point = c - normal * (rs - depth * 0.5);
    Some(Contact::new(sphere_id, capsule_id, normal, depth, point))
}

/// Generates sphere-capsule contacts for `pairs` over `spheres` and `capsules`.
///
/// Returns one slot per pair, in input order: [`Some`] carrying the manifold
/// when the sphere and capsule penetrate, or [`None`] when they are separated or
/// exactly touching. Keeping a slot per pair (rather than compacting) aligns the
/// contact index with the pair index for the device parity test.
///
/// # Panics
///
/// Panics if a pair references a sphere or capsule index outside its slice,
/// which is never valid output from the broad phase over the same input set.
#[must_use]
pub fn cpu_capsule_narrowphase(
    spheres: &[Particle],
    capsules: &[Capsule],
    pairs: &[SphereCapsulePair],
) -> Vec<Option<Contact>> {
    pairs
        .iter()
        .map(|pair| {
            let s = &spheres[pair.sphere as usize];
            let cap = &capsules[pair.capsule as usize];
            sphere_capsule_contact(pair.sphere, pair.capsule, s.position, s.radius, cap)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sphere proxy at `(x, y, z)` with radius `r`.
    fn sphere(x: f32, y: f32, z: f32, r: f32) -> Particle {
        Particle::new(Vec3::new(x, y, z), r)
    }

    #[test]
    fn separated_sphere_and_capsule_report_no_contact() {
        // Capsule along +x, rc 0.5; sphere 3 above the axis midpoint, rs 0.5:
        // closest point is (1, 0, 0), gap 3 >> sum_r 1.
        let caps = [Capsule::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5)];
        let spheres = [sphere(1.0, 3.0, 0.0, 0.5)];
        let pairs = [SphereCapsulePair::new(0, 0)];
        assert_eq!(cpu_capsule_narrowphase(&spheres, &caps, &pairs), vec![None]);
    }

    #[test]
    fn exactly_touching_reports_no_contact() {
        // dist(c, segment) == sum_r == 1 exactly: a shared boundary point with
        // no penetration, rejected by the strict test.
        let caps = [Capsule::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5)];
        let spheres = [sphere(1.0, 1.0, 0.0, 0.5)];
        let pairs = [SphereCapsulePair::new(0, 0)];
        assert_eq!(cpu_capsule_narrowphase(&spheres, &caps, &pairs), vec![None]);
    }

    #[test]
    fn side_overlap_builds_the_expected_manifold() {
        // Sphere alongside the capsule flank, closest point (1, 0, 0).
        // dist 0.8, sum_r 1, depth 0.2, normal +y.
        let caps = [Capsule::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5)];
        let spheres = [sphere(1.0, 0.8, 0.0, 0.5)];
        let pairs = [SphereCapsulePair::new(0, 0)];
        let c = cpu_capsule_narrowphase(&spheres, &caps, &pairs)[0].expect("side overlap");
        assert_eq!(c.a, 0);
        assert_eq!(c.b, 0);
        assert!((c.normal - Vec3::Y).length() < 1.0e-6);
        assert!((c.depth - 0.2).abs() < 1.0e-6);
        // point = c - normal * (rs - depth/2) = (1, 0.8, 0) - (0, 0.4, 0).
        assert!((c.point - Vec3::new(1.0, 0.4, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn endpoint_cap_overlap_clamps_to_the_end() {
        // Sphere beyond p1: t clamps to 1, closest point (2, 0, 0). dist 0.7,
        // sum_r 1, depth 0.3, normal +x. This exercises the spherical end cap.
        let caps = [Capsule::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5)];
        let spheres = [sphere(2.7, 0.0, 0.0, 0.5)];
        let pairs = [SphereCapsulePair::new(0, 0)];
        let c = cpu_capsule_narrowphase(&spheres, &caps, &pairs)[0].expect("cap overlap");
        assert!((c.normal - Vec3::X).length() < 1.0e-6);
        assert!((c.depth - 0.3).abs() < 1.0e-6);
        // point = (2.7, 0, 0) - (0.5 - 0.15) * x = (2.35, 0, 0).
        assert!((c.point - Vec3::new(2.35, 0.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn degenerate_zero_length_capsule_acts_like_a_sphere() {
        // p0 == p1: the capsule collapses to a sphere of radius rc at p0.
        // Sphere at (0.6, 0, 0), dist 0.6, sum_r 1, depth 0.4, normal +x.
        let caps = [Capsule::new(Vec3::ZERO, Vec3::ZERO, 0.5)];
        let spheres = [sphere(0.6, 0.0, 0.0, 0.5)];
        let pairs = [SphereCapsulePair::new(0, 0)];
        let c = cpu_capsule_narrowphase(&spheres, &caps, &pairs)[0].expect("degenerate overlap");
        assert!((c.normal - Vec3::X).length() < 1.0e-6);
        assert!((c.depth - 0.4).abs() < 1.0e-6);
        assert!((c.point - Vec3::new(0.3, 0.0, 0.0)).length() < 1.0e-6);
        assert!(c.normal.is_finite());
    }

    #[test]
    fn sphere_centre_on_axis_falls_back_to_a_stable_axis() {
        // Sphere centre exactly on the segment: direction is undefined, so the
        // test uses +x and the full sum_r depth rather than a NaN normal.
        let caps = [Capsule::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5)];
        let spheres = [sphere(1.0, 0.0, 0.0, 0.5)];
        let pairs = [SphereCapsulePair::new(0, 0)];
        let c = cpu_capsule_narrowphase(&spheres, &caps, &pairs)[0].expect("coincident overlap");
        assert_eq!(c.normal, Vec3::X);
        assert!((c.depth - 1.0).abs() < 1.0e-6);
        // point = c - x * (rs - depth/2) = (1, 0, 0) - x * (0.5 - 0.5) = c.
        assert!((c.point - Vec3::new(1.0, 0.0, 0.0)).length() < 1.0e-6);
        assert!(c.normal.is_finite());
    }

    #[test]
    fn one_slot_per_pair_in_order() {
        // Mix of a hit and a miss over two capsules; every input pair keeps its
        // slot and its order.
        let caps = [
            Capsule::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
            Capsule::new(Vec3::new(0.0, 10.0, 0.0), Vec3::new(2.0, 10.0, 0.0), 0.5),
        ];
        let spheres = [sphere(1.0, 0.8, 0.0, 0.5)];
        let pairs = [SphereCapsulePair::new(0, 0), SphereCapsulePair::new(0, 1)];
        let contacts = cpu_capsule_narrowphase(&spheres, &caps, &pairs);
        assert_eq!(contacts.len(), 2);
        assert!(contacts[0].is_some(), "sphere overlaps capsule 0");
        assert!(contacts[1].is_none(), "sphere far from capsule 1");
    }

    #[test]
    fn empty_pairs_yield_no_contacts() {
        let caps = [Capsule::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5)];
        let spheres = [sphere(0.0, 0.0, 0.0, 0.5)];
        assert!(cpu_capsule_narrowphase(&spheres, &caps, &[]).is_empty());
    }
}
