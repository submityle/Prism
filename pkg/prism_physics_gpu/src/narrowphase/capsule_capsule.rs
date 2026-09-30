//! Capsule-capsule contact geometry, shared bit-for-bit with the `WGSL` kernel.
//!
//! [`capsule_capsule_contact`] is the single source of truth for the collision
//! test and manifold construction; the `CPU` twin
//! ([`cpu_capsule_capsule_narrowphase`]) and the device kernel
//! (`shaders/narrowphase_capsule_capsule.wgsl`) both run this exact arithmetic
//! so their contacts agree to within the floating-point tolerance the parity
//! test allows (only the square root and the reciprocal in the normalisation
//! differ in their low bits).
//!
//! # Geometry
//!
//! Each capsule is a line segment `(p0, p1)` swept by a radius. The dynamic
//! pair test finds the closest point pair `(ca, cb)` between the two segments
//! with the standard clamped `ClosestPtSegmentSegment` routine (Ericson,
//! *Real-Time Collision Detection*), which handles parallel and degenerate
//! segments without ever dividing by zero, then collapses to a sphere-sphere
//! test at those points:
//!
//! ```text
//! delta = cb - ca
//! dist2 = dot(delta, delta)
//! sum_r = ra + rb
//! ```
//!
//! The capsules penetrate when `dist(ca, cb)` is strictly less than `ra + rb`.
//! The contact normal is the unit vector from capsule `a` toward capsule `b`
//! (`delta / dist`), matching the sphere-sphere convention, the penetration
//! depth is `(ra + rb) - dist`, and the contact point sits on the plane midway
//! through the overlap, `ca + normal * (ra - depth / 2)`.
//!
//! # Degenerate directions
//!
//! When the two closest points coincide (or fall within [`COINCIDENT_EPS2`],
//! e.g. intersecting axes) the direction is undefined, so the test falls back to
//! the `+x` axis and a full `ra + rb` depth. This is deterministic and identical
//! on both paths, so crossing capsule axes never produce a `NaN` normal.
//!
//! Provenance: textbook capsule-capsule (segment-segment) closest-feature
//! collision; no Unreal Engine source or derived code.

use glam::Vec3;

use super::capsule::Capsule;
use super::contact::Contact;

/// Squared-length threshold below which a capsule segment (or a solver
/// denominator) is treated as degenerate. Guards every division so a
/// zero-length capsule or a pair of parallel segments never divides by zero.
/// Kept identical to the `WGSL` constant so both paths branch on the same
/// inputs.
pub(crate) const SEG_EPS2: f32 = 1.0e-12;

/// Squared-distance threshold below which the two closest points are treated as
/// coincident. Redefined here (rather than imported) so it stays lock-step with
/// the identical `WGSL` constant in this feature's kernel.
pub(crate) const COINCIDENT_EPS2: f32 = 1.0e-12;

/// A candidate capsule-capsule pair: the two capsule indices.
///
/// The two indices address the same capsule slice; the pair is intrinsically
/// ordered (`a` is the `a` side of the contact, `b` the `b` side) and is not
/// canonicalised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapsuleCapsulePair {
    /// Index of the first capsule; becomes `Contact::a`.
    pub a: u32,
    /// Index of the second capsule; becomes `Contact::b`.
    pub b: u32,
}

impl CapsuleCapsulePair {
    /// Creates a capsule-capsule candidate pair.
    #[must_use]
    pub fn new(a: u32, b: u32) -> CapsuleCapsulePair {
        CapsuleCapsulePair { a, b }
    }
}

/// Closest point pair between segment `(p1, q1)` and segment `(p2, q2)`.
///
/// Returns `(ca, cb)`, the points on the first and second segment respectively
/// that minimise the distance between the segments. This is the clamped Ericson
/// `ClosestPtSegmentSegment` routine: every division is guarded by
/// [`SEG_EPS2`], so parallel segments (zero denominator) and degenerate
/// zero-length segments both take a stable fallback rather than dividing by
/// zero. Mirrors the `WGSL` `closest_pt_segment_segment` helper operation for
/// operation.
#[must_use]
fn closest_pt_segment_segment(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    let s: f32;
    let t: f32;
    if a <= SEG_EPS2 && e <= SEG_EPS2 {
        // Both segments degenerate to points.
        s = 0.0;
        t = 0.0;
    } else if a <= SEG_EPS2 {
        // First segment degenerate: project its point onto the second.
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= SEG_EPS2 {
            // Second segment degenerate: project its point onto the first.
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            // General case: solve the 2x2 system, then clamp t back into range
            // and recompute s for a clamped t.
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            // Parallel segments give a zero denominator; pick s = 0 and let the
            // t recomputation below place the closest point.
            let s0 = if denom > SEG_EPS2 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let t0 = (b * s0 + f) / e;
            if t0 < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t0 > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            } else {
                t = t0;
                s = s0;
            }
        }
    }

    let ca = p1 + d1 * s;
    let cb = p2 + d2 * t;
    (ca, cb)
}

/// Tests whether the two capsules penetrate and, if so, builds the contact.
///
/// Returns [`Some`] with the manifold for the pair `(a_id, b_id)` when the
/// capsules overlap, or [`None`] when they are separated or exactly touching.
/// The normal points from capsule `a` toward capsule `b`, matching the
/// sphere-sphere convention. The arithmetic mirrors
/// `narrowphase_capsule_capsule.wgsl` operation for operation.
#[must_use]
pub(crate) fn capsule_capsule_contact(
    a_id: u32,
    b_id: u32,
    ca_cap: &Capsule,
    cb_cap: &Capsule,
) -> Option<Contact> {
    let (ca, cb) = closest_pt_segment_segment(ca_cap.p0, ca_cap.p1, cb_cap.p0, cb_cap.p1);
    let delta = cb - ca;
    let dist2 = delta.dot(delta);
    let sum_r = ca_cap.radius + cb_cap.radius;
    // Strict overlap: an exactly-touching pair shares only a boundary point and
    // carries no penetration, so it is not a contact.
    if dist2 >= sum_r * sum_r {
        return None;
    }

    let (normal, dist) = if dist2 <= COINCIDENT_EPS2 {
        // Coincident closest points: choose a stable axis rather than dividing
        // by zero.
        (Vec3::X, 0.0)
    } else {
        let dist = dist2.sqrt();
        (delta / dist, dist)
    };

    let depth = sum_r - dist;
    // Contact point on the plane midway through the overlap, measured from the
    // first capsule's closest point along the (a -> b) normal.
    let point = ca + normal * (ca_cap.radius - depth * 0.5);
    Some(Contact::new(a_id, b_id, normal, depth, point))
}

/// Generates capsule-capsule contacts for `pairs` over `capsules`.
///
/// Returns one slot per pair, in input order: [`Some`] carrying the manifold
/// when the two capsules penetrate, or [`None`] when they are separated or
/// exactly touching. Keeping a slot per pair (rather than compacting) aligns the
/// contact index with the pair index for the device parity test.
///
/// # Panics
///
/// Panics if a pair references a capsule index outside `capsules`, which is
/// never valid output from the broad phase over the same capsule set.
#[must_use]
pub fn cpu_capsule_capsule_narrowphase(
    capsules: &[Capsule],
    pairs: &[CapsuleCapsulePair],
) -> Vec<Option<Contact>> {
    pairs
        .iter()
        .map(|pair| {
            let a = &capsules[pair.a as usize];
            let b = &capsules[pair.b as usize];
            capsule_capsule_contact(pair.a, pair.b, a, b)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A capsule from two endpoints and a radius.
    fn cap(p0: Vec3, p1: Vec3, radius: f32) -> Capsule {
        Capsule::new(p0, p1, radius)
    }

    #[test]
    fn parallel_flanks_build_the_expected_manifold() {
        // Two parallel +x segments offset 0.8 in y; closest points at the p0
        // ends, dist 0.8, sum_r 1, depth 0.2, normal +y.
        let capsules = [
            cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
            cap(Vec3::new(0.0, 0.8, 0.0), Vec3::new(2.0, 0.8, 0.0), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        let c = cpu_capsule_capsule_narrowphase(&capsules, &pairs)[0].expect("parallel overlap");
        assert_eq!(c.a, 0);
        assert_eq!(c.b, 1);
        assert!((c.normal - Vec3::Y).length() < 1.0e-6);
        assert!((c.depth - 0.2).abs() < 1.0e-6);
        // point = ca + normal * (ra - depth/2) = (0, 0, 0) + (0, 0.4, 0).
        assert!((c.point - Vec3::new(0.0, 0.4, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn crossing_axes_contact_at_the_mid_span() {
        // Segment a along x, segment b along y offset 0.6 in z; they cross over
        // the origin. Closest points (0,0,0) and (0,0,0.6): dist 0.6, depth 0.4,
        // normal +z.
        let capsules = [
            cap(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5),
            cap(Vec3::new(0.0, -1.0, 0.6), Vec3::new(0.0, 1.0, 0.6), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        let c = cpu_capsule_capsule_narrowphase(&capsules, &pairs)[0].expect("crossing overlap");
        assert!((c.normal - Vec3::Z).length() < 1.0e-6);
        assert!((c.depth - 0.4).abs() < 1.0e-6);
        assert!((c.point - Vec3::new(0.0, 0.0, 0.3)).length() < 1.0e-6);
    }

    #[test]
    fn endpoint_to_endpoint_contact_clamps_both_segments() {
        // Collinear +x segments with a 0.6 gap between a's far end and b's near
        // end; both parameters clamp to the touching ends. dist 0.6, depth 0.4,
        // normal +x.
        let capsules = [
            cap(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5),
            cap(Vec3::new(1.6, 0.0, 0.0), Vec3::new(2.6, 0.0, 0.0), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        let c = cpu_capsule_capsule_narrowphase(&capsules, &pairs)[0].expect("endpoint overlap");
        assert!((c.normal - Vec3::X).length() < 1.0e-6);
        assert!((c.depth - 0.4).abs() < 1.0e-6);
        // point = (1, 0, 0) + x * (0.5 - 0.2) = (1.3, 0, 0).
        assert!((c.point - Vec3::new(1.3, 0.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn clearly_separated_capsules_report_no_contact() {
        let capsules = [
            cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
            cap(Vec3::new(0.0, 5.0, 0.0), Vec3::new(2.0, 5.0, 0.0), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        assert_eq!(
            cpu_capsule_capsule_narrowphase(&capsules, &pairs),
            vec![None]
        );
    }

    #[test]
    fn exactly_touching_reports_no_contact() {
        // Parallel segments exactly sum_r == 1 apart: a shared boundary with no
        // penetration, rejected by the strict test.
        let capsules = [
            cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
            cap(Vec3::new(0.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.0), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        assert_eq!(
            cpu_capsule_capsule_narrowphase(&capsules, &pairs),
            vec![None]
        );
    }

    #[test]
    fn degenerate_zero_length_capsules_act_like_spheres() {
        // Both p0 == p1: the capsules collapse to spheres. Centres 0.6 apart,
        // sum_r 1, depth 0.4, normal +x.
        let capsules = [
            cap(Vec3::ZERO, Vec3::ZERO, 0.5),
            cap(Vec3::new(0.6, 0.0, 0.0), Vec3::new(0.6, 0.0, 0.0), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        let c = cpu_capsule_capsule_narrowphase(&capsules, &pairs)[0].expect("degenerate overlap");
        assert!((c.normal - Vec3::X).length() < 1.0e-6);
        assert!((c.depth - 0.4).abs() < 1.0e-6);
        assert!((c.point - Vec3::new(0.3, 0.0, 0.0)).length() < 1.0e-6);
        assert!(c.normal.is_finite());
    }

    #[test]
    fn coincident_segments_fall_back_to_a_stable_axis() {
        // Identical segments: the closest points coincide, so the direction is
        // undefined and the test uses +x with full sum_r depth instead of a NaN.
        let capsules = [
            cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
            cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        let c = cpu_capsule_capsule_narrowphase(&capsules, &pairs)[0].expect("coincident overlap");
        assert_eq!(c.normal, Vec3::X);
        assert!((c.depth - 1.0).abs() < 1.0e-6);
        // point = ca + x * (ra - depth/2) = (0,0,0) + x * (0.5 - 0.5) = origin.
        assert!((c.point - Vec3::ZERO).length() < 1.0e-6);
        assert!(c.normal.is_finite());
    }

    #[test]
    fn one_slot_per_pair_in_order() {
        // A hit then a miss; every input pair keeps its slot and its order.
        let capsules = [
            cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
            cap(Vec3::new(0.0, 0.8, 0.0), Vec3::new(2.0, 0.8, 0.0), 0.5),
            cap(Vec3::new(0.0, 20.0, 0.0), Vec3::new(2.0, 20.0, 0.0), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1), CapsuleCapsulePair::new(0, 2)];
        let contacts = cpu_capsule_capsule_narrowphase(&capsules, &pairs);
        assert_eq!(contacts.len(), 2);
        assert!(contacts[0].is_some(), "capsules 0 and 1 overlap");
        assert!(contacts[1].is_none(), "capsule 2 is far away");
    }

    #[test]
    fn empty_pairs_yield_no_contacts() {
        let capsules = [cap(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5)];
        assert!(cpu_capsule_capsule_narrowphase(&capsules, &[]).is_empty());
    }
}
