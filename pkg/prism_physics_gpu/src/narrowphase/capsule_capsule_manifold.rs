//! Capsule-capsule two-point contact manifold, shared bit-for-bit with the
//! `WGSL` kernel.
//!
//! The sibling [`capsule_capsule`](super::capsule_capsule) slice reports the
//! single closest-feature contact, which is enough to separate two penetrating
//! capsules but lets a capsule resting *alongside* a parallel neighbour rock and
//! roll about that one point. This module promotes that contact to a persistent
//! up-to-two-point manifold for the parallel case — the representation a solver
//! needs to hold two capsules flush side by side, exactly as
//! [`capsule_obb_manifold`](super::capsule_obb_manifold) does for the
//! capsule-on-box case. The `CPU` twin ([`cpu_capsule_capsule_manifold`]) and
//! the device kernel (`shaders/narrowphase_capsule_capsule_manifold.wgsl`) run
//! the identical arithmetic so their manifolds agree to within the
//! floating-point tolerance the parity test allows.
//!
//! # Geometry
//!
//! The manifold is built in three steps:
//!
//! 1. **Deepest contact.** Run the shared single-point test
//!    ([`capsule_capsule_contact`]). A [`None`] there means no penetration, so
//!    the manifold is [`None`] too. The returned [`Contact`] is both the source
//!    of the shared normal and the honest fallback when a second point cannot be
//!    found.
//! 2. **Parallel test.** Only near-parallel axes can share more than one contact
//!    point: two skew or crossing capsules touch along a single closest-point
//!    pair. The axes are parallel when the squared length of `cross(da, db)` is
//!    within [`PARALLEL_SIN2`] of zero relative to `|da|^2 |db|^2` (a
//!    `sin^2 theta` test that needs no normalisation). A capsule whose segment
//!    has (near) zero length is a sphere, which also carries a single point, so
//!    a degenerate axis falls back too.
//! 3. **Overlap clip.** Project both endpoints of capsule `b` onto capsule `a`'s
//!    axis line and intersect that interval with `a`'s own `[0, La]` span. The
//!    two ends of the overlapping interval are the manifold corners. Each end is
//!    lifted back to a point `pa` on `a`'s axis, the closest point `pb` on `b`'s
//!    segment is found, and the penetration is `depth = (ra + rb) - |pb - pa|`.
//!    Ends with a positive depth are live corners.
//!
//! When both ends are live and distinct, the manifold carries two points sharing
//! the single-contact normal. Otherwise — the axes are not parallel, one is a
//! sphere, the overlap is a sliver, or fewer than two ends penetrate — the
//! manifold honestly reports the single deepest contact (`count == 1`), never a
//! fabricated second point.
//!
//! # Normal convention
//!
//! The shared [`normal`](ContactManifold::normal) is the single-contact normal,
//! pointing from capsule `a` toward capsule `b` (the direction that separates
//! them), matching the sphere-sphere convention the single-point test uses. Each
//! corner's `position` sits on the plane midway through the overlap at that
//! cross-section, `pa + normal * (ra - depth / 2)`, exactly as the single-point
//! contact places its point.
//!
//! Provenance: textbook capsule-capsule manifold (segment-segment closest
//! feature plus parallel-overlap interval clipping); no Unreal Engine source or
//! derived code.

use glam::Vec3;

use super::capsule::Capsule;
use super::capsule_capsule::{capsule_capsule_contact, CapsuleCapsulePair};
use super::contact::Contact;
use super::manifold::{ContactManifold, ManifoldPoint};

/// Squared-length threshold below which a capsule segment is treated as a
/// degenerate point (a sphere), which carries only a single contact. Kept
/// identical to the `WGSL` constant so both paths fall back on the same inputs.
pub(crate) const SEG_EPS2: f32 = 1.0e-12;

/// `sin^2 theta` threshold below which two capsule axes count as parallel.
///
/// The test compares `|cross(da, db)|^2` against `PARALLEL_SIN2 * |da|^2 *
/// |db|^2`, so it needs no normalisation and no square root. The value is
/// `sin^2(2 degrees) ~= 0.00122`: axes within about two degrees of parallel
/// earn a two-point manifold, and anything more skewed falls back to the exact
/// single closest-point contact. Kept identical to the `WGSL` constant.
pub(crate) const PARALLEL_SIN2: f32 = 1.2e-3;

/// World-length threshold below which the overlapping stretch of the two axes is
/// treated as a single touch point, collapsing the manifold to the single
/// deepest contact. Kept identical to the `WGSL` constant.
pub(crate) const OVERLAP_EPS: f32 = 1.0e-6;

/// Squared-distance threshold below which two closest points count as
/// coincident, so the direction is taken from the shared single-contact normal
/// rather than a near-zero difference. Kept identical to the `WGSL` constant.
pub(crate) const COINCIDENT_EPS2: f32 = 1.0e-12;

/// Squared-distance threshold below which the two overlap-end corners are
/// treated as coincident, collapsing the manifold to a single point. Kept
/// identical to the `WGSL` constant.
pub(crate) const SEP_EPS2: f32 = 1.0e-12;

/// A live overlap-end corner: its world position and penetration depth.
///
/// Returned by [`overlap_end_point`]; a [`None`] marks an end whose cross
/// section does not actually penetrate, which forces the single-point fallback.
#[derive(Clone, Copy)]
struct EndPoint {
    /// World-space contact position on the mid-overlap plane.
    position: Vec3,
    /// Penetration depth along the shared normal; strictly positive when live.
    depth: f32,
}

/// Builds the manifold corner for the overlap end at axis parameter `s`.
///
/// `pa` is the point on capsule `a`'s axis at length `s` from `a0`; the closest
/// point `pb` on capsule `b`'s segment gives the cross-section distance and
/// hence the penetration `depth = sum_r - |pb - pa|`. Returns [`None`] when that
/// end does not penetrate (`depth <= 0`). The corner position is placed on the
/// plane midway through the overlap along `normal`, matching the single-point
/// contact convention.
#[must_use]
fn overlap_end_point(
    s: f32,
    a0: Vec3,
    ua: Vec3,
    b0: Vec3,
    ub: Vec3,
    lb: f32,
    ra: f32,
    sum_r: f32,
    normal: Vec3,
) -> Option<EndPoint> {
    let pa = a0 + ua * s;
    // Closest point on capsule b's segment to pa (ub is unit, so the projection
    // is a length clamped to [0, Lb]).
    let tb = (pa - b0).dot(ub).clamp(0.0, lb);
    let pb = b0 + ub * tb;
    let delta = pb - pa;
    let d2 = delta.dot(delta);
    let dist = if d2 <= COINCIDENT_EPS2 {
        0.0
    } else {
        d2.sqrt()
    };
    let depth = sum_r - dist;
    if depth <= 0.0 {
        return None;
    }
    let position = pa + normal * (ra - depth * 0.5);
    Some(EndPoint { position, depth })
}

/// Wraps a single [`Contact`] as a one-point [`ContactManifold`].
///
/// The honest fallback whenever a second contact point cannot be found: the
/// reported normal, position, and depth are exactly the single-point contact's.
#[must_use]
fn single_manifold(contact: &Contact) -> ContactManifold {
    ContactManifold::new(
        contact.a,
        contact.b,
        contact.normal,
        1,
        &[ManifoldPoint::new(contact.point, contact.depth)],
    )
}

/// Tests two capsules and, on penetration, builds their up-to-two-point
/// manifold.
///
/// Returns [`Some`] with the manifold for the pair `(a_id, b_id)` when the
/// capsules overlap, or [`None`] when they are separated or exactly touching.
/// Two near-parallel capsules that overlap along their axes report a two-point
/// manifold spanning the overlap; every other penetrating pair reports the
/// single deepest contact. The arithmetic mirrors
/// `narrowphase_capsule_capsule_manifold.wgsl` operation for operation.
#[must_use]
pub(crate) fn capsule_capsule_manifold_pair(
    a_id: u32,
    b_id: u32,
    a: &Capsule,
    b: &Capsule,
) -> Option<ContactManifold> {
    let contact = capsule_capsule_contact(a_id, b_id, a, b)?;

    let da = a.p1 - a.p0;
    let db = b.p1 - b.p0;
    let la2 = da.dot(da);
    let lb2 = db.dot(db);
    // A zero-length capsule is a sphere: one contact point only.
    if la2 <= SEG_EPS2 || lb2 <= SEG_EPS2 {
        return Some(single_manifold(&contact));
    }

    // Parallel test without normalisation: |cross|^2 <= sin2 * |da|^2 * |db|^2.
    let cross = da.cross(db);
    if cross.dot(cross) > PARALLEL_SIN2 * la2 * lb2 {
        return Some(single_manifold(&contact));
    }

    let la = la2.sqrt();
    let lb = lb2.sqrt();
    let ua = da / la;
    let ub = db / lb;

    // Project b's endpoints onto a's axis line (length from a0) and intersect
    // with a's own [0, La] span.
    let sb0 = (b.p0 - a.p0).dot(ua);
    let sb1 = (b.p1 - a.p0).dot(ua);
    let lo = sb0.min(sb1);
    let hi = sb0.max(sb1);
    let ov_lo = lo.max(0.0);
    let ov_hi = hi.min(la);
    if ov_hi - ov_lo <= OVERLAP_EPS {
        // The axes overlap in at most a single point: keep the deepest contact.
        return Some(single_manifold(&contact));
    }

    let ra = a.radius;
    let sum_r = a.radius + b.radius;
    let normal = contact.normal;
    let end0 = overlap_end_point(ov_lo, a.p0, ua, b.p0, ub, lb, ra, sum_r, normal);
    let end1 = overlap_end_point(ov_hi, a.p0, ua, b.p0, ub, lb, ra, sum_r, normal);

    match (end0, end1) {
        (Some(p0), Some(p1)) => {
            let sep = p1.position - p0.position;
            if sep.dot(sep) <= SEP_EPS2 {
                // The two ends collapsed onto one point: report it once.
                Some(single_manifold(&contact))
            } else {
                Some(ContactManifold::new(
                    a_id,
                    b_id,
                    normal,
                    2,
                    &[
                        ManifoldPoint::new(p0.position, p0.depth),
                        ManifoldPoint::new(p1.position, p1.depth),
                    ],
                ))
            }
        }
        // Fewer than two ends penetrate: the single deepest contact is honest.
        _ => Some(single_manifold(&contact)),
    }
}

/// Generates capsule-capsule manifolds for `pairs` over `capsules`.
///
/// Returns one slot per pair, in input order: [`Some`] carrying the manifold
/// when the two capsules penetrate, or [`None`] when they are separated or
/// exactly touching. Keeping a slot per pair (rather than compacting) aligns the
/// manifold index with the pair index for the device parity test.
///
/// # Panics
///
/// Panics if a pair references a capsule index outside `capsules`, which is
/// never valid output from the broad phase over the same capsule set.
#[must_use]
pub fn cpu_capsule_capsule_manifold(
    capsules: &[Capsule],
    pairs: &[CapsuleCapsulePair],
) -> Vec<Option<ContactManifold>> {
    pairs
        .iter()
        .map(|pair| {
            let a = &capsules[pair.a as usize];
            let b = &capsules[pair.b as usize];
            capsule_capsule_manifold_pair(pair.a, pair.b, a, b)
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
    fn parallel_side_by_side_reports_two_points() {
        // Two fully overlapping +x capsules offset 0.8 in y: sum_r 1, dist 0.8,
        // depth 0.2 at both ends, shared normal +y, corners on the mid plane.
        let capsules = [
            cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
            cap(Vec3::new(0.0, 0.8, 0.0), Vec3::new(2.0, 0.8, 0.0), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        let m = cpu_capsule_capsule_manifold(&capsules, &pairs)[0].expect("overlap");
        assert_eq!(m.count, 2);
        assert_eq!(m.a, 0);
        assert_eq!(m.b, 1);
        assert!(
            (m.normal - Vec3::Y).length() < 1.0e-6,
            "normal {:?}",
            m.normal
        );
        assert!(
            (m.points[0].depth - 0.2).abs() < 1.0e-5,
            "d0 {}",
            m.points[0].depth
        );
        assert!(
            (m.points[1].depth - 0.2).abs() < 1.0e-5,
            "d1 {}",
            m.points[1].depth
        );
        // Corners sit at the overlap ends x = 0 and x = 2, on the mid plane
        // y = 0.4 (ra - depth/2 = 0.5 - 0.1 above capsule a's axis).
        assert!((m.points[0].position - Vec3::new(0.0, 0.4, 0.0)).length() < 1.0e-5);
        assert!((m.points[1].position - Vec3::new(2.0, 0.4, 0.0)).length() < 1.0e-5);
    }

    #[test]
    fn partial_overlap_clips_to_the_shared_span() {
        // b starts halfway along a: the overlap is x in [1, 2], so the two
        // corners land at x = 1 and x = 2.
        let capsules = [
            cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
            cap(Vec3::new(1.0, 0.8, 0.0), Vec3::new(3.0, 0.8, 0.0), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        let m = cpu_capsule_capsule_manifold(&capsules, &pairs)[0].expect("overlap");
        assert_eq!(m.count, 2);
        assert!(
            (m.points[0].position.x - 1.0).abs() < 1.0e-5,
            "x0 {}",
            m.points[0].position.x
        );
        assert!(
            (m.points[1].position.x - 2.0).abs() < 1.0e-5,
            "x1 {}",
            m.points[1].position.x
        );
        assert!((m.points[0].depth - 0.2).abs() < 1.0e-5);
        assert!((m.points[1].depth - 0.2).abs() < 1.0e-5);
    }

    #[test]
    fn crossing_axes_fall_back_to_one_point() {
        // Perpendicular axes are not parallel, so only the single closest-point
        // contact is reported.
        let capsules = [
            cap(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5),
            cap(Vec3::new(0.0, -1.0, 0.3), Vec3::new(0.0, 1.0, 0.3), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        let m = cpu_capsule_capsule_manifold(&capsules, &pairs)[0].expect("overlap");
        assert_eq!(m.count, 1);
        // Closest points (0,0,0) and (0,0,0.3): dist 0.3, depth 0.7, normal +z.
        assert!(
            (m.normal - Vec3::Z).length() < 1.0e-6,
            "normal {:?}",
            m.normal
        );
        assert!(
            (m.points[0].depth - 0.7).abs() < 1.0e-5,
            "d {}",
            m.points[0].depth
        );
    }

    #[test]
    fn end_to_end_touch_falls_back_to_one_point() {
        // Parallel but abutting end to end: the axis overlap is a single point,
        // so the manifold collapses to the deepest contact.
        let capsules = [
            cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
            cap(Vec3::new(2.0, 0.8, 0.0), Vec3::new(4.0, 0.8, 0.0), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        let m = cpu_capsule_capsule_manifold(&capsules, &pairs)[0].expect("overlap");
        assert_eq!(m.count, 1);
        assert!(
            (m.points[0].depth - 0.2).abs() < 1.0e-5,
            "d {}",
            m.points[0].depth
        );
    }

    #[test]
    fn sphere_capsule_reports_one_point() {
        // A zero-length capsule is a sphere: a single contact regardless of the
        // neighbour's orientation.
        let capsules = [
            cap(Vec3::ZERO, Vec3::ZERO, 0.6),
            cap(Vec3::new(0.0, 0.8, 0.0), Vec3::new(2.0, 0.8, 0.0), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        let m = cpu_capsule_capsule_manifold(&capsules, &pairs)[0].expect("overlap");
        assert_eq!(m.count, 1);
        // sum_r 1.1, dist 0.8, depth 0.3, normal +y.
        assert!(
            (m.normal - Vec3::Y).length() < 1.0e-6,
            "normal {:?}",
            m.normal
        );
        assert!(
            (m.points[0].depth - 0.3).abs() < 1.0e-5,
            "d {}",
            m.points[0].depth
        );
    }

    #[test]
    fn separated_reports_none() {
        // Parallel but farther apart than sum_r: no contact at all.
        let capsules = [
            cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
            cap(Vec3::new(0.0, 2.0, 0.0), Vec3::new(2.0, 2.0, 0.0), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        assert!(cpu_capsule_capsule_manifold(&capsules, &pairs)[0].is_none());
    }

    #[test]
    fn antiparallel_axes_still_report_two_points() {
        // b runs in -x but shares a's line: the parallel test uses |cross|, so
        // the opposing direction still earns a two-point overlap manifold.
        let capsules = [
            cap(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.5),
            cap(Vec3::new(2.0, 0.8, 0.0), Vec3::new(0.0, 0.8, 0.0), 0.5),
        ];
        let pairs = [CapsuleCapsulePair::new(0, 1)];
        let m = cpu_capsule_capsule_manifold(&capsules, &pairs)[0].expect("overlap");
        assert_eq!(m.count, 2);
        assert!(
            (m.normal - Vec3::Y).length() < 1.0e-6,
            "normal {:?}",
            m.normal
        );
        assert!((m.points[0].depth - 0.2).abs() < 1.0e-5);
        assert!((m.points[1].depth - 0.2).abs() < 1.0e-5);
    }
}
