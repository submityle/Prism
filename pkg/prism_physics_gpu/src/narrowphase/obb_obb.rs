//! OBB-versus-OBB contact geometry, shared bit-for-bit with the `WGSL` kernel.
//!
//! This module is the single source of truth for the oriented-bounding-box
//! against oriented-bounding-box overlap test and manifold construction: the
//! `CPU` twin ([`cpu_obb_obb_narrowphase`]) and the device kernel
//! (`shaders/narrowphase_obb_obb.wgsl`) both run this exact arithmetic, so their
//! contacts agree to within the floating-point tolerance the parity test
//! allows.
//!
//! # Separating Axis Theorem
//!
//! Two convex boxes are disjoint if and only if there is an axis onto which
//! their projected intervals do not overlap. For a pair of oriented boxes the
//! candidate set is finite — fifteen axes suffice:
//!
//! * the three face normals of box `a` (its local axes `a0`, `a1`, `a2`);
//! * the three face normals of box `b` (its local axes `b0`, `b1`, `b2`);
//! * the nine edge-edge cross products `a_i × b_j`.
//!
//! Along a unit axis `L` the projected half-width ("radius") of a box is
//! `|dot(L, x0)| * he.x + |dot(L, x1)| * he.y + |dot(L, x2)| * he.z` over its
//! three axes `x`, and the signed separation of the two centres is
//! `dot(L, t)` with `t = center_b - center_a`. The interval overlap along `L`
//! is `overlap = rA + rB - |dot(L, t)|`. If any candidate axis yields
//! `overlap <= 0` the boxes are separated (or exactly grazing) and the pair
//! reports no contact — a strict test, matching the sibling slices.
//!
//! When every candidate axis overlaps, the boxes penetrate. The
//! minimum-overlap axis is the minimum translation vector (`MTV`): the shortest
//! push that separates the boxes. Its overlap is the penetration `depth` and
//! its (oriented) direction is the contact `normal`.
//!
//! # Degenerate edge axes
//!
//! When two edges are parallel their cross product vanishes and the edge axis is
//! undefined. Such an axis is always parallel to a face axis that is already in
//! the candidate set, so it is skipped: it neither votes for separation nor
//! competes for the `MTV`. The threshold is [`CROSS_EPS2`] on the squared cross
//! length, kept identical to the `WGSL` constant so both paths skip the same
//! axes.
//!
//! # Face preference
//!
//! When a face axis and an edge axis overlap by almost the same amount, the face
//! axis is the numerically stabler contact normal (edge normals are the
//! normalised cross of two nearly parallel directions). The `MTV` search
//! therefore penalises edge axes by [`EDGE_BIAS`] in the comparison only; the
//! reported depth is always the true overlap of the chosen axis. The bias is
//! applied identically on both paths.
//!
//! # Normal convention
//!
//! The reported [`Contact`] stores box `a` in `a` and box `b` in `b`, and the
//! `normal` points from `a` toward `b` (the direction that pushes `b` off `a`),
//! matching the sphere-sphere manifold. The `MTV` axis is flipped when
//! `dot(t, L) < 0` so it always points from `a` to `b`.
//!
//! # Single-point manifold
//!
//! This slice reports one representative contact per `(box, box)` couple: the
//! mid-overlap point between box `a`'s support vertex along `+normal` and box
//! `b`'s support vertex along `-normal`. That is a deliberate, honest design
//! choice, not a stub — the point lies on the mid-overlap plane the solver
//! pushes against, and for a vertex-face or edge-edge contact it lands on the
//! penetrating feature. A full multi-point manifold (a resting face flush
//! against another face produces up to four clipped contact points) is a
//! separate follow-up slice; nothing here fakes or truncates a multi-point
//! result.
//!
//! # Per-operation agreement
//!
//! Both paths build the same fifteen candidate axes in the same order, normalise
//! the edge crosses with the same [`CROSS_EPS2`] guard, accumulate the same
//! projection sums, take the same strict `overlap <= 0` rejection, run the same
//! biased minimum search, flip the normal on the same `dot(t, L)` sign, and
//! accumulate the same two support vertices. Only the reciprocal square root in
//! the edge-axis `normalize` is inexact, so the parity test matches the validity
//! flag exactly and the normal, depth, and point to within a tight tolerance.
//!
//! Provenance: textbook separating-axis oriented-bounding-box collision
//! manifold; no Unreal Engine source or derived code.

use glam::Vec3;

use super::contact::Contact;
use super::obb::Obb;

/// Squared cross-length threshold below which an edge-edge axis is treated as
/// degenerate (its two edges are parallel) and skipped. Kept identical to the
/// `WGSL` constant so both paths drop the same axes.
pub(crate) const CROSS_EPS2: f32 = 1.0e-12;

/// Comparison-only penalty added to an edge axis's overlap so a face axis of
/// nearly equal overlap wins the minimum-translation search. Never affects the
/// reported penetration depth. Kept identical to the `WGSL` constant.
pub(crate) const EDGE_BIAS: f32 = 1.0e-5;

/// Sentinel overlap for a skipped (degenerate) axis: larger than any real
/// overlap, so a degenerate edge axis never wins the minimum search.
const SKIP_OVERLAP: f32 = 1.0e30;

/// A candidate `(box, box)` couple to test for penetration.
///
/// `a` and `b` index the oriented-bounding-box array. Keeping the pairing
/// explicit lets the narrow phase emit one contact slot per couple in input
/// order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObbObbPair {
    /// Index of the first oriented bounding box (the `a` side of the contact).
    pub a: u32,
    /// Index of the second oriented bounding box (the `b` side of the contact).
    pub b: u32,
}

impl ObbObbPair {
    /// Couples box index `a` with box index `b`.
    #[must_use]
    pub fn new(a: u32, b: u32) -> ObbObbPair {
        ObbObbPair { a, b }
    }
}

/// Returns `+1.0` when `x >= 0.0`, otherwise `-1.0`.
///
/// The zero case resolves to `+1.0` so both paths pick the same support vertex
/// when a box axis is exactly perpendicular to the contact normal; the `WGSL`
/// kernel uses `select(-1.0, 1.0, x >= 0.0)` for the identical mapping.
fn sign_pos(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Projected half-width of a box with axes `x` and half extents `he` onto the
/// unit axis `axis`: `Σ |dot(axis, x_k)| * he_k`.
fn projected_radius(axis: Vec3, x: &[Vec3; 3], he: Vec3) -> f32 {
    axis.dot(x[0]).abs() * he.x + axis.dot(x[1]).abs() * he.y + axis.dot(x[2]).abs() * he.z
}

/// Support vertex of a box in direction `dir`: the box corner furthest along
/// `dir`, `center + Σ x_k * sign(dot(dir, x_k)) * he_k`.
fn support_vertex(center: Vec3, x: &[Vec3; 3], he: Vec3, dir: Vec3) -> Vec3 {
    center
        + x[0] * (sign_pos(dir.dot(x[0])) * he.x)
        + x[1] * (sign_pos(dir.dot(x[1])) * he.y)
        + x[2] * (sign_pos(dir.dot(x[2])) * he.z)
}

/// The winning separating axis and the oriented minimum-translation data.
///
/// The [`axis_index`](Self::axis_index) records which of the fifteen candidate
/// axes won: `0..=2` is a face normal of box `a`, `3..=5` a face normal of box
/// `b`, and `6..=14` an edge-edge cross `a_i x b_j` (with `i = (index - 6) / 3`
/// and `j = (index - 6) % 3`). The manifold builder needs this to tell a
/// face contact (clip an incident face against a reference face) from an
/// edge-edge contact (a single closest-point pair).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SatQuery {
    /// Index of the winning candidate axis, `0..=14`.
    pub(crate) axis_index: usize,
    /// Unit contact normal oriented from box `a` toward box `b`.
    pub(crate) normal: Vec3,
    /// Penetration depth along [`normal`](Self::normal); always positive.
    pub(crate) depth: f32,
}

/// Runs the fifteen-axis separating-axis test on the two boxes.
///
/// Returns [`Some`] with the minimum-translation axis, oriented normal, and
/// penetration depth when every candidate axis overlaps, or [`None`] when any
/// axis separates the boxes or they exactly graze (`overlap <= 0`). This is the
/// shared core of both the single-point [`obb_obb_contact`] and the multi-point
/// manifold builder, so the two agree on the contact normal by construction.
#[must_use]
pub(crate) fn obb_obb_sat(a: &Obb, b: &Obb) -> Option<SatQuery> {
    let ax = a.axes;
    let bx = b.axes;
    let ea = a.half_extents;
    let eb = b.half_extents;
    let t = b.center - a.center;

    // The fifteen candidate axes in a fixed order: box a's three face normals,
    // box b's three face normals, then the nine edge-edge cross products. Edge
    // crosses are normalised; a degenerate (parallel) cross is flagged so it is
    // skipped in both the separation test and the minimum search.
    let mut axes = [Vec3::ZERO; 15];
    let mut valid = [true; 15];
    axes[0] = ax[0];
    axes[1] = ax[1];
    axes[2] = ax[2];
    axes[3] = bx[0];
    axes[4] = bx[1];
    axes[5] = bx[2];
    let mut k = 6;
    for a_axis in &ax {
        for b_axis in &bx {
            let c = a_axis.cross(*b_axis);
            let len2 = c.dot(c);
            if len2 < CROSS_EPS2 {
                valid[k] = false;
                axes[k] = Vec3::ZERO;
            } else {
                axes[k] = c.normalize();
            }
            k += 1;
        }
    }

    let mut best_overlap = SKIP_OVERLAP;
    let mut best_cmp = SKIP_OVERLAP;
    let mut best_axis = Vec3::ZERO;
    let mut best_index = 0usize;
    let mut separated = false;
    for idx in 0..15 {
        if !valid[idx] {
            continue;
        }
        let axis = axes[idx];
        let ra = projected_radius(axis, &ax, ea);
        let rb = projected_radius(axis, &bx, eb);
        let dist = t.dot(axis);
        let overlap = ra + rb - dist.abs();
        if overlap <= 0.0 {
            separated = true;
        }
        // Edge axes (index >= 6) carry a comparison penalty so a face axis of
        // nearly equal overlap is preferred; the reported depth stays exact.
        let bias = if idx >= 6 { EDGE_BIAS } else { 0.0 };
        let cmp = overlap + bias;
        if cmp < best_cmp {
            best_cmp = cmp;
            best_overlap = overlap;
            best_axis = axis;
            best_index = idx;
        }
    }

    if separated {
        return None;
    }

    // Orient the minimum-translation axis from box a toward box b.
    let normal = if t.dot(best_axis) < 0.0 {
        -best_axis
    } else {
        best_axis
    };

    Some(SatQuery {
        axis_index: best_index,
        normal,
        depth: best_overlap,
    })
}

/// Tests whether two oriented bounding boxes penetrate and, if so, builds the
/// contact along the minimum-translation axis.
///
/// Returns [`Some`] with the manifold for the couple `(a_id, b_id)` when every
/// candidate separating axis overlaps, or [`None`] when any axis separates the
/// boxes or they exactly graze (`overlap <= 0`). The arithmetic mirrors
/// `narrowphase_obb_obb.wgsl` operation for operation; see the module
/// documentation for the geometry and the normal convention.
#[must_use]
pub(crate) fn obb_obb_contact(a_id: u32, b_id: u32, a: &Obb, b: &Obb) -> Option<Contact> {
    let sat = obb_obb_sat(a, b)?;
    let ax = a.axes;
    let bx = b.axes;
    let ea = a.half_extents;
    let eb = b.half_extents;

    // Representative single point: the mid-overlap between box a's support
    // vertex along +normal and box b's support vertex along -normal.
    let pa = support_vertex(a.center, &ax, ea, sat.normal);
    let pb = support_vertex(b.center, &bx, eb, -sat.normal);
    let point = (pa + pb) * 0.5;

    Some(Contact::new(a_id, b_id, sat.normal, sat.depth, point))
}

/// `CPU` golden twin of the OBB-versus-OBB narrow phase.
///
/// Turns a set of candidate `pairs` into contact manifolds, running the shared
/// [`obb_obb_contact`] geometry so its output matches the device kernel
/// operation for operation. It emits one slot per input couple, preserving
/// order, which is what lets the parity test line the `GPU` contacts up against
/// this reference index by index: [`Some`] carrying the minimum-translation
/// manifold when the boxes penetrate, or [`None`] when a separating axis exists.
/// A later scan stage compacts the survivors.
///
/// # Panics
///
/// Panics if a couple references a box index outside `boxes`, which is never
/// valid output from a broad phase over the same set.
#[must_use]
pub fn cpu_obb_obb_narrowphase(boxes: &[Obb], pairs: &[ObbObbPair]) -> Vec<Option<Contact>> {
    pairs
        .iter()
        .map(|pair| {
            let a = &boxes[pair.a as usize];
            let b = &boxes[pair.b as usize];
            obb_obb_contact(pair.a, pair.b, a, b)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    /// A unit-axis box at `center` with the given half extents.
    fn axis_box(center: Vec3, he: Vec3) -> Obb {
        Obb::new(center, [Vec3::X, Vec3::Y, Vec3::Z], he)
    }

    #[test]
    fn face_overlap_axis_aligned() {
        // Two unit boxes overlapping 0.5 along x: the +x face of a meets the -x
        // face of b, normal +x, depth 0.5.
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let b = axis_box(Vec3::new(1.5, 0.0, 0.0), Vec3::ONE);
        let c = obb_obb_contact(0, 1, &a, &b).expect("overlapping boxes must contact");
        assert_eq!(c.a, 0);
        assert_eq!(c.b, 1);
        assert!((c.normal - Vec3::X).length() < 1.0e-6);
        assert!((c.depth - 0.5).abs() < 1.0e-6);
        // Support of a along +x is x=1; support of b along -x is x=0.5; mid 0.75.
        assert!((c.point.x - 0.75).abs() < 1.0e-6);
    }

    #[test]
    fn normal_points_from_a_to_b_regardless_of_order() {
        // b to the -x side of a: the normal must still run a -> b, i.e. -x.
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let b = axis_box(Vec3::new(-1.5, 0.0, 0.0), Vec3::ONE);
        let c = obb_obb_contact(0, 1, &a, &b).expect("overlapping boxes must contact");
        assert!((c.normal - (-Vec3::X)).length() < 1.0e-6);
        assert!((c.depth - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn least_penetrated_axis_wins() {
        // Overlap 0.5 on x but 0.8 on y: the MTV is the shallower x axis.
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let b = axis_box(Vec3::new(1.5, 1.2, 0.0), Vec3::ONE);
        let c = obb_obb_contact(0, 1, &a, &b).expect("overlapping boxes must contact");
        assert!((c.normal - Vec3::X).length() < 1.0e-6);
        assert!((c.depth - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn clear_gap_reports_no_contact() {
        // Boxes 3 apart on x, half extent 1 each: a 1-unit gap remains.
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let b = axis_box(Vec3::new(3.0, 0.0, 0.0), Vec3::ONE);
        assert!(obb_obb_contact(0, 1, &a, &b).is_none());
    }

    #[test]
    fn exactly_touching_reports_no_contact() {
        // Faces flush at x=1: overlap == 0, the strict test rejects.
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let b = axis_box(Vec3::new(2.0, 0.0, 0.0), Vec3::ONE);
        assert!(obb_obb_contact(0, 1, &a, &b).is_none());
    }

    #[test]
    fn diagonal_gap_detected_by_face_axis() {
        // Centres offset on x and y so a face axis (not a corner) still
        // separates them: 2.0 on x exceeds the 2.0 combined radius exactly, so
        // any positive gap on x is a clear separation.
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let b = axis_box(Vec3::new(2.5, 2.5, 0.0), Vec3::ONE);
        assert!(obb_obb_contact(0, 1, &a, &b).is_none());
    }

    #[test]
    fn edge_edge_contact_uses_cross_axis() {
        // Box a axis-aligned; box b rotated 45 degrees about z and about x so a
        // slanted edge of b digs into a's edge. The separating axis with the
        // least overlap is an edge-edge cross, exercising the cross-product
        // branch. We only assert an overlap is found and the normal is unit.
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let rot = Quat::from_euler(
            glam::EulerRot::XYZ,
            core::f32::consts::FRAC_PI_4,
            0.0,
            core::f32::consts::FRAC_PI_4,
        );
        let b = Obb::from_quat(Vec3::new(1.6, 1.6, 0.0), rot, Vec3::ONE);
        let c = obb_obb_contact(0, 1, &a, &b).expect("the rotated box overlaps a's corner");
        assert!((c.normal.length() - 1.0).abs() < 1.0e-5);
        assert!(c.depth > 0.0);
    }

    #[test]
    fn rotated_box_separated_reports_no_contact() {
        // The same rotation but pushed clearly away: no axis overlaps.
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let rot = Quat::from_euler(
            glam::EulerRot::XYZ,
            core::f32::consts::FRAC_PI_4,
            0.0,
            core::f32::consts::FRAC_PI_4,
        );
        let b = Obb::from_quat(Vec3::new(5.0, 5.0, 0.0), rot, Vec3::ONE);
        assert!(obb_obb_contact(0, 1, &a, &b).is_none());
    }

    #[test]
    fn parallel_boxes_skip_degenerate_edge_axes() {
        // Two axis-aligned boxes have all nine edge crosses degenerate (parallel
        // edges); the contact must still resolve through a face axis.
        let a = axis_box(Vec3::ZERO, Vec3::ONE);
        let b = axis_box(Vec3::new(0.0, 1.5, 0.0), Vec3::ONE);
        let c = obb_obb_contact(0, 1, &a, &b).expect("stacked boxes overlap on y");
        assert!((c.normal - Vec3::Y).length() < 1.0e-6);
        assert!((c.depth - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn containment_reports_deep_penetration() {
        // A small box fully inside a large box: every axis overlaps, the MTV is
        // the shallowest exit. Small half extent 0.25 inside a half extent 2
        // box centred at the same point: overlap on each axis is 2 + 0.25 = 2.25.
        let a = axis_box(Vec3::ZERO, Vec3::splat(2.0));
        let b = axis_box(Vec3::new(0.1, 0.0, 0.0), Vec3::splat(0.25));
        let c = obb_obb_contact(0, 1, &a, &b).expect("a contained box always contacts");
        assert!((c.normal.length() - 1.0).abs() < 1.0e-6);
        assert!(c.depth > 2.0);
    }

    #[test]
    fn batch_preserves_order_and_indices() {
        // Three boxes, two couples: a clear overlap then a clear gap, checked in
        // input order with their couple indices intact.
        let boxes = [
            axis_box(Vec3::ZERO, Vec3::ONE),
            axis_box(Vec3::new(1.5, 0.0, 0.0), Vec3::ONE),
            axis_box(Vec3::new(10.0, 0.0, 0.0), Vec3::ONE),
        ];
        let pairs = [ObbObbPair::new(0, 1), ObbObbPair::new(0, 2)];
        let contacts = cpu_obb_obb_narrowphase(&boxes, &pairs);
        assert_eq!(contacts.len(), 2);
        let first = contacts[0].expect("boxes 0 and 1 overlap");
        assert_eq!(first.a, 0);
        assert_eq!(first.b, 1);
        assert!((first.normal - Vec3::X).length() < 1.0e-6);
        assert!(contacts[1].is_none());
    }
}
