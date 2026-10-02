//! OBB-versus-triangle contact geometry, shared bit-for-bit with the `WGSL`
//! kernel.
//!
//! This module is the single source of truth for the oriented-bounding-box
//! against triangle overlap test and manifold construction: the `CPU` twin
//! ([`cpu_obb_triangle_narrowphase`]) and the device kernel
//! (`shaders/narrowphase_obb_triangle.wgsl`) both run this exact arithmetic, so
//! their contacts agree to within the floating-point tolerance the parity test
//! allows (only the reciprocal square root in the axis normalisation and the
//! barycentric reciprocals in the closest-point helper are inexact).
//!
//! A single triangle is the atom a trimesh or heightfield collider is built
//! from, so an oriented box resting on arbitrary level geometry reduces to a
//! batch of OBB-versus-triangle tests, one per candidate triangle the broad
//! phase surfaces. This slice sits alongside [`sphere_triangle`] and the
//! capsule-triangle slice as the box half of triangle-mesh collision.
//!
//! [`sphere_triangle`]: super::sphere_triangle
//!
//! # Separating Axis Theorem
//!
//! A convex box and a (flat, convex) triangle are disjoint if and only if there
//! is an axis onto which their projected intervals do not overlap. For a box
//! against a triangle the candidate set is finite — thirteen axes suffice (the
//! Akenine-Möller triangle-box overlap set):
//!
//! * the three face normals of the box (its local axes `a0`, `a1`, `a2`);
//! * the single triangle face normal `(b - a) x (c - a)`;
//! * the nine edge-edge cross products `box_axis_i x tri_edge_j`, with the three
//!   triangle edges `e0 = b - a`, `e1 = c - b`, `e2 = a - c`.
//!
//! Along a unit axis `L` the box projects to the interval
//! `[cB - rB, cB + rB]` with centre projection `cB = dot(L, center)` and
//! half-width ("radius") `rB = |dot(L, a0)| * he.x + |dot(L, a1)| * he.y +
//! |dot(L, a2)| * he.z`. The triangle projects to
//! `[min(p0, p1, p2), max(p0, p1, p2)]` over its three projected vertices
//! `p_k = dot(L, v_k)`. The two intervals overlap by
//! `overlap = min(penL, penR)` with `penL = tmax - bmin` (the push that slides
//! the box off the triangle in `+L`) and `penR = bmax - tmin` (the push in
//! `-L`). If any candidate axis yields `overlap <= 0` the shapes are separated
//! (or exactly grazing) and the pair reports no contact — a strict test,
//! matching the sibling slices.
//!
//! When every candidate axis overlaps, the box penetrates the triangle. The
//! minimum-overlap axis is the minimum translation vector (`MTV`): the shortest
//! push that separates the box from the triangle. Its overlap is the
//! penetration `depth` and its oriented direction is the contact `normal`.
//!
//! # Normal orientation
//!
//! Each candidate axis is oriented by the two one-sided pushes it offers: when
//! `penL <= penR` the box is cleared by sliding it along `+L`, so the oriented
//! axis is `+L`; otherwise it is `-L`. (A tie resolves to `+L`.) This chooses,
//! per axis, the direction that pushes the box *out of* the triangle, so the
//! reported normal always runs **from the triangle toward the box**. Unlike the
//! OBB-OBB slice, there is no separate `dot(t, L)` flip: the triangle has no
//! single centre to flip against, and the one-sided push comparison already
//! fixes the orientation deterministically on both paths.
//!
//! # Face preference
//!
//! When a face axis (a box face or the triangle normal) and an edge-edge axis
//! overlap by almost the same amount, the face axis is the numerically stabler
//! contact normal (edge normals are the normalised cross of two directions that
//! may be nearly parallel). The `MTV` search therefore penalises edge axes by
//! [`EDGE_BIAS`] in the comparison only; the reported depth is always the true
//! overlap of the chosen axis. The bias is applied identically on both paths.
//!
//! # Degenerate axes
//!
//! An edge-edge cross vanishes when the box edge and triangle edge are parallel;
//! such an axis is parallel to a face axis already in the candidate set, so it
//! is skipped (guarded by [`CROSS_EPS2`] on the squared cross length). The
//! triangle normal is likewise skipped when the triangle is degenerate
//! (collinear or zero-area, so `raw_normal` is near zero): the remaining box
//! faces and the still-valid edge crosses then decide the result. This is a
//! deterministic fallback — identical on both paths — so a sliver triangle never
//! yields a `NaN` normal; it simply drops the one undefined axis from the test.
//!
//! # Normal convention
//!
//! The reported [`Contact`] stores the box index in `a` and the triangle index
//! in `b`, and the `normal` points from the triangle toward the box (the
//! direction that pushes the box off the triangle), matching the push-out
//! convention the sibling triangle slice uses for a dynamic shape against a
//! static triangle.
//!
//! # Single-point manifold
//!
//! This slice reports one representative contact per `(box, triangle)` couple:
//! the mid-overlap point between the box's support vertex along `-normal` (the
//! box corner driven deepest into the triangle) and the closest point on the
//! solid triangle to that corner. That is a deliberate, honest design choice,
//! not a stub — the point lies on the mid-overlap region the solver pushes
//! against, and for a vertex, edge, or face contact it lands on the penetrating
//! feature. A full multi-point clipped manifold (a box face resting flush on the
//! triangle produces several clipped contact points) is a separate follow-up
//! slice; nothing here fakes or truncates a multi-point result.
//!
//! # Per-operation agreement
//!
//! Both paths build the same thirteen candidate axes in the same order,
//! normalise the triangle normal and the edge crosses with the same
//! [`CROSS_EPS2`] guard, accumulate the same projection sums, take the same
//! strict `overlap <= 0` rejection, run the same [`EDGE_BIAS`] minimum search,
//! orient the axis by the same `penL <= penR` comparison, and build the contact
//! point from the same support vertex and the same closest-point-on-triangle
//! cascade. Only the reciprocal square roots in the axis normalisation and the
//! barycentric reciprocals in the closest-point helper are inexact, so the
//! parity test matches the validity flag exactly and the normal, depth, and
//! point to within a tight tolerance.
//!
//! Provenance: the thirteen-axis box-triangle separating-axis set is Tomas
//! Akenine-Möller, *Fast 3D Triangle-Box Overlap Testing* (2001); the
//! separating-axis minimum-translation manifold and the closest-point-on-
//! triangle Voronoi cascade (reused from [`sphere_triangle`]) are Christer
//! Ericson, *Real-Time Collision Detection* (2004), sections 5.2.9 and 5.1.5.
//! No Unreal Engine source or derived code.

use glam::Vec3;

use super::contact::Contact;
use super::obb::Obb;
use super::sphere_triangle::{closest_point_on_triangle, Triangle};

/// Squared cross-length threshold below which an edge-edge axis is treated as
/// degenerate (its two edges are parallel) and skipped; also the squared-length
/// threshold below which the triangle normal is treated as degenerate (a
/// collinear or zero-area triangle) and skipped. Kept identical to the `WGSL`
/// constant so both paths drop the same axes.
pub(crate) const CROSS_EPS2: f32 = 1.0e-12;

/// Comparison-only penalty added to an edge-edge axis's overlap so a face axis
/// (a box face or the triangle normal) of nearly equal overlap wins the
/// minimum-translation search. Never affects the reported penetration depth.
/// Kept identical to the `WGSL` constant.
pub(crate) const EDGE_BIAS: f32 = 1.0e-5;

/// Sentinel overlap for a skipped (degenerate) axis: larger than any real
/// overlap, so a degenerate axis never wins the minimum search.
const SKIP_OVERLAP: f32 = 1.0e30;

/// A candidate `(box, triangle)` couple to test for penetration.
///
/// `obb` indexes the oriented-bounding-box array and `triangle` indexes the
/// triangle array. Keeping the pairing explicit lets the narrow phase emit one
/// contact slot per couple in input order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObbTrianglePair {
    /// Index of the oriented bounding box (the `a` side of the contact).
    pub obb: u32,
    /// Index of the triangle (the `b` side of the contact).
    pub triangle: u32,
}

impl ObbTrianglePair {
    /// Couples box index `obb` with triangle index `triangle`.
    #[must_use]
    pub fn new(obb: u32, triangle: u32) -> ObbTrianglePair {
        ObbTrianglePair { obb, triangle }
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
pub(crate) fn support_vertex(center: Vec3, x: &[Vec3; 3], he: Vec3, dir: Vec3) -> Vec3 {
    center
        + x[0] * (sign_pos(dir.dot(x[0])) * he.x)
        + x[1] * (sign_pos(dir.dot(x[1])) * he.y)
        + x[2] * (sign_pos(dir.dot(x[2])) * he.z)
}

/// The winning separating axis and the oriented minimum-translation data.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SatQuery {
    /// Unit contact normal oriented from the triangle toward the box.
    pub(crate) normal: Vec3,
    /// Penetration depth along [`normal`](Self::normal); always positive.
    pub(crate) depth: f32,
    /// Index of the winning candidate axis in the fixed thirteen-axis order:
    /// `0..=2` are the box face normals (local axes), `3` is the triangle face
    /// normal, and `4..=12` are the nine edge-edge crosses. A manifold builder
    /// reads this to classify the contact (box-face reference, triangle-face
    /// reference, or edge-edge) and so pick the clipping that yields multiple
    /// coplanar points.
    pub(crate) axis_index: usize,
}

/// Runs the thirteen-axis separating-axis test on the box and triangle.
///
/// Returns [`Some`] with the minimum-translation axis (oriented from the
/// triangle toward the box) and penetration depth when every candidate axis
/// overlaps, or [`None`] when any axis separates the shapes or they exactly
/// graze (`overlap <= 0`).
#[must_use]
pub(crate) fn obb_triangle_sat(obb: &Obb, tri: &Triangle) -> Option<SatQuery> {
    let x = obb.axes;
    let he = obb.half_extents;

    // Triangle edges in a fixed order; the edge-edge crosses pair each box axis
    // with each of these.
    let e0 = tri.b - tri.a;
    let e1 = tri.c - tri.b;
    let e2 = tri.a - tri.c;
    let tri_edges = [e0, e1, e2];

    // The thirteen candidate axes in a fixed order: the box's three face
    // normals, the triangle face normal, then the nine edge-edge crosses. The
    // triangle normal and the edge crosses are normalised; a degenerate
    // (parallel) cross or a degenerate triangle normal is flagged so it is
    // skipped in both the separation test and the minimum search.
    let mut axes = [Vec3::ZERO; 13];
    let mut valid = [true; 13];
    axes[0] = x[0];
    axes[1] = x[1];
    axes[2] = x[2];

    let raw_normal = tri.raw_normal();
    let n_len2 = raw_normal.dot(raw_normal);
    if n_len2 < CROSS_EPS2 {
        valid[3] = false;
        axes[3] = Vec3::ZERO;
    } else {
        axes[3] = raw_normal.normalize();
    }

    let mut k = 4;
    for box_axis in &x {
        for edge in &tri_edges {
            let c = box_axis.cross(*edge);
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

    let verts = [tri.a, tri.b, tri.c];

    let mut best_overlap = SKIP_OVERLAP;
    let mut best_cmp = SKIP_OVERLAP;
    let mut best_normal = Vec3::ZERO;
    let mut best_axis = 0usize;
    let mut separated = false;
    for idx in 0..13 {
        if !valid[idx] {
            continue;
        }
        let axis = axes[idx];

        // Box interval along the axis.
        let cb = axis.dot(obb.center);
        let rb = projected_radius(axis, &x, he);
        let bmin = cb - rb;
        let bmax = cb + rb;

        // Triangle interval along the axis.
        let p0 = axis.dot(verts[0]);
        let p1 = axis.dot(verts[1]);
        let p2 = axis.dot(verts[2]);
        let tmin = p0.min(p1).min(p2);
        let tmax = p0.max(p1).max(p2);

        // Two one-sided pushes that clear the box off the triangle.
        let pen_l = tmax - bmin;
        let pen_r = bmax - tmin;
        let overlap = pen_l.min(pen_r);
        if overlap <= 0.0 {
            separated = true;
        }

        // Orient the axis toward the shorter push (ties resolve to +L), so the
        // normal always points from the triangle toward the box.
        let oriented = if pen_l <= pen_r { axis } else { -axis };

        // Edge axes (idx >= 4) carry a comparison penalty so a face axis of
        // nearly equal overlap is preferred; the reported depth stays exact.
        let bias = if idx >= 4 { EDGE_BIAS } else { 0.0 };
        let cmp = overlap + bias;
        if cmp < best_cmp {
            best_cmp = cmp;
            best_overlap = overlap;
            best_normal = oriented;
            best_axis = idx;
        }
    }

    if separated {
        return None;
    }

    Some(SatQuery {
        normal: best_normal,
        depth: best_overlap,
        axis_index: best_axis,
    })
}

/// Tests whether the box penetrates the triangle and, if so, builds the contact
/// along the minimum-translation axis.
///
/// Returns [`Some`] with the manifold for the couple `(box_id, tri_id)` when
/// every candidate separating axis overlaps, or [`None`] when any axis separates
/// the shapes or they exactly graze (`overlap <= 0`). The arithmetic mirrors
/// `narrowphase_obb_triangle.wgsl` operation for operation; see the module
/// documentation for the geometry and the normal convention.
#[must_use]
pub(crate) fn obb_triangle_contact(
    box_id: u32,
    tri_id: u32,
    obb: &Obb,
    tri: &Triangle,
) -> Option<Contact> {
    let sat = obb_triangle_sat(obb, tri)?;

    // Representative single point: the mid-overlap between the box's support
    // vertex driven deepest into the triangle (along -normal) and the closest
    // point on the solid triangle to that corner.
    let box_point = support_vertex(obb.center, &obb.axes, obb.half_extents, -sat.normal);
    let tri_point = closest_point_on_triangle(box_point, tri.a, tri.b, tri.c);
    let point = (box_point + tri_point) * 0.5;

    Some(Contact::new(box_id, tri_id, sat.normal, sat.depth, point))
}

/// `CPU` golden twin of the OBB-versus-triangle narrow phase.
///
/// Turns a set of candidate `pairs` into contact manifolds, running the shared
/// [`obb_triangle_contact`] geometry so its output matches the device kernel
/// operation for operation. It emits one slot per input couple, preserving
/// order, which is what lets the parity test line the `GPU` contacts up against
/// this reference index by index: [`Some`] carrying the minimum-translation
/// manifold when the box penetrates the triangle, or [`None`] when a separating
/// axis exists. A later scan stage compacts the survivors.
///
/// # Panics
///
/// Panics if a couple references a box or triangle index outside the
/// corresponding slice, which is never valid output from a broad phase over the
/// same sets.
#[must_use]
pub fn cpu_obb_triangle_narrowphase(
    boxes: &[Obb],
    triangles: &[Triangle],
    pairs: &[ObbTrianglePair],
) -> Vec<Option<Contact>> {
    pairs
        .iter()
        .map(|pair| {
            let obb = &boxes[pair.obb as usize];
            let tri = &triangles[pair.triangle as usize];
            obb_triangle_contact(pair.obb, pair.triangle, obb, tri)
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

    /// The canonical unit triangle in the z = 0 plane with a +z face normal.
    fn unit_triangle() -> Triangle {
        Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }

    #[test]
    fn face_contact_box_above_triangle() {
        // A box hovering just over the triangle interior: its bottom face dips
        // 0.1 below z = 0. The shallowest axis is the +z face normal, so the box
        // is pushed up by depth 0.1.
        let tri = unit_triangle();
        let box_ = axis_box(Vec3::new(0.25, 0.25, 0.4), Vec3::splat(0.5));
        let c = obb_triangle_contact(0, 0, &box_, &tri).expect("box dipping into the face contacts");
        assert_eq!(c.a, 0);
        assert_eq!(c.b, 0);
        assert!((c.normal - Vec3::Z).length() < 1.0e-6);
        assert!((c.depth - 0.1).abs() < 1.0e-6);
        assert!(c.point.is_finite());
    }

    #[test]
    fn face_contact_box_below_triangle() {
        // A box whose top face pokes 0.1 above z = 0 from below: the MTV is the
        // -z face normal, pushing the box further down.
        let tri = unit_triangle();
        let box_ = axis_box(Vec3::new(0.25, 0.25, -0.4), Vec3::splat(0.5));
        let c = obb_triangle_contact(2, 7, &box_, &tri).expect("box poking up from below contacts");
        assert_eq!(c.a, 2);
        assert_eq!(c.b, 7);
        assert!((c.normal + Vec3::Z).length() < 1.0e-6);
        assert!((c.depth - 0.1).abs() < 1.0e-6);
    }

    #[test]
    fn edge_contact_overlaps_hypotenuse() {
        // A box straddling the hypotenuse edge BC: its lower corner (0.4, 0.4)
        // sits inside the triangle while its upper corner pokes outside, so the
        // shallowest separating axis is the in-plane hypotenuse normal
        // (box_z x edge_BC), an edge-edge axis. The contact carries a unit normal
        // and a positive depth. (A box centred at (0.8, 0.8) would only graze the
        // single point (0.5, 0.5) on the hypotenuse and correctly report no
        // contact.)
        let tri = unit_triangle();
        let box_ = axis_box(Vec3::new(0.7, 0.7, 0.0), Vec3::splat(0.3));
        let c = obb_triangle_contact(0, 0, &box_, &tri).expect("box over the hypotenuse contacts");
        assert!((c.normal.length() - 1.0).abs() < 1.0e-6);
        // The winning axis is the in-plane hypotenuse normal, so the contact
        // normal lies in the triangle plane (its z component is ~0) and points
        // outward along +(x + y).
        assert!(c.normal.z.abs() < 1.0e-6);
        let hypot = Vec3::new(1.0, 1.0, 0.0).normalize();
        assert!((c.normal - hypot).length() < 1.0e-6);
        assert!((c.depth - 0.2 / 2.0_f32.sqrt()).abs() < 1.0e-5);
        assert!(c.point.is_finite());
    }

    #[test]
    fn corner_contact_overlaps_vertex() {
        // A box overlapping vertex B at (1, 0, 0) from the +x/-y quadrant: a
        // contact with a unit normal and a positive depth.
        let tri = unit_triangle();
        let box_ = axis_box(Vec3::new(1.1, -0.1, 0.0), Vec3::splat(0.3));
        let c = obb_triangle_contact(0, 0, &box_, &tri).expect("box over vertex B contacts");
        assert!((c.normal.length() - 1.0).abs() < 1.0e-6);
        assert!(c.depth > 0.0);
        assert!(c.point.is_finite());
    }

    #[test]
    fn rotated_box_contact_is_unit_normal() {
        // A box rotated 45 degrees about z, lowered onto the triangle so a slanted
        // edge digs into the face. We assert the contact exists with a unit normal
        // and positive depth (the parity test pins the exact values on-device).
        let tri = unit_triangle();
        let rot = Quat::from_rotation_z(core::f32::consts::FRAC_PI_4);
        let box_ = Obb::from_quat(Vec3::new(0.3, 0.3, 0.3), rot, Vec3::splat(0.5));
        let c = obb_triangle_contact(1, 1, &box_, &tri).expect("rotated box overlaps the face");
        assert!((c.normal.length() - 1.0).abs() < 1.0e-5);
        assert!(c.depth > 0.0);
    }

    #[test]
    fn clear_separation_reports_no_contact() {
        // Box well above the triangle: the +z face axis separates them.
        let tri = unit_triangle();
        let box_ = axis_box(Vec3::new(0.25, 0.25, 3.0), Vec3::splat(0.5));
        assert!(obb_triangle_contact(0, 0, &box_, &tri).is_none());
    }

    #[test]
    fn exactly_touching_reports_no_contact() {
        // Box whose bottom face is flush with z = 0: overlap == 0 on the +z face
        // axis, the strict test rejects.
        let tri = unit_triangle();
        let box_ = axis_box(Vec3::new(0.25, 0.25, 0.5), Vec3::splat(0.5));
        assert!(obb_triangle_contact(0, 0, &box_, &tri).is_none());
    }

    #[test]
    fn degenerate_triangle_falls_back_deterministically() {
        // Three collinear points along x: raw_normal is zero, so the triangle
        // normal axis is dropped. The box still overlaps the segment, and the
        // remaining box-face axes resolve the contact deterministically (no NaN).
        let degenerate = Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        );
        let box_ = axis_box(Vec3::ZERO, Vec3::splat(0.5));
        let c = obb_triangle_contact(0, 0, &box_, &degenerate)
            .expect("box overlapping the degenerate segment contacts");
        assert!((c.normal.length() - 1.0).abs() < 1.0e-6);
        assert!((c.depth - 0.5).abs() < 1.0e-6);
        assert!(c.point.is_finite());
    }

    #[test]
    fn batch_preserves_order_and_indices() {
        // Two boxes and one triangle: a clear face overlap then a clear gap,
        // checked in input order with their couple indices intact.
        let boxes = [
            axis_box(Vec3::new(0.25, 0.25, 0.4), Vec3::splat(0.5)),
            axis_box(Vec3::new(0.25, 0.25, 5.0), Vec3::splat(0.5)),
        ];
        let triangles = [unit_triangle()];
        let pairs = [ObbTrianglePair::new(0, 0), ObbTrianglePair::new(1, 0)];
        let contacts = cpu_obb_triangle_narrowphase(&boxes, &triangles, &pairs);
        assert_eq!(contacts.len(), 2);
        let first = contacts[0].expect("first box overlaps the face");
        assert_eq!(first.a, 0);
        assert_eq!(first.b, 0);
        assert!((first.normal - Vec3::Z).length() < 1.0e-6);
        assert!(contacts[1].is_none());
    }
}
