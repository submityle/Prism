//! Multi-point contact manifold for an oriented bounding box against a
//! halfspace (a resting box on a plane).
//!
//! The single-point [`obb_halfspace_contact`](super::obb_halfspace) reports only
//! the deepest penetrating vertex, which is enough to separate a pair but too
//! sparse to let a solver keep a box *flat* and *stable* on the ground: a
//! one-point contact cannot resist the toppling torque a resting box needs
//! cancelled, so the box jitters and tips. This module promotes that single
//! contact to the full **incident-face manifold** every mainstream engine
//! builds for box-versus-plane: it identifies the box face lying flush against
//! (or dipping deepest into) the plane and reports each of that face's four
//! corners that has crossed the surface.
//!
//! # Geometry
//!
//! A [`Plane`] is a unit outward `normal` `n` and scalar `offset`; a point `p`
//! has signed distance `s = dot(n, p) - offset`, and lies in the solid when
//! `s < 0`. The box's *incident face* is the face whose axis is most aligned
//! with `n` (the axis `k` maximising `|dot(n, axis[k])|`, with a lower-index
//! tie-break), on the solid side of the box. That face's four corners are
//! enumerated by fixing axis `k` at its solid-side extent and sweeping the two
//! remaining axes over `±half_extent`; every corner with `s < 0` becomes a
//! contact, its penetration `depth = -s` and its world position the corner
//! projected onto the surface, `p - n * s`.
//!
//! Because the deepest vertex of the whole box always lies on this incident
//! face (its per-axis signs are exactly the solid-side signs), a penetrating box
//! always yields **at least one** corner — the same contact, to the last bit,
//! the single-point kernel reports — so the manifold degrades honestly to one
//! point for a corner-first poke, two for an edge-first landing, and the full
//! four for a flush rest. A box that floats clear or merely grazes the surface
//! (`s_min >= 0`) reports [`None`], exactly matching the single-point gate.
//!
//! # Correctness model
//!
//! As with every kernel in this crate, [`cpu_obb_halfspace_manifold`] performs
//! the identical `f32` arithmetic, in the identical order, as
//! `shaders/narrowphase_obb_halfspace_manifold.wgsl`, and the real-device parity
//! test bounds the remaining floating-point-reassociation difference with a
//! tight tolerance. The deepest corner is bit-for-bit the single-point contact.
//!
//! # Provenance
//!
//! Textbook oriented-bounding-box-versus-halfspace incident-face clipping (the
//! standard box-on-plane resting manifold). No Unreal Engine source or derived
//! code.

use glam::Vec3;

use super::halfspace::Plane;
use super::manifold::{ContactManifold, ManifoldPoint, MAX_MANIFOLD_POINTS};
use super::obb::Obb;
use super::obb_halfspace::ObbPlanePair;

/// Builds the incident-face contact manifold for one `(box, plane)` couple.
///
/// Returns [`Some`] with the up-to-four-corner manifold when the box's deepest
/// vertex crosses the surface into the solid (`s_min < 0`), or [`None`] when the
/// whole box floats clear or exactly grazes it (`s_min >= 0`). The shared normal
/// is the plane's outward `normal` and the deepest reported corner is identical
/// to the single-point [`obb_halfspace_contact`](super::obb_halfspace) contact.
///
/// The arithmetic mirrors `narrowphase_obb_halfspace_manifold.wgsl` operation
/// for operation; see the module documentation for the geometry.
#[must_use]
pub(crate) fn obb_halfspace_manifold_pair(
    obb_id: u32,
    plane_id: u32,
    box_: &Obb,
    plane: &Plane,
) -> Option<ContactManifold> {
    let n = plane.normal;
    let he = [
        box_.half_extents.x,
        box_.half_extents.y,
        box_.half_extents.z,
    ];
    let axes = box_.axes;

    // Projection of each box axis onto the plane normal.
    let d = [n.dot(axes[0]), n.dot(axes[1]), n.dot(axes[2])];

    // Signed distance of the box centre from the surface and the box's support
    // half-width along the normal: the deepest vertex sits at `s_c - proj`.
    let s_c = n.dot(box_.center) - plane.offset;
    let proj = d[0].abs() * he[0] + d[1].abs() * he[1] + d[2].abs() * he[2];
    let s_min = s_c - proj;

    // Strict overlap: the deepest vertex must cross into the solid; a box that
    // only grazes the surface carries no penetration. Matches the single-point
    // gate exactly so the two kernels agree on validity.
    if s_min >= 0.0 {
        return None;
    }

    // Incident face: the axis most aligned with the plane normal, lower index
    // winning a tie. Its face lies flush against (or dips deepest into) the
    // plane, so its corners are the contact manifold.
    let ad = [d[0].abs(), d[1].abs(), d[2].abs()];
    let mut k = 0usize;
    let mut best = ad[0];
    if ad[1] > best {
        best = ad[1];
        k = 1;
    }
    if ad[2] > best {
        k = 2;
    }

    // The two axes swept over the face, kept in ascending index order so the CPU
    // and device enumerate the four corners identically.
    let (u, v) = match k {
        0 => (1usize, 2usize),
        1 => (0usize, 2usize),
        _ => (0usize, 1usize),
    };

    // Step axis `k` toward the solid side (opposite the normal projection) to
    // reach the incident face; the other two axes stay free to sweep.
    let sign_k = if d[k] >= 0.0 { -1.0 } else { 1.0 };
    let base = box_.center + axes[k] * (sign_k * he[k]);

    // Sweep the two free axes over `±half_extent` in a fixed order and keep every
    // corner that has crossed the surface. The nested order (su outer, sv inner)
    // is mirrored bit-for-bit by the device kernel.
    let mut points = [ManifoldPoint::new(Vec3::ZERO, 0.0); MAX_MANIFOLD_POINTS];
    let mut count = 0usize;
    let signs = [-1.0f32, 1.0f32];
    for &su in &signs {
        for &sv in &signs {
            let vertex = base + axes[u] * (su * he[u]) + axes[v] * (sv * he[v]);
            let s = n.dot(vertex) - plane.offset;
            if s < 0.0 {
                let position = vertex - n * s;
                points[count] = ManifoldPoint::new(position, -s);
                count += 1;
            }
        }
    }

    // The deepest vertex is one of the four swept corners and penetrates by
    // construction, so `count` is always in `1..=4` here.
    Some(ContactManifold::new(obb_id, plane_id, n, count, &points))
}

/// `CPU` golden twin of the `GPU` OBB-versus-halfspace manifold narrow phase.
///
/// Turns a set of candidate `pairs` into contact manifolds, running the shared
/// [`obb_halfspace_manifold_pair`] geometry so its output matches the device
/// kernel operation for operation. It emits one slot per input couple,
/// preserving order, which is what lets the parity test line the `GPU`
/// manifolds up against this reference index by index: [`Some`] carrying the
/// incident-face manifold when the box penetrates, or [`None`] when it floats
/// clear or grazes.
///
/// # Panics
///
/// Panics if a couple references a box or plane index outside the corresponding
/// slice, which is never valid output from a broad phase over the same sets.
#[must_use]
pub fn cpu_obb_halfspace_manifold(
    boxes: &[Obb],
    planes: &[Plane],
    pairs: &[ObbPlanePair],
) -> Vec<Option<ContactManifold>> {
    pairs
        .iter()
        .map(|pair| {
            let box_ = &boxes[pair.obb as usize];
            let plane = &planes[pair.plane as usize];
            obb_halfspace_manifold_pair(pair.obb, pair.plane, box_, plane)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrowphase::obb_halfspace::obb_halfspace_contact;
    use glam::{Quat, Vec3};

    /// The ground plane `y = 0` with an upward outward normal.
    fn ground() -> Plane {
        Plane::new(Vec3::Y, 0.0)
    }

    /// A unit-axis box at `center` with the given half extents.
    fn axis_box(center: Vec3, he: Vec3) -> Obb {
        Obb::new(center, [Vec3::X, Vec3::Y, Vec3::Z], he)
    }

    #[test]
    fn flush_resting_box_reports_four_corners() {
        // A unit box whose bottom face dips 0.1 below the ground: all four bottom
        // corners cross the surface at equal depth.
        let box_ = axis_box(Vec3::new(0.0, 0.9, 0.0), Vec3::ONE);
        let m = obb_halfspace_manifold_pair(0, 0, &box_, &ground()).expect("penetrates");
        assert_eq!(m.count, 4);
        assert_eq!(m.a, 0);
        assert_eq!(m.b, 0);
        assert!(
            (m.normal - Vec3::Y).length() < 1.0e-6,
            "normal {:?}",
            m.normal
        );
        for p in m.points.iter().take(4) {
            assert!((p.depth - 0.1).abs() < 1.0e-5, "depth {}", p.depth);
            // Each corner projects onto the ground surface y = 0.
            assert!(p.position.y.abs() < 1.0e-5, "y {}", p.position.y);
            // The four corners sit at x, z = ±1.
            assert!(
                (p.position.x.abs() - 1.0).abs() < 1.0e-5,
                "x {}",
                p.position.x
            );
            assert!(
                (p.position.z.abs() - 1.0).abs() < 1.0e-5,
                "z {}",
                p.position.z
            );
        }
    }

    #[test]
    fn edge_landing_reports_two_corners() {
        // Tilt the box 12 degrees about x so it rests on its lower edge parallel
        // to x: two corners of the incident face (perpendicular to the y axis)
        // dip below the ground, the other two ride clear.
        let theta = 12.0_f32.to_radians();
        let rot = Quat::from_rotation_x(theta);
        let box_ = Obb::from_quat(Vec3::new(0.0, 0.85, 0.0), rot, Vec3::ONE);
        let m = obb_halfspace_manifold_pair(0, 0, &box_, &ground()).expect("penetrates");
        assert_eq!(m.count, 2);
        assert!(
            (m.normal - Vec3::Y).length() < 1.0e-6,
            "normal {:?}",
            m.normal
        );
        for p in m.points.iter().take(2) {
            assert!(p.depth > 0.0, "depth {}", p.depth);
            assert!(p.position.y.abs() < 1.0e-5, "y {}", p.position.y);
        }
    }

    #[test]
    fn corner_poke_reports_one_corner() {
        // Tilt about two axes so a single vertex is the lowest and only it dips
        // below the surface: the manifold collapses honestly to one point.
        let rot = Quat::from_rotation_x(35.0_f32.to_radians())
            * Quat::from_rotation_z(30.0_f32.to_radians());
        let box_ = Obb::from_quat(Vec3::new(0.0, 1.6, 0.0), rot, Vec3::ONE);
        let m = obb_halfspace_manifold_pair(0, 0, &box_, &ground()).expect("penetrates");
        assert_eq!(m.count, 1);
        assert!(
            (m.normal - Vec3::Y).length() < 1.0e-6,
            "normal {:?}",
            m.normal
        );
        assert!(m.points[0].depth > 0.0, "depth {}", m.points[0].depth);
    }

    #[test]
    fn floating_box_reports_none() {
        // The whole box rides well above the ground: no contact at all.
        let box_ = axis_box(Vec3::new(0.0, 3.0, 0.0), Vec3::ONE);
        assert!(obb_halfspace_manifold_pair(0, 0, &box_, &ground()).is_none());
    }

    #[test]
    fn grazing_box_reports_none() {
        // The bottom face touches the surface exactly (s_min == 0): the strict
        // gate reports no penetration, matching the single-point kernel.
        let box_ = axis_box(Vec3::new(0.0, 1.0, 0.0), Vec3::ONE);
        assert!(obb_halfspace_manifold_pair(0, 0, &box_, &ground()).is_none());
    }

    #[test]
    fn deepest_corner_equals_the_single_point_contact() {
        // Whatever the pose, the deepest manifold corner must reproduce the
        // single-point contact to the last bit: same normal, depth, and point.
        // A compound tilt (about z then x) keeps every axis' projection onto
        // the plane normal non-zero, so the box has a *unique* deepest vertex and
        // the manifold's deepest corner matches the single-point kernel exactly;
        // a pure single-axis tilt would leave two corners at equal depth (a
        // legitimate tie the two kernels break by opposite conventions).
        let rot = Quat::from_rotation_z(15.0_f32.to_radians())
            * Quat::from_rotation_x(20.0_f32.to_radians());
        let box_ = Obb::from_quat(Vec3::new(0.3, 0.8, -0.2), rot, Vec3::ONE);
        let plane = ground();
        let single = obb_halfspace_contact(0, 0, &box_, &plane).expect("penetrates");
        let m = obb_halfspace_manifold_pair(0, 0, &box_, &plane).expect("penetrates");
        // The deepest reported corner is the one with the largest depth.
        let deepest = m
            .points
            .iter()
            .take(m.count as usize)
            .copied()
            .max_by(|x, y| x.depth.partial_cmp(&y.depth).expect("finite depths"))
            .expect("at least one corner");
        assert!((deepest.depth - single.depth).abs() < 1.0e-6, "depth");
        assert!(
            (deepest.position - single.point).length() < 1.0e-6,
            "point {:?} vs {:?}",
            deepest.position,
            single.point
        );
        assert!((m.normal - single.normal).length() < 1.0e-6, "normal");
    }

    #[test]
    fn manifold_batch_preserves_order_and_slots() {
        let boxes = [
            axis_box(Vec3::new(0.0, 0.9, 0.0), Vec3::ONE),
            axis_box(Vec3::new(0.0, 3.0, 0.0), Vec3::ONE),
        ];
        let planes = [ground()];
        let pairs = [ObbPlanePair::new(0, 0), ObbPlanePair::new(1, 0)];
        let out = cpu_obb_halfspace_manifold(&boxes, &planes, &pairs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].expect("penetrates").count, 4);
        assert!(out[1].is_none());
    }
}
