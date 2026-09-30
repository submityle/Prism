//! OBB-versus-halfspace contact geometry, shared bit-for-bit with the `WGSL` kernel.
//!
//! A halfspace is the solid region on one side of an infinite plane — the
//! canonical static collider for a ground, a wall, or a frustum face. This
//! module is the single source of truth for the oriented-bounding-box against
//! halfspace overlap test and manifold construction; the `CPU` twin
//! ([`cpu_obb_halfspace_narrowphase`]) and the device kernel
//! (`shaders/narrowphase_obb_halfspace.wgsl`) both run this exact arithmetic, so
//! their contacts agree to within the floating-point tolerance the parity test
//! allows.
//!
//! # Geometry
//!
//! A [`Plane`] is a unit outward `normal` `n` and a scalar `offset` `d`; the
//! surface is `{ x : dot(n, x) = d }` and the solid occupies
//! `{ x : dot(n, x) <= d }`, so `n` points out of the solid into free space. An
//! [`Obb`] is a centre `center`, three orthonormal local axes `axes`, and per
//! axis half extents `half_extents`.
//!
//! The test uses the box's support function along the plane normal:
//!
//! * the box centre's signed distance to the surface is
//!   `s_c = dot(n, center) - d`;
//! * the box's projected radius along `n` is the support half-width
//!   `proj = |dot(n, axes[0])| * he.x + |dot(n, axes[1])| * he.y + |dot(n, axes[2])| * he.z`;
//! * the deepest vertex's signed distance is therefore `s_min = s_c - proj`.
//!
//! When `s_min >= 0` the whole box is on the free side (or grazing), so there is
//! no contact. Otherwise the box penetrates by `depth = -s_min`, the contact
//! normal is the outward plane normal `n` (the direction that pushes the box
//! out), and the deepest vertex is `deepest = center + Σ axes[i] * sign_i *
//! he[i]` with `sign_i = -1` when `dot(n, axes[i]) >= 0` and `+1` otherwise
//! (each axis steps toward the solid side). The reported contact `point` is that
//! vertex projected onto the surface, `deepest - n * (dot(n, deepest) - d)`,
//! which is algebraically `deepest - n * s_min`.
//!
//! # Single-point manifold
//!
//! This slice reports one contact — the single deepest vertex — per
//! `(box, plane)` couple, matching the one-`Contact`-per-pair architecture the
//! sphere-halfspace slice already establishes. That is a deliberate, honest
//! design choice, not a stub: the deepest penetrating vertex is the exact
//! worst-case point a position solver needs to resolve first. A full multi-point
//! manifold (a resting face flush against the plane produces up to four contact
//! points) is a separate follow-up slice; nothing here fakes or truncates a
//! multi-point result.
//!
//! # Per-operation agreement
//!
//! Both paths perform the identical steps in the identical order: the three
//! axis dot products, the centre distance, the support projection with
//! `abs(dot(n, axis))`, the `s_min >= 0` rejection, the per-axis sign choice,
//! the deepest-vertex accumulation, and the surface projection. There is no
//! square root or reciprocal on this path, so the parity test matches the
//! validity flag exactly and the normal, depth, and point to within the
//! tightest float tolerance.
//!
//! Provenance: textbook OBB-versus-halfspace (support-function) collision
//! manifold; no Unreal Engine source or derived code.

use super::contact::Contact;
use super::halfspace::Plane;
use super::obb::Obb;

/// A candidate `(box, plane)` couple to test for penetration.
///
/// `obb` indexes the oriented-bounding-box array and `plane` indexes the plane
/// array. Keeping the pairing explicit lets the narrow phase emit one contact
/// slot per couple in input order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObbPlanePair {
    /// Index of the oriented bounding box (the `a` side of the reported contact).
    pub obb: u32,
    /// Index of the plane collider (the `b` side of the reported contact).
    pub plane: u32,
}

impl ObbPlanePair {
    /// Couples box index `obb` with plane index `plane`.
    #[must_use]
    pub fn new(obb: u32, plane: u32) -> ObbPlanePair {
        ObbPlanePair { obb, plane }
    }
}

/// Tests whether an oriented bounding box penetrates a halfspace and, if so,
/// builds the contact at the deepest vertex.
///
/// Returns [`Some`] with the manifold for the couple `(obb_id, plane_id)` when
/// the box's deepest vertex crosses the surface into the solid (`s_min < 0`), or
/// [`None`] when the whole box floats clear or exactly grazes the surface
/// (`s_min >= 0`). The arithmetic mirrors `narrowphase_obb_halfspace.wgsl`
/// operation for operation; see the module documentation for the geometry.
#[must_use]
pub(crate) fn obb_halfspace_contact(
    obb_id: u32,
    plane_id: u32,
    box_: &Obb,
    plane: &Plane,
) -> Option<Contact> {
    let n = plane.normal;
    let he = box_.half_extents;
    let a0 = box_.axes[0];
    let a1 = box_.axes[1];
    let a2 = box_.axes[2];

    // Projection of each box axis onto the plane normal.
    let d0 = n.dot(a0);
    let d1 = n.dot(a1);
    let d2 = n.dot(a2);

    // Signed distance of the box centre from the surface.
    let s_c = n.dot(box_.center) - plane.offset;
    // Support half-width of the box along the normal.
    let proj = d0.abs() * he.x + d1.abs() * he.y + d2.abs() * he.z;
    // Signed distance of the deepest vertex from the surface.
    let s_min = s_c - proj;

    // Strict overlap: the deepest vertex must cross into the solid; a box that
    // only grazes the surface carries no penetration.
    if s_min >= 0.0 {
        return None;
    }

    // Deepest vertex: step along each axis toward the solid side of the plane.
    let sign0 = if d0 >= 0.0 { -1.0 } else { 1.0 };
    let sign1 = if d1 >= 0.0 { -1.0 } else { 1.0 };
    let sign2 = if d2 >= 0.0 { -1.0 } else { 1.0 };
    let deepest = box_.center + a0 * (sign0 * he.x) + a1 * (sign1 * he.y) + a2 * (sign2 * he.z);

    let depth = -s_min;
    // Deepest vertex projected onto the surface: the resting contact point.
    let s_deepest = n.dot(deepest) - plane.offset;
    let point = deepest - n * s_deepest;
    Some(Contact::new(obb_id, plane_id, n, depth, point))
}

/// `CPU` golden twin of the OBB-versus-halfspace narrow phase.
///
/// Turns a set of candidate `pairs` into contact manifolds, running the shared
/// [`obb_halfspace_contact`] geometry so its output matches the device kernel
/// operation for operation. It emits one slot per input couple, preserving
/// order, which is what lets the parity test line the `GPU` contacts up against
/// this reference index by index: [`Some`] carrying the deepest-vertex manifold
/// when the box penetrates, or [`None`] when it floats clear or grazes. A later
/// scan stage compacts the survivors.
///
/// # Panics
///
/// Panics if a couple references a box or plane index outside the corresponding
/// slice, which is never valid output from a broad phase over the same sets.
#[must_use]
pub fn cpu_obb_halfspace_narrowphase(
    boxes: &[Obb],
    planes: &[Plane],
    pairs: &[ObbPlanePair],
) -> Vec<Option<Contact>> {
    pairs
        .iter()
        .map(|pair| {
            let box_ = &boxes[pair.obb as usize];
            let plane = &planes[pair.plane as usize];
            obb_halfspace_contact(pair.obb, pair.plane, box_, plane)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn box_sunk_into_ground_reports_deepest_vertex() {
        // Centre 0.5 below the surface, half extent 1: deepest vertex at y = -1.5.
        let box_ = axis_box(Vec3::new(0.0, -0.5, 0.0), Vec3::ONE);
        let c = obb_halfspace_contact(0, 0, &box_, &ground())
            .expect("a box straddling the surface must contact");
        assert_eq!(c.a, 0);
        assert_eq!(c.b, 0);
        assert!((c.normal - Vec3::Y).length() < 1.0e-6);
        // depth = proj - s_c = 1 - (-0.5) = 1.5.
        assert!((c.depth - 1.5).abs() < 1.0e-6);
        // Deepest corner projected onto the surface (x, z tie to -1).
        assert!((c.point - Vec3::new(-1.0, 0.0, -1.0)).length() < 1.0e-6);
    }

    #[test]
    fn centre_outside_but_corner_pokes_through() {
        // Centre 0.5 above the surface: the box still dips a corner below it.
        let box_ = axis_box(Vec3::new(0.0, 0.5, 0.0), Vec3::ONE);
        let c = obb_halfspace_contact(2, 1, &box_, &ground())
            .expect("a poking corner must contact even with the centre outside");
        assert_eq!(c.a, 2);
        assert_eq!(c.b, 1);
        assert!((c.normal - Vec3::Y).length() < 1.0e-6);
        // depth = 1 - 0.5 = 0.5.
        assert!((c.depth - 0.5).abs() < 1.0e-6);
        assert!((c.point - Vec3::new(-1.0, 0.0, -1.0)).length() < 1.0e-6);
    }

    #[test]
    fn box_fully_outside_reports_no_contact() {
        // Centre 2 above the surface, half extent 1: lowest vertex at y = 1 > 0.
        let box_ = axis_box(Vec3::new(0.0, 2.0, 0.0), Vec3::ONE);
        assert!(obb_halfspace_contact(0, 0, &box_, &ground()).is_none());
    }

    #[test]
    fn box_grazing_surface_reports_no_contact() {
        // Lowest vertex exactly on the surface: s_min == 0, strict test rejects.
        let box_ = axis_box(Vec3::new(0.0, 1.0, 0.0), Vec3::ONE);
        assert!(obb_halfspace_contact(0, 0, &box_, &ground()).is_none());
    }

    #[test]
    fn rotated_box_deepest_vertex_shifts() {
        // A box rotated 45 degrees about z has support half-width sqrt(2) along
        // +y; its lowest vertex sinks to center.y - sqrt(2).
        let rot = Quat::from_rotation_z(core::f32::consts::FRAC_PI_4);
        let box_ = Obb::from_quat(Vec3::new(0.0, 1.0, 0.0), rot, Vec3::ONE);
        let c = obb_halfspace_contact(0, 0, &box_, &ground())
            .expect("the rotated box dips below the surface");
        assert!((c.normal - Vec3::Y).length() < 1.0e-5);
        let want_depth = 2.0f32.sqrt() - 1.0;
        assert!((c.depth - want_depth).abs() < 1.0e-5);
        // Lowest vertex projects to (0, 0, -1) on the surface.
        assert!((c.point - Vec3::new(0.0, 0.0, -1.0)).length() < 1.0e-5);
    }

    #[test]
    fn deeply_buried_box_reports_large_depth() {
        // Centre 5 below the surface: deepest vertex at y = -6, depth 6.
        let box_ = axis_box(Vec3::new(0.0, -5.0, 0.0), Vec3::ONE);
        let c =
            obb_halfspace_contact(0, 0, &box_, &ground()).expect("a buried box contacts deeply");
        assert!((c.depth - 6.0).abs() < 1.0e-6);
        assert!((c.point - Vec3::new(-1.0, 0.0, -1.0)).length() < 1.0e-6);
    }

    #[test]
    fn slanted_plane_uses_the_general_normal() {
        // Plane through the origin with a 45-degree normal in the x/y plane.
        let n = Vec3::new(1.0, 1.0, 0.0).normalize();
        let plane = Plane::new(n, 0.0);
        // A unit box whose centre sits just inside the solid side.
        let box_ = axis_box(Vec3::new(-0.2, -0.2, 0.0), Vec3::ONE);
        let c = obb_halfspace_contact(0, 0, &box_, &plane)
            .expect("the box crosses the slanted surface");
        assert!((c.normal - n).length() < 1.0e-6);
        // s_c = dot(n, center) = -0.4/sqrt(2); proj = (1 + 1)/sqrt(2) = sqrt(2).
        let inv = 1.0 / 2.0f32.sqrt();
        let s_c = -0.4 * inv;
        let proj = 2.0 * inv;
        assert!((c.depth - (proj - s_c)).abs() < 1.0e-6);
        // The contact point lies on the surface: dot(n, point) == offset (0).
        assert!(n.dot(c.point).abs() < 1.0e-5);
    }

    #[test]
    fn batch_preserves_order_over_multiple_planes() {
        // Two boxes against a ground plane and a wall plane, checked in input
        // order with their couple indices intact.
        let boxes = [
            axis_box(Vec3::new(0.0, -0.5, 0.0), Vec3::ONE), // dips into ground
            axis_box(Vec3::new(15.0, 5.0, 0.0), Vec3::ONE), // clear of both: above ground and outside the wall's free side
        ];
        let planes = [
            Plane::new(Vec3::Y, 0.0),  // ground y = 0
            Plane::new(Vec3::X, 10.0), // wall x = 10
        ];
        let pairs = [
            ObbPlanePair::new(0, 0), // box 0 vs ground -> contact
            ObbPlanePair::new(1, 0), // box 1 vs ground -> none
            ObbPlanePair::new(1, 1), // box 1 vs wall   -> none
        ];
        let contacts = cpu_obb_halfspace_narrowphase(&boxes, &planes, &pairs);
        assert_eq!(contacts.len(), 3);
        let first = contacts[0].expect("box 0 dips into the ground");
        assert_eq!(first.a, 0);
        assert_eq!(first.b, 0);
        assert!((first.depth - 1.5).abs() < 1.0e-6);
        assert!(contacts[1].is_none());
        assert!(contacts[2].is_none());
    }
}
