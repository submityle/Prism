//! Capsule-versus-halfspace contact geometry, shared bit-for-bit with the
//! `WGSL` kernel.
//!
//! A halfspace is the solid region on one side of an infinite plane — the
//! canonical static collider for a ground, a wall, or a frustum face. This
//! module is the single source of truth for the capsule-against-halfspace
//! overlap test and manifold construction; the `CPU` twin
//! ([`cpu_capsule_halfspace_manifold`]) and the device kernel
//! (`shaders/narrowphase_capsule_halfspace.wgsl`) both run this exact
//! arithmetic, so their manifolds agree to within the floating-point tolerance
//! the parity test allows (this path carries no square root or reciprocal, so
//! the only perturbation is fused-multiply-add reassociation in the axis dot
//! products).
//!
//! # Geometry
//!
//! A [`Plane`] is a unit outward `normal` `n` and a scalar `offset` `d`; the
//! surface is `{ x : dot(n, x) = d }` and the solid occupies
//! `{ x : dot(n, x) <= d }`, so `n` points out of the solid into free space. A
//! [`Capsule`] is a line segment `(p0, p1)` swept by a radius `rc`.
//!
//! Because the signed distance `dot(n, x) - d` is *affine* along the segment,
//! its minimum over the capsule axis is always attained at an endpoint: there is
//! no interior minimum to search for. Each axis endpoint `p_i` therefore behaves
//! exactly like a sphere of radius `rc` centred at `p_i`:
//!
//! * its centre's signed distance is `s_i = dot(n, p_i) - d`;
//! * the capsule surface dips a distance `rc` further toward the solid, so the
//!   deepest surface point below `p_i` has signed distance `s_i - rc`;
//! * that end penetrates when `s_i < rc`, by `depth_i = rc - s_i`, and the
//!   contact point is `p_i` projected onto the surface, `p_i - n * s_i`.
//!
//! # Two-point manifold
//!
//! Unlike the single-vertex OBB-halfspace slice, this slice reports **up to two**
//! contact points — one per penetrating endpoint — because a capsule lying flat
//! on a plane rests on a *segment*, not a point, and a solver given only one
//! point lets the capsule rock and roll about it. Reporting both penetrating
//! ends is the manifold every production engine builds for a horizontal capsule
//! on the ground, and it is the reason this module returns a
//! [`ContactManifold`] rather than a single [`Contact`](super::contact::Contact).
//! A capsule standing on end (only one endpoint below the surface) or a
//! degenerate zero-length capsule (a sphere) collapses to the single-point case,
//! reported honestly as `count == 1`.
//!
//! # Degenerate segment
//!
//! When the segment is degenerate (`dot(ab, ab) <= SEG_EPS2`, a zero-length
//! capsule that is really a sphere) only the `p0` endpoint is tested, so the two
//! coincident ends never produce a duplicate point. The threshold is kept
//! identical to the `WGSL` constant so both paths take the collapse on the same
//! capsules.
//!
//! # Normal convention
//!
//! The shared [`normal`](ContactManifold::normal) is the plane's outward normal
//! `n` — the direction that pushes the capsule out of the solid — matching the
//! sibling sphere- and OBB-halfspace slices. The capsule is the `a` side and the
//! plane the `b` side of every reported manifold.
//!
//! Provenance: textbook capsule-versus-halfspace (affine support) collision
//! manifold; no Unreal Engine source or derived code.

use glam::Vec3;

use super::capsule::Capsule;
use super::halfspace::Plane;
use super::manifold::{ContactManifold, ManifoldPoint, MAX_MANIFOLD_POINTS};

/// Squared-length threshold below which a capsule segment is treated as a single
/// point (a degenerate zero-length capsule). Redefined here (rather than
/// imported from `capsule.rs`) so it stays lock-step with the identical `WGSL`
/// constant in this feature's kernel, keeping both paths on the same collapse.
pub(crate) const SEG_EPS2: f32 = 1.0e-12;

/// A candidate `(capsule, plane)` couple to test for penetration.
///
/// `capsule` indexes the capsule array and `plane` indexes the plane array.
/// Keeping the pairing explicit lets the narrow phase emit one manifold slot per
/// couple in input order, which is what lets the parity test line the `GPU`
/// manifolds up against the reference index by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapsulePlanePair {
    /// Index of the capsule (the `a` side of the reported manifold).
    pub capsule: u32,
    /// Index of the plane collider (the `b` side of the reported manifold).
    pub plane: u32,
}

impl CapsulePlanePair {
    /// Couples capsule index `capsule` with plane index `plane`.
    #[must_use]
    pub fn new(capsule: u32, plane: u32) -> CapsulePlanePair {
        CapsulePlanePair { capsule, plane }
    }
}

/// Tests whether a capsule penetrates a halfspace and, if so, builds the
/// manifold at the penetrating endpoints.
///
/// Returns [`Some`] with a one- or two-point manifold for the couple
/// `(capsule_id, plane_id)` when at least one axis endpoint's swept surface
/// crosses into the solid (`s_i < rc`), or [`None`] when the whole capsule
/// floats clear or exactly grazes the surface. The arithmetic mirrors
/// `narrowphase_capsule_halfspace.wgsl` operation for operation; see the module
/// documentation for the geometry.
///
/// The reported normal is the plane's outward normal (the push-out direction),
/// each point's depth is `rc - s_i`, and each point is the corresponding axis
/// endpoint projected onto the surface. Points are emitted in axis order (`p0`
/// then `p1`).
#[must_use]
pub(crate) fn capsule_halfspace_manifold(
    capsule_id: u32,
    plane_id: u32,
    cap: &Capsule,
    plane: &Plane,
) -> Option<ContactManifold> {
    let n = plane.normal;
    let d = plane.offset;
    let rc = cap.radius;

    // A capsule with a degenerate (zero-length) axis is a sphere: test only p0
    // so the two coincident ends never emit a duplicate contact point.
    let ab = cap.p1 - cap.p0;
    let degenerate = ab.dot(ab) <= SEG_EPS2;

    let mut points = [ManifoldPoint::new(Vec3::ZERO, 0.0); MAX_MANIFOLD_POINTS];
    let mut count = 0usize;

    // Endpoint p0: signed distance of its centre, strict penetration test.
    let s0 = n.dot(cap.p0) - d;
    if s0 < rc {
        points[count] = ManifoldPoint::new(cap.p0 - n * s0, rc - s0);
        count += 1;
    }

    // Endpoint p1, unless the capsule collapsed to a single point.
    if !degenerate {
        let s1 = n.dot(cap.p1) - d;
        if s1 < rc {
            points[count] = ManifoldPoint::new(cap.p1 - n * s1, rc - s1);
            count += 1;
        }
    }

    if count == 0 {
        // Neither end reaches the surface: no penetration.
        return None;
    }
    Some(ContactManifold::new(
        capsule_id, plane_id, n, count, &points,
    ))
}

/// `CPU` golden twin of the capsule-versus-halfspace narrow phase.
///
/// Turns a set of candidate `pairs` into contact manifolds, running the shared
/// [`capsule_halfspace_manifold`] geometry so its output matches the device
/// kernel operation for operation. It emits one slot per input couple,
/// preserving order, which is what lets the parity test line the `GPU` manifolds
/// up against this reference index by index: [`Some`] carrying the one- or
/// two-point manifold when the capsule penetrates, or [`None`] when it floats
/// clear or grazes. A later scan stage compacts the survivors.
///
/// # Panics
///
/// Panics if a couple references a capsule or plane index outside the
/// corresponding slice, which is never valid output from a broad phase over the
/// same sets.
#[must_use]
pub fn cpu_capsule_halfspace_manifold(
    capsules: &[Capsule],
    planes: &[Plane],
    pairs: &[CapsulePlanePair],
) -> Vec<Option<ContactManifold>> {
    pairs
        .iter()
        .map(|pair| {
            let cap = &capsules[pair.capsule as usize];
            let plane = &planes[pair.plane as usize];
            capsule_halfspace_manifold(pair.capsule, pair.plane, cap, plane)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ground plane `y = 0` with an upward outward normal.
    fn ground() -> Plane {
        Plane::new(Vec3::Y, 0.0)
    }

    #[test]
    fn horizontal_capsule_on_ground_reports_both_ends() {
        // A capsule lying flat along +x, radius 0.5, centred 0.3 above the
        // ground: both ends dip 0.2 below the swept surface.
        let cap = Capsule::new(Vec3::new(-1.0, 0.3, 0.0), Vec3::new(1.0, 0.3, 0.0), 0.5);
        let m = capsule_halfspace_manifold(0, 0, &cap, &ground())
            .expect("a flat capsule must contact the ground at both ends");
        assert_eq!(m.a, 0);
        assert_eq!(m.b, 0);
        assert_eq!(m.count, 2);
        assert!((m.normal - Vec3::Y).length() < 1.0e-6);
        // depth = rc - s = 0.5 - 0.3 = 0.2 at both ends.
        assert!((m.points[0].depth - 0.2).abs() < 1.0e-6);
        assert!((m.points[1].depth - 0.2).abs() < 1.0e-6);
        // Points are the endpoints projected onto the surface (y = 0).
        assert!((m.points[0].position - Vec3::new(-1.0, 0.0, 0.0)).length() < 1.0e-6);
        assert!((m.points[1].position - Vec3::new(1.0, 0.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn vertical_capsule_reports_only_the_lower_end() {
        // A capsule standing along +y, radius 0.5: only p0 (the lower end) dips
        // below the surface; p1 is high in free space.
        let cap = Capsule::new(Vec3::new(0.0, 0.2, 0.0), Vec3::new(0.0, 3.0, 0.0), 0.5);
        let m =
            capsule_halfspace_manifold(1, 2, &cap, &ground()).expect("the lower end penetrates");
        assert_eq!(m.a, 1);
        assert_eq!(m.b, 2);
        assert_eq!(m.count, 1);
        // depth = 0.5 - 0.2 = 0.3.
        assert!((m.points[0].depth - 0.3).abs() < 1.0e-6);
        assert!((m.points[0].position - Vec3::new(0.0, 0.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn tilted_capsule_with_one_end_buried() {
        // A capsule tilted so p0 sinks well below the ground and p1 rides clear.
        // p0 = (0, -0.4, 0), s0 = -0.4 < rc; p1 = (2, 2, 0), s1 = 2 > rc.
        let cap = Capsule::new(Vec3::new(0.0, -0.4, 0.0), Vec3::new(2.0, 2.0, 0.0), 0.5);
        let m = capsule_halfspace_manifold(0, 0, &cap, &ground()).expect("p0 is buried");
        assert_eq!(m.count, 1);
        // depth = 0.5 - (-0.4) = 0.9.
        assert!((m.points[0].depth - 0.9).abs() < 1.0e-6);
        // p0 projected onto the surface: (0, -0.4, 0) - Y * (-0.4) = (0, 0, 0).
        assert!((m.points[0].position - Vec3::new(0.0, 0.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn capsule_floating_clear_reports_no_contact() {
        // Both ends sit a full radius-plus above the surface.
        let cap = Capsule::new(Vec3::new(-1.0, 2.0, 0.0), Vec3::new(1.0, 2.0, 0.0), 0.5);
        assert!(capsule_halfspace_manifold(0, 0, &cap, &ground()).is_none());
    }

    #[test]
    fn capsule_grazing_surface_reports_no_contact() {
        // Both ends exactly one radius above the surface: s == rc, the strict
        // test rejects the graze.
        let cap = Capsule::new(Vec3::new(-1.0, 0.5, 0.0), Vec3::new(1.0, 0.5, 0.0), 0.5);
        assert!(capsule_halfspace_manifold(0, 0, &cap, &ground()).is_none());
    }

    #[test]
    fn degenerate_capsule_behaves_like_a_sphere() {
        // p0 == p1: the capsule is a sphere of radius 0.5 centred at (0, 0.2, 0).
        // Only one point is reported, not two coincident ones.
        let cap = Capsule::new(Vec3::new(0.0, 0.2, 0.0), Vec3::new(0.0, 0.2, 0.0), 0.5);
        let m = capsule_halfspace_manifold(0, 0, &cap, &ground()).expect("the sphere penetrates");
        assert_eq!(m.count, 1);
        assert!((m.points[0].depth - 0.3).abs() < 1.0e-6);
        assert!((m.points[0].position - Vec3::new(0.0, 0.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn slanted_plane_uses_the_general_normal() {
        // A plane through the origin with a 45-degree normal in the x/y plane.
        // A horizontal capsule along +z is symmetric about the normal, so both
        // ends share the same signed distance.
        let n = Vec3::new(1.0, 1.0, 0.0).normalize();
        let plane = Plane::new(n, 0.0);
        // Capsule axis along +z through (-0.2, -0.2, +-1), rc 0.5.
        let cap = Capsule::new(Vec3::new(-0.2, -0.2, -1.0), Vec3::new(-0.2, -0.2, 1.0), 0.5);
        let m = capsule_halfspace_manifold(0, 0, &cap, &plane).expect("the capsule crosses");
        assert_eq!(m.count, 2);
        assert!((m.normal - n).length() < 1.0e-6);
        // s = dot(n, (-0.2, -0.2, z)) = -0.4 / sqrt(2); depth = rc - s.
        let s = -0.4 / 2.0f32.sqrt();
        let want_depth = 0.5 - s;
        assert!((m.points[0].depth - want_depth).abs() < 1.0e-6);
        assert!((m.points[1].depth - want_depth).abs() < 1.0e-6);
        // Both points lie on the surface: dot(n, point) == offset (0).
        assert!(n.dot(m.points[0].position).abs() < 1.0e-5);
        assert!(n.dot(m.points[1].position).abs() < 1.0e-5);
    }

    #[test]
    fn batch_preserves_order_over_multiple_planes() {
        // A flat capsule on the ground (two points), a clear capsule (none), and
        // the flat capsule against a distant wall (none), checked in order.
        let capsules = [
            Capsule::new(Vec3::new(-1.0, 0.3, 0.0), Vec3::new(1.0, 0.3, 0.0), 0.5),
            Capsule::new(Vec3::new(-1.0, 8.0, 0.0), Vec3::new(1.0, 8.0, 0.0), 0.5),
        ];
        // The wall's solid occupies x <= -10, so a capsule near the origin sits
        // far on its free side and never contacts it.
        let planes = [Plane::new(Vec3::Y, 0.0), Plane::new(Vec3::X, -10.0)];
        let pairs = [
            CapsulePlanePair::new(0, 0), // flat capsule vs ground -> two points
            CapsulePlanePair::new(1, 0), // high capsule vs ground -> none
            CapsulePlanePair::new(0, 1), // flat capsule on the wall's free side -> none
        ];
        let manifolds = cpu_capsule_halfspace_manifold(&capsules, &planes, &pairs);
        assert_eq!(manifolds.len(), 3);
        let first = manifolds[0].expect("capsule 0 rests on the ground");
        assert_eq!(first.count, 2);
        assert!(manifolds[1].is_none());
        assert!(manifolds[2].is_none());
    }

    #[test]
    fn empty_pairs_yield_no_manifolds() {
        let capsules = [Capsule::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5)];
        let planes = [ground()];
        assert!(cpu_capsule_halfspace_manifold(&capsules, &planes, &[]).is_empty());
    }
}
