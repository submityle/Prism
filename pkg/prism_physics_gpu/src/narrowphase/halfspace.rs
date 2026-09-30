//! Sphere-halfspace contact geometry, shared bit-for-bit with the `WGSL` kernel.
//!
//! A halfspace is the solid region on one side of an infinite plane — the
//! canonical static collider for a ground, a wall, or a frustum face. This
//! module is the single source of truth for the sphere-vs-halfspace overlap
//! test and manifold construction; the `CPU` twin
//! ([`cpu_halfspace_narrowphase`]) and the device kernel both run this exact
//! arithmetic, so their contacts agree to within the floating-point tolerance
//! the parity test allows (only the square root and the reciprocal in the
//! normalisation differ in their low bits).
//!
//! # Geometry
//!
//! A [`Plane`] is a unit outward `normal` `n` and a scalar `offset` `d`; the
//! plane surface is the set `{ x : dot(n, x) = d }` and the solid halfspace is
//! `{ x : dot(n, x) <= d }`, so `n` points out of the solid into free space. A
//! sphere with centre `c` and radius `r` has signed centre distance
//! `s = dot(n, c) - d`: positive when the centre floats in free space, negative
//! when it is buried in the solid. The sphere penetrates the solid when its
//! lowest point along `-n` crosses the surface, i.e. when `s < r`. The
//! penetration `depth` is then `r - s`, the contact `normal` is the outward `n`
//! (the direction that pushes the sphere out), and the contact `point` is the
//! centre projected onto the surface, `c - n * s`.
//!
//! There is no divide-by-zero to guard: the plane normal is assumed already
//! unit (the caller's responsibility, kept identical on both paths so the
//! parity stays exact), and the depth expression touches no reciprocal. A
//! centre exactly on the surface (`s == 0`) still overlaps by the full radius,
//! which is the physically correct resting contact.
//!
//! Provenance: textbook sphere-plane collision; no Unreal Engine source or
//! derived code.

use glam::Vec3;

use super::contact::Contact;
use crate::broadphase::Particle;

/// An infinite plane bounding a solid halfspace.
///
/// The plane surface is `{ x : dot(normal, x) = offset }`; the solid occupies
/// the side `dot(normal, x) <= offset`, so [`normal`](Self::normal) points out
/// of the solid. The normal is assumed to be unit length; supplying a
/// non-unit normal scales the reported depth and skews the contact point, so
/// callers normalise once when they build the collider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// Unit outward normal, pointing from the solid into free space.
    pub normal: Vec3,
    /// Signed plane offset along [`normal`](Self::normal): the surface is
    /// `dot(normal, x) == offset`.
    pub offset: f32,
}

impl Plane {
    /// Builds a plane from its outward `normal` and `offset`.
    ///
    /// `normal` is taken as-is (assumed unit) so the arithmetic stays bit-for-bit
    /// aligned with the device kernel, which likewise does not renormalise.
    #[must_use]
    pub fn new(normal: Vec3, offset: f32) -> Plane {
        Plane { normal, offset }
    }
}

/// A candidate `(sphere, plane)` couple to test for penetration.
///
/// Mirrors the broad-phase [`CandidatePair`](crate::broadphase::CandidatePair)
/// role for the mixed sphere-vs-halfspace phase: `sphere` indexes the particle
/// array and `plane` indexes the plane array. Keeping the pairing explicit lets
/// the narrow phase emit one contact slot per couple in input order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpherePlanePair {
    /// Index of the sphere particle (the `a` side of the reported contact).
    pub sphere: u32,
    /// Index of the plane collider (the `b` side of the reported contact).
    pub plane: u32,
}

impl SpherePlanePair {
    /// Couples sphere index `sphere` with plane index `plane`.
    #[must_use]
    pub fn new(sphere: u32, plane: u32) -> SpherePlanePair {
        SpherePlanePair { sphere, plane }
    }
}

/// Tests whether a sphere penetrates a halfspace and, if so, builds the contact.
///
/// Returns [`Some`] with the manifold for the couple `(sphere_id, plane_id)`
/// when the sphere's centre lies within one radius of the surface on the free
/// side or is buried in the solid (`s < r`), or [`None`] when the sphere floats
/// clear of the surface (`s >= r`). The arithmetic mirrors
/// `narrowphase_halfspace.wgsl` operation for operation.
///
/// The reported `normal` is the plane's outward normal (the push-out direction),
/// the `depth` is `r - s`, and the `point` is the sphere centre projected onto
/// the surface.
#[must_use]
pub(crate) fn sphere_halfspace_contact(
    sphere_id: u32,
    plane_id: u32,
    c: Vec3,
    r: f32,
    plane: &Plane,
) -> Option<Contact> {
    // Signed distance of the centre from the surface along the outward normal.
    let s = plane.normal.dot(c) - plane.offset;
    // Strict overlap: a centre exactly one radius clear only grazes the surface
    // and carries no penetration, so it is not a contact.
    if s >= r {
        return None;
    }

    let depth = r - s;
    // Centre projected onto the surface: the standard resting contact point.
    let point = c - plane.normal * s;
    Some(Contact::new(
        sphere_id,
        plane_id,
        plane.normal,
        depth,
        point,
    ))
}

/// `CPU` golden twin of the sphere-halfspace narrow phase.
///
/// Generates one contact slot per couple in `pairs`, in input order: [`Some`]
/// carrying the manifold when the sphere penetrates the plane's halfspace, or
/// [`None`] when it floats clear. Keeping a slot per couple (rather than
/// compacting) aligns the contact index with the pair index for the device
/// parity test; a later scan stage compacts the survivors.
///
/// # Panics
///
/// Panics if a couple references a sphere or plane index outside the supplied
/// slices, which is never valid output from a broad phase over the same sets.
#[must_use]
pub fn cpu_halfspace_narrowphase(
    spheres: &[Particle],
    planes: &[Plane],
    pairs: &[SpherePlanePair],
) -> Vec<Option<Contact>> {
    pairs
        .iter()
        .map(|pair| {
            let sphere = &spheres[pair.sphere as usize];
            let plane = &planes[pair.plane as usize];
            sphere_halfspace_contact(
                pair.sphere,
                pair.plane,
                sphere.position,
                sphere.radius,
                plane,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sphere at `(x, y, z)` with radius `r`.
    fn sphere(x: f32, y: f32, z: f32, r: f32) -> Particle {
        Particle::new(Vec3::new(x, y, z), r)
    }

    /// The `y = 0` ground plane, solid below, outward normal `+y`.
    fn ground() -> Plane {
        Plane::new(Vec3::Y, 0.0)
    }

    #[test]
    fn sphere_clear_above_ground_reports_no_contact() {
        // Centre at y = 5, radius 1: floats four units clear of the surface.
        let spheres = [sphere(0.0, 5.0, 0.0, 1.0)];
        let planes = [ground()];
        let pairs = [SpherePlanePair::new(0, 0)];
        assert_eq!(
            cpu_halfspace_narrowphase(&spheres, &planes, &pairs),
            vec![None]
        );
    }

    #[test]
    fn sphere_exactly_touching_reports_no_contact() {
        // Centre one radius above the surface: grazes it but does not penetrate.
        let spheres = [sphere(0.0, 1.0, 0.0, 1.0)];
        let planes = [ground()];
        let pairs = [SpherePlanePair::new(0, 0)];
        assert_eq!(
            cpu_halfspace_narrowphase(&spheres, &planes, &pairs),
            vec![None]
        );
    }

    #[test]
    fn sphere_resting_on_ground_builds_the_expected_manifold() {
        // Centre at y = 0.75, radius 1: overlaps by depth 0.25.
        let spheres = [sphere(2.0, 0.75, -3.0, 1.0)];
        let planes = [ground()];
        let pairs = [SpherePlanePair::new(0, 0)];
        let c = cpu_halfspace_narrowphase(&spheres, &planes, &pairs)[0]
            .expect("penetrating sphere must contact");
        assert_eq!(c.a, 0);
        assert_eq!(c.b, 0);
        // Normal is the outward plane normal, +y.
        assert!((c.normal - Vec3::Y).length() < 1.0e-6);
        // Depth = r - s = 1 - 0.75 = 0.25.
        assert!((c.depth - 0.25).abs() < 1.0e-6);
        // Point = centre projected onto y = 0: (2, 0, -3).
        assert!((c.point - Vec3::new(2.0, 0.0, -3.0)).length() < 1.0e-6);
    }

    #[test]
    fn sphere_buried_below_surface_reports_large_depth() {
        // Centre at y = -0.5, radius 1: buried, s = -0.5, depth = 1.5.
        let spheres = [sphere(0.0, -0.5, 0.0, 1.0)];
        let planes = [ground()];
        let pairs = [SpherePlanePair::new(0, 0)];
        let c = cpu_halfspace_narrowphase(&spheres, &planes, &pairs)[0].expect("buried sphere");
        assert!((c.depth - 1.5).abs() < 1.0e-6);
        // Point stays on the surface: y = 0.
        assert!(c.point.y.abs() < 1.0e-6);
    }

    #[test]
    fn tilted_plane_projects_along_its_own_normal() {
        // A 45-degree plane through the origin; normal along (1, 1, 0)/sqrt(2).
        let inv = 1.0 / (2.0f32).sqrt();
        let n = Vec3::new(inv, inv, 0.0);
        let plane = Plane::new(n, 0.0);
        // Place the centre one third of a radius inside the solid along -n.
        let r = 1.0;
        let s = -1.0 / 3.0;
        let c = n * s;
        let spheres = [Particle::new(c, r)];
        let planes = [plane];
        let pairs = [SpherePlanePair::new(0, 0)];
        let contact = cpu_halfspace_narrowphase(&spheres, &planes, &pairs)[0].expect("penetration");
        assert!((contact.normal - n).length() < 1.0e-6);
        assert!((contact.depth - (r - s)).abs() < 1.0e-6);
        // Point = c - n * s lands on the surface, dot(n, point) == offset == 0.
        assert!(n.dot(contact.point).abs() < 1.0e-6);
    }

    #[test]
    fn one_slot_per_pair_in_order_over_multiple_planes() {
        // A hit against the ground and a miss against a high ceiling, in order.
        let spheres = [sphere(0.0, 0.5, 0.0, 1.0)];
        let ceiling = Plane::new(-Vec3::Y, -10.0); // surface at y = 10, solid above
        let planes = [ground(), ceiling];
        let pairs = [SpherePlanePair::new(0, 0), SpherePlanePair::new(0, 1)];
        let contacts = cpu_halfspace_narrowphase(&spheres, &planes, &pairs);
        assert_eq!(contacts.len(), 2);
        assert!(
            contacts[0].is_some(),
            "0.5-radius-1 sphere dips into ground"
        );
        assert!(contacts[1].is_none(), "far below the ceiling");
    }

    #[test]
    fn empty_pairs_yield_no_contacts() {
        let spheres = [sphere(0.0, 0.0, 0.0, 1.0)];
        let planes = [ground()];
        assert!(cpu_halfspace_narrowphase(&spheres, &planes, &[]).is_empty());
    }
}
