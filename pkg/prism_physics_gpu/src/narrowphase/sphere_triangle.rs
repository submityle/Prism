//! Sphere-versus-triangle contact geometry, shared bit-for-bit with the `WGSL` kernel.
//!
//! [`sphere_triangle_contact`] is the single source of truth for the
//! sphere-against-triangle collision test and manifold construction; the `CPU`
//! twin ([`cpu_sphere_triangle_narrowphase`]) and the device kernel
//! (`shaders/narrowphase_sphere_triangle.wgsl`) both run this exact arithmetic
//! so their contacts agree to within the floating-point tolerance the parity
//! test allows (only the square root and the handful of reciprocals in the
//! barycentric clamps differ in their low bits).
//!
//! This is the foundational slice of triangle-mesh collision: a static triangle
//! is the atom a trimesh or heightfield collider is built from, so a sphere (or
//! any bounding-sphere particle) resting on arbitrary level geometry reduces to
//! a batch of sphere-versus-triangle tests, one per candidate triangle the broad
//! phase surfaces. The [`closest_point_on_triangle`] helper it exposes is the
//! shared kernel the capsule-triangle and OBB-triangle slices build on, so the
//! Voronoi-region logic lives in exactly one place.
//!
//! # Geometry
//!
//! A [`Triangle`] is three world-space vertices `a`, `b`, `c`. A sphere is a
//! world centre `p` and radius `r`.
//!
//! The test first finds the closest point `q` on the (solid) triangle to the
//! sphere centre with [`closest_point_on_triangle`], Ericson's Voronoi-region
//! method: the point falls in one of seven regions (three vertices, three
//! edges, or the interior face), and the closest feature is picked by a fixed
//! sequence of sign tests on edge dot products. The offset `diff = p - q` is the
//! vector from that nearest point back to the sphere centre.
//!
//! * **Centre off the triangle** (`dot(diff, diff) > `[`COINCIDENT_EPS2`]): the
//!   distance is `dist = |diff|`; the pair contacts only when `dist < r`
//!   (strict, so a sphere grazing the surface at `dist == r` is *not* a
//!   contact). The normal is `diff / dist`, the penetration is `depth = r -
//!   dist`, and the contact point is the nearest triangle point `q`.
//! * **Centre on the triangle** (`dot(diff, diff) <= `[`COINCIDENT_EPS2`]): the
//!   sphere centre sits on the triangle, so `diff` is degenerate. The normal
//!   falls back to the triangle's geometric face normal
//!   `normalize((b - a) x (c - a))`, the penetration is the full `depth = r`,
//!   and the contact point is `q`. This is deterministic and identical on both
//!   paths, so a sphere pinned to the face never yields a `NaN` normal.
//!
//! # Normal convention
//!
//! The contact normal points **from the triangle toward the sphere**, i.e. the
//! direction that pushes the sphere off the triangle. The reported [`Contact`]
//! stores the sphere index in `a` and the triangle index in `b`; like the
//! sphere-versus-OBB pair, this type's normal runs `b` (triangle) to `a`
//! (sphere), matching the solver's push-out convention for a dynamic sphere
//! against a static triangle.
//!
//! # Per-operation agreement
//!
//! Every step above runs in the identical order on both paths: the same edge
//! vectors, the same dot products, the same seven-way region cascade with its
//! fixed comparison order, the same barycentric reciprocals, the same strict
//! `dist < r` overlap test, and the same degenerate fallback to the face
//! normal. Only `sqrt` and the barycentric reciprocals are inexact, so the
//! parity test matches the validity flag exactly and the normal, depth, and
//! point within a tight tolerance.
//!
//! Provenance: closest-point-on-triangle is the Voronoi-region method from
//! Christer Ericson, *Real-Time Collision Detection* (2004), section 5.1.5; the
//! sphere manifold is textbook. No Unreal Engine source or derived code.

use glam::Vec3;

use super::contact::Contact;

/// Squared-distance threshold below which the offset from the nearest triangle
/// point to the sphere centre is treated as zero, i.e. the centre lies on the
/// triangle and the face-normal fallback runs. Kept identical to the `WGSL`
/// constant so both paths pick the fallback on the same pairs.
pub(crate) const COINCIDENT_EPS2: f32 = 1.0e-12;

/// A triangle collider: three world-space vertices wound counter-clockwise when
/// viewed from the front (`+normal`) side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triangle {
    /// First vertex.
    pub a: Vec3,
    /// Second vertex.
    pub b: Vec3,
    /// Third vertex.
    pub c: Vec3,
}

impl Triangle {
    /// Creates a triangle from its three vertices.
    #[must_use]
    pub fn new(a: Vec3, b: Vec3, c: Vec3) -> Triangle {
        Triangle { a, b, c }
    }

    /// The triangle's unnormalised geometric normal, `(b - a) x (c - a)`.
    ///
    /// Its length is twice the triangle area; a degenerate (collinear or zero
    /// area) triangle returns a near-zero vector, which the contact fallback
    /// treats explicitly rather than normalising.
    #[must_use]
    pub fn raw_normal(&self) -> Vec3 {
        (self.b - self.a).cross(self.c - self.a)
    }
}

/// A candidate sphere-versus-triangle pair: an index into the sphere slice and
/// an index into the triangle slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SphereTrianglePair {
    /// Index of the sphere (bounding-sphere particle) in the sphere slice.
    pub sphere: u32,
    /// Index of the triangle in the triangle slice.
    pub triangle: u32,
}

impl SphereTrianglePair {
    /// Creates a sphere-versus-triangle candidate pair.
    #[must_use]
    pub fn new(sphere: u32, triangle: u32) -> SphereTrianglePair {
        SphereTrianglePair { sphere, triangle }
    }
}

/// Returns the point on the solid triangle `(a, b, c)` closest to `p`.
///
/// This is Ericson's Voronoi-region method (*Real-Time Collision Detection*,
/// section 5.1.5): the query point projects into one of seven regions — the
/// three vertices, the three edges, or the interior face — and the closest
/// feature is selected by a fixed cascade of sign tests on the edge dot
/// products, never falling through to an expensive projection. The result is
/// exact up to the barycentric reciprocals, which the device kernel reproduces
/// operation for operation.
#[must_use]
pub(crate) fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    // Vertex region outside A: the point sits before the AB and AC edges.
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    // Vertex region outside B.
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    // Edge region AB: project onto AB when its barycentric coordinate is valid.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    // Vertex region outside C.
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    // Edge region AC.
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    // Edge region BC.
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    // Interior face region: the barycentric coordinates are all positive.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// Tests whether the sphere penetrates the triangle and, if so, builds the
/// contact.
///
/// Returns [`Some`] with the manifold for the pair `(sphere_id, tri_id)` when
/// the sphere overlaps the triangle, or [`None`] when it is separated or
/// exactly touching (a grazing `dist == r`). The arithmetic mirrors
/// `narrowphase_sphere_triangle.wgsl` operation for operation; see the module
/// documentation for the geometry and the normal convention.
#[must_use]
pub(crate) fn sphere_triangle_contact(
    sphere_id: u32,
    tri_id: u32,
    p: Vec3,
    r: f32,
    tri: &Triangle,
) -> Option<Contact> {
    let q = closest_point_on_triangle(p, tri.a, tri.b, tri.c);
    let diff = p - q;
    let d2 = diff.dot(diff);

    if d2 > COINCIDENT_EPS2 {
        let dist = d2.sqrt();
        // Strict overlap: a sphere exactly touching the surface carries no
        // penetration, so it is not a contact.
        if dist >= r {
            return None;
        }
        let normal = diff / dist;
        let depth = r - dist;
        Some(Contact::new(sphere_id, tri_id, normal, depth, q))
    } else {
        // Centre is on the triangle: fall back to the geometric face normal.
        let raw = tri.raw_normal();
        let len2 = raw.dot(raw);
        let normal = if len2 > COINCIDENT_EPS2 {
            raw / len2.sqrt()
        } else {
            // Degenerate triangle: no face normal, so pick a stable axis.
            Vec3::X
        };
        Some(Contact::new(sphere_id, tri_id, normal, r, q))
    }
}

/// `CPU` golden twin of the sphere-versus-triangle narrow phase.
///
/// Turns a set of candidate `pairs` into contact manifolds, running the shared
/// [`sphere_triangle_contact`] geometry so its output matches the device kernel
/// operation for operation. It emits one slot per input pair, preserving order,
/// which is what lets the parity test line the `GPU` contacts up against this
/// reference index by index: [`Some`] carrying the manifold when the sphere
/// penetrates the triangle, or [`None`] when it is separated or exactly
/// touching. A later scan stage compacts the survivors.
///
/// # Panics
///
/// Panics if a pair references a sphere or triangle index outside the
/// corresponding slice, which is never valid output from a broad phase over the
/// same sets.
#[must_use]
pub fn cpu_sphere_triangle_narrowphase(
    spheres: &[crate::broadphase::Particle],
    triangles: &[Triangle],
    pairs: &[SphereTrianglePair],
) -> Vec<Option<Contact>> {
    pairs
        .iter()
        .map(|pair| {
            let sphere = &spheres[pair.sphere as usize];
            let tri = &triangles[pair.triangle as usize];
            sphere_triangle_contact(
                pair.sphere,
                pair.triangle,
                sphere.position,
                sphere.radius,
                tri,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broadphase::Particle;

    /// The canonical unit triangle in the z = 0 plane with a +z face normal.
    fn unit_triangle() -> Triangle {
        Triangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }

    #[test]
    fn face_contact_above_interior() {
        // Sphere hovering over the triangle interior at (0.25, 0.25, 0.4),
        // radius 0.5: nearest point is directly below on the face, so the
        // contact normal is +z and depth = r - dist = 0.5 - 0.4 = 0.1.
        let tri = unit_triangle();
        let c = sphere_triangle_contact(2, 7, Vec3::new(0.25, 0.25, 0.4), 0.5, &tri)
            .expect("sphere above the face must contact");
        assert_eq!(c.a, 2);
        assert_eq!(c.b, 7);
        assert!((c.normal - Vec3::Z).length() < 1.0e-6);
        assert!((c.depth - 0.1).abs() < 1.0e-6);
        assert!((c.point - Vec3::new(0.25, 0.25, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn below_face_pushes_down() {
        // Sphere under the face: nearest point is still on the face, so the
        // normal flips to -z (push the sphere further down).
        let tri = unit_triangle();
        let c = sphere_triangle_contact(0, 0, Vec3::new(0.25, 0.25, -0.3), 0.5, &tri)
            .expect("sphere below the face must contact");
        assert!((c.normal + Vec3::Z).length() < 1.0e-6);
        assert!((c.depth - 0.2).abs() < 1.0e-6);
        assert!((c.point - Vec3::new(0.25, 0.25, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn vertex_contact_a() {
        // Sphere off vertex A along -x/-y: nearest feature is the vertex (0,0,0).
        let tri = unit_triangle();
        let c = sphere_triangle_contact(0, 0, Vec3::new(-0.3, -0.4, 0.0), 0.6, &tri)
            .expect("sphere off vertex A must contact");
        let diff = Vec3::new(-0.3, -0.4, 0.0);
        let dist = diff.length();
        assert!((c.point - Vec3::ZERO).length() < 1.0e-6);
        assert!((c.normal - diff / dist).length() < 1.0e-6);
        assert!((c.normal.length() - 1.0).abs() < 1.0e-6);
        assert!((c.depth - (0.6 - dist)).abs() < 1.0e-6);
    }

    #[test]
    fn vertex_contact_b() {
        // Sphere past vertex B at +x: nearest feature is (1, 0, 0).
        let tri = unit_triangle();
        let c = sphere_triangle_contact(1, 1, Vec3::new(1.4, -0.2, 0.0), 0.6, &tri)
            .expect("sphere off vertex B must contact");
        let want = Vec3::new(1.0, 0.0, 0.0);
        assert!((c.point - want).length() < 1.0e-6);
        let diff = Vec3::new(1.4, -0.2, 0.0) - want;
        let dist = diff.length();
        assert!((c.normal - diff / dist).length() < 1.0e-6);
        assert!((c.depth - (0.6 - dist)).abs() < 1.0e-6);
    }

    #[test]
    fn edge_contact_ab() {
        // Sphere below edge AB (the y = 0 edge) at (0.5, -0.3, 0.0): nearest
        // point is on the edge at (0.5, 0, 0).
        let tri = unit_triangle();
        let c = sphere_triangle_contact(0, 0, Vec3::new(0.5, -0.3, 0.0), 0.5, &tri)
            .expect("sphere off edge AB must contact");
        assert!((c.point - Vec3::new(0.5, 0.0, 0.0)).length() < 1.0e-6);
        assert!((c.normal - Vec3::new(0.0, -1.0, 0.0)).length() < 1.0e-6);
        assert!((c.depth - 0.2).abs() < 1.0e-6);
    }

    #[test]
    fn edge_contact_bc() {
        // Sphere off the hypotenuse BC (from (1,0,0) to (0,1,0)) at (0.8, 0.8,
        // 0): nearest point is the edge midpoint (0.5, 0.5, 0).
        let tri = unit_triangle();
        let c = sphere_triangle_contact(0, 0, Vec3::new(0.8, 0.8, 0.0), 0.6, &tri)
            .expect("sphere off edge BC must contact");
        assert!((c.point - Vec3::new(0.5, 0.5, 0.0)).length() < 1.0e-6);
        let diff = Vec3::new(0.3, 0.3, 0.0);
        let dist = diff.length();
        assert!((c.normal - diff / dist).length() < 1.0e-6);
        assert!((c.depth - (0.6 - dist)).abs() < 1.0e-6);
    }

    #[test]
    fn centre_on_face_falls_back_to_face_normal() {
        // Sphere centred exactly on the triangle: diff is degenerate, so the
        // normal is the +z face normal and the depth is the full radius.
        let tri = unit_triangle();
        let c = sphere_triangle_contact(4, 9, Vec3::new(0.25, 0.25, 0.0), 0.5, &tri)
            .expect("a sphere centred on the face always contacts");
        assert!((c.normal - Vec3::Z).length() < 1.0e-6);
        assert!((c.depth - 0.5).abs() < 1.0e-6);
        assert!((c.point - Vec3::new(0.25, 0.25, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn clear_separation_reports_no_contact() {
        // Sphere far above the face: dist 2, radius 0.5, no overlap.
        let tri = unit_triangle();
        assert!(sphere_triangle_contact(0, 0, Vec3::new(0.25, 0.25, 2.0), 0.5, &tri).is_none());
    }

    #[test]
    fn exactly_touching_reports_no_contact() {
        // Sphere exactly grazing the face at dist == r: strict test rejects.
        let tri = unit_triangle();
        assert!(sphere_triangle_contact(0, 0, Vec3::new(0.25, 0.25, 0.5), 0.5, &tri).is_none());
    }

    #[test]
    fn batch_preserves_order_and_indices() {
        // Two spheres and one triangle: a clear face overlap then a clear gap,
        // checked in input order with their pair indices intact.
        let spheres = [
            Particle::new(Vec3::new(0.25, 0.25, 0.4), 0.5),
            Particle::new(Vec3::new(0.25, 0.25, 3.0), 0.5),
        ];
        let triangles = [unit_triangle()];
        let pairs = [SphereTrianglePair::new(0, 0), SphereTrianglePair::new(1, 0)];
        let contacts = cpu_sphere_triangle_narrowphase(&spheres, &triangles, &pairs);
        assert_eq!(contacts.len(), 2);
        let first = contacts[0].expect("first sphere overlaps");
        assert_eq!(first.a, 0);
        assert_eq!(first.b, 0);
        assert!((first.normal - Vec3::Z).length() < 1.0e-6);
        assert!(contacts[1].is_none());
    }
}
