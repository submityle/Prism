//! Robust point-in-mesh classification via the generalized winding number.
//!
//! A signed-distance field, a flood-fill voxelizer or a mass integrator all
//! need to answer "is this point inside the mesh?" The classic trick, casting a
//! ray and counting crossings, is cheap but brittle: it misfires on open edges,
//! self-intersections and non-manifold junctions that real authored collision
//! meshes routinely contain.
//!
//! The *generalized winding number* (Jacobson, Kavan & Sorkine, SIGGRAPH 2013)
//! is the robust alternative. It sums the signed solid angle each triangle
//! subtends at the query point and divides by `4*pi`. For a closed,
//! consistently wound surface it evaluates to exactly `1` strictly inside and
//! `0` strictly outside; crucially it *degrades gracefully* on imperfect
//! meshes, returning a smooth fractional field that still rounds to the correct
//! inside/outside decision away from the defects.
//!
//! Each triangle's solid angle is evaluated with the Van Oosterom-Strackee
//! formula, accumulated in `f64` for numerical headroom. This is a published
//! geometry-processing result; nothing here is derived from Unreal Engine
//! source.

use glam::Vec3;

use core::f64::consts::PI;

/// Fraction of a full turn below which a winding number is treated as "outside"
/// / "inside": the field is rounded to the nearest integer multiple of a full
/// turn, so the inside test is `|w| >= 0.5`.
const INSIDE_THRESHOLD: f64 = 0.5;

/// A triangle vertex expressed in `f64`, relative to the query point.
type Rel = [f64; 3];

fn dot(u: Rel, v: Rel) -> f64 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}

fn cross(u: Rel, v: Rel) -> Rel {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

fn length(u: Rel) -> f64 {
    dot(u, u).sqrt()
}

/// Signed solid angle (steradians) subtended by triangle `(a, b, c)` at the
/// origin, via the Van Oosterom-Strackee formula. The vertices are already
/// expressed relative to the query point, in `f64` for numerical headroom.
fn signed_solid_angle(a: Rel, b: Rel, c: Rel) -> f64 {
    let la = length(a);
    let lb = length(b);
    let lc = length(c);
    let denom = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
    let numer = dot(a, cross(b, c));
    // atan2 keeps the result continuous across the +/- pi branch; a zero
    // numerator and denominator (point on the triangle plane through a vertex)
    // yields 0, contributing nothing.
    2.0 * numer.atan2(denom)
}

/// Lifts a mesh vertex into an `f64` triple relative to the query point.
fn rel(v: Vec3, point: Vec3) -> Rel {
    [
        f64::from(v.x) - f64::from(point.x),
        f64::from(v.y) - f64::from(point.y),
        f64::from(v.z) - f64::from(point.z),
    ]
}

/// Computes the generalized winding number of `point` with respect to the mesh.
///
/// Returns `None` when `vertices` or `indices` is empty. Triangles that
/// reference out-of-range vertices are skipped. For a closed, outward-wound
/// mesh the result is `~1` inside and `~0` outside; a globally reversed winding
/// flips the sign.
#[must_use]
pub fn generalized_winding_number(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    point: Vec3,
) -> Option<f32> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }
    let n = vertices.len();
    let mut accum = 0.0f64;
    for tri in indices {
        if (tri[0] as usize) >= n || (tri[1] as usize) >= n || (tri[2] as usize) >= n {
            continue;
        }
        let a = rel(vertices[tri[0] as usize], point);
        let b = rel(vertices[tri[1] as usize], point);
        let c = rel(vertices[tri[2] as usize], point);
        accum += signed_solid_angle(a, b, c);
    }
    Some((accum / (4.0 * PI)) as f32)
}

/// Classifies whether `point` lies inside the mesh using the generalized
/// winding number.
///
/// A point counts as inside when the winding number rounds to a non-zero
/// integer (`|w| >= 0.5`), which is robust to a globally reversed winding and
/// to small holes or self-intersections. Returns `None` on empty input.
#[must_use]
pub fn point_is_inside(vertices: &[Vec3], indices: &[[u32; 3]], point: Vec3) -> Option<bool> {
    let w = generalized_winding_number(vertices, indices, point)?;
    Some(f64::from(w).abs() >= INSIDE_THRESHOLD)
}

/// Evaluates the generalized winding number for many points against one mesh.
///
/// Returns `None` under the same conditions as
/// [`generalized_winding_number`]; otherwise one value per input point in
/// order.
#[must_use]
pub fn winding_numbers(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    points: &[Vec3],
) -> Option<Vec<f32>> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }
    Some(
        points
            .iter()
            .map(|&pt| generalized_winding_number(vertices, indices, pt).unwrap_or(0.0))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;

    /// Builds a closed, welded icosahedron subdivided `levels` times: a
    /// watertight, outward-wound unit sphere.
    fn icosphere(levels: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
        let mut verts: Vec<Vec3> = vec![
            Vec3::new(-1.0, t, 0.0),
            Vec3::new(1.0, t, 0.0),
            Vec3::new(-1.0, -t, 0.0),
            Vec3::new(1.0, -t, 0.0),
            Vec3::new(0.0, -1.0, t),
            Vec3::new(0.0, 1.0, t),
            Vec3::new(0.0, -1.0, -t),
            Vec3::new(0.0, 1.0, -t),
            Vec3::new(t, 0.0, -1.0),
            Vec3::new(t, 0.0, 1.0),
            Vec3::new(-t, 0.0, -1.0),
            Vec3::new(-t, 0.0, 1.0),
        ];
        for v in &mut verts {
            *v = v.normalize();
        }
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..levels {
            let mut cache: StdHashMap<(u32, u32), u32> = StdHashMap::new();
            let mut next: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
            let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&m) = cache.get(&key) {
                    return m;
                }
                let m = verts.len() as u32;
                let p = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
                verts.push(p);
                cache.insert(key, m);
                m
            };
            for f in &faces {
                let a = midpoint(f[0], f[1], &mut verts);
                let b = midpoint(f[1], f[2], &mut verts);
                let c = midpoint(f[2], f[0], &mut verts);
                next.push([f[0], a, c]);
                next.push([f[1], b, a]);
                next.push([f[2], c, b]);
                next.push([a, b, c]);
            }
            faces = next;
        }
        (verts, faces)
    }

    #[test]
    fn center_of_sphere_has_unit_winding() {
        let (v, t) = icosphere(3);
        let w = generalized_winding_number(&v, &t, Vec3::ZERO).expect("non-empty");
        assert!((w - 1.0).abs() < 1e-3, "center winding {w} should be ~1");
    }

    #[test]
    fn far_point_has_zero_winding() {
        let (v, t) = icosphere(3);
        let w = generalized_winding_number(&v, &t, Vec3::new(10.0, 7.0, -4.0)).expect("non-empty");
        assert!(w.abs() < 1e-3, "exterior winding {w} should be ~0");
    }

    #[test]
    fn inside_outside_classification_matches_sphere() {
        let (v, t) = icosphere(3);
        // Deep interior.
        assert_eq!(point_is_inside(&v, &t, Vec3::ZERO), Some(true));
        assert_eq!(
            point_is_inside(&v, &t, Vec3::new(0.5, 0.0, 0.0)),
            Some(true)
        );
        // Clearly exterior.
        assert_eq!(
            point_is_inside(&v, &t, Vec3::new(2.0, 0.0, 0.0)),
            Some(false)
        );
        assert_eq!(
            point_is_inside(&v, &t, Vec3::new(0.0, 3.0, 0.0)),
            Some(false)
        );
    }

    #[test]
    fn reversed_winding_flips_sign_but_not_inside() {
        let (v, mut t) = icosphere(2);
        for f in &mut t {
            f.swap(1, 2); // reverse every triangle's winding
        }
        let w = generalized_winding_number(&v, &t, Vec3::ZERO).expect("non-empty");
        assert!((w + 1.0).abs() < 1e-3, "reversed center winding {w} ~= -1");
        // The inside test is sign-robust.
        assert_eq!(point_is_inside(&v, &t, Vec3::ZERO), Some(true));
    }

    #[test]
    fn small_hole_still_classifies_interior() {
        let (v, mut t) = icosphere(3);
        // Remove a single triangle: the missing solid angle is tiny, so a deep
        // interior point still rounds to "inside".
        t.pop();
        let w = generalized_winding_number(&v, &t, Vec3::ZERO).expect("non-empty");
        assert!(w > 0.9, "interior winding {w} barely changes for one hole");
        assert_eq!(point_is_inside(&v, &t, Vec3::ZERO), Some(true));
    }

    #[test]
    fn batch_matches_scalar() {
        let (v, t) = icosphere(2);
        let pts = [
            Vec3::ZERO,
            Vec3::new(0.3, 0.2, -0.1),
            Vec3::new(5.0, 0.0, 0.0),
        ];
        let batch = winding_numbers(&v, &t, &pts).expect("non-empty");
        assert_eq!(batch.len(), pts.len());
        for (i, &p) in pts.iter().enumerate() {
            let scalar = generalized_winding_number(&v, &t, p).unwrap();
            assert!((batch[i] - scalar).abs() < 1e-6);
        }
    }

    #[test]
    fn out_of_range_triangles_are_skipped() {
        let (v, mut t) = icosphere(1);
        let n = v.len() as u32;
        t.push([n, n + 1, n + 2]); // dangling triangle, ignored
        let w = generalized_winding_number(&v, &t, Vec3::ZERO).expect("non-empty");
        assert!((w - 1.0).abs() < 1e-2);
    }

    #[test]
    fn empty_input_is_rejected() {
        assert!(generalized_winding_number(&[], &[], Vec3::ZERO).is_none());
        assert!(point_is_inside(&[], &[], Vec3::ZERO).is_none());
        assert!(winding_numbers(&[], &[], &[Vec3::ZERO]).is_none());
        let (v, _) = icosphere(0);
        assert!(generalized_winding_number(&v, &[], Vec3::ZERO).is_none());
    }
}
