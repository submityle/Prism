//! Simple-polygon triangulation via ear clipping.
//!
//! GPU fills, mesh colliders, and coverage integrals all want a polygon broken
//! into triangles. [`triangulate`] turns a single simple polygon (no holes, no
//! self-intersections) into a fan of `n - 2` triangles described by index
//! triples into the input vertex array, so the caller keeps its own vertex
//! buffer and only receives topology.
//!
//! Ear clipping repeatedly removes an "ear": a convex vertex `v` whose triangle
//! `(prev, v, next)` contains no other polygon vertex. Removing an ear emits one
//! triangle and shrinks the polygon by a vertex; after `n - 3` clips three
//! vertices remain and form the last triangle. The convexity test is taken in
//! the polygon's own winding (clockwise or counter-clockwise), so either input
//! orientation triangulates correctly.
//!
//! Only `+ - * /` and comparisons are used — no transcendental functions — so
//! the routine is `no_std`-clean and bit-stable across targets. The algorithm
//! is quadratic in the vertex count, which is appropriate for the modest
//! polygons produced by UI vector paths.

use crate::measure::signed_area;
use alloc::vec::Vec;

/// Twice the signed area of triangle `a, b, c`; its sign gives the winding and
/// its magnitude is twice the area.
fn cross(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Returns `true` when `p` lies strictly inside triangle `a, b, c` (not on an
/// edge or vertex). Winding-independent.
fn strictly_inside(p: (f32, f32), a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> bool {
    let d1 = cross(a, b, p);
    let d2 = cross(b, c, p);
    let d3 = cross(c, a, p);
    if d1 == 0.0 || d2 == 0.0 || d3 == 0.0 {
        return false;
    }
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// Triangulates the simple polygon `polygon` by ear clipping, returning
/// `n - 2` triangles as index triples `[i, j, k]` into `polygon`.
///
/// The polygon must be *simple*: a single closed loop with no holes and no
/// self-intersections. It is treated as implicitly closed — do not repeat the
/// first vertex. Either winding is accepted; each emitted triple is wound the
/// same way as the input polygon.
///
/// Returns an empty `Vec` for fewer than three vertices. For a degenerate or
/// non-simple polygon where no ear can be found, triangulation stops early and
/// returns the triangles produced so far rather than looping forever.
#[must_use]
pub fn triangulate(polygon: &[(f32, f32)]) -> Vec<[usize; 3]> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }

    // Sign that makes a convex (ear-candidate) vertex test positive, regardless
    // of the polygon's winding.
    let sign = if signed_area(polygon) >= 0.0 { 1.0_f32 } else { -1.0 };

    let mut indices: Vec<usize> = (0..n).collect();
    let mut triangles: Vec<[usize; 3]> = Vec::with_capacity(n - 2);

    while indices.len() > 3 {
        let m = indices.len();
        let mut ear: Option<usize> = None;
        for (i, &iv) in indices.iter().enumerate() {
            let iu = indices[(i + m - 1) % m];
            let iw = indices[(i + 1) % m];
            let (u, v, w) = (polygon[iu], polygon[iv], polygon[iw]);
            // Reflex or collinear vertices are never ears.
            if sign * cross(u, v, w) <= 0.0 {
                continue;
            }
            // An ear's triangle must enclose no other polygon vertex.
            let mut is_ear = true;
            for &ix in &indices {
                if ix == iu || ix == iv || ix == iw {
                    continue;
                }
                if strictly_inside(polygon[ix], u, v, w) {
                    is_ear = false;
                    break;
                }
            }
            if is_ear {
                triangles.push([iu, iv, iw]);
                ear = Some(i);
                break;
            }
        }
        match ear {
            Some(i) => {
                indices.remove(i);
            }
            // No ear found: polygon is degenerate or non-simple. Stop cleanly.
            None => break,
        }
    }

    if indices.len() == 3 {
        triangles.push([indices[0], indices[1], indices[2]]);
    }

    triangles
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::polygon::sd_polygon;
    use crate::{area, convex_hull, point_in_polygon};

    fn next_rand(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn rand_in(s: &mut u64, lo: f32, hi: f32) -> f32 {
        let b = (next_rand(s) >> 40) as u32;
        lo + (hi - lo) * (b as f32 / 16_777_216.0)
    }

    fn random_convex(state: &mut u64) -> Vec<(f32, f32)> {
        loop {
            let n = 8 + (next_rand(state) % 24) as usize;
            let mut pts = Vec::with_capacity(n);
            for _ in 0..n {
                pts.push((rand_in(state, -20.0, 20.0), rand_in(state, -20.0, 20.0)));
            }
            let hull = convex_hull(&pts);
            if hull.len() >= 3 {
                return hull;
            }
        }
    }

    /// A star-shaped (hence simple) polygon with vertices in strict angular
    /// order and random radii; concave whenever radii vary. Uses `f64`
    /// transcendentals, which are permitted (only `f32` ones are banned).
    fn random_star(state: &mut u64, n: usize) -> Vec<(f32, f32)> {
        let mut v = Vec::with_capacity(n);
        for k in 0..n {
            let theta = core::f64::consts::TAU * (k as f64) / (n as f64);
            let r = f64::from(rand_in(state, 2.0, 12.0));
            v.push(((r * theta.cos()) as f32, (r * theta.sin()) as f32));
        }
        v
    }

    /// Sum of triangle areas computed from a triple.
    fn tri_area(poly: &[(f32, f32)], t: [usize; 3]) -> f32 {
        0.5 * cross(poly[t[0]], poly[t[1]], poly[t[2]]).abs()
    }

    fn total_tri_area(poly: &[(f32, f32)], tris: &[[usize; 3]]) -> f32 {
        tris.iter().map(|&t| tri_area(poly, t)).sum()
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-2 * (a.abs() + b.abs()) + 1e-2
    }

    /// Strongest oracle: the triangle areas must sum to the polygon area
    /// (independently computed by the shoelace formula).
    #[test]
    fn area_sums_to_shoelace_convex() {
        let mut state = 0x1357_9BDF_2468_ACE0_u64;
        for _ in 0..300 {
            let poly = random_convex(&mut state);
            let tris = triangulate(&poly);
            assert_eq!(tris.len(), poly.len() - 2, "wrong triangle count");
            assert!(
                approx(total_tri_area(&poly, &tris), area(&poly)),
                "area mismatch: {} vs {}",
                total_tri_area(&poly, &tris),
                area(&poly)
            );
        }
    }

    /// Same oracle for concave star-shaped polygons.
    #[test]
    fn area_sums_to_shoelace_concave() {
        let mut state = 0x0FED_CBA9_8765_4321_u64;
        for _ in 0..300 {
            let n = 5 + (next_rand(&mut state) % 20) as usize;
            let poly = random_star(&mut state, n);
            let tris = triangulate(&poly);
            assert_eq!(tris.len(), poly.len() - 2, "wrong triangle count");
            assert!(
                approx(total_tri_area(&poly, &tris), area(&poly)),
                "area mismatch: {} vs {}",
                total_tri_area(&poly, &tris),
                area(&poly)
            );
        }
    }

    /// Coverage oracle: a point strictly inside the polygon lies inside exactly
    /// one triangle, and a point outside lies in none. Checked away from
    /// boundaries where classification is ambiguous.
    #[test]
    fn triangles_cover_polygon() {
        let mut state = 0xCAFE_D00D_1234_5678_u64;
        let margin = 0.25_f32;
        for _ in 0..120 {
            let n = 5 + (next_rand(&mut state) % 16) as usize;
            let poly = random_star(&mut state, n);
            let tris = triangulate(&poly);
            for _ in 0..60 {
                let p = (rand_in(&mut state, -14.0, 14.0), rand_in(&mut state, -14.0, 14.0));
                if sd_polygon(p.0, p.1, &poly).abs() <= margin {
                    continue;
                }
                let count = tris
                    .iter()
                    .filter(|&&t| strictly_inside(p, poly[t[0]], poly[t[1]], poly[t[2]]))
                    .count();
                let inside = point_in_polygon(&poly, p);
                if inside {
                    assert_eq!(count, 1, "interior point {p:?} covered {count} times");
                } else {
                    assert_eq!(count, 0, "exterior point {p:?} covered {count} times");
                }
            }
        }
    }

    /// Every emitted index triple references distinct, in-range vertices.
    #[test]
    fn indices_are_valid() {
        let mut state = 0x2222_4444_6666_8888_u64;
        for _ in 0..200 {
            let n = 4 + (next_rand(&mut state) % 18) as usize;
            let poly = random_star(&mut state, n);
            let tris = triangulate(&poly);
            for t in &tris {
                assert!(t[0] < poly.len() && t[1] < poly.len() && t[2] < poly.len());
                assert!(t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
            }
        }
    }

    /// Reversing the winding gives the same triangle count and total area.
    #[test]
    fn winding_independent() {
        let mut state = 0x9999_7777_5555_3333_u64;
        for _ in 0..200 {
            let n = 5 + (next_rand(&mut state) % 16) as usize;
            let poly = random_star(&mut state, n);
            let mut rev = poly.clone();
            rev.reverse();
            let a = triangulate(&poly);
            let b = triangulate(&rev);
            assert_eq!(a.len(), b.len());
            assert!(approx(total_tri_area(&poly, &a), total_tri_area(&rev, &b)));
        }
    }

    /// Fixed concave "L": six vertices, must yield four triangles tiling area 7.
    #[test]
    fn fixed_l_shape() {
        let poly = alloc::vec![
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (0.0, 3.0),
        ];
        let tris = triangulate(&poly);
        assert_eq!(tris.len(), 4);
        assert!(approx(total_tri_area(&poly, &tris), area(&poly)));
        assert!(approx(area(&poly), 5.0), "L area was {}", area(&poly));
    }

    /// Degenerate inputs produce no triangles.
    #[test]
    fn degenerate_inputs() {
        assert!(triangulate(&[]).is_empty());
        assert!(triangulate(&[(0.0, 0.0)]).is_empty());
        assert!(triangulate(&[(0.0, 0.0), (1.0, 1.0)]).is_empty());
        // A single triangle triangulates to itself.
        let tri = alloc::vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        assert_eq!(triangulate(&tri).len(), 1);
    }
}
