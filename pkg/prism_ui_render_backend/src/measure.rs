//! Polygon area and centroid via the shoelace formula.
//!
//! A filled path lowered by [`sd_polygon`](crate::polygon::sd_polygon) or
//! simplified by [`simplify`](crate::simplify::simplify) often needs cheap
//! integral quantities: the enclosed area (for coverage estimates, level-of-
//! detail culling, or splitting large fills) and the area-weighted centroid
//! (for anchoring transforms, labels, or shadows). Both fall straight out of
//! the shoelace (Gauss) formula over the polygon's edges.
//!
//! [`signed_area`] returns a positive value for a counter-clockwise winding and
//! a negative value for a clockwise one, so its sign doubles as an orientation
//! test; [`area`] is its magnitude. [`centroid`] returns the area-weighted
//! centroid, or `None` when the polygon is degenerate (fewer than three
//! vertices, or a vanishing area such as a collinear run) and the centroid is
//! undefined.
//!
//! All arithmetic is pure `f32` (`+ - * /`); there are no transcendental calls,
//! so the routines are `no_std`-clean and bit-stable.

/// Twice the signed area, accumulated once over the closed edge loop. Shared by
/// [`signed_area`] and [`centroid`] so both agree on the degeneracy threshold.
fn signed_area2(verts: &[(f32, f32)]) -> f32 {
    let n = verts.len();
    if n < 3 {
        return 0.0;
    }
    let mut acc = 0.0_f32;
    for (i, &(x0, y0)) in verts.iter().enumerate() {
        let (x1, y1) = verts[(i + 1) % n];
        acc += x0 * y1 - x1 * y0;
    }
    acc
}

/// Signed area of the closed polygon `verts` (shoelace formula).
///
/// Positive for a counter-clockwise winding, negative for clockwise, so the
/// sign is also an orientation test. Returns `0.0` for fewer than three
/// vertices. The polygon is treated as implicitly closed; do not repeat the
/// first vertex at the end.
#[must_use]
pub fn signed_area(verts: &[(f32, f32)]) -> f32 {
    signed_area2(verts) * 0.5
}

/// Unsigned area of the closed polygon `verts`.
///
/// This is the magnitude of [`signed_area`] and is therefore independent of
/// winding direction. Returns `0.0` for fewer than three vertices.
#[must_use]
pub fn area(verts: &[(f32, f32)]) -> f32 {
    signed_area(verts).abs()
}

/// Area that is treated as degenerate for centroid purposes. A twice-area below
/// this magnitude means the vertices are (near-)collinear and the area-weighted
/// centroid would be numerically unstable, so [`centroid`] returns `None`.
const DEGENERATE_AREA2: f32 = 1e-6;

/// Area-weighted centroid of the closed polygon `verts`.
///
/// Returns `None` when the polygon has fewer than three vertices or a vanishing
/// area (for example a collinear run), where the centroid is undefined.
/// Otherwise returns the centroid, which — unlike the plain vertex average — is
/// invariant to how finely the edges are subdivided and lands at the true area
/// center for non-convex shapes too.
///
/// The result is covariant under translation and cyclic vertex rotation, and is
/// unchanged by reversing the winding (only [`signed_area`]'s sign flips).
#[must_use]
pub fn centroid(verts: &[(f32, f32)]) -> Option<(f32, f32)> {
    let n = verts.len();
    if n < 3 {
        return None;
    }
    let mut a2 = 0.0_f32;
    let mut cx = 0.0_f32;
    let mut cy = 0.0_f32;
    for (i, &(x0, y0)) in verts.iter().enumerate() {
        let (x1, y1) = verts[(i + 1) % n];
        let cross = x0 * y1 - x1 * y0;
        a2 += cross;
        cx += (x0 + x1) * cross;
        cy += (y0 + y1) * cross;
    }
    if a2.abs() < DEGENERATE_AREA2 {
        return None;
    }
    // Cx = 1/(6A) * sum, and 6A == 3 * a2 since a2 == 2A.
    let inv = 1.0 / (3.0 * a2);
    Some((cx * inv, cy * inv))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::convex_hull;
    use alloc::vec::Vec;

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

    /// A random convex polygon (CCW) of at least three vertices, so the fan
    /// triangulation from vertex 0 tiles it exactly for the Heron oracle.
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

    fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        (dx * dx + dy * dy).sqrt()
    }

    /// Triangle area from its three side lengths (Heron). Uses only `sqrt`, so
    /// it is an algebraically independent recompute of the cross-product area.
    fn tri_area_heron(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
        let ab = dist(a, b);
        let bc = dist(b, c);
        let ca = dist(c, a);
        let s = 0.5 * (ab + bc + ca);
        (s * (s - ab) * (s - bc) * (s - ca)).max(0.0).sqrt()
    }

    /// Area of a convex polygon by summing the fan triangles from vertex 0 with
    /// Heron's formula — independent of the shoelace cross products.
    fn fan_area_heron(poly: &[(f32, f32)]) -> f32 {
        let mut sum = 0.0_f32;
        for i in 1..poly.len() - 1 {
            sum += tri_area_heron(poly[0], poly[i], poly[i + 1]);
        }
        sum
    }

    #[test]
    fn signed_area_is_zero_for_too_few_vertices() {
        assert_eq!(signed_area(&[]), 0.0);
        assert_eq!(signed_area(&[(1.0, 2.0)]), 0.0);
        assert_eq!(signed_area(&[(1.0, 2.0), (3.0, 4.0)]), 0.0);
        assert_eq!(area(&[(1.0, 2.0), (3.0, 4.0)]), 0.0);
        assert_eq!(centroid(&[(1.0, 2.0), (3.0, 4.0)]), None);
    }

    #[test]
    fn unit_square_area_and_centroid() {
        let sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!((signed_area(&sq) - 1.0).abs() < 1e-6);
        assert!(signed_area(&sq) > 0.0, "CCW winding is positive");
        let (cx, cy) = centroid(&sq).unwrap();
        assert!((cx - 0.5).abs() < 1e-6 && (cy - 0.5).abs() < 1e-6);
    }

    #[test]
    fn winding_reversal_flips_sign_but_not_area_or_centroid() {
        let sq = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)];
        let mut rev = sq.to_vec();
        rev.reverse();
        assert!((signed_area(&sq) + signed_area(&rev)).abs() < 1e-6);
        assert!((area(&sq) - area(&rev)).abs() < 1e-6);
        let a = centroid(&sq).unwrap();
        let b = centroid(&rev).unwrap();
        assert!((a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6);
    }

    #[test]
    fn triangle_centroid_is_vertex_average() {
        // For a triangle the area-weighted centroid equals the plain average of
        // the three vertices — an independent closed form.
        let mut state = 0x1111_2222_3333_4444_u64;
        for _ in 0..200 {
            let a = (rand_in(&mut state, -10.0, 10.0), rand_in(&mut state, -10.0, 10.0));
            let b = (rand_in(&mut state, -10.0, 10.0), rand_in(&mut state, -10.0, 10.0));
            let c = (rand_in(&mut state, -10.0, 10.0), rand_in(&mut state, -10.0, 10.0));
            let tri = [a, b, c];
            if area(&tri) < 1e-2 {
                continue; // skip near-degenerate triangles
            }
            let (cx, cy) = centroid(&tri).unwrap();
            let ax = (a.0 + b.0 + c.0) / 3.0;
            let ay = (a.1 + b.1 + c.1) / 3.0;
            assert!((cx - ax).abs() < 1e-3 && (cy - ay).abs() < 1e-3);
        }
    }

    #[test]
    fn shoelace_area_matches_heron_fan() {
        // THE core oracle: cross-product shoelace vs side-length Heron fan.
        let mut state = 0xDEAD_BEEF_0BAD_F00D_u64;
        for _ in 0..200 {
            let poly = random_convex(&mut state);
            let shoelace = area(&poly);
            let heron = fan_area_heron(&poly);
            assert!(
                (shoelace - heron).abs() <= 1e-2 * heron.max(1.0),
                "shoelace {shoelace} vs heron {heron}"
            );
        }
    }

    #[test]
    fn area_is_translation_invariant() {
        let mut state = 0x0F0F_0F0F_1E1E_1E1E_u64;
        for _ in 0..200 {
            let poly = random_convex(&mut state);
            let before = area(&poly);
            let (tx, ty) = (rand_in(&mut state, -50.0, 50.0), rand_in(&mut state, -50.0, 50.0));
            let moved: Vec<_> = poly.iter().map(|&(x, y)| (x + tx, y + ty)).collect();
            let after = area(&moved);
            assert!((before - after).abs() <= 1e-2 * before.max(1.0));
        }
    }

    #[test]
    fn area_scales_quadratically() {
        let mut state = 0xC0FF_EE00_1234_5678_u64;
        for _ in 0..200 {
            let poly = random_convex(&mut state);
            let base = area(&poly);
            let k = rand_in(&mut state, 0.25, 4.0);
            let scaled: Vec<_> = poly.iter().map(|&(x, y)| (x * k, y * k)).collect();
            let got = area(&scaled);
            let expect = base * k * k;
            assert!((got - expect).abs() <= 1e-2 * expect.max(1.0), "k {k}: {got} vs {expect}");
        }
    }

    #[test]
    fn area_is_invariant_under_quarter_turn() {
        // Rotating by 90 degrees via (x, y) -> (-y, x) is exact (no sin/cos) and
        // preserves both orientation and magnitude of the signed area.
        let mut state = 0xABAB_CDCD_EFEF_0101_u64;
        for _ in 0..200 {
            let poly = random_convex(&mut state);
            let before = signed_area(&poly);
            let turned: Vec<_> = poly.iter().map(|&(x, y)| (-y, x)).collect();
            let after = signed_area(&turned);
            assert!((before - after).abs() <= 1e-2 * before.abs().max(1.0));
        }
    }

    #[test]
    fn centroid_is_translation_covariant() {
        let mut state = 0x5555_AAAA_5555_AAAA_u64;
        for _ in 0..200 {
            let poly = random_convex(&mut state);
            let (cx, cy) = centroid(&poly).unwrap();
            let (tx, ty) = (rand_in(&mut state, -40.0, 40.0), rand_in(&mut state, -40.0, 40.0));
            let moved: Vec<_> = poly.iter().map(|&(x, y)| (x + tx, y + ty)).collect();
            let (mx, my) = centroid(&moved).unwrap();
            assert!((mx - (cx + tx)).abs() <= 1e-2 && (my - (cy + ty)).abs() <= 1e-2);
        }
    }

    #[test]
    fn centroid_is_invariant_under_cyclic_rotation() {
        let mut state = 0x7777_8888_9999_AAAA_u64;
        for _ in 0..200 {
            let poly = random_convex(&mut state);
            let (cx, cy) = centroid(&poly).unwrap();
            let shift = 1 + (next_rand(&mut state) as usize % poly.len());
            let mut rotated = Vec::with_capacity(poly.len());
            rotated.extend_from_slice(&poly[shift..]);
            rotated.extend_from_slice(&poly[..shift]);
            let (rx, ry) = centroid(&rotated).unwrap();
            assert!((rx - cx).abs() <= 1e-2 && (ry - cy).abs() <= 1e-2);
        }
    }

    #[test]
    fn collinear_run_has_no_centroid() {
        let line = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        assert_eq!(centroid(&line), None);
        assert!(area(&line).abs() < 1e-6);
    }
}
