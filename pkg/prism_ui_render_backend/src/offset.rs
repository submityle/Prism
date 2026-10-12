//! Convex polygon offsetting: outward growth and inward shrink.
//!
//! UI needs the same ring at a fixed perpendicular distance from its edges — an
//! inset border (making padding visible), a stroked outline around a convex
//! fill, a hit-test that tolerates a margin, or a drop shadow's silhouette. For
//! a **convex** ring this is exact and cheap: shift every edge line by
//! `distance` along its outward normal and re-intersect adjacent lines
//! ([`offset_convex`]). The result keeps the input's vertex count and winding.
//!
//! Corners join with a **miter** (the sharp intersection of the two adjacent
//! offset lines, as in `stroke-linejoin: miter`), so each output edge lies at
//! distance `distance` from its input edge and each output corner is
//! `distance` from both incident edge *lines*. The miter is not the same as an
//! outward dilation by a disc (that rounds corners with arcs); pick this when
//! the caller wants sharp offset geometry.
//!
//! Positive `distance` grows the ring (outset), negative shrinks it (inset). An
//! inset past the polygon's inradius collapses to nothing and returns `None`, as
//! does any degenerate input (fewer than three vertices, a zero-area ring, a
//! non-convex ring, or a non-finite `distance`). Straight (collinear) vertices
//! are dropped first, exactly as they are for [`is_convex`] and
//! [`convex_hull`](crate::convex_hull), so a ring with a redundant midpoint
//! offsets like the ring without it.
//!
//! Non-convex rings are **not** supported: offsetting them can self-intersect,
//! which needs a full polygon-offset algorithm (miter/bevel/round joins plus
//! self-intersection removal) that this module deliberately does not pretend to
//! be. [`is_convex`] and
//! [`is_simple_polygon`](crate::simplicity::is_simple_polygon) are both checked
//! up front, so a concave *or* self-intersecting ring returns `None` (a star
//! ring is convex-by-turns but not simple).
//!
//! Only `+ - * /` and `sqrt` on `f32` are used — no transcendentals — so the
//! routine is `no_std`-clean and bit-stable.

use alloc::vec::Vec;

use crate::convexity::is_convex;
use crate::measure::signed_area;
use crate::simplicity::is_simple_polygon;

/// Below this normalized magnitude a turn or direction is treated as degenerate.
const EPS: f32 = 1e-6;

use crate::vec2::{cross, dot, length, sub};

/// Drop vertices whose two incident edges are (near-)collinear or zero-length,
/// preserving order. Returns `None` when fewer than three vertices survive.
fn drop_collinear(ring: &[(f32, f32)]) -> Option<Vec<(f32, f32)>> {
    let n = ring.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = ring[(i + n - 1) % n];
        let cur = ring[i];
        let next = ring[(i + 1) % n];
        let a = sub(cur, prev);
        let b = sub(next, cur);
        let (la, lb) = (length(a), length(b));
        if la <= EPS || lb <= EPS {
            continue;
        }
        // |sin(turn angle)| — scale-free, so a shallow-but-real corner survives.
        if cross(a, b).abs() / (la * lb) > EPS {
            out.push(cur);
        }
    }
    if out.len() < 3 { None } else { Some(out) }
}

/// Offset a convex polygon by `distance` with **miter** corners: positive grows
/// it outward, negative shrinks it inward.
///
/// Returns the offset ring in counter-clockwise order (the input may be either
/// winding), or `None` when the result is empty or the input is unusable.
/// Collinear vertices are dropped, so the output can have fewer vertices than a
/// redundant input. For an inset that lands exactly on the inradius the result
/// is a degenerate point and is reported as `None`; a negative `distance` whose
/// magnitude exceeds the inradius also returns `None`.
///
/// Every output edge lies on the line parallel to the corresponding input edge
/// at perpendicular distance `|distance|`, shifted outward for a positive
/// `distance` and inward for a negative one.
///
/// # Preconditions
///
/// The ring must be simple and convex; a concave or self-intersecting input
/// returns `None` rather than a wrong answer.
#[must_use]
pub fn offset_convex(verts: &[(f32, f32)], distance: f32) -> Option<Vec<(f32, f32)>> {
    if verts.len() < 3 || !distance.is_finite() {
        return None;
    }

    // Normalize to counter-clockwise so every outward normal is a right turn.
    let mut ring = verts.to_vec();
    if signed_area(&ring) < 0.0 {
        ring.reverse();
    }
    let ring = drop_collinear(&ring)?;
    // Both guards are required: `is_convex` reports a self-intersecting
    // "star" ring convex (its own docs warn about this), so a pentagram would
    // otherwise be offset into nonsense. `is_simple_polygon` rejects it.
    if !is_convex(&ring) || !is_simple_polygon(&ring) {
        return None;
    }

    let n = ring.len();
    let mut edge = Vec::with_capacity(n);
    let mut normal = Vec::with_capacity(n);
    for i in 0..n {
        let dir = sub(ring[(i + 1) % n], ring[i]);
        let len = length(dir);
        if len <= EPS {
            return None;
        }
        let unit = (dir.0 / len, dir.1 / len);
        edge.push(unit);
        // Outward normal of a CCW edge (e.x, e.y) is its -90° rotation.
        normal.push((unit.1, -unit.0));
    }

    // Shift each edge line outward to a point on the offset line.
    let mut base = Vec::with_capacity(n);
    for i in 0..n {
        base.push((
            ring[i].0 + normal[i].0 * distance,
            ring[i].1 + normal[i].1 * distance,
        ));
    }

    // The offset vertex at ring[i] is the intersection of the two adjacent
    // offset lines.
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = (i + n - 1) % n;
        let (a, u) = (base[prev], edge[prev]);
        let (b, v) = (base[i], edge[i]);
        let denom = cross(u, v);
        if denom.abs() <= EPS {
            // Adjacent edges nearly parallel: the corner is ill-defined.
            return None;
        }
        let t = cross(sub(b, a), v) / denom;
        out.push((a.0 + u.0 * t, a.1 + u.1 * t));
    }

    // A valid offset runs the same way as the input. When an inset is too deep
    // the intersection polygon inverts, so some edge reverses; reject it.
    for i in 0..n {
        let out_edge = sub(out[(i + 1) % n], out[i]);
        if dot(out_edge, sub(ring[(i + 1) % n], ring[i])) <= 0.0 {
            return None;
        }
    }

    Some(out)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::hull::convex_hull;
    use crate::inside::point_in_polygon;

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

    /// A centered regular `n`-gon in CCW order (f64 trig, cast to `f32`).
    fn regular_polygon(n: usize, radius: f32) -> Vec<(f32, f32)> {
        let r = f64::from(radius);
        (0..n)
            .map(|k| {
                let a = core::f64::consts::TAU * (k as f64) / (n as f64);
                ((r * a.cos()) as f32, (r * a.sin()) as f32)
            })
            .collect()
    }

    /// Perpendicular distance from `p` to the infinite line through `a` and `b`.
    fn line_dist(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
        let e = sub(b, a);
        let l = length(e);
        if l <= 1e-9 {
            return length(sub(p, a));
        }
        cross(e, sub(p, a)).abs() / l
    }

    #[test]
    fn unit_square_outset_and_inset() {
        let sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let out = offset_convex(&sq, 1.0).expect("outset");
        let expect = [(-1.0, -1.0), (2.0, -1.0), (2.0, 2.0), (-1.0, 2.0)];
        for (a, b) in out.iter().zip(expect) {
            assert!((a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4, "{a:?}");
        }
        let ins = offset_convex(&sq, -0.25).expect("inset");
        let expect = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)];
        for (a, b) in ins.iter().zip(expect) {
            assert!((a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4, "{a:?}");
        }
    }

    #[test]
    fn inset_past_inradius_collapses_to_none() {
        let sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(offset_convex(&sq, -0.25).is_some());
        assert!(offset_convex(&sq, -0.4).is_some());
        assert!(offset_convex(&sq, -0.6).is_none());
        assert!(offset_convex(&sq, -2.0).is_none());
        // Regular n-gon inradius is R*cos(pi/n): 10% either side of it.
        for n in 5..=12 {
            let radius = 10.0_f64;
            let inradius = radius * (core::f64::consts::PI / n as f64).cos();
            let p = regular_polygon(n, radius as f32);
            let inside = inradius as f32 * 0.9;
            let outside = inradius as f32 * 1.1;
            assert!(offset_convex(&p, -inside).is_some(), "n={n} inside");
            assert!(offset_convex(&p, -outside).is_none(), "n={n} outside");
        }
    }

    #[test]
    fn edges_and_corners_sit_at_distance_d() {
        // The definition of a miter offset: each output edge is the input edge
        // line shifted by `d`, and each output corner is `d` from both incident
        // input edge lines.
        let mut cases: Vec<Vec<(f32, f32)>> =
            (5..=12).map(|n| regular_polygon(n, 10.0)).collect();
        let mut s = 0x5_C11A_u64;
        for _ in 0..40 {
            cases.push(random_convex(&mut s));
        }
        for p in &cases {
            let n = p.len();
            for &d in &[0.5_f32, 1.0] {
                let out = offset_convex(p, d).expect("outset");
                for i in 0..n {
                    let edge = line_dist(p[i], out[i], out[(i + 1) % n]);
                    assert!((edge - d).abs() < 1e-3, "outset edge {i} at {edge} vs {d}");
                    let a = line_dist(out[i], p[(i + n - 1) % n], p[i]);
                    let b = line_dist(out[i], p[i], p[(i + 1) % n]);
                    assert!(
                        (a - d).abs() < 1e-3 && (b - d).abs() < 1e-3,
                        "outset corner {i}: {a}, {b} vs {d}"
                    );
                }
                if let Some(ins) = offset_convex(p, -d) {
                    for i in 0..n {
                        let edge = line_dist(p[i], ins[i], ins[(i + 1) % n]);
                        assert!((edge - d).abs() < 1e-3, "inset edge {i} at {edge} vs {d}");
                    }
                }
            }
        }
    }

    #[test]
    fn offset_edges_stay_parallel_to_input() {
        let mut s = 0x5_C11A_u64;
        for _ in 0..100 {
            let p = random_convex(&mut s);
            let n = p.len();
            for &d in &[0.4_f32, 1.0, -0.3] {
                let Some(out) = offset_convex(&p, d) else {
                    continue;
                };
                assert_eq!(out.len(), n);
                for i in 0..n {
                    let oe = sub(out[(i + 1) % n], out[i]);
                    let ie = sub(p[(i + 1) % n], p[i]);
                    let cr = cross(oe, ie).abs();
                    let scale = length(oe) * length(ie);
                    assert!(cr <= 1e-3 * scale.max(1e-6), "edge {i} not parallel");
                }
            }
        }
    }

    #[test]
    fn outset_then_same_inset_returns_input() {
        let mut s = 0x5_C11A_u64;
        for _ in 0..100 {
            let p = random_convex(&mut s);
            for &d in &[0.5_f32, 1.0] {
                let out = offset_convex(&p, d).expect("outset");
                let back = offset_convex(&out, -d).expect("re-inset");
                assert_eq!(back.len(), p.len());
                for (a, b) in back.iter().zip(p.iter()) {
                    assert!(
                        (a.0 - b.0).abs() < 1e-2 && (a.1 - b.1).abs() < 1e-2,
                        "round trip {a:?} vs {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn outset_contains_input_and_inset_is_contained() {
        let mut s = 0x5_C11A_u64;
        for _ in 0..100 {
            let p = random_convex(&mut s);
            let out = offset_convex(&p, 0.8).expect("outset");
            for &v in &p {
                assert!(point_in_polygon(&out, v), "outset misses {v:?}");
            }
            if let Some(ins) = offset_convex(&p, -0.4) {
                for &v in &ins {
                    assert!(point_in_polygon(&p, v), "inset escapes {v:?}");
                }
            }
        }
    }

    #[test]
    fn translation_invariant_and_scale_covariant() {
        let p = regular_polygon(7, 10.0);
        let base = offset_convex(&p, 1.0).expect("outset");
        let moved: Vec<_> = p.iter().map(|&(x, y)| (x + 5.0, y - 3.0)).collect();
        let moved_out = offset_convex(&moved, 1.0).expect("moved outset");
        for (a, b) in base.iter().zip(moved_out.iter()) {
            assert!((a.0 + 5.0 - b.0).abs() < 1e-3 && (a.1 - 3.0 - b.1).abs() < 1e-3);
        }
        let scaled: Vec<_> = p.iter().map(|&(x, y)| (x * 2.0, y * 2.0)).collect();
        let scaled_out = offset_convex(&scaled, 2.0).expect("scaled outset");
        for (a, b) in base.iter().zip(scaled_out.iter()) {
            assert!((a.0 * 2.0 - b.0).abs() < 1e-2 && (a.1 * 2.0 - b.1).abs() < 1e-2);
        }
    }

    #[test]
    fn winding_and_collinear_vertices_are_normalized() {
        let sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let mut cw = sq.to_vec();
        cw.reverse();
        let a = offset_convex(&sq, 0.5).expect("ccw");
        let b = offset_convex(&cw, 0.5).expect("cw");
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.0 - y.0).abs() < 1e-5 && (x.1 - y.1).abs() < 1e-5);
        }
        // A collinear midpoint is a redundant vertex, not a corner.
        let with_mid = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let out = offset_convex(&with_mid, 0.5).expect("collinear");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn zero_distance_is_identity() {
        let p = regular_polygon(6, 5.0);
        let out = offset_convex(&p, 0.0).expect("identity");
        for (a, b) in out.iter().zip(p.iter()) {
            assert!((a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4);
        }
    }

    #[test]
    fn rejects_degenerate_and_concave_inputs() {
        assert!(offset_convex(&[], 1.0).is_none());
        assert!(offset_convex(&[(0.0, 0.0), (1.0, 0.0)], 1.0).is_none());
        // All collinear: no area.
        assert!(offset_convex(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], 1.0).is_none());
        // Concave L-shape is refused rather than mis-offset.
        let l = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)];
        assert!(offset_convex(&l, 0.5).is_none());
        // Non-finite distance.
        let sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(offset_convex(&sq, f32::NAN).is_none());
        assert!(offset_convex(&sq, f32::INFINITY).is_none());
    }

    #[test]
    fn rejects_a_self_intersecting_star_ring() {
        // A pentagram traced as a star turns the same way at every vertex, so
        // `is_convex` alone accepts it — the simplicity guard must reject it
        // rather than offset a self-intersecting ring into nonsense.
        let pentagon = regular_polygon(5, 10.0);
        let star = [
            pentagon[0],
            pentagon[2],
            pentagon[4],
            pentagon[1],
            pentagon[3],
        ];
        assert!(
            is_convex(&star),
            "the star ring is convex-by-turns"
        );
        assert!(!is_simple_polygon(&star));
        assert!(offset_convex(&star, 1.0).is_none());
    }
}
