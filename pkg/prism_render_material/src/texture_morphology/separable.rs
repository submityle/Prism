//! Separable grayscale morphology with a flat `(2*radius + 1)`-square
//! structuring element (SE).
//!
//! Morphological **dilation** (local max) and **erosion** (local min) are the
//! fixed-function workhorses for operating on coverage masks and signed
//! distance fields (SDFs): growing/shrinking alpha-test or decal coverage,
//! closing pin-holes and opening speckle out of a mask, computing a one-pass
//! outline (`dilate - src`), and the max/min half of a jump-flood / chamfer SDF
//! sweep. For a *flat* SE every tap weight is 0, so the operator is a pure
//! rank filter -- no gamma, no normalisation.
//!
//! The square flat SE is **separable**: a 2D box min equals a per-row 1D min
//! followed by a per-column 1D min (and likewise max), because the minimum over
//! a rectangle is the minimum of the per-row minima taken over the window. This
//! module therefore runs an X pass then a Y pass, exactly like the separable
//! [`box_blur_plane`](crate::box_blur_plane), with border taps resolved through
//! a [`WrapMode`].
//!
//! Four operators are exposed:
//! * [`dilate_plane`] -- local max; **extensive** (`out >= src`) for a
//!   border mode that never invents smaller samples.
//! * [`erode_plane`] -- local min; **anti-extensive** (`out <= src`).
//! * [`open_plane`] -- erosion then dilation; removes bright speckle, is
//!   **idempotent** (`open(open(x)) == open(x)`).
//! * [`close_plane`] -- dilation then erosion; fills dark pin-holes.
//!
//! Dilation and erosion are **dual** under negation: dilating `x` equals
//! negating the erosion of `-x`. Both are **monotone**: `a <= b` elementwise
//! implies `dilate(a) <= dilate(b)` and `erode(a) <= erode(b)`. The oracles
//! below pin all of these down, and cross-check the separable two-pass result
//! against an independent full-2D square-window scan so the separation itself
//! is proven, not assumed.
//!
//! Everything is deterministic analytic `f32` arithmetic -- no AI/ML -- so a CPU
//! golden matches a GPU compute morphology pass exactly.
//!
//! # References
//! * Haralick, Sternberg & Zhuang, "Image Analysis Using Mathematical
//!   Morphology" (1987).
//! * van Herk, "A fast algorithm for local minimum and maximum filters ..."
//!   (1992) -- the O(1)/pixel separable form this mirrors at O(n*r).

use alloc::vec;
use alloc::vec::Vec;

use crate::WrapMode;

/// Which rank the flat structuring element selects.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    /// Local maximum (dilation).
    Max,
    /// Local minimum (erosion).
    Min,
}

/// Wrap an integer tap index into `[0, n)` (same convention as the box blur):
/// `Repeat` / `MirroredRepeat` tile, every other mode clamps to the edge.
#[inline]
#[must_use]
fn wrap_index(i: i64, n: i64, mode: WrapMode) -> usize {
    match mode {
        WrapMode::Repeat => i.rem_euclid(n) as usize,
        WrapMode::MirroredRepeat => {
            let p = i.rem_euclid(2 * n);
            let m = if p >= n { 2 * n - 1 - p } else { p };
            m as usize
        }
        _ => i.clamp(0, n - 1) as usize,
    }
}

/// Rank-filter one length-`n` line (stride `step`, starting at `base`) of
/// `plane` into `out` with a `(2*radius + 1)`-tap flat SE.
fn morph_line(
    plane: &[f32],
    out: &mut [f32],
    base: usize,
    step: usize,
    n: i64,
    radius: i64,
    wrap: WrapMode,
    op: Op,
) {
    let fetch = |x: i64| -> f32 { plane[base + wrap_index(x, n, wrap) * step] };
    for x in 0..n {
        let mut acc = fetch(x - radius);
        for j in (x - radius + 1)..=(x + radius) {
            let s = fetch(j);
            acc = match op {
                Op::Max => acc.max(s),
                Op::Min => acc.min(s),
            };
        }
        out[base + x as usize * step] = acc;
    }
}

/// Shared separable X-then-Y rank filter.
#[must_use]
fn morph_plane(
    plane: &[f32],
    width: u32,
    height: u32,
    radius: u32,
    wrap: WrapMode,
    op: Op,
) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || plane.len() != w * h || radius == 0 {
        return plane.to_vec();
    }
    let r = radius as i64;

    // X pass: plane -> tmp (rows contiguous, step 1).
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        morph_line(plane, &mut tmp, y * w, 1, width as i64, r, wrap, op);
    }
    // Y pass: tmp -> out (columns stride w).
    let mut out = vec![0.0f32; w * h];
    for x in 0..w {
        morph_line(&tmp, &mut out, x, w, height as i64, r, wrap, op);
    }
    out
}

/// Grayscale **dilation** (local max) of a single-channel `width * height`
/// row-major plane with a flat `(2*radius + 1)`-square structuring element,
/// resolving borders through `wrap`.
///
/// Returns a copy when `radius == 0` or either dimension is `0`.
#[must_use]
pub fn dilate_plane(
    plane: &[f32],
    width: u32,
    height: u32,
    radius: u32,
    wrap: WrapMode,
) -> Vec<f32> {
    morph_plane(plane, width, height, radius, wrap, Op::Max)
}

/// Grayscale **erosion** (local min) of a single-channel `width * height`
/// row-major plane with a flat `(2*radius + 1)`-square structuring element,
/// resolving borders through `wrap`.
///
/// Returns a copy when `radius == 0` or either dimension is `0`.
#[must_use]
pub fn erode_plane(
    plane: &[f32],
    width: u32,
    height: u32,
    radius: u32,
    wrap: WrapMode,
) -> Vec<f32> {
    morph_plane(plane, width, height, radius, wrap, Op::Min)
}

/// Morphological **opening** (erosion then dilation): removes bright speckle
/// smaller than the SE while preserving larger shapes. Idempotent.
#[must_use]
pub fn open_plane(plane: &[f32], width: u32, height: u32, radius: u32, wrap: WrapMode) -> Vec<f32> {
    let eroded = erode_plane(plane, width, height, radius, wrap);
    dilate_plane(&eroded, width, height, radius, wrap)
}

/// Morphological **closing** (dilation then erosion): fills dark pin-holes
/// smaller than the SE while preserving larger shapes.
#[must_use]
pub fn close_plane(
    plane: &[f32],
    width: u32,
    height: u32,
    radius: u32,
    wrap: WrapMode,
) -> Vec<f32> {
    let dilated = dilate_plane(plane, width, height, radius, wrap);
    erode_plane(&dilated, width, height, radius, wrap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const WRAPS: [WrapMode; 3] = [
        WrapMode::Repeat,
        WrapMode::MirroredRepeat,
        WrapMode::ClampToEdge,
    ];

    /// Deterministic pseudo-random plane in `[0, 1)` (no transcendentals).
    fn noise_plane(w: u32, h: u32) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut v = Vec::with_capacity(n);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            v.push(((state >> 40) as u32 as f32) / (1u64 << 24) as f32);
        }
        v
    }

    /// Independent full-2D square-window rank filter (the separability oracle).
    fn brute_morph(plane: &[f32], w: u32, h: u32, r: i64, wrap: WrapMode, op: Op) -> Vec<f32> {
        let wi = w as i64;
        let hi = h as i64;
        let mut out = vec![0.0f32; (w * h) as usize];
        for y in 0..hi {
            for x in 0..wi {
                let mut acc = if op == Op::Max {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                };
                for dy in -r..=r {
                    for dx in -r..=r {
                        let sx = wrap_index(x + dx, wi, wrap);
                        let sy = wrap_index(y + dy, hi, wrap);
                        let s = plane[sy * w as usize + sx];
                        acc = if op == Op::Max {
                            acc.max(s)
                        } else {
                            acc.min(s)
                        };
                    }
                }
                out[(y * wi + x) as usize] = acc;
            }
        }
        out
    }

    #[test]
    fn radius_zero_is_identity() {
        let (w, h) = (6u32, 5u32);
        let p = noise_plane(w, h);
        for &wrap in &WRAPS {
            assert_eq!(dilate_plane(&p, w, h, 0, wrap), p);
            assert_eq!(erode_plane(&p, w, h, 0, wrap), p);
        }
    }

    #[test]
    fn constant_plane_is_preserved() {
        let (w, h) = (7u32, 6u32);
        let p = vec![0.37f32; (w * h) as usize];
        for &wrap in &WRAPS {
            for r in 1..=3u32 {
                assert_eq!(dilate_plane(&p, w, h, r, wrap), p);
                assert_eq!(erode_plane(&p, w, h, r, wrap), p);
                assert_eq!(open_plane(&p, w, h, r, wrap), p);
                assert_eq!(close_plane(&p, w, h, r, wrap), p);
            }
        }
    }

    #[test]
    fn separable_matches_full_2d_scan() {
        let (w, h) = (11u32, 9u32);
        let p = noise_plane(w, h);
        for &wrap in &WRAPS {
            for r in 1..=3i64 {
                let d = dilate_plane(&p, w, h, r as u32, wrap);
                let e = erode_plane(&p, w, h, r as u32, wrap);
                let bd = brute_morph(&p, w, h, r, wrap, Op::Max);
                let be = brute_morph(&p, w, h, r, wrap, Op::Min);
                for i in 0..(w * h) as usize {
                    assert!((d[i] - bd[i]).abs() < 1.0e-6, "dilate r={r} i={i}");
                    assert!((e[i] - be[i]).abs() < 1.0e-6, "erode r={r} i={i}");
                }
            }
        }
    }

    #[test]
    fn dilation_erosion_are_dual_under_negation() {
        let (w, h) = (10u32, 8u32);
        let p = noise_plane(w, h);
        let neg: Vec<f32> = p.iter().map(|&x| -x).collect();
        for &wrap in &WRAPS {
            for r in 1..=3u32 {
                let d = dilate_plane(&p, w, h, r, wrap);
                let e_neg = erode_plane(&neg, w, h, r, wrap);
                for i in 0..(w * h) as usize {
                    assert!((d[i] - (-e_neg[i])).abs() < 1.0e-6, "duality r={r} i={i}");
                }
            }
        }
    }

    #[test]
    fn dilate_is_extensive_erode_is_anti_extensive() {
        let (w, h) = (9u32, 9u32);
        let p = noise_plane(w, h);
        for &wrap in &WRAPS {
            for r in 1..=3u32 {
                let d = dilate_plane(&p, w, h, r, wrap);
                let e = erode_plane(&p, w, h, r, wrap);
                for i in 0..(w * h) as usize {
                    assert!(d[i] >= p[i] - 1.0e-6, "dilate>=src");
                    assert!(e[i] <= p[i] + 1.0e-6, "erode<=src");
                }
            }
        }
    }

    #[test]
    fn operators_are_monotone() {
        let (w, h) = (8u32, 7u32);
        let a = noise_plane(w, h);
        // b >= a everywhere (add a non-negative perturbation).
        let bump = noise_plane(w, h);
        let b: Vec<f32> = a.iter().zip(&bump).map(|(&x, &k)| x + k).collect();
        for &wrap in &WRAPS {
            for r in 1..=3u32 {
                let da = dilate_plane(&a, w, h, r, wrap);
                let db = dilate_plane(&b, w, h, r, wrap);
                let ea = erode_plane(&a, w, h, r, wrap);
                let eb = erode_plane(&b, w, h, r, wrap);
                for i in 0..(w * h) as usize {
                    assert!(db[i] >= da[i] - 1.0e-6, "dilate monotone");
                    assert!(eb[i] >= ea[i] - 1.0e-6, "erode monotone");
                }
            }
        }
    }

    #[test]
    fn opening_is_idempotent_and_bounded_by_closing() {
        let (w, h) = (12u32, 10u32);
        let p = noise_plane(w, h);
        for &wrap in &WRAPS {
            for r in 1..=3u32 {
                let o = open_plane(&p, w, h, r, wrap);
                let oo = open_plane(&o, w, h, r, wrap);
                let c = close_plane(&p, w, h, r, wrap);
                for i in 0..(w * h) as usize {
                    assert!((oo[i] - o[i]).abs() < 1.0e-6, "open idempotent i={i}");
                    assert!(o[i] <= p[i] + 1.0e-6, "open<=src");
                    assert!(c[i] >= p[i] - 1.0e-6, "close>=src");
                }
            }
        }
    }

    #[test]
    fn empty_or_mismatched_plane_is_copied() {
        assert_eq!(
            dilate_plane(&[], 0, 0, 2, WrapMode::Repeat),
            Vec::<f32>::new()
        );
        let p = vec![0.5f32; 10];
        assert_eq!(erode_plane(&p, 4, 4, 1, WrapMode::Repeat), p);
    }
}
