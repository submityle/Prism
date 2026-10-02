//! Cubic B-spline smoothing filter for the manual texture sampler, with the
//! Sigg & Hadwiger fast four-bilinear-tap decomposition.
//!
//! [`super::bicubic`] (Catmull-Rom) *interpolates* and therefore overshoots
//! near steep edges (ringing). The uniform cubic B-spline is the complementary
//! choice: an *approximating* kernel whose four weights are all non-negative and
//! sum to one, so the result is a convex blend of the taps and can never over-
//! or undershoot the local texel range. That makes it the AAA choice for smooth
//! upsampling of heightfields / terrain / data textures and for a soft prefilter
//! where ringing is unacceptable.
//!
//! Because every B-spline weight is non-negative, each adjacent weight pair can
//! be folded into a single linearly-filtered fetch: a 2D cubic collapses from 16
//! texel reads to **four** hardware-bilinear reads (Sigg & Hadwiger, GPU Gems 2).
//! [`bspline_cubic_fast`] implements exactly that using the already-tested
//! [`super::bilinear`] fetch, and [`bspline_cubic`] is the straightforward
//! separable 16-tap reference. The two are derived completely differently (tap
//! folding through `bilinear` vs. a direct weighted sum) yet must agree to float
//! precision, giving a non-circular correctness cross-check.
//!
//! Pure `f32` arithmetic with defensive non-finite guards and no AI/ML path, so
//! a CPU golden reproduces a GPU twin (which uses the same four-tap folding)
//! exactly.
//!
//! # Conventions
//! * Texel-centre and mip-sizing conventions match [`super::bilinear`]: the
//!   sample position in texels is `uv * dim - 0.5`, the axis size at `mip` is
//!   `max(1, base_dim >> mip)`.
//! * The 4x4 footprint uses offsets `-1, 0, +1, +2` from `floor(sample)`; each
//!   integer index is folded through [`super::texel_wrap`] exactly like a
//!   bilinear corner, and a [`WrapMode::ClampToBorder`] tap contributes the
//!   caller's border colour with its (non-negative) weight.
//! * Non-finite UVs collapse to `0.0`, never `NaN`.
//!
//! # References
//! * Sigg & Hadwiger, "Fast Third-Order Texture Filtering", GPU Gems 2, ch. 20.
//! * Ruijters, van der Zwaan et al., "Efficient GPU-Based Texture Interpolation
//!   using Uniform B-Splines", J. Graphics Tools 13(4) (2008).
//! * de Boor, *A Practical Guide to Splines* (1978) — uniform cubic B-spline.

use super::super::texture_addressing::WrapMode;
use super::bilinear::{bilinear, dim_at, TexelSource};
use super::texel_wrap::{wrap_texel, TexelAddr};

#[inline]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// The four uniform cubic B-spline weights for a fractional offset `t` in
/// `[0, 1)`, applied to the taps at integer offsets `-1, 0, +1, +2`.
///
/// All four weights are non-negative and sum to 1 for every `t` (a convex
/// partition of unity), and the set is symmetric: reversing `weights(t)` yields
/// `weights(1 - t)`.
#[inline]
#[must_use]
pub fn bspline_cubic_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let om = 1.0 - t;
    [
        (om * om * om) / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

#[inline]
fn accumulate(acc: &mut [f32; 4], c: [f32; 4], w: f32) {
    acc[0] += c[0] * w;
    acc[1] += c[1] * w;
    acc[2] += c[2] * w;
    acc[3] += c[3] * w;
}

/// Separable 16-tap cubic B-spline reference fetch of `src` at mip `mip`.
///
/// Fetches the 4x4 texel neighbourhood (folding each index through the per-axis
/// wrap modes, substituting `border_color` for any out-of-range
/// [`WrapMode::ClampToBorder`] tap) and blends it with the non-negative B-spline
/// kernel. The result is a convex combination of the taps, so it never over- or
/// undershoots the local texel range.
#[must_use]
pub fn bspline_cubic<S: TexelSource>(
    src: &S,
    mip: u32,
    uv: [f32; 2],
    wrap_u: WrapMode,
    wrap_v: WrapMode,
    border_color: [f32; 4],
) -> [f32; 4] {
    let (bw, bh) = src.dimensions();
    let w = dim_at(bw.max(1), mip);
    let h = dim_at(bh.max(1), mip);

    let fx = finite_or_zero(uv[0]) * w as f32 - 0.5;
    let fy = finite_or_zero(uv[1]) * h as f32 - 0.5;
    let x1 = fx.floor();
    let y1 = fy.floor();
    let (ix, iy) = (x1 as i64, y1 as i64);

    let wx = bspline_cubic_weights(fx - x1);
    let wy = bspline_cubic_weights(fy - y1);

    let fetch = |ox: i64, oy: i64| -> [f32; 4] {
        match (wrap_texel(ix + ox, w, wrap_u), wrap_texel(iy + oy, h, wrap_v)) {
            (TexelAddr::In(cx), TexelAddr::In(cy)) => src.texel(mip, cx, cy),
            _ => border_color,
        }
    };

    const OFFS: [i64; 4] = [-1, 0, 1, 2];
    let mut out = [0.0_f32; 4];
    for (ri, &oy) in OFFS.iter().enumerate() {
        let mut row = [0.0_f32; 4];
        for (ci, &ox) in OFFS.iter().enumerate() {
            accumulate(&mut row, fetch(ox, oy), wx[ci]);
        }
        accumulate(&mut out, row, wy[ri]);
    }
    out
}

/// One axis of the fast decomposition: fold the two weight pairs into a combined
/// weight and a fractional sample position (in texel space) for each pair.
///
/// Returns `(g0, g1, p0, p1)` where `g0 = w0 + w1`, `g1 = w2 + w3`, `p0` lands
/// between taps `floor-1` and `floor`, and `p1` between taps `floor+1` and
/// `floor+2`. Denominators are guarded away from zero (B-spline weight pair
/// sums are always `>= 1/6`, but the guard keeps the function total).
#[inline]
fn fold_axis(floor_i: f32, w: [f32; 4]) -> (f32, f32, f32, f32) {
    let g0 = w[0] + w[1];
    let g1 = w[2] + w[3];
    let a = if g0 > 1.0e-8 { w[1] / g0 } else { 0.0 };
    let b = if g1 > 1.0e-8 { w[3] / g1 } else { 0.0 };
    let p0 = floor_i - 1.0 + a;
    let p1 = floor_i + 1.0 + b;
    (g0, g1, p0, p1)
}

/// Fast cubic B-spline fetch: four hardware-bilinear reads via the Sigg &
/// Hadwiger weight-pair folding, equivalent to [`bspline_cubic`] but at a
/// quarter of the texel reads. This is the formulation a GPU uses.
#[must_use]
pub fn bspline_cubic_fast<S: TexelSource>(
    src: &S,
    mip: u32,
    uv: [f32; 2],
    wrap_u: WrapMode,
    wrap_v: WrapMode,
    border_color: [f32; 4],
) -> [f32; 4] {
    let (bw, bh) = src.dimensions();
    let w = dim_at(bw.max(1), mip);
    let h = dim_at(bh.max(1), mip);
    let wf = w as f32;
    let hf = h as f32;

    let fx = finite_or_zero(uv[0]) * wf - 0.5;
    let fy = finite_or_zero(uv[1]) * hf - 0.5;
    let x1 = fx.floor();
    let y1 = fy.floor();

    let wx = bspline_cubic_weights(fx - x1);
    let wy = bspline_cubic_weights(fy - y1);
    let (gx0, gx1, px0, px1) = fold_axis(x1, wx);
    let (gy0, gy1, py0, py1) = fold_axis(y1, wy);

    // Convert texel-space position p back to the UV `bilinear` expects:
    // `bilinear` computes `uv * dim - 0.5`, so `uv = (p + 0.5) / dim` samples
    // texel position `p`.
    let ux0 = (px0 + 0.5) / wf;
    let ux1 = (px1 + 0.5) / wf;
    let uy0 = (py0 + 0.5) / hf;
    let uy1 = (py1 + 0.5) / hf;

    let b00 = bilinear(src, mip, [ux0, uy0], wrap_u, wrap_v, border_color);
    let b10 = bilinear(src, mip, [ux1, uy0], wrap_u, wrap_v, border_color);
    let b01 = bilinear(src, mip, [ux0, uy1], wrap_u, wrap_v, border_color);
    let b11 = bilinear(src, mip, [ux1, uy1], wrap_u, wrap_v, border_color);

    let mut out = [0.0_f32; 4];
    accumulate(&mut out, b00, gx0 * gy0);
    accumulate(&mut out, b10, gx1 * gy0);
    accumulate(&mut out, b01, gx0 * gy1);
    accumulate(&mut out, b11, gx1 * gy1);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 64x64 planar ramp in red: `a + bx*x + by*y`.
    struct Plane {
        a: f32,
        bx: f32,
        by: f32,
    }
    impl TexelSource for Plane {
        fn dimensions(&self) -> (u32, u32) {
            (64, 64)
        }
        fn texel(&self, _mip: u32, x: u32, y: u32) -> [f32; 4] {
            [self.a + self.bx * x as f32 + self.by * y as f32, 0.0, 0.0, 1.0]
        }
    }

    struct Flat([f32; 4]);
    impl TexelSource for Flat {
        fn dimensions(&self) -> (u32, u32) {
            (16, 16)
        }
        fn texel(&self, _mip: u32, _x: u32, _y: u32) -> [f32; 4] {
            self.0
        }
    }

    /// A high-contrast step/checker whose red channel is 0 or 1, used to show
    /// B-spline never overshoots `[0, 1]` (unlike Catmull-Rom).
    struct Step;
    impl TexelSource for Step {
        fn dimensions(&self) -> (u32, u32) {
            (32, 32)
        }
        fn texel(&self, _mip: u32, x: u32, y: u32) -> [f32; 4] {
            let v = ((x / 4 + y / 4) % 2) as f32;
            [v, v, v, 1.0]
        }
    }

    /// Deterministic hash noise for the fast-vs-direct cross check.
    struct Noise;
    impl TexelSource for Noise {
        fn dimensions(&self) -> (u32, u32) {
            (32, 32)
        }
        fn texel(&self, _mip: u32, x: u32, y: u32) -> [f32; 4] {
            let h = (x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)).wrapping_add(1);
            let f = (h % 1000) as f32 / 1000.0;
            [f, 1.0 - f, (f * 0.5) + 0.25, 1.0]
        }
    }

    #[test]
    fn weights_are_partition_of_unity() {
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let s: f32 = bspline_cubic_weights(t).iter().sum();
            assert!((s - 1.0).abs() < 1.0e-6, "t={t} sum={s}");
        }
    }

    #[test]
    fn weights_are_non_negative() {
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            for (k, w) in bspline_cubic_weights(t).iter().enumerate() {
                assert!(*w >= -1.0e-7, "t={t} k={k} w={w}");
            }
        }
    }

    #[test]
    fn weights_are_symmetric() {
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let w = bspline_cubic_weights(t);
            let m = bspline_cubic_weights(1.0 - t);
            for k in 0..4 {
                assert!((w[k] - m[3 - k]).abs() < 1.0e-6, "t={t} k={k} {w:?} {m:?}");
            }
        }
    }

    #[test]
    fn constant_texture_is_preserved() {
        let flat = Flat([0.3, 0.6, 0.9, 1.0]);
        for &uv in &[[0.123, 0.777], [0.5, 0.5], [0.9, 0.1]] {
            let c = bspline_cubic(&flat, 0, uv, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
            let f = bspline_cubic_fast(&flat, 0, uv, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
            for k in 0..4 {
                assert!((c[k] - flat.0[k]).abs() < 1.0e-6, "direct uv={uv:?} {c:?}");
                assert!((f[k] - flat.0[k]).abs() < 1.0e-6, "fast uv={uv:?} {f:?}");
            }
        }
    }

    #[test]
    fn reproduces_linear_ramp() {
        // B-spline reproduces linear polynomials, so a planar ramp is returned
        // as the continuous plane at an interior UV.
        let p = Plane { a: 1.0, bx: 0.5, by: 0.3 };
        let fx = 24.37_f32;
        let fy = 19.62_f32;
        let uv = [(fx + 0.5) / 64.0, (fy + 0.5) / 64.0];
        let c = bspline_cubic(&p, 0, uv, WrapMode::ClampToEdge, WrapMode::ClampToEdge, [0.0; 4]);
        let want = p.a + p.bx * fx + p.by * fy;
        assert!((c[0] - want).abs() < 1.0e-3, "got {} want {want}", c[0]);
    }

    #[test]
    fn never_overshoots_step() {
        // Convex (non-negative) weights -> output stays within the [0,1] source
        // range everywhere, including across the hard step edges.
        for i in 0..=40 {
            for j in 0..=40 {
                let uv = [i as f32 / 40.0, j as f32 / 40.0];
                let c = bspline_cubic(&Step, 0, uv, WrapMode::ClampToEdge, WrapMode::ClampToEdge, [0.0; 4]);
                assert!(c[0] >= -1.0e-6 && c[0] <= 1.0 + 1.0e-6, "uv={uv:?} r={}", c[0]);
            }
        }
    }

    #[test]
    fn fast_matches_direct() {
        for &uv in &[[0.137, 0.482], [0.5, 0.5], [0.91, 0.04], [0.26, 0.73], [0.03, 0.97]] {
            for (wu, wv) in [
                (WrapMode::Repeat, WrapMode::Repeat),
                (WrapMode::ClampToEdge, WrapMode::ClampToEdge),
                (WrapMode::MirroredRepeat, WrapMode::MirroredRepeat),
                (WrapMode::ClampToBorder, WrapMode::ClampToBorder),
            ] {
                let border = [0.11, 0.22, 0.33, 0.44];
                let a = bspline_cubic(&Noise, 0, uv, wu, wv, border);
                let b = bspline_cubic_fast(&Noise, 0, uv, wu, wv, border);
                for k in 0..4 {
                    assert!((a[k] - b[k]).abs() < 1.0e-4, "uv={uv:?} {wu:?}/{wv:?} {a:?} vs {b:?}");
                }
            }
        }
    }

    #[test]
    fn non_finite_uv_is_safe() {
        let a = bspline_cubic(&Flat([0.5, 0.5, 0.5, 1.0]), 0, [f32::NAN, f32::INFINITY], WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        let b = bspline_cubic_fast(&Flat([0.5, 0.5, 0.5, 1.0]), 0, [f32::NAN, f32::INFINITY], WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        assert!(a.iter().all(|v| v.is_finite()), "{a:?}");
        assert!(b.iter().all(|v| v.is_finite()), "{b:?}");
    }
}
