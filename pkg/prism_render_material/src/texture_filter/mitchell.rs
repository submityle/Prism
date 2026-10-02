//! Mitchell-Netravali parametric cubic filter — the general `(B, C)` family
//! that unifies this crate's two concrete cubics and adds the recommended
//! balanced resampling kernel.
//!
//! [`super::bicubic`] (Catmull-Rom) is sharp but rings; [`super::bspline`] is
//! ringing-free but soft. Mitchell & Netravali parameterise the whole separable
//! cubic family by two scalars `(B, C)` so a project can dial the ring/blur
//! trade-off, and recommend `B = C = 1/3` as the subjectively best general
//! image resampler (used by most offline resamplers). This module exposes that
//! family directly; the two existing kernels are its `(0, 1/2)` and `(1, 0)`
//! corners, which the tests use as independent oracles.
//!
//! Pure `f32` arithmetic with defensive non-finite guards and no AI/ML path, so
//! a CPU golden reproduces a GPU twin exactly. For `C > 0` the kernel has a
//! negative lobe and can overshoot by design; values are not clamped.
//!
//! # Conventions
//! * Texel-centre / mip-sizing conventions match [`super::bilinear`].
//! * The 4x4 footprint uses offsets `-1, 0, +1, +2` from `floor(sample)`; each
//!   index is folded through [`super::texel_wrap`] like a bilinear corner, and a
//!   [`WrapMode::ClampToBorder`] tap contributes the border colour with its
//!   (possibly negative) weight.
//! * Non-finite UVs collapse to `0.0`, never `NaN`.
//!
//! # References
//! * Mitchell & Netravali, "Reconstruction Filters in Computer Graphics",
//!   SIGGRAPH 1988 — the `(B, C)` family and the `1/3, 1/3` recommendation.
//! * Catmull-Rom is `(0, 1/2)`; the cubic B-spline is `(1, 0)`.

use super::super::texture_addressing::WrapMode;
use super::bilinear::{dim_at, TexelSource};
use super::texel_wrap::{wrap_texel, TexelAddr};

/// Recommended Mitchell-Netravali `B` parameter (`1/3`).
pub const MITCHELL_B: f32 = 1.0 / 3.0;
/// Recommended Mitchell-Netravali `C` parameter (`1/3`).
pub const MITCHELL_C: f32 = 1.0 / 3.0;

#[inline]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

#[inline]
fn accumulate(acc: &mut [f32; 4], c: [f32; 4], w: f32) {
    acc[0] += c[0] * w;
    acc[1] += c[1] * w;
    acc[2] += c[2] * w;
    acc[3] += c[3] * w;
}

/// The Mitchell-Netravali cubic kernel `k(x)` for a non-negative distance `x`
/// and parameters `(b, c)`. `C1`-continuous at `x = 1`, zero for `x >= 2`.
#[inline]
#[must_use]
fn mn_kernel(x: f32, b: f32, c: f32) -> f32 {
    let x2 = x * x;
    let x3 = x2 * x;
    if x < 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * x3 + (-18.0 + 12.0 * b + 6.0 * c) * x2 + (6.0 - 2.0 * b)) / 6.0
    } else if x < 2.0 {
        ((-b - 6.0 * c) * x3 + (6.0 * b + 30.0 * c) * x2 + (-12.0 * b - 48.0 * c) * x + (8.0 * b + 24.0 * c)) / 6.0
    } else {
        0.0
    }
}

/// The four Mitchell-Netravali weights for a fractional offset `t` in `[0, 1)`
/// and parameters `(b, c)`, applied to the taps at integer offsets
/// `-1, 0, +1, +2`. The weights sum to 1 for every `t` (partition of unity).
///
/// `(b, c) = (0, 1/2)` reproduces Catmull-Rom; `(1, 0)` reproduces the cubic
/// B-spline.
#[inline]
#[must_use]
pub fn mitchell_netravali_weights(b: f32, c: f32, t: f32) -> [f32; 4] {
    [
        mn_kernel(t + 1.0, b, c),
        mn_kernel(t, b, c),
        mn_kernel(1.0 - t, b, c),
        mn_kernel(2.0 - t, b, c),
    ]
}

/// Mitchell-Netravali `(b, c)` cubic sample of `src` at mip `mip` for UV `uv`
/// (separable 16-tap). Pass [`MITCHELL_B`]/[`MITCHELL_C`] for the recommended
/// balanced resampler.
#[must_use]
pub fn cubic_mitchell<S: TexelSource>(
    src: &S,
    mip: u32,
    uv: [f32; 2],
    b: f32,
    c: f32,
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

    let wx = mitchell_netravali_weights(b, c, fx - x1);
    let wy = mitchell_netravali_weights(b, c, fy - y1);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bicubic_catmull_rom, bspline_cubic, catmull_rom_weights, bspline_cubic_weights};

    struct Flat([f32; 4]);
    impl TexelSource for Flat {
        fn dimensions(&self) -> (u32, u32) {
            (16, 16)
        }
        fn texel(&self, _mip: u32, _x: u32, _y: u32) -> [f32; 4] {
            self.0
        }
    }

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
    fn partition_of_unity_for_many_bc() {
        for &(b, c) in &[(1.0 / 3.0, 1.0 / 3.0), (0.0, 0.5), (1.0, 0.0), (0.0, 0.0), (0.0, 0.75)] {
            for i in 0..=10 {
                let t = i as f32 / 10.0;
                let s: f32 = mitchell_netravali_weights(b, c, t).iter().sum();
                assert!((s - 1.0).abs() < 1.0e-5, "b={b} c={c} t={t} sum={s}");
            }
        }
    }

    #[test]
    fn reduces_to_catmull_rom_weights() {
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let mn = mitchell_netravali_weights(0.0, 0.5, t);
            let cr = catmull_rom_weights(t);
            for k in 0..4 {
                assert!((mn[k] - cr[k]).abs() < 1.0e-6, "t={t} k={k} {mn:?} vs {cr:?}");
            }
        }
    }

    #[test]
    fn reduces_to_bspline_weights() {
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let mn = mitchell_netravali_weights(1.0, 0.0, t);
            let bs = bspline_cubic_weights(t);
            for k in 0..4 {
                assert!((mn[k] - bs[k]).abs() < 1.0e-6, "t={t} k={k} {mn:?} vs {bs:?}");
            }
        }
    }

    #[test]
    fn sampler_matches_catmull_rom_at_0_half() {
        for &uv in &[[0.137, 0.482], [0.5, 0.5], [0.91, 0.04]] {
            let a = cubic_mitchell(&Noise, 0, uv, 0.0, 0.5, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
            let b = bicubic_catmull_rom(&Noise, 0, uv, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
            for k in 0..4 {
                assert!((a[k] - b[k]).abs() < 1.0e-5, "uv={uv:?} {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn sampler_matches_bspline_at_1_zero() {
        for &uv in &[[0.137, 0.482], [0.5, 0.5], [0.91, 0.04]] {
            let a = cubic_mitchell(&Noise, 0, uv, 1.0, 0.0, WrapMode::ClampToEdge, WrapMode::ClampToEdge, [0.0; 4]);
            let b = bspline_cubic(&Noise, 0, uv, WrapMode::ClampToEdge, WrapMode::ClampToEdge, [0.0; 4]);
            for k in 0..4 {
                assert!((a[k] - b[k]).abs() < 1.0e-5, "uv={uv:?} {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn recommended_constant_is_preserved() {
        let flat = Flat([0.3, 0.6, 0.9, 1.0]);
        for &uv in &[[0.123, 0.777], [0.5, 0.5], [0.9, 0.1]] {
            let c = cubic_mitchell(&flat, 0, uv, MITCHELL_B, MITCHELL_C, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
            for k in 0..4 {
                assert!((c[k] - flat.0[k]).abs() < 1.0e-6, "uv={uv:?} {c:?}");
            }
        }
    }

    #[test]
    fn non_finite_uv_is_safe() {
        let c = cubic_mitchell(
            &Flat([0.5, 0.5, 0.5, 1.0]),
            0,
            [f32::NAN, f32::INFINITY],
            MITCHELL_B,
            MITCHELL_C,
            WrapMode::Repeat,
            WrapMode::Repeat,
            [0.0; 4],
        );
        assert!(c.iter().all(|v| v.is_finite()), "{c:?}");
    }
}
