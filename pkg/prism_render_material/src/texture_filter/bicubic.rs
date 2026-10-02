//! Catmull-Rom bicubic magnification filter for the manual texture sampler.
//!
//! [`super::bilinear`] blends the four neighbour texels around a sample; that
//! is the hardware `LINEAR` filter and is C0 only, so strong magnification of a
//! lightmap, a UI atlas, or a pre-upsample prefilter shows the characteristic
//! "diamond" bilinear seams. This module adds a higher-quality 4x4 cubic fetch
//! using the Catmull-Rom (Keys cubic-convolution, a = -1/2) kernel: it is C1,
//! *interpolating* (passes through the original texels), and reproduces linear
//! ramps exactly, giving a smooth magnified result at a fixed 16-tap cost.
//!
//! Two formulations live here. [`bicubic_catmull_rom`] is the production
//! separable evaluation (four horizontal cubics blended by one vertical cubic).
//! The test module carries an independent full 4x4 double-sum reference; the two
//! are derived differently yet must agree bit-closely, which catches tap
//! indexing / weight-ordering mistakes without a circular self-check.
//!
//! Pure `f32` arithmetic with defensive non-finite guards and no AI/ML path, so
//! a CPU golden reproduces a GPU twin exactly. Cubic kernels overshoot by design
//! (ringing near steep edges); values are intentionally **not** clamped to the
//! input range so the result matches a GPU cubic twin.
//!
//! # Conventions
//! * Texel-centre and mip-sizing conventions match [`super::bilinear`]: the
//!   sample position in texels is `uv * dim - 0.5`, and the axis size at `mip`
//!   is `max(1, base_dim >> mip)`.
//! * The 4x4 footprint uses offsets `-1, 0, +1, +2` from `floor(sample)`; each
//!   integer index is folded through [`super::texel_wrap`] exactly like a
//!   bilinear corner, and a [`WrapMode::ClampToBorder`] tap contributes the
//!   caller's border colour with its (possibly negative) cubic weight.
//! * Non-finite UVs collapse to `0.0`, never `NaN`.
//!
//! # References
//! * Keys, "Cubic Convolution Interpolation for Digital Image Processing",
//!   IEEE ASSP-29 (1981) — the a = -1/2 kernel reproduced here.
//! * Catmull & Rom, "A Class of Local Interpolating Splines" (1974).
//! * Sigg & Hadwiger, "Fast Third-Order Texture Filtering", GPU Gems 2 — the
//!   separable structure this module mirrors on the CPU.

use super::super::texture_addressing::WrapMode;
use super::bilinear::{dim_at, TexelSource};
use super::texel_wrap::{wrap_texel, TexelAddr};

#[inline]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// The four Catmull-Rom (Keys a = -1/2) weights for a fractional offset `t` in
/// `[0, 1)`, applied to the taps at integer offsets `-1, 0, +1, +2`.
///
/// The weights sum to 1 for every `t` (partition of unity) and the weight set
/// is symmetric: reversing `weights(t)` yields `weights(1 - t)`.
#[inline]
#[must_use]
pub fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Weight a tap colour into an accumulator in place.
#[inline]
fn accumulate(acc: &mut [f32; 4], c: [f32; 4], w: f32) {
    acc[0] += c[0] * w;
    acc[1] += c[1] * w;
    acc[2] += c[2] * w;
    acc[3] += c[3] * w;
}

/// Bicubic (Catmull-Rom) sample of `src` at mip `mip` for UV `uv`.
///
/// Fetches the 4x4 texel neighbourhood around the sample position, folding each
/// integer index through the per-axis wrap modes (substituting `border_color`
/// for any [`WrapMode::ClampToBorder`] tap out of range), and blends it with the
/// separable Catmull-Rom kernel. Interpolates the original texels at texel
/// centres and reproduces linear ramps exactly.
#[must_use]
pub fn bicubic_catmull_rom<S: TexelSource>(
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
    let tx = fx - x1;
    let ty = fy - y1;
    let (ix, iy) = (x1 as i64, y1 as i64);

    let wx = catmull_rom_weights(tx);
    let wy = catmull_rom_weights(ty);

    let fetch = |ox: i64, oy: i64| -> [f32; 4] {
        match (wrap_texel(ix + ox, w, wrap_u), wrap_texel(iy + oy, h, wrap_v)) {
            (TexelAddr::In(cx), TexelAddr::In(cy)) => src.texel(mip, cx, cy),
            _ => border_color,
        }
    };

    // Separable: collapse each of the four rows horizontally, then blend the
    // four row results vertically.
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

    /// Independent reference: full 4x4 double sum with the product weight
    /// `wx[j] * wy[i]`. Derived separately from the production separable form.
    fn bicubic_direct<S: TexelSource>(
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
        let (tx, ty) = (fx - x1, fy - y1);
        let (ix, iy) = (x1 as i64, y1 as i64);
        let wx = catmull_rom_weights(tx);
        let wy = catmull_rom_weights(ty);
        const OFFS: [i64; 4] = [-1, 0, 1, 2];
        let mut out = [0.0_f32; 4];
        for (i, &oy) in OFFS.iter().enumerate() {
            for (j, &ox) in OFFS.iter().enumerate() {
                let c = match (wrap_texel(ix + ox, w, wrap_u), wrap_texel(iy + oy, h, wrap_v)) {
                    (TexelAddr::In(cx), TexelAddr::In(cy)) => src.texel(mip, cx, cy),
                    _ => border_color,
                };
                accumulate(&mut out, c, wx[j] * wy[i]);
            }
        }
        out
    }

    /// A 64x64 source whose red channel is a planar ramp `a + bx*x + by*y`, so a
    /// linear-reproducing filter returns the continuous plane at any interior UV.
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

    /// Constant source: every texel is the same colour, so any filter that is a
    /// partition of unity returns that colour unchanged.
    struct Flat([f32; 4]);
    impl TexelSource for Flat {
        fn dimensions(&self) -> (u32, u32) {
            (16, 16)
        }
        fn texel(&self, _mip: u32, _x: u32, _y: u32) -> [f32; 4] {
            self.0
        }
    }

    /// A deterministic pseudo-random source for the direct-vs-separable cross
    /// check (no RNG dependency; a hash of the coordinates).
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
            let s: f32 = catmull_rom_weights(t).iter().sum();
            assert!((s - 1.0).abs() < 1.0e-6, "t={t} sum={s}");
        }
    }

    #[test]
    fn weights_are_symmetric() {
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let w = catmull_rom_weights(t);
            let m = catmull_rom_weights(1.0 - t);
            // reverse(w) == weights(1 - t).
            for k in 0..4 {
                assert!((w[k] - m[3 - k]).abs() < 1.0e-6, "t={t} k={k} {w:?} {m:?}");
            }
        }
    }

    #[test]
    fn weights_at_zero_pick_center_tap() {
        let w = catmull_rom_weights(0.0);
        assert_eq!(w, [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn interpolates_at_texel_centers() {
        // uv = (k + 0.5)/dim lands tx = 0 -> center-tap weight picks texel k.
        let p = Plane { a: 2.0, bx: 0.5, by: -0.25 };
        for (kx, ky) in [(10u32, 12u32), (30, 5), (40, 40)] {
            let uv = [(kx as f32 + 0.5) / 64.0, (ky as f32 + 0.5) / 64.0];
            let c = bicubic_catmull_rom(&p, 0, uv, WrapMode::ClampToEdge, WrapMode::ClampToEdge, [0.0; 4]);
            let want = p.texel(0, kx, ky)[0];
            assert!((c[0] - want).abs() < 1.0e-4, "centre ({kx},{ky}) got {} want {want}", c[0]);
        }
    }

    #[test]
    fn reproduces_linear_ramp_between_centers() {
        // Interior fractional UV: the continuous plane value is reproduced.
        let p = Plane { a: 1.0, bx: 0.75, by: 0.3 };
        let uv = [(20.37 + 0.5) / 64.0, (18.62 + 0.5) / 64.0];
        let c = bicubic_catmull_rom(&p, 0, uv, WrapMode::ClampToEdge, WrapMode::ClampToEdge, [0.0; 4]);
        let want = p.a + p.bx * 20.37 + p.by * 18.62;
        assert!((c[0] - want).abs() < 1.0e-3, "got {} want {want}", c[0]);
    }

    #[test]
    fn constant_texture_is_preserved() {
        let flat = Flat([0.3, 0.6, 0.9, 1.0]);
        for &uv in &[[0.123, 0.777], [0.5, 0.5], [0.9, 0.1]] {
            let c = bicubic_catmull_rom(&flat, 0, uv, WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
            for k in 0..4 {
                assert!((c[k] - flat.0[k]).abs() < 1.0e-6, "uv={uv:?} {c:?}");
            }
        }
    }

    #[test]
    fn separable_matches_direct_double_sum() {
        for &uv in &[[0.137, 0.482], [0.5, 0.5], [0.91, 0.04], [0.26, 0.73]] {
            for (wu, wv) in [
                (WrapMode::Repeat, WrapMode::Repeat),
                (WrapMode::ClampToEdge, WrapMode::ClampToEdge),
                (WrapMode::ClampToBorder, WrapMode::ClampToBorder),
            ] {
                let border = [0.11, 0.22, 0.33, 0.44];
                let a = bicubic_catmull_rom(&Noise, 0, uv, wu, wv, border);
                let b = bicubic_direct(&Noise, 0, uv, wu, wv, border);
                for k in 0..4 {
                    assert!((a[k] - b[k]).abs() < 1.0e-5, "uv={uv:?} {wu:?}/{wv:?} {a:?} vs {b:?}");
                }
            }
        }
    }

    #[test]
    fn non_finite_uv_is_safe() {
        let c = bicubic_catmull_rom(
            &Flat([0.5, 0.5, 0.5, 1.0]),
            0,
            [f32::NAN, f32::INFINITY],
            WrapMode::Repeat,
            WrapMode::Repeat,
            [0.0; 4],
        );
        assert!(c.iter().all(|v| v.is_finite()), "{c:?}");
    }
}
