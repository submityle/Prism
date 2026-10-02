//! Height-field -> tangent-space normal map generation.
//!
//! AAA material authoring routinely derives a normal map from a greyscale
//! height / bump field (sculpt cavity masks, tiling detail, decals, terrain).
//! The normal is the unit surface normal of the height field: given the local
//! slope (gradient) `g = (dh/du, dh/dv)` the tangent-space normal is
//! `normalize(-g.x, -g.y, 1)` -- exactly the [`slope_to_normal`] round trip this
//! crate already verifies for the strength control. This module estimates that
//! gradient from a discrete height grid and feeds it through the same slope
//! conversion, so the generator and the strength slider stay consistent.
//!
//! Three gradient estimators are offered:
//! * [`HeightGradient::CentralDifference`] -- the 2-tap `(h[+1] - h[-1]) / 2`
//!   estimator; cheapest and exact for a locally linear field.
//! * [`HeightGradient::Sobel`] -- the 3x3 Sobel operator, which averages three
//!   rows/columns so it is far less sensitive to single-texel noise while still
//!   being exact for a planar ramp. This is the common content-pipeline choice.
//! * [`HeightGradient::Scharr`] -- the 3x3 Scharr operator, an optimized
//!   `{3, 10, 3}` weighting whose Fourier response best approximates an ideal
//!   rotation-invariant gradient, so slanted and curved detail keeps a more
//!   accurate normal direction than Sobel while remaining exact for a planar
//!   ramp.
//!
//! The per-texel world spacing `texel_world_size` lets non-square texels and an
//! explicit bump scale map to a physically meaningful slope; `strength` scales
//! the slope the same way [`scale_strength`] does (steepen `> 1`, flatten
//! `< 1`). Border texels fetch their neighbours through a [`WrapMode`] so a
//! tiling source stays seamless under [`WrapMode::Repeat`].
//!
//! Pure analytic `f32` math, no AI/ML, so a CPU golden matches a GPU compute
//! twin to floating-point tolerance.
//!
//! # References
//! * Akenine-Moller et al., *Real-Time Rendering* 4th ed., Section 16.3
//!   (height-field gradient / bump-to-normal).
//! * Sobel & Feldman, "A 3x3 Isotropic Gradient Operator for Image Processing"
//!   (1968) -- the Sobel kernel.
//! * Scharr, "Optimale Operatoren in der digitalen Bildverarbeitung" (2000)
//!   -- the rotation-optimized `{3, 10, 3}` gradient kernel.

use alloc::vec::Vec;

use super::slope_to_normal;
use crate::WrapMode;

/// Gradient estimator used by [`height_to_normal`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeightGradient {
    /// 2-tap central difference `(h[+1] - h[-1]) / 2` per axis.
    CentralDifference,
    /// 3x3 Sobel operator (row/column averaged central difference).
    Sobel,
    /// 3x3 Scharr operator (`{3, 10, 3}` rotation-optimized gradient).
    Scharr,
}

/// Wrap an integer texel index into `[0, n)` for a gradient tap.
///
/// `Repeat` and `MirroredRepeat` keep tiling / mirrored sources seamless; every
/// other mode (clamp, border, mirror-clamp) collapses to edge clamp, which is
/// the correct behaviour for a non-tiling height field.
#[inline]
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

/// Generate a tangent-space (z-up) unit normal per texel from a height grid.
///
/// `heights` is a row-major `width * height` greyscale field. `strength` scales
/// the estimated slope (`1` = as authored, `> 1` steeper, `< 1` flatter, `0`
/// flat); `texel_world_size` is the world-space spacing of one texel on each
/// axis (use `[1.0, 1.0]` for pixel-space). `wrap` controls how border taps
/// fetch out-of-range neighbours.
///
/// Returns `None` if the dimensions are zero, the buffer length does not match
/// `width * height`, or a texel size is not strictly positive.
#[must_use]
pub fn height_to_normal(
    heights: &[f32],
    width: u32,
    height: u32,
    strength: f32,
    texel_world_size: [f32; 2],
    kernel: HeightGradient,
    wrap: WrapMode,
) -> Option<Vec<[f32; 3]>> {
    if width == 0 || height == 0 {
        return None;
    }
    let count = width as usize * height as usize;
    if heights.len() != count {
        return None;
    }
    let (dx, dy) = (texel_world_size[0], texel_world_size[1]);
    if !dx.is_finite() || !dy.is_finite() || dx <= 0.0 || dy <= 0.0 {
        return None;
    }

    let w = width as i64;
    let h = height as i64;
    let at = |x: i64, y: i64| -> f32 {
        heights[wrap_index(y, h, wrap) * width as usize + wrap_index(x, w, wrap)]
    };

    let mut out = Vec::with_capacity(count);
    for y in 0..h {
        for x in 0..w {
            let (gx, gy) = match kernel {
                HeightGradient::CentralDifference => (
                    (at(x + 1, y) - at(x - 1, y)) / (2.0 * dx),
                    (at(x, y + 1) - at(x, y - 1)) / (2.0 * dy),
                ),
                HeightGradient::Sobel => {
                    let gx = ((at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1))
                        - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1)))
                        / (8.0 * dx);
                    let gy = ((at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1))
                        - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1)))
                        / (8.0 * dy);
                    (gx, gy)
                }
                HeightGradient::Scharr => {
                    let gx = ((3.0 * at(x + 1, y - 1)
                        + 10.0 * at(x + 1, y)
                        + 3.0 * at(x + 1, y + 1))
                        - (3.0 * at(x - 1, y - 1) + 10.0 * at(x - 1, y) + 3.0 * at(x - 1, y + 1)))
                        / (32.0 * dx);
                    let gy = ((3.0 * at(x - 1, y + 1)
                        + 10.0 * at(x, y + 1)
                        + 3.0 * at(x + 1, y + 1))
                        - (3.0 * at(x - 1, y - 1) + 10.0 * at(x, y - 1) + 3.0 * at(x + 1, y - 1)))
                        / (32.0 * dy);
                    (gx, gy)
                }
            };
            out.push(slope_to_normal([strength * gx, strength * gy]));
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scale_strength;

    const KERNELS: [HeightGradient; 3] = [
        HeightGradient::CentralDifference,
        HeightGradient::Sobel,
        HeightGradient::Scharr,
    ];

    fn idx(x: u32, y: u32, w: u32) -> usize {
        (y * w + x) as usize
    }

    fn unit(n: [f32; 3]) {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((l - 1.0).abs() < 1.0e-5, "len={l} n={n:?}");
    }

    #[test]
    fn rejects_bad_input() {
        let h = vec![0.0f32; 12];
        assert!(height_to_normal(
            &h,
            4,
            3,
            1.0,
            [1.0, 1.0],
            HeightGradient::Sobel,
            WrapMode::ClampToEdge
        )
        .is_some());
        // wrong length
        assert!(height_to_normal(
            &h,
            4,
            4,
            1.0,
            [1.0, 1.0],
            HeightGradient::Sobel,
            WrapMode::ClampToEdge
        )
        .is_none());
        // zero dim
        assert!(height_to_normal(
            &h,
            0,
            3,
            1.0,
            [1.0, 1.0],
            HeightGradient::Sobel,
            WrapMode::ClampToEdge
        )
        .is_none());
        // bad texel size
        assert!(height_to_normal(
            &h,
            4,
            3,
            1.0,
            [0.0, 1.0],
            HeightGradient::Sobel,
            WrapMode::ClampToEdge
        )
        .is_none());
        assert!(height_to_normal(
            &h,
            4,
            3,
            1.0,
            [1.0, -2.0],
            HeightGradient::Sobel,
            WrapMode::ClampToEdge
        )
        .is_none());
    }

    #[test]
    fn flat_height_is_up_for_every_kernel_and_wrap() {
        let h = vec![0.37f32; 6 * 5];
        for k in KERNELS {
            for wrap in [
                WrapMode::ClampToEdge,
                WrapMode::Repeat,
                WrapMode::MirroredRepeat,
            ] {
                for &s in &[0.0f32, 1.0, 4.0] {
                    let n = height_to_normal(&h, 6, 5, s, [1.0, 1.0], k, wrap).unwrap();
                    for v in n {
                        assert!(
                            (v[0]).abs() < 1.0e-6
                                && (v[1]).abs() < 1.0e-6
                                && (v[2] - 1.0).abs() < 1.0e-6,
                            "{k:?} {wrap:?} {v:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn planar_ramp_matches_analytic_slope() {
        // h = m * x (per texel), dx = 1 -> gradient is exactly m on both kernels.
        let (w, hgt) = (8u32, 5u32);
        let m = 0.25f32;
        let mut h = vec![0.0f32; (w * hgt) as usize];
        for y in 0..hgt {
            for x in 0..w {
                h[idx(x, y, w)] = m * x as f32;
            }
        }
        let expect = slope_to_normal([m, 0.0]);
        for k in KERNELS {
            let n =
                height_to_normal(&h, w, hgt, 1.0, [1.0, 1.0], k, WrapMode::ClampToEdge).unwrap();
            // interior columns only (edges clamp, which breaks the ramp there).
            for y in 0..hgt {
                for x in 1..w - 1 {
                    let got = n[idx(x, y, w)];
                    for c in 0..3 {
                        assert!(
                            (got[c] - expect[c]).abs() < 1.0e-5,
                            "{k:?} x={x} y={y} {got:?} vs {expect:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn central_and_sobel_agree_on_a_linear_ramp() {
        // Both estimators are exact for a planar field, so they must agree
        // pointwise on the interior.
        let (w, hgt) = (7u32, 6u32);
        let mut h = vec![0.0f32; (w * hgt) as usize];
        for y in 0..hgt {
            for x in 0..w {
                h[idx(x, y, w)] = 0.3 * x as f32 - 0.17 * y as f32;
            }
        }
        let a = height_to_normal(
            &h,
            w,
            hgt,
            1.0,
            [1.0, 1.0],
            HeightGradient::CentralDifference,
            WrapMode::ClampToEdge,
        )
        .unwrap();
        let b = height_to_normal(
            &h,
            w,
            hgt,
            1.0,
            [1.0, 1.0],
            HeightGradient::Sobel,
            WrapMode::ClampToEdge,
        )
        .unwrap();
        for y in 1..hgt - 1 {
            for x in 1..w - 1 {
                let (va, vb) = (a[idx(x, y, w)], b[idx(x, y, w)]);
                for c in 0..3 {
                    assert!(
                        (va[c] - vb[c]).abs() < 1.0e-5,
                        "x={x} y={y} {va:?} vs {vb:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn output_is_always_unit() {
        // Arbitrary bumpy field.
        let (w, hgt) = (9u32, 9u32);
        let mut h = vec![0.0f32; (w * hgt) as usize];
        for y in 0..hgt {
            for x in 0..w {
                h[idx(x, y, w)] = ((x * 7 + y * 13) % 11) as f32 * 0.1;
            }
        }
        for k in KERNELS {
            for wrap in [
                WrapMode::ClampToEdge,
                WrapMode::Repeat,
                WrapMode::MirroredRepeat,
            ] {
                let n = height_to_normal(&h, w, hgt, 2.0, [1.0, 1.3], k, wrap).unwrap();
                for v in n {
                    unit(v);
                }
            }
        }
    }

    #[test]
    fn strength_matches_slope_space_scaling() {
        // Generating with strength s equals scaling the s=1 normals through the
        // verified slope-space strength control, texel for texel.
        let (w, hgt) = (6u32, 6u32);
        let mut h = vec![0.0f32; (w * hgt) as usize];
        for y in 0..hgt {
            for x in 0..w {
                h[idx(x, y, w)] = (x as f32) * 0.37 + (y as f32) * (y as f32) * 0.11;
            }
        }
        let base = height_to_normal(
            &h,
            w,
            hgt,
            1.0,
            [1.0, 1.0],
            HeightGradient::Sobel,
            WrapMode::Repeat,
        )
        .unwrap();
        for &s in &[0.5f32, 2.0, 4.0] {
            let scaled = height_to_normal(
                &h,
                w,
                hgt,
                s,
                [1.0, 1.0],
                HeightGradient::Sobel,
                WrapMode::Repeat,
            )
            .unwrap();
            for (b, sc) in base.iter().zip(scaled.iter()) {
                let via = scale_strength(*b, s);
                for c in 0..3 {
                    assert!((sc[c] - via[c]).abs() < 1.0e-5, "s={s} {sc:?} vs {via:?}");
                }
            }
        }
    }

    #[test]
    fn horizontal_mirror_flips_normal_x() {
        // Mirroring the height field in x negates the x-gradient, so the normal
        // x-component flips while y and z are preserved.
        let (w, hgt) = (7u32, 4u32);
        let mut h = vec![0.0f32; (w * hgt) as usize];
        for y in 0..hgt {
            for x in 0..w {
                h[idx(x, y, w)] = (x * x + 3 * y) as f32 * 0.05;
            }
        }
        let mut hm = vec![0.0f32; (w * hgt) as usize];
        for y in 0..hgt {
            for x in 0..w {
                hm[idx(x, y, w)] = h[idx(w - 1 - x, y, w)];
            }
        }
        let n = height_to_normal(
            &h,
            w,
            hgt,
            1.5,
            [1.0, 1.0],
            HeightGradient::Sobel,
            WrapMode::ClampToEdge,
        )
        .unwrap();
        let nm = height_to_normal(
            &hm,
            w,
            hgt,
            1.5,
            [1.0, 1.0],
            HeightGradient::Sobel,
            WrapMode::ClampToEdge,
        )
        .unwrap();
        for y in 0..hgt {
            for x in 0..w {
                let a = n[idx(x, y, w)];
                let b = nm[idx(w - 1 - x, y, w)];
                assert!(
                    (a[0] + b[0]).abs() < 1.0e-5,
                    "x flip x={x} y={y} {a:?} {b:?}"
                );
                assert!((a[1] - b[1]).abs() < 1.0e-5, "y keep x={x} y={y}");
                assert!((a[2] - b[2]).abs() < 1.0e-5, "z keep x={x} y={y}");
            }
        }
    }

    #[test]
    fn column_only_variation_has_zero_normal_x_everywhere() {
        // A field that varies only along y has no x-gradient, so normal.x is 0
        // even at the left/right borders -- a direct check of edge wrapping.
        let (w, hgt) = (5u32, 6u32);
        let mut h = vec![0.0f32; (w * hgt) as usize];
        for y in 0..hgt {
            for x in 0..w {
                h[idx(x, y, w)] = y as f32 * 0.3;
            }
        }
        for k in KERNELS {
            for wrap in [WrapMode::ClampToEdge, WrapMode::Repeat] {
                let n = height_to_normal(&h, w, hgt, 1.0, [1.0, 1.0], k, wrap).unwrap();
                for v in &n {
                    assert!(v[0].abs() < 1.0e-6, "{k:?} {wrap:?} nx={}", v[0]);
                }
            }
        }
    }

    #[test]
    fn scharr_exact_on_planar_ramp() {
        // On a planar ramp h = a*x + b*y the true slope is a constant (a, b),
        // and the Scharr `{3,10,3}` weights are normalized so the interior
        // gradient reproduces it exactly -- the primary anti-fake oracle.
        let (w, hgt) = (6u32, 5u32);
        let (a, b) = (0.37f32, -0.21f32);
        let mut h = vec![0.0f32; (w * hgt) as usize];
        for y in 0..hgt {
            for x in 0..w {
                h[idx(x, y, w)] = a * x as f32 + b * y as f32;
            }
        }
        let n = height_to_normal(
            &h,
            w,
            hgt,
            1.0,
            [1.0, 1.0],
            HeightGradient::Scharr,
            WrapMode::ClampToEdge,
        )
        .unwrap();
        let want = slope_to_normal([a, b]);
        for y in 1..hgt - 1 {
            for x in 1..w - 1 {
                let v = n[idx(x, y, w)];
                for c in 0..3 {
                    assert!(
                        (v[c] - want[c]).abs() < 1.0e-5,
                        "x={x} y={y} {v:?} vs {want:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn scharr_agrees_with_other_kernels_on_planar() {
        // Every estimator here is exact on a linear field, so Scharr must match
        // central-difference and Sobel texel-for-texel in the interior.
        let (w, hgt) = (7u32, 6u32);
        let mut h = vec![0.0f32; (w * hgt) as usize];
        for y in 0..hgt {
            for x in 0..w {
                h[idx(x, y, w)] = 0.5 * x as f32 - 0.3 * y as f32 + 2.0;
            }
        }
        let args =
            |k| height_to_normal(&h, w, hgt, 1.3, [1.1, 0.9], k, WrapMode::ClampToEdge).unwrap();
        let sc = args(HeightGradient::Scharr);
        let so = args(HeightGradient::Sobel);
        let cd = args(HeightGradient::CentralDifference);
        for y in 1..hgt - 1 {
            for x in 1..w - 1 {
                let (a, b, c) = (sc[idx(x, y, w)], so[idx(x, y, w)], cd[idx(x, y, w)]);
                for k in 0..3 {
                    assert!((a[k] - b[k]).abs() < 1.0e-5, "scharr vs sobel {a:?} {b:?}");
                    assert!(
                        (a[k] - c[k]).abs() < 1.0e-5,
                        "scharr vs central {a:?} {c:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn scharr_horizontal_mirror_flips_normal_x() {
        // Mirroring the height field in x negates the x-gradient, so the Scharr
        // normal x-component flips while y and z are preserved.
        let (w, hgt) = (7u32, 4u32);
        let mut h = vec![0.0f32; (w * hgt) as usize];
        for y in 0..hgt {
            for x in 0..w {
                h[idx(x, y, w)] = (x * x + 3 * y) as f32 * 0.05;
            }
        }
        let mut hm = vec![0.0f32; (w * hgt) as usize];
        for y in 0..hgt {
            for x in 0..w {
                hm[idx(x, y, w)] = h[idx(w - 1 - x, y, w)];
            }
        }
        let n = height_to_normal(
            &h,
            w,
            hgt,
            1.5,
            [1.0, 1.0],
            HeightGradient::Scharr,
            WrapMode::ClampToEdge,
        )
        .unwrap();
        let nm = height_to_normal(
            &hm,
            w,
            hgt,
            1.5,
            [1.0, 1.0],
            HeightGradient::Scharr,
            WrapMode::ClampToEdge,
        )
        .unwrap();
        for y in 0..hgt {
            for x in 0..w {
                let a = n[idx(x, y, w)];
                let b = nm[idx(w - 1 - x, y, w)];
                assert!(
                    (a[0] + b[0]).abs() < 1.0e-5,
                    "x flip x={x} y={y} {a:?} {b:?}"
                );
                assert!((a[1] - b[1]).abs() < 1.0e-5, "y keep x={x} y={y}");
                assert!((a[2] - b[2]).abs() < 1.0e-5, "z keep x={x} y={y}");
            }
        }
    }
}
