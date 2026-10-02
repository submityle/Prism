//! Alpha-test coverage preservation for mip chains (`RGBA8`).
//!
//! Alpha-tested materials (foliage, chain-link, decals) compare a texture's
//! alpha against a fixed cutoff. A naively down-sampled mip averages alpha, so
//! the fraction of texels that pass the cutoff drifts with mip level: foliage
//! thins out and shimmers in the distance. AAA pipelines (`UE`'s *Alpha
//! Coverage* preserve, `NVIDIA` Texture Tools `setAlphaTestCoverage`, Unity's
//! "preserve coverage") fix this by rescaling each mip's alpha so its passing
//! fraction matches mip 0. This module implements that purely numerically on
//! the shared [`Rgba8Image`] model -- a deterministic CPU golden, no AI/ML.
//!
//! The reduction itself is produced by any mip filter in this module
//! ([`windowed`](super::windowed), [`kaiser`](super::kaiser),
//! [`box_filter`](super::box_filter)); coverage preservation is a decoupled
//! post-pass over an already-built chain, so it composes with every filter.
//!
//! # Conventions
//! * `threshold` is the alpha-test cutoff as a raw `u8` (e.g. `128` for 0.5).
//! * Coverage is the fraction of texels whose (scaled) alpha is `>= threshold`.
//!   It is monotonically non-decreasing in the alpha scale, so a bisection
//!   recovers the scale that reproduces a target coverage.
//! * Only alpha is modified; RGB is untouched. Alpha is treated as linear
//!   coverage regardless of the colour space used to build the chain.
//!
//! # References
//! * I. Castano, "Computing Alpha Mipmaps" (`NVIDIA`, 2010).
//! * `NVTT` `nvtt::Surface::setAlphaTestCoverage` / `scaleAlphaToCoverage`.

use alloc::vec::Vec;

use super::box_filter::Rgba8Image;

/// Fraction of texels whose alpha, multiplied by `alpha_scale` and clamped to
/// `[0,1]`, is at least the `threshold` cutoff. Returns `0` for an empty image.
#[must_use]
pub fn alpha_test_coverage(img: &Rgba8Image, threshold: u8, alpha_scale: f32) -> f32 {
    let texels = img.as_slice();
    let n = texels.len();
    if n == 0 {
        return 0.0;
    }
    let t = f32::from(threshold) / 255.0;
    let mut passed = 0u64;
    for px in texels {
        let a = (f32::from(px[3]) / 255.0 * alpha_scale).clamp(0.0, 1.0);
        if a >= t {
            passed += 1;
        }
    }
    passed as f32 / n as f32
}

/// Bisect for the alpha scale that makes [`alpha_test_coverage`] match
/// `desired_coverage` as closely as the per-texel granularity allows.
///
/// Coverage is monotone non-decreasing in the scale, so bisection is exact up
/// to one texel. The upper bound is grown geometrically (capped) so even a mip
/// that has lost most of its passing texels can be pushed back to the target.
#[must_use]
pub fn solve_alpha_scale(img: &Rgba8Image, threshold: u8, desired_coverage: f32) -> f32 {
    let target = desired_coverage.clamp(0.0, 1.0);
    if target <= 0.0 {
        return 0.0;
    }
    // A fully transparent image can never reach a positive coverage; report the
    // largest scale we tried so callers get a well-defined (saturating) value.
    let mut lo = 0.0f32;
    let mut hi = 4.0f32;
    while alpha_test_coverage(img, threshold, hi) < target && hi < 1_048_576.0 {
        hi *= 2.0;
    }
    // 24 iterations resolve the scale to ~1e-7 of the final bracket width.
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if alpha_test_coverage(img, threshold, mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Return a copy of `img` with alpha multiplied by `scale` (clamped to `[0,1]`
/// then re-quantised round-half-up); RGB is copied unchanged.
#[must_use]
pub fn apply_alpha_scale(img: &Rgba8Image, scale: f32) -> Rgba8Image {
    let texels: Vec<[u8; 4]> = img
        .as_slice()
        .iter()
        .map(|p| {
            let a = (f32::from(p[3]) / 255.0 * scale).clamp(0.0, 1.0);
            let au8 = (a * 255.0 + 0.5).floor() as u8;
            [p[0], p[1], p[2], au8]
        })
        .collect();
    // Dimensions are preserved and non-zero, so reconstruction always succeeds.
    Rgba8Image::new(img.width(), img.height(), texels)
        .expect("alpha rescale preserves the (non-zero) image dimensions")
}

/// Rescale the alpha of every mip below level 0 so its alpha-test coverage
/// matches mip 0's coverage at `threshold`. Mip 0 and RGB are left untouched.
pub fn preserve_alpha_coverage(chain: &mut [Rgba8Image], threshold: u8) {
    if chain.is_empty() {
        return;
    }
    let desired = alpha_test_coverage(&chain[0], threshold, 1.0);
    for level in chain.iter_mut().skip(1) {
        let scale = solve_alpha_scale(level, threshold, desired);
        *level = apply_alpha_scale(level, scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn img(w: u32, h: u32, texels: Vec<[u8; 4]>) -> Rgba8Image {
        Rgba8Image::new(w, h, texels).unwrap()
    }

    fn solid(w: u32, h: u32, c: [u8; 4]) -> Rgba8Image {
        img(w, h, vec![c; (w * h) as usize])
    }

    #[test]
    fn coverage_bounds() {
        let opaque = solid(4, 4, [10, 20, 30, 255]);
        assert_eq!(alpha_test_coverage(&opaque, 128, 1.0), 1.0);
        let clear = solid(4, 4, [10, 20, 30, 0]);
        assert_eq!(alpha_test_coverage(&clear, 128, 1.0), 0.0);
    }

    #[test]
    fn coverage_is_monotone_in_scale() {
        // A graded alpha ramp 0,16,32,...; higher scale cannot lower coverage.
        let mut texels = Vec::new();
        for i in 0..16u32 {
            let a = (i * 16).min(255) as u8;
            texels.push([0, 0, 0, a]);
        }
        let g = img(4, 4, texels);
        let mut prev = 0.0f32;
        let mut s = 0.0f32;
        while s <= 8.0 {
            let c = alpha_test_coverage(&g, 128, s);
            assert!(c + 1e-6 >= prev, "coverage dropped at scale {s}: {c} < {prev}");
            prev = c;
            s += 0.25;
        }
    }

    #[test]
    fn scale_one_is_identity() {
        let src = img(2, 2, vec![[1, 2, 3, 40], [4, 5, 6, 200], [7, 8, 9, 0], [1, 1, 1, 255]]);
        let out = apply_alpha_scale(&src, 1.0);
        assert_eq!(out.as_slice(), src.as_slice());
    }

    #[test]
    fn apply_scale_clamps_alpha() {
        let src = solid(2, 2, [0, 0, 0, 200]);
        let out = apply_alpha_scale(&src, 4.0);
        for p in out.as_slice() {
            assert_eq!(p[3], 255);
        }
    }

    #[test]
    fn solve_recovers_target_coverage() {
        // 8x8 graded alpha; ask for a few target coverages and check the solved
        // scale reproduces them to within one texel (the count granularity).
        let n = 64u32;
        let mut texels = Vec::new();
        for i in 0..n {
            let a = ((i * 255) / (n - 1)) as u8;
            texels.push([0, 0, 0, a]);
        }
        let g = img(8, 8, texels);
        let gran = 1.0 / n as f32;
        for &target in &[0.25f32, 0.5, 0.75] {
            let scale = solve_alpha_scale(&g, 128, target);
            let got = alpha_test_coverage(&g, 128, scale);
            assert!((got - target).abs() <= gran + 1e-6, "target {target}: got {got}");
        }
    }

    #[test]
    fn preserve_restores_coverage_across_chain() {
        // Base: 16x16, exactly half opaque / half fully transparent -> 0.5
        // coverage at threshold 128. A box-blurred chain would drift; after the
        // preserve pass every level must match 0.5 to within its own texel
        // granularity.
        let mut base_texels = Vec::new();
        for y in 0..16u32 {
            for _x in 0..16u32 {
                let a = if y < 8 { 255 } else { 0 };
                base_texels.push([200, 100, 50, a]);
            }
        }
        let base = img(16, 16, base_texels);
        let desired = alpha_test_coverage(&base, 128, 1.0);
        assert!((desired - 0.5).abs() < 1e-6);

        // Build a simple averaged chain by hand (box 2x2 on alpha) so this test
        // does not depend on another filter module.
        let mut chain = vec![base.clone()];
        let mut cur = base;
        while cur.width() > 1 || cur.height() > 1 {
            let (w, h) = (cur.width(), cur.height());
            let (nw, nh) = (w.div_ceil(2).max(1), h.div_ceil(2).max(1));
            let mut nt = Vec::new();
            for oy in 0..nh {
                for ox in 0..nw {
                    let mut sum = [0u32; 4];
                    let mut cnt = 0u32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let sx = (ox * 2 + dx).min(w - 1);
                            let sy = (oy * 2 + dy).min(h - 1);
                            let p = cur.as_slice()[(sy * w + sx) as usize];
                            for c in 0..4 {
                                sum[c] += u32::from(p[c]);
                            }
                            cnt += 1;
                        }
                    }
                    nt.push([
                        (sum[0] / cnt) as u8,
                        (sum[1] / cnt) as u8,
                        (sum[2] / cnt) as u8,
                        (sum[3] / cnt) as u8,
                    ]);
                }
            }
            cur = img(nw, nh, nt);
            chain.push(cur.clone());
        }

        preserve_alpha_coverage(&mut chain, 128);
        for level in &chain {
            let n = (level.width() * level.height()) as f32;
            let cov = alpha_test_coverage(level, 128, 1.0);
            assert!(
                (cov - desired).abs() <= 1.0 / n + 1e-6,
                "level {}x{} coverage {cov} off target {desired}",
                level.width(),
                level.height()
            );
        }
    }

    #[test]
    fn fully_transparent_target_zero_gives_zero_scale() {
        let clear = solid(4, 4, [0, 0, 0, 0]);
        assert_eq!(solve_alpha_scale(&clear, 128, 0.0), 0.0);
    }

    #[test]
    fn empty_chain_is_noop() {
        let mut chain: Vec<Rgba8Image> = Vec::new();
        preserve_alpha_coverage(&mut chain, 128);
        assert!(chain.is_empty());
    }
}
