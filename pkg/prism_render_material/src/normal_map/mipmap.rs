//! Variance-preserving normal-map mip reduction with Toksvig specular AA.
//!
//! Averaging unit tangent-space normals across a 2x2 footprint yields a mean
//! vector that is *shorter* than unit whenever the sub-texels disagree. That
//! lost length is exactly the normal variance the coarser mip can no longer
//! represent as geometry -- if it is simply renormalised away, distant surfaces
//! shimmer because the specular lobe is now too tight for the real sub-texel
//! roughness. AAA pipelines (`UE`, Frostbite, `CryEngine`) instead *bake* the
//! lost variance into roughness so the material looks the same at every mip:
//! this is Toksvig's "mipmapping normal maps" correction. Everything here is
//! pure analytic math -- no AI/ML -- so a CPU golden matches a GPU twin.
//!
//! The reducer works on decoded unit normals (see
//! [`decode_rg`](super::decode_rg)) and a parallel roughness grid, so it is
//! decoupled from any particular texture storage; callers decode, reduce, then
//! re-encode. It composes with the colour-path mip filters in
//! [`texture_mipgen`](crate::texture_mipgen).
//!
//! # Conventions
//! * Reduction factor is exactly 2 per axis; a dimension of `1` is carried
//!   through unchanged (GL `max(1, dim >> 1)` rule), matching the colour path.
//! * The averaged normal is renormalised to unit length (falling back to the
//!   geometric `(0,0,1)` when the mean cancels), and its pre-normalisation mean
//!   length drives the roughness correction.
//! * Roughness is treated as the GGX-style parameter `alpha` in `[0,1]`; the
//!   corrected roughness is never *less* than the box-averaged roughness.
//!
//! # References
//! * M. Toksvig, "Mipmapping Normal Maps" (`NVIDIA`, 2005).
//! * Olano & Baker, "LEAN Mapping" (2010) -- related variance approach.
//! * Hill & Baker / Kaplanyan, specular anti-aliasing via roughness.

use alloc::vec::Vec;


/// Geometric fallback when a 2x2 mean normal cancels to near zero.
const GEOMETRIC_NORMAL: [f32; 3] = [0.0, 0.0, 1.0];

/// Average a set of unit normals, returning the renormalised mean direction and
/// the pre-normalisation mean length in `[0, 1]`.
///
/// A mean length of `1` means every input agreed (no lost detail); a length
/// near `0` means the normals cancelled (maximum lost detail). An empty input,
/// or one that cancels, yields the geometric normal and length `0`.
#[must_use]
pub fn average_unit_normals(normals: &[[f32; 3]]) -> ([f32; 3], f32) {
    let n = normals.len();
    if n == 0 {
        return (GEOMETRIC_NORMAL, 0.0);
    }
    let mut sum = [0.0f32; 3];
    for v in normals {
        sum[0] += v[0];
        sum[1] += v[1];
        sum[2] += v[2];
    }
    let inv_n = 1.0 / n as f32;
    let mean = [sum[0] * inv_n, sum[1] * inv_n, sum[2] * inv_n];
    let len2 = mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2];
    if len2 <= 1.0e-12 {
        return (GEOMETRIC_NORMAL, 0.0);
    }
    let len = len2.sqrt();
    let inv = 1.0 / len;
    ([mean[0] * inv, mean[1] * inv, mean[2] * inv], len.clamp(0.0, 1.0))
}

/// Blinn-Phong specular power equivalent to a GGX-style roughness `alpha`.
///
/// `s = 2 / alpha^2 - 2`; `alpha` is clamped away from `0` so a mirror surface
/// maps to a large (but finite) power.
#[must_use]
pub fn power_from_roughness(alpha: f32) -> f32 {
    let a = alpha.clamp(1.0e-3, 1.0);
    2.0 / (a * a) - 2.0
}

/// Inverse of [`power_from_roughness`]: `alpha = sqrt(2 / (s + 2))`.
#[must_use]
pub fn roughness_from_power(power: f32) -> f32 {
    let s = power.max(0.0);
    (2.0 / (s + 2.0)).sqrt().clamp(0.0, 1.0)
}

/// Toksvig attenuation factor `len / (len + s*(1 - len))` in `(0, 1]`.
///
/// `s` is the base specular power; `len` is the mean normal length. A length of
/// `1` returns `1` (nothing lost); shorter lengths return smaller factors,
/// which shrink the effective specular power (i.e. roughen the surface).
#[must_use]
pub fn toksvig_factor(mean_len: f32, power: f32) -> f32 {
    let len = mean_len.clamp(0.0, 1.0);
    let s = power.max(0.0);
    let denom = len + s * (1.0 - len);
    if denom <= 1.0e-8 {
        return 1.0;
    }
    (len / denom).clamp(0.0, 1.0)
}

/// Correct `base_roughness` for the normal variance implied by `mean_len`.
///
/// Converts roughness to specular power, applies the Toksvig factor, and
/// converts back. The result is clamped to `[base_roughness, 1]` so filtering
/// can only *add* roughness, never sharpen -- preserving the specular footprint
/// of the finer mip.
#[must_use]
pub fn toksvig_roughness(mean_len: f32, base_roughness: f32) -> f32 {
    let base = base_roughness.clamp(0.0, 1.0);
    let s = power_from_roughness(base);
    let ft = toksvig_factor(mean_len, s);
    let s_eff = (ft * s).max(1.0e-4);
    roughness_from_power(s_eff).clamp(base, 1.0)
}

/// A dimension can be halved only when it is `1` (carried through) or even.
#[inline]
#[must_use]
fn reducible_dim(dim: u32) -> bool {
    dim == 1 || dim.is_multiple_of(2)
}

/// Reduce a decoded unit-normal grid and its parallel roughness grid by exactly
/// 2 per axis, baking lost normal variance into the output roughness.
///
/// Returns `(normals, roughness, out_w, out_h)` with renormalised normals and
/// Toksvig-corrected roughness, or `None` when the input is already `1x1`, has a
/// non-reducible (odd, >1) dimension, or the two grids disagree in size.
#[must_use]
pub fn reduce_normal_roughness_2x(
    normals: &[[f32; 3]],
    roughness: &[f32],
    w: u32,
    h: u32,
) -> Option<(Vec<[f32; 3]>, Vec<f32>, u32, u32)> {
    let count = (w as usize).checked_mul(h as usize)?;
    if normals.len() != count || roughness.len() != count {
        return None;
    }
    if w == 1 && h == 1 {
        return None;
    }
    if !reducible_dim(w) || !reducible_dim(h) {
        return None;
    }
    let nw = (w / 2).max(1);
    let nh = (h / 2).max(1);
    let step_x = if w > 1 { 2 } else { 1 };
    let step_y = if h > 1 { 2 } else { 1 };
    let span_x = if w > 1 { 2 } else { 1 };
    let span_y = if h > 1 { 2 } else { 1 };

    let mut out_n = Vec::with_capacity((nw * nh) as usize);
    let mut out_r = Vec::with_capacity((nw * nh) as usize);
    for oy in 0..nh {
        for ox in 0..nw {
            let mut group = [[0.0f32; 3]; 4];
            let mut rough_sum = 0.0f32;
            let mut k = 0usize;
            for dy in 0..span_y {
                for dx in 0..span_x {
                    let sx = (ox * step_x + dx).min(w - 1);
                    let sy = (oy * step_y + dy).min(h - 1);
                    let idx = (sy * w + sx) as usize;
                    group[k] = normals[idx];
                    rough_sum += roughness[idx];
                    k += 1;
                }
            }
            let (mean_n, mean_len) = average_unit_normals(&group[..k]);
            let base_rough = rough_sum / k as f32;
            out_n.push(mean_n);
            out_r.push(toksvig_roughness(mean_len, base_rough));
        }
    }
    Some((out_n, out_r, nw, nh))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normal_map::reconstruct::reconstruct_z;
    use alloc::vec;

    fn is_unit(n: [f32; 3]) {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((l - 1.0).abs() < 1.0e-5, "len={l} n={n:?}");
    }

    #[test]
    fn equal_normals_keep_unit_length() {
        let n = [0.0, 0.0, 1.0];
        let (mean, len) = average_unit_normals(&[n, n, n, n]);
        assert!((len - 1.0).abs() < 1.0e-6);
        is_unit(mean);
        assert!((mean[2] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn opposing_normals_cancel_to_geometric() {
        let a = reconstruct_z([0.9, 0.0]);
        let b = [-a[0], -a[1], a[2]];
        let (mean, len) = average_unit_normals(&[a, b]);
        // x/y cancel; the mean is short -> heavily roughened later.
        assert!(len < 1.0);
        is_unit(mean);
    }

    #[test]
    fn empty_is_geometric_zero_len() {
        let (mean, len) = average_unit_normals(&[]);
        assert_eq!(mean, [0.0, 0.0, 1.0]);
        assert_eq!(len, 0.0);
    }

    #[test]
    fn mean_length_is_bounded() {
        let a = reconstruct_z([0.3, -0.2]);
        let b = reconstruct_z([-0.1, 0.4]);
        let (_m, len) = average_unit_normals(&[a, b]);
        assert!((0.0..=1.0).contains(&len));
    }

    #[test]
    fn roughness_power_roundtrips() {
        for &a in &[0.05f32, 0.1, 0.25, 0.5, 0.8, 1.0] {
            let s = power_from_roughness(a);
            let back = roughness_from_power(s);
            assert!((back - a).abs() < 1.0e-4, "a={a} back={back}");
        }
    }

    #[test]
    fn toksvig_factor_identity_and_monotone() {
        let s = power_from_roughness(0.2);
        assert!((toksvig_factor(1.0, s) - 1.0).abs() < 1.0e-6);
        let mut prev = 0.0f32;
        let mut l = 0.0f32;
        while l <= 1.0 {
            let f = toksvig_factor(l, s);
            assert!((0.0..=1.0 + 1e-6).contains(&f));
            assert!(f + 1e-6 >= prev, "factor dropped at len {l}: {f} < {prev}");
            prev = f;
            l += 0.05;
        }
    }

    #[test]
    fn full_length_keeps_base_roughness() {
        for &base in &[0.05f32, 0.3, 0.7] {
            let out = toksvig_roughness(1.0, base);
            assert!((out - base).abs() < 1.0e-3, "base={base} out={out}");
        }
    }

    #[test]
    fn shorter_length_only_roughens() {
        let base = 0.2f32;
        let full = toksvig_roughness(1.0, base);
        let mid = toksvig_roughness(0.7, base);
        let low = toksvig_roughness(0.4, base);
        assert!(full <= mid + 1e-6 && mid <= low + 1e-6, "{full} {mid} {low}");
        assert!(low >= base);
        assert!(low <= 1.0);
    }

    #[test]
    fn reduce_flat_map_is_flat_and_unchanged_roughness() {
        let n = [0.0, 0.0, 1.0];
        let normals = vec![n; 16];
        let roughness = vec![0.3f32; 16];
        let (on, or, nw, nh) = reduce_normal_roughness_2x(&normals, &roughness, 4, 4).unwrap();
        assert_eq!((nw, nh), (2, 2));
        for v in &on {
            is_unit(*v);
            assert!((v[2] - 1.0).abs() < 1e-6);
        }
        for &r in &or {
            // Flat normals lose no detail -> roughness unchanged.
            assert!((r - 0.3).abs() < 1e-3, "r={r}");
        }
    }

    #[test]
    fn reduce_noisy_map_roughens() {
        // A checker of two opposing tilts: box-averaged roughness is 0.1 but the
        // normals disagree, so Toksvig must push roughness up.
        let a = reconstruct_z([0.8, 0.0]);
        let b = reconstruct_z([-0.8, 0.0]);
        let mut normals = alloc::vec::Vec::new();
        for y in 0..4u32 {
            for x in 0..4u32 {
                normals.push(if (x + y) % 2 == 0 { a } else { b });
            }
        }
        let roughness = vec![0.1f32; 16];
        let (_on, or, _nw, _nh) = reduce_normal_roughness_2x(&normals, &roughness, 4, 4).unwrap();
        for &r in &or {
            assert!(r > 0.1, "expected roughening, got {r}");
            assert!(r <= 1.0);
        }
    }

    #[test]
    fn one_by_n_carries_dimension() {
        let n = [0.0, 0.0, 1.0];
        let (on, _or, nw, nh) =
            reduce_normal_roughness_2x(&[n, n, n, n], &[0.5; 4], 4, 1).unwrap();
        assert_eq!((nw, nh), (2, 1));
        assert_eq!(on.len(), 2);
    }

    #[test]
    fn one_by_one_and_odd_and_mismatch_are_none() {
        let n = [0.0, 0.0, 1.0];
        assert!(reduce_normal_roughness_2x(&[n], &[0.5], 1, 1).is_none());
        assert!(reduce_normal_roughness_2x(&[n, n, n], &[0.5; 3], 3, 1).is_none());
        assert!(reduce_normal_roughness_2x(&[n, n, n, n], &[0.5; 3], 4, 1).is_none());
    }
}
