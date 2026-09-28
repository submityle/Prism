//! GGX-prefiltered environment radiance (the "L" half of Karis 2013 split-sum).
//!
//! The split-sum approximation factors the specular IBL integral into a scalar
//! environment BRDF (see [`super::brdf_lut`]) and a *prefiltered* radiance map
//! that pre-convolves the environment with the GGX lobe for a set of
//! roughness levels.  Rougher reflections read coarser mips, so a single
//! `textureSampleLevel` reconstructs blurry specular in one tap, matching
//! Unreal's `FilterCubeMap` / `PrefilterEnvMap`.
//!
//! This module is the backend-neutral CPU reference: it prefilters a source
//! [`CubemapFaces`] into a mip chain of [`CubemapFaces`], each baked at an
//! increasing roughness, and reconstructs a trilinear sample the GPU
//! `env_prefilter.wesl` twin reproduces.  All transcendentals route through
//! `bevy_math::ops` for cross-platform determinism.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use super::cubemap::{face_direction, CubemapFaces, FACE_COUNT};
use super::sampling::{hammersley, importance_sample_ggx};

/// Convolves the `source` environment with the GGX lobe about `direction`.
///
/// Following Karis, the normal, view, and reflection are all assumed equal to
/// `direction` (the isotropic split-sum approximation), so the lobe is centred
/// on the reflected view.  For each Hammersley sample a GGX half vector `H` is
/// drawn in the tangent frame around `direction`, the light direction
/// `L = reflect(-V, H)` is formed, and samples with `n_dot_l > 0` are
/// accumulated weighted by `n_dot_l`.  At `roughness == 0` every `H` collapses
/// to the normal, so the result is the sharp mirror sample.
///
/// Returns a linear-RGB radiance; a source with no lit samples (all `L` below
/// the horizon, which cannot happen for the centred lobe) falls back to the
/// direct `direction` sample.
pub fn prefilter_radiance(
    source: &CubemapFaces,
    direction: [f32; 3],
    roughness: f32,
    sample_count: u32,
) -> [f32; 3] {
    let n = normalize(direction).unwrap_or([0.0, 1.0, 0.0]);
    let roughness = roughness.clamp(0.0, 1.0);
    let samples = sample_count.max(1);

    // Mirror path: the lobe is a delta at the normal, so skip the loop and
    // return the sharp reflection to avoid importance-sampling jitter.
    if roughness == 0.0 {
        return source.sample(n);
    }

    // Orthonormal tangent frame around the normal `n` (Duff et al. would be
    // faster, but this mirrors the shader's explicit `up`-cross construction).
    let up = if n[2].abs() < 0.999 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let tangent = normalize(cross(up, n)).unwrap_or([1.0, 0.0, 0.0]);
    let bitangent = cross(n, tangent);

    // V == R == N in the isotropic approximation.
    let view = n;

    let mut color = [0.0f32; 3];
    let mut total_weight = 0.0f32;
    for i in 0..samples {
        let xi = hammersley(i, samples);
        // Tangent-space GGX half vector, then to world via the frame.
        let ht = importance_sample_ggx(xi, roughness);
        let h = [
            tangent[0] * ht[0] + bitangent[0] * ht[1] + n[0] * ht[2],
            tangent[1] * ht[0] + bitangent[1] * ht[1] + n[1] * ht[2],
            tangent[2] * ht[0] + bitangent[2] * ht[1] + n[2] * ht[2],
        ];
        let v_dot_h = dot(view, h);
        // L = reflect(-V, H) = 2 (V.H) H - V.
        let light = [
            2.0 * v_dot_h * h[0] - view[0],
            2.0 * v_dot_h * h[1] - view[1],
            2.0 * v_dot_h * h[2] - view[2],
        ];
        let n_dot_l = dot(n, light);
        if n_dot_l > 0.0 {
            let radiance = source.sample(light);
            color[0] += radiance[0] * n_dot_l;
            color[1] += radiance[1] * n_dot_l;
            color[2] += radiance[2] * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    if total_weight > 1.0e-6 {
        let inv = total_weight.recip();
        [color[0] * inv, color[1] * inv, color[2] * inv]
    } else {
        source.sample(n)
    }
}

/// A GGX-prefiltered environment radiance map: one cube per roughness mip.
///
/// Mip `0` is the sharpest (roughness `0`) and each subsequent mip bakes a
/// higher roughness across `[0, 1]`, with the face resolution halving per mip
/// (clamped to a `1x1` floor) to match a GPU mip chain.  Reflections read the
/// mip whose baked roughness matches the surface, trilinearly blending across
/// adjacent mips.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefilteredEnvMap {
    /// Per-mip prefiltered cubes, index `0` = roughness `0` (sharpest).
    pub mips: Vec<CubemapFaces>,
}

impl PrefilteredEnvMap {
    /// Prefilters `source` into `mip_count` roughness levels.
    ///
    /// Mip `i` bakes roughness `i / (mip_count - 1)` at face resolution
    /// `max(base_size >> i, 1)` using `sample_count` GGX samples per texel.
    /// Returns `None` when `mip_count` or `base_size` is zero so callers can
    /// fall back to the low-frequency SH probe.
    pub fn generate(
        source: &CubemapFaces,
        mip_count: u32,
        base_size: u32,
        sample_count: u32,
    ) -> Option<Self> {
        if mip_count == 0 || base_size == 0 {
            return None;
        }
        let denom = (mip_count.max(2) - 1) as f32;
        let mut mips = Vec::with_capacity(mip_count as usize);
        for mip in 0..mip_count {
            let size = (base_size >> mip).max(1);
            let roughness = if mip_count == 1 {
                0.0
            } else {
                (mip as f32) / denom
            };
            let inv = (size as f32).recip();
            let mut faces: [Vec<[f32; 3]>; 6] = core::array::from_fn(|_| {
                vec![[0.0f32; 3]; (size as usize) * (size as usize)]
            });
            for (face_index, face) in faces.iter_mut().enumerate().take(FACE_COUNT) {
                for y in 0..size {
                    // Texel centres mapped to the in-face [-1, 1] extent.
                    let v = ((y as f32) + 0.5) * inv * 2.0 - 1.0;
                    for x in 0..size {
                        let u = ((x as f32) + 0.5) * inv * 2.0 - 1.0;
                        let dir = face_direction(face_index, u, v);
                        face[(y * size + x) as usize] =
                            prefilter_radiance(source, dir, roughness, sample_count);
                    }
                }
            }
            // SAFETY of unwrap: `size >= 1` and every face is `size * size`.
            mips.push(CubemapFaces::new(size, faces)?);
        }
        Some(Self { mips })
    }

    /// Number of baked roughness mips.
    pub fn mip_count(&self) -> u32 {
        self.mips.len() as u32
    }

    /// Trilinearly samples the prefiltered radiance along `direction` for a
    /// surface `roughness`.
    ///
    /// `roughness` selects a fractional mip `lod = roughness * (mip_count - 1)`;
    /// the two bracketing mips are sampled (each already bilinear within a
    /// face) and linearly blended, mirroring a GPU `textureSampleLevel`.
    pub fn sample(&self, direction: [f32; 3], roughness: f32) -> [f32; 3] {
        if self.mips.is_empty() {
            return [0.0; 3];
        }
        if self.mips.len() == 1 {
            return self.mips[0].sample(direction);
        }
        let max_mip = (self.mips.len() - 1) as f32;
        let lod = (roughness.clamp(0.0, 1.0) * max_mip).clamp(0.0, max_mip);
        let lo = lod.floor() as usize;
        let hi = (lo + 1).min(self.mips.len() - 1);
        let t = lod - (lo as f32);
        let a = self.mips[lo].sample(direction);
        let b = self.mips[hi].sample(direction);
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = ops::sqrt(dot(v, v));
    if len > 1.0e-8 {
        let inv = len.recip();
        Some([v[0] * inv, v[1] * inv, v[2] * inv])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Builds a cube whose every texel is `color`.
    fn constant_cube(size: u32, color: [f32; 3]) -> CubemapFaces {
        let face = vec![color; (size as usize) * (size as usize)];
        CubemapFaces::new(size, core::array::from_fn(|_| face.clone())).unwrap()
    }

    /// Builds a cube where each face is a single flat `color`.
    fn faces_from_colors(size: u32, colors: [[f32; 3]; 6]) -> CubemapFaces {
        let faces = core::array::from_fn(|i| vec![colors[i]; (size as usize) * (size as usize)]);
        CubemapFaces::new(size, faces).unwrap()
    }

    #[test]
    fn constant_environment_stays_constant() {
        let src = constant_cube(8, [0.3, 0.6, 0.9]);
        for roughness in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for dir in [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-0.4, 0.7, 0.3],
            ] {
                let c = prefilter_radiance(&src, dir, roughness, 64);
                assert!((c[0] - 0.3).abs() < 1.0e-3, "r={roughness} dir={dir:?} -> {c:?}");
                assert!((c[1] - 0.6).abs() < 1.0e-3);
                assert!((c[2] - 0.9).abs() < 1.0e-3);
            }
        }
    }

    #[test]
    fn mirror_roughness_matches_source_sample() {
        // A directional gradient across faces; roughness 0 must equal a direct
        // sharp sample of the source.
        let src = faces_from_colors(
            4,
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
        );
        for dir in [
            [1.0, 0.1, 0.0],
            [0.0, 1.0, 0.2],
            [0.1, 0.0, 1.0],
            [-1.0, 0.0, 0.1],
        ] {
            let sharp = src.sample(dir);
            let filtered = prefilter_radiance(&src, dir, 0.0, 32);
            for k in 0..3 {
                assert!((sharp[k] - filtered[k]).abs() < 1.0e-5);
            }
        }
    }

    #[test]
    fn bright_face_dominates_its_hemisphere() {
        // Only +Y is bright; the prefiltered radiance looking up must be
        // brighter than looking down at any non-mirror roughness.
        let mut colors = [[0.0f32; 3]; 6];
        colors[2] = [4.0, 4.0, 4.0]; // +Y
        let src = faces_from_colors(16, colors);
        let up = prefilter_radiance(&src, [0.0, 1.0, 0.0], 0.4, 256);
        let down = prefilter_radiance(&src, [0.0, -1.0, 0.0], 0.4, 256);
        assert!(up[0] > down[0], "up={up:?} down={down:?}");
        assert!(up[0] > 0.0);
    }

    #[test]
    fn higher_roughness_blurs_toward_neighbourhood() {
        // Sharp step between +X (bright) and its neighbours. A rougher filter
        // pulls the +X sample down as it averages in darker directions.
        let mut colors = [[0.0f32; 3]; 6];
        colors[0] = [8.0, 8.0, 8.0]; // +X only
        let src = faces_from_colors(16, colors);
        let dir = [1.0, 0.0, 0.0];
        let sharp = prefilter_radiance(&src, dir, 0.05, 256);
        let rough = prefilter_radiance(&src, dir, 0.8, 256);
        assert!(rough[0] < sharp[0], "sharp={sharp:?} rough={rough:?}");
        assert!(rough[0] >= 0.0);
    }

    #[test]
    fn results_are_finite_and_non_negative() {
        let mut colors = [[0.0f32; 3]; 6];
        colors[4] = [2.0, 3.0, 1.0];
        let src = faces_from_colors(8, colors);
        for roughness in [0.0, 0.3, 0.6, 1.0] {
            for i in 0..12 {
                let a = (i as f32) * 0.5;
                let dir = [ops::cos(a), 0.2, ops::sin(a)];
                let c = prefilter_radiance(&src, dir, roughness, 64);
                for &value in c.iter().take(3) {
                    assert!(value.is_finite());
                    assert!(value >= -1.0e-6);
                }
            }
        }
    }

    #[test]
    fn generate_builds_halving_mip_chain() {
        let src = constant_cube(16, [0.5, 0.5, 0.5]);
        let map = PrefilteredEnvMap::generate(&src, 4, 16, 32).unwrap();
        assert_eq!(map.mip_count(), 4);
        assert_eq!(map.mips[0].size, 16);
        assert_eq!(map.mips[1].size, 8);
        assert_eq!(map.mips[2].size, 4);
        assert_eq!(map.mips[3].size, 2);
    }

    #[test]
    fn generate_preserves_constant_across_mips() {
        let src = constant_cube(8, [0.2, 0.4, 0.8]);
        let map = PrefilteredEnvMap::generate(&src, 5, 8, 32).unwrap();
        for roughness in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let c = map.sample([0.3, 0.5, -0.2], roughness);
            assert!((c[0] - 0.2).abs() < 2.0e-3, "r={roughness} -> {c:?}");
            assert!((c[1] - 0.4).abs() < 2.0e-3);
            assert!((c[2] - 0.8).abs() < 2.0e-3);
        }
    }

    #[test]
    fn trilinear_sample_blends_between_mips() {
        // Two-mip map with a known contrast; a mid roughness must land between
        // the sharp and rough mip samples.
        let mut colors = [[0.0f32; 3]; 6];
        colors[2] = [4.0, 4.0, 4.0];
        let src = faces_from_colors(16, colors);
        let map = PrefilteredEnvMap::generate(&src, 3, 16, 128).unwrap();
        let dir = [0.0, 1.0, 0.0];
        let sharp = map.mips[0].sample(dir);
        let rough = map.mips[2].sample(dir);
        let mid = map.sample(dir, 0.5);
        let lo = sharp[0].min(rough[0]);
        let hi = sharp[0].max(rough[0]);
        assert!(mid[0] >= lo - 1.0e-4 && mid[0] <= hi + 1.0e-4, "mid={mid:?} sharp={sharp:?} rough={rough:?}");
    }

    #[test]
    fn zero_mip_count_returns_none() {
        let src = constant_cube(4, [1.0, 1.0, 1.0]);
        assert!(PrefilteredEnvMap::generate(&src, 0, 4, 16).is_none());
        assert!(PrefilteredEnvMap::generate(&src, 4, 0, 16).is_none());
    }
}
