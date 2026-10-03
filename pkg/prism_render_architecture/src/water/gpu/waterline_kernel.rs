//! Water waterline-mask compute kernel: the `WESL` shader plus its bit-exact
//! `CPU` twin.
//!
//! The waterline is where the animated water surface crosses solid geometry;
//! rendering needs a soft above/below transition weight and a shallow-water
//! shoreline band there (see [`waterline`](super::super::waterline) for the
//! golden derivation). This module runs that same derivation as a per-pixel
//! screen-space compute pass.
//!
//! [`WATER_WATERLINE_WESL`] is the shader (entry point `water_waterline_mask`);
//! [`dispatch_waterline_mask`] is its bit-exact `CPU` twin. Because the sandbox
//! has no `GPU`, the twin is the correctness proof: it consumes the identical
//! buffer `ABI` ([`WaterKernel::WaterlineMask`](super::super::kernels::WaterKernel)
//! — no storage buffer, one uniform param block, one sampled scene texture, one
//! `rg32float` storage output texture, dispatched over an 8x8 pixel tile on the
//! `Screen` domain) and reconstructs the shader arithmetic pixel-for-pixel, so
//! the parity test can diff it against the golden
//! [`waterline::waterline_weight`](super::super::waterline::waterline_weight)
//! and [`waterline::shoreline_band`](super::super::waterline::shoreline_band).
//!
//! Only `+ - * /`, comparisons, `clamp` and `max` appear, matching the
//! workspace determinism policy that forbids every float intrinsic but `sqrt`.

use alloc::vec;
use alloc::vec::Vec;

use super::super::waterline::WaterlineParams;
use super::super::EPS;

/// `WESL` source of the water waterline-mask compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_WATERLINE_WESL: &str = include_str!("water_waterline.wesl");

/// Number of `f32` lanes in one sampled scene texel: `(sample_y,
/// water_surface_y, water_depth, pad)`.
///
/// Mirrors the shader's `textureLoad(scene_in, …)` which reads an `rgba` texel
/// and uses the first three channels; the fourth is reserved padding.
pub const WATERLINE_SCENE_FLOATS: usize = 4;

/// Number of `f32` lanes in one output texel: `(waterline_weight,
/// shoreline_band)`.
///
/// Mirrors the shader's `textureStore(mask_out, …, vec4(w, band, 0, 0))` into
/// the `rg32float` target.
pub const WATERLINE_OUT_FLOATS: usize = 2;

/// Soft waterline weight in `0..=1`, reconstructing the shader's
/// `waterline_weight` from the signed submersion `depth` and the transition
/// `half`-width.
///
/// Kept independent of
/// [`waterline::waterline_weight`](super::super::waterline::waterline_weight)
/// so the parity test proves the transcription rather than asserting a
/// tautology.
#[must_use]
fn waterline_weight_twin(depth: f32, half: f32) -> f32 {
    if half <= EPS {
        return if depth >= 0.0 { 1.0 } else { 0.0 };
    }
    ((depth + half) / (2.0 * half)).clamp(0.0, 1.0)
}

/// Shoreline-band weight in `0..=1`, reconstructing the shader's
/// `shoreline_band` from the signed submersion `depth`, the total
/// `water_depth`, and the `shoreline_depth` reach.
#[must_use]
fn shoreline_band_twin(depth: f32, water_depth: f32, shoreline_depth: f32) -> f32 {
    if depth < 0.0 {
        return 0.0;
    }
    if shoreline_depth <= EPS {
        return 0.0;
    }
    let clamped_depth = water_depth.max(0.0);
    (1.0 - clamped_depth / shoreline_depth).clamp(0.0, 1.0)
}

/// Bit-exact `CPU` twin of the `water_waterline_mask` compute pass.
///
/// `scene` is the row-major sampled texture, [`WATERLINE_SCENE_FLOATS`] lanes
/// per texel: `(sample_y, water_surface_y, water_depth, pad)`. The returned
/// buffer is row-major with [`WATERLINE_OUT_FLOATS`] lanes per texel:
/// `(waterline_weight, shoreline_band)`. A `scene` buffer shorter than
/// `width * height` texels leaves the trailing output texels at zero rather
/// than reading out of bounds, matching the shader's guarded dispatch.
#[must_use]
pub fn dispatch_waterline_mask(
    scene: &[f32],
    params: WaterlineParams,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let texels = width * height;
    let mut out = vec![0.0_f32; texels * WATERLINE_OUT_FLOATS];
    let mut t = 0usize;
    while t < texels {
        let base = t * WATERLINE_SCENE_FLOATS;
        if base + WATERLINE_SCENE_FLOATS > scene.len() {
            break;
        }
        let sample_y = scene[base];
        let water_surface_y = scene[base + 1];
        let water_depth = scene[base + 2];

        let depth = water_surface_y - sample_y;
        let w = waterline_weight_twin(depth, params.transition_half_width);
        let band = shoreline_band_twin(depth, water_depth, params.shoreline_depth);

        let o = t * WATERLINE_OUT_FLOATS;
        out[o] = w;
        out[o + 1] = band;
        t += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::super::waterline::{shoreline_band, waterline_weight};
    use super::*;

    const PARAMS: WaterlineParams = WaterlineParams {
        transition_half_width: 0.25,
        shoreline_depth: 1.0,
    };

    fn scene_grid(width: usize, height: usize) -> Vec<f32> {
        // Deterministic ramp of sample heights, water surface, and depth so the
        // kernel exercises air, waterline and submerged shoreline pixels.
        let mut v = Vec::with_capacity(width * height * WATERLINE_SCENE_FLOATS);
        let mut idx = 0usize;
        let total = width * height;
        while idx < total {
            let f = idx as f32;
            let sample_y = -0.6 + f * 0.05;
            let water_surface_y = 0.1 + f * 0.01;
            let water_depth = 0.2 + f * 0.03;
            v.push(sample_y);
            v.push(water_surface_y);
            v.push(water_depth);
            v.push(0.0);
            idx += 1;
        }
        v
    }

    #[test]
    fn twin_matches_cpu_golden_bit_for_bit() {
        let width = 9;
        let height = 7;
        let scene = scene_grid(width, height);
        let out = dispatch_waterline_mask(&scene, PARAMS, width, height);
        let total = width * height;
        let mut t = 0usize;
        while t < total {
            let base = t * WATERLINE_SCENE_FLOATS;
            let sample_y = scene[base];
            let water_surface_y = scene[base + 1];
            let water_depth = scene[base + 2];
            let want_w = waterline_weight(sample_y, water_surface_y, PARAMS);
            let want_band = shoreline_band(sample_y, water_surface_y, water_depth, PARAMS);
            let o = t * WATERLINE_OUT_FLOATS;
            assert_eq!(out[o].to_bits(), want_w.to_bits(), "weight texel {t}");
            assert_eq!(out[o + 1].to_bits(), want_band.to_bits(), "band texel {t}");
            t += 1;
        }
    }

    #[test]
    fn weight_is_zero_in_air_and_one_submerged() {
        // One pixel high in the air, one deep underwater.
        let scene = [
            10.0, 0.0, 0.0, 0.0, // sample_y well above surface -> dry
            -10.0, 0.0, 5.0, 0.0, // sample_y well below surface -> submerged
        ];
        let out = dispatch_waterline_mask(&scene, PARAMS, 2, 1);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[2], 1.0);
    }

    #[test]
    fn degenerate_band_is_a_hard_step() {
        let params = WaterlineParams {
            transition_half_width: 0.0,
            shoreline_depth: 1.0,
        };
        // Exactly at the surface counts as submerged (depth >= 0 -> 1.0).
        let scene = [0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0];
        let out = dispatch_waterline_mask(&scene, params, 2, 1);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn shoreline_band_rises_as_water_gets_shallower() {
        // Submerged samples; shallower total depth -> larger band weight.
        let scene = [
            -1.0, 0.0, 0.1, 0.0, // very shallow
            -1.0, 0.0, 0.9, 0.0, // nearly at the shoreline reach
        ];
        let out = dispatch_waterline_mask(&scene, PARAMS, 2, 1);
        assert!(out[1] > out[3], "shallower water must band harder");
    }

    #[test]
    fn short_buffers_do_not_panic() {
        // Claim a 4x4 grid but only supply two texels of scene data.
        let scene = [0.0, 1.0, 0.5, 0.0, -1.0, 1.0, 0.5, 0.0];
        let out = dispatch_waterline_mask(&scene, PARAMS, 4, 4);
        assert_eq!(out.len(), 16 * WATERLINE_OUT_FLOATS);
        // Trailing texels past the supplied data stay zero.
        assert_eq!(out[2 * WATERLINE_OUT_FLOATS], 0.0);
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        assert!(WATER_WATERLINE_WESL.contains("fn water_waterline_mask"));
        assert!(WATER_WATERLINE_WESL.contains("@workgroup_size(8, 8, 1)"));
        assert!(WATER_WATERLINE_WESL.contains("texture_storage_2d<rg32float, write>"));
        assert!(WATER_WATERLINE_WESL.contains("struct WaterlineMaskParams"));
    }

    #[test]
    fn scene_and_out_strides_are_consistent() {
        assert_eq!(WATERLINE_SCENE_FLOATS, 4);
        assert_eq!(WATERLINE_OUT_FLOATS, 2);
    }
}
