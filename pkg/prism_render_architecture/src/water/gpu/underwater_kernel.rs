//! Water underwater-volume froxel compute kernel: the `WESL` shader plus its
//! bit-exact `CPU` twin.
//!
//! Below the surface the water is a participating medium: light is absorbed and
//! out-scattered along every eye path, so distant geometry fades, warm colours
//! vanish first, and shafts of light (god rays) glow in the haze (see
//! [`underwater`](super::super::underwater) for the golden derivation). This
//! pass runs that same per-cell math across the shared froxel volume: a
//! per-channel `Beer-Lambert` depth colour shift, a bounded multiple-scattering
//! boost, and an achromatic god-ray in-scatter term.
//!
//! [`WATER_UNDERWATER_VOLUME_WESL`] is the shader (entry point
//! `water_underwater_volume`); [`dispatch_underwater_volume`] is its bit-exact
//! `CPU` twin. Because the sandbox has no `GPU`, the twin is the correctness
//! proof: it consumes the identical buffer `ABI`
//! ([`WaterKernel::UnderwaterVolume`](super::super::kernels::WaterKernel) — no
//! storage buffer, one uniform param block, one sampled `texture_3d`, one
//! `rgba32float` storage output, 4x4x4 brick, `Grid3d` domain) and reconstructs
//! the shader arithmetic cell-for-cell. The composite is diffed against the
//! golden [`underwater::depth_color_shift`](super::super::underwater::depth_color_shift),
//! [`underwater::multiple_scatter_boost`](super::super::underwater::multiple_scatter_boost)
//! and [`underwater::godray_inscatter`](super::super::underwater::godray_inscatter),
//! so the whole pass is bit-exact rather than a floating approximation.
//!
//! Extinction uses the crate's monotone
//! [`exp_approx`](super::super::exp_approx), the one shared transcendental leaf
//! both sides call; the composite formulae are reconstructed inline. Only
//! `+ - * /`, comparisons and `clamp`/`max` appear, matching the workspace
//! determinism policy that forbids every float intrinsic but `sqrt`.

use alloc::vec;
use alloc::vec::Vec;

use super::super::exp_approx;

/// `WESL` source of the water underwater-volume froxel compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_UNDERWATER_VOLUME_WESL: &str = include_str!("water_underwater_volume.wesl");

/// Number of `f32` lanes in one froxel input cell: `(color_r, color_g,
/// color_b, path_length)`.
pub const UNDERWATER_FROXEL_FLOATS: usize = 4;

/// Number of `f32` lanes in one output cell: `(r, g, b, a)`.
pub const UNDERWATER_OUT_FLOATS: usize = 4;

/// Uniform parameter block for the underwater-volume pass.
///
/// `ext_r`/`ext_g`/`ext_b` are the per-channel extinction coefficients
/// (absorption plus out-scatter); real water attenuates red fastest, so
/// `r > g > b` drives the blue-green depth cast. `scatter_albedo` feeds both
/// the bounded multiple-scattering boost and the god-ray glow; `surface_light`
/// is the radiance focused through the surface, and `godray_extinction` is the
/// shaft extinction used for the in-scatter term. The grid dimensions travel as
/// explicit [`dispatch_underwater_volume`] arguments rather than struct fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnderwaterParams {
    /// Red-channel extinction coefficient.
    pub ext_r: f32,
    /// Green-channel extinction coefficient.
    pub ext_g: f32,
    /// Blue-channel extinction coefficient.
    pub ext_b: f32,
    /// Single-scattering albedo for the glow boost and god-ray term.
    pub scatter_albedo: f32,
    /// Surface light focused into the shaft.
    pub surface_light: f32,
    /// Extinction coefficient for the god-ray shaft.
    pub godray_extinction: f32,
}

/// `Beer-Lambert` transmittance `exp(-extinction * distance)` clamped to
/// `0..=1`, reconstructing the shader's `beer_lambert_transmittance`.
///
/// Kept independent of
/// [`underwater::beer_lambert_transmittance`](super::super::underwater::beer_lambert_transmittance)
/// so the parity test proves the transcription rather than asserting a
/// tautology. `exp_approx` is the shared transcendental leaf (like `sqrt`), so
/// it is called directly on both sides.
#[must_use]
fn beer_lambert_twin(extinction: f32, distance: f32) -> f32 {
    let e = extinction.max(0.0);
    let d = distance.max(0.0);
    exp_approx(-(e * d)).clamp(0.0, 1.0)
}

/// Bounded multiple-scattering amplification `single / (1 - albedo)`,
/// reconstructing the shader's `multiple_scatter_boost`.
#[must_use]
fn multiple_scatter_boost_twin(single: f32, albedo: f32) -> f32 {
    let a = albedo.clamp(0.0, 0.999);
    single.max(0.0) / (1.0 - a)
}

/// God-ray in-scatter `surface_light * albedo * (1 - transmittance)`,
/// reconstructing the shader's `godray_inscatter`.
#[must_use]
fn godray_twin(surface_light: f32, scatter_albedo: f32, extinction: f32, path_length: f32) -> f32 {
    let light = surface_light.max(0.0);
    let albedo = scatter_albedo.clamp(0.0, 1.0);
    let transmittance = beer_lambert_twin(extinction, path_length);
    light * albedo * (1.0 - transmittance)
}

/// Bit-exact `CPU` twin of the `water_underwater_volume` compute pass.
///
/// `froxel` is the `z`-major flattened volume, [`UNDERWATER_FROXEL_FLOATS`]
/// lanes per cell: `(color_r, color_g, color_b, path_length)`, indexed
/// `((z * height + y) * width + x)`. The returned buffer has the same layout
/// with [`UNDERWATER_OUT_FLOATS`] lanes per cell: the attenuated, scattered
/// in-water radiance with alpha pinned to one. A `froxel` buffer shorter than
/// `width * height * depth` cells leaves the trailing output cells at zero
/// rather than reading out of bounds, matching the shader's guarded dispatch.
#[must_use]
pub fn dispatch_underwater_volume(
    froxel: &[f32],
    params: UnderwaterParams,
    width: usize,
    height: usize,
    depth: usize,
) -> Vec<f32> {
    let cells = width * height * depth;
    let mut out = vec![0.0_f32; cells * UNDERWATER_OUT_FLOATS];
    let mut idx = 0usize;
    while idx < cells {
        let fbase = idx * UNDERWATER_FROXEL_FLOATS;
        if fbase + UNDERWATER_FROXEL_FLOATS > froxel.len() {
            break;
        }
        let o = idx * UNDERWATER_OUT_FLOATS;
        let color_r = froxel[fbase].max(0.0);
        let color_g = froxel[fbase + 1].max(0.0);
        let color_b = froxel[fbase + 2].max(0.0);
        let path_length = froxel[fbase + 3];

        // Per-channel `Beer-Lambert` depth colour shift (red collapses first).
        let shifted_r = color_r * beer_lambert_twin(params.ext_r, path_length);
        let shifted_g = color_g * beer_lambert_twin(params.ext_g, path_length);
        let shifted_b = color_b * beer_lambert_twin(params.ext_b, path_length);

        // Bounded multiple-scattering glow lift.
        let boosted_r = multiple_scatter_boost_twin(shifted_r, params.scatter_albedo);
        let boosted_g = multiple_scatter_boost_twin(shifted_g, params.scatter_albedo);
        let boosted_b = multiple_scatter_boost_twin(shifted_b, params.scatter_albedo);

        // Achromatic god-ray in-scatter added along the shaft.
        let glow = godray_twin(
            params.surface_light,
            params.scatter_albedo,
            params.godray_extinction,
            path_length,
        );

        out[o] = boosted_r + glow;
        out[o + 1] = boosted_g + glow;
        out[o + 2] = boosted_b + glow;
        out[o + 3] = 1.0;
        idx += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::super::underwater::{
        depth_color_shift, godray_inscatter, multiple_scatter_boost, RgbColor, RgbExtinction,
    };
    use super::*;

    const PARAMS: UnderwaterParams = UnderwaterParams {
        ext_r: 0.6,
        ext_g: 0.2,
        ext_b: 0.05,
        scatter_albedo: 0.4,
        surface_light: 1.5,
        godray_extinction: 0.3,
    };

    #[test]
    fn composite_matches_golden_bit_for_bit() {
        // A handful of cells with varying colour and path length; every output
        // channel must equal the golden depth-shift -> boost -> + glow pipeline
        // reconstructed from the independent `underwater` module, bit-for-bit.
        let width = 2;
        let height = 2;
        let depth = 2;
        let cells = width * height * depth;
        let mut froxel = Vec::with_capacity(cells * UNDERWATER_FROXEL_FLOATS);
        let mut c = 0usize;
        while c < cells {
            let f = c as f32;
            froxel.push(0.9 - 0.05 * f);
            froxel.push(0.6 + 0.02 * f);
            froxel.push(0.3 + 0.04 * f);
            froxel.push(0.5 + 0.7 * f);
            c += 1;
        }
        let out = dispatch_underwater_volume(&froxel, PARAMS, width, height, depth);

        let ext = RgbExtinction {
            r: PARAMS.ext_r,
            g: PARAMS.ext_g,
            b: PARAMS.ext_b,
        };
        let mut i = 0usize;
        while i < cells {
            let fbase = i * UNDERWATER_FROXEL_FLOATS;
            let color = RgbColor {
                r: froxel[fbase],
                g: froxel[fbase + 1],
                b: froxel[fbase + 2],
            };
            let path = froxel[fbase + 3];
            let shifted = depth_color_shift(color, ext, path);
            let glow = godray_inscatter(
                PARAMS.surface_light,
                PARAMS.scatter_albedo,
                PARAMS.godray_extinction,
                path,
            );
            let want_r = multiple_scatter_boost(shifted.r, PARAMS.scatter_albedo) + glow;
            let want_g = multiple_scatter_boost(shifted.g, PARAMS.scatter_albedo) + glow;
            let want_b = multiple_scatter_boost(shifted.b, PARAMS.scatter_albedo) + glow;
            let o = i * UNDERWATER_OUT_FLOATS;
            assert_eq!(out[o].to_bits(), want_r.to_bits(), "red cell {i}");
            assert_eq!(out[o + 1].to_bits(), want_g.to_bits(), "green cell {i}");
            assert_eq!(out[o + 2].to_bits(), want_b.to_bits(), "blue cell {i}");
            assert_eq!(out[o + 3].to_bits(), 1.0_f32.to_bits(), "alpha cell {i}");
            i += 1;
        }
    }

    #[test]
    fn red_attenuates_before_blue_with_depth() {
        // A single deep cell with a white-ish colour: because red extinction is
        // largest and the god-ray glow is achromatic (equal on every channel),
        // the ordering red < green < blue must survive into the output.
        let froxel = [1.0_f32, 1.0, 1.0, 6.0];
        let out = dispatch_underwater_volume(&froxel, PARAMS, 1, 1, 1);
        assert!(out[0] < out[1], "red collapses before green");
        assert!(out[1] < out[2], "green collapses before blue");
        assert!(out[0] >= 0.0);
    }

    #[test]
    fn godray_glow_rises_with_path_length() {
        // Zero colour isolates the glow term: longer shafts scatter more light
        // toward the eye, so the output rises monotonically with path length.
        let near = [0.0_f32, 0.0, 0.0, 1.0];
        let far = [0.0_f32, 0.0, 0.0, 20.0];
        let out_near = dispatch_underwater_volume(&near, PARAMS, 1, 1, 1);
        let out_far = dispatch_underwater_volume(&far, PARAMS, 1, 1, 1);
        assert!(out_far[0] > out_near[0], "longer shaft glows more");
        assert!(out_near[0] >= 0.0);
    }

    #[test]
    fn short_buffers_do_not_panic() {
        // Claim a 2x2x2 grid but only supply one cell of data.
        let froxel = [0.5_f32, 0.5, 0.5, 2.0];
        let out = dispatch_underwater_volume(&froxel, PARAMS, 2, 2, 2);
        assert_eq!(out.len(), 8 * UNDERWATER_OUT_FLOATS);
        // Trailing cells past the supplied data stay zero.
        assert_eq!(out[UNDERWATER_OUT_FLOATS].to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        assert!(WATER_UNDERWATER_VOLUME_WESL.contains("fn water_underwater_volume"));
        assert!(WATER_UNDERWATER_VOLUME_WESL.contains("@workgroup_size(4, 4, 4)"));
        assert!(WATER_UNDERWATER_VOLUME_WESL.contains("texture_storage_3d<rgba32float, write>"));
        assert!(WATER_UNDERWATER_VOLUME_WESL.contains("struct UnderwaterParams"));
    }

    #[test]
    fn strides_are_consistent() {
        assert_eq!(UNDERWATER_FROXEL_FLOATS, 4);
        assert_eq!(UNDERWATER_OUT_FLOATS, 4);
    }
}
