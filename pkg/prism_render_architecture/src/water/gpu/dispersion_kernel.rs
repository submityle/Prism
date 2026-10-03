//! Water spectral-dispersion / chromatic-refraction compute kernel: the `WESL`
//! shader plus its bit-exact `CPU` twin.
//!
//! Water bends short (blue) wavelengths slightly more than long (red) ones, so
//! a refracted ray splits into a faint rainbow fringe (see
//! [`dispersion`](super::super::dispersion) for the golden derivation). This
//! pass runs that same `Cauchy` index law per pixel, turns the per-channel
//! transmitted sines into screen-space refraction offsets, and resamples the
//! scene colour once per channel along the per-pixel refraction direction to
//! paint the coloured fringe.
//!
//! [`WATER_DISPERSION_REFRACT_WESL`] is the shader (entry point
//! `water_dispersion_refract`); [`dispatch_dispersion_refract`] is its
//! bit-exact `CPU` twin. Because the sandbox has no `GPU`, the twin is the
//! correctness proof: it consumes the identical buffer `ABI`
//! ([`WaterKernel::DispersionRefract`](super::super::kernels::WaterKernel) — no
//! storage buffer, one uniform param block, two sampled textures, one
//! `rgba32float` storage output, 8x8 pixel tile, `Screen` domain) and
//! reconstructs the shader arithmetic texel-for-texel. The numeric core (the
//! per-channel indices and offsets) is diffed against the golden
//! [`dispersion::spectral_iors`](super::super::dispersion::spectral_iors) and
//! [`dispersion::dispersion_offsets`](super::super::dispersion::dispersion_offsets),
//! and the scene resampling uses the same truncate-toward-zero integer offset
//! on both sides, so the whole composite is bit-exact rather than a floating
//! sampler approximation.
//!
//! Only `+ - * /`, comparisons, `clamp` and `max` appear, matching the
//! workspace determinism policy that forbids every float intrinsic but `sqrt`.

use alloc::vec;
use alloc::vec::Vec;

use super::super::EPS;

/// `WESL` source of the water dispersion-refraction compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_DISPERSION_REFRACT_WESL: &str = include_str!("water_dispersion_refract.wesl");

/// Reference RGB wavelengths in micrometres (red, green, blue), matching the
/// shader constants and
/// [`dispersion::RGB_WAVELENGTHS_UM`](super::super::dispersion::RGB_WAVELENGTHS_UM).
const WAVELENGTH_R: f32 = 0.700;
const WAVELENGTH_G: f32 = 0.546;
const WAVELENGTH_B: f32 = 0.440;

/// Number of `f32` lanes in one sampled scene texel: `(r, g, b, a)`.
pub const DISPERSION_SCENE_FLOATS: usize = 4;

/// Number of `f32` lanes in one sampled gbuffer texel: `(sin_incidence,
/// refract_dir_x, refract_dir_y, water_mask)`.
pub const DISPERSION_GBUFFER_FLOATS: usize = 4;

/// Number of `f32` lanes in one output texel: `(r, g, b, a)`.
pub const DISPERSION_OUT_FLOATS: usize = 4;

/// Uniform parameter block for the dispersion-refraction pass.
///
/// `cauchy_a` is the baseline water index (about `1.324`) and `cauchy_b` the
/// dispersion coefficient in micrometre-squared; `strength` is the screen-space
/// refraction gain folded with the water thickness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DispersionRefractParams {
    /// Baseline `Cauchy` index `a`.
    pub cauchy_a: f32,
    /// `Cauchy` dispersion coefficient `b` in micrometre-squared.
    pub cauchy_b: f32,
    /// Screen-space refraction offset gain.
    pub strength: f32,
}

/// `Cauchy` refractive-index law, reconstructing the shader's `cauchy_ior`.
///
/// Kept independent of
/// [`dispersion::cauchy_ior`](super::super::dispersion::cauchy_ior) so the
/// parity test proves the transcription rather than asserting a tautology.
#[must_use]
fn cauchy_ior_twin(a: f32, b: f32, wavelength_um: f32) -> f32 {
    let lambda = wavelength_um.max(EPS);
    a + b / (lambda * lambda)
}

/// Transmitted-ray sine for one channel, reconstructing the shader's
/// `channel_transmitted_sine`.
#[must_use]
fn channel_transmitted_sine_twin(sin_incidence: f32, ior: f32) -> f32 {
    let n = ior.max(EPS);
    (sin_incidence.clamp(0.0, 1.0) / n).clamp(0.0, 1.0)
}

/// Bit-exact `CPU` twin of the `water_dispersion_refract` compute pass.
///
/// `scene` is the row-major sampled scene colour, [`DISPERSION_SCENE_FLOATS`]
/// lanes per texel; `gbuffer` is row-major, [`DISPERSION_GBUFFER_FLOATS`] lanes
/// per texel: `(sin_incidence, refract_dir_x, refract_dir_y, water_mask)`. The
/// returned buffer is row-major with [`DISPERSION_OUT_FLOATS`] lanes per texel.
/// Pixels whose `water_mask` is below `0.5` pass the scene colour straight
/// through. A `scene` or `gbuffer` buffer shorter than `width * height` texels
/// leaves the trailing output texels at zero rather than reading out of
/// bounds, matching the shader's guarded dispatch.
#[must_use]
pub fn dispatch_dispersion_refract(
    scene: &[f32],
    gbuffer: &[f32],
    params: DispersionRefractParams,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let texels = width * height;
    let mut out = vec![0.0_f32; texels * DISPERSION_OUT_FLOATS];
    let w = width as i32;
    let h = height as i32;
    let mut t = 0usize;
    while t < texels {
        let sbase = t * DISPERSION_SCENE_FLOATS;
        let gbase = t * DISPERSION_GBUFFER_FLOATS;
        if sbase + DISPERSION_SCENE_FLOATS > scene.len()
            || gbase + DISPERSION_GBUFFER_FLOATS > gbuffer.len()
        {
            break;
        }
        let o = t * DISPERSION_OUT_FLOATS;
        let sin_incidence = gbuffer[gbase];
        let dir_x = gbuffer[gbase + 1];
        let dir_y = gbuffer[gbase + 2];
        let mask = gbuffer[gbase + 3];

        if mask < 0.5 {
            out[o] = scene[sbase];
            out[o + 1] = scene[sbase + 1];
            out[o + 2] = scene[sbase + 2];
            out[o + 3] = scene[sbase + 3];
            t += 1;
            continue;
        }

        let x = (t % width) as f32;
        let y = (t / width) as f32;

        let ior_r = cauchy_ior_twin(params.cauchy_a, params.cauchy_b, WAVELENGTH_R);
        let ior_g = cauchy_ior_twin(params.cauchy_a, params.cauchy_b, WAVELENGTH_G);
        let ior_b = cauchy_ior_twin(params.cauchy_a, params.cauchy_b, WAVELENGTH_B);

        let s = sin_incidence.clamp(0.0, 1.0);
        let gain = params.strength.max(0.0);
        let off_r = channel_transmitted_sine_twin(s, ior_r) * gain;
        let off_g = channel_transmitted_sine_twin(s, ior_g) * gain;
        let off_b = channel_transmitted_sine_twin(s, ior_b) * gain;

        // Nearest-neighbour resample with truncate-toward-zero integer offset
        // and edge clamp, identical on both sides of the parity boundary.
        let sample = |offset_px: f32, channel: usize| -> f32 {
            let px = x + offset_px * dir_x;
            let py = y + offset_px * dir_y;
            let sx = (px as i32).clamp(0, w - 1) as usize;
            let sy = (py as i32).clamp(0, h - 1) as usize;
            scene[(sy * width + sx) * DISPERSION_SCENE_FLOATS + channel]
        };

        out[o] = sample(off_r, 0);
        out[o + 1] = sample(off_g, 1);
        out[o + 2] = sample(off_b, 2);
        out[o + 3] = scene[sbase + 3];
        t += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::super::dispersion::{dispersion_offsets, spectral_iors};
    use super::*;

    const PARAMS: DispersionRefractParams = DispersionRefractParams {
        cauchy_a: 1.324,
        cauchy_b: 0.006,
        strength: 3.0,
    };

    // Deterministic scene gradient and a water gbuffer that disperses rightward.
    fn fields(width: usize, height: usize) -> (Vec<f32>, Vec<f32>) {
        let total = width * height;
        let mut scene = Vec::with_capacity(total * DISPERSION_SCENE_FLOATS);
        let mut gbuffer = Vec::with_capacity(total * DISPERSION_GBUFFER_FLOATS);
        let mut idx = 0usize;
        while idx < total {
            let f = idx as f32;
            scene.push(0.02 * f);
            scene.push(0.03 * f + 0.1);
            scene.push(0.05 * f + 0.2);
            scene.push(1.0);
            // sin_incidence sweeps 0..~1 across the grid; horizontal refract dir.
            let sin = ((f * 0.017) % 1.0).clamp(0.0, 1.0);
            gbuffer.push(sin);
            gbuffer.push(1.0);
            gbuffer.push(0.0);
            gbuffer.push(1.0);
            idx += 1;
        }
        (scene, gbuffer)
    }

    #[test]
    fn offsets_match_cpu_golden_bit_for_bit() {
        // The per-channel offsets the twin bakes into its sampling must equal
        // the golden `dispersion_offsets` exactly for every sampled incidence.
        let iors = spectral_iors(PARAMS.cauchy_a, PARAMS.cauchy_b);
        let mut i = 0u32;
        while i < 64 {
            let sin = i as f32 / 63.0;
            let golden = dispersion_offsets(iors, sin, PARAMS.strength);
            let s = sin.clamp(0.0, 1.0);
            let gain = PARAMS.strength.max(0.0);
            let twin = [
                channel_transmitted_sine_twin(s, iors.r) * gain,
                channel_transmitted_sine_twin(s, iors.g) * gain,
                channel_transmitted_sine_twin(s, iors.b) * gain,
            ];
            assert_eq!(
                twin[0].to_bits(),
                golden[0].to_bits(),
                "red offset sin {sin}"
            );
            assert_eq!(
                twin[1].to_bits(),
                golden[1].to_bits(),
                "green offset sin {sin}"
            );
            assert_eq!(
                twin[2].to_bits(),
                golden[2].to_bits(),
                "blue offset sin {sin}"
            );
            i += 1;
        }
    }

    #[test]
    fn iors_match_cpu_golden_bit_for_bit() {
        let golden = spectral_iors(PARAMS.cauchy_a, PARAMS.cauchy_b);
        let r = cauchy_ior_twin(PARAMS.cauchy_a, PARAMS.cauchy_b, WAVELENGTH_R);
        let g = cauchy_ior_twin(PARAMS.cauchy_a, PARAMS.cauchy_b, WAVELENGTH_G);
        let b = cauchy_ior_twin(PARAMS.cauchy_a, PARAMS.cauchy_b, WAVELENGTH_B);
        assert_eq!(r.to_bits(), golden.r.to_bits());
        assert_eq!(g.to_bits(), golden.g.to_bits());
        assert_eq!(b.to_bits(), golden.b.to_bits());
    }

    #[test]
    fn non_water_pixels_pass_scene_through() {
        // mask < 0.5 copies the scene colour verbatim, no resampling.
        let scene = [0.3, 0.6, 0.9, 1.0];
        let gbuffer = [0.8, 1.0, 0.0, 0.0];
        let out = dispatch_dispersion_refract(&scene, &gbuffer, PARAMS, 1, 1);
        assert_eq!(out[0].to_bits(), 0.3_f32.to_bits());
        assert_eq!(out[1].to_bits(), 0.6_f32.to_bits());
        assert_eq!(out[2].to_bits(), 0.9_f32.to_bits());
        assert_eq!(out[3].to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn zero_strength_leaves_colour_in_place() {
        // No offset -> every channel samples its own pixel, so a water pixel
        // reproduces its own scene colour.
        let params = DispersionRefractParams {
            cauchy_a: 1.324,
            cauchy_b: 0.006,
            strength: 0.0,
        };
        let scene = [0.25, 0.5, 0.75, 1.0];
        let gbuffer = [0.9, 1.0, 0.0, 1.0];
        let out = dispatch_dispersion_refract(&scene, &gbuffer, params, 1, 1);
        assert_eq!(out[0].to_bits(), 0.25_f32.to_bits());
        assert_eq!(out[1].to_bits(), 0.5_f32.to_bits());
        assert_eq!(out[2].to_bits(), 0.75_f32.to_bits());
    }

    #[test]
    fn blue_shifts_least_so_fringe_is_ordered() {
        // A horizontal brightness ramp dispersed rightward over a scene whose
        // three colour channels share the same per-column ramp. Blue has the
        // highest index so the smallest offset and samples closest to the
        // pixel's own column; red has the largest offset and reaches further
        // along the increasing ramp. Because the offset ordering is
        // red >= green >= blue and the truncate-toward-zero sample column is
        // monotonic in the (non-negative) offset, every pixel must satisfy
        // red_sample >= green_sample >= blue_sample.
        let width = 32;
        let height = 1;
        let total = width * height;
        let mut scene = Vec::with_capacity(total * DISPERSION_SCENE_FLOATS);
        let mut gbuffer = Vec::with_capacity(total * DISPERSION_GBUFFER_FLOATS);
        let mut col = 0usize;
        while col < total {
            // Identical ramp on r, g, b so only the offset — not the base
            // channel value — drives the comparison.
            let ramp = 0.03 * col as f32;
            scene.push(ramp);
            scene.push(ramp);
            scene.push(ramp);
            scene.push(1.0);
            // Strong grazing incidence, horizontal rightward refraction, water.
            gbuffer.push(0.95);
            gbuffer.push(1.0);
            gbuffer.push(0.0);
            gbuffer.push(1.0);
            col += 1;
        }
        // Large gain so the per-channel offsets land on distinct columns.
        let params = DispersionRefractParams {
            cauchy_a: 1.324,
            cauchy_b: 0.02,
            strength: 40.0,
        };
        let out = dispatch_dispersion_refract(&scene, &gbuffer, params, width, height);
        let mut t = 0usize;
        while t < total {
            let o = t * DISPERSION_OUT_FLOATS;
            assert!(out[o] >= out[o + 1] - EPS, "red >= green at texel {t}");
            assert!(out[o + 1] >= out[o + 2] - EPS, "green >= blue at texel {t}");
            t += 1;
        }
        // And the fringe is strictly visible somewhere: at the left edge the
        // red channel reaches a brighter column than the blue channel.
        assert!(
            out[0] > out[2],
            "dispersion fringe must be visible at the edge"
        );
    }

    #[test]
    fn short_buffers_do_not_panic() {
        // Claim a 4x4 grid but only supply two texels of data.
        let scene = [0.1, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 1.0];
        let gbuffer = [0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.0];
        let out = dispatch_dispersion_refract(&scene, &gbuffer, PARAMS, 4, 4);
        assert_eq!(out.len(), 16 * DISPERSION_OUT_FLOATS);
        // Trailing texels past the supplied data stay zero.
        assert_eq!(out[2 * DISPERSION_OUT_FLOATS], 0.0);
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        assert!(WATER_DISPERSION_REFRACT_WESL.contains("fn water_dispersion_refract"));
        assert!(WATER_DISPERSION_REFRACT_WESL.contains("@workgroup_size(8, 8, 1)"));
        assert!(WATER_DISPERSION_REFRACT_WESL.contains("texture_storage_2d<rgba32float, write>"));
        assert!(WATER_DISPERSION_REFRACT_WESL.contains("struct DispersionParams"));
    }

    #[test]
    fn scene_and_out_strides_are_consistent() {
        assert_eq!(DISPERSION_SCENE_FLOATS, 4);
        assert_eq!(DISPERSION_GBUFFER_FLOATS, 4);
        assert_eq!(DISPERSION_OUT_FLOATS, 4);
    }
}
