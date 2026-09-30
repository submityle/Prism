//! High-level ocean authoring presets.
//!
//! [`WaterBody`](crate::water::body::WaterBody) is the low-level `#[repr(C)]`
//! mirror the shaders read directly: filling it by hand means drawing an entire
//! `Tessendorf` initial spectrum, packing an analytic `Gerstner` train and
//! wiring every count, extent and pass flag. That is the right contract for the
//! solver, but far too much ceremony for a game that just wants "a stormy
//! ocean here". This module is the single-responsibility authoring layer that
//! turns a small, art-directable [`OceanPreset`] into a fully live body,
//! mirroring the one-line ocean/lake/river actors of `UE5` Water and `Crest`.
//!
//! The flagship [`WaterBody::ocean`](crate::water::body::WaterBody::ocean) preset is
//! not a stub: it draws the deterministic initial spectrum from the
//! dependency-free architecture core
//! ([`build_cascade_spectra`](prism_render_architecture::water::initial_spectrum::build_cascade_spectra)),
//! synthesises a physically bounded `Gerstner` wave fan whose summed steepness
//! can never fold the surface onto itself, and lights exactly the spectral +
//! `Gerstner` passes the golden schedule expects. A calm sea (no wind) carries
//! no energy, so the preset returns an honest no-op body rather than a
//! fabricated one.

use std::f32::consts::TAU;

use bevy_math::ops;

use prism_render_architecture::water::cascade::pack_cascades;
use prism_render_architecture::water::gpu::buffers::WaterBufferCounts;
use prism_render_architecture::water::gpu::pipeline::WaterPasses;
use prism_render_architecture::water::initial_spectrum::build_cascade_spectra;
use prism_render_architecture::water::spectrum::{SpectrumKind, SpectrumParams};
use prism_render_architecture::water::Vec2;

use crate::water::abi::{GpuGerstnerWave, GpuWaterGerstnerParams, GpuWaterSpectrumParams};
use crate::water::bind_groups::WaterSurfaceExtent;
use crate::water::body::WaterBody;

/// Standard gravity `g` (m/s^2), shared with the architecture spectral core.
const GRAVITY: f32 = 9.806_65;

/// How many octaves of wavelength the analytic `Gerstner` fan spans below the
/// longest wind-sustained wave. Four octaves keeps the analytic train coarse
/// (it carries the readable swell); the spectral cascades own the fine detail.
const GERSTNER_OCTAVE_SPAN: f32 = 4.0;

/// The golden angle (rad); walking phases by it de-correlates successive
/// `Gerstner` trains without a random draw, so the fan is deterministic.
const GOLDEN_ANGLE: f32 = 2.399_963_2;

/// An art-directable description of a wind-driven ocean surface.
///
/// Every field is a plain scalar so the preset is safe to expose across the
/// crate boundary; the heavy `#[repr(C)]` records it expands into stay
/// internal. Construct one with [`OceanPreset::default`] and override the few
/// fields that matter, the way a game tweaks a `UE5` Water body.
#[derive(Clone, Copy, Debug)]
pub struct OceanPreset {
    /// Which statistical spectrum shapes the sea state.
    pub kind: SpectrumKind,
    /// Grid resolution `N`; the spectrum holds `N*N` complex amplitudes and the
    /// output height/displacement textures are `N*N` texels. Powers of two feed
    /// the `IFFT` best.
    pub resolution: u32,
    /// Physical patch size `L` (m) the wave-number grid tiles across the world.
    pub patch_size: f32,
    /// Wind speed `V` (m/s); the single strongest driver of sea energy.
    pub wind_speed: f32,
    /// Downwind heading (rad) in the world `xz` plane; the swell travels along
    /// it and the spectral energy aligns to it.
    pub wind_direction: f32,
    /// Overall energy scale (the `Phillips` `A` constant / `JONSWAP` alpha).
    pub amplitude: f32,
    /// Horizontal choppiness `lambda` fed to the `Tessendorf` `-i*k_hat` term;
    /// `0` is a smooth swell, larger sharpens crests.
    pub choppiness: f32,
    /// `JONSWAP` peak-enhancement `gamma` (`1.0` reduces it to
    /// Pierson-Moskowitz); ignored by the other spectra.
    pub peak_enhancement: f32,
    /// Shortest wavelength (m) kept before high-frequency suppression, the
    /// anti-alias floor of the highest cascade.
    pub min_wavelength: f32,
    /// Directional-spread exponent; higher pins energy tighter to the wind.
    pub directional_exponent: u32,
    /// Number of spectral cascades the ocean pass loops over (clamped to at
    /// least one live cascade).
    pub cascades: u32,
    /// Geometric patch-size ratio between neighbouring cascades (`> 1`). Each
    /// finer cascade tiles a patch `1 / cascade_ratio` the size of the coarser
    /// one, the way `UE5` Water, `Crest` and `WaveWorks` stack `FFT` bands
    /// across scales. A ratio `<= 1` degenerates to no ocean.
    pub cascade_ratio: f32,
    /// How many analytic `Gerstner` trains to synthesise for the readable
    /// swell; `0` leaves the ocean spectrum-only.
    pub gerstner_waves: u32,
    /// Total surface steepness budget in `[0, 1]` shared across the `Gerstner`
    /// fan; the per-wave steepness sums to at most this, so the composited
    /// surface can never fold onto itself (no self-intersecting crests).
    pub steepness: f32,
    /// Rest water level (world `y`, m) the summed vertical displacement rides
    /// on.
    pub base_level: f32,
    /// Jacobian fold threshold: `J <= foam_threshold` flags a breaking crest
    /// for the foam field.
    pub foam_threshold: f32,
    /// Seed selecting the spectrum's random draw; distinct bodies get
    /// uncorrelated seas while any one seed is fully reproducible.
    pub seed: u32,
}

impl Default for OceanPreset {
    /// A moderate, art-friendly open sea: a `256` grid over a `256 m` patch, a
    /// `12 m/s` wind (a fresh breeze) and a four-cascade, eight-wave swell.
    fn default() -> Self {
        Self {
            kind: SpectrumKind::Phillips,
            resolution: 256,
            patch_size: 256.0,
            wind_speed: 12.0,
            wind_direction: 0.0,
            amplitude: 4.0e-3,
            choppiness: 1.2,
            peak_enhancement: 3.3,
            min_wavelength: 0.08,
            directional_exponent: 2,
            cascades: 4,
            cascade_ratio: 4.0,
            gerstner_waves: 8,
            steepness: 0.75,
            base_level: 0.0,
            foam_threshold: 0.6,
            seed: 0x0cea_0cea,
        }
    }
}

impl WaterBody {
    /// Builds a fully live ocean body from a high-level [`OceanPreset`].
    ///
    /// The initial spectrum is drawn deterministically by the architecture core
    /// and packed into the `spectrum_h0` / `spectrum_h0_neg` pools; a physically
    /// bounded `Gerstner` fan is synthesised for the readable swell; and the
    /// spectral + `Gerstner` passes are lit with the counts and extents the
    /// golden schedule sizes its resident buffers from. A calm sea (zero wind or
    /// a degenerate grid) carries no spectral energy, so this returns the
    /// default no-op body rather than an empty-but-lit ocean that would dispatch
    /// over nothing.
    #[must_use]
    pub fn ocean(preset: OceanPreset) -> Self {
        // Draw the deterministic initial spectrum from the dependency-free core.
        let wind = Vec2::new(
            preset.wind_speed * ops::cos(preset.wind_direction),
            preset.wind_speed * ops::sin(preset.wind_direction),
        );
        let params = SpectrumParams {
            kind: preset.kind,
            wind,
            amplitude: preset.amplitude,
            peak_enhancement: preset.peak_enhancement,
            min_wavelength: preset.min_wavelength,
            directional_exponent: preset.directional_exponent,
        };

        // Draw one band-limited spectrum per cascade and flatten the ragged set
        // into the device-friendly stacked atlas the shader addresses by a
        // per-cascade `h0_offset` / `tile_origin_y`, the way `UE5` Water,
        // `Crest` and `WaveWorks` resolve waves across scales.
        let fields = build_cascade_spectra(
            preset.resolution,
            preset.patch_size,
            preset.cascade_ratio,
            preset.cascades,
            params,
            preset.seed,
        );
        let Some(packed) = pack_cascades(&fields) else {
            return Self::default();
        };

        // A calm sea (no wind, so every drawn amplitude is zero) carries no
        // energy across any cascade: keep the body an honest no-op.
        let total_energy: f32 = packed.h0.iter().map(|a| a[0] * a[0] + a[1] * a[1]).sum();
        if total_energy <= f32::EPSILON {
            return Self::default();
        }

        let layout = packed.layout;
        let n = layout.resolution();
        // Per-cascade grid size: the packed/scratch buffers and the counts are
        // sized for one `N*N` tile that the spectral pass reuses per cascade.
        let texels = n * n;

        // The concatenated `M*N*N` amplitude pools, coarsest cascade first.
        let spectrum_h0 = packed.h0;
        let spectrum_h0_neg = packed.h0_neg;

        // One spectral uniform per cascade: patch size plus the cascade's slice
        // offset into the concatenated `h0` pool and its tile's top atlas row.
        let cascade_params: Vec<GpuWaterSpectrumParams> = (0..layout.cascade_count())
            .map(|c| GpuWaterSpectrumParams {
                grid_size: n,
                patch_size: packed.patch_sizes[c as usize],
                time: 0.0,
                choppiness: preset.choppiness,
                foam_threshold: preset.foam_threshold,
                h0_offset: layout.h0_offset(c),
                tile_origin_y: layout.tile_origin(c).1,
                _pad: 0,
            })
            .collect();

        let gerstner_waves = gerstner_fan(&preset);
        let wave_count = gerstner_waves.len() as u32;
        // Cascade 0 is the coarsest patch (the readable swell) the analytic fan rides.
        let gerstner_patch = packed.patch_sizes[0];

        let extent = WaterSurfaceExtent {
            width: layout.atlas_width(),
            height: layout.atlas_height(),
        };

        Self {
            spectrum_h0,
            spectrum_h0_neg,
            // The ocean group's direct-sum reference binds a single spectral
            // uniform; production runs the per-cascade `spectrum_fft` groups.
            spectrum_params: cascade_params[0],
            cascade_params,
            gerstner_waves,
            gerstner_params: GpuWaterGerstnerParams {
                grid_size: n,
                patch_size: gerstner_patch,
                time: 0.0,
                wave_count,
                base_level: preset.base_level,
                _pad: [0; 3],
            },
            ocean_extent: extent,

            passes: WaterPasses {
                ocean_spectrum: true,
                gerstner: wave_count > 0,
                ..WaterPasses::default()
            },
            ocean_cascades: layout.cascade_count(),
            spectrum_texels: texels,
            grid2d_texels: texels,
            counts: WaterBufferCounts {
                spectrum_texels: texels,
                gerstner_waves: wave_count,
                ..WaterBufferCounts::default()
            },
            ..Self::default()
        }
    }
}

/// Synthesises the deterministic, physically bounded analytic `Gerstner` fan.
///
/// The trains span [`GERSTNER_OCTAVE_SPAN`] octaves of wavelength below the
/// longest wave the wind sustains (`L = V^2 / g`), fan out symmetrically around
/// the wind heading, and share the preset's steepness budget so the summed
/// steepness `sum Q_i` never exceeds `1` — the condition that keeps a `Gerstner`
/// surface from folding onto itself.
fn gerstner_fan(preset: &OceanPreset) -> Vec<GpuGerstnerWave> {
    let count = preset.gerstner_waves;
    if count == 0 || preset.wind_speed <= 0.0 {
        return Vec::new();
    }

    // Longest wave the wind can build; the fan hangs its octave ladder off it.
    let largest = (preset.wind_speed * preset.wind_speed / GRAVITY).max(1.0);
    // Split the shared steepness budget evenly so the sum stays bounded by it.
    let per_wave_steepness = (preset.steepness / count as f32).clamp(0.0, 1.0);
    // Widen the directional fan as more trains are added, capped at ~60 deg.
    let spread = (TAU / 12.0) * (count as f32 - 1.0).min(3.0) / 3.0;

    let mut waves = Vec::with_capacity(count as usize);
    for i in 0..count {
        let frac = if count > 1 {
            i as f32 / (count as f32 - 1.0)
        } else {
            0.0
        };
        // Longest train first, halving wavelength across the octave span.
        let wavelength = (largest * ops::powf(2.0, -frac * GERSTNER_OCTAVE_SPAN)).max(0.1);
        let k = TAU / wavelength;
        // Bound amplitude by the per-wave steepness: Q = k*A <= per_wave_steepness.
        let amplitude = per_wave_steepness / k;
        // Symmetric fan around the wind heading.
        let angle = preset.wind_direction + spread * (frac * 2.0 - 1.0);

        waves.push(GpuGerstnerWave {
            dir_x: ops::cos(angle),
            dir_z: ops::sin(angle),
            amplitude,
            wavelength,
            steepness: per_wave_steepness,
            speed: 1.0,
            phase: (i as f32 * GOLDEN_ANGLE) % TAU,
            _pad: 0.0,
        });
    }
    waves
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::water::gpu::pipeline::prepare;
    use prism_render_architecture::water::kernels::WaterKernel;

    /// The flagship preset must expand into a live spectral + `Gerstner` body
    /// whose golden schedule leads with the spectrum `IFFT`.
    #[test]
    fn ocean_preset_builds_a_live_ordered_ocean() {
        let preset = OceanPreset {
            resolution: 64,
            gerstner_waves: 8,
            ..OceanPreset::default()
        };
        let body = WaterBody::ocean(preset);

        assert_eq!(body.spectrum_h0.len(), 4 * 64 * 64);
        assert_eq!(body.spectrum_h0_neg.len(), 4 * 64 * 64);
        assert_eq!(body.gerstner_waves.len(), 8);
        assert_eq!(body.counts.gerstner_waves, 8);
        assert_eq!(body.counts.spectrum_texels, 64 * 64);
        assert_eq!(body.ocean_extent.width, 64);
        assert_eq!(body.ocean_extent.height, 4 * 64);

        let ex = body.as_extract();
        assert!(ex.ocean_spectrum);
        assert!(ex.gerstner);
        assert_eq!(ex.ocean_cascades, 4);

        let plan = prepare(&ex);
        assert!(!plan.dispatches.is_empty());
        assert_eq!(plan.dispatches[0].kernel, WaterKernel::SpectrumIfft);
    }

    /// A calm sea (no wind) carries no energy, so the preset is an honest no-op:
    /// an empty body whose schedule dispatches nothing.
    #[test]
    fn calm_sea_is_an_honest_noop() {
        let body = WaterBody::ocean(OceanPreset {
            wind_speed: 0.0,
            ..OceanPreset::default()
        });
        assert!(body.spectrum_h0.is_empty());
        assert!(body.gerstner_waves.is_empty());
        assert!(prepare(&body.as_extract()).dispatches.is_empty());
    }

    /// The same seed reproduces the identical spectrum; a different seed draws
    /// an uncorrelated one.
    #[test]
    fn spectrum_is_deterministic_in_the_seed() {
        let base = OceanPreset {
            resolution: 32,
            ..OceanPreset::default()
        };
        let a = WaterBody::ocean(base);
        let b = WaterBody::ocean(base);
        assert_eq!(a.spectrum_h0, b.spectrum_h0);

        let c = WaterBody::ocean(OceanPreset {
            seed: base.seed ^ 0x5555_5555,
            ..base
        });
        assert_ne!(a.spectrum_h0, c.spectrum_h0);
    }

    /// The `Gerstner` fan is physically bounded: every train is unit-directed,
    /// its steepness is in `[0, 1]`, and the summed steepness never exceeds the
    /// preset budget, so the composited surface cannot fold onto itself.
    #[test]
    fn gerstner_fan_is_physically_bounded() {
        let preset = OceanPreset {
            resolution: 32,
            gerstner_waves: 8,
            steepness: 0.8,
            ..OceanPreset::default()
        };
        let body = WaterBody::ocean(preset);

        let mut steepness_sum = 0.0;
        for wave in &body.gerstner_waves {
            let dir_len_sq = wave.dir_x * wave.dir_x + wave.dir_z * wave.dir_z;
            assert!((dir_len_sq - 1.0).abs() < 1.0e-5, "direction must be unit");
            assert!((0.0..=1.0).contains(&wave.steepness));
            assert!(wave.amplitude > 0.0);
            assert!(wave.wavelength > 0.0);
            steepness_sum += wave.steepness;
        }
        assert!(
            steepness_sum <= preset.steepness + 1.0e-5,
            "summed steepness {steepness_sum} must stay within the budget"
        );
    }

    /// A stronger wind builds a more energetic sea: the total spectral variance
    /// grows monotonically with wind speed.
    #[test]
    fn stronger_wind_builds_more_energy() {
        let energy = |wind: f32| -> f32 {
            let body = WaterBody::ocean(OceanPreset {
                resolution: 48,
                wind_speed: wind,
                ..OceanPreset::default()
            });
            body.spectrum_h0
                .iter()
                .map(|c| c[0] * c[0] + c[1] * c[1])
                .sum::<f32>()
        };
        assert!(energy(16.0) > energy(8.0));
    }
}
