//! Runtime dynamic channels: the analytic modulation of the *frozen* über
//! surface (§17.1, roadmap P2-g).
//!
//! # Why modulation rather than more lobes
//!
//! The bake (§6) freezes a material's static über fields, which is optimal for
//! static surfaces but too rigid for "the same material under a continuously
//! changing state": a wet car drying, snow accumulating, paint chipping, a
//! surface burning away. Expressing those as extra lobes or extra bakes explodes
//! the permutation space (§8.4); expressing them as a texture blend loses the
//! physics. §17.1 instead opens a small, explicitly-declared runtime axis and
//! **remaps existing über fields analytically**: no new lobe, no new shader
//! permutation, no per-pixel budget gamble — and, because every remap is a
//! closed-form expression, the CPU golden and the WESL twin stay comparable.
//!
//! # The contract
//!
//! * **Identity at zero.** Every channel value `0` is a bit-for-bit identity
//!   (the per-channel branch is taken per *material*, not per pixel, so the
//!   wavefront stays coherent), which is why a material that never drives a
//!   channel shades exactly as before.
//! * **Order is part of the contract.** Channels apply in a fixed order —
//!   `wetness → snow → damage → dissolve` — so "wet snow" and "snow on a wet
//!   surface" cannot drift apart between implementations. Snow sits on top of
//!   wetness, wear cuts through both, dissolve is last because it consumes the
//!   result.
//! * **No new parameters.** The remaps use fixed, physically-motivated targets
//!   (water's index, snow's albedo, a chip's roughness) rather than new author
//!   knobs: the authored über fields are the *input*, and the channel is the
//!   state. That keeps the authoring surface stable and the ABI unchanged.
//! * **Deterministic and total.** Only `min`/`max`/`mix` arithmetic; the dissolve
//!   mask is a hash of the quantised surface cell, so it needs no binding and no
//!   authoring data, and both lines derive it identically.
//!
//! Mirrored by `shaders/dynamic.wesl`.

use crate::lighting::SurfaceSample;
use crate::lobe_sampling::{lobe_hash_combine, lobe_hash_u32, lobe_uniform_from_hash};
use prism_render_material::DynamicChannels;

/// Relative darkening of the base colour under a full water film.
///
/// Water fills the pores and adds a specular interface, so most of what was
/// diffuse scattering now reflects at the surface: the diffuse albedo visibly
/// darkens (`0.55` is the usual "wet stone/wood" factor) while the specular
/// terms below take over.
pub const WET_BASE_DARKEN: f32 = 0.55;
/// Multiplier on the specular roughness under a full film: a film is *smoother*
/// than the substrate it covers, so highlights tighten instead of blurring.
pub const WET_ROUGHNESS_SCALE: f32 = 0.35;
/// Refractive index of the water film (20 °C, 589 nm).
pub const WATER_IOR: f32 = 1.333;

/// Diffuse albedo of fresh snow (slightly cool, as snow is).
pub const SNOW_ALBEDO: [f32; 3] = [0.95, 0.97, 1.0];
/// Specular roughness of a snow cover: a broad, diffuse-dominant highlight.
pub const SNOW_ROUGHNESS: f32 = 0.9;
/// Diffuse (Oren-Nayar micro-slope) roughness of snow.
pub const SNOW_DIFFUSE_ROUGHNESS: f32 = 0.9;
/// Specular weight of snow — it is a dielectric powder, not a lacquer.
pub const SNOW_SPECULAR_WEIGHT: f32 = 0.35;
/// Sheen a snow cover retains from its granular top layer.
pub const SNOW_SHEEN: f32 = 0.15;

/// Relative darkening of the base colour under full wear (oxidation, grime).
pub const DAMAGE_BASE_DARKEN: f32 = 0.45;
/// Fraction of the metal interface destroyed by full wear.
pub const DAMAGE_METAL_LOSS: f32 = 0.6;
/// How far full wear pushes the specular roughness (`d * this` towards 1).
pub const DAMAGE_ROUGHEN: f32 = 0.6;

/// Mask cells per unit of the surface cell space used by the procedural
/// dissolve mask. Fine enough to read as a ragged cut, coarse enough that a
/// pixel never straddles a cell boundary on its own.
pub const DISSOLVE_MASK_CELLS: f32 = 96.0;
/// Width of the dissolve cut line, in mask units: the band over which coverage
/// falls from `1` to `0` and the emissive rim peaks.
pub const DISSOLVE_EDGE_WIDTH: f32 = 0.12;
/// Peak emissive gain of the cut line, relative to the material's own emission.
pub const DISSOLVE_EDGE_GAIN: f32 = 4.0;

/// The procedural dissolve mask for one surface cell, in `[0, 1)`.
///
/// Hashing the quantised cell (rather than the raw UV) keeps the cut stable
/// under sub-cell motion, so a dissolving surface does not shimmer. Mirrored by
/// `dyn_dissolve_mask` in WESL.
#[must_use]
pub fn dissolve_mask(cell: [f32; 2]) -> f32 {
    let cu = quantize_cell(cell[0]);
    let cv = quantize_cell(cell[1]);
    lobe_uniform_from_hash(lobe_hash_combine(lobe_hash_u32(cu), cv))
}

/// Quantise one cell coordinate into a mask lattice index.
fn quantize_cell(coord: f32) -> u32 {
    let scaled = coord * DISSOLVE_MASK_CELLS;
    // `floor` then a saturating cast: a non-finite or absurd coordinate lands on
    // a defined cell instead of an undefined one.
    let floored = bevy_math::ops::floor(scaled).clamp(-2_147_483_600.0, 2_147_483_600.0);
    (floored as i32) as u32
}

/// Apply the four runtime dynamic channels to `surface` (§17.1).
///
/// `surface_cell` is the same shade-time cell the glint seed quantises (the
/// surface UV or a tangent-frame projection of the position) and is used for the
/// procedural dissolve mask.
pub fn apply_dynamic_channels(
    surface: &mut SurfaceSample,
    channels: DynamicChannels,
    surface_cell: [f32; 2],
) {
    let wetness = channels.wetness.clamp(0.0, 1.0);
    if wetness > 0.0 {
        for channel in &mut surface.base_color {
            *channel *= mix(1.0, WET_BASE_DARKEN, wetness);
        }
        surface.diffuse_roughness = mix(surface.diffuse_roughness, 0.0, wetness);
        surface.perceptual_roughness = mix(
            surface.perceptual_roughness,
            surface.perceptual_roughness * WET_ROUGHNESS_SCALE,
            wetness,
        );
        surface.specular_weight = mix(surface.specular_weight, 1.0, wetness);
        surface.specular_ior = mix(surface.specular_ior, WATER_IOR, wetness);
        surface.sheen *= 1.0 - wetness;
    }

    let snow = channels.snow.clamp(0.0, 1.0);
    if snow > 0.0 {
        for (channel, target) in surface.base_color.iter_mut().zip(SNOW_ALBEDO) {
            *channel = mix(*channel, target, snow);
        }
        surface.metallic = mix(surface.metallic, 0.0, snow);
        surface.perceptual_roughness = mix(surface.perceptual_roughness, SNOW_ROUGHNESS, snow);
        surface.diffuse_roughness = mix(surface.diffuse_roughness, SNOW_DIFFUSE_ROUGHNESS, snow);
        surface.specular_weight = mix(surface.specular_weight, SNOW_SPECULAR_WEIGHT, snow);
        surface.sheen = mix(surface.sheen, SNOW_SHEEN, snow);
    }

    let damage = channels.damage.clamp(0.0, 1.0);
    if damage > 0.0 {
        for channel in &mut surface.base_color {
            *channel *= mix(1.0, DAMAGE_BASE_DARKEN, damage);
        }
        surface.metallic *= 1.0 - DAMAGE_METAL_LOSS * damage;
        surface.perceptual_roughness =
            mix(surface.perceptual_roughness, 1.0, damage * DAMAGE_ROUGHEN);
        // Chipped wear breaks the coat open.
        surface.clearcoat *= 1.0 - damage;
    }

    let dissolve = channels.dissolve.clamp(0.0, 1.0);
    if dissolve > 0.0 {
        // The *coverage* half of the cut is applied where alpha is consumed
        // (alpha test / blend) via [`dissolve_coverage`]; the shading resolve
        // produces radiance, not coverage. What belongs here is the cut *line*:
        // the band where coverage is between 0 and 1 is lit by the material's own
        // emission, so the rim takes the colour the author already chose (and a
        // non-emissive material gets a clean cut with no rim at all).
        let coverage = dissolve_coverage(dissolve, surface_cell);
        let rim = DISSOLVE_EDGE_GAIN * coverage * (1.0 - coverage);
        for channel in &mut surface.emissive {
            *channel *= 1.0 + rim;
        }
    }
}

/// Share of a dissolving pixel that survives the cut, in `[0, 1]`.
///
/// Multiply this into the material's alpha where alpha is consumed (alpha test /
/// blend), and use the same value's complement to light the rim in the shading
/// resolve — one mask, so the visible edge and the coverage edge cannot
/// disagree. `0` progress is an exact identity (and skips the mask hash
/// entirely).
///
/// Mirrored by `dyn_dissolve_coverage` in `dynamic_mask.wesl`.
#[must_use]
pub fn dissolve_coverage(dissolve: f32, surface_cell: [f32; 2]) -> f32 {
    let progress = dissolve.clamp(0.0, 1.0);
    if progress <= 0.0 {
        return 1.0;
    }
    clamp01((dissolve_mask(surface_cell) - progress) / DISSOLVE_EDGE_WIDTH)
}

/// `a*(1 - t) + b*t`.
///
/// Deliberately *not* `a + (b - a) * t` (the expression WESL's built-in `mix`
/// lowers to): this form is exact at both endpoints, so full coverage lands on
/// the target value bit for bit and `t == 0` returns `a` unchanged. That is what
/// makes "fully snowy" mean exactly the snow albedo and keeps the zero-channel
/// identity airtight. The WESL twin spells the same expression out, for the same
/// reason.
#[inline]
fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

#[inline]
#[expect(
    clippy::manual_clamp,
    reason = "clamp propagates NaN; max/min absorb it into a defined endpoint"
)]
fn clamp01(value: f32) -> f32 {
    // Deliberately not `clamp`: a NaN here would poison the coverage/rim term
    // (and through it the whole surface), whereas `max`/`min` map it to `0`.
    value.max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface() -> SurfaceSample {
        SurfaceSample {
            base_color: [0.6, 0.5, 0.4],
            base_weight: 1.0,
            diffuse_roughness: 0.6,
            metallic: 0.8,
            perceptual_roughness: 0.4,
            specular_weight: 0.7,
            specular_ior: 1.5,
            sheen: 0.5,
            clearcoat: 0.9,
            emissive: [0.2, 0.1, 0.05],
            geometry_opacity: 1.0,
            ..SurfaceSample::default()
        }
    }

    fn modulated(channels: DynamicChannels) -> SurfaceSample {
        let mut surface = surface();
        apply_dynamic_channels(&mut surface, channels, [0.3, 0.7]);
        surface
    }

    #[test]
    fn every_channel_is_a_bit_for_bit_identity_at_zero() {
        let mut base = surface();
        apply_dynamic_channels(&mut base, DynamicChannels::NONE, [0.31, 0.77]);
        assert_eq!(base, surface());
        // …and each channel alone leaves the others' inputs untouched.
        let only_snow = modulated(DynamicChannels {
            snow: 0.5,
            ..DynamicChannels::NONE
        });
        let only_damage = modulated(DynamicChannels {
            damage: 0.5,
            ..DynamicChannels::NONE
        });
        assert_eq!(only_snow.geometry_opacity, 1.0);
        assert_eq!(only_damage.sheen, surface().sheen);
    }

    #[test]
    fn wetness_darkens_smooths_and_raises_the_interface() {
        let dry = surface();
        let wet = modulated(DynamicChannels {
            wetness: 1.0,
            ..DynamicChannels::NONE
        });
        assert!(wet.base_color[0] < dry.base_color[0]);
        assert!(wet.base_color[0].abs() > 0.0);
        assert!(wet.diffuse_roughness < dry.diffuse_roughness);
        assert!(wet.perceptual_roughness < dry.perceptual_roughness);
        assert!(wet.specular_weight > dry.specular_weight);
        assert!((wet.specular_ior - WATER_IOR).abs() < 1.0e-6);
        assert!(wet.sheen < dry.sheen);
        // Half wetness lands between the two endpoints (monotone, no snap).
        let half = modulated(DynamicChannels {
            wetness: 0.5,
            ..DynamicChannels::NONE
        });
        assert!(half.base_color[0] < dry.base_color[0] && half.base_color[0] > wet.base_color[0]);
    }

    #[test]
    fn snow_whitens_roughens_and_kills_the_metal_interface() {
        let snow = modulated(DynamicChannels {
            snow: 1.0,
            ..DynamicChannels::NONE
        });
        assert_eq!(snow.base_color, SNOW_ALBEDO);
        assert!(snow.metallic < 0.05);
        assert_eq!(snow.perceptual_roughness, SNOW_ROUGHNESS);
        assert_eq!(snow.diffuse_roughness, SNOW_DIFFUSE_ROUGHNESS);
        assert_eq!(snow.specular_weight, SNOW_SPECULAR_WEIGHT);
    }

    #[test]
    fn damage_breaks_the_interface_and_opens_the_coat() {
        let damage = modulated(DynamicChannels {
            damage: 1.0,
            ..DynamicChannels::NONE
        });
        assert!(damage.base_color[0] < surface().base_color[0]);
        assert!(damage.metallic < surface().metallic);
        assert!(damage.perceptual_roughness > surface().perceptual_roughness);
        assert!(damage.clearcoat < 1.0e-6);
    }

    #[test]
    fn dissolve_cuts_coverage_and_lights_an_emissive_rim() {
        let base = surface();
        // Pick a cell and a progress that straddles its mask value so the pixel
        // sits on the cut line.
        let cell = [0.42, 0.17];
        let mask = dissolve_mask(cell);
        let on_the_line = mask - DISSOLVE_EDGE_WIDTH * 0.5;
        let coverage = dissolve_coverage(on_the_line, cell);
        assert!(
            (coverage - 0.5).abs() < 1.0e-6,
            "coverage {coverage} should be mid-cut"
        );

        let mut rimmed = base;
        apply_dynamic_channels(
            &mut rimmed,
            DynamicChannels {
                dissolve: on_the_line,
                ..DynamicChannels::NONE
            },
            cell,
        );
        assert!(
            rimmed.emissive[0] > base.emissive[0],
            "the cut line must glow"
        );

        // Fully dissolved: no coverage, so no rim either.
        let mut gone = base;
        apply_dynamic_channels(
            &mut gone,
            DynamicChannels {
                dissolve: 1.0,
                ..DynamicChannels::NONE
            },
            cell,
        );
        assert_eq!(dissolve_coverage(1.0, cell), 0.0);
        assert_eq!(gone.emissive, base.emissive);

        // Inert: an exact identity, and the mask hash is never even evaluated.
        assert_eq!(dissolve_coverage(0.0, cell), 1.0);
        let mut intact = base;
        apply_dynamic_channels(&mut intact, DynamicChannels::NONE, cell);
        assert_eq!(intact, base);
    }

    #[test]
    fn dissolve_coverage_is_monotone_in_progress_and_cell() {
        // More progress = less coverage, always, and never outside `[0, 1]`.
        let cell = [0.42, 0.17];
        let mask = dissolve_mask(cell);
        let mut previous = f32::INFINITY;
        for step in 0..=16 {
            let dissolve = mask - DISSOLVE_EDGE_WIDTH + step as f32 * DISSOLVE_EDGE_WIDTH / 8.0;
            let coverage = dissolve_coverage(dissolve, cell);
            assert!((0.0..=1.0).contains(&coverage), "coverage {coverage}");
            assert!(coverage <= previous + 1.0e-6, "{coverage} after {previous}");
            previous = coverage;
        }
        // A cell whose mask sits above the cut is untouched by that same state.
        let high_cell = (0..64)
            .map(|i| [i as f32 * 0.37, 0.11])
            .find(|c| dissolve_mask(*c) > 0.9)
            .expect("a cell with a high mask value");
        assert_eq!(dissolve_coverage(mask, high_cell), 1.0);
    }

    #[test]
    fn the_procedural_mask_is_deterministic_and_spread_out() {
        // Deterministic: the same cell always cuts the same way.
        assert_eq!(dissolve_mask([0.3, 0.7]), dissolve_mask([0.3, 0.7]));
        // Spread: neighbouring cells are not correlated (a dissolve must read as
        // a ragged cut, not a gradient or a single block).
        let values: Vec<f32> = (0..64)
            .map(|i| dissolve_mask([i as f32 / 64.0, 0.5]))
            .collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        assert!((0.3..0.7).contains(&mean), "mean {mean}");
        let below = values.iter().filter(|v| **v < 0.5).count();
        assert!(below > 8 && below < values.len() - 8, "below {below}");
    }

    #[test]
    fn channels_compose_in_the_documented_order() {
        // Snow on top of wetness: the snow albedo wins, but the water film still
        // raised the interface before the snow overwrote the specular weight.
        let wet_snow = modulated(DynamicChannels {
            wetness: 1.0,
            snow: 1.0,
            ..DynamicChannels::NONE
        });
        assert_eq!(wet_snow.base_color, SNOW_ALBEDO);
        assert_eq!(wet_snow.specular_weight, SNOW_SPECULAR_WEIGHT);
        // Damage applies after snow and still opens a (now snow-covered) coat.
        let damaged = modulated(DynamicChannels {
            snow: 1.0,
            damage: 1.0,
            ..DynamicChannels::NONE
        });
        assert!(
            damaged.perceptual_roughness > wet_snow.perceptual_roughness,
            "wear roughs the snow cover further"
        );
        assert!(damaged.clearcoat < 1.0e-6);
    }
}
