//! Octave-summed volumetric *multiple-scattering* energy compensation for the
//! §20 volumetric smoke path, so dense media never go "dead black" (design
//! `docs/prism_particle_engine_design_zh.md` §20, "单次 + 多次散射 ... 多散射用
//! 近似（blur/ambient 项或 Guerrilla 风格能量补偿），避免烟发死黑"; it feeds the
//! §17 `PBR` volumetric closure's multiple-scattering requirement).
//!
//! The decision layer in [`super::shading`] and the baking layer in
//! [`super::volumetrics`] only cover *single* scattering: the Henyey-Greenstein
//! lobe [`super::volumetrics::henyey_greenstein`], the authored double-lobe
//! response [`super::volumetrics::double_lobe_phase`], the banded `NPR` bridge
//! [`super::volumetrics::phase_response_banded`], and the six-way rig
//! [`super::shading::six_way_response`]. A single-scatter-only medium darkens
//! toward black in its interior and on its back side, because every photon that
//! would have bounced a second, third, or Nth time is simply dropped. This
//! module owns the `CPU` reference for the *multiple-scattering* term that puts
//! that energy back.
//!
//! The approximation is the public octave-sum model from production volume
//! rendering — Wrenninge, Kulla, et al.'s "Oz" multiple-scattering octaves and
//! the equivalent Frostbite (Hillaire) and Guerrilla/Decima volumetric energy
//! compensation. Each successive scattering order is approximated by one
//! *octave*: its throughput is attenuated by the scattering albedo and its
//! phase lobe is broadened toward isotropic. The textbook spelling raises the
//! albedo and the bandwidth to the octave index (`albedo^i`, `g·b^i`), which
//! needs `powf`; this crate's determinism contract forbids transcendental math,
//! so the powers are formed as a running product inside a fixed integer loop
//! (`×` only) instead. The scattering cosine comes straight from a direction
//! dot product, so no `acos` ever appears, and the per-octave phase reuses the
//! existing `sqrt`-only [`super::volumetrics::double_lobe_phase`].
//!
//! Energy stays bounded because the octave count is fixed and finite and the
//! per-octave throughput is a geometric product of factors in `0..=1`. Dead
//! black is avoided two ways: the deeper octaves tend to isotropic (so they
//! lift the back-scatter directions a forward lobe leaves dark), and an
//! explicit isotropic ambient term — scaled by the total scattered throughput —
//! guarantees a positive floor under a high-albedo white furnace. Like its
//! sibling modules, every routine here is pure and deterministic and uses only
//! `+ − × ÷` and `sqrt`, so the `CPU` reference stays bit-reproducible against a
//! future `GPU` kernel.

use super::shading::PhaseParams;
use super::volumetrics::{double_lobe_phase, FOUR_PI};
use super::Vec3;

/// The isotropic phase value `1 / (4·π)`, the floor every scattering lobe
/// shares and the value a broadened octave tends toward.
///
/// Derived from [`FOUR_PI`] by division (ordinary arithmetic, never a
/// transcendental call), so it stays consistent with the single-scatter lobes.
pub const ISOTROPIC_PHASE: f32 = 1.0 / FOUR_PI;

/// Upper bound on the octave count folded into one response.
///
/// A scattering order whose throughput has decayed below `f32` relevance adds
/// nothing, and an unbounded loop would be a denial-of-service hazard; `64`
/// orders is far past the point where a `0..=1` geometric throughput vanishes.
pub const MAX_OCTAVES: u32 = 64;

/// Clamps a value into `0..=1` (ordinary comparison arithmetic, no transcendental math).
#[must_use]
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// Clamps a value to be non-negative (used for the ambient-lift strength).
#[must_use]
fn clamp_low(v: f32) -> f32 {
    v.max(0.0)
}

/// Authored parameters for the octave-summed multiple-scattering response.
///
/// The response folds [`Self::octaves`] scattering orders. Order `i` carries a
/// per-channel throughput `(albedo · octave_decay)^i` and a phase lobe whose
/// anisotropy is the base [`Self::phase`] scaled by `anisotropy_falloff^i`, so
/// deeper orders are both dimmer and more isotropic. Order `0` is exactly the
/// single-scatter [`super::volumetrics::double_lobe_phase`], so a one-octave
/// response degenerates to the existing single-scatter lobe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiScatterParams {
    /// Per-channel single-scatter albedo (`RGB`), each component in `0..=1`.
    ///
    /// This both tints and attenuates each successive scattering order, so a
    /// colored medium shifts hue as multiple scattering dominates.
    pub albedo: Vec3,
    /// The base single-scatter phase (reused verbatim for order `0`).
    pub phase: PhaseParams,
    /// Number of scattering orders to fold, clamped to `1..=MAX_OCTAVES`.
    pub octaves: u32,
    /// Per-octave anisotropy bandwidth factor in `0..=1`.
    ///
    /// The order-`i` lobe uses anisotropy `g · anisotropy_falloff^i`, so values
    /// below `1` broaden deeper octaves toward isotropic (the physical effect
    /// of successive scattering randomizing direction).
    pub anisotropy_falloff: f32,
    /// Extra per-octave throughput decay in `0..=1`, multiplying the albedo.
    ///
    /// `1.0` leaves the physical `albedo^i` decay untouched; smaller values let
    /// an artist pull energy out of the higher orders.
    pub octave_decay: f32,
    /// Strength of the isotropic ambient floor (non-negative).
    ///
    /// The ambient term is `ambient_lift · ISOTROPIC_PHASE · Σ throughput`, an
    /// energy-conserving lift that scales with how much light the medium
    /// actually scatters, so a high-albedo interior never collapses to black.
    pub ambient_lift: f32,
}

impl Default for MultiScatterParams {
    fn default() -> Self {
        Self {
            // A mildly forward-scattering grey smoke: 0.8 albedo is a typical
            // lit-smoke value, 4 octaves captures the visible energy lift.
            albedo: Vec3::splat(0.8),
            phase: PhaseParams {
                // 0.3 is a gentle forward bias; no back lobe by default.
                g: 0.3,
                back_lobe_weight: 0.0,
                back_g: 0.0,
            },
            octaves: 4,
            // Halve the anisotropy each octave (common octave bandwidth).
            anisotropy_falloff: 0.5,
            // Keep the physical albedo decay untouched by default.
            octave_decay: 1.0,
            // A modest ambient floor to guarantee no dead black.
            ambient_lift: 0.4,
        }
    }
}

impl MultiScatterParams {
    /// Returns a copy with every field clamped into its valid range.
    #[must_use]
    pub fn sanitized(self) -> Self {
        // Fold at least one scattering order and never loop past the cap.
        let octaves = self.octaves.clamp(1, MAX_OCTAVES);
        Self {
            albedo: Vec3::new(
                clamp01(self.albedo.x),
                clamp01(self.albedo.y),
                clamp01(self.albedo.z),
            ),
            phase: self.phase,
            octaves,
            anisotropy_falloff: clamp01(self.anisotropy_falloff),
            octave_decay: clamp01(self.octave_decay),
            ambient_lift: clamp_low(self.ambient_lift),
        }
    }

    /// Evaluates the multiple-scattering response for a scattering cosine.
    ///
    /// `cos_theta` is the cosine of the scattering angle (the dot product of the
    /// normalized incoming and outgoing directions); forward scatter is `+1`.
    /// Returns the per-channel (`RGB`) multiple-scattering contribution: the
    /// octave sum plus the isotropic ambient floor. The sum is a fixed integer
    /// loop whose per-octave throughput and anisotropy are running products, so
    /// no `powf` or `acos` is used.
    #[must_use]
    pub fn response_cos(self, cos_theta: f32) -> Vec3 {
        let params = self.sanitized();
        // Per-octave throughput ratio: albedo tinted by the extra decay.
        let decay_rgb = params.albedo.scale(params.octave_decay);
        // Order-0 throughput is unit (every channel fully lit before scatter).
        let mut throughput = Vec3::splat(1.0);
        // Order-0 anisotropy scale is 1 (base lobe), narrowing each octave.
        let mut anisotropy_scale = 1.0_f32;
        let mut directional = Vec3::ZERO;
        let mut throughput_sum = Vec3::ZERO;
        for _ in 0..params.octaves {
            let octave_phase = double_lobe_phase(
                PhaseParams {
                    g: params.phase.g * anisotropy_scale,
                    back_lobe_weight: params.phase.back_lobe_weight,
                    back_g: params.phase.back_g * anisotropy_scale,
                },
                cos_theta,
            );
            directional = directional.add(throughput.scale(octave_phase));
            throughput_sum = throughput_sum.add(throughput);
            // Advance the running products for the next octave (× only).
            throughput = throughput.mul(decay_rgb);
            anisotropy_scale *= params.anisotropy_falloff;
        }
        let ambient = throughput_sum.scale(params.ambient_lift * ISOTROPIC_PHASE);
        directional.add(ambient)
    }

    /// Evaluates the response from an incoming and an outgoing direction.
    ///
    /// `incoming` is the propagation direction of the light reaching the sample
    /// and `outgoing` is the propagation direction of the scattered light
    /// toward the viewer; forward scatter (parallel directions) gives
    /// `cos_theta = +1`. Both are normalized internally; a zero-length input
    /// collapses its contribution to the isotropic case (`cos_theta = 0`)
    /// rather than producing `NaN`.
    #[must_use]
    pub fn response(self, incoming: Vec3, outgoing: Vec3) -> Vec3 {
        let cos_theta = incoming
            .normalize_or_zero()
            .dot(outgoing.normalize_or_zero());
        self.response_cos(cos_theta)
    }

    /// The scalar luminance of [`Self::response_cos`] (mean of the channels).
    #[must_use]
    pub fn luminance_cos(self, cos_theta: f32) -> f32 {
        let rgb = self.response_cos(cos_theta);
        // Mean of the three `RGB` channels.
        (rgb.x + rgb.y + rgb.z) / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::volumetrics::double_lobe_phase;

    /// Approximate-equality tolerance for the energy-level assertions.
    const TOL: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL
    }

    fn forward_phase() -> PhaseParams {
        PhaseParams {
            // Gentle forward bias with no back lobe.
            g: 0.4,
            back_lobe_weight: 0.0,
            back_g: 0.0,
        }
    }

    #[test]
    fn one_octave_degenerates_to_single_scatter_phase() {
        let params = MultiScatterParams {
            albedo: Vec3::splat(0.7),
            phase: forward_phase(),
            octaves: 1,
            anisotropy_falloff: 0.5,
            octave_decay: 1.0,
            // No ambient lift so the result is purely the single-scatter lobe.
            ambient_lift: 0.0,
        };
        // The scattering cosine for this check.
        let cos_theta = 0.3;
        let single = double_lobe_phase(forward_phase(), cos_theta);
        let rgb = params.response_cos(cos_theta);
        // Order-0 throughput is unit on every channel, so all channels equal
        // the single-scatter lobe.
        assert!(approx(rgb.x, single));
        assert!(approx(rgb.y, single));
        assert!(approx(rgb.z, single));
    }

    #[test]
    fn energy_grows_with_octaves_but_stays_bounded() {
        let base = MultiScatterParams {
            albedo: Vec3::splat(0.8),
            phase: forward_phase(),
            octaves: 2,
            anisotropy_falloff: 0.5,
            octave_decay: 1.0,
            ambient_lift: 0.2,
        };
        // A side-scatter cosine where several octaves all contribute.
        let cos_theta = 0.5;
        let r2 = MultiScatterParams {
            octaves: 2,
            ..base
        }
        .response_cos(cos_theta)
        .x;
        let r4 = MultiScatterParams {
            octaves: 4,
            ..base
        }
        .response_cos(cos_theta)
        .x;
        let r8 = MultiScatterParams {
            octaves: 8,
            ..base
        }
        .response_cos(cos_theta)
        .x;
        // Each added octave contributes strictly positive energy.
        assert!(r2 < r4);
        assert!(r4 < r8);
        // The geometric throughput converges, so a large octave count stays
        // finite and barely above the 8-octave value.
        let r32 = MultiScatterParams {
            octaves: 32,
            ..base
        }
        .response_cos(cos_theta)
        .x;
        let r64 = MultiScatterParams {
            octaves: 64,
            ..base
        }
        .response_cos(cos_theta)
        .x;
        assert!(r64.is_finite());
        assert!(r8 <= r64);
        // Converged: the tail past 32 octaves adds almost nothing.
        assert!((r64 - r32).abs() < 1e-3);
    }

    #[test]
    fn high_albedo_white_furnace_is_not_dead_black() {
        let params = MultiScatterParams {
            // Near-white, strongly forward-scattering medium.
            albedo: Vec3::splat(0.95),
            phase: PhaseParams {
                g: 0.8,
                back_lobe_weight: 0.0,
                back_g: 0.0,
            },
            octaves: 8,
            anisotropy_falloff: 0.5,
            octave_decay: 1.0,
            ambient_lift: 1.0,
        };
        // Full back-scatter, where a single forward lobe is darkest.
        let back = params.response_cos(-1.0).x;
        // Comfortably lifted away from black.
        assert!(back > 0.1, "back-scatter went dead black: {back}");
        // And strictly brighter than the bare single-scatter back lobe.
        let single_back = double_lobe_phase(params.phase, -1.0);
        assert!(back > single_back);
    }

    #[test]
    fn isotropic_albedo_is_symmetric_in_cosine() {
        let params = MultiScatterParams {
            albedo: Vec3::splat(0.6),
            // Fully isotropic base phase: g = 0, no back lobe.
            phase: PhaseParams::isotropic(),
            octaves: 5,
            anisotropy_falloff: 0.5,
            octave_decay: 1.0,
            ambient_lift: 0.3,
        };
        // An isotropic medium responds identically for mirrored cosines.
        let plus = params.response_cos(0.7).x;
        let minus = params.response_cos(-0.7).x;
        assert!(approx(plus, minus));
    }

    #[test]
    fn response_is_bit_for_bit_deterministic() {
        let params = MultiScatterParams::default();
        // The same inputs must reproduce the same bits on every evaluation.
        let a = params.response(Vec3::new(1.0, 0.2, -0.3), Vec3::new(-0.4, 1.0, 0.1));
        let b = params.response(Vec3::new(1.0, 0.2, -0.3), Vec3::new(-0.4, 1.0, 0.1));
        assert_eq!(a.x.to_bits(), b.x.to_bits());
        assert_eq!(a.y.to_bits(), b.y.to_bits());
        assert_eq!(a.z.to_bits(), b.z.to_bits());
    }

    #[test]
    fn ambient_floor_matches_closed_form() {
        let params = MultiScatterParams {
            albedo: Vec3::splat(0.5),
            // Isotropic base so every octave phase equals ISOTROPIC_PHASE.
            phase: PhaseParams::isotropic(),
            octaves: 2,
            anisotropy_falloff: 0.5,
            octave_decay: 1.0,
            ambient_lift: 1.0,
        };
        // Throughput sum over two octaves: 1 + 0.5 = 1.5 per channel.
        let throughput_sum = 1.5;
        // Directional sum: (1 + 0.5) · ISOTROPIC_PHASE.
        let directional = throughput_sum * ISOTROPIC_PHASE;
        // Ambient: throughput_sum · ambient_lift · ISOTROPIC_PHASE.
        let ambient = throughput_sum * 1.0 * ISOTROPIC_PHASE;
        let expected = directional + ambient;
        assert!(approx(params.response_cos(0.25).x, expected));
    }

    #[test]
    fn octave_count_is_clamped() {
        let params = MultiScatterParams {
            // Zero octaves must clamp up to a single scattering order.
            octaves: 0,
            ..MultiScatterParams::default()
        };
        assert_eq!(params.sanitized().octaves, 1);
        let over = MultiScatterParams {
            octaves: 10_000,
            ..MultiScatterParams::default()
        };
        assert_eq!(over.sanitized().octaves, MAX_OCTAVES);
    }

    #[test]
    fn zero_direction_falls_back_to_isotropic_cosine() {
        let params = MultiScatterParams::default();
        // A zero outgoing direction collapses to cos_theta = 0, never NaN.
        let rgb = params.response(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO);
        assert!(rgb.x.is_finite());
        assert!(approx(rgb.x, params.response_cos(0.0).x));
    }
}
