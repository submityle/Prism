//! Wet-hair saturation coupling: a deterministic map from a scalar *wetness*
//! fraction to the physical parameter *modifiers* that make wet hair behave and
//! read differently from dry hair.
//!
//! When hair takes on water the change is physically coherent and driven by a
//! single quantity, the water saturation `w` in `[0, 1]` (`0` fully dry, `1`
//! fully saturated):
//!
//! * **Clumping tightens.** Surface tension pulls neighbouring fibres into
//!   wet locks, so the authored clump radius effectively shrinks
//!   ([`WetHairResponse::clump_scale`], a multiplier `<= 1`).
//! * **Mass rises.** Absorbed water adds mass to each strand
//!   ([`WetHairResponse::mass_mul`], a multiplier `>= 1`), which the dynamics
//!   solver uses via inverse mass.
//! * **Damping rises.** Viscous water between fibres bleeds motion faster
//!   ([`WetHairResponse::damping_mul`], a multiplier `>= 1`).
//! * **Absorption deepens.** A water film raises the effective path absorption
//!   so wet hair looks darker ([`WetHairResponse::sigma_a_mul`], a multiplier
//!   `>= 1` applied to the pigment `sigma_a` from [`crate::hair::melanin`]).
//! * **Specular sharpens.** The smooth water layer lowers the apparent
//!   roughness, giving the characteristic wet sheen
//!   ([`WetHairResponse::roughness_delta`], an additive offset `<= 0`).
//!
//! Every output is a straight linear interpolation between the dry endpoint
//! (`w = 0`) and the wet endpoint (`w = 1`), so there is **no** transcendental
//! math here (no `libm` determinism shim needed) and the map is exactly
//! golden-comparable: array in, array out, panic-free. The dynamics, LOD and
//! shading sides *consume* these modifiers; this module owns none of their
//! budgets and only produces the deterministic coupling, mirroring the
//! array-in/array-out contract of [`crate::hair::melanin`].
//!
//! Inputs that are negative, greater than one, or non-finite (`NaN`/inf) are
//! unphysical and are sanitised to the valid `[0, 1]` range by
//! [`sanitize_wetness`] before use; they never panic.

use alloc::vec::Vec;

/// Clump-radius multiplier at full saturation (`w = 1`). Wet locks draw fibres
/// together, so the authored clump radius shrinks to this fraction. The dry
/// endpoint is always `1.0` (authored radius unchanged).
pub const CLUMP_SCALE_WET: f32 = 0.45;

/// Per-strand mass multiplier at full saturation. Absorbed water makes a
/// saturated strand noticeably heavier; the dry endpoint is `1.0`.
pub const MASS_MUL_WET: f32 = 1.6;

/// Velocity-damping multiplier at full saturation. Viscous inter-fibre water
/// removes motion faster; the dry endpoint is `1.0`.
pub const DAMPING_MUL_WET: f32 = 1.8;

/// Pigment-absorption (`sigma_a`) multiplier at full saturation. A water film
/// deepens the effective absorption so wet hair reads darker; the dry endpoint
/// is `1.0`.
pub const SIGMA_A_MUL_WET: f32 = 1.5;

/// Additive roughness offset at full saturation. The smooth water layer lowers
/// apparent roughness (sharper, wetter specular); the dry endpoint is `0.0`.
pub const ROUGHNESS_DELTA_WET: f32 = -0.35;

/// Clamp an arbitrary wetness input to the physical `[0, 1]` saturation range.
/// Non-finite inputs (`NaN`/inf) are treated as fully dry (`0.0`) so every
/// downstream modifier stays finite and bounded.
#[must_use]
pub fn sanitize_wetness(wetness: f32) -> f32 {
    if wetness.is_finite() {
        wetness.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Linear interpolation between the dry endpoint `dry` (at `w = 0`) and the wet
/// endpoint `wet` (at `w = 1`). Pure arithmetic, no transcendentals.
#[must_use]
fn lerp(dry: f32, wet: f32, w: f32) -> f32 {
    dry + (wet - dry) * w
}

/// The set of physical parameter modifiers produced for a given wetness. Each
/// field is applied by a different consumer (dynamics, interpolation/clumping,
/// shading), but all are produced here from the single saturation scalar so the
/// coupling stays consistent and deterministic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WetHairResponse {
    /// Multiplier on the authored clump radius; `1.0` dry down to
    /// [`CLUMP_SCALE_WET`] when fully saturated.
    pub clump_scale: f32,
    /// Multiplier on per-strand mass; `1.0` dry up to [`MASS_MUL_WET`].
    pub mass_mul: f32,
    /// Multiplier on velocity damping; `1.0` dry up to [`DAMPING_MUL_WET`].
    pub damping_mul: f32,
    /// Multiplier on the pigment `sigma_a`; `1.0` dry up to [`SIGMA_A_MUL_WET`].
    pub sigma_a_mul: f32,
    /// Additive roughness offset; `0.0` dry down to [`ROUGHNESS_DELTA_WET`].
    pub roughness_delta: f32,
}

impl WetHairResponse {
    /// The fully dry response: every modifier is a no-op.
    pub const DRY: Self = Self {
        clump_scale: 1.0,
        mass_mul: 1.0,
        damping_mul: 1.0,
        sigma_a_mul: 1.0,
        roughness_delta: 0.0,
    };

    /// Apply the absorption multiplier to a base RGB `sigma_a` (for example the
    /// pigment absorption from [`crate::hair::melanin::melanin_absorption`]),
    /// returning the wet-deepened absorption. Pure per-channel scale.
    #[must_use]
    pub fn apply_absorption(self, base: [f32; 3]) -> [f32; 3] {
        [
            base[0] * self.sigma_a_mul,
            base[1] * self.sigma_a_mul,
            base[2] * self.sigma_a_mul,
        ]
    }

    /// Apply the roughness offset to a base roughness, clamped to the valid
    /// `[0, 1]` range so a very wet strand cannot drive roughness negative.
    #[must_use]
    pub fn apply_roughness(self, base: f32) -> f32 {
        (base + self.roughness_delta).clamp(0.0, 1.0)
    }
}

impl Default for WetHairResponse {
    fn default() -> Self {
        Self::DRY
    }
}

/// Map a single wetness fraction to its physical parameter modifiers. The input
/// is sanitised to `[0, 1]` first, so any value is accepted without panicking;
/// `0` (or any non-finite input) returns [`WetHairResponse::DRY`].
#[must_use]
pub fn wet_hair_response(wetness: f32) -> WetHairResponse {
    let w = sanitize_wetness(wetness);
    WetHairResponse {
        clump_scale: lerp(1.0, CLUMP_SCALE_WET, w),
        mass_mul: lerp(1.0, MASS_MUL_WET, w),
        damping_mul: lerp(1.0, DAMPING_MUL_WET, w),
        sigma_a_mul: lerp(1.0, SIGMA_A_MUL_WET, w),
        roughness_delta: lerp(0.0, ROUGHNESS_DELTA_WET, w),
    }
}

/// Map a slice of wetness fractions to their responses, preserving order. An
/// empty input yields an empty output; the result length always equals the
/// input length.
#[must_use]
pub fn wet_hair_response_map(wetness: &[f32]) -> Vec<WetHairResponse> {
    wetness.iter().map(|&w| wet_hair_response(w)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn dry_is_identity() {
        let r = wet_hair_response(0.0);
        assert_eq!(r, WetHairResponse::DRY);
        assert!(close(r.clump_scale, 1.0));
        assert!(close(r.mass_mul, 1.0));
        assert!(close(r.damping_mul, 1.0));
        assert!(close(r.sigma_a_mul, 1.0));
        assert!(close(r.roughness_delta, 0.0));
    }

    #[test]
    fn fully_wet_hits_endpoints() {
        let r = wet_hair_response(1.0);
        assert!(close(r.clump_scale, CLUMP_SCALE_WET));
        assert!(close(r.mass_mul, MASS_MUL_WET));
        assert!(close(r.damping_mul, DAMPING_MUL_WET));
        assert!(close(r.sigma_a_mul, SIGMA_A_MUL_WET));
        assert!(close(r.roughness_delta, ROUGHNESS_DELTA_WET));
    }

    #[test]
    fn half_wet_is_midpoint() {
        let r = wet_hair_response(0.5);
        assert!(close(r.clump_scale, (1.0 + CLUMP_SCALE_WET) * 0.5));
        assert!(close(r.mass_mul, (1.0 + MASS_MUL_WET) * 0.5));
        assert!(close(r.sigma_a_mul, (1.0 + SIGMA_A_MUL_WET) * 0.5));
        assert!(close(r.roughness_delta, ROUGHNESS_DELTA_WET * 0.5));
    }

    #[test]
    fn monotonic_in_wetness() {
        let dry = wet_hair_response(0.25);
        let wet = wet_hair_response(0.75);
        // Clumping tightens and roughness drops as it gets wetter; the other
        // modifiers grow.
        assert!(wet.clump_scale < dry.clump_scale);
        assert!(wet.roughness_delta < dry.roughness_delta);
        assert!(wet.mass_mul > dry.mass_mul);
        assert!(wet.damping_mul > dry.damping_mul);
        assert!(wet.sigma_a_mul > dry.sigma_a_mul);
    }

    #[test]
    fn out_of_range_and_non_finite_sanitize() {
        assert!(close(sanitize_wetness(-3.0), 0.0));
        assert!(close(sanitize_wetness(2.5), 1.0));
        assert!(close(sanitize_wetness(f32::NAN), 0.0));
        assert!(close(sanitize_wetness(f32::INFINITY), 0.0));
        // Over-saturated input behaves exactly like fully wet.
        assert_eq!(wet_hair_response(9.0), wet_hair_response(1.0));
        // Non-finite input behaves exactly like dry.
        assert_eq!(wet_hair_response(f32::NAN), WetHairResponse::DRY);
    }

    #[test]
    fn apply_absorption_scales_each_channel() {
        let r = wet_hair_response(1.0);
        let out = r.apply_absorption([0.4, 0.7, 1.3]);
        assert!(close(out[0], 0.4 * SIGMA_A_MUL_WET));
        assert!(close(out[1], 0.7 * SIGMA_A_MUL_WET));
        assert!(close(out[2], 1.3 * SIGMA_A_MUL_WET));
    }

    #[test]
    fn apply_roughness_clamps_to_unit_range() {
        let r = wet_hair_response(1.0);
        // A low base roughness cannot be pushed below zero.
        assert!(close(r.apply_roughness(0.1), 0.0));
        // A mid base roughness drops by the offset.
        assert!(close(r.apply_roughness(0.8), 0.8 + ROUGHNESS_DELTA_WET));
        // Dry leaves roughness untouched.
        assert!(close(WetHairResponse::DRY.apply_roughness(0.6), 0.6));
    }

    #[test]
    fn map_matches_scalar_and_preserves_order() {
        let input = [0.0, 0.5, 1.0, -1.0];
        let mapped = wet_hair_response_map(&input);
        assert_eq!(mapped.len(), input.len());
        for (i, &w) in input.iter().enumerate() {
            assert_eq!(mapped[i], wet_hair_response(w));
        }
        // Order is preserved: first is dry, last (sanitised from -1) is dry too.
        assert_eq!(mapped[0], WetHairResponse::DRY);
        assert_eq!(mapped[3], WetHairResponse::DRY);
    }

    #[test]
    fn empty_map_is_empty_without_panic() {
        assert!(wet_hair_response_map(&[]).is_empty());
    }
}
