//! Melanin-pigment absorption parameterisation for physically based hair colour.
//!
//! A production hair BSDF (Marschner/Chiang R/TT/TRT in
//! [`crate::hair::dual_scattering`]'s sibling shading closures) is driven by a
//! spectral *absorption* coefficient `sigma_a`, not by an RGB "tint". Real hair
//! colour comes almost entirely from two pigments: **eumelanin** (brown-black)
//! and **pheomelanin** (red-yellow). Black, brown, blond and red hair are all
//! the *same* model at different pigment concentrations, which is why melanin
//! parameterisation (rather than a free tint colour) gives physically plausible
//! colour, natural dye/ombre gradients, and energy-consistent multiple
//! scattering. This is the parameterisation used by the film-grade `Chiang`
//! 2016 production hair model and the earlier energy-conserving `d'Eon` 2011
//! model, and it is what `UE5` Groom and `pbrt` expose as the pigment inputs.
//!
//! The concentration -> `sigma_a` map is a *material-independent, deterministic*
//! quantity (a fixed per-pigment absorption spectrum times a scalar
//! concentration, summed), so it lives in this architecture crate exactly like
//! the forward-scatter crossing count in [`crate::hair::dual_scattering`]: array
//! in, array out, golden-comparable, panic-free. The shading side then feeds the
//! returned `sigma_a` into the Beer-Lambert transmittance of each lobe (the only
//! transcendental step, which belongs to the material closure, not here). This
//! module performs **no** transcendental math — it is a pure non-negative linear
//! combination — so it needs no `libm` determinism shims.
//!
//! The per-pigment absorption spectra sampled at the renderer's RGB primaries
//! are the canonical values from the `Chiang` 2016 model (as tabulated by
//! `pbrt`): eumelanin absorbs most strongly in blue (giving the warm brown-black
//! cast), pheomelanin is comparatively flat with a red-yellow bias. See
//! [`EUMELANIN_SIGMA_A`] / [`PHEOMELANIN_SIGMA_A`].

use alloc::vec::Vec;

/// Per-unit-concentration eumelanin absorption at the renderer's RGB primaries
/// (`Chiang` 2016 / `pbrt`). Blue is absorbed most strongly, which is what makes
/// dense eumelanin read as brown-to-black rather than neutral grey.
pub const EUMELANIN_SIGMA_A: [f32; 3] = [0.419, 0.697, 1.37];

/// Per-unit-concentration pheomelanin absorption at the renderer's RGB primaries
/// (`Chiang` 2016 / `pbrt`). Flatter and lower than eumelanin with a red-yellow
/// bias, so pheomelanin-dominant grooms read as red/ginger.
pub const PHEOMELANIN_SIGMA_A: [f32; 3] = [0.187, 0.4, 1.05];

/// A hair fibre's pigment content: non-negative eumelanin and pheomelanin
/// concentrations in the same arbitrary unit as [`EUMELANIN_SIGMA_A`] /
/// [`PHEOMELANIN_SIGMA_A`] (so a concentration of `1.0` reproduces that spectrum
/// exactly). Negative inputs are unphysical and are clamped to `0` by
/// [`MelaninProfile::sanitized`] before use; they never panic.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MelaninProfile {
    /// Eumelanin (brown-black pigment) concentration; clamped to `>= 0`.
    pub eumelanin: f32,
    /// Pheomelanin (red-yellow pigment) concentration; clamped to `>= 0`.
    pub pheomelanin: f32,
}

impl MelaninProfile {
    /// A pigment profile from explicit eumelanin / pheomelanin concentrations.
    #[must_use]
    pub const fn new(eumelanin: f32, pheomelanin: f32) -> Self {
        Self {
            eumelanin,
            pheomelanin,
        }
    }

    /// This profile with any negative (unphysical) concentration clamped to `0`.
    /// Non-finite inputs (`NaN`/inf) are treated as `0` so downstream absorption
    /// stays finite and non-negative.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            eumelanin: clamp_concentration(self.eumelanin),
            pheomelanin: clamp_concentration(self.pheomelanin),
        }
    }
}

/// Clamps one concentration to a finite, non-negative value (negative or
/// non-finite -> `0`), without any floating-point equality test.
#[must_use]
fn clamp_concentration(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

/// The RGB absorption coefficient `sigma_a` of a fibre with the given pigment
/// content: `eumelanin * EUMELANIN_SIGMA_A + pheomelanin * PHEOMELANIN_SIGMA_A`,
/// per channel. The result is finite and non-negative for every input (negative
/// / non-finite concentrations are clamped first). This is the quantity the
/// shading closure raises to Beer-Lambert transmittance per lobe.
#[must_use]
pub fn melanin_absorption(profile: MelaninProfile) -> [f32; 3] {
    let clean = profile.sanitized();
    [
        clean.eumelanin * EUMELANIN_SIGMA_A[0] + clean.pheomelanin * PHEOMELANIN_SIGMA_A[0],
        clean.eumelanin * EUMELANIN_SIGMA_A[1] + clean.pheomelanin * PHEOMELANIN_SIGMA_A[1],
        clean.eumelanin * EUMELANIN_SIGMA_A[2] + clean.pheomelanin * PHEOMELANIN_SIGMA_A[2],
    ]
}

/// Per-fibre absorption for a whole groom: maps each [`MelaninProfile`] through
/// [`melanin_absorption`], preserving input order. An empty slice returns an
/// empty [`Vec`] (no panic); this is the array-in/array-out form used for a root
/// melanin texture or per-strand pigment attribute.
#[must_use]
pub fn melanin_absorption_map(profiles: &[MelaninProfile]) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(profiles.len());
    for &profile in profiles {
        out.push(melanin_absorption(profile));
    }
    out
}

/// Representative natural hair colours as pigment concentrations, usable as
/// artist-facing starting points before per-groom tuning. The concentrations
/// are illustrative values in the literature's documented ranges (black and
/// brown are eumelanin-dominant with little pheomelanin; blond is low total
/// eumelanin; red is pheomelanin-dominant), not a measurement of any specific
/// person. Feed [`NaturalHairColor::profile`] into [`melanin_absorption`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NaturalHairColor {
    /// Dense eumelanin, negligible pheomelanin.
    Black,
    /// High eumelanin with a little pheomelanin.
    Brown,
    /// Low total eumelanin, slight pheomelanin.
    Blond,
    /// Pheomelanin-dominant with modest eumelanin.
    Red,
}

impl NaturalHairColor {
    /// The representative pigment concentrations for this colour.
    #[must_use]
    pub const fn profile(self) -> MelaninProfile {
        match self {
            Self::Black => MelaninProfile::new(8.0, 0.0),
            Self::Brown => MelaninProfile::new(1.3, 0.1),
            Self::Blond => MelaninProfile::new(0.2, 0.05),
            Self::Red => MelaninProfile::new(0.3, 1.5),
        }
    }

    /// The RGB absorption coefficient for this representative colour.
    #[must_use]
    pub fn absorption(self) -> [f32; 3] {
        melanin_absorption(self.profile())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (a[0] - b[0]).abs() < EPS && (a[1] - b[1]).abs() < EPS && (a[2] - b[2]).abs() < EPS
    }

    #[test]
    fn zero_pigment_is_zero_absorption() {
        let sigma = melanin_absorption(MelaninProfile::new(0.0, 0.0));
        assert!(close(sigma, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn unit_eumelanin_reproduces_spectrum() {
        let sigma = melanin_absorption(MelaninProfile::new(1.0, 0.0));
        assert!(close(sigma, EUMELANIN_SIGMA_A));
    }

    #[test]
    fn unit_pheomelanin_reproduces_spectrum() {
        let sigma = melanin_absorption(MelaninProfile::new(0.0, 1.0));
        assert!(close(sigma, PHEOMELANIN_SIGMA_A));
    }

    #[test]
    fn mix_is_linear_combination() {
        let eu = 2.0;
        let pheo = 3.0;
        let sigma = melanin_absorption(MelaninProfile::new(eu, pheo));
        let expected = [
            eu * EUMELANIN_SIGMA_A[0] + pheo * PHEOMELANIN_SIGMA_A[0],
            eu * EUMELANIN_SIGMA_A[1] + pheo * PHEOMELANIN_SIGMA_A[1],
            eu * EUMELANIN_SIGMA_A[2] + pheo * PHEOMELANIN_SIGMA_A[2],
        ];
        assert!(close(sigma, expected));
    }

    #[test]
    fn negative_and_non_finite_concentrations_clamp_to_zero() {
        let from_negative = melanin_absorption(MelaninProfile::new(-5.0, -1.0));
        assert!(close(from_negative, [0.0, 0.0, 0.0]));

        let from_nan = melanin_absorption(MelaninProfile::new(f32::NAN, f32::INFINITY));
        assert!(from_nan.iter().all(|c| c.is_finite()));
        assert!(close(from_nan, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn eumelanin_absorbs_blue_more_than_red() {
        // The warm brown-black cast comes from blue being absorbed hardest.
        let sigma = melanin_absorption(MelaninProfile::new(1.0, 0.0));
        assert!(sigma[2] > sigma[1]);
        assert!(sigma[1] > sigma[0]);
    }

    #[test]
    fn map_matches_scalar_and_preserves_order() {
        let profiles = [
            MelaninProfile::new(1.0, 0.0),
            MelaninProfile::new(0.0, 1.0),
            MelaninProfile::new(2.0, 0.5),
        ];
        let mapped = melanin_absorption_map(&profiles);
        assert_eq!(mapped.len(), profiles.len());
        for (profile, got) in profiles.iter().zip(mapped.iter()) {
            assert!(close(*got, melanin_absorption(*profile)));
        }
    }

    #[test]
    fn empty_map_is_empty_without_panic() {
        let mapped = melanin_absorption_map(&[]);
        assert!(mapped.is_empty());
    }

    #[test]
    fn black_absorbs_more_than_blond() {
        let black_total: f32 = NaturalHairColor::Black.absorption().iter().sum();
        let blond_total: f32 = NaturalHairColor::Blond.absorption().iter().sum();
        assert!(black_total > blond_total);
    }

    #[test]
    fn red_is_pheomelanin_dominant() {
        // A ginger groom should carry more pheomelanin than eumelanin.
        let red = NaturalHairColor::Red.profile();
        assert!(red.pheomelanin > red.eumelanin);
    }
}
