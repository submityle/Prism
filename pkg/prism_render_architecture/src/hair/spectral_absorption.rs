//! Spectral melanin absorption and `Hero-wavelength` sampling for hair colour.
//!
//! The RGB melanin parameterisation in [`crate::hair::melanin`] gives three
//! absorption coefficients at the renderer's RGB primaries. That is enough for
//! an RGB pipeline, but a spectral renderer needs the full `lambda -> sigma_a`
//! curve so that dispersion, fluorescent dye, and narrow-band studio lights read
//! correctly, and so that pigmented hair avoids the characteristic RGB
//! "hue-shift under saturated light" error. This module extends the two per-unit
//! pigment coefficients to a sampled visible-light spectrum (380-730nm) and adds
//! stratified `Hero-wavelength` sampling in the sense of `Wilkie` 2014
//! ("Hero Wavelength Spectral Sampling").
//!
//! As with the `Chiang` 2016 / `d'Eon` 2011 pigment model that the RGB sibling
//! follows, the concentration -> `sigma_a` map is a *material-independent,
//! deterministic* linear combination of two fixed per-pigment spectra, so it
//! belongs in this architecture crate exactly like the RGB form: table in,
//! sample out, golden-comparable, panic-free. The shading closure still owns the
//! single transcendental step (Beer-Lambert transmittance per lobe); this module
//! performs **no** transcendental math — it is pure table lookup plus linear
//! interpolation — so it needs no `libm` determinism shims.
//!
//! The per-pigment spectra are monotonically decreasing with wavelength (blue is
//! absorbed hardest, red weakest), calibrated so that the sampled values near the
//! renderer's RGB primaries match the canonical `Chiang` 2016 / `pbrt` RGB
//! constants in [`crate::hair::melanin`]: eumelanin runs from strong violet
//! absorption down to a weak red tail (giving the warm brown-black cast),
//! pheomelanin is lower and flatter with a red-yellow bias (giving red/ginger).
//! Values are illustrative physically plausible curves, not a spectrophotometer
//! measurement of any specific fibre.

use alloc::vec::Vec;

/// Number of uniformly spaced samples in each per-pigment absorption spectrum.
pub const SPECTRUM_SAMPLES: usize = 15;

/// Shortest sampled wavelength, in nanometres (violet end of visible light).
pub const SPECTRUM_MIN_NM: f32 = 380.0;

/// Longest sampled wavelength, in nanometres (red end of visible light).
pub const SPECTRUM_MAX_NM: f32 = 730.0;

/// Per-unit-concentration eumelanin absorption sampled uniformly across
/// `[SPECTRUM_MIN_NM, SPECTRUM_MAX_NM]`. Monotonically decreasing from strong
/// violet absorption to a weak red tail; the samples near the RGB primaries
/// reproduce the magnitudes of [`crate::hair::melanin::EUMELANIN_SIGMA_A`].
pub const EUMELANIN_SPECTRUM: [f32; SPECTRUM_SAMPLES] = [
    1.95,  // 380 nm
    1.75,  // 405 nm
    1.56,  // 430 nm
    1.40,  // 455 nm
    1.18,  // 480 nm
    0.92,  // 505 nm
    0.70,  // 530 nm
    0.60,  // 555 nm
    0.52,  // 580 nm
    0.46,  // 605 nm
    0.419, // 630 nm
    0.39,  // 655 nm
    0.37,  // 680 nm
    0.35,  // 705 nm
    0.33,  // 730 nm
];

/// Per-unit-concentration pheomelanin absorption sampled uniformly across
/// `[SPECTRUM_MIN_NM, SPECTRUM_MAX_NM]`. Lower and flatter than eumelanin with a
/// red-yellow bias; the samples near the RGB primaries reproduce the magnitudes
/// of [`crate::hair::melanin::PHEOMELANIN_SIGMA_A`].
pub const PHEOMELANIN_SPECTRUM: [f32; SPECTRUM_SAMPLES] = [
    1.30,  // 380 nm
    1.20,  // 405 nm
    1.12,  // 430 nm
    1.06,  // 455 nm
    0.85,  // 480 nm
    0.60,  // 505 nm
    0.40,  // 530 nm
    0.33,  // 555 nm
    0.27,  // 580 nm
    0.22,  // 605 nm
    0.187, // 630 nm
    0.17,  // 655 nm
    0.155, // 680 nm
    0.14,  // 705 nm
    0.13,  // 730 nm
];

/// Clamps one concentration to a finite, non-negative value (negative or
/// non-finite -> `0`), without any floating-point equality test. Mirrors the
/// sanitisation used by [`crate::hair::melanin`].
#[must_use]
fn clamp_concentration(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

/// Clamps a wavelength to the sampled range, mapping non-finite inputs to the
/// short-wavelength endpoint. This keeps interpolation in-bounds and panic-free
/// for any input (the table is clamped, not extrapolated, outside its range).
#[must_use]
fn sanitized_wavelength(wavelength_nm: f32) -> f32 {
    if wavelength_nm.is_finite() {
        wavelength_nm.clamp(SPECTRUM_MIN_NM, SPECTRUM_MAX_NM)
    } else {
        SPECTRUM_MIN_NM
    }
}

/// Wraps a value into `[0, 1)` by subtracting its floor, mapping non-finite
/// inputs to `0`. Used to rotate `Hero-wavelength` samples cyclically through
/// the visible band without any transcendental call.
#[must_use]
fn wrap_unit(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    let fractional = value - value.floor();
    if !(0.0..1.0).contains(&fractional) {
        0.0
    } else {
        fractional
    }
}

/// Linearly interpolates a uniformly sampled spectrum at `wavelength_nm`,
/// clamping to the endpoints outside the sampled range. Never panics.
#[must_use]
fn sample_spectrum(table: &[f32; SPECTRUM_SAMPLES], wavelength_nm: f32) -> f32 {
    let clamped = sanitized_wavelength(wavelength_nm);
    let span = SPECTRUM_MAX_NM - SPECTRUM_MIN_NM;
    let step = span / (SPECTRUM_SAMPLES as f32 - 1.0);
    let position = (clamped - SPECTRUM_MIN_NM) / step;
    let lower_f = position.floor();
    let lower = lower_f as usize;
    if lower >= SPECTRUM_SAMPLES - 1 {
        return table[SPECTRUM_SAMPLES - 1];
    }
    let frac = position - lower_f;
    table[lower] * (1.0 - frac) + table[lower + 1] * frac
}

/// Per-unit-concentration eumelanin absorption at an arbitrary wavelength,
/// linearly interpolated from [`EUMELANIN_SPECTRUM`] and clamped to the sampled
/// range at the endpoints.
#[must_use]
pub fn eumelanin_sigma_a_at(wavelength_nm: f32) -> f32 {
    sample_spectrum(&EUMELANIN_SPECTRUM, wavelength_nm)
}

/// Per-unit-concentration pheomelanin absorption at an arbitrary wavelength,
/// linearly interpolated from [`PHEOMELANIN_SPECTRUM`] and clamped to the sampled
/// range at the endpoints.
#[must_use]
pub fn pheomelanin_sigma_a_at(wavelength_nm: f32) -> f32 {
    sample_spectrum(&PHEOMELANIN_SPECTRUM, wavelength_nm)
}

/// The spectral absorption coefficient `sigma_a` at `wavelength_nm` for a fibre
/// with the given pigment concentrations:
/// `eumelanin * eumelanin_sigma_a_at + pheomelanin * pheomelanin_sigma_a_at`.
/// Negative / non-finite concentrations are clamped to `0` first, so the result
/// is always finite and non-negative.
#[must_use]
pub fn melanin_sigma_a_at(eumelanin: f32, pheomelanin: f32, wavelength_nm: f32) -> f32 {
    let eu = clamp_concentration(eumelanin);
    let pheo = clamp_concentration(pheomelanin);
    eu * eumelanin_sigma_a_at(wavelength_nm) + pheo * pheomelanin_sigma_a_at(wavelength_nm)
}

/// Four stratified wavelengths for one `Hero-wavelength` spectral sample
/// (`Wilkie` 2014). Index `0` is the hero (primary) wavelength; the other three
/// are its equally spaced cyclic companions through the visible band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeroWavelengths {
    /// The four sampled wavelengths in nanometres, all within
    /// `[SPECTRUM_MIN_NM, SPECTRUM_MAX_NM)`.
    pub lambdas: [f32; 4],
}

/// Builds a `Hero-wavelength` sample from a stratified unit sample `u` and a
/// per-sample `rotate` offset (both in units of the visible band, cyclically
/// wrapped). The four wavelengths are spaced a quarter of the band apart and
/// wrap around the red->violet boundary, following `Wilkie` 2014. Non-finite or
/// out-of-range inputs are wrapped into `[0, 1)`, so this never panics.
#[must_use]
pub fn hero_wavelengths(u: f32, rotate: f32) -> HeroWavelengths {
    let span = SPECTRUM_MAX_NM - SPECTRUM_MIN_NM;
    let base = wrap_unit(wrap_unit(u) + wrap_unit(rotate));
    let mut lambdas = [0.0f32; 4];
    let mut j = 0usize;
    while j < 4 {
        let offset = base + 0.25 * j as f32;
        lambdas[j] = SPECTRUM_MIN_NM + wrap_unit(offset) * span;
        j += 1;
    }
    HeroWavelengths { lambdas }
}

/// The spectral absorption coefficient `sigma_a` at each of the four
/// `Hero-wavelength` samples for the given pigment concentrations. Array in,
/// array out; finite and non-negative for every input.
#[must_use]
pub fn hero_sigma_a(eumelanin: f32, pheomelanin: f32, hero: HeroWavelengths) -> [f32; 4] {
    [
        melanin_sigma_a_at(eumelanin, pheomelanin, hero.lambdas[0]),
        melanin_sigma_a_at(eumelanin, pheomelanin, hero.lambdas[1]),
        melanin_sigma_a_at(eumelanin, pheomelanin, hero.lambdas[2]),
        melanin_sigma_a_at(eumelanin, pheomelanin, hero.lambdas[3]),
    ]
}

/// Maps a slice of wavelengths to their `sigma_a` for the given pigment
/// concentrations, preserving input order. An empty slice returns an empty
/// [`Vec`] (no panic); this is the array-in/array-out form used to resample a
/// whole spectral tabulation.
#[must_use]
pub fn spectrum_sample_map(eumelanin: f32, pheomelanin: f32, wavelengths: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(wavelengths.len());
    for &wavelength_nm in wavelengths {
        out.push(melanin_sigma_a_at(eumelanin, pheomelanin, wavelength_nm));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn blue_is_absorbed_more_than_red() {
        // Blue (~450nm) absorption must exceed red (~650nm) for both pigments.
        assert!(eumelanin_sigma_a_at(450.0) > eumelanin_sigma_a_at(650.0));
        assert!(pheomelanin_sigma_a_at(450.0) > pheomelanin_sigma_a_at(650.0));
    }

    #[test]
    fn spectrum_is_monotonically_decreasing() {
        let mut i = 1usize;
        while i < SPECTRUM_SAMPLES {
            assert!(EUMELANIN_SPECTRUM[i] <= EUMELANIN_SPECTRUM[i - 1]);
            assert!(PHEOMELANIN_SPECTRUM[i] <= PHEOMELANIN_SPECTRUM[i - 1]);
            i += 1;
        }
    }

    #[test]
    fn zero_concentration_is_zero_absorption() {
        assert!(close(melanin_sigma_a_at(0.0, 0.0, 500.0), 0.0));
    }

    #[test]
    fn absorption_is_linear_in_concentration() {
        let single = melanin_sigma_a_at(1.0, 0.5, 520.0);
        let doubled = melanin_sigma_a_at(2.0, 1.0, 520.0);
        assert!(close(doubled, 2.0 * single));
    }

    #[test]
    fn unit_concentration_matches_component_samples() {
        let wavelength = 500.0;
        let combined = melanin_sigma_a_at(1.0, 1.0, wavelength);
        let expected = eumelanin_sigma_a_at(wavelength) + pheomelanin_sigma_a_at(wavelength);
        assert!(close(combined, expected));
    }

    #[test]
    fn out_of_range_wavelengths_clamp_to_endpoints() {
        assert!(close(eumelanin_sigma_a_at(100.0), EUMELANIN_SPECTRUM[0],));
        assert!(close(
            eumelanin_sigma_a_at(2000.0),
            EUMELANIN_SPECTRUM[SPECTRUM_SAMPLES - 1],
        ));
        assert!(close(
            pheomelanin_sigma_a_at(f32::NAN),
            PHEOMELANIN_SPECTRUM[0],
        ));
        // Exact endpoints hit the first / last table entries.
        assert!(close(
            eumelanin_sigma_a_at(SPECTRUM_MIN_NM),
            EUMELANIN_SPECTRUM[0]
        ));
        assert!(close(
            eumelanin_sigma_a_at(SPECTRUM_MAX_NM),
            EUMELANIN_SPECTRUM[SPECTRUM_SAMPLES - 1],
        ));
    }

    #[test]
    fn hero_wavelengths_are_in_range() {
        let hero = hero_wavelengths(0.3, 0.1);
        for &lambda in &hero.lambdas {
            assert!(lambda >= SPECTRUM_MIN_NM);
            assert!(lambda <= SPECTRUM_MAX_NM);
            assert!(lambda.is_finite());
        }
    }

    #[test]
    fn hero_wavelengths_are_equally_spaced_cyclically() {
        let span = SPECTRUM_MAX_NM - SPECTRUM_MIN_NM;
        let quarter = 0.25 * span;
        let hero = hero_wavelengths(0.2, 0.05);
        let mut j = 0usize;
        while j < 4 {
            let next = hero.lambdas[(j + 1) % 4];
            let mut delta = next - hero.lambdas[j];
            if delta < 0.0 {
                delta += span;
            }
            assert!(close(delta, quarter));
            j += 1;
        }
    }

    #[test]
    fn hero_wavelengths_sanitize_bad_inputs() {
        let from_nan = hero_wavelengths(f32::NAN, f32::INFINITY);
        for &lambda in &from_nan.lambdas {
            assert!(lambda.is_finite());
            assert!(lambda >= SPECTRUM_MIN_NM);
            assert!(lambda <= SPECTRUM_MAX_NM);
        }
        let from_large = hero_wavelengths(12.7, -3.4);
        for &lambda in &from_large.lambdas {
            assert!(lambda.is_finite());
            assert!(lambda >= SPECTRUM_MIN_NM);
            assert!(lambda <= SPECTRUM_MAX_NM);
        }
    }

    #[test]
    fn hero_sigma_a_is_finite_for_bad_concentrations() {
        let hero = hero_wavelengths(0.5, 0.0);
        let sigma = hero_sigma_a(f32::NAN, f32::INFINITY, hero);
        for &value in &sigma {
            assert!(value.is_finite());
            assert!(close(value, 0.0));
        }
    }

    #[test]
    fn negative_concentration_clamps_to_zero() {
        let sigma = melanin_sigma_a_at(-4.0, -2.0, 500.0);
        assert!(sigma.is_finite());
        assert!(close(sigma, 0.0));
    }

    #[test]
    fn empty_map_is_empty_without_panic() {
        let mapped = spectrum_sample_map(1.0, 1.0, &[]);
        assert!(mapped.is_empty());
    }

    #[test]
    fn map_matches_scalar_and_preserves_order() {
        let wavelengths = [450.0, 520.0, 600.0, 680.0];
        let mapped = spectrum_sample_map(1.3, 0.4, &wavelengths);
        assert_eq!(mapped.len(), wavelengths.len());
        let mut i = 0usize;
        while i < wavelengths.len() {
            assert!(close(
                mapped[i],
                melanin_sigma_a_at(1.3, 0.4, wavelengths[i])
            ));
            i += 1;
        }
    }
}
