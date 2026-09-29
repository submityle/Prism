//! Ocean wave spectra, dispersion, phase advance, and the wave-folding
//! (whitecap) criterion — the classical spectral core of a `Tessendorf` ocean.
//!
//! An open ocean is not a handful of animated sine waves; it is a random field
//! whose energy is distributed across wave numbers by a statistical spectrum.
//! This module is the pure-CPU math for that field:
//!
//! * energy density functions for the `Phillips`, `JONSWAP`, and
//!   Pierson-Moskowitz spectra (`PM`);
//! * the deep-water dispersion relation `omega(k) = sqrt(g*k)`;
//! * time evolution of a complex spectral amplitude
//!   `h(k, t) = h0(k) e^{i*omega*t} + conj(h0(-k)) e^{-i*omega*t}`, which keeps
//!   the inverse-`FFT` height field real; and
//! * the Jacobian fold test that flags wave crests steep enough to break into
//!   whitecap foam.
//!
//! The inverse `FFT` itself is a GPU kernel (see the WESL scaffolding); this
//! module produces the per-wave-number inputs that kernel consumes and the
//! classification the foam field reads. Only `sqrt` and the hand-rolled
//! [`super::exp_approx`] / [`super::sin_approx`] / [`super::cos_approx`] are
//! used — no `libm`, no transcendental intrinsics.

use super::{cos_approx, exp_approx, sin_approx, Vec2, GRAVITY};

/// A hand-rolled complex number for spectral amplitudes.
///
/// The spectral field is complex-valued; `prism_render_architecture` is
/// dependency-free, so the two-component complex type is spelled out here.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl Complex {
    /// The additive identity.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Builds a complex number from parts.
    #[must_use]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Complex conjugate `re - i*im`.
    #[must_use]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Complex sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The spectral math API uses named add/mul methods for call-site uniformity with Vec2/Vec3; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }

    /// Complex product.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: named mul for call-site uniformity, not an operator trait."
    )]
    pub fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }

    /// Multiplies by the unit phasor `e^{i*theta} = cos(theta) + i*sin(theta)`.
    #[must_use]
    pub fn mul_phasor(self, theta: f32) -> Self {
        self.mul(Self::new(cos_approx(theta), sin_approx(theta)))
    }

    /// Squared magnitude `re^2 + im^2`.
    #[must_use]
    pub fn norm_squared(self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

/// Which statistical spectrum shapes the initial energy distribution.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SpectrumKind {
    /// `Phillips` spectrum: the classic wind-driven ocean spectrum, simple and
    /// art-directable via wind speed and direction.
    Phillips,
    /// `JONSWAP` spectrum: a fetch-limited peak-enhanced spectrum for growing
    /// wind seas (sharper, younger waves).
    Jonswap,
    /// Pierson-Moskowitz spectrum: a fully developed sea in equilibrium with a
    /// sustained wind.
    PiersonMoskowitz,
}

/// Parameters describing one wind-driven sea state.
///
/// `wind` points downwind; its length is the wind speed `V` in m/s. `amplitude`
/// scales overall energy. `min_wavelength` suppresses ripples below a chosen
/// wavelength (numerical stability / anti-alias of the highest cascade).
/// `directional_exponent` sharpens alignment of energy to the wind direction.
#[derive(Clone, Copy, Debug)]
pub struct SpectrumParams {
    /// Statistical spectrum shape.
    pub kind: SpectrumKind,
    /// Downwind vector; its magnitude is the wind speed `V` (m/s).
    pub wind: Vec2,
    /// Overall energy scale (the `Phillips` `A` constant / `JONSWAP` alpha).
    pub amplitude: f32,
    /// Peak-enhancement factor `gamma` for `JONSWAP` (`1.0` reduces to `PM`).
    pub peak_enhancement: f32,
    /// Shortest wavelength (m) kept before high-frequency suppression.
    pub min_wavelength: f32,
    /// Directional-spread exponent; higher aligns energy tighter to the wind.
    pub directional_exponent: u32,
}

impl SpectrumParams {
    /// Wind speed `V` (m/s), the magnitude of [`SpectrumParams::wind`].
    #[must_use]
    pub fn wind_speed(self) -> f32 {
        self.wind.length()
    }

    /// The largest wave the wind sustains, `L = V^2 / g` (m). Longer waves
    /// carry more energy, so `L` grows with wind speed.
    #[must_use]
    pub fn largest_wave(self) -> f32 {
        let v = self.wind_speed();
        v * v / GRAVITY
    }
}

/// Deep-water dispersion relation `omega(k) = sqrt(g*k)`.
///
/// `k` is the wave-number magnitude (rad/m). Monotonically increasing in `k`,
/// so longer waves (small `k`) travel slower in angular frequency — the
/// property the phase-advance and cascade code relies on. Non-positive `k`
/// yields `0`.
#[must_use]
pub fn dispersion(k: f32) -> f32 {
    if k <= 0.0 {
        return 0.0;
    }
    (GRAVITY * k).sqrt()
}

/// `Phillips` energy density at wave vector `k_vec`.
///
/// `P(k) = A * exp(-1/(k L)^2) / k^4 * |k_hat . w_hat|^{2n} * exp(-(k l)^2)`,
/// where `L = V^2/g` and `l = min_wavelength`. The `1/k^4` favors long waves,
/// the first exponential cuts waves longer than the wind sustains, the
/// directional term aligns energy downwind, and the last exponential suppresses
/// ripples below `min_wavelength`. Always non-negative; `0` at `k = 0`.
#[must_use]
pub fn phillips(k_vec: Vec2, params: SpectrumParams) -> f32 {
    let k_sq = k_vec.length_squared();
    if k_sq < super::EPS_LEN_SQ {
        return 0.0;
    }
    let k = k_sq.sqrt();
    let big_l = params.largest_wave();
    if big_l <= 0.0 {
        return 0.0;
    }
    let k_l = k * big_l;
    // exp(-1/(kL)^2): cuts waves longer than the wind can build.
    let long_cut = exp_approx(-1.0 / (k_l * k_l));
    // 1/k^4 spectral falloff.
    let falloff = 1.0 / (k_sq * k_sq);
    // Directional alignment |k_hat . w_hat|^{2n}.
    let wind_dir = params.wind.normalize_or_zero();
    let k_dir = k_vec.scale(1.0 / k);
    let mut align = k_dir.dot(wind_dir);
    if align < 0.0 {
        align = -align;
    }
    let mut directional = 1.0;
    let mut i = 0;
    while i < params.directional_exponent {
        directional *= align * align;
        i += 1;
    }
    // exp(-(k l)^2): suppress sub-min_wavelength ripples.
    let l = params.min_wavelength;
    let short_cut = exp_approx(-(k * l) * (k * l));
    params.amplitude * long_cut * falloff * directional * short_cut
}

/// Pierson-Moskowitz omnidirectional density at angular frequency `omega`.
///
/// `S(omega) = alpha * g^2 / omega^5 * exp(-1.25 * (omega_p / omega)^4)`, with a
/// peak frequency `omega_p = g / V` set by the wind speed. Non-negative; `0`
/// for non-positive `omega`.
#[must_use]
pub fn pierson_moskowitz(omega: f32, params: SpectrumParams) -> f32 {
    if omega <= 0.0 {
        return 0.0;
    }
    let v = params.wind_speed();
    if v <= 0.0 {
        return 0.0;
    }
    let omega_p = GRAVITY / v;
    let ratio = omega_p / omega;
    let ratio4 = {
        let r2 = ratio * ratio;
        r2 * r2
    };
    let omega2 = omega * omega;
    let omega5 = omega2 * omega2 * omega;
    let g2 = GRAVITY * GRAVITY;
    params.amplitude * g2 / omega5 * exp_approx(-1.25 * ratio4)
}

/// `JONSWAP` omnidirectional density at angular frequency `omega`.
///
/// A fetch-limited sea: Pierson-Moskowitz multiplied by the peak-enhancement
/// `gamma^r`, where `r = exp(-(omega - omega_p)^2 / (2 sigma^2 omega_p^2))` and
/// `sigma` widens above the peak. `peak_enhancement = 1` recovers `PM`.
/// Non-negative.
#[must_use]
pub fn jonswap(omega: f32, params: SpectrumParams) -> f32 {
    let base = pierson_moskowitz(omega, params);
    if base <= 0.0 {
        return 0.0;
    }
    let v = params.wind_speed();
    let omega_p = GRAVITY / v;
    let sigma = if omega <= omega_p { 0.07 } else { 0.09 };
    let denom = 2.0 * sigma * sigma * omega_p * omega_p;
    let diff = omega - omega_p;
    let r = exp_approx(-(diff * diff) / denom);
    // gamma^r via exp/ln-free power: gamma^r = exp(r * ln(gamma)) is disallowed,
    // so approximate the (usually small) exponent by a monotone blend that is
    // exact at r=0 (factor 1) and r=1 (factor gamma).
    let gamma = params.peak_enhancement;
    let enhancement = 1.0 + (gamma - 1.0) * r;
    base * enhancement
}

/// Evaluates the initial energy density for a wave vector under any spectrum.
///
/// `Phillips` is inherently directional and evaluated from `k_vec`; the
/// frequency-space `JONSWAP` / `PM` densities are converted through the
/// dispersion relation and given the same `|k_hat . w_hat|^{2n}` directional
/// weighting so all three kinds share one call site. Always non-negative.
#[must_use]
pub fn energy_density(k_vec: Vec2, params: SpectrumParams) -> f32 {
    match params.kind {
        SpectrumKind::Phillips => phillips(k_vec, params),
        SpectrumKind::Jonswap | SpectrumKind::PiersonMoskowitz => {
            let k_sq = k_vec.length_squared();
            if k_sq < super::EPS_LEN_SQ {
                return 0.0;
            }
            let k = k_sq.sqrt();
            let omega = dispersion(k);
            let omni = match params.kind {
                SpectrumKind::Jonswap => jonswap(omega, params),
                _ => pierson_moskowitz(omega, params),
            };
            let wind_dir = params.wind.normalize_or_zero();
            let k_dir = k_vec.scale(1.0 / k);
            let mut align = k_dir.dot(wind_dir);
            if align < 0.0 {
                align = -align;
            }
            let mut directional = 1.0;
            let mut i = 0;
            while i < params.directional_exponent {
                directional *= align * align;
                i += 1;
            }
            omni * directional * params.amplitude
        }
    }
}

/// Advances a complex spectral amplitude to time `t`.
///
/// `h(k, t) = h0 * e^{i*omega*t} + conj(h0_neg) * e^{-i*omega*t}`, where `h0` is
/// the amplitude at `+k`, `h0_neg` the amplitude at `-k`, and `omega` the
/// dispersion frequency. This Hermitian pairing is what keeps the inverse `FFT`
/// height field real-valued. The result is exactly periodic with period
/// `2*PI/omega` (up to the phase approximation).
#[must_use]
pub fn advance_amplitude(h0: Complex, h0_neg: Complex, omega: f32, t: f32) -> Complex {
    let theta = omega * t;
    let forward = h0.mul_phasor(theta);
    let backward = h0_neg.conj().mul_phasor(-theta);
    forward.add(backward)
}

/// `true` when the surface Jacobian indicates a folding crest (whitecap).
///
/// The choppy-displacement Jacobian `J` measures horizontal area compression;
/// `J` at or below `threshold` (commonly a small positive value, with `J < 0`
/// meaning the surface has folded over itself) marks a breaking crest that
/// seeds foam. Pure comparison, no float equality.
#[must_use]
pub fn is_wave_folding(jacobian: f32, threshold: f32) -> bool {
    jacobian <= threshold
}

/// The choppy-surface Jacobian from the displacement partial derivatives.
///
/// `J = (1 + dDx_dx)(1 + dDz_dz) - dDx_dz * dDz_dx`, the determinant of the
/// horizontal displacement gradient plus identity. Values below `1` indicate
/// compression toward a crest; values at or below `0` indicate a fold.
#[must_use]
pub fn surface_jacobian(d_dx_dx: f32, d_dz_dz: f32, d_dx_dz: f32, d_dz_dx: f32) -> f32 {
    (1.0 + d_dx_dx) * (1.0 + d_dz_dz) - d_dx_dz * d_dz_dx
}

#[cfg(test)]
mod tests {
    use super::*;

    const PHILLIPS: SpectrumParams = SpectrumParams {
        kind: SpectrumKind::Phillips,
        wind: Vec2 { x: 12.0, y: 0.0 },
        amplitude: 0.5,
        peak_enhancement: 3.3,
        min_wavelength: 0.5,
        directional_exponent: 1,
    };

    #[test]
    fn dispersion_is_monotonic_and_zero_at_origin() {
        assert_eq!(dispersion(0.0), 0.0);
        let mut prev = dispersion(0.01);
        let mut k = 0.02;
        while k <= 10.0 {
            let cur = dispersion(k);
            assert!(cur > prev);
            prev = cur;
            k += 0.05;
        }
        // sqrt(g * 1) = sqrt(9.81).
        assert!((dispersion(1.0) - GRAVITY.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn phillips_is_nonnegative_everywhere() {
        let mut ky = -5.0;
        while ky <= 5.0 {
            let mut kx = -5.0;
            while kx <= 5.0 {
                assert!(phillips(Vec2::new(kx, ky), PHILLIPS) >= 0.0);
                kx += 0.5;
            }
            ky += 0.5;
        }
        // Zero wave vector carries no energy.
        assert_eq!(phillips(Vec2::ZERO, PHILLIPS), 0.0);
    }

    #[test]
    fn energy_grows_with_wind_speed() {
        // A representative wave vector aligned with the wind.
        let k_vec = Vec2::new(0.3, 0.0);
        let mut prev = 0.0;
        let mut speed = 4.0;
        while speed <= 24.0 {
            let params = SpectrumParams {
                wind: Vec2::new(speed, 0.0),
                ..PHILLIPS
            };
            let e = phillips(k_vec, params);
            assert!(e >= prev);
            prev = e;
            speed += 2.0;
        }
        assert!(prev > 0.0);
    }

    #[test]
    fn directional_term_favors_downwind() {
        // Downwind wave carries more energy than a cross-wind wave of equal k.
        let downwind = phillips(Vec2::new(0.4, 0.0), PHILLIPS);
        let crosswind = phillips(Vec2::new(0.0, 0.4), PHILLIPS);
        assert!(downwind > crosswind);
    }

    #[test]
    fn pm_and_jonswap_are_nonnegative_and_peaked() {
        let params = SpectrumParams {
            kind: SpectrumKind::Jonswap,
            ..PHILLIPS
        };
        let mut omega = 0.1;
        let mut peak_value = 0.0_f32;
        while omega <= 6.0 {
            let pm = pierson_moskowitz(omega, params);
            let js = jonswap(omega, params);
            assert!(pm >= 0.0);
            assert!(js >= 0.0);
            // JONSWAP peak-enhances above PM (gamma > 1).
            assert!(js >= pm - 1e-6);
            peak_value = peak_value.max(js);
            omega += 0.1;
        }
        assert!(peak_value > 0.0);
    }

    #[test]
    fn amplitude_is_periodic_in_time() {
        let h0 = Complex::new(0.7, -0.2);
        let h0_neg = Complex::new(0.1, 0.5);
        let omega = dispersion(0.8);
        let period = super::super::TWO_PI / omega;
        let a = advance_amplitude(h0, h0_neg, omega, 0.3);
        let b = advance_amplitude(h0, h0_neg, omega, 0.3 + period);
        assert!((a.re - b.re).abs() < 5e-3);
        assert!((a.im - b.im).abs() < 5e-3);
    }

    #[test]
    fn amplitude_evolution_stays_bounded() {
        let h0 = Complex::new(1.0, 0.0);
        let h0_neg = Complex::new(1.0, 0.0);
        let omega = dispersion(1.0);
        let mut t = 0.0;
        while t <= 20.0 {
            let h = advance_amplitude(h0, h0_neg, omega, t);
            // Sum of two unit-scaled phasors: bounded by their magnitudes.
            assert!(h.norm_squared().sqrt() <= 2.0 + 1e-2);
            t += 0.5;
        }
    }

    #[test]
    fn jacobian_fold_classification_is_deterministic() {
        // Flat surface: J = 1, not folding.
        assert!(!is_wave_folding(surface_jacobian(0.0, 0.0, 0.0, 0.0), 0.0));
        // Strong compression along one axis drives J negative: folding.
        // (1 - 2.5)(1 - 0.5) - 0.2 * 0.2 = -0.75 - 0.04 = -0.79.
        let j = surface_jacobian(-2.5, -0.5, 0.2, 0.2);
        assert!(j < 0.0);
        assert!(is_wave_folding(j, 0.0));
        // Threshold classification is a plain comparison.
        assert!(is_wave_folding(0.3, 0.5));
        assert!(!is_wave_folding(0.6, 0.5));
    }

    #[test]
    fn complex_algebra_is_exact() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, -1.0);
        assert_eq!(a.add(b), Complex::new(4.0, 1.0));
        assert_eq!(a.mul(b), Complex::new(5.0, 5.0));
        assert_eq!(a.conj(), Complex::new(1.0, -2.0));
        assert_eq!(a.norm_squared(), 5.0);
    }
}
