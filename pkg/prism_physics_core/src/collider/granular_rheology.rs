//! Local granular rheology: the inertial number and the `μ(I)` friction law.
//!
//! Dense granular flows are well described by the dimensionless *inertial
//! number*
//!
//! ```text
//! I = γ̇ d / √(P / ρ_s)
//! ```
//!
//! comparing the macroscopic shear time `1/γ̇` with the microscopic
//! rearrangement time `d √(ρ_s / P)` (grain diameter `d`, grain material
//! density `ρ_s`, confining pressure `P`, shear rate `γ̇`). The GDR-MiDi /
//! Jop–Forterre–Pouliquen phenomenology then sets the effective friction
//!
//! ```text
//! μ(I) = μ_s + (μ_2 − μ_s) / (I_0 / I + 1)
//! ```
//!
//! which interpolates from the static (quasi-static) friction `μ_s` at
//! `I → 0` toward the dynamic limit `μ_2` at large `I`, with crossover scale
//! `I_0`. The shear stress is `τ = μ(I) · P` and the effective viscosity is
//! `η = τ / γ̇`. A companion linear dilatancy law `φ(I) = φ_max − a·I` captures
//! the loss of packing fraction with increasing agitation.
//!
//! This module is a pure constitutive correlation and does not couple to the
//! simulation step.

/// Parameters and evaluation of the `μ(I)` granular friction law.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GranularRheology {
    mu_static: f32,
    mu_dynamic: f32,
    i_ref: f32,
    grain_diameter: f32,
    grain_density: f32,
}

impl GranularRheology {
    /// Builds the rheology from its parameters.
    ///
    /// * `mu_static` — `μ_s`, quasi-static friction (`> 0`).
    /// * `mu_dynamic` — `μ_2`, high-`I` friction limit (`> μ_s`).
    /// * `i_ref` — `I_0`, crossover inertial number (`> 0`).
    /// * `grain_diameter` — `d` (`> 0`).
    /// * `grain_density` — `ρ_s`, grain material density (`> 0`).
    ///
    /// Returns `None` for non-finite inputs, non-positive magnitudes, or when
    /// `mu_dynamic ≤ mu_static`.
    pub fn new(
        mu_static: f32,
        mu_dynamic: f32,
        i_ref: f32,
        grain_diameter: f32,
        grain_density: f32,
    ) -> Option<Self> {
        for v in [mu_static, mu_dynamic, i_ref, grain_diameter, grain_density] {
            if !v.is_finite() || v <= 0.0 {
                return None;
            }
        }
        if mu_dynamic <= mu_static {
            return None;
        }
        Some(Self {
            mu_static,
            mu_dynamic,
            i_ref,
            grain_diameter,
            grain_density,
        })
    }

    /// Quasi-static friction `μ_s`.
    pub fn mu_static(&self) -> f32 {
        self.mu_static
    }

    /// Dynamic friction limit `μ_2`.
    pub fn mu_dynamic(&self) -> f32 {
        self.mu_dynamic
    }

    /// Crossover inertial number `I_0`.
    pub fn i_ref(&self) -> f32 {
        self.i_ref
    }

    /// Grain diameter `d`.
    pub fn grain_diameter(&self) -> f32 {
        self.grain_diameter
    }

    /// Grain material density `ρ_s`.
    pub fn grain_density(&self) -> f32 {
        self.grain_density
    }

    /// Inertial number `I = γ̇ d / √(P / ρ_s)` for `shear_rate ≥ 0` and
    /// `pressure > 0`. Returns `None` on non-finite or non-positive pressure,
    /// or negative/non-finite shear rate.
    pub fn inertial_number(&self, shear_rate: f32, pressure: f32) -> Option<f32> {
        if !shear_rate.is_finite() || shear_rate < 0.0 {
            return None;
        }
        if !pressure.is_finite() || pressure <= 0.0 {
            return None;
        }
        let micro_time_sq = pressure / self.grain_density; // (P/ρ)
        let denom = micro_time_sq.sqrt();
        if denom <= 0.0 {
            return None;
        }
        Some(shear_rate * self.grain_diameter / denom)
    }

    /// Effective friction `μ(I) = μ_s + (μ_2 − μ_s)/(I_0/I + 1)`.
    ///
    /// Clamps negative `I` to `0` (static limit). At `I = 0` returns exactly
    /// `μ_s`; as `I → ∞` it approaches `μ_2`.
    pub fn friction(&self, inertial_number: f32) -> f32 {
        if !inertial_number.is_finite() {
            // Treat +∞ as the dynamic limit; anything else degrades to static.
            return if inertial_number.is_sign_positive() && inertial_number.is_infinite() {
                self.mu_dynamic
            } else {
                self.mu_static
            };
        }
        if inertial_number <= 0.0 {
            return self.mu_static;
        }
        let extra = (self.mu_dynamic - self.mu_static) / (self.i_ref / inertial_number + 1.0);
        self.mu_static + extra
    }

    /// Shear stress `τ = μ(I) · P` for the given shear rate and pressure.
    pub fn shear_stress(&self, shear_rate: f32, pressure: f32) -> Option<f32> {
        let i = self.inertial_number(shear_rate, pressure)?;
        Some(self.friction(i) * pressure)
    }

    /// Effective viscosity `η = τ / γ̇` for `shear_rate > 0`.
    pub fn effective_viscosity(&self, shear_rate: f32, pressure: f32) -> Option<f32> {
        if !shear_rate.is_finite() || shear_rate <= 0.0 {
            return None;
        }
        let tau = self.shear_stress(shear_rate, pressure)?;
        Some(tau / shear_rate)
    }
}

/// Linear dilatancy law `φ(I) = φ_max − a·I`.
///
/// Captures the decrease of packing fraction from its dense static value
/// `φ_max` as the inertial number grows, with slope `a` (the dilatancy
/// coefficient). The result is clamped to `[0, φ_max]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DilatancyLaw {
    phi_max: f32,
    slope: f32,
}

impl DilatancyLaw {
    /// Builds the law from the dense packing fraction `phi_max ∈ (0, 1]` and
    /// dilatancy slope `slope ≥ 0`.
    pub fn new(phi_max: f32, slope: f32) -> Option<Self> {
        if !phi_max.is_finite() || phi_max <= 0.0 || phi_max > 1.0 {
            return None;
        }
        if !slope.is_finite() || slope < 0.0 {
            return None;
        }
        Some(Self { phi_max, slope })
    }

    /// Dense (static) packing fraction `φ_max`.
    pub fn phi_max(&self) -> f32 {
        self.phi_max
    }

    /// Dilatancy slope `a`.
    pub fn slope(&self) -> f32 {
        self.slope
    }

    /// Packing fraction `φ(I) = φ_max − a·I`, clamped to `[0, φ_max]`.
    /// Negative `I` clamps to the dense limit `φ_max`.
    pub fn volume_fraction(&self, inertial_number: f32) -> f32 {
        if !inertial_number.is_finite() || inertial_number <= 0.0 {
            return self.phi_max;
        }
        (self.phi_max - self.slope * inertial_number).clamp(0.0, self.phi_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> GranularRheology {
        // Typical glass-bead values: μ_s ≈ 0.38, μ_2 ≈ 0.64, I_0 ≈ 0.279.
        GranularRheology::new(0.38, 0.64, 0.279, 1.0e-3, 2.5e3).unwrap()
    }

    #[test]
    fn rejects_bad_input() {
        assert!(GranularRheology::new(0.0, 0.6, 0.3, 1e-3, 2500.0).is_none());
        assert!(GranularRheology::new(0.4, 0.3, 0.3, 1e-3, 2500.0).is_none()); // μ2 ≤ μs
        assert!(GranularRheology::new(0.4, 0.6, 0.0, 1e-3, 2500.0).is_none());
        assert!(GranularRheology::new(0.4, 0.6, 0.3, -1.0, 2500.0).is_none());
        assert!(GranularRheology::new(0.4, 0.6, 0.3, 1e-3, f32::NAN).is_none());
    }

    #[test]
    fn inertial_number_formula() {
        let r = sample();
        // γ̇ = 100, d = 1e-3, P = 1000, ρ = 2500 → √(P/ρ) = √0.4 ≈ 0.6324555.
        // I = 100·1e-3 / 0.6324555 ≈ 0.158114.
        let i = r.inertial_number(100.0, 1000.0).unwrap();
        assert!((i - 0.158_114).abs() < 1e-4, "I = {i}");
        // Zero shear rate → zero inertial number.
        assert_eq!(r.inertial_number(0.0, 1000.0).unwrap(), 0.0);
        // Bad pressure / shear rate.
        assert!(r.inertial_number(100.0, 0.0).is_none());
        assert!(r.inertial_number(-1.0, 1000.0).is_none());
    }

    #[test]
    fn friction_limits() {
        let r = sample();
        // I = 0 → μ_s exactly.
        assert!((r.friction(0.0) - r.mu_static()).abs() < 1e-7);
        assert!((r.friction(-5.0) - r.mu_static()).abs() < 1e-7);
        // I → ∞ → μ_2.
        assert!((r.friction(f32::INFINITY) - r.mu_dynamic()).abs() < 1e-7);
        // Large but finite I approaches μ_2 from below.
        let big = r.friction(1.0e6);
        assert!(big < r.mu_dynamic() && big > r.mu_dynamic() - 1e-3);
    }

    #[test]
    fn friction_at_crossover_is_midpoint() {
        let r = sample();
        // At I = I_0, μ = μ_s + (μ_2-μ_s)/2.
        let mid = r.mu_static() + 0.5 * (r.mu_dynamic() - r.mu_static());
        assert!((r.friction(r.i_ref()) - mid).abs() < 1e-6);
    }

    #[test]
    fn friction_is_monotone_increasing() {
        let r = sample();
        let mut prev = r.friction(0.0);
        for k in 1..=200 {
            let i = k as f32 * 0.01;
            let mu = r.friction(i);
            assert!(mu >= prev - 1e-7, "monotone at I={i}");
            assert!(mu <= r.mu_dynamic() + 1e-6);
            prev = mu;
        }
    }

    #[test]
    fn shear_stress_and_viscosity() {
        let r = sample();
        let gamma = 100.0;
        let p = 1000.0;
        let i = r.inertial_number(gamma, p).unwrap();
        let tau = r.shear_stress(gamma, p).unwrap();
        assert!((tau - r.friction(i) * p).abs() < 1e-3);
        let eta = r.effective_viscosity(gamma, p).unwrap();
        assert!((eta - tau / gamma).abs() < 1e-5);
        assert!(r.effective_viscosity(0.0, p).is_none());
    }

    #[test]
    fn dilatancy_law_behaviour() {
        let d = DilatancyLaw::new(0.62, 0.2).unwrap();
        // I = 0 → φ_max.
        assert!((d.volume_fraction(0.0) - 0.62).abs() < 1e-7);
        assert!((d.volume_fraction(-1.0) - 0.62).abs() < 1e-7);
        // I = 0.5 → 0.62 - 0.1 = 0.52.
        assert!((d.volume_fraction(0.5) - 0.52).abs() < 1e-6);
        // Large I clamps at 0.
        assert_eq!(d.volume_fraction(1000.0), 0.0);
        // Rejects bad params.
        assert!(DilatancyLaw::new(0.0, 0.1).is_none());
        assert!(DilatancyLaw::new(1.1, 0.1).is_none());
        assert!(DilatancyLaw::new(0.6, -0.1).is_none());
    }
}
