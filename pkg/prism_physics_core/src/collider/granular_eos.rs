//! Equation of state for a dense, inelastic granular gas.
//!
//! Kinetic theory of granular flow (Lun et al. 1984; Jenkins–Savage) models the
//! granular pressure as the sum of a *kinetic* (streaming) part and a
//! *collisional* part. For smooth inelastic spheres of restitution `e` at bulk
//! density `ρ = ρ_s · φ`, granular temperature `T`, solid fraction `φ`, and
//! contact radial-distribution value `g0`:
//!
//! ```text
//! p = ρ T [ 1 + 2 (1 + e) φ g0 ]
//! ```
//!
//! The bracket is the compressibility factor `Z = p / (ρ T)`. The leading `1`
//! is the ideal-gas (kinetic) term; `2 (1 + e) φ g0` is the collisional
//! enhancement, which grows with packing fraction and vanishes in the dilute
//! limit `φ → 0`. The contact value `g0` is supplied by the caller (for
//! example the Carnahan–Starling form exposed by the kinetic-theory module),
//! keeping this a pure analytic constitutive relation with no coupling to the
//! simulation pipeline.
//!
//! No Unreal Engine source or derived code.

/// Granular-gas equation of state parameterised by the coefficient of
/// restitution.
///
/// Build with [`GranularEquationOfState::new`], then evaluate the pressure or
/// its components from a bulk-density / temperature / packing state.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GranularEquationOfState {
    /// Coefficient of normal restitution `e`, in `[0, 1]`.
    restitution: f32,
}

impl GranularEquationOfState {
    /// Builds the equation of state from a coefficient of restitution.
    ///
    /// Returns `None` unless `e` is finite and lies in `[0, 1]`.
    #[must_use]
    pub fn new(restitution: f32) -> Option<Self> {
        if !restitution.is_finite() || !(0.0..=1.0).contains(&restitution) {
            return None;
        }
        Some(Self { restitution })
    }

    /// Coefficient of restitution `e`.
    #[must_use]
    pub fn restitution(&self) -> f32 {
        self.restitution
    }

    /// Validates the packing arguments shared by the pressure accessors.
    ///
    /// Requires finite inputs, `φ ∈ [0, 1)`, and `g0 >= 1`.
    fn valid_packing(solid_fraction: f32, g0: f32) -> bool {
        solid_fraction.is_finite()
            && g0.is_finite()
            && (0.0..1.0).contains(&solid_fraction)
            && g0 >= 1.0
    }

    /// Compressibility factor `Z = 1 + 2 (1 + e) φ g0`.
    ///
    /// Returns `None` unless `φ ∈ [0, 1)` and `g0 >= 1` are finite.
    #[must_use]
    pub fn compressibility_factor(&self, solid_fraction: f32, g0: f32) -> Option<f32> {
        if !Self::valid_packing(solid_fraction, g0) {
            return None;
        }
        Some(1.0 + 2.0 * (1.0 + self.restitution) * solid_fraction * g0)
    }

    /// The collisional enhancement term `2 (1 + e) φ g0` (i.e. `Z - 1`).
    ///
    /// Returns `None` unless `φ ∈ [0, 1)` and `g0 >= 1` are finite.
    #[must_use]
    pub fn collisional_enhancement(&self, solid_fraction: f32, g0: f32) -> Option<f32> {
        if !Self::valid_packing(solid_fraction, g0) {
            return None;
        }
        Some(2.0 * (1.0 + self.restitution) * solid_fraction * g0)
    }

    /// Kinetic (ideal-gas) pressure `p_k = ρ T`.
    ///
    /// Returns `None` unless `ρ > 0` and `T >= 0` are finite.
    #[must_use]
    pub fn kinetic_pressure(&self, bulk_density: f32, temperature: f32) -> Option<f32> {
        if !bulk_density.is_finite() || !temperature.is_finite() {
            return None;
        }
        if bulk_density <= 0.0 || temperature < 0.0 {
            return None;
        }
        Some(bulk_density * temperature)
    }

    /// Collisional pressure `p_c = ρ T · 2 (1 + e) φ g0`.
    ///
    /// Returns `None` unless all state arguments are valid.
    #[must_use]
    pub fn collisional_pressure(
        &self,
        bulk_density: f32,
        temperature: f32,
        solid_fraction: f32,
        g0: f32,
    ) -> Option<f32> {
        let kinetic = self.kinetic_pressure(bulk_density, temperature)?;
        let enhancement = self.collisional_enhancement(solid_fraction, g0)?;
        Some(kinetic * enhancement)
    }

    /// Total granular pressure `p = ρ T [1 + 2 (1 + e) φ g0]`.
    ///
    /// Returns `None` unless `ρ > 0`, `T >= 0`, `φ ∈ [0, 1)`, and `g0 >= 1` are
    /// finite.
    #[must_use]
    pub fn pressure(
        &self,
        bulk_density: f32,
        temperature: f32,
        solid_fraction: f32,
        g0: f32,
    ) -> Option<f32> {
        let kinetic = self.kinetic_pressure(bulk_density, temperature)?;
        let z = self.compressibility_factor(solid_fraction, g0)?;
        Some(kinetic * z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0e-4;

    #[test]
    fn new_rejects_invalid_restitution() {
        assert!(GranularEquationOfState::new(-0.1).is_none());
        assert!(GranularEquationOfState::new(1.1).is_none());
        assert!(GranularEquationOfState::new(f32::NAN).is_none());
        assert!(GranularEquationOfState::new(0.0).is_some());
        assert!(GranularEquationOfState::new(1.0).is_some());
    }

    #[test]
    fn dilute_limit_is_ideal_gas() {
        let eos = GranularEquationOfState::new(0.9).expect("valid");
        // phi -> 0: Z -> 1, p -> rho T.
        let z = eos.compressibility_factor(0.0, 1.0).expect("valid");
        assert!((z - 1.0).abs() < TOL);
        let p = eos.pressure(10.0, 2.0, 0.0, 1.0).expect("valid");
        assert!((p - 20.0).abs() < TOL);
    }

    #[test]
    fn closed_form_matches_hand_computation() {
        // e=0.8, phi=0.3, g0=2.5 => Z = 1 + 2*1.8*0.3*2.5 = 3.7.
        let eos = GranularEquationOfState::new(0.8).expect("valid");
        let z = eos.compressibility_factor(0.3, 2.5).expect("valid");
        assert!((z - 3.7).abs() < TOL);
        // rho=10, T=2 => p = 10*2*3.7 = 74.
        let p = eos.pressure(10.0, 2.0, 0.3, 2.5).expect("valid");
        assert!((p - 74.0).abs() < 1.0e-3);
    }

    #[test]
    fn kinetic_plus_collisional_equals_total() {
        let eos = GranularEquationOfState::new(0.5).expect("valid");
        let (rho, t, phi, g0) = (12.0, 3.0, 0.4, 2.0);
        let total = eos.pressure(rho, t, phi, g0).expect("valid");
        let kinetic = eos.kinetic_pressure(rho, t).expect("valid");
        let collisional = eos.collisional_pressure(rho, t, phi, g0).expect("valid");
        assert!((kinetic + collisional - total).abs() < 1.0e-3);
    }

    #[test]
    fn pressure_grows_with_restitution_and_fraction() {
        let low = GranularEquationOfState::new(0.2).expect("valid");
        let high = GranularEquationOfState::new(0.9).expect("valid");
        let p_low = low.pressure(10.0, 2.0, 0.3, 2.0).expect("valid");
        let p_high = high.pressure(10.0, 2.0, 0.3, 2.0).expect("valid");
        assert!(p_high > p_low);

        let eos = GranularEquationOfState::new(0.6).expect("valid");
        let p_sparse = eos.pressure(10.0, 2.0, 0.1, 2.0).expect("valid");
        let p_dense = eos.pressure(10.0, 2.0, 0.5, 2.0).expect("valid");
        assert!(p_dense > p_sparse);
    }

    #[test]
    fn pressure_is_linear_in_temperature() {
        let eos = GranularEquationOfState::new(0.7).expect("valid");
        let p1 = eos.pressure(10.0, 1.0, 0.3, 2.0).expect("valid");
        let p2 = eos.pressure(10.0, 2.0, 0.3, 2.0).expect("valid");
        assert!((p2 - 2.0 * p1).abs() < 1.0e-3);
        // Zero temperature => zero pressure.
        assert!(eos.pressure(10.0, 0.0, 0.3, 2.0).expect("valid").abs() < TOL);
    }

    #[test]
    fn rejects_out_of_range_state() {
        let eos = GranularEquationOfState::new(0.8).expect("valid");
        assert!(eos.pressure(0.0, 2.0, 0.3, 2.0).is_none());
        assert!(eos.pressure(10.0, -1.0, 0.3, 2.0).is_none());
        assert!(eos.pressure(10.0, 2.0, 1.0, 2.0).is_none());
        assert!(eos.pressure(10.0, 2.0, 0.3, 0.5).is_none());
        assert!(eos.compressibility_factor(f32::NAN, 2.0).is_none());
    }
}
