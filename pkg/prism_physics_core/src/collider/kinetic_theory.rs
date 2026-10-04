//! Dense-gas kinetic-theory descriptors for granular media.
//!
//! Treating an agitated granular assembly as a gas of inelastic hard spheres
//! lets Enskog kinetic theory predict microscopic transport scales from three
//! coarse fields: the number density `n`, the grain diameter `d`, the solid
//! fraction `φ`, and the granular temperature `T` (velocity-variance units, as
//! produced by [`crate::collider::granular_temperature`]).
//!
//! The excluded-volume correction enters through the pair-correlation function
//! at contact. We use the Carnahan–Starling form for spheres,
//!
//! ```text
//! g0(φ) = (2 − φ) / (2 (1 − φ)³)
//! ```
//!
//! which rises from `1` in the dilute limit toward large values as `φ → 1`.
//! The Enskog mean free path and per-particle collision frequency are then
//!
//! ```text
//! ℓ = 1 / (√2 · π · n · d² · g0),      ω = 4 · n · d² · g0 · √(π T)
//! ```
//!
//! so collisions become more frequent and the mean free path shrinks as the
//! packing densifies (`g0 ↑`) or the agitation grows (`√T ↑`). This module is
//! a pure constitutive correlation and does not couple to the simulation step.

/// Enskog kinetic-theory descriptors for a granular gas state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GranularKineticState {
    number_density: f32,
    grain_diameter: f32,
    solid_fraction: f32,
    granular_temperature: f32,
    pair_correlation: f32,
    mean_free_path: f32,
    collision_frequency: f32,
}

impl GranularKineticState {
    /// Builds the kinetic state from the coarse fields.
    ///
    /// * `number_density` — `n`, grains per unit volume (`> 0`).
    /// * `grain_diameter` — `d` (`> 0`).
    /// * `solid_fraction` — `φ ∈ [0, 1)` (packing fraction).
    /// * `granular_temperature` — `T ≥ 0`, velocity-variance units.
    ///
    /// Returns `None` for non-finite inputs, non-positive `n`/`d`, `φ` outside
    /// `[0, 1)`, or negative `T`.
    pub fn new(
        number_density: f32,
        grain_diameter: f32,
        solid_fraction: f32,
        granular_temperature: f32,
    ) -> Option<Self> {
        if !number_density.is_finite() || number_density <= 0.0 {
            return None;
        }
        if !grain_diameter.is_finite() || grain_diameter <= 0.0 {
            return None;
        }
        if !solid_fraction.is_finite() || !(0.0..1.0).contains(&solid_fraction) {
            return None;
        }
        if !granular_temperature.is_finite() || granular_temperature < 0.0 {
            return None;
        }

        // Carnahan–Starling pair correlation at contact (f64 for the cube).
        let phi = solid_fraction as f64;
        let one_minus = 1.0 - phi;
        let g0 = (2.0 - phi) / (2.0 * one_minus * one_minus * one_minus);
        if !g0.is_finite() || g0 <= 0.0 {
            return None;
        }
        let pair_correlation = g0 as f32;

        let d = grain_diameter;
        let d2 = d * d;
        let sqrt2 = std::f32::consts::SQRT_2;
        let pi = std::f32::consts::PI;

        // Mean free path ℓ = 1 / (√2 π n d² g0).
        let denom = sqrt2 * pi * number_density * d2 * pair_correlation;
        if !denom.is_finite() || denom <= 0.0 {
            return None;
        }
        let mean_free_path = 1.0 / denom;

        // Collision frequency ω = 4 n d² g0 √(π T).
        let thermal = (pi * granular_temperature).max(0.0).sqrt();
        let collision_frequency = 4.0 * number_density * d2 * pair_correlation * thermal;
        if !mean_free_path.is_finite() || !collision_frequency.is_finite() {
            return None;
        }

        Some(Self {
            number_density,
            grain_diameter,
            solid_fraction,
            granular_temperature,
            pair_correlation,
            mean_free_path,
            collision_frequency,
        })
    }

    /// Number density `n`.
    pub fn number_density(&self) -> f32 {
        self.number_density
    }

    /// Grain diameter `d`.
    pub fn grain_diameter(&self) -> f32 {
        self.grain_diameter
    }

    /// Solid (packing) fraction `φ`.
    pub fn solid_fraction(&self) -> f32 {
        self.solid_fraction
    }

    /// Granular temperature `T`.
    pub fn granular_temperature(&self) -> f32 {
        self.granular_temperature
    }

    /// Carnahan–Starling pair-correlation value at contact `g0(φ)`.
    pub fn pair_correlation(&self) -> f32 {
        self.pair_correlation
    }

    /// Enskog mean free path `ℓ = 1/(√2 π n d² g0)`.
    pub fn mean_free_path(&self) -> f32 {
        self.mean_free_path
    }

    /// Per-particle collision frequency `ω = 4 n d² g0 √(π T)`.
    pub fn collision_frequency(&self) -> f32 {
        self.collision_frequency
    }

    /// Mean time between collisions `1/ω`. `None` when the gas is frozen
    /// (`T = 0 ⇒ ω = 0`).
    pub fn mean_collision_time(&self) -> Option<f32> {
        if self.collision_frequency <= 0.0 {
            return None;
        }
        Some(1.0 / self.collision_frequency)
    }

    /// Characteristic thermal velocity scale `√T`.
    pub fn thermal_velocity_scale(&self) -> f32 {
        self.granular_temperature.max(0.0).sqrt()
    }

    /// RMS velocity fluctuation `√(3T)` (full three-dimensional speed).
    pub fn rms_fluctuation_speed(&self) -> f32 {
        (3.0 * self.granular_temperature).max(0.0).sqrt()
    }

    /// Knudsen number `Kn = ℓ / L` for a system length `L > 0`.
    ///
    /// `Kn ≳ 1` flags a rarefied regime where continuum closures break down.
    pub fn knudsen_number(&self, system_length: f32) -> Option<f32> {
        if !system_length.is_finite() || system_length <= 0.0 {
            return None;
        }
        Some(self.mean_free_path / system_length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_input() {
        assert!(GranularKineticState::new(0.0, 1.0, 0.3, 1.0).is_none());
        assert!(GranularKineticState::new(1.0, -1.0, 0.3, 1.0).is_none());
        assert!(GranularKineticState::new(1.0, 1.0, 1.0, 1.0).is_none()); // φ = 1
        assert!(GranularKineticState::new(1.0, 1.0, -0.1, 1.0).is_none());
        assert!(GranularKineticState::new(1.0, 1.0, 0.3, -1.0).is_none());
        assert!(GranularKineticState::new(f32::NAN, 1.0, 0.3, 1.0).is_none());
    }

    #[test]
    fn dilute_pair_correlation_approaches_one() {
        // g0(φ→0) = (2-0)/(2·1) = 1.
        let s = GranularKineticState::new(1.0, 1.0, 1e-6, 1.0).unwrap();
        assert!((s.pair_correlation() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn carnahan_starling_known_value() {
        // φ = 0.5 → g0 = (2-0.5)/(2·0.125) = 1.5/0.25 = 6.
        let s = GranularKineticState::new(1.0, 1.0, 0.5, 1.0).unwrap();
        assert!((s.pair_correlation() - 6.0).abs() < 1e-4);
        // g0 grows with φ.
        let dense = GranularKineticState::new(1.0, 1.0, 0.6, 1.0).unwrap();
        assert!(dense.pair_correlation() > s.pair_correlation());
    }

    #[test]
    fn mean_free_path_known_formula() {
        // Dilute (φ→0, g0≈1): ℓ = 1/(√2 π n d²).
        let n = 2.0;
        let d = 0.5;
        let s = GranularKineticState::new(n, d, 1e-6, 1.0).unwrap();
        let expected = 1.0 / (std::f32::consts::SQRT_2 * std::f32::consts::PI * n * d * d);
        assert!((s.mean_free_path() - expected).abs() / expected < 1e-3);
    }

    #[test]
    fn mean_free_path_shrinks_with_density_and_packing() {
        let sparse = GranularKineticState::new(1.0, 1.0, 0.1, 1.0).unwrap();
        let denser_n = GranularKineticState::new(2.0, 1.0, 0.1, 1.0).unwrap();
        assert!(denser_n.mean_free_path() < sparse.mean_free_path());
        let denser_phi = GranularKineticState::new(1.0, 1.0, 0.4, 1.0).unwrap();
        assert!(denser_phi.mean_free_path() < sparse.mean_free_path());
    }

    #[test]
    fn collision_frequency_scaling() {
        // ω ∝ √T: quadrupling T doubles ω.
        let a = GranularKineticState::new(1.0, 1.0, 0.3, 1.0).unwrap();
        let b = GranularKineticState::new(1.0, 1.0, 0.3, 4.0).unwrap();
        assert!((b.collision_frequency() / a.collision_frequency() - 2.0).abs() < 1e-3);
        // ω ∝ d²: doubling d quadruples ω.
        let c = GranularKineticState::new(1.0, 2.0, 0.3, 1.0).unwrap();
        assert!((c.collision_frequency() / a.collision_frequency() - 4.0).abs() < 1e-3);
    }

    #[test]
    fn frozen_gas_has_no_collisions() {
        // T = 0 → ω = 0, mean collision time undefined.
        let s = GranularKineticState::new(1.0, 1.0, 0.3, 0.0).unwrap();
        assert_eq!(s.collision_frequency(), 0.0);
        assert!(s.mean_collision_time().is_none());
        assert_eq!(s.thermal_velocity_scale(), 0.0);
        // Mean free path is still finite and positive.
        assert!(s.mean_free_path() > 0.0);
    }

    #[test]
    fn collision_time_is_reciprocal_frequency() {
        let s = GranularKineticState::new(1.0, 1.0, 0.3, 2.0).unwrap();
        let t = s.mean_collision_time().unwrap();
        assert!((t * s.collision_frequency() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn thermal_speeds_and_knudsen() {
        let s = GranularKineticState::new(1.0, 1.0, 0.3, 3.0).unwrap();
        // √T and √(3T).
        assert!((s.thermal_velocity_scale() - 3.0_f32.sqrt()).abs() < 1e-5);
        assert!((s.rms_fluctuation_speed() - 3.0_f32).abs() < 1e-5);
        // Kn = ℓ / L.
        let kn = s.knudsen_number(2.0).unwrap();
        assert!((kn - s.mean_free_path() / 2.0).abs() < 1e-6);
        assert!(s.knudsen_number(0.0).is_none());
    }
}
