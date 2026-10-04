//! Haff's law for the homogeneous cooling of an inelastic granular gas.
//!
//! An unforced granular gas of inelastic hard spheres loses kinetic energy at
//! every collision, so its granular temperature `T` decays monotonically. In
//! the homogeneous cooling state the fractional loss rate is set by the
//! collision frequency `ω` and the normal restitution `e`:
//!
//! ```text
//! ζ0 = (1 − e²) · ω0 / 3            (initial cooling rate, 1/time)
//! ```
//!
//! Because `ω ∝ √T`, integrating `dT/dt = −ζ0 (T/T0)^{1/2} · T` yields Haff's
//! algebraic decay
//!
//! ```text
//! T(t) = T0 / (1 + t/τ)²,     τ = 2 / ζ0,
//! ```
//!
//! while the collision frequency relaxes as `ω(t) = ω0 / (1 + t/τ)`. Perfectly
//! elastic grains (`e = 1`) do not cool and are rejected here. This module is a
//! pure analytic correlation; it composes
//! [`crate::collider::kinetic_theory::GranularKineticState`] but does not couple
//! to the simulation step.

use crate::collider::kinetic_theory::GranularKineticState;

/// Haff homogeneous-cooling descriptor for an inelastic granular gas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HaffCooling {
    initial_temperature: f32,
    initial_collision_frequency: f32,
    restitution: f32,
    cooling_rate: f32,
    cooling_time: f32,
}

impl HaffCooling {
    /// Builds the cooling law from explicit initial fields.
    ///
    /// * `initial_temperature` — `T0 ≥ 0`, velocity-variance units.
    /// * `initial_collision_frequency` — `ω0 ≥ 0`.
    /// * `restitution` — normal coefficient `e ∈ [0, 1)`.
    ///
    /// Returns `None` for non-finite inputs, negative `T0`/`ω0`, `e` outside
    /// `[0, 1)`, or a non-positive cooling rate (no dissipation to drive the
    /// decay, e.g. `ω0 = 0`).
    pub fn new(
        initial_temperature: f32,
        initial_collision_frequency: f32,
        restitution: f32,
    ) -> Option<Self> {
        if !initial_temperature.is_finite() || initial_temperature < 0.0 {
            return None;
        }
        if !initial_collision_frequency.is_finite() || initial_collision_frequency < 0.0 {
            return None;
        }
        if !restitution.is_finite() || !(0.0..1.0).contains(&restitution) {
            return None;
        }

        // ζ0 = (1 − e²) ω0 / 3.
        let inelasticity = 1.0 - restitution * restitution;
        let cooling_rate = inelasticity * initial_collision_frequency / 3.0;
        if !cooling_rate.is_finite() || cooling_rate <= 0.0 {
            return None;
        }

        // τ = 2 / ζ0.
        let cooling_time = 2.0 / cooling_rate;
        if !cooling_time.is_finite() || cooling_time <= 0.0 {
            return None;
        }

        Some(Self {
            initial_temperature,
            initial_collision_frequency,
            restitution,
            cooling_rate,
            cooling_time,
        })
    }

    /// Builds the cooling law from a kinetic state and a restitution value.
    ///
    /// The initial temperature and collision frequency are read from `state`.
    pub fn from_kinetic_state(state: &GranularKineticState, restitution: f32) -> Option<Self> {
        Self::new(
            state.granular_temperature(),
            state.collision_frequency(),
            restitution,
        )
    }

    /// Initial granular temperature `T0`.
    pub fn initial_temperature(&self) -> f32 {
        self.initial_temperature
    }

    /// Initial collision frequency `ω0`.
    pub fn initial_collision_frequency(&self) -> f32 {
        self.initial_collision_frequency
    }

    /// Normal restitution `e`.
    pub fn restitution(&self) -> f32 {
        self.restitution
    }

    /// Initial cooling rate `ζ0 = (1 − e²) ω0 / 3`.
    pub fn cooling_rate(&self) -> f32 {
        self.cooling_rate
    }

    /// Characteristic cooling time `τ = 2/ζ0`.
    pub fn cooling_time(&self) -> f32 {
        self.cooling_time
    }

    /// Granular temperature `T(t) = T0 / (1 + t/τ)²` at elapsed time `t ≥ 0`.
    ///
    /// Returns `None` for non-finite or negative `t`.
    pub fn temperature_at(&self, time: f32) -> Option<f32> {
        if !time.is_finite() || time < 0.0 {
            return None;
        }
        let r = 1.0 + time / self.cooling_time;
        Some(self.initial_temperature / (r * r))
    }

    /// Collision frequency `ω(t) = ω0 / (1 + t/τ)` at elapsed time `t ≥ 0`.
    ///
    /// Returns `None` for non-finite or negative `t`.
    pub fn collision_frequency_at(&self, time: f32) -> Option<f32> {
        if !time.is_finite() || time < 0.0 {
            return None;
        }
        let r = 1.0 + time / self.cooling_time;
        Some(self.initial_collision_frequency / r)
    }

    /// Time for the granular temperature to fall to a fraction `f ∈ (0, 1]` of
    /// `T0`. From `f = (1 + t/τ)^{-2}` this is `t = τ (f^{-1/2} − 1)`.
    ///
    /// Returns `None` when `f` is non-finite or outside `(0, 1]`.
    pub fn time_to_fraction(&self, fraction: f32) -> Option<f32> {
        if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
            return None;
        }
        let inv_sqrt = 1.0 / fraction.sqrt();
        Some(self.cooling_time * (inv_sqrt - 1.0))
    }

    /// Time for the granular temperature to halve, `t = τ (√2 − 1)`.
    pub fn half_life(&self) -> f32 {
        self.cooling_time * (std::f32::consts::SQRT_2 - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_input() {
        assert!(HaffCooling::new(-1.0, 1.0, 0.5).is_none());
        assert!(HaffCooling::new(1.0, -1.0, 0.5).is_none());
        assert!(HaffCooling::new(1.0, 1.0, 1.0).is_none()); // elastic
        assert!(HaffCooling::new(1.0, 1.0, -0.1).is_none());
        assert!(HaffCooling::new(1.0, 0.0, 0.5).is_none()); // no collisions → no cooling
        assert!(HaffCooling::new(f32::NAN, 1.0, 0.5).is_none());
    }

    #[test]
    fn cooling_rate_formula() {
        // ζ0 = (1 − e²) ω0 / 3; e = 0.5, ω0 = 3 → (0.75)·3/3 = 0.75.
        let h = HaffCooling::new(1.0, 3.0, 0.5).unwrap();
        assert!((h.cooling_rate() - 0.75).abs() < 1e-5);
        // τ = 2/ζ0.
        assert!((h.cooling_time() - 2.0 / 0.75).abs() < 1e-4);
    }

    #[test]
    fn temperature_follows_haff_law() {
        let h = HaffCooling::new(4.0, 2.0, 0.6).unwrap();
        // t = 0 → T0.
        assert!((h.temperature_at(0.0).unwrap() - 4.0).abs() < 1e-5);
        // t = τ → T0/4.
        let tau = h.cooling_time();
        assert!((h.temperature_at(tau).unwrap() - 1.0).abs() < 1e-4);
        // t = 3τ → T0/16.
        assert!((h.temperature_at(3.0 * tau).unwrap() - 4.0 / 16.0).abs() < 1e-4);
        assert!(h.temperature_at(-1.0).is_none());
    }

    #[test]
    fn temperature_is_monotonically_decreasing() {
        let h = HaffCooling::new(5.0, 1.5, 0.4).unwrap();
        let mut prev = f32::INFINITY;
        for k in 0..20 {
            let t = k as f32 * 0.5;
            let temp = h.temperature_at(t).unwrap();
            assert!(temp < prev);
            prev = temp;
        }
    }

    #[test]
    fn collision_frequency_decays_slower_than_temperature() {
        // ω ∝ (1+t/τ)^{-1}, T ∝ (1+t/τ)^{-2}: ω²/ω0² = T/T0.
        let h = HaffCooling::new(2.0, 4.0, 0.7).unwrap();
        let tau = h.cooling_time();
        let w = h.collision_frequency_at(tau).unwrap();
        assert!((w - 2.0).abs() < 1e-4); // ω0/2
        let ratio_w = w / h.initial_collision_frequency();
        let ratio_t = h.temperature_at(tau).unwrap() / h.initial_temperature();
        assert!((ratio_w * ratio_w - ratio_t).abs() < 1e-4);
    }

    #[test]
    fn half_life_and_fraction() {
        let h = HaffCooling::new(1.0, 2.0, 0.5).unwrap();
        let t_half = h.half_life();
        assert!((h.temperature_at(t_half).unwrap() - 0.5).abs() < 1e-4);
        // time_to_fraction(0.5) matches half_life.
        assert!((h.time_to_fraction(0.5).unwrap() - t_half).abs() < 1e-5);
        // Quarter temperature at t = τ.
        assert!((h.time_to_fraction(0.25).unwrap() - h.cooling_time()).abs() < 1e-4);
        assert!(h.time_to_fraction(0.0).is_none());
        assert!(h.time_to_fraction(1.5).is_none());
    }

    #[test]
    fn more_inelastic_cools_faster() {
        // Smaller e → larger (1−e²) → larger ζ0 → shorter τ.
        let soft = HaffCooling::new(1.0, 2.0, 0.9).unwrap();
        let hard = HaffCooling::new(1.0, 2.0, 0.2).unwrap();
        assert!(hard.cooling_rate() > soft.cooling_rate());
        assert!(hard.cooling_time() < soft.cooling_time());
    }

    #[test]
    fn composes_with_kinetic_state() {
        let state = GranularKineticState::new(1.0, 1.0, 0.3, 2.0).unwrap();
        let h = HaffCooling::from_kinetic_state(&state, 0.5).unwrap();
        assert_eq!(h.initial_temperature(), state.granular_temperature());
        assert_eq!(h.initial_collision_frequency(), state.collision_frequency());
        // Frozen state (T = 0 → ω = 0) cannot cool.
        let frozen = GranularKineticState::new(1.0, 1.0, 0.3, 0.0).unwrap();
        assert!(HaffCooling::from_kinetic_state(&frozen, 0.5).is_none());
    }
}
