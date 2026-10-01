//! Soft-constraint coefficients for the Temporal Gauss-Seidel (`TGS`) solver.
//!
//! A soft constraint is a damped harmonic oscillator pulling the constraint
//! error `C` to zero. Rather than exposing the raw spring stiffness and damping
//! as forces (which couple awkwardly to the time step), the constraint is
//! specified by an *angular* response: a target frequency `hertz` and a
//! dimensionless `damping_ratio` (`1` is critical damping). From those and the
//! substep size `h` we precompute three per-iteration coefficients that the
//! velocity solver applies directly:
//!
//! * `bias_rate` — converts the position error into a target bias velocity
//!   (`bias = bias_rate * C`), the amount of error the solver tries to remove.
//! * `mass_scale` — scales the effective mass so a soft constraint resists the
//!   corrective impulse, bleeding the correction across iterations instead of
//!   snapping in one step.
//! * `impulse_scale` — decays the accumulated impulse each iteration so a soft
//!   constraint relaxes rather than locking rigidly.
//!
//! Taking `hertz` to infinity recovers the rigid limit: `bias_rate -> 1/h`,
//! `mass_scale -> 1`, `impulse_scale -> 0`, i.e. a full Baumgarte correction
//! with no impulse decay, which is exactly what a hard distance joint wants.
//!
//! # Provenance
//!
//! The soft-constraint parametrisation (frequency / damping ratio mapped to
//! bias, mass, and impulse coefficients) is the standard implicit-spring form
//! used by modern substepping velocity solvers (Catto, "Soft Constraints",
//! GDC 2011; the same coefficients appear in `Box2D`'s TGS Soft solver). No
//! Unreal Engine source or derived code.

use core::f32::consts::TAU;

/// Precomputed per-iteration coefficients of a soft constraint at a fixed
/// substep size.
///
/// Build with [`SoftParams::from_hertz`] (a damped spring) or
/// [`SoftParams::rigid`] (the stiff limit). The velocity solver reads the three
/// public fields; they are inert data with no further invariants beyond being
/// finite and non-negative, which the constructors guarantee for finite inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftParams {
    /// Converts position error into target bias velocity: `bias = bias_rate*C`.
    pub bias_rate: f32,
    /// Effective-mass scale applied to the corrective impulse, in `[0, 1]`.
    pub mass_scale: f32,
    /// Decay applied to the accumulated impulse each iteration, in `[0, 1]`.
    pub impulse_scale: f32,
}

impl SoftParams {
    /// Builds the coefficients of a damped spring at frequency `hertz` and
    /// damping ratio `damping_ratio`, solved at substep size `h`.
    ///
    /// A non-positive `hertz` (or non-positive `h`) collapses to the rigid
    /// limit via [`SoftParams::rigid`], so a caller can request "as stiff as
    /// possible" with `hertz = 0`. `damping_ratio` is clamped to be
    /// non-negative; `1` is critical damping, below `1` is underdamped
    /// (springy), above `1` is overdamped (sluggish).
    #[must_use]
    pub fn from_hertz(hertz: f32, damping_ratio: f32, h: f32) -> SoftParams {
        if hertz <= 0.0 || h <= 0.0 {
            return SoftParams::rigid(h);
        }
        let zeta = damping_ratio.max(0.0);
        // omega is the undamped angular frequency; a1 is the damped response
        // denominator (velocity term), a2 the position term, a3 its normaliser.
        let omega = TAU * hertz;
        let a1 = 2.0 * zeta + h * omega;
        let a2 = h * omega * a1;
        let a3 = 1.0 / (1.0 + a2);
        SoftParams {
            bias_rate: omega / a1,
            mass_scale: a2 * a3,
            impulse_scale: a3,
        }
    }

    /// The rigid limit (`hertz -> infinity`): a full Baumgarte correction with
    /// no impulse decay.
    ///
    /// `bias_rate` removes the entire position error over one substep
    /// (`1/h`), the effective mass is used in full (`mass_scale = 1`), and the
    /// accumulated impulse is preserved (`impulse_scale = 0`). A non-positive
    /// `h` yields a zero `bias_rate`, which turns the biased solve into a plain
    /// velocity projection (the subsequent position integration is a no-op for
    /// such a step anyway).
    #[must_use]
    pub fn rigid(h: f32) -> SoftParams {
        let bias_rate = if h > 0.0 { 1.0 / h } else { 0.0 };
        SoftParams {
            bias_rate,
            mass_scale: 1.0,
            impulse_scale: 0.0,
        }
    }

    /// The relaxation coefficients used by the bias-free solve pass: no bias
    /// velocity, full effective mass, no impulse decay.
    ///
    /// The relax pass runs *after* positions are integrated to remove the bias
    /// velocity the biased pass injected, so it must not re-apply any bias.
    #[must_use]
    pub fn relax() -> SoftParams {
        SoftParams {
            bias_rate: 0.0,
            mass_scale: 1.0,
            impulse_scale: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rigid_preset_is_full_baumgarte_with_no_decay() {
        let p = SoftParams::rigid(0.01);
        assert!(
            (p.bias_rate - 100.0).abs() < 1e-4,
            "bias_rate was {}",
            p.bias_rate
        );
        assert_eq!(p.mass_scale, 1.0);
        assert_eq!(p.impulse_scale, 0.0);
    }

    #[test]
    fn non_positive_hertz_collapses_to_rigid() {
        let h = 1.0 / 240.0;
        assert_eq!(SoftParams::from_hertz(0.0, 1.0, h), SoftParams::rigid(h));
        assert_eq!(SoftParams::from_hertz(-5.0, 1.0, h), SoftParams::rigid(h));
    }

    #[test]
    fn non_positive_substep_yields_zero_bias_rate() {
        assert_eq!(SoftParams::from_hertz(30.0, 1.0, 0.0).bias_rate, 0.0);
        assert_eq!(SoftParams::rigid(0.0).bias_rate, 0.0);
    }

    #[test]
    fn coefficients_stay_in_the_unit_range() {
        let h = 1.0 / 240.0;
        for &hertz in &[1.0f32, 10.0, 30.0, 120.0, 1000.0] {
            for &zeta in &[0.25f32, 1.0, 4.0] {
                let p = SoftParams::from_hertz(hertz, zeta, h);
                assert!(p.bias_rate > 0.0 && p.bias_rate.is_finite());
                assert!(
                    (0.0..=1.0).contains(&p.mass_scale),
                    "mass_scale {} out of range for hertz {hertz} zeta {zeta}",
                    p.mass_scale
                );
                assert!(
                    (0.0..=1.0).contains(&p.impulse_scale),
                    "impulse_scale {} out of range",
                    p.impulse_scale
                );
            }
        }
    }

    #[test]
    fn mass_and_impulse_scales_are_complementary() {
        // By construction mass_scale = a2*a3 and impulse_scale = a3 with
        // a3 = 1/(1+a2), so mass_scale + impulse_scale == 1 exactly.
        let p = SoftParams::from_hertz(30.0, 1.0, 1.0 / 240.0);
        assert!((p.mass_scale + p.impulse_scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stiffer_springs_approach_the_rigid_limit() {
        let h = 1.0 / 240.0;
        let soft = SoftParams::from_hertz(5.0, 1.0, h);
        let stiff = SoftParams::from_hertz(5000.0, 1.0, h);
        // Higher frequency means more mass used and less impulse decay, i.e.
        // closer to the rigid preset.
        assert!(stiff.mass_scale > soft.mass_scale);
        assert!(stiff.impulse_scale < soft.impulse_scale);
        assert!((stiff.mass_scale - 1.0).abs() < (soft.mass_scale - 1.0).abs());
    }

    #[test]
    fn known_value_matches_hand_computation() {
        // h = 1/60, hertz = 60/(2*pi) so omega = 60 exactly; zeta = 1.
        let h = 1.0 / 60.0;
        let hertz = 60.0 / TAU;
        let p = SoftParams::from_hertz(hertz, 1.0, h);
        // omega = 60; a1 = 2 + (1/60)*60 = 3; a2 = (1/60)*60*3 = 3; a3 = 1/4.
        // bias_rate = 60/3 = 20; mass_scale = 3/4; impulse_scale = 1/4.
        assert!(
            (p.bias_rate - 20.0).abs() < 1e-3,
            "bias_rate {}",
            p.bias_rate
        );
        assert!((p.mass_scale - 0.75).abs() < 1e-5);
        assert!((p.impulse_scale - 0.25).abs() < 1e-5);
    }
}
