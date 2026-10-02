//! Opt-in configuration for the cloth↔rigid two-way coupling bridge.
//!
//! The coupling bridge is wired into the step flow as an *opt-in* stage so that
//! existing rigid-only goldens stay bit-identical: with the default
//! [`ClothRigidCouplingConfig::disabled`] the step driver never touches the
//! soft arrays or the rigid bodies, and the whole bridge is a no-op. A caller
//! that wants deformable props resting on cloth flips the flag on explicitly.
//!
//! Two levels of coupling are exposed:
//!
//! * [`enabled`](ClothRigidCouplingConfig::enabled) runs the linear bridge
//!   ([`super::couple_cloth_to_rigid`]): particles and proxies exchange a
//!   mass-weighted push and each body receives the net linear reaction impulse.
//! * [`angular`](ClothRigidCouplingConfig::angular) additionally arms the
//!   per-contact angular bridge ([`super::couple_cloth_to_rigid_angular`]),
//!   which recovers each contact's lever arm and applies the net torque to the
//!   body's angular velocity. It is a strict superset of the linear bridge and
//!   is off by default so the linear goldens stay bit-identical.
//! * [`friction`](ClothRigidCouplingConfig::friction) additionally arms the
//!   per-contact Coulomb *friction* bridge
//!   ([`super::couple_cloth_to_rigid_friction`]), the final coupling pass. It
//!   brakes the tangential slide between each particle and the proxy and writes
//!   the equal-and-opposite tangential (and its lever-arm) reaction back onto
//!   the body, clamped by the `μ_s`/`μ_d` cone. It is off by default so the
//!   linear/angular goldens stay bit-identical.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is a
//! plain configuration flag with no physics in it.

/// Toggles the cloth↔rigid two-way coupling stage in the step flow.
///
/// Defaults to disabled so that worlds that never opt in behave exactly as they
/// did before the bridge existed (the rigid-only path is untouched and
/// bit-identical). Enable it to let soft bodies push back on the rigid proxies
/// they rest against; additionally set [`angular`](Self::angular) to let the
/// per-contact bridge spin the proxies about their contact lever arms.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ClothRigidCouplingConfig {
    /// Whether the (linear) coupling stage runs. `false` makes the whole bridge
    /// a no-op.
    pub enabled: bool,
    /// Whether the per-contact *angular* bridge runs on top of the linear one.
    /// Requires [`enabled`](Self::enabled); `false` leaves the linear path
    /// bit-identical and applies no torque.
    pub angular: bool,
    /// Whether the per-contact Coulomb *friction* bridge runs as the final
    /// coupling pass. Requires [`enabled`](Self::enabled); `false` leaves the
    /// linear/angular path bit-identical and applies no tangential reaction.
    pub friction: bool,
    /// Static Coulomb friction coefficient `μ_s`. The tangential impulse stays
    /// in the static (full-arrest) regime while it does not exceed
    /// `μ_s · |jₙ|`. Sanitised to `[0, 1]` (NaN → 0) by
    /// [`friction_static`](Self::friction_static).
    pub mu_static: crate::math::scalar::Real,
    /// Dynamic (sliding) Coulomb friction coefficient `μ_d`. Once the static
    /// cone is exceeded the tangential impulse is clamped to `μ_d · |jₙ|`.
    /// Sanitised to `[0, 1]` (NaN → 0) by
    /// [`friction_dynamic`](Self::friction_dynamic).
    pub mu_dynamic: crate::math::scalar::Real,
}

impl ClothRigidCouplingConfig {
    /// Returns a disabled configuration (the default): the coupling stage is a
    /// no-op and the rigid-only path stays bit-identical.
    #[must_use]
    pub fn disabled() -> ClothRigidCouplingConfig {
        ClothRigidCouplingConfig {
            enabled: false,
            angular: false,
            friction: false,
            mu_static: 0.0,
            mu_dynamic: 0.0,
        }
    }

    /// Returns a configuration with the linear coupling stage enabled and the
    /// angular bridge off (so the linear goldens stay bit-identical).
    #[must_use]
    pub fn active() -> ClothRigidCouplingConfig {
        ClothRigidCouplingConfig {
            enabled: true,
            angular: false,
            friction: false,
            mu_static: 0.0,
            mu_dynamic: 0.0,
        }
    }

    /// Returns a configuration with both the linear stage and the per-contact
    /// angular bridge enabled.
    #[must_use]
    pub fn active_angular() -> ClothRigidCouplingConfig {
        ClothRigidCouplingConfig {
            enabled: true,
            angular: true,
            friction: false,
            mu_static: 0.0,
            mu_dynamic: 0.0,
        }
    }

    /// Returns a configuration with the given linear-enabled flag and the
    /// angular bridge off.
    #[must_use]
    pub fn with_enabled(enabled: bool) -> ClothRigidCouplingConfig {
        ClothRigidCouplingConfig {
            enabled,
            angular: false,
            friction: false,
            mu_static: 0.0,
            mu_dynamic: 0.0,
        }
    }

    /// Returns a copy of `self` with the angular bridge flag set to `angular`.
    /// The angular bridge only runs when [`enabled`](Self::enabled) is also set.
    #[must_use]
    pub fn with_angular(mut self, angular: bool) -> ClothRigidCouplingConfig {
        self.angular = angular;
        self
    }

    /// Returns `true` when the (linear) coupling stage should run.
    #[must_use]
    pub fn is_enabled(self) -> bool {
        self.enabled
    }

    /// Returns `true` when the per-contact angular bridge should run. It
    /// requires the linear stage to be enabled too.
    #[must_use]
    pub fn is_angular(self) -> bool {
        self.enabled && self.angular
    }

    /// Returns a configuration with the linear stage and the per-contact
    /// Coulomb friction bridge enabled (the angular bridge left off), using the
    /// given static/dynamic coefficients.
    ///
    /// The friction driver still runs the linear normal pass first, so a caller
    /// that wants friction *and* torque should instead combine
    /// [`active_angular`](Self::active_angular) with
    /// [`with_friction`](Self::with_friction).
    #[must_use]
    pub fn active_friction(
        mu_static: crate::math::scalar::Real,
        mu_dynamic: crate::math::scalar::Real,
    ) -> ClothRigidCouplingConfig {
        ClothRigidCouplingConfig {
            enabled: true,
            angular: false,
            friction: true,
            mu_static,
            mu_dynamic,
        }
    }

    /// Returns a copy of `self` with the friction bridge flag and its
    /// coefficients set. The friction bridge only runs when
    /// [`enabled`](Self::enabled) is also set.
    #[must_use]
    pub fn with_friction(
        mut self,
        friction: bool,
        mu_static: crate::math::scalar::Real,
        mu_dynamic: crate::math::scalar::Real,
    ) -> ClothRigidCouplingConfig {
        self.friction = friction;
        self.mu_static = mu_static;
        self.mu_dynamic = mu_dynamic;
        self
    }

    /// Returns `true` when the per-contact Coulomb friction bridge should run.
    /// It requires the linear stage to be enabled too.
    #[must_use]
    pub fn is_friction(self) -> bool {
        self.enabled && self.friction
    }

    /// Returns the sanitised static Coulomb coefficient `μ_s`, clamped to
    /// `[0, 1]` with a NaN mapped to `0` so the friction cone is always a
    /// finite, non-negative fraction of the normal impulse.
    #[must_use]
    pub fn friction_static(self) -> crate::math::scalar::Real {
        sanitize_mu(self.mu_static)
    }

    /// Returns the sanitised dynamic Coulomb coefficient `μ_d`, clamped to
    /// `[0, 1]` with a NaN mapped to `0`.
    #[must_use]
    pub fn friction_dynamic(self) -> crate::math::scalar::Real {
        sanitize_mu(self.mu_dynamic)
    }
}

/// Clamps a Coulomb friction coefficient into the physically meaningful
/// `[0, 1]` range, mapping a `NaN` to `0` so a mis-configured coefficient can
/// never produce a [`f32::NAN`] tangential impulse or a cone wider than the
/// normal impulse.
fn sanitize_mu(mu: crate::math::scalar::Real) -> crate::math::scalar::Real {
    if mu.is_nan() {
        0.0
    } else {
        mu.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_disabled() {
        assert!(!ClothRigidCouplingConfig::default().is_enabled());
        assert!(!ClothRigidCouplingConfig::default().is_angular());
        assert_eq!(
            ClothRigidCouplingConfig::default(),
            ClothRigidCouplingConfig::disabled()
        );
    }

    #[test]
    fn active_is_enabled() {
        assert!(ClothRigidCouplingConfig::active().is_enabled());
    }

    #[test]
    fn active_is_linear_only_by_default() {
        // `active` keeps the angular bridge off so the linear goldens stay
        // bit-identical; only `active_angular`/`with_angular` arm the torque path.
        assert!(!ClothRigidCouplingConfig::active().is_angular());
    }

    #[test]
    fn active_angular_enables_both() {
        let cfg = ClothRigidCouplingConfig::active_angular();
        assert!(cfg.is_enabled());
        assert!(cfg.is_angular());
    }

    #[test]
    fn angular_requires_enabled() {
        // Setting only the angular flag without `enabled` must not run anything.
        let cfg = ClothRigidCouplingConfig::with_enabled(false).with_angular(true);
        assert!(!cfg.is_enabled());
        assert!(!cfg.is_angular());
    }

    #[test]
    fn with_enabled_round_trips() {
        assert!(ClothRigidCouplingConfig::with_enabled(true).is_enabled());
        assert!(!ClothRigidCouplingConfig::with_enabled(false).is_enabled());
    }

    #[test]
    fn with_angular_round_trips() {
        assert!(ClothRigidCouplingConfig::active()
            .with_angular(true)
            .is_angular());
        assert!(!ClothRigidCouplingConfig::active()
            .with_angular(false)
            .is_angular());
    }

    #[test]
    fn default_has_no_friction() {
        let cfg = ClothRigidCouplingConfig::default();
        assert!(!cfg.friction);
        assert!(!cfg.is_friction());
        assert_eq!(cfg.mu_static, 0.0);
        assert_eq!(cfg.mu_dynamic, 0.0);
    }

    #[test]
    fn active_friction_enables_linear_and_friction_only() {
        let cfg = ClothRigidCouplingConfig::active_friction(0.8, 0.5);
        assert!(cfg.is_enabled());
        assert!(cfg.is_friction());
        // Friction does not implicitly arm the angular bridge.
        assert!(!cfg.is_angular());
        assert_eq!(cfg.friction_static(), 0.8);
        assert_eq!(cfg.friction_dynamic(), 0.5);
    }

    #[test]
    fn friction_requires_enabled() {
        // Setting only the friction flag without `enabled` must not run anything.
        let cfg = ClothRigidCouplingConfig::with_enabled(false).with_friction(true, 0.9, 0.6);
        assert!(!cfg.is_enabled());
        assert!(!cfg.is_friction());
    }

    #[test]
    fn with_friction_round_trips() {
        let on = ClothRigidCouplingConfig::active().with_friction(true, 0.7, 0.4);
        assert!(on.is_friction());
        assert_eq!(on.mu_static, 0.7);
        assert_eq!(on.mu_dynamic, 0.4);
        assert!(!ClothRigidCouplingConfig::active()
            .with_friction(false, 0.7, 0.4)
            .is_friction());
    }

    #[test]
    fn coefficients_are_sanitised_into_unit_range() {
        // Out-of-range and NaN coefficients are clamped/zeroed by the accessors
        // so the friction cone can never widen past the normal impulse or NaN.
        let cfg = ClothRigidCouplingConfig::active_friction(5.0, -2.0);
        assert_eq!(cfg.friction_static(), 1.0);
        assert_eq!(cfg.friction_dynamic(), 0.0);
        let nan = ClothRigidCouplingConfig::active_friction(f32::NAN, f32::NAN);
        assert_eq!(nan.friction_static(), 0.0);
        assert_eq!(nan.friction_dynamic(), 0.0);
    }
}
