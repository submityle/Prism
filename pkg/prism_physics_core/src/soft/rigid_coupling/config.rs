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
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ClothRigidCouplingConfig {
    /// Whether the (linear) coupling stage runs. `false` makes the whole bridge
    /// a no-op.
    pub enabled: bool,
    /// Whether the per-contact *angular* bridge runs on top of the linear one.
    /// Requires [`enabled`](Self::enabled); `false` leaves the linear path
    /// bit-identical and applies no torque.
    pub angular: bool,
}

impl ClothRigidCouplingConfig {
    /// Returns a disabled configuration (the default): the coupling stage is a
    /// no-op and the rigid-only path stays bit-identical.
    #[must_use]
    pub fn disabled() -> ClothRigidCouplingConfig {
        ClothRigidCouplingConfig {
            enabled: false,
            angular: false,
        }
    }

    /// Returns a configuration with the linear coupling stage enabled and the
    /// angular bridge off (so the linear goldens stay bit-identical).
    #[must_use]
    pub fn active() -> ClothRigidCouplingConfig {
        ClothRigidCouplingConfig {
            enabled: true,
            angular: false,
        }
    }

    /// Returns a configuration with both the linear stage and the per-contact
    /// angular bridge enabled.
    #[must_use]
    pub fn active_angular() -> ClothRigidCouplingConfig {
        ClothRigidCouplingConfig {
            enabled: true,
            angular: true,
        }
    }

    /// Returns a configuration with the given linear-enabled flag and the
    /// angular bridge off.
    #[must_use]
    pub fn with_enabled(enabled: bool) -> ClothRigidCouplingConfig {
        ClothRigidCouplingConfig {
            enabled,
            angular: false,
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
}
