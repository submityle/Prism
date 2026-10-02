//! Opt-in configuration for the cloth↔rigid two-way coupling bridge.
//!
//! The coupling bridge is wired into the step flow as an *opt-in* stage so that
//! existing rigid-only goldens stay bit-identical: with the default
//! [`ClothRigidCouplingConfig::disabled`] the step driver never touches the
//! soft arrays or the rigid bodies, and the whole bridge is a no-op. A caller
//! that wants deformable props resting on cloth flips the flag on explicitly.
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
/// they rest against.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ClothRigidCouplingConfig {
    /// Whether the coupling stage runs. `false` makes the whole bridge a no-op.
    pub enabled: bool,
}

impl ClothRigidCouplingConfig {
    /// Returns a disabled configuration (the default): the coupling stage is a
    /// no-op and the rigid-only path stays bit-identical.
    #[must_use]
    pub fn disabled() -> ClothRigidCouplingConfig {
        ClothRigidCouplingConfig { enabled: false }
    }

    /// Returns an enabled configuration: the coupling stage runs each substep.
    #[must_use]
    pub fn active() -> ClothRigidCouplingConfig {
        ClothRigidCouplingConfig { enabled: true }
    }

    /// Returns a configuration with the given enabled flag.
    #[must_use]
    pub fn with_enabled(enabled: bool) -> ClothRigidCouplingConfig {
        ClothRigidCouplingConfig { enabled }
    }

    /// Returns `true` when the coupling stage should run.
    #[must_use]
    pub fn is_enabled(self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_disabled() {
        assert!(!ClothRigidCouplingConfig::default().is_enabled());
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
    fn with_enabled_round_trips() {
        assert!(ClothRigidCouplingConfig::with_enabled(true).is_enabled());
        assert!(!ClothRigidCouplingConfig::with_enabled(false).is_enabled());
    }
}
