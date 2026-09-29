//! A `CPU`-testable fake [`RenderBackend`].
//!
//! The real [`RenderBackend`] owns a live `GPU` device: it creates resources,
//! records command buffers, and submits them. None of that is available on a
//! headless test runner, yet the contract around it — the enabled capability
//! set, feature gating, and orderly shutdown — is worth testing on its own.
//!
//! [`FakeBackend`] implements the trait entirely on the `CPU`. It holds the
//! [`CapabilitySet`] a negotiated wgpu device would run with and records
//! shutdown so tests can assert that [`RenderBackend::wait_idle_for_shutdown`]
//! was actually driven. Real device submission remains pending the GPU backend;
//! this stand-in exists purely to exercise the surrounding contract
//! deterministically.

use super::capability::{Capability, CapabilitySet};
use super::RenderBackend;

/// A deterministic, device-free [`RenderBackend`] for tests.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct FakeBackend {
    enabled: CapabilitySet,
    /// Number of times [`RenderBackend::wait_idle_for_shutdown`] was called.
    shutdown_waits: u32,
}

impl FakeBackend {
    /// Creates a fake backend whose device runs with `enabled`.
    ///
    /// The [`CapabilitySet::WEBGPU_BASELINE`] is folded in unconditionally, so
    /// the backend always reports at least the baseline just like a real
    /// negotiated device.
    #[must_use]
    pub const fn new(enabled: CapabilitySet) -> Self {
        Self {
            enabled: CapabilitySet::WEBGPU_BASELINE.union(enabled),
            shutdown_waits: 0,
        }
    }

    /// A device with only the guaranteed `WebGPU` baseline enabled.
    #[must_use]
    pub const fn baseline() -> Self {
        Self::new(CapabilitySet::WEBGPU_BASELINE)
    }

    /// A device with every extension feature enabled, for tests that want the
    /// full capability set available.
    #[must_use]
    pub fn with_all_extensions() -> Self {
        Self::new(CapabilitySet::from_slice(&Capability::ALL))
    }

    /// How many times shutdown has been awaited on this backend.
    #[must_use]
    pub const fn shutdown_waits(&self) -> u32 {
        self.shutdown_waits
    }

    /// Whether shutdown has been awaited at least once.
    #[must_use]
    pub const fn is_shut_down(&self) -> bool {
        self.shutdown_waits > 0
    }
}

impl RenderBackend for FakeBackend {
    fn capabilities(&self) -> CapabilitySet {
        self.enabled
    }

    fn wait_idle_for_shutdown(&mut self) {
        self.shutdown_waits = self.shutdown_waits.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::super::capability::{Capability, CapabilitySet, FeatureRequirement};
    use super::*;

    #[test]
    fn baseline_reports_only_baseline() {
        let be = FakeBackend::baseline();
        assert_eq!(be.capabilities(), CapabilitySet::WEBGPU_BASELINE);
        assert!(be.capabilities().contains(Capability::Compute));
        assert!(!be.capabilities().contains(Capability::RayQuery));
    }

    #[test]
    fn new_always_folds_in_baseline() {
        let be = FakeBackend::new(CapabilitySet::EMPTY.with(Capability::MeshShading));
        assert!(be.capabilities().contains(Capability::Compute));
        assert!(be.capabilities().contains(Capability::MeshShading));
    }

    #[test]
    fn all_extensions_covers_every_capability() {
        let be = FakeBackend::with_all_extensions();
        for cap in Capability::ALL {
            assert!(be.capabilities().contains(cap), "{cap:?} should be enabled");
        }
    }

    #[test]
    fn supports_feature_default_method() {
        let be = FakeBackend::new(CapabilitySet::EMPTY.with(Capability::MeshShading));
        let mesh = FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::MeshShading));
        let ray = FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::RayQuery));
        assert!(be.supports(mesh));
        assert!(!be.supports(ray));
    }

    #[test]
    fn shutdown_is_recorded() {
        let mut be = FakeBackend::with_all_extensions();
        assert!(!be.is_shut_down());
        be.wait_idle_for_shutdown();
        be.wait_idle_for_shutdown();
        assert_eq!(be.shutdown_waits(), 2);
        assert!(be.is_shut_down());
    }

    #[test]
    fn usable_through_trait_object() {
        let mut be = FakeBackend::baseline();
        let dyn_be: &mut dyn RenderBackend = &mut be;
        dyn_be.wait_idle_for_shutdown();
        assert_eq!(dyn_be.capabilities(), CapabilitySet::WEBGPU_BASELINE);
        assert_eq!(be.shutdown_waits(), 1);
    }

    #[test]
    fn determinism_of_capabilities() {
        let be = FakeBackend::with_all_extensions();
        assert_eq!(be.capabilities(), be.capabilities());
    }
}
