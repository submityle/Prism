//! A `CPU`-testable fake [`RenderBackend`].
//!
//! The real [`RenderBackend`] owns a live `GPU` device: it creates resources,
//! records command buffers, and submits them. None of that is available on a
//! headless test runner, yet the contract around it — mode, tier, capability
//! gating, and orderly shutdown — is worth testing on its own.
//!
//! [`FakeBackend`] implements the trait entirely on the `CPU`. It reports a
//! configured [`BackendMode`] and [`VulkanTier`], derives its capabilities from
//! that tier, and records shutdown so tests can assert that
//! [`RenderBackend::wait_idle_for_shutdown`] was actually driven. Real device
//! submission remains pending the GPU backend; this stand-in exists purely to
//! exercise the surrounding contract deterministically.

use super::{BackendMode, RenderBackend, VulkanTier};

/// A deterministic, device-free [`RenderBackend`] for tests.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct FakeBackend {
    mode: BackendMode,
    tier: VulkanTier,
    /// Number of times [`RenderBackend::wait_idle_for_shutdown`] was called.
    shutdown_waits: u32,
}

impl FakeBackend {
    /// Creates a fake backend advertising `mode` and `tier`.
    #[must_use]
    pub const fn new(mode: BackendMode, tier: VulkanTier) -> Self {
        Self {
            mode,
            tier,
            shutdown_waits: 0,
        }
    }

    /// A `Vulkan`-first backend at the highest tier, for tests that want every
    /// capability available.
    #[must_use]
    pub const fn full_vulkan() -> Self {
        Self::new(BackendMode::VulkanFirst, VulkanTier::Full)
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
    fn mode(&self) -> BackendMode {
        self.mode
    }

    fn tier(&self) -> VulkanTier {
        self.tier
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
    fn reports_configured_mode_and_tier() {
        let be = FakeBackend::new(BackendMode::WgpuCompatibility, VulkanTier::Core13);
        assert_eq!(be.mode(), BackendMode::WgpuCompatibility);
        assert_eq!(be.tier(), VulkanTier::Core13);
    }

    #[test]
    fn capabilities_follow_tier_via_default_method() {
        let be = FakeBackend::full_vulkan();
        assert_eq!(be.capabilities(), VulkanTier::Full.capabilities());
        assert!(be.capabilities().contains(Capability::RayTracingPipeline));
    }

    #[test]
    fn supports_feature_default_method() {
        let be = FakeBackend::new(BackendMode::VulkanFirst, VulkanTier::MeshShader);
        let mesh = FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::MeshShading));
        let ray = FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::RayQuery));
        assert!(be.supports(mesh));
        assert!(!be.supports(ray));
    }

    #[test]
    fn shutdown_is_recorded() {
        let mut be = FakeBackend::full_vulkan();
        assert!(!be.is_shut_down());
        be.wait_idle_for_shutdown();
        be.wait_idle_for_shutdown();
        assert_eq!(be.shutdown_waits(), 2);
        assert!(be.is_shut_down());
    }

    #[test]
    fn usable_through_trait_object() {
        let mut be = FakeBackend::full_vulkan();
        let dyn_be: &mut dyn RenderBackend = &mut be;
        dyn_be.wait_idle_for_shutdown();
        assert_eq!(dyn_be.mode(), BackendMode::VulkanFirst);
        assert_eq!(be.shutdown_waits(), 1);
    }

    #[test]
    fn determinism_of_capabilities() {
        let be = FakeBackend::new(BackendMode::VulkanFirst, VulkanTier::RayQuery);
        assert_eq!(be.capabilities(), be.capabilities());
    }
}
