//! Single-owner `GPU` backend contracts.
//!
//! Prism runs exactly one `GPU` backend at a time — a `Vulkan`-first path or a
//! portable `wgpu` compatibility path — and that backend is the sole owner of
//! resource creation, command recording, and submission. This module owns the
//! deterministic, device-free contracts around that ownership:
//!
//! * [`capability`] — the [`Capability`] bitset, per-[`VulkanTier`] capability
//!   mapping, and [`FeatureRequirement`] validation that reports exactly which
//!   capabilities a tier is missing.
//! * [`selection`] — [`select_backend`], the policy that picks a
//!   [`BackendMode`] from what the platform offers and what features demand.
//! * [`fake`] — a `CPU`-testable [`FakeBackend`] implementation of
//!   [`RenderBackend`].
//!
//! The trait's device-facing operations (resource creation, command recording,
//! submission) are pending the GPU backend; the pieces modeled here are the
//! capability negotiation and lifecycle contract the renderer is built against.

pub mod capability;
pub mod fake;
pub mod selection;

pub use capability::{Capability, CapabilitySet, FeatureRequirement, MissingCapabilities};
pub use fake::FakeBackend;
pub use selection::{select_backend, BackendAvailability, BackendSelection, SelectionError};

/// Selects one independently owned backend. Resources never cross modes
/// without an explicit interop implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BackendMode {
    VulkanFirst,
    WgpuCompatibility,
}

/// `Vulkan` capability tiers consumed by higher-level render features.
///
/// The [`Ord`] derivation ranks tiers by breadth in the common case, but
/// feature gating must go through [`VulkanTier::capabilities`] rather than tier
/// ordering: `MeshShader` and `RayQuery` are siblings, not a linear scale.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum VulkanTier {
    #[default]
    Core13,
    MeshShader,
    RayQuery,
    Full,
}

/// The sole owner of resource creation, command recording, and submission.
///
/// Capability queries have default implementations in terms of [`Self::tier`],
/// so implementors only supply mode, tier, and shutdown. Device-facing methods
/// (resource creation, command recording, submission) are pending the GPU
/// backend.
pub trait RenderBackend {
    /// The mode this backend implements.
    fn mode(&self) -> BackendMode;

    /// The capability tier this backend guarantees.
    fn tier(&self) -> VulkanTier;

    /// Blocks until the `GPU` is idle so resources can be torn down safely.
    fn wait_idle_for_shutdown(&mut self);

    /// The capability set this backend provides, derived from its tier.
    fn capabilities(&self) -> CapabilitySet {
        self.tier().capabilities()
    }

    /// Whether this backend satisfies `requirement`.
    fn supports(&self, requirement: FeatureRequirement) -> bool {
        requirement.validate(self.capabilities()).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_default_is_core13() {
        assert_eq!(VulkanTier::default(), VulkanTier::Core13);
    }

    #[test]
    fn end_to_end_negotiation() {
        // Platform offers a mesh-shader Vulkan device and a baseline wgpu path.
        let avail = BackendAvailability {
            vulkan_tier: Some(VulkanTier::MeshShader),
            wgpu_capabilities: CapabilitySet::from_slice(&[
                Capability::Compute,
                Capability::BindlessDescriptors,
                Capability::IndirectDrawCount,
            ]),
        };
        let needs_mesh =
            FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::MeshShading));
        let sel = select_backend(BackendMode::VulkanFirst, avail, needs_mesh).unwrap();
        assert_eq!(sel.mode, BackendMode::VulkanFirst);

        // Build the chosen backend and confirm it honors the same requirement.
        let mut backend = FakeBackend::new(sel.mode, VulkanTier::MeshShader);
        assert!(backend.supports(needs_mesh));
        assert_eq!(backend.capabilities(), sel.capabilities);
        backend.wait_idle_for_shutdown();
        assert!(backend.is_shut_down());
    }

    #[test]
    fn missing_capability_is_reported_exactly() {
        let backend = FakeBackend::new(BackendMode::VulkanFirst, VulkanTier::Core13);
        let needs_ray = FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::RayQuery));
        assert!(!backend.supports(needs_ray));
        let missing = needs_ray.validate(backend.capabilities()).unwrap_err();
        assert_eq!(
            missing.missing,
            CapabilitySet::EMPTY.with(Capability::RayQuery)
        );
    }
}
