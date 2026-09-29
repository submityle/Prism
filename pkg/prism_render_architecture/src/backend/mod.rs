//! Single-backend `GPU` contracts.
//!
//! Prism runs on exactly one backend: the portable **wgpu** runtime, which
//! itself covers Metal, `Vulkan`, `D3D12`, and `WebGPU`. There is no second,
//! "native" backend to escalate to. Everything the renderer needs is expressed
//! as wgpu capabilities: the guaranteed `WebGPU` baseline plus opt-in wgpu
//! extension features (bindless, mesh shading, ray query, multi-draw indirect
//! count, ray tracing). wgpu maps those extensions onto the underlying
//! `Vulkan`/`Metal`/`D3D12` extensions; capabilities an adapter cannot provide
//! are simply unavailable and drive a fallback path, not a backend switch.
//!
//! This module owns the deterministic, device-free contracts around that single
//! backend:
//!
//! * [`capability`] — the [`Capability`] bitset, the [`CapabilitySet`] set math,
//!   the always-present [`CapabilitySet::WEBGPU_BASELINE`], and
//!   [`FeatureRequirement`] validation that reports exactly which capabilities a
//!   feature is missing.
//! * [`negotiation`] — [`negotiate_features`], which intersects the capabilities
//!   an adapter advertises with the extension features the app requests and
//!   yields the [`EnabledFeatures`] the device will run with, plus the requested
//!   extensions the adapter could not provide.
//! * [`fake`] — a `CPU`-testable [`FakeBackend`] implementation of
//!   [`RenderBackend`].
//!
//! The trait's device-facing operations (resource creation, command recording,
//! submission) are pending the GPU backend; the pieces modeled here are the
//! capability negotiation and lifecycle contract the renderer is built against.

pub mod capability;
pub mod fake;
pub mod negotiation;

pub use capability::{Capability, CapabilitySet, FeatureRequirement, MissingCapabilities};
pub use fake::FakeBackend;
pub use negotiation::{negotiate_features, EnabledFeatures};

/// The sole owner of resource creation, command recording, and submission.
///
/// There is only one backend (wgpu), so there is no backend identity to report;
/// an implementation is characterized purely by the [`CapabilitySet`] its
/// negotiated device runs with. Capability gating has a default implementation
/// in terms of [`Self::capabilities`], so implementors only supply the enabled
/// capabilities and shutdown. Device-facing methods (resource creation, command
/// recording, submission) are pending the GPU backend.
pub trait RenderBackend {
    /// The capability set the negotiated device runs with.
    ///
    /// Always a superset of [`CapabilitySet::WEBGPU_BASELINE`].
    fn capabilities(&self) -> CapabilitySet;

    /// Blocks until the `GPU` is idle so resources can be torn down safely.
    fn wait_idle_for_shutdown(&mut self);

    /// Whether this backend satisfies `requirement`.
    fn supports(&self, requirement: FeatureRequirement) -> bool {
        requirement.validate(self.capabilities()).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_backend_supports_only_baseline_features() {
        let backend = FakeBackend::baseline();
        assert_eq!(backend.capabilities(), CapabilitySet::WEBGPU_BASELINE);

        let needs_compute = FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::Compute));
        assert!(backend.supports(needs_compute));

        let needs_mesh =
            FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::MeshShading));
        assert!(!backend.supports(needs_mesh));
    }

    #[test]
    fn end_to_end_negotiation_then_backend() {
        // The adapter advertises mesh shading among its extensions; the app asks
        // for it. Negotiation enables it and the backend then honors the need.
        let adapter = CapabilitySet::WEBGPU_BASELINE
            .with(Capability::BindlessDescriptors)
            .with(Capability::MeshShading);
        let requested = CapabilitySet::EMPTY.with(Capability::MeshShading);

        let enabled = negotiate_features(adapter, requested);
        assert!(enabled.unavailable().is_empty());

        let mut backend = FakeBackend::new(enabled.enabled());
        let needs_mesh =
            FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::MeshShading));
        assert!(backend.supports(needs_mesh));
        backend.wait_idle_for_shutdown();
        assert!(backend.is_shut_down());
    }

    #[test]
    fn missing_capability_is_reported_exactly() {
        let backend = FakeBackend::baseline();
        let needs_ray = FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::RayQuery));
        assert!(!backend.supports(needs_ray));
        let missing = needs_ray.validate(backend.capabilities()).unwrap_err();
        assert_eq!(
            missing.missing,
            CapabilitySet::EMPTY.with(Capability::RayQuery)
        );
    }

    #[test]
    fn usable_through_trait_object() {
        let mut backend = FakeBackend::with_all_extensions();
        let dyn_backend: &mut dyn RenderBackend = &mut backend;
        assert!(dyn_backend
            .capabilities()
            .contains(Capability::RayTracingPipeline));
        dyn_backend.wait_idle_for_shutdown();
        assert!(backend.is_shut_down());
    }
}
