//! wgpu extension-feature negotiation.
//!
//! With a single wgpu backend there is nothing to *select* — the only decision
//! is which optional extension features to enable on the device. An adapter
//! advertises the capabilities it can provide; the app requests the extensions
//! its render features need; the device is created with the ones both agree on.
//!
//! [`negotiate_features`] models exactly that handshake as a pure, device-free
//! function:
//!
//! * The [`CapabilitySet::WEBGPU_BASELINE`] is always enabled — every wgpu
//!   device guarantees it, so it is enabled whether or not it was requested.
//! * Requested extensions the adapter supports are enabled.
//! * Requested extensions the adapter does **not** support are reported as
//!   [`EnabledFeatures::unavailable`], so higher layers can pick a fallback
//!   (for example: no ray query → screen-space reflections).
//!
//! Extensions the adapter supports but the app did not request are left off:
//! enabling an unused feature can cost performance and there is nothing to gate
//! on it.

use super::capability::{CapabilitySet, FeatureRequirement};

/// The outcome of negotiating requested extensions against an adapter.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EnabledFeatures {
    enabled: CapabilitySet,
    unavailable: CapabilitySet,
}

impl EnabledFeatures {
    /// The capability set the device will run with.
    ///
    /// Always a superset of [`CapabilitySet::WEBGPU_BASELINE`].
    #[must_use]
    pub const fn enabled(self) -> CapabilitySet {
        self.enabled
    }

    /// Requested extensions the adapter could not provide.
    ///
    /// These drive fallback paths, never a backend switch. Empty means every
    /// requested extension was enabled.
    #[must_use]
    pub const fn unavailable(self) -> CapabilitySet {
        self.unavailable
    }

    /// Whether the negotiated device satisfies `requirement`.
    #[must_use]
    pub fn satisfies(self, requirement: FeatureRequirement) -> bool {
        requirement.validate(self.enabled).is_ok()
    }
}

/// Negotiates the extension features to enable on the single wgpu device.
///
/// `adapter_supported` is what the physical adapter can provide (typically a
/// superset of the baseline). `requested` is what the app's render features ask
/// for. The result enables the baseline plus every requested extension the
/// adapter supports, and reports the requested extensions it does not.
#[must_use]
pub fn negotiate_features(
    adapter_supported: CapabilitySet,
    requested: CapabilitySet,
) -> EnabledFeatures {
    let enabled = CapabilitySet::WEBGPU_BASELINE.union(adapter_supported.intersection(requested));
    let unavailable = requested.difference(adapter_supported);
    EnabledFeatures {
        enabled,
        unavailable,
    }
}

#[cfg(test)]
mod tests {
    use super::super::capability::Capability;
    use super::*;

    #[test]
    fn baseline_is_always_enabled_even_when_unrequested() {
        let enabled = negotiate_features(CapabilitySet::WEBGPU_BASELINE, CapabilitySet::EMPTY);
        assert_eq!(enabled.enabled(), CapabilitySet::WEBGPU_BASELINE);
        assert!(enabled.unavailable().is_empty());
    }

    #[test]
    fn supported_request_is_enabled() {
        let adapter = CapabilitySet::WEBGPU_BASELINE
            .with(Capability::MeshShading)
            .with(Capability::RayQuery);
        let requested = CapabilitySet::EMPTY.with(Capability::MeshShading);
        let enabled = negotiate_features(adapter, requested);
        assert!(enabled.enabled().contains(Capability::MeshShading));
        assert!(enabled.unavailable().is_empty());
    }

    #[test]
    fn unsupported_request_is_unavailable_not_enabled() {
        let adapter = CapabilitySet::WEBGPU_BASELINE.with(Capability::BindlessDescriptors);
        let requested = CapabilitySet::from_slice(&[
            Capability::BindlessDescriptors,
            Capability::RayTracingPipeline,
        ]);
        let enabled = negotiate_features(adapter, requested);
        assert!(enabled.enabled().contains(Capability::BindlessDescriptors));
        assert!(!enabled.enabled().contains(Capability::RayTracingPipeline));
        assert_eq!(
            enabled.unavailable(),
            CapabilitySet::EMPTY.with(Capability::RayTracingPipeline)
        );
    }

    #[test]
    fn supported_but_unrequested_extension_stays_off() {
        let adapter = CapabilitySet::WEBGPU_BASELINE
            .with(Capability::MeshShading)
            .with(Capability::RayQuery);
        let enabled = negotiate_features(adapter, CapabilitySet::EMPTY);
        assert_eq!(enabled.enabled(), CapabilitySet::WEBGPU_BASELINE);
        assert!(!enabled.enabled().contains(Capability::MeshShading));
    }

    #[test]
    fn satisfies_reflects_enabled_set() {
        let adapter = CapabilitySet::WEBGPU_BASELINE.with(Capability::MeshShading);
        let requested = CapabilitySet::EMPTY.with(Capability::MeshShading);
        let enabled = negotiate_features(adapter, requested);
        let needs_mesh =
            FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::MeshShading));
        let needs_ray = FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::RayQuery));
        assert!(enabled.satisfies(needs_mesh));
        assert!(!enabled.satisfies(needs_ray));
    }

    #[test]
    fn negotiation_is_deterministic() {
        let adapter = CapabilitySet::WEBGPU_BASELINE.with(Capability::RayQuery);
        let requested = CapabilitySet::EMPTY.with(Capability::RayQuery);
        assert_eq!(
            negotiate_features(adapter, requested),
            negotiate_features(adapter, requested)
        );
    }
}
