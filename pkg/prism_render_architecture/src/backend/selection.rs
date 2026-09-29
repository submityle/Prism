//! `BackendMode` selection strategy.
//!
//! Prism can run its `Vulkan`-first backend or a portable `wgpu` compatibility
//! backend, but never both at once: resources never cross modes without an
//! explicit interop path. Choosing which mode owns the frame is a deterministic
//! policy over what the platform offers and what the enabled features demand.
//!
//! [`select_backend`] encodes that policy. It prefers the caller's requested
//! mode, checks whether that mode's advertised capabilities satisfy the required
//! set, and falls back to the other mode only when it can. If neither mode
//! satisfies the requirements it fails with the exact shortfall, so the caller
//! can disable features rather than crash later on a missing `GPU` feature.
//!
//! This layer is pure: it inspects capability sets and never touches a device.

use super::capability::{CapabilitySet, FeatureRequirement, MissingCapabilities};
use super::{BackendMode, VulkanTier};

/// What the platform can offer to the selector.
///
/// The `Vulkan` path is described by an optional [`VulkanTier`] (absent when no
/// suitable `Vulkan` device exists); the `wgpu` path is described directly by
/// the capability set its adapter reports.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BackendAvailability {
    /// The best `Vulkan` tier available, or [`None`] if `Vulkan` is unusable.
    pub vulkan_tier: Option<VulkanTier>,
    /// Capabilities the `wgpu` compatibility adapter reports.
    pub wgpu_capabilities: CapabilitySet,
}

impl BackendAvailability {
    /// Capabilities offered by a given mode under this availability.
    #[must_use]
    pub fn capabilities(self, mode: BackendMode) -> CapabilitySet {
        match mode {
            BackendMode::VulkanFirst => self
                .vulkan_tier
                .map_or(CapabilitySet::EMPTY, VulkanTier::capabilities),
            BackendMode::WgpuCompatibility => self.wgpu_capabilities,
        }
    }

    /// Whether `mode` is usable at all (its `Vulkan` device exists, or `wgpu`
    /// is always considered present).
    #[must_use]
    pub fn is_present(self, mode: BackendMode) -> bool {
        match mode {
            BackendMode::VulkanFirst => self.vulkan_tier.is_some(),
            BackendMode::WgpuCompatibility => true,
        }
    }
}

/// The chosen backend and the capabilities it will provide.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BackendSelection {
    /// The mode that will own the frame.
    pub mode: BackendMode,
    /// The capability set that mode provides.
    pub capabilities: CapabilitySet,
    /// Whether the requested mode was honored (`false` means a fallback ran).
    pub used_fallback: bool,
}

/// Why no backend could be selected.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SelectionError {
    /// No mode's capabilities satisfied the requirement. Carries the smallest
    /// observed shortfall (from the preferred mode when it is present).
    Unsatisfiable(MissingCapabilities),
}

/// Selects a backend for `requirement`, preferring `preferred`.
///
/// The preferred mode wins when it is present and satisfies the requirement.
/// Otherwise the other mode is tried; if it satisfies the requirement the result
/// is flagged as a fallback. When neither works, the shortfall reported is the
/// preferred mode's when that mode is present, else the fallback mode's.
pub fn select_backend(
    preferred: BackendMode,
    availability: BackendAvailability,
    requirement: FeatureRequirement,
) -> Result<BackendSelection, SelectionError> {
    let other = match preferred {
        BackendMode::VulkanFirst => BackendMode::WgpuCompatibility,
        BackendMode::WgpuCompatibility => BackendMode::VulkanFirst,
    };

    if availability.is_present(preferred) {
        let caps = availability.capabilities(preferred);
        if requirement.validate(caps).is_ok() {
            return Ok(BackendSelection {
                mode: preferred,
                capabilities: caps,
                used_fallback: false,
            });
        }
    }

    if availability.is_present(other) {
        let caps = availability.capabilities(other);
        if requirement.validate(caps).is_ok() {
            return Ok(BackendSelection {
                mode: other,
                capabilities: caps,
                used_fallback: true,
            });
        }
    }

    // Report the most relevant shortfall: the preferred mode's if present,
    // otherwise the other mode's.
    let shortfall_mode = if availability.is_present(preferred) {
        preferred
    } else {
        other
    };
    let missing = requirement
        .validate(availability.capabilities(shortfall_mode))
        .err()
        .unwrap_or(MissingCapabilities {
            missing: CapabilitySet::EMPTY,
        });
    Err(SelectionError::Unsatisfiable(missing))
}

#[cfg(test)]
mod tests {
    use super::super::capability::Capability;
    use super::*;

    fn wgpu_baseline() -> CapabilitySet {
        CapabilitySet::from_slice(&[
            Capability::Compute,
            Capability::BindlessDescriptors,
            Capability::IndirectDrawCount,
        ])
    }

    #[test]
    fn preferred_vulkan_wins_when_sufficient() {
        let avail = BackendAvailability {
            vulkan_tier: Some(VulkanTier::Full),
            wgpu_capabilities: wgpu_baseline(),
        };
        let req = FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::RayQuery));
        let sel = select_backend(BackendMode::VulkanFirst, avail, req).unwrap();
        assert_eq!(sel.mode, BackendMode::VulkanFirst);
        assert!(!sel.used_fallback);
        assert!(sel.capabilities.contains(Capability::RayQuery));
    }

    #[test]
    fn falls_back_to_wgpu_when_vulkan_absent() {
        let avail = BackendAvailability {
            vulkan_tier: None,
            wgpu_capabilities: wgpu_baseline(),
        };
        let req = FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::Compute));
        let sel = select_backend(BackendMode::VulkanFirst, avail, req).unwrap();
        assert_eq!(sel.mode, BackendMode::WgpuCompatibility);
        assert!(sel.used_fallback);
    }

    #[test]
    fn falls_back_when_vulkan_tier_too_low() {
        let avail = BackendAvailability {
            vulkan_tier: Some(VulkanTier::Core13),
            // Pretend a hypothetical wgpu adapter exposes mesh shading.
            wgpu_capabilities: wgpu_baseline().with(Capability::MeshShading),
        };
        let req = FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::MeshShading));
        let sel = select_backend(BackendMode::VulkanFirst, avail, req).unwrap();
        assert_eq!(sel.mode, BackendMode::WgpuCompatibility);
        assert!(sel.used_fallback);
    }

    #[test]
    fn unsatisfiable_reports_preferred_shortfall() {
        let avail = BackendAvailability {
            vulkan_tier: Some(VulkanTier::Core13),
            wgpu_capabilities: wgpu_baseline(),
        };
        let req =
            FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::RayTracingPipeline));
        let err = select_backend(BackendMode::VulkanFirst, avail, req).unwrap_err();
        match err {
            SelectionError::Unsatisfiable(m) => {
                assert!(m.missing.contains(Capability::RayTracingPipeline));
            }
        }
    }

    #[test]
    fn preferred_wgpu_is_honored() {
        let avail = BackendAvailability {
            vulkan_tier: Some(VulkanTier::Full),
            wgpu_capabilities: wgpu_baseline(),
        };
        let req = FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::Compute));
        let sel = select_backend(BackendMode::WgpuCompatibility, avail, req).unwrap();
        assert_eq!(sel.mode, BackendMode::WgpuCompatibility);
        assert!(!sel.used_fallback);
    }

    #[test]
    fn determinism_of_selection() {
        let avail = BackendAvailability {
            vulkan_tier: Some(VulkanTier::MeshShader),
            wgpu_capabilities: wgpu_baseline(),
        };
        let req = FeatureRequirement::new(CapabilitySet::EMPTY.with(Capability::MeshShading));
        let a = select_backend(BackendMode::VulkanFirst, avail, req);
        let b = select_backend(BackendMode::VulkanFirst, avail, req);
        assert_eq!(a, b);
    }
}
