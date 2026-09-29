//! `GPU` capability gating for `Vulkan` tiers.
//!
//! Higher-level render features declare the `GPU` capabilities they need — mesh
//! shading, ray queries, indirect draw-count, and so on. A concrete backend
//! advertises a [`VulkanTier`], and each tier maps to the fixed
//! [`CapabilitySet`] it guarantees. This module owns that mapping and the
//! validation that answers one question deterministically: does this tier
//! satisfy a feature's requirements, and if not, exactly which capabilities are
//! missing?
//!
//! Capabilities are a small `u32` bitset, so set math is exact and cheap and the
//! whole layer is `GPU`-independent. Actual device-feature queries against a
//! live `Vulkan` physical device are pending the GPU backend; here we model the
//! tier-to-capability contract the renderer is written against.

use super::VulkanTier;

/// A single `GPU` capability the renderer can depend on.
///
/// Each capability occupies one bit in a [`CapabilitySet`]. The discriminants
/// are the bit indices and are stable, so serialized sets stay comparable.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum Capability {
    /// Compute shaders and storage buffers: the universal baseline.
    Compute = 0,
    /// Bindless descriptor indexing of large descriptor arrays.
    BindlessDescriptors = 1,
    /// `vkCmdDrawIndirectCount`-style `GPU`-driven draw submission.
    IndirectDrawCount = 2,
    /// Mesh and task shaders (the `MeshShader` tier).
    MeshShading = 3,
    /// Inline ray queries from any shader stage (the `RayQuery` tier).
    RayQuery = 4,
    /// Full ray-tracing pipelines with shader binding tables.
    RayTracingPipeline = 5,
}

impl Capability {
    /// Every capability, in ascending bit order.
    pub const ALL: [Self; 6] = [
        Self::Compute,
        Self::BindlessDescriptors,
        Self::IndirectDrawCount,
        Self::MeshShading,
        Self::RayQuery,
        Self::RayTracingPipeline,
    ];

    /// The single-bit mask for this capability.
    #[must_use]
    pub const fn bit(self) -> u32 {
        1u32 << (self as u8)
    }
}

/// An exact set of [`Capability`] flags backed by a `u32` bitmask.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CapabilitySet(u32);

impl CapabilitySet {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// Builds a set from a slice of capabilities.
    #[must_use]
    pub const fn from_slice(caps: &[Capability]) -> Self {
        let mut bits = 0u32;
        let mut i = 0;
        while i < caps.len() {
            bits |= caps[i].bit();
            i += 1;
        }
        Self(bits)
    }

    /// The raw bitmask.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Returns `true` when the set contains no capabilities.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Number of capabilities in the set.
    #[must_use]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    /// Returns `true` when `cap` is present.
    #[must_use]
    pub const fn contains(self, cap: Capability) -> bool {
        self.0 & cap.bit() != 0
    }

    /// Returns `true` when every capability in `other` is present in `self`.
    #[must_use]
    pub const fn contains_all(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// The set with `cap` added.
    #[must_use]
    pub const fn with(self, cap: Capability) -> Self {
        Self(self.0 | cap.bit())
    }

    /// Union of two sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Intersection of two sets.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Capabilities present in `self` but absent from `other`.
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Capabilities `required` asks for that `self` does not provide.
    ///
    /// This is the gating primitive: an empty result means `self` satisfies
    /// `required`.
    #[must_use]
    pub const fn missing_from(self, required: Self) -> Self {
        required.difference(self)
    }
}

/// The capabilities a render feature requires from the backend.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct FeatureRequirement {
    required: CapabilitySet,
}

impl FeatureRequirement {
    /// A requirement for exactly the given capabilities.
    #[must_use]
    pub const fn new(required: CapabilitySet) -> Self {
        Self { required }
    }

    /// The required capability set.
    #[must_use]
    pub const fn required(self) -> CapabilitySet {
        self.required
    }

    /// Validates `available` against this requirement.
    ///
    /// Returns [`Ok`] when every required capability is present, otherwise
    /// [`Err`] carrying the exact set of missing capabilities.
    pub const fn validate(self, available: CapabilitySet) -> Result<(), MissingCapabilities> {
        let missing = available.missing_from(self.required);
        if missing.is_empty() {
            Ok(())
        } else {
            Err(MissingCapabilities { missing })
        }
    }
}

/// A feature's requirement was not met; carries the shortfall.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MissingCapabilities {
    /// Capabilities the feature needs that the backend does not provide.
    pub missing: CapabilitySet,
}

impl VulkanTier {
    /// The capability set every backend at this tier guarantees.
    ///
    /// * `Core13` — the universal baseline: compute, bindless, indirect count.
    /// * `MeshShader` — baseline plus mesh/task shading.
    /// * `RayQuery` — baseline plus inline ray queries.
    /// * `Full` — baseline plus mesh shading, ray queries, and ray-tracing
    ///   pipelines.
    #[must_use]
    pub const fn capabilities(self) -> CapabilitySet {
        let baseline = CapabilitySet::from_slice(&[
            Capability::Compute,
            Capability::BindlessDescriptors,
            Capability::IndirectDrawCount,
        ]);
        match self {
            Self::Core13 => baseline,
            Self::MeshShader => baseline.with(Capability::MeshShading),
            Self::RayQuery => baseline.with(Capability::RayQuery),
            Self::Full => baseline
                .with(Capability::MeshShading)
                .with(Capability::RayQuery)
                .with(Capability::RayTracingPipeline),
        }
    }

    /// Returns `true` when this tier satisfies `requirement`.
    #[must_use]
    pub const fn satisfies(self, requirement: FeatureRequirement) -> bool {
        requirement.validate(self.capabilities()).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_are_distinct_powers_of_two() {
        let mut seen = 0u32;
        for cap in Capability::ALL {
            let bit = cap.bit();
            assert_eq!(bit.count_ones(), 1, "each capability is a single bit");
            assert_eq!(seen & bit, 0, "bits must not collide");
            seen |= bit;
        }
        assert_eq!(seen.count_ones(), Capability::ALL.len() as u32);
    }

    #[test]
    fn set_membership_and_math() {
        let s = CapabilitySet::from_slice(&[Capability::Compute, Capability::RayQuery]);
        assert!(s.contains(Capability::Compute));
        assert!(!s.contains(Capability::MeshShading));
        assert_eq!(s.len(), 2);
        let t = CapabilitySet::EMPTY.with(Capability::RayQuery);
        assert!(s.contains_all(t));
        assert_eq!(s.intersection(t), t);
        assert_eq!(
            s.difference(t),
            CapabilitySet::EMPTY.with(Capability::Compute)
        );
    }

    #[test]
    fn tier_capability_gating() {
        assert!(VulkanTier::Core13
            .capabilities()
            .contains(Capability::Compute));
        assert!(!VulkanTier::Core13
            .capabilities()
            .contains(Capability::MeshShading));
        assert!(VulkanTier::MeshShader
            .capabilities()
            .contains(Capability::MeshShading));
        assert!(!VulkanTier::MeshShader
            .capabilities()
            .contains(Capability::RayQuery));
        assert!(VulkanTier::RayQuery
            .capabilities()
            .contains(Capability::RayQuery));
        assert!(VulkanTier::Full
            .capabilities()
            .contains(Capability::RayTracingPipeline));
        assert!(VulkanTier::Full
            .capabilities()
            .contains(Capability::MeshShading));
    }

    #[test]
    fn validate_reports_exact_missing_set() {
        let needs_mesh_and_ray = FeatureRequirement::new(CapabilitySet::from_slice(&[
            Capability::MeshShading,
            Capability::RayQuery,
        ]));
        // Core13 is missing both.
        let err = needs_mesh_and_ray
            .validate(VulkanTier::Core13.capabilities())
            .unwrap_err();
        assert!(err.missing.contains(Capability::MeshShading));
        assert!(err.missing.contains(Capability::RayQuery));
        assert_eq!(err.missing.len(), 2);

        // MeshShader tier still misses ray query.
        let err = needs_mesh_and_ray
            .validate(VulkanTier::MeshShader.capabilities())
            .unwrap_err();
        assert_eq!(err.missing, CapabilitySet::EMPTY.with(Capability::RayQuery));

        // Full tier satisfies everything.
        assert!(needs_mesh_and_ray
            .validate(VulkanTier::Full.capabilities())
            .is_ok());
        assert!(VulkanTier::Full.satisfies(needs_mesh_and_ray));
    }

    #[test]
    fn empty_requirement_is_always_satisfied() {
        let none = FeatureRequirement::default();
        assert!(none.validate(CapabilitySet::EMPTY).is_ok());
        assert!(VulkanTier::Core13.satisfies(none));
    }

    #[test]
    fn determinism_of_missing_set() {
        let req = FeatureRequirement::new(CapabilitySet::from_slice(&[
            Capability::MeshShading,
            Capability::RayTracingPipeline,
        ]));
        let a = req.validate(VulkanTier::Core13.capabilities());
        let b = req.validate(VulkanTier::Core13.capabilities());
        assert_eq!(a, b);
    }
}
