//! `GPU` capability gating over wgpu features.
//!
//! Prism runs on a single portable **wgpu** backend. Beyond the guaranteed
//! `WebGPU` baseline, extra capabilities — bindless descriptor arrays, mesh
//! shading, ray queries, multi-draw indirect count, and so on — are opt-in
//! **wgpu features** (many are native-only extensions that wgpu maps onto the
//! underlying `Vulkan`/`Metal`/`D3D12` extensions). An adapter advertises which
//! features it supports, the device enables the subset the app requests, and
//! higher-level render features declare the capabilities they need.
//!
//! This module owns the deterministic, device-free contract: a small bitset of
//! capabilities, the `WebGPU` baseline that is always present, and the
//! [`FeatureRequirement`] validation that answers one question — does an enabled
//! feature set satisfy a render feature's needs, and if not, exactly which
//! capabilities are missing?
//!
//! Capabilities are a small `u32` bitset, so set math is exact and cheap and the
//! whole layer is `GPU`-independent. Actual feature queries against a live wgpu
//! adapter/device are pending the GPU backend; here we model the
//! feature-to-capability contract the renderer is written against.

/// A single `GPU` capability the renderer can depend on, expressed as a wgpu
/// feature.
///
/// Each capability occupies one bit in a [`CapabilitySet`]. The discriminants
/// are the bit indices and are stable, so serialized sets stay comparable.
/// [`Capability::Compute`] is part of the `WebGPU` baseline (see
/// [`CapabilitySet::WEBGPU_BASELINE`]); the rest are opt-in wgpu extension
/// features whose availability varies by adapter.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum Capability {
    /// Compute shaders and storage buffers: the universal `WebGPU` baseline.
    Compute = 0,
    /// Bindless descriptor indexing of large texture arrays
    /// (wgpu `TEXTURE_BINDING_ARRAY` + non-uniform indexing).
    BindlessDescriptors = 1,
    /// `GPU`-driven multi-draw indirect with a device count buffer
    /// (wgpu `MULTI_DRAW_INDIRECT_COUNT`).
    IndirectDrawCount = 2,
    /// Mesh and task shaders (wgpu `EXPERIMENTAL_MESH_SHADER`).
    MeshShading = 3,
    /// Inline ray queries from any shader stage (wgpu `EXPERIMENTAL_RAY_QUERY`).
    RayQuery = 4,
    /// Ray-tracing acceleration structures / pipelines
    /// (wgpu `EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE`).
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

    /// Whether this capability is an opt-in wgpu extension feature rather than
    /// part of the guaranteed `WebGPU` baseline.
    #[must_use]
    pub const fn is_extension(self) -> bool {
        !matches!(self, Self::Compute)
    }
}

/// An exact set of [`Capability`] flags backed by a `u32` bitmask.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CapabilitySet(u32);

impl CapabilitySet {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// The capabilities every wgpu device guarantees without any extension.
    ///
    /// A negotiated device always has at least these enabled, regardless of
    /// which extension features the adapter supports.
    pub const WEBGPU_BASELINE: Self = Self(Capability::Compute.bit());

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
    /// [`Err`] carrying exactly the capabilities that are missing.
    pub fn validate(self, available: CapabilitySet) -> Result<(), MissingCapabilities> {
        let missing = available.missing_from(self.required);
        if missing.is_empty() {
            Ok(())
        } else {
            Err(MissingCapabilities { missing })
        }
    }
}

/// The shortfall when a feature's requirement was not met; carries the missing
/// capabilities.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MissingCapabilities {
    /// Capabilities the feature needs that the enabled feature set lacks.
    pub missing: CapabilitySet,
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
    fn only_compute_is_baseline() {
        assert!(!Capability::Compute.is_extension());
        for cap in Capability::ALL {
            if cap != Capability::Compute {
                assert!(cap.is_extension(), "{cap:?} should be an extension");
            }
        }
        assert_eq!(
            CapabilitySet::WEBGPU_BASELINE,
            CapabilitySet::EMPTY.with(Capability::Compute)
        );
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
    fn validate_reports_exact_missing_set() {
        let needs_mesh_and_ray = FeatureRequirement::new(CapabilitySet::from_slice(&[
            Capability::MeshShading,
            Capability::RayQuery,
        ]));
        // Baseline is missing both.
        let err = needs_mesh_and_ray
            .validate(CapabilitySet::WEBGPU_BASELINE)
            .unwrap_err();
        assert!(err.missing.contains(Capability::MeshShading));
        assert!(err.missing.contains(Capability::RayQuery));
        assert_eq!(err.missing.len(), 2);

        // Baseline plus mesh shading still misses ray query.
        let err = needs_mesh_and_ray
            .validate(CapabilitySet::WEBGPU_BASELINE.with(Capability::MeshShading))
            .unwrap_err();
        assert_eq!(err.missing, CapabilitySet::EMPTY.with(Capability::RayQuery));

        // Enabling both satisfies the requirement.
        assert!(needs_mesh_and_ray
            .validate(
                CapabilitySet::WEBGPU_BASELINE
                    .with(Capability::MeshShading)
                    .with(Capability::RayQuery)
            )
            .is_ok());
    }

    #[test]
    fn empty_requirement_is_always_satisfied() {
        let none = FeatureRequirement::default();
        assert!(none.validate(CapabilitySet::EMPTY).is_ok());
        assert!(none.validate(CapabilitySet::WEBGPU_BASELINE).is_ok());
    }

    #[test]
    fn determinism_of_missing_set() {
        let req = FeatureRequirement::new(CapabilitySet::from_slice(&[
            Capability::MeshShading,
            Capability::RayTracingPipeline,
        ]));
        let a = req.validate(CapabilitySet::WEBGPU_BASELINE);
        let b = req.validate(CapabilitySet::WEBGPU_BASELINE);
        assert_eq!(a, b);
    }
}
