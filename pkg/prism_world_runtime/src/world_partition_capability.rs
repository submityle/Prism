//! Explicit capability boundary for the engine-neutral world runtime.
//!
//! The runtime owns bounded cell streaming and its traceable load/reload/evict
//! lifecycle. It is not a complete scene-owned World Partition implementation.
//! Callers that need a broader product capability can fail closed with a typed
//! error instead of silently treating this vertical slice as equivalent.

use core::fmt;

/// A World Partition capability known to this runtime boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WorldPartitionCapability {
    /// Bounded cell load and resident replacement through [`crate::SceneHost`].
    CellStreaming,
    /// Manifest-versioned resident cell replacement.
    CellReload,
    /// Resident cell eviction.
    CellEviction,
    /// Deterministic validation of the runtime load/evict trace.
    TraceReplay,
    /// Runtime data-layer activation and filtering.
    DataLayers,
    /// Hierarchical distant proxies and HLOD generation/selection.
    HierarchicalLod,
    /// One-file-per-actor authoring and merge semantics.
    OneFilePerActor,
    /// Reusable level-instance composition and ownership.
    LevelInstances,
    /// Persistent cross-cell object references and coordinated relocation.
    CrossCellReferences,
    /// Runtime world-origin rebasing for scene-owned objects.
    WorldOriginRebasing,
    /// Multiplayer authority handoff across streamed sectors.
    MultiplayerHandoff,
    /// Crash-atomic persistence spanning scene, derived caches and journals.
    CrossResourceCrashAtomicity,
    /// The complete product-level World Partition feature set.
    FullWorldPartition,
}

impl WorldPartitionCapability {
    /// Returns whether this runtime boundary implements the capability.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        matches!(
            self,
            Self::CellStreaming | Self::CellReload | Self::CellEviction | Self::TraceReplay
        )
    }

    /// Fails closed for capabilities outside this vertical slice.
    pub fn require(self) -> Result<(), WorldPartitionCapabilityError> {
        if self.is_supported() {
            Ok(())
        } else {
            Err(WorldPartitionCapabilityError::Unsupported { capability: self })
        }
    }
}

/// Structured failure for a capability not implemented by this runtime slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldPartitionCapabilityError {
    /// The requested capability is outside the implemented cell-runtime seam.
    Unsupported {
        /// Capability that was requested.
        capability: WorldPartitionCapability,
    },
}

impl WorldPartitionCapabilityError {
    /// Returns the capability that was rejected.
    #[must_use]
    pub const fn capability(self) -> WorldPartitionCapability {
        match self {
            Self::Unsupported { capability } => capability,
        }
    }
}

impl fmt::Display for WorldPartitionCapabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { capability } => {
                write!(
                    formatter,
                    "unsupported World Partition capability: {capability:?}"
                )
            }
        }
    }
}

impl core::error::Error for WorldPartitionCapabilityError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implemented_cell_runtime_capabilities_are_admitted() {
        for capability in [
            WorldPartitionCapability::CellStreaming,
            WorldPartitionCapability::CellReload,
            WorldPartitionCapability::CellEviction,
            WorldPartitionCapability::TraceReplay,
        ] {
            assert!(capability.is_supported());
            assert_eq!(capability.require(), Ok(()));
        }
    }

    #[test]
    fn broader_world_partition_capabilities_fail_with_the_exact_variant() {
        let capability = WorldPartitionCapability::DataLayers;
        let error = capability
            .require()
            .expect_err("capability is not implemented");
        assert_eq!(error.capability(), capability);
        assert_eq!(
            error,
            WorldPartitionCapabilityError::Unsupported { capability }
        );
    }
}
