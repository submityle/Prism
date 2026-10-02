//! CPU-side model for a precompiled pipeline-state-object (PSO) cache.
//!
//! Runtime shader/pipeline compilation is the AAA "shader comp stutter"
//! problem: the first time a material × pass × render-state combination is
//! drawn, the driver compiles its pipeline, dropping a frame. The fix is to
//! precompile ("warm") the needed pipelines ahead of time and persist the
//! driver's pipeline binaries across sessions so later launches skip
//! compilation entirely.
//!
//! This module models the deterministic, backend-agnostic core of that system,
//! leaving actual pipeline compilation, threading, and disk/driver I/O to the
//! engine-side consumer (consistent with this crate's "contracts, not backend"
//! charter). It is split into focused submodules:
//!
//! - [`fingerprint`] — [`DeviceFingerprint`] gating reuse of a persisted cache
//!   against the current device/driver/`ABI`.
//! - [`warm_set`] — [`WarmSetPlanner`] turning static requirements plus runtime
//!   miss feedback into a deterministic, prioritized [`WarmSetPlan`].
//! - [`eviction`] — [`LruPsoCache`], a byte-budgeted LRU residency model.
//! - [`warm_enumerate`] — [`PackageWarmSpec`] deriving warm requests from a
//!   package's [`PermutationSpace`](super::PermutationSpace) × render states.
//!
//! Pipelines are addressed by a [`PsoCacheKey`] — the shader package, its dense
//! permutation index, and a hash of the fixed-function / render-target state
//! the pipeline is compiled against.

mod eviction;
mod fingerprint;
mod warm_enumerate;
mod warm_set;

pub use eviction::{AdmitOutcome, LruPsoCache};
pub use fingerprint::{DeviceFingerprint, FingerprintMismatch, GraphicsBackend};
pub use warm_enumerate::{PackageWarmSpec, WarmEnumerationError};
pub use warm_set::{WarmPriority, WarmRequest, WarmSetPlan, WarmSetPlanner};

use super::ShaderPackageId;

/// Hash of the fixed-function and render-target state a pipeline is compiled
/// against (blend, depth/stencil, rasterizer, attachment formats, …).
///
/// The same shader permutation yields distinct pipelines under different render
/// state, so the state hash is part of a pipeline's cache identity.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PipelineStateHash(pub u64);

/// Fully-qualified identity of a single pipeline-state object.
///
/// Combines the owning shader package, the dense permutation index within that
/// package's [`PermutationSpace`](super::PermutationSpace), and the render-state
/// hash. The ordering derives field-by-field so the key is a deterministic
/// `BTreeMap`/`BTreeSet` key and warm/eviction order is reproducible.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PsoCacheKey {
    /// Shader package the pipeline belongs to.
    pub package: ShaderPackageId,
    /// Dense permutation index within the package's permutation space.
    pub permutation_index: u64,
    /// Hash of the render state the pipeline is compiled against.
    pub state: PipelineStateHash,
}

impl PsoCacheKey {
    /// Builds a key from its components.
    #[must_use]
    pub fn new(package: ShaderPackageId, permutation_index: u64, state: PipelineStateHash) -> Self {
        Self {
            package,
            permutation_index,
            state,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_orders_by_package_then_permutation_then_state() {
        let a = PsoCacheKey::new(ShaderPackageId::new("a"), 0, PipelineStateHash(0));
        let a2 = PsoCacheKey::new(ShaderPackageId::new("a"), 0, PipelineStateHash(1));
        let a3 = PsoCacheKey::new(ShaderPackageId::new("a"), 1, PipelineStateHash(0));
        let b = PsoCacheKey::new(ShaderPackageId::new("b"), 0, PipelineStateHash(0));
        assert!(a < a2);
        assert!(a2 < a3);
        assert!(a3 < b);
    }
}
