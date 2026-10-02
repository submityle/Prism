//! Build-time `HLSL`-to-`SPIR-V` shader package metadata and validation.
//!
//! A *shader package* is the build system's unit of compiled output: a named
//! bundle of `SPIR-V` entry points that were generated against a specific
//! render `ABI`. This module models the metadata and lifecycle around such a
//! package on the `CPU` side, independent of any `GPU` backend:
//!
//! - [`ShaderPackageManifest`] captures the identity, provenance, and `ABI`
//!   binding (`AbiVersion` + `AbiHash`) of a package.
//! - [`PackageState`] is a small lifecycle state machine
//!   (`Missing` -> `Validating` -> `Ready`/`Rejected`) with checked transitions.
//! - [`ShaderPackageRegistry`] indexes manifests by [`ShaderPackageId`] and
//!   tracks each package's current [`PackageState`].
//! - Manifest validation compares a package's `ABI` binding against an
//!   expectation and produces an actionable rejection reason on mismatch.
//! - Permutation utilities count and select shader variants deterministically.
//!
//! Everything here is deterministic and allocation-light so build outputs and
//! golden tests reproduce exactly across runs.

use crate::abi::{AbiHash, AbiVersion};

mod manifest;
mod permutation;
mod registry;
mod state;
mod pso_cache;

pub use manifest::{ManifestExpectation, RejectionReason, ValidationOutcome};
pub use permutation::{PermutationKey, PermutationSelector, PermutationSpace};
pub use registry::{RegisterError, RegistryEntry, ShaderPackageRegistry};
pub use state::TransitionError;
pub use pso_cache::{
    AdmitOutcome, DeviceFingerprint, FingerprintMismatch, GraphicsBackend, LruPsoCache,
    PackageWarmSpec, PipelineStateHash, PsoCacheKey, WarmEnumerationError, WarmPriority,
    WarmRequest, WarmSetPlan, WarmSetPlanner,
};

/// Stable, human-readable identifier for a shader package.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ShaderPackageId(pub String);

impl ShaderPackageId {
    /// Creates an identifier from anything string-like.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrows the identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Identity, provenance, and `ABI` binding for one shader package.
#[derive(Clone, Debug)]
pub struct ShaderPackageManifest {
    pub id: ShaderPackageId,
    pub source_revision: String,
    pub entry_point: String,
    pub abi_version: AbiVersion,
    pub abi_hash: AbiHash,
    pub permutation_count: u32,
}

/// Lifecycle of a shader package as the build system validates it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PackageState {
    Missing,
    Validating,
    Ready,
    Rejected,
}
