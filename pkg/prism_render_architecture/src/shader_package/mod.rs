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

mod artifact;
mod execution;
mod manifest;
mod module;
mod permutation;
mod pso_cache;
mod registry;
mod state;

pub use artifact::{ArtifactDigest, ArtifactValidationError, VerifiedShaderArtifact};
pub use execution::{ExecutionFamily, PipelineIdentity};
pub use module::{InternedModule, ModuleId, ModuleIdentity, ModuleInternError, ModuleInterner};
pub use manifest::{ManifestExpectation, RejectionReason, ValidationOutcome};
pub use permutation::{PermutationError, PermutationIndex, PermutationKey, PermutationSelector, PermutationSpace};
pub use pso_cache::{
    AdmitOutcome, DeviceFingerprint, DrawDecision, FingerprintMismatch, GraphicsBackend,
    LruPsoCache, PackageWarmSpec, PersistError, PersistLoadError, PersistedPipeline,
    PersistedPsoCache, PipelineReadiness, PipelineStateHash, PsoCacheKey, WarmCounters,
    WarmEnumerationError, WarmPriority, WarmRequest, WarmRuntime, WarmSetPlan, WarmSetPlanner,
};
pub use registry::{RegisterError, RegistryEntry, ShaderPackageRegistry};
pub use state::TransitionError;

/// Process-interned package identifier. Text is retained only at the cold seam;
/// equality and hashing use the numeric handle. Cold registry iteration retains
/// lexical ordering for reproducible manifests; hot pipeline keys use ModuleId.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ShaderPackageId(usize);

impl Ord for ShaderPackageId {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        if self == other { return core::cmp::Ordering::Equal; }
        self.as_str().cmp(other.as_str())
    }
}

impl PartialOrd for ShaderPackageId {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> { Some(self.cmp(other)) }
}

#[derive(Default)]
struct PackageNames {
    by_name: std::collections::BTreeMap<&'static str, ShaderPackageId>,
    names: Vec<&'static str>,
}

fn package_names() -> &'static std::sync::Mutex<PackageNames> {
    static NAMES: std::sync::OnceLock<std::sync::Mutex<PackageNames>> = std::sync::OnceLock::new();
    NAMES.get_or_init(Default::default)
}

impl ShaderPackageId {
    /// Creates an identifier from anything string-like.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        let mut names = package_names().lock().unwrap_or_else(|error| error.into_inner());
        if let Some(existing) = names.by_name.get(id.as_str()) { return *existing; }
        // Package names intentionally live as long as the process. This avoids
        // a lock or reference count in every hot key comparison and copy.
        let text = Box::leak(id.into_boxed_str());
        let handle = Self(names.names.len());
        names.names.push(text);
        names.by_name.insert(text, handle);
        handle
    }

    /// Borrows the identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        package_names().lock().unwrap_or_else(|error| error.into_inner()).names[self.0]
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
