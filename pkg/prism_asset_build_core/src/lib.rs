//! Protocol primitives and a small, local, transactional asset-build store.
//!
//! Artifact keys describe build semantics; content hashes describe bytes. The
//! store deliberately provides no remote/distributed cache or build execution.

extern crate alloc;

mod journal;

/// Immutable, content-addressed artifact byte storage.
pub mod cas;
/// Persistent local derived-data descriptor index.
pub mod ddc;
/// Executes a build graph through DDC/CAS into transactional publication.
pub mod executor;
/// Deterministic dependency graph and serial processor execution.
pub mod graph;
/// Versioned deterministic artifact-key derivation.
pub mod hash;
/// Processor contracts and build-node declarations.
pub mod job;
/// Deterministic bridge from build manifests/CAS blobs to runtime containers.
pub mod package;
/// Immutable binding between a build revision and its runtime package outputs.
pub mod publication;
/// Revision manifests and atomic local publication.
pub mod transaction;
/// Stable IDs, hashes, descriptors, and manifest protocol types.
pub mod types;
/// Independent release validation for manifests and CAS blobs.
pub mod validation;
/// Canonical virtual-geometry artifact validation and encoding.
pub mod virtual_geometry;

pub use cas::{CasError, CasSweepReport, ContentAddressedStore};
pub use ddc::{Ddc, DdcError, DerivedDataCache, LocalDdc};
pub use executor::{ArtifactSpec, BuildExecution, BuildExecutionError, BuildExecutor};
pub use graph::{BuildError, BuildGraph, BuildOutputs};
pub use hash::{derive_artifact_key, ArtifactKeyInputs, HashError};
pub use job::{BuildJob, JobId, Processor, ProcessorError};
pub use package::{
    package_runtime, package_runtime_with_options, AssetPackageMapping, PackageError,
    PackageOptions, PackageRequest, RuntimePackage,
};
pub use publication::{PublicationDescriptor, PublishedContainer};
pub use transaction::{
    PublishError, PublishTransaction, PublishedManifest, RevisionStore, TransactionState,
};
pub use types::{
    ArtifactDescriptor, ArtifactKey, AssetId, BuildRevision, ContentHash, Manifest, ManifestEntry,
    TransactionId,
};
pub use validation::{ArtifactQualityGate, ValidationError, ValidationPolicy, ValidationReport};
pub use virtual_geometry::{
    CanonicalVirtualGeometryProcessor, VirtualGeometryArtifact, VIRTUAL_GEOMETRY_ARTIFACT_KIND,
    VIRTUAL_GEOMETRY_ARTIFACT_SCHEMA,
};
