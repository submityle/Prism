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
/// Mature-library-backed source importers and canonical runtime artifacts.
pub mod import;
/// Processor contracts and build-node declarations.
pub mod job;
/// Deterministic bridge from build manifests/CAS blobs to runtime containers.
pub mod package;
/// Deterministic diff and byte-accounting plans between build revisions.
pub mod patch;
/// Deterministic, schema-checked build profile composition and projections.
pub mod profile;
/// Immutable binding between a build revision and its runtime package outputs.
pub mod publication;
/// Explicit importer/cook recipe identity.
pub mod recipe;
/// Final release integrity and anti-rollback gate.
pub mod release;
/// Declared CPU, memory, I/O and device admission budgets.
pub mod resources;
/// Deterministic incremental runtime package generation and patch binding.
pub mod runtime_patch;
/// Portable shader-package asset references and complete variant manifests.
pub mod shader_package;
/// Immutable source snapshots and source identity.
pub mod source;
/// Revision manifests and atomic local publication.
pub mod transaction;
/// Stable IDs, hashes, descriptors, and manifest protocol types.
pub mod types;
/// Independent release validation for manifests and CAS blobs.
pub mod validation;
/// Canonical virtual-geometry artifact validation and encoding.
pub mod virtual_geometry;
/// Deterministic static mesh clusterization and page packing.
pub mod virtual_geometry_builder;
/// QEM-backed static virtual-geometry LOD chain construction.
pub mod virtual_geometry_lod;
/// Versioned cluster-local position quantization payload codec.
pub mod virtual_geometry_payload;

pub use cas::{CasError, CasSweepReport, ContentAddressedStore};
pub use ddc::{Ddc, DdcCapacity, DdcError, DdcEviction, DerivedDataCache, LocalDdc};
pub use executor::{
    ArtifactSpec, BuildExecution, BuildExecutionError, BuildExecutor, DeterminismDifference,
};
pub use graph::{BuildError, BuildGraph, BuildOutputs};
pub use hash::{derive_artifact_key, ArtifactKeyInputs, HashError};
pub use job::{BuildJob, JobId, Processor, ProcessorError};
pub use package::{
    package_runtime, package_runtime_selected_with_options, package_runtime_with_options,
    AssetPackageMapping, PackageError, PackageOptions, PackageRequest, RuntimePackage,
};
pub use patch::{plan_patch, PatchChange, PatchEntry, PatchPlan, PatchPlanError};
pub use profile::{
    BuildProfile, CapabilityCandidate, ProfileError, ProfileField, ProfileFieldType, ProfileHash,
    ProfileLayer, ProfileMergePolicy, ProfileOverridePolicy, ProfileValue,
};
pub use publication::{PublicationDescriptor, PublishedContainer};
pub use recipe::{ImportPlan, ImportProcessor, ImportRecipe, ImportRecipeError};
pub use release::{
    ReleaseDescriptor, ReleaseError, ReleaseGate, ReleaseKeyId, ReleaseKeyStatus, ReleasePublicKey,
    ReleaseSignatureError, ReleaseSigner, ReleaseTrustError, ReleaseTrustStore,
    ReleaseVerification, ReleaseVerifier, SignedReleaseDescriptor, TrustPolicy,
};
pub use resources::{BuildResources, ResourceError};
pub use runtime_patch::{
    package_runtime_patch, package_runtime_patch_with_options, RuntimePatchError,
    RuntimePatchPackage, RuntimePatchRequest,
};
pub use shader_package::{ShaderPackageAsset, ShaderPackageAssetError};
pub use source::{SourceError, SourceSnapshot};
pub use transaction::{
    PublishError, PublishTransaction, PublishedManifest, RevisionStore, TransactionState,
};
pub use types::{
    ArtifactDescriptor, ArtifactKey, ArtifactStorageClass, AssetId, BuildRevision, ContentHash,
    Manifest, ManifestEntry, TransactionId,
};
pub use validation::{ArtifactQualityGate, ValidationError, ValidationPolicy, ValidationReport};
pub use virtual_geometry::{
    CanonicalVirtualGeometryProcessor, VirtualGeometryArtifact, VIRTUAL_GEOMETRY_ARTIFACT_KIND,
    VIRTUAL_GEOMETRY_ARTIFACT_SCHEMA,
};
pub use virtual_geometry_builder::{
    build_static_virtual_geometry, StaticClusterBuildError, StaticClusterBuildSettings,
    STATIC_CLUSTER_PAYLOAD_MAGIC, STATIC_CLUSTER_PAYLOAD_VERSION,
};
pub use virtual_geometry_lod::{
    build_static_virtual_geometry_lod_chain, build_static_virtual_geometry_lod_chain_with_policy,
    LodAttributeMetric, LodAttributePolicy, StaticLodPolicy, StaticVirtualGeometryLod,
    StaticVirtualGeometryLodError, StaticVirtualGeometryLodSet, STATIC_LOD_ALGORITHM_VERSION,
};
pub use virtual_geometry_payload::{
    cluster_position_error, cluster_position_error_squared, decode_cluster_payload,
    encode_cluster_payload, encode_cluster_payload_with_attributes, ClusterPayloadError,
    ClusterVertexAttribute, QuantizedClusterPayload, MAX_CLUSTER_ATTRIBUTES,
    MAX_CLUSTER_ATTRIBUTE_NAME_BYTES, POSITION_ONLY_CLUSTER_PAYLOAD_VERSION,
    STATIC_CLUSTER_PAYLOAD_MAGIC as QUANTIZED_CLUSTER_PAYLOAD_MAGIC,
    STATIC_CLUSTER_PAYLOAD_VERSION as QUANTIZED_CLUSTER_PAYLOAD_VERSION,
};
