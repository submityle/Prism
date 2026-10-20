//! Frozen editable payloads cooked through the ordinary DAG/DDC/CAS pipeline.
//!
//! Source records belong to import/reimport. Cooking consumes the saved editing
//! representation, so edits cannot be silently replaced by reimporting a source.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use prism_asset::StableGuid;

use crate::editable::{
    EditableAsset, EditableAssetError, EditableDependency, EditableDependencyKind,
    EditableSubAsset, EditableSubAssetReference, EditableWorkspace,
    MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
};
use crate::{
    package_runtime_with_options, ArtifactKey, ArtifactSpec, AssetId, AssetPackageMapping,
    BuildExecution, BuildExecutionError, BuildExecutor, BuildGraph, BuildInputSnapshot, BuildJob,
    BuildResources, BuildRevision, ContentAddressedStore, DerivedDataCache, JobId, PackageError,
    PackageOptions, PackageRequest, ProcessorError, ProcessorIdentity, RevisionStore,
    RuntimePackage,
};

const PROTOCOL_VERSION: u32 = 1;
const MAX_ASSETS: usize = 65_536;
const MAX_EDGES: usize = 1_048_576;
const MAX_COOKERS: usize = 1024;
const MAX_FINGERPRINT_BYTES: usize = 1024 * 1024;
const MAX_OUTPUT_BYTES: u64 = 256 * 1024 * 1024;
const MAX_IMAGE_OUTPUT_BYTES: u64 = 32 * 1024 * 1024;

/// All semantic fields visible to a cooker. Source and preview are excluded.
/// Editor-only dependencies/references are removed before this input is frozen.
#[derive(Debug)]
pub struct EditableCookInput {
    pub asset: AssetId,
    pub kind: u128,
    pub flags: u32,
    pub canonical_settings: Vec<u8>,
    pub subassets: Vec<EditableSubAsset>,
    pub dependencies: Vec<EditableDependency>,
    pub subasset_references: Vec<EditableSubAssetReference>,
    pub authoring_payload: Vec<u8>,
}

/// Borrowed completed artifact of one hard dependency, in ascending AssetId order.
pub struct EditableCookDependency<'a> {
    pub asset: AssetId,
    pub bytes: &'a [u8],
}

/// Registration-time output contract. Resource limits bound declared demand,
/// not arbitrary allocations performed inside caller-owned cooker code.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditableCookerDescriptor {
    pub authoring_kind: u128,
    pub identity: ProcessorIdentity,
    pub output_kind: u128,
    pub output_schema: u32,
    pub alignment: u32,
    /// Per-artifact encoded output limit, at most 256 MiB. This is checked
    /// after caller code returns; it is not a heap or aggregate-output budget.
    pub max_output_bytes: u64,
    pub resources: BuildResources,
}

/// Deterministic transformation of saved editing data. Implementations must
/// depend only on input, ordered hard artifacts and captured immutable settings.
/// Ambient files, clocks and environment reads violate the cache contract.
pub trait EditableCooker: Send + Sync + 'static {
    fn descriptor(&self) -> EditableCookerDescriptor;
    fn cook(
        &self,
        input: &EditableCookInput,
        dependencies: &[EditableCookDependency<'_>],
    ) -> Result<Vec<u8>, ProcessorError>;
}

struct Registration {
    descriptor: EditableCookerDescriptor,
    cooker: Arc<dyn EditableCooker>,
}

/// Caller-owned registry with one frozen contract per authoring kind.
#[derive(Default)]
pub struct EditableCookerRegistry {
    registrations: BTreeMap<u128, Registration>,
}

impl EditableCookerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers the canonical RGBA8 mip-chain image cooker.
    pub fn with_builtin() -> Self {
        let mut registry = Self::new();
        registry
            .register(CanonicalEditableImageCooker)
            .expect("built-in editable image cooker has a valid unique contract");
        registry
    }

    pub fn register<C: EditableCooker>(&mut self, cooker: C) -> Result<(), EditableBuildError> {
        let descriptor = cooker.descriptor();
        if descriptor.authoring_kind == 0
            || descriptor.identity.id == 0
            || descriptor.identity.version == 0
            || descriptor.output_kind == 0
            || descriptor.output_schema == 0
            || !descriptor.alignment.is_power_of_two()
            || descriptor.resources.cpu_threads == 0
            || descriptor.max_output_bytes == 0
            || descriptor.max_output_bytes > MAX_OUTPUT_BYTES
        {
            return Err(EditableBuildError::InvalidCooker);
        }
        if self.registrations.contains_key(&descriptor.authoring_kind)
            || self
                .registrations
                .values()
                .any(|registered| registered.descriptor.identity.id == descriptor.identity.id)
        {
            return Err(EditableBuildError::DuplicateCooker);
        }
        if self.registrations.len() >= MAX_COOKERS {
            return Err(EditableBuildError::Limit("cooker registrations"));
        }
        self.registrations.insert(
            descriptor.authoring_kind,
            Registration {
                descriptor,
                cooker: Arc::new(cooker),
            },
        );
        Ok(())
    }
}

/// Explicit build environment. Fingerprints are never read from ambient state.
#[derive(Clone, Debug)]
pub struct EditableBuildOptions {
    pub profile_hash: [u8; 32],
    /// Explicit canonical bytes, hashed once before graph construction. Nodes
    /// retain a fixed-size domain-separated digest rather than these bytes.
    pub toolchain_fingerprint: Vec<u8>,
    /// Explicit canonical bytes, with the same bounded storage as the toolchain.
    pub environment_fingerprint: Vec<u8>,
    /// Positive encoded snapshot budget, at most 512 MiB.
    pub max_snapshot_bytes: u64,
}

impl Default for EditableBuildOptions {
    fn default() -> Self {
        Self {
            profile_hash: [0; 32],
            toolchain_fingerprint: Vec::new(),
            environment_fingerprint: Vec::new(),
            max_snapshot_bytes: MAX_EDITABLE_WORKSPACE_ENCODED_BYTES,
        }
    }
}

/// Immutable build closure. Payload allocations move from the snapshot into
/// Arc-owned jobs; graph fan-out borrows dependency outputs without copying.
pub struct EditableBuildPlan {
    graph: BuildGraph,
    specs: BTreeMap<JobId, ArtifactSpec>,
    inputs: BuildInputSnapshot,
    hard: BTreeMap<AssetId, Vec<AssetId>>,
    soft: BTreeMap<AssetId, Vec<AssetId>>,
    output_limits: BTreeMap<AssetId, u64>,
    artifact_keys: BTreeMap<JobId, ArtifactKey>,
}

impl EditableBuildPlan {
    /// Captures a bounded coherent workspace under its publication lock, then
    /// releases the lock before graph planning and all cooker execution.
    pub fn from_workspace(
        workspace: &EditableWorkspace,
        roots: &[AssetId],
        registry: &EditableCookerRegistry,
        options: &EditableBuildOptions,
    ) -> Result<Self, EditableBuildError> {
        validate_options(options)?;
        Self::from_snapshot(
            workspace
                .snapshot(options.max_snapshot_bytes)
                .map_err(EditableBuildError::Editable)?,
            roots,
            registry,
            options,
        )
    }

    /// Selects only the hard dependency closure of roots. Missing soft targets
    /// are retained as metadata; editor-only targets do not enter the build.
    pub fn from_snapshot(
        mut snapshot: BTreeMap<AssetId, EditableAsset>,
        roots: &[AssetId],
        registry: &EditableCookerRegistry,
        options: &EditableBuildOptions,
    ) -> Result<Self, EditableBuildError> {
        validate_options(options)?;
        if roots.is_empty() {
            return Err(EditableBuildError::EmptyRoots);
        }
        if snapshot.len() > MAX_ASSETS || roots.len() > MAX_ASSETS {
            return Err(EditableBuildError::Limit("assets or roots"));
        }
        let mut total = 0u64;
        for (id, asset) in &snapshot {
            if id.0 == 0 || *id != asset.asset {
                return Err(EditableBuildError::InvalidIdentity(*id));
            }
            total = total
                .checked_add(
                    asset
                        .validated_encoded_len()
                        .map_err(EditableBuildError::Editable)? as u64,
                )
                .ok_or(EditableBuildError::Limit("snapshot bytes"))?;
            if total > options.max_snapshot_bytes {
                return Err(EditableBuildError::Limit("snapshot bytes"));
            }
        }
        let mut pending = roots.iter().copied().collect::<BTreeSet<_>>();
        let mut hard = BTreeMap::new();
        let mut soft = BTreeMap::new();
        let mut edges = 0usize;
        while let Some(id) = pending.pop_first() {
            if hard.contains_key(&id) {
                continue;
            }
            if id.0 == 0 {
                return Err(EditableBuildError::InvalidIdentity(id));
            }
            let asset = snapshot
                .get(&id)
                .ok_or(EditableBuildError::MissingAsset(id))?;
            let mut hard_ids = BTreeSet::new();
            let mut soft_ids = BTreeSet::new();
            for (target, kind) in asset
                .dependencies
                .iter()
                .map(|d| (d.asset, d.kind))
                .chain(asset.subasset_references.iter().map(|r| (r.asset, r.kind)))
            {
                if kind == EditableDependencyKind::EditorOnly {
                    continue;
                }
                if target.0 == 0 {
                    return Err(EditableBuildError::InvalidIdentity(target));
                }
                edges += 1;
                if edges > MAX_EDGES {
                    return Err(EditableBuildError::Limit("dependency edges"));
                }
                // A reference to a subasset in this same package is resolved
                // below, but is not an artifact dependency on its own output.
                if target == id {
                    continue;
                }
                match kind {
                    EditableDependencyKind::Hard => {
                        if !snapshot.contains_key(&target) {
                            return Err(EditableBuildError::MissingAsset(target));
                        }
                        hard_ids.insert(target);
                    }
                    EditableDependencyKind::Soft => {
                        soft_ids.insert(target);
                    }
                    EditableDependencyKind::EditorOnly => unreachable!(),
                }
            }
            for reference in &asset.subasset_references {
                if reference.kind == EditableDependencyKind::EditorOnly {
                    continue;
                }
                if let Some(target) = snapshot.get(&reference.asset) {
                    if target
                        .subassets
                        .binary_search_by_key(&reference.subasset, |s| s.id)
                        .is_err()
                    {
                        return Err(EditableBuildError::MissingSubAsset {
                            asset: reference.asset,
                            subasset: reference.subasset,
                        });
                    }
                }
            }
            soft_ids.retain(|id| !hard_ids.contains(id));
            pending.extend(hard_ids.iter().copied().filter(|id| !hard.contains_key(id)));
            hard.insert(id, hard_ids.into_iter().collect::<Vec<_>>());
            soft.insert(id, soft_ids.into_iter().collect::<Vec<_>>());
        }

        let mut graph = BuildGraph::new();
        let mut specs = BTreeMap::new();
        let mut inputs = BuildInputSnapshot::new();
        let mut output_limits = BTreeMap::new();
        // Hash each potentially large environment record only once. The graph
        // may contain 65,536 nodes, so copying the allowed 1 MiB records into
        // every spec would otherwise amplify a small snapshot into 128 GiB.
        let toolchain = blake3::derive_key(
            "Prism editable cook toolchain fingerprint v1",
            &options.toolchain_fingerprint,
        );
        let environment = blake3::derive_key(
            "Prism editable cook environment fingerprint v1",
            &options.environment_fingerprint,
        );
        for (id, dependencies) in &hard {
            let asset = snapshot.remove(id).expect("closure was validated");
            let registration = registry.registrations.get(&asset.kind).ok_or(
                EditableBuildError::MissingCooker {
                    asset: *id,
                    kind: asset.kind,
                },
            )?;
            let contract = &registration.descriptor;
            output_limits.insert(*id, contract.max_output_bytes);
            let input = Arc::new(EditableCookInput {
                asset: asset.asset,
                kind: asset.kind,
                flags: asset.flags,
                canonical_settings: asset.canonical_settings,
                subassets: asset.subassets,
                dependencies: asset
                    .dependencies
                    .into_iter()
                    .filter(|d| d.kind != EditableDependencyKind::EditorOnly)
                    .collect(),
                subasset_references: asset
                    .subasset_references
                    .into_iter()
                    .filter(|r| r.kind != EditableDependencyKind::EditorOnly)
                    .collect(),
                authoring_payload: asset.authoring_payload,
            });
            let mut spec = ArtifactSpec {
                asset: *id,
                processor_id: contract.identity.id,
                processor_version: contract.identity.version,
                canonical_settings: semantic_hash(&input, contract.max_output_bytes).to_vec(),
                profile_hash: [0; 32],
                toolchain_fingerprint: toolchain.to_vec(),
                environment_fingerprint: environment.to_vec(),
                kind: contract.output_kind,
                schema: contract.output_schema,
                alignment: contract.alignment,
            };
            let job = JobId(id.0);
            inputs.set_settings(
                job,
                spec.artifact_key(PROTOCOL_VERSION, &[])
                    .map_err(EditableBuildError::Hash)?
                    .0
                    .to_vec(),
            );
            spec.profile_hash = options.profile_hash;
            inputs.set_profile_hash(job, options.profile_hash);
            let dependency_ids = dependencies.clone();
            let cooker = Arc::clone(&registration.cooker);
            let max_output_bytes = contract.max_output_bytes;
            graph
                .add_job(
                    BuildJob::new(
                        job,
                        dependencies
                            .iter()
                            .map(|id| JobId(id.0))
                            .collect::<Vec<_>>(),
                        move |bytes: &[&[u8]]| {
                            let dependencies = dependency_ids
                                .iter()
                                .zip(bytes)
                                .map(|(asset, bytes)| EditableCookDependency {
                                    asset: *asset,
                                    bytes,
                                })
                                .collect::<Vec<_>>();
                            cooker.cook(&input, &dependencies)
                        },
                    )
                    .with_resources(contract.resources)
                    .with_max_output_bytes(max_output_bytes),
                )
                .map_err(EditableBuildError::Graph)?;
            specs.insert(job, spec);
        }
        let mut artifact_keys = BTreeMap::new();
        for job in graph.execution_order().map_err(EditableBuildError::Graph)? {
            let dependencies = graph
                .job(job)
                .expect("ordered job exists")
                .dependencies()
                .iter()
                .map(|dependency| artifact_keys[dependency])
                .collect::<Vec<_>>();
            let key = specs[&job]
                .artifact_key(PROTOCOL_VERSION, &dependencies)
                .map_err(EditableBuildError::Hash)?;
            artifact_keys.insert(job, key);
        }
        Ok(Self {
            graph,
            specs,
            inputs,
            hard,
            soft,
            output_limits,
            artifact_keys,
        })
    }

    pub fn graph(&self) -> &BuildGraph {
        &self.graph
    }
    pub fn specs(&self) -> &BTreeMap<JobId, ArtifactSpec> {
        &self.specs
    }
    pub fn inputs(&self) -> &BuildInputSnapshot {
        &self.inputs
    }

    /// Executes with durable invalidation explanations and the ordinary
    /// all-or-nothing Revision publication. Failed builds may leave valid cache.
    /// Every cache read and fresh output is checked against the frozen job's
    /// output limit, including entries observed after DDC index refreshes.
    /// Determinism verification checks two bounded fresh executions.
    pub fn execute(
        &self,
        store: &RevisionStore,
        ddc: &mut DerivedDataCache,
        revision: BuildRevision,
        max_concurrency: usize,
        budget: BuildResources,
        verify_determinism: bool,
    ) -> Result<BuildExecution, EditableBuildError> {
        BuildExecutor {
            protocol_version: PROTOCOL_VERSION,
        }
        .execute_explained_with_resources(
            &self.graph,
            &self.specs,
            &self.inputs,
            store,
            ddc,
            revision,
            max_concurrency,
            budget,
            verify_determinism,
        )
        .map_err(EditableBuildError::Execution)
    }

    /// Packages a matching execution, preserving soft routes even when targets
    /// are absent. Stable runtime GUIDs preserve every AssetId bit.
    pub fn package(
        &self,
        execution: &BuildExecution,
        cas: &ContentAddressedStore,
        container_name: &str,
        options: PackageOptions,
    ) -> Result<RuntimePackage, EditableBuildError> {
        let manifest = &execution.published.manifest;
        if manifest.entries.len() != self.specs.len()
            || execution.artifact_keys.len() != self.specs.len()
        {
            return Err(EditableBuildError::ManifestMismatch);
        }
        for (job, spec) in &self.specs {
            if execution.artifact_keys.get(job)
                != manifest
                    .entries
                    .get(&spec.asset)
                    .map(|descriptor| &descriptor.key)
            {
                return Err(EditableBuildError::ManifestMismatch);
            }
        }
        self.package_manifest(manifest, cas, container_name, options)
    }

    /// Packages a committed manifest only when every entry matches this frozen
    /// authoring closure and output contract. This performs no cooking and
    /// preserves optional references without adding them to the build DAG.
    /// CAS contents are verified by the ordinary runtime package bridge.
    pub fn package_manifest(
        &self,
        manifest: &crate::Manifest,
        cas: &ContentAddressedStore,
        container_name: &str,
        options: PackageOptions,
    ) -> Result<RuntimePackage, EditableBuildError> {
        if manifest.entries.len() != self.specs.len() {
            return Err(EditableBuildError::ManifestMismatch);
        }
        for spec in self.specs.values() {
            let descriptor = manifest
                .entries
                .get(&spec.asset)
                .ok_or(EditableBuildError::ManifestMismatch)?;
            let dependencies = self.hard[&spec.asset]
                .iter()
                .map(|id| self.artifact_keys[&JobId(id.0)])
                .collect::<Vec<_>>();
            let key = self.artifact_keys[&JobId(spec.asset.0)];
            if descriptor.key != key
                || descriptor.kind != spec.kind
                || descriptor.schema != spec.schema
                || descriptor.alignment != spec.alignment
                || descriptor.dependencies != dependencies
                || descriptor.byte_len > self.output_limits[&spec.asset]
            {
                return Err(EditableBuildError::ManifestMismatch);
            }
        }
        let mappings = self
            .specs
            .values()
            .map(|spec| AssetPackageMapping {
                asset: spec.asset,
                guid: StableGuid::from_u128(spec.asset.0),
                container_name: container_name.to_owned(),
                priority: 0,
                chunk_priority: 0,
            })
            .collect();
        let mut package = package_runtime_with_options(
            PackageRequest {
                manifest,
                cas,
                mappings,
            },
            options,
        )
        .map_err(EditableBuildError::Package)?;
        for (id, targets) in &self.soft {
            let guid = StableGuid::from_u128(id.0);
            let mut metadata = package
                .runtime_manifest
                .metadata(guid)
                .ok_or(EditableBuildError::ManifestMismatch)?
                .clone();
            metadata.optional_dependencies = targets
                .iter()
                .map(|id| StableGuid::from_u128(id.0))
                .collect();
            package
                .runtime_manifest
                .set_metadata(guid, metadata)
                .map_err(|e| EditableBuildError::Package(PackageError::RuntimeManifest(e)))?;
        }
        package.runtime_manifest_bytes = package
            .runtime_manifest
            .encode()
            .map_err(|e| EditableBuildError::Package(PackageError::RuntimeManifest(e)))?;
        Ok(package)
    }
}

fn validate_options(options: &EditableBuildOptions) -> Result<(), EditableBuildError> {
    if options.max_snapshot_bytes == 0
        || options.max_snapshot_bytes > MAX_EDITABLE_WORKSPACE_ENCODED_BYTES
    {
        return Err(EditableBuildError::Limit("snapshot byte budget"));
    }
    if options.toolchain_fingerprint.len() > MAX_FINGERPRINT_BYTES
        || options.environment_fingerprint.len() > MAX_FINGERPRINT_BYTES
    {
        return Err(EditableBuildError::Limit("fingerprint bytes"));
    }
    Ok(())
}

fn dependency_tag(kind: EditableDependencyKind) -> u8 {
    match kind {
        EditableDependencyKind::Hard => 0,
        EditableDependencyKind::Soft => 1,
        EditableDependencyKind::EditorOnly => unreachable!("editor-only input was filtered"),
    }
}

fn semantic_hash(input: &EditableCookInput, max_output_bytes: u64) -> [u8; 32] {
    fn bytes(hash: &mut blake3::Hasher, value: &[u8]) {
        hash.update(&(value.len() as u64).to_le_bytes());
        hash.update(value);
    }
    let mut hash = blake3::Hasher::new_derive_key("Prism editable cook input v1");
    // Admission is part of the output contract, including for warm DDC hits.
    hash.update(&max_output_bytes.to_le_bytes());
    hash.update(&input.asset.0.to_le_bytes());
    hash.update(&input.kind.to_le_bytes());
    hash.update(&input.flags.to_le_bytes());
    bytes(&mut hash, &input.canonical_settings);
    hash.update(&(input.subassets.len() as u64).to_le_bytes());
    for subasset in &input.subassets {
        hash.update(&subasset.id.to_le_bytes());
        bytes(&mut hash, subasset.name.as_bytes());
        hash.update(&subasset.kind.to_le_bytes());
    }
    hash.update(&(input.dependencies.len() as u64).to_le_bytes());
    for dependency in &input.dependencies {
        hash.update(&dependency.asset.0.to_le_bytes());
        hash.update(&[dependency_tag(dependency.kind)]);
    }
    hash.update(&(input.subasset_references.len() as u64).to_le_bytes());
    for reference in &input.subasset_references {
        hash.update(&reference.asset.0.to_le_bytes());
        hash.update(&reference.subasset.to_le_bytes());
        hash.update(&[dependency_tag(reference.kind)]);
    }
    bytes(&mut hash, &input.authoring_payload);
    *hash.finalize().as_bytes()
}

/// Validates a complete semantic RGBA8 mip-chain and emits its canonical
/// runtime representation. Import/reimport owns mip generation and pixel edits;
/// no external or embedded source is read by this cooker.
pub struct CanonicalEditableImageCooker;

impl EditableCooker for CanonicalEditableImageCooker {
    fn descriptor(&self) -> EditableCookerDescriptor {
        EditableCookerDescriptor {
            authoring_kind: crate::import::IMAGE_ARTIFACT_KIND,
            // Version 2 rejects inconsistent image headers. A version-1 cache
            // entry must not bypass that validation on a warm build.
            identity: ProcessorIdentity::new(0x5052_4544_4954_494d_4147_4543_4f4f_4b31, 2),
            output_kind: crate::import::IMAGE_ARTIFACT_KIND,
            output_schema: prism_image::IMAGE_MIP_ARTIFACT_SCHEMA,
            alignment: 4,
            max_output_bytes: MAX_IMAGE_OUTPUT_BYTES,
            resources: BuildResources {
                memory_bytes: 128 * 1024 * 1024,
                ..BuildResources::default()
            },
        }
    }

    fn cook(
        &self,
        input: &EditableCookInput,
        _: &[EditableCookDependency<'_>],
    ) -> Result<Vec<u8>, ProcessorError> {
        // This cooker preserves validated wire bytes, so their length is also
        // the output length. Reject before decoding allocates any mip pixels.
        if input.authoring_payload.len() as u64 > MAX_IMAGE_OUTPUT_BYTES {
            return Err(ProcessorError::new(
                "editable image payload exceeds declared output byte bound",
            ));
        }
        let image = prism_image::ImageMipArtifact::decode(&input.authoring_payload)
            .map_err(|e| ProcessorError::new(e.to_string()))?;
        // The image decoder validates the level chain independently of the
        // top-level dimensions. Runtime consumers must see a consistent header.
        if image
            .levels
            .first()
            .is_none_or(|level| level.width != image.width || level.height != image.height)
        {
            return Err(ProcessorError::new(
                "editable image header dimensions differ from the first mip",
            ));
        }
        // Decoding checks every level and trailing byte. These validated wire
        // bytes are canonical; avoid encoding and cloning the mip chain again.
        // Release decoded pixels before allocating the one owned output copy.
        drop(image);
        Ok(input.authoring_payload.clone())
    }
}

/// Planning, execution or publication mismatch before package delivery.
#[derive(Debug)]
pub enum EditableBuildError {
    Editable(EditableAssetError),
    InvalidCooker,
    DuplicateCooker,
    Limit(&'static str),
    InvalidIdentity(AssetId),
    EmptyRoots,
    MissingAsset(AssetId),
    MissingCooker { asset: AssetId, kind: u128 },
    MissingSubAsset { asset: AssetId, subasset: u64 },
    Graph(crate::BuildError),
    Hash(crate::HashError),
    Execution(BuildExecutionError),
    Package(PackageError),
    ManifestMismatch,
}

impl fmt::Display for EditableBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Editable(e) => e.fmt(f),
            Self::Graph(e) => e.fmt(f),
            Self::Hash(e) => e.fmt(f),
            Self::Execution(e) => e.fmt(f),
            Self::Package(e) => e.fmt(f),
            Self::InvalidCooker => f.write_str("invalid editable cooker contract"),
            Self::DuplicateCooker => f.write_str("duplicate editable kind or processor namespace"),
            Self::Limit(field) => write!(f, "editable build exceeds {field} limit"),
            Self::InvalidIdentity(id) => write!(f, "invalid editable identity {:032x}", id.0),
            Self::EmptyRoots => f.write_str("editable build requires at least one root"),
            Self::MissingAsset(id) => write!(f, "editable asset {:032x} is missing", id.0),
            Self::MissingCooker { asset, kind } => write!(
                f,
                "asset {:032x} has no cooker for kind {kind:032x}",
                asset.0
            ),
            Self::MissingSubAsset { asset, subasset } => {
                write!(f, "asset {:032x} has no subasset {subasset}", asset.0)
            }
            Self::ManifestMismatch => {
                f.write_str("execution does not match frozen editable build plan")
            }
        }
    }
}

impl std::error::Error for EditableBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Editable(e) => Some(e),
            Self::Graph(e) => Some(e),
            Self::Hash(e) => Some(e),
            Self::Execution(e) => Some(e),
            Self::Package(e) => Some(e),
            _ => None,
        }
    }
}
