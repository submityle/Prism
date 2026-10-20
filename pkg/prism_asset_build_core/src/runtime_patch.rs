//! Build-to-runtime incremental package creation.
//!
//! This module turns a deterministic Build Manifest diff into a real runtime
//! delta: only added/changed payloads are emitted, removed or replaced routes
//! become tombstones, and the resulting descriptor binds both revisions and
//! both runtime-manifest identities.

use std::collections::{BTreeMap, BTreeSet};

use prism_asset::StableGuid;
use prism_asset_container::{
    ManifestError, ManifestMetadata, ManifestRoute, PatchDescriptorError, RuntimeManifest,
    RuntimePatchDescriptor,
};

use crate::cas::ContentAddressedStore;
use crate::package::{
    package_runtime_selected_with_options, validate_package_mappings, AssetPackageMapping,
    PackageError, PackageOptions, PackageRequest, RuntimePackage,
};
use crate::patch::{plan_patch, PatchChange, PatchPlanError};
use crate::provenance::ReleaseProvenance;
use crate::publication::PublicationDescriptor;
use crate::types::{ArtifactKey, AssetId, BuildRevision, ContentHash, Manifest};

/// Inputs required to build a runtime patch from two committed revisions.
pub struct RuntimePatchRequest<'a> {
    /// Previously packaged base revision.
    pub base_manifest: &'a Manifest,
    /// New complete target build manifest.
    pub target_manifest: &'a Manifest,
    /// Runtime package corresponding exactly to `base_manifest`.
    pub base_package: &'a RuntimePackage,
    /// Immutable payload store used for newly introduced artifacts.
    pub cas: &'a ContentAddressedStore,
    /// Complete target mapping table. Changed routes are placed in a patch
    /// namespace; unchanged mappings remain dependency context only.
    pub target_mappings: Vec<AssetPackageMapping>,
}

/// A self-contained runtime patch payload and its application descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimePatchPackage {
    /// Patch/base/target binding and tombstone list.
    pub descriptor: RuntimePatchDescriptor,
    /// Canonical encoded descriptor bytes suitable for publication.
    pub descriptor_bytes: Vec<u8>,
    /// Incremental containers and routes introduced by the patch.
    pub payload: RuntimePackage,
    /// Complete target routing manifest. It carries metadata for dependencies
    /// that remain in the base package without copying their payload bytes.
    pub target_runtime_manifest: RuntimeManifest,
    /// Canonical bytes of [`Self::target_runtime_manifest`].
    pub target_runtime_manifest_bytes: Vec<u8>,
    /// Complete target-package provenance reconstructed from base and patch
    /// container bytes.
    pub target_provenance: ReleaseProvenance,
    /// Canonical bytes of [`Self::target_provenance`].
    pub target_provenance_bytes: Vec<u8>,
}

impl RuntimePatchPackage {
    /// Verifies and applies this patch's routing state to a base manifest.
    ///
    /// The returned manifest is the authoritative target snapshot carried by
    /// the patch. The incremental payload is checked against it first, so a
    /// caller can mount `payload.containers` and then replace its routing
    /// snapshot without silently accepting a mismatched patch.
    pub fn apply_to_manifest(
        &self,
        base_build_manifest_hash: ContentHash,
        base_runtime_manifest: &RuntimeManifest,
    ) -> Result<RuntimeManifest, RuntimePatchError> {
        let decoded = RuntimePatchDescriptor::decode(&self.descriptor_bytes)?;
        if decoded != self.descriptor {
            return Err(RuntimePatchError::DescriptorObjectMismatch);
        }
        if base_build_manifest_hash.0 != self.descriptor.base_build_manifest_hash {
            return Err(RuntimePatchError::BaseBuildManifestHashMismatch);
        }

        let base_runtime_manifest_bytes = base_runtime_manifest
            .encode()
            .map_err(RuntimePatchError::Manifest)?;
        if ContentHash::digest(&base_runtime_manifest_bytes).0
            != self.descriptor.base_runtime_manifest_hash
        {
            return Err(RuntimePatchError::BaseRuntimeManifestHashMismatch);
        }

        let payload_manifest_bytes = self
            .payload
            .runtime_manifest
            .encode()
            .map_err(RuntimePatchError::Manifest)?;
        if payload_manifest_bytes != self.payload.runtime_manifest_bytes
            || ContentHash::digest(&payload_manifest_bytes).0
                != self.descriptor.patch_runtime_manifest_hash
        {
            return Err(RuntimePatchError::PatchManifestMismatch);
        }

        let target_manifest_bytes = self
            .target_runtime_manifest
            .encode()
            .map_err(RuntimePatchError::Manifest)?;
        if target_manifest_bytes != self.target_runtime_manifest_bytes
            || ContentHash::digest(&target_manifest_bytes).0
                != self.descriptor.target_runtime_manifest_hash
        {
            return Err(RuntimePatchError::TargetManifestMismatch);
        }

        validate_target_provenance(
            &self.target_provenance,
            &self.target_provenance_bytes,
            &self.descriptor,
            &self.target_runtime_manifest,
            &self.target_runtime_manifest_bytes,
        )?;

        let mut merged = base_runtime_manifest.clone();
        merged.apply_patch(
            &self.payload.runtime_manifest,
            &self.descriptor.removed_guids,
        )?;
        validate_patch_routes(&merged, &self.target_runtime_manifest)?;

        for route in self.payload.runtime_manifest.routes() {
            let target = self
                .target_runtime_manifest
                .route(route.guid)
                .ok_or(RuntimePatchError::TargetRouteMissing { guid: route.guid })?;
            if target != route {
                return Err(RuntimePatchError::TargetRouteMismatch { guid: route.guid });
            }
            if self
                .target_runtime_manifest
                .container_identity(&route.container_name)
                != self
                    .payload
                    .runtime_manifest
                    .container_identity(&route.container_name)
            {
                return Err(RuntimePatchError::TargetContainerMismatch {
                    name: route.container_name.clone(),
                });
            }
        }

        for route in base_runtime_manifest.routes() {
            if self.descriptor.removed_guids.contains(&route.guid)
                || self.payload.runtime_manifest.route(route.guid).is_some()
            {
                continue;
            }
            let target = self
                .target_runtime_manifest
                .route(route.guid)
                .ok_or(RuntimePatchError::TargetRouteMissing { guid: route.guid })?;
            if self.target_runtime_manifest.metadata(route.guid)
                != base_runtime_manifest.metadata(route.guid)
            {
                return Err(RuntimePatchError::TargetMetadataMismatch { guid: route.guid });
            }
            if self
                .target_runtime_manifest
                .container_identity(&target.container_name)
                != base_runtime_manifest.container_identity(&route.container_name)
            {
                return Err(RuntimePatchError::TargetContainerMismatch {
                    name: target.container_name.clone(),
                });
            }
        }

        for guid in &self.descriptor.removed_guids {
            if self.payload.runtime_manifest.route(*guid).is_none()
                && self.target_runtime_manifest.route(*guid).is_some()
            {
                return Err(RuntimePatchError::RemovedRouteStillPresent { guid: *guid });
            }
        }
        Ok(self.target_runtime_manifest.clone())
    }
}

/// Errors raised while producing an incremental runtime package.
#[derive(Debug)]
pub enum RuntimePatchError {
    /// The base package was produced from another build manifest.
    BasePackageManifestMismatch {
        expected: ContentHash,
        actual: ContentHash,
    },
    /// The base package's encoded runtime manifest does not match its object.
    BasePackageRuntimeManifestMismatch,
    /// A removed or replaced asset has no stable base route.
    MissingBaseRoute { asset: AssetId },
    /// The deterministic Build Manifest diff failed.
    Plan(PatchPlanError),
    /// The selected runtime payload could not be packaged.
    Package(PackageError),
    /// Applying the generated delta to the base runtime manifest failed.
    Manifest(ManifestError),
    /// The descriptor could not be constructed or encoded.
    Descriptor(PatchDescriptorError),
    /// The target publication/provenance descriptor could not be constructed.
    Provenance(&'static str),
    /// The target provenance object and its canonical bytes disagree.
    ProvenanceObjectMismatch,
    /// The target provenance does not bind the patch target revision/manifests.
    ProvenanceBindingMismatch,
    /// The target provenance container facts do not match the target manifest.
    ProvenanceContainerMismatch { name: String },
    /// A target container is required by the complete target manifest but is
    /// unavailable from either the base package or patch payload.
    MissingTargetContainer { name: String },
    /// The descriptor bytes decode to a different descriptor object.
    DescriptorObjectMismatch,
    /// The supplied base build manifest hash does not match the descriptor.
    BaseBuildManifestHashMismatch,
    /// The supplied base runtime manifest does not match the descriptor.
    BaseRuntimeManifestHashMismatch,
    /// The incremental runtime manifest object/bytes/hash is inconsistent.
    PatchManifestMismatch,
    /// The target runtime manifest object/bytes/hash is inconsistent.
    TargetManifestMismatch,
    /// A target snapshot route was not produced by the base-plus-patch route merge.
    TargetRouteMissing { guid: StableGuid },
    /// A patch route differs from the target snapshot route.
    TargetRouteMismatch { guid: StableGuid },
    /// An unchanged base route has different target metadata.
    TargetMetadataMismatch { guid: StableGuid },
    /// A patch route binds a different immutable container identity in the target.
    TargetContainerMismatch { name: String },
    /// A tombstoned route unexpectedly remains in the target snapshot.
    RemovedRouteStillPresent { guid: StableGuid },
    /// The base runtime package is internally inconsistent.
    InvalidBasePackage(&'static str),
}

impl std::fmt::Display for RuntimePatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BasePackageManifestMismatch { expected, actual } => write!(
                f,
                "base runtime package binds manifest {}, expected {}",
                actual.to_hex(),
                expected.to_hex()
            ),
            Self::BasePackageRuntimeManifestMismatch => {
                f.write_str("base runtime manifest object and bytes differ")
            }
            Self::MissingBaseRoute { asset } => {
                write!(f, "base runtime package has no route for asset {asset:?}")
            }
            Self::Plan(error) => write!(f, "runtime patch planning failed: {error}"),
            Self::Package(error) => write!(f, "runtime patch packaging failed: {error}"),
            Self::Manifest(error) => write!(f, "runtime patch manifest merge failed: {error}"),
            Self::Descriptor(error) => write!(f, "runtime patch descriptor failed: {error}"),
            Self::Provenance(error) => write!(f, "runtime patch provenance failed: {error}"),
            Self::ProvenanceObjectMismatch => {
                f.write_str("runtime patch provenance bytes do not match the descriptor object")
            }
            Self::ProvenanceBindingMismatch => {
                f.write_str("runtime patch provenance is not bound to the patch target")
            }
            Self::ProvenanceContainerMismatch { name } => {
                write!(
                    f,
                    "runtime patch provenance container facts mismatch {name:?}"
                )
            }
            Self::MissingTargetContainer { name } => {
                write!(
                    f,
                    "runtime patch target container bytes are missing for {name:?}"
                )
            }
            Self::DescriptorObjectMismatch => {
                f.write_str("runtime patch descriptor bytes do not match the descriptor object")
            }
            Self::BaseBuildManifestHashMismatch => {
                f.write_str("base build manifest hash does not match runtime patch descriptor")
            }
            Self::BaseRuntimeManifestHashMismatch => {
                f.write_str("base runtime manifest hash does not match runtime patch descriptor")
            }
            Self::PatchManifestMismatch => {
                f.write_str("runtime patch payload manifest object or hash is inconsistent")
            }
            Self::TargetManifestMismatch => {
                f.write_str("runtime patch target manifest object or hash is inconsistent")
            }
            Self::TargetRouteMissing { guid } => {
                write!(f, "runtime patch target manifest is missing route {guid:?}")
            }
            Self::TargetRouteMismatch { guid } => {
                write!(
                    f,
                    "runtime patch route {guid:?} differs from target manifest"
                )
            }
            Self::TargetMetadataMismatch { guid } => {
                write!(
                    f,
                    "unchanged runtime route {guid:?} has different target metadata"
                )
            }
            Self::TargetContainerMismatch { name } => {
                write!(
                    f,
                    "runtime patch target container identity differs for {name:?}"
                )
            }
            Self::RemovedRouteStillPresent { guid } => {
                write!(
                    f,
                    "runtime patch tombstone route {guid:?} remains in target manifest"
                )
            }
            Self::InvalidBasePackage(reason) => {
                write!(f, "base runtime package is invalid: {reason}")
            }
        }
    }
}

impl std::error::Error for RuntimePatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Plan(error) => Some(error),
            Self::Package(error) => Some(error),
            Self::Manifest(error) => Some(error),
            Self::Descriptor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PatchPlanError> for RuntimePatchError {
    fn from(value: PatchPlanError) -> Self {
        Self::Plan(value)
    }
}

impl From<PackageError> for RuntimePatchError {
    fn from(value: PackageError) -> Self {
        Self::Package(value)
    }
}

impl From<ManifestError> for RuntimePatchError {
    fn from(value: ManifestError) -> Self {
        Self::Manifest(value)
    }
}

impl From<PatchDescriptorError> for RuntimePatchError {
    fn from(value: PatchDescriptorError) -> Self {
        Self::Descriptor(value)
    }
}

/// Builds a runtime patch using the default chunk layout policy.
pub fn package_runtime_patch(
    request: RuntimePatchRequest<'_>,
) -> Result<RuntimePatchPackage, RuntimePatchError> {
    package_runtime_patch_with_options(request, PackageOptions::default())
}

/// Builds a runtime patch with an explicit chunk layout policy.
pub fn package_runtime_patch_with_options(
    request: RuntimePatchRequest<'_>,
    options: PackageOptions,
) -> Result<RuntimePatchPackage, RuntimePatchError> {
    let RuntimePatchRequest {
        base_manifest,
        target_manifest,
        base_package,
        cas,
        target_mappings,
    } = request;
    let base_manifest_hash = ContentHash::digest(&base_manifest.canonical_bytes());
    if base_package.source_manifest_hash != base_manifest_hash {
        return Err(RuntimePatchError::BasePackageManifestMismatch {
            expected: base_manifest_hash,
            actual: base_package.source_manifest_hash,
        });
    }
    let encoded_base_manifest = base_package
        .runtime_manifest
        .encode()
        .map_err(RuntimePatchError::Manifest)?;
    if encoded_base_manifest != base_package.runtime_manifest_bytes {
        return Err(RuntimePatchError::BasePackageRuntimeManifestMismatch);
    }

    validate_base_package(base_package)?;
    let target_by_asset = validate_package_mappings(target_manifest, &target_mappings)?;
    let plan = plan_patch(base_manifest, target_manifest)?;
    let mut selected_assets = BTreeSet::new();
    let mut removed_guids = BTreeSet::new();
    for entry in &plan.entries {
        match entry.change {
            PatchChange::Added => {
                selected_assets.insert(entry.asset);
            }
            PatchChange::Changed => {
                selected_assets.insert(entry.asset);
                let base_guid = base_package
                    .asset_guids
                    .get(&entry.asset)
                    .copied()
                    .ok_or(RuntimePatchError::MissingBaseRoute { asset: entry.asset })?;
                if target_by_asset[&entry.asset].guid != base_guid {
                    removed_guids.insert(base_guid);
                }
            }
            PatchChange::Removed => {
                let base_guid = base_package
                    .asset_guids
                    .get(&entry.asset)
                    .copied()
                    .ok_or(RuntimePatchError::MissingBaseRoute { asset: entry.asset })?;
                removed_guids.insert(base_guid);
            }
        }
    }

    for asset in target_manifest.entries.keys() {
        if selected_assets.contains(asset) {
            continue;
        }
        let base_guid = base_package
            .asset_guids
            .get(asset)
            .copied()
            .ok_or(RuntimePatchError::MissingBaseRoute { asset: *asset })?;
        let base_route = base_package
            .runtime_manifest
            .route(base_guid)
            .ok_or(RuntimePatchError::MissingBaseRoute { asset: *asset })?;
        let target_mapping = &target_by_asset[asset];
        if base_guid != target_mapping.guid
            || base_route.container_name != target_mapping.container_name
            || base_route.priority != target_mapping.priority
            || base_route.container_guid != target_mapping.guid
        {
            selected_assets.insert(*asset);
            if base_guid != target_mapping.guid {
                removed_guids.insert(base_guid);
            }
        }
    }

    let patch_mappings =
        namespace_changed_mappings(target_mappings, &selected_assets, target_manifest.revision);
    let mut payload = package_runtime_selected_with_options(
        PackageRequest {
            manifest: target_manifest,
            cas,
            mappings: patch_mappings,
        },
        options,
        selected_assets.iter().copied(),
    )?;

    let removed_guids = removed_guids.into_iter().collect::<Vec<_>>();
    let target_runtime_manifest = build_target_runtime_manifest(
        base_package,
        target_manifest,
        &target_by_asset,
        &selected_assets,
        &payload.runtime_manifest,
    )?;
    complete_patch_dependency_context(
        &mut payload.runtime_manifest,
        &target_runtime_manifest,
        selected_assets
            .iter()
            .map(|asset| target_by_asset[asset].guid),
    )?;
    payload.runtime_manifest_bytes = payload
        .runtime_manifest
        .encode()
        .map_err(RuntimePatchError::Manifest)?;
    let target_runtime_manifest_bytes = target_runtime_manifest
        .encode()
        .map_err(RuntimePatchError::Manifest)?;
    let (target_provenance, target_provenance_bytes) = build_target_provenance(
        base_package,
        target_manifest,
        &target_runtime_manifest,
        &target_runtime_manifest_bytes,
        &payload,
    )?;
    let descriptor = RuntimePatchDescriptor::new(
        plan.base_revision.0,
        plan.target_revision.0,
        base_manifest_hash.0,
        payload.source_manifest_hash.0,
        ContentHash::digest(&base_package.runtime_manifest_bytes).0,
        ContentHash::digest(&target_runtime_manifest_bytes).0,
        ContentHash::digest(&payload.runtime_manifest_bytes).0,
        removed_guids,
    )?;
    let descriptor_bytes = descriptor.encode()?;
    Ok(RuntimePatchPackage {
        descriptor,
        descriptor_bytes,
        payload,
        target_runtime_manifest,
        target_runtime_manifest_bytes,
        target_provenance,
        target_provenance_bytes,
    })
}

fn build_target_provenance(
    base_package: &RuntimePackage,
    target_manifest: &Manifest,
    target_runtime_manifest: &RuntimeManifest,
    target_runtime_manifest_bytes: &[u8],
    patch_payload: &RuntimePackage,
) -> Result<(ReleaseProvenance, Vec<u8>), RuntimePatchError> {
    let mut containers = BTreeMap::new();
    for route in target_runtime_manifest.routes() {
        if containers.contains_key(&route.container_name) {
            continue;
        }
        let bytes = patch_payload
            .containers
            .get(&route.container_name)
            .or_else(|| base_package.containers.get(&route.container_name))
            .ok_or_else(|| RuntimePatchError::MissingTargetContainer {
                name: route.container_name.clone(),
            })?;
        let identity = target_runtime_manifest
            .container_identity(&route.container_name)
            .ok_or_else(|| RuntimePatchError::ProvenanceContainerMismatch {
                name: route.container_name.clone(),
            })?;
        if identity.byte_len != bytes.len() as u64
            || identity.content_hash != ContentHash::digest(bytes).0
        {
            return Err(RuntimePatchError::ProvenanceContainerMismatch {
                name: route.container_name.clone(),
            });
        }
        containers.insert(route.container_name.clone(), bytes.clone());
    }
    let build_manifest_hash = ContentHash::digest(&target_manifest.canonical_bytes());
    let publication = PublicationDescriptor::new(
        target_manifest.revision,
        build_manifest_hash,
        target_runtime_manifest_bytes,
        &containers,
    )
    .map_err(RuntimePatchError::Provenance)?;
    let provenance =
        ReleaseProvenance::from_manifest(target_manifest, build_manifest_hash, &publication)
            .map_err(RuntimePatchError::Provenance)?;
    let bytes = provenance.encode().map_err(RuntimePatchError::Provenance)?;
    Ok((provenance, bytes))
}

fn validate_target_provenance(
    provenance: &ReleaseProvenance,
    provenance_bytes: &[u8],
    descriptor: &RuntimePatchDescriptor,
    target_runtime_manifest: &RuntimeManifest,
    target_runtime_manifest_bytes: &[u8],
) -> Result<(), RuntimePatchError> {
    let decoded =
        ReleaseProvenance::decode(provenance_bytes).map_err(RuntimePatchError::Provenance)?;
    if decoded != *provenance
        || provenance.encode().map_err(RuntimePatchError::Provenance)? != provenance_bytes
    {
        return Err(RuntimePatchError::ProvenanceObjectMismatch);
    }
    if provenance.revision.0 != descriptor.target_revision
        || provenance.build_manifest_hash.0 != descriptor.target_build_manifest_hash
        || provenance.runtime_manifest_hash.0 != descriptor.target_runtime_manifest_hash
    {
        return Err(RuntimePatchError::ProvenanceBindingMismatch);
    }

    let publication = PublicationDescriptor {
        revision: BuildRevision(descriptor.target_revision),
        build_manifest_hash: ContentHash(descriptor.target_build_manifest_hash),
        runtime_manifest_hash: ContentHash::digest(target_runtime_manifest_bytes),
        containers: provenance
            .containers
            .iter()
            .map(|container| crate::publication::PublishedContainer {
                name: container.name.clone(),
                byte_len: container.byte_len,
                content_hash: container.content_hash,
            })
            .collect(),
    };
    if provenance.publication_hash != publication.content_hash() {
        return Err(RuntimePatchError::ProvenanceBindingMismatch);
    }
    let provenance_names = provenance
        .containers
        .iter()
        .map(|container| container.name.as_str())
        .collect::<BTreeSet<_>>();
    let target_names = target_runtime_manifest
        .routes()
        .map(|route| route.container_name.as_str())
        .collect::<BTreeSet<_>>();
    if provenance_names != target_names {
        let name = target_names
            .symmetric_difference(&provenance_names)
            .next()
            .map(|name| (*name).to_owned())
            .expect("different sets have a symmetric difference");
        return Err(RuntimePatchError::ProvenanceContainerMismatch { name });
    }
    for container in &provenance.containers {
        let Some(identity) = target_runtime_manifest.container_identity(&container.name) else {
            return Err(RuntimePatchError::ProvenanceContainerMismatch {
                name: container.name.clone(),
            });
        };
        if identity.byte_len != container.byte_len
            || identity.content_hash != container.content_hash.0
        {
            return Err(RuntimePatchError::ProvenanceContainerMismatch {
                name: container.name.clone(),
            });
        }
    }
    Ok(())
}

fn namespace_changed_mappings(
    mappings: Vec<AssetPackageMapping>,
    selected_assets: &BTreeSet<AssetId>,
    target_revision: BuildRevision,
) -> Vec<AssetPackageMapping> {
    mappings
        .into_iter()
        .map(|mut mapping| {
            if selected_assets.contains(&mapping.asset) {
                mapping.container_name =
                    format!("patch_{}_{}", target_revision.0, mapping.container_name);
            }
            mapping
        })
        .collect()
}

/// Adds target routes needed only to validate and resolve patch dependencies.
///
/// These context routes intentionally do not add payload bytes: unchanged
/// containers remain owned by the base package. Keeping the context routes in
/// the patch manifest makes the delta self-describing while preserving the
/// incremental payload guarantee. Optional targets contribute context only
/// when present; missing hard targets remain errors.
fn complete_patch_dependency_context(
    patch: &mut RuntimeManifest,
    target: &RuntimeManifest,
    roots: impl IntoIterator<Item = StableGuid>,
) -> Result<(), RuntimePatchError> {
    let mut pending = roots.into_iter().collect::<BTreeSet<_>>();
    let mut closure = BTreeSet::new();
    while let Some(guid) = pending.pop_first() {
        if !closure.insert(guid) {
            continue;
        }
        let metadata = target
            .metadata(guid)
            .ok_or(RuntimePatchError::TargetRouteMissing { guid })?;
        pending.extend(metadata.hard_dependencies.iter().copied());
        pending.extend(
            metadata
                .optional_dependencies
                .iter()
                .copied()
                .filter(|dependency| target.route(*dependency).is_some()),
        );
    }

    for guid in &closure {
        let route = target
            .route(*guid)
            .ok_or(RuntimePatchError::TargetRouteMissing { guid: *guid })?;
        let metadata = target
            .metadata(*guid)
            .ok_or(RuntimePatchError::TargetRouteMissing { guid: *guid })?;
        if let Some(existing) = patch.route(*guid) {
            if existing != route {
                return Err(RuntimePatchError::TargetRouteMismatch { guid: *guid });
            }
        } else {
            patch
                .insert_with_metadata(route.clone(), metadata.clone())
                .map_err(RuntimePatchError::Manifest)?;
        }
        let identity = target.container_identity(&route.container_name).ok_or(
            RuntimePatchError::TargetContainerMismatch {
                name: route.container_name.clone(),
            },
        )?;
        patch
            .bind_container(route.container_name.clone(), identity)
            .map_err(RuntimePatchError::Manifest)?;
    }

    for guid in &closure {
        let metadata = target
            .metadata(*guid)
            .ok_or(RuntimePatchError::TargetRouteMissing { guid: *guid })?;
        patch
            .set_metadata(*guid, metadata.clone())
            .map_err(RuntimePatchError::Manifest)?;
    }
    Ok(())
}

fn validate_patch_routes(
    merged: &RuntimeManifest,
    target: &RuntimeManifest,
) -> Result<(), RuntimePatchError> {
    for route in merged.routes() {
        let target_route = target
            .route(route.guid)
            .ok_or(RuntimePatchError::TargetRouteMissing { guid: route.guid })?;
        if target_route != route {
            return Err(RuntimePatchError::TargetRouteMismatch { guid: route.guid });
        }
    }
    for route in target.routes() {
        if merged.route(route.guid).is_none() {
            return Err(RuntimePatchError::TargetRouteMissing { guid: route.guid });
        }
    }
    Ok(())
}

fn validate_base_package(base_package: &RuntimePackage) -> Result<(), RuntimePatchError> {
    if base_package.asset_guids.len() != base_package.runtime_manifest.routes().count() {
        return Err(RuntimePatchError::InvalidBasePackage(
            "asset-to-guid map does not cover exactly the runtime routes",
        ));
    }
    for guid in base_package.asset_guids.values() {
        if base_package.runtime_manifest.route(*guid).is_none() {
            return Err(RuntimePatchError::InvalidBasePackage(
                "asset-to-guid map references a missing runtime route",
            ));
        }
    }
    Ok(())
}

fn build_target_runtime_manifest(
    base_package: &RuntimePackage,
    target_manifest: &Manifest,
    target_by_asset: &std::collections::BTreeMap<AssetId, AssetPackageMapping>,
    selected_assets: &BTreeSet<AssetId>,
    patch_manifest: &RuntimeManifest,
) -> Result<RuntimeManifest, RuntimePatchError> {
    let mut artifact_assets = std::collections::BTreeMap::<ArtifactKey, BTreeSet<AssetId>>::new();
    for (asset, descriptor) in &target_manifest.entries {
        artifact_assets
            .entry(descriptor.key)
            .or_default()
            .insert(*asset);
    }

    let final_guid = |asset: AssetId| -> Result<StableGuid, RuntimePatchError> {
        if selected_assets.contains(&asset) {
            return Ok(target_by_asset[&asset].guid);
        }
        base_package
            .asset_guids
            .get(&asset)
            .copied()
            .ok_or(RuntimePatchError::MissingBaseRoute { asset })
    };

    let mut target = RuntimeManifest::new();
    for (asset, descriptor) in &target_manifest.entries {
        let mapping = &target_by_asset[asset];
        let guid = final_guid(*asset)?;
        let (container_name, container_guid, identity) = if selected_assets.contains(asset) {
            let container_name = format!(
                "patch_{}_{}",
                target_manifest.revision.0, mapping.container_name
            );
            let route = patch_manifest
                .route(guid)
                .ok_or(RuntimePatchError::TargetRouteMissing { guid })?;
            let identity = patch_manifest.container_identity(&container_name).ok_or(
                RuntimePatchError::TargetContainerMismatch {
                    name: container_name.clone(),
                },
            )?;
            (container_name, route.container_guid, identity)
        } else {
            let base_route = base_package
                .runtime_manifest
                .route(guid)
                .ok_or(RuntimePatchError::MissingBaseRoute { asset: *asset })?;
            let identity = base_package
                .runtime_manifest
                .container_identity(&base_route.container_name)
                .ok_or(RuntimePatchError::TargetContainerMismatch {
                    name: base_route.container_name.clone(),
                })?;
            (
                base_route.container_name.clone(),
                base_route.container_guid,
                identity,
            )
        };

        let hard_dependencies = descriptor
            .dependencies
            .iter()
            .flat_map(|dependency| artifact_assets.get(dependency).into_iter().flatten())
            .map(|dependency_asset| final_guid(*dependency_asset))
            .collect::<Result<BTreeSet<_>, _>>()?
            .into_iter()
            .collect::<Vec<_>>();
        // Build descriptors carry only hard edges. Preserve soft references
        // on reused routes, including targets removed by this patch.
        let optional_dependencies = if selected_assets.contains(asset) {
            Vec::new()
        } else {
            base_package
                .runtime_manifest
                .metadata(guid)
                .ok_or(RuntimePatchError::TargetRouteMissing { guid })?
                .optional_dependencies
                .clone()
        };
        target
            .insert_with_metadata(
                ManifestRoute {
                    guid,
                    container_name: container_name.clone(),
                    priority: mapping.priority,
                    container_guid,
                },
                ManifestMetadata {
                    hard_dependencies,
                    optional_dependencies,
                    initial_residency: 0,
                    streaming_level: 0,
                    codec: prism_asset_container::Codec::Raw,
                    schema: descriptor.schema,
                    install_capabilities: 0,
                },
            )
            .map_err(RuntimePatchError::Manifest)?;
        target
            .bind_container(container_name, identity)
            .map_err(RuntimePatchError::Manifest)?;
    }
    target.encode().map_err(RuntimePatchError::Manifest)?;
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ArtifactDescriptor, ArtifactKey, ManifestEntry};
    use tempfile::tempdir;

    fn descriptor(key: u8, bytes: &[u8], dependencies: Vec<ArtifactKey>) -> ArtifactDescriptor {
        ArtifactDescriptor::for_bytes(ArtifactKey([key; 32]), 1, 1, 16, dependencies, bytes)
    }

    #[test]
    fn patch_emits_only_changed_payload_and_applies_tombstone() {
        let directory = tempdir().unwrap();
        let cas = ContentAddressedStore::open(directory.keep()).unwrap();
        let base_a = b"base-a";
        let base_b = b"base-b";
        let target_a = b"target-a";
        let base_a_hash = cas.put(base_a).unwrap();
        let base_b_hash = cas.put(base_b).unwrap();
        let target_a_hash = cas.put(target_a).unwrap();
        let base = Manifest::new(
            BuildRevision(1),
            [
                ManifestEntry {
                    asset: AssetId(1),
                    descriptor: descriptor(1, base_a, vec![]),
                },
                ManifestEntry {
                    asset: AssetId(2),
                    descriptor: descriptor(2, base_b, vec![]),
                },
            ],
        )
        .unwrap();
        let target = Manifest::new(
            BuildRevision(2),
            [ManifestEntry {
                asset: AssetId(1),
                descriptor: ArtifactDescriptor {
                    content_hash: target_a_hash,
                    ..descriptor(9, target_a, vec![])
                },
            }],
        )
        .unwrap();
        assert_eq!(base.entries[&AssetId(1)].content_hash, base_a_hash);
        assert_eq!(base.entries[&AssetId(2)].content_hash, base_b_hash);

        let base_package = crate::package::package_runtime(PackageRequest {
            manifest: &base,
            cas: &cas,
            mappings: vec![
                AssetPackageMapping {
                    asset: AssetId(1),
                    guid: StableGuid::from_u128(101),
                    container_name: "base_a".into(),
                    priority: 0,
                    chunk_priority: 0,
                },
                AssetPackageMapping {
                    asset: AssetId(2),
                    guid: StableGuid::from_u128(202),
                    container_name: "base_b".into(),
                    priority: 0,
                    chunk_priority: 0,
                },
            ],
        })
        .unwrap();
        let patch = package_runtime_patch(RuntimePatchRequest {
            base_manifest: &base,
            target_manifest: &target,
            base_package: &base_package,
            cas: &cas,
            target_mappings: vec![AssetPackageMapping {
                asset: AssetId(1),
                guid: StableGuid::from_u128(101),
                container_name: "target_a".into(),
                priority: 10,
                chunk_priority: 0,
            }],
        })
        .unwrap();

        assert_eq!(patch.payload.asset_guids.len(), 1);
        assert_eq!(patch.payload.containers.len(), 1);
        assert_eq!(
            patch.descriptor.removed_guids,
            vec![StableGuid::from_u128(202)]
        );
        assert_eq!(
            RuntimePatchDescriptor::decode(&patch.descriptor_bytes).unwrap(),
            patch.descriptor
        );
        let mut merged = base_package.runtime_manifest.clone();
        merged
            .apply_patch(
                &patch.payload.runtime_manifest,
                &patch.descriptor.removed_guids,
            )
            .unwrap();
        assert!(merged.route(StableGuid::from_u128(101)).is_some());
        assert!(merged.route(StableGuid::from_u128(202)).is_none());
        assert_eq!(
            merged
                .route(StableGuid::from_u128(101))
                .unwrap()
                .container_name,
            "patch_2_target_a"
        );
        assert_eq!(
            ContentHash::digest(&patch.target_runtime_manifest_bytes).0,
            patch.descriptor.target_runtime_manifest_hash
        );
        assert_eq!(
            patch
                .apply_to_manifest(
                    base_package.source_manifest_hash,
                    &base_package.runtime_manifest,
                )
                .unwrap(),
            patch.target_runtime_manifest
        );
    }

    #[test]
    fn patch_keeps_unchanged_dependencies_in_delta_metadata() {
        let directory = tempdir().unwrap();
        let cas = ContentAddressedStore::open(directory.keep()).unwrap();
        let stable = b"stable";
        let changed = b"changed";
        let stable_hash = cas.put(stable).unwrap();
        let changed_hash = cas.put(changed).unwrap();
        let dependency = ArtifactKey([1; 32]);
        let base = Manifest::new(
            BuildRevision(1),
            [
                ManifestEntry {
                    asset: AssetId(1),
                    descriptor: descriptor(1, stable, vec![]),
                },
                ManifestEntry {
                    asset: AssetId(2),
                    descriptor: ArtifactDescriptor {
                        content_hash: changed_hash,
                        ..descriptor(2, changed, vec![dependency])
                    },
                },
            ],
        )
        .unwrap();
        let target = Manifest::new(
            BuildRevision(2),
            [
                ManifestEntry {
                    asset: AssetId(1),
                    descriptor: descriptor(1, stable, vec![]),
                },
                ManifestEntry {
                    asset: AssetId(2),
                    descriptor: ArtifactDescriptor {
                        content_hash: changed_hash,
                        ..descriptor(3, changed, vec![dependency])
                    },
                },
            ],
        )
        .unwrap();
        assert_eq!(base.entries[&AssetId(1)].content_hash, stable_hash);
        let mappings = vec![
            AssetPackageMapping {
                asset: AssetId(1),
                guid: StableGuid::from_u128(11),
                container_name: "base_stable".into(),
                priority: 0,
                chunk_priority: 0,
            },
            AssetPackageMapping {
                asset: AssetId(2),
                guid: StableGuid::from_u128(22),
                container_name: "base_changed".into(),
                priority: 0,
                chunk_priority: 0,
            },
        ];
        let base_package = crate::package::package_runtime(PackageRequest {
            manifest: &base,
            cas: &cas,
            mappings: mappings.clone(),
        })
        .unwrap();
        let patch = package_runtime_patch(RuntimePatchRequest {
            base_manifest: &base,
            target_manifest: &target,
            base_package: &base_package,
            cas: &cas,
            target_mappings: mappings,
        })
        .unwrap();
        let metadata = patch
            .target_runtime_manifest
            .metadata(StableGuid::from_u128(22))
            .unwrap();
        assert_eq!(metadata.hard_dependencies, vec![StableGuid::from_u128(11)]);
        assert!(patch
            .payload
            .runtime_manifest
            .route(StableGuid::from_u128(11))
            .is_some());
        assert_eq!(patch.payload.containers.len(), 1);
        assert_eq!(
            patch
                .apply_to_manifest(
                    base_package.source_manifest_hash,
                    &base_package.runtime_manifest,
                )
                .unwrap(),
            patch.target_runtime_manifest
        );
    }

    #[test]
    fn empty_patch_reuses_base_manifest_without_reading_payloads() {
        let directory = tempdir().unwrap();
        let cas = ContentAddressedStore::open(directory.keep()).unwrap();
        let bytes = b"stable";
        cas.put(bytes).unwrap();
        let base = Manifest::new(
            BuildRevision(3),
            [ManifestEntry {
                asset: AssetId(1),
                descriptor: descriptor(1, bytes, vec![]),
            }],
        )
        .unwrap();
        let target = Manifest::new(
            BuildRevision(4),
            [ManifestEntry {
                asset: AssetId(1),
                descriptor: descriptor(1, bytes, vec![]),
            }],
        )
        .unwrap();
        let mapping = AssetPackageMapping {
            asset: AssetId(1),
            guid: StableGuid::from_u128(11),
            container_name: "base".into(),
            priority: 0,
            chunk_priority: 0,
        };
        let base_package = crate::package::package_runtime(PackageRequest {
            manifest: &base,
            cas: &cas,
            mappings: vec![mapping.clone()],
        })
        .unwrap();
        let patch = package_runtime_patch(RuntimePatchRequest {
            base_manifest: &base,
            target_manifest: &target,
            base_package: &base_package,
            cas: &cas,
            target_mappings: vec![mapping],
        })
        .unwrap();
        assert!(patch.payload.containers.is_empty());
        assert!(patch.payload.runtime_manifest.routes().next().is_none());
        assert_eq!(
            patch
                .apply_to_manifest(
                    base_package.source_manifest_hash,
                    &base_package.runtime_manifest,
                )
                .unwrap(),
            base_package.runtime_manifest
        );
    }

    #[test]
    fn patch_rejects_base_and_target_integrity_mismatches() {
        let directory = tempdir().unwrap();
        let cas = ContentAddressedStore::open(directory.keep()).unwrap();
        let base_bytes = b"base";
        let target_bytes = b"target";
        cas.put(base_bytes).unwrap();
        cas.put(target_bytes).unwrap();
        let base = Manifest::new(
            BuildRevision(1),
            [ManifestEntry {
                asset: AssetId(1),
                descriptor: descriptor(1, base_bytes, vec![]),
            }],
        )
        .unwrap();
        let target = Manifest::new(
            BuildRevision(2),
            [ManifestEntry {
                asset: AssetId(1),
                descriptor: descriptor(2, target_bytes, vec![]),
            }],
        )
        .unwrap();
        let mapping = AssetPackageMapping {
            asset: AssetId(1),
            guid: StableGuid::from_u128(11),
            container_name: "asset".into(),
            priority: 0,
            chunk_priority: 0,
        };
        let base_package = crate::package::package_runtime(PackageRequest {
            manifest: &base,
            cas: &cas,
            mappings: vec![mapping.clone()],
        })
        .unwrap();
        let patch = package_runtime_patch(RuntimePatchRequest {
            base_manifest: &base,
            target_manifest: &target,
            base_package: &base_package,
            cas: &cas,
            target_mappings: vec![mapping],
        })
        .unwrap();

        let changed_base = RuntimeManifest::new();
        let mut changed_patch = patch.clone();
        changed_patch.descriptor.target_revision += 1;
        assert!(matches!(
            changed_patch.apply_to_manifest(
                base_package.source_manifest_hash,
                &base_package.runtime_manifest,
            ),
            Err(RuntimePatchError::DescriptorObjectMismatch)
        ));
        assert!(matches!(
            patch.apply_to_manifest(ContentHash([7; 32]), &base_package.runtime_manifest),
            Err(RuntimePatchError::BaseBuildManifestHashMismatch)
        ));
        assert!(matches!(
            patch.apply_to_manifest(base_package.source_manifest_hash, &changed_base),
            Err(RuntimePatchError::BaseRuntimeManifestHashMismatch)
        ));
        let mut tampered_target = patch.clone();
        tampered_target.target_runtime_manifest_bytes[0] ^= 1;
        assert!(matches!(
            tampered_target.apply_to_manifest(
                base_package.source_manifest_hash,
                &base_package.runtime_manifest,
            ),
            Err(RuntimePatchError::TargetManifestMismatch)
        ));

        let mut tampered_provenance_bytes = patch.clone();
        tampered_provenance_bytes.target_provenance_bytes[0] ^= 1;
        assert!(matches!(
            tampered_provenance_bytes.apply_to_manifest(
                base_package.source_manifest_hash,
                &base_package.runtime_manifest,
            ),
            Err(RuntimePatchError::Provenance(_))
        ));

        let mut tampered_provenance_object = patch.clone();
        tampered_provenance_object
            .target_provenance
            .publication_hash = ContentHash([8; 32]);
        assert!(matches!(
            tampered_provenance_object.apply_to_manifest(
                base_package.source_manifest_hash,
                &base_package.runtime_manifest,
            ),
            Err(RuntimePatchError::ProvenanceObjectMismatch)
        ));
    }

    #[test]
    fn patch_rejects_incomplete_or_duplicate_target_mappings() {
        let directory = tempdir().unwrap();
        let cas = ContentAddressedStore::open(directory.keep()).unwrap();
        let bytes = b"asset";
        cas.put(bytes).unwrap();
        let base = Manifest::new(
            BuildRevision(1),
            [ManifestEntry {
                asset: AssetId(1),
                descriptor: descriptor(1, bytes, vec![]),
            }],
        )
        .unwrap();
        let target = Manifest::new(
            BuildRevision(2),
            [ManifestEntry {
                asset: AssetId(1),
                descriptor: descriptor(2, bytes, vec![]),
            }],
        )
        .unwrap();
        let base_mapping = AssetPackageMapping {
            asset: AssetId(1),
            guid: StableGuid::from_u128(11),
            container_name: "base".into(),
            priority: 0,
            chunk_priority: 0,
        };
        let base_package = crate::package::package_runtime(PackageRequest {
            manifest: &base,
            cas: &cas,
            mappings: vec![base_mapping],
        })
        .unwrap();
        assert!(matches!(
            package_runtime_patch(RuntimePatchRequest {
                base_manifest: &base,
                target_manifest: &target,
                base_package: &base_package,
                cas: &cas,
                target_mappings: vec![],
            }),
            Err(RuntimePatchError::Package(PackageError::MissingMapping {
                asset: AssetId(1)
            }))
        ));

        let mapping = AssetPackageMapping {
            asset: AssetId(1),
            guid: StableGuid::from_u128(12),
            container_name: "target".into(),
            priority: 0,
            chunk_priority: 0,
        };
        assert!(matches!(
            package_runtime_patch(RuntimePatchRequest {
                base_manifest: &base,
                target_manifest: &target,
                base_package: &base_package,
                cas: &cas,
                target_mappings: vec![mapping.clone(), mapping],
            }),
            Err(RuntimePatchError::Package(
                PackageError::DuplicateAssetMapping { asset: AssetId(1) }
            ))
        ));
    }
}
