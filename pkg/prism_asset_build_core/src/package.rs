//! Deterministic build-to-runtime packaging.
//!
//! This module is deliberately a bridge, not a second asset database.  The
//! build manifest remains the source of truth for artifact identity and
//! dependencies, while the CAS remains the source of truth for bytes.  The
//! package step verifies both before handing raw payloads to the canonical
//! [`prism_asset_container::ContainerWriter`].

use alloc::collections::{BTreeMap, BTreeSet};
use std::io::ErrorKind;

use prism_asset::StableGuid;
use prism_asset_container::{
    ChunkInput, ChunkPlanError, ChunkPlanner, ChunkPolicy, ContainerError, ContainerIdentity,
    ContainerReader, ContainerWriter, ManifestError, ManifestMetadata, ManifestRoute,
    RuntimeManifest,
};

use crate::cas::{CasError, ContentAddressedStore};
use crate::types::{ArtifactKey, ArtifactStorageClass, AssetId, ContentHash, Manifest};

/// The runtime identity and container route assigned to one build asset.
///
/// Every manifest asset must have exactly one mapping.  `guid` is both the
/// logical route identity and the container-local record identity; this keeps
/// the bridge lossless while making the mapping explicit at the build/runtime
/// boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetPackageMapping {
    pub asset: AssetId,
    pub guid: StableGuid,
    pub container_name: String,
    /// Priority recorded in the runtime manifest. Higher values win overlay
    /// resolution; it is part of the package contract, not writer metadata.
    pub priority: i32,
    /// Scheduling priority used by the deterministic chunk planner.
    pub chunk_priority: u32,
}

/// Inputs consumed by [`package_runtime`].
pub struct PackageRequest<'a> {
    pub manifest: &'a Manifest,
    pub cas: &'a ContentAddressedStore,
    pub mappings: Vec<AssetPackageMapping>,
}

/// Optional physical layout policy for runtime package containers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackageOptions {
    /// Maximum raw bytes per planned chunk, including internal padding.
    pub chunk_policy: ChunkPolicy,
    /// Per-payload encoding used by the runtime container writer.
    pub codec: prism_asset_container::Codec,
    /// Optional independent chunk encoding. When set, the package uses the
    /// chunk-table v2 layout and stores all payloads raw inside the planned
    /// chunks before encoding each chunk independently.
    pub chunk_codec: Option<prism_asset_container::Codec>,
}

impl Default for PackageOptions {
    fn default() -> Self {
        Self {
            chunk_policy: ChunkPolicy::default(),
            codec: prism_asset_container::Codec::Raw,
            chunk_codec: None,
        }
    }
}

/// Fully encoded runtime package output.
///
/// A runtime manifest can route to multiple containers, so `containers` is a
/// canonical name-to-bytes map rather than an ambiguous single byte string.
/// Each value is exactly the output of one [`ContainerWriter::finish`] call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimePackage {
    /// Digest of the canonical build manifest used to produce this package.
    /// Publication refuses to combine a package with a different revision
    /// manifest.
    pub source_manifest_hash: ContentHash,
    /// Deterministic proof that every build asset has one runtime route.
    ///
    /// The runtime manifest keys routes by `StableGuid`, while the build
    /// manifest keys entries by `AssetId`. Keeping this bridge explicit lets
    /// the publication boundary reject a package that is internally valid but
    /// omits build assets.
    pub asset_guids: BTreeMap<AssetId, StableGuid>,
    pub containers: BTreeMap<String, Vec<u8>>,
    pub runtime_manifest: RuntimeManifest,
    pub runtime_manifest_bytes: Vec<u8>,
}

impl RuntimePackage {
    /// Returns one encoded container by its manifest route name.
    #[must_use]
    pub fn container_bytes(&self, name: &str) -> Option<&[u8]> {
        self.containers.get(name).map(Vec::as_slice)
    }
}

/// Validates a complete asset-to-runtime mapping table without reading CAS.
///
/// Patch planning uses this before deciding which payloads to emit. Keeping
/// mapping validation separate from payload packaging prevents a malformed
/// target mapping from being hidden by an otherwise empty or partial patch.
pub(crate) fn validate_package_mappings(
    manifest: &Manifest,
    mappings: &[AssetPackageMapping],
) -> Result<BTreeMap<AssetId, AssetPackageMapping>, PackageError> {
    let mut by_asset = BTreeMap::<AssetId, AssetPackageMapping>::new();
    let mut by_guid = BTreeMap::<StableGuid, AssetId>::new();
    for mapping in mappings {
        if !manifest.entries.contains_key(&mapping.asset) {
            return Err(PackageError::UnknownAsset {
                asset: mapping.asset,
            });
        }
        if by_asset.insert(mapping.asset, mapping.clone()).is_some() {
            return Err(PackageError::DuplicateAssetMapping {
                asset: mapping.asset,
            });
        }
        if mapping.guid.is_nil() {
            return Err(PackageError::InvalidGuid {
                asset: mapping.asset,
            });
        }
        if let Some(first_asset) = by_guid.insert(mapping.guid, mapping.asset)
            && first_asset != mapping.asset
        {
            return Err(PackageError::DuplicateGuid {
                guid: mapping.guid,
                first_asset,
                second_asset: mapping.asset,
            });
        }
    }
    for asset in manifest.entries.keys() {
        if !by_asset.contains_key(asset) {
            return Err(PackageError::MissingMapping { asset: *asset });
        }
    }
    Ok(by_asset)
}

/// Errors raised before a runtime package is emitted.
#[derive(Debug)]
pub enum PackageError {
    /// A manifest asset has more than one mapping.
    DuplicateAssetMapping { asset: AssetId },
    /// A mapping refers to an asset absent from the manifest.
    UnknownAsset { asset: AssetId },
    /// A manifest asset has no runtime mapping.
    MissingMapping { asset: AssetId },
    /// Two assets would occupy one container record identity.
    DuplicateGuid {
        guid: StableGuid,
        first_asset: AssetId,
        second_asset: AssetId,
    },
    /// The reserved nil GUID cannot identify a packaged asset.
    InvalidGuid { asset: AssetId },
    /// A dependency key is not represented by any manifest descriptor.
    MissingDependency {
        asset: AssetId,
        dependency: ArtifactKey,
    },
    /// A dependency key is present in the manifest but has no runtime route.
    UnmappedDependency {
        asset: AssetId,
        dependency: ArtifactKey,
    },
    /// One semantic artifact key was published with incompatible descriptors.
    ConflictingDescriptor { key: ArtifactKey },
    /// The artifact dependency graph contains a cycle.
    DependencyCycle { artifacts: Vec<ArtifactKey> },
    /// The requested CAS blob does not exist.
    MissingPayload {
        asset: AssetId,
        key: ArtifactKey,
        hash: ContentHash,
    },
    /// CAS returned an I/O or integrity error other than a missing blob.
    Cas {
        asset: AssetId,
        key: ArtifactKey,
        source: CasError,
    },
    /// Kept explicit even though the current CAS reader verifies hashes.
    HashMismatch {
        asset: AssetId,
        key: ArtifactKey,
        expected: ContentHash,
        actual: ContentHash,
    },
    /// The descriptor's declared length differs from the CAS bytes.
    LengthMismatch {
        asset: AssetId,
        key: ArtifactKey,
        expected: u64,
        actual: u64,
    },
    /// A descriptor failed its structural validation.
    InvalidDescriptor {
        asset: AssetId,
        key: ArtifactKey,
        reason: &'static str,
    },
    /// A driver-local artifact cannot be emitted into a portable package.
    DriverLocalArtifact { asset: AssetId, key: ArtifactKey },
    /// A container writer rejected an encoded record.
    Container {
        name: String,
        source: ContainerError,
    },
    /// Chunk planning rejected the requested runtime layout.
    ChunkPlan {
        name: String,
        source: ChunkPlanError,
    },
    /// Runtime manifest encoding failed.
    RuntimeManifest(ManifestError),
}

impl std::fmt::Display for PackageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateAssetMapping { asset } => {
                write!(f, "asset {asset:?} has duplicate package mappings")
            }
            Self::UnknownAsset { asset } => {
                write!(f, "package mapping references unknown {asset:?}")
            }
            Self::MissingMapping { asset } => {
                write!(f, "manifest asset {asset:?} has no package mapping")
            }
            Self::DuplicateGuid {
                guid,
                first_asset,
                second_asset,
            } => write!(
                f,
                "runtime guid {guid:?} is used by assets {first_asset:?} and {second_asset:?}"
            ),
            Self::InvalidGuid { asset } => {
                write!(f, "asset {asset:?} uses the reserved nil runtime guid")
            }
            Self::MissingDependency { asset, dependency } => write!(
                f,
                "asset {asset:?} references missing artifact dependency {}",
                dependency.to_hex()
            ),
            Self::UnmappedDependency { asset, dependency } => write!(
                f,
                "asset {asset:?} references unmapped artifact dependency {}",
                dependency.to_hex()
            ),
            Self::ConflictingDescriptor { key } => {
                write!(
                    f,
                    "artifact key {} has conflicting descriptors",
                    key.to_hex()
                )
            }
            Self::DependencyCycle { artifacts } => {
                write!(
                    f,
                    "artifact dependency graph contains a cycle of {} artifact(s)",
                    artifacts.len()
                )
            }
            Self::MissingPayload { asset, key, hash } => write!(
                f,
                "asset {asset:?} artifact {} is missing CAS payload {}",
                key.to_hex(),
                hash.to_hex()
            ),
            Self::Cas { asset, key, source } => {
                write!(
                    f,
                    "asset {asset:?} artifact {} CAS error: {source}",
                    key.to_hex()
                )
            }
            Self::HashMismatch {
                asset,
                key,
                expected,
                actual,
            } => write!(
                f,
                "asset {asset:?} artifact {} hash mismatch: expected {}, got {}",
                key.to_hex(),
                expected.to_hex(),
                actual.to_hex()
            ),
            Self::LengthMismatch {
                asset,
                key,
                expected,
                actual,
            } => write!(
                f,
                "asset {asset:?} artifact {} length mismatch: expected {expected}, got {actual}",
                key.to_hex()
            ),
            Self::InvalidDescriptor { asset, key, reason } => {
                write!(
                    f,
                    "asset {asset:?} descriptor {} is invalid: {reason}",
                    key.to_hex()
                )
            }
            Self::DriverLocalArtifact { asset, key } => write!(
                f,
                "asset {asset:?} artifact {} is driver-local and cannot be packaged",
                key.to_hex()
            ),
            Self::Container { name, source } => write!(f, "container {name:?} failed: {source}"),
            Self::ChunkPlan { name, source } => {
                write!(f, "container {name:?} chunk plan failed: {source}")
            }
            Self::RuntimeManifest(source) => write!(f, "runtime manifest failed: {source}"),
        }
    }
}

impl std::error::Error for PackageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cas { source, .. } => Some(source),
            Self::Container { source, .. } => Some(source),
            Self::ChunkPlan { source, .. } => Some(source),
            Self::RuntimeManifest(source) => Some(source),
            _ => None,
        }
    }
}

/// Packages a complete build manifest into deterministic runtime containers
/// and a deterministic routing manifest.
pub fn package_runtime(request: PackageRequest<'_>) -> Result<RuntimePackage, PackageError> {
    package_runtime_with_options(request, PackageOptions::default())
}

/// Packages a manifest using an explicit physical chunk-layout policy.
///
/// The original [`package_runtime`] remains source-compatible and uses the
/// default 64 MiB chunk capacity, 4 KiB chunk alignment, and unlimited chunk
/// count. Each named container is planned independently; dependencies crossing
/// container boundaries are validated by the package closure pass but do not
/// constrain the local placement order.
pub fn package_runtime_with_options(
    request: PackageRequest<'_>,
    options: PackageOptions,
) -> Result<RuntimePackage, PackageError> {
    package_runtime_impl(request, options, None)
}

/// Packages only the selected assets while retaining the complete manifest as
/// dependency context. This is the primitive used by runtime Patch creation:
/// unchanged artifacts are not read from CAS or emitted into patch containers,
/// but changed routes may still depend on unchanged base routes.
pub fn package_runtime_selected_with_options(
    request: PackageRequest<'_>,
    options: PackageOptions,
    selected_assets: impl IntoIterator<Item = AssetId>,
) -> Result<RuntimePackage, PackageError> {
    let selected_assets = selected_assets.into_iter().collect::<BTreeSet<_>>();
    package_runtime_impl(request, options, Some(selected_assets))
}

fn package_runtime_impl(
    request: PackageRequest<'_>,
    options: PackageOptions,
    selected_assets: Option<BTreeSet<AssetId>>,
) -> Result<RuntimePackage, PackageError> {
    let PackageRequest {
        manifest,
        cas,
        mappings,
    } = request;

    let by_asset = validate_package_mappings(manifest, &mappings)?;

    let selected_assets =
        selected_assets.unwrap_or_else(|| manifest.entries.keys().copied().collect());
    for asset in &selected_assets {
        if !manifest.entries.contains_key(asset) {
            return Err(PackageError::UnknownAsset { asset: *asset });
        }
        if !by_asset.contains_key(asset) {
            return Err(PackageError::MissingMapping { asset: *asset });
        }
    }

    // Resolve every artifact key first.  This validates dependency references
    // independently of payload I/O and makes malformed manifests fail before
    // any writer bytes are produced.
    let mut artifact_assets = BTreeMap::<ArtifactKey, BTreeSet<AssetId>>::new();
    let mut artifact_descriptors = BTreeMap::new();
    for (asset, descriptor) in &manifest.entries {
        if descriptor.storage_class != ArtifactStorageClass::Shared {
            return Err(PackageError::DriverLocalArtifact {
                asset: *asset,
                key: descriptor.key,
            });
        }
        descriptor
            .validate()
            .map_err(|reason| PackageError::InvalidDescriptor {
                asset: *asset,
                key: descriptor.key,
                reason,
            })?;
        artifact_assets
            .entry(descriptor.key)
            .or_default()
            .insert(*asset);
        if let Some(existing) = artifact_descriptors.insert(descriptor.key, descriptor)
            && existing != descriptor
        {
            return Err(PackageError::ConflictingDescriptor {
                key: descriptor.key,
            });
        }
    }
    let mut indegree = artifact_descriptors
        .keys()
        .copied()
        .map(|key| (key, 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut dependents = BTreeMap::<ArtifactKey, BTreeSet<ArtifactKey>>::new();
    for (asset, descriptor) in &manifest.entries {
        for dependency in &descriptor.dependencies {
            let Some(dependency_assets) = artifact_assets.get(dependency) else {
                return Err(PackageError::MissingDependency {
                    asset: *asset,
                    dependency: *dependency,
                });
            };
            if !dependency_assets
                .iter()
                .any(|dependency_asset| by_asset.contains_key(dependency_asset))
            {
                return Err(PackageError::UnmappedDependency {
                    asset: *asset,
                    dependency: *dependency,
                });
            }
            let inserted = dependents
                .entry(*dependency)
                .or_default()
                .insert(descriptor.key);
            if inserted {
                *indegree
                    .get_mut(&descriptor.key)
                    .expect("dependency keys were inserted above") += 1;
            }
        }
    }
    let mut ready = indegree
        .iter()
        .filter_map(|(key, degree)| (*degree == 0).then_some(*key))
        .collect::<BTreeSet<_>>();
    let mut visited = 0usize;
    while let Some(key) = ready.pop_first() {
        visited += 1;
        if let Some(children) = dependents.get(&key) {
            for child in children {
                let degree = indegree.get_mut(child).expect("child has an indegree");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(*child);
                }
            }
        }
    }
    if visited != indegree.len() {
        return Err(PackageError::DependencyCycle {
            artifacts: indegree
                .into_iter()
                .filter_map(|(key, degree)| (degree != 0).then_some(key))
                .collect(),
        });
    }

    let mut writers = BTreeMap::<String, ContainerWriter>::new();
    let mut chunk_inputs = BTreeMap::<String, Vec<ChunkInput>>::new();
    let mut payload_cache = BTreeMap::<ContentHash, Vec<u8>>::new();
    for asset in &selected_assets {
        let descriptor = manifest
            .entries
            .get(asset)
            .expect("selected assets were validated above");
        let mapping = by_asset
            .get(asset)
            .expect("selected assets were checked above");
        let payload = if let Some(payload) = payload_cache.get(&descriptor.content_hash) {
            payload.clone()
        } else {
            let payload = match cas.read(descriptor.content_hash) {
                Ok(payload) => payload,
                Err(CasError::Io(error)) if error.kind() == ErrorKind::NotFound => {
                    return Err(PackageError::MissingPayload {
                        asset: *asset,
                        key: descriptor.key,
                        hash: descriptor.content_hash,
                    });
                }
                Err(CasError::Corrupt { expected, actual }) => {
                    return Err(PackageError::HashMismatch {
                        asset: *asset,
                        key: descriptor.key,
                        expected,
                        actual,
                    });
                }
                Err(source) => {
                    return Err(PackageError::Cas {
                        asset: *asset,
                        key: descriptor.key,
                        source,
                    });
                }
            };
            let actual_len =
                u64::try_from(payload.len()).map_err(|_| PackageError::LengthMismatch {
                    asset: *asset,
                    key: descriptor.key,
                    expected: descriptor.byte_len,
                    actual: u64::MAX,
                })?;
            if actual_len != descriptor.byte_len {
                return Err(PackageError::LengthMismatch {
                    asset: *asset,
                    key: descriptor.key,
                    expected: descriptor.byte_len,
                    actual: actual_len,
                });
            }
            if ContentHash::digest(&payload) != descriptor.content_hash {
                return Err(PackageError::HashMismatch {
                    asset: *asset,
                    key: descriptor.key,
                    expected: descriptor.content_hash,
                    actual: ContentHash::digest(&payload),
                });
            }
            payload_cache.insert(descriptor.content_hash, payload.clone());
            payload
        };
        let actual_len =
            u64::try_from(payload.len()).map_err(|_| PackageError::LengthMismatch {
                asset: *asset,
                key: descriptor.key,
                expected: descriptor.byte_len,
                actual: u64::MAX,
            })?;
        if actual_len != descriptor.byte_len {
            return Err(PackageError::LengthMismatch {
                asset: *asset,
                key: descriptor.key,
                expected: descriptor.byte_len,
                actual: actual_len,
            });
        }

        let local_dependencies = descriptor
            .dependencies
            .iter()
            .flat_map(|dependency| artifact_assets.get(dependency).into_iter().flatten())
            .filter_map(|dependency_asset| by_asset.get(dependency_asset))
            .filter(|dependency_mapping| {
                dependency_mapping.container_name == mapping.container_name
            })
            .map(|dependency_mapping| dependency_mapping.guid)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        chunk_inputs
            .entry(mapping.container_name.clone())
            .or_default()
            .push(ChunkInput {
                guid: mapping.guid,
                size: descriptor.byte_len,
                alignment: u64::from(descriptor.alignment),
                dependencies: local_dependencies,
                priority: mapping.chunk_priority,
            });

        let writer = writers.entry(mapping.container_name.clone()).or_default();
        if options.chunk_codec.is_some() {
            writer.add_aligned(mapping.guid, payload, descriptor.alignment)
        } else {
            match options.codec {
                prism_asset_container::Codec::Raw => {
                    writer.add_aligned(mapping.guid, payload, descriptor.alignment)
                }
                prism_asset_container::Codec::Deflate => {
                    writer.add_deflate_aligned(mapping.guid, payload, descriptor.alignment)
                }
            }
        }
        .map_err(|source| PackageError::Container {
            name: mapping.container_name.clone(),
            source,
        })?;
    }

    let mut containers = BTreeMap::new();
    for (name, writer) in writers {
        let plan = ChunkPlanner::new(options.chunk_policy)
            .plan(chunk_inputs.remove(&name).unwrap_or_default())
            .map_err(|source| PackageError::ChunkPlan {
                name: name.clone(),
                source,
            })?;
        let bytes = if let Some(codec) = options.chunk_codec {
            writer.finish_with_chunk_compression(&plan, options.chunk_policy.chunk_alignment, codec)
        } else if options.codec == prism_asset_container::Codec::Raw {
            writer.finish_with_chunk_plan(&plan, options.chunk_policy.chunk_alignment)
        } else {
            writer.finish()
        }
        .map_err(|source| PackageError::Container {
            name: name.clone(),
            source,
        })?;
        containers.insert(name, bytes);
    }

    // The route identity is derived only after the final container bytes are
    // fixed. This prevents a manifest from accepting a same-named replacement
    // whose length, content hash, or generation differs from the package.
    let mut runtime_manifest = RuntimeManifest::new();
    for asset in &selected_assets {
        let mapping = by_asset
            .get(asset)
            .expect("selected assets were checked above");
        let descriptor = manifest
            .entries
            .get(&mapping.asset)
            .expect("mapping assets were validated above");
        let container = containers
            .get(&mapping.container_name)
            .expect("every mapping has a finalized container");
        let info = ContainerReader::new(container.clone())
            .map_err(|source| PackageError::Container {
                name: mapping.container_name.clone(),
                source,
            })?
            .info();
        let hard_dependencies =
            descriptor
                .dependencies
                .iter()
                .try_fold(BTreeSet::new(), |mut routes, dependency| {
                    let dependency_assets = artifact_assets.get(dependency).ok_or(
                        PackageError::UnmappedDependency {
                            asset: mapping.asset,
                            dependency: *dependency,
                        },
                    )?;
                    for dependency_asset in dependency_assets {
                        if !selected_assets.contains(dependency_asset) {
                            continue;
                        }
                        let dependency_mapping = by_asset.get(dependency_asset).ok_or(
                            PackageError::UnmappedDependency {
                                asset: mapping.asset,
                                dependency: *dependency,
                            },
                        )?;
                        routes.insert(dependency_mapping.guid);
                    }
                    Ok::<_, PackageError>(routes)
                })?
                .into_iter()
                .collect::<Vec<_>>();
        let entry_codec = ContainerReader::new(container.clone())
            .map_err(|source| PackageError::Container {
                name: mapping.container_name.clone(),
                source,
            })?
            .entry(mapping.guid)
            .ok_or_else(|| PackageError::Container {
                name: mapping.container_name.clone(),
                source: ContainerError::MissingAsset(mapping.guid),
            })?
            .codec;
        runtime_manifest
            .insert_with_metadata(
                ManifestRoute {
                    guid: mapping.guid,
                    container_name: mapping.container_name.clone(),
                    priority: mapping.priority,
                    container_guid: mapping.guid,
                },
                ManifestMetadata {
                    hard_dependencies,
                    optional_dependencies: Vec::new(),
                    initial_residency: 0,
                    streaming_level: 0,
                    codec: entry_codec,
                    schema: descriptor.schema,
                    install_capabilities: 0,
                },
            )
            .map_err(PackageError::RuntimeManifest)?;
        runtime_manifest
            .bind_container(
                mapping.container_name.clone(),
                ContainerIdentity {
                    byte_len: info.total_len,
                    content_hash: info.content_hash,
                    generation: info.generation,
                },
            )
            .map_err(PackageError::RuntimeManifest)?;
    }
    let runtime_manifest_bytes = runtime_manifest
        .encode()
        .map_err(PackageError::RuntimeManifest)?;
    Ok(RuntimePackage {
        source_manifest_hash: ContentHash::digest(&manifest.canonical_bytes()),
        asset_guids: selected_assets
            .iter()
            .map(|asset| {
                (
                    *asset,
                    by_asset
                        .get(asset)
                        .expect("selected assets were checked above")
                        .guid,
                )
            })
            .collect(),
        containers,
        runtime_manifest,
        runtime_manifest_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_asset_container::{
        ChunkPolicy, Codec, ContainerReader, CHUNK_FORMAT_VERSION, FORMAT_VERSION,
    };
    use tempfile::tempdir;

    fn key(n: u8) -> ArtifactKey {
        ArtifactKey([n; 32])
    }

    fn manifest_and_cas() -> (Manifest, ContentAddressedStore, Vec<AssetPackageMapping>) {
        let dir = tempdir().unwrap();
        let cas = ContentAddressedStore::open(dir.keep()).unwrap();
        let root_bytes = b"root-payload";
        let leaf_bytes = b"leaf-payload";
        let root_hash = cas.put(root_bytes).unwrap();
        let leaf_hash = cas.put(leaf_bytes).unwrap();
        let manifest = Manifest::new(
            crate::types::BuildRevision(7),
            [
                crate::types::ManifestEntry {
                    asset: AssetId(1),
                    descriptor: crate::types::ArtifactDescriptor {
                        key: key(1),
                        kind: 1,
                        schema: 1,
                        storage_class: crate::types::ArtifactStorageClass::Shared,
                        content_hash: root_hash,
                        byte_len: root_bytes.len() as u64,
                        alignment: 16,
                        dependencies: vec![key(2)],
                    },
                },
                crate::types::ManifestEntry {
                    asset: AssetId(2),
                    descriptor: crate::types::ArtifactDescriptor {
                        key: key(2),
                        kind: 1,
                        schema: 1,
                        storage_class: crate::types::ArtifactStorageClass::Shared,
                        content_hash: leaf_hash,
                        byte_len: leaf_bytes.len() as u64,
                        alignment: 16,
                        dependencies: vec![],
                    },
                },
            ],
        )
        .unwrap();
        let mappings = vec![
            AssetPackageMapping {
                asset: AssetId(1),
                guid: StableGuid::from_u128(101),
                container_name: "z.container".into(),
                priority: 0,
                chunk_priority: 0,
            },
            AssetPackageMapping {
                asset: AssetId(2),
                guid: StableGuid::from_u128(202),
                container_name: "a.container".into(),
                priority: 0,
                chunk_priority: 0,
            },
        ];
        (manifest, cas, mappings)
    }

    #[test]
    fn packages_manifest_cas_and_runtime_manifest_end_to_end() {
        let (manifest, cas, mappings) = manifest_and_cas();
        let package = package_runtime(PackageRequest {
            manifest: &manifest,
            cas: &cas,
            mappings,
        })
        .unwrap();
        let decoded = RuntimeManifest::decode(&package.runtime_manifest_bytes).unwrap();
        assert_eq!(decoded, package.runtime_manifest);
        assert_eq!(
            package.source_manifest_hash,
            ContentHash::digest(&manifest.canonical_bytes())
        );
        assert_eq!(
            package.asset_guids,
            [
                (AssetId(1), StableGuid::from_u128(101)),
                (AssetId(2), StableGuid::from_u128(202)),
            ]
            .into_iter()
            .collect()
        );
        let root_metadata = decoded
            .metadata(StableGuid::from_u128(101))
            .expect("package emits route metadata");
        assert_eq!(
            root_metadata.hard_dependencies,
            vec![StableGuid::from_u128(202)]
        );
        assert_eq!(root_metadata.schema, 1);
        assert_eq!(
            package.containers.keys().collect::<Vec<_>>(),
            vec!["a.container", "z.container"]
        );
        for (name, guid, expected) in [
            (
                "a.container",
                StableGuid::from_u128(202),
                b"leaf-payload".as_slice(),
            ),
            (
                "z.container",
                StableGuid::from_u128(101),
                b"root-payload".as_slice(),
            ),
        ] {
            let reader =
                ContainerReader::new(package.container_bytes(name).unwrap().to_vec()).unwrap();
            assert_eq!(reader.read_all(guid).unwrap(), expected);
            let route = decoded.route(guid).unwrap();
            assert_eq!(route.container_name, name);
            assert_eq!(route.container_guid, guid);
        }
    }

    #[test]
    fn package_preserves_all_routes_for_a_shared_artifact_dependency() {
        let (mut manifest, cas, mut mappings) = manifest_and_cas();
        let shared = manifest.entries.get(&AssetId(2)).unwrap().clone();
        manifest.entries.insert(AssetId(3), shared);
        mappings.push(AssetPackageMapping {
            asset: AssetId(3),
            guid: StableGuid::from_u128(303),
            container_name: "a.container".into(),
            priority: 0,
            chunk_priority: 0,
        });

        let package = package_runtime(PackageRequest {
            manifest: &manifest,
            cas: &cas,
            mappings,
        })
        .unwrap();
        let root = package
            .runtime_manifest
            .metadata(StableGuid::from_u128(101))
            .unwrap();
        assert_eq!(
            root.hard_dependencies,
            vec![StableGuid::from_u128(202), StableGuid::from_u128(303)]
        );
    }

    #[test]
    fn packaging_is_deterministic_across_mapping_order() {
        let (manifest, cas, mappings) = manifest_and_cas();
        let mut reversed = mappings.clone();
        reversed.reverse();
        let first = package_runtime(PackageRequest {
            manifest: &manifest,
            cas: &cas,
            mappings,
        })
        .unwrap();
        let second = package_runtime(PackageRequest {
            manifest: &manifest,
            cas: &cas,
            mappings: reversed,
        })
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn package_runtime_consumes_chunk_capacity_alignment_and_manifest_routes() {
        let (manifest, cas, mut mappings) = manifest_and_cas();
        mappings[1].container_name = mappings[0].container_name.clone();
        let package = package_runtime_with_options(
            PackageRequest {
                manifest: &manifest,
                cas: &cas,
                mappings,
            },
            PackageOptions {
                chunk_policy: ChunkPolicy {
                    max_chunk_size: 16,
                    chunk_alignment: 64,
                    max_chunks: 2,
                },
                chunk_codec: None,
                codec: prism_asset_container::Codec::Raw,
            },
        )
        .unwrap();

        let name = "z.container";
        let reader = ContainerReader::new(package.container_bytes(name).unwrap().to_vec()).unwrap();
        assert_eq!(reader.info().version, FORMAT_VERSION);
        let root = reader.entry(StableGuid::from_u128(101)).unwrap();
        let leaf = reader.entry(StableGuid::from_u128(202)).unwrap();
        let payload_start = u64::from_le_bytes(
            package.container_bytes(name).unwrap()[36..44]
                .try_into()
                .unwrap(),
        );
        assert_eq!((root.offset - payload_start) % 64, 0);
        assert_eq!((leaf.offset - payload_start) % 64, 0);
        assert!(leaf.offset < root.offset);
        assert_eq!(root.offset - leaf.offset, 64);
        assert_eq!(reader.read_all(root.guid).unwrap(), b"root-payload");
        assert_eq!(reader.read_all(leaf.guid).unwrap(), b"leaf-payload");

        let decoded = RuntimeManifest::decode(&package.runtime_manifest_bytes).unwrap();
        assert_eq!(decoded, package.runtime_manifest);
        for guid in [root.guid, leaf.guid] {
            let route = decoded.route(guid).unwrap();
            assert_eq!(route.container_name, name);
            assert_eq!(route.container_guid, guid);
        }
    }

    #[test]
    fn package_runtime_can_emit_deterministic_deflate_payloads() {
        let (manifest, cas, mappings) = manifest_and_cas();
        let package = package_runtime_with_options(
            PackageRequest {
                manifest: &manifest,
                cas: &cas,
                mappings,
            },
            PackageOptions {
                chunk_policy: ChunkPolicy::default(),
                codec: prism_asset_container::Codec::Deflate,
                chunk_codec: None,
            },
        )
        .unwrap();

        let container = package.container_bytes("z.container").unwrap();
        let reader = ContainerReader::new(container.to_vec()).unwrap();
        let root = StableGuid::from_u128(101);
        assert_eq!(reader.entry(root).unwrap().codec, Codec::Deflate);
        assert!(!reader.read_all(root).unwrap().is_empty());
        assert_eq!(
            package.runtime_manifest.metadata(root).unwrap().codec,
            Codec::Deflate
        );
        let leaf = StableGuid::from_u128(202);
        let leaf_reader =
            ContainerReader::new(package.container_bytes("a.container").unwrap().to_vec()).unwrap();
        assert_eq!(leaf_reader.entry(leaf).unwrap().codec, Codec::Deflate);
        assert!(!leaf_reader.read_all(leaf).unwrap().is_empty());
        assert_eq!(
            package.runtime_manifest.metadata(leaf).unwrap().codec,
            Codec::Deflate
        );
        let decoded = RuntimeManifest::decode(&package.runtime_manifest_bytes).unwrap();
        assert_eq!(decoded, package.runtime_manifest);
    }

    #[test]
    fn package_runtime_can_emit_independently_compressed_chunks() {
        let (manifest, cas, mut mappings) = manifest_and_cas();
        mappings[1].container_name = mappings[0].container_name.clone();
        let package = package_runtime_with_options(
            PackageRequest {
                manifest: &manifest,
                cas: &cas,
                mappings,
            },
            PackageOptions {
                chunk_policy: ChunkPolicy {
                    max_chunk_size: 64,
                    chunk_alignment: 64,
                    max_chunks: 2,
                },
                codec: Codec::Raw,
                chunk_codec: Some(Codec::Deflate),
            },
        )
        .unwrap();
        let container = package.container_bytes("z.container").unwrap();
        let reader = ContainerReader::new(container.to_vec()).unwrap();
        assert_eq!(reader.info().version, CHUNK_FORMAT_VERSION);
        assert_eq!(reader.info().chunk_count, 2);
        for guid in [StableGuid::from_u128(101), StableGuid::from_u128(202)] {
            assert_eq!(reader.entry(guid).unwrap().codec, Codec::Deflate);
            assert!(!reader.read_all(guid).unwrap().is_empty());
            assert_eq!(
                package.runtime_manifest.metadata(guid).unwrap().codec,
                Codec::Deflate
            );
        }
    }

    #[test]
    fn package_runtime_rejects_chunk_capacity_and_count_overflow() {
        let (manifest, cas, mappings) = manifest_and_cas();
        assert!(matches!(
            package_runtime_with_options(
                PackageRequest {
                    manifest: &manifest,
                    cas: &cas,
                    mappings: mappings.clone(),
                },
                PackageOptions {
                    chunk_policy: ChunkPolicy {
                        max_chunk_size: 4,
                        chunk_alignment: 4,
                        max_chunks: 1,
                    },
                    chunk_codec: None,
                    codec: prism_asset_container::Codec::Raw,
                },
            ),
            Err(PackageError::ChunkPlan { .. })
        ));

        let mut together = mappings;
        together[1].container_name = together[0].container_name.clone();
        assert!(matches!(
            package_runtime_with_options(
                PackageRequest {
                    manifest: &manifest,
                    cas: &cas,
                    mappings: together,
                },
                PackageOptions {
                    chunk_policy: ChunkPolicy {
                        max_chunk_size: 16,
                        chunk_alignment: 64,
                        max_chunks: 1,
                    },
                    chunk_codec: None,
                    codec: prism_asset_container::Codec::Raw,
                },
            ),
            Err(PackageError::ChunkPlan { .. })
        ));
    }

    #[test]
    fn rejects_missing_unknown_duplicate_and_unmapped_dependency() {
        let (manifest, cas, mut mappings) = manifest_and_cas();
        mappings.pop();
        assert!(matches!(
            package_runtime(PackageRequest {
                manifest: &manifest,
                cas: &cas,
                mappings
            }),
            Err(PackageError::MissingMapping { asset: AssetId(2) })
        ));

        let (_, cas, mut mappings) = manifest_and_cas();
        mappings[0].asset = AssetId(99);
        assert!(matches!(
            package_runtime(PackageRequest {
                manifest: &manifest,
                cas: &cas,
                mappings
            }),
            Err(PackageError::UnknownAsset { asset: AssetId(99) })
        ));

        let (_, cas, mut mappings) = manifest_and_cas();
        mappings.push(mappings[0].clone());
        assert!(matches!(
            package_runtime(PackageRequest {
                manifest: &manifest,
                cas: &cas,
                mappings
            }),
            Err(PackageError::DuplicateAssetMapping { asset: AssetId(1) })
        ));

        let (manifest, cas, mut mappings) = manifest_and_cas();
        mappings[1].guid = mappings[0].guid;
        assert!(matches!(
            package_runtime(PackageRequest {
                manifest: &manifest,
                cas: &cas,
                mappings
            }),
            Err(PackageError::DuplicateGuid { .. })
        ));
    }

    #[test]
    fn rejects_missing_dependency_payload_hash_and_length() {
        let (manifest, cas, mappings) = manifest_and_cas();
        let missing = Manifest::new(
            manifest.revision,
            manifest
                .entries
                .values()
                .cloned()
                .enumerate()
                .map(|(i, mut descriptor)| {
                    if i == 0 {
                        descriptor.dependencies = vec![key(99)];
                    }
                    crate::types::ManifestEntry {
                        asset: AssetId((i + 1) as u128),
                        descriptor,
                    }
                }),
        )
        .unwrap();
        assert!(matches!(
            package_runtime(PackageRequest {
                manifest: &missing,
                cas: &cas,
                mappings: mappings.clone()
            }),
            Err(PackageError::MissingDependency { .. })
        ));

        let mut no_payload = manifest.clone();
        no_payload
            .entries
            .get_mut(&AssetId(1))
            .unwrap()
            .content_hash = ContentHash([9; 32]);
        assert!(matches!(
            package_runtime(PackageRequest {
                manifest: &no_payload,
                cas: &cas,
                mappings: mappings.clone()
            }),
            Err(PackageError::MissingPayload {
                asset: AssetId(1),
                ..
            })
        ));

        let mut wrong_length = manifest;
        wrong_length.entries.get_mut(&AssetId(1)).unwrap().byte_len += 1;
        assert!(matches!(
            package_runtime(PackageRequest {
                manifest: &wrong_length,
                cas: &cas,
                mappings
            }),
            Err(PackageError::LengthMismatch {
                asset: AssetId(1),
                ..
            })
        ));
    }

    #[test]
    fn rejects_conflicting_descriptors_and_dependency_cycles() {
        let (manifest, cas, mappings) = manifest_and_cas();
        let mut conflicting = manifest.clone();
        let second = conflicting.entries.get_mut(&AssetId(2)).unwrap();
        second.key = key(1);
        second.schema = 2;
        assert!(matches!(
            package_runtime(PackageRequest {
                manifest: &conflicting,
                cas: &cas,
                mappings: mappings.clone()
            }),
            Err(PackageError::ConflictingDescriptor { key: actual_key }) if actual_key == key(1)
        ));

        let mut cyclic = manifest;
        cyclic.entries.get_mut(&AssetId(2)).unwrap().dependencies = vec![key(1)];
        assert!(matches!(
            package_runtime(PackageRequest {
                manifest: &cyclic,
                cas: &cas,
                mappings
            }),
            Err(PackageError::DependencyCycle { .. })
        ));
    }

    #[test]
    fn rejects_runtime_invalid_container_names() {
        let (manifest, cas, mut mappings) = manifest_and_cas();
        mappings[0].container_name = "".into();
        assert!(matches!(
            package_runtime(PackageRequest {
                manifest: &manifest,
                cas: &cas,
                mappings
            }),
            Err(PackageError::RuntimeManifest(ManifestError::InvalidName))
        ));

        let (_, cas, mut mappings) = manifest_and_cas();
        mappings[0].guid = StableGuid::NIL;
        assert!(matches!(
            package_runtime(PackageRequest {
                manifest: &manifest,
                cas: &cas,
                mappings
            }),
            Err(PackageError::InvalidGuid { asset: AssetId(1) })
        ));
    }
}
