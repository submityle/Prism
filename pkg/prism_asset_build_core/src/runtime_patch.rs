//! Build-to-runtime incremental package creation.
//!
//! This module turns a deterministic Build Manifest diff into a real runtime
//! delta: only added/changed payloads are emitted, removed or replaced routes
//! become tombstones, and the resulting descriptor binds both revisions and
//! both runtime-manifest identities.

use std::collections::BTreeSet;

use prism_asset::StableGuid;
use prism_asset_container::{ManifestError, PatchDescriptorError, RuntimePatchDescriptor};

use crate::cas::ContentAddressedStore;
use crate::package::{
    package_runtime_selected_with_options, AssetPackageMapping, PackageError, PackageOptions,
    PackageRequest, RuntimePackage,
};
use crate::patch::{plan_patch, PatchChange, PatchPlanError};
use crate::types::{AssetId, BuildRevision, ContentHash, Manifest};

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
                let target_guid = target_mappings
                    .iter()
                    .find(|mapping| mapping.asset == entry.asset)
                    .map(|mapping| mapping.guid);
                if target_guid.is_some_and(|target_guid| target_guid != base_guid) {
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

    let target_mappings =
        namespace_changed_mappings(target_mappings, &selected_assets, target_manifest.revision);
    let payload = package_runtime_selected_with_options(
        PackageRequest {
            manifest: target_manifest,
            cas,
            mappings: target_mappings,
        },
        options,
        selected_assets.iter().copied(),
    )?;

    let mut target_runtime_manifest = base_package.runtime_manifest.clone();
    let removed_guids = removed_guids.into_iter().collect::<Vec<_>>();
    target_runtime_manifest.apply_patch(&payload.runtime_manifest, &removed_guids)?;
    let target_runtime_manifest_bytes = target_runtime_manifest
        .encode()
        .map_err(RuntimePatchError::Manifest)?;
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
    })
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
            ContentHash::digest(&merged.encode().unwrap()).0,
            patch.descriptor.target_runtime_manifest_hash
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
            .payload
            .runtime_manifest
            .metadata(StableGuid::from_u128(22))
            .unwrap();
        assert_eq!(metadata.hard_dependencies, vec![StableGuid::from_u128(11)]);
    }
}
