//! Real package/patch regressions for optional dependency context and hard edges.

use prism_asset::StableGuid;
use prism_asset_build_core::{
    package_runtime, package_runtime_patch, ArtifactDescriptor, ArtifactKey, AssetId,
    AssetPackageMapping, BuildRevision, ContentAddressedStore, Manifest, ManifestEntry,
    PackageError, PackageRequest, RuntimePackage, RuntimePatchError, RuntimePatchPackage,
    RuntimePatchRequest,
};
use prism_asset_container::{ContainerReader, OverlaySet, RuntimeManifest};
use tempfile::{tempdir, TempDir};

const CHANGED: &[u8] = b"changed root";
const CONTEXT: &[u8] = b"stable hard context";

fn guid(asset: u128) -> StableGuid {
    StableGuid::from_u128(asset * 11)
}

fn key(value: u8) -> ArtifactKey {
    ArtifactKey([value; 32])
}

fn entry(asset: u128, artifact: u8, bytes: &[u8], dependencies: Vec<ArtifactKey>) -> ManifestEntry {
    ManifestEntry {
        asset: AssetId(asset),
        descriptor: ArtifactDescriptor::for_bytes(key(artifact), 1, 1, 16, dependencies, bytes),
    }
}

fn mappings(manifest: &Manifest) -> Vec<AssetPackageMapping> {
    manifest
        .entries
        .keys()
        .map(|asset| AssetPackageMapping {
            asset: *asset,
            guid: guid(asset.0),
            container_name: format!("base_{}", asset.0),
            priority: 0,
            chunk_priority: 0,
        })
        .collect()
}

struct Fixture {
    _directory: TempDir,
    cas: ContentAddressedStore,
    base: Manifest,
    package: RuntimePackage,
}

impl Fixture {
    fn new(optional_edges: &[(u128, &[u128])]) -> Self {
        let directory = tempdir().unwrap();
        let cas = ContentAddressedStore::open(directory.path().join("cas")).unwrap();
        for bytes in [
            b"base root".as_slice(),
            CONTEXT,
            b"optional target",
            CHANGED,
        ] {
            cas.put(bytes).unwrap();
        }
        let base = Manifest::new(
            BuildRevision(1),
            [
                entry(1, 1, b"base root", vec![key(2)]),
                entry(2, 2, CONTEXT, vec![]),
                entry(3, 3, b"optional target", vec![]),
            ],
        )
        .unwrap();
        let mut package = package_runtime(PackageRequest {
            manifest: &base,
            cas: &cas,
            mappings: mappings(&base),
        })
        .unwrap();
        // Soft references live in runtime metadata, not ArtifactDescriptor.
        // Keep the real package's object and canonical bytes in sync.
        for &(asset, dependencies) in optional_edges {
            let mut metadata = package
                .runtime_manifest
                .metadata(guid(asset))
                .unwrap()
                .clone();
            metadata.optional_dependencies = dependencies.iter().copied().map(guid).collect();
            package
                .runtime_manifest
                .set_metadata(guid(asset), metadata)
                .unwrap();
        }
        package.runtime_manifest_bytes = package.runtime_manifest.encode().unwrap();
        Self {
            _directory: directory,
            cas,
            base,
            package,
        }
    }

    fn target(&self) -> Manifest {
        let mut target = self.base.clone();
        target.revision = BuildRevision(2);
        target
            .entries
            .insert(AssetId(1), entry(1, 9, CHANGED, vec![key(2)]).descriptor);
        target
    }

    fn patch(&self, target: &Manifest) -> Result<RuntimePatchPackage, RuntimePatchError> {
        package_runtime_patch(RuntimePatchRequest {
            base_manifest: &self.base,
            target_manifest: target,
            base_package: &self.package,
            cas: &self.cas,
            target_mappings: mappings(target),
        })
    }

    fn assert_applies(&self, patch: &RuntimePatchPackage, expected_root: &[u8]) -> RuntimeManifest {
        assert_eq!(
            RuntimeManifest::decode(&patch.payload.runtime_manifest_bytes).unwrap(),
            patch.payload.runtime_manifest,
        );
        assert_eq!(
            RuntimeManifest::decode(&patch.target_runtime_manifest_bytes).unwrap(),
            patch.target_runtime_manifest,
        );
        let applied = patch
            .apply_to_manifest(
                self.package.source_manifest_hash,
                &self.package.runtime_manifest,
            )
            .unwrap();
        assert_eq!(applied, patch.target_runtime_manifest);

        let mut overlays = OverlaySet::new();
        for package in [&self.package, &patch.payload] {
            for (name, bytes) in &package.containers {
                overlays.mount(
                    name.clone(),
                    0,
                    ContainerReader::new(bytes.clone()).unwrap(),
                );
            }
        }
        overlays.mount_tombstones("removed", 1, patch.descriptor.removed_guids.iter().copied());
        let reader = applied.bind(&overlays).unwrap();
        assert_eq!(reader.read_all(guid(1)).unwrap(), expected_root);
        assert_eq!(reader.read_all(guid(2)).unwrap(), CONTEXT);
        applied
    }
}

#[test]
fn patch_allows_missing_optional_context_target() {
    let fixture = Fixture::new(&[(2, &[99])]);
    let original = fixture.package.clone();
    let patch = fixture.patch(&fixture.target()).unwrap();

    assert_eq!(patch.payload.asset_guids.len(), 1);
    assert_eq!(patch.payload.containers.len(), 1);
    assert_eq!(patch.payload.runtime_manifest.routes().count(), 2);
    assert!(patch.payload.runtime_manifest.route(guid(99)).is_none());
    assert_eq!(
        patch
            .payload
            .runtime_manifest
            .metadata(guid(2))
            .unwrap()
            .optional_dependencies,
        vec![guid(99)],
    );
    let applied = fixture.assert_applies(&patch, CHANGED);
    assert_eq!(
        applied.metadata(guid(2)),
        fixture.package.runtime_manifest.metadata(guid(2))
    );
    assert_eq!(fixture.package, original);
}

#[test]
fn patch_includes_existing_optional_context_without_copying_its_payload() {
    let fixture = Fixture::new(&[(2, &[3])]);
    let patch = fixture.patch(&fixture.target()).unwrap();

    assert_eq!(patch.payload.runtime_manifest.routes().count(), 3);
    assert_eq!(patch.payload.asset_guids.len(), 1);
    assert_eq!(patch.payload.containers.len(), 1);
    let context = patch.payload.runtime_manifest.route(guid(3)).unwrap();
    assert_eq!(
        context,
        fixture.package.runtime_manifest.route(guid(3)).unwrap()
    );
    assert!(!patch
        .payload
        .containers
        .contains_key(&context.container_name));
    fixture.assert_applies(&patch, CHANGED);
}

#[test]
fn patch_optional_context_cycles_and_self_references_terminate() {
    let cases: &[&[(u128, &[u128])]] = &[&[(2, &[3]), (3, &[2])], &[(2, &[2])]];
    for edges in cases {
        let fixture = Fixture::new(edges);
        let patch = fixture.patch(&fixture.target()).unwrap();
        assert_eq!(patch.payload.asset_guids.len(), 1);
        assert_eq!(patch.payload.containers.len(), 1);
        for &(asset, _) in *edges {
            assert_eq!(
                patch.payload.runtime_manifest.metadata(guid(asset)),
                fixture.package.runtime_manifest.metadata(guid(asset)),
            );
        }
        fixture.assert_applies(&patch, CHANGED);
    }
}

#[test]
fn patch_can_remove_an_optional_context_target() {
    let fixture = Fixture::new(&[(2, &[3])]);
    let mut target = fixture.target();
    target.entries.remove(&AssetId(3));
    let patch = fixture.patch(&target).unwrap();

    assert_eq!(patch.descriptor.removed_guids, vec![guid(3)]);
    assert!(patch.payload.runtime_manifest.route(guid(3)).is_none());
    assert_eq!(patch.payload.runtime_manifest.routes().count(), 2);
    assert_eq!(patch.payload.containers.len(), 1);
    let applied = fixture.assert_applies(&patch, CHANGED);
    assert!(applied.route(guid(3)).is_none());
    assert_eq!(
        applied.metadata(guid(2)).unwrap().optional_dependencies,
        vec![guid(3)]
    );
}

#[test]
fn empty_patch_preserves_missing_optional_references() {
    let fixture = Fixture::new(&[(2, &[99])]);
    let mut target = fixture.base.clone();
    target.revision = BuildRevision(2);
    let patch = fixture.patch(&target).unwrap();

    assert!(patch.payload.containers.is_empty());
    assert!(patch.payload.runtime_manifest.routes().next().is_none());
    assert_eq!(
        fixture.assert_applies(&patch, b"base root"),
        fixture.package.runtime_manifest
    );
}

#[test]
fn patch_rejects_missing_hard_target_even_with_optional_references() {
    let fixture = Fixture::new(&[(2, &[99])]);
    let original = fixture.package.clone();
    let mut target = fixture.target();
    target.entries.remove(&AssetId(2));

    assert!(matches!(
        fixture.patch(&target),
        Err(RuntimePatchError::Package(PackageError::MissingDependency {
            asset: AssetId(1),
            dependency,
        })) if dependency == key(2)
    ));
    assert_eq!(fixture.package, original);
}

#[test]
fn patch_rejects_hard_cycles_even_with_optional_references() {
    let fixture = Fixture::new(&[(2, &[3])]);
    let mut target = fixture.target();
    target.entries.get_mut(&AssetId(2)).unwrap().dependencies = vec![key(9)];

    assert!(matches!(
        fixture.patch(&target),
        Err(RuntimePatchError::Package(
            PackageError::DependencyCycle { .. }
        ))
    ));
}
