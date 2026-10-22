//! Bridge committed mesh-derived artifacts into the runtime consumer contract.

use crate::types::{ArtifactDescriptor, ArtifactKey, ArtifactStorageClass, AssetId, Manifest};
use prism_render_architecture::geometry::MeshDerivedArtifact;

/// Why a committed manifest cannot bind a derived mesh product to its source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshDerivedBindingError {
    /// The source mesh asset is absent from the selected manifest revision.
    MissingSourceAsset,
    /// The requested derived product is absent from the selected manifest revision.
    MissingDerivedAsset,
    /// The manifest revision cannot be represented by the runtime mesh contract.
    InvalidRevision,
    /// The expected protocol kind or schema is zero.
    InvalidExpectedProtocol,
    /// A source or derived descriptor is malformed or driver-local.
    InvalidDescriptor,
    /// The derived descriptor's kind does not match the requested consumer product.
    KindMismatch,
    /// The derived descriptor's schema does not match the requested consumer product.
    SchemaMismatch,
    /// The derived descriptor does not depend on the exact source mesh artifact key.
    MissingSourceDependency,
}

/// Resolves a derived artifact from one committed manifest and binds it to the
/// exact mesh artifact key and manifest revision it was built from.
///
/// The caller supplies the product kind/schema expected by the consumer. The
/// manifest is rechecked here because its fields are public and it may have
/// been assembled outside the publication validator. Only shared artifacts
/// with valid descriptors are eligible for runtime geometry consumers.
pub fn bind_mesh_derived_artifact(
    manifest: &Manifest,
    source_asset: AssetId,
    derived_asset: AssetId,
    expected_kind: u128,
    expected_schema: u32,
) -> Result<MeshDerivedArtifact, MeshDerivedBindingError> {
    let source_revision = u32::try_from(manifest.revision.0)
        .ok()
        .filter(|revision| *revision != 0)
        .ok_or(MeshDerivedBindingError::InvalidRevision)?;
    if expected_kind == 0 || expected_schema == 0 {
        return Err(MeshDerivedBindingError::InvalidExpectedProtocol);
    }

    let source = manifest
        .entries
        .get(&source_asset)
        .ok_or(MeshDerivedBindingError::MissingSourceAsset)?;
    let derived = manifest
        .entries
        .get(&derived_asset)
        .ok_or(MeshDerivedBindingError::MissingDerivedAsset)?;
    validate_shared_descriptor(source)?;
    validate_shared_descriptor(derived)?;
    if derived.kind != expected_kind {
        return Err(MeshDerivedBindingError::KindMismatch);
    }
    if derived.schema != expected_schema {
        return Err(MeshDerivedBindingError::SchemaMismatch);
    }
    if !derived.dependencies.contains(&source.key) {
        return Err(MeshDerivedBindingError::MissingSourceDependency);
    }

    Ok(MeshDerivedArtifact {
        key: derived.key.0,
        kind: derived.kind,
        schema: derived.schema,
        source_revision,
    })
}

fn validate_shared_descriptor(
    descriptor: &ArtifactDescriptor,
) -> Result<(), MeshDerivedBindingError> {
    if descriptor.storage_class != ArtifactStorageClass::Shared || descriptor.validate().is_err() {
        return Err(MeshDerivedBindingError::InvalidDescriptor);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BuildRevision, ContentHash, ManifestEntry};

    const COLLISION_KIND: u128 = 0x5052_4953_4d5f_434f_4c4c_4953_494f_4e31;
    const COLLISION_SCHEMA: u32 = 2;
    const SOURCE_ASSET: AssetId = AssetId(10);
    const DERIVED_ASSET: AssetId = AssetId(20);

    fn descriptor(
        key: u8,
        kind: u128,
        schema: u32,
        dependencies: Vec<ArtifactKey>,
    ) -> ArtifactDescriptor {
        ArtifactDescriptor {
            key: ArtifactKey([key; 32]),
            kind,
            schema,
            storage_class: ArtifactStorageClass::Shared,
            content_hash: ContentHash([key; 32]),
            byte_len: 4,
            alignment: 4,
            dependencies,
        }
    }

    fn manifest(revision: u64) -> Manifest {
        let source = descriptor(1, 0x4d455348, 1, Vec::new());
        let derived = descriptor(2, COLLISION_KIND, COLLISION_SCHEMA, vec![source.key]);
        Manifest::new(
            BuildRevision(revision),
            [
                ManifestEntry {
                    asset: SOURCE_ASSET,
                    descriptor: source,
                },
                ManifestEntry {
                    asset: DERIVED_ASSET,
                    descriptor: derived,
                },
            ],
        )
        .unwrap()
    }

    fn bind(manifest: &Manifest) -> Result<MeshDerivedArtifact, MeshDerivedBindingError> {
        bind_mesh_derived_artifact(
            manifest,
            SOURCE_ASSET,
            DERIVED_ASSET,
            COLLISION_KIND,
            COLLISION_SCHEMA,
        )
    }

    #[test]
    fn binds_manifest_key_kind_schema_and_committed_revision() {
        let artifact = bind(&manifest(42)).unwrap();
        assert_eq!(artifact.key, [2; 32]);
        assert_eq!(artifact.kind, COLLISION_KIND);
        assert_eq!(artifact.schema, COLLISION_SCHEMA);
        assert_eq!(artifact.source_revision, 42);
    }

    #[test]
    fn committed_manifest_binding_is_accepted_by_collision_consumer_contract() {
        use prism_render_architecture::geometry::{
            MeshConsumer, MeshConsumerBinding, MeshConsumerSelection, MeshRepresentation,
        };
        use prism_render_architecture::gpu_scene::GeometryHandle;

        let artifact = bind(&manifest(42)).unwrap();
        let binding = MeshConsumerBinding {
            geometry: GeometryHandle::new(3, 1),
            revision: 42,
            current: MeshRepresentation::StaticLod { level: 0 },
            previous: MeshRepresentation::StaticLod { level: 0 },
            history_valid: false,
            fallback: MeshRepresentation::StaticLod { level: 0 },
            derived: Some(artifact),
        };
        assert!(matches!(
            binding.select(MeshConsumer::Collision),
            Ok(MeshConsumerSelection::Derived { artifact: selected, .. }) if selected == artifact
        ));
    }

    #[test]
    fn rejects_missing_assets_wrong_protocol_or_missing_dependency() {
        let mut missing_source = manifest(1);
        missing_source.entries.remove(&SOURCE_ASSET);
        assert_eq!(
            bind(&missing_source),
            Err(MeshDerivedBindingError::MissingSourceAsset)
        );

        let mut missing_product = manifest(1);
        missing_product.entries.remove(&DERIVED_ASSET);
        assert_eq!(
            bind(&missing_product),
            Err(MeshDerivedBindingError::MissingDerivedAsset)
        );

        assert_eq!(
            bind_mesh_derived_artifact(&manifest(1), SOURCE_ASSET, DERIVED_ASSET, 0, 1),
            Err(MeshDerivedBindingError::InvalidExpectedProtocol)
        );

        let mut wrong_kind = manifest(1);
        wrong_kind.entries.get_mut(&DERIVED_ASSET).unwrap().kind += 1;
        assert_eq!(
            bind(&wrong_kind),
            Err(MeshDerivedBindingError::KindMismatch)
        );

        let mut wrong_schema = manifest(1);
        wrong_schema.entries.get_mut(&DERIVED_ASSET).unwrap().schema += 1;
        assert_eq!(
            bind(&wrong_schema),
            Err(MeshDerivedBindingError::SchemaMismatch)
        );

        let mut detached = manifest(1);
        detached
            .entries
            .get_mut(&DERIVED_ASSET)
            .unwrap()
            .dependencies
            .clear();
        assert_eq!(
            bind(&detached),
            Err(MeshDerivedBindingError::MissingSourceDependency)
        );
    }

    #[test]
    fn rejects_unrepresentable_revision_and_driver_local_descriptors() {
        assert_eq!(
            bind(&manifest(u64::from(u32::MAX) + 1)),
            Err(MeshDerivedBindingError::InvalidRevision)
        );

        let mut local = manifest(1);
        local.entries.get_mut(&DERIVED_ASSET).unwrap().storage_class =
            ArtifactStorageClass::DriverLocal;
        assert_eq!(
            bind(&local),
            Err(MeshDerivedBindingError::InvalidDescriptor)
        );
    }
}
