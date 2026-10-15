//! Scene-side bridge for the shared mesh consumer contract.
//!
//! `prism_render_scene` owns the Bevy adapter and therefore resolves the
//! engine-neutral identity before a render command touches a prepared mesh or
//! allocator slice.  The physical source remains the Bevy render asset; this
//! module is the single semantic gate for the scene consumers.

use bevy_asset::AssetId;
use bevy_mesh::Mesh;
use prism_render_architecture::geometry::{
    MeshConsumer, MeshConsumerBinding, MeshConsumerBindingError, MeshConsumerSelection,
    MeshRepresentation,
};
use prism_render_architecture::gpu_scene::GeometryHandle;

use super::runtime::RenderGeometryRegistry;

/// Logical selection returned to a scene draw consumer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RenderSceneMeshSelection {
    /// Stable geometry identity used by the GPU Scene.
    pub geometry: GeometryHandle,
    /// Representation selected by the shared contract.
    pub representation: MeshRepresentation,
    /// Whether the declared fallback was selected.
    pub used_fallback: bool,
}

/// Resolve one prepared asset through the shared render-consumer contract.
pub(crate) fn select_render_mesh(
    registry: &RenderGeometryRegistry,
    asset: AssetId<Mesh>,
    consumer: MeshConsumer,
    requested_lod: u32,
) -> Result<RenderSceneMeshSelection, MeshConsumerBindingError> {
    let geometry = registry
        .handle(asset)
        .ok_or(MeshConsumerBindingError::InvalidGeometry)?;
    let record = registry
        .record(geometry)
        .ok_or(MeshConsumerBindingError::InvalidGeometry)?;
    let fallback = record
        .lods
        .iter()
        .find(|lod| lod.resident && lod.fallback)
        .map(|lod| lod.level)
        .ok_or(MeshConsumerBindingError::InvalidRevision)?;
    let binding = MeshConsumerBinding {
        geometry,
        revision: record.revision,
        current: MeshRepresentation::StaticLod {
            level: requested_lod,
        },
        previous: MeshRepresentation::StaticLod {
            level: requested_lod,
        },
        history_valid: false,
        fallback: MeshRepresentation::StaticLod { level: fallback },
        derived: None,
    };
    let preferred_resident = record
        .lods
        .iter()
        .any(|lod| lod.level == requested_lod && lod.resident);
    let selection = binding.select_with_fallback(consumer, preferred_resident)?;
    match selection {
        MeshConsumerSelection::Render {
            geometry,
            representation,
            used_fallback,
        } => Ok(RenderSceneMeshSelection {
            geometry,
            representation,
            used_fallback,
        }),
        MeshConsumerSelection::Motion { .. } | MeshConsumerSelection::Derived { .. } => {
            Err(MeshConsumerBindingError::MissingDerivedArtifact)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::uuid::Uuid;
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::geometry::{GeometryLodRecord, GeometryRecord};

    fn asset() -> AssetId<Mesh> {
        AssetId::Uuid {
            uuid: Uuid::from_u128(99),
        }
    }

    fn registry() -> RenderGeometryRegistry {
        let mut registry = RenderGeometryRegistry::default();
        registry.upsert(
            asset(),
            GeometryRecord {
                handle: GenerationalHandle::new(2, 1),
                revision: 4,
                lods: vec![
                    GeometryLodRecord {
                        level: 0,
                        resident: true,
                        fallback: true,
                        ..Default::default()
                    },
                    GeometryLodRecord {
                        level: 1,
                        resident: false,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
        );
        registry
    }

    #[test]
    fn resident_lod_uses_shared_main_view_contract() {
        let selected = select_render_mesh(&registry(), asset(), MeshConsumer::MainView, 0).unwrap();
        assert_eq!(selected.geometry, GenerationalHandle::new(2, 1));
        assert_eq!(
            selected.representation,
            MeshRepresentation::StaticLod { level: 0 }
        );
        assert!(!selected.used_fallback);
    }

    #[test]
    fn nonresident_lod_uses_explicit_fallback_for_shadow() {
        let selected = select_render_mesh(&registry(), asset(), MeshConsumer::Shadow, 1).unwrap();
        assert_eq!(
            selected.representation,
            MeshRepresentation::StaticLod { level: 0 }
        );
        assert!(selected.used_fallback);
    }
}
