use bevy_asset::AssetEvent;
use bevy_ecs::{
    entity::Entity,
    message::MessageReader,
    prelude::{ResMut, Resource},
};
use bevy_render::Extract;
use prism_render_architecture::gpu_scene::{SceneOperation, SceneTransactionBuilder};

use crate::{
    buffers::GpuSceneBuffers, completion::GpuCompletionTracker, diagnostics::GpuSceneDiagnostics,
    geometry::RenderGeometryRegistry, scene::RenderGpuScene,
};

const EXTRACT_PRODUCER: u32 = 1;

fn retire_geometry_event(
    event: &AssetEvent<bevy_mesh::Mesh>,
    scene: &mut RenderGpuScene,
    geometry: &mut RenderGeometryRegistry,
) {
    let id = match event {
        AssetEvent::Removed { id } | AssetEvent::Unused { id } => *id,
        _ => return,
    };

    scene.retire_geometry(id);
    scene.retire_geometry_lod(id);
    if geometry.retire(id).is_none() {
        geometry.retire_lod_asset(id);
    }
}

pub(crate) fn retire_unused_geometry(
    mut events: Extract<MessageReader<AssetEvent<bevy_mesh::Mesh>>>,
    mut scene: ResMut<RenderGpuScene>,
    mut geometry: ResMut<RenderGeometryRegistry>,
) {
    for event in events.read() {
        retire_geometry_event(event, &mut scene, &mut geometry);
    }
}

pub(crate) fn destroy_removed_entities(
    removed_entities: &[Entity],
    scene: &mut RenderGpuScene,
    buffers: &mut GpuSceneBuffers,
    completion: &GpuCompletionTracker,
    diagnostics: &mut GpuSceneDiagnostics,
    clock: &mut ExtractionClock,
) {
    let removed: Vec<_> = removed_entities
        .iter()
        .filter_map(|&entity| {
            scene
                .handle_for_entity(entity)
                .map(|handle| (entity, handle))
        })
        .collect();
    if removed.is_empty() {
        return;
    }
    clock.advance();
    let mut transaction =
        SceneTransactionBuilder::for_producer(clock.frame_epoch, clock.sequence, EXTRACT_PRODUCER);
    for &(_, handle) in &removed {
        transaction.push(SceneOperation::Destroy { handle });
    }
    let report = scene.apply_entity_transaction(buffers, &transaction.finish());
    diagnostics.destroyed = report.destroyed;
    diagnostics.transaction_errors = report.errors.len() as u32;
    diagnostics.active_instances = scene.snapshot().instance_count;
    diagnostics.scene_epoch = report.scene_epoch;
    if report.errors.is_empty() {
        for (entity, handle) in removed {
            scene.remove_entity(entity);
            let _ = scene.retire(handle, completion);
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct ExtractionClock {
    pub frame_epoch: u64,
    pub sequence: u64,
}

impl ExtractionClock {
    pub fn advance(&mut self) {
        self.frame_epoch += 1;
        self.sequence += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::{uuid::Uuid, AssetId};
    use prism_render_architecture::{
        abi::GenerationalHandle,
        geometry::{GeometryLodRecord, GeometryRecord},
    };

    fn seeded_geometry() -> (
        AssetId<bevy_mesh::Mesh>,
        AssetId<bevy_mesh::Mesh>,
        GenerationalHandle,
        RenderGpuScene,
        RenderGeometryRegistry,
    ) {
        let base = AssetId::Uuid {
            uuid: Uuid::from_u128(100),
        };
        let lod = AssetId::Uuid {
            uuid: Uuid::from_u128(101),
        };
        let handle = GenerationalHandle::new(1, 1);

        let mut scene = RenderGpuScene::new(8);
        scene.bind_geometry_mesh(handle, base);
        scene.bind_geometry_lod_meshes(handle, &[(lod, 1.0)]);

        let mut geometry = RenderGeometryRegistry::default();
        geometry.upsert(
            base,
            GeometryRecord {
                handle,
                lods: vec![
                    GeometryLodRecord {
                        level: 0,
                        resident: true,
                        fallback: true,
                        ..Default::default()
                    },
                    GeometryLodRecord {
                        level: 1,
                        resident: true,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
        );
        assert!(geometry.upsert_lod_asset(handle, lod, 1, 2, 3));

        (base, lod, handle, scene, geometry)
    }

    #[test]
    fn removed_and_unused_retire_base_and_lod_geometry() {
        for removed in [true, false] {
            let (base, lod, handle, mut scene, mut geometry) = seeded_geometry();
            let event = if removed {
                AssetEvent::Removed { id: base }
            } else {
                AssetEvent::Unused { id: base }
            };
            retire_geometry_event(&event, &mut scene, &mut geometry);

            assert_eq!(scene.geometry_handle(base), None);
            assert_eq!(scene.geometry_lod_assets(handle).next(), None);
            assert_eq!(geometry.handle(base), None);
            assert_eq!(geometry.handle(lod), None);

            let (base, lod, handle, mut scene, mut geometry) = seeded_geometry();
            let event = if removed {
                AssetEvent::Removed { id: lod }
            } else {
                AssetEvent::Unused { id: lod }
            };
            retire_geometry_event(&event, &mut scene, &mut geometry);

            assert_eq!(scene.geometry_handle(base), Some(handle));
            assert_eq!(scene.geometry_lod_assets(handle).next(), None);
            assert_eq!(geometry.handle(base), Some(handle));
            assert_eq!(geometry.handle(lod), None);
            assert!(geometry.record(handle).is_some());
        }
    }
}
