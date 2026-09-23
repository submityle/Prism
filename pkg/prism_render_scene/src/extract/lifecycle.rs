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

pub(crate) fn retire_unused_geometry(
    mut events: Extract<MessageReader<AssetEvent<bevy_mesh::Mesh>>>,
    mut scene: ResMut<RenderGpuScene>,
    mut geometry: ResMut<RenderGeometryRegistry>,
) {
    for event in events.read() {
        if let AssetEvent::Unused { id } = event {
            scene.retire_geometry(*id);
            geometry.retire(*id);
        }
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
