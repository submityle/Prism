use bevy_camera::primitives::Aabb;
use bevy_ecs::{lifecycle::RemovedComponents, prelude::*};
use bevy_math::{Affine3, Affine3Ext as _};
use bevy_mesh::Mesh3d;
use bevy_render::{sync_world::RenderEntity, Extract};
use bevy_transform::components::GlobalTransform;
use prism_render_architecture::gpu_scene::{
    GeometryHandle, InstanceRecord, SceneBounds, SceneOperation, SceneTransactionBuilder,
    SceneTransform,
};

use crate::{
    buffers::GpuSceneBuffers,
    completion::GpuCompletionTracker,
    diagnostics::GpuSceneDiagnostics,
    extract::{ExtractedSceneInstance, PrismGpuSceneEntity},
    scene::RenderGpuScene,
};

const EXTRACT_PRODUCER: u32 = 1;

pub(crate) fn extract_scene_instances(
    changed: Extract<
        Query<
            (
                RenderEntity,
                &PrismGpuSceneEntity,
                &GlobalTransform,
                Option<&Aabb>,
                &Mesh3d,
            ),
            (
                With<PrismGpuSceneEntity>,
                Or<(
                    Added<PrismGpuSceneEntity>,
                    Changed<GlobalTransform>,
                    Changed<Aabb>,
                    Changed<Mesh3d>,
                )>,
            ),
        >,
    >,
    mut removed: Extract<RemovedComponents<PrismGpuSceneEntity>>,
    render_entities: Extract<Query<RenderEntity>>,
    mut extracted_instances: Query<&mut ExtractedSceneInstance>,
    mut commands: Commands,
) {
    for main_entity in removed.read() {
        if let Ok(render_entity) = render_entities.get(main_entity) {
            commands
                .entity(render_entity)
                .remove::<ExtractedSceneInstance>();
        }
    }

    for (render_entity, config, transform, bounds, mesh) in &changed {
        let update = ExtractedSceneInstance {
            handle: None,
            transform: *transform,
            bounds: bounds.copied(),
            mesh: mesh.clone(),
            geometry: config.geometry,
            material: config.material,
            flags: config.flags,
            render_layers: if config.render_layers == 0 {
                1
            } else {
                config.render_layers
            },
        };
        if let Ok(mut existing) = extracted_instances.get_mut(render_entity) {
            let handle = existing.handle;
            *existing = ExtractedSceneInstance { handle, ..update };
        } else {
            commands.entity(render_entity).insert(update);
        }
    }
}

pub(crate) fn apply_extracted_scene_changes(
    mut changed: Query<(Entity, &mut ExtractedSceneInstance), Changed<ExtractedSceneInstance>>,
    mut removed: RemovedComponents<ExtractedSceneInstance>,
    mut scene: ResMut<RenderGpuScene>,
    mut buffers: ResMut<GpuSceneBuffers>,
    completion: Res<GpuCompletionTracker>,
    mode: Res<crate::GpuSceneMode>,
    mut diagnostics: ResMut<GpuSceneDiagnostics>,
    mut frame_epoch: Local<u64>,
    mut sequence: Local<u64>,
) {
    *diagnostics = GpuSceneDiagnostics {
        active_instances: scene.snapshot().instance_count,
        scene_epoch: scene.snapshot().scene_epoch,
        ..GpuSceneDiagnostics::default()
    };
    if *mode == crate::GpuSceneMode::Disabled {
        return;
    }
    *frame_epoch += 1;
    *sequence += 1;
    let mut transaction =
        SceneTransactionBuilder::for_producer(*frame_epoch, *sequence, EXTRACT_PRODUCER);
    let mut new_bindings = Vec::new();
    let mut allocated_handles = Vec::new();
    let mut allocation_failures = 0_u32;

    for (entity, mut extracted) in &mut changed {
        let handle = match extracted.handle {
            Some(handle) => handle,
            None => match scene.allocate() {
                Ok(handle) => {
                    extracted.handle = Some(handle);
                    new_bindings.push((entity, handle));
                    allocated_handles.push(handle);
                    handle
                }
                Err(_) => {
                    allocation_failures += 1;
                    continue;
                }
            },
        };
        let geometry = extracted
            .geometry
            .unwrap_or_else(|| scene.geometry_for_mesh(extracted.mesh.id()));
        let record = instance_record(&extracted, geometry);
        if scene.mirror().get(handle).is_some() {
            transaction
                .push(SceneOperation::SetTransform {
                    handle,
                    current: record.current_transform,
                })
                .push(SceneOperation::SetBounds {
                    handle,
                    bounds: record.bounds,
                })
                .push(SceneOperation::SetGeometry {
                    handle,
                    geometry: record.geometry,
                })
                .push(SceneOperation::SetMaterial {
                    handle,
                    material: record.material,
                })
                .push(SceneOperation::SetFlags {
                    handle,
                    mask: u32::MAX,
                    value: record.flags,
                })
                .push(SceneOperation::SetRenderLayers {
                    handle,
                    render_layers: record.render_layers,
                });
        } else {
            transaction.push(SceneOperation::Create { handle, record });
        }
    }

    let removed_entities: Vec<_> = removed.read().collect();
    let removed_handles: Vec<_> = removed_entities
        .iter()
        .filter_map(|&entity| {
            scene
                .handle_for_entity(entity)
                .map(|handle| (entity, handle))
        })
        .collect();
    for &(_, handle) in &removed_handles {
        // Removal happens after the component value is gone. The retained map
        // in `RenderGpuScene` provides the stable handle for retirement.
        transaction.push(SceneOperation::Destroy { handle });
    }

    let transaction = transaction.finish();
    if !transaction.operations.is_empty() {
        let report = scene.apply_entity_transaction(&mut buffers, &transaction);
        diagnostics.active_instances = scene.snapshot().instance_count;
        diagnostics.created = report.created;
        diagnostics.destroyed = report.destroyed;
        diagnostics.updated_fields = report.updated;
        diagnostics.transaction_errors = report.errors.len() as u32;
        diagnostics.allocation_failures = allocation_failures;
        diagnostics.scene_epoch = report.scene_epoch;
        if report.errors.is_empty() {
            for (entity, handle) in new_bindings {
                scene.bind_entity(entity, handle);
            }
            for (entity, handle) in removed_handles {
                scene.remove_entity(entity);
                let _ = scene.retire(handle, &completion);
            }
        } else {
            for handle in allocated_handles {
                let _ = scene.retire(handle, &completion);
            }
        }
    }
}

fn instance_record(extracted: &ExtractedSceneInstance, geometry: GeometryHandle) -> InstanceRecord {
    let current_transform = scene_transform(extracted.transform);
    let bounds = extracted
        .bounds
        .map_or_else(SceneBounds::default, |bounds| {
            let half_extents = bounds.half_extents.to_array();
            SceneBounds {
                center: bounds.center.to_array(),
                radius: half_extents[0]
                    .hypot(half_extents[1])
                    .hypot(half_extents[2]),
                half_extents,
                _padding: 0.0,
            }
        });
    InstanceRecord {
        current_transform,
        previous_transform: current_transform,
        bounds,
        geometry,
        material: extracted.material,
        render_layers: extracted.render_layers,
        flags: extracted.flags,
    }
}

fn scene_transform(transform: GlobalTransform) -> SceneTransform {
    let rows = Affine3::from(transform.affine()).to_transpose();
    SceneTransform {
        rows: rows.map(|row| row.to_array()),
    }
}
