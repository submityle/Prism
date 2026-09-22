use bevy_app::App;
use bevy_ecs::{entity::Entity, world::FromWorld};
use bevy_mesh::Mesh;
use prism_render_architecture::gpu_scene::{
    InstanceRecord, SceneOperation, SceneTransaction, SceneTransform, UploadBudget, UploadStrategy,
};

use crate::{buffers::GpuSceneBuffers, RenderGpuScene};

#[test]
fn plugin_contract_types_can_be_initialized_without_touching_bevy_sources() {
    let _app = App::new();
    let mut scene = RenderGpuScene::new(8);
    let handle = scene.allocate().unwrap();
    scene.bind_entity(Entity::from_raw_u32(1).unwrap(), handle);
    assert_eq!(
        scene.handle_for_entity(Entity::from_raw_u32(1).unwrap()),
        Some(handle)
    );
}

#[test]
fn entity_transaction_publishes_snapshot_and_previous_transform() {
    let mut world = bevy_ecs::world::World::new();
    let mut buffers = GpuSceneBuffers::from_world(&mut world);
    let mut scene = RenderGpuScene::new(8);
    let handle = scene.allocate().unwrap();
    let mut first = InstanceRecord::default();
    first.current_transform.rows[0][3] = 1.0;
    first.previous_transform = first.current_transform;
    let report = scene.apply_transaction(
        &mut buffers,
        &SceneTransaction {
            frame_epoch: 1,
            sequence: 1,
            producer: 1,
            operations: vec![SceneOperation::Create {
                handle,
                record: first,
            }],
        },
    );
    assert!(report.errors.is_empty());
    assert_eq!(scene.snapshot().instance_count, 1);

    let mut current = SceneTransform::IDENTITY;
    current.rows[0][3] = 2.0;
    scene.apply_transaction(
        &mut buffers,
        &SceneTransaction {
            frame_epoch: 2,
            sequence: 2,
            producer: 1,
            operations: vec![SceneOperation::SetTransform { handle, current }],
        },
    );
    let record = scene.mirror().get(handle).unwrap();
    assert_eq!(record.previous_transform.rows[0][3], 1.0);
    assert_eq!(record.current_transform.rows[0][3], 2.0);
}

#[test]
fn geometry_handles_are_stable_per_asset() {
    let mut scene = RenderGpuScene::new(8);
    let mesh = bevy_asset::Assets::<Mesh>::default().add(Mesh::new(
        bevy_mesh::PrimitiveTopology::TriangleList,
        bevy_asset::RenderAssetUsages::default(),
    ));
    assert_eq!(
        scene.geometry_for_mesh(mesh.id()),
        scene.geometry_for_mesh(mesh.id())
    );
}

#[test]
fn transaction_publishes_upload_plan_and_budget_pressure() {
    let mut world = bevy_ecs::world::World::new();
    let mut buffers = GpuSceneBuffers::from_world(&mut world);
    buffers.set_upload_budget(UploadBudget {
        max_bytes_per_frame: 1,
        ..UploadBudget::default()
    });
    let mut scene = RenderGpuScene::new(8);
    let handle = scene.allocate().unwrap();
    let report = scene.apply_transaction(
        &mut buffers,
        &SceneTransaction {
            frame_epoch: 1,
            sequence: 1,
            producer: 1,
            operations: vec![SceneOperation::Create {
                handle,
                record: InstanceRecord::default(),
            }],
        },
    );
    assert!(report.errors.is_empty());
    let plan = buffers.last_upload_plan();
    assert!(plan.budget_exceeded);
    assert_ne!(plan.instances.strategy, UploadStrategy::None);
    // Slot zero is intentionally reserved, so the first live slot makes the
    // four full-table uploads cover two rows.
    assert_eq!(plan.estimated_bytes, 320);
}
