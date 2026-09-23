use bevy_app::App;
use bevy_ecs::{entity::Entity, schedule::Schedule, world::FromWorld};
use bevy_mesh::{Mesh, Mesh3d};
use prism_render_architecture::{
    abi::GenerationalHandle,
    gpu_scene::{
        InstanceRecord, SceneOperation, SceneTransaction, SceneTransform, UploadBudget,
        UploadStrategy,
    },
};

use crate::{
    buffers::GpuSceneBuffers,
    completion::GpuCompletionTracker,
    diagnostics::{GpuSceneDiagnostics, GpuSceneUploadSettings},
    extract::{lifecycle::ExtractionClock, ExtractedSceneInstance, GpuSceneInstanceAddress},
    GpuSceneMode, RenderGpuScene,
};

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
fn retired_geometry_gets_a_fresh_runtime_identity() {
    let mut scene = RenderGpuScene::new(8);
    let mesh = bevy_asset::Assets::<Mesh>::default().add(Mesh::new(
        bevy_mesh::PrimitiveTopology::TriangleList,
        bevy_asset::RenderAssetUsages::default(),
    ));
    let first = scene.geometry_for_mesh(mesh.id());
    assert_eq!(scene.retire_geometry(mesh.id()), Some(first));
    assert_eq!(scene.geometry_handle(mesh.id()), None);
    let second = scene.geometry_for_mesh(mesh.id());
    assert_ne!(first, second);
}

#[test]
fn material_registry_requires_monotonic_generations() {
    let mut scene = RenderGpuScene::new(8);
    let first = GenerationalHandle {
        index: 4,
        generation: 1,
    };
    assert!(scene.register_material(first));
    assert!(scene.material_is_current(first));
    assert!(!scene.register_material(first));
    assert!(scene.retire_material(first));
    assert!(!scene.material_is_current(first));
    assert!(!scene.register_material(first));
    let second = GenerationalHandle {
        index: 4,
        generation: 2,
    };
    assert!(scene.register_material(second));
    assert!(!scene.register_material(first));
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

fn gpu_scene_test_world() -> (bevy_ecs::world::World, Schedule) {
    let mut world = bevy_ecs::world::World::new();
    let buffers = GpuSceneBuffers::from_world(&mut world);
    world.insert_resource(buffers);
    world.insert_resource(RenderGpuScene::new(32));
    world.insert_resource(GpuCompletionTracker::default());
    world.insert_resource(GpuSceneMode::Enabled);
    world.insert_resource(GpuSceneDiagnostics::default());
    world.insert_resource(GpuSceneUploadSettings::default());
    world.insert_resource(ExtractionClock::default());
    world.insert_resource(crate::material::runtime::RenderMaterialRegistry::default());
    let mut schedule = Schedule::default();
    schedule.add_systems(crate::extract::apply_extracted_scene_changes);
    (world, schedule)
}

fn extracted_instance() -> ExtractedSceneInstance {
    ExtractedSceneInstance {
        handle: None,
        transform: bevy_transform::components::GlobalTransform::IDENTITY,
        bounds: None,
        mesh: Mesh3d::default(),
        geometry: None,
        material: Default::default(),
        material_asset: None,
        flags: 7,
        render_layers: 1,
    }
}

#[test]
fn ecs_spawn_update_remove_and_mode_switch_are_transactional() {
    let (mut world, mut schedule) = gpu_scene_test_world();
    let entity = world.spawn(extracted_instance()).id();
    schedule.run(&mut world);
    let address = *world
        .entity(entity)
        .get::<GpuSceneInstanceAddress>()
        .unwrap();
    assert_eq!(
        world.resource::<RenderGpuScene>().snapshot().instance_count,
        1
    );

    world.clear_trackers();
    world
        .entity_mut(entity)
        .get_mut::<ExtractedSceneInstance>()
        .unwrap()
        .transform = bevy_transform::components::GlobalTransform::from_xyz(3.0, 0.0, 0.0);
    schedule.run(&mut world);
    let scene = world.resource::<RenderGpuScene>();
    let handle = scene.handle_for_entity(entity).unwrap();
    let record = scene.mirror().get(handle).unwrap();
    assert_eq!(record.current_transform.rows[0][3], 3.0);
    assert_eq!(record.previous_transform.rows[0][3], 0.0);
    assert_eq!(address.index, handle.index);

    world.clear_trackers();
    *world.resource_mut::<GpuSceneMode>() = GpuSceneMode::Disabled;
    world.entity_mut(entity).remove::<ExtractedSceneInstance>();
    schedule.run(&mut world);
    assert!(world
        .entity(entity)
        .get::<GpuSceneInstanceAddress>()
        .is_none());
    assert_eq!(
        world.resource::<RenderGpuScene>().snapshot().instance_count,
        0
    );

    world.clear_trackers();
    *world.resource_mut::<GpuSceneMode>() = GpuSceneMode::Enabled;
    world.entity_mut(entity).insert(extracted_instance());
    schedule.run(&mut world);
    let replacement = *world
        .entity(entity)
        .get::<GpuSceneInstanceAddress>()
        .unwrap();
    assert_ne!(replacement.index, 0);
    assert_eq!(
        world.resource::<RenderGpuScene>().snapshot().instance_count,
        1
    );
}
