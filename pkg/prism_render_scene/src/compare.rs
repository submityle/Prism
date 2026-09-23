use bevy_ecs::prelude::*;
use prism_render_architecture::gpu_scene::SceneHandle;

use crate::{ExtractedSceneInstance, GpuSceneInstanceAddress, GpuSceneMode, RenderGpuScene};

#[derive(Resource, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuSceneParityDiagnostics {
    pub checked: u32,
    pub missing_address: u32,
    pub stale_address: u32,
    pub transform_mismatches: u32,
    pub bounds_mismatches: u32,
    pub geometry_mismatches: u32,
    pub material_mismatches: u32,
}

pub(crate) fn compare_scene_mirror(
    mode: Res<GpuSceneMode>,
    scene: Res<RenderGpuScene>,
    instances: Query<(&ExtractedSceneInstance, Option<&GpuSceneInstanceAddress>)>,
    mut diagnostics: ResMut<GpuSceneParityDiagnostics>,
) {
    *diagnostics = GpuSceneParityDiagnostics::default();
    if *mode != GpuSceneMode::Compare {
        return;
    }
    for (extracted, address) in &instances {
        diagnostics.checked += 1;
        let Some(address) = address else {
            diagnostics.missing_address += 1;
            continue;
        };
        let handle = SceneHandle {
            index: address.index,
            generation: address.generation,
        };
        let Some(record) = scene.mirror().get(handle) else {
            diagnostics.stale_address += 1;
            continue;
        };
        if record.current_transform != super::extract::scene_transform(extracted.transform) {
            diagnostics.transform_mismatches += 1;
        }
        let expected_geometry = extracted
            .geometry
            .or_else(|| scene.geometry_handle(extracted.mesh.id()));
        if expected_geometry != Some(record.geometry) {
            diagnostics.geometry_mismatches += 1;
        }
        if extracted.material != record.material {
            diagnostics.material_mismatches += 1;
        }
        if let Some(bounds) = extracted.bounds {
            let half = bounds.half_extents.to_array();
            if record.bounds.center != bounds.center.to_array()
                || record.bounds.half_extents != half
            {
                diagnostics.bounds_mismatches += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy_ecs::{schedule::Schedule, world::World};

    use super::*;

    #[test]
    fn compare_reports_missing_and_stale_addresses() {
        let mut world = World::new();
        world.insert_resource(GpuSceneMode::Compare);
        world.insert_resource(RenderGpuScene::new(8));
        world.insert_resource(GpuSceneParityDiagnostics::default());
        world.spawn(ExtractedSceneInstance {
            main_entity: bevy_render::sync_world::MainEntity::from(Entity::PLACEHOLDER),
            handle: None,
            transform: bevy_transform::components::GlobalTransform::IDENTITY,
            bounds: None,
            mesh: bevy_mesh::Mesh3d::default(),
            geometry: None,
            material: Default::default(),
            material_asset: None,
            flags: 0,
            render_layers: 1,
        });
        let mut schedule = Schedule::default();
        schedule.add_systems(compare_scene_mirror);
        schedule.run(&mut world);
        let diagnostics = world.resource::<GpuSceneParityDiagnostics>();
        assert_eq!(diagnostics.checked, 1);
        assert_eq!(diagnostics.missing_address, 1);
    }
}
