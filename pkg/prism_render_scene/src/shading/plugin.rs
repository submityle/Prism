use bevy_app::{App, Plugin};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{Render, RenderApp, RenderSystems};

use super::runtime::{prepare_shading_work, PrismShadingDiagnostics, PrismShadingSettings};

pub struct PrismShadingPlugin;

impl Plugin for PrismShadingPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<PrismShadingSettings>()
            .init_resource::<PrismShadingDiagnostics>()
            .add_systems(
                Render,
                prepare_shading_work
                    .after(super::super::visibility::systems::build_unified_visibility)
                    .in_set(RenderSystems::PrepareResources),
            );
    }
}
