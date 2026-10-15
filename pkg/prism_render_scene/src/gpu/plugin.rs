//! [`PrismGpuPlugin`]: creates the Prism RHI GPU context for the render world.

use bevy_app::{App, Plugin};
use bevy_render::{RenderApp, RenderStartup};

use crate::gpu::context::PrismGpu;

/// Installs the Prism RHI GPU context into Bevy's render sub-application.
///
/// The context is best-effort during the `bevy_render` → Prism RHI port: on a
/// host with no adapter the resource is left absent and consumers that require
/// it no-op, so GPU-less CI keeps building and running the CPU paths. A real
/// renderer treats a missing context as a hard error.
///
/// Until the port finishes the render world still owns Bevy's device as well;
/// this plugin only adds the Prism one so subsystems can migrate one at a time.
#[derive(Default)]
pub struct PrismGpuPlugin;

impl Plugin for PrismGpuPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(RenderStartup, super::fabric::init_scene_fabric);
        if let Ok(context) = PrismGpu::new() {
            render_app.insert_resource(context);
        }
    }
}
