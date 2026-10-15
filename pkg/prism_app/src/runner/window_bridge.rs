//! Main-thread seam between the OS runner and an optional renderer.

use prism_ecs::prelude::Resource;
use prism_window::WindowId;
use prism_window_winit::WinitBackend;

use crate::App;

/// Called before extraction with the live managed window ids. On suspension,
/// `suspended` is true and the renderer must release platform surfaces before
/// returning. The runner synchronizes pipelined sub-apps before this callback.
/// Installing this resource is optional; headless Apps need no renderer.
#[derive(Clone, Copy, Resource)]
pub struct WindowRenderBridge(pub fn(&mut App, &WinitBackend, &[WindowId], bool));

pub(super) fn publish(
    app: &mut App,
    backend: &WinitBackend,
    windows: &[WindowId],
    suspended: bool,
) {
    if let Some(bridge) = app.world().get_resource::<WindowRenderBridge>().copied() {
        app.sync_sub_apps();
        (bridge.0)(app, backend, windows, suspended);
    }
}
