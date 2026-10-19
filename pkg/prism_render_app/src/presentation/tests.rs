use super::*;

use prism_app::presentation::{PresentationBridge, PresentationClockDomain, PresentationHostClock};
use prism_render_driver::{Extent3d, Instance as _, TextureFormat};
use prism_render_driver_null::NullInstance;
use prism_render_runtime::RenderRuntime;
use std::sync::Arc;

struct FakeWindow;

impl raw_window_handle::HasWindowHandle for FakeWindow {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::Unavailable)
    }
}

impl raw_window_handle::HasDisplayHandle for FakeWindow {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::Unavailable)
    }
}

#[test]
fn null_window_request_lifecycle_is_explicit() {
    let id = WindowId(7);
    let mut requests = WindowRenderRequests::<NullInstance>::default();
    assert!(!requests.contains(id));

    requests.attach_factory(id, RenderViewport::new(64, 32), |_runtime| {
        Err(SurfaceError::Unsupported)
    });
    assert!(requests.contains(id));
    assert!(requests.update(id, RenderViewport::new(80, 40), false));
    assert_eq!(requests.ids().collect::<Vec<_>>(), vec![id]);
    assert!(requests.remove(id));
    assert!(!requests.contains(id));

    requests.use_offscreen();
    assert!(!requests.enabled);
}

#[test]
fn null_surface_presents_and_releases_window_resources() {
    let extent = Extent3d::new_2d(16, 16);
    let runtime =
        RenderRuntime::<NullInstance>::headless(NullInstance::new(), extent).expect("null runtime");
    let id = WindowId(1);
    let mut requests = WindowRenderRequests::<NullInstance>::default();
    requests.attach_factory(id, RenderViewport::from_extent(extent), |runtime| {
        runtime
            .instance()
            .create_surface(Arc::new(FakeWindow), runtime.device(), runtime.queue())
    });

    let bridge = PresentationBridge::new(PresentationHostClock::new(PresentationClockDomain(1)));
    let mut targets = WindowRenderTargets::with_bridge(bridge);
    let queues = RenderQueues::new();
    let functions = DrawFunctions::new();
    let mut stats = RenderFrameStats::default();
    assert!(targets.drive(
        &runtime,
        &requests,
        &WindowRenderQueues::default(),
        &queues,
        &functions,
        None,
        &mut stats,
    ));
    assert_eq!(targets.len(), 1);
    assert!(matches!(
        targets.status(id),
        Some(WindowFrameStatus::Presented { .. })
    ));
    assert!(runtime.wait_idle());
    assert!(targets.release(&runtime));
    assert!(targets.is_empty());
}

#[test]
fn hardware_present_pipeline_constructs_when_adapter_is_available() {
    let extent = Extent3d::new_2d(8, 8);
    let Ok(runtime) = RenderRuntime::<prism_render_driver_wgpu::WgpuInstance>::wgpu(extent) else {
        return;
    };
    let target = RenderTarget::new(runtime.device(), extent);
    let lifetime = WindowLifetime::new();
    let pipeline = PresentPipeline::new(
        runtime.device_handle(),
        &target,
        TextureFormat::Rgba8Unorm,
        Arc::clone(&lifetime.in_flight),
        Arc::clone(&lifetime.destroy_allowed),
    );
    drop(pipeline);
    target.destroy(runtime.device());
}

#[cfg(feature = "winit")]
#[test]
#[ignore = "requires a real OS window and display compositor"]
fn real_window_presentation_acceptance_entry() {
    use crate::PrismRenderPlugin;
    use prism_app::runner::WinitRunner;
    use prism_app::{App, Plugin};
    use prism_render_driver_wgpu::WgpuInstance;

    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::new(WgpuInstance::new()));
    app.add_plugins(WinitPresentationPlugin::<WgpuInstance>::default());
    let _exit = WinitRunner::new(app).run();
}
