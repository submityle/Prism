//! Physical GPU acceptance for caller-owned camera output textures.

use prism_render_app::{TextureTargetRegistration, TextureTargetRegistry, RENDER_TARGET_FORMAT};
use prism_render_driver::{
    Color, Extent3d, RenderDevice, TextureDescriptor, TextureState, TextureUsages,
    TextureViewDescriptor,
};
use prism_render_driver_wgpu::{require_hardware_gpu, WgpuInstance};
use prism_render_graph::{ImportedTexture, PassFlags, RenderGraph, TextureDesc};
use prism_render_runtime::RenderRuntime;
use std::sync::Arc;

#[test]
fn external_camera_target_is_a_real_graph_attachment_and_readback_source() {
    let runtime = match RenderRuntime::headless(WgpuInstance::new(), Extent3d::new_2d(4, 2)) {
        Ok(runtime) => runtime,
        Err(error) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but hardware runtime creation failed: {error:?}"
            );
            eprintln!("SKIPPED external target GPU readback: {error:?}");
            return;
        }
    };
    let extent = Extent3d::new_2d(4, 2);
    let device = runtime.device_handle();
    let descriptor = TextureDescriptor::new_2d(
        extent,
        RENDER_TARGET_FORMAT,
        TextureUsages::RENDER_ATTACHMENT
            .union(TextureUsages::STORAGE_BINDING)
            .union(TextureUsages::COPY_SRC)
            .union(TextureUsages::COPY_DST)
            .union(TextureUsages::TEXTURE_BINDING),
    );
    let texture = device.create_texture(&descriptor);
    let view_descriptor = TextureViewDescriptor::default();
    let view = device.create_texture_view(texture, &view_descriptor);
    let owner = Arc::new(());
    let mut registry = TextureTargetRegistry::default();
    let handle = registry
        .register_external(
            &runtime,
            7,
            TextureTargetRegistration {
                device: Arc::clone(&device),
                device_generation: runtime.device_generation(),
                texture,
                view,
                descriptor,
                view_descriptor,
                entry_state: TextureState::initial(),
                exit_state: TextureState::copy_src(),
                owner: owner.clone(),
            },
        )
        .expect("external texture registration is valid");
    assert_eq!(
        registry
            .get(handle)
            .expect("fresh handle resolves")
            .texture(),
        texture
    );

    let mut graph = RenderGraph::new(extent);
    graph.retain_external_owner(owner);
    let color = graph.import_texture(
        "external-camera-target",
        TextureDesc::color(RENDER_TARGET_FORMAT).with_size(
            prism_render_graph::SizeClass::absolute_2d(extent.width, extent.height),
        ),
        ImportedTexture {
            texture,
            default_view: view,
            entry_state: TextureState::initial(),
            exit_state: TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    graph.add_raster_pass("external-clear", PassFlags::SIDE_EFFECT, |builder| {
        builder.clear_color_attachment(color, Color::new(2.0, 1.0, 0.5, 1.0));
        |_context| {}
    });
    let plan = graph.compile().expect("external output graph compiles");
    assert!(plan.attachments.iter().flatten().any(|attachments| {
        attachments
            .colors
            .iter()
            .any(|attachment| attachment.resource.get() == 0)
    }));
    runtime
        .execute_tracked(graph)
        .expect("external attachment graph submits");
    assert!(runtime.wait_idle());
    let pixels = runtime
        .queue()
        .read_texture_rgba16f(texture, extent.width, extent.height);
    assert_eq!(pixels.len(), (extent.width * extent.height) as usize);
    assert!(pixels.iter().all(|pixel| {
        (pixel[0] - 2.0).abs() < 0.002
            && (pixel[1] - 1.0).abs() < 0.002
            && (pixel[2] - 0.5).abs() < 0.002
            && (pixel[3] - 1.0).abs() < 0.002
    }));
    eprintln!(
        "GPU external-target evidence: route={:#x}, texture={texture:?}, view={view:?}, pixels={pixels:?}",
        handle.route_id()
    );
    registry
        .unregister(&runtime, handle)
        .expect("queue is idle before unregister");
    registry.collect_completed();
    device.destroy_texture_view(view);
    device.destroy_texture(texture);
}
