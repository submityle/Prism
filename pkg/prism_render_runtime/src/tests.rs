//! GPU-gated tests for the RHI render spine.
//!
//! Each test first attempts to bring up a real device. When no adapter is
//! present (the usual GPU-less CI sandbox) the test returns early and is a
//! no-op, so the suite always passes without hardware. On a machine with a GPU
//! it exercises the full facade -> graph -> readback path through the frozen
//! RHI, which is exactly the ground the Bevy render stack used to provide.

use prism_render_driver as rhi;
use rhi::{
    Adapter, Instance, RenderDevice, TextureState, TextureUsages,
};
use prism_render_driver_wgpu::WgpuInstance;
use prism_render_graph::{ImportedTexture, PassFlags, TextureDesc};

use crate::RenderRuntime;

/// The offscreen target edge length used by the readback test.
const SIZE: u32 = 64;

#[test]
fn instance_enumerates_adapters_through_the_rhi() {
    let instance = WgpuInstance::new();
    let adapters = <WgpuInstance as Instance>::enumerate_adapters(&instance);
    // A GPU-less sandbox legitimately reports nothing; skip in that case.
    for adapter in &adapters {
        let info = adapter.info();
        assert!(!info.name.is_empty(), "adapter reports a name");
        let caps = adapter.capabilities();
        assert_eq!(caps.backend, info.backend, "caps and info agree on backend");
    }
}

#[test]
fn rhi_graph_clears_imported_target() {
    let Some(rt) = RenderRuntime::wgpu(rhi::Extent3d::new_2d(SIZE, SIZE)).ok() else {
        return;
    };
    let device = rt.device();

    // An offscreen target the graph imports (so it survives culling and its
    // result outlives the frame) and clears.
    let texture = device.create_texture(&rhi::TextureDescriptor::new_2d(
        rhi::Extent3d::new_2d(SIZE, SIZE),
        rhi::TextureFormat::Rgba8Unorm,
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    ));
    let view = device.create_texture_view(texture, &rhi::TextureViewDescriptor::default());

    let mut graph = rt.new_graph();
    let target = graph.import_texture(
        "offscreen",
        TextureDesc::color(rhi::TextureFormat::Rgba8Unorm).with_usage(TextureUsages::COPY_SRC),
        ImportedTexture {
            texture,
            default_view: view,
            entry_state: TextureState::initial(),
            exit_state: TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    graph.add_raster_pass("clear", PassFlags::SIDE_EFFECT, |b| {
        let _ = b.color_attachment(target);
        |_ctx| {}
    });

    rt.execute(graph).expect("graph executes on the RHI");
    assert!(rt.device().poll_wait(), "the device stays healthy");

    let pixels = rt.queue().read_texture_rgba8(texture, SIZE, SIZE);
    assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize);
    assert!(
        pixels.chunks_exact(4).all(|p| p == [0, 0, 0, 255]),
        "a freshly written version-0 target is cleared to opaque black"
    );

    device.destroy_texture_view(view);
    device.destroy_texture(texture);
}
