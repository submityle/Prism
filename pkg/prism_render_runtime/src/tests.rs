//! GPU-gated tests for the RHI render spine.
//!
//! Each test first attempts to bring up a real device. When no adapter is
//! present (the usual GPU-less CI sandbox) the test returns early and is a
//! no-op, so the suite always passes without hardware. On a machine with a GPU
//! it exercises the full declare -> compile -> realize -> submit -> readback
//! path through the frozen RHI, which is exactly the ground the Bevy render
//! stack used to provide.

use prism_render_driver as rhi;
use rhi::{RenderDevice, TextureState, TextureUsages};
use prism_render_graph::{ImportedTexture, PassFlags, TextureDesc};

use crate::RenderRuntime;

/// The offscreen target edge length used by the readback test.
const SIZE: u32 = 64;

#[test]
fn rhi_graph_clears_imported_target() {
    let Some(rt) = RenderRuntime::headless(rhi::Extent3d::new_2d(SIZE, SIZE)).ok() else {
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
    assert!(rt.wait(), "the device stays healthy through the frame");

    let pixels = rt.queue().read_texture_rgba8(texture, SIZE, SIZE);
    assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize);
    assert!(
        pixels.chunks_exact(4).all(|p| p == [0, 0, 0, 255]),
        "a freshly written version-0 target is cleared to opaque black"
    );

    device.destroy_texture_view(view);
    device.destroy_texture(texture);
}
