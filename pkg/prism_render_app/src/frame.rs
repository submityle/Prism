//! Building the frame's render graph.

use prism_render_driver::{TextureFormat, TextureState, TextureUsages};
use prism_render_graph::{ImportedTexture, PassFlags, RenderGraph, TextureDesc};

use crate::target::RenderTarget;

/// The texel format of the render sub-app's offscreen frame target.
///
/// `Rgba8Unorm` is the portable baseline every backend can render to and read
/// back, so a frame recorded here is verifiable on any GPU.
pub const RENDER_TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

/// Builds the frame graph for one frame: import the offscreen target and clear
/// it.
///
/// The graph is deliberately minimal — this layer exists to prove the ECS ↔ RHI
/// wiring, not to shade geometry. The target is **imported** rather than
/// allocated transiently so the same texture backs every frame and its contents
/// survive the graph (which is what lets a test read the frame back). The pass
/// is flagged [`PassFlags::SIDE_EFFECT`] so the compiler never culls it: it
/// writes an externally-owned target whose result no in-graph consumer reads.
///
/// The graph is sized to the target's viewport, so a resize in the main world
/// (via [`RenderCamera`](crate::RenderCamera)) is reflected in the frame the
/// graph describes.
pub(crate) fn clear_target_graph(target: &RenderTarget) -> RenderGraph {
    let mut graph = RenderGraph::new(target.viewport().extent());
    let handle = graph.import_texture(
        "prism-render-app-target",
        TextureDesc::color(target.format()).with_usage(TextureUsages::COPY_SRC),
        ImportedTexture {
            texture: target.texture(),
            default_view: target.view(),
            entry_state: TextureState::initial(),
            exit_state: TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    graph.add_raster_pass(
        "prism-render-app-clear",
        PassFlags::SIDE_EFFECT,
        move |builder| {
            let _ = builder.color_attachment(handle);
            |_ctx| {}
        },
    );
    graph
}
