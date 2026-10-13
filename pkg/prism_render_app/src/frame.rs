//! Building the frame's render graph.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_render_driver::{
    BufferId, BufferState, BufferUsages, TextureFormat, TextureState, TextureUsages,
};
use prism_render_graph::{
    BufferDesc, BufferHandle, ImportedBuffer, ImportedTexture, PassFlags, RenderGraph, TextureDesc,
};

use crate::draw::DrawFunctions;
use crate::queue::RenderQueues;
use crate::sets::RenderPhase;
use crate::target::RenderTarget;

/// The texel format of the render sub-app's offscreen frame target.
///
/// `Rgba8Unorm` is the portable baseline every backend can render to and read
/// back, so a frame recorded here is verifiable on any GPU.
pub const RENDER_TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

/// The depth format of the render sub-app's frame.
///
/// `Depth32Float` is what reversed-Z wants: the convention is near 1 / far 0
/// tested with `GreaterEqual`, which needs the float buffer's precision near the
/// far plane (see [`DepthStencilState::reverse_z`]).
///
/// [`DepthStencilState::reverse_z`]: prism_render_driver::DepthStencilState::reverse_z
pub const RENDER_DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

/// Builds the frame graph: clear the target, then record the queued draws.
///
/// One raster pass per phase that has items, in [`RenderPhase::ORDER`], each
/// drawing into the frame target. A buffer an item binds must be *declared* by
/// the pass that reads it — that is what orders the pass against writers and
/// lets validation see the access — so every bound buffer is imported into the
/// graph once and declared as a vertex or index read. The recorder then binds
/// the same ids, so the declaration and the recorded command agree by
/// construction.
///
/// The target is **imported** rather than allocated transiently so the same
/// texture backs every frame and its contents survive the graph (which is what
/// lets a test read the frame back), and every pass is flagged
/// [`PassFlags::SIDE_EFFECT`] so the compiler never culls a frame that draws into
/// an externally-owned target.
///
/// An item carrying GPU-written indirect arguments declares that read as well;
/// its [`IndirectArgs`](crate::IndirectArgs) handle must come from *this* graph,
/// which is why a bin stream is filled once the frame graph exists (see the port
/// notes on [`RenderQueues`]).
pub(crate) fn target_graph(
    target: &RenderTarget,
    queues: &RenderQueues,
    functions: &DrawFunctions,
) -> RenderGraph {
    let mut graph = RenderGraph::new(target.viewport().extent());
    let color = graph.import_texture(
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

    // Import each bound buffer once. An imported buffer's descriptor is only the
    // usage this frame declares — the graph neither owns nor sizes it, and the
    // size is the owner's business (a mesh buffer, a scene buffer, a bin stream).
    let mut imported: BTreeMap<BufferId, BufferHandle> = BTreeMap::new();
    for item in RenderPhase::ORDER
        .into_iter()
        .flat_map(|phase| queues.phase(phase))
    {
        let mut bind = |graph: &mut RenderGraph, id: BufferId, usage: BufferUsages| {
            if let Some(handle) = imported.get(&id) {
                return *handle;
            }
            let handle = graph.import_buffer(
                "frame-bound-buffer",
                BufferDesc::new(0, usage),
                ImportedBuffer {
                    buffer: id,
                    entry_state: BufferState::initial(),
                },
            );
            imported.insert(id, handle);
            handle
        };
        for buffer in &item.vertices {
            let _ = bind(&mut graph, *buffer, BufferUsages::VERTEX);
        }
        if let Some((indices, _)) = item.indices {
            let _ = bind(&mut graph, indices, BufferUsages::INDEX);
        }
    }

    // A frame that rasterizes geometry needs depth, and the pipelines it
    // specializes must agree with the attachment: the queue stage passes
    // `RENDER_DEPTH_FORMAT` into the variant for exactly this reason.
    let depth = graph.import_texture(
        "prism-render-app-depth",
        TextureDesc::depth(target.depth_format()),
        ImportedTexture {
            texture: target.depth_texture(),
            default_view: target.depth_view(),
            entry_state: TextureState::initial(),
            exit_state: TextureState::depth_write(),
            mip_levels: 1,
            array_layers: 1,
        },
    );

    // The clear runs first and is never culled: the frame overwrites whatever the
    // previous frame left in the target.
    graph.add_raster_pass(
        "prism-render-app-clear",
        PassFlags::SIDE_EFFECT,
        move |builder| {
            let _ = builder.color_attachment(color);
            let _ = builder.depth_attachment(depth);
            |_ctx| {}
        },
    );

    for phase in RenderPhase::ORDER {
        if queues.phase(phase).is_empty() {
            continue;
        }
        // Resolve this phase's declarations before building the pass, so the
        // setup closure captures handles rather than the import table.
        let mut vertex_reads = Vec::new();
        let mut index_reads = Vec::new();
        let mut indirect_reads = Vec::new();
        for item in queues.phase(phase) {
            for buffer in &item.vertices {
                if let Some(handle) = imported.get(buffer) {
                    vertex_reads.push(*handle);
                }
            }
            if let Some((indices, _)) = item.indices
                && let Some(handle) = imported.get(&indices)
            {
                index_reads.push(*handle);
            }
            if let Some(args) = item.indirect {
                indirect_reads.push(args.args);
            }
        }
        graph.add_raster_pass(
            format!("prism-render-app-{phase:?}"),
            PassFlags::SIDE_EFFECT,
            move |builder| {
                let _ = builder.color_attachment(color);
                let _ = builder.depth_attachment(depth);
                for handle in &vertex_reads {
                    let _ = builder.read_vertex(*handle);
                }
                for handle in &index_reads {
                    let _ = builder.read_index(*handle);
                }
                for handle in &indirect_reads {
                    let _ = builder.read_indirect(*handle);
                }
                // A pass closure outlives this borrow, so it owns what it needs:
                // a cheap clone of the queue and of the draw-function registry
                // (shared functions, cloned handles). Recording then goes through
                // the one `RenderQueues::record` path, exactly as a standalone
                // pass does.
                let queues = queues.clone();
                let functions = functions.clone();
                move |ctx| queues.record(phase, ctx, &functions)
            },
        );
    }
    graph
}
