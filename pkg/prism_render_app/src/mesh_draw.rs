//! The canonical mesh draw function.
//!
//! A phase item is self-contained: it carries the pipeline to bind and the
//! exact draw call (vertex buffer, optional index buffer, element range, base
//! vertex). So the draw function needs no external state — it records the item
//! straight into the frame graph's pass recorder, which is what makes it usable
//! as a `DrawFunctions<ExecuteContext>` entry.
//!
//! The instance count is `1`: per-instance data rides the GPU scene (instance
//! index) and, for the GPU-driven path, a bin's indirect arguments; a direct
//! per-item mesh draw binds its geometry once and draws one instance.

use crate::draw::MeshRecorder;
use crate::queue::PhaseItem;

/// Records one item's mesh draw: bind the pipeline, bind the geometry, draw.
pub fn mesh_draw(item: &PhaseItem, ctx: &mut dyn MeshRecorder) {
    ctx.set_pipeline(item.pipeline);
    ctx.set_vertex_buffer(0, item.vertices, 0);
    match item.indices {
        Some((indices, format)) => {
            ctx.set_index_buffer(indices, format, 0);
            ctx.draw_indexed(
                item.first..item.first + item.count,
                item.base_vertex,
                0..1,
            );
        }
        None => {
            ctx.draw(item.first..item.first + item.count, 0..1);
        }
    }
}

/// Records a GPU-driven bin draw: bind the pipeline and geometry, then issue one
/// multi-draw over the item's indirect argument range.
///
/// This is the bin path of the GPU-driven base: the arguments (element count,
/// first index, base vertex, and — for the GPU-driven cull — the instance count
/// the CPU never learns) come from a buffer the frame graph declared, so the
/// read is synchronized like any other declared access. An item with no indirect
/// range records nothing here.
pub fn mesh_bin_draw(item: &PhaseItem, ctx: &mut dyn MeshRecorder) {
    let Some(args) = item.indirect else {
        return;
    };
    ctx.set_pipeline(item.pipeline);
    ctx.set_vertex_buffer(0, item.vertices, 0);
    if let Some((indices, format)) = item.indices {
        ctx.set_index_buffer(indices, format, 0);
    }
    ctx.multi_draw_indirect(args.args, args.offset, args.count, args.stride);
}
