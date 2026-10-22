//! The canonical mesh draw function.
//!
//! A phase item is self-contained: it carries the pipeline to bind and the
//! exact draw call (vertex buffer, optional index buffer, element range, base
//! vertex). A bind-group-aware registration wraps this function with the
//! explicit binding contract before this function records geometry, so the
//! mesh path remains independent of material-resource lookup while still
//! producing real forward-lit commands.
//!
//! The instance count is `1`: per-instance data rides the GPU scene (instance
//! index) and, for the GPU-driven path, a bin's indirect arguments; a direct
//! per-item mesh draw binds its geometry once and draws one instance.

use crate::draw::MeshRecorder;
use crate::queue::PhaseItem;
use core::num::NonZeroU64;
use prism_render_driver::DRAW_INDEXED_INDIRECT_ARGS_SIZE;

fn indirect_range_is_valid(offset: u64, count: u32, stride: u32) -> bool {
    if count == 0 || offset % 4 != 0 || stride < DRAW_INDEXED_INDIRECT_ARGS_SIZE || stride % 4 != 0
    {
        return false;
    }
    offset
        .checked_add(u64::from(count - 1) * u64::from(stride))
        .and_then(|last| last.checked_add(u64::from(DRAW_INDEXED_INDIRECT_ARGS_SIZE)))
        .is_some()
}

fn ranges_are_valid(item: &PhaseItem, indirect: bool) -> bool {
    if item.vertices.len() != item.vertex_sizes.len()
        || item.vertex_sizes.iter().any(|&size| size == 0)
    {
        return false;
    }
    match (item.indices, item.index_size) {
        (Some((_, format)), Some(size)) if size > 0 && size % format.size() == 0 => {
            if indirect {
                true
            } else {
                item.first
                    .checked_add(item.count)
                    .and_then(|end| u64::from(end).checked_mul(format.size()))
                    .is_some_and(|required| required <= size)
            }
        }
        (None, None) => !indirect && item.first.checked_add(item.count).is_some(),
        _ => false,
    }
}

/// Records one item's mesh draw: bind the pipeline, bind the geometry, draw.
pub fn mesh_draw(item: &PhaseItem, ctx: &mut dyn MeshRecorder) {
    if !ranges_are_valid(item, false) {
        return;
    }
    ctx.set_pipeline(item.pipeline);
    for (slot, (buffer, size)) in item
        .vertices
        .iter()
        .copied()
        .zip(&item.vertex_sizes)
        .enumerate()
    {
        ctx.set_vertex_buffer(slot as u32, buffer, 0, NonZeroU64::new(*size));
    }
    match item.indices {
        Some((indices, format)) => {
            ctx.set_index_buffer(
                indices,
                format,
                0,
                item.index_size.and_then(NonZeroU64::new),
            );
            ctx.draw_indexed(item.first..item.first + item.count, item.base_vertex, 0..1);
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
    if item.indices.is_none()
        || !ranges_are_valid(item, true)
        || !indirect_range_is_valid(args.offset, args.count, args.stride)
    {
        return;
    }
    ctx.set_pipeline(item.pipeline);
    for (slot, (buffer, size)) in item
        .vertices
        .iter()
        .copied()
        .zip(&item.vertex_sizes)
        .enumerate()
    {
        ctx.set_vertex_buffer(slot as u32, buffer, 0, NonZeroU64::new(*size));
    }
    if let Some((indices, format)) = item.indices {
        ctx.set_index_buffer(
            indices,
            format,
            0,
            item.index_size.and_then(NonZeroU64::new),
        );
    }
    ctx.multi_draw_indirect(args.args, args.offset, args.count, args.stride);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::MeshRecorder;
    use alloc::vec::Vec;
    use core::ops::Range;
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_driver::{BindGroupId, BufferId, IndexFormat, RenderPipelineId};
    use prism_render_graph::BufferHandle;

    fn item() -> PhaseItem {
        PhaseItem {
            pipeline: RenderPipelineId::from_parts(1, 1),
            draw: crate::DrawFunctionId(0),
            geometry: GenerationalHandle::new(1, 1),
            vertices: vec![BufferId::from_parts(2, 1)],
            vertex_sizes: vec![48],
            indices: Some((BufferId::from_parts(3, 1), IndexFormat::Uint16)),
            index_size: Some(12),
            first: 0,
            count: 6,
            base_vertex: 0,
            indirect: None,
            lod_level: 0,
            material: GenerationalHandle::new(4, 1),
            scene: GenerationalHandle::new(5, 1),
            order: 0,
        }
    }

    fn indirect_args() -> crate::queue::IndirectArgs {
        let mut graph = prism_render_graph::RenderGraph::new(prism_render_driver::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        });
        let mut handle = None;
        graph.add_compute_pass(
            "mesh-draw-indirect-test",
            prism_render_graph::PassFlags::EMPTY,
            |pass| {
                handle = Some(pass.create_buffer(
                    "mesh-draw-indirect-test",
                    prism_render_graph::BufferDesc::new(
                        96,
                        prism_render_driver::BufferUsages::INDIRECT,
                    ),
                ));
                |_: &mut prism_render_graph::ExecuteContext<'_>| {}
            },
        );
        crate::queue::IndirectArgs {
            args: handle.expect("the test pass creates its indirect handle"),
            offset: 0,
            count: 1,
            stride: DRAW_INDEXED_INDIRECT_ARGS_SIZE,
        }
    }

    #[derive(Default)]
    struct Recorder {
        commands: usize,
    }

    impl MeshRecorder for Recorder {
        fn set_pipeline(&mut self, _: RenderPipelineId) {
            self.commands += 1;
        }
        fn set_render_bind_group(&mut self, _: u32, _: BindGroupId, _: Vec<u32>) {
            self.commands += 1;
        }
        fn set_vertex_buffer(&mut self, _: u32, _: BufferId, _: u64, _: Option<NonZeroU64>) {
            self.commands += 1;
        }
        fn set_index_buffer(&mut self, _: BufferId, _: IndexFormat, _: u64, _: Option<NonZeroU64>) {
            self.commands += 1;
        }
        fn draw(&mut self, _: Range<u32>, _: Range<u32>) {
            self.commands += 1;
        }
        fn draw_indexed(&mut self, _: Range<u32>, _: i32, _: Range<u32>) {
            self.commands += 1;
        }
        fn draw_indexed_indirect(&mut self, _: BufferHandle, _: u64) {
            self.commands += 1;
        }
        fn multi_draw_indirect(&mut self, _: BufferHandle, _: u64, _: u32, _: u32) {
            self.commands += 1;
        }
    }

    #[test]
    fn direct_draw_rejects_zero_and_malformed_suballocation_ranges() {
        let mut draw = item();
        draw.vertex_sizes[0] = 0;
        assert!(!ranges_are_valid(&draw, false));

        let mut draw = item();
        draw.vertex_sizes.clear();
        assert!(!ranges_are_valid(&draw, false));

        let mut draw = item();
        draw.index_size = Some(0);
        assert!(!ranges_are_valid(&draw, false));

        let mut draw = item();
        draw.index_size = None;
        assert!(!ranges_are_valid(&draw, false));

        let mut draw = item();
        draw.index_size = Some(13);
        assert!(!ranges_are_valid(&draw, false));

        let mut draw = item();
        draw.first = 4;
        draw.count = u32::MAX;
        assert!(!ranges_are_valid(&draw, false));

        let mut draw = item();
        draw.count = 7;
        assert!(!ranges_are_valid(&draw, false));

        let mut draw = item();
        draw.indices = None;
        draw.index_size = None;
        draw.first = u32::MAX - 1;
        draw.count = 4;
        assert!(!ranges_are_valid(&draw, false));

        let mut recorder = Recorder::default();
        mesh_draw(&draw, &mut recorder);
        assert_eq!(recorder.commands, 0, "invalid range records no commands");
    }

    #[test]
    fn bin_draw_requires_exact_nonzero_vertex_and_index_ranges() {
        let mut draw = item();
        draw.indirect = Some(indirect_args());
        assert!(ranges_are_valid(&draw, true));
        let mut valid_recorder = Recorder::default();
        mesh_bin_draw(&draw, &mut valid_recorder);
        assert_eq!(
            valid_recorder.commands, 4,
            "valid indexed bin records all commands"
        );

        draw.index_size = Some(0);
        assert!(!ranges_are_valid(&draw, true));
        draw.index_size = Some(12);
        draw.vertex_sizes[0] = 0;
        assert!(!ranges_are_valid(&draw, true));
        draw.vertex_sizes[0] = 48;
        draw.indices = None;
        draw.index_size = None;
        assert!(!ranges_are_valid(&draw, true));

        let mut recorder = Recorder::default();
        mesh_bin_draw(&draw, &mut recorder);
        assert_eq!(recorder.commands, 0, "invalid range records no commands");
    }

    #[test]
    fn indirect_ranges_match_indexed_argument_abi_and_do_not_overflow() {
        let stride = DRAW_INDEXED_INDIRECT_ARGS_SIZE;
        assert!(indirect_range_is_valid(0, 1, stride));
        assert!(indirect_range_is_valid(4, 3, stride + 4));
        assert!(!indirect_range_is_valid(2, 1, stride));
        assert!(!indirect_range_is_valid(0, 0, stride));
        assert!(!indirect_range_is_valid(0, 1, stride - 4));
        assert!(!indirect_range_is_valid(0, 1, stride + 2));
        assert!(!indirect_range_is_valid(u64::MAX - 3, 2, stride));

        let mut draw = item();
        let mut args = indirect_args();
        args.offset = 2;
        draw.indirect = Some(args);
        let mut recorder = Recorder::default();
        mesh_bin_draw(&draw, &mut recorder);
        assert_eq!(
            recorder.commands, 0,
            "invalid indirect range records no commands"
        );
    }
}
