//! Draw functions and the recording context they record into.
//!
//! Bevy's render phases reference a `DrawFunctionId` into a per-phase
//! `DrawFunctions` registry; the draw command then records real GPU work. Prism
//! keeps that shape, but decouples the queue layer from the frame graph: a draw
//! function records into the object-safe [`MeshRecorder`] trait, which the graph
//! implements for its pass recorder ([`prism_render_graph::ExecuteContext`]) and
//! a test implements for any scratch type.
//!
//! A trait object (rather than a generic `C`) is deliberate: a pass hands out a
//! fresh `ExecuteContext<'_>` per call, so a draw function must be callable for
//! any lifetime — exactly what `&mut dyn MeshRecorder` gives and a monomorphic
//! `DrawFunctions<ExecuteContext<'a>>` cannot.

use core::ops::Range;

use prism_render_driver::{BufferId, IndexFormat, RenderPipelineId};

use crate::queue::PhaseItem;

/// The render commands a mesh draw function records.
///
/// Implemented for the frame graph's pass recorder; a minimal surface keeps the
/// queue layer independent of the graph's internals and lets a test capture the
/// exact command sequence.
pub trait MeshRecorder {
    /// Binds the active render pipeline.
    fn set_pipeline(&mut self, pipeline: RenderPipelineId);
    /// Binds a vertex buffer at `slot`.
    fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferId, offset: u64);
    /// Binds an index buffer.
    fn set_index_buffer(&mut self, buffer: BufferId, format: IndexFormat, offset: u64);
    /// Records a non-indexed draw.
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);
    /// Records an indexed draw.
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>);
    /// Records a single indexed indirect draw from a declared argument buffer.
    fn draw_indexed_indirect(&mut self, args: prism_render_graph::BufferHandle, offset: u64);
    /// Records `count` packed indexed indirect draws from a declared argument
    /// array (the GPU-driven bin path).
    fn multi_draw_indirect(
        &mut self,
        args: prism_render_graph::BufferHandle,
        offset: u64,
        count: u32,
        stride: u32,
    );
}

/// A registered draw function.
type DrawFn = Box<dyn Fn(&PhaseItem, &mut dyn MeshRecorder)>;

/// A stable handle to a registered draw function.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct DrawFunctionId(pub u32);

/// A registry of draw functions keyed by [`DrawFunctionId`].
#[derive(Default)]
pub struct DrawFunctions {
    functions: Vec<DrawFn>,
}

impl DrawFunctions {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
        }
    }

    /// Registers `function` and returns its id.
    pub fn register(&mut self, function: impl Fn(&PhaseItem, &mut dyn MeshRecorder) + 'static) -> DrawFunctionId {
        let id = DrawFunctionId(self.functions.len() as u32);
        self.functions.push(Box::new(function));
        id
    }

    /// The number of registered draw functions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.functions.len()
    }

    /// Whether no draw functions are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// Records `item` by invoking the draw function `id`.
    ///
    /// A no-op for an unknown id, so a queue entry can never panic the frame.
    pub fn call(&self, id: DrawFunctionId, item: &PhaseItem, context: &mut dyn MeshRecorder) {
        if let Some(function) = self.functions.get(id.0 as usize) {
            function(item, context);
        }
    }
}

impl MeshRecorder for prism_render_graph::ExecuteContext<'_> {
    fn set_pipeline(&mut self, pipeline: RenderPipelineId) {
        self.set_render_pipeline(pipeline);
    }

    fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferId, offset: u64) {
        self.set_vertex_buffer(slot, buffer, offset);
    }

    fn set_index_buffer(&mut self, buffer: BufferId, format: IndexFormat, offset: u64) {
        self.set_index_buffer(buffer, format, offset);
    }

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.draw(vertices, instances);
    }

    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.draw_indexed(indices, base_vertex, instances);
    }

    fn draw_indexed_indirect(&mut self, args: prism_render_graph::BufferHandle, offset: u64) {
        self.draw_indexed_indirect(args, offset);
    }

    fn multi_draw_indirect(
        &mut self,
        args: prism_render_graph::BufferHandle,
        offset: u64,
        count: u32,
        stride: u32,
    ) {
        self.multi_draw_indirect(args, offset, count, stride);
    }
}
