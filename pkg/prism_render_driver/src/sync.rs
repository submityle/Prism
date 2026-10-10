//! GPU synchronization model: pipeline-stage/access masks, image layouts, and
//! the barrier vocabulary the auto-barrier solver ([`crate::tracker`]) emits.
//!
//! This module is the single source of truth for *what state a resource is in*
//! and *how to describe a transition between two states*. It is deliberately
//! backend-agnostic: a Vulkan backend maps [`PipelineStages`]/[`Accesses`] onto
//! `VkPipelineStageFlags2`/`VkAccessFlags2` and [`TextureLayout`] onto
//! `VkImageLayout`; a Metal/D3D12 backend maps the same vocabulary onto its
//! native resource-state model (`MTLBarrierScope`/`D3D12_RESOURCE_STATES`); a
//! wgpu backend ignores it entirely because wgpu tracks state internally. The
//! vocabulary is modeled on Vulkan's synchronization2 because it is the most
//! explicit of the mainstream APIs and the others are strict subsets.
//!
//! Everything here is pure data: `Copy`, `no_std`, no `unsafe`, with checked
//! construction. The solver composes these into batches of [`Barrier`]s.

use crate::flags::bitflags;

bitflags! {
    /// Pipeline stages at which a memory access happens, used to scope the
    /// source and destination of a barrier so the backend only waits for the
    /// stages that actually matter (modeled on `VkPipelineStageFlags2`).
    ///
    /// Finer stages mean less over-synchronization: transitioning a texture
    /// from `COLOR_ATTACHMENT_OUTPUT` to `FRAGMENT_SHADER` lets vertex work of
    /// the next pass overlap the previous pass's fragment work.
    pub struct PipelineStages {
        /// Draw/dispatch indirect argument fetch.
        const DRAW_INDIRECT = 1 << 0;
        /// Vertex and index buffer input assembly.
        const VERTEX_INPUT = 1 << 1;
        /// Vertex shader execution.
        const VERTEX_SHADER = 1 << 2;
        /// Fragment shader execution.
        const FRAGMENT_SHADER = 1 << 3;
        /// Early depth/stencil tests (before the fragment shader).
        const EARLY_FRAGMENT_TESTS = 1 << 4;
        /// Late depth/stencil tests (after the fragment shader).
        const LATE_FRAGMENT_TESTS = 1 << 5;
        /// Color attachment blending and writeback.
        const COLOR_ATTACHMENT_OUTPUT = 1 << 6;
        /// Compute shader execution.
        const COMPUTE_SHADER = 1 << 7;
        /// Copy / blit / resolve transfer operations.
        const TRANSFER = 1 << 8;
        /// Acceleration-structure build (ray tracing).
        const ACCELERATION_STRUCTURE_BUILD = 1 << 9;
        /// Ray-tracing pipeline shader execution.
        const RAY_TRACING_SHADER = 1 << 10;
        /// Presentation to a surface.
        const PRESENT = 1 << 11;
        /// Host (CPU) access, e.g. mapped-memory reads/writes.
        const HOST = 1 << 12;
    }
}

bitflags! {
    /// How memory is accessed at a [`PipelineStages`] scope. Separating reads
    /// from writes lets the solver skip redundant read→read barriers while
    /// still ordering write→read and write→write hazards (modeled on
    /// `VkAccessFlags2`).
    pub struct Accesses {
        /// Indirect command buffer read.
        const INDIRECT_COMMAND_READ = 1 << 0;
        /// Index buffer read.
        const INDEX_READ = 1 << 1;
        /// Vertex attribute read.
        const VERTEX_ATTRIBUTE_READ = 1 << 2;
        /// Uniform buffer read.
        const UNIFORM_READ = 1 << 3;
        /// Sampled texture / read-only storage read in a shader.
        const SHADER_READ = 1 << 4;
        /// Storage resource write in a shader.
        const SHADER_WRITE = 1 << 5;
        /// Color attachment read (blending/load).
        const COLOR_ATTACHMENT_READ = 1 << 6;
        /// Color attachment write.
        const COLOR_ATTACHMENT_WRITE = 1 << 7;
        /// Depth/stencil attachment read (test/load).
        const DEPTH_STENCIL_READ = 1 << 8;
        /// Depth/stencil attachment write.
        const DEPTH_STENCIL_WRITE = 1 << 9;
        /// Transfer source read.
        const TRANSFER_READ = 1 << 10;
        /// Transfer destination write.
        const TRANSFER_WRITE = 1 << 11;
        /// Host read of mapped memory.
        const HOST_READ = 1 << 12;
        /// Host write to mapped memory.
        const HOST_WRITE = 1 << 13;
        /// Acceleration-structure read.
        const ACCELERATION_STRUCTURE_READ = 1 << 14;
        /// Acceleration-structure write (build).
        const ACCELERATION_STRUCTURE_WRITE = 1 << 15;
    }
}

impl Accesses {
    /// The mask of every write access. A state containing any of these bits is
    /// a *writing* state and participates in write-after-write and
    /// read-after-write hazards.
    #[must_use]
    pub const fn write_mask() -> Self {
        Self::SHADER_WRITE
            .union(Self::COLOR_ATTACHMENT_WRITE)
            .union(Self::DEPTH_STENCIL_WRITE)
            .union(Self::TRANSFER_WRITE)
            .union(Self::HOST_WRITE)
            .union(Self::ACCELERATION_STRUCTURE_WRITE)
    }

    /// Whether these accesses include any write.
    #[must_use]
    pub const fn is_write(self) -> bool {
        self.intersects(Self::write_mask())
    }

    /// Whether these accesses are read-only (no write bits set).
    #[must_use]
    pub const fn is_read_only(self) -> bool {
        !self.is_write()
    }
}

/// The layout a texture's memory is organized in. Native APIs require a
/// texture to be in the right layout for each use (e.g. a Vulkan image must be
/// `COLOR_ATTACHMENT_OPTIMAL` to be rendered to and `SHADER_READ_ONLY_OPTIMAL`
/// to be sampled); switching layouts is the primary reason texture barriers
/// exist. wgpu and Metal hide this, but the enum is backend-neutral and simply
/// collapses to a no-op where the API manages layout implicitly.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TextureLayout {
    /// Contents are undefined; used as the "before" layout when the previous
    /// contents may be discarded (cheaper than preserving them).
    Undefined,
    /// General-purpose layout valid for any access; may be slower than a
    /// specialized layout but needs no transition between dissimilar uses.
    General,
    /// Optimal for use as a color render target.
    ColorAttachment,
    /// Optimal for read/write use as a depth/stencil attachment.
    DepthStencilAttachment,
    /// Optimal for read-only use as a depth/stencil attachment (allows
    /// simultaneous sampling of depth).
    DepthStencilReadOnly,
    /// Optimal for sampling / read-only shader access.
    ShaderReadOnly,
    /// Optimal as a transfer source.
    TransferSrc,
    /// Optimal as a transfer destination.
    TransferDst,
    /// Optimal for presentation to a surface.
    Present,
}

impl TextureLayout {
    /// Whether this layout preserves the texture's existing contents across a
    /// transition *into* it. [`TextureLayout::Undefined`] does not, which lets
    /// the backend skip the (potentially expensive) decompress/preserve step.
    #[must_use]
    pub const fn preserves_contents(self) -> bool {
        !matches!(self, Self::Undefined)
    }
}

/// Which hardware queue a piece of work runs on. Cross-queue transitions need a
/// queue-ownership transfer (a release on the source queue paired with an
/// acquire on the destination queue) in explicit APIs; same-queue transitions
/// only need an execution+memory barrier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum QueueKind {
    /// The graphics (and by convention universal) queue.
    Graphics,
    /// An async-compute queue.
    Compute,
    /// A dedicated transfer/DMA queue.
    Transfer,
}

/// The synchronization state of a buffer: the stages and accesses that last
/// touched it. Buffers have no layout, so a transition is purely an
/// execution+memory dependency.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferState {
    /// Pipeline stages that (will) access the buffer.
    pub stages: PipelineStages,
    /// How those stages access it.
    pub accesses: Accesses,
    /// The queue the access runs on.
    pub queue: QueueKind,
}

impl BufferState {
    /// A fresh/unused buffer state: no stages, no accesses, on the graphics
    /// queue. Transitioning *from* this state needs no source barrier.
    #[must_use]
    pub const fn initial() -> Self {
        Self {
            stages: PipelineStages::NONE,
            accesses: Accesses::NONE,
            queue: QueueKind::Graphics,
        }
    }

    /// Builds a state from explicit stages/accesses on the graphics queue.
    #[must_use]
    pub const fn new(stages: PipelineStages, accesses: Accesses) -> Self {
        Self {
            stages,
            accesses,
            queue: QueueKind::Graphics,
        }
    }

    /// Returns this state rebound to a different queue.
    #[must_use]
    pub const fn on_queue(mut self, queue: QueueKind) -> Self {
        self.queue = queue;
        self
    }

    /// Whether this state writes the buffer.
    #[must_use]
    pub const fn is_write(self) -> bool {
        self.accesses.is_write()
    }

    /// Read as the source of an indirect draw/dispatch.
    #[must_use]
    pub const fn indirect() -> Self {
        Self::new(
            PipelineStages::DRAW_INDIRECT,
            Accesses::INDIRECT_COMMAND_READ,
        )
    }

    /// Read as an index buffer.
    #[must_use]
    pub const fn index() -> Self {
        Self::new(PipelineStages::VERTEX_INPUT, Accesses::INDEX_READ)
    }

    /// Read as a vertex buffer.
    #[must_use]
    pub const fn vertex() -> Self {
        Self::new(
            PipelineStages::VERTEX_INPUT,
            Accesses::VERTEX_ATTRIBUTE_READ,
        )
    }

    /// Read as a uniform buffer in the given stages.
    #[must_use]
    pub const fn uniform(stages: PipelineStages) -> Self {
        Self::new(stages, Accesses::UNIFORM_READ)
    }

    /// Read as a storage buffer in the given stages.
    #[must_use]
    pub const fn storage_read(stages: PipelineStages) -> Self {
        Self::new(stages, Accesses::SHADER_READ)
    }

    /// Read-write as a storage buffer in the given stages.
    #[must_use]
    pub const fn storage_write(stages: PipelineStages) -> Self {
        Self::new(stages, Accesses::SHADER_READ.union(Accesses::SHADER_WRITE))
    }

    /// Transfer (copy) source.
    #[must_use]
    pub const fn copy_src() -> Self {
        Self::new(PipelineStages::TRANSFER, Accesses::TRANSFER_READ)
    }

    /// Transfer (copy) destination.
    #[must_use]
    pub const fn copy_dst() -> Self {
        Self::new(PipelineStages::TRANSFER, Accesses::TRANSFER_WRITE)
    }
}

/// A half-open range of texture subresources (mip levels and array layers) a
/// barrier applies to. Transitioning a subresource range rather than the whole
/// texture avoids over-synchronizing mips/layers that are not involved.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SubresourceRange {
    /// First mip level (inclusive).
    pub base_mip_level: u32,
    /// Number of mip levels covered.
    pub mip_level_count: u32,
    /// First array layer (inclusive).
    pub base_array_layer: u32,
    /// Number of array layers covered.
    pub array_layer_count: u32,
}

impl SubresourceRange {
    /// A range covering every mip level and array layer of a texture.
    #[must_use]
    pub const fn all() -> Self {
        Self {
            base_mip_level: 0,
            mip_level_count: u32::MAX,
            base_array_layer: 0,
            array_layer_count: u32::MAX,
        }
    }

    /// A range covering a single mip level and array layer.
    #[must_use]
    pub const fn single(mip_level: u32, array_layer: u32) -> Self {
        Self {
            base_mip_level: mip_level,
            mip_level_count: 1,
            base_array_layer: array_layer,
            array_layer_count: 1,
        }
    }

    /// Whether this range covers the whole texture (used to fast-path merges).
    #[must_use]
    pub const fn is_full(self) -> bool {
        self.base_mip_level == 0
            && self.mip_level_count == u32::MAX
            && self.base_array_layer == 0
            && self.array_layer_count == u32::MAX
    }
}

/// The synchronization state of a texture (or a subresource of one): stages,
/// accesses, and the all-important image [`TextureLayout`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TextureState {
    /// Pipeline stages that (will) access the texture.
    pub stages: PipelineStages,
    /// How those stages access it.
    pub accesses: Accesses,
    /// The memory layout the texture is in.
    pub layout: TextureLayout,
    /// The queue the access runs on.
    pub queue: QueueKind,
}

impl TextureState {
    /// A fresh/undefined texture state: no stages/accesses, `Undefined`
    /// layout, graphics queue. The natural "before" state for a newly created
    /// or discard-loaded texture.
    #[must_use]
    pub const fn initial() -> Self {
        Self {
            stages: PipelineStages::NONE,
            accesses: Accesses::NONE,
            layout: TextureLayout::Undefined,
            queue: QueueKind::Graphics,
        }
    }

    /// Builds a state from explicit parts on the graphics queue.
    #[must_use]
    pub const fn new(stages: PipelineStages, accesses: Accesses, layout: TextureLayout) -> Self {
        Self {
            stages,
            accesses,
            layout,
            queue: QueueKind::Graphics,
        }
    }

    /// Returns this state rebound to a different queue.
    #[must_use]
    pub const fn on_queue(mut self, queue: QueueKind) -> Self {
        self.queue = queue;
        self
    }

    /// Whether this state writes the texture.
    #[must_use]
    pub const fn is_write(self) -> bool {
        self.accesses.is_write()
    }

    /// Use as a color render target.
    #[must_use]
    pub const fn color_target() -> Self {
        Self::new(
            PipelineStages::COLOR_ATTACHMENT_OUTPUT,
            Accesses::COLOR_ATTACHMENT_WRITE.union(Accesses::COLOR_ATTACHMENT_READ),
            TextureLayout::ColorAttachment,
        )
    }

    /// Use as a writable depth/stencil attachment.
    #[must_use]
    pub const fn depth_write() -> Self {
        Self::new(
            PipelineStages::EARLY_FRAGMENT_TESTS.union(PipelineStages::LATE_FRAGMENT_TESTS),
            Accesses::DEPTH_STENCIL_WRITE.union(Accesses::DEPTH_STENCIL_READ),
            TextureLayout::DepthStencilAttachment,
        )
    }

    /// Use as a read-only depth/stencil attachment.
    #[must_use]
    pub const fn depth_read() -> Self {
        Self::new(
            PipelineStages::EARLY_FRAGMENT_TESTS.union(PipelineStages::LATE_FRAGMENT_TESTS),
            Accesses::DEPTH_STENCIL_READ,
            TextureLayout::DepthStencilReadOnly,
        )
    }

    /// Sample / read-only shader access in the given stages.
    #[must_use]
    pub const fn shader_read(stages: PipelineStages) -> Self {
        Self::new(stages, Accesses::SHADER_READ, TextureLayout::ShaderReadOnly)
    }

    /// Read-write storage image access in the given stages (uses the `General`
    /// layout, which is required for storage writes).
    #[must_use]
    pub const fn storage_write(stages: PipelineStages) -> Self {
        Self::new(
            stages,
            Accesses::SHADER_READ.union(Accesses::SHADER_WRITE),
            TextureLayout::General,
        )
    }

    /// Transfer (copy/blit) source.
    #[must_use]
    pub const fn copy_src() -> Self {
        Self::new(
            PipelineStages::TRANSFER,
            Accesses::TRANSFER_READ,
            TextureLayout::TransferSrc,
        )
    }

    /// Transfer (copy/blit) destination.
    #[must_use]
    pub const fn copy_dst() -> Self {
        Self::new(
            PipelineStages::TRANSFER,
            Accesses::TRANSFER_WRITE,
            TextureLayout::TransferDst,
        )
    }

    /// Ready for presentation to a surface.
    #[must_use]
    pub const fn present() -> Self {
        Self::new(
            PipelineStages::PRESENT,
            Accesses::NONE,
            TextureLayout::Present,
        )
    }

    /// The undefined "before" state carrying a specific queue, used when a
    /// texture's contents may be discarded.
    #[must_use]
    pub const fn undefined() -> Self {
        Self::initial()
    }
}

/// Whether a transition starts, finishes, or performs a barrier in one step.
///
/// Split barriers (a [`BarrierKind::Begin`] right after the last use followed
/// by a [`BarrierKind::End`] right before the next use) let the GPU overlap the
/// transition's latency with unrelated work in between, which is a key AAA
/// optimization. A backend that cannot split simply treats `Begin` as a no-op
/// and `End` as a full [`BarrierKind::Immediate`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BarrierKind {
    /// A complete barrier executed at a single point.
    Immediate,
    /// The start of a split barrier (signal); safe to no-op if unsupported.
    Begin,
    /// The end of a split barrier (wait).
    End,
}

/// A buffer memory/execution barrier describing the transition between two
/// [`BufferState`]s.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferBarrier<B: Copy> {
    /// The buffer being transitioned.
    pub buffer: B,
    /// State before the barrier.
    pub before: BufferState,
    /// State after the barrier.
    pub after: BufferState,
    /// Whether this is an immediate or split barrier.
    pub kind: BarrierKind,
}

/// A texture barrier describing a layout + memory/execution transition over a
/// subresource range.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TextureBarrier<T: Copy> {
    /// The texture being transitioned.
    pub texture: T,
    /// The subresource range the barrier applies to.
    pub range: SubresourceRange,
    /// State before the barrier.
    pub before: TextureState,
    /// State after the barrier.
    pub after: TextureState,
    /// Whether this is an immediate or split barrier.
    pub kind: BarrierKind,
}

/// A single barrier over either a buffer or a texture, generic over the
/// backend's handle types so the shared solver stays free of RHI id details.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Barrier<B: Copy, T: Copy> {
    /// A buffer barrier.
    Buffer(BufferBarrier<B>),
    /// A texture barrier.
    Texture(TextureBarrier<T>),
}

impl<B: Copy, T: Copy> Barrier<B, T> {
    /// The [`BarrierKind`] of this barrier.
    #[must_use]
    pub const fn kind(&self) -> BarrierKind {
        match self {
            Self::Buffer(b) => b.kind,
            Self::Texture(t) => t.kind,
        }
    }

    /// Whether this barrier crosses a queue boundary (needs an ownership
    /// transfer in explicit APIs).
    #[must_use]
    pub const fn is_queue_transfer(&self) -> bool {
        match self {
            Self::Buffer(b) => {
                !matches!(
                    (b.before.queue, b.after.queue),
                    (QueueKind::Graphics, QueueKind::Graphics)
                        | (QueueKind::Compute, QueueKind::Compute)
                        | (QueueKind::Transfer, QueueKind::Transfer)
                ) && queues_differ(b.before.queue, b.after.queue)
            }
            Self::Texture(t) => queues_differ(t.before.queue, t.after.queue),
        }
    }
}

/// Whether two queues are different kinds.
#[must_use]
const fn queues_differ(a: QueueKind, b: QueueKind) -> bool {
    !matches!(
        (a, b),
        (QueueKind::Graphics, QueueKind::Graphics)
            | (QueueKind::Compute, QueueKind::Compute)
            | (QueueKind::Transfer, QueueKind::Transfer)
    )
}

/// Merges two *read-only* buffer states into one that covers both, so a
/// resource read by several consumers transitions once into the union of their
/// stages/accesses. Returns `None` if either state writes, because writes may
/// not be merged (they must be ordered).
#[must_use]
pub fn merge_read_buffer_states(a: BufferState, b: BufferState) -> Option<BufferState> {
    if a.is_write() || b.is_write() || a.queue as u8 != b.queue as u8 {
        return None;
    }
    Some(BufferState {
        stages: a.stages.union(b.stages),
        accesses: a.accesses.union(b.accesses),
        queue: a.queue,
    })
}

/// Merges two *read-only* texture states that share a layout and queue into
/// their union. Returns `None` if either writes, their layouts differ, or they
/// run on different queues — any of which forces a real transition.
#[must_use]
pub fn merge_read_texture_states(a: TextureState, b: TextureState) -> Option<TextureState> {
    if a.is_write() || b.is_write() || a.layout != b.layout || (a.queue as u8) != (b.queue as u8) {
        return None;
    }
    Some(TextureState {
        stages: a.stages.union(b.stages),
        accesses: a.accesses.union(b.accesses),
        layout: a.layout,
        queue: a.queue,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_mask_classification() {
        assert!(Accesses::SHADER_WRITE.is_write());
        assert!(Accesses::COLOR_ATTACHMENT_WRITE.is_write());
        assert!(Accesses::SHADER_READ.is_read_only());
        assert!(Accesses::NONE.is_read_only());
        assert!(Accesses::SHADER_READ
            .union(Accesses::SHADER_WRITE)
            .is_write());
    }

    #[test]
    fn layout_contents_preservation() {
        assert!(!TextureLayout::Undefined.preserves_contents());
        assert!(TextureLayout::ShaderReadOnly.preserves_contents());
        assert!(TextureLayout::ColorAttachment.preserves_contents());
    }

    #[test]
    fn buffer_common_states() {
        assert!(BufferState::copy_dst().is_write());
        assert!(!BufferState::copy_src().is_write());
        assert!(!BufferState::uniform(PipelineStages::VERTEX_SHADER).is_write());
        assert!(BufferState::storage_write(PipelineStages::COMPUTE_SHADER).is_write());
        assert_eq!(BufferState::initial().accesses, Accesses::NONE);
    }

    #[test]
    fn texture_common_states() {
        assert_eq!(
            TextureState::color_target().layout,
            TextureLayout::ColorAttachment
        );
        assert_eq!(TextureState::present().layout, TextureLayout::Present);
        assert!(TextureState::copy_dst().is_write());
        assert!(!TextureState::shader_read(PipelineStages::FRAGMENT_SHADER).is_write());
        assert_eq!(TextureState::initial().layout, TextureLayout::Undefined);
    }

    #[test]
    fn subresource_range_helpers() {
        assert!(SubresourceRange::all().is_full());
        assert!(!SubresourceRange::single(0, 0).is_full());
        let s = SubresourceRange::single(2, 3);
        assert_eq!(s.base_mip_level, 2);
        assert_eq!(s.base_array_layer, 3);
        assert_eq!(s.mip_level_count, 1);
    }

    #[test]
    fn merge_reads() {
        let a = BufferState::uniform(PipelineStages::VERTEX_SHADER);
        let b = BufferState::storage_read(PipelineStages::FRAGMENT_SHADER);
        let m = merge_read_buffer_states(a, b).expect("two reads merge");
        assert!(m.stages.contains(PipelineStages::VERTEX_SHADER));
        assert!(m.stages.contains(PipelineStages::FRAGMENT_SHADER));
        assert!(m.accesses.contains(Accesses::UNIFORM_READ));
        assert!(m.accesses.contains(Accesses::SHADER_READ));

        // A write cannot merge.
        assert!(merge_read_buffer_states(a, BufferState::copy_dst()).is_none());
    }

    #[test]
    fn merge_texture_reads_requires_same_layout() {
        let a = TextureState::shader_read(PipelineStages::VERTEX_SHADER);
        let b = TextureState::shader_read(PipelineStages::FRAGMENT_SHADER);
        assert!(merge_read_texture_states(a, b).is_some());
        // Different layout -> no merge.
        assert!(merge_read_texture_states(a, TextureState::depth_read()).is_none());
    }

    #[test]
    fn queue_transfer_detection() {
        let b: Barrier<u32, u32> = Barrier::Buffer(BufferBarrier {
            buffer: 0,
            before: BufferState::storage_write(PipelineStages::COMPUTE_SHADER)
                .on_queue(QueueKind::Compute),
            after: BufferState::vertex().on_queue(QueueKind::Graphics),
            kind: BarrierKind::Immediate,
        });
        assert!(b.is_queue_transfer());

        let same: Barrier<u32, u32> = Barrier::Buffer(BufferBarrier {
            buffer: 0,
            before: BufferState::copy_dst(),
            after: BufferState::vertex(),
            kind: BarrierKind::Immediate,
        });
        assert!(!same.is_queue_transfer());
    }
}
