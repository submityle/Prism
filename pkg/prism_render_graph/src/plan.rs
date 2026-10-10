//! The immutable, replayable product of compilation.
//!
//! [`RenderGraph::compile`](crate::RenderGraph::compile) borrows the graph and
//! returns an [`ExecutionPlan`]: the surviving passes in execution order plus
//! all the metadata a backend needs to run the frame — per-pass barriers,
//! transient memory aliasing offsets, resolved attachment load/store, and which
//! resources must be realized. The plan references passes by index so it never
//! copies the move-only execute closures; execution consumes those from the
//! graph itself.

use alloc::vec::Vec;

use prism_render_driver::{Barrier, DepthLoadOp, LoadOp, StoreOp};

use crate::handle::ResourceIndex;

/// A synchronization barrier keyed by virtual-resource index.
///
/// The barrier solver runs at compile time, before realization, so barriers
/// name [`ResourceIndex`] rather than driver ids. Backends that need explicit
/// barriers (Vulkan/D3D12/Metal) map each index to its realized id; auto-synced
/// backends (wgpu) treat the plan as a verification oracle.
pub type GraphBarrier = Barrier<ResourceIndex, ResourceIndex>;

/// Why compilation failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CompileError {
    /// The pass dependency graph contains a cycle and cannot be scheduled.
    /// Carries the number of passes that could not be ordered.
    Cycle {
        /// How many passes remained unscheduled when progress stalled.
        unscheduled: usize,
    },
}

impl core::fmt::Display for CompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Cycle { unscheduled } => write!(
                f,
                "render graph has a dependency cycle ({unscheduled} passes unscheduled)"
            ),
        }
    }
}

/// A physical-memory slot assigned to a transient resource.
///
/// Produced by lifetime-interval aliasing. Transients whose live ranges do not
/// overlap receive the same `offset`, so a backend with placed resources can
/// bind them to one heap region. Backends without placed resources create a
/// distinct object per resource and ignore the offset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AliasSlot {
    /// The byte offset into the transient heap.
    pub offset: u64,
    /// The byte size reserved.
    pub size: u64,
}

/// Transient memory aliasing assignment for a frame.
///
/// Indexed by [`ResourceIndex`]; `None` means the resource is not an aliasable
/// transient (imported, persistent, or culled). `heap_size` is the peak
/// simultaneous footprint — the size of the single heap that backs every slot.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct AliasPlan {
    /// Per-texture slot assignment.
    pub texture_slots: Vec<Option<AliasSlot>>,
    /// Per-buffer slot assignment.
    pub buffer_slots: Vec<Option<AliasSlot>>,
    /// The peak simultaneous transient footprint in bytes.
    pub heap_size: u64,
}

/// A resolved color attachment of a raster pass.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ColorAttachmentPlan {
    /// The virtual texture bound.
    pub resource: ResourceIndex,
    /// The SSA version written by this attachment.
    pub output_version: u32,
    /// Derived load behavior.
    pub load: LoadOp,
    /// Derived store behavior (`Discard` when the result is never consumed).
    pub store: StoreOp,
}

/// A resolved depth/stencil attachment of a raster pass.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DepthAttachmentPlan {
    /// The virtual texture bound.
    pub resource: ResourceIndex,
    /// The SSA version written (or read) by this attachment.
    pub output_version: u32,
    /// Whether the pass writes depth (vs. a read-only depth test).
    pub writes: bool,
    /// Derived depth load behavior.
    pub depth_load: DepthLoadOp,
    /// Derived depth store behavior.
    pub depth_store: StoreOp,
}

/// The resolved attachments of one raster pass.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct RasterAttachments {
    /// Color attachments in `@location` order.
    pub colors: Vec<ColorAttachmentPlan>,
    /// The optional depth/stencil attachment.
    pub depth: Option<DepthAttachmentPlan>,
}

/// The compiled plan for one frame.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct ExecutionPlan {
    /// Surviving pass indices (into the graph's pass list) in execution order.
    pub order: Vec<usize>,
    /// Per-original-pass liveness, indexed by pass index.
    pub alive: Vec<bool>,
    /// Barriers to run before each ordered pass, parallel to [`Self::order`].
    pub barriers: Vec<Vec<GraphBarrier>>,
    /// Barriers to run after the last pass to satisfy imported exit states.
    pub final_barriers: Vec<GraphBarrier>,
    /// Resolved attachments for each ordered pass (`Some` for raster passes),
    /// parallel to [`Self::order`].
    pub attachments: Vec<Option<RasterAttachments>>,
    /// Transient/persistent memory aliasing assignment.
    pub alias: AliasPlan,
    /// Whether each texture (by [`ResourceIndex`]) is referenced by a surviving
    /// pass and must be realized.
    pub used_textures: Vec<bool>,
    /// Whether each buffer (by [`ResourceIndex`]) is referenced by a surviving
    /// pass and must be realized.
    pub used_buffers: Vec<bool>,
}

impl ExecutionPlan {
    /// The number of surviving passes.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.order.len()
    }

    /// The total number of barriers across every surviving pass plus the final
    /// imported-exit transitions.
    #[must_use]
    pub fn barrier_count(&self) -> usize {
        self.barriers.iter().map(Vec::len).sum::<usize>() + self.final_barriers.len()
    }
}
