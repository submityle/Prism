//! # `prism_render_graph`
//!
//! Prism's **frame render graph**: a declarative, backend-agnostic dependency
//! graph that compiles a single frame's passes into an immutable, replayable
//! execution plan over [`prism_render_driver`].
//!
//! Each frame the renderer *declares* passes — raster, compute, transfer,
//! present — in terms of **virtual resources** ([`TextureHandle`] /
//! [`BufferHandle`]) rather than real GPU objects. A pass states only *how* it
//! touches each resource ([`TextureUse`] / [`BufferUse`]); the compiler derives
//! everything else:
//!
//! - **SSA write-versioning** ([`handle`]). Every write mints a new resource
//!   version, so "reader of version `v` depends on the writer of `v`" is exact,
//!   with no manual edge wiring. Physical WAW and WAR dependencies protect
//!   storage even when a writer uses a stale input handle.
//! - **Dead-pass culling** ([`compile::cull`]). Reverse reachability from the
//!   frame's side-effecting roots (present, imported writes, explicit
//!   [`PassFlags::SIDE_EFFECT`]) drops passes whose results nobody consumes.
//! - **Topological scheduling** ([`compile::schedule`]). A deterministic Kahn
//!   sort respecting read-after-write *and* write-after-read ordering, with
//!   insertion-order tie-breaking for stable, reproducible frames.
//! - **Transient memory aliasing** ([`compile::alias`]). Lifetime analysis
//!   plus the shared kernel's deterministic first-fit interval assignment
//!   lets non-overlapping transients share physical memory, recorded as plan
//!   metadata for native placed-resource backends.
//! - **Automatic barriers** ([`compile::barrier`]). The driver
//!   [`StateTracker`] turns declared uses into the minimal set of
//!   synchronization barriers, per pass, carried as plan metadata.
//!
//! Compilation is pure and allocation-light; it borrows the graph and yields an
//! [`ExecutionPlan`] of indices and metadata. Execution then realizes
//! transients, runs each surviving pass's recorded closure to assemble a driver
//! [`CommandBuffer`], and submits it.
//!
//! The crate is `no_std + alloc` and `unsafe`-free: all GPU interaction flows
//! through the [`prism_render_driver`] traits, so the graph is fully
//! deterministic and testable against a mock device with no real GPU.
//!
//! ## Parallel recording
//!
//! The optional `parallel` feature (which pulls in `std`) records a frame's
//! surviving passes concurrently on a scoped worker pool, then assembles their
//! commands in the compiled plan's deterministic order. The assembled
//! [`CommandBuffer`] is byte-for-byte identical to the sequential one, and the
//! feature is off by default so the crate stays `no_std`.
//!
//! [`prism_render_driver`]: prism_render_driver
//! [`StateTracker`]: prism_render_driver::StateTracker
//! [`CommandBuffer`]: prism_render_driver::CommandBuffer

#![cfg_attr(all(not(test), not(feature = "parallel")), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]

extern crate alloc;
#[cfg(feature = "parallel")]
extern crate std;

pub mod access;
pub mod blackboard;
pub mod builder;
pub mod capacity;
pub mod compile;
pub mod desc;
pub mod frame_graph;
pub mod graph;
pub mod handle;
pub mod pass;
pub mod plan;
pub mod resource;
pub mod systems;

mod record;

#[cfg(feature = "parallel")]
mod parallel;

pub use access::{full_range, BufferUse, TextureUse};
pub use blackboard::Blackboard;
pub use builder::{ExecuteContext, PassBuilder};
pub use desc::{BufferDesc, SizeClass, TextureDesc};
pub use graph::{ExecuteFn, GraphAllocations, PreparedGraph, RecordedGraph, RenderGraph};
pub use handle::{
    BufferHandle, BufferMarker, ResourceHandle, ResourceIndex, TextureHandle, TextureMarker,
};
pub use pass::{
    BufferAccessRecord, PassFlags, PassKind, PassNode, TextureAccessRecord, ThreadBound,
};
pub use plan::{
    AliasPlan, AliasSlot, ColorAttachmentPlan, CompileError, CompiledAccess, DepthAttachmentPlan,
    ExecutionPlan, GraphBarrier, LogicalBarrier, LogicalResource, RasterAttachments,
};
pub use resource::{
    BufferResource, ImportedBuffer, ImportedTexture, Lifetime, RealizedTexture, TextureResource,
};
pub use systems::{configure_render_graph_sets, RenderGraphSystems};

#[cfg(test)]
#[path = "../text/unit/graph.rs"]
mod tests;

#[cfg(test)]
#[path = "../text/unit/compiler.rs"]
mod compiler_tests;

#[cfg(test)]
#[path = "../text/unit/recording.rs"]
mod recording_tests;
