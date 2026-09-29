//! `GPU`-driven cloth solve: dispatch contract, persistent buffers, and the
//! async frame pipeline.
//!
//! The `CPU` cloth modules ([`super::dynamics`], [`super::constraints`],
//! [`super::collision`], [`super::embed`]) own the golden reference solver and
//! every per-frame routing decision. This module owns the contract for running
//! that same solver as a persistent, `GPU`-driven pipeline:
//!
//! * [`kernels`] — the compute-kernel dispatch contract ([`ClothKernel`] and
//!   its per-kernel [`kernels::KernelDescriptor`]): bind-group shape, workgroup
//!   tiling, dispatch domain and stable `WESL` entry-point name for each pass.
//! * [`buffers`] — the sizing of the device-resident buffers the solver reads
//!   and writes in place across frames, and the [`buffers::AsyncFrameState`]
//!   double-buffer state machine that sequences record → submit → retire so a
//!   slot is never read while the `GPU` is still writing it.
//! * [`pipeline`] — the Extract → Prepare → Queue frame plan that expands the
//!   substep loop and graph-color batching into a flat, ordered, budget-able
//!   dispatch schedule.
//!
//! Everything here is pure integer bookkeeping with no `GPU` handles, no floats
//! and no wall clock, so the whole `GPU` schedule is deterministic and
//! `CPU`-testable. The numerical passes themselves are authored in `WESL` and
//! mirror the `CPU` golden reference byte-for-byte.

pub mod buffers;
pub mod kernels;
pub mod pipeline;

pub use buffers::{
    AsyncFrameState, BufferCounts, BufferParity, FrameSlot, PersistentBufferSet, PipelineError,
    SlotState,
};
pub use kernels::{
    linear_group_count, BindGroupLayout, ClothKernel, DispatchDomain, KernelDescriptor,
    WorkgroupSize,
};
pub use pipeline::{
    extract, plan_frame, prepare, queue, ClothGpuExtract, ClothGpuFramePlan, ClothGpuPrepare,
    ClothGpuQueue, PlannedDispatch,
};
