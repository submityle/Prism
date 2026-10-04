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
//! * [`upload`] — the class-separated constraint packing that turns an
//!   authored constraint list into the per-class per-color counts and
//!   contiguous by-color buffer layout the dispatch schedule addresses.
//!
//! Everything here is pure integer bookkeeping with no `GPU` handles, no floats
//! and no wall clock, so the whole `GPU` schedule is deterministic and
//! `CPU`-testable.
//!
//! # Single source of truth (no double-implementation)
//!
//! This module never re-derives any cloth-solver math. The one authoritative
//! algorithm lives in [`prism_physics_core`]`::soft`; the sibling `CPU` cloth
//! modules ([`super::dynamics`], [`super::constraints`], [`super::collision`],
//! [`super::bending`], …) are thin render-side façades that delegate every
//! numeric step to it. The `WESL` numerical passes this schedule dispatches
//! (authored in `prism_render_scene`'s `shaders/cloth_*.wesl`) are a *verified
//! binding* of that same golden, not an independent implementation: each pass
//! is pinned to the `CPU` golden by a parity test in
//! `prism_render_scene::cloth` (`*_parity`). Integer bookkeeping — cell
//! assignment, buffer strides, dispatch order — is held exact; the float
//! projection/collision passes match within a tight tolerance (`GPU` fused
//! multiply-add and div/sqrt rounding), the same parity model the physics
//! engine's own GPU backend uses.
//!
//! The render frame-graph `WESL` path here and the physics engine's standalone
//! `wgpu` compute backend (`prism_physics_gpu::cloth`, authored in `WGSL`)
//! are therefore two runtime bindings of the one `prism_physics_core` golden —
//! one driven inside the renderer's GPU scene, one for headless/physics-only
//! simulation — rather than two competing solvers. New GPU cloth passes must be
//! added by binding to that golden (and its parity harness), never by hand-
//! porting fresh solver arithmetic into either shader language.

pub mod buffers;
pub mod kernels;
pub mod pipeline;
pub mod upload;

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
pub use upload::{color_bending, plan_constraint_upload, BendingUploadPlan, ConstraintUploadPlan};
