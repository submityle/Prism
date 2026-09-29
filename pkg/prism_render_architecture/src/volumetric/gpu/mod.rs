//! `GPU`-driven volumetric cloud/atmosphere solve: the compute-dispatch
//! contract, the persistent device buffers, the async frame pipeline, and the
//! shared-service wiring (design section 17, milestone M8).
//!
//! The `CPU` volumetric modules ([`super::noise`], [`super::modeling`],
//! [`super::weather`], [`super::raymarch`], [`super::scatter`],
//! [`super::multiscatter`], [`super::shadow`], [`super::avsm`],
//! [`super::temporal`]) own the golden reference numerics and every per-frame
//! routing decision. This module owns the contract for running those same
//! passes as a persistent, `GPU`-driven pipeline authored in `WESL`:
//!
//! * [`kernels`] — the compute-kernel dispatch contract
//!   ([`kernels::VolumetricKernel`] and its per-kernel
//!   [`kernels::KernelDescriptor`]): bind-group shape, workgroup tiling,
//!   dispatch-domain shape and stable `WESL` entry-point name for each pass.
//! * [`buffers`] — the sizing of the device-resident buffers the solve reads
//!   and writes in place across frames (the density cache, weather map, low-res
//!   ray-march target, reprojection history, cloud-shadow map and
//!   multiple-scatter `LUT`), and the [`buffers::AsyncFrameState`]
//!   double-buffer state machine that sequences record -> submit -> retire so a
//!   history slot is never sampled while the `GPU` is still writing it.
//! * [`pipeline`] — the Extract -> Prepare -> Queue frame plan that expands the
//!   fixed producer-before-consumer chain into a flat, ordered, budget-able
//!   dispatch schedule.
//! * [`shared_services`] — the signature of how each kernel consumes or writes
//!   the shared advanced base (hybrid `GI`, `ReSTIR`, the virtual shadow map,
//!   the froxel volume, the atmosphere `LUT`, the path-traced reference and the
//!   temporal-upsampling history), which the volumetric subsystem consumes and
//!   never re-implements.
//!
//! Everything here is pure integer/enum bookkeeping with no `GPU` handles, no
//! floats and no wall clock, so the whole `GPU` schedule is deterministic and
//! `CPU`-testable. The numerical passes themselves are authored in `WESL` and
//! mirror the `CPU` golden reference.
//!
//! **Not machine-verified.** The sandbox has no `GPU` and no `WESL` compiler,
//! so nothing below is validated against a real backend; the workgroup tiles,
//! buffer strides and bindings are design targets to be re-tuned once the
//! backend lands.

pub mod buffers;
pub mod kernels;
pub mod pipeline;
pub mod shared_services;

pub use buffers::{
    AsyncFrameState, BufferCounts, BufferParity, FrameSlot, PersistentBufferSet, PipelineError,
    SlotState,
};
pub use kernels::{
    linear_group_count, BindGroupLayout, DispatchDomain, KernelDescriptor, VolumetricKernel,
    WorkgroupSize,
};
pub use pipeline::{
    extract, plan_frame, prepare, queue, PassMask, PlannedDispatch, VolumetricGpuExtract,
    VolumetricGpuFramePlan, VolumetricGpuPrepare, VolumetricGpuQueue,
};
pub use shared_services::{
    produced_services, shared_service_bindings, touched_services, ServiceAccess,
    SharedServiceBinding,
};
