//! `GPU`-driven water solve: dispatch contract, persistent buffers, and (in a
//! follow-up slice) the async frame pipeline.
//!
//! The `CPU` water modules ([`super::spectrum`], [`super::swe`], [`super::pbf`],
//! [`super::flip`], [`super::foam`], [`super::reconstruct`], …) own the golden
//! reference numerics and every per-frame routing decision. This module owns
//! the contract for running those same solvers as a persistent, `GPU`-driven
//! pipeline:
//!
//! * [`super::kernels`] — the compute-kernel dispatch contract
//!   ([`super::kernels::WaterKernel`] and its per-kernel descriptor):
//!   bind-group shape, workgroup tiling, dispatch domain and stable `WESL`
//!   entry-point name for each of the sixteen water passes.
//! * [`buffers`] — the sizing of the device-resident buffers the solvers read
//!   and write in place across frames, and the [`buffers::AsyncFrameState`]
//!   double-buffer state machine that sequences record → submit → retire so a
//!   slot is never read while the `GPU` is still writing it.
//!
//! Everything here is pure integer bookkeeping with no `GPU` handles, no floats
//! and no wall clock, so the whole `GPU` schedule is deterministic and
//! `CPU`-testable. The numerical passes themselves are authored in `WESL`
//! (`water_ocean`, `water_flip`, `water_pbf`, `water_render_fx`,
//! `water_surface`) and mirror the `CPU` golden reference byte-for-byte.

pub mod buffers;
pub mod pipeline;

pub use buffers::{
    AsyncFrameState, BufferParity, FrameSlot, PipelineError, SlotState, WaterBufferCounts,
    WaterPersistentBufferSet, DISPLACEMENT_TEXEL_STRIDE, FLIP_PARTICLE_STRIDE, FOAM_CELL_STRIDE,
    FROXEL_STRIDE, GERSTNER_WAVE_STRIDE, GRID_SCALAR_STRIDE, NORMAL_TEXEL_STRIDE,
    PBF_PARTICLE_STRIDE, SPECTRUM_AMPLITUDE_STRIDE, SWE_CELL_STRIDE, WETNESS_CELL_STRIDE,
};

pub use pipeline::{
    extract, plan_frame, prepare, queue, PlannedDispatch, WaterGpuExtract, WaterGpuFramePlan,
    WaterGpuPrepare, WaterGpuQueue, WaterPasses,
};
