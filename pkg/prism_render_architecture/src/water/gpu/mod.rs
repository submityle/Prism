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
//!   entry-point name for each water compute pass.
//! * [`buffers`] — the sizing of the device-resident buffers the solvers read
//!   and write in place across frames, and the [`buffers::AsyncFrameState`]
//!   double-buffer state machine that sequences record → submit → retire so a
//!   slot is never read while the `GPU` is still writing it.
//! * [`fft_plan`] — the deterministic ping-pong pass schedule for the
//!   separable inverse butterfly `FFT` that replaces the ocean spectrum's
//!   `O(N^4)` direct-sum reference with the `O(N^2 log N)` transform every
//!   shipping ocean (`WaveWorks`, `Crest`, `UE5` Water) uses.
//! * [`spectral_plan`] — the per-cascade evolve → butterfly → assemble stage
//!   schedule that drives the spectral ocean over the [`fft_plan`] passes,
//!   packing the eight real output fields into four complex transforms.
//!
//! Everything here is pure integer bookkeeping with no `GPU` handles, no floats
//! and no wall clock, so the whole `GPU` schedule is deterministic and
//! `CPU`-testable. The numerical passes themselves are authored in `WESL`
//! (`water_ocean`, `water_flip`, `water_pbf`, `water_render_fx`,
//! `water_surface`) and mirror the `CPU` golden reference byte-for-byte.

pub mod buffers;
pub mod coupling_readback_kernel;
pub mod dispersion_kernel;
pub mod fft_plan;
pub mod pipeline;
pub mod render_fx_kernel;
pub mod spectral_plan;
pub mod surface_bindings;
pub mod surface_mesh;
pub mod surface_pass;
pub mod swe_kernel;
pub mod underwater_kernel;
pub mod waterline_kernel;

pub use buffers::{
    AsyncFrameState, BufferParity, FrameSlot, PipelineError, SlotState, WaterBufferCounts,
    WaterPersistentBufferSet, DISPLACEMENT_TEXEL_STRIDE, FLIP_MAC_SCATTER_SLOTS,
    FLIP_PARTICLE_STRIDE, FOAM_CELL_STRIDE, FROXEL_STRIDE, GERSTNER_WAVE_STRIDE,
    GRID_SCALAR_STRIDE, NORMAL_TEXEL_STRIDE, PBF_PARTICLE_STRIDE, SPECTRUM_AMPLITUDE_STRIDE,
    SWE_CELL_STRIDE, WETNESS_CELL_STRIDE,
};

pub use fft_plan::{
    fft_pass_ping_pong, fft_result_buffer, inverse_fft2_pass_count, plan_inverse_fft2, FftAxis,
    FftEntry, FftPass, FftPassParams, FftPingPong,
};

pub use pipeline::{
    extract, plan_frame, prepare, queue, PlannedDispatch, WaterGpuExtract, WaterGpuFramePlan,
    WaterGpuPrepare, WaterGpuQueue, WaterPasses,
};

pub use render_fx_kernel::{
    dispatch_foam_advect, dispatch_wetness_step, FOAM_DRIVE_FLOATS, FOAM_DRIVE_STRIDE,
    WATER_RENDER_FX_WESL, WETNESS_DRIVE_FLOATS, WETNESS_DRIVE_STRIDE, WETNESS_OUT_FLOATS,
};

pub use swe_kernel::{
    dispatch_swe_step, SweStepOutput, SWE_VEL_FLOATS, SWE_VEL_STRIDE, WATER_SWE_WESL,
};

pub use waterline_kernel::{
    dispatch_waterline_mask, WATERLINE_OUT_FLOATS, WATERLINE_SCENE_FLOATS, WATER_WATERLINE_WESL,
};

pub use spectral_plan::{
    cascade_spectral_stage_count, field_slot, ocean_spectral_pass_count, plan_cascade_spectral,
    plan_ocean_spectral, ComplexPart, FieldSlot, SpectralPass, SpectralRealField, SpectralStage,
    SPECTRAL_COMPLEX_FIELD_COUNT, SPECTRAL_REAL_FIELD_COUNT,
};

pub use surface_bindings::{
    plan_surface_draw_call, SurfaceBinding, SurfaceBindingKind, SurfaceDrawCall, SurfaceGrid,
    SURFACE_VERTEX_RECORD_STRIDE,
};

pub use surface_mesh::{
    plan_surface_mesh_dispatch, surface_mesh_output_bytes, SurfaceMeshDispatch, SurfaceMeshOutput,
    SurfaceMeshSource, SURFACE_MESH_LANES,
};

pub use surface_pass::{
    plan_surface_draw, DepthTest, SurfaceBlend, SurfaceDepth, SurfaceDrawDescriptor,
    WaterRenderTarget,
};

pub use dispersion_kernel::{
    dispatch_dispersion_refract, DispersionRefractParams, DISPERSION_GBUFFER_FLOATS,
    DISPERSION_OUT_FLOATS, DISPERSION_SCENE_FLOATS, WATER_DISPERSION_REFRACT_WESL,
};

pub use underwater_kernel::{
    dispatch_underwater_volume, UnderwaterParams, UNDERWATER_FROXEL_FLOATS, UNDERWATER_OUT_FLOATS,
    WATER_UNDERWATER_VOLUME_WESL,
};

pub use coupling_readback_kernel::{
    dispatch_coupling_readback, CouplingReadbackParams, COUPLING_BODY_FLOATS,
    COUPLING_FORCE_FLOATS, WATER_COUPLING_READBACK_WESL,
};
