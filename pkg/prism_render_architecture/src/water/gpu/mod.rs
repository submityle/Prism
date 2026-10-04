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
pub mod caustics_kernel;
pub mod coupling_readback_kernel;
pub mod dispersion_kernel;
pub mod fft_bitrev_kernel;
pub mod fft_normalize_kernel;
pub mod fft_plan;
pub mod fft_stage_kernel;
pub mod flip_cell_histogram_kernel;
pub mod flip_cell_scan_kernel;
pub mod flip_cell_scatter_kernel;
pub mod flip_mac_divergence_kernel;
pub mod flip_mac_faces_normalize_kernel;
pub mod flip_mac_g2p_kernel;
pub mod flip_mac_p2g_kernel;
pub mod flip_mac_pressure_kernel;
pub mod flip_mac_project_kernel;
pub mod gerstner_displace_kernel;
pub mod pbf_density_kernel;
pub mod pipeline;
pub mod render_fx_kernel;
pub mod spectral_plan;
pub mod spectrum_assemble_kernel;
pub mod spectrum_evolve_kernel;
pub mod spectrum_ifft_kernel;
pub mod surface_bindings;
pub mod surface_mesh;
pub mod surface_mesh_kernel;
pub mod surface_pass;
pub mod surface_reconstruct_kernel;
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

pub use spectrum_assemble_kernel::{
    dispatch_spectrum_assemble, AssembleFields, AssembleParams, ASSEMBLE_MAX_CASCADES,
    ASSEMBLE_OUT_FLOATS, CASCADE_TEXEL_FLOATS, WATER_SPECTRUM_ASSEMBLE_WESL,
};

pub use spectrum_evolve_kernel::{
    dispatch_spectrum_evolve, EvolveParams, SpectrumEvolveBuffers, EVOLVE_COMPLEX_FLOATS,
    EVOLVE_PACKED_FIELDS, WATER_SPECTRUM_EVOLVE_WESL,
};

pub use spectrum_ifft_kernel::{
    dispatch_spectrum_ifft, IfftParams, SpectrumIfftFields, IFFT_TEXEL_FLOATS,
    WATER_SPECTRUM_IFFT_WESL,
};

pub use surface_bindings::{
    plan_surface_draw_call, SurfaceBinding, SurfaceBindingKind, SurfaceDrawCall, SurfaceGrid,
    SURFACE_VERTEX_RECORD_STRIDE,
};

pub use surface_mesh::{
    plan_surface_mesh_dispatch, surface_mesh_output_bytes, SurfaceMeshDispatch, SurfaceMeshOutput,
    SurfaceMeshSource, SURFACE_MESH_LANES,
};

pub use surface_mesh_kernel::{
    dispatch_surface_mesh, MeshParams, SurfaceMeshFields, SURFACE_MESH_VERTEX_FLOATS,
    SURFACE_TEXEL_FLOATS, WATER_SURFACE_MESH_WESL,
};

pub use surface_pass::{
    plan_surface_draw, DepthTest, SurfaceBlend, SurfaceDepth, SurfaceDrawDescriptor,
    WaterRenderTarget,
};

pub use surface_reconstruct_kernel::{
    dispatch_surface_reconstruct, ReconstructParams, RECONSTRUCT_OUT_FLOATS,
    WATER_SURFACE_RECONSTRUCT_WESL,
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

pub use caustics_kernel::{
    dispatch_caustics_project, CausticsParams, CAUSTICS_OUT_FLOATS, CAUSTICS_RECEIVER_FLOATS,
    CAUSTICS_SCENE_FLOATS, WATER_CAUSTICS_PROJECT_WESL,
};

pub use fft_bitrev_kernel::{
    dispatch_fft_bit_reverse, FftBitReverseParams, FFT_COMPLEX_FLOATS, WATER_FFT_BITREV_WESL,
};

pub use fft_normalize_kernel::{
    dispatch_fft_normalize, FftNormalizeParams, WATER_FFT_NORMALIZE_WESL,
};

pub use fft_stage_kernel::{dispatch_fft_stage, FftStageParams, WATER_FFT_STAGE_WESL};

pub use flip_cell_histogram_kernel::{
    dispatch_flip_cell_histogram, FlipHistogramParams, FLIP_HISTOGRAM_POSITION_FLOATS,
    WATER_FLIP_CELL_HISTOGRAM_WESL,
};

pub use flip_cell_scan_kernel::{
    dispatch_flip_cell_scan, FlipScanParams, WATER_FLIP_CELL_SCAN_WESL,
};

pub use flip_cell_scatter_kernel::{
    dispatch_flip_cell_scatter, FlipScatterParams, FLIP_SCATTER_POSITION_FLOATS,
    WATER_FLIP_CELL_SCATTER_WESL,
};

pub use flip_mac_divergence_kernel::{
    dispatch_mac_divergence, MacDivergenceParams, WATER_FLIP_MAC_DIVERGENCE_WESL,
};

pub use flip_mac_faces_normalize_kernel::{
    dispatch_mac_faces_normalize, FacesNormalizeParams, WATER_FLIP_MAC_FACES_NORMALIZE_WESL,
};

pub use flip_mac_g2p_kernel::{dispatch_mac_g2p, FlipG2PParams, WATER_FLIP_MAC_G2P_WESL};

pub use flip_mac_p2g_kernel::{
    dispatch_mac_p2g, FlipP2GParams, FLIP_P2G_PARTICLE_FLOATS, WATER_FLIP_MAC_P2G_WESL,
};

pub use flip_mac_pressure_kernel::{
    dispatch_mac_pressure, MacPressureParams, WATER_FLIP_MAC_PRESSURE_WESL,
};

pub use flip_mac_project_kernel::{
    dispatch_mac_project, MacProjectParams, WATER_FLIP_MAC_PROJECT_WESL,
};

pub use gerstner_displace_kernel::{
    dispatch_gerstner_displace, GerstnerFields, GerstnerParams, GERSTNER_OUT_FLOATS,
    GERSTNER_WAVE_FLOATS, WATER_GERSTNER_DISPLACE_WESL,
};

pub use pbf_density_kernel::{
    dispatch_pbf_compute_lambda, dispatch_pbf_density_solve, PbfDensityParams, PBF_POSITION_FLOATS,
    WATER_PBF_DENSITY_WESL,
};
