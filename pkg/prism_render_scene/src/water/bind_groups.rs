//! Per-body `GPU` buffer allocation and bind-group assembly for the water /
//! fluid subsystem.
//!
//! [`super::pipeline`] built the sixteen compute pipelines and the twelve
//! bind-group layouts once at startup; this slice turns one water body's
//! `CPU`-golden solver state into the resident device buffers, storage
//! textures, sampled textures and sampler those pipelines read and write, and
//! wires them into the twelve bind groups the dispatch slice records against.
//!
//! The resident buffer sizing mirrors the golden contract in
//! [`prism_render_architecture::water::gpu::buffers`]: the ocean spectrum `h0`
//! amplitude pair and analytic `Gerstner` wave train, the `FLIP`/`APIC`
//! particle pool and its `MAC` grid scalars, the `PBF` particle pool and its
//! spatial-hash, the Shallow-Water height/velocity grid, the semi-Lagrangian
//! foam field, the surface wetness field and the underwater froxel volume all
//! live in persistent storage across frames. Each buffered record is uploaded
//! with a plain `bytemuck` cast of the `#[repr(C)]` mirrors in [`super::abi`],
//! so the host bytes and the `WESL` `struct`s stay layout-identical (the
//! `size_of` contract tests in that module pin the strides).
//!
//! Unlike the buffers-only cloth subsystem, the water render passes also bind
//! storage textures (spectrum displacement/normal `rgba32float`, the `FLIP`
//! surface normal, the caustics `r32float` projection, the dispersion and
//! wetness `rgba16float` targets and the underwater `rgba16float` `3D`
//! volume) plus a handful of sampled textures and a filtering sampler for the
//! refraction lookups; those are created here as real, format-correct device
//! objects so every bind group is always valid even for an empty body.
//!
//! An empty optional array (no waves, no particles, no foam on a body) is
//! padded up to a single zeroed element and every texture extent clamps to at
//! least `1x1`(`x1`) so the binding is always valid; the matching per-pass
//! uniform count is zero, so the dispatch bounds check discards the placeholder
//! before it is ever read.

#![allow(
    dead_code,
    reason = "the per-body buffer set, its storage/sampled textures and its twelve bind groups are now built by the prepare author and recorded by the Core3d dispatch node; the only unread item left is the device-free `WaterBufferPlan` sizing view, which exists to pin the scene-side storage clamps against the golden `WaterPersistentBufferSet` in the contract tests below"
)]

use bevy_render::{
    render_resource::{
        AddressMode, BindGroup, BindGroupEntries, Buffer, BufferDescriptor, BufferInitDescriptor,
        BufferUsages, Extent3d, FilterMode, MipmapFilterMode, Sampler, SamplerDescriptor, Texture,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
        TextureViewDescriptor,
    },
    renderer::RenderDevice,
};
use bytemuck::Pod;
use prism_render_architecture::water::gpu::buffers::{
    WaterBufferCounts, WaterPersistentBufferSet, FLIP_SCATTER_SLOTS_PER_CELL, GRID_SCALAR_STRIDE,
    PBF_PARTICLE_STRIDE,
};
use prism_render_architecture::water::gpu::fft_pass_ping_pong;
use prism_render_architecture::water::gpu::pipeline::mac_face_count;

use super::abi::{
    GpuFlipParticle, GpuFlipSimParams, GpuFlipSurfaceParams, GpuGerstnerWave, GpuMacParams,
    GpuPbfParams, GpuSprayParams, GpuSprayParticle, GpuSpraySource, GpuSpraySpawnHeader,
    GpuWaterCausticsParams, GpuWaterCouplingParams, GpuWaterCouplingQuery,
    GpuWaterDispersionParams, GpuWaterFoamParams, GpuWaterGerstnerParams, GpuWaterSpectrumParams,
    GpuWaterSurfaceMeshParams, GpuWaterSweParams, GpuWaterUnderwaterParams,
    GpuWaterWaterlineParams, GpuWaterWetnessParams,
};
use super::pipeline::WaterComputePipelines;

/// Fixed-point atomic slots per staggered-`MAC` velocity face in the `P2G`
/// scatter accumulator: `[momentum, mass]` packed at `face_flat * 2 + {0, 1}`,
/// matching `water_flip_mac_p2g.wesl`. The scatter buffer therefore holds two
/// `u32` atomics per face.
pub(crate) const MAC_FACE_SCATTER_SLOTS: u32 = 2;

/// Byte size of the staggered-`MAC` `P2G` face scatter accumulator for a grid
/// with `mac_faces_total` velocity faces: [`MAC_FACE_SCATTER_SLOTS`]
/// (`[momentum, mass]`) fixed-point `u32` atomics per face, each
/// [`GRID_SCALAR_STRIDE`] bytes. Sizing it at one scalar per face (or per cell)
/// under-allocates it and lets `water_flip_mac_p2g.wesl` scatter past the
/// buffer's end at `face_scatter[face_flat * 2 + {0, 1}]`.
#[must_use]
pub(crate) fn mac_face_scatter_bytes(mac_faces_total: u32) -> u64 {
    u64::from(mac_faces_total) * u64::from(MAC_FACE_SCATTER_SLOTS) * u64::from(GRID_SCALAR_STRIDE)
}

/// Byte stride of one Shallow-Water working scalar (`h`, `u`, `v` are unpacked
/// into separate `array<f32>` working buffers by `water_surface.wesl`, distinct
/// from the packed `vec4<f32>` resident cell the sizing contract counts).
pub(crate) const SWE_SCALAR_STRIDE: u32 = 4;

/// Byte stride of one Shallow-Water source injection record, a `vec4<f32>`
/// (`xyz` momentum + `w` height source) matching `swe_sources` in the shader.
pub(crate) const SWE_SOURCE_STRIDE: u32 = 16;

/// Byte stride of one foam working scalar (`foam_in`/`foam_out`/velocity/source
/// are `array<f32>` in `water_surface.wesl`).
pub(crate) const FOAM_SCALAR_STRIDE: u32 = 4;

/// Byte stride of one waterline sample (`array<f32>` inputs) and its packed
/// `vec4<f32>` output row.
pub(crate) const WATERLINE_SCALAR_STRIDE: u32 = 4;
/// Byte stride of one packed waterline output row (`vec4<f32>`).
pub(crate) const WATERLINE_OUT_STRIDE: u32 = 16;

/// Byte stride of one wetness working cell (`array<vec2<f32>>` state in
/// `water_render_fx.wesl`, distinct from the single-`f32` resident cell the
/// sizing contract counts).
pub(crate) const WETNESS_STATE_STRIDE: u32 = 8;

/// Byte stride of one caustics photon-count bin (`array<u32>`).
pub(crate) const CAUSTICS_BIN_STRIDE: u32 = 4;

/// Byte stride of one coupling read-back row (`array<vec4<f32>>`).
pub(crate) const COUPLING_READBACK_STRIDE: u32 = 16;

/// Minimum storage-buffer size in bytes.
///
/// `wgpu` rejects a zero-sized storage buffer, so every resident buffer is
/// clamped to at least one 16-byte `std430` row even when its element count is
/// zero. The placeholder is never read: the owning pass's uniform count is zero
/// and its per-element bounds check discards the sole padded element.
const MIN_STORAGE_BYTES: u64 = 16;

/// A `2D` texture extent (in texels), clamped to at least `1x1` at creation.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct WaterSurfaceExtent {
    /// Width in texels.
    pub(crate) width: u32,
    /// Height in texels.
    pub(crate) height: u32,
}

/// A `3D` texture extent (in texels), clamped to at least `1x1x1` at creation.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct WaterVolumeExtent {
    /// Width in texels.
    pub(crate) width: u32,
    /// Height in texels.
    pub(crate) height: u32,
    /// Depth in texels (froxel slices).
    pub(crate) depth: u32,
}

/// The `CPU`-golden host state uploaded to build one water body's resident
/// resources.
///
/// Every buffered slice is a `#[repr(C)]` mirror ready for a `bytemuck` cast;
/// the packing from the architecture-layer solver types into these mirrors is
/// the extract slice's job, so this factory stays a pure "host state → device
/// resources" step that can be reasoned about without the golden solvers in
/// scope. Element counts that the persistent-buffer contract does not size
/// (spray sources, waterline samples, texture extents) are passed explicitly.
pub(crate) struct WaterBodyUpload<'a> {
    // -- Ocean: spectrum IFFT + analytic Gerstner --------------------------
    /// Initial `Tessendorf` spectrum amplitudes `h0` (`array<vec2<f32>>`).
    pub(crate) spectrum_h0: &'a [[f32; 2]],
    /// Conjugate spectrum `h0(-k)` (`array<vec2<f32>>`).
    pub(crate) spectrum_h0_neg: &'a [[f32; 2]],
    /// Per-frame spectrum scalars (the ocean group's direct-sum reference).
    pub(crate) spectrum_params: GpuWaterSpectrumParams,
    /// Per-cascade spectral uniforms, one per stacked atlas tile; the
    /// production `spectrum_fft` groups bind these in cascade order.
    pub(crate) cascade_params: &'a [GpuWaterSpectrumParams],
    /// Analytic `Gerstner` wave trains; empty when the body is spectrum-only.
    pub(crate) gerstner_waves: &'a [GpuGerstnerWave],
    /// Per-frame `Gerstner` scalars.
    pub(crate) gerstner_params: GpuWaterGerstnerParams,
    /// Displacement / normal storage-texture extent (`N x N` cascade atlas).
    pub(crate) ocean_extent: WaterSurfaceExtent,

    // -- Volume: FLIP / APIC ------------------------------------------------
    /// `FLIP`/`APIC` particle pool; empty when the body carries no volume sim.
    pub(crate) flip_particles: &'a [GpuFlipParticle],
    /// Number of `MAC` grid scalar cells (sizes the scatter/pressure buffers).
    pub(crate) flip_grid_cells: u32,
    /// `FLIP` per-substep uniform.
    pub(crate) flip_params: GpuFlipSimParams,
    /// `FLIP` surface-reconstruction uniform.
    pub(crate) flip_surface_params: GpuFlipSurfaceParams,
    /// Screen-space surface reconstruction extent (`surface_normal_tex`).
    pub(crate) flip_surface_extent: WaterSurfaceExtent,

    // -- PBF ---------------------------------------------------------------
    /// `PBF` particle positions (`array<vec4<f32>>`); ping-ponged in-solve.
    pub(crate) pbf_positions: &'a [[f32; 4]],
    /// Number of `PBF` spatial-hash entries (one `u32` per hash slot).
    pub(crate) pbf_hash_entries: u32,
    /// `PBF` density-solve uniform.
    pub(crate) pbf_params: GpuPbfParams,

    // -- Spray -------------------------------------------------------------
    /// Spray emitter sources; empty when the body sheds no spray.
    pub(crate) spray_sources: &'a [GpuSpraySource],
    /// Spray-pool particle capacity (sizes the spawn append buffer).
    pub(crate) spray_capacity: u32,
    /// Spray-emit uniform.
    pub(crate) spray_params: GpuSprayParams,

    // -- Shallow Water -----------------------------------------------------
    /// Number of Shallow-Water grid cells (height / velocity field).
    pub(crate) swe_cells: u32,
    /// Shallow-Water source injections (`array<vec4<f32>>`).
    pub(crate) swe_sources: &'a [[f32; 4]],
    /// Shallow-Water step uniform.
    pub(crate) swe_params: GpuWaterSweParams,

    // -- Foam --------------------------------------------------------------
    /// Number of foam coverage cells (semi-Lagrangian, double-buffered).
    pub(crate) foam_cells: u32,
    /// Reactive foam source field (`array<f32>`).
    pub(crate) foam_sources: &'a [f32],
    /// Foam advection uniform.
    pub(crate) foam_params: GpuWaterFoamParams,

    // -- Waterline ---------------------------------------------------------
    /// Number of waterline samples (sizes the sample / output arrays).
    pub(crate) waterline_samples: u32,
    /// Waterline mask uniform.
    pub(crate) waterline_params: GpuWaterWaterlineParams,

    // -- Caustics ----------------------------------------------------------
    /// Photon-count bins projected onto the caustics grid (`array<u32>`).
    pub(crate) caustics_photon_counts: &'a [u32],
    /// Caustics projection uniform.
    pub(crate) caustics_params: GpuWaterCausticsParams,
    /// Caustics `r32float` projection-target extent.
    pub(crate) caustics_extent: WaterSurfaceExtent,
    /// Sampled photon-offset texture extent.
    pub(crate) caustics_offset_extent: WaterSurfaceExtent,

    // -- Dispersion --------------------------------------------------------
    /// Dispersion refraction uniform.
    pub(crate) dispersion_params: GpuWaterDispersionParams,
    /// Dispersion `rgba16float` output + sampled scene / normal extent.
    pub(crate) dispersion_extent: WaterSurfaceExtent,

    // -- Underwater volume -------------------------------------------------
    /// Underwater single-scatter uniform.
    pub(crate) underwater_params: GpuWaterUnderwaterParams,
    /// Underwater `rgba16float` `3D` froxel-volume extent.
    pub(crate) underwater_extent: WaterVolumeExtent,
    /// Sampled surface-light texture extent.
    pub(crate) underwater_light_extent: WaterSurfaceExtent,

    // -- Wetness -----------------------------------------------------------
    /// Number of surface wetness cells (`array<vec2<f32>>` state).
    pub(crate) wetness_cells: u32,
    /// Wetness step uniform.
    pub(crate) wetness_params: GpuWaterWetnessParams,
    /// Wetness `rgba16float` output extent.
    pub(crate) wetness_extent: WaterSurfaceExtent,

    // -- Coupling ----------------------------------------------------------
    /// Two-way coupling buoyancy queries (`array<WaterCouplingQuery>`).
    pub(crate) coupling_queries: &'a [GpuWaterCouplingQuery],
    /// Number of coupling read-back rows (`array<vec4<f32>>`).
    pub(crate) coupling_readback_rows: u32,
    /// Coupling read-back uniform.
    pub(crate) coupling_params: GpuWaterCouplingParams,
    /// Number of displaced surface-mesh vertices the `water_surface_mesh`
    /// kernel scatters (and the raster draw later reads).
    pub(crate) surface_vertex_count: u32,
    /// Surface-mesh grid / patch uniform.
    pub(crate) surface_mesh_params: GpuWaterSurfaceMeshParams,
}

/// Every resident device buffer, storage / sampled texture view and sampler
/// backing one water body, held so the bind groups (and the async read-back)
/// keep them alive for as long as the body is resident.
pub(crate) struct WaterBodyGpuBuffers {
    // Ocean buffers.
    pub(crate) spectrum_h0: Buffer,
    pub(crate) spectrum_h0_neg: Buffer,
    /// The ocean group's single direct-sum reference spectral uniform.
    pub(crate) spectrum_params: Buffer,
    /// One spectral uniform per cascade, in cascade order; the `spectrum_fft`
    /// groups bind these so each inverse `FFT` transforms its own atlas tile.
    pub(crate) cascade_params: Vec<Buffer>,
    pub(crate) gerstner_waves: Buffer,
    pub(crate) gerstner_params: Buffer,
    // Ocean storage textures (`rgba32float`, write).
    pub(crate) spectrum_displacement: TextureView,
    pub(crate) spectrum_normal: TextureView,
    pub(crate) gerstner_displacement: TextureView,
    pub(crate) gerstner_normal: TextureView,
    // FLIP buffers + surface texture.
    pub(crate) flip_particles: Buffer,
    pub(crate) flip_grid_scatter: Buffer,
    pub(crate) flip_pressure_in: Buffer,
    pub(crate) flip_pressure_out: Buffer,
    pub(crate) flip_params: Buffer,
    pub(crate) flip_surface_depth: Buffer,
    pub(crate) flip_surface_thickness: Buffer,
    pub(crate) flip_surface_normal: TextureView,
    pub(crate) flip_surface_params: Buffer,
    // Face-centered staggered `MAC` projection buffers (shared `FLIP`/`APIC`
    // grid): the packed `[u | v | w]` faces, the `P2G` momentum/mass atomic
    // scatter, a pre-projection snapshot for the `G2P` `FLIP` delta, and the
    // compact single-sided divergence / `Jacobi` pressure ping-pong.
    pub(crate) mac_faces: Buffer,
    pub(crate) mac_face_scatter: Buffer,
    pub(crate) mac_faces_preproj: Buffer,
    pub(crate) mac_divergence: Buffer,
    pub(crate) mac_pressure_in: Buffer,
    pub(crate) mac_pressure_out: Buffer,
    pub(crate) mac_params: Buffer,
    // PBF buffers.
    pub(crate) pbf_positions_in: Buffer,
    pub(crate) pbf_positions_out: Buffer,
    pub(crate) pbf_hash: Buffer,
    pub(crate) pbf_params: Buffer,
    // Spray buffers.
    pub(crate) spray_sources: Buffer,
    pub(crate) spray_spawn: Buffer,
    pub(crate) spray_params: Buffer,
    // Shallow-Water buffers.
    pub(crate) swe_h_in: Buffer,
    pub(crate) swe_u_in: Buffer,
    pub(crate) swe_v_in: Buffer,
    pub(crate) swe_h_out: Buffer,
    pub(crate) swe_u_out: Buffer,
    pub(crate) swe_v_out: Buffer,
    pub(crate) swe_sources: Buffer,
    pub(crate) swe_params: Buffer,
    // Foam buffers.
    pub(crate) foam_in: Buffer,
    pub(crate) foam_u: Buffer,
    pub(crate) foam_v: Buffer,
    pub(crate) foam_sources: Buffer,
    pub(crate) foam_out: Buffer,
    pub(crate) foam_params: Buffer,
    // Waterline buffers.
    pub(crate) waterline_sample_y: Buffer,
    pub(crate) waterline_surface_y: Buffer,
    pub(crate) waterline_depth: Buffer,
    pub(crate) waterline_out: Buffer,
    pub(crate) waterline_params: Buffer,
    // Caustics.
    pub(crate) caustics_photon_counts: Buffer,
    pub(crate) caustics_params: Buffer,
    pub(crate) caustics_out: TextureView,
    pub(crate) caustics_offsets: TextureView,
    // Dispersion.
    pub(crate) dispersion_params: Buffer,
    pub(crate) dispersion_out: TextureView,
    pub(crate) dispersion_scene: TextureView,
    pub(crate) dispersion_normal: TextureView,
    pub(crate) dispersion_sampler: Sampler,
    // Underwater.
    pub(crate) underwater_params: Buffer,
    pub(crate) underwater_out: TextureView,
    pub(crate) underwater_surface_light: TextureView,
    /// Froxel-volume depth-mapping uniform the surface fragment stage binds
    /// at `@group(5)` to composite the underwater single-scatter volume into
    /// its refraction (see [`super::surface_froxel`]).
    pub(crate) surface_froxel_params: Buffer,
    // Wetness.
    pub(crate) wetness_state: Buffer,
    pub(crate) wetness_params: Buffer,
    pub(crate) wetness_out: TextureView,
    // Spectral `FFT` (`Tessendorf`) ping-pong pools.
    pub(crate) packed_g0: Buffer,
    pub(crate) packed_g1: Buffer,
    pub(crate) packed_g2: Buffer,
    pub(crate) packed_g3: Buffer,
    pub(crate) scratch_g0: Buffer,
    pub(crate) scratch_g1: Buffer,
    pub(crate) scratch_g2: Buffer,
    pub(crate) scratch_g3: Buffer,
    /// One small uniform per scheduled inverse-`FFT` pass, in dispatch
    /// order; the butterfly bind groups bind the matching payload per pass.
    pub(crate) fft_pass_params: Vec<Buffer>,
    /// The ocean grid edge `N`, forwarded so the dispatch node can re-derive
    /// the per-pass dispatch dimensions and kernels from the golden plan.
    pub(crate) ocean_n: u32,
    // Coupling.
    pub(crate) coupling_queries: Buffer,
    pub(crate) coupling_readback: Buffer,
    pub(crate) coupling_params: Buffer,
    // Surface mesh (slice2c): the per-vertex uniform, the four write-only
    // scatter pools the raster draw reads, and the non-filtering sampler for
    // the assembled displacement / normal cascade lookups.
    pub(crate) surface_mesh_params: Buffer,
    pub(crate) surface_mesh_base_positions: Buffer,
    pub(crate) surface_mesh_surface_uvs: Buffer,
    pub(crate) surface_mesh_displacement: Buffer,
    // Previous-frame displacement, copied from `surface_mesh_displacement`
    // before each frame's assembly overwrites it, so the raster vertex stage
    // can emit wave self-motion into the motion G-buffer.
    pub(crate) surface_mesh_displacement_prev: Buffer,
    pub(crate) surface_mesh_normal_foam: Buffer,
    pub(crate) surface_mesh_sampler: Sampler,
}

impl WaterBodyGpuBuffers {
    /// Allocates and uploads every resident resource for one water body.
    ///
    /// Particle and grid pools that the async read-back may copy back for a
    /// `CPU` fallback carry `COPY_SRC`; the read-only inputs are device-local
    /// (`STORAGE | COPY_DST`). Storage textures are format-matched to the
    /// `WESL` bindings and every extent clamps to at least one texel so the
    /// bind group is always valid, even for an empty body.
    pub(crate) fn create(device: &RenderDevice, upload: &WaterBodyUpload<'_>) -> Self {
        // ---- Ocean ----
        let spectrum_h0 = read_only_storage(device, "prism water spectrum h0", upload.spectrum_h0);
        let spectrum_h0_neg = read_only_storage(
            device,
            "prism water spectrum h0 neg",
            upload.spectrum_h0_neg,
        );
        let spectrum_params = uniform(
            device,
            "prism water spectrum params",
            &upload.spectrum_params,
        );
        let gerstner_waves =
            read_only_storage(device, "prism water gerstner waves", upload.gerstner_waves);
        let gerstner_params = uniform(
            device,
            "prism water gerstner params",
            &upload.gerstner_params,
        );
        // Written by the ocean kernels as storage and additionally sampled by
        // the surface-mesh kernel, so these two carry `TEXTURE_BINDING`.
        let spectrum_displacement = sampled_storage_texture_2d(
            device,
            "prism water spectrum displacement",
            upload.ocean_extent,
            TextureFormat::Rgba32Float,
        );
        let spectrum_normal = sampled_storage_texture_2d(
            device,
            "prism water spectrum normal",
            upload.ocean_extent,
            TextureFormat::Rgba32Float,
        );
        let gerstner_displacement = storage_texture_2d(
            device,
            "prism water gerstner displacement",
            upload.ocean_extent,
            TextureFormat::Rgba32Float,
        );
        let gerstner_normal = storage_texture_2d(
            device,
            "prism water gerstner normal",
            upload.ocean_extent,
            TextureFormat::Rgba32Float,
        );

        // ---- Spectral FFT (Tessendorf) ping-pong pools ----
        // One complex `[f32; 2]` (eight bytes) per spectrum texel; the butterfly
        // passes ping-pong between the packed and scratch grids per cascade.
        // These grids hold one `N*N` tile that the spectral pass reuses across
        // every cascade, so they are sized from the atlas edge `N` (not the
        // concatenated `M*N*N` amplitude pool, which would over-allocate `M`x).
        let cascade_bytes =
            (upload.ocean_extent.width as u64) * (upload.ocean_extent.width as u64) * 8;
        let packed_g0 = zeroed_storage(device, "prism water packed g0", cascade_bytes);
        let packed_g1 = zeroed_storage(device, "prism water packed g1", cascade_bytes);
        let packed_g2 = zeroed_storage(device, "prism water packed g2", cascade_bytes);
        let packed_g3 = zeroed_storage(device, "prism water packed g3", cascade_bytes);
        let scratch_g0 = zeroed_storage(device, "prism water scratch g0", cascade_bytes);
        let scratch_g1 = zeroed_storage(device, "prism water scratch g1", cascade_bytes);
        let scratch_g2 = zeroed_storage(device, "prism water scratch g2", cascade_bytes);
        let scratch_g3 = zeroed_storage(device, "prism water scratch g3", cascade_bytes);
        let ocean_n = upload.ocean_extent.width;
        // One small uniform per scheduled inverse-`FFT` pass, in dispatch order,
        // so the recorder can re-bind the correct pass payload before every
        // butterfly launch. A non-power-of-two edge plans no passes and yields
        // an empty vector (a deterministic no-op, matching the arch contract).
        let fft_pass_params: Vec<Buffer> = super::fft_upload::fft_pass_uniforms_for(ocean_n)
            .iter()
            .map(|params| uniform(device, "prism water fft pass params", params))
            .collect();
        // One spectral uniform buffer per cascade, in cascade order; the
        // `spectrum_fft` groups bind these so each inverse `FFT` transforms its
        // own atlas tile via the cascade's `h0_offset` / `tile_origin_y`.
        let cascade_params: Vec<Buffer> = upload
            .cascade_params
            .iter()
            .map(|p| uniform(device, "prism water spectrum cascade params", p))
            .collect();

        // ---- FLIP / APIC ----
        let flip_particles =
            readable_storage(device, "prism water flip particles", upload.flip_particles);
        // Pressure/divergence hold one f32 per cell; the P2G scatter accumulator
        // holds FLIP_SCATTER_SLOTS_PER_CELL fixed-point atomics per cell
        // (momentum_x, momentum_y, momentum_z, mass), so it is four times larger.
        // Sizing the scatter buffer at `grid_bytes` under-allocates it four-fold
        // and lets the `water_flip` P2G kernel scatter past the buffer's end.
        let grid_bytes = u64::from(upload.flip_grid_cells) * u64::from(GRID_SCALAR_STRIDE);
        let scatter_bytes = grid_bytes * u64::from(FLIP_SCATTER_SLOTS_PER_CELL);
        let flip_grid_scatter = zeroed_storage(device, "prism water flip scatter", scatter_bytes);
        let flip_pressure_in = zeroed_storage(device, "prism water flip pressure in", grid_bytes);
        let flip_pressure_out = zeroed_storage(device, "prism water flip pressure out", grid_bytes);
        let flip_params = uniform(device, "prism water flip params", &upload.flip_params);
        let surface_scalar_bytes = u64::from(upload.flip_surface_extent.width)
            * u64::from(upload.flip_surface_extent.height)
            * u64::from(GRID_SCALAR_STRIDE);
        let flip_surface_depth = zeroed_storage(
            device,
            "prism water flip surface depth",
            surface_scalar_bytes,
        );
        let flip_surface_thickness = zeroed_storage(
            device,
            "prism water flip surface thickness",
            surface_scalar_bytes,
        );
        let flip_surface_normal = storage_texture_2d(
            device,
            "prism water flip surface normal",
            upload.flip_surface_extent,
            TextureFormat::Rgba16Float,
        );
        let flip_surface_params = uniform(
            device,
            "prism water flip surface params",
            &upload.flip_surface_params,
        );

        // ---- Face-centered staggered MAC projection ----
        // The staggered grid stores one velocity component per cell face, so the
        // face total differs from the cell total: `mac_face_count` sums the
        // `(nx+1)·ny·nz` u-faces, `nx·(ny+1)·nz` v-faces and `nx·ny·(nz+1)`
        // w-faces. The faces buffer and its pre-projection snapshot hold one
        // `f32` per face; the `P2G` scatter holds `MAC_FACE_SCATTER_SLOTS`
        // (`[momentum, mass]`) atomics per face. Divergence and the pressure
        // ping-pong hold one `f32` per cell, exactly like the collocated path.
        let [mac_nx, mac_ny, mac_nz, _] = upload.flip_params.dim;
        let mac_faces_total = mac_face_count(mac_nx, mac_ny, mac_nz);
        let mac_face_bytes = u64::from(mac_faces_total) * u64::from(GRID_SCALAR_STRIDE);
        let mac_scatter_bytes = mac_face_scatter_bytes(mac_faces_total);
        let mac_faces = zeroed_storage(device, "prism water mac faces", mac_face_bytes);
        let mac_face_scatter =
            zeroed_storage(device, "prism water mac face scatter", mac_scatter_bytes);
        let mac_faces_preproj =
            zeroed_storage(device, "prism water mac faces preproj", mac_face_bytes);
        let mac_divergence = zeroed_storage(device, "prism water mac divergence", grid_bytes);
        let mac_pressure_in = zeroed_storage(device, "prism water mac pressure in", grid_bytes);
        let mac_pressure_out = zeroed_storage(device, "prism water mac pressure out", grid_bytes);
        let mac_params = uniform(
            device,
            "prism water mac params",
            &GpuMacParams::from_flip(&upload.flip_params),
        );

        // ---- PBF ----
        let pbf_positions_in =
            readable_storage(device, "prism water pbf positions in", upload.pbf_positions);
        let pbf_positions_out = zeroed_storage(
            device,
            "prism water pbf positions out",
            (upload.pbf_positions.len() as u64) * u64::from(PBF_PARTICLE_STRIDE),
        );
        let pbf_hash = zeroed_storage(
            device,
            "prism water pbf hash",
            u64::from(upload.pbf_hash_entries) * u64::from(GRID_SCALAR_STRIDE),
        );
        let pbf_params = uniform(device, "prism water pbf params", &upload.pbf_params);

        // ---- Spray ----
        let spray_sources =
            read_only_storage(device, "prism water spray sources", upload.spray_sources);
        let spray_spawn = zeroed_storage(
            device,
            "prism water spray spawn",
            size_of::<GpuSpraySpawnHeader>() as u64
                + u64::from(upload.spray_capacity) * size_of::<GpuSprayParticle>() as u64,
        );
        let spray_params = uniform(device, "prism water spray params", &upload.spray_params);

        // ---- Shallow Water ----
        let swe_bytes = u64::from(upload.swe_cells) * u64::from(SWE_SCALAR_STRIDE);
        let swe_h_in = zeroed_storage(device, "prism water swe h in", swe_bytes);
        let swe_u_in = zeroed_storage(device, "prism water swe u in", swe_bytes);
        let swe_v_in = zeroed_storage(device, "prism water swe v in", swe_bytes);
        let swe_h_out = zeroed_storage(device, "prism water swe h out", swe_bytes);
        let swe_u_out = zeroed_storage(device, "prism water swe u out", swe_bytes);
        let swe_v_out = zeroed_storage(device, "prism water swe v out", swe_bytes);
        let swe_sources = read_only_storage(device, "prism water swe sources", upload.swe_sources);
        let swe_params = uniform(device, "prism water swe params", &upload.swe_params);

        // ---- Foam ----
        let foam_bytes = u64::from(upload.foam_cells) * u64::from(FOAM_SCALAR_STRIDE);
        let foam_in = zeroed_storage(device, "prism water foam in", foam_bytes);
        let foam_u = zeroed_storage(device, "prism water foam u", foam_bytes);
        let foam_v = zeroed_storage(device, "prism water foam v", foam_bytes);
        let foam_sources =
            read_only_storage(device, "prism water foam sources", upload.foam_sources);
        let foam_out = zeroed_storage(device, "prism water foam out", foam_bytes);
        let foam_params = uniform(device, "prism water foam params", &upload.foam_params);

        // ---- Waterline ----
        let waterline_scalar_bytes =
            u64::from(upload.waterline_samples) * u64::from(WATERLINE_SCALAR_STRIDE);
        let waterline_sample_y = zeroed_storage(
            device,
            "prism water waterline sample y",
            waterline_scalar_bytes,
        );
        let waterline_surface_y = zeroed_storage(
            device,
            "prism water waterline surface y",
            waterline_scalar_bytes,
        );
        let waterline_depth = zeroed_storage(
            device,
            "prism water waterline depth",
            waterline_scalar_bytes,
        );
        let waterline_out = zeroed_storage(
            device,
            "prism water waterline out",
            u64::from(upload.waterline_samples) * u64::from(WATERLINE_OUT_STRIDE),
        );
        let waterline_params = uniform(
            device,
            "prism water waterline params",
            &upload.waterline_params,
        );

        // ---- Caustics ----
        let caustics_photon_counts = read_only_storage(
            device,
            "prism water caustics photon counts",
            upload.caustics_photon_counts,
        );
        let caustics_params = uniform(
            device,
            "prism water caustics params",
            &upload.caustics_params,
        );
        let caustics_out = sampled_storage_texture_2d(
            device,
            "prism water caustics out",
            upload.caustics_extent,
            TextureFormat::R32Float,
        );
        let caustics_offsets = sampled_texture_2d(
            device,
            "prism water caustics offsets",
            upload.caustics_offset_extent,
            TextureFormat::Rgba16Float,
        );

        // ---- Dispersion ----
        let dispersion_params = uniform(
            device,
            "prism water dispersion params",
            &upload.dispersion_params,
        );
        let dispersion_out = storage_texture_2d(
            device,
            "prism water dispersion out",
            upload.dispersion_extent,
            TextureFormat::Rgba16Float,
        );
        let dispersion_scene = sampled_texture_2d(
            device,
            "prism water dispersion scene",
            upload.dispersion_extent,
            TextureFormat::Rgba16Float,
        );
        let dispersion_normal = sampled_texture_2d(
            device,
            "prism water dispersion normal",
            upload.dispersion_extent,
            TextureFormat::Rgba16Float,
        );
        let dispersion_sampler = filtering_sampler(device, "prism water dispersion sampler");

        // ---- Underwater volume ----
        let underwater_params = uniform(
            device,
            "prism water underwater params",
            &upload.underwater_params,
        );
        let underwater_out = storage_texture_3d(
            device,
            "prism water underwater out",
            upload.underwater_extent,
            TextureFormat::Rgba16Float,
        );
        let underwater_surface_light = sampled_texture_2d(
            device,
            "prism water underwater surface light",
            upload.underwater_light_extent,
            TextureFormat::Rgba16Float,
        );
        // Precompute the froxel depth-mapping the surface fragment stage reads
        // to sample the above volume: `1 / (slice_thickness * depth_slices)`.
        let surface_froxel_params = uniform(
            device,
            "prism water surface froxel params",
            &crate::water::surface_froxel::GpuWaterFroxelParams::new(
                upload.underwater_params.slice_thickness,
                upload.underwater_extent.depth,
                true,
            ),
        );

        // ---- Wetness ----
        let wetness_state = zeroed_storage(
            device,
            "prism water wetness state",
            u64::from(upload.wetness_cells) * u64::from(WETNESS_STATE_STRIDE),
        );
        let wetness_params = uniform(device, "prism water wetness params", &upload.wetness_params);
        let wetness_out = storage_texture_2d(
            device,
            "prism water wetness out",
            upload.wetness_extent,
            TextureFormat::Rgba16Float,
        );

        // ---- Coupling ----
        let coupling_queries = read_only_storage(
            device,
            "prism water coupling queries",
            upload.coupling_queries,
        );
        let coupling_readback = zeroed_storage(
            device,
            "prism water coupling readback",
            u64::from(upload.coupling_readback_rows) * u64::from(COUPLING_READBACK_STRIDE),
        );
        let coupling_params = uniform(
            device,
            "prism water coupling params",
            &upload.coupling_params,
        );

        // ---- Surface mesh (slice2c) ----
        // The displaced render mesh the `water_surface_mesh` kernel scatters
        // into four write-only `array<vec4<f32>>` pools (base positions,
        // surface `UV`s, displacement and normal/foam), later read by the raster
        // draw. Each pool is sized to the body's vertex count (clamped to one
        // vertex so an empty body still binds a valid buffer); sixteen bytes per
        // `vec4<f32>` vertex.
        let surface_mesh_vertex_bytes = (upload.surface_vertex_count.max(1) as u64) * 16;
        let surface_mesh_params = uniform(
            device,
            "prism water surface mesh params",
            &upload.surface_mesh_params,
        );
        let surface_mesh_base_positions = zeroed_storage(
            device,
            "prism water surface mesh base positions",
            surface_mesh_vertex_bytes,
        );
        let surface_mesh_surface_uvs = zeroed_storage(
            device,
            "prism water surface mesh surface uvs",
            surface_mesh_vertex_bytes,
        );
        let surface_mesh_displacement = zeroed_storage(
            device,
            "prism water surface mesh displacement",
            surface_mesh_vertex_bytes,
        );
        let surface_mesh_displacement_prev = zeroed_storage(
            device,
            "prism water surface mesh displacement prev",
            surface_mesh_vertex_bytes,
        );
        let surface_mesh_normal_foam = zeroed_storage(
            device,
            "prism water surface mesh normal foam",
            surface_mesh_vertex_bytes,
        );
        let surface_mesh_sampler = nonfiltering_sampler(device, "prism water surface mesh sampler");

        Self {
            spectrum_h0,
            spectrum_h0_neg,
            spectrum_params,
            gerstner_waves,
            gerstner_params,
            spectrum_displacement,
            spectrum_normal,
            gerstner_displacement,
            gerstner_normal,
            flip_particles,
            flip_grid_scatter,
            flip_pressure_in,
            flip_pressure_out,
            flip_params,
            flip_surface_depth,
            flip_surface_thickness,
            flip_surface_normal,
            flip_surface_params,
            mac_faces,
            mac_face_scatter,
            mac_faces_preproj,
            mac_divergence,
            mac_pressure_in,
            mac_pressure_out,
            mac_params,
            pbf_positions_in,
            pbf_positions_out,
            pbf_hash,
            pbf_params,
            spray_sources,
            spray_spawn,
            spray_params,
            swe_h_in,
            swe_u_in,
            swe_v_in,
            swe_h_out,
            swe_u_out,
            swe_v_out,
            swe_sources,
            swe_params,
            foam_in,
            foam_u,
            foam_v,
            foam_sources,
            foam_out,
            foam_params,
            waterline_sample_y,
            waterline_surface_y,
            waterline_depth,
            waterline_out,
            waterline_params,
            caustics_photon_counts,
            caustics_params,
            caustics_out,
            caustics_offsets,
            dispersion_params,
            dispersion_out,
            dispersion_scene,
            dispersion_normal,
            dispersion_sampler,
            underwater_params,
            underwater_out,
            underwater_surface_light,
            surface_froxel_params,
            wetness_state,
            wetness_params,
            wetness_out,
            packed_g0,
            packed_g1,
            packed_g2,
            packed_g3,
            scratch_g0,
            scratch_g1,
            scratch_g2,
            scratch_g3,
            fft_pass_params,
            cascade_params,
            ocean_n,
            coupling_queries,
            coupling_readback,
            coupling_params,
            surface_mesh_params,
            surface_mesh_base_positions,
            surface_mesh_surface_uvs,
            surface_mesh_displacement,
            surface_mesh_displacement_prev,
            surface_mesh_normal_foam,
            surface_mesh_sampler,
        }
    }
}

/// The twelve bind groups one water body dispatches against, one per layout in
/// [`WaterComputePipelines`].
///
/// Present only once its backing [`WaterBodyGpuBuffers`] exists; the dispatch
/// node treats a present set as "safe to record". The kernel-to-group mapping
/// (`ocean`/`swe`/`pbf`/`spray`/`flip` at `@group(0)`, the render-effect passes
/// at `@group(1..=4)`) is owned by [`super::pipeline::wesl_group`].
pub(crate) struct WaterBodyBindGroups {
    /// `@group(0)` for `spectrum_ifft` + `gerstner_displace`.
    pub(crate) ocean: BindGroup,
    /// `@group(0)` for the three collocated `FLIP` solve passes.
    pub(crate) flip: BindGroup,
    /// `@group(0)` for the standalone `water_surface_reconstruct` pass (the
    /// splatted depth + thickness buffers, the reconstructed-normal storage
    /// texture, and the filter/projection uniform).
    pub(crate) surface_reconstruct: BindGroup,
    /// `@group(0)` for `pbf_density_solve`.
    pub(crate) pbf: BindGroup,
    /// `@group(0)` for `spray_emit`.
    pub(crate) spray: BindGroup,
    /// `@group(0)` for `swe_step`.
    pub(crate) swe: BindGroup,
    /// `@group(1)` for `foam_advect`.
    pub(crate) foam: BindGroup,
    /// `@group(2)` for `waterline_mask`.
    pub(crate) waterline: BindGroup,
    /// `@group(0)` for `caustics_project`.
    pub(crate) caustics: BindGroup,
    /// `@group(1)` for `dispersion_refract`.
    pub(crate) dispersion: BindGroup,
    /// `@group(2)` for `underwater_volume`.
    pub(crate) underwater: BindGroup,
    /// `@group(3)` for `wetness_step`.
    pub(crate) wetness: BindGroup,
    /// `@group(4)` for `coupling_readback`.
    pub(crate) coupling: BindGroup,
    /// `@group(0)` for `water_flip_mac_p2g` + `water_flip_mac_faces_normalize`
    /// (particles, the momentum/mass atomic scatter, the packed faces and the
    /// shared `FLIP` uniform).
    pub(crate) mac_p2g: BindGroup,
    /// `@group(0)` for the three `water_flip_mac.wesl` solve passes
    /// (`mac_divergence` + `mac_pressure` + `mac_project`): faces, divergence,
    /// the pressure ping-pong and the `MAC` uniform.
    pub(crate) mac_solve: BindGroup,
    /// `@group(0)` for `water_flip_mac_g2p` (particles, the projected faces, the
    /// pre-projection snapshot and the shared `FLIP` uniform).
    pub(crate) mac_g2p: BindGroup,
    /// `@group(0)` for `spectrum_evolve` + `spectrum_assemble` (thirteen
    /// bindings), one bind group per cascade in cascade order. The recorder
    /// binds `spectrum_fft[c]` so each inverse `FFT` transforms atlas tile `c`.
    pub(crate) spectrum_fft: Vec<BindGroup>,
    /// `@group(0)` butterfly `FFT` bind groups, one per `(pass, complex grid)`.
    ///
    /// The outer index is the pass ordinal in [`plan_inverse_fft2`] order; the
    /// inner four are the packed<->scratch ping-pong bindings for the packed
    /// complex grids `g0..g3`. Each binds `(src, dst, fft_pass_params[pass])`
    /// with `src`/`dst` chosen by the golden `fft_pass_ping_pong` parity, so the
    /// dispatch node walks the matrix without re-deriving the routing.
    ///
    /// [`plan_inverse_fft2`]: prism_render_architecture::water::gpu::plan_inverse_fft2
    pub(crate) butterfly_passes: Vec<[BindGroup; 4]>,
    /// `@group(0)` for `water_surface_mesh` (the surface-mesh uniform, the
    /// assembled displacement / normal cascade textures, their non-filtering
    /// sampler, and the four write-only per-vertex scatter pools).
    pub(crate) surface_mesh: BindGroup,
    /// The ocean grid edge `N`, so the dispatch node re-derives per-pass
    /// dispatch dimensions and kernels from the golden inverse-`FFT` plan.
    pub(crate) ocean_n: u32,
}

impl WaterBodyBindGroups {
    /// Builds the twelve bind groups binding `buffers` against the shared
    /// pipeline layouts, in the exact `@binding` order declared by each `WESL`
    /// entry point.
    pub(crate) fn create(
        device: &RenderDevice,
        pipelines: &WaterComputePipelines,
        buffers: &WaterBodyGpuBuffers,
    ) -> Self {
        let ocean = device.create_bind_group(
            "prism water ocean",
            &pipelines.ocean_layout,
            &BindGroupEntries::sequential((
                buffers.spectrum_h0.as_entire_binding(),
                buffers.spectrum_h0_neg.as_entire_binding(),
                buffers.spectrum_params.as_entire_binding(),
                &buffers.spectrum_displacement,
                &buffers.spectrum_normal,
                buffers.gerstner_waves.as_entire_binding(),
                buffers.gerstner_params.as_entire_binding(),
                &buffers.gerstner_displacement,
                &buffers.gerstner_normal,
            )),
        );
        let flip = device.create_bind_group(
            "prism water flip",
            &pipelines.flip_layout,
            &BindGroupEntries::sequential((
                buffers.flip_particles.as_entire_binding(),
                buffers.flip_grid_scatter.as_entire_binding(),
                buffers.flip_pressure_in.as_entire_binding(),
                buffers.flip_pressure_out.as_entire_binding(),
                buffers.flip_params.as_entire_binding(),
                buffers.flip_surface_depth.as_entire_binding(),
                buffers.flip_surface_thickness.as_entire_binding(),
                &buffers.flip_surface_normal,
                buffers.flip_surface_params.as_entire_binding(),
            )),
        );
        let surface_reconstruct = device.create_bind_group(
            "prism water surface reconstruct",
            &pipelines.surface_reconstruct_layout,
            &BindGroupEntries::sequential((
                buffers.flip_surface_depth.as_entire_binding(),
                buffers.flip_surface_thickness.as_entire_binding(),
                &buffers.flip_surface_normal,
                buffers.flip_surface_params.as_entire_binding(),
            )),
        );
        let pbf = device.create_bind_group(
            "prism water pbf",
            &pipelines.pbf_layout,
            &BindGroupEntries::sequential((
                buffers.pbf_positions_in.as_entire_binding(),
                buffers.pbf_positions_out.as_entire_binding(),
                buffers.pbf_hash.as_entire_binding(),
                buffers.pbf_params.as_entire_binding(),
            )),
        );
        let spray = device.create_bind_group(
            "prism water spray",
            &pipelines.spray_layout,
            &BindGroupEntries::sequential((
                buffers.spray_sources.as_entire_binding(),
                buffers.spray_spawn.as_entire_binding(),
                buffers.spray_params.as_entire_binding(),
            )),
        );
        let swe = device.create_bind_group(
            "prism water swe",
            &pipelines.swe_layout,
            &BindGroupEntries::sequential((
                buffers.swe_h_in.as_entire_binding(),
                buffers.swe_u_in.as_entire_binding(),
                buffers.swe_v_in.as_entire_binding(),
                buffers.swe_h_out.as_entire_binding(),
                buffers.swe_u_out.as_entire_binding(),
                buffers.swe_v_out.as_entire_binding(),
                buffers.swe_sources.as_entire_binding(),
                buffers.swe_params.as_entire_binding(),
            )),
        );
        let foam = device.create_bind_group(
            "prism water foam",
            &pipelines.foam_layout,
            &BindGroupEntries::sequential((
                buffers.foam_in.as_entire_binding(),
                buffers.foam_u.as_entire_binding(),
                buffers.foam_v.as_entire_binding(),
                buffers.foam_sources.as_entire_binding(),
                buffers.foam_out.as_entire_binding(),
                buffers.foam_params.as_entire_binding(),
            )),
        );
        let waterline = device.create_bind_group(
            "prism water waterline",
            &pipelines.waterline_layout,
            &BindGroupEntries::sequential((
                buffers.waterline_sample_y.as_entire_binding(),
                buffers.waterline_surface_y.as_entire_binding(),
                buffers.waterline_depth.as_entire_binding(),
                buffers.waterline_out.as_entire_binding(),
                buffers.waterline_params.as_entire_binding(),
            )),
        );
        let caustics = device.create_bind_group(
            "prism water caustics",
            &pipelines.caustics_layout,
            &BindGroupEntries::sequential((
                buffers.caustics_photon_counts.as_entire_binding(),
                buffers.caustics_params.as_entire_binding(),
                &buffers.caustics_out,
                &buffers.caustics_offsets,
            )),
        );
        let dispersion = device.create_bind_group(
            "prism water dispersion",
            &pipelines.dispersion_layout,
            &BindGroupEntries::sequential((
                buffers.dispersion_params.as_entire_binding(),
                &buffers.dispersion_out,
                &buffers.dispersion_scene,
                &buffers.dispersion_normal,
                &buffers.dispersion_sampler,
            )),
        );
        let underwater = device.create_bind_group(
            "prism water underwater",
            &pipelines.underwater_layout,
            &BindGroupEntries::sequential((
                buffers.underwater_params.as_entire_binding(),
                &buffers.underwater_out,
                &buffers.underwater_surface_light,
            )),
        );
        let wetness = device.create_bind_group(
            "prism water wetness",
            &pipelines.wetness_layout,
            &BindGroupEntries::sequential((
                buffers.wetness_state.as_entire_binding(),
                buffers.wetness_params.as_entire_binding(),
                &buffers.wetness_out,
            )),
        );
        let coupling = device.create_bind_group(
            "prism water coupling",
            &pipelines.coupling_layout,
            &BindGroupEntries::sequential((
                buffers.coupling_queries.as_entire_binding(),
                buffers.coupling_readback.as_entire_binding(),
                buffers.coupling_params.as_entire_binding(),
            )),
        );
        // Face-centered staggered `MAC` projection groups. `P2G` scatters the
        // particle momentum/mass into the atomic accumulator and normalizes into
        // the packed faces; the solve measures compact divergence, relaxes the
        // `Jacobi` pressure ping-pong and subtracts the compact gradient in
        // place; `G2P` gathers the projected faces back, differencing against the
        // pre-projection snapshot for the `FLIP` delta. All three reuse the live
        // `FLIP`/`APIC` particle pool and uniform so the staggered chain shares
        // one authored grid with the collocated path it supersedes.
        let mac_p2g = device.create_bind_group(
            "prism water mac p2g",
            &pipelines.mac_p2g_layout,
            &BindGroupEntries::sequential((
                buffers.flip_particles.as_entire_binding(),
                buffers.mac_face_scatter.as_entire_binding(),
                buffers.mac_faces.as_entire_binding(),
                buffers.flip_params.as_entire_binding(),
            )),
        );
        let mac_solve = device.create_bind_group(
            "prism water mac solve",
            &pipelines.mac_solve_layout,
            &BindGroupEntries::sequential((
                buffers.mac_faces.as_entire_binding(),
                buffers.mac_divergence.as_entire_binding(),
                buffers.mac_pressure_in.as_entire_binding(),
                buffers.mac_pressure_out.as_entire_binding(),
                buffers.mac_params.as_entire_binding(),
            )),
        );
        let mac_g2p = device.create_bind_group(
            "prism water mac g2p",
            &pipelines.mac_g2p_layout,
            &BindGroupEntries::sequential((
                buffers.flip_particles.as_entire_binding(),
                buffers.mac_faces.as_entire_binding(),
                buffers.mac_faces_preproj.as_entire_binding(),
                buffers.flip_params.as_entire_binding(),
            )),
        );
        // One spectral bind group per cascade: the concatenated `h0` pools and
        // shared ping-pong grids stay bound, and `@binding(2)` selects the
        // cascade's own spectral uniform so the evolve reads its `h0_offset`
        // slice and the assemble writes its `tile_origin_y` atlas tile.
        let spectrum_fft: Vec<BindGroup> = buffers
            .cascade_params
            .iter()
            .map(|cascade_params| {
                device.create_bind_group(
                    "prism water spectrum fft",
                    &pipelines.spectrum_fft_layout,
                    &BindGroupEntries::sequential((
                        buffers.spectrum_h0.as_entire_binding(),
                        buffers.spectrum_h0_neg.as_entire_binding(),
                        cascade_params.as_entire_binding(),
                        buffers.packed_g0.as_entire_binding(),
                        buffers.packed_g1.as_entire_binding(),
                        buffers.packed_g2.as_entire_binding(),
                        buffers.packed_g3.as_entire_binding(),
                        buffers.scratch_g0.as_entire_binding(),
                        buffers.scratch_g1.as_entire_binding(),
                        buffers.scratch_g2.as_entire_binding(),
                        buffers.scratch_g3.as_entire_binding(),
                        &buffers.spectrum_displacement,
                        &buffers.spectrum_normal,
                    )),
                )
            })
            .collect();
        // Per-pass, per-complex-grid butterfly bind groups. Each of the four
        // packed complex grids `g0..g3` keeps its own packed<->scratch ping-pong
        // pair; the golden `fft_pass_ping_pong` routing picks which buffer is
        // read and which is written for every pass (pass 0 reads the packed seed
        // and writes scratch, and every subsequent pass flips). One bind group
        // per `(pass, grid)` so the dispatch recorder only walks the matrix.
        let packed = [
            &buffers.packed_g0,
            &buffers.packed_g1,
            &buffers.packed_g2,
            &buffers.packed_g3,
        ];
        let scratch = [
            &buffers.scratch_g0,
            &buffers.scratch_g1,
            &buffers.scratch_g2,
            &buffers.scratch_g3,
        ];
        let butterfly_passes: Vec<[BindGroup; 4]> = buffers
            .fft_pass_params
            .iter()
            .enumerate()
            .map(|(ordinal, pass_params)| {
                let route = fft_pass_ping_pong(ordinal);
                core::array::from_fn(|grid| {
                    let pair = [packed[grid], scratch[grid]];
                    let src = pair[route.src as usize];
                    let dst = pair[route.dst as usize];
                    device.create_bind_group(
                        "prism water butterfly pass",
                        &pipelines.butterfly_layout,
                        &BindGroupEntries::sequential((
                            src.as_entire_binding(),
                            dst.as_entire_binding(),
                            pass_params.as_entire_binding(),
                        )),
                    )
                })
            })
            .collect();
        let surface_mesh = device.create_bind_group(
            "prism water surface mesh",
            &pipelines.surface_mesh_layout,
            &BindGroupEntries::sequential((
                buffers.surface_mesh_params.as_entire_binding(),
                &buffers.spectrum_displacement,
                &buffers.spectrum_normal,
                &buffers.surface_mesh_sampler,
                buffers.surface_mesh_base_positions.as_entire_binding(),
                buffers.surface_mesh_surface_uvs.as_entire_binding(),
                buffers.surface_mesh_displacement.as_entire_binding(),
                buffers.surface_mesh_normal_foam.as_entire_binding(),
            )),
        );
        Self {
            ocean,
            flip,
            surface_reconstruct,
            pbf,
            spray,
            swe,
            foam,
            waterline,
            caustics,
            dispersion,
            underwater,
            wetness,
            coupling,
            mac_p2g,
            mac_solve,
            mac_g2p,
            spectrum_fft,
            butterfly_passes,
            surface_mesh,
            ocean_n: buffers.ocean_n,
        }
    }
}

/// The device-free per-body resident-buffer sizing view.
///
/// It reproduces the golden byte sizes from
/// [`WaterPersistentBufferSet`] (clamped up to the storage floor exactly as the
/// runtime padding does), so the render allocation can be reasoned about and
/// unit-tested without a `GPU` in scope.
pub(crate) struct WaterBufferPlan {
    set: WaterPersistentBufferSet,
}

impl WaterBufferPlan {
    /// Builds the sizing view from the golden element counts.
    #[must_use]
    pub(crate) fn new(counts: WaterBufferCounts) -> Self {
        Self {
            set: WaterPersistentBufferSet::new(counts),
        }
    }

    /// Bytes for one spectrum `h0` amplitude array (the conjugate array is a
    /// second buffer of the same size).
    #[must_use]
    pub(crate) fn spectrum_amplitude_bytes(&self) -> u64 {
        clamp_storage(self.set.spectrum_amplitude_bytes())
    }

    /// Bytes for the transformed displacement storage texture.
    #[must_use]
    pub(crate) fn displacement_bytes(&self) -> u64 {
        clamp_storage(self.set.displacement_bytes())
    }

    /// Bytes for the surface normal / foldover storage texture.
    #[must_use]
    pub(crate) fn normal_bytes(&self) -> u64 {
        clamp_storage(self.set.normal_bytes())
    }

    /// Bytes for the analytic `Gerstner` wave-train buffer.
    #[must_use]
    pub(crate) fn gerstner_bytes(&self) -> u64 {
        clamp_storage(self.set.gerstner_bytes())
    }

    /// Bytes for the Shallow-Water height / velocity grid.
    #[must_use]
    pub(crate) fn swe_bytes(&self) -> u64 {
        clamp_storage(self.set.swe_bytes())
    }

    /// Bytes for one `PBF` particle-position pool (double-buffered).
    #[must_use]
    pub(crate) fn pbf_particle_bytes(&self) -> u64 {
        clamp_storage(self.set.pbf_particle_bytes())
    }

    /// Bytes for the `PBF` spatial-hash entry buffer.
    #[must_use]
    pub(crate) fn pbf_hash_bytes(&self) -> u64 {
        clamp_storage(self.set.pbf_hash_bytes())
    }

    /// Bytes for the `FLIP`/`APIC` particle pool.
    #[must_use]
    pub(crate) fn flip_particle_bytes(&self) -> u64 {
        clamp_storage(self.set.flip_particle_bytes())
    }

    /// Bytes for one `FLIP` `MAC` grid scalar buffer (double-buffered).
    #[must_use]
    pub(crate) fn flip_grid_bytes(&self) -> u64 {
        clamp_storage(self.set.flip_grid_bytes())
    }

    /// Bytes for the single-buffered `FLIP` `P2G` scatter accumulator (four
    /// fixed-point atomics per cell: `momentum_x`/`y`/`z` + `mass`).
    #[must_use]
    pub(crate) fn flip_scatter_bytes(&self) -> u64 {
        clamp_storage(self.set.flip_scatter_bytes())
    }

    /// Bytes for one foam coverage field (double-buffered).
    #[must_use]
    pub(crate) fn foam_bytes(&self) -> u64 {
        clamp_storage(self.set.foam_bytes())
    }

    /// Bytes for the surface wetness field.
    #[must_use]
    pub(crate) fn wetness_bytes(&self) -> u64 {
        clamp_storage(self.set.wetness_bytes())
    }

    /// Bytes for the underwater single-scatter froxel volume.
    #[must_use]
    pub(crate) fn froxel_bytes(&self) -> u64 {
        clamp_storage(self.set.froxel_bytes())
    }

    /// Total resident bytes for every persistent water buffer (double-buffered
    /// pools counted twice), matching the golden set.
    #[must_use]
    pub(crate) fn total_bytes(&self) -> u64 {
        u64::from(self.set.total_bytes())
    }

    /// Whether at least one solver has resident state, i.e. a dispatch can do
    /// real work.
    #[must_use]
    pub(crate) fn is_simulatable(&self) -> bool {
        self.set.is_simulatable()
    }
}

/// Clamps a golden byte size up to the storage-buffer floor, matching the
/// runtime padding [`zeroed_storage`] and [`storage_with_data`] apply.
#[must_use]
fn clamp_storage(bytes: u32) -> u64 {
    u64::from(bytes).max(MIN_STORAGE_BYTES)
}

/// Uploads a read-write storage buffer that is also copyable back to the host
/// (`STORAGE | COPY_DST | COPY_SRC`), padding an empty slice to one zeroed
/// element so the binding is valid.
fn readable_storage<T: Pod>(device: &RenderDevice, label: &str, data: &[T]) -> Buffer {
    storage_with_data(
        device,
        label,
        data,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    )
}

/// Uploads a device-local read-only storage buffer (`STORAGE | COPY_DST`),
/// padding an empty slice to one zeroed element so the binding is valid.
fn read_only_storage<T: Pod>(device: &RenderDevice, label: &str, data: &[T]) -> Buffer {
    storage_with_data(
        device,
        label,
        data,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    )
}

/// Uploads `data` (or one zeroed element when empty) into a storage buffer with
/// the given usage.
fn storage_with_data<T: Pod>(
    device: &RenderDevice,
    label: &str,
    data: &[T],
    usage: BufferUsages,
) -> Buffer {
    let placeholder = [T::zeroed()];
    let contents: &[T] = if data.is_empty() { &placeholder } else { data };
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(contents),
        usage,
    })
}

/// Allocates a zeroed device-local read-write storage buffer of `byte_len`
/// bytes (clamped to at least one `std430` row), for the pass-produced pools.
fn zeroed_storage(device: &RenderDevice, label: &str, byte_len: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: byte_len.max(MIN_STORAGE_BYTES),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Uploads a `#[repr(C)]` uniform record into a `UNIFORM | COPY_DST` buffer.
fn uniform<T: Pod>(device: &RenderDevice, label: &str, value: &T) -> Buffer {
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    })
}

/// Creates a `2D` storage texture (`STORAGE_BINDING | COPY_SRC`) and returns its
/// default view, clamping the extent to at least one texel.
fn storage_texture_2d(
    device: &RenderDevice,
    label: &str,
    extent: WaterSurfaceExtent,
    format: TextureFormat,
) -> TextureView {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: extent.width.max(1),
            height: extent.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    default_view(&texture)
}

/// Creates a `2D` storage texture that is also sampleable
/// (`STORAGE_BINDING | COPY_SRC | TEXTURE_BINDING`) and returns its default
/// view, clamping the extent to at least one texel. Used for the assembled
/// displacement / normal cascade, which the ocean kernels write as storage and
/// the surface-mesh kernel samples.
fn sampled_storage_texture_2d(
    device: &RenderDevice,
    label: &str,
    extent: WaterSurfaceExtent,
    format: TextureFormat,
) -> TextureView {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: extent.width.max(1),
            height: extent.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_SRC
            | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    default_view(&texture)
}

/// Creates a `3D` storage texture (`STORAGE_BINDING | COPY_SRC`) and returns its
/// default view, clamping the extent to at least one texel per axis.
fn storage_texture_3d(
    device: &RenderDevice,
    label: &str,
    extent: WaterVolumeExtent,
    format: TextureFormat,
) -> TextureView {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: extent.width.max(1),
            height: extent.height.max(1),
            depth_or_array_layers: extent.depth.max(1),
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format,
        usage: TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_SRC
            | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    default_view(&texture)
}

/// Creates a `2D` sampled texture (`TEXTURE_BINDING | COPY_DST`) and returns its
/// default view, clamping the extent to at least one texel.
fn sampled_texture_2d(
    device: &RenderDevice,
    label: &str,
    extent: WaterSurfaceExtent,
    format: TextureFormat,
) -> TextureView {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: extent.width.max(1),
            height: extent.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    default_view(&texture)
}

/// The default full-resource view of `texture`.
fn default_view(texture: &Texture) -> TextureView {
    texture.create_view(&TextureViewDescriptor::default())
}

/// A trilinear filtering sampler for the refraction scene lookups.
pub(crate) fn filtering_sampler(device: &RenderDevice, label: &str) -> Sampler {
    device.create_sampler(&SamplerDescriptor {
        label: Some(label),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Linear,
        ..Default::default()
    })
}

/// A non-filtering (nearest) sampler for the surface-mesh kernel's
/// `Rgba32Float` displacement / normal lookups, which are not filterable
/// without the `float32-filterable` feature.
fn nonfiltering_sampler(device: &RenderDevice, label: &str) -> Sampler {
    device.create_sampler(&SamplerDescriptor {
        label: Some(label),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: MipmapFilterMode::Nearest,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::water::gpu::buffers::{
        DISPLACEMENT_TEXEL_STRIDE, FLIP_PARTICLE_STRIDE, FLIP_SCATTER_SLOTS_PER_CELL,
        FOAM_CELL_STRIDE, FROXEL_STRIDE, GERSTNER_WAVE_STRIDE, NORMAL_TEXEL_STRIDE,
        SPECTRUM_AMPLITUDE_STRIDE, SWE_CELL_STRIDE, WETNESS_CELL_STRIDE,
    };

    fn counts() -> WaterBufferCounts {
        WaterBufferCounts {
            spectrum_texels: 256,
            gerstner_waves: 32,
            swe_cells: 4096,
            pbf_particles: 2048,
            pbf_hash_entries: 4096,
            flip_particles: 8192,
            flip_grid_cells: 32768,
            foam_cells: 4096,
            wetness_cells: 1024,
            froxels: 65536,
        }
    }

    /// A populated plan reproduces the golden per-buffer byte sizes so the
    /// render allocation can never drift from the architecture-layer contract.
    #[test]
    fn plan_reproduces_golden_byte_sizes() {
        let plan = WaterBufferPlan::new(counts());
        let golden = WaterPersistentBufferSet::new(counts());
        assert_eq!(
            plan.spectrum_amplitude_bytes(),
            u64::from(golden.spectrum_amplitude_bytes())
        );
        assert_eq!(
            plan.displacement_bytes(),
            u64::from(golden.displacement_bytes())
        );
        assert_eq!(plan.normal_bytes(), u64::from(golden.normal_bytes()));
        assert_eq!(plan.gerstner_bytes(), u64::from(golden.gerstner_bytes()));
        assert_eq!(plan.swe_bytes(), u64::from(golden.swe_bytes()));
        assert_eq!(
            plan.pbf_particle_bytes(),
            u64::from(golden.pbf_particle_bytes())
        );
        assert_eq!(plan.pbf_hash_bytes(), u64::from(golden.pbf_hash_bytes()));
        assert_eq!(
            plan.flip_particle_bytes(),
            u64::from(golden.flip_particle_bytes())
        );
        assert_eq!(plan.flip_grid_bytes(), u64::from(golden.flip_grid_bytes()));
        assert_eq!(
            plan.flip_scatter_bytes(),
            u64::from(golden.flip_scatter_bytes())
        );
        assert_eq!(plan.foam_bytes(), u64::from(golden.foam_bytes()));
        assert_eq!(plan.wetness_bytes(), u64::from(golden.wetness_bytes()));
        assert_eq!(plan.froxel_bytes(), u64::from(golden.froxel_bytes()));
        assert_eq!(plan.total_bytes(), u64::from(golden.total_bytes()));
    }

    /// The `FLIP` `P2G` scatter accumulator must be four times the single
    /// pressure/divergence scalar buffer: the `water_flip` kernel scatters into
    /// `grid_scatter[cell * 4 + {0..3}]` (`momentum_x`/`y`/`z` + `mass`), so a
    /// buffer sized at one scalar per cell would be read and written past its
    /// end. This pins the four-slot sizing so a regression back to `grid_bytes`
    /// fails the build.
    #[test]
    fn flip_scatter_buffer_holds_four_atomics_per_cell() {
        let plan = WaterBufferPlan::new(counts());
        assert_eq!(plan.flip_scatter_bytes(), plan.flip_grid_bytes() * 4);
        // Enough u32 slots for the kernel's largest write index + 1.
        let slots_needed =
            u64::from(counts().flip_grid_cells) * u64::from(FLIP_SCATTER_SLOTS_PER_CELL);
        assert!(plan.flip_scatter_bytes() >= slots_needed * u64::from(GRID_SCALAR_STRIDE));
    }

    /// The staggered-`MAC` `P2G` face scatter accumulator must hold
    /// [`MAC_FACE_SCATTER_SLOTS`] (`[momentum, mass]`) atomics per velocity
    /// face, not per cell: the `water_flip_mac_p2g` kernel scatters into
    /// `face_scatter[face_flat * 2 + {0, 1}]` over the `[u | v | w]` face
    /// blocks, so sizing it at one scalar per face (or at the smaller cell
    /// total) is read and written past its end. This pins the two-slot face
    /// sizing against the kernel's largest write index + 1 on a non-cubic grid
    /// so a per-axis miscount or a regression to one slot fails the build.
    #[test]
    fn mac_face_scatter_buffer_holds_two_atomics_per_face() {
        assert_eq!(MAC_FACE_SCATTER_SLOTS, 2);
        // Distinct nx/ny/nz so the u/v/w face families have different counts
        // and a dropped axis in the sizing would change the total.
        let (nx, ny, nz) = (6u32, 5u32, 4u32);
        let faces = mac_face_count(nx, ny, nz);
        let bytes = mac_face_scatter_bytes(faces);
        // Two u32 atomics (`[momentum, mass]`) per face.
        assert_eq!(
            bytes,
            u64::from(faces) * u64::from(MAC_FACE_SCATTER_SLOTS) * u64::from(GRID_SCALAR_STRIDE)
        );
        // The face total exceeds the cell total, so a cell-sized buffer would
        // under-allocate even before the two-slot factor.
        assert!(faces > nx * ny * nz);
        // The kernel's largest write is `face_scatter[(faces - 1) * 2 + 1]`, so
        // the buffer must hold at least `faces * 2` u32 slots.
        let max_index_plus_one = u64::from(faces) * u64::from(MAC_FACE_SCATTER_SLOTS);
        assert!(bytes >= max_index_plus_one * u64::from(GRID_SCALAR_STRIDE));
    }

    /// An empty body clamps every resident buffer to the storage floor so no
    /// binding is ever zero-sized, mirroring the runtime placeholder padding.
    #[test]
    fn empty_plan_clamps_to_storage_floor() {
        let plan = WaterBufferPlan::new(WaterBufferCounts::default());
        assert_eq!(plan.spectrum_amplitude_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.displacement_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.normal_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.gerstner_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.swe_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.pbf_particle_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.pbf_hash_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.flip_particle_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.flip_grid_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.flip_scatter_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.foam_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.wetness_bytes(), MIN_STORAGE_BYTES);
        assert_eq!(plan.froxel_bytes(), MIN_STORAGE_BYTES);
        assert!(!plan.is_simulatable());
    }

    /// A body with any resident solver state reports itself simulatable, so the
    /// dispatch node records it rather than skipping it.
    #[test]
    fn populated_plan_is_simulatable() {
        assert!(WaterBufferPlan::new(counts()).is_simulatable());
    }

    /// The golden strides the resident buffers derive from; pin them so a
    /// golden drift is caught here rather than at dispatch time.
    #[test]
    fn golden_strides_are_stable() {
        assert_eq!(SPECTRUM_AMPLITUDE_STRIDE, 8);
        assert_eq!(DISPLACEMENT_TEXEL_STRIDE, 16);
        assert_eq!(NORMAL_TEXEL_STRIDE, 16);
        assert_eq!(GERSTNER_WAVE_STRIDE, 32);
        assert_eq!(SWE_CELL_STRIDE, 16);
        assert_eq!(PBF_PARTICLE_STRIDE, 16);
        assert_eq!(GRID_SCALAR_STRIDE, 4);
        assert_eq!(FLIP_PARTICLE_STRIDE, 80);
        assert_eq!(FOAM_CELL_STRIDE, 4);
        assert_eq!(WETNESS_CELL_STRIDE, 4);
        assert_eq!(FROXEL_STRIDE, 16);
    }

    /// The local working strides used by the device-side `create` path stay in
    /// step with the shader bindings (`f32` scalars, `vec2`/`vec4` rows).
    #[test]
    fn working_strides_match_shader_bindings() {
        assert_eq!(SWE_SCALAR_STRIDE, 4);
        assert_eq!(SWE_SOURCE_STRIDE, 16);
        assert_eq!(FOAM_SCALAR_STRIDE, 4);
        assert_eq!(WATERLINE_SCALAR_STRIDE, 4);
        assert_eq!(WATERLINE_OUT_STRIDE, 16);
        assert_eq!(WETNESS_STATE_STRIDE, 8);
        assert_eq!(CAUSTICS_BIN_STRIDE, 4);
        assert_eq!(COUPLING_READBACK_STRIDE, 16);
    }
}
