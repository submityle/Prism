//! Compute pipelines and bind-group layouts for the `GPU` water subsystem.
//!
//! The `CPU`-golden solvers in `prism_render_architecture::water` decide *what*
//! runs each frame; this slice builds the concrete `wgpu` compute pipelines and
//! the bind-group layouts that the sixteen water kernels dispatch against. The
//! kernels are authored across five `WESL` shaders, each declaring one or more
//! `@group(N)` resource interfaces:
//!
//! * `shaders/water_ocean.wesl` — the spectral inverse-`FFT` (`Tessendorf`) and
//!   the analytic `Gerstner` superposition. Both entry points share one
//!   nine-binding `@group(0)`: two read-only spectrum pools, the spectrum
//!   uniform, two write-only displacement/normal storage textures, the read-only
//!   `Gerstner` wave train, the `Gerstner` uniform, and two more write-only
//!   displacement/normal textures.
//! * `shaders/water_flip.wesl` — the three `FLIP`/`APIC` passes (`P2G` scatter,
//!   pressure projection, `G2P` gather) and the surface reconstruction. All four
//!   share one nine-binding `@group(0)`: the read-write particle pool and grid
//!   scatter buffer, the read-only / read-write pressure ping-pong, the sim
//!   uniform, the read-only surface depth/thickness buffers, the write-only
//!   surface-normal texture, and the surface uniform.
//! * `shaders/water_pbf.wesl` — the `PBF` density solve and the crest-spray
//!   emitter. These two rebind `@group(0)` to *different* resource sets (a
//!   four-binding density interface versus a three-binding spray interface), so
//!   they need separate layouts even though they share one shader file.
//! * `shaders/water_surface.wesl` — the `SWE` step (`@group(0)`, eight
//!   bindings), the foam advection (`@group(1)`, six bindings) and the waterline
//!   mask (`@group(2)`, five bindings). Each pass owns its own group index so
//!   `naga` never sees two incompatible declarations of one group.
//! * `shaders/water_render_fx.wesl` — the caustics projection (`@group(0)`), the
//!   spectral dispersion refract (`@group(1)`), the underwater volume
//!   (`@group(2)`), the wetness step (`@group(3)`) and the coupling readback
//!   (`@group(4)`), each on its own group index.
//!
//! Because several passes alias the *same* group index to incompatible resource
//! sets (the `water_pbf.wesl` density and spray passes both bind `@group(0)`),
//! each distinct interface needs its own bind-group layout. That is why this
//! module owns *twelve* layouts, one per distinct `@group(N)` interface, not one
//! per shader file.
//!
//! Every pipeline binds the layout that matches the group index its shader
//! declares. Passes on `@group(1..=4)` reserve empty placeholder layouts for the
//! lower, unused group slots so the `wgpu` pipeline layout stays contiguous; the
//! dispatch slice only binds the single real group each kernel reads. The water
//! shaders carry no `var<immediate>` push constants, so every pipeline declares
//! `immediate_size == 0`. The pipeline handles are keyed by [`WaterKernel`] so
//! the dispatch slice can look one up directly from the golden kernel schedule.

#![allow(
    dead_code,
    reason = "the water compute pipelines and the twelve bind-group layouts are the render-resource foundation of the GPU water subsystem; the bind-group preparation and Core3d dispatch slices that consume `WaterComputePipelines`, its accessors, `wesl_group` and `init_water_compute_pipelines` land in the following slices, and the layout grouping is exercised now by the kernel-contract test below"
)]

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{
            sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d,
            texture_storage_2d, texture_storage_3d, uniform_buffer_sized,
        },
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        SamplerBindingType, ShaderStages, StorageTextureAccess, TextureFormat, TextureSampleType,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;
use prism_render_architecture::water::kernels::WaterKernel;

/// The compute pipelines and bind-group layouts for every water kernel.
///
/// Inserted at `RenderStartup` by [`init_water_compute_pipelines`]. The sixteen
/// pipeline handles are queued into the [`PipelineCache`] and resolve
/// asynchronously; the dispatch slice skips a pass whose handle is not yet ready
/// rather than stalling the frame. The twelve layouts are created eagerly so the
/// bind-group slice can allocate against them the moment a body uploads.
#[derive(Resource)]
pub(crate) struct WaterComputePipelines {
    /// `@group(0)` for both `water_ocean.wesl` entry points (nine bindings).
    pub(crate) ocean_layout: BindGroupLayout,
    /// `@group(0)` for the four `water_flip.wesl` passes (nine bindings).
    pub(crate) flip_layout: BindGroupLayout,
    /// `@group(0)` for the `water_pbf_density_solve` pass (four bindings).
    pub(crate) pbf_layout: BindGroupLayout,
    /// `@group(0)` for the `water_spray_emit` pass (three bindings).
    pub(crate) spray_layout: BindGroupLayout,
    /// `@group(0)` for the `water_swe_step` pass (eight bindings).
    pub(crate) swe_layout: BindGroupLayout,
    /// `@group(1)` for the `water_foam_advect` pass (six bindings).
    pub(crate) foam_layout: BindGroupLayout,
    /// `@group(2)` for the `water_waterline_mask` pass (five bindings).
    pub(crate) waterline_layout: BindGroupLayout,
    /// `@group(0)` for the `water_caustics_project` pass (four bindings).
    pub(crate) caustics_layout: BindGroupLayout,
    /// `@group(1)` for the `water_dispersion_refract` pass (five bindings).
    pub(crate) dispersion_layout: BindGroupLayout,
    /// `@group(2)` for the `water_underwater_volume` pass (three bindings).
    pub(crate) underwater_layout: BindGroupLayout,
    /// `@group(3)` for the `water_wetness_step` pass (three bindings).
    pub(crate) wetness_layout: BindGroupLayout,
    /// `@group(4)` for the `water_coupling_readback` pass (three bindings).
    pub(crate) coupling_layout: BindGroupLayout,
    /// `@group(0)` for both `water_spectrum_fft.wesl` entry points
    /// (`water_spectrum_evolve` + `water_spectrum_assemble`, thirteen bindings).
    pub(crate) spectrum_fft_layout: BindGroupLayout,
    /// `@group(0)` for the three `water_butterfly.wesl` passes (three bindings).
    pub(crate) butterfly_layout: BindGroupLayout,
    /// `@group(0)` for `water_flip_mac_p2g.wesl` (`water_flip_mac_p2g` +
    /// `water_flip_mac_faces_normalize`, four bindings including the `atomic`
    /// face-scatter accumulator).
    pub(crate) mac_p2g_layout: BindGroupLayout,
    /// `@group(0)` for the three `water_flip_mac.wesl` passes (`mac_divergence`
    /// + `mac_pressure` + `mac_project`, five bindings).
    pub(crate) mac_solve_layout: BindGroupLayout,
    /// `@group(0)` for `water_flip_mac_g2p.wesl` (`water_flip_mac_g2p`, four
    /// bindings: the projected and pre-projection face snapshots).
    pub(crate) mac_g2p_layout: BindGroupLayout,
    /// `@group(0)` for `water_surface_mesh.wesl` (`water_surface_mesh`, eight
    /// bindings: the mesh uniform, the sampled displacement / normal cascade
    /// textures, the sampler, and the four write-only per-vertex arrays).
    pub(crate) surface_mesh_layout: BindGroupLayout,

    /// `water_spectrum_ifft`: evolve and inverse-`FFT` the wave spectrum.
    pub(crate) spectrum_ifft: CachedComputePipelineId,
    /// `water_gerstner_displace`: sum the analytic `Gerstner` wave trains.
    pub(crate) gerstner_displace: CachedComputePipelineId,
    /// `water_swe_step`: one Shallow-Water Equations height/velocity step.
    pub(crate) swe_step: CachedComputePipelineId,
    /// `water_pbf_density_solve`: one `PBF` density-constraint iteration.
    pub(crate) pbf_density_solve: CachedComputePipelineId,
    /// `water_flip_p2g`: `FLIP`/`APIC` particle-to-grid scatter.
    pub(crate) flip_p2g: CachedComputePipelineId,
    /// `water_flip_pressure_solve`: the grid incompressibility projection.
    pub(crate) flip_pressure_solve: CachedComputePipelineId,
    /// `water_flip_g2p`: `FLIP`/`APIC` grid-to-particle gather.
    pub(crate) flip_g2p: CachedComputePipelineId,
    /// `water_flip_mac_p2g`: face-centered `MAC` particle-to-grid momentum sum.
    pub(crate) mac_p2g: CachedComputePipelineId,
    /// `water_flip_mac_faces_normalize`: divide the summed `MAC` momentum by
    /// weight to recover face-centered velocities.
    pub(crate) mac_faces_normalize: CachedComputePipelineId,
    /// `mac_divergence`: compact single-sided divergence of the `MAC` faces.
    pub(crate) mac_divergence: CachedComputePipelineId,
    /// `mac_pressure`: one `Jacobi` pressure-relaxation iteration.
    pub(crate) mac_pressure: CachedComputePipelineId,
    /// `mac_project`: subtract the compact pressure gradient from the faces.
    pub(crate) mac_project: CachedComputePipelineId,
    /// `water_flip_mac_g2p`: face-centered `MAC` grid-to-particle gather.
    pub(crate) mac_g2p: CachedComputePipelineId,
    /// `water_surface_reconstruct`: build a renderable surface from particles.
    pub(crate) surface_reconstruct: CachedComputePipelineId,
    /// `water_caustics_project`: project caustic intensity onto receivers.
    pub(crate) caustics_project: CachedComputePipelineId,
    /// `water_foam_advect`: advect and decay the foam coverage field.
    pub(crate) foam_advect: CachedComputePipelineId,
    /// `water_spray_emit`: append crest-spray particles into the pool.
    pub(crate) spray_emit: CachedComputePipelineId,
    /// `water_wetness_step`: advance the per-cell surface wetness field.
    pub(crate) wetness_step: CachedComputePipelineId,
    /// `water_waterline_mask`: rasterize the soft above/below waterline mask.
    pub(crate) waterline_mask: CachedComputePipelineId,
    /// `water_dispersion_refract`: per-channel (`RGB`) `IOR` screen refraction.
    pub(crate) dispersion_refract: CachedComputePipelineId,
    /// `water_underwater_volume`: accumulate underwater single-scatter froxels.
    pub(crate) underwater_volume: CachedComputePipelineId,
    /// `water_coupling_readback`: read back bounded two-way coupling queries.
    pub(crate) coupling_readback: CachedComputePipelineId,
    /// `water_spectrum_evolve`: advance `h0` to the time-`t` complex spectrum.
    pub(crate) spectrum_evolve: CachedComputePipelineId,
    /// `water_fft_bitrev`: bit-reversal permutation before the butterfly stages.
    pub(crate) fft_bit_reverse: CachedComputePipelineId,
    /// `water_fft_stage`: one radix-2 butterfly stage of the inverse `FFT`.
    pub(crate) fft_stage: CachedComputePipelineId,
    /// `water_fft_normalize`: scale the transformed grid by `1/N`.
    pub(crate) fft_normalize: CachedComputePipelineId,
    /// `water_spectrum_assemble`: pack displacement/normal from the `FFT` grids.
    pub(crate) spectrum_assemble: CachedComputePipelineId,
    /// `water_surface_mesh`: sample the assembled displacement / normal cascade
    /// textures and scatter the four per-vertex storage arrays the raster draw
    /// consumes.
    pub(crate) surface_mesh: CachedComputePipelineId,
}

impl WaterComputePipelines {
    /// Returns the queued pipeline handle that runs the given kernel.
    ///
    /// This maps the golden [`WaterKernel`] schedule onto the concrete pipeline
    /// handles so the dispatch slice can iterate [`WaterKernel::ALL`] and record
    /// each pass without duplicating the kernel-to-pipeline mapping.
    #[must_use]
    pub(crate) fn pipeline(&self, kernel: WaterKernel) -> CachedComputePipelineId {
        match kernel {
            WaterKernel::SpectrumIfft => self.spectrum_ifft,
            WaterKernel::GerstnerDisplace => self.gerstner_displace,
            WaterKernel::SweStep => self.swe_step,
            WaterKernel::PbfDensitySolve => self.pbf_density_solve,
            WaterKernel::FlipP2G => self.flip_p2g,
            WaterKernel::FlipPressureSolve => self.flip_pressure_solve,
            WaterKernel::FlipG2P => self.flip_g2p,
            WaterKernel::FlipMacP2G => self.mac_p2g,
            WaterKernel::FlipMacFacesNormalize => self.mac_faces_normalize,
            WaterKernel::FlipMacDivergence => self.mac_divergence,
            WaterKernel::FlipMacPressure => self.mac_pressure,
            WaterKernel::FlipMacProject => self.mac_project,
            WaterKernel::FlipMacG2P => self.mac_g2p,
            WaterKernel::SurfaceReconstruct => self.surface_reconstruct,
            WaterKernel::CausticsProject => self.caustics_project,
            WaterKernel::FoamAdvect => self.foam_advect,
            WaterKernel::SprayEmit => self.spray_emit,
            WaterKernel::WetnessStep => self.wetness_step,
            WaterKernel::WaterlineMask => self.waterline_mask,
            WaterKernel::DispersionRefract => self.dispersion_refract,
            WaterKernel::UnderwaterVolume => self.underwater_volume,
            WaterKernel::CouplingReadback => self.coupling_readback,
            WaterKernel::SpectrumEvolve => self.spectrum_evolve,
            WaterKernel::FftBitReverse => self.fft_bit_reverse,
            WaterKernel::FftStage => self.fft_stage,
            WaterKernel::FftNormalize => self.fft_normalize,
            WaterKernel::SpectrumAssemble => self.spectrum_assemble,
            WaterKernel::SurfaceMesh => self.surface_mesh,
        }
    }

    /// Returns the bind-group layout the given kernel's bind group must target.
    ///
    /// Mirrors the shader interface: the two `water_ocean.wesl` kernels share the
    /// ocean layout, the four `water_flip.wesl` kernels share the flip layout,
    /// and every other kernel binds its own distinct interface.
    #[must_use]
    pub(crate) fn layout(&self, kernel: WaterKernel) -> &BindGroupLayout {
        match kernel {
            WaterKernel::SpectrumIfft | WaterKernel::GerstnerDisplace => &self.ocean_layout,
            WaterKernel::FlipP2G
            | WaterKernel::FlipPressureSolve
            | WaterKernel::FlipG2P
            | WaterKernel::SurfaceReconstruct => &self.flip_layout,
            WaterKernel::FlipMacP2G | WaterKernel::FlipMacFacesNormalize => &self.mac_p2g_layout,
            WaterKernel::FlipMacDivergence
            | WaterKernel::FlipMacPressure
            | WaterKernel::FlipMacProject => &self.mac_solve_layout,
            WaterKernel::FlipMacG2P => &self.mac_g2p_layout,
            WaterKernel::PbfDensitySolve => &self.pbf_layout,
            WaterKernel::SprayEmit => &self.spray_layout,
            WaterKernel::SweStep => &self.swe_layout,
            WaterKernel::FoamAdvect => &self.foam_layout,
            WaterKernel::WaterlineMask => &self.waterline_layout,
            WaterKernel::CausticsProject => &self.caustics_layout,
            WaterKernel::DispersionRefract => &self.dispersion_layout,
            WaterKernel::UnderwaterVolume => &self.underwater_layout,
            WaterKernel::WetnessStep => &self.wetness_layout,
            WaterKernel::CouplingReadback => &self.coupling_layout,
            WaterKernel::SurfaceMesh => &self.surface_mesh_layout,
            WaterKernel::SpectrumEvolve | WaterKernel::SpectrumAssemble => {
                &self.spectrum_fft_layout
            }
            WaterKernel::FftBitReverse | WaterKernel::FftStage | WaterKernel::FftNormalize => {
                &self.butterfly_layout
            }
        }
    }
}

/// The `@group(N)` index the given kernel's shader declares its interface on.
///
/// Most kernels bind `@group(0)`, but the passes that share `water_surface.wesl`
/// and `water_render_fx.wesl` each own a distinct group index so `naga` never
/// sees two incompatible declarations of one group. The dispatch slice uses this
/// to call `set_bind_group(wesl_group(kernel), ..)`, and the pipeline builder
/// uses it to pad the lower placeholder slots.
#[must_use]
pub(crate) fn wesl_group(kernel: WaterKernel) -> u32 {
    match kernel {
        WaterKernel::FoamAdvect | WaterKernel::DispersionRefract => 1,
        WaterKernel::WaterlineMask | WaterKernel::UnderwaterVolume => 2,
        WaterKernel::WetnessStep => 3,
        WaterKernel::CouplingReadback => 4,
        WaterKernel::SpectrumIfft
        | WaterKernel::GerstnerDisplace
        | WaterKernel::SweStep
        | WaterKernel::PbfDensitySolve
        | WaterKernel::FlipP2G
        | WaterKernel::FlipPressureSolve
        | WaterKernel::FlipG2P
        | WaterKernel::FlipMacP2G
        | WaterKernel::FlipMacFacesNormalize
        | WaterKernel::FlipMacDivergence
        | WaterKernel::FlipMacPressure
        | WaterKernel::FlipMacProject
        | WaterKernel::FlipMacG2P
        | WaterKernel::SurfaceReconstruct
        | WaterKernel::CausticsProject
        | WaterKernel::SprayEmit
        | WaterKernel::SpectrumEvolve
        | WaterKernel::FftBitReverse
        | WaterKernel::FftStage
        | WaterKernel::FftNormalize
        | WaterKernel::SpectrumAssemble
        | WaterKernel::SurfaceMesh => 0,
    }
}

/// Builds the `water_ocean.wesl` `@group(0)` layout entries (nine bindings).
///
/// Bindings `0..2` are the read-only spectrum pools, `2` the spectrum uniform,
/// `3..5` the write-only displacement/normal storage textures, `5` the read-only
/// `Gerstner` wave train, `6` the `Gerstner` uniform, and `7..9` the write-only
/// `Gerstner` displacement/normal textures. `None` min-binding-size keeps the
/// layout agnostic to each pool's run-time length.
fn ocean_layout_entries() -> BindGroupLayoutEntries<9> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            uniform_buffer_sized(false, None),
            texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            storage_buffer_read_only_sized(false, None),
            uniform_buffer_sized(false, None),
            texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// Builds the `water_flip.wesl` `@group(0)` layout entries (nine bindings): the
/// read-write particle pool and atomic grid-scatter buffer, the read-only /
/// read-write pressure ping-pong, the sim uniform, the read-only surface
/// depth/thickness buffers, the write-only surface-normal texture, and the
/// surface uniform.
fn flip_layout_entries() -> BindGroupLayoutEntries<9> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            uniform_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `water_flip_mac_p2g.wesl` `@group(0)` layout entries (four
/// bindings): the read-only particle pool, the read-write `atomic` face-scatter
/// accumulator, the read-write face-velocity buffer, and the sim uniform. Both
/// `water_flip_mac_p2g` and `water_flip_mac_faces_normalize` bind this shape.
fn mac_p2g_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `water_flip_mac.wesl` `@group(0)` layout entries (five bindings):
/// the read-write `MAC` faces, the read-write divergence scratch, the read-only
/// pressure ping and read-write pressure pong, and the `MAC` uniform. The
/// `mac_divergence`, `mac_pressure`, and `mac_project` passes all bind this.
fn mac_solve_layout_entries() -> BindGroupLayoutEntries<5> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `water_flip_mac_g2p.wesl` `@group(0)` layout entries (four
/// bindings): the read-write particle pool, the read-only projected faces, the
/// read-only pre-projection faces (for the `FLIP` delta), and the sim uniform.
fn mac_g2p_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `water_pbf_density_solve` `@group(0)` layout entries (four
/// bindings): the read-only input positions, the read-write output positions,
/// the read-only spatial-hash table, and the `PBF` uniform.
fn pbf_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `water_spray_emit` `@group(0)` layout entries (three bindings): the
/// read-only spray sources, the read-write spawn record, and the spray uniform.
fn spray_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `water_swe_step` `@group(0)` layout entries (eight bindings): the
/// three read-only input fields (height, `u`, `v`), the three read-write output
/// fields, the read-only per-cell interaction sources, and the `SWE` uniform.
fn swe_layout_entries() -> BindGroupLayoutEntries<8> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `water_foam_advect` `@group(1)` layout entries (six bindings): the
/// read-only foam field, the two read-only advection velocity fields, the
/// read-only reactive foam sources, the read-write output foam field, and the
/// foam uniform.
fn foam_layout_entries() -> BindGroupLayoutEntries<6> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `water_waterline_mask` `@group(2)` layout entries (five bindings):
/// the read-only sample/surface heights, the read-only depth field, the
/// read-write output mask, and the waterline uniform.
fn waterline_layout_entries() -> BindGroupLayoutEntries<5> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `water_caustics_project` `@group(0)` layout entries (four
/// bindings): the read-only photon-count buffer, the caustics uniform, the
/// write-only single-channel caustics texture, and the read-only offset texture.
/// The offset texture has no companion sampler, so it is declared
/// non-filterable.
fn caustics_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            uniform_buffer_sized(false, None),
            texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    )
}

/// Builds the `water_dispersion_refract` `@group(1)` layout entries (five
/// bindings): the dispersion uniform, the write-only refraction output texture,
/// the read-only scene and normal textures, and the filtering sampler. The scene
/// and normal textures are sampled through the filtering sampler, so `wgpu`
/// requires them to be declared filterable.
fn dispersion_layout_entries() -> BindGroupLayoutEntries<5> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            uniform_buffer_sized(false, None),
            texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
            texture_2d(TextureSampleType::Float { filterable: true }),
            texture_2d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
        ),
    )
}

/// Builds the `water_underwater_volume` `@group(2)` layout entries (three
/// bindings): the underwater uniform, the write-only 3D froxel scatter texture,
/// and the read-only surface-light texture. The surface-light texture has no
/// companion sampler, so it is declared non-filterable.
fn underwater_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            uniform_buffer_sized(false, None),
            texture_storage_3d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    )
}

/// Builds the `water_wetness_step` `@group(3)` layout entries (three bindings):
/// the read-write per-cell wetness state, the wetness uniform, and the
/// write-only wetness output texture.
fn wetness_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_sized(false, None),
            uniform_buffer_sized(false, None),
            texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// Builds the two-entry-point `water_spectrum_fft.wesl` `@group(0)` layout
/// entries (thirteen bindings): the two read-only spectrum `h0` pools, the
/// spectrum uniform, the four read-write packed `FFT` grids, the four read-only
/// assembled `FFT` grids, and the two write-only displacement/normal storage
/// textures.
fn spectrum_fft_layout_entries() -> BindGroupLayoutEntries<13> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            uniform_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// Builds the `water_butterfly.wesl` `@group(0)` layout entries (three
/// bindings): the read-only `FFT` source grid, the read-write `FFT` destination
/// grid, and the per-pass butterfly uniform.
fn butterfly_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Builds the `water_coupling_readback` `@group(4)` layout entries (three
/// bindings): the read-only coupling queries, the read-write readback buffer,
/// and the coupling uniform.
fn coupling_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer: creates the twelve water bind-group layouts and
/// queues the sixteen water compute pipelines into the [`PipelineCache`].
///
/// The five water shaders must be registered as embedded assets before this
/// runs (see the water plugin slice); `load_embedded_asset!` resolves them by
/// their path relative to this file. Each pipeline names its `WESL` entry point
/// and binds the layout that matches the `@group(N)` index its shader declares,
/// padding the lower group slots of the `@group(1..=4)` passes with an empty
/// placeholder layout so the `wgpu` pipeline layout stays contiguous.
/// Builds the `water_surface_mesh.wesl` `@group(0)` layout entries (eight
/// bindings): the surface-mesh uniform, the assembled displacement and normal
/// cascade textures, their non-filtering sampler, and the four write-only
/// per-vertex storage arrays (base positions, surface `UV`s, displacement, and
/// normal/foam) the raster draw later reads.
fn surface_mesh_layout_entries() -> BindGroupLayoutEntries<8> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            uniform_buffer_sized(false, None),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            sampler(SamplerBindingType::NonFiltering),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
        ),
    )
}

pub(crate) fn init_water_compute_pipelines(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let ocean_entries = ocean_layout_entries();
    let flip_entries = flip_layout_entries();
    let pbf_entries = pbf_layout_entries();
    let spray_entries = spray_layout_entries();
    let swe_entries = swe_layout_entries();
    let foam_entries = foam_layout_entries();
    let waterline_entries = waterline_layout_entries();
    let caustics_entries = caustics_layout_entries();
    let dispersion_entries = dispersion_layout_entries();
    let underwater_entries = underwater_layout_entries();
    let wetness_entries = wetness_layout_entries();
    let coupling_entries = coupling_layout_entries();
    let spectrum_fft_entries = spectrum_fft_layout_entries();
    let butterfly_entries = butterfly_layout_entries();
    let mac_p2g_entries = mac_p2g_layout_entries();
    let mac_solve_entries = mac_solve_layout_entries();
    let mac_g2p_entries = mac_g2p_layout_entries();
    let surface_mesh_entries = surface_mesh_layout_entries();

    let ocean_descriptor = BindGroupLayoutDescriptor::new("prism water ocean", &ocean_entries);
    let flip_descriptor = BindGroupLayoutDescriptor::new("prism water flip", &flip_entries);
    let pbf_descriptor = BindGroupLayoutDescriptor::new("prism water pbf", &pbf_entries);
    let spray_descriptor = BindGroupLayoutDescriptor::new("prism water spray", &spray_entries);
    let swe_descriptor = BindGroupLayoutDescriptor::new("prism water swe", &swe_entries);
    let foam_descriptor = BindGroupLayoutDescriptor::new("prism water foam", &foam_entries);
    let waterline_descriptor =
        BindGroupLayoutDescriptor::new("prism water waterline", &waterline_entries);
    let caustics_descriptor =
        BindGroupLayoutDescriptor::new("prism water caustics", &caustics_entries);
    let dispersion_descriptor =
        BindGroupLayoutDescriptor::new("prism water dispersion", &dispersion_entries);
    let underwater_descriptor =
        BindGroupLayoutDescriptor::new("prism water underwater", &underwater_entries);
    let wetness_descriptor =
        BindGroupLayoutDescriptor::new("prism water wetness", &wetness_entries);
    let coupling_descriptor =
        BindGroupLayoutDescriptor::new("prism water coupling", &coupling_entries);
    let spectrum_fft_descriptor =
        BindGroupLayoutDescriptor::new("prism water spectrum fft", &spectrum_fft_entries);
    let butterfly_descriptor =
        BindGroupLayoutDescriptor::new("prism water butterfly", &butterfly_entries);
    let mac_p2g_descriptor =
        BindGroupLayoutDescriptor::new("prism water mac p2g", &mac_p2g_entries);
    let mac_solve_descriptor =
        BindGroupLayoutDescriptor::new("prism water mac solve", &mac_solve_entries);
    let mac_g2p_descriptor =
        BindGroupLayoutDescriptor::new("prism water mac g2p", &mac_g2p_entries);
    let surface_mesh_descriptor =
        BindGroupLayoutDescriptor::new("prism water surface mesh", &surface_mesh_entries);

    // Empty placeholder layout padding the lower, unused group slots of the
    // `@group(1..=4)` passes so the `wgpu` pipeline layout stays contiguous; the
    // dispatch slice only binds the single real group each kernel reads.
    let empty = BindGroupLayoutDescriptor::new("prism water empty", &[]);

    let ocean_layout = device.create_bind_group_layout("prism water ocean", &ocean_entries);
    let flip_layout = device.create_bind_group_layout("prism water flip", &flip_entries);
    let pbf_layout = device.create_bind_group_layout("prism water pbf", &pbf_entries);
    let spray_layout = device.create_bind_group_layout("prism water spray", &spray_entries);
    let swe_layout = device.create_bind_group_layout("prism water swe", &swe_entries);
    let foam_layout = device.create_bind_group_layout("prism water foam", &foam_entries);
    let waterline_layout =
        device.create_bind_group_layout("prism water waterline", &waterline_entries);
    let caustics_layout =
        device.create_bind_group_layout("prism water caustics", &caustics_entries);
    let dispersion_layout =
        device.create_bind_group_layout("prism water dispersion", &dispersion_entries);
    let underwater_layout =
        device.create_bind_group_layout("prism water underwater", &underwater_entries);
    let wetness_layout = device.create_bind_group_layout("prism water wetness", &wetness_entries);
    let coupling_layout =
        device.create_bind_group_layout("prism water coupling", &coupling_entries);
    let spectrum_fft_layout =
        device.create_bind_group_layout("prism water spectrum fft", &spectrum_fft_entries);
    let butterfly_layout =
        device.create_bind_group_layout("prism water butterfly", &butterfly_entries);
    let mac_p2g_layout = device.create_bind_group_layout("prism water mac p2g", &mac_p2g_entries);
    let mac_solve_layout =
        device.create_bind_group_layout("prism water mac solve", &mac_solve_entries);
    let mac_g2p_layout = device.create_bind_group_layout("prism water mac g2p", &mac_g2p_entries);
    let surface_mesh_layout =
        device.create_bind_group_layout("prism water surface mesh", &surface_mesh_entries);

    let ocean_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/water_ocean.wesl");
    let flip_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/water_flip.wesl");
    let pbf_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/water_pbf.wesl");
    let surface_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/water_surface.wesl");
    let render_fx_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/water_render_fx.wesl");
    let spectrum_fft_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/water_spectrum_fft.wesl");
    let butterfly_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/water_butterfly.wesl");
    let mac_p2g_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/water_flip_mac_p2g.wesl");
    let mac_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/water_flip_mac.wesl");
    let mac_g2p_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/water_flip_mac_g2p.wesl");
    let surface_mesh_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/water_surface_mesh.wesl");

    // Every water pipeline binds the layout(s) matching its shader's declared
    // `@group(N)` index; passes on `@group(1..=4)` prepend empty placeholder
    // layouts for the lower slots. No water kernel reads a `var<immediate>`
    // push constant, so `immediate_size` is always zero.
    let queue = |label: &str,
                 layout: Vec<BindGroupLayoutDescriptor>,
                 shader: &Handle<Shader>,
                 kernel: WaterKernel| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.to_owned().into()),
            layout,
            immediate_size: 0,
            shader: shader.clone(),
            entry_point: Some(kernel.wesl_entry_point().to_owned().into()),
            ..Default::default()
        })
    };

    let spectrum_ifft = queue(
        "prism water spectrum ifft",
        vec![ocean_descriptor.clone()],
        &ocean_shader,
        WaterKernel::SpectrumIfft,
    );
    let gerstner_displace = queue(
        "prism water gerstner displace",
        vec![ocean_descriptor.clone()],
        &ocean_shader,
        WaterKernel::GerstnerDisplace,
    );
    let swe_step = queue(
        "prism water swe step",
        vec![swe_descriptor.clone()],
        &surface_shader,
        WaterKernel::SweStep,
    );
    let pbf_density_solve = queue(
        "prism water pbf density solve",
        vec![pbf_descriptor.clone()],
        &pbf_shader,
        WaterKernel::PbfDensitySolve,
    );
    let flip_p2g = queue(
        "prism water flip p2g",
        vec![flip_descriptor.clone()],
        &flip_shader,
        WaterKernel::FlipP2G,
    );
    let flip_pressure_solve = queue(
        "prism water flip pressure solve",
        vec![flip_descriptor.clone()],
        &flip_shader,
        WaterKernel::FlipPressureSolve,
    );
    let flip_g2p = queue(
        "prism water flip g2p",
        vec![flip_descriptor.clone()],
        &flip_shader,
        WaterKernel::FlipG2P,
    );
    let surface_reconstruct = queue(
        "prism water surface reconstruct",
        vec![flip_descriptor.clone()],
        &flip_shader,
        WaterKernel::SurfaceReconstruct,
    );
    let caustics_project = queue(
        "prism water caustics project",
        vec![caustics_descriptor.clone()],
        &render_fx_shader,
        WaterKernel::CausticsProject,
    );
    let foam_advect = queue(
        "prism water foam advect",
        vec![empty.clone(), foam_descriptor.clone()],
        &surface_shader,
        WaterKernel::FoamAdvect,
    );
    let spray_emit = queue(
        "prism water spray emit",
        vec![spray_descriptor.clone()],
        &pbf_shader,
        WaterKernel::SprayEmit,
    );
    let wetness_step = queue(
        "prism water wetness step",
        vec![
            empty.clone(),
            empty.clone(),
            empty.clone(),
            wetness_descriptor.clone(),
        ],
        &render_fx_shader,
        WaterKernel::WetnessStep,
    );
    let waterline_mask = queue(
        "prism water waterline mask",
        vec![empty.clone(), empty.clone(), waterline_descriptor.clone()],
        &surface_shader,
        WaterKernel::WaterlineMask,
    );
    let dispersion_refract = queue(
        "prism water dispersion refract",
        vec![empty.clone(), dispersion_descriptor.clone()],
        &render_fx_shader,
        WaterKernel::DispersionRefract,
    );
    let underwater_volume = queue(
        "prism water underwater volume",
        vec![empty.clone(), empty.clone(), underwater_descriptor.clone()],
        &render_fx_shader,
        WaterKernel::UnderwaterVolume,
    );
    let coupling_readback = queue(
        "prism water coupling readback",
        vec![
            empty.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
            coupling_descriptor.clone(),
        ],
        &render_fx_shader,
        WaterKernel::CouplingReadback,
    );
    let spectrum_evolve = queue(
        "prism water spectrum evolve",
        vec![spectrum_fft_descriptor.clone()],
        &spectrum_fft_shader,
        WaterKernel::SpectrumEvolve,
    );
    let spectrum_assemble = queue(
        "prism water spectrum assemble",
        vec![spectrum_fft_descriptor.clone()],
        &spectrum_fft_shader,
        WaterKernel::SpectrumAssemble,
    );
    let fft_bit_reverse = queue(
        "prism water fft bit reverse",
        vec![butterfly_descriptor.clone()],
        &butterfly_shader,
        WaterKernel::FftBitReverse,
    );
    let fft_stage = queue(
        "prism water fft stage",
        vec![butterfly_descriptor.clone()],
        &butterfly_shader,
        WaterKernel::FftStage,
    );
    let fft_normalize = queue(
        "prism water fft normalize",
        vec![butterfly_descriptor.clone()],
        &butterfly_shader,
        WaterKernel::FftNormalize,
    );
    let mac_p2g = queue(
        "prism water mac p2g",
        vec![mac_p2g_descriptor.clone()],
        &mac_p2g_shader,
        WaterKernel::FlipMacP2G,
    );
    let mac_faces_normalize = queue(
        "prism water mac faces normalize",
        vec![mac_p2g_descriptor.clone()],
        &mac_p2g_shader,
        WaterKernel::FlipMacFacesNormalize,
    );
    let mac_divergence = queue(
        "prism water mac divergence",
        vec![mac_solve_descriptor.clone()],
        &mac_shader,
        WaterKernel::FlipMacDivergence,
    );
    let mac_pressure = queue(
        "prism water mac pressure",
        vec![mac_solve_descriptor.clone()],
        &mac_shader,
        WaterKernel::FlipMacPressure,
    );
    let mac_project = queue(
        "prism water mac project",
        vec![mac_solve_descriptor.clone()],
        &mac_shader,
        WaterKernel::FlipMacProject,
    );
    let mac_g2p = queue(
        "prism water mac g2p",
        vec![mac_g2p_descriptor.clone()],
        &mac_g2p_shader,
        WaterKernel::FlipMacG2P,
    );
    let surface_mesh = queue(
        "prism water surface mesh",
        vec![surface_mesh_descriptor.clone()],
        &surface_mesh_shader,
        WaterKernel::SurfaceMesh,
    );

    commands.insert_resource(WaterComputePipelines {
        ocean_layout,
        flip_layout,
        pbf_layout,
        spray_layout,
        swe_layout,
        foam_layout,
        waterline_layout,
        caustics_layout,
        dispersion_layout,
        underwater_layout,
        wetness_layout,
        coupling_layout,
        spectrum_ifft,
        gerstner_displace,
        swe_step,
        pbf_density_solve,
        flip_p2g,
        flip_pressure_solve,
        flip_g2p,
        surface_reconstruct,
        caustics_project,
        foam_advect,
        spray_emit,
        wetness_step,
        waterline_mask,
        dispersion_refract,
        underwater_volume,
        coupling_readback,
        spectrum_fft_layout,
        butterfly_layout,
        mac_p2g_layout,
        mac_solve_layout,
        mac_g2p_layout,
        surface_mesh_layout,
        spectrum_evolve,
        spectrum_assemble,
        fft_bit_reverse,
        fft_stage,
        fft_normalize,
        mac_p2g,
        mac_faces_normalize,
        mac_divergence,
        mac_pressure,
        mac_project,
        mac_g2p,
        surface_mesh,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel-to-`@group` map must place every kernel on the group index its
    /// shader declares. This pins the two shared-shader group families
    /// (`water_surface.wesl` and `water_render_fx.wesl`) so a shader edit that
    /// renumbers a group without updating [`wesl_group`] fails here.
    #[test]
    fn wesl_group_matches_the_shader_group_index() {
        assert_eq!(wesl_group(WaterKernel::SpectrumIfft), 0);
        assert_eq!(wesl_group(WaterKernel::GerstnerDisplace), 0);
        assert_eq!(wesl_group(WaterKernel::SweStep), 0);
        assert_eq!(wesl_group(WaterKernel::PbfDensitySolve), 0);
        assert_eq!(wesl_group(WaterKernel::FlipP2G), 0);
        assert_eq!(wesl_group(WaterKernel::FlipPressureSolve), 0);
        assert_eq!(wesl_group(WaterKernel::FlipG2P), 0);
        assert_eq!(wesl_group(WaterKernel::SurfaceReconstruct), 0);
        assert_eq!(wesl_group(WaterKernel::CausticsProject), 0);
        assert_eq!(wesl_group(WaterKernel::SprayEmit), 0);
        assert_eq!(wesl_group(WaterKernel::FoamAdvect), 1);
        assert_eq!(wesl_group(WaterKernel::DispersionRefract), 1);
        assert_eq!(wesl_group(WaterKernel::WaterlineMask), 2);
        assert_eq!(wesl_group(WaterKernel::UnderwaterVolume), 2);
        assert_eq!(wesl_group(WaterKernel::WetnessStep), 3);
        assert_eq!(wesl_group(WaterKernel::CouplingReadback), 4);
        assert_eq!(wesl_group(WaterKernel::SpectrumEvolve), 0);
        assert_eq!(wesl_group(WaterKernel::SpectrumAssemble), 0);
        assert_eq!(wesl_group(WaterKernel::FftBitReverse), 0);
        assert_eq!(wesl_group(WaterKernel::FftStage), 0);
        assert_eq!(wesl_group(WaterKernel::FftNormalize), 0);
    }

    /// The layout-entry builders must report the binding counts the shaders
    /// declare, and every kernel's golden descriptor must bind at least one
    /// resource. The concrete `BindGroupLayout` handles need a `RenderDevice`,
    /// which a headless unit test has no access to, so this validates the
    /// interface *shape* through the entry builders and the golden contract
    /// rather than the device layouts.
    #[test]
    fn layout_entries_match_the_declared_binding_counts() {
        assert_eq!(ocean_layout_entries().len(), 9);
        assert_eq!(flip_layout_entries().len(), 9);
        assert_eq!(pbf_layout_entries().len(), 4);
        assert_eq!(spray_layout_entries().len(), 3);
        assert_eq!(swe_layout_entries().len(), 8);
        assert_eq!(foam_layout_entries().len(), 6);
        assert_eq!(waterline_layout_entries().len(), 5);
        assert_eq!(caustics_layout_entries().len(), 4);
        assert_eq!(dispersion_layout_entries().len(), 5);
        assert_eq!(underwater_layout_entries().len(), 3);
        assert_eq!(wetness_layout_entries().len(), 3);
        assert_eq!(coupling_layout_entries().len(), 3);
        assert_eq!(spectrum_fft_layout_entries().len(), 13);
        assert_eq!(butterfly_layout_entries().len(), 3);
        assert_eq!(mac_p2g_layout_entries().len(), 4);
        assert_eq!(mac_solve_layout_entries().len(), 5);
        assert_eq!(mac_g2p_layout_entries().len(), 4);
        assert_eq!(surface_mesh_layout_entries().len(), 8);

        for kernel in WaterKernel::ALL {
            assert!(
                kernel.descriptor().layout.total() > 0,
                "{kernel:?} bound nothing"
            );
        }
    }

    /// The two `water_ocean.wesl` kernels and the four `water_flip.wesl` kernels
    /// must each collapse onto one shared group interface, matching the shared
    /// layout arms in [`WaterComputePipelines::layout`]. This exercises the
    /// grouping through the golden storage-texture counts the contract declares.
    #[test]
    fn shared_shader_kernels_agree_on_their_interface() {
        // Both ocean kernels write two displacement/normal storage textures.
        for kernel in [WaterKernel::SpectrumIfft, WaterKernel::GerstnerDisplace] {
            assert_eq!(
                kernel.descriptor().layout.storage_textures,
                2,
                "{kernel:?} ocean layout expects two storage textures"
            );
        }
        // The three particle/grid flip solve kernels share three storage
        // buffers and one uniform with no sampled textures.
        for kernel in [
            WaterKernel::PbfDensitySolve,
            WaterKernel::FlipP2G,
            WaterKernel::FlipG2P,
        ] {
            assert_eq!(kernel.descriptor().layout.storage_buffers, 3);
            assert_eq!(kernel.descriptor().layout.sampled_textures, 0);
        }
    }
}
