//! `wgpu` compute twin of the multiple-scattering energy-gain `LUT` bake
//! ([`MultiScatterLut::build_energy_gain`](prism_render_architecture::volumetric::multiscatter::MultiScatterLut::build_energy_gain)).
//!
//! The energy-gain table is a three-axis `LUT` indexed by view-zenith cosine in
//! `[-1, 1]`, accumulated `optical_depth` in `[0, DEFAULT_MAX_OPTICAL_DEPTH]`,
//! and single-scatter `albedo` in `[0, 1]`, each cell storing a bounded
//! multiple-scatter energy gain in `[0, 1]` (design section 7b). Single-scatter
//! ray-marching loses the energy real clouds redistribute through many
//! scattering events; this table recovers it toward the analytic
//! multiple-scattering ceiling without ever amplifying past unit `albedo`. The
//! sample side already has a device twin (`GpuMultiScatterLutSample`); this
//! kernel is the on-device *bake* that fills every cell in parallel, one
//! invocation per cell, exactly as a startup or streaming pass would when the
//! `octave` schedule changes.
//!
//! # Correctness model
//!
//! The kernel mirrors the `CPU` reduction exactly. Each invocation reconstructs
//! its cell's three axis indices from its flat index with the `cos` axis
//! outermost, then `optical_depth`, then `albedo` (matching the golden
//! `MultiScatterLut::from_fn` fill order), maps each index to its physical
//! axis coordinate with the same `axis_value` lerp, and writes the same
//! saturated `energy_gain`: the product of an analytic isotropic diffusion
//! reflectance, an `optical_depth` ramp, and the `albedo`-independent
//! `octave`-phase modulation. The parity test rebuilds the same table on the
//! `CPU` through [`MultiScatterLut::build_energy_gain`] and reads back every
//! cell at its exact grid coordinate via `MultiScatterLut::sample` (where the
//! `trilinear` blend degenerates to the stored cell), so a mis-decomposed index
//! or a wrong axis coordinate could not pass.
//!
//! # Portability
//!
//! `exp` uses the same base-two range-reduction polynomial as the golden
//! `math::exp_approx`; every other operation is multiply-add plus compare in
//! the portable core-`WGSL` subset, so the kernel runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: `Frostbite`-style pre-integrated multiple-scattering energy gain
//! plus Wrenninge-style `octave` decay and standard Henyey-Greenstein phase,
//! baked with a `wgpu` compute dispatch; no Unreal Engine source or derived
//! code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::volumetric::scatter::OctaveParams;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Largest accumulated `optical_depth` represented on the depth axis, matching
/// `multiscatter::DEFAULT_MAX_OPTICAL_DEPTH`.
const DEFAULT_MAX_OPTICAL_DEPTH: f32 = 8.0;

/// Uniform parameters for one bake dispatch. Layout matches `Params` in
/// `shaders/multiscatter_lut_build.wesl`: three axis cell counts and the octave
/// count, then the three octave decays and the depth-axis maximum.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    cos_dim: u32,
    depth_dim: u32,
    albedo_dim: u32,
    octave_count: u32,
    attenuation: f32,
    contribution: f32,
    eccentricity_attenuation: f32,
    max_optical_depth: f32,
}

/// A compiled, reusable multiple-scattering energy-gain bake pipeline.
pub struct GpuMultiScatterLutBuild {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMultiScatterLutBuild {
    /// Compiles the energy-gain bake kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMultiScatterLutBuild {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_build"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/multiscatter_lut_build.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_build_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_build_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_build_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("multiscatter_lut_build_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMultiScatterLutBuild {
            module,
            layout,
            pipeline,
        }
    }

    /// Bakes the energy-gain table for the given per-axis cell counts and
    /// `octave` schedule, returning the row-major cells in fill order (the
    /// `cos` axis outermost, then `optical_depth`, then `albedo`).
    ///
    /// Each returned cell equals the `CPU`
    /// [`MultiScatterLut::build_energy_gain`](prism_render_architecture::volumetric::multiscatter::MultiScatterLut::build_energy_gain)
    /// value for the same grid to within the tolerance documented on this
    /// module. Every dimension is clamped to at least one cell (mirroring the
    /// golden), so the returned length is always the product of the clamped
    /// dimensions and never zero.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, dims: [u32; 3], params: OctaveParams) -> Vec<f32> {
        let dims = [dims[0].max(1), dims[1].max(1), dims[2].max(1)];
        let total = (dims[0] as u64) * (dims[1] as u64) * (dims[2] as u64);

        let device = ctx.device();

        let gpu_params = Params {
            cos_dim: dims[0],
            depth_dim: dims[1],
            albedo_dim: dims[2],
            octave_count: params.octave_count,
            attenuation: params.attenuation,
            contribution: params.contribution,
            eccentricity_attenuation: params.eccentricity_attenuation,
            max_optical_depth: DEFAULT_MAX_OPTICAL_DEPTH,
        };

        let out_bytes = total * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_build_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_build_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_build_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_build_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_build_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_multiscatter_lut_build_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (total as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len() as u64, total);
        raw
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
