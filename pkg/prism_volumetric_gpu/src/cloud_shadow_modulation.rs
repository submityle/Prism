//! `wgpu` compute twin of the cloud-shadow ground-modulation factor
//! ([`cloud_shadow_modulation`](prism_render_architecture::volumetric::coupling::cloud_shadow_modulation)).
//!
//! Two-way scene coupling (design section 9d) closes here: the cloud's
//! transmittance controls how much sunlight reaches the ground, and the ground
//! albedo controls how much of that light is reflected back to modulate the
//! aerial perspective / `GI` sky-light. The lit-ground factor is the product of
//! the two saturated inputs and therefore stays in `0..=1`; a fully opaque
//! cloud (`transmittance == 0`) drives the ground contribution to `0`. The
//! `CPU` golden
//! [`cloud_shadow_modulation`](prism_render_architecture::volumetric::coupling::cloud_shadow_modulation)
//! owns that math; [`GpuCloudShadowModulation`] is the on-device twin that runs
//! one thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The kernel contains no transcendental call at all — it is two saturating
//! clamps and a multiply — so `CPU` and `GPU` evaluate the same closed-form
//! algebra. The only slack is a legal multiply-add contraction of a few `ULP`,
//! so the parity test asserts a tight tolerance (`abs_diff < 1e-6` or
//! `rel_diff < 1e-5`). The scenes also assert the documented `[0, 1]` range,
//! that an opaque cloud kills the ground contribution regardless of albedo, and
//! that fully open sky returns the (saturated) albedo, so a degenerate kernel
//! could not pass.
//!
//! # Portability
//!
//! The kernel is `clamp` and multiply in the portable core-`WGSL` subset — no
//! `exp`, `pow` or optional device feature — so it runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard cloud-shadow ground coupling plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One modulation query: the cloud transmittance reaching the ground and the
/// ground albedo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudShadowModulationQuery {
    /// Cloud transmittance reaching the ground; the fraction of sunlight that
    /// survives the cloud shadow.
    pub cloud_transmittance: f32,
    /// Ground albedo; the fraction of received light reflected back.
    pub ground_albedo: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/cloud_shadow_modulation.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    cloud_transmittance: f32,
    ground_albedo: f32,
    pad0: f32,
    pad1: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/cloud_shadow_modulation.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable cloud-shadow-modulation pipeline.
pub struct GpuCloudShadowModulation {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCloudShadowModulation {
    /// Compiles the cloud-shadow-modulation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCloudShadowModulation {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloud_shadow_modulation"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/cloud_shadow_modulation.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloud_shadow_modulation_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloud_shadow_modulation_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloud_shadow_modulation_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cloud_shadow_modulation_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCloudShadowModulation {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the cloud-shadow-modulation factor for every query in `queries`,
    /// returning one value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`cloud_shadow_modulation`](prism_render_architecture::volumetric::coupling::cloud_shadow_modulation)`(q.cloud_transmittance, q.ground_albedo)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[CloudShadowModulationQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                cloud_transmittance: q.cloud_transmittance,
                ground_albedo: q.ground_albedo,
                pad0: 0.0,
                pad1: 0.0,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloud_shadow_modulation_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloud_shadow_modulation_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloud_shadow_modulation_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloud_shadow_modulation_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloud_shadow_modulation_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloud_shadow_modulation_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloud_shadow_modulation_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
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
        let values = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(values.len(), queries.len());
        values
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
