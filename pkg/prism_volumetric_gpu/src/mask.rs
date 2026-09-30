//! `wgpu` compute twin of the god-ray scattering mask
//! ([`scattering_mask`](prism_render_architecture::volumetric::shadow::scattering_mask)).
//!
//! Volumetric beams are visible where light reaches the fragment (high
//! `shadow_transmittance`) *and* there is medium to scatter off (nonzero
//! density). The mask is the product of the two saturated terms, so it is
//! always in `[0, 1]` and vanishes in either full shadow or clear air (design
//! section 12). The `CPU` golden
//! [`scattering_mask`](prism_render_architecture::volumetric::shadow::scattering_mask)
//! owns that math; [`GpuScatteringMask`] is the on-device twin that runs one
//! thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The mask contains no transcendental call — it is two saturates and a single
//! multiply — so `CPU` and `GPU` evaluate the identical closed-form algebra
//! with no room for fma contraction (a lone multiply cannot be fused into a
//! multiply-add). The parity test therefore asserts the tightest tolerance
//! (`abs_diff < 1e-6` or `rel_diff < 1e-5`), and the scenes also assert the
//! documented `[0, 1]` range plus the two vanishing boundaries (full shadow and
//! clear air), so a degenerate kernel could not pass.
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
//! Provenance: standard screen-space god-ray gating plus `wgpu` compute
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

/// One mask query: the fragment's shadow transmittance and local density.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskQuery {
    /// Fraction of light reaching the fragment (clamped into `[0, 1]`).
    pub shadow_transmittance: f32,
    /// Local scattering density at the fragment (clamped into `[0, 1]`).
    pub density: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/mask.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    shadow_transmittance: f32,
    density: f32,
    pad0: f32,
    pad1: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/mask.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable god-ray scattering-mask pipeline.
pub struct GpuScatteringMask {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuScatteringMask {
    /// Compiles the scattering-mask kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuScatteringMask {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mask"),
            source: ShaderSource::Wgsl(include_str!("../shaders/mask.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mask_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mask_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mask_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("mask_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuScatteringMask {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the scattering mask for every query in `queries`, returning
    /// one value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`scattering_mask`](prism_render_architecture::volumetric::shadow::scattering_mask)`(q.shadow_transmittance, q.density)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[MaskQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                shadow_transmittance: q.shadow_transmittance,
                density: q.density,
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
            label: Some("prism_volumetric_mask_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mask_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mask_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mask_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mask_bind_group"),
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
            label: Some("prism_volumetric_mask_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mask_pass"),
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
