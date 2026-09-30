//! `wgpu` compute twin of the multi-scatter `LUT` sampler
//! ([`MultiScatterLut::sample`](prism_render_architecture::volumetric::multiscatter::MultiScatterLut::sample)).
//!
//! The multi-scatter energy-gain table is a 3D `LUT` over `cos` (view/light
//! angle cosine), accumulated `optical_depth` and `albedo` (design section 7b).
//! A query maps each raw axis coordinate to a normalized fraction (clamped into
//! range), resolves the two bracketing cells per axis, blends the eight
//! surrounding cells with trilinear weights, and saturates the result. Because
//! the corner weights sum to one and every stored cell is in `[0, 1]`, the
//! returned gain is always in `[0, 1]`; out-of-range coordinates resolve to the
//! nearest edge value without a `panic`. The `CPU` golden
//! [`MultiScatterLut::sample`](prism_render_architecture::volumetric::multiscatter::MultiScatterLut::sample)
//! owns that math; [`GpuMultiScatterLutSample`] is the on-device twin that runs
//! one thread per query.
//!
//! # Correctness model
//!
//! The sampler is only multiply/add on the raw table plus integer address
//! arithmetic — no transcendental, no lattice `hash`, no optional device
//! feature — so `CPU` and `GPU` evaluate the identical closed-form algebra. The
//! per-corner weight is grouped as `(wx * wy * wz)` and then multiplied by the
//! cell value, matching the `CPU` `weights[k] * corners[k]` accumulation order,
//! so they differ at most by a legal multiply-add contraction of a few `ULP`.
//! The parity test asserts each sampled gain to within `abs_diff < 1e-6`,
//! additionally checks the `[0, 1]` range and clamp-to-edge behaviour, so a
//! swapped axis, a dropped corner, or a mis-ordered accumulation could not pass.
//!
//! # Portability
//!
//! The kernel is integer address math plus multiply/add in the portable
//! core-`WGSL` subset, so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard trilinear `LUT` sampling with clamp-to-edge plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

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

/// One multi-scatter `LUT` query: the three physical axis coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiScatterSampleQuery {
    /// View/light angle cosine (first axis).
    pub cos: f32,
    /// Accumulated `optical_depth` (second axis).
    pub depth: f32,
    /// Single-scatter `albedo` (third axis).
    pub albedo: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/multiscatter_lut_sample.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    cos: f32,
    depth: f32,
    albedo: f32,
    pad0: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/multiscatter_lut_sample.wesl`: the query count, the three per-axis
/// cell counts, and the axis min/max ranges (each `vec3` padded to 16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    dim0: u32,
    dim1: u32,
    dim2: u32,
    min0: f32,
    min1: f32,
    min2: f32,
    pad0: f32,
    max0: f32,
    max1: f32,
    max2: f32,
    pad1: f32,
}

/// A compiled, reusable multi-scatter `LUT` sampler pipeline.
pub struct GpuMultiScatterLutSample {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMultiScatterLutSample {
    /// Compiles the multi-scatter `LUT` sampler kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMultiScatterLutSample {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_sample"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/multiscatter_lut_sample.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_sample_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_sample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_sample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("multiscatter_lut_sample_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMultiScatterLutSample {
            module,
            layout,
            pipeline,
        }
    }

    /// Samples the row-major table `lut` (cell counts `dims`, axis ranges
    /// `mins`/`maxs`) at every query in `queries`, returning one gain per query
    /// in input order.
    ///
    /// The `lut` slice is addressed as
    /// `(cos * dims[1] + depth) * dims[2] + albedo` and must have length
    /// `dims[0] * dims[1] * dims[2]`; the returned gain for query `q` equals
    /// the `CPU`
    /// [`MultiScatterLut::sample`](prism_render_architecture::volumetric::multiscatter::MultiScatterLut::sample)
    /// of a table carrying the same cells and ranges, to within the tolerance
    /// documented on this module. An empty `queries` slice yields an empty
    /// result — storage buffers cannot be zero-sized, so it is handled by an
    /// early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        lut: &[f32],
        dims: [usize; 3],
        mins: [f32; 3],
        maxs: [f32; 3],
        queries: &[MultiScatterSampleQuery],
    ) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        assert_eq!(
            lut.len(),
            dims[0] * dims[1] * dims[2],
            "lut length must equal dims[0] * dims[1] * dims[2]"
        );
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                cos: q.cos,
                depth: q.depth,
                albedo: q.albedo,
                pad0: 0.0,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            dim0: dims[0] as u32,
            dim1: dims[1] as u32,
            dim2: dims[2] as u32,
            min0: mins[0],
            min1: mins[1],
            min2: mins[2],
            pad0: 0.0,
            max0: maxs[0],
            max1: maxs[1],
            max2: maxs[2],
            pad1: 0.0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_sample_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let lut_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_sample_lut"),
            contents: bytemuck::cast_slice(lut),
            usage: BufferUsages::STORAGE,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_sample_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_sample_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_sample_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_sample_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: lut_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_multiscatter_lut_sample_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_multiscatter_lut_sample_pass"),
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
