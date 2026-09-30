//! `wgpu` compute twin of the aerial-perspective fade
//! ([`aerial_perspective_weight`](prism_render_architecture::volumetric::atmosphere::aerial_perspective_weight)).
//!
//! Aerial perspective blends distant geometry toward the atmosphere in-scatter
//! with a normalised exponential opacity `(1 - e^{-k t}) / (1 - e^{-k})`, where
//! `t = distance / max_distance` clamped to `0..=1` and `k` is the fixed aerial
//! extinction (design section 8). The weight is monotone non-decreasing in
//! distance, exactly `0` at the camera and exactly `1` at the far plane, so it
//! mirrors the froxel fade without reading or rewriting the shared atmosphere
//! `LUT`. The `CPU` golden
//! [`aerial_perspective_weight`](prism_render_architecture::volumetric::atmosphere::aerial_perspective_weight)
//! owns that math; [`GpuAerialPerspective`] is the on-device twin that runs one
//! thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The exponential is evaluated with the *same* hand-rolled `exp_approx` the
//! reference uses — base-two range reduction with a fractional seven-term
//! polynomial times an integer power assembled from the `f32` exponent field —
//! not the device-native `exp`. Mirroring the polynomial keeps the twin
//! bit-close to the reference, so the parity test asserts a tight tolerance
//! (`abs_diff < 1e-6` or `rel_diff < 1e-5`). The scenes also assert the
//! documented boundary conditions (exactly `0` at the camera, exactly `1` at
//! the far plane), monotonicity and the `[0, 1]` range, plus the degenerate
//! non-positive `max_distance` branch, so a degenerate kernel could not pass.
//!
//! # Portability
//!
//! The kernel is `floor`, `bitcast`, integer ops and multiply/add in the
//! portable core-`WGSL` subset — no `exp`, `pow` or optional device feature —
//! so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard normalised exponential aerial-perspective fade plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

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

/// One aerial-perspective query: the fragment distance and the fade's far
/// distance (both in metres).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AerialQuery {
    /// Distance from the camera to the fragment, in metres.
    pub distance: f32,
    /// Farthest distance the fade reaches full opacity, in metres.
    pub max_distance: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/aerial.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    distance: f32,
    max_distance: f32,
    pad0: f32,
    pad1: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/aerial.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable aerial-perspective pipeline.
pub struct GpuAerialPerspective {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAerialPerspective {
    /// Compiles the aerial-perspective kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAerialPerspective {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_aerial"),
            source: ShaderSource::Wgsl(include_str!("../shaders/aerial.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_aerial_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_aerial_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_aerial_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("aerial_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAerialPerspective {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the aerial-perspective weight for every query in `queries`,
    /// returning one value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`aerial_perspective_weight`](prism_render_architecture::volumetric::atmosphere::aerial_perspective_weight)`(q.distance, q.max_distance)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[AerialQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                distance: q.distance,
                max_distance: q.max_distance,
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
            label: Some("prism_volumetric_aerial_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_aerial_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_aerial_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_aerial_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_aerial_bind_group"),
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
            label: Some("prism_volumetric_aerial_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_aerial_pass"),
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
