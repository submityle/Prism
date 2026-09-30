//! `wgpu` compute twin of the trilinear corner-weights builder
//! ([`trilinear_weights`](prism_render_architecture::volumetric::multiscatter::trilinear_weights)).
//!
//! Both the multi-scatter `LUT` and the scatter probe grid interpolate their
//! tabulated energy with a trilinear blend (design section 7b). Corner `k` uses
//! bit `2` for `fx`, bit `1` for `fy` and bit `0` for `fz`; each axis
//! contributes `frac` or `1 - frac`, so the eight weights sum to exactly one
//! for any fractions (a partition of unity), which keeps interpolated energy
//! bounded. The `CPU` golden
//! [`trilinear_weights`](prism_render_architecture::volumetric::multiscatter::trilinear_weights)
//! owns that math; [`GpuTrilinear`] is the on-device twin that runs one thread
//! per query and returns the same eight weights.
//!
//! # Correctness model
//!
//! The blend is only multiply/add on the raw fractions — no transcendental, no
//! `clamp`, no lattice `hash` — so `CPU` and `GPU` evaluate the identical
//! closed-form algebra. They are not bit-exact only because a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few `ULP`. The parity test asserts each of the eight
//! weights to within `abs_diff < 1e-6` or `rel_diff < 1e-5`, additionally
//! asserts the partition-of-unity sum stays within `1e-6` of one, so a dropped
//! bit-selection or a swapped `frac`/`1 - frac` factor could not pass.
//!
//! # Portability
//!
//! The kernel is integer bit tests plus multiply/add in the portable
//! core-`WGSL` subset — no `exp`, `pow` or optional device feature — so it runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard trilinear partition-of-unity interpolation plus `wgpu`
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

/// One trilinear query: the three per-axis interpolation fractions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrilinearQuery {
    /// Interpolation fraction along the `x` axis.
    pub fx: f32,
    /// Interpolation fraction along the `y` axis.
    pub fy: f32,
    /// Interpolation fraction along the `z` axis.
    pub fz: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/trilinear.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    fx: f32,
    fy: f32,
    fz: f32,
    pad0: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/trilinear.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable trilinear-weights pipeline.
pub struct GpuTrilinear {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTrilinear {
    /// Compiles the trilinear-weights kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTrilinear {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_trilinear"),
            source: ShaderSource::Wgsl(include_str!("../shaders/trilinear.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_trilinear_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_trilinear_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_trilinear_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("trilinear_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTrilinear {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the eight trilinear corner weights for every query in
    /// `queries`, returning one `[f32; 8]` per query in input order.
    ///
    /// The returned array for query `q` equals
    /// [`trilinear_weights`](prism_render_architecture::volumetric::multiscatter::trilinear_weights)`(q.fx, q.fy, q.fz)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[TrilinearQuery]) -> Vec<[f32; 8]> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                fx: q.fx,
                fy: q.fy,
                fz: q.fz,
                pad0: 0.0,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Eight f32 per query: the unit-cube corner weights, tightly packed.
        let out_bytes = (queries.len() as u64) * 8 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_trilinear_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_trilinear_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_trilinear_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_trilinear_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_trilinear_bind_group"),
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
            label: Some("prism_volumetric_trilinear_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_trilinear_pass"),
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
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(flat.len(), queries.len() * 8);
        let values: Vec<[f32; 8]> = flat
            .chunks_exact(8)
            .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
            .collect();
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
