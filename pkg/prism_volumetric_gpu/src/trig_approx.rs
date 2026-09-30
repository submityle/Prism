//! `wgpu` compute twin of the hand-rolled trigonometry
//! ([`sin_approx`](prism_render_architecture::volumetric::math::sin_approx) /
//! [`cos_approx`](prism_render_architecture::volumetric::math::cos_approx)).
//!
//! The determinism policy forbids the hardware transcendental `sin`/`cos`, so
//! the `CPU` reduces the angle to `[-PI, PI]` via `wrap_pi` (round-half-away),
//! folds it into `[-PI/2, PI/2]` through `sin(PI - x)`, and evaluates a
//! seventh-order Taylor polynomial; cosine is `sin(x + PI/2)`. Several volume
//! kernels (`velocity_at`, `gravity_wave`) embed this trig inline;
//! [`GpuTrigApprox`] isolates it so a wide-angle sweep exercises the full range
//! reduction — the round-half-away tie handling and both folding branches — in
//! one place.
//!
//! # Correctness model
//!
//! The kernel mirrors the `CPU` reduction and polynomial exactly. `WGSL`'s
//! native `round` rounds ties to even, so `round_ties_away` reproduces Rust's
//! `f32::round` explicitly; every other step is multiply/add on the reduced
//! angle. `CPU` and `GPU` therefore evaluate the identical closed-form algebra
//! and differ at most by a legal multiply-add contraction of a few `ULP`. The
//! parity test asserts both `sin` and `cos` to within `abs_diff < 1e-6` across
//! several full periods (including negatives and near-tie angles), so a wrong
//! reduction branch or a dropped polynomial term could not pass.
//!
//! # Portability
//!
//! The kernel is `floor`/`select` plus multiply/add in the portable core-`WGSL`
//! subset, so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard range-reduced Taylor sine plus `wgpu` compute dispatch;
//! no Unreal Engine source or derived code.

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

/// One trigonometry query: the angle in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrigApproxQuery {
    /// Angle in radians.
    pub angle: f32,
}

/// The sine and cosine of one query angle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrigApproxResult {
    /// `sin_approx(angle)`.
    pub sin: f32,
    /// `cos_approx(angle)`.
    pub cos: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/trig_approx.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    angle: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// One result as read back. `8`-byte `repr(C)` matching `Res` in
/// `shaders/trig_approx.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    sin: f32,
    cos: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/trig_approx.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable trigonometry pipeline.
pub struct GpuTrigApprox {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTrigApprox {
    /// Compiles the trigonometry kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTrigApprox {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_trig_approx"),
            source: ShaderSource::Wgsl(include_str!("../shaders/trig_approx.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_trig_approx_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_trig_approx_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_trig_approx_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("trig_approx_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTrigApprox {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates `sin_approx` and `cos_approx` for every query in `queries`,
    /// returning one [`TrigApproxResult`] per query in input order.
    ///
    /// The returned pair for query `q` equals the `CPU`
    /// [`sin_approx`](prism_render_architecture::volumetric::math::sin_approx)`(q.angle)`
    /// and
    /// [`cos_approx`](prism_render_architecture::volumetric::math::cos_approx)`(q.angle)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[TrigApproxQuery]) -> Vec<TrigApproxResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                angle: q.angle,
                pad0: 0.0,
                pad1: 0.0,
                pad2: 0.0,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_trig_approx_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_trig_approx_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_trig_approx_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_trig_approx_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_trig_approx_bind_group"),
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
            label: Some("prism_volumetric_trig_approx_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_trig_approx_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());
        raw.into_iter()
            .map(|r| TrigApproxResult {
                sin: r.sin,
                cos: r.cos,
            })
            .collect()
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
