//! `wgpu` compute twin of the wind-velocity kernel
//! ([`WindField::velocity_at`](prism_render_architecture::volumetric::weather::WindField::velocity_at)).
//!
//! The wind velocity at a point combines the base flow `direction * speed` with
//! a divergence-free curl perturbation from the stream function
//! `psi = sin(f x) sin(f y)` — shape `(sin(f x) cos(f y), -cos(f x) sin(f y))`
//! with `f = CURL_FREQUENCY` — scaled by the gust
//! `speed * curl_strength * (1 + saturate(local_disturbance))` (design section 4
//! curl advection / section 9 weather). Keeping the perturbation
//! divergence-free is what makes wind advection near mass-conserving.
//!
//! The `CPU` golden
//! [`velocity_at`](prism_render_architecture::volumetric::weather::WindField::velocity_at)
//! owns that logic; [`GpuVelocityAt`] is the on-device twin that runs one thread
//! per query. The [`WindField`] normalizes its direction and clamps its speed /
//! curl strength on construction, so this kernel uploads the already-conditioned
//! accessor values and does no further conditioning of those inputs.
//!
//! # Correctness model
//!
//! The `CPU` golden reduces angles with a hand-rolled `sin_approx` / `cos_approx`
//! (range reduction to `[-PI/2, PI/2]` then a seventh-order Taylor polynomial,
//! no float intrinsic, for cross-target determinism); this kernel mirrors that
//! same reduction and polynomial so the on-device result matches bit-close,
//! rather than calling the `WGSL` `sin` / `cos` builtins. The parity test
//! asserts both velocity components within a tight tolerance.
//!
//! # Portability
//!
//! The kernel is arithmetic plus the polynomial `sin` / `cos` in the portable
//! core-`WGSL` subset — no trig builtin or optional device feature — so it runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard divergence-free curl-noise wind advection plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::volumetric::math::Vec2;
use prism_render_architecture::volumetric::weather::WindField;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One wind-velocity query: the wind field, the sample position (cell units)
/// and the per-cell wind-disturbance channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VelocityAtQuery {
    /// The wind field (its direction is normalized and speed / curl clamped on
    /// construction).
    pub wind: WindField,
    /// The sample position in cell units.
    pub pos: Vec2,
    /// The per-cell wind-disturbance channel (saturated to `0..=1`).
    pub local_disturbance: f32,
}

/// One query as uploaded. `32`-byte `repr(C)` matching `Query` in
/// `shaders/velocity_at.wesl`: the conditioned wind accessors, the position and
/// the disturbance, plus one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    direction_x: f32,
    direction_y: f32,
    speed: f32,
    curl_strength: f32,
    pos_x: f32,
    pos_y: f32,
    local_disturbance: f32,
    pad0: u32,
}

/// One result as read back. `8`-byte `repr(C)` matching `Res` in
/// `shaders/velocity_at.wesl`: the velocity vector.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    x: f32,
    y: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/velocity_at.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable wind-velocity pipeline.
pub struct GpuVelocityAt {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVelocityAt {
    /// Compiles the wind-velocity kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVelocityAt {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_velocity_at_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/velocity_at.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_velocity_at_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_velocity_at_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_velocity_at_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("velocity_at_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVelocityAt {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning one velocity [`Vec2`] per
    /// query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`velocity_at`](prism_render_architecture::volumetric::weather::WindField::velocity_at)
    /// applied to the same wind, position and disturbance, matching within a
    /// tight floating-point tolerance. An empty `queries` slice yields an empty
    /// result — storage buffers cannot be zero-sized, so it is handled by an
    /// early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[VelocityAtQuery]) -> Vec<Vec2> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| {
                let dir = q.wind.direction();
                GpuQuery {
                    direction_x: dir.x,
                    direction_y: dir.y,
                    speed: q.wind.speed(),
                    curl_strength: q.wind.curl_strength(),
                    pos_x: q.pos.x,
                    pos_y: q.pos.y,
                    local_disturbance: q.local_disturbance,
                    pad0: 0,
                }
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
            label: Some("prism_volumetric_velocity_at_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_velocity_at_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_velocity_at_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_velocity_at_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_velocity_at_bind_group"),
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
            label: Some("prism_volumetric_velocity_at_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_velocity_at_pass"),
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
        raw.into_iter().map(|r| Vec2::new(r.x, r.y)).collect()
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
