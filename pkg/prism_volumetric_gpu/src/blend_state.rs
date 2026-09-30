//! `wgpu` compute twin of the sky-state coverage blend
//! ([`blend_state`](prism_render_architecture::volumetric::weather::blend_state)).
//!
//! The sky-state machine (`Clear` -> `Fair` -> `Overcast` -> `Storm`) drives a
//! target cloud coverage per state; when the weather transitions between two
//! states the target coverage is interpolated with a Hermite `smoothstep` so it
//! starts and ends with zero slope — no visible jump at either endpoint (design
//! section 9). For a transition parameter `t` running `0..=1`:
//!
//! ```text
//! a = target_coverage(from)
//! b = target_coverage(to)
//! w = smoothstep(0, 1, t)   (t saturated, then cubic Hermite)
//! result = saturate(a + (b - a) * w)
//! ```
//!
//! The `CPU` golden
//! [`blend_state`](prism_render_architecture::volumetric::weather::blend_state)
//! owns that logic; [`GpuBlendState`] is the on-device twin that runs one thread
//! per query.
//!
//! # Correctness model
//!
//! The blend is monotone in `t`, the endpoints are exact (`t = 0` yields `a`,
//! `t = 1` yields `b`), and the result stays within the coverage interval
//! spanned by the two states, hence within `0..=1`. The kernel mirrors
//! `math::smoothstep` bit for bit, so the parity test asserts agreement within
//! a tight floating-point tolerance.
//!
//! # Portability
//!
//! The kernel is `clamp`, arithmetic and the mirrored smoothstep in the
//! portable core-`WGSL` subset — no `exp`, `pow` or optional device feature —
//! so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard smoothstep coverage blend / weather state machine plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::volumetric::weather::SkyState;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One blend query: the source and destination sky-states and the transition
/// parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendStateQuery {
    /// The sky-state the transition starts from (`t = 0`).
    pub from: SkyState,
    /// The sky-state the transition ends at (`t = 1`).
    pub to: SkyState,
    /// The transition parameter, saturated to `0..=1`.
    pub t: f32,
}

/// Maps a [`SkyState`] to the ordinal the shader expects.
fn state_ordinal(state: SkyState) -> u32 {
    match state {
        SkyState::Clear => 0,
        SkyState::Fair => 1,
        SkyState::Overcast => 2,
        SkyState::Storm => 3,
    }
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/blend_state.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    from_state: u32,
    to_state: u32,
    t: f32,
    pad: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/blend_state.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable sky-state blend pipeline.
pub struct GpuBlendState {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBlendState {
    /// Compiles the sky-state blend kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBlendState {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_blend_state"),
            source: ShaderSource::Wgsl(include_str!("../shaders/blend_state.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_blend_state_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_blend_state_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_blend_state_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("blend_state_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBlendState {
            module,
            layout,
            pipeline,
        }
    }

    /// Blends every query in `queries`, returning one coverage per query in
    /// input order.
    ///
    /// The returned value for query `q` equals
    /// [`blend_state`](prism_render_architecture::volumetric::weather::blend_state)`(q.from, q.to, q.t)`
    /// within a tight floating-point tolerance (the mirrored smoothstep). An
    /// empty `queries` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[BlendStateQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                from_state: state_ordinal(q.from),
                to_state: state_ordinal(q.to),
                t: q.t,
                pad: 0,
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
            label: Some("prism_volumetric_blend_state_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_blend_state_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_blend_state_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_blend_state_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_blend_state_bind_group"),
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
            label: Some("prism_volumetric_blend_state_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_blend_state_pass"),
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
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(out.len(), queries.len());
        out
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
