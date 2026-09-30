//! `wgpu` compute twin of the discrete sky-state machine
//! ([`advance_state`](prism_render_architecture::volumetric::weather::advance_state),
//! [`dissipate_state`](prism_render_architecture::volumetric::weather::dissipate_state)
//! and
//! [`target_coverage`](prism_render_architecture::volumetric::weather::target_coverage)).
//!
//! The sky-state ladder is `Clear` -> `Fair` -> `Overcast` -> `Storm` (design
//! section 9, "生消"). `advance_state` climbs one rung and saturates at `Storm`;
//! `dissipate_state` descends one rung and saturates at `Clear`;
//! `target_coverage` maps each state to the steady-state coverage it relaxes
//! toward. The `CPU` goldens own that logic; [`GpuSkyStateTransition`] is the
//! on-device twin that runs one thread per query and returns the advanced and
//! dissipated states plus the input state's target coverage.
//!
//! # Correctness model
//!
//! All three goldens are total, branch-only integer / lookup logic — no floats
//! beyond the four coverage constants — so the twin reproduces them exactly
//! (not merely close). The parity test asserts bit-exact equality of the two
//! transitioned states and exact equality of the target coverage, and checks
//! the ladder invariants (advance is non-decreasing and saturates at `Storm`,
//! dissipate is non-increasing and saturates at `Clear`).
//!
//! # Portability
//!
//! The kernel is integer branching in the portable core-`WGSL` subset — no
//! optional device feature — so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: original Prism weather state machine plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.
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

/// One sky-state transition result: the neighbouring states on the
/// intensification ladder plus the input state's target coverage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyStateTransition {
    /// The next intensifying state (saturates at `Storm`).
    pub advanced: SkyState,
    /// The next dissipating state (saturates at `Clear`).
    pub dissipated: SkyState,
    /// The input state's steady-state target coverage in `0..=1`.
    pub target_coverage: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/sky_state_transition.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    state: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One result as read back. `16`-byte `repr(C)` matching `Res` in
/// `shaders/sky_state_transition.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    advanced: u32,
    dissipated: u32,
    target_coverage: f32,
    pad0: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/sky_state_transition.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Maps a [`SkyState`] to the ordinal the kernel expects.
fn state_ordinal(state: SkyState) -> u32 {
    match state {
        SkyState::Clear => 0,
        SkyState::Fair => 1,
        SkyState::Overcast => 2,
        SkyState::Storm => 3,
    }
}

/// Maps a ladder ordinal back to a [`SkyState`] (positions above `3` clamp to
/// `Storm`, matching `SkyState::from_ordinal`).
fn state_from_ordinal(ordinal: u32) -> SkyState {
    match ordinal {
        0 => SkyState::Clear,
        1 => SkyState::Fair,
        2 => SkyState::Overcast,
        _ => SkyState::Storm,
    }
}

/// A compiled, reusable sky-state transition pipeline.
pub struct GpuSkyStateTransition {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSkyStateTransition {
    /// Compiles the sky-state transition kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSkyStateTransition {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sky_state_transition_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/sky_state_transition.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sky_state_transition_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sky_state_transition_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sky_state_transition_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sky_state_transition_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSkyStateTransition {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every state in `states`, returning one [`SkyStateTransition`]
    /// per state in input order.
    ///
    /// For state `s` the result has `advanced ==`
    /// [`advance_state`](prism_render_architecture::volumetric::weather::advance_state)`(s)`,
    /// `dissipated ==`
    /// [`dissipate_state`](prism_render_architecture::volumetric::weather::dissipate_state)`(s)`
    /// and `target_coverage ==`
    /// [`target_coverage`](prism_render_architecture::volumetric::weather::target_coverage)`(s)`.
    /// An empty `states` slice yields an empty result — storage buffers cannot
    /// be zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, states: &[SkyState]) -> Vec<SkyStateTransition> {
        if states.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = states
            .iter()
            .map(|&s| GpuQuery {
                state: state_ordinal(s),
                pad0: 0,
                pad1: 0,
                pad2: 0,
            })
            .collect();

        let gpu_params = Params {
            count: states.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (states.len() as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sky_state_transition_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sky_state_transition_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sky_state_transition_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sky_state_transition_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sky_state_transition_bind_group"),
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
            label: Some("prism_volumetric_sky_state_transition_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sky_state_transition_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (states.len() as u32).div_ceil(64);
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
        debug_assert_eq!(raw.len(), states.len());
        raw.into_iter()
            .map(|r| SkyStateTransition {
                advanced: state_from_ordinal(r.advanced),
                dissipated: state_from_ordinal(r.dissipated),
                target_coverage: r.target_coverage,
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
