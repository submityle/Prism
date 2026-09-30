//! `wgpu` compute twin of the storm vertical-profile seam
//! ([`StormState::vertical_profile`](prism_render_architecture::volumetric::storm::StormState::vertical_profile)).
//!
//! The cumulonimbus vertical-development model (design section 9b) folds the
//! storm state machine's evolving fields back through the module's authored
//! curves into a single `0..=1` weight the density field multiplies the deep
//! convective column by. [`StormState::vertical_profile`] unions the spreading
//! `anvil_profile` (driven by `anvil_spread`) with the `overshooting_bump` dome
//! (driven by `overshooting_top`, reinforced by the `pyrocumulus` plume) via a
//! probabilistic `OR` `a + b - a*b`:
//!
//! ```text
//! anvil      = anvil_profile(height_fraction, anvil_spread)
//! dome_drive = saturate(overshooting_top + pyrocumulus * (1 - overshooting_top))
//! dome       = overshooting_bump(height_fraction, dome_drive)
//! result     = saturate(anvil + dome - anvil * dome)
//! ```
//!
//! so two overlapping contributions never sum past one and the result stays in
//! `0..=1`, monotonically non-decreasing in every driving field. The `CPU`
//! golden
//! [`StormState::vertical_profile`](prism_render_architecture::volumetric::storm::StormState::vertical_profile)
//! owns that math; [`GpuStormVerticalProfile`] is the on-device twin that runs
//! one thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The `lerp` and `smoothstep` are expanded to the *same* closed forms the
//! `CPU` `math` module uses, and the exponential in the dome uses the *same*
//! hand-rolled `exp_approx` the reference uses — base-two range reduction with a
//! fractional seven-term polynomial times an integer power assembled from the
//! `f32` exponent field — not the device-native `exp`. Mirroring those forms
//! keeps the twin bit-close to the reference, so the parity test asserts a
//! tight tolerance (`abs_diff < 1e-6` or `rel_diff < 1e-5`). The scenes also
//! assert the profile is bounded to `0..=1` and monotonically non-decreasing in
//! each of the three driving fields, so a degenerate kernel could not pass.
//!
//! # Portability
//!
//! The kernel is `floor`, `bitcast`, integer ops, `clamp`/`saturate` and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow` or
//! optional device feature — so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard cumulonimbus vertical-development composite plus `wgpu`
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

/// One storm vertical-profile query: the normalized band `height_fraction` plus
/// the three driving storm-state fields, each a normalized `0..=1` signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StormVerticalProfileQuery {
    /// Normalized band height (`0` at the cloud base, `1` at the tropopause).
    pub height_fraction: f32,
    /// `anvil` spread `0..=1` driving the flat anvil profile.
    pub anvil_spread: f32,
    /// `overshooting` top prominence `0..=1` driving the dome bump.
    pub overshooting_top: f32,
    /// `pyrocumulus` plume strength `0..=1` reinforcing the dome drive.
    pub pyrocumulus: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/storm_vertical_profile.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    height_fraction: f32,
    anvil_spread: f32,
    overshooting_top: f32,
    pyrocumulus: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/storm_vertical_profile.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable storm vertical-profile pipeline.
pub struct GpuStormVerticalProfile {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuStormVerticalProfile {
    /// Compiles the storm vertical-profile shader on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuStormVerticalProfile {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_storm_vertical_profile"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/storm_vertical_profile.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_storm_vertical_profile_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_storm_vertical_profile_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_storm_vertical_profile_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("storm_vertical_profile_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuStormVerticalProfile {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the storm vertical profile for every query in `queries`,
    /// returning one value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`StormState::vertical_profile`](prism_render_architecture::volumetric::storm::StormState::vertical_profile)
    /// evaluated on a `StormState` whose `anvil_spread`, `overshooting_top` and
    /// `pyrocumulus` fields are `q.anvil_spread`, `q.overshooting_top` and
    /// `q.pyrocumulus` (all other fields left at their default), called with
    /// `q.height_fraction`, to within the tolerance documented on this module.
    /// An empty `queries` slice yields an empty result — storage buffers cannot
    /// be zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[StormVerticalProfileQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                height_fraction: q.height_fraction,
                anvil_spread: q.anvil_spread,
                overshooting_top: q.overshooting_top,
                pyrocumulus: q.pyrocumulus,
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
            label: Some("prism_volumetric_storm_vertical_profile_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_storm_vertical_profile_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_storm_vertical_profile_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_storm_vertical_profile_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_storm_vertical_profile_bind_group"),
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
            label: Some("prism_volumetric_storm_vertical_profile_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_storm_vertical_profile_pass"),
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
