//! `wgpu` compute twin of the Monte-Carlo transmittance oracles
//! ([`delta_tracking_transmittance`](prism_render_architecture::volumetric::reference::delta_tracking_transmittance)
//! and
//! [`ratio_tracking_transmittance`](prism_render_architecture::volumetric::reference::ratio_tracking_transmittance))
//! for a **homogeneous** medium.
//!
//! The `CPU` goldens take an arbitrary extinction closure `sigma_t_fn` so they
//! can integrate a heterogeneous field. Their on-device twin restricts that
//! closure to the constant `sigma_t`, which is exactly the regime the parity
//! test drives the goldens with, so the two sides run the identical random
//! walk. Each of `samples` walks re-seeds a deterministic Weyl+hash `RNG` from
//! `sample_seed(seed, i)` and integrates free-flight steps
//! `t += -ln(1 - u) / majorant`:
//!
//! * delta tracking counts the fraction of walks that reach `distance` before a
//!   real collision (accepted with probability `saturate(sigma_t / majorant)`
//!   at each tentative event); it draws two unit floats per interior event.
//! * ratio tracking accumulates the residual-ratio product
//!   `prod (1 - saturate(sigma_t / majorant))` over the null-collision events;
//!   it draws one unit float per event.
//!
//! Both estimators are the unbiased reference oracles the analytic
//! transmittance of design sections 6 / 9c must converge to.
//! [`GpuTrackingTransmittance`] runs one thread per query and returns both
//! estimates.
//!
//! # Correctness model
//!
//! The `RNG`, `ln_approx` and arithmetic mirror the goldens exactly: `WGSL`
//! `u32` wraps like `wrapping_*`, and `ln_approx` is the same
//! bit-reconstruction + `atanh` series. Floating-point division is
//! spec-allowed to differ by a few `ULP` across devices, though; on a
//! correctly-rounding device the walks stay bit-identical, but the parity test
//! uses tolerances (plus a discrete-flip margin for the delta count) rather
//! than bit equality so it stays robust across adapters. The test also checks
//! the `[0, 1]` range, the zero-distance / empty-budget identity, that both
//! estimates converge to the analytic transmittance as `samples` grows, and
//! that ratio tracking sits no further from the analytic value than delta
//! tracking.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — no optional device
//! feature — so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard delta / ratio tracking estimators (Novak et al.); no
//! Unreal Engine source or derived code.
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

/// One tracking query: the homogeneous medium's extinction and majorant, the
/// path distance, and the deterministic seed / sample budget for the walk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackingTransmittanceQuery {
    /// The homogeneous medium extinction (accepted with probability
    /// `saturate(sigma_t / majorant)`; negatives clamp to `0`).
    pub sigma_t: f32,
    /// The free-flight majorant (upper bound on `sigma_t`; floored at `EPS`).
    pub majorant: f32,
    /// The path distance through the medium (clamped to `>= 0`).
    pub distance: f32,
    /// The base `RNG` seed; each sample decorrelates from it.
    pub seed: u32,
    /// The number of independent free-flight walks to average.
    pub samples: u32,
}

/// One tracking result: the delta-tracking and ratio-tracking transmittance
/// estimates, both in `0..=1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackingEstimate {
    /// The delta-tracking (survival-fraction) estimate.
    pub delta: f32,
    /// The ratio-tracking (residual-product) estimate.
    pub ratio: f32,
}

/// One query as uploaded. `32`-byte `repr(C)` matching `Query` in
/// `shaders/tracking_transmittance.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    sigma_t: f32,
    majorant: f32,
    distance: f32,
    seed: u32,
    samples: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One result as read back. `8`-byte `repr(C)` matching `Res` in
/// `shaders/tracking_transmittance.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    delta: f32,
    ratio: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/tracking_transmittance.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable tracking-transmittance pipeline.
pub struct GpuTrackingTransmittance {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTrackingTransmittance {
    /// Compiles the tracking-transmittance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTrackingTransmittance {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_tracking_transmittance_shader"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/tracking_transmittance.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_tracking_transmittance_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_tracking_transmittance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_tracking_transmittance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("tracking_transmittance_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTrackingTransmittance {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning one [`TrackingEstimate`]
    /// per query in input order.
    ///
    /// For a query `q` the result's `delta` equals
    /// [`delta_tracking_transmittance`](prism_render_architecture::volumetric::reference::delta_tracking_transmittance)`(|_| q.sigma_t, q.majorant, q.distance, q.seed, q.samples)`
    /// and `ratio` equals the matching
    /// [`ratio_tracking_transmittance`](prism_render_architecture::volumetric::reference::ratio_tracking_transmittance)
    /// call, up to floating-point division tolerance. An empty `queries` slice
    /// yields an empty result — storage buffers cannot be zero-sized, so it is
    /// handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[TrackingTransmittanceQuery],
    ) -> Vec<TrackingEstimate> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                sigma_t: q.sigma_t,
                majorant: q.majorant,
                distance: q.distance,
                seed: q.seed,
                samples: q.samples,
                pad0: 0,
                pad1: 0,
                pad2: 0,
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
            label: Some("prism_volumetric_tracking_transmittance_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_tracking_transmittance_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_tracking_transmittance_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_tracking_transmittance_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_tracking_transmittance_bind_group"),
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
            label: Some("prism_volumetric_tracking_transmittance_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_tracking_transmittance_pass"),
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
            .map(|r| TrackingEstimate {
                delta: r.delta,
                ratio: r.ratio,
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
