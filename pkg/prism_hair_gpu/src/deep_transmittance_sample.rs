//! `wgpu` compute twin of Prism's deep-opacity transmittance decode
//! ([`sample_transmittance`](prism_render_architecture::hair::deep_transmittance::sample_transmittance)).
//!
//! [`GpuHairDeepOpacity`](crate::deep_opacity::GpuHairDeepOpacity) (and the
//! golden [`build_deep_opacity`](prism_render_architecture::hair::deep_transmittance::build_deep_opacity))
//! *pack* a light ray's self-shadow into a layered stack of far-boundary depths
//! and cumulative transmittance `T = product(1 - alpha)`. A shading pass then
//! *decodes* it: for a receiver at light-space `depth`, what fraction `T` of the
//! light survives to reach it? This crate is the on-device twin of that decode:
//! one thread per receiver query walks the identical bracket-and-interpolate the
//! golden does, so a passing real-device parity test is direct evidence the
//! ported decode reads the same transmittance as the reference — not merely that
//! its shader compiles.
//!
//! It is the read side of the [`deep_opacity`](crate::deep_opacity) packing
//! (pack once, decode per shaded receiver) and the **multiplicative dual** of
//! [`forward_scatter_sample`](crate::forward_scatter_sample): the same layered
//! bracket-and-interpolate, but the packed curve is the monotonically
//! non-increasing transmittance `T` in `0..=1` rather than the additive,
//! uncapped crossing count, so the out-of-band reads are fully transmissive
//! (`1.0`, fully lit) instead of `0`. Unlike the discrete prefix walk in
//! [`voxel_transmittance`](crate::voxel_transmittance), it decodes a layered
//! curve with **linear interpolation** between the two bracketing boundaries — a
//! genuinely distinct sampling kernel.
//!
//! # Batched decode shape
//!
//! [`GpuHairDeepTransmittanceSample::eval`] takes a batch of packed
//! [`DeepOpacityLayers`] curves and a batch of [`TransmittanceQuery`] receivers
//! (each naming its curve index and depth), and returns one transmittance per
//! query. The curves are flattened into shared `layer_depths` /
//! `layer_transmittance` pools with a compacted per-curve `[start, count)`
//! range; one thread per query resolves its curve and writes one output, so
//! every query is independent and race-free.
//!
//! # Correctness model
//!
//! The decode mirrors the golden branch-for-branch (empty curve and a receiver
//! in front of the frontmost boundary read `1.0`; a receiver at or beyond the
//! deepest boundary reads the last transmittance; otherwise the first window
//! with `depth <= d1` interpolates `t0 + (t1 - t0) * f`). The interpolation is a
//! closed-form multiply/add/divide with no transcendental call, so `CPU` and
//! `GPU` evaluate the same arithmetic but are not bit-exact — a `GPU` may fuse a
//! multiply-add, perturbing the low mantissa bits by a few `ULP`. The parity
//! test therefore asserts a per-value tolerance (`abs_diff < 1e-4` or `rel_diff
//! < 1e-3`) while the front/empty/beyond reads are exact.
//!
//! # Portability
//!
//! The kernel uses only multiply/add/divide and comparisons in the portable
//! core-`WGSL` subset — no `exp`, `pow` or optional device feature — so the twin
//! runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard deep-opacity transmittance curve decode
//! (bracket-and-interpolate over cumulative transmittance layers) plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::deep_transmittance::DeepOpacityLayers;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One receiver query: which packed curve to decode and the receiver's
/// light-space depth.
///
/// `curve` indexes the `curves` slice handed to
/// [`GpuHairDeepTransmittanceSample::eval`]; an out-of-range index decodes to
/// `1.0` (no occluders) rather than panicking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransmittanceQuery {
    /// Index into the batch of packed curves.
    pub curve: u32,
    /// Receiver light-space depth to decode the transmittance at.
    pub depth: f32,
}

/// Uniform decode parameters. `16`-byte scalar-packed `repr(C)` matching
/// `HairDeepSampleParams` in `shaders/deep_transmittance_sample.wesl` (padded to
/// a multiple of 16 for the uniform block).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    query_count: u32,
    curve_count: u32,
    pad0: u32,
    pad1: u32,
}

/// One compacted per-curve slice descriptor uploaded to the kernel. `8`-byte
/// `repr(C)` matching `HairDeepCurveRange` in
/// `shaders/deep_transmittance_sample.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCurveRange {
    start: u32,
    count: u32,
}

/// One receiver query uploaded to the kernel. `8`-byte `repr(C)` matching
/// `HairDeepQuery` in `shaders/deep_transmittance_sample.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    curve: u32,
    depth: f32,
}

/// A compiled, reusable deep-opacity transmittance decode pipeline.
pub struct GpuHairDeepTransmittanceSample {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairDeepTransmittanceSample {
    /// Compiles the deep-opacity transmittance decode kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairDeepTransmittanceSample {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_deep_transmittance_sample"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/deep_transmittance_sample.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_deep_transmittance_sample_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_deep_transmittance_sample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_deep_transmittance_sample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairDeepTransmittanceSample {
            module,
            layout,
            pipeline,
        }
    }

    /// Decodes the surviving transmittance each [`TransmittanceQuery`] receiver
    /// sees in its packed [`DeepOpacityLayers`] curve.
    ///
    /// The result equals
    /// [`sample_transmittance`](prism_render_architecture::hair::deep_transmittance::sample_transmittance)
    /// applied to `curves[query.curve]` at `query.depth`, to within the
    /// fused-multiply-add tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector without a dispatch (storage
    /// buffers cannot be zero-sized); a query naming an out-of-range curve
    /// decodes to `1.0` (fully transmissive).
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        curves: &[DeepOpacityLayers],
        queries: &[TransmittanceQuery],
    ) -> Vec<f32> {
        // No queries: nothing to decode; mirror the empty result without
        // touching the device (buffers cannot be zero-sized).
        if queries.is_empty() {
            return Vec::new();
        }

        // Flatten every curve's parallel depth/transmittance arrays into shared
        // pools with a compacted per-curve range. A curve's two arrays are
        // parallel and equal length in the golden type; use the shorter to stay
        // panic-free on a malformed pair.
        let mut depths: Vec<f32> = Vec::new();
        let mut transmittance: Vec<f32> = Vec::new();
        let mut ranges: Vec<GpuCurveRange> = Vec::with_capacity(curves.len());
        for curve in curves {
            let count = curve
                .layer_depths
                .len()
                .min(curve.layer_transmittance.len());
            let start = depths.len() as u32;
            depths.extend_from_slice(&curve.layer_depths[..count]);
            transmittance.extend_from_slice(&curve.layer_transmittance[..count]);
            ranges.push(GpuCurveRange {
                start,
                count: count as u32,
            });
        }

        // Storage buffers cannot be zero-sized. Pad each empty pool with a
        // single dummy entry; the `count == 0` ranges (and the `curve_count`
        // guard) keep every query off the dummy.
        if depths.is_empty() {
            depths.push(0.0);
            transmittance.push(1.0);
        }
        if ranges.is_empty() {
            ranges.push(GpuCurveRange { start: 0, count: 0 });
        }

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                curve: q.curve,
                depth: q.depth,
            })
            .collect();

        let uniform = Params {
            query_count: queries.len() as u32,
            curve_count: curves.len() as u32,
            pad0: 0,
            pad1: 0,
        };

        let device = ctx.device();
        let out_bytes = (queries.len() * size_of::<f32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_deep_transmittance_sample_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let ranges_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_deep_transmittance_sample_ranges"),
            contents: bytemuck::cast_slice(&ranges),
            usage: BufferUsages::STORAGE,
        });
        let depths_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_deep_transmittance_sample_depths"),
            contents: bytemuck::cast_slice(&depths),
            usage: BufferUsages::STORAGE,
        });
        let transmittance_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_deep_transmittance_sample_transmittance"),
            contents: bytemuck::cast_slice(&transmittance),
            usage: BufferUsages::STORAGE,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_deep_transmittance_sample_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_deep_transmittance_sample_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_deep_transmittance_sample_out_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_deep_transmittance_sample_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: ranges_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: depths_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: transmittance_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_deep_transmittance_sample_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_deep_transmittance_sample_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        read_f32(&out_stage)
    }
}

/// Reads a mapped staging buffer back into an owned `f32` vector, then unmaps.
fn read_f32(stage: &wgpu::Buffer) -> Vec<f32> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    stage.unmap();
    out
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
