//! `wgpu` compute twin of the per-sample firefly (outlier) luminance clamp from
//! the `CPU` golden `prism_render_architecture::reference_pt::firefly`.
//!
//! Unbiased path tracing occasionally returns a single sample with enormous
//! energy — a near-grazing specular bounce onto a small bright emitter, a
//! low-probability caustic, or a sampling combination with a tiny density in
//! the denominator. Averaged over a finite budget those spikes survive as
//! isolated bright "firefly" pixels. The reference `FireflyClamp::apply` trades
//! a small bounded bias for a large noise reduction by scaling an over-bright
//! sample down so its luminance equals a configured maximum while preserving
//! its chromaticity. This module ports that stateless, no-`RNG` policy onto the
//! device: one thread clamps one radiance sample.
//!
//! [`GpuFireflyClamp`] is the on-device twin, so a passing real-device parity
//! test is direct evidence the ported kernel takes the same `Off` /
//! non-positive-maximum / below-threshold / scaled branch the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `luminance(v) = 0.2126 r + 0.7152 g +
//! 0.0722 b` and `FireflyClamp::apply`: `Off` (`mode = 0`) passes the sample
//! through; `MaxLuminance` (`mode = 1`) passes it through when the maximum is
//! non-positive or the luminance is already at or below it, and otherwise
//! scales every channel by `max_lum / luma` so the clamped luminance equals the
//! maximum and the channel ratios (hue and saturation) are preserved. There is
//! no loop: each thread performs a fixed, bounded sequence of arithmetic, so
//! the kernel provably terminates.
//!
//! # Correctness model
//!
//! Every quantity threads through multiplies, adds and one guarded division, so
//! `CPU` and `GPU` are not necessarily bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous channel; the discrete `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! A `mode` outside `{0, 1}` has no reference meaning, so the twin reports
//! `valid = 0` with cleared outputs. In the scaling branch the maximum is
//! strictly positive and the luminance strictly larger, so the divisor is
//! safe; a `select` guard keeps every lane on the same branch arithmetic and
//! never divides by zero. An empty query batch short-circuits on the host with
//! no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `select`, `+ - * /`,
//! `u32` comparisons and ordered `f32` comparisons — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no `smoothstep`, no `round`, no `copysign` and
//! no optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::firefly`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` firefly luminance-clamp kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `luminance` and `FireflyClamp::apply` branch for
/// branch; see the module documentation for the algorithm.
const FIREFLY_LUMINANCE_CLAMP_WGSL: &str = r#"
// Firefly luminance-clamp twin: one thread per query reproduces luminance(v)
// and FireflyClamp::apply, scaling an over-bright sample down to a target
// luminance while preserving chromaticity. It mirrors the CPU golden branch for
// branch, uses only the portable core-WGSL subset (select and + - * / plus u32
// and ordered f32 comparisons), takes no optional feature, and has no loop, so
// the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::reference_pt::firefly；无第三方引擎源码
// 或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Clamp policy: 0 = Off, 1 = MaxLuminance; any other value is invalid.
    mode: u32,
    // Maximum luminance for the MaxLuminance policy; non-positive disables it.
    max_lum: f32,
    // Linear RGB radiance sample.
    vx: f32,
    vy: f32,
    vz: f32,
}

struct Result {
    // Clamped radiance sample; zero when invalid.
    ox: f32,
    oy: f32,
    oz: f32,
    // 1 when the mode is a known policy, 0 for an unknown mode.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Standard luma coefficients summing to one, so a unit-white sample has unit
// luminance.
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

// Relative luminance of a linear RGB radiance value.
fn luminance(v: vec3<f32>) -> f32 {
    return LUMA_R * v.x + LUMA_G * v.y + LUMA_B * v.z;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let value = vec3<f32>(q.vx, q.vy, q.vz);

    var out: Result;
    out.ox = 0.0;
    out.oy = 0.0;
    out.oz = 0.0;
    out.valid = 0u;

    // A mode outside {Off, MaxLuminance} has no reference meaning.
    if (q.mode >= 2u) {
        results[idx] = out;
        return;
    }

    // Off (mode == 0) passes the sample through unchanged. MaxLuminance
    // (mode == 1) scales only when the maximum is strictly positive and the
    // luminance strictly exceeds it; then max_lum > 0 and luma > max_lum, so
    // the divisor is strictly positive. The select guard keeps every lane on
    // the same branch arithmetic and never divides by zero.
    let luma = luminance(value);
    let clamp_active = (q.mode == 1u) && (q.max_lum > 0.0) && (luma > q.max_lum);
    let safe_luma = select(1.0, luma, clamp_active);
    let scale = select(1.0, q.max_lum / safe_luma, clamp_active);
    let clamped = value * scale;

    out.ox = clamped.x;
    out.oy = clamped.y;
    out.oz = clamped.z;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`FIREFLY_LUMINANCE_CLAMP_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The `vec3` sample is flattened to scalar lanes; the kernel rebuilds the
/// `vec3<f32>`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    mode: u32,
    max_lum: f32,
    vx: f32,
    vy: f32,
    vz: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Four `4`-byte scalars fill one `16`-byte slot exactly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    ox: f32,
    oy: f32,
    oz: f32,
    valid: u32,
}

/// One query for the firefly luminance-clamp twin: the clamp `mode`, the
/// `max_lum` target and the linear `RGB` radiance `value`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FireflyClampQuery {
    /// Clamp policy: `0` = `Off`, `1` = `MaxLuminance`; any other value is
    /// treated as invalid and reports `valid = 0`.
    pub mode: u32,
    /// Maximum luminance for the `MaxLuminance` policy; a non-positive value
    /// disables the clamp (treated like `Off`).
    pub max_lum: f32,
    /// Linear `RGB` radiance sample to clamp.
    pub value: [f32; 3],
}

impl FireflyClampQuery {
    /// Builds a query from the clamp mode, the maximum luminance and the
    /// radiance sample.
    #[must_use]
    pub fn new(mode: u32, max_lum: f32, value: [f32; 3]) -> FireflyClampQuery {
        FireflyClampQuery {
            mode,
            max_lum,
            value,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `FireflyClamp::apply` output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FireflyClampResult {
    /// The clamped radiance sample; cleared to zero when invalid.
    pub clamped: [f32; 3],
    /// `1` when the mode is a known policy, `0` for an unknown mode.
    pub valid: u32,
}

/// Encodes one [`FireflyClampQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &FireflyClampQuery) -> GpuQuery {
    GpuQuery {
        mode: q.mode,
        max_lum: q.max_lum,
        vx: q.value[0],
        vy: q.value[1],
        vz: q.value[2],
    }
}

/// Decodes one packed [`GpuResult`] into the public [`FireflyClampResult`].
fn decode_result(raw: &GpuResult) -> FireflyClampResult {
    FireflyClampResult {
        clamped: [raw.ox, raw.oy, raw.oz],
        valid: raw.valid,
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

/// A compiled, reusable firefly luminance-clamp compute pipeline, twinning the
/// `CPU` golden `luminance` and `FireflyClamp::apply`.
pub struct GpuFireflyClamp {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFireflyClamp {
    /// Compiles the firefly luminance-clamp kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFireflyClamp {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_firefly_luminance_clamp"),
            source: ShaderSource::Wgsl(FIREFLY_LUMINANCE_CLAMP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_firefly_luminance_clamp_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_firefly_luminance_clamp_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_firefly_luminance_clamp_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFireflyClamp {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`FireflyClampResult`]
    /// per input, in order.
    ///
    /// Each continuous channel matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[FireflyClampQuery],
    ) -> Vec<FireflyClampResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_firefly_luminance_clamp_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_firefly_luminance_clamp_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_firefly_luminance_clamp_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_firefly_luminance_clamp_bind_group"),
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
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_firefly_luminance_clamp_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_firefly_luminance_clamp_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_firefly_luminance_clamp_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
