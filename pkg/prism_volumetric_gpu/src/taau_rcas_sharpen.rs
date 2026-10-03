//! `wgpu` compute twin of the Robust Contrast-Adaptive Sharpening (`RCAS`)
//! per-pixel resolve inside the temporal upscaler
//! ([`sharpen`](prism_render_architecture::temporal_upscale::sharpen)).
//!
//! The `CPU` golden
//! [`rcas`](prism_render_architecture::temporal_upscale::sharpen::rcas) is
//! `AMD`'s `RCAS` sharpening filter (the pass shipped with `FSR`): a `5`-tap
//! cross whose single sharpening coefficient — the *lobe* — is derived per
//! pixel from the local ring contrast so flat regions are left untouched and
//! only edges are sharpened, with the lobe limited by
//! [`RCAS_LIMIT`](prism_render_architecture::temporal_upscale::sharpen::RCAS_LIMIT)
//! so a channel can never blow out into a halo. Every operation is `+`, `-`,
//! `*`, `/`, `min`/`max` and `abs`, so a `GPU` kernel reproduces the result
//! without any transcendental.
//!
//! [`GpuTaauRcasSharpen`] is the on-device twin of that per-pixel closed form:
//! one thread sharpens one pixel from its [`CrossTaps`] cross
//! (`center` plus the `north`/`south`/`west`/`east` ring), the `sharpness`
//! knob and the `denoise` flag, reproducing the reference's exact arithmetic so
//! a passing real-device parity test is direct evidence the ported kernel
//! computes the same sharpened color the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For one pixel the twin reproduces the whole scalar core of
//! [`rcas`](prism_render_architecture::temporal_upscale::sharpen::rcas):
//!
//! - the per-channel ring minimum and maximum over the four neighbor taps
//!   (`ring_min_max`);
//! - the per-channel sharpening lobe candidate `channel_lobe`, the headroom
//!   `max(-hit_min, hit_max)` with both reciprocals guarded against the
//!   degenerate flat-ring case;
//! - the most conservative (largest, closest to zero) lobe across the three
//!   channels, clamped to `[-RCAS_LIMIT, 0]` and scaled by `sharpness`;
//! - the optional luma-spike `denoise_weight` attenuation (built from the
//!   `Rec. 709` [`luminance`](prism_render_architecture::temporal_upscale::color::luminance)
//!   of the center and the ring);
//! - the normalized weighted combine
//!   `(center + lobe * ring_sum) / (1 + 4 * lobe)` per channel.
//!
//! # What stays on the host
//!
//! The variable-length aggregate around a single pixel stays on the host: the
//! image-wide tap gather with clamp addressing, the per-pixel `[0, 1)` tone-map
//! domain conversion, the final largest-channel clamp below `1`, and the
//! variable-length batch assembly. The host enqueues one
//! [`TaauRcasSharpenQuery`] per pixel it wants sharpened, so a storage buffer
//! is never zero-sized; an empty batch short-circuits on the host with no
//! dispatch.
//!
//! # Correctness model
//!
//! Every quantity threads through subtracts, reciprocals and multiplies, so the
//! `CPU` and `GPU` are not bit-exact: a `GPU` reciprocal may land a few units in
//! the last place from the scalar reference. The parity test therefore asserts
//! a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on each output
//! channel, tight enough to catch a genuinely wrong port (a dropped clamp, a
//! swapped tap, a missing `denoise` term) yet loose enough to admit a legal
//! last-place reciprocal difference. The lobe clamp keeps the denominator
//! `1 + 4 * lobe` at or above `0.25`, so the reciprocal is always well-defined;
//! the `denoise` flag crosses the device boundary as a [`u32`] and is read back
//! exactly.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `round`, no `smoothstep`.
//! No optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. There is no loop: each thread performs a fixed, bounded
//! sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::sharpen`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `RCAS` per-pixel kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`rcas`](prism_render_architecture::temporal_upscale::sharpen::rcas) per-pixel
/// closed form; see the module documentation for the algorithm.
const TAAU_RCAS_SHARPEN_WGSL: &str = r#"
// RCAS per-pixel twin: one thread sharpens one pixel from its 5-tap cross
// (center + north/south/west/east ring), a sharpness knob and a denoise flag,
// mirroring the CPU golden `temporal_upscale::sharpen::rcas` closed form with
// only min/max/abs and + - * /. It owns no image-wide tap gather, no tone-map
// domain conversion and no final peak clamp; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::temporal_upscale::sharpen；无第三方
// 引擎源码或衍生代码。

// AMD's RCAS lobe magnitude cap: 0.25 - 1/16 = 0.1875, so the normalizing
// denominator 1 + 4 * lobe stays at or above 0.25.
const RCAS_LIMIT: f32 = 0.25 - 1.0 / 16.0;

struct Params {
    // Number of pixels in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // center.rgb
    cx: f32, cy: f32, cz: f32,
    // north.rgb
    nx: f32, ny: f32, nz: f32,
    // south.rgb
    sx: f32, sy: f32, sz: f32,
    // west.rgb
    wx: f32, wy: f32, wz: f32,
    // east.rgb
    ex: f32, ey: f32, ez: f32,
    // Normalized [0, 1] sharpness knob.
    sharpness: f32,
    // 1 enables the luma-spike denoise attenuation, 0 disables it.
    denoise: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Sharpened color.rgb for this pixel.
    r: f32,
    g: f32,
    b: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Rec. 709 relative luminance, matching `temporal_upscale::color::luminance`.
fn luminance(r: f32, g: f32, b: f32) -> f32 {
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// Per-channel sharpening lobe candidate (always <= 0): the least aggressive of
// the headroom toward black and the headroom toward the peak of 1, with both
// reciprocals guarded against the degenerate flat-ring case.
fn channel_lobe(mn: f32, mx: f32) -> f32 {
    var hit_min: f32 = 0.0;
    if (mx > 0.0) {
        hit_min = mn / (4.0 * mx);
    }
    let denom = 4.0 * (mn - 1.0);
    var hit_max: f32 = 0.0;
    if (denom < 0.0) {
        hit_max = (1.0 - mx) / denom;
    }
    return max(-hit_min, hit_max);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Per-channel ring min/max over the four neighbor taps (north, south, west,
    // east), seeded from north and folded over the other three.
    var mn_r = q.nx;
    var mx_r = q.nx;
    var mn_g = q.ny;
    var mx_g = q.ny;
    var mn_b = q.nz;
    var mx_b = q.nz;

    mn_r = min(mn_r, q.sx);
    mx_r = max(mx_r, q.sx);
    mn_g = min(mn_g, q.sy);
    mx_g = max(mx_g, q.sy);
    mn_b = min(mn_b, q.sz);
    mx_b = max(mx_b, q.sz);

    mn_r = min(mn_r, q.wx);
    mx_r = max(mx_r, q.wx);
    mn_g = min(mn_g, q.wy);
    mx_g = max(mx_g, q.wy);
    mn_b = min(mn_b, q.wz);
    mx_b = max(mx_b, q.wz);

    mn_r = min(mn_r, q.ex);
    mx_r = max(mx_r, q.ex);
    mn_g = min(mn_g, q.ey);
    mx_g = max(mx_g, q.ey);
    mn_b = min(mn_b, q.ez);
    mx_b = max(mx_b, q.ez);

    // The lobe is the most conservative (closest to zero, i.e. largest) of the
    // three per-channel limits so no channel clips.
    let lobe_r = channel_lobe(mn_r, mx_r);
    let lobe_g = channel_lobe(mn_g, mx_g);
    let lobe_b = channel_lobe(mn_b, mx_b);
    var lobe = max(max(lobe_r, lobe_g), lobe_b);

    // Clamp to the RCAS range [-RCAS_LIMIT, 0] (spelled with min/max) and scale
    // by the sharpness knob.
    lobe = min(max(lobe, -RCAS_LIMIT), 0.0) * q.sharpness;

    // Optional luma-spike denoise attenuation in [0.5, 1].
    if (q.denoise != 0u) {
        let center_l = luminance(q.cx, q.cy, q.cz);
        let ln = luminance(q.nx, q.ny, q.nz);
        let ls = luminance(q.sx, q.sy, q.sz);
        let lw = luminance(q.wx, q.wy, q.wz);
        let le = luminance(q.ex, q.ey, q.ez);
        let mean = 0.25 * (ln + ls + lw + le);
        var lo = center_l;
        var hi = center_l;
        lo = min(lo, ln);
        hi = max(hi, ln);
        lo = min(lo, ls);
        hi = max(hi, ls);
        lo = min(lo, lw);
        hi = max(hi, lw);
        lo = min(lo, le);
        hi = max(hi, le);
        let range = hi - lo;
        // Degenerate (non-finite or non-positive) range disables the attenuation.
        // `!(range > 0.0)` is true for both NaN and range <= 0, matching the
        // golden's `range.is_nan() || range <= 0.0` guard without any `==`.
        var weight: f32 = 1.0;
        if (range > 0.0) {
            let spike = min(abs(center_l - mean) / range, 1.0);
            weight = 1.0 - 0.5 * spike;
        }
        lobe = lobe * weight;
    }

    // Weighted combine: center weight 1, each ring tap weight `lobe` (negative),
    // normalized by the weight sum 1 + 4 * lobe. The clamped lobe keeps the
    // denominator at or above 0.25, so the reciprocal is always well-defined.
    let norm = 1.0 / (4.0 * lobe + 1.0);
    let ring_sum_r = q.nx + q.sx + q.wx + q.ex;
    let ring_sum_g = q.ny + q.sy + q.wy + q.ey;
    let ring_sum_b = q.nz + q.sz + q.wz + q.ez;

    var out: Result;
    out.r = (q.cx + lobe * ring_sum_r) * norm;
    out.g = (q.cy + lobe * ring_sum_g) * norm;
    out.b = (q.cz + lobe * ring_sum_b) * norm;
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the pixel count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`TAAU_RCAS_SHARPEN_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid pixels in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one pixel query: the five cross taps as
/// `15` scalar channels, the `sharpness` knob, the `denoise` flag as a [`u32`],
/// and three pad words to an `80`-byte (`16`-byte-multiple) stride matching the
/// `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `center` red.
    cx: f32,
    /// `center` green.
    cy: f32,
    /// `center` blue.
    cz: f32,
    /// `north` red.
    nx: f32,
    /// `north` green.
    ny: f32,
    /// `north` blue.
    nz: f32,
    /// `south` red.
    sx: f32,
    /// `south` green.
    sy: f32,
    /// `south` blue.
    sz: f32,
    /// `west` red.
    wx: f32,
    /// `west` green.
    wy: f32,
    /// `west` blue.
    wz: f32,
    /// `east` red.
    ex: f32,
    /// `east` green.
    ey: f32,
    /// `east` blue.
    ez: f32,
    /// Normalized `[0, 1]` sharpness knob.
    sharpness: f32,
    /// `1` enables the luma-spike denoise attenuation, `0` disables it.
    denoise: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one sharpened pixel: the output color's three
/// channels plus one pad word to a `16`-byte stride matching the `WGSL`
/// `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Sharpened red channel.
    r: f32,
    /// Sharpened green channel.
    g: f32,
    /// Sharpened blue channel.
    b: f32,
    /// Padding word.
    pad0: f32,
}

/// One per-pixel query for the `RCAS` sharpening twin: the five cross taps, the
/// `sharpness` knob and the `denoise` flag.
///
/// The taps form the plus/cross pattern centered on the pixel being sharpened
/// (`north`/`south`/`west`/`east` surrounding `center`), expected in the bounded
/// tone-mapped domain the golden
/// [`rcas`](prism_render_architecture::temporal_upscale::sharpen::rcas) documents.
/// The host owns the image-wide tap gather and the surrounding domain
/// conversion, and enqueues one [`TaauRcasSharpenQuery`] per pixel it wants
/// sharpened.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauRcasSharpenQuery {
    /// The pixel being sharpened; the only tap whose kernel weight is positive.
    pub center: [f32; 3],
    /// The tap one pixel above `center`.
    pub north: [f32; 3],
    /// The tap one pixel below `center`.
    pub south: [f32; 3],
    /// The tap one pixel left of `center`.
    pub west: [f32; 3],
    /// The tap one pixel right of `center`.
    pub east: [f32; 3],
    /// Normalized `[0, 1]` sharpness knob: `0` returns `center` unchanged, `1`
    /// applies the full contrast-limited lobe.
    pub sharpness: f32,
    /// When `true`, the lobe is attenuated in luma-spike regions.
    pub denoise: bool,
}

impl TaauRcasSharpenQuery {
    /// Builds a query from the five cross taps, the `sharpness` knob and the
    /// `denoise` flag.
    #[must_use]
    pub const fn new(
        center: [f32; 3],
        north: [f32; 3],
        south: [f32; 3],
        west: [f32; 3],
        east: [f32; 3],
        sharpness: f32,
        denoise: bool,
    ) -> TaauRcasSharpenQuery {
        TaauRcasSharpenQuery {
            center,
            north,
            south,
            west,
            east,
            sharpness,
            denoise,
        }
    }
}

/// One resolved pixel of the `RCAS` sharpening twin, mirroring the `[f32; 3]`
/// color the golden
/// [`rcas`](prism_render_architecture::temporal_upscale::sharpen::rcas) returns.
///
/// The color is in the same (tone-mapped) domain as the input; it may overshoot
/// slightly above the ring maximum or below `0` at an edge, which is the
/// intended sharpening response. The caller clamps the largest channel back
/// below `1` before `untonemap`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauRcasSharpenResult {
    /// The sharpened color for this pixel.
    pub color: [f32; 3],
}

/// Encodes one [`TaauRcasSharpenQuery`] into its `std430` [`GpuQuery`] slot,
/// turning the `denoise` [`bool`] into a `0`/`1` [`u32`].
fn encode_query(q: &TaauRcasSharpenQuery) -> GpuQuery {
    GpuQuery {
        cx: q.center[0],
        cy: q.center[1],
        cz: q.center[2],
        nx: q.north[0],
        ny: q.north[1],
        nz: q.north[2],
        sx: q.south[0],
        sy: q.south[1],
        sz: q.south[2],
        wx: q.west[0],
        wy: q.west[1],
        wz: q.west[2],
        ex: q.east[0],
        ey: q.east[1],
        ez: q.east[2],
        sharpness: q.sharpness,
        denoise: u32::from(q.denoise),
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`TaauRcasSharpenResult`].
fn decode_result(raw: &GpuResult) -> TaauRcasSharpenResult {
    TaauRcasSharpenResult {
        color: [raw.r, raw.g, raw.b],
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

/// A compiled, reusable `RCAS` per-pixel compute pipeline, twinning the scalar
/// core of the `CPU` golden
/// [`rcas`](prism_render_architecture::temporal_upscale::sharpen::rcas).
pub struct GpuTaauRcasSharpen {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTaauRcasSharpen {
    /// Compiles the `RCAS` per-pixel kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTaauRcasSharpen {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_taau_rcas_sharpen"),
            source: ShaderSource::Wgsl(TAAU_RCAS_SHARPEN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_taau_rcas_sharpen_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_taau_rcas_sharpen_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_taau_rcas_sharpen_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTaauRcasSharpen {
            module,
            layout,
            pipeline,
        }
    }

    /// Sharpens every pixel in `queries` and returns one
    /// [`TaauRcasSharpenResult`] per input, in order.
    ///
    /// Each channel equals the reference within the tolerance documented on this
    /// module. An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TaauRcasSharpenQuery],
    ) -> Vec<TaauRcasSharpenResult> {
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
            label: Some("prism_volumetric_taau_rcas_sharpen_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_taau_rcas_sharpen_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_taau_rcas_sharpen_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_taau_rcas_sharpen_bind_group"),
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
            label: Some("prism_volumetric_taau_rcas_sharpen_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_taau_rcas_sharpen_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_taau_rcas_sharpen_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pixel, flattened to a 1-D dispatch.
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
