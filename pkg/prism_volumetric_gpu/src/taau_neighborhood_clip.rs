//! `wgpu` compute twin of the per-pixel history-clipping core inside the
//! temporal-upscale neighborhood contract
//! ([`neighborhood`](prism_render_architecture::temporal_upscale::neighborhood)).
//!
//! The anti-ghosting rule of a temporal upsampler is that a reprojected history
//! color is only trustworthy if it resembles the current frame's local
//! neighborhood. The `CPU` golden
//! [`NeighborhoodStats`](prism_render_architecture::temporal_upscale::neighborhood::NeighborhoodStats)
//! summarizes a gathered window of current-frame samples as a per-channel
//! `min`/`max`/`mean`/`m2` region, derives Marco Salvi's variance box
//! `mean +/- gamma * stddev` (intersected with the hard `min`/`max`), and then
//! clips the history into that box along the ray toward the box center
//! (Karis "clip to `AABB`") with
//! [`clip_to_aabb`](prism_render_architecture::temporal_upscale::neighborhood::clip_to_aabb).
//!
//! [`GpuTaauNeighborhoodClip`] is the on-device twin of the *numeric core* of
//! that pipeline: given a window already reduced to its moments, one thread
//! reproduces one output pixel's `stddev`, variance box, and clipped history,
//! using only `+`, `-`, `*`, `/`, `min`, `max`, and `sqrt`.
//!
//! # What is twinned
//!
//! Per output pixel the kernel reproduces, from the window moments `mean`,
//! `m2`, `hard_min`, `hard_max`, a `gamma`, and a `history_point`:
//! - the per-channel standard deviation
//!   `stddev = sqrt(max(m2 - mean * mean, 0))`, mirroring
//!   [`NeighborhoodStats::stddev`](prism_render_architecture::temporal_upscale::neighborhood::NeighborhoodStats::stddev);
//! - the variance box `lo = max(mean - gamma * stddev, hard_min)` and
//!   `hi = min(mean + gamma * stddev, hard_max)`, with a negative or zero
//!   `gamma` treated as `0`, mirroring
//!   [`NeighborhoodStats::variance_aabb`](prism_render_architecture::temporal_upscale::neighborhood::NeighborhoodStats::variance_aabb);
//! - the clipped history `clip_to_aabb(lo, hi, history_point)`, the Karis
//!   clip-toward-center form `t = max_c(|v_c| / extent_c)` with `center + v / t`
//!   on the box surface when `t > 1`, mirroring
//!   [`clip_to_aabb`](prism_render_architecture::temporal_upscale::neighborhood::clip_to_aabb).
//!
//! # What stays on the host
//!
//! The variable-length window reduction
//! [`NeighborhoodStats::from_samples`](prism_render_architecture::temporal_upscale::neighborhood::NeighborhoodStats::from_samples)
//! is a one-pass accumulation over a slice with no fixed-width device analogue,
//! so it stays on the host; the host feeds the kernel the already-reduced
//! moments. The empty-window degenerate case (no neighborhood to summarize)
//! also stays on the host: an empty batch short-circuits before any dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! Every output is a continuous quantity threaded through subtracts, a `sqrt`,
//! and a divide, so the `CPU` and `GPU` are not bit-exact: a `GPU` `sqrt` or
//! reciprocal may land a few units in the last place from the scalar reference.
//! The parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on `lo`, `hi`, and `clipped`, tight enough to catch a
//! genuinely wrong port yet loose enough to admit a legal last-place
//! difference. Fixtures use non-degenerate boxes and keep the history clearly
//! inside or clearly outside the box, away from the `t = 1` surface tie.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `sqrt`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`, `round`
//! or `cbrt`. The golden's zero-extent axis sentinel `f32::INFINITY` (force a
//! clip straight to the center) is reproduced with a large finite `t` so the
//! reciprocal collapses the offset toward zero without ever constructing a
//! `NaN`; the logic is equivalent. No optional device feature is required, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each
//! thread performs a fixed, bounded sequence of arithmetic.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::neighborhood`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` neighborhood history-clip per-pixel kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`neighborhood`](prism_render_architecture::temporal_upscale::neighborhood)
/// `stddev` / `variance_aabb` / `clip_to_aabb` closed form; see the module
/// documentation for the algorithm.
const TAAU_NEIGHBORHOOD_CLIP_WGSL: &str = r#"
// Temporal-upscale neighborhood history-clip twin: one thread turns one output
// pixel's reduced window moments into the standard deviation, Marco Salvi
// variance box, and Karis clip-toward-center history, mirroring the CPU golden
// `temporal_upscale::neighborhood` closed form with only min/max/abs/sqrt and
// + - * /. The variable-length window reduction `from_samples` stays host-side.
//
// Provenance: 孪生自本仓 prism_render_architecture::temporal_upscale::neighborhood；
// 无第三方引擎源码或衍生代码。

// Large finite stand-in for the golden's `f32::INFINITY` zero-extent-axis
// sentinel: with t this large the reciprocal collapses the offset toward zero,
// clipping the point to the box center without ever constructing a NaN.
const CLIP_FORCE: f32 = 1.0e30;

struct Params {
    // Number of pixels in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Component-wise window mean.
    mean0: f32,
    mean1: f32,
    mean2: f32,
    // Component-wise window mean of squares (second raw moment).
    m20: f32,
    m21: f32,
    m22: f32,
    // Hard axis-aligned box lower bound (window min).
    min0: f32,
    min1: f32,
    min2: f32,
    // Hard axis-aligned box upper bound (window max).
    max0: f32,
    max1: f32,
    max2: f32,
    // Reprojected history point to clip into the variance box.
    hist0: f32,
    hist1: f32,
    hist2: f32,
    // Variance-box scale; a negative or zero value clips straight to the mean.
    gamma: f32,
}

struct Result {
    // Variance-box lower bound, intersected with the hard min.
    lo0: f32,
    lo1: f32,
    lo2: f32,
    // Variance-box upper bound, intersected with the hard max.
    hi0: f32,
    hi1: f32,
    hi2: f32,
    // History clipped into the variance box toward its center.
    clipped0: f32,
    clipped1: f32,
    clipped2: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let mean = vec3<f32>(q.mean0, q.mean1, q.mean2);
    let m2 = vec3<f32>(q.m20, q.m21, q.m22);
    let hard_min = vec3<f32>(q.min0, q.min1, q.min2);
    let hard_max = vec3<f32>(q.max0, q.max1, q.max2);
    let history = vec3<f32>(q.hist0, q.hist1, q.hist2);

    // stddev = sqrt(max(m2 - mean^2, 0)): the inner max guards the tiny negative
    // variance that floating-point cancellation can produce before the sqrt.
    let variance = max(m2 - mean * mean, vec3<f32>(0.0, 0.0, 0.0));
    let stddev = sqrt(variance);

    // A negative or zero gamma is treated as 0 (clip straight to the mean),
    // mirroring the golden `if gamma > 0 { gamma } else { 0 }`.
    var g = 0.0;
    if (q.gamma > 0.0) {
        g = q.gamma;
    }
    let spread = stddev * g;

    // Variance box centered on the mean, then intersected with the hard box so
    // the region can only tighten, never invent colors the window never held.
    let lo = max(mean - spread, hard_min);
    let hi = min(mean + spread, hard_max);

    // Karis clip toward the box center: scale the offset-from-center by the
    // largest per-axis overshoot t = max_c(|v_c| / extent_c); when t > 1 the
    // point is outside and center + v / t lands on the box surface.
    let center = (lo + hi) * 0.5;
    let extent = (hi - lo) * 0.5;
    let v = history - center;

    var t = 0.0;
    if (extent.x > 0.0) {
        t = max(t, abs(v.x) / extent.x);
    } else if (abs(v.x) > 0.0) {
        t = CLIP_FORCE;
    }
    if (extent.y > 0.0) {
        t = max(t, abs(v.y) / extent.y);
    } else if (abs(v.y) > 0.0) {
        t = CLIP_FORCE;
    }
    if (extent.z > 0.0) {
        t = max(t, abs(v.z) / extent.z);
    } else if (abs(v.z) > 0.0) {
        t = CLIP_FORCE;
    }

    var clipped = history;
    if (t > 1.0) {
        let inv_t = 1.0 / t;
        clipped = center + v * inv_t;
    }

    var out: Result;
    out.lo0 = lo.x;
    out.lo1 = lo.y;
    out.lo2 = lo.z;
    out.hi0 = hi.x;
    out.hi1 = hi.y;
    out.hi2 = hi.z;
    out.clipped0 = clipped.x;
    out.clipped1 = clipped.y;
    out.clipped2 = clipped.z;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the pixel count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`TAAU_NEIGHBORHOOD_CLIP_WGSL`].
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

/// `repr(C)` `std430` layout of one pixel query: the window moments, the hard
/// box, the history point, and the `gamma`, all as scalar `f32` lanes matching
/// the `WGSL` `Query` struct (`16` scalars, a `64`-byte stride).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Mean component `0`.
    mean0: f32,
    /// Mean component `1`.
    mean1: f32,
    /// Mean component `2`.
    mean2: f32,
    /// Mean-of-squares component `0`.
    m20: f32,
    /// Mean-of-squares component `1`.
    m21: f32,
    /// Mean-of-squares component `2`.
    m22: f32,
    /// Hard min component `0`.
    min0: f32,
    /// Hard min component `1`.
    min1: f32,
    /// Hard min component `2`.
    min2: f32,
    /// Hard max component `0`.
    max0: f32,
    /// Hard max component `1`.
    max1: f32,
    /// Hard max component `2`.
    max2: f32,
    /// History component `0`.
    hist0: f32,
    /// History component `1`.
    hist1: f32,
    /// History component `2`.
    hist2: f32,
    /// Variance-box scale `gamma`.
    gamma: f32,
}

/// `repr(C)` `std430` layout of one pixel result matching the `WGSL` `Result`
/// struct: the variance box `lo`/`hi` and the `clipped` history plus three pad
/// words to a `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Variance-box lower bound component `0`.
    lo0: f32,
    /// Variance-box lower bound component `1`.
    lo1: f32,
    /// Variance-box lower bound component `2`.
    lo2: f32,
    /// Variance-box upper bound component `0`.
    hi0: f32,
    /// Variance-box upper bound component `1`.
    hi1: f32,
    /// Variance-box upper bound component `2`.
    hi2: f32,
    /// Clipped history component `0`.
    clipped0: f32,
    /// Clipped history component `1`.
    clipped1: f32,
    /// Clipped history component `2`.
    clipped2: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// One per-pixel query for the neighborhood history-clip twin: the reduced
/// window moments, the hard axis-aligned box, the reprojected `history_point`,
/// and the variance-box scale `gamma`.
///
/// The host owns the variable-length window reduction
/// [`NeighborhoodStats::from_samples`](prism_render_architecture::temporal_upscale::neighborhood::NeighborhoodStats::from_samples)
/// and enqueues one [`TaauNeighborhoodClipQuery`] per output pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauNeighborhoodClipQuery {
    /// Component-wise window `mean`.
    pub mean: [f32; 3],
    /// Component-wise window mean of squares (`m2`), used to recover the
    /// variance as `m2 - mean * mean`.
    pub m2: [f32; 3],
    /// Hard box lower bound (window `min`).
    pub hard_min: [f32; 3],
    /// Hard box upper bound (window `max`).
    pub hard_max: [f32; 3],
    /// Reprojected history point to clip into the variance box.
    pub history_point: [f32; 3],
    /// Variance-box scale; a negative or zero value clips straight to the mean.
    pub gamma: f32,
}

/// One resolved output pixel of the neighborhood history clip: the variance box
/// `lo`/`hi` and the `clipped` history.
///
/// `lo` and `hi` mirror
/// [`NeighborhoodStats::variance_aabb`](prism_render_architecture::temporal_upscale::neighborhood::NeighborhoodStats::variance_aabb)
/// and `clipped` mirrors
/// [`clip_to_aabb`](prism_render_architecture::temporal_upscale::neighborhood::clip_to_aabb)
/// of `history_point` into `[lo, hi]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauNeighborhoodClipResult {
    /// Variance-box lower bound, intersected with the hard min.
    pub lo: [f32; 3],
    /// Variance-box upper bound, intersected with the hard max.
    pub hi: [f32; 3],
    /// History clipped into the variance box toward its center.
    pub clipped: [f32; 3],
}

/// Encodes one [`TaauNeighborhoodClipQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &TaauNeighborhoodClipQuery) -> GpuQuery {
    GpuQuery {
        mean0: q.mean[0],
        mean1: q.mean[1],
        mean2: q.mean[2],
        m20: q.m2[0],
        m21: q.m2[1],
        m22: q.m2[2],
        min0: q.hard_min[0],
        min1: q.hard_min[1],
        min2: q.hard_min[2],
        max0: q.hard_max[0],
        max1: q.hard_max[1],
        max2: q.hard_max[2],
        hist0: q.history_point[0],
        hist1: q.history_point[1],
        hist2: q.history_point[2],
        gamma: q.gamma,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`TaauNeighborhoodClipResult`].
fn decode_result(raw: &GpuResult) -> TaauNeighborhoodClipResult {
    TaauNeighborhoodClipResult {
        lo: [raw.lo0, raw.lo1, raw.lo2],
        hi: [raw.hi0, raw.hi1, raw.hi2],
        clipped: [raw.clipped0, raw.clipped1, raw.clipped2],
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

/// A compiled, reusable neighborhood history-clip per-pixel compute pipeline,
/// twinning the numeric core of the `CPU` golden
/// [`neighborhood`](prism_render_architecture::temporal_upscale::neighborhood).
pub struct GpuTaauNeighborhoodClip {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTaauNeighborhoodClip {
    /// Compiles the neighborhood history-clip kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTaauNeighborhoodClip {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_taau_neighborhood_clip"),
            source: ShaderSource::Wgsl(TAAU_NEIGHBORHOOD_CLIP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_taau_neighborhood_clip_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_taau_neighborhood_clip_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_taau_neighborhood_clip_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTaauNeighborhoodClip {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every pixel in `queries` and returns one
    /// [`TaauNeighborhoodClipResult`] per input, in order.
    ///
    /// The variance box `lo`/`hi` and the `clipped` history match the reference
    /// to within the tolerance documented on this module. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TaauNeighborhoodClipQuery],
    ) -> Vec<TaauNeighborhoodClipResult> {
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
            label: Some("prism_volumetric_taau_neighborhood_clip_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_taau_neighborhood_clip_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_taau_neighborhood_clip_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_taau_neighborhood_clip_bind_group"),
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
            label: Some("prism_volumetric_taau_neighborhood_clip_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_taau_neighborhood_clip_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_taau_neighborhood_clip_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per output pixel, flattened to a 1-D dispatch.
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
