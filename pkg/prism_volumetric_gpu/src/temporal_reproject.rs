//! `wgpu` compute twin of the particle temporal-reprojection history resolve
//! ([`temporal_reprojection`](prism_render_architecture::particle::temporal_reprojection),
//! design section 21).
//!
//! The `CPU` golden
//! [`particle::temporal_reprojection`](prism_render_architecture::particle::temporal_reprojection)
//! owns the *history-resolve* half of a temporal anti-aliasing (`TAA`) /
//! temporal-accumulation pipeline: given an upstream screen-space motion vector
//! it reprojects the previous frame's resolved colour into the current pixel,
//! decides whether that history is trustworthy, constrains it against the
//! current frame's local neighbourhood to fight ghosting, and blends the
//! constrained history against the current sample by confidence. This module is
//! the on-device twin: [`GpuTemporalReproject`] runs *one thread per pixel* and
//! reproduces the same resolved colour, so a passing real-device parity test is
//! direct evidence the ported kernel resolves history the same way the
//! reference does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! Each thread reproduces the reference resolve chain end to end for its pixel:
//!
//! 1. **Reproject** —
//!    [`reproject_uv`](prism_render_architecture::particle::temporal_reprojection::reproject_uv):
//!    the history sample lives at `current_uv - motion_uv`.
//! 2. **Validate** —
//!    [`history_valid`](prism_render_architecture::particle::temporal_reprojection::history_valid),
//!    which rejects off-screen reprojection
//!    ([`is_on_screen`](prism_render_architecture::particle::temporal_reprojection::is_on_screen)),
//!    multiplies the depth-disocclusion confidence
//!    ([`depth_confidence`](prism_render_architecture::particle::temporal_reprojection::depth_confidence))
//!    by the velocity confidence
//!    ([`velocity_confidence`](prism_render_architecture::particle::temporal_reprojection::velocity_confidence))
//!    and returns a soft weight in `[0, 1]`, never a hard boolean.
//! 3. **Constrain** —
//!    [`clip_history_ycocg`](prism_render_architecture::particle::temporal_reprojection::clip_history_ycocg):
//!    the sharper `YCoCg`-space `AABB` line clip of the reprojected history
//!    toward the current sample, using the current frame's `3x3` neighbourhood
//!    box widened by `clamp_widen`.
//! 4. **Blend** —
//!    [`blend_history`](prism_render_architecture::particle::temporal_reprojection::blend_history)
//!    by
//!    [`history_weight`](prism_render_architecture::particle::temporal_reprojection::history_weight):
//!    at full confidence the result trends to the constrained history, at zero
//!    confidence it collapses to the current sample.
//!
//! [`GpuTemporalReproject::eval`] returns a [`TemporalReprojectResult`] per
//! pixel carrying the resolved colour, the history confidence and the
//! reprojected `UV`, so a parity test can pin every stage, not just the final
//! blend.
//!
//! # Layout parity
//!
//! The per-dispatch uniform mirrors the reference `std430` parameter image: its
//! first four `f32` scalars are laid out exactly as
//! [`ReprojectionParams::to_std430`](prism_render_architecture::particle::temporal_reprojection::ReprojectionParams::to_std430)
//! (and therefore
//! [`pack_params_std430`](prism_render_architecture::particle::temporal_reprojection::pack_params_std430))
//! writes them, which a `debug_assert` in [`GpuTemporalReproject::eval`] checks
//! byte for byte before dispatch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `+ - * /` and one `sqrt` for the motion-vector length — with
//! no `exp`, `pow`, `sin`/`cos` or optional device feature, so it runs
//! unmodified on Metal, Vulkan and DX12. The soft confidence falloff is the
//! cubic `smoothstep` polynomial, exactly as the reference, so no transcendental
//! curve is introduced.
//!
//! # Correctness model
//!
//! The reproject, the `YCoCg` transform, the neighbourhood box and the blend are
//! fixed closed-form algebra, so `CPU` and `GPU` evaluate the same expression.
//! They are not bit-exact in general: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate (the `YCoCg` combinations, the clip crossing
//! fraction, the `smoothstep` cubic, the lerp), perturbing the low mantissa bits
//! by a few units in the last place. The parity test therefore asserts an
//! absolute tolerance tight enough to catch a genuinely wrong port yet loose
//! enough to admit legal fused multiply-add contraction, and additionally pins
//! the saturated cases bit for bit (off-screen or depth-rejected history makes
//! the blend weight an exact `0.0`, so the resolved colour collapses to exactly
//! the current sample).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `TAA` / temporal-accumulation history resolve
//! (reproject, disocclusion / velocity rejection, `YCoCg` neighbourhood clip,
//! confidence blend) plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::temporal_reprojection::{ReprojectionParams, Rgba, Vec2};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of colour taps in the current-frame neighbourhood window: a `3x3`
/// box, centre at index four (row-major), matching the reference `[Rgba; 9]`.
pub const NEIGHBORHOOD_TAPS: usize = 9;

/// The portable core-`WGSL` temporal-reprojection kernel, embedded inline so the
/// twin ships as a single source file. Mirrors the `CPU` resolve chain stage for
/// stage; see the module documentation for the algorithm.
///
/// The `Params` struct lays its first four `f32` scalars out exactly as
/// [`ReprojectionParams::to_std430`](prism_render_architecture::particle::temporal_reprojection::ReprojectionParams::to_std430)
/// does (one `vec4` slot), reusing the trailing scalar slot to carry the
/// per-dispatch pixel count.
const TEMPORAL_REPROJECT_WGSL: &str = r#"
// Temporal-reprojection twin: one thread per pixel reprojects the history
// sample, scores its confidence, YCoCg-clips it into the current 3x3
// neighbourhood box and blends it against the current sample by confidence. It
// mirrors the CPU golden `particle::temporal_reprojection` resolve chain, uses
// only the portable core-WGSL subset (min/max/clamp/abs and + - * / plus one
// sqrt for the motion length), and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard TAA / temporal-accumulation history resolve; no Unreal
// Engine source or derived code.

struct Params {
    // Steady-state history weight in [0, 1] retained at full confidence.
    max_history_weight: f32,
    // Relative depth difference at or above which history is rejected.
    depth_reject_relative: f32,
    // Screen-space motion magnitude (UV) at which velocity confidence hits zero.
    velocity_reject_uv: f32,
    // Fractional widening of the neighbourhood constraint box before clipping.
    clamp_widen: f32,
    // Pixel count, one thread each. Reuses the trailing scalar of the shared
    // std430 params slot.
    pixel_count: u32,
    // Padding to a 32-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One pixel's resolve inputs. 64-byte std430 stride matching the host GpuQuery.
struct Query {
    // Current-frame colour sample (linear RGBA).
    current: vec4<f32>,
    // Reprojected previous-frame colour sample (linear RGBA).
    history: vec4<f32>,
    // Current pixel UV.
    current_uv: vec2<f32>,
    // Screen-space motion vector (current - previous) in UV units.
    motion_uv: vec2<f32>,
    // Current surface depth.
    current_depth: f32,
    // History surface depth at the reprojected UV.
    history_depth: f32,
    pad0: f32,
    pad1: f32,
}

// One pixel's resolve outputs. 32-byte std430 stride matching the host result.
struct Resolved {
    // Final resolved colour (linear RGBA).
    resolved: vec4<f32>,
    // History confidence in [0, 1].
    confidence: f32,
    // Reprojected history UV (x, y).
    history_x: f32,
    history_y: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read> neighborhood: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> results: array<Resolved>;

// Floating-point comparison tolerance, matching the reference CMP_EPS.
const CMP_EPS: f32 = 1e-6;
// Floor on the depth denominator, matching the reference DEPTH_EPS.
const DEPTH_EPS: f32 = 1e-6;
// Number of neighbourhood taps per pixel (3x3), matching the reference window.
const TAPS: u32 = 9u;

// Cubic smoothstep t*t*(3 - 2t) clamped to [0, 1], matching the reference
// `smoothstep`. Inputs outside [0, 1] saturate to the endpoints.
fn smoothstep_unit(t: f32) -> f32 {
    let x = clamp(t, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

// Depth-based history confidence in [0, 1], matching the reference
// `depth_confidence`. At or above the relative-difference threshold the
// surfaces differ (a disocclusion) and confidence is zero; below it, confidence
// falls off with the cubic smoothstep.
fn depth_confidence(current_depth: f32, history_depth: f32, reject_relative: f32) -> f32 {
    let denom = max(abs(current_depth), DEPTH_EPS);
    let rel = abs(current_depth - history_depth) / denom;
    let thr = max(reject_relative, CMP_EPS);
    if (rel >= thr) {
        return 0.0;
    }
    return 1.0 - smoothstep_unit(rel / thr);
}

// Velocity-based history confidence in [0, 1], matching the reference
// `velocity_confidence`. A non-positive reject limit disables the penalty and
// returns full confidence.
fn velocity_confidence(motion: vec2<f32>, reject_uv: f32) -> f32 {
    let limit = max(reject_uv, 0.0);
    if (limit <= CMP_EPS) {
        return 1.0;
    }
    let speed = sqrt(motion.x * motion.x + motion.y * motion.y);
    return clamp(1.0 - smoothstep_unit(speed / limit), 0.0, 1.0);
}

// Whether a reprojected UV lands inside the inclusive [0, 1] screen rectangle,
// matching the reference `is_on_screen`. A NaN component fails every compare and
// is treated as off-screen, exactly as the reference range `contains` does.
fn is_on_screen(uv: vec2<f32>) -> bool {
    return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

// Combined history confidence in [0, 1], matching the reference `history_valid`:
// zero off-screen, else the depth confidence times the velocity confidence,
// clamped.
fn history_valid(history_uv: vec2<f32>, current_depth: f32, history_depth: f32, motion: vec2<f32>) -> f32 {
    if (!is_on_screen(history_uv)) {
        return 0.0;
    }
    let depth = depth_confidence(current_depth, history_depth, params.depth_reject_relative);
    let velocity = velocity_confidence(motion, params.velocity_reject_uv);
    return clamp(depth * velocity, 0.0, 1.0);
}

// Linear RGB to YCoCg (alpha ignored), matching the reference `rgb_to_ycocg`.
fn rgb_to_ycocg(color: vec4<f32>) -> vec3<f32> {
    let r = color.x;
    let g = color.y;
    let b = color.z;
    let y = r * 0.25 + g * 0.5 + b * 0.25;
    let co = r * 0.5 - b * 0.5;
    let cg = -r * 0.25 + g * 0.5 - b * 0.25;
    return vec3<f32>(y, co, cg);
}

// Inverse of `rgb_to_ycocg`, carrying alpha through unchanged, matching the
// reference `ycocg_to_rgb`.
fn ycocg_to_rgb(ycocg: vec3<f32>, alpha: f32) -> vec4<f32> {
    let y = ycocg.x;
    let co = ycocg.y;
    let cg = ycocg.z;
    let r = y + co - cg;
    let g = y + cg;
    let b = y - co - cg;
    return vec4<f32>(r, g, b, alpha);
}

// Tightens the retained fraction `t` for one axis of the YCoCg clip, matching
// one iteration of the reference `clip_toward`. An axis whose component barely
// moves (within CMP_EPS) imposes no constraint, so a vanishing extent never
// divides by zero.
fn clip_axis(t: f32, lo: f32, hi: f32, qa: f32, pa: f32) -> f32 {
    let delta = pa - qa;
    if (delta > CMP_EPS) {
        return min(t, (hi - qa) / delta);
    } else if (delta < -CMP_EPS) {
        return min(t, (lo - qa) / delta);
    }
    return t;
}

// Moves point `p` toward point `q` (assumed inside the box) until it lies on or
// inside the [box_min, box_max] AABB, matching the reference `clip_toward`.
fn clip_toward(box_min: vec3<f32>, box_max: vec3<f32>, q: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    var t = 1.0;
    t = clip_axis(t, box_min.x, box_max.x, q.x, p.x);
    t = clip_axis(t, box_min.y, box_max.y, q.y, p.y);
    t = clip_axis(t, box_min.z, box_max.z, q.z, p.z);
    t = clamp(t, 0.0, 1.0);
    return vec3<f32>(
        q.x + t * (p.x - q.x),
        q.y + t * (p.y - q.y),
        q.z + t * (p.z - q.z),
    );
}

// Clips the reprojected history toward the current sample until it lands inside
// the current neighbourhood's widened YCoCg AABB, matching the reference
// `clip_history_ycocg`. The history alpha is preserved.
fn clip_history_ycocg(base: u32, current: vec4<f32>, history: vec4<f32>) -> vec4<f32> {
    let q = rgb_to_ycocg(current);
    let p = rgb_to_ycocg(history);

    var lo = rgb_to_ycocg(neighborhood[base]);
    var hi = lo;
    for (var i = 1u; i < TAPS; i = i + 1u) {
        let value = rgb_to_ycocg(neighborhood[base + i]);
        lo = min(lo, value);
        hi = max(hi, value);
    }

    let factor = 1.0 + max(params.clamp_widen, 0.0);
    let centre = (lo + hi) * 0.5;
    let half_extent = (hi - lo) * 0.5 * factor;
    lo = centre - half_extent;
    hi = centre + half_extent;

    let clipped = clip_toward(lo, hi, q, p);
    return ycocg_to_rgb(clipped, history.w);
}

// History blend weight in [0, 1], matching the reference `history_weight`.
fn history_weight(confidence: f32) -> f32 {
    return clamp(clamp(confidence, 0.0, 1.0) * params.max_history_weight, 0.0, 1.0);
}

// Per-channel lerp current*(1 - w) + history*w, matching the reference
// `blend_history`.
fn blend_history(current: vec4<f32>, clamped_history: vec4<f32>, w: f32) -> vec4<f32> {
    return current * (1.0 - w) + clamped_history * w;
}

@compute @workgroup_size(64)
fn resolve_history(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_index = gid.x;
    if (pixel_index >= params.pixel_count) {
        return;
    }
    let query = queries[pixel_index];

    // 1. Reproject: history lives at current_uv - motion_uv.
    let history_uv = query.current_uv - query.motion_uv;

    // 2. Validate: soft confidence in [0, 1].
    let confidence = history_valid(
        history_uv,
        query.current_depth,
        query.history_depth,
        query.motion_uv,
    );

    // 3. Constrain: YCoCg neighbourhood line clip toward the current sample.
    let base = pixel_index * TAPS;
    let clipped = clip_history_ycocg(base, query.current, query.history);

    // 4. Blend: lerp current and constrained history by confidence weight.
    let w = history_weight(confidence);
    let resolved = blend_history(query.current, clipped, w);

    var out: Resolved;
    out.resolved = resolved;
    out.confidence = confidence;
    out.history_x = history_uv.x;
    out.history_y = history_uv.y;
    out.pad = 0.0;
    results[pixel_index] = out;
}
"#;

/// One pixel's temporal-reprojection query: the twin of a single reference
/// resolve-chain evaluation.
///
/// `neighborhood` is the current frame's `3x3` colour window around the pixel
/// (centre at index four, row-major), the plausible-colour box the history is
/// `YCoCg`-clipped into. The reprojected history `UV` is derived on device as
/// `current_uv - motion_uv`; the caller supplies `history` as the colour already
/// fetched at that reprojected location (the upstream history-sample tap), along
/// with its `history_depth`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalReprojectQuery {
    /// Current pixel `UV`.
    pub current_uv: Vec2,
    /// Screen-space motion vector (`current - previous`) in `UV` units.
    pub motion_uv: Vec2,
    /// Current surface depth at this pixel.
    pub current_depth: f32,
    /// History surface depth at the reprojected `UV`.
    pub history_depth: f32,
    /// Current-frame colour sample (linear `RGBA`).
    pub current: Rgba,
    /// Reprojected previous-frame colour sample (linear `RGBA`).
    pub history: Rgba,
    /// Current frame's `3x3` colour neighbourhood ([`NEIGHBORHOOD_TAPS`] taps,
    /// row-major, centre at index four).
    pub neighborhood: [Rgba; NEIGHBORHOOD_TAPS],
}

/// One pixel's resolved temporal-reprojection output.
///
/// Carries the final blended colour plus the two intermediate values a parity
/// test pins directly: the history `confidence` and the reprojected
/// `history_uv`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalReprojectResult {
    /// Final resolved colour (linear `RGBA`).
    pub resolved: Rgba,
    /// History confidence in `[0, 1]`.
    pub confidence: f32,
    /// Reprojected history `UV` (`current_uv - motion_uv`).
    pub history_uv: Vec2,
}

/// Uniform parameters for one temporal-reprojection dispatch. `repr(C)` `std430`
/// layout matching `Params` in [`TEMPORAL_REPROJECT_WGSL`]: the four
/// [`ReprojectionParams`] scalars in
/// [`ReprojectionParams::to_std430`](prism_render_architecture::particle::temporal_reprojection::ReprojectionParams::to_std430)
/// order, then the pixel count (reusing the trailing scalar slot) and three pad
/// words — `32` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    max_history_weight: f32,
    depth_reject_relative: f32,
    velocity_reject_uv: f32,
    clamp_widen: f32,
    pixel_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One pixel's query as uploaded. `64`-byte `std430` stride matching `Query` in
/// the shader: the two colour vectors, the two `UV` vectors, the two depths and
/// two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    current: [f32; 4],
    history: [f32; 4],
    current_uv: [f32; 2],
    motion_uv: [f32; 2],
    current_depth: f32,
    history_depth: f32,
    pad0: f32,
    pad1: f32,
}

/// One pixel's result as read back. `32`-byte `std430` stride matching
/// `Resolved` in the shader: the resolved colour, the confidence, the two
/// reprojected `UV` components and one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResolved {
    resolved: [f32; 4],
    confidence: f32,
    history_x: f32,
    history_y: f32,
    pad: f32,
}

/// A compiled, reusable temporal-reprojection history-resolve pipeline.
pub struct GpuTemporalReproject {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTemporalReproject {
    /// Compiles the temporal-reprojection resolve kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTemporalReproject {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_temporal_reproject"),
            source: ShaderSource::Wgsl(TEMPORAL_REPROJECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_temporal_reproject_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_temporal_reproject_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_temporal_reproject_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("resolve_history"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTemporalReproject {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every pixel in `queries` under the shared `params`, returning one
    /// [`TemporalReprojectResult`] per pixel in input order.
    ///
    /// The returned result for pixel `q` reproduces the reference resolve chain:
    /// `history_uv` equals
    /// [`reproject_uv`](prism_render_architecture::particle::temporal_reprojection::reproject_uv)`(q.current_uv, q.motion_uv)`,
    /// `confidence` equals
    /// [`history_valid`](prism_render_architecture::particle::temporal_reprojection::history_valid)`(history_uv, q.current_depth, q.history_depth, q.motion_uv, params)`,
    /// and `resolved` equals
    /// [`blend_history`](prism_render_architecture::particle::temporal_reprojection::blend_history)`(q.current, clip_history_ycocg(q.current, q.history, &q.neighborhood, params), confidence, params)`,
    /// all to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so it
    /// is handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        params: &ReprojectionParams,
        queries: &[TemporalReprojectQuery],
    ) -> Vec<TemporalReprojectResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        // Flatten the per-pixel 3x3 neighbourhood windows into one contiguous
        // buffer; pixel `i` owns the run `[i * TAPS .. i * TAPS + TAPS]`, which
        // the kernel reads back by the same stride.
        let mut flat_neighborhood: Vec<[f32; 4]> =
            Vec::with_capacity(queries.len() * NEIGHBORHOOD_TAPS);
        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| {
                flat_neighborhood.extend_from_slice(&q.neighborhood);
                GpuQuery {
                    current: q.current,
                    history: q.history,
                    current_uv: [q.current_uv.x, q.current_uv.y],
                    motion_uv: [q.motion_uv.x, q.motion_uv.y],
                    current_depth: q.current_depth,
                    history_depth: q.history_depth,
                    pad0: 0.0,
                    pad1: 0.0,
                }
            })
            .collect();

        let gpu_params = GpuParams {
            max_history_weight: params.max_history_weight,
            depth_reject_relative: params.depth_reject_relative,
            velocity_reject_uv: params.velocity_reject_uv,
            clamp_widen: params.clamp_widen,
            pixel_count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // The uniform's first vec4 must mirror the reference std430 parameter
        // image byte for byte; a drift here would desynchronise the kernel from
        // the golden layout silently.
        debug_assert_eq!(
            bytemuck::bytes_of(&gpu_params)[..params.to_std430().len()],
            params.to_std430()[..],
            "GPU params prefix must match the golden std430 parameter image"
        );

        let out_bytes = (queries.len() as u64) * (size_of::<GpuResolved>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_temporal_reproject_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_temporal_reproject_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let neighborhood_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_temporal_reproject_neighborhood"),
            contents: bytemuck::cast_slice(&flat_neighborhood),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_temporal_reproject_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_temporal_reproject_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_temporal_reproject_bind_group"),
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
                    resource: neighborhood_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_temporal_reproject_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_temporal_reproject_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pixel, in workgroups of 64 (the kernel's size).
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuResolved>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
            .into_iter()
            .map(|r| TemporalReprojectResult {
                resolved: r.resolved,
                confidence: r.confidence,
                history_uv: Vec2 {
                    x: r.history_x,
                    y: r.history_y,
                },
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
