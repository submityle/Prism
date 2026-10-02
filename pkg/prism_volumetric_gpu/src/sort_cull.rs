//! `wgpu` compute twin of the sort/cull decision golden
//! ([`sort_cull`](prism_render_architecture::particle::sort_cull), design §12,
//! §13).
//!
//! Before a renderer composites its particles it decides *whether* to sort and
//! *how*, *which* bounding spheres survive distance / frustum / `HZB` culling,
//! and *whether* a barely-visible emitter can sleep. The `CPU` golden
//! `prism_render_architecture::particle::sort_cull` owns that math — a 16-bit
//! depth key, the blend-driven strategy matrix, inward-pointing frustum planes,
//! an axis-aligned bounds algebra, the continuous significance blend and the
//! per-frame sleep accumulator. [`GpuSortCull`] is the on-device twin that runs
//! one thread per aggregated query and reproduces every lane, so a passing
//! real-device parity test is direct evidence the ported kernel folds the same
//! quantization, the same comparisons and the same classification the reference
//! does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! One aggregate [`GpuSortCullQuery`] drives, per thread, the whole decision
//! surface and the kernel writes one aggregate [`GpuSortCullResult`]:
//!
//! * [`quantize_depth`](prism_render_architecture::particle::sort_cull::quantize_depth)
//!   and
//!   [`sort_key`](prism_render_architecture::particle::sort_cull::sort_key),
//!   returned as `u32` in `0..=65535` because `WGSL` has no `u16`; the host
//!   widens the golden `u16` with `as u32` and compares exactly.
//! * [`BlendMode::needs_sort`](prism_render_architecture::particle::sort_cull::BlendMode::needs_sort)
//!   and
//!   [`choose_sort_strategy`](prism_render_architecture::particle::sort_cull::choose_sort_strategy),
//!   as a discrete strategy code.
//! * the [`Aabb`](prism_render_architecture::particle::sort_cull::Aabb) algebra
//!   (`empty` / `from_point` / `expand` / `union` / `center` / `half_extents` /
//!   `bounding_radius` / `is_valid`), the `empty` box materialized with a
//!   `bitcast` infinity sentinel because `WGSL` has no infinity literal. The
//!   variable-length `reduce_bounds` is *not* twinned here (it is a reduction);
//!   the fixed-arity constructors it is built from are.
//! * [`Plane::signed_distance`](prism_render_architecture::particle::sort_cull::Plane::signed_distance),
//!   [`Frustum::intersects_sphere`](prism_render_architecture::particle::sort_cull::Frustum::intersects_sphere),
//!   [`cull_sphere`](prism_render_architecture::particle::sort_cull::cull_sphere)
//!   (distance before frustum, with a classified
//!   [`CullReason`](prism_render_architecture::particle::sort_cull::CullReason))
//!   and
//!   [`apply_hzb`](prism_render_architecture::particle::sort_cull::apply_hzb).
//! * [`significance`](prism_render_architecture::particle::sort_cull::significance)
//!   and the
//!   [`update_sleep`](prism_render_architecture::particle::sort_cull::update_sleep)
//!   accumulator over
//!   [`SleepState`](prism_render_architecture::particle::sort_cull::SleepState) /
//!   [`SleepParams`](prism_render_architecture::particle::sort_cull::SleepParams).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `floor`, `sqrt`, `dot`, `+ - * /`, integer compares and two `bitcast`s to
//! build the `IEEE`-754 infinity sentinels — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `round` or optional device feature, so it runs unmodified on
//! Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of comparisons and
//! arithmetic, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the continuous values
//! (`signed_distance`, `center`, `half_extents`, `bounding_radius`,
//! `significance`) yet an *exact* match on the quantized keys, the discrete
//! classification codes and every boolean. Fixtures are placed clear of the
//! quantization half-step and the `+/-radius` tangent boundary so a legal `ULP`
//! perturbation never flips a verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sort_cull`；
//! standard depth-key quantization, frustum-plane sphere culling and bounds
//! algebra plus `wgpu` compute dispatch; no third-party engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::sort_cull::{
    apply_hzb, choose_sort_strategy, cull_sphere, quantize_depth, significance, sort_key,
    update_sleep, Aabb, BlendMode, CullReason, Frustum, SleepParams, SleepState, SortDecision,
};
use prism_render_architecture::particle::{SortStrategy, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Discrete `true` code written by the kernel for every boolean lane, matching
/// the host `== 1` decode. A direct `f32` equality is forbidden, so booleans
/// cross the device boundary as integer flags.
const CODE_TRUE: u32 = 1;

/// Blend-mode discriminant for the one order-dependent mode, matching the host
/// `BLEND_ALPHA` packing; only this mode reports `needs_sort`.
const BLEND_ALPHA: u32 = 3;

/// The portable core-`WGSL` sort/cull kernel, embedded inline so the twin ships
/// as a single source file. Mirrors the `CPU` golden
/// [`sort_cull`](prism_render_architecture::particle::sort_cull) decision by
/// decision; see the module documentation for the algorithm.
const SORT_CULL_WGSL: &str = r#"
// Sort/cull decision twin: one thread per aggregated query reproduces the depth
// key, the strategy matrix, the AABB algebra, the frustum/distance/HZB cull
// classification, the significance blend and the sleep accumulator of the CPU
// golden `particle::sort_cull`. It uses only the portable core-WGSL subset
// (min/max/abs/floor/sqrt/dot, + - * /, integer compares and two bitcasts for
// the infinity sentinels) and takes no optional feature, so it runs unmodified
// on Metal, Vulkan and DX12.
//
// Provenance: standard depth-key quantization, frustum-plane sphere culling and
// bounds algebra; no third-party engine source or derived code.

struct Params {
    // Number of valid queries dispatched; threads past it return early.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One aggregated query. 272-byte std430 stride matching the host `RawQuery`:
// twelve vec4 lanes (three AABB points, the plane-distance probe point, the six
// frustum planes as (normal.xyz, d), the test sphere as (center.xyz, radius)
// and the camera as (pos.xyz, max_distance)) followed by twenty scalar words.
struct Query {
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    plane_point: vec4<f32>,
    plane0: vec4<f32>,
    plane1: vec4<f32>,
    plane2: vec4<f32>,
    plane3: vec4<f32>,
    plane4: vec4<f32>,
    plane5: vec4<f32>,
    sphere: vec4<f32>,
    camera: vec4<f32>,
    depth: f32,
    near: f32,
    far: f32,
    back_to_front: u32,
    blend: u32,
    particle_count: u32,
    radix_min_count: u32,
    prefer_shared_oit: u32,
    screen_coverage: f32,
    speed: f32,
    speed_ref: f32,
    occluded: u32,
    in_idle_frames: u32,
    in_asleep: u32,
    sleep_sig: f32,
    sleep_below: f32,
    frames_to_sleep: u32,
    qpad0: u32,
    qpad1: u32,
    qpad2: u32,
}

// One aggregated result. 112-byte std430 stride matching the host `RawResult`:
// three vec4 lanes (the box center, half-extents and union center) then sixteen
// scalar words carrying the keys, the discrete classification codes, the
// booleans (as 0u/1u) and the continuous outputs.
struct Res {
    center: vec4<f32>,
    half_extents: vec4<f32>,
    union_center: vec4<f32>,
    quantized_depth: u32,
    sort_key: u32,
    needs_sort: u32,
    sort_strategy: u32,
    bounding_radius: f32,
    is_valid: u32,
    empty_valid: u32,
    plane0_sd: f32,
    frustum_intersects: u32,
    cull_visible: u32,
    cull_reason: u32,
    hzb_visible: u32,
    hzb_reason: u32,
    significance: f32,
    out_idle_frames: u32,
    out_asleep: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// The one order-dependent blend mode; only it needs a sort. Matches the host
// `BLEND_ALPHA` packing.
const BLEND_ALPHA: u32 = 3u;

// Sort-strategy discriminants, matching the host `strategy_from_code` decode.
const STRATEGY_NONE: u32 = 0u;
const STRATEGY_SHARED_OIT: u32 = 1u;
const STRATEGY_RADIX: u32 = 2u;
const STRATEGY_BITONIC: u32 = 3u;

// Cull-reason discriminants, matching the host `reason_from_code` decode.
const REASON_VISIBLE: u32 = 0u;
const REASON_OUTSIDE_FRUSTUM: u32 = 1u;
const REASON_BEYOND_DISTANCE: u32 = 2u;
const REASON_HZB_OCCLUDED: u32 = 3u;

// An axis-aligned box, the WGSL mirror of the golden `Aabb`.
struct AabbW {
    mn: vec3<f32>,
    mx: vec3<f32>,
}

// The empty box: +inf min, -inf max, the identity for `aabb_union`. Built with
// bitcast because WGSL has no infinity literal, mirroring the golden
// `f32::INFINITY` / `f32::NEG_INFINITY` corners exactly.
fn aabb_empty() -> AabbW {
    let pos_inf = bitcast<f32>(0x7f800000u);
    let neg_inf = bitcast<f32>(0xff800000u);
    return AabbW(vec3<f32>(pos_inf, pos_inf, pos_inf), vec3<f32>(neg_inf, neg_inf, neg_inf));
}

// A zero-volume box at one point.
fn aabb_from_point(p: vec3<f32>) -> AabbW {
    return AabbW(p, p);
}

// Grows the box to include `p`.
fn aabb_expand(b: AabbW, p: vec3<f32>) -> AabbW {
    return AabbW(min(b.mn, p), max(b.mx, p));
}

// The smallest box containing both boxes.
fn aabb_union(a: AabbW, b: AabbW) -> AabbW {
    return AabbW(min(a.mn, b.mn), max(a.mx, b.mx));
}

// Center point (defined only for a valid box).
fn aabb_center(b: AabbW) -> vec3<f32> {
    return (b.mn + b.mx) * 0.5;
}

// Half the diagonal extent (defined only for a valid box).
fn aabb_half_extents(b: AabbW) -> vec3<f32> {
    return (b.mx - b.mn) * 0.5;
}

// Radius of the bounding sphere sharing the box center.
fn aabb_bounding_radius(b: AabbW) -> f32 {
    let h = aabb_half_extents(b);
    return sqrt(dot(h, h));
}

// True when the box encloses a non-negative volume, mirroring the golden
// per-axis min <= max test.
fn aabb_is_valid(b: AabbW) -> bool {
    return b.mn.x <= b.mx.x && b.mn.y <= b.mx.y && b.mn.z <= b.mx.z;
}

// Quantizes a view-space depth to a 16-bit sort key, returned as u32 in
// 0..=65535. Mirrors the golden `quantize_depth`: a degenerate range short
// circuits to 0, otherwise the clamped depth maps linearly and the `* 65535 +
// 0.5` rounding is reproduced with floor (WGSL has neither `round` nor `u16`).
fn quantize_depth(depth: f32, near: f32, far: f32) -> u32 {
    if (far <= near) {
        return 0u;
    }
    let clamped = min(max(depth, near), far);
    let t = (clamped - near) / (far - near);
    return u32(floor(t * 65535.0 + 0.5));
}

// Depth sort key oriented for a blend mode. Back-to-front inverts the key with
// `65535 - q`, the u32 complement reproducing the golden `u16::MAX - q`.
fn sort_key(depth: f32, near: f32, far: f32, back_to_front: u32) -> u32 {
    let q = quantize_depth(depth, near, far);
    if (back_to_front != 0u) {
        return 65535u - q;
    }
    return q;
}

// Significance of an emitter this frame: both terms clamped to 0..=1 with the
// larger winning, mirroring the golden `max(0).min(1)` ordering and the
// non-positive `speed_ref` short circuit.
fn significance_of(screen_coverage: f32, speed: f32, speed_ref: f32) -> f32 {
    let coverage = min(max(screen_coverage, 0.0), 1.0);
    var motion = 0.0;
    if (speed_ref > 0.0) {
        motion = min(max(speed / speed_ref, 0.0), 1.0);
    }
    return max(coverage, motion);
}

@compute @workgroup_size(64)
fn sort_cull_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // --- Depth keys (returned as u32 in 0..=65535). ---
    let qd = quantize_depth(q.depth, q.near, q.far);
    let sk = sort_key(q.depth, q.near, q.far, q.back_to_front);

    // --- Blend-driven strategy matrix. ---
    let needs = q.blend == BLEND_ALPHA;
    var strat = STRATEGY_NONE;
    if (!needs || q.particle_count <= 1u) {
        strat = STRATEGY_NONE;
    } else if (q.prefer_shared_oit != 0u) {
        strat = STRATEGY_SHARED_OIT;
    } else if (q.particle_count >= q.radix_min_count) {
        strat = STRATEGY_RADIX;
    } else {
        strat = STRATEGY_BITONIC;
    }

    // --- AABB algebra: from_point + two expands reproduces reduce over the
    // three points; a separate union exercises the identity path. ---
    let b = aabb_expand(aabb_expand(aabb_from_point(q.p0.xyz), q.p1.xyz), q.p2.xyz);
    let box_center = aabb_center(b);
    let box_half = aabb_half_extents(b);
    let box_radius = aabb_bounding_radius(b);
    let box_valid = aabb_is_valid(b);
    let empty_valid = aabb_is_valid(aabb_empty());
    let left = aabb_from_point(q.p0.xyz);
    let right = aabb_expand(aabb_from_point(q.p1.xyz), q.p2.xyz);
    let union_center = aabb_center(aabb_union(left, right));

    // --- Plane signed distance (probe plane0 at the probe point). ---
    let plane0_sd = dot(q.plane0.xyz, q.plane_point.xyz) + q.plane0.w;

    // --- Frustum sphere test over the six inward planes. ---
    let center = q.sphere.xyz;
    let radius = q.sphere.w;
    var planes = array<vec4<f32>, 6>(
        q.plane0, q.plane1, q.plane2, q.plane3, q.plane4, q.plane5,
    );
    var inside = true;
    for (var i = 0u; i < 6u; i = i + 1u) {
        let sd = dot(planes[i].xyz, center) + planes[i].w;
        if (sd < -radius) {
            inside = false;
        }
    }

    // --- Cull: distance first (cheapest), then frustum. ---
    let camera = q.camera.xyz;
    let max_distance = q.camera.w;
    let dc = camera - center;
    let d2 = dot(dc, dc);
    let cull_at = max_distance + radius;
    var cull_visible = true;
    var cull_reason = REASON_VISIBLE;
    if (max_distance > 0.0 && d2 > cull_at * cull_at) {
        cull_visible = false;
        cull_reason = REASON_BEYOND_DISTANCE;
    } else if (!inside) {
        cull_visible = false;
        cull_reason = REASON_OUTSIDE_FRUSTUM;
    }

    // --- HZB fold: only a currently-visible decision can be occlusion-culled. ---
    var hzb_visible = cull_visible;
    var hzb_reason = cull_reason;
    if (cull_visible && q.occluded != 0u) {
        hzb_visible = false;
        hzb_reason = REASON_HZB_OCCLUDED;
    }

    // --- Significance. ---
    let sig = significance_of(q.screen_coverage, q.speed, q.speed_ref);

    // --- Sleep accumulator: wake instantly above threshold, else accrue. ---
    var out_idle = 0u;
    var out_asleep = 0u;
    if (q.sleep_sig > q.sleep_below) {
        out_idle = 0u;
        out_asleep = 0u;
    } else {
        var idle = q.in_idle_frames;
        if (idle < 0xffffffffu) {
            idle = idle + 1u;
        }
        out_idle = idle;
        if (idle >= q.frames_to_sleep) {
            out_asleep = 1u;
        } else {
            out_asleep = 0u;
        }
    }

    var res: Res;
    res.center = vec4<f32>(box_center, 0.0);
    res.half_extents = vec4<f32>(box_half, 0.0);
    res.union_center = vec4<f32>(union_center, 0.0);
    res.quantized_depth = qd;
    res.sort_key = sk;
    res.needs_sort = select(0u, 1u, needs);
    res.sort_strategy = strat;
    res.bounding_radius = box_radius;
    res.is_valid = select(0u, 1u, box_valid);
    res.empty_valid = select(0u, 1u, empty_valid);
    res.plane0_sd = plane0_sd;
    res.frustum_intersects = select(0u, 1u, inside);
    res.cull_visible = select(0u, 1u, cull_visible);
    res.cull_reason = cull_reason;
    res.hzb_visible = select(0u, 1u, hzb_visible);
    res.hzb_reason = hzb_reason;
    res.significance = sig;
    res.out_idle_frames = out_idle;
    res.out_asleep = out_asleep;
    results[idx] = res;
}
"#;

/// One aggregated sort/cull query: every scalar, point, plane and flag the
/// decision surface needs, carried with the `CPU` golden types so a caller
/// builds it from the same values the reference consumes.
///
/// Derives only [`PartialEq`] (no `Eq` / `Hash`) because it holds `f32`
/// geometry. One dispatch may mix wholly unrelated queries.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sort_cull`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSortCullQuery {
    /// View-space depth fed to `quantize_depth` and `sort_key`.
    pub depth: f32,
    /// Near plane of the depth range.
    pub near: f32,
    /// Far plane of the depth range.
    pub far: f32,
    /// Whether the sort key is inverted for back-to-front ordering.
    pub back_to_front: bool,
    /// The sort decision whose `needs_sort` and strategy are twinned.
    pub sort: SortDecision,
    /// Three points folded into a bounds box (`from_point` then two `expand`s).
    pub points: [Vec3; 3],
    /// Probe point for the `plane0` `signed_distance` reading.
    pub plane_point: Vec3,
    /// The six-plane frustum tested against the sphere.
    pub frustum: Frustum,
    /// Center of the sphere passed to the frustum and cull tests.
    pub sphere_center: Vec3,
    /// Radius of that sphere.
    pub sphere_radius: f32,
    /// Camera position for the distance cull.
    pub camera_position: Vec3,
    /// Cull distance; a non-positive value disables the distance test.
    pub max_distance: f32,
    /// Whether the `HZB` reported the sphere occluded.
    pub occluded: bool,
    /// Screen coverage term of `significance`.
    pub screen_coverage: f32,
    /// Speed term of `significance`.
    pub speed: f32,
    /// Reference speed treated as fully significant.
    pub speed_ref: f32,
    /// Incoming sleep bookkeeping advanced by `update_sleep`.
    pub sleep_state: SleepState,
    /// Significance fed to `update_sleep` this frame.
    pub sleep_significance: f32,
    /// Sleep thresholds for `update_sleep`.
    pub sleep_params: SleepParams,
}

/// The aggregated verdict for one query, the host-side mirror of the kernel's
/// `Res` lane.
///
/// Keys and classification codes are decoded to the golden
/// [`SortStrategy`](prism_render_architecture::particle::SortStrategy) and
/// [`CullReason`](prism_render_architecture::particle::sort_cull::CullReason)
/// enums; booleans decode from the `0`/`1` flags; continuous values pass
/// through unchanged. Derives only [`PartialEq`] (no `Eq` / `Hash`) because it
/// holds `f32` outputs.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sort_cull`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSortCullResult {
    /// The quantized depth key, widened from the golden `u16`.
    pub quantized_depth: u32,
    /// The oriented sort key, widened from the golden `u16`.
    pub sort_key: u32,
    /// Whether the blend mode requires an explicit sort.
    pub needs_sort: bool,
    /// The chosen sort strategy.
    pub sort_strategy: SortStrategy,
    /// Center of the bounds box over the three points.
    pub aabb_center: Vec3,
    /// Half-extents of that box.
    pub aabb_half_extents: Vec3,
    /// Bounding-sphere radius of that box.
    pub aabb_bounding_radius: f32,
    /// Whether that box is valid.
    pub aabb_is_valid: bool,
    /// Whether the empty box is valid (always `false`; checks the inf sentinel).
    pub aabb_empty_valid: bool,
    /// Center of the union of two boxes built from the three points.
    pub aabb_union_center: Vec3,
    /// Signed distance from `plane_point` to the frustum's first plane.
    pub plane0_signed_distance: f32,
    /// Whether the sphere is not fully outside any frustum plane.
    pub frustum_intersects: bool,
    /// Whether the sphere survives the distance-and-frustum cull.
    pub cull_visible: bool,
    /// Why the cull kept or rejected the sphere.
    pub cull_reason: CullReason,
    /// Whether the sphere survives after the `HZB` fold.
    pub hzb_visible: bool,
    /// The reason after the `HZB` fold.
    pub hzb_reason: CullReason,
    /// The blended significance this frame.
    pub significance: f32,
    /// Idle-frame count after advancing the sleep state.
    pub out_idle_frames: u32,
    /// Whether the emitter is asleep after advancing the sleep state.
    pub out_asleep: bool,
}

/// One query as uploaded. `272`-byte `repr(C)` `std430` layout matching `Query`
/// in [`SORT_CULL_WGSL`]: twelve `vec4` lanes then twenty scalar words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RawQuery {
    p0: [f32; 4],
    p1: [f32; 4],
    p2: [f32; 4],
    plane_point: [f32; 4],
    plane0: [f32; 4],
    plane1: [f32; 4],
    plane2: [f32; 4],
    plane3: [f32; 4],
    plane4: [f32; 4],
    plane5: [f32; 4],
    sphere: [f32; 4],
    camera: [f32; 4],
    depth: f32,
    near: f32,
    far: f32,
    back_to_front: u32,
    blend: u32,
    particle_count: u32,
    radix_min_count: u32,
    prefer_shared_oit: u32,
    screen_coverage: f32,
    speed: f32,
    speed_ref: f32,
    occluded: u32,
    in_idle_frames: u32,
    in_asleep: u32,
    sleep_sig: f32,
    sleep_below: f32,
    frames_to_sleep: u32,
    qpad0: u32,
    qpad1: u32,
    qpad2: u32,
}

/// One result as read back. `112`-byte `repr(C)` `std430` layout matching `Res`
/// in [`SORT_CULL_WGSL`]: three `vec4` lanes then sixteen scalar words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RawResult {
    center: [f32; 4],
    half_extents: [f32; 4],
    union_center: [f32; 4],
    quantized_depth: u32,
    sort_key: u32,
    needs_sort: u32,
    sort_strategy: u32,
    bounding_radius: f32,
    is_valid: u32,
    empty_valid: u32,
    plane0_sd: f32,
    frustum_intersects: u32,
    cull_visible: u32,
    cull_reason: u32,
    hzb_visible: u32,
    hzb_reason: u32,
    significance: f32,
    out_idle_frames: u32,
    out_asleep: u32,
}

/// Uniform parameters for one dispatch. `16`-byte `repr(C)` matching `Params`
/// in [`SORT_CULL_WGSL`]: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Packs a [`Vec3`] into a padded `vec4` upload lane.
fn vec4_of(v: Vec3) -> [f32; 4] {
    [v.x, v.y, v.z, 0.0]
}

/// Maps a [`BlendMode`] to its upload discriminant, matching the kernel's
/// `BLEND_ALPHA` test.
fn blend_code(blend: BlendMode) -> u32 {
    match blend {
        BlendMode::Opaque => 0,
        BlendMode::Additive => 1,
        BlendMode::Premultiplied => 2,
        BlendMode::AlphaBlend => BLEND_ALPHA,
    }
}

/// Maps a strategy code read back from the kernel to a [`SortStrategy`]; every
/// code outside `1..=3` is the order-independent [`SortStrategy::None`].
fn strategy_from_code(code: u32) -> SortStrategy {
    match code {
        1 => SortStrategy::SharedOit,
        2 => SortStrategy::ViewDepthRadix,
        3 => SortStrategy::ViewDepthBitonic,
        _ => SortStrategy::None,
    }
}

/// Maps a reason code read back from the kernel to a [`CullReason`]; every code
/// outside `1..=3` is [`CullReason::Visible`].
fn reason_from_code(code: u32) -> CullReason {
    match code {
        1 => CullReason::OutsideFrustum,
        2 => CullReason::BeyondDistance,
        3 => CullReason::HzbOccluded,
        _ => CullReason::Visible,
    }
}

impl RawQuery {
    /// Packs a [`GpuSortCullQuery`] into the `std430` upload layout.
    fn from_query(q: &GpuSortCullQuery) -> RawQuery {
        let planes = q.frustum.planes;
        let plane_lane = |p: usize| {
            let pl = planes[p];
            [pl.normal.x, pl.normal.y, pl.normal.z, pl.d]
        };
        RawQuery {
            p0: vec4_of(q.points[0]),
            p1: vec4_of(q.points[1]),
            p2: vec4_of(q.points[2]),
            plane_point: vec4_of(q.plane_point),
            plane0: plane_lane(0),
            plane1: plane_lane(1),
            plane2: plane_lane(2),
            plane3: plane_lane(3),
            plane4: plane_lane(4),
            plane5: plane_lane(5),
            sphere: [
                q.sphere_center.x,
                q.sphere_center.y,
                q.sphere_center.z,
                q.sphere_radius,
            ],
            camera: [
                q.camera_position.x,
                q.camera_position.y,
                q.camera_position.z,
                q.max_distance,
            ],
            depth: q.depth,
            near: q.near,
            far: q.far,
            back_to_front: u32::from(q.back_to_front),
            blend: blend_code(q.sort.blend),
            particle_count: q.sort.particle_count,
            radix_min_count: q.sort.radix_min_count,
            prefer_shared_oit: u32::from(q.sort.prefer_shared_oit),
            screen_coverage: q.screen_coverage,
            speed: q.speed,
            speed_ref: q.speed_ref,
            occluded: u32::from(q.occluded),
            in_idle_frames: q.sleep_state.idle_frames,
            in_asleep: u32::from(q.sleep_state.asleep),
            sleep_sig: q.sleep_significance,
            sleep_below: q.sleep_params.sleep_below,
            frames_to_sleep: q.sleep_params.frames_to_sleep,
            qpad0: 0,
            qpad1: 0,
            qpad2: 0,
        }
    }
}

/// Maps one kernel `Res` lane back to the host [`GpuSortCullResult`].
fn decode_result(raw: &RawResult) -> GpuSortCullResult {
    GpuSortCullResult {
        quantized_depth: raw.quantized_depth,
        sort_key: raw.sort_key,
        needs_sort: raw.needs_sort == CODE_TRUE,
        sort_strategy: strategy_from_code(raw.sort_strategy),
        aabb_center: Vec3::new(raw.center[0], raw.center[1], raw.center[2]),
        aabb_half_extents: Vec3::new(
            raw.half_extents[0],
            raw.half_extents[1],
            raw.half_extents[2],
        ),
        aabb_bounding_radius: raw.bounding_radius,
        aabb_is_valid: raw.is_valid == CODE_TRUE,
        aabb_empty_valid: raw.empty_valid == CODE_TRUE,
        aabb_union_center: Vec3::new(
            raw.union_center[0],
            raw.union_center[1],
            raw.union_center[2],
        ),
        plane0_signed_distance: raw.plane0_sd,
        frustum_intersects: raw.frustum_intersects == CODE_TRUE,
        cull_visible: raw.cull_visible == CODE_TRUE,
        cull_reason: reason_from_code(raw.cull_reason),
        hzb_visible: raw.hzb_visible == CODE_TRUE,
        hzb_reason: reason_from_code(raw.hzb_reason),
        significance: raw.significance,
        out_idle_frames: raw.out_idle_frames,
        out_asleep: raw.out_asleep == CODE_TRUE,
    }
}

/// The `CPU` golden verdict for one query, dispatching to the reference entry
/// points so callers (and the parity test) can pin the twin lane for lane.
///
/// Every field is computed by the golden
/// [`sort_cull`](prism_render_architecture::particle::sort_cull) functions, so a
/// disagreement with [`GpuSortCull::eval`] is a genuine port error.
#[must_use]
pub fn cpu_reference(query: &GpuSortCullQuery) -> GpuSortCullResult {
    let quantized_depth = u32::from(quantize_depth(query.depth, query.near, query.far));
    let key = u32::from(sort_key(
        query.depth,
        query.near,
        query.far,
        query.back_to_front,
    ));
    let strategy = choose_sort_strategy(query.sort);
    let box_bounds = Aabb::from_point(query.points[0])
        .expand(query.points[1])
        .expand(query.points[2]);
    let union_box = Aabb::from_point(query.points[0])
        .union(Aabb::from_point(query.points[1]).expand(query.points[2]));
    let cull = cull_sphere(
        query.frustum,
        query.camera_position,
        query.sphere_center,
        query.sphere_radius,
        query.max_distance,
    );
    let hzb = apply_hzb(cull, query.occluded);
    let sleep = update_sleep(
        query.sleep_state,
        query.sleep_significance,
        query.sleep_params,
    );
    GpuSortCullResult {
        quantized_depth,
        sort_key: key,
        needs_sort: query.sort.blend.needs_sort(),
        sort_strategy: strategy,
        aabb_center: box_bounds.center(),
        aabb_half_extents: box_bounds.half_extents(),
        aabb_bounding_radius: box_bounds.bounding_radius(),
        aabb_is_valid: box_bounds.is_valid(),
        aabb_empty_valid: Aabb::empty().is_valid(),
        aabb_union_center: union_box.center(),
        plane0_signed_distance: query.frustum.planes[0].signed_distance(query.plane_point),
        frustum_intersects: query
            .frustum
            .intersects_sphere(query.sphere_center, query.sphere_radius),
        cull_visible: cull.visible,
        cull_reason: cull.reason,
        hzb_visible: hzb.visible,
        hzb_reason: hzb.reason,
        significance: significance(query.screen_coverage, query.speed, query.speed_ref),
        out_idle_frames: sleep.idle_frames,
        out_asleep: sleep.asleep,
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

/// A compiled, reusable sort/cull decision pipeline.
pub struct GpuSortCull {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSortCull {
    /// Compiles the sort/cull kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSortCull {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sort_cull"),
            source: ShaderSource::Wgsl(SORT_CULL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sort_cull_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sort_cull_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sort_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sort_cull_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSortCull {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries`, returning one [`GpuSortCullResult`] per
    /// query in input order.
    ///
    /// The returned result for query `q` mirrors every golden
    /// [`sort_cull`](prism_render_architecture::particle::sort_cull) decision
    /// evaluated on `q`, matching [`cpu_reference`] lane for lane. An empty
    /// `queries` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so it is handled by an early return before any dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[GpuSortCullQuery]) -> Vec<GpuSortCullResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<RawQuery> = queries.iter().map(RawQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<RawResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sort_cull_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sort_cull_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sort_cull_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sort_cull_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sort_cull_bind_group"),
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
            label: Some("prism_volumetric_sort_cull_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sort_cull_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
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
        let raw = bytemuck::cast_slice::<u8, RawResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
    }
}
