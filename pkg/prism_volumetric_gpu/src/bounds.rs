//! `wgpu` compute twin of the per-query axis-aligned bounding box (`AABB`)
//! algebra and the first-pass dispatch sizing from the parallel bounds-reduction
//! contract ([`bounds`](prism_render_architecture::particle::bounds), particle
//! design §12, §13).
//!
//! The `CPU` golden [`bounds`](prism_render_architecture::particle::bounds) owns
//! two things: the small, verifiable box algebra every culling and
//! indirect-draw stage shares
//! ([`Aabb::expand_point`](prism_render_architecture::particle::bounds::Aabb::expand_point),
//! [`Aabb::union`](prism_render_architecture::particle::bounds::Aabb::union),
//! [`Aabb::center`](prism_render_architecture::particle::bounds::Aabb::center),
//! [`Aabb::half_extent`](prism_render_architecture::particle::bounds::Aabb::half_extent),
//! [`Aabb::surface_area`](prism_render_architecture::particle::bounds::Aabb::surface_area),
//! [`Aabb::longest_axis`](prism_render_architecture::particle::bounds::Aabb::longest_axis)
//! and the empty-box test
//! [`Aabb::is_empty`](prism_render_architecture::particle::bounds::Aabb::is_empty)),
//! and the integer sizing arithmetic the first reduction pass needs
//! ([`BoundsReduction::workgroup_count`](prism_render_architecture::particle::bounds::BoundsReduction::workgroup_count)
//! and
//! [`BoundsReduction::partial_count`](prism_render_architecture::particle::bounds::BoundsReduction::partial_count)).
//! [`GpuBounds`] is the on-device twin: one thread solves one query, so a
//! passing real-device parity test is direct evidence the ported kernel computes
//! the same box geometry and the same workgroup sizing the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! For a batch of independent queries, each carrying a primary box `a`, a
//! secondary box `b`, a probe point and a `(particle_count, workgroup_size)`
//! sizing pair, the kernel reproduces: the center and half-extent of `a`, its
//! surface area, its longest-axis code, its empty flag, the union of `a` and
//! `b`, the box `a` grown to contain the probe point, and the first-pass
//! workgroup / partial counts. The full two-level tree reduction (the
//! `while count > 1` fold and the `std430` partial-buffer byte sizing, which
//! needs `u64`) stays on the host; this wave twins only the per-query box
//! algebra and the single-level sizing, all of which is per-thread work with no
//! cross-thread dependency.
//!
//! # Correctness model
//!
//! The longest-axis code and the empty flag are discrete classifications built
//! from `f32` magnitude comparisons, and the workgroup / partial counts are
//! integer `div_ceil` arithmetic, so for inputs clear of the degeneracy and tie
//! thresholds the `CPU` and `GPU` agree exactly and the parity test asserts an
//! exact `==`. The center, half-extent, surface area, union corners and expanded
//! corners thread through adds, multiplies and `min` / `max`, so the parity test
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every
//! continuous quantity, tight enough to catch a genuinely wrong port yet loose
//! enough to admit legal fused multiply-add contraction in
//! [`Aabb::surface_area`](prism_render_architecture::particle::bounds::Aabb::surface_area).
//!
//! # Degenerate inputs
//!
//! An empty box seeds `min` at `f32::MAX` and `max` at `f32::MIN`, so its
//! center, half-extent, surface area and longest axis are meaningless (they
//! involve subtracting two sentinels and overflow to a non-finite value). The
//! host guards those: a parity fixture only compares the continuous geometry of
//! `a` when `a` is non-empty, and the empty flag is always compared exactly. The
//! union and the point-expansion are well defined even when `a` is empty — the
//! empty box is the union identity, so the union of an empty `a` with `b`
//! equals `b`, and expanding an empty `a` by a point yields the degenerate box
//! at that point — and remain finite, so they are always compared. A
//! `particle_count` of zero yields a workgroup count of zero, matching the
//! reference. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `+ - * /`, unsigned integer arithmetic and comparisons — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `sqrt` and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. There is no loop: each thread performs a fixed, bounded sequence of
//! arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::bounds`；无第三方引擎源码或衍生代码。
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

/// Longest-axis code for the x axis, matching the reference `0`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bounds::Aabb::longest_axis`；无第三方引擎源码或衍生代码。
pub const AXIS_X: u32 = 0;
/// Longest-axis code for the y axis, matching the reference `1`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bounds::Aabb::longest_axis`；无第三方引擎源码或衍生代码。
pub const AXIS_Y: u32 = 1;
/// Longest-axis code for the z axis, matching the reference `2`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bounds::Aabb::longest_axis`；无第三方引擎源码或衍生代码。
pub const AXIS_Z: u32 = 2;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
///
/// Provenance: 本仓孪生约定；无第三方引擎源码或衍生代码。
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` bounds kernel, embedded inline so the twin ships as
/// a single source file. The single entry point `solve` mirrors the `CPU` golden
/// [`bounds`](prism_render_architecture::particle::bounds) box algebra and
/// first-pass sizing; see the module documentation for the algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bounds`；无第三方引擎源码或衍生代码。
const BOUNDS_WGSL: &str = r#"
// Bounds twin: one thread per query reproduces the per-box center, half-extent,
// surface area, longest-axis code and empty flag, the union of two boxes, the
// box grown to contain a probe point, and the first-pass workgroup / partial
// counts. It mirrors the CPU golden `particle::bounds` function for function,
// uses only the portable core-WGSL subset (min/max and + - * / plus unsigned
// integer math), needs no sqrt and no transcendental call and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12. There is no loop, so
// the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::bounds；无第三方引擎
// 源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Primary box `a`: min and max corners, each a vec3 with a trailing pad lane.
    a_min: vec3<f32>,
    pad_amin: f32,
    a_max: vec3<f32>,
    pad_amax: f32,
    // Secondary box `b` for the union, same padded layout.
    b_min: vec3<f32>,
    pad_bmin: f32,
    b_max: vec3<f32>,
    pad_bmax: f32,
    // Probe point grown into box `a` by expand_point.
    point: vec3<f32>,
    pad_pt: f32,
    // First-pass sizing: particle count and the (unclamped) workgroup size.
    particle_count: u32,
    workgroup_size: u32,
    pad_s0: u32,
    pad_s1: u32,
}

struct Result {
    // center(a) and half_extent(a), each a vec3 with a trailing pad lane.
    center: vec3<f32>,
    pad_c: f32,
    half_extent: vec3<f32>,
    pad_h: f32,
    // union(a, b) corners.
    union_min: vec3<f32>,
    pad_umin: f32,
    union_max: vec3<f32>,
    pad_umax: f32,
    // box `a` grown to contain the probe point.
    expanded_min: vec3<f32>,
    pad_emin: f32,
    expanded_max: vec3<f32>,
    pad_emax: f32,
    // Scalar / integer answers packed into two vec4-aligned lanes.
    surface_area: f32,
    longest_axis: u32,
    is_empty: u32,
    workgroup_count: u32,
    partial_count: u32,
    pad_r0: u32,
    pad_r1: u32,
    pad_r2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Longest-axis code (0 = x, 1 = y, 2 = z) for the box spanned by `mn`..`mx`.
// Ties resolve toward the lower index with `>=`, mirroring the reference.
fn longest_axis(mn: vec3<f32>, mx: vec3<f32>) -> u32 {
    let dx = mx.x - mn.x;
    let dy = mx.y - mn.y;
    let dz = mx.z - mn.z;
    if (dx >= dy && dx >= dz) {
        return 0u;
    } else if (dy >= dz) {
        return 1u;
    }
    return 2u;
}

// Empty-box test: any axis with min > max holds no points, mirroring the
// reference `is_empty` (which never uses `==`).
fn is_empty(mn: vec3<f32>, mx: vec3<f32>) -> u32 {
    if (mn.x > mx.x || mn.y > mx.y || mn.z > mx.z) {
        return 1u;
    }
    return 0u;
}

// Total surface area 2 * (dx*dy + dy*dz + dz*dx), using only multiply-add,
// mirroring the reference `surface_area`.
fn surface_area(mn: vec3<f32>, mx: vec3<f32>) -> f32 {
    let dx = mx.x - mn.x;
    let dy = mx.y - mn.y;
    let dz = mx.z - mn.z;
    return 2.0 * (dx * dy + dy * dz + dz * dx);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // center and half-extent of box `a`.
    let center = (q.a_min + q.a_max) * 0.5;
    let half_extent = (q.a_max - q.a_min) * 0.5;

    // union(a, b): component-wise min of mins and max of maxes.
    let union_min = min(q.a_min, q.b_min);
    let union_max = max(q.a_max, q.b_max);

    // expand_point: grow box `a` to contain the probe point.
    let expanded_min = min(q.a_min, q.point);
    let expanded_max = max(q.a_max, q.point);

    // First-pass sizing: clamp the workgroup size to the valid [1, 1024] range,
    // then div_ceil(particle_count, workgroup_size) via the overflow-safe form.
    let ws = clamp(q.workgroup_size, 1u, 1024u);
    var workgroup_count: u32 = 0u;
    if (q.particle_count != 0u) {
        workgroup_count = (q.particle_count - 1u) / ws + 1u;
    }

    var out: Result;
    out.center = center;
    out.pad_c = 0.0;
    out.half_extent = half_extent;
    out.pad_h = 0.0;
    out.union_min = union_min;
    out.pad_umin = 0.0;
    out.union_max = union_max;
    out.pad_umax = 0.0;
    out.expanded_min = expanded_min;
    out.pad_emin = 0.0;
    out.expanded_max = expanded_max;
    out.pad_emax = 0.0;
    out.surface_area = surface_area(q.a_min, q.a_max);
    out.longest_axis = longest_axis(q.a_min, q.a_max);
    out.is_empty = is_empty(q.a_min, q.a_max);
    out.workgroup_count = workgroup_count;
    out.partial_count = workgroup_count;
    out.pad_r0 = 0u;
    out.pad_r1 = 0u;
    out.pad_r2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`BOUNDS_WGSL`].
///
/// Provenance: 本仓孪生 dispatch 约定；无第三方引擎源码或衍生代码。
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
/// Every `vec3` lane carries a trailing pad word so each stays `16`-byte aligned
/// on device.
///
/// Provenance: 本仓孪生 `std430` 打包约定；无第三方引擎源码或衍生代码。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Primary box `a` min corner.
    a_min: [f32; 3],
    /// Pad lane after `a_min`.
    pad_amin: f32,
    /// Primary box `a` max corner.
    a_max: [f32; 3],
    /// Pad lane after `a_max`.
    pad_amax: f32,
    /// Secondary box `b` min corner.
    b_min: [f32; 3],
    /// Pad lane after `b_min`.
    pad_bmin: f32,
    /// Secondary box `b` max corner.
    b_max: [f32; 3],
    /// Pad lane after `b_max`.
    pad_bmax: f32,
    /// Probe point grown into box `a`.
    point: [f32; 3],
    /// Pad lane after `point`.
    pad_pt: f32,
    /// Particle count for the first-pass sizing.
    particle_count: u32,
    /// Unclamped workgroup size for the first-pass sizing.
    workgroup_size: u32,
    /// Padding word.
    pad_s0: u32,
    /// Padding word.
    pad_s1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
///
/// Provenance: 本仓孪生 `std430` 打包约定；无第三方引擎源码或衍生代码。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `center` of box `a`.
    center: [f32; 3],
    /// Pad lane after `center`.
    pad_c: f32,
    /// `half_extent` of box `a`.
    half_extent: [f32; 3],
    /// Pad lane after `half_extent`.
    pad_h: f32,
    /// `union` min corner.
    union_min: [f32; 3],
    /// Pad lane after `union_min`.
    pad_umin: f32,
    /// `union` max corner.
    union_max: [f32; 3],
    /// Pad lane after `union_max`.
    pad_umax: f32,
    /// `expand_point` min corner.
    expanded_min: [f32; 3],
    /// Pad lane after `expanded_min`.
    pad_emin: f32,
    /// `expand_point` max corner.
    expanded_max: [f32; 3],
    /// Pad lane after `expanded_max`.
    pad_emax: f32,
    /// `surface_area` of box `a`.
    surface_area: f32,
    /// `longest_axis` code of box `a` (`0` = x, `1` = y, `2` = z).
    longest_axis: u32,
    /// `1` when box `a` is empty, `0` otherwise.
    is_empty: u32,
    /// First-pass `workgroup_count`.
    workgroup_count: u32,
    /// First-pass `partial_count` (equals `workgroup_count`).
    partial_count: u32,
    /// Padding word.
    pad_r0: u32,
    /// Padding word.
    pad_r1: u32,
    /// Padding word.
    pad_r2: u32,
}

/// One query for the bounds twin: a primary box `a`, a secondary box `b` for the
/// union, a probe point for the point-expansion, and a `(particle_count,
/// workgroup_size)` pair for the first-pass sizing.
///
/// The box algebra and the sizing are independent, so a single query exercises
/// every twinned function at once.
///
/// Provenance: 本模块新建输入类型；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBoundsQuery {
    /// Primary box `a` min corner.
    pub a_min: [f32; 3],
    /// Primary box `a` max corner.
    pub a_max: [f32; 3],
    /// Secondary box `b` min corner, unioned with `a`.
    pub b_min: [f32; 3],
    /// Secondary box `b` max corner, unioned with `a`.
    pub b_max: [f32; 3],
    /// Probe point grown into box `a` by `expand_point`.
    pub point: [f32; 3],
    /// Particle count fed to the first-pass sizing.
    pub particle_count: u32,
    /// Workgroup size fed to the first-pass sizing; clamped to `[1, 1024]` on
    /// device, matching the reference.
    pub workgroup_size: u32,
}

/// One resolved answer for a single query, mirroring every value the reference
/// reports across its twinned functions.
///
/// Provenance: 本模块新建输出类型；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBoundsResult {
    /// `center` of box `a`, matching
    /// [`Aabb::center`](prism_render_architecture::particle::bounds::Aabb::center).
    pub center: [f32; 3],
    /// `half_extent` of box `a`, matching
    /// [`Aabb::half_extent`](prism_render_architecture::particle::bounds::Aabb::half_extent).
    pub half_extent: [f32; 3],
    /// `surface_area` of box `a`, matching
    /// [`Aabb::surface_area`](prism_render_architecture::particle::bounds::Aabb::surface_area).
    pub surface_area: f32,
    /// Longest-axis code of box `a` (`AXIS_X`, `AXIS_Y` or `AXIS_Z`), matching
    /// [`Aabb::longest_axis`](prism_render_architecture::particle::bounds::Aabb::longest_axis).
    pub longest_axis: u32,
    /// `true` when box `a` is empty, matching
    /// [`Aabb::is_empty`](prism_render_architecture::particle::bounds::Aabb::is_empty).
    pub is_empty: bool,
    /// `union(a, b)` min corner, matching
    /// [`Aabb::union`](prism_render_architecture::particle::bounds::Aabb::union).
    pub union_min: [f32; 3],
    /// `union(a, b)` max corner, matching
    /// [`Aabb::union`](prism_render_architecture::particle::bounds::Aabb::union).
    pub union_max: [f32; 3],
    /// Box `a` grown to contain the probe point, min corner, matching
    /// [`Aabb::expand_point`](prism_render_architecture::particle::bounds::Aabb::expand_point).
    pub expanded_min: [f32; 3],
    /// Box `a` grown to contain the probe point, max corner, matching
    /// [`Aabb::expand_point`](prism_render_architecture::particle::bounds::Aabb::expand_point).
    pub expanded_max: [f32; 3],
    /// First-pass workgroup count, matching
    /// [`BoundsReduction::workgroup_count`](prism_render_architecture::particle::bounds::BoundsReduction::workgroup_count).
    pub workgroup_count: u32,
    /// First-pass partial count, matching
    /// [`BoundsReduction::partial_count`](prism_render_architecture::particle::bounds::BoundsReduction::partial_count).
    pub partial_count: u32,
}

/// Encodes one [`GpuBoundsQuery`] into its `std430` [`GpuQuery`] slot.
///
/// Provenance: 本模块主机侧打包；无第三方引擎源码或衍生代码。
fn encode_query(q: &GpuBoundsQuery) -> GpuQuery {
    GpuQuery {
        a_min: q.a_min,
        pad_amin: 0.0,
        a_max: q.a_max,
        pad_amax: 0.0,
        b_min: q.b_min,
        pad_bmin: 0.0,
        b_max: q.b_max,
        pad_bmax: 0.0,
        point: q.point,
        pad_pt: 0.0,
        particle_count: q.particle_count,
        workgroup_size: q.workgroup_size,
        pad_s0: 0,
        pad_s1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GpuBoundsResult`].
///
/// Provenance: 本模块主机侧解包；无第三方引擎源码或衍生代码。
fn decode_result(raw: &GpuResult) -> GpuBoundsResult {
    GpuBoundsResult {
        center: raw.center,
        half_extent: raw.half_extent,
        surface_area: raw.surface_area,
        longest_axis: raw.longest_axis,
        is_empty: raw.is_empty != 0,
        union_min: raw.union_min,
        union_max: raw.union_max,
        expanded_min: raw.expanded_min,
        expanded_max: raw.expanded_max,
        workgroup_count: raw.workgroup_count,
        partial_count: raw.partial_count,
    }
}

/// Builds a compute-visible buffer binding layout entry.
///
/// Provenance: 本仓孪生绑定布局约定；无第三方引擎源码或衍生代码。
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

/// A compiled, reusable bounds compute pipeline, twinning the `CPU` golden
/// [`bounds`](prism_render_architecture::particle::bounds).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bounds`；无第三方引擎源码或衍生代码。
pub struct GpuBounds {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBounds {
    /// Compiles the bounds kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::bounds`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBounds {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bounds"),
            source: ShaderSource::Wgsl(BOUNDS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bounds_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bounds_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bounds_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBounds {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`GpuBoundsResult`] per
    /// input, in order.
    ///
    /// The longest-axis code, empty flag and workgroup / partial counts equal
    /// the reference exactly for inputs clear of the degeneracy and tie
    /// thresholds; the continuous box geometry matches to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::bounds`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[GpuBoundsQuery]) -> Vec<GpuBoundsResult> {
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
            label: Some("prism_volumetric_bounds_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bounds_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bounds_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bounds_bind_group"),
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
            label: Some("prism_volumetric_bounds_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bounds_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bounds_pass"),
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
