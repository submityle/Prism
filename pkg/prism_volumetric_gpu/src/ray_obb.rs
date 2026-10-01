//! `wgpu` compute twin of the analytic ray vs *oriented bounding box* (`OBB`)
//! slab-method golden
//! ([`ray_obb`](prism_render_architecture::particle::ray_obb), design §10, §14).
//!
//! The particle subsystem's picking, collision-probe and analytic-primitive
//! raytrace contracts need, per `(ray, box)` pair, the ordered slab chord
//! `[t_enter, t_exit]`, the unsigned "does the forward half-line hit" predicate
//! and the single nearest forward surface crossing with its world-space point
//! and *outward* unit face normal. The `CPU` golden
//! [`Obb::span`](prism_render_architecture::particle::ray_obb::Obb::span),
//! [`Obb::first_hit`](prism_render_architecture::particle::ray_obb::Obb::first_hit)
//! and
//! [`Obb::intersects`](prism_render_architecture::particle::ray_obb::Obb::intersects)
//! own that math; [`GpuRayObb`] is the on-device twin that runs one thread per
//! query and reproduces every lane. A passing real-device parity test is
//! therefore direct evidence the ported kernel transforms the ray into the box
//! local frame and folds the same three guarded divisions the reference does,
//! not merely that its shader compiles.
//!
//! # What is twinned
//!
//! An `OBB` is an axis-aligned box rotated into world space: a `center`, three
//! mutually orthogonal **unit** axes `axis_u`, `axis_v`, `axis_w`, and a
//! non-negative half-extent along each. Solving the ray intersection reduces to
//! *projecting the ray onto the three local axes* — equivalently transforming
//! the ray into the box local frame by dot products — and running the classic
//! three-slab test, intersecting the per-axis parameter intervals into a single
//! ordered `[t_enter, t_exit]` span. The kernel reproduces the reference
//! `slab_span` guard-for-guard: a zero-length direction is rejected first
//! (`dot(dir, dir)` below [`EPS`](prism_render_architecture::particle::ray_obb::EPS)
//! squared); an axis whose *projected* direction magnitude is below `EPS` is
//! *parallel* and divides nothing — it contributes a miss when the projected
//! origin lies outside that `[-half, half]` slab and the sentinel interval
//! `(-inf, +inf)` otherwise; every other axis forms the reciprocal `1 / d`,
//! orders its `[-half, +half]` face parameters into `[t_near, t_far]` carrying
//! the outward face normal (the signed axis), and folds them into the running
//! chord. From that one chord the kernel derives the forward-ray hit flag
//! (`t_exit >= 0`) and the first visible forward crossing — the entry face when
//! it is in front, otherwise the exit face (the origin is inside the box) —
//! together with that face's outward normal and the world point `origin + dir *
//! t`, each matching the reference `span`, `first_hit` and `intersects`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `+ - * /`, `dot` and one `bitcast` to materialize the `IEEE`-754 infinity
//! sentinel that mirrors the reference's `f32::INFINITY` running bounds — with
//! no `sin`, `cos`, `exp`, `log`, `pow`, `sqrt` or optional device feature (the
//! box faces are flat, so there is no quadratic). It therefore runs unmodified
//! on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of three guarded divisions
//! and a running interval intersection, so `CPU` and `GPU` evaluate the same
//! closed form in the same associativity. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the chord,
//! crossing, point and normal values yet an *exact* match on the discrete hit
//! flags, which are routed through the same `EPS` magnitude band the reference
//! uses so a query placed clear of a face boundary folds the identical boolean
//! verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ray_obb`；
//! standard slab-method ray/`OBB` intersection plus `wgpu` compute dispatch; no
//! third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::ray_obb::{Obb, Ray, Vec3};
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

/// Discrete hit code written by the kernel for a lane that intersects: matches
/// the host `== 1` decode in [`decode_result`]. A direct `f32` equality is
/// forbidden, so the kernel emits an integer flag rather than a sentinel float.
const CODE_HIT: u32 = 1;

/// The portable core-`WGSL` ray/`OBB` kernel, embedded inline so the twin ships
/// as a single source file. Mirrors the `CPU` golden
/// [`slab_span`](prism_render_architecture::particle::ray_obb) and its public
/// `span` / `first_hit` / `intersects` readings guard-for-guard; see the module
/// documentation for the algorithm.
const RAY_OBB_WGSL: &str = r#"
// Ray vs OBB slab-method twin: one thread per (ray, box) query transforms the
// ray into the box local frame by projecting onto the three unit axes, folds
// the three per-axis slab intervals into the ordered chord [t_enter, t_exit]
// (carrying the outward face normal at each crossing), then writes the span hit
// flag, the forward-ray hit flag (t_exit >= 0) and the first visible forward
// crossing with its outward normal and world point (or +inf / zero sentinels on
// a forward miss). It mirrors the CPU golden `particle::ray_obb` division guard
// for guard, uses only the portable core-WGSL subset (abs/min/max, dot,
// + - * / and one bitcast for the infinity sentinel), and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard slab-method ray/OBB intersection; no third-party engine
// source or derived code.

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 112-byte std430 stride matching the host `GpuQuery`: the ray
// origin and direction, the box center, its three local unit axes and its
// half-extents, each padded to a vec4 so the storage array needs no manual
// vec3 alignment arithmetic.
struct Query {
    origin: vec4<f32>,
    dir: vec4<f32>,
    center: vec4<f32>,
    axis_u: vec4<f32>,
    axis_v: vec4<f32>,
    axis_w: vec4<f32>,
    half_extents: vec4<f32>,
}

// One result. 64-byte std430 stride matching the host `GpuResult`: the span and
// forward hit flags as 0u/1u, the ordered chord endpoints, the first visible
// forward crossing (a +inf sentinel on a forward miss), the outward face normal
// and the world hit point (each in a vec4 lane), plus three pad words.
struct Result {
    span_hit: u32,
    forward_hit: u32,
    t_enter: f32,
    t_exit: f32,
    first_t: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    normal: vec4<f32>,
    point: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Magnitude floor guarding the parallel-slab `1 / d` division and classifying a
// projected direction component as parallel, matching the reference `EPS`. A
// direct f32 `==`/`!=` is forbidden, so the zero-direction and parallel tests
// compare magnitudes against this floor instead of exact zero.
const EPS: f32 = 1.0e-6;

// The resolved slab span: whether the three per-axis intervals overlap, the
// ordered entry/exit parameters when they do, and the outward face normal at
// each of the two crossings.
struct Span {
    hit: bool,
    t_enter: f32,
    t_exit: f32,
    enter_normal: vec3<f32>,
    exit_normal: vec3<f32>,
}

// The contribution of one box axis to the running slab span: whether the ray is
// parallel to and outside this slab (a whole-query miss), whether the axis
// actually constrains the interval (false for the parallel-but-inside case the
// caller skips), and the ordered near/far parameters with their outward normals.
struct AxisSlab {
    miss: bool,
    constrains: bool,
    t_near: f32,
    n_near: vec3<f32>,
    t_far: f32,
    n_far: vec3<f32>,
}

// Projects the ray onto one box axis and solves that single slab, mirroring one
// iteration of the reference `slab_span` loop: `o` and `d` are the origin-offset
// and direction projected onto the (unit) axis. When |d| is below EPS the ray
// is parallel: it misses if the projected origin is outside [-half, half] and
// otherwise contributes nothing. Otherwise `t_a` reaches the -half face
// (outward normal -axis) and `t_b` the +half face (outward normal +axis),
// ordered into [t_near, t_far].
fn axis_slab(axis: vec3<f32>, half: f32, rel: vec3<f32>, dir: vec3<f32>) -> AxisSlab {
    var r: AxisSlab;
    r.miss = false;
    r.constrains = false;
    r.t_near = 0.0;
    r.t_far = 0.0;
    r.n_near = vec3<f32>(0.0, 0.0, 0.0);
    r.n_far = vec3<f32>(0.0, 0.0, 0.0);

    let o = dot(axis, rel);
    let d = dot(axis, dir);

    if (abs(d) <= EPS) {
        if (o < -half - EPS || o > half + EPS) {
            r.miss = true;
        }
        return r;
    }

    let inv = 1.0 / d;
    let t_a = (-half - o) * inv;
    let t_b = (half - o) * inv;
    if (t_a <= t_b) {
        r.t_near = t_a;
        r.n_near = -axis;
        r.t_far = t_b;
        r.n_far = axis;
    } else {
        r.t_near = t_b;
        r.n_near = axis;
        r.t_far = t_a;
        r.n_far = -axis;
    }
    r.constrains = true;
    return r;
}

// Folds one axis slab into the running [t_enter, t_exit] span, matching the
// reference's strict `>` / `<` running-bound updates (ties keep the earlier
// normal; the fixtures stay clear of ties). Returns the updated span and sets
// `hit = false` the moment the interval becomes empty.
fn fold_axis(span: Span, a: AxisSlab) -> Span {
    var s = span;
    if (a.t_near > s.t_enter) {
        s.t_enter = a.t_near;
        s.enter_normal = a.n_near;
    }
    if (a.t_far < s.t_exit) {
        s.t_exit = a.t_far;
        s.exit_normal = a.n_far;
    }
    if (s.t_enter > s.t_exit) {
        s.hit = false;
    }
    return s;
}

// The shared slab solve: transforms the ray into the box local frame and
// intersects the three per-axis parameter intervals of the infinite line,
// returning the ordered span with its crossing normals when they overlap.
// Mirrors the reference `slab_span` guard for guard: a zero-length direction is
// rejected first (so a point never spans the sentinel interval); an axis whose
// projected direction magnitude is below EPS is parallel and divides nothing,
// contributing a miss when the projected origin lies outside that slab and the
// infinite sentinel interval (which the running min/max then skips) otherwise.
fn slab_span(
    origin: vec3<f32>,
    dir: vec3<f32>,
    center: vec3<f32>,
    axis_u: vec3<f32>,
    axis_v: vec3<f32>,
    axis_w: vec3<f32>,
    half_extents: vec3<f32>,
) -> Span {
    var out: Span;
    out.hit = false;
    out.t_enter = 0.0;
    out.t_exit = 0.0;
    out.enter_normal = vec3<f32>(0.0, 0.0, 0.0);
    out.exit_normal = vec3<f32>(0.0, 0.0, 0.0);

    let dd = dot(dir, dir);
    if (dd <= EPS * EPS) {
        return out;
    }

    // IEEE-754 +/-infinity sentinels, mirroring the reference's `f32::INFINITY`
    // running bounds exactly; built by bitcast because WGSL has no inf literal.
    let pos_inf = bitcast<f32>(0x7f800000u);
    let neg_inf = bitcast<f32>(0xff800000u);

    out.hit = true;
    out.t_enter = neg_inf;
    out.t_exit = pos_inf;

    let rel = origin - center;

    // Axis u.
    let au = axis_slab(axis_u, half_extents.x, rel, dir);
    if (au.miss) {
        out.hit = false;
        return out;
    }
    if (au.constrains) {
        out = fold_axis(out, au);
        if (!out.hit) {
            return out;
        }
    }

    // Axis v.
    let av = axis_slab(axis_v, half_extents.y, rel, dir);
    if (av.miss) {
        out.hit = false;
        return out;
    }
    if (av.constrains) {
        out = fold_axis(out, av);
        if (!out.hit) {
            return out;
        }
    }

    // Axis w.
    let aw = axis_slab(axis_w, half_extents.z, rel, dir);
    if (aw.miss) {
        out.hit = false;
        return out;
    }
    if (aw.constrains) {
        out = fold_axis(out, aw);
        if (!out.hit) {
            return out;
        }
    }

    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let origin = queries[idx].origin.xyz;
    let dir = queries[idx].dir.xyz;
    let center = queries[idx].center.xyz;
    let axis_u = queries[idx].axis_u.xyz;
    let axis_v = queries[idx].axis_v.xyz;
    let axis_w = queries[idx].axis_w.xyz;
    let half_extents = queries[idx].half_extents.xyz;

    let span = slab_span(origin, dir, center, axis_u, axis_v, axis_w, half_extents);
    let pos_inf = bitcast<f32>(0x7f800000u);

    var res: Result;
    res.pad0 = 0u;
    res.pad1 = 0u;
    res.pad2 = 0u;
    res.normal = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    res.point = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    if (span.hit) {
        res.span_hit = 1u;
        res.t_enter = span.t_enter;
        res.t_exit = span.t_exit;
    } else {
        res.span_hit = 0u;
        res.t_enter = pos_inf;
        res.t_exit = pos_inf;
    }

    // Forward-ray reading: visible only when some of the chord lies at or ahead
    // of the origin (t_exit >= 0), matching the reference `first_hit`.
    let forward = span.hit && span.t_exit >= 0.0;
    if (forward) {
        res.forward_hit = 1u;
        // The entry face when in front, else the exit face (the origin is inside
        // the box), matching `Obb::first_hit`.
        var t = span.t_exit;
        var n = span.exit_normal;
        if (span.t_enter >= 0.0) {
            t = span.t_enter;
            n = span.enter_normal;
        }
        res.first_t = t;
        res.normal = vec4<f32>(n, 0.0);
        res.point = vec4<f32>(origin + dir * t, 0.0);
    } else {
        res.forward_hit = 0u;
        res.first_t = pos_inf;
    }

    results[idx] = res;
}
"#;

/// One ray vs `OBB` query: the ray and the oriented box to test it against.
///
/// Mirrors a single reference
/// [`Obb::span`](prism_render_architecture::particle::ray_obb::Obb::span) /
/// [`Obb::first_hit`](prism_render_architecture::particle::ray_obb::Obb::first_hit)
/// call. Carrying the box per query lets one dispatch mix rays against many
/// distinct boxes. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds
/// `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayObbQuery {
    /// The ray whose chord and forward half-line are tested.
    pub ray: Ray,
    /// The oriented box the ray is tested against.
    pub obb: Obb,
}

/// The resolved verdict for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `span_hit` is the chord reading
/// ([`Obb::span`](prism_render_architecture::particle::ray_obb::Obb::span)) and
/// `forward_hit` the forward-ray reading
/// ([`Obb::intersects`](prism_render_architecture::particle::ray_obb::Obb::intersects),
/// equal to
/// [`Obb::first_hit`](prism_render_architecture::particle::ray_obb::Obb::first_hit)`.is_some()`).
/// `t_enter` and `t_exit` carry the ordered chord and are meaningful only when
/// `span_hit` is `true`; `first_t`, `normal` and `point` carry the nearest
/// forward crossing and are meaningful only when `forward_hit` is `true`
/// (otherwise `first_t` holds a `+inf` sentinel and `normal` / `point` are
/// zero). Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32`
/// parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayObbResult {
    /// Whether the *infinite line* through the ray crosses the box.
    pub span_hit: bool,
    /// Whether the *forward half-line* (`t >= 0`) strikes the box surface.
    pub forward_hit: bool,
    /// The chord entry parameter; meaningful only when `span_hit`.
    pub t_enter: f32,
    /// The chord exit parameter; meaningful only when `span_hit`.
    pub t_exit: f32,
    /// The nearest forward crossing parameter; meaningful only when `forward_hit`.
    pub first_t: f32,
    /// The outward unit face normal at the forward crossing; meaningful only
    /// when `forward_hit`.
    pub normal: Vec3,
    /// The world-space forward crossing point; meaningful only when `forward_hit`.
    pub point: Vec3,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`RAY_OBB_WGSL`]: the query count and three pad words — `16`
/// bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `112`-byte `std430` stride matching `Query` in the
/// shader: the ray origin and direction, the box center, its three local unit
/// axes and its half-extents, each padded to a `vec4` lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin in `xyz`; the `w` lane is unused padding.
    origin: [f32; 4],
    /// Ray direction in `xyz`; the `w` lane is unused padding.
    dir: [f32; 4],
    /// Box center in `xyz`; the `w` lane is unused padding.
    center: [f32; 4],
    /// First local unit axis in `xyz`; the `w` lane is unused padding.
    axis_u: [f32; 4],
    /// Second local unit axis in `xyz`; the `w` lane is unused padding.
    axis_v: [f32; 4],
    /// Third local unit axis in `xyz`; the `w` lane is unused padding.
    axis_w: [f32; 4],
    /// Per-axis half-extents in `xyz`; the `w` lane is unused padding.
    half_extents: [f32; 4],
}

impl GpuQuery {
    /// Packs a [`RayObbQuery`] into the `std430` upload layout.
    fn from_query(query: &RayObbQuery) -> GpuQuery {
        let o = query.ray.origin;
        let d = query.ray.dir;
        let c = query.obb.center;
        let u = query.obb.axis_u;
        let v = query.obb.axis_v;
        let w = query.obb.axis_w;
        let h = query.obb.half_extents;
        GpuQuery {
            origin: [o.x, o.y, o.z, 0.0],
            dir: [d.x, d.y, d.z, 0.0],
            center: [c.x, c.y, c.z, 0.0],
            axis_u: [u.x, u.y, u.z, 0.0],
            axis_v: [v.x, v.y, v.z, 0.0],
            axis_w: [w.x, w.y, w.z, 0.0],
            half_extents: [h.x, h.y, h.z, 0.0],
        }
    }
}

/// One result as read back. `64`-byte `std430` stride matching `Result` in the
/// shader: the two hit flags as `0`/`1`, the ordered chord endpoints, the
/// nearest forward crossing, the outward face normal and world point (each in a
/// `vec4` lane) and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Span (infinite-line) hit flag (`1` = hit).
    span_hit: u32,
    /// Forward-ray hit flag (`1` = hit).
    forward_hit: u32,
    /// Chord entry parameter.
    t_enter: f32,
    /// Chord exit parameter.
    t_exit: f32,
    /// Nearest forward crossing (or a `+inf` sentinel on a forward miss).
    first_t: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Outward face normal in `xyz`; the `w` lane is unused padding.
    normal: [f32; 4],
    /// World hit point in `xyz`; the `w` lane is unused padding.
    point: [f32; 4],
}

/// Maps one kernel `Result` lane back to the host [`RayObbResult`].
fn decode_result(raw: &GpuResult) -> RayObbResult {
    RayObbResult {
        span_hit: raw.span_hit == CODE_HIT,
        forward_hit: raw.forward_hit == CODE_HIT,
        t_enter: raw.t_enter,
        t_exit: raw.t_exit,
        first_t: raw.first_t,
        normal: Vec3::new(raw.normal[0], raw.normal[1], raw.normal[2]),
        point: Vec3::new(raw.point[0], raw.point[1], raw.point[2]),
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

/// A compiled, reusable ray/`OBB` slab-test pipeline.
pub struct GpuRayObb {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayObb {
    /// Compiles the ray/`OBB` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayObb {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_obb"),
            source: ShaderSource::Wgsl(RAY_OBB_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_obb_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_obb_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_obb_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayObb {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries`, returning one [`RayObbResult`] per query
    /// in input order.
    ///
    /// The returned result for query `q` mirrors
    /// [`Obb::span`](prism_render_architecture::particle::ray_obb::Obb::span),
    /// [`Obb::first_hit`](prism_render_architecture::particle::ray_obb::Obb::first_hit)
    /// and
    /// [`Obb::intersects`](prism_render_architecture::particle::ray_obb::Obb::intersects)
    /// evaluated on `q.obb` and `q.ray`. An empty `queries` slice yields an
    /// empty result — storage buffers cannot be zero-sized, so it is handled by
    /// an early return before any dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[RayObbQuery]) -> Vec<RayObbResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_obb_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_obb_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_obb_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_obb_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_obb_bind_group"),
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
            label: Some("prism_volumetric_ray_obb_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_obb_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
    }
}
