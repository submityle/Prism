//! `wgpu` compute twin of the analytic ray-cone intersection contract
//! ([`ray_cone`](prism_render_architecture::particle::ray_cone), particle
//! design §10, §14).
//!
//! The `CPU` golden
//! [`ray_cone`](prism_render_architecture::particle::ray_cone) owns the
//! closed-form solution of the ray-against-a-quadric-cone problem. The cone is
//! described by an apex point, a unit `axis` pointing toward the opening
//! direction, and the **cosine of its half-angle** (`cos_half_angle`, strictly
//! inside `(0, 1)`). The infinite variant
//! [`ray_infinite_cone`](prism_render_architecture::particle::ray_cone::ray_infinite_cone)
//! solves the scalar quadratic obtained by substituting `origin + t * dir` into
//! the cone identity `dot(v, axis)^2 = cos^2 * dot(v, v)` (with
//! `v = origin + t * dir - apex`); the finite variant
//! [`ray_finite_cone`](prism_render_architecture::particle::ray_cone::ray_finite_cone)
//! additionally clips the lateral surface to the axial band
//! `0 <= proj <= height` and closes the wide end with a flat base cap.
//! [`GpuRayCone`] is the on-device twin: it runs one thread per query and
//! reproduces the same answers the reference produces, so a passing real-device
//! parity test is direct evidence the ported kernel solves the same geometry
//! and classifies the same degenerate cases, not merely that its shader
//! compiles.
//!
//! # What is twinned
//!
//! Both reference entry points are reproduced per query, selected by the
//! `kind` discriminant (`0` = infinite, `1` = finite): the nearest forward hit
//! (`RayConeHit`'s ray parameter `t` and world-space `point`), or a clean miss.
//! The degenerate cases the reference handles explicitly are mirrored branch
//! for branch: an out-of-range cosine (`cos_out_of_range`) is rejected; a
//! non-positive `height` is rejected for the finite cone; a near-zero direction
//! is rejected rather than producing a `NaN`; the quadratic solver
//! (`solve_quadratic`) degrades to the linear case when the leading coefficient
//! is negligible and to a single double root when the discriminant is grazing;
//! and every candidate root on the reflected backward nappe (where
//! `dot(hit - apex, axis) < 0`) is culled.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `sqrt`, `+ - * /` — with no `sin`, `cos`, `tan`, `exp`, `log`, `pow` or
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. The only non-rational operation is the `sqrt` of the discriminant,
//! matching the reference's single `f32::sqrt` call.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 4e-3 + 1e-3 * (|a| + |b|)`, matching the reference's grazing
//! double-root `sqrt`-scale slack) on the `f32` fields while pinning the hit
//! flag exactly, tight enough to catch a genuinely wrong port (a dropped nappe
//! cull, a wrong cap radius, a wrong band clip) yet loose enough to admit legal
//! fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`ray_cone`](prism_render_architecture::particle::ray_cone) plus `wgpu`
//! compute dispatch; no third-party engine source or derived code.

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
/// shared by every one-thread-per-element kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Query discriminant selecting the **infinite** cone solver, mirroring the
/// reference
/// [`ray_infinite_cone`](prism_render_architecture::particle::ray_cone::ray_infinite_cone).
pub const KIND_INFINITE: u32 = 0;
/// Query discriminant selecting the **finite** capped cone solver, mirroring
/// the reference
/// [`ray_finite_cone`](prism_render_architecture::particle::ray_cone::ray_finite_cone).
pub const KIND_FINITE: u32 = 1;

/// Discrete hit code written by the kernel for a lane that strikes the cone:
/// matches the host `== 1` decode in [`decode_result`]. A direct `f32` equality
/// is forbidden, so the kernel emits an integer flag rather than a sentinel
/// float.
const CODE_HIT: u32 = 1;

/// The portable core-`WGSL` ray-cone kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the `CPU`
/// golden
/// [`ray_cone`](prism_render_architecture::particle::ray_cone) branch for
/// branch; see the module documentation for the algorithm.
const RAY_CONE_WGSL: &str = r#"
// Analytic ray-cone twin: one thread per query substitutes the ray into the
// quadric cone identity, solves the scalar quadratic, culls the reflected
// backward nappe and (for the finite kind) clips the lateral band and closes
// the wide end with a flat base cap, reporting the nearest forward hit flag,
// its ray parameter and world-space point. It mirrors the CPU golden
// `particle::ray_cone`, uses only the portable core-WGSL subset (min/max/abs/
// sqrt and + - * /) and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::ray_cone; no third-party
// engine source or derived code.

// Epsilon used to guard divisions, classify the quadratic discriminant, and
// compare quantities against zero without ever writing an exact == / != on an
// f32, matching the reference `CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;
// A finite sentinel standing in for the "no candidate yet" running best ray
// parameter; every real fixture parameter is many orders of magnitude smaller.
const T_SENTINEL: f32 = 1.0e30;
// Query discriminant selecting the infinite cone solver.
const KIND_INFINITE: u32 = 0u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 64-byte std430 stride matching the host `GpuQuery`: the ray origin
// and direction, the apex and axis, each on its own 16-byte-aligned slot with a
// trailing scalar (cos_half_angle, height, kind) packed into the padding lane.
struct Query {
    origin: vec3<f32>,
    cos_half_angle: f32,
    dir: vec3<f32>,
    height: f32,
    apex: vec3<f32>,
    kind: u32,
    axis: vec3<f32>,
    pad0: f32,
}

// One result. 32-byte std430 stride matching the host `GpuResult`: the hit flag
// as 0u/1u, the forward-hit parameter, two pad words, then the world-space hit
// point on its own 16-byte slot.
struct Result {
    hit: u32,
    t: f32,
    pad0: u32,
    pad1: u32,
    point: vec3<f32>,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Real roots of a t^2 + b t + c = 0 in the first `count` lanes; `count` is 0,
// 1, or 2. Roots are not sorted; callers scan them for the best candidate.
struct Roots {
    t0: f32,
    t1: f32,
    count: u32,
}

// The cone quadratic coefficients (a, b, c) plus the axial scalars cd and dd,
// so the axial coordinate of a hit at parameter t is cd + t * dd.
struct Coeffs {
    a: f32,
    b: f32,
    c: f32,
    cd: f32,
    dd: f32,
}

// A single forward hit: whether it exists, its ray parameter and its point.
struct Hit {
    hit: bool,
    t: f32,
    point: vec3<f32>,
}

// Solves `a t^2 + b t + c = 0`, degrading to the linear case when `a` is
// negligible and to a single double root when the discriminant is negligible,
// mirroring the reference `solve_quadratic`.
fn solve_quadratic(a: f32, b: f32, c: f32) -> Roots {
    var out: Roots;
    out.t0 = 0.0;
    out.t1 = 0.0;
    out.count = 0u;
    if (abs(a) < CMP_EPS) {
        // Near-linear: b t + c = 0.
        if (abs(b) < CMP_EPS) {
            return out;
        }
        out.t0 = -c / b;
        out.count = 1u;
        return out;
    }
    let disc = b * b - 4.0 * a * c;
    if (disc < -CMP_EPS) {
        return out;
    }
    let sqrt_disc = sqrt(max(disc, 0.0));
    if (sqrt_disc < CMP_EPS) {
        // Grazing: a single double root.
        out.t0 = -b / (2.0 * a);
        out.count = 1u;
        return out;
    }
    let inv = 0.5 / a;
    out.t0 = (-b - sqrt_disc) * inv;
    out.t1 = (-b + sqrt_disc) * inv;
    out.count = 2u;
    return out;
}

// Rejects a half-angle cosine that is not strictly inside (0, 1), mirroring the
// reference `cos_out_of_range`.
fn cos_out_of_range(cos_half_angle: f32) -> bool {
    if (cos_half_angle <= CMP_EPS) {
        return true;
    }
    return cos_half_angle >= 1.0;
}

// Builds the cone quadratic coefficients and axial scalars shared by both the
// infinite and finite solvers, mirroring the reference `cone_coefficients`.
fn cone_coefficients(
    origin: vec3<f32>,
    dir: vec3<f32>,
    apex: vec3<f32>,
    axis: vec3<f32>,
    cos_half_angle: f32,
) -> Coeffs {
    let co = origin - apex;
    let dd = dot(dir, axis);
    let cd = dot(co, axis);
    let dirdir = dot(dir, dir);
    let codir = dot(co, dir);
    let coco = dot(co, co);
    let k2 = cos_half_angle * cos_half_angle;
    var out: Coeffs;
    out.a = dd * dd - k2 * dirdir;
    out.b = 2.0 * (cd * dd - k2 * codir);
    out.c = cd * cd - k2 * coco;
    out.cd = cd;
    out.dd = dd;
    return out;
}

// Intersects a ray with an infinite cone, returning the nearest forward hit on
// the forward nappe; backward-nappe roots are culled. Mirrors the reference
// `ray_infinite_cone`.
fn ray_infinite_cone(
    origin: vec3<f32>,
    dir: vec3<f32>,
    apex: vec3<f32>,
    axis: vec3<f32>,
    cos_half_angle: f32,
) -> Hit {
    var out: Hit;
    out.hit = false;
    out.t = 0.0;
    out.point = vec3<f32>(0.0, 0.0, 0.0);
    if (cos_out_of_range(cos_half_angle)) {
        return out;
    }
    if (dot(dir, dir) < CMP_EPS) {
        return out;
    }
    let co = cone_coefficients(origin, dir, apex, axis, cos_half_angle);
    let roots = solve_quadratic(co.a, co.b, co.c);
    var best_t: f32 = T_SENTINEL;
    var found = false;
    if (roots.count >= 1u) {
        let t = roots.t0;
        // Forward only, and cull the reflected backward nappe.
        if (t >= 0.0 && co.cd + t * co.dd >= -CMP_EPS && t < best_t) {
            best_t = t;
            found = true;
        }
    }
    if (roots.count >= 2u) {
        let t = roots.t1;
        if (t >= 0.0 && co.cd + t * co.dd >= -CMP_EPS && t < best_t) {
            best_t = t;
            found = true;
        }
    }
    if (found) {
        out.hit = true;
        out.t = best_t;
        out.point = origin + dir * best_t;
    }
    return out;
}

// Intersects a ray with a finite capped cone: the infinite solve restricted to
// the axial band [0, height] plus a flat base cap of radius height * sin / cos.
// Mirrors the reference `ray_finite_cone`.
fn ray_finite_cone(
    origin: vec3<f32>,
    dir: vec3<f32>,
    apex: vec3<f32>,
    axis: vec3<f32>,
    cos_half_angle: f32,
    height: f32,
) -> Hit {
    var out: Hit;
    out.hit = false;
    out.t = 0.0;
    out.point = vec3<f32>(0.0, 0.0, 0.0);
    if (cos_out_of_range(cos_half_angle)) {
        return out;
    }
    if (height <= CMP_EPS) {
        return out;
    }
    if (dot(dir, dir) < CMP_EPS) {
        return out;
    }
    let co = cone_coefficients(origin, dir, apex, axis, cos_half_angle);
    let roots = solve_quadratic(co.a, co.b, co.c);
    var best_t: f32 = T_SENTINEL;
    var found = false;

    // Lateral surface, clipped to the axial band [0, height].
    if (roots.count >= 1u) {
        let t = roots.t0;
        let axial = co.cd + t * co.dd;
        if (t >= 0.0 && axial >= -CMP_EPS && axial <= height + CMP_EPS && t < best_t) {
            best_t = t;
            found = true;
        }
    }
    if (roots.count >= 2u) {
        let t = roots.t1;
        let axial = co.cd + t * co.dd;
        if (t >= 0.0 && axial >= -CMP_EPS && axial <= height + CMP_EPS && t < best_t) {
            best_t = t;
            found = true;
        }
    }

    // Base cap: the flat disk that closes the wide end.
    if (abs(co.dd) > CMP_EPS) {
        let t = (height - co.cd) / co.dd;
        if (t >= 0.0) {
            let hit = origin + dir * t;
            let center = apex + axis * height;
            let radial = hit - center;
            let k2 = cos_half_angle * cos_half_angle;
            let sin2 = 1.0 - k2;
            let radius2 = height * height * sin2 / k2;
            if (dot(radial, radial) <= radius2 + CMP_EPS && t < best_t) {
                best_t = t;
                found = true;
            }
        }
    }

    if (found) {
        out.hit = true;
        out.t = best_t;
        out.point = origin + dir * best_t;
    }
    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var hit: Hit;
    if (q.kind == KIND_INFINITE) {
        hit = ray_infinite_cone(q.origin, q.dir, q.apex, q.axis, q.cos_half_angle);
    } else {
        hit = ray_finite_cone(q.origin, q.dir, q.apex, q.axis, q.cos_half_angle, q.height);
    }

    var out: Result;
    out.hit = 0u;
    out.t = 0.0;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.point = vec3<f32>(0.0, 0.0, 0.0);
    out.pad2 = 0.0;
    if (hit.hit) {
        // Option -> u32 flag re-mapping: miss leaves t / point at zero.
        out.hit = 1u;
        out.t = hit.t;
        out.point = hit.point;
    }
    results[idx] = out;
}
"#;

/// One ray-cone query: the ray, the cone geometry and a `kind` discriminant
/// selecting the infinite ([`KIND_INFINITE`]) or finite ([`KIND_FINITE`])
/// solver.
///
/// Mirrors a single reference
/// [`ray_infinite_cone`](prism_render_architecture::particle::ray_cone::ray_infinite_cone)
/// or
/// [`ray_finite_cone`](prism_render_architecture::particle::ray_cone::ray_finite_cone)
/// call. The `axis` is expected pre-normalized by the caller, exactly as the
/// reference expects. Derives only [`PartialEq`] (no `Eq` / `Hash`) because it
/// holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayConeQuery {
    /// Ray origin.
    pub origin: [f32; 3],
    /// Ray direction (need not be unit length; a near-zero direction misses).
    pub dir: [f32; 3],
    /// Cone apex point.
    pub apex: [f32; 3],
    /// Unit axis pointing toward the cone's opening direction.
    pub axis: [f32; 3],
    /// Cosine of the cone half-angle, strictly inside `(0, 1)`.
    pub cos_half_angle: f32,
    /// Axial height of the finite cone; ignored when `kind` is
    /// [`KIND_INFINITE`].
    pub height: f32,
    /// Solver discriminant: [`KIND_INFINITE`] or [`KIND_FINITE`].
    pub kind: u32,
}

impl RayConeQuery {
    /// Builds an infinite-cone query ([`KIND_INFINITE`]); `height` is unused and
    /// stored as `0.0`.
    #[must_use]
    pub const fn infinite(
        origin: [f32; 3],
        dir: [f32; 3],
        apex: [f32; 3],
        axis: [f32; 3],
        cos_half_angle: f32,
    ) -> RayConeQuery {
        RayConeQuery {
            origin,
            dir,
            apex,
            axis,
            cos_half_angle,
            height: 0.0,
            kind: KIND_INFINITE,
        }
    }

    /// Builds a finite capped-cone query ([`KIND_FINITE`]) with axial `height`.
    #[must_use]
    pub const fn finite(
        origin: [f32; 3],
        dir: [f32; 3],
        apex: [f32; 3],
        axis: [f32; 3],
        cos_half_angle: f32,
        height: f32,
    ) -> RayConeQuery {
        RayConeQuery {
            origin,
            dir,
            apex,
            axis,
            cos_half_angle,
            height,
            kind: KIND_FINITE,
        }
    }
}

/// The resolved verdict for one query, the host-side mirror of the kernel's
/// `Result` lane and of the reference's `Option<RayConeHit>`.
///
/// `hit` is the exact flag (`1` when the forward ray strikes the cone, `0`
/// otherwise); `t` and `point` carry the nearest forward hit and are meaningful
/// only when `hit` is `1` (otherwise both are zero). Derives only [`PartialEq`]
/// (no `Eq` / `Hash`) because it holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayConeResult {
    /// Hit flag: `1` when the forward ray strikes the cone, `0` on a miss.
    pub hit: u32,
    /// Ray parameter of the nearest forward hit; meaningful only when `hit`.
    pub t: f32,
    /// World-space hit point; meaningful only when `hit`, else zero.
    pub point: [f32; 3],
}

/// `repr(C)` `std430` layout of one packed query: four `vec4` slots holding
/// `(origin.xyz, cos_half_angle)`, `(dir.xyz, height)`, `(apex.xyz, kind)` and
/// `(axis.xyz, pad)` — `64` bytes, each `vec3` on its `16`-byte-aligned slot
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin.
    origin: [f32; 3],
    /// Cone half-angle cosine, packed into the origin slot's fourth lane.
    cos_half_angle: f32,
    /// Ray direction.
    dir: [f32; 3],
    /// Finite-cone height, packed into the direction slot's fourth lane.
    height: f32,
    /// Cone apex point.
    apex: [f32; 3],
    /// Solver discriminant, packed into the apex slot's fourth lane.
    kind: u32,
    /// Unit cone axis.
    axis: [f32; 3],
    /// Padding lane after the axis.
    pad0: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &RayConeQuery) -> GpuQuery {
        GpuQuery {
            origin: query.origin,
            cos_half_angle: query.cos_half_angle,
            dir: query.dir,
            height: query.height,
            apex: query.apex,
            kind: query.kind,
            axis: query.axis,
            pad0: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: the hit flag and parameter with two
/// pad words, then one `vec4` slot for the hit point — `32` bytes matching the
/// `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Hit flag (`1` = hit).
    hit: u32,
    /// Nearest forward-hit ray parameter.
    t: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// World-space hit point.
    point: [f32; 3],
    /// Padding lane after the hit point.
    pad2: f32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Maps one kernel `Result` lane back to the host [`RayConeResult`], unpacking
/// the integer hit flag into the public shape.
fn decode_result(raw: &GpuResult) -> RayConeResult {
    let hit = u32::from(raw.hit == CODE_HIT);
    RayConeResult {
        hit,
        t: raw.t,
        point: raw.point,
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

/// A compiled, reusable ray-cone compute pipeline.
pub struct GpuRayCone {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayCone {
    /// Compiles the ray-cone kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayCone {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_cone"),
            source: ShaderSource::Wgsl(RAY_CONE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_cone_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_cone_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_cone_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayCone {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`RayConeResult`] per input,
    /// in order.
    ///
    /// Each result equals the reference answer (`ray_infinite_cone` or
    /// `ray_finite_cone`, selected by `kind`) to within the tolerance documented
    /// on this module. An empty input returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn intersect(&self, ctx: &GpuContext, queries: &[RayConeQuery]) -> Vec<RayConeResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_cone_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_cone_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_cone_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_cone_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_cone_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ray_cone_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_cone_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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
        debug_assert_eq!(raw.len(), count);

        raw.iter().map(decode_result).collect()
    }
}
