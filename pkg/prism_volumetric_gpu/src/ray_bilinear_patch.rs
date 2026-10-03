//! `wgpu` compute twin of the analytic ray vs *bilinear patch* intersector
//! (`prism_render_architecture::ray_scene::bilinear_patch`).
//!
//! A bilinear patch is the ruled surface spanned by four corners `p00`, `p10`,
//! `p11`, `p01`:
//!
//! ```text
//! P(u, v) = (1 - u)(1 - v)*p00 + u(1 - v)*p10 + u*v*p11 + (1 - u)*v*p01
//! ```
//!
//! with `u, v` in `[0, 1]`. The golden `BilinearPatch::intersect` follows
//! Reshetov's *"Cool Patches"* (Ray Tracing Gems, 2019): for a fixed `u` the
//! patch is a straight segment in `v`, so the ray is reduced to a quadratic in
//! `u` whose (at most two) roots are each back-substituted to recover `v` and
//! the ray parameter `t`. [`GpuRayBilinearPatch`] is the on-device twin that
//! runs one thread per `(ray, patch)` query and reproduces every lane: the
//! quadratic coefficients `a`, `b`, `c`, the stable root pair, the per-root
//! `solve_v` back-substitution, and the oriented analytic normal.
//!
//! # What is twinned
//!
//! The kernel reproduces the golden closed form step for step. The quadratic
//! `a*u^2 + b*u + c = 0` is formed from the translation-invariant edge vectors
//! and the ray; a negative discriminant is a miss. When the quadratic is
//! numerically planar in `u` (coefficient `c` within [`EPS`] of zero) the kernel
//! falls back to the linear branch `b*u + a = 0`, exactly as the reference does
//! for `c == 0`. Each root in `[0, 1]` is back-substituted: the vertical line at
//! that `u` is intersected to recover `(t, v)`, rejecting the root when the ray
//! is parallel to that line (`dot(n0, n0) <= 0`), when `v` leaves `[0, 1]`, or
//! when `t` leaves the ray interval. The surviving nearest root yields the unit
//! surface normal `dP/du x dP/dv` oriented against the ray, with `front_face`
//! recording whether the ray struck the outward side first.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `select`, `dot`, `cross`, a single `sqrt` and `+ - * /` — with no `sin`,
//! `cos`, `exp`, `log`, `pow` or optional device feature, so it runs unmodified
//! on Metal, Vulkan and DX12. A direct `f32` equality is forbidden, so the
//! planar-quadratic test compares `abs(c)` and `abs(b)` against [`EPS`] rather
//! than exact zero.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of the same products,
//! quotients and one `sqrt` the scalar reference evaluates, in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the continuous `t`, `u`, `v`
//! and normal components yet an *exact* match on the discrete `hit` and
//! `front_face` flags, with fixtures and the randomized sweep kept clear of the
//! branch-switch loci so both sides fold the identical verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::bilinear_patch`；无第三方引擎源码或衍生代码。

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` ray/bilinear-patch kernel, embedded inline so the
/// twin ships as a single source file. Mirrors the `CPU` golden
/// `BilinearPatch::intersect` step for step; see the module documentation for
/// the algorithm.
const RAY_BILINEAR_PATCH_WGSL: &str = r#"
// Ray vs bilinear-patch twin: one thread per (ray, patch) query reduces the ray
// to a quadratic in u (Reshetov "Cool Patches", 2019), back-substitutes each
// root in [0, 1] to recover (t, v), and keeps the nearest valid hit with its
// oriented unit normal. It mirrors the CPU golden
// `ray_scene::bilinear_patch::BilinearPatch::intersect` guard for guard, uses
// only the portable core-WGSL subset (abs/min/max/select/dot/cross, one sqrt
// and + - * /), and takes no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::bilinear_patch;
// no third-party engine source or derived code.

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 96-byte std430 stride matching the host `GpuQuery`: the ray origin
// (with t_min in w), the ray direction (with t_max in w) and the four patch
// corners, each padded to a vec4 so the storage array needs no manual vec3
// alignment arithmetic.
struct Query {
    origin: vec4<f32>,
    dir: vec4<f32>,
    p00: vec4<f32>,
    p10: vec4<f32>,
    p11: vec4<f32>,
    p01: vec4<f32>,
}

// One result. 48-byte std430 stride matching the host `GpuResult`: the oriented
// unit normal (xyz, w pad), the hit parameter t, the patch coordinates u and v,
// the hit flag and the front-face flag as 0u/1u, plus three pad words.
struct Result {
    normal: vec4<f32>,
    t: f32,
    u: f32,
    v: f32,
    hit: u32,
    front_face: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Magnitude floor classifying the quadratic as numerically planar in u. The
// reference uses an exact `c == 0` (and `b == 0`) test; a direct f32 `==` is
// forbidden here, so the twin and its host oracle both compare magnitudes
// against this shared floor. Fixtures and the sweep stay well clear of it so
// both sides pick the same branch.
const EPS: f32 = 1.0e-5;

// One back-substituted root. `ok` is 1u when the root yielded a valid hit.
struct RootHit {
    ok: u32,
    t: f32,
    v: f32,
    normal: vec3<f32>,
    front: u32,
}

// Back-substitutes a single u root: intersects the vertical line at that u to
// recover (t, v), rejecting roots outside [0, 1], a ray parallel to the line
// (dot(n0, n0) <= 0), v outside [0, 1], or t outside the ray interval, then
// builds the oriented unit normal. Mirrors the reference `solve_v` plus
// `oriented_normal` guard for guard.
fn solve_root(
    u: f32,
    rd: vec3<f32>,
    q00: vec3<f32>,
    q10: vec3<f32>,
    e00: vec3<f32>,
    e10: vec3<f32>,
    e11: vec3<f32>,
    f: vec3<f32>,
    t_lo: f32,
    t_hi: f32,
) -> RootHit {
    var out: RootHit;
    out.ok = 0u;
    out.t = 0.0;
    out.v = 0.0;
    out.normal = vec3<f32>(0.0, 0.0, 0.0);
    out.front = 0u;

    if (u < 0.0 || u > 1.0) {
        return out;
    }

    // solve_v: the vertical line at u, relative to the ray origin.
    let pa = q00 + (q10 - q00) * u;
    let pb = e00 + (e11 - e00) * u;
    let n0 = cross(rd, pb);
    let det2 = dot(n0, n0);
    if (det2 <= 0.0) {
        return out;
    }
    let m = cross(n0, pa);
    let t = dot(m, pb) / det2;
    let v = dot(m, rd) / det2;
    if (v < 0.0 || v > 1.0) {
        return out;
    }
    if (t < t_lo || t > t_hi) {
        return out;
    }

    // oriented_normal: dP/du x dP/dv, flipped to face the incident ray.
    let dpdu = e10 * (1.0 - v) + f * v;
    let dpdv = e00 * (1.0 - u) + e11 * u;
    let g = cross(dpdu, dpdv);
    let len2 = dot(g, g);
    if (len2 <= 0.0) {
        return out;
    }
    let inv_len = 1.0 / sqrt(len2);
    let outward = g * inv_len;
    let facing = dot(rd, outward) < 0.0;
    var normal = outward;
    if (!facing) {
        normal = -outward;
    }

    out.ok = 1u;
    out.t = t;
    out.v = v;
    out.normal = normal;
    out.front = select(0u, 1u, facing);
    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    let ro = q.origin.xyz;
    let rd = q.dir.xyz;
    let p00 = q.p00.xyz;
    let p10 = q.p10.xyz;
    let p11 = q.p11.xyz;
    let p01 = q.p01.xyz;

    // Ray interval clamp mirroring `Ray::new`: t_min floored at 0, t_max at
    // least t_min. Fixtures pass finite, ordered intervals so this is identity.
    let t_lo = max(q.origin.w, 0.0);
    let t_hi_init = max(q.dir.w, t_lo);

    // Translation-invariant edge vectors.
    let e10 = p10 - p00;
    let e11 = p11 - p10;
    let e00 = p01 - p00;
    let f = p11 - p01;
    let qn = cross(e10, p01 - p11);
    let q00 = p00 - ro;
    let q10 = p10 - ro;

    // Quadratic a*u^2 + b*u + c = 0 (Reshetov's formulation).
    let a = dot(cross(q00, rd), e00);
    let c = dot(qn, rd);
    let b = dot(cross(q10, rd), e11) - a - c;

    var res: Result;
    res.normal = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    res.t = 0.0;
    res.u = 0.0;
    res.v = 0.0;
    res.hit = 0u;
    res.front_face = 0u;
    res.pad0 = 0u;
    res.pad1 = 0u;
    res.pad2 = 0u;

    let det = b * b - 4.0 * a * c;
    if (det < 0.0) {
        results[idx] = res;
        return;
    }
    let sq = sqrt(det);

    var u1 = 0.0;
    var u2 = 0.0;
    var have_roots = true;
    if (abs(c) <= EPS) {
        // Degenerate (planar in u): linear equation b*u + a = 0.
        if (abs(b) <= EPS) {
            have_roots = false;
        } else {
            u1 = -a / b;
            // Sentinel outside [0, 1]; the second root is skipped.
            u2 = -1.0;
        }
    } else {
        // Numerically stable roots: the large root uses a same-sign addition,
        // the small one via the product a / big (reference `sq.copysign(b)`).
        let signed_sq = select(-sq, sq, b >= 0.0);
        let big = (-b - signed_sq) * 0.5;
        u1 = big / c;
        u2 = a / big;
    }

    if (!have_roots) {
        results[idx] = res;
        return;
    }

    var t_hi = t_hi_init;
    // Root 1 (processed first, exactly as the reference iterates [u1, u2]).
    let r1 = solve_root(u1, rd, q00, q10, e00, e10, e11, f, t_lo, t_hi);
    if (r1.ok == 1u) {
        t_hi = r1.t;
        res.hit = 1u;
        res.t = r1.t;
        res.u = u1;
        res.v = r1.v;
        res.normal = vec4<f32>(r1.normal, 0.0);
        res.front_face = r1.front;
    }
    // Root 2 (tested against the possibly-tightened t_hi).
    let r2 = solve_root(u2, rd, q00, q10, e00, e10, e11, f, t_lo, t_hi);
    if (r2.ok == 1u) {
        t_hi = r2.t;
        res.hit = 1u;
        res.t = r2.t;
        res.u = u2;
        res.v = r2.v;
        res.normal = vec4<f32>(r2.normal, 0.0);
        res.front_face = r2.front;
    }

    results[idx] = res;
}
"#;

/// One ray vs bilinear-patch query: the ray and the four patch corners.
///
/// Mirrors a single reference `BilinearPatch::intersect` call. Carrying the
/// corners per query lets one dispatch mix one ray against many distinct
/// patches. `t_min` is floored at zero and `t_max` raised to at least `t_min`
/// on device, mirroring `Ray::new`. Derives only [`PartialEq`] (no `Eq`/`Hash`)
/// because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayBilinearPatchQuery {
    /// Ray origin.
    pub origin: [f32; 3],
    /// Ray direction (never assumed unit; `t` is in `direction` lengths).
    pub direction: [f32; 3],
    /// Lower bound of the ray interval (floored at zero on device).
    pub t_min: f32,
    /// Upper bound of the ray interval (raised to at least `t_min` on device).
    pub t_max: f32,
    /// Corner at `(u, v) = (0, 0)`.
    pub p00: [f32; 3],
    /// Corner at `(u, v) = (1, 0)`.
    pub p10: [f32; 3],
    /// Corner at `(u, v) = (1, 1)`.
    pub p11: [f32; 3],
    /// Corner at `(u, v) = (0, 1)`.
    pub p01: [f32; 3],
}

/// The resolved verdict for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `hit` is `1` when the ray struck the patch inside its interval and `0`
/// otherwise. `t`, `normal`, `u` and `v` are meaningful only when `hit` is `1`
/// (they are left zeroed on a miss); `front_face` is `1` when the ray struck
/// the outward-facing side. Derives only [`PartialEq`] (no `Eq`/`Hash`) because
/// it holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayBilinearPatchResult {
    /// Whether the ray crossed the patch inside its interval (`1` = hit).
    pub hit: u32,
    /// Ray parameter at the intersection; meaningful only when `hit`.
    pub t: f32,
    /// Unit surface normal, oriented against the ray; meaningful only when `hit`.
    pub normal: [f32; 3],
    /// Whether the ray struck the outward-facing side (`1` = front).
    pub front_face: u32,
    /// Patch `u` parameter of the hit, in `[0, 1]`; meaningful only when `hit`.
    pub u: f32,
    /// Patch `v` parameter of the hit, in `[0, 1]`; meaningful only when `hit`.
    pub v: f32,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`RAY_BILINEAR_PATCH_WGSL`]: the query count and three pad words
/// — `16` bytes.
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

/// One query as uploaded. `96`-byte `std430` stride matching `Query` in the
/// shader: the ray origin (with `t_min` in `w`), the ray direction (with
/// `t_max` in `w`) and the four corners, each padded to a `vec4` lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin in `xyz`; `w` carries `t_min`.
    origin: [f32; 4],
    /// Ray direction in `xyz`; `w` carries `t_max`.
    dir: [f32; 4],
    /// Corner `p00` in `xyz`; the `w` lane is unused padding.
    p00: [f32; 4],
    /// Corner `p10` in `xyz`; the `w` lane is unused padding.
    p10: [f32; 4],
    /// Corner `p11` in `xyz`; the `w` lane is unused padding.
    p11: [f32; 4],
    /// Corner `p01` in `xyz`; the `w` lane is unused padding.
    p01: [f32; 4],
}

/// Packs a [`RayBilinearPatchQuery`] into the `std430` upload layout.
fn encode_query(q: &RayBilinearPatchQuery) -> GpuQuery {
    GpuQuery {
        origin: [q.origin[0], q.origin[1], q.origin[2], q.t_min],
        dir: [q.direction[0], q.direction[1], q.direction[2], q.t_max],
        p00: [q.p00[0], q.p00[1], q.p00[2], 0.0],
        p10: [q.p10[0], q.p10[1], q.p10[2], 0.0],
        p11: [q.p11[0], q.p11[1], q.p11[2], 0.0],
        p01: [q.p01[0], q.p01[1], q.p01[2], 0.0],
    }
}

/// One result as read back. `48`-byte `std430` stride matching `Result` in the
/// shader: the oriented unit normal, the hit parameter, the patch coordinates,
/// the two flags and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Oriented unit normal in `xyz`; the `w` lane is unused padding.
    normal: [f32; 4],
    /// Ray parameter at the intersection.
    t: f32,
    /// Patch `u` coordinate.
    u: f32,
    /// Patch `v` coordinate.
    v: f32,
    /// Hit flag (`1` = hit).
    hit: u32,
    /// Front-face flag (`1` = front).
    front_face: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Maps one kernel `Result` lane back to the host [`RayBilinearPatchResult`].
fn decode_result(raw: &GpuResult) -> RayBilinearPatchResult {
    RayBilinearPatchResult {
        hit: raw.hit,
        t: raw.t,
        normal: [raw.normal[0], raw.normal[1], raw.normal[2]],
        front_face: raw.front_face,
        u: raw.u,
        v: raw.v,
    }
}

/// A compiled, reusable ray/bilinear-patch intersection pipeline.
pub struct GpuRayBilinearPatch {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayBilinearPatch {
    /// Compiles the ray/bilinear-patch kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayBilinearPatch {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_bilinear_patch"),
            source: ShaderSource::Wgsl(RAY_BILINEAR_PATCH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_bilinear_patch_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_bilinear_patch_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_bilinear_patch_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayBilinearPatch {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries`, returning one [`RayBilinearPatchResult`]
    /// per query in input order.
    ///
    /// The returned result for query `q` mirrors `BilinearPatch::intersect`
    /// evaluated on `q`'s corners and ray. An empty `queries` slice yields an
    /// empty result — storage buffers cannot be zero-sized, so it is handled by
    /// an early return before any dispatch.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RayBilinearPatchQuery],
    ) -> Vec<RayBilinearPatchResult> {
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
            label: Some("prism_volumetric_ray_bilinear_patch_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_bilinear_patch_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_bilinear_patch_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_bilinear_patch_bind_group"),
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
            label: Some("prism_volumetric_ray_bilinear_patch_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ray_bilinear_patch_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_bilinear_patch_pass"),
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
        debug_assert_eq!(raw.len(), count);

        raw.iter().map(decode_result).collect()
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
