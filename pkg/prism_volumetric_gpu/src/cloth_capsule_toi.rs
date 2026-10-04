//! `wgpu` compute twin of the swept-point-versus-capsule continuous-collision
//! time-of-impact (TOI) closed form from the `CPU` golden
//! `prism_physics_core::soft::collision::ccd`.
//!
//! The soft-body substep solver only projects particles out of the analytic
//! body colliders at their end-of-step position, so a fast, thin garment can
//! tunnel straight through a thin capsule in one substep. The reference closes
//! that gap by sweeping the segment `prev -> curr` of a particle against a
//! capsule (segment `p0`..`p1` inflated by `radius`) and solving for the
//! earliest time of impact in `0..=1`. A capsule is the union of an infinite
//! cylinder about its axis (restricted to the segment slab) with a sphere at
//! each end cap, so the TOI is the earliest of a cylinder-slab entry and the
//! two end-cap sphere entries. This module ports that stateless, loop-free
//! closed form onto the device: one thread resolves one `(particle, capsule)`
//! sweep.
//!
//! [`GpuClothCapsuleToi`] is the on-device twin, so a passing real-device
//! parity test is direct evidence the ported kernel takes the same
//! quadratic / interval-intersection branch the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `capsule_toi(prev, curr, p0, p1,
//! radius)` and its private helpers branch for branch:
//!
//! * `sphere_toi` — substitutes the point path into `|p(t) - center|^2 ==
//!   radius^2`, giving a quadratic whose earliest entry root (via
//!   `first_entry_time`) is the crossing; a point starting on or inside reports
//!   `t == 0`.
//! * `cylinder_slab_toi` — intersects the radial sub-`radius` interval with the
//!   axial slab interval `0..=len` and with `0..=1`, returning the lower bound
//!   of the resulting interval.
//! * `earliest` — merges two optional times, preferring the smaller present
//!   value.
//! * `capsule_toi` — a non-positive radius is inert; a collapsed capsule
//!   (`len_sq <= EPS_LEN_SQ`) degenerates to a single sphere at `p0`; otherwise
//!   the earliest of the cylinder slab and the two end-cap spheres wins.
//!
//! There is no loop: each thread performs a fixed, bounded sequence, so the
//! kernel provably terminates. The outer `resolve_ccd` batch sweep is *not*
//! twinned; this module is the per-pair scalar core.
//!
//! # Correctness model
//!
//! Every path threads through multiplies, adds and guarded divisions and a
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a `GPU` may contract a multiply-add). The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! the continuous `t` and compares the discrete `hit` flag exactly. The `valid`
//! flag is always `1` (the closed form never fails to classify).
//!
//! # Degenerate inputs
//!
//! A non-positive radius misses; a collapsed capsule falls back to a sphere; a
//! quadratic with a negative discriminant, a (near) zero leading coefficient,
//! or an axial projection outside the slab misses on the corresponding branch.
//! The infinite radial/axial intervals the reference expresses with
//! `f32::INFINITY` are represented on the device with large finite sentinels
//! that can never win the `max(.., 0.0)` / `min(.., 1.0)` clamp, so they never
//! leak into the returned time. All divisions are guarded by the same ordered
//! `EPS_COEF` / `EPS_LEN_SQ` comparisons the reference uses, so no path divides
//! by zero. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `select`, `abs`,
//! `min`, `max`, `sqrt`, `+ - * /` and ordered `f32` comparisons — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, no `round`, no `f32` modulo, no `u64` /
//! `i64` / `f64` and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::ccd`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` capsule-TOI kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `capsule_toi` and its helpers branch for branch; see the module
/// documentation for the algorithm.
const CLOTH_CAPSULE_TOI_WGSL: &str = r#"
// Cloth capsule-TOI twin: one thread per query reproduces capsule_toi(prev,
// curr, p0, p1, radius). It mirrors the CPU golden branch for branch, uses only
// the portable core-WGSL subset (select, abs, min, max, sqrt, ordered f32
// comparisons), takes no optional feature, and has no loop, so the kernel
// provably terminates.
//
// Provenance: 孪生自本仓 prism_physics_core::soft::collision::ccd；无第三方引擎源码
// 或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Segment start of the swept point.
    prev_x: f32,
    prev_y: f32,
    prev_z: f32,
    // Segment end of the swept point.
    curr_x: f32,
    curr_y: f32,
    curr_z: f32,
    // First capsule axis endpoint.
    p0_x: f32,
    p0_y: f32,
    p0_z: f32,
    // Second capsule axis endpoint.
    p1_x: f32,
    p1_y: f32,
    p1_z: f32,
    // Capsule inflation radius.
    radius: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // 1 when the swept segment reaches the capsule within 0..=1.
    hit: u32,
    // Earliest time of impact in 0..=1 when hit, otherwise 0.
    t: f32,
    // 1 always: the closed form never fails to classify.
    valid: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Numerical floor for treating a scalar coefficient as zero, matching the
// reference EPS_COEF.
const EPS_COEF: f32 = 1.0e-12;
// Squared-length floor below which the capsule axis is collapsed, matching the
// reference EPS_LEN_SQ.
const EPS_LEN_SQ: f32 = 1.0e-12;
// Large finite sentinels standing in for the reference -inf / +inf interval
// bounds; they can never win the max(.., 0.0) / min(.., 1.0) clamp, so they
// never leak into the returned time.
const SENTINEL_LO: f32 = -1.0e30;
const SENTINEL_HI: f32 = 1.0e30;

// A time-of-impact answer: hit = 1 with a time, or hit = 0 (the reference None).
struct Toi {
    hit: u32,
    t: f32,
}

fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// The earliest root in 0..=1 of a*t^2 + b*t + c <= 0 for a non-negative leading
// coefficient, i.e. the first time the value becomes non-positive, or a miss.
// Mirrors the golden first_entry_time branch for branch.
fn first_entry_time(a: f32, b: f32, c: f32) -> Toi {
    var out: Toi;
    out.hit = 0u;
    out.t = 0.0;
    if (c <= 0.0) {
        out.hit = 1u;
        out.t = 0.0;
        return out;
    }
    if (a <= EPS_COEF) {
        // Linear: b*t + c <= 0. With c > 0 this needs b < 0.
        if (b >= -EPS_COEF) {
            return out;
        }
        let t = -c / b;
        if (t <= 1.0) {
            out.hit = 1u;
            out.t = max(t, 0.0);
        }
        return out;
    }
    let disc = b * b - 4.0 * a * c;
    if (disc < 0.0) {
        return out;
    }
    let root = sqrt(disc);
    // With c > 0 and a > 0 the earlier root is the entry into the region.
    let t = ((-b) - root) / (2.0 * a);
    if (t >= 0.0 && t <= 1.0) {
        out.hit = 1u;
        out.t = t;
    }
    return out;
}

// Earliest entry time of the point sweep into the sphere, or a miss.
fn sphere_toi(prev: vec3<f32>, curr: vec3<f32>, center: vec3<f32>, radius: f32) -> Toi {
    var out: Toi;
    out.hit = 0u;
    out.t = 0.0;
    if (radius <= 0.0) {
        return out;
    }
    let m = curr - prev;
    let e = prev - center;
    let a = dot3(m, m);
    let b = 2.0 * dot3(e, m);
    let c = dot3(e, e) - radius * radius;
    return first_entry_time(a, b, c);
}

// Earliest cylindrical-side entry time whose contact projects onto the segment
// slab 0..=len, or a miss. Mirrors the golden cylinder_slab_toi.
fn cylinder_slab_toi(
    prev: vec3<f32>,
    curr: vec3<f32>,
    p0: vec3<f32>,
    axis: vec3<f32>,
    radius: f32,
) -> Toi {
    var out: Toi;
    out.hit = 0u;
    out.t = 0.0;
    let len = sqrt(dot3(axis, axis));
    if (len <= EPS_COEF) {
        return out;
    }
    let u = axis * (1.0 / len);
    let e0 = prev - p0;
    let m = curr - prev;
    let mu = dot3(m, u);
    let e0u = dot3(e0, u);

    // Radial interval [rad_lo, rad_hi] where perpendicular distance <= radius.
    let a = dot3(m, m) - mu * mu;
    let b = 2.0 * (dot3(e0, m) - e0u * mu);
    let c = dot3(e0, e0) - e0u * e0u - radius * radius;
    var rad_lo: f32;
    var rad_hi: f32;
    if (a > EPS_COEF) {
        let disc = b * b - 4.0 * a * c;
        if (disc < 0.0) {
            return out;
        }
        let root = sqrt(disc);
        rad_lo = ((-b) - root) / (2.0 * a);
        rad_hi = ((-b) + root) / (2.0 * a);
    } else if (c <= 0.0) {
        // Parallel to the axis and already within radius: radially inside for
        // the entire segment.
        rad_lo = SENTINEL_LO;
        rad_hi = SENTINEL_HI;
    } else {
        return out;
    }

    // Axial interval [ax_lo, ax_hi] where the projection lies in [0, len].
    var ax_lo: f32;
    var ax_hi: f32;
    if (abs(mu) > EPS_COEF) {
        let t_at_zero = -e0u / mu;
        let t_at_len = (len - e0u) / mu;
        ax_lo = min(t_at_zero, t_at_len);
        ax_hi = max(t_at_zero, t_at_len);
    } else if (e0u >= 0.0 && e0u <= len) {
        ax_lo = SENTINEL_LO;
        ax_hi = SENTINEL_HI;
    } else {
        return out;
    }

    let lo = max(max(rad_lo, ax_lo), 0.0);
    let hi = min(min(rad_hi, ax_hi), 1.0);
    if (lo <= hi) {
        out.hit = 1u;
        out.t = lo;
    }
    return out;
}

// Whichever of the two times is earlier, preferring a present value over a miss.
fn earliest(lhs: Toi, rhs: Toi) -> Toi {
    var out: Toi;
    if (lhs.hit == 1u && rhs.hit == 1u) {
        out.hit = 1u;
        out.t = min(lhs.t, rhs.t);
        return out;
    }
    if (lhs.hit == 1u) {
        return lhs;
    }
    return rhs;
}

// Earliest time the swept point reaches the capsule, or a miss. Mirrors the
// golden capsule_toi: the union of the cylinder slab with the two end caps.
fn capsule_toi(
    prev: vec3<f32>,
    curr: vec3<f32>,
    p0: vec3<f32>,
    p1: vec3<f32>,
    radius: f32,
) -> Toi {
    var out: Toi;
    out.hit = 0u;
    out.t = 0.0;
    if (radius <= 0.0) {
        return out;
    }
    let axis = p1 - p0;
    let len_sq = dot3(axis, axis);
    if (len_sq <= EPS_LEN_SQ) {
        // Degenerate capsule behaves like a sphere at p0.
        return sphere_toi(prev, curr, p0, radius);
    }
    var best = cylinder_slab_toi(prev, curr, p0, axis, radius);
    best = earliest(best, sphere_toi(prev, curr, p0, radius));
    best = earliest(best, sphere_toi(prev, curr, p1, radius));
    return best;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let prev = vec3<f32>(q.prev_x, q.prev_y, q.prev_z);
    let curr = vec3<f32>(q.curr_x, q.curr_y, q.curr_z);
    let p0 = vec3<f32>(q.p0_x, q.p0_y, q.p0_z);
    let p1 = vec3<f32>(q.p1_x, q.p1_y, q.p1_z);
    let toi = capsule_toi(prev, curr, p0, p1, q.radius);

    var r: Result;
    r.hit = toi.hit;
    r.t = toi.t;
    r.valid = 1u;
    r.pad0 = 0u;
    results[idx] = r;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`CLOTH_CAPSULE_TOI_WGSL`].
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
/// Thirteen `f32` plus three pad words fill exactly four `16`-byte slots, so the
/// host stride matches the shader stride for batches of two or more elements.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    prev_x: f32,
    prev_y: f32,
    prev_z: f32,
    curr_x: f32,
    curr_y: f32,
    curr_z: f32,
    p0_x: f32,
    p0_y: f32,
    p0_z: f32,
    p1_x: f32,
    p1_y: f32,
    p1_z: f32,
    radius: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. `hit`, `t` and `valid` plus a trailing pad word fill exactly one
/// `16`-byte slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    hit: u32,
    t: f32,
    valid: u32,
    pad0: u32,
}

/// One query for the cloth capsule-TOI twin: the swept segment `prev -> curr`
/// against the capsule (segment `p0`..`p1` inflated by `radius`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothCapsuleToiQuery {
    /// `x` component of the segment start.
    pub prev_x: f32,
    /// `y` component of the segment start.
    pub prev_y: f32,
    /// `z` component of the segment start.
    pub prev_z: f32,
    /// `x` component of the segment end.
    pub curr_x: f32,
    /// `y` component of the segment end.
    pub curr_y: f32,
    /// `z` component of the segment end.
    pub curr_z: f32,
    /// `x` component of the first capsule axis endpoint.
    pub p0_x: f32,
    /// `y` component of the first capsule axis endpoint.
    pub p0_y: f32,
    /// `z` component of the first capsule axis endpoint.
    pub p0_z: f32,
    /// `x` component of the second capsule axis endpoint.
    pub p1_x: f32,
    /// `y` component of the second capsule axis endpoint.
    pub p1_y: f32,
    /// `z` component of the second capsule axis endpoint.
    pub p1_z: f32,
    /// Capsule inflation radius.
    pub radius: f32,
}

impl ClothCapsuleToiQuery {
    /// Builds a query from the three points and the radius.
    #[must_use]
    pub fn new(prev: [f32; 3], curr: [f32; 3], p0: [f32; 3], p1: [f32; 3], radius: f32) -> Self {
        ClothCapsuleToiQuery {
            prev_x: prev[0],
            prev_y: prev[1],
            prev_z: prev[2],
            curr_x: curr[0],
            curr_y: curr[1],
            curr_z: curr[2],
            p0_x: p0[0],
            p0_y: p0[1],
            p0_z: p0[2],
            p1_x: p1[0],
            p1_y: p1[1],
            p1_z: p1[2],
            radius,
        }
    }
}

/// One resolved answer, mirroring the reference `capsule_toi` return.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothCapsuleToiResult {
    /// `1` when the swept segment reaches the capsule within `0..=1`.
    pub hit: u32,
    /// Earliest time of impact in `0..=1` when `hit`, otherwise `0`.
    pub t: f32,
    /// `1` always: the closed form never fails to classify.
    pub valid: u32,
}

/// Encodes one [`ClothCapsuleToiQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ClothCapsuleToiQuery) -> GpuQuery {
    GpuQuery {
        prev_x: q.prev_x,
        prev_y: q.prev_y,
        prev_z: q.prev_z,
        curr_x: q.curr_x,
        curr_y: q.curr_y,
        curr_z: q.curr_z,
        p0_x: q.p0_x,
        p0_y: q.p0_y,
        p0_z: q.p0_z,
        p1_x: q.p1_x,
        p1_y: q.p1_y,
        p1_z: q.p1_z,
        radius: q.radius,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ClothCapsuleToiResult`].
fn decode_result(raw: &GpuResult) -> ClothCapsuleToiResult {
    ClothCapsuleToiResult {
        hit: raw.hit,
        t: raw.t,
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

/// A compiled, reusable capsule-TOI compute pipeline, twinning the `CPU` golden
/// `capsule_toi`.
pub struct GpuClothCapsuleToi {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothCapsuleToi {
    /// Compiles the capsule-TOI kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothCapsuleToi {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_capsule_toi"),
            source: ShaderSource::Wgsl(CLOTH_CAPSULE_TOI_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_capsule_toi_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_capsule_toi_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_capsule_toi_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothCapsuleToi {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`ClothCapsuleToiResult`]
    /// per input, in order.
    ///
    /// The continuous `t` matches the reference to within the tolerance
    /// documented on this module; the `hit` and `valid` flags match exactly. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothCapsuleToiQuery],
    ) -> Vec<ClothCapsuleToiResult> {
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
            label: Some("prism_volumetric_cloth_capsule_toi_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_capsule_toi_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_capsule_toi_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_capsule_toi_bind_group"),
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
            label: Some("prism_volumetric_cloth_capsule_toi_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_capsule_toi_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_capsule_toi_pass"),
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
