//! `wgpu` compute twin of the finite-volume moving-sphere continuous-collision
//! contract
//! ([`sphere_sweep`](prism_render_architecture::particle::sphere_sweep),
//! particle design §10, §13).
//!
//! The `CPU` golden
//! [`sphere_sweep`](prism_render_architecture::particle::sphere_sweep) owns the
//! purely geometric swept-sphere question: given a sphere of a fixed radius
//! moving at a constant velocity across the unit timestep `t ∈ [0, 1]`, at what
//! fraction of the step does it *first* touch another moving sphere, an infinite
//! plane, or a stationary point, and with what outward contact normal? It
//! answers that with three functions —
//! [`sweep_sphere_vs_sphere`](prism_render_architecture::particle::sphere_sweep::sweep_sphere_vs_sphere),
//! [`sweep_sphere_vs_plane`](prism_render_architecture::particle::sphere_sweep::sweep_sphere_vs_plane)
//! and
//! [`sweep_sphere_vs_point`](prism_render_architecture::particle::sphere_sweep::sweep_sphere_vs_point)
//! — each returning a
//! [`SweepHit`](prism_render_architecture::particle::sphere_sweep::SweepHit).
//! [`GpuSphereSweep`] is the on-device twin: one thread per query reproduces the
//! same answer the batch reference produces, so a passing real-device parity
//! test is direct evidence the ported kernel solves the same geometry and
//! classifies the same degenerate cases the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced: the
//! `Miss` / `Hit { toi, normal, point }` discriminant and, on a hit, the
//! time-of-impact fraction, the outward unit contact normal and the world
//! contact point. The reference's three routines are selected by a per-query op
//! code (`0` = versus sphere, `1` = versus plane, `2` = versus point) and the
//! degenerate cases are mirrored branch for branch: an initial overlap (start
//! distance below the summed radius) reports a `toi` of `0` with a separation
//! normal; a coincident overlap falls back to the stable default normal
//! `(1, 0, 0)`; a separated pair with no relative motion misses; a negative
//! discriminant misses; and a smaller root outside `[0, 1]` misses. The
//! versus-point routine is the versus-sphere solver with a zero-radius,
//! stationary target, exactly as the reference forwards it.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `sqrt`, `+ - * /` and unsigned integer compares — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan` or optional device feature, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. The only non-rational operation is the
//! `sqrt` of the clamped discriminant, matching the reference's `f32::sqrt`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields while pinning
//! the `Miss` / `Hit` classification exactly, tight enough to catch a genuinely
//! wrong port (a dropped branch, a swapped coefficient, a wrong normal) yet
//! loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`sphere_sweep`](prism_render_architecture::particle::sphere_sweep); no
//! third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::sphere_sweep::SweepHit;
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

/// The portable core-`WGSL` swept-sphere kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`sphere_sweep`](prism_render_architecture::particle::sphere_sweep) branch
/// for branch; see the module documentation for the algorithm.
const SPHERE_SWEEP_WGSL: &str = r#"
// Finite-volume moving-sphere continuous-collision twin: one thread per query
// reproduces the Miss / Hit { toi, normal, point } answer for a sphere swept
// over t in [0, 1] against another moving sphere (op 0), an infinite plane
// (op 1), or a stationary point (op 2, the zero-radius stationary sphere the
// reference forwards to). It mirrors the CPU golden particle::sphere_sweep,
// uses only the portable core-WGSL subset (min/max/abs/sqrt and + - * / plus
// unsigned integer compares) and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::sphere_sweep; no
// third-party engine source or derived code.

// Epsilon guarding divisions, classifying the overlap, discriminant and
// near-parallel cases, and comparing against zero without ever writing an exact
// == / != on an f32. It matches the reference EPS of 1e-6.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Swept sphere A: center in the first three lanes, radius in the fourth.
    center_a: vec3<f32>,
    radius_a: f32,
    // Swept sphere A velocity; the fourth lane carries sphere B's radius.
    vel_a: vec3<f32>,
    radius_b: f32,
    // Geometry of the target B: sphere center (op 0), plane unit normal (op 1)
    // or stationary point (op 2); the fourth lane carries the plane offset d.
    geom_b: vec3<f32>,
    plane_d: f32,
    // Target sphere velocity (zero for the plane and point ops); a pad follows.
    vel_b: vec3<f32>,
    pad0: f32,
    // Op classification code (0 = vs sphere, 1 = vs plane, 2 = vs point) plus
    // three pad words to fill the slot.
    op: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

struct Result {
    // Time-of-impact (valid only when hit is set), the hit flag (0 = Miss,
    // 1 = Hit), and two pad lanes to fill the slot.
    toi: f32,
    hit: u32,
    pad0: f32,
    pad1: f32,
    // Outward unit contact normal; a pad lane follows.
    normal: vec3<f32>,
    pad2: f32,
    // World contact point at the moment of first touch; a pad lane follows.
    point: vec3<f32>,
    pad3: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The decoded answer for one query before packing into the Result slot.
struct Hit {
    hit: u32,
    toi: f32,
    normal: vec3<f32>,
    point: vec3<f32>,
}

// Unit vector along v, or fallback when v is shorter than CMP_EPS, mirroring the
// reference v_normalize_or so a near-zero vector never yields a NaN.
fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = sqrt(dot(v, v));
    if (len < CMP_EPS) {
        return fallback;
    }
    return v * (1.0 / len);
}

// Swept-sphere versus swept-sphere, reduced to a moving point against a
// stationary sphere of radius ra + rb; mirrors sweep_sphere_vs_sphere.
fn sweep_vs_sphere(
    ca: vec3<f32>,
    ra: f32,
    va: vec3<f32>,
    cb: vec3<f32>,
    rb: f32,
    vb: vec3<f32>,
) -> Hit {
    var res: Hit;
    res.hit = 0u;
    res.toi = 0.0;
    res.normal = vec3<f32>(0.0, 0.0, 0.0);
    res.point = vec3<f32>(0.0, 0.0, 0.0);

    let d0 = ca - cb;
    let dv = va - vb;
    let r = ra + rb;
    let r_sq = r * r;
    let d0_sq = dot(d0, d0);
    let c = d0_sq - r_sq;

    // Already overlapping (or exactly touching): report an immediate contact.
    if (c <= CMP_EPS) {
        let normal = normalize_or(d0, vec3<f32>(1.0, 0.0, 0.0));
        res.hit = 1u;
        res.toi = 0.0;
        res.normal = normal;
        res.point = ca - normal * ra;
        return res;
    }

    let a = dot(dv, dv);
    // No relative motion while separated: the pair can never meet.
    if (a < CMP_EPS) {
        return res;
    }

    let b = 2.0 * dot(d0, dv);
    let disc = b * b - 4.0 * a * c;
    if (disc < -CMP_EPS) {
        return res;
    }

    // Smaller root is the entry (first-touch) time; a > 0 here.
    let sqrt_disc = sqrt(max(disc, 0.0));
    let toi = (-b - sqrt_disc) / (2.0 * a);
    if (toi < 0.0 || toi > 1.0) {
        return res;
    }

    let d_t = d0 + dv * toi;
    let normal = normalize_or(d_t, vec3<f32>(1.0, 0.0, 0.0));
    let ca_t = ca + va * toi;
    res.hit = 1u;
    res.toi = toi;
    res.normal = normal;
    res.point = ca_t - normal * ra;
    return res;
}

// Swept-sphere versus infinite plane dot(n, x) = plane_d; mirrors
// sweep_sphere_vs_plane.
fn sweep_vs_plane(center: vec3<f32>, radius: f32, vel: vec3<f32>, n: vec3<f32>, plane_d: f32) -> Hit {
    var res: Hit;
    res.hit = 0u;
    res.toi = 0.0;
    res.normal = vec3<f32>(0.0, 0.0, 0.0);
    res.point = vec3<f32>(0.0, 0.0, 0.0);

    let s0 = dot(n, center) - plane_d;
    let sn = dot(n, vel);

    // Orient the contact normal toward the side the centre currently lies on.
    var side = -1.0;
    if (s0 >= 0.0) {
        side = 1.0;
    }
    let normal = n * side;

    // Already touching or penetrating: immediate contact.
    if (abs(s0) <= radius + CMP_EPS) {
        res.hit = 1u;
        res.toi = 0.0;
        res.normal = normal;
        res.point = center - normal * radius;
        return res;
    }

    // Near-parallel sweep with the centre farther than the radius never meets.
    if (abs(sn) < CMP_EPS) {
        return res;
    }

    // First touch happens when the signed distance reaches +/- radius. needle
    // stands in for the reference target (a WGSL-safe identifier).
    let needle = side * radius;
    let toi = (needle - s0) / sn;
    if (toi < 0.0 || toi > 1.0) {
        return res;
    }

    let center_t = center + vel * toi;
    res.hit = 1u;
    res.toi = toi;
    res.normal = normal;
    res.point = center_t - normal * radius;
    return res;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var res: Hit;
    if (q.op == 1u) {
        res = sweep_vs_plane(q.center_a, q.radius_a, q.vel_a, q.geom_b, q.plane_d);
    } else {
        // op 0 (vs sphere) and op 2 (vs point, with radius_b = 0 and vel_b = 0
        // packed by the host) share the moving-point-versus-sphere solver.
        res = sweep_vs_sphere(q.center_a, q.radius_a, q.vel_a, q.geom_b, q.radius_b, q.vel_b);
    }

    var out: Result;
    out.toi = res.toi;
    out.hit = res.hit;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.normal = res.normal;
    out.pad2 = 0.0;
    out.point = res.point;
    out.pad3 = 0.0;
    results[idx] = out;
}
"#;

/// Selects which of the three reference routines a [`SphereSweepQuery`]
/// evaluates.
///
/// The discriminant order matches the `u32` op codes the kernel branches on, so
/// the host encode is a plain cast.
///
/// Provenance: twinned from this repository's
/// [`sphere_sweep`](prism_render_architecture::particle::sphere_sweep); no
/// third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SphereSweepOp {
    /// Swept-sphere versus swept-sphere
    /// ([`sweep_sphere_vs_sphere`](prism_render_architecture::particle::sphere_sweep::sweep_sphere_vs_sphere)).
    Sphere,
    /// Swept-sphere versus infinite plane
    /// ([`sweep_sphere_vs_plane`](prism_render_architecture::particle::sphere_sweep::sweep_sphere_vs_plane)).
    Plane,
    /// Swept-sphere versus stationary point
    /// ([`sweep_sphere_vs_point`](prism_render_architecture::particle::sphere_sweep::sweep_sphere_vs_point)).
    Point,
}

impl SphereSweepOp {
    /// Returns the `u32` op code the kernel branches on for this routine.
    #[must_use]
    const fn code(self) -> u32 {
        match self {
            SphereSweepOp::Sphere => 0,
            SphereSweepOp::Plane => 1,
            SphereSweepOp::Point => 2,
        }
    }
}

/// One swept-sphere query: the routine selected by `op` with its flattened
/// geometry.
///
/// The swept sphere A is always described by `center_a`, `radius_a` and
/// `vel_a`. The target `B` reuses the shared slots per op: `geom_b` is the
/// sphere centre (op [`SphereSweepOp::Sphere`]), the plane unit normal
/// (op [`SphereSweepOp::Plane`]) or the stationary point
/// (op [`SphereSweepOp::Point`]); `radius_b` and `vel_b` apply only to the
/// sphere op; and `plane_d` is the plane offset read only by the plane op. A
/// field not read by the chosen op is ignored. Build queries through
/// [`SphereSweepQuery::sphere`], [`SphereSweepQuery::plane`] and
/// [`SphereSweepQuery::point`] so the unused slots are zeroed consistently.
///
/// Provenance: twinned from this repository's
/// [`sphere_sweep`](prism_render_architecture::particle::sphere_sweep); no
/// third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereSweepQuery {
    /// The routine this query evaluates.
    pub op: SphereSweepOp,
    /// Swept sphere A centre at the start of the step.
    pub center_a: [f32; 3],
    /// Swept sphere A radius.
    pub radius_a: f32,
    /// Swept sphere A constant velocity over the unit step.
    pub vel_a: [f32; 3],
    /// Target geometry: sphere centre, plane unit normal, or stationary point.
    pub geom_b: [f32; 3],
    /// Target sphere radius (read only by [`SphereSweepOp::Sphere`]).
    pub radius_b: f32,
    /// Target sphere velocity (read only by [`SphereSweepOp::Sphere`]).
    pub vel_b: [f32; 3],
    /// Plane offset `d` (read only by [`SphereSweepOp::Plane`]).
    pub plane_d: f32,
}

impl SphereSweepQuery {
    /// Builds a swept-sphere-versus-swept-sphere query, matching
    /// [`sweep_sphere_vs_sphere`](prism_render_architecture::particle::sphere_sweep::sweep_sphere_vs_sphere).
    #[must_use]
    pub const fn sphere(
        center_a: [f32; 3],
        radius_a: f32,
        vel_a: [f32; 3],
        center_b: [f32; 3],
        radius_b: f32,
        vel_b: [f32; 3],
    ) -> SphereSweepQuery {
        SphereSweepQuery {
            op: SphereSweepOp::Sphere,
            center_a,
            radius_a,
            vel_a,
            geom_b: center_b,
            radius_b,
            vel_b,
            plane_d: 0.0,
        }
    }

    /// Builds a swept-sphere-versus-plane query, matching
    /// [`sweep_sphere_vs_plane`](prism_render_architecture::particle::sphere_sweep::sweep_sphere_vs_plane).
    /// The `plane_normal_unit` must be a unit vector, exactly as the reference
    /// requires.
    #[must_use]
    pub const fn plane(
        center: [f32; 3],
        radius: f32,
        vel: [f32; 3],
        plane_normal_unit: [f32; 3],
        plane_d: f32,
    ) -> SphereSweepQuery {
        SphereSweepQuery {
            op: SphereSweepOp::Plane,
            center_a: center,
            radius_a: radius,
            vel_a: vel,
            geom_b: plane_normal_unit,
            radius_b: 0.0,
            vel_b: [0.0, 0.0, 0.0],
            plane_d,
        }
    }

    /// Builds a swept-sphere-versus-point query, matching
    /// [`sweep_sphere_vs_point`](prism_render_architecture::particle::sphere_sweep::sweep_sphere_vs_point)
    /// (the zero-radius, stationary target the reference forwards to).
    #[must_use]
    pub const fn point(
        center: [f32; 3],
        radius: f32,
        vel: [f32; 3],
        point: [f32; 3],
    ) -> SphereSweepQuery {
        SphereSweepQuery {
            op: SphereSweepOp::Point,
            center_a: center,
            radius_a: radius,
            vel_a: vel,
            geom_b: point,
            radius_b: 0.0,
            vel_b: [0.0, 0.0, 0.0],
            plane_d: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one packed query: five `vec4` slots holding
/// `(center_a.xyz, radius_a)`, `(vel_a.xyz, radius_b)`, `(geom_b.xyz, plane_d)`,
/// `(vel_b.xyz, pad)` and `(op, pad, pad, pad)` — `80` bytes, each `vec3` on its
/// `16`-byte-aligned slot exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Swept sphere A centre.
    center_a: [f32; 3],
    /// Swept sphere A radius, packed into the centre slot's fourth lane.
    radius_a: f32,
    /// Swept sphere A velocity.
    vel_a: [f32; 3],
    /// Target sphere radius, packed into the velocity slot's fourth lane.
    radius_b: f32,
    /// Target geometry (sphere centre, plane normal, or point).
    geom_b: [f32; 3],
    /// Plane offset, packed into the geometry slot's fourth lane.
    plane_d: f32,
    /// Target sphere velocity.
    vel_b: [f32; 3],
    /// Padding lane after the target velocity.
    pad0: f32,
    /// Op classification code (`0` = vs sphere, `1` = vs plane, `2` = vs point).
    op: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Padding word.
    pad3: u32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &SphereSweepQuery) -> GpuQuery {
        GpuQuery {
            center_a: query.center_a,
            radius_a: query.radius_a,
            vel_a: query.vel_a,
            radius_b: query.radius_b,
            geom_b: query.geom_b,
            plane_d: query.plane_d,
            vel_b: query.vel_b,
            pad0: 0.0,
            op: query.op.code(),
            pad1: 0,
            pad2: 0,
            pad3: 0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: two scalars `(toi, hit)` with two
/// pad lanes, then two `vec4` slots for the outward normal and the contact
/// point — `48` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Time-of-impact fraction (valid only when `hit` is `1`).
    toi: f32,
    /// Hit flag (`0` = `Miss`, `1` = `Hit`).
    hit: u32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// Outward unit contact normal.
    normal: [f32; 3],
    /// Padding lane after the normal.
    pad2: f32,
    /// World contact point at the moment of first touch.
    point: [f32; 3],
    /// Padding lane after the point.
    pad3: f32,
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

/// A compiled, reusable swept-sphere compute pipeline.
pub struct GpuSphereSweep {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSphereSweep {
    /// Compiles the swept-sphere kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSphereSweep {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sphere_sweep"),
            source: ShaderSource::Wgsl(SPHERE_SWEEP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sphere_sweep_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sphere_sweep_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sphere_sweep_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSphereSweep {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one
    /// [`SweepHit`](prism_render_architecture::particle::sphere_sweep::SweepHit)
    /// per input, in order.
    ///
    /// Each result equals the reference answer (`sweep_sphere_vs_sphere`,
    /// `sweep_sphere_vs_plane` or `sweep_sphere_vs_point`) to within the
    /// tolerance documented on this module, with the `Miss` / `Hit`
    /// classification exact. An empty input returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[SphereSweepQuery]) -> Vec<SweepHit> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sphere_sweep_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sphere_sweep_output"),
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
            label: Some("prism_volumetric_sphere_sweep_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sphere_sweep_bind_group"),
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
            label: Some("prism_volumetric_sphere_sweep_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sphere_sweep_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sphere_sweep_pass"),
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

        raw.iter().map(decode_result).collect()
    }
}

/// Decodes one packed [`GpuResult`] into the public reference
/// [`SweepHit`](prism_render_architecture::particle::sphere_sweep::SweepHit),
/// unpacking the hit flag into the `Miss` / `Hit` shape.
fn decode_result(raw: &GpuResult) -> SweepHit {
    if raw.hit == 0 {
        SweepHit::Miss
    } else {
        SweepHit::Hit {
            toi: raw.toi,
            normal: raw.normal,
            point: raw.point,
        }
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
