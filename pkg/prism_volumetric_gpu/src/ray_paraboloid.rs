//! `wgpu` compute twin of the analytic *paraboloid* (parabolic dish / bowl)
//! ray intersection of the `CPU` golden path —
//! `Paraboloid::intersect` in
//! `prism_render_architecture::ray_scene::paraboloid`.
//!
//! A paraboloid of revolution is the natural proxy for a reflector dish, a
//! rounded goblet wall, or any surface whose radius grows as the square root of
//! the axial distance. The surface is swept from an `apex` vertex (radius `0`)
//! to a `top` rim of radius `radius`, with the perpendicular distance to the
//! axis growing as `rho(z) = radius * sqrt(z / h)` for axial distance
//! `z` in `[0, h]` (`h = |top - apex|`); equivalently the surface satisfies the
//! implicit quadric `k * rho^2 = z` with `k = h / radius^2`. The ray is
//! substituted into that quadric, yielding a quadratic in `t` whose roots are
//! clipped to the finite `z` in `[0, h]` band; the nearest valid root is the
//! hit and the analytic gradient `2k * rho_vec - n_hat` (normalized) is the
//! surface normal. Every step is add/sub/mul/div/`sqrt` and ordered
//! comparisons, so it is reproducible on the `GPU` free of any transcendental
//! call. [`GpuRayParaboloid`] is the on-device twin: each thread intersects one
//! ray with one paraboloid and writes the hit flag, parameter, oriented normal
//! and front-face flag.
//!
//! # What is twinned
//!
//! Each thread reads one [`RayParaboloidQuery`] — the ray `origin` and
//! `direction` with its `[t_min, t_max]` interval, and the paraboloid `apex`,
//! `top` and `radius` — and writes one [`RayParaboloidResult`] holding the
//! `hit` flag (`1`/`0`), the ray parameter `t`, the unit `normal` oriented
//! against the incident ray, and the `front_face` flag (`1` when the ray struck
//! the outward-facing convex side). A miss reports `hit = 0` with the remaining
//! fields zeroed.
//!
//! The kernel reproduces the reference closed form exactly: the degenerate
//! guards (zero-length axis, zero radius, zero-length direction), the quadric
//! coefficients `A = k*(dd - zd^2)`, `B = 2k*(ad - za*zd) - zd`,
//! `C = k*(aa - za^2) - za`, the quadratic (or linear, when the ray runs
//! parallel to the axis) root solve, the clip to the `z` in `[0, h]` band, the
//! nearest-root selection, the analytic gradient normal, and the front-face
//! orientation.
//!
//! # What stays on the host
//!
//! The paraboloid `BVH` build and its ordered slab traversal, the acceleration
//! structure that narrows a scene to one candidate primitive, and the stable
//! `primitive` id bookkeeping all stay on the host; the device sees only the
//! stateless, fixed-width ray/paraboloid test, one query at a time, so a
//! storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! The intersection threads through products, sums, a discriminant `sqrt` and
//! several branch decisions, so the `CPU` and `GPU` are not bit-exact: a fused
//! multiply-add or a differently ordered sum may land a few units in the last
//! place from the scalar reference. The parity test asserts the discrete `hit`
//! and `front_face` flags exactly and each continuous component (`t`, `normal`)
//! within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. Fixtures and the random
//! sweep reject-sample away from every branch cliff — the discriminant near
//! zero (a tangent ray), a root near the `[t_min, t_max]` ends, the axial
//! coordinate near the `0`/`h` band edges, the leading coefficient near zero (a
//! ray near-parallel to the axis), and the incidence dot near zero (a grazing
//! hit) — so the `CPU` and `GPU` never pick different sides of a comparison.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`,
//! `min`, `max`, `dot`, `+ - * /` and unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round`/`ceil`/`%`, and no `f64`/`u64`/`u16`/`i64`/`i16`. The zero tests use
//! ordered comparisons and an `abs(x) > eps` guard, never a bare float
//! equality. It runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::paraboloid`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` paraboloid-intersection kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `Paraboloid::intersect`, one query per thread.
const RAY_PARABOLOID_WGSL: &str = r#"
// Analytic paraboloid ray intersection twinned from the CPU golden path. Each
// thread intersects one ray with one paraboloid and writes the hit flag, ray
// parameter, oriented unit normal and front-face flag; the BVH build and the
// ordered slab traversal stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::paraboloid；无第三方
// 引擎源码或衍生代码。

// Guard separating a "nonzero" leading coefficient from the parallel-to-axis
// degenerate branch; the parity harness keeps real queries well clear of it.
const A_EPS: f32 = 1.0e-7;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Ray origin.
    ox: f32,
    oy: f32,
    oz: f32,
    // Ray direction (not required to be unit length).
    dx: f32,
    dy: f32,
    dz: f32,
    // Ray parameter interval.
    t_min: f32,
    t_max: f32,
    // Paraboloid apex (axis start, radius 0).
    apx: f32,
    apy: f32,
    apz: f32,
    // Paraboloid rim center (axis end).
    topx: f32,
    topy: f32,
    topz: f32,
    // Rim radius (assumed non-negative by the caller).
    radius: f32,
    pad0: f32,
}

struct Outputs {
    // Hit flag: 1u on a hit, 0u on a miss.
    hit: u32,
    // Front-face flag: 1u when the ray struck the outward convex side.
    front_face: u32,
    // Ray parameter at the nearest valid hit; 0.0 on a miss.
    t: f32,
    // Unit surface normal oriented against the ray; zero on a miss.
    nx: f32,
    ny: f32,
    nz: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Outputs>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Outputs;
    out.hit = 0u;
    out.front_face = 0u;
    out.t = 0.0;
    out.nx = 0.0;
    out.ny = 0.0;
    out.nz = 0.0;
    out.pad0 = 0.0;
    out.pad1 = 0.0;

    // Axis vector apex -> rim and its squared length.
    let w = vec3<f32>(q.topx - q.apx, q.topy - q.apy, q.topz - q.apz);
    let h2 = dot(w, w);
    let r2 = q.radius * q.radius;
    let dir = vec3<f32>(q.dx, q.dy, q.dz);
    let dd = dot(dir, dir);
    // A zero-length axis, zero radius, or zero-length direction never hits.
    if (h2 <= 0.0 || r2 <= 0.0 || dd <= 0.0) {
        results[idx] = out;
        return;
    }

    let h = sqrt(h2);
    let inv_h = 1.0 / h;
    // Unit axis from apex toward the rim.
    let n = w * inv_h;
    // Quadric coefficient: z = k * rho^2 with k = h / radius^2.
    let k = h / r2;

    // Ray origin relative to the apex and the axial / dot projections.
    let a = vec3<f32>(q.ox - q.apx, q.oy - q.apy, q.oz - q.apz);
    let za = dot(a, n);
    let zd = dot(dir, n);
    let ad = dot(a, dir);
    let aa = dot(a, a);

    // A*t^2 + B*t + C = 0: the perpendicular pieces are dd - zd^2 (perp
    // speed^2), ad - za*zd (perp dot), aa - za^2 (perp offset^2).
    let coeff_a = k * (dd - zd * zd);
    let coeff_b = 2.0 * k * (ad - za * zd) - zd;
    let coeff_c = k * (aa - za * za) - za;

    // Candidate roots: quadratic, or linear when the ray runs parallel to the
    // axis (A == 0). The band clip and nearest-root pick happen below.
    var root0: f32 = 0.0;
    var root1: f32 = 0.0;
    var root_count: u32 = 0u;
    if (abs(coeff_a) > A_EPS) {
        let disc = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c;
        if (disc < 0.0) {
            results[idx] = out;
            return;
        }
        let sqrt_disc = sqrt(disc);
        let inv_2a = 1.0 / (2.0 * coeff_a);
        let ra = (-coeff_b - sqrt_disc) * inv_2a;
        let rb = (-coeff_b + sqrt_disc) * inv_2a;
        root0 = min(ra, rb);
        root1 = max(ra, rb);
        root_count = 2u;
    } else if (abs(coeff_b) > A_EPS) {
        root0 = -coeff_c / coeff_b;
        root1 = root0;
        root_count = 1u;
    } else {
        results[idx] = out;
        return;
    }

    let t_min = q.t_min;
    let t_max = q.t_max;

    var best_t: f32 = 0.0;
    var best_out = vec3<f32>(0.0, 0.0, 0.0);
    var found: bool = false;

    // Walk the (ascending) candidate roots; the first valid one wins because a
    // later, larger root is skipped by the `found && t >= best_t` guard.
    for (var i: u32 = 0u; i < root_count; i = i + 1u) {
        var t: f32 = root0;
        if (i == 1u) {
            t = root1;
        }
        if (!(t >= t_min && t <= t_max)) {
            continue;
        }
        if (found && !(t < best_t)) {
            continue;
        }
        // Axial distance from the apex; valid on the dish in [0, h].
        let z = za + t * zd;
        if (z < 0.0 || z > h) {
            continue;
        }
        // Point relative to the apex and its perpendicular (radial) part.
        let p = a + dir * t;
        let perp = p - n * z;
        // Outward gradient of k * rho^2 - z: 2k * rho_vec - n_hat.
        let grad = perp * (2.0 * k) - n;
        let nn = dot(grad, grad);
        if (nn <= 0.0) {
            continue;
        }
        best_t = t;
        best_out = grad * (1.0 / sqrt(nn));
        found = true;
    }

    if (!found) {
        results[idx] = out;
        return;
    }

    // Orient the normal against the incident ray.
    let incidence = dot(dir, best_out);
    var normal = best_out;
    if (!(incidence < 0.0)) {
        normal = best_out * -1.0;
    }
    out.hit = 1u;
    if (incidence < 0.0) {
        out.front_face = 1u;
    }
    out.t = best_t;
    out.nx = normal.x;
    out.ny = normal.y;
    out.nz = normal.z;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RAY_PARABOLOID_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the ray `origin`/`direction` with its `[t_min, t_max]` interval and the
/// paraboloid `apex`/`top`/`radius` — `16` `f32` words at a `64`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin `x`.
    ox: f32,
    /// Ray origin `y`.
    oy: f32,
    /// Ray origin `z`.
    oz: f32,
    /// Ray direction `x`.
    dx: f32,
    /// Ray direction `y`.
    dy: f32,
    /// Ray direction `z`.
    dz: f32,
    /// Lower bound of the ray parameter interval.
    t_min: f32,
    /// Upper bound of the ray parameter interval.
    t_max: f32,
    /// Paraboloid apex `x`.
    apx: f32,
    /// Paraboloid apex `y`.
    apy: f32,
    /// Paraboloid apex `z`.
    apz: f32,
    /// Paraboloid rim-center `x`.
    topx: f32,
    /// Paraboloid rim-center `y`.
    topy: f32,
    /// Paraboloid rim-center `z`.
    topz: f32,
    /// Rim radius.
    radius: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Outputs`
/// struct: the hit and front-face flags, the ray parameter and the oriented
/// normal, plus two pad words to a `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Hit flag (`1` = hit, `0` = miss).
    hit: u32,
    /// Front-face flag (`1` = outward convex side struck).
    front_face: u32,
    /// Ray parameter at the nearest valid hit.
    t: f32,
    /// Oriented unit normal `x`.
    nx: f32,
    /// Oriented unit normal `y`.
    ny: f32,
    /// Oriented unit normal `z`.
    nz: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One ray/paraboloid intersection query: the ray and the paraboloid.
///
/// `origin` and `direction` define the ray (the direction need not be unit
/// length), restricted to the `[t_min, t_max]` parameter interval. The
/// paraboloid is the surface of revolution swept from `apex` (radius `0`) to
/// the `top` rim of radius `radius`. The reference folds a negative radius to
/// its magnitude on construction, so callers should pass a non-negative
/// `radius`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayParaboloidQuery {
    /// Ray origin `[x, y, z]`.
    pub origin: [f32; 3],
    /// Ray direction `[x, y, z]` (not required to be unit length).
    pub direction: [f32; 3],
    /// Lower bound of the ray parameter interval.
    pub t_min: f32,
    /// Upper bound of the ray parameter interval.
    pub t_max: f32,
    /// Paraboloid apex (axis start, radius `0`).
    pub apex: [f32; 3],
    /// Paraboloid rim center (axis end).
    pub top: [f32; 3],
    /// Rim radius (non-negative).
    pub radius: f32,
}

impl RayParaboloidQuery {
    /// Builds a query from the ray and the paraboloid parameters.
    #[must_use]
    pub const fn new(
        origin: [f32; 3],
        direction: [f32; 3],
        t_min: f32,
        t_max: f32,
        apex: [f32; 3],
        top: [f32; 3],
        radius: f32,
    ) -> RayParaboloidQuery {
        RayParaboloidQuery {
            origin,
            direction,
            t_min,
            t_max,
            apex,
            top,
            radius,
        }
    }
}

/// One resolved ray/paraboloid intersection.
///
/// `hit` is the exact flag (`1` when the ray strikes the dish inside its
/// `[t_min, t_max]` interval, `0` otherwise). `t`, `normal` and `front_face`
/// are meaningful only when `hit` is `1` (otherwise `t` and `normal` are zero
/// and `front_face` is `0`). `normal` is the unit surface normal oriented
/// against the incident ray, and `front_face` is `1` when the ray struck the
/// outward-facing convex side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayParaboloidResult {
    /// Hit flag (`1` = hit, `0` = miss).
    pub hit: u32,
    /// Ray parameter at the nearest valid hit; meaningful only when `hit`.
    pub t: f32,
    /// Unit surface normal oriented against the ray; meaningful only when `hit`.
    pub normal: [f32; 3],
    /// Front-face flag (`1` = outward convex side); meaningful only when `hit`.
    pub front_face: u32,
}

/// Encodes one [`RayParaboloidQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &RayParaboloidQuery) -> GpuQuery {
    GpuQuery {
        ox: q.origin[0],
        oy: q.origin[1],
        oz: q.origin[2],
        dx: q.direction[0],
        dy: q.direction[1],
        dz: q.direction[2],
        t_min: q.t_min,
        t_max: q.t_max,
        apx: q.apex[0],
        apy: q.apex[1],
        apz: q.apex[2],
        topx: q.top[0],
        topy: q.top[1],
        topz: q.top[2],
        radius: q.radius,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`RayParaboloidResult`].
fn decode_result(raw: &GpuResult) -> RayParaboloidResult {
    RayParaboloidResult {
        hit: raw.hit,
        t: raw.t,
        normal: [raw.nx, raw.ny, raw.nz],
        front_face: raw.front_face,
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

/// A compiled, reusable paraboloid-intersection compute pipeline, twinning the
/// `CPU` golden `Paraboloid::intersect` of
/// `prism_render_architecture::ray_scene::paraboloid`.
pub struct GpuRayParaboloid {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayParaboloid {
    /// Compiles the paraboloid-intersection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayParaboloid {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_paraboloid"),
            source: ShaderSource::Wgsl(RAY_PARABOLOID_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_paraboloid_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_paraboloid_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_paraboloid_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayParaboloid {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`RayParaboloidResult`]
    /// per input, in order.
    ///
    /// The reported hit matches the reference: the discrete flags exactly and
    /// the continuous fields to within the tolerance documented on this module.
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RayParaboloidQuery],
    ) -> Vec<RayParaboloidResult> {
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
            label: Some("prism_volumetric_ray_paraboloid_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_paraboloid_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_paraboloid_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_paraboloid_bind_group"),
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
            label: Some("prism_volumetric_ray_paraboloid_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ray_paraboloid_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_paraboloid_pass"),
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
