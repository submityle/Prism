//! `wgpu` compute twin of the analytic ray/ellipsoid intersection from the
//! reference ray-scene ellipsoid primitive
//! (`prism_render_architecture::ray_scene::ellipsoid`).
//!
//! The `CPU` golden `Ellipsoid::intersect` owns the closed-form solution of the
//! ray/ellipsoid problem. It scales the ray into the ellipsoid's unit-sphere
//! frame — dividing the center-relative origin and the direction componentwise
//! by the radii — then runs the same numerically stable reduced-quadratic solve
//! as the sphere: it forms the reduced discriminant `half_b^2 - a * c_term` and
//! picks the root branch by the sign of the linear coefficient to dodge the
//! catastrophic cancellation a naive `(-b +- sqrt(disc)) / 2a` suffers on a
//! grazing ray. Because the direction is scaled rather than assumed unit, the
//! `t^2` coefficient carries the full `dot(scaled_dir, scaled_dir)` and the
//! solve stays correct for any non-degenerate ray length. The surface normal is
//! the implicit gradient `(P - center) / radii^2`, which is not unit for an
//! ellipsoid, so it is explicitly normalized and then oriented against the
//! incident ray. [`GpuRayEllipsoid`] is the on-device twin: it runs one thread
//! per query and reproduces the same nearest hit the reference produces, so a
//! passing real-device parity test is direct evidence the ported kernel solves
//! the same quadratic and classifies the same degenerate cases, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! One thread resolves one query. Each [`RayEllipsoidQuery`] carries the ray
//! (origin, direction and the `[t_min, t_max]` interval) and the ellipsoid
//! (center and radii); the kernel reproduces `Ellipsoid::intersect` branch for
//! branch and writes one [`RayEllipsoidResult`] holding the hit flag, the ray
//! parameter `t`, the unit normal oriented against the ray and the front-face
//! flag. The degenerate cases the reference handles explicitly are mirrored: a
//! zero or negative radius on any axis is rejected, a zero-length scaled
//! direction (`a <= 0`) is rejected, a negative discriminant misses, and a hit
//! whose only roots fall outside `[t_min, t_max]` misses. A ray that starts
//! inside the ellipsoid reports `front_face == false` with the normal flipped
//! inward, exactly as the reference does.
//!
//! # What stays on the host
//!
//! Nothing of the per-query math stays on the host: the whole solve is a fixed,
//! bounded sequence of products, adds, comparisons and two `sqrt` calls that
//! runs entirely on device. The host only flattens the query batch into a
//! `std430` storage buffer and short-circuits an empty batch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! The ray parameter `t` and the normal are *continuous* quantities threaded
//! through `sqrt`, divisions and products, so the `CPU` and `GPU` are not
//! bit-exact: a device may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore compares the continuous fields with an
//! absolute-or-relative tolerance (`abs <= 1e-4 || rel <= 1e-3`, relative floor
//! `1e-6`) while pinning the integer `hit` and `front_face` flags exactly. The
//! genuine degeneracies — a near-zero discriminant graze, a root pressed
//! against `t_min` or `t_max`, a ray origin on the surface, and a near-zero
//! radius — are kept off their thresholds by the fixtures and the randomized
//! sweep so the two sides agree on the discrete classification.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `sqrt`, `select`, `+ - * /` and ordered comparisons — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round`, no `%` and
//! no `u64` / `u16` / `i64` / `f64`. The `copysign` of the reference is spelled
//! as a sign-selecting `select`. No optional device feature is required, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::ellipsoid`；无第三方引擎源码或衍生代码。
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

/// Dot product of two 3-vectors, matching the golden `dot` helper.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::ellipsoid`'s `Ellipsoid::intersect`, reproduced without
/// importing the golden so the twin stays self-contained.
///
/// Scales the ray into the ellipsoid's unit-sphere frame, solves the stable
/// reduced quadratic, selects the nearest root in `[t_min, t_max]`, and reports
/// the ray parameter, the normalized gradient normal oriented against the ray
/// and the front-face flag. Returns `None` on a degenerate radius, a
/// zero-length scaled direction, a negative discriminant or no root in range.
/// The radii are folded to their magnitude first, mirroring `Ellipsoid::new`.
#[must_use]
pub fn intersect_ellipsoid(
    origin: [f32; 3],
    direction: [f32; 3],
    t_min: f32,
    t_max: f32,
    center: [f32; 3],
    radii: [f32; 3],
) -> Option<(f32, [f32; 3], bool)> {
    let r = [radii[0].abs(), radii[1].abs(), radii[2].abs()];
    if r[0] <= 0.0 || r[1] <= 0.0 || r[2] <= 0.0 {
        return None;
    }
    let so = [
        (origin[0] - center[0]) / r[0],
        (origin[1] - center[1]) / r[1],
        (origin[2] - center[2]) / r[2],
    ];
    let sd = [
        direction[0] / r[0],
        direction[1] / r[1],
        direction[2] / r[2],
    ];
    let a = dot3(sd, sd);
    if a <= 0.0 {
        return None;
    }
    let half_b = dot3(so, sd);
    let c_term = dot3(so, so) - 1.0;
    let disc = half_b * half_b - a * c_term;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    // `copysign(sqrt_disc, half_b)`: magnitude of `sqrt_disc` with the sign of
    // `half_b`, keeping the numerator off the cancellation a direct sum suffers.
    let signed = if half_b < 0.0 { -sqrt_disc } else { sqrt_disc };
    let k = -(half_b + signed);
    let (t_near, t_far) = if k.abs() > 0.0 {
        let r0 = k / a;
        let r1 = c_term / k;
        if r0 <= r1 {
            (r0, r1)
        } else {
            (r1, r0)
        }
    } else {
        let rr = -half_b / a;
        (rr, rr)
    };

    let t = if t_near >= t_min && t_near <= t_max {
        t_near
    } else if t_far >= t_min && t_far <= t_max {
        t_far
    } else {
        return None;
    };

    let point = [
        origin[0] + t * direction[0],
        origin[1] + t * direction[1],
        origin[2] + t * direction[2],
    ];
    let grad = [
        (point[0] - center[0]) / (r[0] * r[0]),
        (point[1] - center[1]) / (r[1] * r[1]),
        (point[2] - center[2]) / (r[2] * r[2]),
    ];
    let inv_len = 1.0 / dot3(grad, grad).sqrt();
    let outward = [grad[0] * inv_len, grad[1] * inv_len, grad[2] * inv_len];
    let front_face = dot3(direction, outward) < 0.0;
    let normal = if front_face {
        outward
    } else {
        [-outward[0], -outward[1], -outward[2]]
    };
    Some((t, normal, front_face))
}

/// The portable core-`WGSL` ray/ellipsoid kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// golden `Ellipsoid::intersect`; see the module documentation.
const RAY_ELLIPSOID_WGSL: &str = r#"
// Ray/ellipsoid intersection twin: one thread solves one query, mirroring the
// golden Ellipsoid::intersect with the stable reduced quadratic (scale into the
// unit-sphere frame, pick the root branch by the sign of the linear
// coefficient), the nearest in-range root, and the normalized gradient normal
// oriented against the ray. Transcendental-free: only products, adds,
// comparisons, select and sqrt.
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::ellipsoid；无第三方引擎源码或衍生代码。

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
    // Ray direction (not necessarily normalized).
    dx: f32,
    dy: f32,
    dz: f32,
    // Valid ray parameter interval.
    t_lo: f32,
    t_hi: f32,
    // Ellipsoid center.
    cx: f32,
    cy: f32,
    cz: f32,
    // Ellipsoid semi-axis radii.
    rx: f32,
    ry: f32,
    rz: f32,
}

struct HitResult {
    hit: u32,
    t: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    front_face: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<HitResult>;

fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: HitResult;
    out.hit = 0u;
    out.t = 0.0;
    out.nx = 0.0;
    out.ny = 0.0;
    out.nz = 0.0;
    out.front_face = 0u;

    let center = vec3<f32>(q.cx, q.cy, q.cz);
    let direction = vec3<f32>(q.dx, q.dy, q.dz);
    let origin = vec3<f32>(q.ox, q.oy, q.oz);
    // Radii folded to magnitude, mirroring the golden Ellipsoid::new.
    let r = vec3<f32>(abs(q.rx), abs(q.ry), abs(q.rz));

    if (r.x <= 0.0 || r.y <= 0.0 || r.z <= 0.0) {
        results[idx] = out;
        return;
    }

    let so = vec3<f32>(
        (origin.x - center.x) / r.x,
        (origin.y - center.y) / r.y,
        (origin.z - center.z) / r.z,
    );
    let sd = vec3<f32>(direction.x / r.x, direction.y / r.y, direction.z / r.z);

    let a = dot3(sd, sd);
    if (a <= 0.0) {
        results[idx] = out;
        return;
    }
    let half_b = dot3(so, sd);
    let c_term = dot3(so, so) - 1.0;
    let disc = half_b * half_b - a * c_term;
    if (disc < 0.0) {
        results[idx] = out;
        return;
    }
    let sqrt_disc = sqrt(disc);
    // copysign(sqrt_disc, half_b): magnitude with the sign of half_b.
    let signed = select(sqrt_disc, -sqrt_disc, half_b < 0.0);
    let k = -(half_b + signed);

    var t_near = 0.0;
    var t_far = 0.0;
    if (abs(k) > 0.0) {
        let r0 = k / a;
        let r1 = c_term / k;
        t_near = min(r0, r1);
        t_far = max(r0, r1);
    } else {
        let rr = -half_b / a;
        t_near = rr;
        t_far = rr;
    }

    let t_lo = q.t_lo;
    let t_hi = q.t_hi;
    var t = 0.0;
    var found = false;
    if (t_near >= t_lo && t_near <= t_hi) {
        t = t_near;
        found = true;
    } else if (t_far >= t_lo && t_far <= t_hi) {
        t = t_far;
        found = true;
    }
    if (!found) {
        results[idx] = out;
        return;
    }

    let point = vec3<f32>(
        origin.x + t * direction.x,
        origin.y + t * direction.y,
        origin.z + t * direction.z,
    );
    let grad = vec3<f32>(
        (point.x - center.x) / (r.x * r.x),
        (point.y - center.y) / (r.y * r.y),
        (point.z - center.z) / (r.z * r.z),
    );
    let inv_len = 1.0 / sqrt(dot3(grad, grad));
    let outward = vec3<f32>(grad.x * inv_len, grad.y * inv_len, grad.z * inv_len);
    let is_front = dot3(direction, outward) < 0.0;
    let normal = select(
        vec3<f32>(-outward.x, -outward.y, -outward.z),
        outward,
        is_front,
    );

    out.hit = 1u;
    out.t = t;
    out.nx = normal.x;
    out.ny = normal.y;
    out.nz = normal.z;
    out.front_face = select(0u, 1u, is_front);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in [`RAY_ELLIPSOID_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the ray and the ellipsoid, matching
/// the `WGSL` `Query` struct (fourteen `f32`).
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
    /// Lower bound of the valid `t` interval.
    t_lo: f32,
    /// Upper bound of the valid `t` interval.
    t_hi: f32,
    /// Ellipsoid center `x`.
    cx: f32,
    /// Ellipsoid center `y`.
    cy: f32,
    /// Ellipsoid center `z`.
    cz: f32,
    /// Ellipsoid semi-axis radius `x`.
    rx: f32,
    /// Ellipsoid semi-axis radius `y`.
    ry: f32,
    /// Ellipsoid semi-axis radius `z`.
    rz: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `HitResult`
/// struct: the hit flag, the ray parameter, the normal and the front-face flag.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` when the ray hit the ellipsoid in range, `0` otherwise.
    hit: u32,
    /// Ray parameter at the intersection (zero on a miss).
    t: f32,
    /// Unit normal `x`, oriented against the ray (zero on a miss).
    nx: f32,
    /// Unit normal `y`, oriented against the ray (zero on a miss).
    ny: f32,
    /// Unit normal `z`, oriented against the ray (zero on a miss).
    nz: f32,
    /// `1` on a front-face hit, `0` on a back-face hit or a miss.
    front_face: u32,
}

/// One query: the ray and the ellipsoid to test against it.
///
/// The `o*` triple is the ray origin, the `d*` triple the direction (not
/// necessarily normalized), `t_lo`/`t_hi` the valid parameter interval, the
/// `c*` triple the ellipsoid center and the `r*` triple its semi-axis radii.
/// The host enqueues one query per evaluation, and an empty batch is
/// short-circuited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayEllipsoidQuery {
    /// Ray origin `x`.
    pub ox: f32,
    /// Ray origin `y`.
    pub oy: f32,
    /// Ray origin `z`.
    pub oz: f32,
    /// Ray direction `x`.
    pub dx: f32,
    /// Ray direction `y`.
    pub dy: f32,
    /// Ray direction `z`.
    pub dz: f32,
    /// Lower bound of the valid `t` interval.
    pub t_lo: f32,
    /// Upper bound of the valid `t` interval.
    pub t_hi: f32,
    /// Ellipsoid center `x`.
    pub cx: f32,
    /// Ellipsoid center `y`.
    pub cy: f32,
    /// Ellipsoid center `z`.
    pub cz: f32,
    /// Ellipsoid semi-axis radius `x`.
    pub rx: f32,
    /// Ellipsoid semi-axis radius `y`.
    pub ry: f32,
    /// Ellipsoid semi-axis radius `z`.
    pub rz: f32,
}

impl RayEllipsoidQuery {
    /// Builds a query from the ray and the ellipsoid parameters.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query packs the ray and the ellipsoid as flat scalars for a std430 slot"
    )]
    pub const fn new(
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dy: f32,
        dz: f32,
        t_lo: f32,
        t_hi: f32,
        cx: f32,
        cy: f32,
        cz: f32,
        rx: f32,
        ry: f32,
        rz: f32,
    ) -> RayEllipsoidQuery {
        RayEllipsoidQuery {
            ox,
            oy,
            oz,
            dx,
            dy,
            dz,
            t_lo,
            t_hi,
            cx,
            cy,
            cz,
            rx,
            ry,
            rz,
        }
    }
}

/// One resolved query: the hit flag, the ray parameter, the normal and the
/// front-face flag. All fields are zero on a miss.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayEllipsoidResult {
    /// `true` when the ray hit the ellipsoid inside `[t_lo, t_hi]`.
    pub hit: bool,
    /// Ray parameter at the intersection (zero on a miss).
    pub t: f32,
    /// Unit surface normal oriented against the ray (zero on a miss).
    pub normal: [f32; 3],
    /// `true` on a front-face hit, `false` on a back-face hit or a miss.
    pub front_face: bool,
}

/// Encodes one [`RayEllipsoidQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &RayEllipsoidQuery) -> GpuQuery {
    GpuQuery {
        ox: q.ox,
        oy: q.oy,
        oz: q.oz,
        dx: q.dx,
        dy: q.dy,
        dz: q.dz,
        t_lo: q.t_lo,
        t_hi: q.t_hi,
        cx: q.cx,
        cy: q.cy,
        cz: q.cz,
        rx: q.rx,
        ry: q.ry,
        rz: q.rz,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`RayEllipsoidResult`],
/// unpacking the integer flags into the reference's boolean shape.
fn decode_result(raw: &GpuResult) -> RayEllipsoidResult {
    RayEllipsoidResult {
        hit: raw.hit != 0,
        t: raw.t,
        normal: [raw.nx, raw.ny, raw.nz],
        front_face: raw.front_face != 0,
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

/// A compiled, reusable ray/ellipsoid compute pipeline, twinning the golden
/// `ray_scene::ellipsoid`'s `Ellipsoid::intersect`.
pub struct GpuRayEllipsoid {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayEllipsoid {
    /// Compiles the ray/ellipsoid kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayEllipsoid {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_ellipsoid"),
            source: ShaderSource::Wgsl(RAY_ELLIPSOID_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_ellipsoid_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_ellipsoid_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_ellipsoid_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayEllipsoid {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one [`RayEllipsoidResult`]
    /// per input, in order.
    ///
    /// Each continuous field equals the reference within floating-point
    /// tolerance and the `hit`/`front_face` flags match exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RayEllipsoidQuery],
    ) -> Vec<RayEllipsoidResult> {
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
            label: Some("prism_volumetric_ray_ellipsoid_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_ellipsoid_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_ellipsoid_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_ellipsoid_bind_group"),
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
            label: Some("prism_volumetric_ray_ellipsoid_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ray_ellipsoid_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_ellipsoid_pass"),
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
