//! `wgpu` compute twin of the analytic ray/round-cone intersection.
//!
//! A **round cone** is the convex hull of two spheres: a `radius_a`-sphere at
//! endpoint `a` and a `radius_b`-sphere at endpoint `b`. Its surface is a
//! lateral cone band tangent to both spheres, capped by a spherical cap at each
//! end. When `radius_a` equals `radius_b` it degenerates to a capsule; when the
//! axis is shorter than the radius difference, one sphere swallows the other and
//! the solid reduces to the larger end sphere.
//!
//! The `CPU` golden `prism_render_architecture::ray_scene::round_cone` owns the
//! closed form. It intersects three analytic pieces and keeps the nearest valid
//! root inside the ray interval: the external tangent cone band (clipped to the
//! axial band between the two tangent circles), and the two endpoint spheres
//! (each clipped to the cap region its tangent circle leaves exposed). Every
//! `t²` coefficient carries `dd = dot(direction, direction)`, so the test is
//! correct for non-unit ray directions, and every step is add/sub/mul/div/`sqrt`
//! and comparisons.
//!
//! [`GpuRayRoundCone`] is the on-device twin: it runs one thread per query and
//! reproduces the golden answer branch for branch. Degenerate inputs are
//! handled explicitly, mirroring the reference: a zero-length direction never
//! hits; both radii non-positive never hits; a zero-length axis (or one shorter
//! than `|radius_a − radius_b|`) reduces to a single sphere at the larger
//! endpoint. The reference folds both radii to their magnitude, so the twin
//! folds `abs(radius_a)` and `abs(radius_b)` as well.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `sqrt`, `select`, `+ - * /` and `bitcast` — with no
//! transcendental call and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. The only non-rational operations are the
//! `sqrt` of each discriminant and the axis length, matching the reference.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and a few
//! `sqrt`s, so `CPU` and `GPU` evaluate the same closed form. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on the `f32` fields while pinning the hit flag and the
//! `front_face` classification exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::round_cone`；
//! 无第三方引擎源码或衍生代码。

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

/// The portable core-`WGSL` ray/round-cone kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `prism_render_architecture::ray_scene::round_cone` branch for
/// branch; see the module documentation for the algorithm.
const RAY_ROUND_CONE_WGSL: &str = r#"
// Analytic ray/round-cone twin: one thread per query intersects the tangent
// cone band plus the two clipped end spheres and keeps the nearest valid root
// inside the ray interval. It mirrors the CPU golden
// prism_render_architecture::ray_scene::round_cone, uses only the portable
// core-WGSL subset (min/max/clamp/abs/sqrt/select and + - * /) and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::round_cone;
// 无第三方引擎源码或衍生代码。

// Guard magnitude for a quadratic leading coefficient treated as zero; below it
// the piece contributes no quadratic root, matching the reference `!= 0.0`
// branch away from the razor-thin critical band the fixtures avoid.
const COEFF_EPS: f32 = 1.0e-20;
// Guard magnitude for the signed radius difference `rr`; at or below it the
// lateral band is treated as a straight cylinder (the capsule construction).
const RR_EPS: f32 = 1.0e-9;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Ray origin in the first three lanes; the ray's t_min in the fourth.
    origin: vec3<f32>,
    t_min: f32,
    // Ray direction (not required to be unit length); the ray's t_max follows.
    direction: vec3<f32>,
    t_max: f32,
    // Near endpoint / a-side sphere center; radius_a in the fourth lane.
    a: vec3<f32>,
    radius_a: f32,
    // Far endpoint / b-side sphere center; radius_b in the fourth lane.
    b: vec3<f32>,
    radius_b: f32,
}

struct Result {
    // Chosen hit parameter (valid only when hit is set).
    t: f32,
    // 1 when a nearest in-interval hit exists, else 0.
    hit: u32,
    // 1 when the ray struck the outward-facing side, else 0.
    front_face: u32,
    pad0: f32,
    // Unit surface normal, oriented against the incident ray; a pad lane follows.
    normal: vec3<f32>,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Running best (t, outward-normal) across every analytic piece.
struct Best {
    t: f32,
    outward: vec3<f32>,
}

// Two roots of a quadratic (or none).
struct Roots {
    ok: bool,
    t0: f32,
    t1: f32,
}

// Keeps the nearest in-interval root, mirroring the reference `consider`
// closure: reject when t is outside [t_min, t_max] or not strictly nearer.
fn consider(best: ptr<function, Best>, t: f32, outward: vec3<f32>, t_min: f32, t_max: f32) {
    if (!(t >= t_min && t <= t_max) || t >= (*best).t) {
        return;
    }
    (*best).t = t;
    (*best).outward = outward;
}

// Solves the sphere quadratic |rel + t*d|^2 = r2, returning both roots or none,
// mirroring the reference `sphere_roots`.
fn sphere_roots(rel: vec3<f32>, direction: vec3<f32>, dd: f32, r2: f32) -> Roots {
    let b = 2.0 * dot(rel, direction);
    let c = dot(rel, rel) - r2;
    let disc = b * b - 4.0 * dd * c;
    if (disc < 0.0) {
        return Roots(false, 0.0, 0.0);
    }
    let sqrt_disc = sqrt(disc);
    let inv_2a = 1.0 / (2.0 * dd);
    return Roots(true, (-b - sqrt_disc) * inv_2a, (-b + sqrt_disc) * inv_2a);
}

// Emits both roots of a sphere of radius `radius` at `center`, with outward
// normal normalize(p - center), mirroring the reference `emit_sphere`.
fn emit_sphere(
    best: ptr<function, Best>,
    origin: vec3<f32>,
    direction: vec3<f32>,
    dd: f32,
    center: vec3<f32>,
    radius: f32,
    t_min: f32,
    t_max: f32,
) {
    let rel = origin - center;
    let r = sphere_roots(rel, direction, dd, radius * radius);
    if (r.ok) {
        var roots = array<f32, 2>(r.t0, r.t1);
        for (var i = 0; i < 2; i = i + 1) {
            let t = roots[i];
            let p = origin + direction * t;
            let radial = p - center;
            let len2 = dot(radial, radial);
            if (len2 > 0.0) {
                consider(best, t, radial * (1.0 / sqrt(len2)), t_min, t_max);
            }
        }
    }
}

// Equal-radius lateral band: an infinite cylinder of radius `r` clipped to the
// axial range [0, l], plus hemispheres clipped at the band ends. Mirrors the
// reference `intersect_cylinder`.
fn intersect_cylinder(
    best: ptr<function, Best>,
    origin: vec3<f32>,
    direction: vec3<f32>,
    dd: f32,
    a: vec3<f32>,
    b: vec3<f32>,
    n: vec3<f32>,
    l: f32,
    r: f32,
    t_min: f32,
    t_max: f32,
) {
    let r2 = r * r;
    let oa = origin - a;
    let za = dot(oa, n);
    let zd = dot(direction, n);
    let ad = dot(oa, direction);
    let aa = dot(oa, oa);

    let coeff_a = dd - zd * zd;
    let coeff_b = 2.0 * (ad - za * zd);
    let coeff_c = aa - za * za - r2;
    if (abs(coeff_a) > COEFF_EPS) {
        let disc = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c;
        if (disc >= 0.0) {
            let sqrt_disc = sqrt(disc);
            let inv_2a = 1.0 / (2.0 * coeff_a);
            var roots = array<f32, 2>((-coeff_b - sqrt_disc) * inv_2a, (-coeff_b + sqrt_disc) * inv_2a);
            for (var i = 0; i < 2; i = i + 1) {
                let t = roots[i];
                let z = za + t * zd;
                if (z >= 0.0 && z <= l) {
                    let axis_point = a + n * z;
                    let p = origin + direction * t;
                    let radial = p - axis_point;
                    let len2 = dot(radial, radial);
                    if (len2 > 0.0) {
                        consider(best, t, radial * (1.0 / sqrt(len2)), t_min, t_max);
                    }
                }
            }
        }
    }

    // Hemisphere at `a` (band start, axial <= 0).
    let ra = sphere_roots(oa, direction, dd, r2);
    if (ra.ok) {
        var roots = array<f32, 2>(ra.t0, ra.t1);
        for (var i = 0; i < 2; i = i + 1) {
            let t = roots[i];
            let p = origin + direction * t;
            if (dot(p - a, n) <= 0.0) {
                let radial = p - a;
                let len2 = dot(radial, radial);
                if (len2 > 0.0) {
                    consider(best, t, radial * (1.0 / sqrt(len2)), t_min, t_max);
                }
            }
        }
    }

    // Hemisphere at `b` (band end, axial >= l). Note the clip uses `p - a`.
    let ob = origin - b;
    let rb = sphere_roots(ob, direction, dd, r2);
    if (rb.ok) {
        var roots = array<f32, 2>(rb.t0, rb.t1);
        for (var i = 0; i < 2; i = i + 1) {
            let t = roots[i];
            let p = origin + direction * t;
            if (dot(p - a, n) >= l) {
                let radial = p - b;
                let len2 = dot(radial, radial);
                if (len2 > 0.0) {
                    consider(best, t, radial * (1.0 / sqrt(len2)), t_min, t_max);
                }
            }
        }
    }
}

// Unequal-radius lateral band: the external tangent cone clipped to the axial
// band between the tangent circles, plus the end spheres clipped to the caps
// those circles leave exposed. Mirrors the reference `intersect_tapered`.
fn intersect_tapered(
    best: ptr<function, Best>,
    origin: vec3<f32>,
    direction: vec3<f32>,
    dd: f32,
    a: vec3<f32>,
    b: vec3<f32>,
    n: vec3<f32>,
    l: f32,
    rr: f32,
    radius_a: f32,
    radius_b: f32,
    t_min: f32,
    t_max: f32,
) {
    let sin_a = rr / l;
    let cos_a = sqrt(1.0 - sin_a * sin_a);
    let band_lo = radius_a * sin_a;
    let band_hi = l + radius_b * sin_a;

    let oa = origin - a;
    let za = dot(oa, n);
    let zd = dot(direction, n);
    let ad = dot(oa, direction);
    let aa = dot(oa, oa);
    let k = cos_a * cos_a;
    let rhs0 = radius_a - sin_a * za;
    let rhs1 = -sin_a * zd;
    let coeff_a = k * (dd - zd * zd) - rhs1 * rhs1;
    let coeff_b = k * (ad - za * zd) - rhs0 * rhs1;
    let coeff_c = k * (aa - za * za) - rhs0 * rhs0;
    if (abs(coeff_a) > COEFF_EPS) {
        // `coeff_b` is the half-b form, so the discriminant drops the factor 4.
        let disc = coeff_b * coeff_b - coeff_a * coeff_c;
        if (disc >= 0.0) {
            let sqrt_disc = sqrt(disc);
            let inv_a = 1.0 / coeff_a;
            var roots = array<f32, 2>((-coeff_b - sqrt_disc) * inv_a, (-coeff_b + sqrt_disc) * inv_a);
            for (var i = 0; i < 2; i = i + 1) {
                let t = roots[i];
                let axial = za + t * zd;
                if (!(axial >= band_lo && axial <= band_hi)) {
                    continue;
                }
                // Keep only the physical nappe: radius_a - axial * sin_a >= 0.
                if (rhs0 + t * rhs1 < 0.0) {
                    continue;
                }
                let p = origin + direction * t;
                let radial = p - a - n * axial;
                let len2 = dot(radial, radial);
                if (len2 <= 0.0) {
                    continue;
                }
                let radial_u = radial * (1.0 / sqrt(len2));
                // Outward normal tilts off the radial toward the narrow end by
                // the half-angle; unit by construction (cos^2 + sin^2 = 1).
                let outward = radial_u * cos_a + n * sin_a;
                consider(best, t, outward, t_min, t_max);
            }
        }
    }

    // Sphere at `a`, clipped to the cap below the tangent circle.
    let ra = sphere_roots(oa, direction, dd, radius_a * radius_a);
    if (ra.ok) {
        var roots = array<f32, 2>(ra.t0, ra.t1);
        for (var i = 0; i < 2; i = i + 1) {
            let t = roots[i];
            let p = origin + direction * t;
            if (dot(p - a, n) <= band_lo) {
                let radial = p - a;
                let len2 = dot(radial, radial);
                if (len2 > 0.0) {
                    consider(best, t, radial * (1.0 / sqrt(len2)), t_min, t_max);
                }
            }
        }
    }

    // Sphere at `b`, clipped to the cap above the tangent circle.
    let ob = origin - b;
    let rb = sphere_roots(ob, direction, dd, radius_b * radius_b);
    if (rb.ok) {
        var roots = array<f32, 2>(rb.t0, rb.t1);
        for (var i = 0; i < 2; i = i + 1) {
            let t = roots[i];
            let p = origin + direction * t;
            if (dot(p - b, n) >= radius_b * sin_a) {
                let radial = p - b;
                let len2 = dot(radial, radial);
                if (len2 > 0.0) {
                    consider(best, t, radial * (1.0 / sqrt(len2)), t_min, t_max);
                }
            }
        }
    }
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let origin = q.origin;
    let direction = q.direction;
    let t_min = q.t_min;
    let t_max = q.t_max;
    let a = q.a;
    let b = q.b;
    // Fold both radii to their magnitude, mirroring `RoundCone::new`.
    let radius_a = abs(q.radius_a);
    let radius_b = abs(q.radius_b);

    var out: Result;
    out.t = 0.0;
    out.hit = 0u;
    out.front_face = 0u;
    out.pad0 = 0.0;
    out.normal = vec3<f32>(0.0, 0.0, 0.0);
    out.pad1 = 0.0;

    let dd = dot(direction, direction);
    if (dd <= 0.0) {
        results[idx] = out;
        return;
    }
    if (radius_a <= 0.0 && radius_b <= 0.0) {
        results[idx] = out;
        return;
    }

    let inf = bitcast<f32>(0x7f800000u);
    var best: Best;
    best.t = inf;
    best.outward = vec3<f32>(0.0, 0.0, 0.0);

    let w = b - a;
    let l2 = dot(w, w);
    let rr = radius_a - radius_b;

    if (l2 <= rr * rr) {
        // One sphere engulfs the other (or the axis is zero): larger end sphere.
        var center = a;
        var radius = radius_a;
        if (radius_a < radius_b) {
            center = b;
            radius = radius_b;
        }
        emit_sphere(&best, origin, direction, dd, center, radius, t_min, t_max);
    } else {
        let inv_l = 1.0 / sqrt(l2);
        let n = w * inv_l;
        let l = l2 * inv_l;
        if (abs(rr) <= RR_EPS) {
            intersect_cylinder(&best, origin, direction, dd, a, b, n, l, radius_a, t_min, t_max);
        } else {
            intersect_tapered(
                &best, origin, direction, dd, a, b, n, l, rr, radius_a, radius_b, t_min, t_max,
            );
        }
    }

    if (best.t < inf) {
        let front = dot(direction, best.outward) < 0.0;
        var normal = best.outward;
        if (!front) {
            normal = -best.outward;
        }
        out.t = best.t;
        out.hit = 1u;
        out.front_face = select(0u, 1u, front);
        out.normal = normal;
    }
    results[idx] = out;
}
"#;

/// One ray/round-cone query: a ray interval and the round cone's two endpoints
/// and radii. The ray direction need not be unit length; both radii are folded
/// to their magnitude on device, mirroring `RoundCone::new`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayRoundConeQuery {
    /// Ray origin.
    pub origin: [f32; 3],
    /// Ray direction (not required to be unit length).
    pub direction: [f32; 3],
    /// Lower bound of the ray parameter interval.
    pub t_min: f32,
    /// Upper bound of the ray parameter interval.
    pub t_max: f32,
    /// First segment endpoint (center of the `a`-side sphere).
    pub a: [f32; 3],
    /// Second segment endpoint (center of the `b`-side sphere).
    pub b: [f32; 3],
    /// Sphere radius at endpoint `a` (folded to its magnitude on device).
    pub radius_a: f32,
    /// Sphere radius at endpoint `b` (folded to its magnitude on device).
    pub radius_b: f32,
}

/// The resolved answer for one query, mirroring the reference `RoundConeHit`
/// plus the miss case: `hit` is `0` on a miss (with all other fields zero) and
/// `1` on a hit; `front_face` is `1` when the ray struck the outward-facing side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayRoundConeResult {
    /// `1` when a nearest in-interval hit exists, else `0`.
    pub hit: u32,
    /// Ray parameter at the intersection (valid only when `hit` is `1`).
    pub t: f32,
    /// Unit surface normal, oriented against the incident ray.
    pub normal: [f32; 3],
    /// `1` when the ray struck the outward-facing side, else `0`.
    pub front_face: u32,
}

/// `repr(C)` `std430` layout of one packed query: four `vec4` slots holding
/// `(origin.xyz, t_min)`, `(direction.xyz, t_max)`, `(a.xyz, radius_a)` and
/// `(b.xyz, radius_b)` — `64` bytes, each `vec3` on its `16`-byte-aligned slot
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin.
    origin: [f32; 3],
    /// Lower bound of the ray parameter interval, packed into the origin slot.
    t_min: f32,
    /// Ray direction.
    direction: [f32; 3],
    /// Upper bound of the ray parameter interval, packed into the direction slot.
    t_max: f32,
    /// First segment endpoint.
    a: [f32; 3],
    /// Radius at endpoint `a`, packed into the `a` slot.
    radius_a: f32,
    /// Second segment endpoint.
    b: [f32; 3],
    /// Radius at endpoint `b`, packed into the `b` slot.
    radius_b: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &RayRoundConeQuery) -> GpuQuery {
        GpuQuery {
            origin: query.origin,
            t_min: query.t_min,
            direction: query.direction,
            t_max: query.t_max,
            a: query.a,
            radius_a: query.radius_a,
            b: query.b,
            radius_b: query.radius_b,
        }
    }
}

/// `repr(C)` `std430` layout of one result: `(t, hit, front_face, pad)` then a
/// `vec4` slot for the normal — `32` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Chosen hit parameter (valid only when `hit` is `1`).
    t: f32,
    /// `1` when a nearest in-interval hit exists, else `0`.
    hit: u32,
    /// `1` when the ray struck the outward-facing side, else `0`.
    front_face: u32,
    /// Padding lane.
    pad0: f32,
    /// Unit surface normal, oriented against the incident ray.
    normal: [f32; 3],
    /// Padding lane after the normal.
    pad1: f32,
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

/// A compiled, reusable ray/round-cone compute pipeline.
pub struct GpuRayRoundCone {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayRoundCone {
    /// Compiles the ray/round-cone kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayRoundCone {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_round_cone"),
            source: ShaderSource::Wgsl(RAY_ROUND_CONE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_round_cone_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_round_cone_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_round_cone_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayRoundCone {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`RayRoundConeResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference `RoundCone::intersect` answer to within
    /// the tolerance documented on this module. An empty input returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RayRoundConeQuery],
    ) -> Vec<RayRoundConeResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_round_cone_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_round_cone_output"),
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
            label: Some("prism_volumetric_ray_round_cone_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_round_cone_bind_group"),
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
            label: Some("prism_volumetric_ray_round_cone_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ray_round_cone_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_round_cone_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`RayRoundConeResult`].
fn decode_result(raw: &GpuResult) -> RayRoundConeResult {
    RayRoundConeResult {
        hit: raw.hit,
        t: raw.t,
        normal: raw.normal,
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
