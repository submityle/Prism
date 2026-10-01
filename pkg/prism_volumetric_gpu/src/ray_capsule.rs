//! `wgpu` compute twin of the analytic ray-capsule intersection contract
//! ([`ray_capsule`](prism_render_architecture::particle::ray_capsule), particle
//! design §10, §14).
//!
//! The `CPU` golden
//! [`ray_capsule`](prism_render_architecture::particle::ray_capsule) owns the
//! closed-form solution of the ray-capsule problem. A **capsule** is the set of
//! points within `radius` of the finite segment `a -> b`: a cylindrical side
//! wall closed off by a hemisphere at each end. The reference solves the surface
//! as the union of three analytic pieces — the side wall (a scalar quadratic in
//! the plane perpendicular to the unit axis, a root counting only when its axial
//! coordinate `m` lands inside `[0, height]`), the near hemisphere (a ray-sphere
//! solve about `a`, counted only on its outer half `m <= 0`) and the far
//! hemisphere (a ray-sphere solve about `b`, counted only on `m >= height`) —
//! and the smallest non-negative ray parameter across all three wins.
//! [`GpuRayCapsule`] is the on-device twin: it runs one thread per query and
//! reproduces the same answers the batch reference produces, so a passing
//! real-device parity test is direct evidence the ported kernel solves the same
//! geometry and classifies the same degenerate cases the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced:
//! `Capsule::closest_on_segment` (the clamped projection onto the axis segment),
//! `Capsule::contains` (the strict-inside predicate), `Capsule::first_hit` (the
//! nearest forward `RayCapsuleHit`: its ray parameter, world-space point and
//! outward unit normal) and `Capsule::intersects` (the unsigned boolean, which
//! the reference defines as `first_hit(..).is_some()`). The degenerate cases the
//! reference handles explicitly are mirrored branch for branch: a radius at or
//! below the compare epsilon is rejected as non-intersecting and contains
//! nothing; a zero-length ray direction is rejected rather than producing a
//! `NaN`; coincident endpoints collapse the segment to the single point `a` that
//! the two hemispheres then cover between them; and the hand-rolled
//! `normalize_or_zero` guards the reconstructed normal against a zero-length
//! divide exactly as the reference does.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `sqrt`, `+ - * /` and unsigned bit arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan` or optional device feature, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`. The only non-rational
//! operations are the `sqrt` of each discriminant and axis length, matching the
//! reference's `f32::sqrt` calls.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and a few
//! `sqrt`s, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields while pinning
//! the boolean classification exactly, tight enough to catch a genuinely wrong
//! port (a dropped surface piece, a wrong cap half, a wrong normal) yet loose
//! enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`ray_capsule`](prism_render_architecture::particle::ray_capsule) plus `wgpu`
//! compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::ray_capsule::{Capsule, Ray, RayCapsuleHit, Vec3};
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

/// Bit flag set when the forward half-line strikes the capsule surface
/// (`Capsule::intersects`, equal to a nearest forward hit existing).
const FLAG_INTERSECTS: u32 = 1;
/// Bit flag set when the query point lies strictly inside the solid capsule
/// (`Capsule::contains`).
const FLAG_CONTAINS: u32 = 2;
/// Bit flag set when a nearest forward hit exists (`Capsule::first_hit`).
const FLAG_HIT: u32 = 4;

/// The portable core-`WGSL` ray-capsule kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`ray_capsule`](prism_render_architecture::particle::ray_capsule) branch for
/// branch; see the module documentation for the algorithm.
const RAY_CAPSULE_WGSL: &str = r#"
// Analytic ray-capsule twin: one thread per query solves the capsule surface as
// the union of a side-wall quadratic plus two hemisphere ray-sphere solves,
// reports the closest axis-segment point, the strict-inside predicate, the
// unsigned intersect flag and the nearest forward hit with its outward unit
// normal. It mirrors the CPU golden `particle::ray_capsule`, uses only the
// portable core-WGSL subset (min/max/clamp/abs/sqrt and + - * / plus unsigned
// bit math) and takes no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::ray_capsule; no
// third-party engine source or derived code.

// Epsilon used to guard divisions, classify discriminants, and compare
// parameters against zero without ever writing an exact == / != on an f32.
const CMP_EPS: f32 = 1.0e-6;
// A finite sentinel standing in for +inf as the running best ray parameter;
// every real fixture parameter is many orders of magnitude smaller.
const T_SENTINEL: f32 = 1.0e30;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Ray origin in the first three lanes; capsule radius in the fourth.
    origin: vec3<f32>,
    radius: f32,
    // Ray direction (not required to be unit length); a pad lane follows.
    dir: vec3<f32>,
    pad0: f32,
    // Near segment endpoint / near hemisphere center; a pad lane follows.
    a: vec3<f32>,
    pad1: f32,
    // Far segment endpoint / far hemisphere center; a pad lane follows.
    b: vec3<f32>,
    pad2: f32,
    // Containment / closest-point query point; a pad lane follows.
    point: vec3<f32>,
    pad3: f32,
}

struct Result {
    // Chosen forward-hit parameter (valid only when the hit flag is set) and the
    // packed classification flags, with two pad lanes to fill the slot.
    hit_t: f32,
    flags: u32,
    pad0: f32,
    pad1: f32,
    // World-space hit point; a pad lane follows.
    hit_point: vec3<f32>,
    pad2: f32,
    // Outward unit surface normal at the hit; a pad lane follows.
    normal: vec3<f32>,
    pad3: f32,
    // Closest point on the axis segment to the query point; a pad lane follows.
    closest: vec3<f32>,
    pad4: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Unit vector in the same direction, or the zero vector when the input is
// shorter than CMP_EPS, mirroring the reference `normalize_or_zero` so a
// degenerate input can never yield a NaN.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len = sqrt(dot(v, v));
    if (len < CMP_EPS) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return v * (1.0 / len);
}

// The point on the axis segment a -> b closest to p, clamped to the endpoints,
// mirroring `Capsule::closest_on_segment`. Coincident endpoints collapse to a.
fn closest_on_segment(a: vec3<f32>, b: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let ba = b - a;
    let baba = dot(ba, ba);
    if (baba < CMP_EPS) {
        return a;
    }
    let h = clamp(dot(p - a, ba) / baba, 0.0, 1.0);
    return a + ba * h;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let origin = q.origin;
    let dir = q.dir;
    let a = q.a;
    let b = q.b;
    let radius = q.radius;
    let p = q.point;
    let r2 = radius * radius;

    var flags: u32 = 0u;
    var hit_t: f32 = 0.0;
    var hit_point = vec3<f32>(0.0, 0.0, 0.0);
    var normal = vec3<f32>(0.0, 0.0, 0.0);

    // closest_on_segment: always reported, independent of the ray.
    let closest = closest_on_segment(a, b, p);

    // is_degenerate: a radius at or below the compare epsilon encloses nothing.
    let degenerate = radius <= CMP_EPS;

    // Capsule::contains: strictly inside the solid; a degenerate capsule
    // contains nothing.
    if (!degenerate) {
        let diff = p - closest;
        if (dot(diff, diff) < r2 - CMP_EPS) {
            flags = flags | 2u;
        }
    }

    // Capsule::first_hit: reject a degenerate capsule or a near-zero direction,
    // then take the smallest non-negative candidate across the three pieces.
    let dir_len2 = dot(dir, dir);
    if (!degenerate && dir_len2 > CMP_EPS) {
        let ca = normalize_or_zero(b - a);
        let height = sqrt(dot(b - a, b - a));

        var best_t: f32 = T_SENTINEL;
        var best_point = vec3<f32>(0.0, 0.0, 0.0);
        var hit_any = false;

        // --- Side wall: quadratic in the plane perpendicular to the axis. ---
        let oc = origin - a;
        let d_par = dot(dir, ca);
        let oc_par = dot(oc, ca);
        let d_perp = dir - ca * d_par;
        let oc_perp = oc - ca * oc_par;
        let a_wall = dot(d_perp, d_perp);
        if (a_wall > CMP_EPS) {
            let bw = 2.0 * dot(d_perp, oc_perp);
            let cw = dot(oc_perp, oc_perp) - r2;
            let disc = bw * bw - 4.0 * a_wall * cw;
            if (disc >= 0.0) {
                let sqrt_disc = sqrt(max(disc, 0.0));
                let inv_2a = 1.0 / (2.0 * a_wall);
                let root0 = (-bw - sqrt_disc) * inv_2a;
                let root1 = (-bw + sqrt_disc) * inv_2a;
                if (root0 >= -CMP_EPS) {
                    let t = max(root0, 0.0);
                    let point = origin + dir * t;
                    let m = dot(point - a, ca);
                    if (m >= -CMP_EPS && m <= height + CMP_EPS && t < best_t) {
                        best_t = t;
                        best_point = point;
                        hit_any = true;
                    }
                }
                if (root1 >= -CMP_EPS) {
                    let t = max(root1, 0.0);
                    let point = origin + dir * t;
                    let m = dot(point - a, ca);
                    if (m >= -CMP_EPS && m <= height + CMP_EPS && t < best_t) {
                        best_t = t;
                        best_point = point;
                        hit_any = true;
                    }
                }
            }
        }

        // --- Hemisphere caps: a ray-sphere solve about each endpoint, accepted
        //     only on its outer hemisphere (near: m <= 0, far: m >= height). ---
        let a_sph = dot(dir, dir);
        let inv_2a_sph = 1.0 / (2.0 * a_sph);

        // Near cap, centered at a.
        let oc_n = origin - a;
        let bn = 2.0 * dot(dir, oc_n);
        let cn = dot(oc_n, oc_n) - r2;
        let disc_n = bn * bn - 4.0 * a_sph * cn;
        if (disc_n >= 0.0) {
            let sqrt_disc = sqrt(max(disc_n, 0.0));
            let root0 = (-bn - sqrt_disc) * inv_2a_sph;
            let root1 = (-bn + sqrt_disc) * inv_2a_sph;
            if (root0 >= -CMP_EPS) {
                let t = max(root0, 0.0);
                let point = origin + dir * t;
                let m = dot(point - a, ca);
                if (m <= CMP_EPS && t < best_t) {
                    best_t = t;
                    best_point = point;
                    hit_any = true;
                }
            }
            if (root1 >= -CMP_EPS) {
                let t = max(root1, 0.0);
                let point = origin + dir * t;
                let m = dot(point - a, ca);
                if (m <= CMP_EPS && t < best_t) {
                    best_t = t;
                    best_point = point;
                    hit_any = true;
                }
            }
        }

        // Far cap, centered at b.
        let oc_f = origin - b;
        let bf = 2.0 * dot(dir, oc_f);
        let cf = dot(oc_f, oc_f) - r2;
        let disc_f = bf * bf - 4.0 * a_sph * cf;
        if (disc_f >= 0.0) {
            let sqrt_disc = sqrt(max(disc_f, 0.0));
            let root0 = (-bf - sqrt_disc) * inv_2a_sph;
            let root1 = (-bf + sqrt_disc) * inv_2a_sph;
            if (root0 >= -CMP_EPS) {
                let t = max(root0, 0.0);
                let point = origin + dir * t;
                let m = dot(point - a, ca);
                if (m >= height - CMP_EPS && t < best_t) {
                    best_t = t;
                    best_point = point;
                    hit_any = true;
                }
            }
            if (root1 >= -CMP_EPS) {
                let t = max(root1, 0.0);
                let point = origin + dir * t;
                let m = dot(point - a, ca);
                if (m >= height - CMP_EPS && t < best_t) {
                    best_t = t;
                    best_point = point;
                    hit_any = true;
                }
            }
        }

        if (hit_any) {
            hit_t = best_t;
            hit_point = best_point;
            // outward_normal: from the closest axis-segment point to the hit.
            normal = normalize_or_zero(best_point - closest_on_segment(a, b, best_point));
            // intersects is defined as first_hit being present; set both.
            flags = flags | 1u;
            flags = flags | 4u;
        }
    }

    var out: Result;
    out.hit_t = hit_t;
    out.flags = flags;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.hit_point = hit_point;
    out.pad2 = 0.0;
    out.normal = normal;
    out.pad3 = 0.0;
    out.closest = closest;
    out.pad4 = 0.0;
    results[idx] = out;
}
"#;

/// One ray-capsule query: a [`Ray`] and a query point tested against a
/// [`Capsule`], the same inputs the reference `Capsule::first_hit`,
/// `Capsule::intersects`, `Capsule::contains` and `Capsule::closest_on_segment`
/// consume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayCapsuleQuery {
    /// The ray being traced (direction need not be unit length).
    pub ray: Ray,
    /// The capsule being intersected and classified against.
    pub capsule: Capsule,
    /// The point classified by `contains` and projected by `closest_on_segment`.
    pub point: Vec3,
}

impl RayCapsuleQuery {
    /// Builds a query from a ray, a capsule and a containment / closest-point
    /// query point.
    #[must_use]
    pub const fn new(ray: Ray, capsule: Capsule, point: Vec3) -> RayCapsuleQuery {
        RayCapsuleQuery {
            ray,
            capsule,
            point,
        }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// reports: the unsigned intersect predicate, the containment predicate, the
/// closest axis-segment point and the nearest forward hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayCapsuleResult {
    /// Whether the forward half-line strikes the capsule surface, matching
    /// `Capsule::intersects` (equal to [`RayCapsuleResult::hit`] being
    /// [`Some`]).
    pub intersects: bool,
    /// Whether the query point lies strictly inside the solid capsule, matching
    /// `Capsule::contains`.
    pub contains: bool,
    /// The point on the axis segment closest to the query point, matching
    /// `Capsule::closest_on_segment`.
    pub closest: Vec3,
    /// The nearest forward hit, matching `Capsule::first_hit`, or [`None`].
    pub hit: Option<RayCapsuleHit>,
}

/// `repr(C)` `std430` layout of one packed query: five `vec4` slots holding
/// `(origin.xyz, radius)`, `(dir.xyz, pad)`, `(a.xyz, pad)`, `(b.xyz, pad)` and
/// `(point.xyz, pad)` — `80` bytes, each `vec3` on its `16`-byte-aligned slot
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin.
    origin: [f32; 3],
    /// Capsule radius, packed into the origin slot's fourth lane.
    radius: f32,
    /// Ray direction.
    dir: [f32; 3],
    /// Padding lane after the direction.
    pad0: f32,
    /// Near segment endpoint.
    a: [f32; 3],
    /// Padding lane after the near endpoint.
    pad1: f32,
    /// Far segment endpoint.
    b: [f32; 3],
    /// Padding lane after the far endpoint.
    pad2: f32,
    /// Containment / closest-point query point.
    point: [f32; 3],
    /// Padding lane after the query point.
    pad3: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &RayCapsuleQuery) -> GpuQuery {
        GpuQuery {
            origin: [query.ray.origin.x, query.ray.origin.y, query.ray.origin.z],
            radius: query.capsule.radius,
            dir: [query.ray.dir.x, query.ray.dir.y, query.ray.dir.z],
            pad0: 0.0,
            a: [query.capsule.a.x, query.capsule.a.y, query.capsule.a.z],
            pad1: 0.0,
            b: [query.capsule.b.x, query.capsule.b.y, query.capsule.b.z],
            pad2: 0.0,
            point: [query.point.x, query.point.y, query.point.z],
            pad3: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: two scalars `(hit_t, flags)` with
/// two pad lanes, then three `vec4` slots for the hit point, the outward normal
/// and the closest axis-segment point — `64` bytes matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Chosen forward-hit parameter (valid only when the hit flag is set).
    hit_t: f32,
    /// Packed classification flags (`FLAG_INTERSECTS` and friends).
    flags: u32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// World-space hit point.
    hit_point: [f32; 3],
    /// Padding lane after the hit point.
    pad2: f32,
    /// Outward unit surface normal.
    normal: [f32; 3],
    /// Padding lane after the normal.
    pad3: f32,
    /// Closest point on the axis segment to the query point.
    closest: [f32; 3],
    /// Padding lane after the closest point.
    pad4: f32,
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

/// A compiled, reusable ray-capsule compute pipeline.
pub struct GpuRayCapsule {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayCapsule {
    /// Compiles the ray-capsule kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayCapsule {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_capsule"),
            source: ShaderSource::Wgsl(RAY_CAPSULE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_capsule_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_capsule_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_capsule_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayCapsule {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`RayCapsuleResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference answers (`Capsule::intersects`,
    /// `Capsule::contains`, `Capsule::closest_on_segment` and
    /// `Capsule::first_hit`) to within the tolerance documented on this module.
    /// An empty input returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[RayCapsuleQuery]) -> Vec<RayCapsuleResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_capsule_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_capsule_output"),
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
            label: Some("prism_volumetric_ray_capsule_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_capsule_bind_group"),
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
            label: Some("prism_volumetric_ray_capsule_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ray_capsule_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_capsule_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`RayCapsuleResult`],
/// unpacking the flag bits into the reference's boolean / `Option` shape.
fn decode_result(raw: &GpuResult) -> RayCapsuleResult {
    let intersects = raw.flags & FLAG_INTERSECTS != 0;
    let contains = raw.flags & FLAG_CONTAINS != 0;
    let has_hit = raw.flags & FLAG_HIT != 0;
    let closest = Vec3::new(raw.closest[0], raw.closest[1], raw.closest[2]);
    let hit = if has_hit {
        Some(RayCapsuleHit {
            t: raw.hit_t,
            point: Vec3::new(raw.hit_point[0], raw.hit_point[1], raw.hit_point[2]),
            normal: Vec3::new(raw.normal[0], raw.normal[1], raw.normal[2]),
        })
    } else {
        None
    };
    RayCapsuleResult {
        intersects,
        contains,
        closest,
        hit,
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
