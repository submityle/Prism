//! `wgpu` compute twin of the analytic ray-finite-cylinder intersection
//! contract ([`ray_cylinder`](prism_render_architecture::particle::ray_cylinder),
//! particle design §10, §14).
//!
//! The `CPU` golden
//! [`ray_cylinder`](prism_render_architecture::particle::ray_cylinder) owns the
//! closed-form solution of the ray against a **finite, capped cylinder**: a
//! segment of axis `base -> base + axis * height` swept at a radius `r` and
//! closed by two flat circular end caps. It solves the surface in two pieces —
//! the **side wall** (projecting the ray into the plane perpendicular to the
//! axis reduces the infinite cylinder to a scalar quadratic `a t^2 + b t + c`
//! whose root counts only when its axial coordinate lands in `[0, height]`) and
//! the two **end caps** (disk-bounded ray-plane solves accepted only when the
//! hit is within `r` of the cap center) — then reports the nearest forward
//! surface hit with its outward unit normal. [`GpuRayCylinder`] is the on-device
//! twin: it runs one thread per query and reproduces the same answers the batch
//! reference produces, so a passing real-device parity test is direct evidence
//! the ported kernel solves the same quadratic, applies the same axial-span and
//! cap-radius acceptance tests, and classifies the same degenerate cases the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced:
//! [`Cylinder::contains`] (the strict inside/outside point classification),
//! [`Cylinder::intersects`] (the unsigned boolean) and
//! [`Cylinder::first_hit`] (the [`RayCylinderHit`]: the nearest forward `t`, its
//! world-space point and the outward unit normal, radial on the wall and axial
//! on a cap). The degenerate cases the reference rejects rather than turning
//! into a `NaN` are mirrored branch for branch: a near-zero radius, a near-zero
//! height or a near-zero axis makes the solid enclose nothing (no hit, contains
//! nothing); a near-zero ray direction is not a ray; and a ray parallel to the
//! axis contributes no side-wall root (its perpendicular projection collapses)
//! and is answered by the caps alone. The hand-rolled `normalize_or_zero` guards
//! every normal against a zero-length divide exactly as the reference does.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `clamp`, `sqrt`, `+ - * /`, unsigned bit arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan` or optional device feature, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. The only non-rational operations are the
//! `sqrt` of the quadratic discriminant and the `sqrt` inside
//! `normalize_or_zero`, matching the reference's `f32::sqrt` calls.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! `sqrt`s, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields while pinning
//! the boolean classification (`contains`, `intersects`, hit presence) exactly,
//! tight enough to catch a genuinely wrong port (a dropped cap branch, a wrong
//! axial-span test, a flipped normal) yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`ray_cylinder`](prism_render_architecture::particle::ray_cylinder) plus
//! `wgpu` compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::ray_cylinder::{Cylinder, Ray, RayCylinderHit, Vec3};
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

/// Bit flag set when the test point lies strictly inside the solid
/// ([`Cylinder::contains`]).
const FLAG_CONTAINS: u32 = 1;
/// Bit flag set when the forward half-line strikes the surface
/// ([`Cylinder::intersects`]).
const FLAG_INTERSECTS: u32 = 2;
/// Bit flag set when a nearest forward hit exists ([`Cylinder::first_hit`]).
const FLAG_HIT: u32 = 4;

/// The portable core-`WGSL` ray-cylinder kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`ray_cylinder`](prism_render_architecture::particle::ray_cylinder) branch
/// for branch; see the module documentation for the algorithm.
const RAY_CYLINDER_WGSL: &str = r#"
// Analytic ray-finite-cylinder twin: one thread per query classifies a test
// point against the solid, solves the side-wall quadratic and the two end-cap
// planes, and emits the contains / intersects / hit flags and the nearest
// forward hit with its outward unit normal. It mirrors the CPU golden
// `particle::ray_cylinder`, uses only the portable core-WGSL subset
// (min/max/abs/clamp/sqrt and + - * / plus unsigned bit math) and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::ray_cylinder; no
// third-party engine source or derived code.

// Epsilon used to guard divisions, classify the quadratic discriminant, and
// compare parameters against zero without ever writing an exact == / != on an
// f32.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Ray origin in the first three lanes; cylinder radius in the fourth.
    origin: vec3<f32>,
    radius: f32,
    // Ray direction (not required to be unit length); cylinder height follows.
    dir: vec3<f32>,
    height: f32,
    // Cylinder base (center of the base cap); a pad lane follows.
    base: vec3<f32>,
    pad0: f32,
    // Cylinder axis (not required to be unit length); a pad lane follows.
    axis: vec3<f32>,
    pad1: f32,
    // The point tested by `contains`; a pad lane follows.
    point: vec3<f32>,
    pad2: f32,
}

struct Result {
    // The chosen forward-hit parameter and the packed classification flags.
    hit_t: f32,
    flags: u32,
    rpad0: f32,
    rpad1: f32,
    // World-space hit point; a pad lane follows.
    point: vec3<f32>,
    rpad2: f32,
    // Outward unit surface normal; a pad lane follows.
    normal: vec3<f32>,
    rpad3: f32,
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

// Whether the cylinder encloses no volume: a near-zero radius, a near-zero
// height, or an axis too short to define a direction, mirroring
// `Cylinder::is_degenerate`.
fn is_degenerate(radius: f32, height: f32, axis: vec3<f32>) -> bool {
    return radius <= CMP_EPS || height <= CMP_EPS || dot(axis, axis) < CMP_EPS;
}

// `Cylinder::contains`: strictly inside axially within (0, height) and radially
// within the radius, both by more than CMP_EPS. A degenerate cylinder contains
// nothing.
fn contains(p: vec3<f32>, base: vec3<f32>, ca: vec3<f32>, radius: f32, height: f32) -> bool {
    let rel = p - base;
    let m = dot(rel, ca);
    if (m <= CMP_EPS || m >= height - CMP_EPS) {
        return false;
    }
    let perp = rel - ca * m;
    return dot(perp, perp) < radius * radius - CMP_EPS;
}

// Keeps the nearest forward candidate, mirroring the reference `push` closure
// that only overwrites `best` when the new parameter is strictly closer.
fn consider(
    cand_t: f32,
    cand_point: vec3<f32>,
    cand_normal: vec3<f32>,
    have: ptr<function, bool>,
    best_t: ptr<function, f32>,
    best_point: ptr<function, vec3<f32>>,
    best_normal: ptr<function, vec3<f32>>,
) {
    let closer = !(*have) || cand_t < *best_t;
    if (closer) {
        *have = true;
        *best_t = cand_t;
        *best_point = cand_point;
        *best_normal = cand_normal;
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
    let dir = q.dir;
    let base = q.base;
    let axis = q.axis;
    let radius = q.radius;
    let height = q.height;
    let test_point = q.point;

    var hit_t: f32 = 0.0;
    var best_point = vec3<f32>(0.0, 0.0, 0.0);
    var best_normal = vec3<f32>(0.0, 0.0, 0.0);
    var have_hit = false;
    var flags: u32 = 0u;

    let degenerate = is_degenerate(radius, height, axis);

    // A degenerate cylinder encloses nothing and never intersects, exactly as
    // the reference short-circuits both `contains` and `first_hit`.
    if (!degenerate) {
        let ca = normalize_or_zero(axis);

        // Cylinder::contains on the test point.
        if (contains(test_point, base, ca, radius, height)) {
            flags = flags | 1u;
        }

        // first_hit: a zero-length direction is not a ray.
        if (dot(dir, dir) > CMP_EPS) {
            // --- Side wall: quadratic in the plane perpendicular to the axis.
            let oc = origin - base;
            let d_par = dot(dir, ca);
            let oc_par = dot(oc, ca);
            let d_perp = dir - ca * d_par;
            let oc_perp = oc - ca * oc_par;
            let a = dot(d_perp, d_perp);
            if (a > CMP_EPS) {
                let b = 2.0 * dot(d_perp, oc_perp);
                let c = dot(oc_perp, oc_perp) - radius * radius;
                let disc = b * b - 4.0 * a * c;
                if (disc >= -CMP_EPS) {
                    let sqrt_disc = sqrt(max(disc, 0.0));
                    let inv_2a = 1.0 / (2.0 * a);
                    var roots = array<f32, 2>(
                        (-b - sqrt_disc) * inv_2a,
                        (-b + sqrt_disc) * inv_2a
                    );
                    for (var i = 0; i < 2; i = i + 1) {
                        let raw_t = roots[i];
                        if (raw_t >= -CMP_EPS) {
                            let t = max(raw_t, 0.0);
                            let point = origin + dir * t;
                            let m = dot(point - base, ca);
                            if (m >= -CMP_EPS && m <= height + CMP_EPS) {
                                let axis_point = base + ca * m;
                                let normal = normalize_or_zero(point - axis_point);
                                consider(
                                    t, point, normal,
                                    &have_hit, &hit_t, &best_point, &best_normal
                                );
                            }
                        }
                    }
                }
            }

            // --- End caps: two disk-bounded planes at the base and the top. ---
            let denom = dot(dir, ca);
            if (abs(denom) > CMP_EPS) {
                let inv_denom = 1.0 / denom;
                var centers = array<vec3<f32>, 2>(base, base + ca * height);
                var normals = array<vec3<f32>, 2>(ca * -1.0, ca);
                for (var i = 0; i < 2; i = i + 1) {
                    let center = centers[i];
                    let out_normal = normals[i];
                    let raw_t = dot(center - origin, ca) * inv_denom;
                    if (raw_t >= -CMP_EPS) {
                        let t = max(raw_t, 0.0);
                        let point = origin + dir * t;
                        let rel = point - center;
                        let axial = dot(rel, ca);
                        let perp = rel - ca * axial;
                        if (dot(perp, perp) <= radius * radius + CMP_EPS) {
                            consider(
                                t, point, out_normal,
                                &have_hit, &hit_t, &best_point, &best_normal
                            );
                        }
                    }
                }
            }
        }
    }

    if (have_hit) {
        // Cylinder::intersects is `first_hit(ray).is_some()`.
        flags = flags | 2u;
        flags = flags | 4u;
    }

    var out: Result;
    out.hit_t = hit_t;
    out.flags = flags;
    out.rpad0 = 0.0;
    out.rpad1 = 0.0;
    out.point = best_point;
    out.rpad2 = 0.0;
    out.normal = best_normal;
    out.rpad3 = 0.0;
    results[idx] = out;
}
"#;

/// One ray-cylinder query: a [`Ray`] and a test [`Vec3`] tested against a
/// [`Cylinder`], the same inputs the reference [`Cylinder::contains`] /
/// [`Cylinder::first_hit`] / [`Cylinder::intersects`] consume. Carrying the
/// cylinder per query lets one dispatch mix rays against many distinct
/// cylinders. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds
/// `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayCylinderQuery {
    /// The ray being traced (direction need not be unit length).
    pub ray: Ray,
    /// The cylinder being intersected and whose volume is point-tested.
    pub cylinder: Cylinder,
    /// The point classified by [`Cylinder::contains`].
    pub point: Vec3,
}

impl RayCylinderQuery {
    /// Builds a query from a ray, a cylinder and the point to classify.
    #[must_use]
    pub const fn new(ray: Ray, cylinder: Cylinder, point: Vec3) -> RayCylinderQuery {
        RayCylinderQuery {
            ray,
            cylinder,
            point,
        }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// reports: the point-containment predicate, the surface-intersection predicate
/// and the nearest forward hit. Derives only [`PartialEq`] (no `Eq`/`Hash`)
/// because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayCylinderResult {
    /// Whether the test point lies strictly inside the solid, matching
    /// [`Cylinder::contains`].
    pub contains: bool,
    /// Whether the forward half-line strikes the surface, matching
    /// [`Cylinder::intersects`].
    pub intersects: bool,
    /// The nearest forward hit, matching [`Cylinder::first_hit`], or [`None`].
    pub hit: Option<RayCylinderHit>,
}

/// `repr(C)` `std430` layout of one packed query: five `vec4` slots holding
/// `(origin.xyz, radius)`, `(dir.xyz, height)`, `(base.xyz, pad)`,
/// `(axis.xyz, pad)` and `(point.xyz, pad)` — `80` bytes, each `vec3` on its
/// `16`-byte-aligned slot exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin.
    origin: [f32; 3],
    /// Cylinder radius, packed into the origin slot's fourth lane.
    radius: f32,
    /// Ray direction.
    dir: [f32; 3],
    /// Cylinder height, packed into the direction slot's fourth lane.
    height: f32,
    /// Cylinder base (center of the base cap).
    base: [f32; 3],
    /// Padding lane after the base.
    pad0: f32,
    /// Cylinder axis direction.
    axis: [f32; 3],
    /// Padding lane after the axis.
    pad1: f32,
    /// The point classified by `contains`.
    point: [f32; 3],
    /// Padding lane after the point.
    pad2: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn from_query(query: &RayCylinderQuery) -> GpuQuery {
        let o = query.ray.origin;
        let d = query.ray.dir;
        let cyl = query.cylinder;
        let p = query.point;
        GpuQuery {
            origin: [o.x, o.y, o.z],
            radius: cyl.radius,
            dir: [d.x, d.y, d.z],
            height: cyl.height,
            base: [cyl.base.x, cyl.base.y, cyl.base.z],
            pad0: 0.0,
            axis: [cyl.axis.x, cyl.axis.y, cyl.axis.z],
            pad1: 0.0,
            point: [p.x, p.y, p.z],
            pad2: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: four scalars `(hit_t, flags, pad,`
/// `pad)` then two `vec4` slots for the hit point and the outward normal — `48`
/// bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Chosen forward-hit parameter (valid only when the hit flag is set).
    hit_t: f32,
    /// Packed classification flags (`FLAG_CONTAINS` and friends).
    flags: u32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// World-space hit point.
    point: [f32; 3],
    /// Padding lane after the point.
    pad2: f32,
    /// Outward unit surface normal.
    normal: [f32; 3],
    /// Padding lane after the normal.
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

/// A compiled, reusable ray-cylinder compute pipeline.
pub struct GpuRayCylinder {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayCylinder {
    /// Compiles the ray-cylinder kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayCylinder {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_cylinder"),
            source: ShaderSource::Wgsl(RAY_CYLINDER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_cylinder_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_cylinder_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_cylinder_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayCylinder {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`RayCylinderResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference answers
    /// ([`Cylinder::contains`], [`Cylinder::intersects`] and
    /// [`Cylinder::first_hit`]) to within the tolerance documented on this
    /// module. An empty input returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[RayCylinderQuery]) -> Vec<RayCylinderResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_cylinder_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_cylinder_output"),
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
            label: Some("prism_volumetric_ray_cylinder_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_cylinder_bind_group"),
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
            label: Some("prism_volumetric_ray_cylinder_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ray_cylinder_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_cylinder_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`RayCylinderResult`],
/// unpacking the flag bits into the reference's boolean / `Option` shape.
fn decode_result(raw: &GpuResult) -> RayCylinderResult {
    let contains = raw.flags & FLAG_CONTAINS != 0;
    let intersects = raw.flags & FLAG_INTERSECTS != 0;
    let has_hit = raw.flags & FLAG_HIT != 0;
    let hit = if has_hit {
        Some(RayCylinderHit {
            t: raw.hit_t,
            point: Vec3::new(raw.point[0], raw.point[1], raw.point[2]),
            normal: Vec3::new(raw.normal[0], raw.normal[1], raw.normal[2]),
        })
    } else {
        None
    };
    RayCylinderResult {
        contains,
        intersects,
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
