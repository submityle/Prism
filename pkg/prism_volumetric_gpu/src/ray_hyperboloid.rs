//! `wgpu` compute twin of the analytic *hyperboloid of one sheet* ray
//! intersection from this repository's
//! `prism_render_architecture::ray_scene::hyperboloid`.
//!
//! The `CPU` golden `Hyperboloid::intersect` owns the closed-form solution of
//! the ray-against-a-quadric-hyperboloid problem. A hyperboloid of revolution
//! is the surface of revolution whose radius grows away from a circular waist:
//! at signed axial distance `z` from the waist plane the radius obeys
//! `rho(z)^2 = waist^2 + flare^2 * z^2`. It is centered on the `center` waist
//! point, its axis and half-height come from `top` (the axis spans
//! `center +/- (top - center)`, symmetric about the waist), `waist` is the
//! throat radius and `flare` the radial growth per axial unit. `flare = 0`
//! degenerates to a cylinder of radius `waist`; `waist = 0` degenerates to a
//! double cone through the throat.
//!
//! # What is twinned
//!
//! The intersection substitutes the ray `origin + t * dir` into the implicit
//! quadric `F = |p - center|^2 - (1 + flare^2) * dot(p - center, n_hat)^2 -
//! waist^2 = 0`, giving a quadratic in `t` whose `t^2` coefficient carries
//! `dd = dot(dir, dir)` (so the test is correct for the non-unit ray
//! directions the scene feeds it), clips both roots to the finite axial band
//! `z` in `[-h, h]`, and returns the nearest valid root. The surface normal is
//! the analytic gradient `2 * (p - center) - 2 * (1 + flare^2) * z * n_hat`
//! (normalized), oriented against the incident ray. [`GpuRayHyperboloid`] is
//! the on-device twin: it runs one thread per query and reproduces the same
//! answers the reference produces, so a passing real-device parity test is
//! direct evidence the ported kernel solves the same geometry and classifies
//! the same degenerate cases, not merely that its shader compiles.
//!
//! The degenerate cases the reference handles explicitly are mirrored branch
//! for branch: a zero-length axis (`h2 <= 0`) is rejected; a near-zero
//! direction (`dd <= 0`) is rejected rather than producing a `NaN`; the
//! quadratic solver degrades to the linear case when the leading coefficient
//! `coeff_a` is exactly zero (the ray runs along a surface asymptote
//! direction) and misses when both leading coefficients vanish; a negative
//! discriminant misses; every candidate root outside the finite axial band or
//! the ray interval `[t_min, t_max]` is culled; and a vanishing gradient is
//! skipped.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `sqrt`, `select`, `+ - * /` — with no `sin`, `cos`, `tan`, `exp`, `log`,
//! `pow` or optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. The only non-rational operations are the two `sqrt`
//! calls (axis length and discriminant) plus the normalization `sqrt`,
//! matching the reference's `f32::sqrt` calls.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units
//! in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4 || rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on the `f32`
//! fields while pinning the hit flag and the front-face flag exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::hyperboloid`；无第三方引擎源码或衍生代码。

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly
/// default shared by every one-thread-per-element kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Discrete hit code written by the kernel for a lane that strikes the surface,
/// matching the host decode in [`decode_result`]. A direct `f32` equality is
/// forbidden, so the hit verdict is carried as this integer flag.
const CODE_HIT: u32 = 1;

/// Inlined `WGSL` compute shader source. Keeping it in the Rust binary avoids
/// shipping a sidecar asset and keeps the twin and its kernel versioned as a
/// single source file. The single entry point `solve` mirrors the `CPU` golden
/// `Hyperboloid::intersect` branch for branch; see the module documentation
/// for the algorithm.
const RAY_HYPERBOLOID_WGSL: &str = r#"
// Analytic ray-hyperboloid-of-one-sheet twin: one thread per query substitutes
// the ray into the implicit quadric, solves the scalar quadratic (degrading to
// the linear case when the leading coefficient vanishes), clips both roots to
// the finite axial band [-h, h] and the ray interval [t_min, t_max], and
// reports the nearest valid hit flag, its ray parameter, the oriented unit
// normal and the front-face flag. It mirrors the CPU golden
// Hyperboloid::intersect, uses only the portable core-WGSL subset
// (min/max/abs/sqrt/select and + - * /) and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::hyperboloid；无第三方引擎源码或衍生代码。

// A finite sentinel standing in for the "no candidate yet" running best ray
// parameter; every real fixture parameter is many orders of magnitude smaller.
const T_SENTINEL: f32 = 1.0e30;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 64-byte std430 stride matching the host `GpuQuery`: the ray origin
// and direction, the waist center and top rim, each vec3 on its own 16-byte
// slot with a trailing scalar (waist, flare, t_min, t_max) packed into the pad.
struct Query {
    origin: vec3<f32>,
    waist: f32,
    dir: vec3<f32>,
    flare: f32,
    center: vec3<f32>,
    t_min: f32,
    top: vec3<f32>,
    t_max: f32,
}

// One result. 32-byte std430 stride matching the host `GpuResult`: the hit flag
// as 0u/1u, the ray parameter, the front-face flag and a pad word, then the
// oriented unit normal on its own 16-byte slot.
struct Result {
    hit: u32,
    t: f32,
    front_face: u32,
    pad0: u32,
    normal: vec3<f32>,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Default miss: a zeroed result lane.
    var out: Result;
    out.hit = 0u;
    out.t = 0.0;
    out.front_face = 0u;
    out.pad0 = 0u;
    out.normal = vec3<f32>(0.0, 0.0, 0.0);
    out.pad1 = 0.0;

    let w = q.top - q.center;
    let h2 = dot(w, w);
    let dd = dot(q.dir, q.dir);
    // Degenerate axis or near-zero direction: reject rather than divide by zero.
    if (h2 <= 0.0 || dd <= 0.0) {
        results[idx] = out;
        return;
    }

    let h = sqrt(h2);
    let inv_h = 1.0 / h;
    // Unit axis from the waist toward `top`.
    let n = w * inv_h;
    // `1 + flare^2` weights the axial term in the quadric.
    let g = 1.0 + q.flare * q.flare;

    // Ray origin relative to the waist center.
    let a = q.origin - q.center;
    let za = dot(a, n);
    let zd = dot(q.dir, n);
    let ad = dot(a, q.dir);
    let aa = dot(a, a);

    // `coeff_a*t^2 + coeff_b*t + coeff_c = 0`.
    let coeff_a = dd - g * zd * zd;
    let coeff_b = 2.0 * (ad - g * za * zd);
    let coeff_c = aa - g * za * za - q.waist * q.waist;

    // Collect candidate roots: quadratic, or linear when coeff_a is zero (the
    // ray runs along a surface asymptote direction). `abs(x) > 0.0` reproduces
    // the reference `x != 0.0` with an ordered comparison only.
    var r0: f32 = 0.0;
    var r1: f32 = 0.0;
    var has_roots = false;
    if (abs(coeff_a) > 0.0) {
        let disc = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c;
        if (disc >= 0.0) {
            let sqrt_disc = sqrt(disc);
            let inv_2a = 1.0 / (2.0 * coeff_a);
            let ra = (-coeff_b - sqrt_disc) * inv_2a;
            let rb = (-coeff_b + sqrt_disc) * inv_2a;
            r0 = min(ra, rb);
            r1 = max(ra, rb);
            has_roots = true;
        }
    } else if (abs(coeff_b) > 0.0) {
        let r = -coeff_c / coeff_b;
        r0 = r;
        r1 = r;
        has_roots = true;
    }

    if (!has_roots) {
        results[idx] = out;
        return;
    }

    var best_t: f32 = T_SENTINEL;
    var best_outward = vec3<f32>(0.0, 0.0, 0.0);
    var found = false;

    // Candidate r0 (the smaller root).
    if (r0 >= q.t_min && r0 <= q.t_max && r0 < best_t) {
        let z = za + r0 * zd;
        if (z >= -h && z <= h) {
            let p = a + q.dir * r0;
            let gz = g * z;
            let grad = 2.0 * (p - gz * n);
            let nn = dot(grad, grad);
            if (nn > 0.0) {
                best_t = r0;
                best_outward = grad * (1.0 / sqrt(nn));
                found = true;
            }
        }
    }

    // Candidate r1 (the larger root); skipped when r0 already won.
    if (r1 >= q.t_min && r1 <= q.t_max && r1 < best_t) {
        let z = za + r1 * zd;
        if (z >= -h && z <= h) {
            let p = a + q.dir * r1;
            let gz = g * z;
            let grad = 2.0 * (p - gz * n);
            let nn = dot(grad, grad);
            if (nn > 0.0) {
                best_t = r1;
                best_outward = grad * (1.0 / sqrt(nn));
                found = true;
            }
        }
    }

    if (!found) {
        results[idx] = out;
        return;
    }

    // Orient the outward gradient against the incident ray.
    let front_face = dot(q.dir, best_outward) < 0.0;
    let normal = select(
        vec3<f32>(-best_outward.x, -best_outward.y, -best_outward.z),
        best_outward,
        front_face,
    );

    out.hit = 1u;
    out.t = best_t;
    out.front_face = select(0u, 1u, front_face);
    out.normal = normal;
    results[idx] = out;
}
"#;

/// One ray-hyperboloid query: the ray interval and the hyperboloid geometry.
///
/// Mirrors a single reference `Hyperboloid::intersect` call. The ray
/// `direction` need not be unit length; a near-zero direction misses. `waist`
/// and `flare` are stored as supplied (the reference folds caller negatives to
/// their magnitude; pass non-negative values to match). Derives only
/// [`PartialEq`] (no [`Eq`] / [`Hash`]) because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHyperboloidQuery {
    /// Ray origin.
    pub origin: [f32; 3],
    /// Ray direction (need not be unit length; a near-zero direction misses).
    pub dir: [f32; 3],
    /// Waist (throat) center, on the axis, midplane of the symmetric surface.
    pub center: [f32; 3],
    /// One rim end of the axis; the axis spans `center +/- (top - center)`.
    pub top: [f32; 3],
    /// Non-negative throat radius at the waist plane.
    pub waist: f32,
    /// Non-negative radial growth per axial unit away from the waist.
    pub flare: f32,
    /// Inclusive lower bound of the valid ray interval.
    pub t_min: f32,
    /// Inclusive upper bound of the valid ray interval.
    pub t_max: f32,
}

impl RayHyperboloidQuery {
    /// Builds a ray-hyperboloid query over the interval `[t_min, t_max]`.
    ///
    /// The caller is expected to pass `t_min >= 0` and `t_max >= t_min`
    /// (the reference `Ray::new` clamps those invariants; this twin consumes
    /// them directly).
    #[must_use]
    pub const fn new(
        origin: [f32; 3],
        dir: [f32; 3],
        center: [f32; 3],
        top: [f32; 3],
        waist: f32,
        flare: f32,
        t_min: f32,
        t_max: f32,
    ) -> RayHyperboloidQuery {
        RayHyperboloidQuery {
            origin,
            dir,
            center,
            top,
            waist,
            flare,
            t_min,
            t_max,
        }
    }
}

/// The resolved verdict for one query, the host-side mirror of the kernel's
/// `Result` lane and of the reference's `Option<HyperboloidHit>`.
///
/// `hit` is the exact flag (`1` when the ray strikes the surface within the
/// band and interval, `0` otherwise); `t`, `normal` and `front_face` carry the
/// nearest valid hit and are meaningful only when `hit` is `1` (otherwise all
/// are zero). Derives only [`PartialEq`] (no [`Eq`] / [`Hash`]) because it
/// holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHyperboloidResult {
    /// Hit flag: `1` when the ray strikes the surface, `0` on a miss.
    pub hit: u32,
    /// Ray parameter of the nearest valid hit; meaningful only when `hit`.
    pub t: f32,
    /// Oriented unit surface normal; meaningful only when `hit`, else zero.
    pub normal: [f32; 3],
    /// Front-face flag: `1` when the ray struck the outward-facing side,
    /// `0` otherwise; meaningful only when `hit`.
    pub front_face: u32,
}

/// `repr(C)` `std430` layout of one packed query: four `vec4` slots holding
/// `(origin.xyz, waist)`, `(dir.xyz, flare)`, `(center.xyz, t_min)` and
/// `(top.xyz, t_max)` — `64` bytes, each `vec3` on its `16`-byte-aligned slot
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin.
    origin: [f32; 3],
    /// Throat radius, packed into the origin slot's fourth lane.
    waist: f32,
    /// Ray direction.
    dir: [f32; 3],
    /// Radial growth slope, packed into the direction slot's fourth lane.
    flare: f32,
    /// Waist center.
    center: [f32; 3],
    /// Lower ray-interval bound, packed into the center slot's fourth lane.
    t_min: f32,
    /// Axis rim end.
    top: [f32; 3],
    /// Upper ray-interval bound, packed into the top slot's fourth lane.
    t_max: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &RayHyperboloidQuery) -> GpuQuery {
        GpuQuery {
            origin: query.origin,
            waist: query.waist,
            dir: query.dir,
            flare: query.flare,
            center: query.center,
            t_min: query.t_min,
            top: query.top,
            t_max: query.t_max,
        }
    }
}

/// `repr(C)` `std430` layout of one result: the hit flag, ray parameter,
/// front-face flag and a pad word, then one `vec4` slot for the oriented unit
/// normal — `32` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Hit flag (`1` = hit).
    hit: u32,
    /// Nearest valid-hit ray parameter.
    t: f32,
    /// Front-face flag (`1` = outward-facing side struck).
    front_face: u32,
    /// Padding word.
    pad0: u32,
    /// Oriented unit surface normal.
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

/// Maps one kernel `Result` lane back to the host [`RayHyperboloidResult`],
/// unpacking the integer hit flag into the public shape.
fn decode_result(raw: &GpuResult) -> RayHyperboloidResult {
    let hit = u32::from(raw.hit == CODE_HIT);
    RayHyperboloidResult {
        hit,
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

/// A compiled, reusable ray-hyperboloid compute pipeline.
pub struct GpuRayHyperboloid {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayHyperboloid {
    /// Compiles the ray-hyperboloid kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayHyperboloid {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_hyperboloid"),
            source: ShaderSource::Wgsl(RAY_HYPERBOLOID_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_hyperboloid_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_hyperboloid_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_hyperboloid_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayHyperboloid {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`RayHyperboloidResult`]
    /// per input, in order.
    ///
    /// Each result equals the reference `Hyperboloid::intersect` answer to
    /// within the tolerance documented on this module. An empty input returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot
    /// be zero-sized.
    #[must_use]
    pub fn intersect(
        &self,
        ctx: &GpuContext,
        queries: &[RayHyperboloidQuery],
    ) -> Vec<RayHyperboloidResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_hyperboloid_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_hyperboloid_output"),
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
            label: Some("prism_volumetric_ray_hyperboloid_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_hyperboloid_bind_group"),
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
            label: Some("prism_volumetric_ray_hyperboloid_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ray_hyperboloid_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_hyperboloid_pass"),
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
