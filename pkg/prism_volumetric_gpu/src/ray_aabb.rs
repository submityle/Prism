//! `wgpu` compute twin of the analytic ray vs *axis-aligned bounding box*
//! (`AABB`) slab-method golden
//! ([`ray_aabb`](prism_render_architecture::particle::ray_aabb), design §10,
//! §14).
//!
//! The particle subsystem's picking, culling-probe and analytic-primitive
//! raytrace contracts need, per `(ray, box)` pair, the ordered slab chord
//! `[t_enter, t_exit]`, the two "does it hit" readings (the *infinite line* and
//! the *forward half-line*) and the single first *visible* forward crossing.
//! The `CPU` golden
//! [`Aabb::intersect_line`](prism_render_architecture::particle::ray_aabb::Aabb::intersect_line),
//! [`Aabb::intersect_ray`](prism_render_architecture::particle::ray_aabb::Aabb::intersect_ray)
//! and
//! [`Aabb::first_hit_t`](prism_render_architecture::particle::ray_aabb::Aabb::first_hit_t)
//! own that math; [`GpuRayAabb`] is the on-device twin that runs one thread per
//! query and reproduces every lane. A passing real-device parity test is
//! therefore direct evidence the ported kernel folds the same three per-axis
//! divisions, the same division guards and the same running interval
//! intersection the reference does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces the shared
//! [`slab_chord`](prism_render_architecture::particle::ray_aabb) solve
//! guard-for-guard: a zero-length direction is rejected first (`dot(dir, dir)`
//! below [`EPS`](prism_render_architecture::particle::ray_aabb::EPS) squared);
//! an axis whose direction component magnitude is below `EPS` is *parallel* and
//! divides nothing — it contributes a miss when the origin is outside that slab
//! and the sentinel interval `(-inf, +inf)` otherwise; every other axis forms
//! the reciprocal `1 / d` and folds its `[t_near, t_far]` into the running
//! `[t_enter, t_exit]` with `min`/`max`. The chord is reported when
//! `t_enter <= t_exit` (compared with `<=`, never with `==`). From that one
//! chord the kernel derives the infinite-line hit flag, the forward-ray hit
//! flag (`t_exit >= 0`) and the first visible forward crossing (`t_enter` when
//! it is in front, otherwise `t_exit`), each matching the reference
//! `intersect_line`, `intersect_ray`, `first_hit_t`, `intersects_line` and
//! `intersects_ray`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `+ - * /` and one `bitcast` to materialize the `IEEE`-754 infinity sentinel
//! that mirrors the reference's `f32::INFINITY` running bounds — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `sqrt` or optional device feature (the box faces
//! are flat, so there is no quadratic). It therefore runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of three guarded divisions
//! and a running interval intersection, so `CPU` and `GPU` evaluate the same
//! closed form in the same associativity. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the chord
//! and crossing values yet an *exact* match on the discrete hit flags, which
//! are routed through the same `EPS` magnitude band the reference uses so a
//! query placed clear of a face boundary folds the identical boolean verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ray_aabb`；
//! standard slab-method ray/`AABB` intersection plus `wgpu` compute dispatch;
//! no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::ray_aabb::{Aabb, Ray, RayAabbHit};
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

/// Discrete hit code written by the kernel for a lane that intersects: matches
/// the host `== 1` decode in [`decode_result`]. A direct `f32` equality is
/// forbidden, so the kernel emits an integer flag rather than a sentinel float.
const CODE_HIT: u32 = 1;

/// The portable core-`WGSL` ray/`AABB` kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` golden
/// [`slab_chord`](prism_render_architecture::particle::ray_aabb) and its three
/// public readings guard-for-guard; see the module documentation for the
/// algorithm.
const RAY_AABB_WGSL: &str = r#"
// Ray vs AABB slab-method twin: one thread per (ray, box) query folds the three
// per-axis slab intervals into the ordered chord [t_enter, t_exit], then writes
// the infinite-line hit flag, the forward-ray hit flag (t_exit >= 0) and the
// first visible forward crossing (or a +inf sentinel on a forward miss). It
// mirrors the CPU golden `particle::ray_aabb` division guard for guard, uses
// only the portable core-WGSL subset (abs/min/max, + - * / and one bitcast for
// the infinity sentinel), and takes no optional feature, so it runs unmodified
// on Metal, Vulkan and DX12.
//
// Provenance: standard slab-method ray/AABB intersection; no third-party engine
// source or derived code.

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 64-byte std430 stride matching the host `GpuQuery`: the ray origin
// and direction and the box min and max corners, each padded to a vec4 so the
// storage array needs no manual vec3 alignment arithmetic.
struct Query {
    origin: vec4<f32>,
    dir: vec4<f32>,
    bmin: vec4<f32>,
    bmax: vec4<f32>,
}

// One result. 32-byte std430 stride matching the host `GpuResult`: the two hit
// flags as 0u/1u, the ordered chord endpoints and the first visible forward
// crossing (a +inf sentinel on a forward miss), plus three pad words.
struct Result {
    line_hit: u32,
    ray_hit: u32,
    t_enter: f32,
    t_exit: f32,
    first_hit_t: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Magnitude floor guarding the parallel-slab `1 / d` division and classifying a
// direction component as parallel, matching the reference `EPS`. A direct f32
// `==`/`!=` is forbidden, so the zero-direction and parallel tests compare
// magnitudes against this floor instead of exact zero.
const EPS: f32 = 1.0e-6;

// The resolved slab chord: whether the three per-axis intervals overlap and the
// ordered entry/exit parameters when they do.
struct Chord {
    hit: bool,
    t_enter: f32,
    t_exit: f32,
}

// The shared slab solve: intersects the three per-axis parameter intervals of
// the infinite line through the ray and returns the ordered chord when they
// overlap. Mirrors the reference `slab_chord` guard for guard: a zero-length
// direction is rejected first (so a point never spans the sentinel interval);
// an axis whose direction component magnitude is below `EPS` is parallel and
// divides nothing, contributing a miss when the origin lies outside that slab
// and the infinite sentinel interval (which the running min/max then skips)
// otherwise.
fn slab_chord(origin: vec3<f32>, dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> Chord {
    var out: Chord;
    out.hit = false;
    out.t_enter = 0.0;
    out.t_exit = 0.0;

    let dd = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
    if (dd <= EPS * EPS) {
        return out;
    }

    // IEEE-754 +/-infinity sentinels, mirroring the reference's `f32::INFINITY`
    // running bounds exactly; built by bitcast because WGSL has no inf literal.
    let pos_inf = bitcast<f32>(0x7f800000u);
    let neg_inf = bitcast<f32>(0xff800000u);
    var t_enter = neg_inf;
    var t_exit = pos_inf;

    // Axis x.
    if (abs(dir.x) < EPS) {
        if (origin.x < bmin.x || origin.x > bmax.x) {
            return out;
        }
    } else {
        let inv = 1.0 / dir.x;
        let t1 = (bmin.x - origin.x) * inv;
        let t2 = (bmax.x - origin.x) * inv;
        let t_near = min(t1, t2);
        let t_far = max(t1, t2);
        t_enter = max(t_enter, t_near);
        t_exit = min(t_exit, t_far);
    }
    // Axis y.
    if (abs(dir.y) < EPS) {
        if (origin.y < bmin.y || origin.y > bmax.y) {
            return out;
        }
    } else {
        let inv = 1.0 / dir.y;
        let t1 = (bmin.y - origin.y) * inv;
        let t2 = (bmax.y - origin.y) * inv;
        let t_near = min(t1, t2);
        let t_far = max(t1, t2);
        t_enter = max(t_enter, t_near);
        t_exit = min(t_exit, t_far);
    }
    // Axis z.
    if (abs(dir.z) < EPS) {
        if (origin.z < bmin.z || origin.z > bmax.z) {
            return out;
        }
    } else {
        let inv = 1.0 / dir.z;
        let t1 = (bmin.z - origin.z) * inv;
        let t2 = (bmax.z - origin.z) * inv;
        let t_near = min(t1, t2);
        let t_far = max(t1, t2);
        t_enter = max(t_enter, t_near);
        t_exit = min(t_exit, t_far);
    }

    if (t_enter <= t_exit) {
        out.hit = true;
        out.t_enter = t_enter;
        out.t_exit = t_exit;
    }
    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let origin = queries[idx].origin.xyz;
    let dir = queries[idx].dir.xyz;
    let bmin = queries[idx].bmin.xyz;
    let bmax = queries[idx].bmax.xyz;

    let chord = slab_chord(origin, dir, bmin, bmax);
    let pos_inf = bitcast<f32>(0x7f800000u);

    var res: Result;
    res.pad0 = 0u;
    res.pad1 = 0u;
    res.pad2 = 0u;

    if (chord.hit) {
        res.line_hit = 1u;
        res.t_enter = chord.t_enter;
        res.t_exit = chord.t_exit;
    } else {
        res.line_hit = 0u;
        res.t_enter = pos_inf;
        res.t_exit = pos_inf;
    }

    // Forward-ray reading: visible only when some of the chord lies at or ahead
    // of the origin (t_exit >= 0), matching the reference `intersect_ray`.
    let forward = chord.hit && chord.t_exit >= 0.0;
    if (forward) {
        res.ray_hit = 1u;
        // first_visible_t: the entry face when in front, else the exit face
        // (the origin is inside the box), matching `RayAabbHit::first_visible_t`.
        var first = chord.t_exit;
        if (chord.t_enter >= 0.0) {
            first = chord.t_enter;
        }
        res.first_hit_t = first;
    } else {
        res.ray_hit = 0u;
        res.first_hit_t = pos_inf;
    }

    results[idx] = res;
}
"#;

/// One ray vs `AABB` query: the ray and the box to test it against.
///
/// Mirrors a single reference
/// [`Aabb::intersect_line`](prism_render_architecture::particle::ray_aabb::Aabb::intersect_line)
/// / [`Aabb::intersect_ray`](prism_render_architecture::particle::ray_aabb::Aabb::intersect_ray)
/// call. Carrying the box per query lets one dispatch mix rays against many
/// distinct boxes. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds
/// `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayAabbQuery {
    /// The ray whose infinite line and forward half-line are both tested.
    pub ray: Ray,
    /// The axis-aligned box the ray is tested against.
    pub aabb: Aabb,
}

/// The resolved verdict for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `line_hit` is the *infinite-line* reading
/// ([`Aabb::intersects_line`](prism_render_architecture::particle::ray_aabb::Aabb::intersects_line))
/// and `ray_hit` the *forward-ray* reading
/// ([`Aabb::intersects_ray`](prism_render_architecture::particle::ray_aabb::Aabb::intersects_ray)).
/// `t_enter` and `t_exit` carry the ordered chord and are meaningful only when
/// `line_hit` is `true`; `first_hit_t` carries the first visible forward
/// crossing and is meaningful only when `ray_hit` is `true` (otherwise both
/// hold a `+inf` sentinel). Derives only [`PartialEq`] (no `Eq`/`Hash`) because
/// it holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayAabbResult {
    /// Whether the *infinite line* through the ray crosses the box.
    pub line_hit: bool,
    /// Whether the *forward half-line* (`t >= 0`) crosses the box.
    pub ray_hit: bool,
    /// The chord entry parameter; meaningful only when `line_hit`.
    pub t_enter: f32,
    /// The chord exit parameter; meaningful only when `line_hit`.
    pub t_exit: f32,
    /// The first visible forward crossing; meaningful only when `ray_hit`.
    pub first_hit_t: f32,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`RAY_AABB_WGSL`]: the query count and three pad words — `16`
/// bytes, each field at the uniform offset the shader expects.
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

/// One query as uploaded. `64`-byte `std430` stride matching `Query` in the
/// shader: the ray origin and direction and the box min and max corners, each
/// padded to a `vec4` lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin in `xyz`; the `w` lane is unused padding.
    origin: [f32; 4],
    /// Ray direction in `xyz`; the `w` lane is unused padding.
    dir: [f32; 4],
    /// Box `min` corner in `xyz`; the `w` lane is unused padding.
    bmin: [f32; 4],
    /// Box `max` corner in `xyz`; the `w` lane is unused padding.
    bmax: [f32; 4],
}

impl GpuQuery {
    /// Packs a [`RayAabbQuery`] into the `std430` upload layout.
    fn from_query(query: &RayAabbQuery) -> GpuQuery {
        let o = query.ray.origin;
        let d = query.ray.dir;
        let lo = query.aabb.min;
        let hi = query.aabb.max;
        GpuQuery {
            origin: [o.x, o.y, o.z, 0.0],
            dir: [d.x, d.y, d.z, 0.0],
            bmin: [lo.x, lo.y, lo.z, 0.0],
            bmax: [hi.x, hi.y, hi.z, 0.0],
        }
    }
}

/// One result as read back. `32`-byte `std430` stride matching `Result` in the
/// shader: the two hit flags as `0`/`1`, the ordered chord endpoints, the first
/// visible forward crossing and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Infinite-line hit flag (`1` = hit).
    line_hit: u32,
    /// Forward-ray hit flag (`1` = hit).
    ray_hit: u32,
    /// Chord entry parameter.
    t_enter: f32,
    /// Chord exit parameter.
    t_exit: f32,
    /// First visible forward crossing (or a `+inf` sentinel on a forward miss).
    first_hit_t: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Maps one kernel `Result` lane back to the host [`RayAabbResult`].
fn decode_result(raw: &GpuResult) -> RayAabbResult {
    RayAabbResult {
        line_hit: raw.line_hit == CODE_HIT,
        ray_hit: raw.ray_hit == CODE_HIT,
        t_enter: raw.t_enter,
        t_exit: raw.t_exit,
        first_hit_t: raw.first_hit_t,
    }
}

/// A compiled, reusable ray/`AABB` slab-test pipeline.
pub struct GpuRayAabb {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayAabb {
    /// Compiles the ray/`AABB` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayAabb {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_aabb"),
            source: ShaderSource::Wgsl(RAY_AABB_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_aabb_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_aabb_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_aabb_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayAabb {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries`, returning one [`RayAabbResult`] per
    /// query in input order.
    ///
    /// The returned result for query `q` mirrors
    /// [`Aabb::intersect_line`](prism_render_architecture::particle::ray_aabb::Aabb::intersect_line),
    /// [`Aabb::intersect_ray`](prism_render_architecture::particle::ray_aabb::Aabb::intersect_ray)
    /// and
    /// [`Aabb::first_hit_t`](prism_render_architecture::particle::ray_aabb::Aabb::first_hit_t)
    /// evaluated on `q.aabb` and `q.ray`. An empty `queries` slice yields an
    /// empty result — storage buffers cannot be zero-sized, so it is handled by
    /// an early return before any dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[RayAabbQuery]) -> Vec<RayAabbResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_aabb_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_aabb_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_aabb_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_aabb_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_aabb_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ray_aabb_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_aabb_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
    }
}

/// The `CPU` golden verdict for one query, dispatching to the reference entry
/// points so callers (and the parity test) can pin the twin lane for lane.
///
/// Returns the infinite-line and forward-ray hit flags, the ordered chord (as
/// an [`Option`] because the reference returns `None` on a miss) and the first
/// visible forward crossing (`None` on a forward miss).
#[must_use]
pub fn cpu_reference(query: &RayAabbQuery) -> (bool, bool, Option<RayAabbHit>, Option<f32>) {
    let line = query.aabb.intersect_line(query.ray);
    let fwd = query.aabb.intersect_ray(query.ray);
    let first = query.aabb.first_hit_t(query.ray);
    (line.is_some(), fwd.is_some(), line, first)
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
