//! `wgpu` compute twin of the analytic ray vs oriented rectangle
//! (parallelogram) golden
//! [`Rectangle::intersect`](prism_render_architecture::ray_scene::rectangle::Rectangle::intersect).
//!
//! A [`Rectangle`](prism_render_architecture::ray_scene::rectangle::Rectangle)
//! is a center-based parallelogram: the set of points
//! `center + a·axis_u + b·axis_v` with `a, b ∈ [-1, 1]`, where `axis_u` and
//! `axis_v` are *half-edge* vectors (each spans half the corresponding side).
//! The plane normal is derived as `axis_u × axis_v`. The intersection solves
//! the single ray/plane equation `t = (center - origin) · n / (dir · n)` and
//! then solves the in-plane `2×2` parallelogram-coordinate system to accept or
//! reject the hit. Every step is add/sub/mul/div/`sqrt` and comparisons, so it
//! is bit-reproducible on the `GPU` and free of any transcendental call.
//! [`GpuRayRectangle`] is the on-device twin that runs one thread per query and
//! reproduces every lane; a passing real-device parity test is therefore direct
//! evidence the ported kernel folds the same plane solve, the same reciprocal
//! basis containment test and the same incident-oriented normal the reference
//! does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces
//! [`Rectangle::intersect`](prism_render_architecture::ray_scene::rectangle::Rectangle::intersect)
//! guard-for-guard: a zero-area rectangle (parallel or zero-length edges,
//! `nn = |axis_u × axis_v|² <= 0`) never reports a hit; a ray parallel to the
//! plane (`|dir · n|` below a magnitude floor) never reports a hit; the plane
//! parameter `t` outside `[t_min, t_max]` is a miss; the in-plane coordinates
//! `a` and `b` are solved via the reciprocal basis (the system determinant
//! equals `nn` by Lagrange's identity) and rejected unless both lie in
//! `[-1, 1]`; and the reported unit normal is flipped to oppose the incident
//! ray, with `front_face` set when the ray struck the side the derived normal
//! faces (`dir · n < 0`). A miss writes an all-zero result lane.
//!
//! # What stays on the host
//!
//! Only the per-query closed form is twinned. The reference `RectangleBvh`
//! acceleration structure (variable-length node and primitive vectors, slab
//! traversal) is not a per-lane kernel and stays on the host; the twin tests
//! one `(ray, rectangle)` pair per thread. The reference `primitive` id is not
//! carried through the kernel because the query layout omits it, so the twin
//! reports only the geometric verdict (`hit`, `t`, `normal`, `front_face`).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `sqrt`, `select`, `+ - * /` and comparisons — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `round`, `smoothstep`, `cbrt`, no `u64`/`i64`/`u16`/`i16`/`f64`
//! and no optional device feature. It therefore runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of one plane division, a
//! reciprocal-basis `2×2` solve and one `sqrt`, so `CPU` and `GPU` evaluate the
//! same closed form in the same associativity. They are not bit-exact: a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits by a few units in the last place. The parity test
//! therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on
//! the continuous `t` and `normal` values yet an *exact* match on the discrete
//! `hit` and `front_face` flags, which are routed through the same magnitude
//! guards the reference uses so a query placed clear of a boundary folds the
//! identical verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::rectangle`；
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Discrete hit code written by the kernel for a lane that intersects: matches
/// the host `== 1` decode in [`decode_result`]. A direct `f32` equality is
/// forbidden, so the kernel emits an integer flag rather than a sentinel float.
const CODE_HIT: u32 = 1;

/// The portable core-`WGSL` ray/rectangle kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` golden
/// [`Rectangle::intersect`](prism_render_architecture::ray_scene::rectangle::Rectangle::intersect)
/// guard-for-guard; see the module documentation for the algorithm.
const RAY_RECTANGLE_WGSL: &str = r#"
// Ray vs oriented rectangle (parallelogram) twin: one thread per (ray,
// rectangle) query solves the single plane crossing
// t = (center - origin) . n / (dir . n), then solves the in-plane 2x2
// parallelogram coordinates a, b via the reciprocal basis and accepts the hit
// only when both lie in [-1, 1]. It mirrors the CPU golden
// `ray_scene::rectangle::Rectangle::intersect` guard for guard, uses only the
// portable core-WGSL subset (abs/min/max/sqrt/select, + - * / and comparisons),
// and takes no optional feature, so it runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::rectangle；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 80-byte std430 stride matching the host `GpuQuery`: the ray origin
// (with `t_min` in `w`), the ray direction (with `t_max` in `w`) and the
// rectangle center and two half-edge vectors, each padded to a vec4 so the
// storage array needs no manual vec3 alignment arithmetic.
struct Query {
    origin: vec4<f32>,
    dir: vec4<f32>,
    center: vec4<f32>,
    axis_u: vec4<f32>,
    axis_v: vec4<f32>,
}

// One result. 32-byte std430 stride matching the host `GpuResult`: the hit flag
// and front-face flag as 0u/1u, the plane parameter and the incident-oriented
// unit normal, plus two pad words.
struct Result {
    hit: u32,
    front_face: u32,
    t: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Magnitude floor guarding the ray/plane `1 / denom` division: a direction
// whose component along the plane normal has magnitude at or below this floor
// runs parallel to the plane and can never cross it. A direct f32 `==`/`!=` is
// forbidden, so the parallel test compares the magnitude against this floor
// instead of exact zero. It mirrors the reference `denom == 0.0` reject on
// well-conditioned queries (any larger near-parallel denom drives `t` out of
// `[t_min, t_max]`, a miss in both paths).
const DENOM_EPS: f32 = 1.0e-12;

// A miss lane: every field zeroed, matching the host `decode_result` which maps
// a zero `hit` to `None`-equivalent defaults.
fn miss_result() -> Result {
    var r: Result;
    r.hit = 0u;
    r.front_face = 0u;
    r.t = 0.0;
    r.nx = 0.0;
    r.ny = 0.0;
    r.nz = 0.0;
    r.pad0 = 0u;
    r.pad1 = 0u;
    return r;
}

// The shared ray/rectangle solve, mirroring the reference `Rectangle::intersect`
// guard for guard and returning a zeroed lane on any miss.
fn intersect(
    origin: vec3<f32>,
    dir: vec3<f32>,
    t_min: f32,
    t_max: f32,
    center: vec3<f32>,
    axis_u: vec3<f32>,
    axis_v: vec3<f32>,
) -> Result {
    // A zero-area (parallel or zero-length edges) rectangle has no plane.
    let n = cross(axis_u, axis_v);
    let nn = dot(n, n);
    if (nn <= 0.0) {
        return miss_result();
    }
    // `|denom|` at or below the floor means the ray runs parallel to the plane
    // (or has zero length) and can never cross it.
    let denom = dot(dir, n);
    if (abs(denom) <= DENOM_EPS) {
        return miss_result();
    }
    let oc = center - origin;
    let t = dot(oc, n) / denom;
    if (t < t_min || t > t_max) {
        return miss_result();
    }
    // Solve `p = a·u + b·v` in the plane via the reciprocal basis. The
    // determinant `uu·vv - uv²` equals `nn` (Lagrange identity) and is strictly
    // positive here, so the divide is well defined.
    let p = (origin + dir * t) - center;
    let uu = dot(axis_u, axis_u);
    let vv = dot(axis_v, axis_v);
    let uv = dot(axis_u, axis_v);
    let pu = dot(p, axis_u);
    let pv = dot(p, axis_v);
    let a = (vv * pu - uv * pv) / nn;
    let b = (uu * pv - uv * pu) / nn;
    if (a < -1.0 || a > 1.0 || b < -1.0 || b > 1.0) {
        return miss_result();
    }
    let inv = 1.0 / sqrt(nn);
    let unit = n * inv;
    let front = denom < 0.0;
    var nrm = unit;
    if (!front) {
        nrm = unit * -1.0;
    }
    var r: Result;
    r.hit = 1u;
    r.front_face = select(0u, 1u, front);
    r.t = t;
    r.nx = nrm.x;
    r.ny = nrm.y;
    r.nz = nrm.z;
    r.pad0 = 0u;
    r.pad1 = 0u;
    return r;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    results[idx] = intersect(
        q.origin.xyz,
        q.dir.xyz,
        q.origin.w,
        q.dir.w,
        q.center.xyz,
        q.axis_u.xyz,
        q.axis_v.xyz,
    );
}
"#;

/// One ray vs rectangle query: the ray (origin, direction and the `[t_min,
/// t_max]` valid parameter interval) and the oriented rectangle to test it
/// against.
///
/// Mirrors a single reference
/// [`Rectangle::intersect`](prism_render_architecture::ray_scene::rectangle::Rectangle::intersect)
/// call. `axis_u` and `axis_v` are the rectangle's *half-edge* vectors (so the
/// full side lengths are `2·|axis_u|` and `2·|axis_v|`). Carrying the rectangle
/// per query lets one dispatch mix rays against many distinct rectangles.
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayRectangleQuery {
    /// Ray origin in world space.
    pub origin: [f32; 3],
    /// Ray direction (need not be unit length).
    pub dir: [f32; 3],
    /// Lower bound of the accepted plane parameter interval.
    pub t_min: f32,
    /// Upper bound of the accepted plane parameter interval.
    pub t_max: f32,
    /// Center of the rectangle (a point on its plane).
    pub center: [f32; 3],
    /// Half-edge vector along the first side.
    pub axis_u: [f32; 3],
    /// Half-edge vector along the second side.
    pub axis_v: [f32; 3],
}

impl RayRectangleQuery {
    /// Builds one ray/rectangle query from its components.
    #[must_use]
    pub fn new(
        origin: [f32; 3],
        dir: [f32; 3],
        t_min: f32,
        t_max: f32,
        center: [f32; 3],
        axis_u: [f32; 3],
        axis_v: [f32; 3],
    ) -> RayRectangleQuery {
        RayRectangleQuery {
            origin,
            dir,
            t_min,
            t_max,
            center,
            axis_u,
            axis_v,
        }
    }
}

/// The resolved verdict for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `hit` is the containment reading (whether the ray crosses the plane inside
/// the parallelogram and within `[t_min, t_max]`). `t` and `normal` are
/// meaningful only when `hit` is `true` (otherwise both are zeroed); `normal`
/// is the unit surface normal oriented against the incident ray, and
/// `front_face` is `true` when the ray struck the side the derived normal
/// faces (a ray arriving from behind reports `front_face == false` with a
/// flipped normal). Derives only [`PartialEq`] (no `Eq`/`Hash`) because it
/// holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayRectangleResult {
    /// Whether the ray strikes the rectangle inside `[t_min, t_max]`.
    pub hit: bool,
    /// The plane parameter at the hit; meaningful only when `hit`.
    pub t: f32,
    /// The incident-oriented unit normal; meaningful only when `hit`.
    pub normal: [f32; 3],
    /// Whether the ray struck the side the derived normal faces; meaningful
    /// only when `hit`.
    pub front_face: bool,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`RAY_RECTANGLE_WGSL`]: the query count and three pad words —
/// `16` bytes, each field at the uniform offset the shader expects.
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

/// One query as uploaded. `80`-byte `std430` stride matching `Query` in the
/// shader: the ray origin (with `t_min` in `w`), the ray direction (with
/// `t_max` in `w`) and the rectangle center and two half-edge vectors, each
/// padded to a `vec4` lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin in `xyz`; the `w` lane carries `t_min`.
    origin: [f32; 4],
    /// Ray direction in `xyz`; the `w` lane carries `t_max`.
    dir: [f32; 4],
    /// Rectangle center in `xyz`; the `w` lane is unused padding.
    center: [f32; 4],
    /// First half-edge vector in `xyz`; the `w` lane is unused padding.
    axis_u: [f32; 4],
    /// Second half-edge vector in `xyz`; the `w` lane is unused padding.
    axis_v: [f32; 4],
}

impl GpuQuery {
    /// Packs a [`RayRectangleQuery`] into the `std430` upload layout.
    fn from_query(query: &RayRectangleQuery) -> GpuQuery {
        let o = query.origin;
        let d = query.dir;
        let c = query.center;
        let u = query.axis_u;
        let v = query.axis_v;
        GpuQuery {
            origin: [o[0], o[1], o[2], query.t_min],
            dir: [d[0], d[1], d[2], query.t_max],
            center: [c[0], c[1], c[2], 0.0],
            axis_u: [u[0], u[1], u[2], 0.0],
            axis_v: [v[0], v[1], v[2], 0.0],
        }
    }
}

/// One result as read back. `32`-byte `std430` stride matching `Result` in the
/// shader: the hit and front-face flags as `0`/`1`, the plane parameter, the
/// incident-oriented unit normal and two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Hit flag (`1` = hit).
    hit: u32,
    /// Front-face flag (`1` = struck the derived-normal side).
    front_face: u32,
    /// Plane parameter at the hit.
    t: f32,
    /// Normal `x`.
    nx: f32,
    /// Normal `y`.
    ny: f32,
    /// Normal `z`.
    nz: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// Maps one kernel `Result` lane back to the host [`RayRectangleResult`].
fn decode_result(raw: &GpuResult) -> RayRectangleResult {
    RayRectangleResult {
        hit: raw.hit == CODE_HIT,
        t: raw.t,
        normal: [raw.nx, raw.ny, raw.nz],
        front_face: raw.front_face == CODE_HIT,
    }
}

/// A compiled, reusable ray/rectangle intersection pipeline.
pub struct GpuRayRectangle {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayRectangle {
    /// Compiles the ray/rectangle kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayRectangle {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_rectangle"),
            source: ShaderSource::Wgsl(RAY_RECTANGLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_rectangle_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_rectangle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_rectangle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayRectangle {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries`, returning one [`RayRectangleResult`]
    /// per query in input order.
    ///
    /// The returned result for query `q` mirrors
    /// [`Rectangle::intersect`](prism_render_architecture::ray_scene::rectangle::Rectangle::intersect)
    /// evaluated on `q`'s rectangle and ray. An empty `queries` slice yields an
    /// empty result — storage buffers cannot be zero-sized, so it is handled by
    /// an early return before any dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[RayRectangleQuery]) -> Vec<RayRectangleResult> {
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
            label: Some("prism_volumetric_ray_rectangle_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_rectangle_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_rectangle_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_rectangle_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_rectangle_bind_group"),
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
            label: Some("prism_volumetric_ray_rectangle_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_rectangle_pass"),
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
