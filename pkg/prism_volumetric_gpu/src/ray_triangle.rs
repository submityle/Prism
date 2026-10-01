//! `wgpu` compute twin of the closed-form `ray`-triangle `Moller-Trumbore`
//! golden ([`ray_triangle`](prism_render_architecture::particle::ray_triangle),
//! particle design §14, §22).
//!
//! Particle collision against static geometry, decal projection and beam/trace
//! effects all reduce to the same analytic question: where does a `ray` pierce
//! a triangle, what are the surface `barycentric` coordinates there, and which
//! face was struck? The `CPU` golden
//! [`intersect_moller_trumbore`](prism_render_architecture::particle::ray_triangle::intersect_moller_trumbore)
//! owns that branch-free closed form; [`GpuRayTriangle`] is the on-device twin
//! that runs one thread per `(ray, triangle)` pair and reproduces every lane. A
//! passing real-device parity test is therefore direct evidence the ported
//! kernel folds the same cross products, the same guarded determinant division
//! and the same `barycentric` rejections the reference does, not merely that its
//! shader compiles.
//!
//! # What is twinned
//!
//! Each lane reproduces the reference solve guard for guard: `edge1 = v1 - v0`,
//! `edge2 = v2 - v0`, `pvec = dir x edge2`, `det = edge1 . pvec`; a determinant
//! whose magnitude is below [`EPS`] is a *parallel / degenerate* miss (the
//! guarded division never amplifies round-off into a `NaN` or a spurious far
//! hit); otherwise `inv_det = 1 / det`, `u = (origin - v0) . pvec * inv_det`,
//! `qvec = (origin - v0) x edge1`, `v = dir . qvec * inv_det` and
//! `t = edge2 . qvec * inv_det`, accepted only when `0 <= u <= 1`, `v >= 0`,
//! `u + v <= 1` and `t > EPS`. The kernel reports the discrete hit flag, the
//! front-face flag (a positive determinant, matching the reference's
//! `cull_backface` verdict) and the continuous `t`, `u`, `v`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `select`,
//! `+ - * /` plus the hand-rolled `dot` and `cross` — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `sqrt` or optional device feature (the triangle test is
//! polynomial with a single reciprocal), so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of cross products, dot
//! products and one guarded division, so `CPU` and `GPU` evaluate the same
//! closed form in the same associativity. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on `t`, `u`,
//! `v` yet an *exact* match on the discrete hit and front-face flags, whose
//! fixtures are placed clear of every branch tie so both devices fold the
//! identical boolean verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ray_triangle`；
//! 非第三方引擎源码或衍生代码。

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::ray_triangle::{Ray, Vec3};
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

/// The portable core-`WGSL` `ray`-triangle kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`intersect_moller_trumbore`](prism_render_architecture::particle::ray_triangle::intersect_moller_trumbore)
/// branch for branch; see the module documentation for the algorithm.
const RAY_TRIANGLE_WGSL: &str = r#"
// Ray-triangle Moller-Trumbore twin: one thread per (ray, triangle) pair
// reproduces the discrete hit flag, the front-face flag and the barycentric
// (t, u, v) the CPU golden `particle::ray_triangle` reports. It mirrors the
// reference branch for branch, uses only the portable core-WGSL subset
// (abs/select and + - * / plus hand-rolled dot/cross), needs no sqrt and takes
// no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 particle::ray_triangle; 非第三方引擎源码或衍生代码。

// Magnitude below which the determinant is treated as degenerate, so the
// guarded division falls back to a miss instead of amplifying round-off.
// Matches the reference `EPS`.
const EPS: f32 = 1.0e-7;

struct Params {
    // Number of (ray, triangle) pairs in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Ray origin point; a pad lane follows.
    origin: vec3<f32>,
    pad0: f32,
    // Ray direction (not required to be unit length); a pad lane follows.
    dir: vec3<f32>,
    pad1: f32,
    // First triangle vertex; a pad lane follows.
    v0: vec3<f32>,
    pad2: f32,
    // Second triangle vertex; a pad lane follows.
    v1: vec3<f32>,
    pad3: f32,
    // Third triangle vertex; a pad lane follows.
    v2: vec3<f32>,
    pad4: f32,
}

struct Result {
    // 1 when the ray crosses the triangle in front of its origin, else 0.
    hit: u32,
    // 1 when the struck face is the front face (positive determinant), else 0.
    front_face: u32,
    // Ray parameter of the hit (0 on a miss).
    t: f32,
    // Barycentric weight of v1 (0 on a miss).
    u: f32,
    // Barycentric weight of v2 (0 on a miss).
    v: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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
    let origin = q.origin;
    let dir = q.dir;
    let v0 = q.v0;
    let v1 = q.v1;
    let v2 = q.v2;

    var out: Result;
    out.hit = 0u;
    out.front_face = 0u;
    out.t = 0.0;
    out.u = 0.0;
    out.v = 0.0;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let pvec = cross(dir, edge2);
    let det = dot(edge1, pvec);

    // Guard the determinant: a ray in the triangle's plane (|det| < EPS) is a
    // miss, never a division by a near-zero quantity. This mirrors the
    // reference no-cull guard `det.abs() < EPS`.
    if (abs(det) >= EPS) {
        let inv_det = 1.0 / det;
        let tvec = origin - v0;

        let uu = dot(tvec, pvec) * inv_det;
        // Accept only 0 <= u <= 1 (reference `(0.0..=1.0).contains(&u)`).
        if (uu >= 0.0 && uu <= 1.0) {
            let qvec = cross(tvec, edge1);
            let vv = dot(dir, qvec) * inv_det;
            // Accept only v >= 0 and u + v <= 1.
            if (vv >= 0.0 && uu + vv <= 1.0) {
                let tt = dot(edge2, qvec) * inv_det;
                // Accept only a strictly forward crossing (reference `t > EPS`).
                if (tt > EPS) {
                    out.hit = 1u;
                    out.t = tt;
                    out.u = uu;
                    out.v = vv;
                    // Front face is the positive-determinant side, matching the
                    // reference `cull_backface` verdict `det >= EPS`.
                    out.front_face = select(0u, 1u, det >= EPS);
                }
            }
        }
    }

    results[idx] = out;
}
"#;

/// One `ray`-triangle query: a [`Ray`] and the three triangle vertices, the same
/// inputs the reference
/// [`intersect_moller_trumbore`](prism_render_architecture::particle::ray_triangle::intersect_moller_trumbore)
/// consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayTriangleQuery {
    /// The `ray` being tested against the triangle.
    pub ray: Ray,
    /// First triangle vertex (`v0`), the vertex the implied weight `1 - u - v`
    /// multiplies.
    pub v0: Vec3,
    /// Second triangle vertex (`v1`), weighted by `u`.
    pub v1: Vec3,
    /// Third triangle vertex (`v2`), weighted by `v`.
    pub v2: Vec3,
}

impl RayTriangleQuery {
    /// Builds a query from a `ray` and the three triangle vertices.
    #[must_use]
    pub const fn new(ray: Ray, v0: Vec3, v1: Vec3, v2: Vec3) -> RayTriangleQuery {
        RayTriangleQuery { ray, v0, v1, v2 }
    }
}

/// The resolved answer for one `ray`-triangle query, mirroring the reference
/// [`Hit`](prism_render_architecture::particle::ray_triangle::Hit) plus the
/// discrete hit and facing verdicts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayTriangleHit {
    /// Whether the `ray` crosses the triangle strictly in front of its origin
    /// (matching whether the reference returns `Some` with `cull_backface`
    /// disabled).
    pub hit: bool,
    /// Whether the struck face is the front face (a positive determinant),
    /// matching whether the reference still returns `Some` with
    /// `cull_backface` enabled. Meaningful only when `hit` is `true`.
    pub front_face: bool,
    /// Distance along the `ray` (in units of the `ray` direction) to the hit;
    /// `0.0` on a miss. Matches `Hit::t`.
    pub t: f32,
    /// `barycentric` weight of the second vertex `v1`; `0.0` on a miss. Matches
    /// `Hit::u`.
    pub u: f32,
    /// `barycentric` weight of the third vertex `v2`; `0.0` on a miss. Matches
    /// `Hit::v`.
    pub v: f32,
}

/// `repr(C)` `std430` layout of one packed query: five `vec4` slots holding
/// `(origin.xyz, pad)`, `(dir.xyz, pad)`, `(v0.xyz, pad)`, `(v1.xyz, pad)` and
/// `(v2.xyz, pad)` — `80` bytes, each `vec3` on its `16`-byte-aligned slot
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin point.
    origin: [f32; 3],
    /// Padding lane after the origin.
    pad0: f32,
    /// Ray direction.
    dir: [f32; 3],
    /// Padding lane after the direction.
    pad1: f32,
    /// First triangle vertex.
    v0: [f32; 3],
    /// Padding lane after the first vertex.
    pad2: f32,
    /// Second triangle vertex.
    v1: [f32; 3],
    /// Padding lane after the second vertex.
    pad3: f32,
    /// Third triangle vertex.
    v2: [f32; 3],
    /// Padding lane after the third vertex.
    pad4: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &RayTriangleQuery) -> GpuQuery {
        GpuQuery {
            origin: [query.ray.origin.x, query.ray.origin.y, query.ray.origin.z],
            pad0: 0.0,
            dir: [query.ray.dir.x, query.ray.dir.y, query.ray.dir.z],
            pad1: 0.0,
            v0: [query.v0.x, query.v0.y, query.v0.z],
            pad2: 0.0,
            v1: [query.v1.x, query.v1.y, query.v1.z],
            pad3: 0.0,
            v2: [query.v2.x, query.v2.y, query.v2.z],
            pad4: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: the discrete hit and front-face
/// flags, the `t`, `u`, `v` scalars and three pad words — `32` bytes matching
/// the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` when the `ray` crosses the triangle, else `0`.
    hit: u32,
    /// `1` when the struck face is the front face, else `0`.
    front_face: u32,
    /// Ray parameter of the hit.
    t: f32,
    /// `barycentric` weight of `v1`.
    u: f32,
    /// `barycentric` weight of `v2`.
    v: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Uniform parameters for one dispatch: the pair count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of `(ray, triangle)` pairs in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable `ray`-triangle compute pipeline.
pub struct GpuRayTriangle {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayTriangle {
    /// Compiles the `ray`-triangle kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayTriangle {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_triangle"),
            source: ShaderSource::Wgsl(RAY_TRIANGLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_triangle_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_triangle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_triangle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayTriangle {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`RayTriangleHit`] per
    /// input, in order.
    ///
    /// Each result equals the reference
    /// [`intersect_moller_trumbore`](prism_render_architecture::particle::ray_triangle::intersect_moller_trumbore)
    /// answer to within the tolerance documented on this module, with the hit
    /// and front-face flags matched exactly. An empty input returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[RayTriangleQuery]) -> Vec<RayTriangleHit> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_triangle_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_triangle_output"),
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
            label: Some("prism_volumetric_ray_triangle_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_triangle_bind_group"),
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
            label: Some("prism_volumetric_ray_triangle_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ray_triangle_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_triangle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per (ray, triangle) pair, flattened to a 1-D dispatch.
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

/// Decodes one packed [`GpuResult`] into the public [`RayTriangleHit`].
fn decode_result(raw: &GpuResult) -> RayTriangleHit {
    RayTriangleHit {
        hit: raw.hit != 0,
        front_face: raw.front_face != 0,
        t: raw.t,
        u: raw.u,
        v: raw.v,
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
