//! `wgpu` compute twin of the triangle surface-point and normal reconstruction
//! ([`barycentric_to_point`](prism_render_architecture::particle::ray_triangle::barycentric_to_point)
//! and
//! [`triangle_normal`](prism_render_architecture::particle::ray_triangle::triangle_normal)).
//!
//! After a ray-triangle intersection reports barycentric coordinates `(u, v)`,
//! the surface point is recovered as `w*v0 + u*v1 + v*v2` with `w = 1 - u - v`,
//! and the triangle's shading normal is the normalized right-handed cross
//! product `(v1 - v0) × (v2 - v0)`. Both are pure closed-form vector algebra —
//! multiply-add plus one guarded normalization — so they port to the device
//! unchanged, and a passing real-device parity run is direct evidence the
//! ported kernel folds the same arithmetic the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! One thread serves one query. It reproduces both golden functions exactly:
//! the barycentric point reconstruction, and the normal with its degenerate
//! guard — a triangle whose cross-product length falls below `EPS = 1.0e-7`
//! (zero-area or collinear) yields the exact zero vector rather than a `NaN`
//! from dividing by a near-zero length.
//!
//! # What stays on the host
//!
//! The intersection test that produces `(u, v)` and the vertex gather are
//! upstream responsibilities; this twin only reconstructs the point and normal
//! from the supplied vertices and coordinates.
//!
//! # Correctness model
//!
//! The point is a continuous `f32` triple built from multiply-adds, compared
//! with `abs <= 1e-5 || rel <= 1e-5`. The normal passes through a `sqrt` and a
//! reciprocal, so it is compared with the slightly looser
//! `abs <= 1e-5 || rel <= 1e-4`. A degenerate triangle's normal is the exact
//! zero vector on both sides and is asserted directly.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — multiply-add, a
//! cross product, one `sqrt`, one comparison, and a `select` — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`,
//! no `round`, and no `cbrt`. Each thread performs a bounded, branch-free
//! sequence, so the kernel provably terminates. No optional device feature is
//! required, so it runs unmodified across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ray_triangle`；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup. The reconstruction is a
/// single-thread-per-element kernel, so a one-dimensional dispatch of this
/// width keeps every lane busy on real hardware.
const WORKGROUP_SIZE: u32 = 64;

/// The inlined `WGSL` twin of the triangle reconstruction: one thread per
/// query, computing the barycentric point and the guarded unit normal.
const RAY_TRIANGLE_RECONSTRUCT_WGSL: &str = r#"
// Twin of particle::ray_triangle::{barycentric_to_point, triangle_normal}.
// One thread per query:
//   w     = 1 - u - v
//   point = v0*w + v1*u + v2*v
//   n     = cross(v1 - v0, v2 - v0)
//   len   = sqrt(dot(n, n)); normal = (len < EPS) ? 0 : n / len
// Multiply-add, one cross product, one sqrt; no transcendental.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::ray_triangle；无第三方引擎源码或衍生代码。

// Degenerate-length threshold matching ray_triangle::EPS.
const EPS: f32 = 1.0e-7;

struct Params {
    // Number of queries in the storage arrays; threads past this stop.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Vertex 0.
    v0x: f32,
    v0y: f32,
    v0z: f32,
    // Vertex 1.
    v1x: f32,
    v1y: f32,
    v1z: f32,
    // Vertex 2.
    v2x: f32,
    v2y: f32,
    v2z: f32,
    // Barycentric coordinate u (weight on v1).
    bu: f32,
    // Barycentric coordinate v (weight on v2).
    bv: f32,
}

struct Reconstructed {
    // Reconstructed surface point.
    px: f32,
    py: f32,
    pz: f32,
    // Unit surface normal (zero vector when degenerate).
    nx: f32,
    ny: f32,
    nz: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Reconstructed>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    let v0 = vec3<f32>(q.v0x, q.v0y, q.v0z);
    let v1 = vec3<f32>(q.v1x, q.v1y, q.v1z);
    let v2 = vec3<f32>(q.v2x, q.v2y, q.v2z);

    let w = 1.0 - q.bu - q.bv;
    let point = v0 * w + v1 * q.bu + v2 * q.bv;

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let n = cross(edge1, edge2);
    let len = sqrt(dot(n, n));
    let unit = n * (1.0 / len);
    let normal = select(unit, vec3<f32>(0.0, 0.0, 0.0), len < EPS);

    results[idx].px = point.x;
    results[idx].py = point.y;
    results[idx].pz = point.z;
    results[idx].nx = normal.x;
    results[idx].ny = normal.y;
    results[idx].nz = normal.z;
}
"#;

/// Uniform parameters for one dispatch: the count plus three pad words to fill
/// a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RAY_TRIANGLE_RECONSTRUCT_WGSL`].
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
/// the three vertices and the two barycentric coordinates (alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Vertex `0`, component `x`.
    v0x: f32,
    /// Vertex `0`, component `y`.
    v0y: f32,
    /// Vertex `0`, component `z`.
    v0z: f32,
    /// Vertex `1`, component `x`.
    v1x: f32,
    /// Vertex `1`, component `y`.
    v1y: f32,
    /// Vertex `1`, component `z`.
    v1z: f32,
    /// Vertex `2`, component `x`.
    v2x: f32,
    /// Vertex `2`, component `y`.
    v2y: f32,
    /// Vertex `2`, component `z`.
    v2z: f32,
    /// Barycentric coordinate `u` (weight on `v1`).
    bu: f32,
    /// Barycentric coordinate `v` (weight on `v2`).
    bv: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL`
/// `Reconstructed` struct: the surface point and the unit normal (alignment
/// `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Reconstructed point, component `x`.
    px: f32,
    /// Reconstructed point, component `y`.
    py: f32,
    /// Reconstructed point, component `z`.
    pz: f32,
    /// Unit normal, component `x`.
    nx: f32,
    /// Unit normal, component `y`.
    ny: f32,
    /// Unit normal, component `z`.
    nz: f32,
}

/// One triangle reconstruction query to run on the device: the three vertices
/// and the barycentric coordinates `(u, v)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayTriangleReconstructQuery {
    /// Vertex `v0`, as `[x, y, z]`.
    pub v0: [f32; 3],
    /// Vertex `v1`, as `[x, y, z]`.
    pub v1: [f32; 3],
    /// Vertex `v2`, as `[x, y, z]`.
    pub v2: [f32; 3],
    /// Barycentric coordinate `u` (weight on `v1`).
    pub u: f32,
    /// Barycentric coordinate `v` (weight on `v2`).
    pub v: f32,
}

/// One triangle reconstruction result: the surface point and the unit normal
/// (the zero vector when the triangle is degenerate).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayTriangleReconstructResult {
    /// Reconstructed surface point, as `[x, y, z]`.
    pub point: [f32; 3],
    /// Unit surface normal, as `[x, y, z]` (zero vector when degenerate).
    pub normal: [f32; 3],
}

/// Encodes one [`RayTriangleReconstructQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &RayTriangleReconstructQuery) -> GpuQuery {
    GpuQuery {
        v0x: q.v0[0],
        v0y: q.v0[1],
        v0z: q.v0[2],
        v1x: q.v1[0],
        v1y: q.v1[1],
        v1z: q.v1[2],
        v2x: q.v2[0],
        v2y: q.v2[1],
        v2z: q.v2[2],
        bu: q.u,
        bv: q.v,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`RayTriangleReconstructResult`].
fn decode_result(raw: &GpuResult) -> RayTriangleReconstructResult {
    RayTriangleReconstructResult {
        point: [raw.px, raw.py, raw.pz],
        normal: [raw.nx, raw.ny, raw.nz],
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

/// A compiled, reusable triangle reconstruction compute pipeline, twinning the
/// `CPU` golden
/// [`barycentric_to_point`](prism_render_architecture::particle::ray_triangle::barycentric_to_point)
/// and
/// [`triangle_normal`](prism_render_architecture::particle::ray_triangle::triangle_normal).
pub struct GpuRayTriangleReconstruct {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRayTriangleReconstruct {
    /// Compiles the triangle reconstruction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayTriangleReconstruct {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ray_triangle_reconstruct"),
            source: ShaderSource::Wgsl(RAY_TRIANGLE_RECONSTRUCT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ray_triangle_reconstruct_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ray_triangle_reconstruct_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ray_triangle_reconstruct_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayTriangleReconstruct {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every query in `queries` and returns one
    /// [`RayTriangleReconstructResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RayTriangleReconstructQuery],
    ) -> Vec<RayTriangleReconstructResult> {
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
            label: Some("prism_volumetric_ray_triangle_reconstruct_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ray_triangle_reconstruct_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ray_triangle_reconstruct_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ray_triangle_reconstruct_bind_group"),
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
            label: Some("prism_volumetric_ray_triangle_reconstruct_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ray_triangle_reconstruct_encoder"),
        });
        {
            // One thread per query.
            let threads = count as u32;
            let groups = threads.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ray_triangle_reconstruct_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
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
