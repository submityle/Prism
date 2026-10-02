//! `wgpu` compute twin of the `CPU` golden closed-mesh point-containment test
//! [`Mesh::contains_point`](prism_render_architecture::particle::mesh_emission::Mesh::contains_point)
//! (design §8).
//!
//! The golden test casts one fixed, non-axis-aligned ray from the query point
//! and counts forward Möller–Trumbore triangle crossings; an odd count means
//! the point lies inside the (assumed closed) mesh. The `CPU`
//! [`mesh_emission`](prism_render_architecture::particle::mesh_emission) module
//! owns that math; [`GpuMeshRayContains`] is the on-device twin, validated
//! against the public reference so a passing real-device parity test is direct
//! evidence the ported kernel evaluates the same intersection algebra, not
//! merely that the shader compiles.
//!
//! # Algorithm
//!
//! The kernel is a pure per-point gather: one thread owns one query point,
//! reads every triangle, and never scatters. For point `p` it walks all
//! triangles with the shared ray direction `RAY_DIR` and, for each, runs the
//! Möller–Trumbore test replicated verbatim from the golden private
//! `ray_hits_triangle`: it forms the edges `e1 = b − a`, `e2 = c − a`, the
//! determinant `det = e1 · (dir × e2)`, rejects a back-parallel or degenerate
//! triangle when `|det| < EPS`, then solves the barycentric coordinates
//! `u`, `v` and the ray parameter `t`, counting a forward crossing only when
//! `0 ≤ u ≤ 1`, `0 ≤ v`, `u + v ≤ 1` and `t > EPS`. The parity output is the
//! crossing count's low bit: `crossings & 1`, which equals the golden
//! `crossings % 2 == 1`.
//!
//! The fixed ray direction `RAY_DIR = (1.0, 0.372_133_1, 0.211_907_3)` is a
//! golden private constant; its skewed, non-axis-aligned components keep the
//! ray from grazing shared triangle edges and vertices on axis-aligned meshes,
//! so a boundary sample is counted once rather than twice. The literal is
//! replicated in the shader (and mirrored in the parity test for its
//! near-branch rejection sampling).
//!
//! # Boundaries
//!
//! A mesh with no triangles yields zero crossings, so every point is reported
//! outside. The host short-circuits that case to an all-`false` vector so the
//! device never binds a zero-length triangle storage buffer. An empty query
//! point list short-circuits to an empty vector without dispatching.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer index
//! arithmetic, `+ − × ÷` and comparisons on `f32`, one `abs` and one division
//! per triangle — with no `sin`, `cos`, `exp`, `log`, `pow` or optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The dot and
//! cross products are hand-written so their evaluation order matches the scalar
//! reference exactly.
//!
//! # Correctness model
//!
//! The per-point output is a boolean parity, not a continuous quantity, so the
//! parity test asserts an exact `==` on the recovered flag. Fixtures use
//! rejection sampling to keep every query point well clear of the Möller–
//! Trumbore decision thresholds (`det`, `u ∈ {0, 1}`, `v = 0`, `u + v = 1`,
//! `t ≈ EPS`), so a legal `GPU` fused multiply-add that perturbs the low
//! mantissa bits can never flip a comparison and hence never flip the boolean.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twins the `CPU` golden `Mesh::contains_point` and its private
//! `ray_hits_triangle` and `RAY_DIR` in
//! `prism_render_architecture::particle::mesh_emission`; no Unreal Engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The number of threads per workgroup. `64` is a portable, warp-friendly size
/// used across this crate's kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` point-containment kernel, embedded inline so the
/// twin ships as a single source file. Mirrors the `CPU` ray-crossing count
/// exactly; see the module documentation for the algorithm.
const CONTAINS_WGSL: &str = r#"
// Closed-mesh point-containment twin: one thread per query point casts the
// fixed `RAY_DIR` ray and counts forward Möller–Trumbore crossings, writing the
// count's low bit (odd -> inside). It mirrors the CPU golden
// `particle::mesh_emission::Mesh::contains_point` and its private
// `ray_hits_triangle`, uses only the portable core-WGSL subset (integer index
// math plus + - * / and comparisons on scalars, one abs and one division per
// triangle), and takes no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: twins the CPU golden Mesh::contains_point and ray_hits_triangle;
// no Unreal Engine source or derived code.

struct Params {
    // Number of triangles in the mesh (each three padded vec4 entries).
    tri_count: u32,
    // Number of query points (one thread each).
    point_count: u32,
    // Padding to keep the uniform a multiple of 16 bytes.
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Triangle vertices, three padded vec4 per triangle: a, b, c at 3*i, 3*i+1,
// 3*i+2.
@group(0) @binding(1) var<storage, read> triangles: array<vec4<f32>>;
// Query points, one padded vec4 each.
@group(0) @binding(2) var<storage, read> points: array<vec4<f32>>;
// Per-point containment flag (0 = outside, 1 = inside).
@group(0) @binding(3) var<storage, read_write> out_flags: array<u32>;

// The absolute tolerance from the golden `mesh_emission::EPS`.
const EPS: f32 = 0.000001;

// Hand-written dot so the summation order matches the scalar reference.
fn vdot(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Hand-written cross so the component order matches the scalar reference.
fn vcross(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    );
}

// Möller–Trumbore forward crossing, transcribed from the golden private
// `ray_hits_triangle`: returns true when the ray from `origin` along `dir`
// crosses triangle (a, b, c) in the forward (t > EPS) direction.
fn ray_hits(origin: vec3<f32>, dir: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> bool {
    let e1 = b - a;
    let e2 = c - a;
    let h = vcross(dir, e2);
    let det = vdot(e1, h);
    if (abs(det) < EPS) {
        return false;
    }
    let inv_det = 1.0 / det;
    let s = origin - a;
    let u = inv_det * vdot(s, h);
    if (u < 0.0 || u > 1.0) {
        return false;
    }
    let q = vcross(s, e1);
    let v = inv_det * vdot(dir, q);
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }
    let t = inv_det * vdot(e2, q);
    return t > EPS;
}

@compute @workgroup_size(64)
fn contains(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.point_count) {
        return;
    }

    let origin = points[idx].xyz;
    // The golden fixed, non-axis-aligned ray direction `RAY_DIR`.
    let dir = vec3<f32>(1.0, 0.3721331, 0.2119073);

    var crossings = 0u;
    // Bounded loop over all triangles; the upper bound is the triangle count.
    for (var i = 0u; i < params.tri_count; i = i + 1u) {
        let a = triangles[3u * i + 0u].xyz;
        let b = triangles[3u * i + 1u].xyz;
        let c = triangles[3u * i + 2u].xyz;
        if (ray_hits(origin, dir, a, b, c)) {
            crossings = crossings + 1u;
        }
    }

    // Odd crossing count means inside; the low bit equals `crossings % 2 == 1`.
    out_flags[idx] = crossings & 1u;
}
"#;

/// Uniform parameters for the kernel. `repr(C)` `std430` layout matching
/// `Params` in [`CONTAINS_WGSL`]: the triangle and point counts then two pad
/// words — `16` bytes total with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ContainsParams {
    /// Number of triangles in the mesh.
    tri_count: u32,
    /// Number of query points.
    point_count: u32,
    /// Padding to keep the struct a multiple of `16` bytes.
    pad0: u32,
    /// Padding to keep the struct a multiple of `16` bytes.
    pad1: u32,
}

/// One `Vec3` as uploaded to the device. `16`-byte `std430` stride matching
/// `vec4<f32>` in the shader: the three components plus a zero pad lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVec3Pad {
    /// X component.
    x: f32,
    /// Y component.
    y: f32,
    /// Z component.
    z: f32,
    /// Padding lane, held at zero so it never perturbs the arithmetic.
    pad: f32,
}

impl GpuVec3Pad {
    /// Packs a [`Vec3`] into the padded device layout.
    fn from_vec3(v: Vec3) -> GpuVec3Pad {
        GpuVec3Pad {
            x: v.x,
            y: v.y,
            z: v.z,
            pad: 0.0,
        }
    }
}

/// One triangle of the containment mesh, three object-space vertices.
///
/// The twin only needs the positions; the kernel reads them as the Möller–
/// Trumbore `a`, `b`, `c` in this order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuTriangle {
    /// First vertex `a`.
    pub a: Vec3,
    /// Second vertex `b`.
    pub b: Vec3,
    /// Third vertex `c`.
    pub c: Vec3,
}

impl GpuTriangle {
    /// Builds a triangle from its three vertices.
    #[must_use]
    pub fn new(a: Vec3, b: Vec3, c: Vec3) -> GpuTriangle {
        GpuTriangle { a, b, c }
    }
}

/// A containment request: the closed mesh's triangles and the query points.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMeshContainsQuery {
    /// The mesh triangles the ray is tested against.
    pub triangles: Vec<GpuTriangle>,
    /// The query points, each classified inside (`true`) or outside (`false`).
    pub points: Vec<Vec3>,
}

/// A compiled, reusable point-containment pipeline.
pub struct GpuMeshRayContains {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMeshRayContains {
    /// Compiles the point-containment kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMeshRayContains {
        let device = ctx.device();

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_ray_contains"),
            source: ShaderSource::Wgsl(CONTAINS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_ray_contains_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_ray_contains_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_ray_contains_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("contains"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuMeshRayContains {
            module,
            layout,
            pipeline,
        }
    }

    /// Classifies each `query.points` entry against the closed `query.triangles`
    /// mesh, one thread per point.
    ///
    /// The returned flag equals
    /// [`Mesh::contains_point`](prism_render_architecture::particle::mesh_emission::Mesh::contains_point)
    /// for the mesh with the same triangle positions, element for element. An
    /// empty point list yields an empty vector without dispatching; a mesh with
    /// no triangles yields all `false` (every point is outside) without binding
    /// a zero-length triangle buffer.
    #[must_use]
    pub fn contains(&self, ctx: &GpuContext, query: &GpuMeshContainsQuery) -> Vec<bool> {
        let point_count = query.points.len();
        if point_count == 0 {
            return Vec::new();
        }
        let tri_count = query.triangles.len();
        if tri_count == 0 {
            return vec![false; point_count];
        }

        let device = ctx.device();
        let gpu_params = ContainsParams {
            tri_count: tri_count as u32,
            point_count: point_count as u32,
            pad0: 0,
            pad1: 0,
        };

        // Three padded vec4 per triangle, in a, b, c order.
        let mut packed_tris: Vec<GpuVec3Pad> = Vec::with_capacity(tri_count * 3);
        for tri in &query.triangles {
            packed_tris.push(GpuVec3Pad::from_vec3(tri.a));
            packed_tris.push(GpuVec3Pad::from_vec3(tri.b));
            packed_tris.push(GpuVec3Pad::from_vec3(tri.c));
        }
        let packed_points: Vec<GpuVec3Pad> = query
            .points
            .iter()
            .copied()
            .map(GpuVec3Pad::from_vec3)
            .collect();
        let out_bytes = (point_count as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_ray_contains_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let triangles_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_ray_contains_triangles"),
            contents: bytemuck::cast_slice(&packed_tris),
            usage: BufferUsages::STORAGE,
        });
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_ray_contains_points"),
            contents: bytemuck::cast_slice(&packed_points),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_ray_contains_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_ray_contains_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_ray_contains_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: triangles_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: points_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let groups = (point_count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_ray_contains_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_ray_contains_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
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
        let gpu_flags = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(gpu_flags.len(), point_count);

        gpu_flags.into_iter().map(|flag| flag == 1).collect()
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
