//! `wgpu` compute twin of the per-triangle surface-evaluation contract
//! ([`mesh_emission`](prism_render_architecture::particle::mesh_emission),
//! particle design §8).
//!
//! The `CPU` golden
//! [`MeshTriangle`](prism_render_architecture::particle::mesh_emission::MeshTriangle)
//! owns the small, verifiable surface math a mesh emitter needs once a triangle
//! has been picked: the surface area
//! ([`MeshTriangle::area`](prism_render_architecture::particle::mesh_emission::MeshTriangle::area)),
//! the unit geometric normal
//! ([`MeshTriangle::geometric_normal`](prism_render_architecture::particle::mesh_emission::MeshTriangle::geometric_normal)),
//! and the barycentric interpolation of a position
//! ([`MeshTriangle::position_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::position_at)),
//! a renormalized shading normal
//! ([`MeshTriangle::normal_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::normal_at)),
//! and a texture coordinate
//! ([`MeshTriangle::uv_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::uv_at)).
//! [`GpuMeshTriangleEval`] is the on-device twin: one thread evaluates one
//! triangle, so a passing real-device parity test is direct evidence the ported
//! kernel computes the same five quantities and classifies the same degenerate
//! normal case the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For a batch of independent triangles, every per-triangle answer the
//! reference computes is reproduced: the scalar area, the unit geometric
//! normal, the interpolated position, the renormalized shading normal (with the
//! geometric-normal fallback), and the interpolated `UV`. Each triangle is
//! supplied with its three vertices (position, shading normal and `UV`) and one
//! barycentric weight triple `bary`; the five answers are evaluated at `bary`.
//!
//! # Correctness model
//!
//! The area, the normals, the position and the `UV` all thread through
//! multiplies, adds, a cross product and one guarded reciprocal `sqrt`, so `CPU`
//! and `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits. The parity test
//! therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on
//! every continuous quantity, tight enough to catch a genuinely wrong port (a
//! dropped term, a swapped corner, a wrong weight) yet loose enough to admit
//! legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! The geometric normal of a collinear or coincident triangle has a cross
//! product whose squared length is at or below [`EPS_LEN_SQ`]; the kernel then
//! returns the zero vector instead of dividing by a near-zero length, matching
//! the reference `normalize_or_zero`. The shading normal falls back to the
//! geometric normal when the interpolated normal's squared length is at or below
//! [`EPS_LEN_SQ`] (an unauthored, all-zero normal set), mirroring the reference
//! `normal_at`. An empty batch short-circuits on the host with no dispatch, since
//! a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `dot`, `cross`,
//! `sqrt`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no
//! loop: each thread performs a fixed, bounded sequence of arithmetic, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::mesh_emission`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::mesh_emission::MeshTriangle;
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

/// The portable core-`WGSL` mesh-triangle-evaluation kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`MeshTriangle`](prism_render_architecture::particle::mesh_emission::MeshTriangle)
/// method for method; see the module documentation for the algorithm.
const MESH_TRIANGLE_EVAL_WGSL: &str = r#"
// Mesh-triangle-evaluation twin: one thread per triangle reproduces the surface
// area, the unit geometric normal, the barycentric position, the renormalized
// shading normal (with the geometric-normal fallback) and the barycentric UV. It
// mirrors the CPU golden `particle::mesh_emission::MeshTriangle` method for
// method, uses only the portable core-WGSL subset (dot/cross/sqrt and + - * /
// plus unsigned index math), needs no transcendental call and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12. There is no loop, so
// the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::mesh_emission；无第三方
// 引擎源码或衍生代码。

// Squared-length magnitude below which a vector is treated as zero, so a
// normalization yields the zero vector rather than dividing by a near-zero
// length. Matches the reference `EPS_LEN_SQ`; the compare rule used instead of
// an f32 `==`.
const EPS_LEN_SQ: f32 = 1.0e-12;

struct Params {
    // Number of triangles in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Vertex a position; a pad lane follows.
    a_pos: vec3<f32>,
    pad0: f32,
    // Vertex a shading normal; a pad lane follows.
    a_nrm: vec3<f32>,
    pad1: f32,
    // Vertex b position; a pad lane follows.
    b_pos: vec3<f32>,
    pad2: f32,
    // Vertex b shading normal; a pad lane follows.
    b_nrm: vec3<f32>,
    pad3: f32,
    // Vertex c position; a pad lane follows.
    c_pos: vec3<f32>,
    pad4: f32,
    // Vertex c shading normal; a pad lane follows.
    c_nrm: vec3<f32>,
    pad5: f32,
    // Barycentric weight triple (w_a, w_b, w_c); a pad lane follows.
    bary: vec3<f32>,
    pad6: f32,
    // Per-vertex texture coordinates, with a trailing pad vec2.
    a_uv: vec2<f32>,
    b_uv: vec2<f32>,
    c_uv: vec2<f32>,
    pad7: vec2<f32>,
}

struct Result {
    // Surface area, with three pad lanes filling the first 16-byte slot.
    area: f32,
    pad_a0: f32,
    pad_a1: f32,
    pad_a2: f32,
    // Unit geometric normal; a pad lane follows.
    geo_n: vec3<f32>,
    pad_g: f32,
    // Interpolated position; a pad lane follows.
    pos: vec3<f32>,
    pad_p: f32,
    // Renormalized shading normal (geometric-normal fallback); a pad lane follows.
    nrm: vec3<f32>,
    pad_n: f32,
    // Interpolated texture coordinate, with two pad lanes.
    uv: vec2<f32>,
    pad_u0: f32,
    pad_u1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Unit vector along `v`, or the zero vector when `v` is (numerically) zero, so
// normalization never yields NaN. Mirrors the reference `Vec3::normalize_or_zero`
// using the squared-length guard `EPS_LEN_SQ`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Two triangle edges from vertex a; their cross product is twice the area
    // vector and points along the geometric normal.
    let e1 = q.b_pos - q.a_pos;
    let e2 = q.c_pos - q.a_pos;
    let cr = cross(e1, e2);

    // area = 0.5 * |(b - a) x (c - a)|; one cross and one sqrt.
    let area = sqrt(dot(cr, cr)) * 0.5;

    // geometric_normal = normalize_or_zero((b - a) x (c - a)); zero for a
    // degenerate triangle, never NaN.
    let geo_n = normalize_or_zero(cr);

    // position_at(bary): barycentric blend of the three corner positions.
    let pos = q.a_pos * q.bary.x + q.b_pos * q.bary.y + q.c_pos * q.bary.z;

    // normal_at(bary): blend the shading normals, renormalize, and fall back to
    // the geometric normal when the blended normal is (numerically) zero.
    let blended = q.a_nrm * q.bary.x + q.b_nrm * q.bary.y + q.c_nrm * q.bary.z;
    let blended_len_sq = dot(blended, blended);
    var nrm: vec3<f32>;
    if (blended_len_sq > EPS_LEN_SQ) {
        nrm = blended * (1.0 / sqrt(blended_len_sq));
    } else {
        nrm = geo_n;
    }

    // uv_at(bary): barycentric blend of the three corner texture coordinates.
    let uv = vec2<f32>(
        q.a_uv.x * q.bary.x + q.b_uv.x * q.bary.y + q.c_uv.x * q.bary.z,
        q.a_uv.y * q.bary.x + q.b_uv.y * q.bary.y + q.c_uv.y * q.bary.z,
    );

    var out: Result;
    out.area = area;
    out.pad_a0 = 0.0;
    out.pad_a1 = 0.0;
    out.pad_a2 = 0.0;
    out.geo_n = geo_n;
    out.pad_g = 0.0;
    out.pos = pos;
    out.pad_p = 0.0;
    out.nrm = nrm;
    out.pad_n = 0.0;
    out.uv = uv;
    out.pad_u0 = 0.0;
    out.pad_u1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the triangle count plus three pad words
/// to fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MESH_TRIANGLE_EVAL_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid triangles in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one triangle, matching the `WGSL` `Query`
/// struct. Every `vec3` lane carries a trailing pad word so each stays `16`-byte
/// aligned on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Vertex `a` position.
    a_pos: [f32; 3],
    /// Pad lane after `a_pos`.
    pad0: f32,
    /// Vertex `a` shading normal.
    a_nrm: [f32; 3],
    /// Pad lane after `a_nrm`.
    pad1: f32,
    /// Vertex `b` position.
    b_pos: [f32; 3],
    /// Pad lane after `b_pos`.
    pad2: f32,
    /// Vertex `b` shading normal.
    b_nrm: [f32; 3],
    /// Pad lane after `b_nrm`.
    pad3: f32,
    /// Vertex `c` position.
    c_pos: [f32; 3],
    /// Pad lane after `c_pos`.
    pad4: f32,
    /// Vertex `c` shading normal.
    c_nrm: [f32; 3],
    /// Pad lane after `c_nrm`.
    pad5: f32,
    /// Barycentric weight triple.
    bary: [f32; 3],
    /// Pad lane after `bary`.
    pad6: f32,
    /// Vertex `a` texture coordinate.
    a_uv: [f32; 2],
    /// Vertex `b` texture coordinate.
    b_uv: [f32; 2],
    /// Vertex `c` texture coordinate.
    c_uv: [f32; 2],
    /// Trailing pad lanes to keep the `std430` stride a multiple of `16`.
    pad7: [f32; 2],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Surface area.
    area: f32,
    /// Padding lane.
    pad_a0: f32,
    /// Padding lane.
    pad_a1: f32,
    /// Padding lane.
    pad_a2: f32,
    /// Unit geometric normal (zero for a degenerate triangle).
    geo_n: [f32; 3],
    /// Pad lane after `geo_n`.
    pad_g: f32,
    /// Interpolated position.
    pos: [f32; 3],
    /// Pad lane after `pos`.
    pad_p: f32,
    /// Renormalized shading normal (geometric-normal fallback).
    nrm: [f32; 3],
    /// Pad lane after `nrm`.
    pad_n: f32,
    /// Interpolated texture coordinate.
    uv: [f32; 2],
    /// Padding lane.
    pad_u0: f32,
    /// Padding lane.
    pad_u1: f32,
}

/// One triangle for the mesh-triangle-evaluation twin: the three resolved
/// vertices (position, shading normal and `UV`) and the barycentric weight
/// triple at which the five quantities are evaluated.
///
/// The `triangle` reuses the `CPU` golden
/// [`MeshTriangle`](prism_render_architecture::particle::mesh_emission::MeshTriangle)
/// so the host builds one value and both the reference and the device read the
/// same vertices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMeshTriangleEvalQuery {
    /// The resolved triangle whose surface quantities are evaluated.
    pub triangle: MeshTriangle,
    /// Barycentric weight triple `(w_a, w_b, w_c)` for the interpolated outputs.
    pub bary: [f32; 3],
}

/// One resolved answer for a single triangle, mirroring every value the
/// reference reports across its five twinned methods.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMeshTriangleEvalResult {
    /// Surface area, matching
    /// [`MeshTriangle::area`](prism_render_architecture::particle::mesh_emission::MeshTriangle::area).
    pub area: f32,
    /// Unit geometric normal (zero for a degenerate triangle), matching
    /// [`MeshTriangle::geometric_normal`](prism_render_architecture::particle::mesh_emission::MeshTriangle::geometric_normal).
    pub geometric_normal: [f32; 3],
    /// Interpolated position, matching
    /// [`MeshTriangle::position_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::position_at).
    pub position: [f32; 3],
    /// Renormalized shading normal with the geometric-normal fallback, matching
    /// [`MeshTriangle::normal_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::normal_at).
    pub normal: [f32; 3],
    /// Interpolated texture coordinate, matching
    /// [`MeshTriangle::uv_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::uv_at).
    pub uv: [f32; 2],
}

/// Encodes one [`GpuMeshTriangleEvalQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &GpuMeshTriangleEvalQuery) -> GpuQuery {
    let t = q.triangle;
    GpuQuery {
        a_pos: [t.a.position.x, t.a.position.y, t.a.position.z],
        pad0: 0.0,
        a_nrm: [t.a.normal.x, t.a.normal.y, t.a.normal.z],
        pad1: 0.0,
        b_pos: [t.b.position.x, t.b.position.y, t.b.position.z],
        pad2: 0.0,
        b_nrm: [t.b.normal.x, t.b.normal.y, t.b.normal.z],
        pad3: 0.0,
        c_pos: [t.c.position.x, t.c.position.y, t.c.position.z],
        pad4: 0.0,
        c_nrm: [t.c.normal.x, t.c.normal.y, t.c.normal.z],
        pad5: 0.0,
        bary: q.bary,
        pad6: 0.0,
        a_uv: t.a.uv,
        b_uv: t.b.uv,
        c_uv: t.c.uv,
        pad7: [0.0, 0.0],
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GpuMeshTriangleEvalResult`].
fn decode_result(raw: &GpuResult) -> GpuMeshTriangleEvalResult {
    GpuMeshTriangleEvalResult {
        area: raw.area,
        geometric_normal: raw.geo_n,
        position: raw.pos,
        normal: raw.nrm,
        uv: raw.uv,
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

/// A compiled, reusable mesh-triangle-evaluation compute pipeline, twinning the
/// `CPU` golden
/// [`MeshTriangle`](prism_render_architecture::particle::mesh_emission::MeshTriangle).
pub struct GpuMeshTriangleEval {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMeshTriangleEval {
    /// Compiles the mesh-triangle-evaluation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMeshTriangleEval {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_triangle_eval"),
            source: ShaderSource::Wgsl(MESH_TRIANGLE_EVAL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_triangle_eval_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_triangle_eval_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_triangle_eval_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMeshTriangleEval {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every triangle in `queries` and returns one
    /// [`GpuMeshTriangleEvalResult`] per input, in order.
    ///
    /// The five quantities match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GpuMeshTriangleEvalQuery],
    ) -> Vec<GpuMeshTriangleEvalResult> {
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
            label: Some("prism_volumetric_mesh_triangle_eval_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_triangle_eval_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_triangle_eval_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_triangle_eval_bind_group"),
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
            label: Some("prism_volumetric_mesh_triangle_eval_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_triangle_eval_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_triangle_eval_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per triangle, flattened to a 1-D dispatch.
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
