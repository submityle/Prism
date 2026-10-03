//! `wgpu` compute twin of the per-triangle tangent-frame kernel underlying
//! this repository's
//! `prism_render_architecture::ray_scene::mesh_tangents::compute_tangents`.
//!
//! The `CPU` golden `compute_tangents` walks a whole [`TriangleMesh`],
//! accumulating each triangle's un-normalized tangent and bitangent into every
//! incident vertex (Lengyel's method, the `MikkTSpace` convention) before a
//! per-vertex Gram-Schmidt and handedness pass. That whole-mesh scatter-add is
//! an irregular, data-dependent gather that stays on the host.
//!
//! This twin isolates the **per-triangle closed form**: it treats a single
//! triangle as the *sole* contributor to each of its three vertices and
//! reproduces, branch for branch, the exact arithmetic the golden path applies
//! once per triangle:
//!
//! - solve the `2x2` `UV` edge system for the surface tangent `t` and bitangent
//!   `b` (`det = du1*dv2 - du2*dv1`; a degenerate `|det| <= 1e-20` triangle
//!   contributes nothing, so its vertices fall back to an arbitrary basis),
//! - per vertex normalize the shading normal, Gram-Schmidt the accumulated
//!   tangent against it, and
//! - record the handedness `w` so the bitangent reconstructs as `w * (N x T)`.
//!
//! # What is twinned
//!
//! [`GpuMeshTriangleTangent`] reproduces, for one triangle per query, the three
//! `[tx, ty, tz, w]` tangents the golden path would assign to that triangle's
//! vertices if it were the only incident triangle (`acc = t`, `bitan = b`), plus
//! a `degenerate` flag that is `1` exactly when the `UV` determinant collapses
//! and the fallback basis is used.
//!
//! # What stays on the host
//!
//! The whole-mesh accumulation (scatter-adding every incident triangle's `t`
//! and `b` into shared vertices, area-weighted by the un-normalized edge
//! products), the attribute-presence checks and the `Vec`-aligned output pool
//! are a variable-length gather that is not twinned; only the per-triangle
//! closed form above runs on the device.
//!
//! # Degeneracy
//!
//! A `UV`-degenerate triangle (`|det| <= 1e-20`, collinear texture
//! coordinates) contributes a zero tangent and bitangent in the golden path.
//! The twin mirrors that by zeroing `t` and `b`, so each vertex's Gram-Schmidt
//! residual collapses and the deterministic [`fallback_tangent`] basis is used
//! with handedness `+1`, matching the golden `continue`. The `degenerate` flag
//! is `1` in exactly this case.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `dot`, `cross`, `sqrt`, `+ - * /` and `select`-style ordered branches — with
//! no `sin`, `cos`, `tan`, `exp`, `log`, `pow`, `round` or optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The only
//! non-rational operation is the normalization `sqrt`, matching the reference's
//! `f32::sqrt`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and a
//! handful of `sqrt`s, so `CPU` and `GPU` evaluate the same closed form in the
//! same order. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4 || rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on the tangent
//! components while pinning the `degenerate` flag and the handedness `w`
//! exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_tangents`；无第三方引擎源码或衍生代码。

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

/// Inlined `WGSL` compute shader source. Keeping it in the Rust binary avoids
/// shipping a sidecar asset and keeps the twin and its kernel versioned as a
/// single source file. The single entry point `solve` mirrors the per-triangle
/// closed form of the `CPU` golden `compute_tangents`; see the module
/// documentation for the algorithm.
const MESH_TANGENTS_WGSL: &str = r#"
// Per-triangle tangent-frame twin: one thread per query solves the 2x2 UV edge
// system for the surface tangent/bitangent, then for each of the triangle's
// three vertices normalizes the shading normal, Gram-Schmidt-orthogonalizes the
// tangent and records the handedness w so the bitangent is w * (N x T). It
// mirrors the CPU golden compute_tangents treating one triangle as the sole
// contributor, uses only the portable core-WGSL subset (abs/min/max/dot/cross/
// sqrt and + - * /) and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::mesh_tangents；无第三方引擎源码或衍生代码。

// Normalization and determinant guards matching the reference thresholds.
const DET_EPS: f32 = 1.0e-20;
const LEN2_EPS: f32 = 1.0e-24;
const ORTHO_EPS: f32 = 1.0e-16;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: a single triangle's three positions, three shading normals and
// three UV coordinates, each vec3 lifted to a vec4 slot and the UVs packed two
// per vec4 so the std430 layout is explicit.
struct Query {
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
    // uv0.xy, uv1.xy
    uv01: vec4<f32>,
    // uv2.xy, pad, pad
    uv2: vec4<f32>,
}

// One result: the three per-vertex tangents (xyz tangent + w handedness) and
// the degenerate flag with std430 padding.
struct Result {
    t0: vec4<f32>,
    t1: vec4<f32>,
    t2: vec4<f32>,
    degenerate: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Normalizes v, returning fallback when v is too short to normalize.
fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if (len2 > LEN2_EPS) {
        let inv = 1.0 / sqrt(len2);
        return v * inv;
    }
    return fallback;
}

// Deterministic unit tangent orthogonal to n: pick the world axis least aligned
// with n and project it onto the tangent plane.
fn fallback_tangent(n: vec3<f32>) -> vec3<f32> {
    let ax = abs(n.x);
    let ay = abs(n.y);
    let az = abs(n.z);
    var axis = vec3<f32>(0.0, 0.0, 1.0);
    if (ax <= ay && ax <= az) {
        axis = vec3<f32>(1.0, 0.0, 0.0);
    } else if (ay <= az) {
        axis = vec3<f32>(0.0, 1.0, 0.0);
    }
    let d = dot(axis, n);
    let ortho = axis - n * d;
    return normalize_or(ortho, vec3<f32>(1.0, 0.0, 0.0));
}

// Gram-Schmidt the accumulated tangent t_dir against the vertex normal, then
// record the handedness against the accumulated bitangent b_dir.
fn vertex_tangent(t_dir: vec3<f32>, b_dir: vec3<f32>, raw_n: vec3<f32>) -> vec4<f32> {
    let n = normalize_or(raw_n, vec3<f32>(0.0, 0.0, 1.0));
    let acc = t_dir;
    let ndt = dot(n, acc);
    let ortho = acc - n * ndt;
    var tangent = fallback_tangent(n);
    if (dot(ortho, ortho) > ORTHO_EPS) {
        tangent = normalize_or(ortho, fallback_tangent(n));
    }
    var w = 1.0;
    if (dot(cross(n, tangent), b_dir) < 0.0) {
        w = -1.0;
    }
    return vec4<f32>(tangent.x, tangent.y, tangent.z, w);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let p0 = q.p0.xyz;
    let p1 = q.p1.xyz;
    let p2 = q.p2.xyz;
    let w0 = q.uv01.xy;
    let w1 = q.uv01.zw;
    let w2 = q.uv2.xy;

    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let du1 = w1.x - w0.x;
    let dv1 = w1.y - w0.y;
    let du2 = w2.x - w0.x;
    let dv2 = w2.y - w0.y;

    // Determinant of the UV edge matrix; a degenerate triangle contributes a
    // zero tangent/bitangent so its vertices use the fallback basis.
    let det = du1 * dv2 - du2 * dv1;
    var degenerate: u32 = 0u;
    var t_dir = vec3<f32>(0.0, 0.0, 0.0);
    var b_dir = vec3<f32>(0.0, 0.0, 0.0);
    if (abs(det) <= DET_EPS) {
        degenerate = 1u;
    } else {
        let r = 1.0 / det;
        t_dir = (e1 * dv2 - e2 * dv1) * r;
        b_dir = (e2 * du1 - e1 * du2) * r;
    }

    var out: Result;
    out.t0 = vertex_tangent(t_dir, b_dir, q.n0.xyz);
    out.t1 = vertex_tangent(t_dir, b_dir, q.n1.xyz);
    out.t2 = vertex_tangent(t_dir, b_dir, q.n2.xyz);
    out.degenerate = degenerate;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// One per-triangle tangent query: a single triangle's three positions, three
/// shading normals and three `UV` coordinates.
///
/// Mirrors a single reference triangle treated as the sole contributor to each
/// of its vertices. Vertices are index-aligned: `p0`/`n0`/`uv0` describe the
/// first vertex, and so on. Derives only [`PartialEq`] (no [`Eq`] / [`Hash`])
/// because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshTriangleTangentQuery {
    /// Position of the first vertex.
    pub p0: [f32; 3],
    /// Position of the second vertex.
    pub p1: [f32; 3],
    /// Position of the third vertex.
    pub p2: [f32; 3],
    /// Texture coordinate of the first vertex.
    pub uv0: [f32; 2],
    /// Texture coordinate of the second vertex.
    pub uv1: [f32; 2],
    /// Texture coordinate of the third vertex.
    pub uv2: [f32; 2],
    /// Shading normal of the first vertex.
    pub n0: [f32; 3],
    /// Shading normal of the second vertex.
    pub n1: [f32; 3],
    /// Shading normal of the third vertex.
    pub n2: [f32; 3],
}

impl MeshTriangleTangentQuery {
    /// Builds a query over the triangle's positions, texture coordinates and
    /// shading normals.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "one argument per triangle attribute mirrors the golden triangle inputs"
    )]
    pub const fn new(
        p0: [f32; 3],
        p1: [f32; 3],
        p2: [f32; 3],
        uv0: [f32; 2],
        uv1: [f32; 2],
        uv2: [f32; 2],
        n0: [f32; 3],
        n1: [f32; 3],
        n2: [f32; 3],
    ) -> MeshTriangleTangentQuery {
        MeshTriangleTangentQuery {
            p0,
            p1,
            p2,
            uv0,
            uv1,
            uv2,
            n0,
            n1,
            n2,
        }
    }
}

/// The per-vertex tangents for one triangle query, the host-side mirror of the
/// kernel's `Result` lane.
///
/// `tangent0`, `tangent1` and `tangent2` are the `[tx, ty, tz, w]` tangents
/// assigned to the triangle's three vertices (unit tangent plus handedness
/// `w ∈ {-1, +1}`); `degenerate` is `1` exactly when the `UV` determinant
/// collapses and the fallback basis is used. Derives only [`PartialEq`] (no
/// [`Eq`] / [`Hash`]) because it holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshTriangleTangentResult {
    /// Tangent `[tx, ty, tz, w]` of the first vertex.
    pub tangent0: [f32; 4],
    /// Tangent `[tx, ty, tz, w]` of the second vertex.
    pub tangent1: [f32; 4],
    /// Tangent `[tx, ty, tz, w]` of the third vertex.
    pub tangent2: [f32; 4],
    /// Degenerate flag: `1` when the `UV` determinant collapsed.
    pub degenerate: u32,
}

/// `repr(C)` `std430` layout of one packed query: three positions, three
/// normals (each `vec3` lifted to a `vec4` slot) and the three `UV`s packed two
/// per `vec4` — `128` bytes, exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First position, `xyz` plus a pad lane.
    p0: [f32; 4],
    /// Second position, `xyz` plus a pad lane.
    p1: [f32; 4],
    /// Third position, `xyz` plus a pad lane.
    p2: [f32; 4],
    /// First normal, `xyz` plus a pad lane.
    n0: [f32; 4],
    /// Second normal, `xyz` plus a pad lane.
    n1: [f32; 4],
    /// Third normal, `xyz` plus a pad lane.
    n2: [f32; 4],
    /// `uv0.xy` then `uv1.xy`.
    uv01: [f32; 4],
    /// `uv2.xy` then two pad lanes.
    uv2: [f32; 4],
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &MeshTriangleTangentQuery) -> GpuQuery {
        GpuQuery {
            p0: [query.p0[0], query.p0[1], query.p0[2], 0.0],
            p1: [query.p1[0], query.p1[1], query.p1[2], 0.0],
            p2: [query.p2[0], query.p2[1], query.p2[2], 0.0],
            n0: [query.n0[0], query.n0[1], query.n0[2], 0.0],
            n1: [query.n1[0], query.n1[1], query.n1[2], 0.0],
            n2: [query.n2[0], query.n2[1], query.n2[2], 0.0],
            uv01: [query.uv0[0], query.uv0[1], query.uv1[0], query.uv1[1]],
            uv2: [query.uv2[0], query.uv2[1], 0.0, 0.0],
        }
    }
}

/// `repr(C)` `std430` layout of one result: three tangent `vec4`s then the
/// degenerate flag with three pad words — `64` bytes matching the `WGSL`
/// `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Tangent of the first vertex.
    t0: [f32; 4],
    /// Tangent of the second vertex.
    t1: [f32; 4],
    /// Tangent of the third vertex.
    t2: [f32; 4],
    /// Degenerate flag (`1` = fallback basis used).
    degenerate: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// round the uniform block out to `16` bytes.
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

/// Decodes one packed `GpuResult` into the public [`MeshTriangleTangentResult`].
fn decode_result(raw: &GpuResult) -> MeshTriangleTangentResult {
    MeshTriangleTangentResult {
        tangent0: raw.t0,
        tangent1: raw.t1,
        tangent2: raw.t2,
        degenerate: raw.degenerate,
    }
}

/// Builds one storage/uniform buffer bind-group-layout entry.
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

/// On-device twin of the per-triangle tangent-frame kernel.
///
/// Owns the compiled [`ComputePipeline`] and its [`BindGroupLayout`]; build it
/// once with [`GpuMeshTriangleTangent::new`] and reuse it across
/// [`GpuMeshTriangleTangent::evaluate`] calls.
pub struct GpuMeshTriangleTangent {
    /// The compiled shader module (retained so the pipeline stays valid).
    #[expect(
        dead_code,
        reason = "retained so the compiled module outlives the pipeline"
    )]
    module: ShaderModule,
    /// The bind group layout shared by every dispatch.
    layout: BindGroupLayout,
    /// The compute pipeline running the `solve` entry point.
    pipeline: ComputePipeline,
}

impl GpuMeshTriangleTangent {
    /// Compiles the kernel and builds the reusable pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMeshTriangleTangent {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_tangents_shader"),
            source: ShaderSource::Wgsl(MESH_TANGENTS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_tangents_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_tangents_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_tangents_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMeshTriangleTangent {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every triangle query on-device and returns one
    /// [`MeshTriangleTangentResult`] per input, in order.
    ///
    /// Each result equals the per-triangle reference answer to within the
    /// tolerance documented on this module. An empty input returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MeshTriangleTangentQuery],
    ) -> Vec<MeshTriangleTangentResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_tangents_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_tangents_output"),
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
            label: Some("prism_volumetric_mesh_tangents_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_tangents_bind_group"),
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
            label: Some("prism_volumetric_mesh_tangents_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_tangents_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_tangents_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per triangle query, flattened to a 1-D dispatch.
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
