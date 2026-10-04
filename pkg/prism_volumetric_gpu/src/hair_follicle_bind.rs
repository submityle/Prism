//! `wgpu` compute twin of follicle binding and deformation transfer from the
//! `CPU` golden `prism_render_architecture::hair::follicle_bind`.
//!
//! A groom is authored once in a rest pose, but the scalp it grows from is an
//! animated skinned mesh. Each follicle (hair root) is bound to one scalp
//! triangle by its barycentric coordinates plus a small signed offset along the
//! surface normal, so a root that sits a hair's breadth above the skin keeps
//! that height after the triangle deforms. When the scalp triangle moves
//! (skinning, blendshapes, head turns) the root position and its local tangent
//! frame are reconstructed from the deformed triangle, so hair stays glued to
//! the scalp without sliding or poking through.
//!
//! [`GpuHairFollicleBind`] is the on-device twin: one thread binds one root to
//! its rest triangle and then transfers it onto the deformed triangle in a
//! single pass. A passing real-device parity test is direct evidence the ported
//! kernel reproduces the same barycentric solve, the same normal-offset
//! reconstruction and the same Gram-Schmidt frame the reference computes, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces four stateless, `RNG`-free closed forms
//! of the golden module:
//!
//! - `compute_barycentric`: Ericson's projected-plane barycentric solve, with a
//!   centroid fallback for a degenerate (zero-area) triangle.
//! - `bind_follicle`: the sanitised barycentric position plus the signed height
//!   of the root above the rest surface along the interpolated normal.
//! - `transfer_root`: the barycentric surface point on the deformed triangle
//!   plus the stored offset along the deformed interpolated normal.
//! - `transfer_frame`: the interpolated normal with a Gram-Schmidt tangent and a
//!   right-handed bitangent, each re-normalised with a canonical-axis fallback.
//!
//! The array batch helper `transfer_root_map` is intentionally not twinned: it
//! is just a per-element loop over `transfer_root`, which this kernel already
//! runs one thread per query.
//!
//! # What stays on the host
//!
//! The skinning pass that produces the deformed triangle each frame, the groom
//! authoring and the follicle-to-triangle assignment all stay on the host; the
//! device sees only the stateless, fixed-width bind-and-transfer, one follicle
//! at a time, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! The solve threads through guarded divisions and `sqrt` (barycentric
//! denominator, normal re-normalisation), so the `CPU` and `GPU` are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate. The parity test asserts every continuous quantity within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (relative floor `1e-6`). The
//! discrete `valid` word is asserted exactly: `1` for a non-degenerate rest
//! triangle, else `0`.
//!
//! # Degenerate inputs
//!
//! A zero-area rest triangle has a barycentric denominator at or below
//! `1e-12`; the reference falls back to the centroid weighting and canonical
//! frame axes, and the twin reports `valid = 0` while still producing those
//! same finite fallback outputs. Fixtures and the sweep keep the random
//! triangles well away from zero area so parity never sits on that knife edge.
//! An empty query batch short-circuits on the host with no dispatch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `dot`, `cross`, `select`, `sqrt`, `+ - * /` and unsigned index arithmetic —
//! with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`, no inverse trigonometry, no
//! `round` and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::follicle_bind`；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` follicle-bind kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// `hair::follicle_bind::{compute_barycentric, bind_follicle, transfer_root, transfer_frame}`.
const HAIR_FOLLICLE_BIND_WGSL: &str = r#"
// Follicle-bind twin: one thread binds one hair root to its rest scalp triangle
// (barycentric position + signed normal offset) and transfers it onto the
// deformed triangle, mirroring the CPU golden
// `hair::follicle_bind::{compute_barycentric, bind_follicle, transfer_root,
// transfer_frame}` with only ordered comparisons, dot/cross, sqrt, products and
// quotients. The skinning pass and follicle assignment stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::hair::follicle_bind；无第三方
// 引擎源码或衍生代码。

const DEGENERATE_EPS: f32 = 1e-12;
const THIRD: f32 = 0.3333333333333333;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Query {
    // Hair root world position.
    root: array<f32, 3>,
    // Rest triangle: three vertex positions, normals, tangents (xyz each).
    rest_pos: array<f32, 9>,
    rest_nrm: array<f32, 9>,
    rest_tan: array<f32, 9>,
    // Deformed triangle: three vertex positions, normals, tangents (xyz each).
    def_pos: array<f32, 9>,
    def_nrm: array<f32, 9>,
    def_tan: array<f32, 9>,
};

struct Outcome {
    root: array<f32, 3>,
    normal: array<f32, 3>,
    tangent: array<f32, 3>,
    bitangent: array<f32, 3>,
    valid: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Outcome>;

// Reads a packed xyz triple starting at `base` from a 9-wide array.
fn tri_vertex(a: array<f32, 9>, base: u32) -> vec3<f32> {
    return vec3<f32>(a[base], a[base + 1u], a[base + 2u]);
}

// Normalises `a`, returning `fallback` when `a` is (near) zero-length.
fn normalize_or(a: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = sqrt(dot(a, a));
    return select(fallback, a / len, len > DEGENERATE_EPS);
}

// Non-negative weights that sum to one; an all-zero set falls back to centroid.
fn sanitize_bary(b: vec3<f32>) -> vec3<f32> {
    let u = select(0.0, b.x, b.x > 0.0);
    let v = select(0.0, b.y, b.y > 0.0);
    let w = select(0.0, b.z, b.z > 0.0);
    let sum = u + v + w;
    let centroid = vec3<f32>(THIRD, THIRD, THIRD);
    return select(centroid, vec3<f32>(u / sum, v / sum, w / sum), sum > DEGENERATE_EPS);
}

// Interpolates a per-vertex attribute by barycentric weights.
fn bary_mix(w: vec3<f32>, a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>) -> vec3<f32> {
    return a0 * w.x + a1 * w.y + a2 * w.z;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let root = vec3<f32>(q.root[0], q.root[1], q.root[2]);

    let rp0 = tri_vertex(q.rest_pos, 0u);
    let rp1 = tri_vertex(q.rest_pos, 3u);
    let rp2 = tri_vertex(q.rest_pos, 6u);
    let rn0 = tri_vertex(q.rest_nrm, 0u);
    let rn1 = tri_vertex(q.rest_nrm, 3u);
    let rn2 = tri_vertex(q.rest_nrm, 6u);

    let dp0 = tri_vertex(q.def_pos, 0u);
    let dp1 = tri_vertex(q.def_pos, 3u);
    let dp2 = tri_vertex(q.def_pos, 6u);
    let dn0 = tri_vertex(q.def_nrm, 0u);
    let dn1 = tri_vertex(q.def_nrm, 3u);
    let dn2 = tri_vertex(q.def_nrm, 6u);
    let dt0 = tri_vertex(q.def_tan, 0u);
    let dt1 = tri_vertex(q.def_tan, 3u);
    let dt2 = tri_vertex(q.def_tan, 6u);

    // compute_barycentric on the rest triangle (Ericson's method).
    let v0 = rp1 - rp0;
    let v1 = rp2 - rp0;
    let v2 = root - rp0;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    let degenerate = abs(denom) <= DEGENERATE_EPS;
    // Guard the reciprocal so a degenerate triangle never divides by ~zero; the
    // centroid branch discards the guarded value anyway.
    let safe_denom = select(denom, 1.0, degenerate);
    let inv = 1.0 / safe_denom;
    let bv = (d11 * d20 - d01 * d21) * inv;
    let bw = (d00 * d21 - d01 * d20) * inv;
    let bu = 1.0 - bv - bw;
    let centroid = vec3<f32>(THIRD, THIRD, THIRD);
    let bary_raw = select(vec3<f32>(bu, bv, bw), centroid, degenerate);

    // bind_follicle: sanitised weights, rest surface point and signed offset.
    let bary = sanitize_bary(bary_raw);
    let surface_rest = bary_mix(bary, rp0, rp1, rp2);
    let normal_rest = normalize_or(bary_mix(bary, rn0, rn1, rn2), vec3<f32>(0.0, 0.0, 1.0));
    let offset = dot(root - surface_rest, normal_rest);

    // transfer_root onto the deformed triangle.
    let surface_def = bary_mix(bary, dp0, dp1, dp2);
    let normal_def = normalize_or(bary_mix(bary, dn0, dn1, dn2), vec3<f32>(0.0, 0.0, 1.0));
    let out_root = surface_def + normal_def * offset;

    // transfer_frame: Gram-Schmidt tangent, right-handed bitangent.
    let raw_tangent = bary_mix(bary, dt0, dt1, dt2);
    let projected = raw_tangent - normal_def * dot(raw_tangent, normal_def);
    let fb_lo = normalize_or(cross(normal_def, vec3<f32>(1.0, 0.0, 0.0)), vec3<f32>(0.0, 1.0, 0.0));
    let fb_hi = normalize_or(cross(normal_def, vec3<f32>(0.0, 1.0, 0.0)), vec3<f32>(1.0, 0.0, 0.0));
    let fallback = select(fb_hi, fb_lo, abs(normal_def.x) < 0.9);
    let tangent = normalize_or(projected, fallback);
    let bitangent = normalize_or(cross(normal_def, tangent), vec3<f32>(0.0, 1.0, 0.0));

    var out: Outcome;
    out.root = array<f32, 3>(out_root.x, out_root.y, out_root.z);
    out.normal = array<f32, 3>(normal_def.x, normal_def.y, normal_def.z);
    out.tangent = array<f32, 3>(tangent.x, tangent.y, tangent.z);
    out.bitangent = array<f32, 3>(bitangent.x, bitangent.y, bitangent.z);
    out.valid = select(1u, 0u, degenerate);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte-aligned uniform struct matching `Params` in
/// [`HAIR_FOLLICLE_BIND_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
///
/// Every field is a flat `f32` array (no `vec3`) so the host and device agree
/// on byte strides without any `16`-byte `vec3` padding ambiguity. Positions,
/// normals and tangents are stored vertex-major: `[x0, y0, z0, x1, ...]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Hair root world position `[x, y, z]`.
    root: [f32; 3],
    /// Rest triangle vertex positions, three `xyz` triples.
    rest_pos: [f32; 9],
    /// Rest triangle per-vertex normals, three `xyz` triples.
    rest_nrm: [f32; 9],
    /// Rest triangle per-vertex tangents, three `xyz` triples.
    rest_tan: [f32; 9],
    /// Deformed triangle vertex positions, three `xyz` triples.
    def_pos: [f32; 9],
    /// Deformed triangle per-vertex normals, three `xyz` triples.
    def_nrm: [f32; 9],
    /// Deformed triangle per-vertex tangents, three `xyz` triples.
    def_tan: [f32; 9],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Outcome`
/// struct: the transferred root, the orthonormal frame and the `valid` word.
/// All members are `4`-byte aligned, so the stride matches on host and device
/// with no padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Transferred root world position.
    root: [f32; 3],
    /// Unit surface normal at the root.
    normal: [f32; 3],
    /// Unit surface tangent, orthogonal to the normal.
    tangent: [f32; 3],
    /// Unit bitangent completing the right-handed basis.
    bitangent: [f32; 3],
    /// `1` for a non-degenerate rest triangle, else `0`.
    valid: u32,
}

/// A scalp triangle with a per-vertex local frame, mirroring the golden
/// `TriangleFrame`. `positions[i]` is the world position of vertex `i`;
/// `normals[i]` and `tangents[i]` are that vertex's local frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleFrame {
    /// World-space positions of the three triangle vertices.
    pub positions: [[f32; 3]; 3],
    /// Per-vertex surface normals.
    pub normals: [[f32; 3]; 3],
    /// Per-vertex surface tangents.
    pub tangents: [[f32; 3]; 3],
}

/// One query for the follicle-bind twin: the hair root and the rest and
/// deformed scalp triangles.
///
/// The thread binds `root` to `rest` (barycentric position plus signed normal
/// offset) and transfers that binding onto `deformed`, producing the moved root
/// and its local frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairFollicleBindQuery {
    /// Hair root world position.
    pub root: [f32; 3],
    /// Rest-pose scalp triangle the root is bound to.
    pub rest: TriangleFrame,
    /// Deformed scalp triangle the binding is transferred onto.
    pub deformed: TriangleFrame,
}

impl HairFollicleBindQuery {
    /// Builds a query from the root and the rest and deformed triangles.
    #[must_use]
    pub const fn new(
        root: [f32; 3],
        rest: TriangleFrame,
        deformed: TriangleFrame,
    ) -> HairFollicleBindQuery {
        HairFollicleBindQuery {
            root,
            rest,
            deformed,
        }
    }
}

/// One resolved follicle binding transferred onto the deformed triangle: the
/// moved root, its orthonormal frame and the degeneracy flag.
///
/// `valid` is `1` when the rest triangle is non-degenerate; it is `0` for a
/// zero-area rest triangle, in which case the outputs are the reference's
/// centroid-and-canonical-axis fallbacks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairFollicleBindResult {
    /// Transferred root world position.
    pub root: [f32; 3],
    /// Unit surface normal at the root.
    pub normal: [f32; 3],
    /// Unit surface tangent, orthogonal to the normal.
    pub tangent: [f32; 3],
    /// Unit bitangent completing the right-handed basis.
    pub bitangent: [f32; 3],
    /// `1` for a non-degenerate rest triangle, else `0`.
    pub valid: u32,
}

/// Flattens a triangle's three `xyz` triples into a `9`-wide vertex-major array.
fn flatten_tri(tri: &[[f32; 3]; 3]) -> [f32; 9] {
    [
        tri[0][0], tri[0][1], tri[0][2], tri[1][0], tri[1][1], tri[1][2], tri[2][0], tri[2][1],
        tri[2][2],
    ]
}

/// Encodes one public [`HairFollicleBindQuery`] into its packed `std430` form.
fn encode_query(q: &HairFollicleBindQuery) -> GpuQuery {
    GpuQuery {
        root: q.root,
        rest_pos: flatten_tri(&q.rest.positions),
        rest_nrm: flatten_tri(&q.rest.normals),
        rest_tan: flatten_tri(&q.rest.tangents),
        def_pos: flatten_tri(&q.deformed.positions),
        def_nrm: flatten_tri(&q.deformed.normals),
        def_tan: flatten_tri(&q.deformed.tangents),
    }
}

/// Decodes one packed [`GpuResult`] into the public [`HairFollicleBindResult`].
fn decode_result(raw: &GpuResult) -> HairFollicleBindResult {
    HairFollicleBindResult {
        root: raw.root,
        normal: raw.normal,
        tangent: raw.tangent,
        bitangent: raw.bitangent,
        valid: raw.valid,
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

/// Compute twin of the hair follicle bind-and-transfer closed form.
///
/// Build once with [`GpuHairFollicleBind::new`], then call
/// [`GpuHairFollicleBind::evaluate`] for each batch of follicles.
pub struct GpuHairFollicleBind {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairFollicleBind {
    /// Compiles the follicle-bind kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairFollicleBind {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hair_follicle_bind"),
            source: ShaderSource::Wgsl(HAIR_FOLLICLE_BIND_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hair_follicle_bind_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hair_follicle_bind_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hair_follicle_bind_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairFollicleBind {
            module,
            layout,
            pipeline,
        }
    }

    /// Binds and transfers every follicle in `queries` and returns one
    /// [`HairFollicleBindResult`] per input, in order.
    ///
    /// The continuous outputs match the reference to within the tolerance
    /// documented on this module; the `valid` flag is exact. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HairFollicleBindQuery],
    ) -> Vec<HairFollicleBindResult> {
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
            label: Some("prism_volumetric_hair_follicle_bind_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_follicle_bind_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_follicle_bind_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hair_follicle_bind_bind_group"),
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
            label: Some("prism_volumetric_hair_follicle_bind_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hair_follicle_bind_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hair_follicle_bind_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per follicle, flattened to a 1-D dispatch.
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
