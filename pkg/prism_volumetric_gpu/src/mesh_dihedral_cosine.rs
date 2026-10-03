//! `wgpu` compute twin of the per-interior-edge dihedral geometry golden
//! [`dihedral_cosines`](prism_render_architecture::ray_scene::mesh_dihedral_cosine::dihedral_cosines).
//!
//! The dihedral angle across a shared edge — how sharply its two incident
//! triangles fold — drives several geometry stages: cloth and hair bending
//! energy, feature-aware adaptive tessellation and simplification, and
//! crease-preserving level-of-detail. Rather than the angle itself (whose
//! recovery needs an inverse trigonometric function), the reference exposes the
//! two transcendental-free components that fully determine it:
//!
//! * the `cosine` `dot(n0, n1)` of the two unit incident-face normals, which is
//!   `1` for coplanar faces and decreases toward `-1` as the fold sharpens, and
//! * the `signed_sine` `dot(cross(n0, n1), e)` along the unit edge direction
//!   `e`, whose sign separates a ridge from a valley fold.
//!
//! # What is twinned
//!
//! The whole-mesh reference [`dihedral_cosines`](prism_render_architecture::ray_scene::mesh_dihedral_cosine::dihedral_cosines)
//! first builds an incidence `HashMap` to find each undirected edge's two
//! incident faces; that variable-length topology pass is inherently host work
//! and is **not** twinned. This kernel twins only the per-edge closed form the
//! reference evaluates once two incident faces and the shared edge are known:
//! given each triangle's three vertex positions (in its own winding) and the
//! shared edge's lower and higher endpoints `a` and `b`, it reproduces the
//! reference `face_unit_normal`, the `cosine = dot(n0, n1)` clamped to
//! `[-1, 1]`, and the `signed_sine = dot(cross(n0, n1), e)` clamped to
//! `[-1, 1]`, where `e` is the unit edge direction `b - a`. The caller must
//! pass the two faces in ascending face-index order, exactly as the reference
//! orders them so the signed-sine sign is deterministic.
//!
//! # Degenerate handling
//!
//! The reference skips an edge entirely when either incident face is
//! zero-area (so it has no unit normal). This per-lane kernel cannot drop a
//! lane, so it emits a defined sentinel instead: when either face normalizes
//! from a (near) zero cross product, the lane sets `degenerate = 1` and writes
//! `cosine = 0` and `signed_sine = 0`. When the edge vector itself is
//! degenerate (zero length) the reference leaves `signed_sine = 0` while still
//! recording the edge; this kernel mirrors that fallback exactly.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `cross`, `dot`,
//! `clamp`, `sqrt`, `+ - * /` and comparisons — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `round`, `smoothstep`, `cbrt`, no `u64`/`i64`/`u16`/`i16`/`f64`
//! and no optional device feature. It therefore runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of cross and dot products
//! and two `sqrt`-based normalizations, so `CPU` and `GPU` evaluate the same
//! closed form in the same associativity. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the
//! continuous `cosine` and `signed_sine` values yet an *exact* match on the
//! discrete `degenerate` flag, which is routed through the same zero-area
//! guard the reference uses so a well-conditioned fold folds the identical
//! verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_dihedral_cosine`；
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

/// Discrete flag written by the kernel when either incident face is zero-area
/// and has no unit normal. A direct `f32` equality is forbidden, so the kernel
/// emits this integer flag rather than a sentinel float.
const DEGENERATE_FLAG: u32 = 1;

/// The portable core-`WGSL` per-edge dihedral kernel source. Inlined so the
/// crate ships as a single source file. Mirrors the per-edge block of the `CPU`
/// golden
/// [`dihedral_cosines`](prism_render_architecture::ray_scene::mesh_dihedral_cosine::dihedral_cosines);
/// see the module documentation for the algorithm.
const MESH_DIHEDRAL_COSINE_WGSL: &str = r#"
// Per-interior-edge dihedral geometry twin: one thread per (two faces, shared
// edge) query reproduces the CPU golden
// `ray_scene::mesh_dihedral_cosine` per-edge block. Each face's unit normal is
// the normalized cross product of two of its edge vectors; the cosine is the
// clamped dot of the two normals and the signed sine is the clamped dot of
// their cross with the unit edge direction. It uses only the portable
// core-WGSL subset (cross/dot/clamp/sqrt, + - * / and comparisons) and takes
// no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::mesh_dihedral_cosine；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. Each of the two triangles carries its three vertex positions in
// its own winding (padded to vec4 lanes), followed by the shared edge's lower
// and higher endpoint positions. Face normal uses the triangle's own winding.
struct Query {
    tri0a: vec4<f32>,
    tri0b: vec4<f32>,
    tri0c: vec4<f32>,
    tri1a: vec4<f32>,
    tri1b: vec4<f32>,
    tri1c: vec4<f32>,
    edge_a: vec4<f32>,
    edge_b: vec4<f32>,
}

// One result: the clamped cosine, the clamped signed sine, a degenerate flag
// and a padding word (16-byte stride).
struct Result {
    cosine: f32,
    signed_sine: f32,
    degenerate: u32,
    pad0: u32,
}

// Normalization outcome: the unit vector and whether the input had positive
// squared length (`ok == 1u`) or was (near) zero (`ok == 0u`).
struct Unit {
    v: vec3<f32>,
    ok: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Unit-length form of `v`, or `ok == 0u` when `v` is (near) zero. Mirrors the
// reference `normalize`, which fails when `len_sq <= 0`.
fn normalize_or_fail(v: vec3<f32>) -> Unit {
    var out: Unit;
    let len_sq = dot(v, v);
    if (len_sq <= 0.0) {
        out.v = vec3<f32>(0.0, 0.0, 0.0);
        out.ok = 0u;
        return out;
    }
    let inv = 1.0 / sqrt(len_sq);
    out.v = v * inv;
    out.ok = 1u;
    return out;
}

// Unit normal of the triangle (a, b, c) in its own winding, or `ok == 0u` when
// the triangle is zero-area. Mirrors the reference `face_unit_normal`:
// normalize(cross(b - a, c - a)).
fn face_normal(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> Unit {
    return normalize_or_fail(cross(b - a, c - a));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    let n0 = face_normal(q.tri0a.xyz, q.tri0b.xyz, q.tri0c.xyz);
    let n1 = face_normal(q.tri1a.xyz, q.tri1b.xyz, q.tri1c.xyz);

    var r: Result;
    r.pad0 = 0u;

    if (n0.ok == 0u || n1.ok == 0u) {
        // Either incident face is zero-area: the reference drops the edge; this
        // per-lane kernel emits the defined sentinel instead.
        r.cosine = 0.0;
        r.signed_sine = 0.0;
        r.degenerate = 1u;
        results[idx] = r;
        return;
    }

    let cosine = clamp(dot(n0.v, n1.v), -1.0, 1.0);

    // Edge direction from the lower to the higher endpoint. A zero-length edge
    // leaves the signed sine at 0, matching the reference fallback.
    let edge = normalize_or_fail(q.edge_b.xyz - q.edge_a.xyz);
    var signed_sine = 0.0;
    if (edge.ok == 1u) {
        signed_sine = clamp(dot(cross(n0.v, n1.v), edge.v), -1.0, 1.0);
    }

    r.cosine = cosine;
    r.signed_sine = signed_sine;
    r.degenerate = 0u;
    results[idx] = r;
}
"#;

/// One dihedral query: the two incident triangles' vertex positions (each in
/// its own winding) and the shared edge's lower and higher endpoints.
///
/// Mirrors a single per-edge evaluation inside the reference
/// [`dihedral_cosines`](prism_render_architecture::ray_scene::mesh_dihedral_cosine::dihedral_cosines).
/// `tri0` and `tri1` must be passed in ascending face-index order so the
/// `signed_sine` sign is deterministic, and `edge_a`/`edge_b` must be the
/// shared edge's lower and higher endpoint positions. Derives only
/// [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DihedralCosineQuery {
    /// The first (lower-index) incident triangle's three vertex positions in
    /// its own winding.
    pub tri0: [[f32; 3]; 3],
    /// The second (higher-index) incident triangle's three vertex positions in
    /// its own winding.
    pub tri1: [[f32; 3]; 3],
    /// The shared edge's lower endpoint position.
    pub edge_a: [f32; 3],
    /// The shared edge's higher endpoint position.
    pub edge_b: [f32; 3],
}

impl DihedralCosineQuery {
    /// Builds one dihedral query from the two faces and the shared edge.
    #[must_use]
    pub fn new(
        tri0: [[f32; 3]; 3],
        tri1: [[f32; 3]; 3],
        edge_a: [f32; 3],
        edge_b: [f32; 3],
    ) -> DihedralCosineQuery {
        DihedralCosineQuery {
            tri0,
            tri1,
            edge_a,
            edge_b,
        }
    }
}

/// The evaluated dihedral geometry for one query, the host-side mirror of the
/// kernel's `Result` lane.
///
/// `cosine` is `dot(n0, n1)` clamped to `[-1, 1]` and `signed_sine` is
/// `dot(cross(n0, n1), e)` clamped to `[-1, 1]` along the unit edge direction
/// `e`. `degenerate` is `true` when either incident face is zero-area; then
/// both `cosine` and `signed_sine` are the defined sentinel `0`. Derives only
/// [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DihedralCosineResult {
    /// Cosine `dot(n0, n1)` of the two unit incident-face normals.
    pub cosine: f32,
    /// Signed sine `dot(cross(n0, n1), e)` along the unit edge direction `e`.
    pub signed_sine: f32,
    /// Whether an incident face was zero-area (then `cosine`/`signed_sine` are
    /// the sentinel `0`).
    pub degenerate: bool,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`MESH_DIHEDRAL_COSINE_WGSL`]: the query count and three pad
/// words — `16` bytes, each field at the uniform offset the shader expects.
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

/// One query as uploaded. `128`-byte `std430` stride matching `Query` in the
/// shader: the six triangle vertices and the two edge endpoints, each padded to
/// a `vec4` lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First face vertex `a` in `xyz`; `w` is unused padding.
    tri0a: [f32; 4],
    /// First face vertex `b` in `xyz`; `w` is unused padding.
    tri0b: [f32; 4],
    /// First face vertex `c` in `xyz`; `w` is unused padding.
    tri0c: [f32; 4],
    /// Second face vertex `a` in `xyz`; `w` is unused padding.
    tri1a: [f32; 4],
    /// Second face vertex `b` in `xyz`; `w` is unused padding.
    tri1b: [f32; 4],
    /// Second face vertex `c` in `xyz`; `w` is unused padding.
    tri1c: [f32; 4],
    /// Shared-edge lower endpoint in `xyz`; `w` is unused padding.
    edge_a: [f32; 4],
    /// Shared-edge higher endpoint in `xyz`; `w` is unused padding.
    edge_b: [f32; 4],
}

impl GpuQuery {
    /// Packs a [`DihedralCosineQuery`] into the `std430` upload layout.
    fn from_query(query: &DihedralCosineQuery) -> GpuQuery {
        let pad = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
        GpuQuery {
            tri0a: pad(query.tri0[0]),
            tri0b: pad(query.tri0[1]),
            tri0c: pad(query.tri0[2]),
            tri1a: pad(query.tri1[0]),
            tri1b: pad(query.tri1[1]),
            tri1c: pad(query.tri1[2]),
            edge_a: pad(query.edge_a),
            edge_b: pad(query.edge_b),
        }
    }
}

/// One result as read back. `16`-byte `std430` stride matching `Result` in the
/// shader: the cosine, the signed sine, the degenerate flag and one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Cosine `dot(n0, n1)`.
    cosine: f32,
    /// Signed sine `dot(cross(n0, n1), e)`.
    signed_sine: f32,
    /// Degenerate flag (`1` = an incident face was zero-area).
    degenerate: u32,
    /// Padding word.
    pad0: u32,
}

/// Maps one kernel `Result` lane back to the host [`DihedralCosineResult`].
fn decode_result(raw: &GpuResult) -> DihedralCosineResult {
    DihedralCosineResult {
        cosine: raw.cosine,
        signed_sine: raw.signed_sine,
        degenerate: raw.degenerate == DEGENERATE_FLAG,
    }
}

/// A compiled, reusable per-edge dihedral evaluation pipeline.
pub struct GpuDihedralCosine {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDihedralCosine {
    /// Compiles the per-edge dihedral kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDihedralCosine {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_dihedral_cosine"),
            source: ShaderSource::Wgsl(MESH_DIHEDRAL_COSINE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_dihedral_cosine_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_dihedral_cosine_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_dihedral_cosine_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDihedralCosine {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning one
    /// [`DihedralCosineResult`] per query in input order.
    ///
    /// The returned result for query `q` mirrors the per-edge block of the
    /// reference
    /// [`dihedral_cosines`](prism_render_architecture::ray_scene::mesh_dihedral_cosine::dihedral_cosines)
    /// evaluated on `q`'s two faces and shared edge. An empty `queries` slice
    /// yields an empty result — storage buffers cannot be zero-sized, so it is
    /// handled by an early return before any dispatch.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[DihedralCosineQuery],
    ) -> Vec<DihedralCosineResult> {
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
            label: Some("prism_volumetric_mesh_dihedral_cosine_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_dihedral_cosine_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_dihedral_cosine_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_dihedral_cosine_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_dihedral_cosine_bind_group"),
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
            label: Some("prism_volumetric_mesh_dihedral_cosine_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_dihedral_cosine_pass"),
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
