//! `wgpu` compute twin of the per-edge numeric core inside the triangle-mesh
//! edge-length statistics contract
//! ([`mesh_edge_length_stats`](prism_render_architecture::ray_scene::mesh_edge_length_stats)).
//!
//! The `CPU` golden
//! [`edge_length_stats`](prism_render_architecture::ray_scene::mesh_edge_length_stats::edge_length_stats)
//! turns a triangle mesh into a de-duplicated, sorted list of undirected edges
//! paired with their Euclidean lengths, then answers aggregate queries such as
//! [`count_shorter_than`](prism_render_architecture::ray_scene::mesh_edge_length_stats)
//! and
//! [`count_longer_than`](prism_render_architecture::ray_scene::mesh_edge_length_stats).
//! That whole-mesh reduction is a *variable-length aggregate*: it hashes each
//! triangle's three edges into a `HashSet` so every undirected edge is measured
//! exactly once, sorts the survivors, and reduces them to counts. The dedup,
//! sort and reduction all stay on the host — they are variable-length container
//! work with no fixed-width device analogue.
//!
//! [`GpuMeshEdgeLength`] is the on-device twin of the *numeric core* that
//! aggregate is built from: the length of one edge and its strict classification
//! against a threshold. One thread solves one edge, reproducing the reference's
//! exact closed form, so a passing real-device parity test is direct evidence
//! the ported kernel computes the same length and the same strict comparisons
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For an edge with endpoints `a` and `b` and a `threshold`, the reference
//! `distance` computes `sqrt(dx^2 + dy^2 + dz^2)` with `dx = a.x - b.x` etc.,
//! and the two counters keep an edge whose length `l` is *strictly* shorter
//! (`l < threshold`) or *strictly* longer (`l > threshold`). The twin
//! reproduces each of those quantities per edge: the Euclidean `length`, the
//! `shorter` flag (`1` when `length < threshold`), and the `longer` flag (`1`
//! when `length > threshold`). An edge exactly at the threshold sets neither
//! flag, matching the reference's strict inequalities.
//!
//! # What stays on the host
//!
//! The `HashSet` edge dedup, the `sorted_pair` keying, the ascending sort, the
//! `mean_length` and the count reductions are all variable-length container
//! work that the host owns; the device never sees them. The host enqueues one
//! query per edge it wants measured, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! The `length` is continuous and threads through a `sqrt`, so `CPU` and `GPU`
//! are not bit-exact: a `GPU` `sqrt` may land a few units in the last place from
//! the scalar reference. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on it. The `shorter` and `longer`
//! flags are discrete decisions built from strict ordered comparisons; they
//! must agree exactly. To keep both sides on the same side of each comparison,
//! the fixtures and the random sweep keep `|length - threshold|` well clear of
//! zero before asserting the flags; a separate named fixture exercises the exact
//! at-threshold tie where both flags are `0`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `select`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `round`, no `floor`, no
//! `f32` `%` and no `powi` (each square is an explicit multiply). No optional
//! device feature is required, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. There is no loop: each thread performs a fixed, bounded sequence of
//! arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_edge_length_stats`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` per-edge length-and-classification kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden per-edge closed form; see the module
/// documentation for the algorithm.
const MESH_EDGE_LENGTH_WGSL: &str = r#"
// Mesh edge-length per-edge twin: one thread computes one undirected edge's
// Euclidean length and its strict classification against a threshold, mirroring
// the CPU golden `ray_scene::mesh_edge_length_stats` closed form with only
// sqrt/select and + - * /. It owns no HashSet dedup, no sort and no count
// reduction; those variable-length aggregates stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::mesh_edge_length_stats；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of edges in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Endpoint a (ax, ay, az).
    ax: f32,
    ay: f32,
    az: f32,
    // Endpoint b (bx, by, bz).
    bx: f32,
    by: f32,
    bz: f32,
    // The classification threshold.
    threshold: f32,
    pad0: f32,
}

struct Result {
    // Euclidean edge length |a - b|.
    length: f32,
    // 1 when length < threshold (strict), else 0.
    shorter: u32,
    // 1 when length > threshold (strict), else 0.
    longer: u32,
    pad0: u32,
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

    // Per-axis differences, matching the golden `distance` helper exactly.
    let dx = q.ax - q.bx;
    let dy = q.ay - q.by;
    let dz = q.az - q.bz;

    // Euclidean length: each square is an explicit multiply (no banned powi),
    // and the radicand is a sum of squares so it is never negative.
    let length = sqrt(dx * dx + dy * dy + dz * dz);

    var out: Result;
    out.length = length;
    // Strict ordered comparisons mirror the golden `count_shorter_than` /
    // `count_longer_than` filters; no f32 `==` is used.
    out.shorter = select(0u, 1u, length < q.threshold);
    out.longer = select(0u, 1u, length > q.threshold);
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the edge count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MESH_EDGE_LENGTH_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid edges in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one edge query: two endpoints and a threshold
/// plus one pad word to a `32`-byte stride, matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Endpoint `a.x`.
    ax: f32,
    /// Endpoint `a.y`.
    ay: f32,
    /// Endpoint `a.z`.
    az: f32,
    /// Endpoint `b.x`.
    bx: f32,
    /// Endpoint `b.y`.
    by: f32,
    /// Endpoint `b.z`.
    bz: f32,
    /// The classification threshold.
    threshold: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one edge result, matching the `WGSL` `Result`
/// struct: a length, two flags, and one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Euclidean edge length `|a - b|`.
    length: f32,
    /// `1` when `length < threshold`, else `0`.
    shorter: u32,
    /// `1` when `length > threshold`, else `0`.
    longer: u32,
    /// Padding word.
    pad0: u32,
}

/// One per-edge query for the mesh edge-length twin: the two endpoints `a` and
/// `b` of the undirected edge and the classification `threshold`.
///
/// The host owns the surrounding aggregate — the `HashSet` edge dedup, the
/// sort, and the count reductions — and enqueues one [`MeshEdgeLengthQuery`] per
/// edge it wants measured, matching the reference `edge_length_stats` edge walk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshEdgeLengthQuery {
    /// Endpoint `a` of the edge.
    pub a: [f32; 3],
    /// Endpoint `b` of the edge.
    pub b: [f32; 3],
    /// The classification threshold the length is compared against.
    pub threshold: f32,
}

impl MeshEdgeLengthQuery {
    /// Builds a query for the edge `a`-`b` classified against `threshold`.
    #[must_use]
    pub const fn new(a: [f32; 3], b: [f32; 3], threshold: f32) -> MeshEdgeLengthQuery {
        MeshEdgeLengthQuery { a, b, threshold }
    }
}

/// One resolved edge of the mesh edge-length twin, mirroring the length and the
/// strict threshold classification the reference `distance` and the
/// `count_shorter_than` / `count_longer_than` filters produce.
///
/// `shorter` is `true` when `length < threshold` and `longer` is `true` when
/// `length > threshold`; an edge exactly at the threshold sets neither, matching
/// the reference's strict inequalities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshEdgeLengthResult {
    /// Euclidean edge length `|a - b|`.
    pub length: f32,
    /// `true` when `length < threshold` (strict).
    pub shorter: bool,
    /// `true` when `length > threshold` (strict).
    pub longer: bool,
}

/// Encodes one [`MeshEdgeLengthQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &MeshEdgeLengthQuery) -> GpuQuery {
    GpuQuery {
        ax: q.a[0],
        ay: q.a[1],
        az: q.a[2],
        bx: q.b[0],
        by: q.b[1],
        bz: q.b[2],
        threshold: q.threshold,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`MeshEdgeLengthResult`],
/// turning the flag words back into [`bool`]s.
fn decode_result(raw: &GpuResult) -> MeshEdgeLengthResult {
    MeshEdgeLengthResult {
        length: raw.length,
        shorter: raw.shorter != 0,
        longer: raw.longer != 0,
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

/// A compiled, reusable mesh edge-length per-edge compute pipeline, twinning the
/// numeric core of the `CPU` golden
/// [`mesh_edge_length_stats`](prism_render_architecture::ray_scene::mesh_edge_length_stats).
pub struct GpuMeshEdgeLength {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMeshEdgeLength {
    /// Compiles the mesh edge-length per-edge kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMeshEdgeLength {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_edge_length"),
            source: ShaderSource::Wgsl(MESH_EDGE_LENGTH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_edge_length_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_edge_length_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_edge_length_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMeshEdgeLength {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every edge in `queries` and returns one [`MeshEdgeLengthResult`]
    /// per input, in order.
    ///
    /// The `length` matches the reference to within the tolerance documented on
    /// this module; the `shorter` and `longer` flags equal the reference exactly
    /// for edges clear of the at-threshold tie. An empty `queries` batch returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MeshEdgeLengthQuery],
    ) -> Vec<MeshEdgeLengthResult> {
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
            label: Some("prism_volumetric_mesh_edge_length_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_edge_length_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_edge_length_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_edge_length_bind_group"),
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
            label: Some("prism_volumetric_mesh_edge_length_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_edge_length_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_edge_length_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per edge, flattened to a 1-D dispatch.
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
