//! `wgpu` compute twin of the per-triangle shape-quality metric
//! (`prism_render_architecture::ray_scene::mesh_triangle_quality`).
//!
//! Meshing, tessellation and simulation all degrade on *sliver* triangles —
//! long thin faces whose tiny area relative to their edge lengths wrecks
//! interpolation, shadow bias and numerical conditioning. The canonical
//! scale-invariant score is the **normalized shape quality** (the mean-ratio
//! metric):
//!
//! ```text
//! q = 4 * sqrt(3) * A / (l0^2 + l1^2 + l2^2)
//! ```
//!
//! where `A` is the triangle area and `l0..l2` its edge lengths. It equals `1`
//! for an equilateral triangle and falls toward `0` as the triangle
//! degenerates. The `CPU` golden `triangle_quality` evaluates this per face in
//! one pass and then reduces to whole-mesh totals; this twin reproduces only
//! the **single-triangle core**: three vertices in, `(area, quality,
//! degenerate)` out, running one thread per triangle. The whole-mesh reduction
//! (total area, min / average quality, sliver counts) stays on the host and is
//! not twinned here.
//!
//! # What is twinned
//!
//! For one triangle `(v0, v1, v2)` the kernel forms the edge vectors
//! `e0 = v1 - v0`, `e1 = v2 - v1`, `e2 = v0 - v2`, the doubled-area cross
//! product `e0 x (v2 - v0)`, the area `A = 0.5 * |cross|`, and the squared edge
//! sum `sum_sq = e0.e0 + e1.e1 + e2.e2`. When `sum_sq > 0` the raw score is
//! `q_raw = 4 * sqrt(3) * A / sum_sq`, clamped to `[0, 1]` exactly as the
//! reference. A triangle whose vertices coincide (`sum_sq <= 0`) or whose score
//! collapses (`q_raw <= `[`QUALITY_EPS`]) is reported as *degenerate*: its
//! `degenerate` flag is `1` and its `area` and `quality` are forced to `0`.
//!
//! The area uses one `sqrt` of the cross-product magnitude and the `sqrt(3)`
//! numerator is itself a `sqrt`; everything else is squared edge lengths and
//! dot / cross products, so no `f32` transcendental function appears.
//!
//! # Parity
//!
//! The companion test reruns an independent host reimplementation of the same
//! closed form (it does not depend on the golden crate) and asserts the kernel
//! reproduces every lane: `area` and `quality` within `abs <= 1e-4` or
//! `rel <= 1e-3` (relative floor `1e-6`), and the discrete `degenerate` flag
//! for exact equality.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_triangle_quality`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` per-triangle quality kernel, embedded inline so the
/// twin ships as a single source file. Mirrors the `CPU` golden
/// `triangle_quality` single-triangle body step for step; see the module
/// documentation for the metric.
const MESH_TRIANGLE_QUALITY_WGSL: &str = r#"
// Per-triangle shape-quality twin: one thread per triangle computes the area
// A = 0.5*|e0 x (v2 - v0)| and the normalized mean-ratio score
// q = 4*sqrt(3)*A / (|e0|^2 + |e1|^2 + |e2|^2), clamped to [0, 1], mirroring
// the CPU golden `ray_scene::mesh_triangle_quality::triangle_quality` body. It
// uses only the portable core-WGSL subset (dot/cross, two sqrt and + - * /),
// takes no optional feature, and so runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::mesh_triangle_quality;
// no third-party engine source or derived code.

struct Params {
    // Number of valid triangles in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query triangle. 48-byte std430 stride matching the host `GpuQuery`: three
// vertices each padded to a vec4 so the storage array needs no manual vec3
// alignment arithmetic. The w lanes are unused padding.
struct Query {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
}

// One result. 16-byte std430 stride matching the host `GpuResult`: the area,
// the normalized quality, the degenerate flag as 0u/1u and one pad word.
struct Result {
    area: f32,
    quality: f32,
    degenerate: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Scale-invariant floor classifying a triangle as degenerate: when the clamped
// numerator 4*sqrt(3)*A/sum_sq falls at or below this, the face is treated as
// zero-area (collinear) and its area and quality are forced to zero. The host
// oracle shares the same floor so both pick the identical branch.
const QUALITY_EPS: f32 = 1.0e-6;

@compute @workgroup_size(64)
fn measure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    let v0 = q.v0.xyz;
    let v1 = q.v1.xyz;
    let v2 = q.v2.xyz;

    // Edge vectors, matching the reference ordering.
    let e0 = v1 - v0;
    let e1 = v2 - v1;
    let e2 = v0 - v2;

    // Doubled-area cross product e0 x (v2 - v0) == e0 x (-e2).
    let crs = cross(e0, v2 - v0);
    let cross_mag = sqrt(dot(crs, crs));
    let area = 0.5 * cross_mag;

    let sum_sq = dot(e0, e0) + dot(e1, e1) + dot(e2, e2);

    // sqrt(3) numerator of the normalized mean-ratio metric.
    let sqrt3 = sqrt(3.0);

    var res: Result;
    res.area = 0.0;
    res.quality = 0.0;
    res.degenerate = 1u;
    res.pad0 = 0u;

    if (sum_sq > 0.0) {
        let q_raw = 4.0 * sqrt3 * area / sum_sq;
        if (q_raw > QUALITY_EPS) {
            res.area = area;
            res.quality = clamp(q_raw, 0.0, 1.0);
            res.degenerate = 0u;
        }
    }

    results[idx] = res;
}
"#;

/// One triangle to score: its three vertices.
///
/// Mirrors a single reference `triangle_quality` face evaluation. Derives only
/// [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshTriangleQualityQuery {
    /// First triangle vertex.
    pub v0: [f32; 3],
    /// Second triangle vertex.
    pub v1: [f32; 3],
    /// Third triangle vertex.
    pub v2: [f32; 3],
}

/// The scored result for one triangle, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `area` is the triangle area and `quality` the normalized mean-ratio score in
/// `[0, 1]` (`1` = equilateral). `degenerate` is `1` for a coincident or
/// collinear (effectively zero-area) face, in which case `area` and `quality`
/// are both `0`. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds
/// `f32` fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshTriangleQualityResult {
    /// Triangle area; `0` when `degenerate`.
    pub area: f32,
    /// Normalized mean-ratio quality in `[0, 1]`; `0` when `degenerate`.
    pub quality: f32,
    /// Whether the triangle is degenerate (`1` = coincident/collinear).
    pub degenerate: u32,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`MESH_TRIANGLE_QUALITY_WGSL`]: the triangle count and three pad
/// words — `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid triangles.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One triangle as uploaded. `48`-byte `std430` stride matching `Query` in the
/// shader: the three vertices, each padded to a `vec4` lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Vertex `v0` in `xyz`; the `w` lane is unused padding.
    v0: [f32; 4],
    /// Vertex `v1` in `xyz`; the `w` lane is unused padding.
    v1: [f32; 4],
    /// Vertex `v2` in `xyz`; the `w` lane is unused padding.
    v2: [f32; 4],
}

/// Packs a [`MeshTriangleQualityQuery`] into the `std430` upload layout.
fn encode_query(q: &MeshTriangleQualityQuery) -> GpuQuery {
    GpuQuery {
        v0: [q.v0[0], q.v0[1], q.v0[2], 0.0],
        v1: [q.v1[0], q.v1[1], q.v1[2], 0.0],
        v2: [q.v2[0], q.v2[1], q.v2[2], 0.0],
    }
}

/// One result as read back. `16`-byte `std430` stride matching `Result` in the
/// shader: the area, the quality, the degenerate flag and one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Triangle area.
    area: f32,
    /// Normalized mean-ratio quality.
    quality: f32,
    /// Degenerate flag (`1` = degenerate).
    degenerate: u32,
    /// Padding word.
    pad0: u32,
}

/// Maps one kernel `Result` lane back to the host [`MeshTriangleQualityResult`].
fn decode_result(raw: &GpuResult) -> MeshTriangleQualityResult {
    MeshTriangleQualityResult {
        area: raw.area,
        quality: raw.quality,
        degenerate: raw.degenerate,
    }
}

/// A compiled, reusable per-triangle quality pipeline.
pub struct GpuMeshTriangleQuality {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMeshTriangleQuality {
    /// Compiles the per-triangle quality kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMeshTriangleQuality {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_triangle_quality"),
            source: ShaderSource::Wgsl(MESH_TRIANGLE_QUALITY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_triangle_quality_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_triangle_quality_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_triangle_quality_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("measure"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMeshTriangleQuality {
            module,
            layout,
            pipeline,
        }
    }

    /// Scores every triangle in `queries`, returning one
    /// [`MeshTriangleQualityResult`] per triangle in input order.
    ///
    /// The returned result for triangle `q` mirrors the reference
    /// `triangle_quality` single-face body evaluated on `q`'s vertices. An empty
    /// `queries` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so it is handled by an early return before any dispatch.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MeshTriangleQualityQuery],
    ) -> Vec<MeshTriangleQualityResult> {
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
            label: Some("prism_volumetric_mesh_triangle_quality_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_triangle_quality_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_triangle_quality_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_triangle_quality_bind_group"),
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
            label: Some("prism_volumetric_mesh_triangle_quality_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_triangle_quality_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_triangle_quality_pass"),
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
        debug_assert_eq!(raw.len(), count);

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
