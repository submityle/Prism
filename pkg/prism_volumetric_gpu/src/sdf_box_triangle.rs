//! `wgpu` compute twin of two analytic signed-distance primitives from the
//! `CPU` golden path
//! (`prism_render_architecture::ray_scene::sdf_primitives`): the exact
//! axis-aligned box distance `box_sdf` and the exact triangle distance
//! `triangle_sdf`.
//!
//! Implicit modelling and procedural ray-marching evaluate analytic primitives
//! whose exact Euclidean distance is known in closed form rather than sampling
//! a baked grid. This module is the on-device twin of two of those atoms:
//!
//! - `box_sdf`: the signed distance to an axis-aligned box of a given
//!   half-extent, negative inside and positive outside, using the exact
//!   interior/exterior split so the field is a true distance on both sides.
//! - `triangle_sdf`: Inigo Quilez's exact `udTriangle` unsigned distance to a
//!   triangular patch, partitioning space into the face-interior region (a
//!   perpendicular projection onto the plane) versus an edge/vertex region (the
//!   minimum distance to the clamped foot on each of the three edges).
//!
//! [`GpuSdfBoxTriangle`] evaluates both for one query per thread, reproducing
//! the reference closed forms with only `abs`, `min`, `max`, `clamp`, dot and
//! cross products, a sign test and a final `sqrt` — no transcendental — so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same distances the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfBoxTriangleQuery`] — the query point, the box
//! half-extent and the three triangle vertices — and writes one
//! [`SdfBoxTriangleResult`] holding the signed box distance and the unsigned
//! triangle distance. The box kernel folds the point into the first octant by
//! `abs`, subtracts the half-extent, and sums the positive exterior overshoot
//! length with the clamped interior term. The triangle kernel forms the three
//! edge vectors and the face normal `nor`, sums the three edge-sign tests to
//! classify the query, then takes either the squared perpendicular projection
//! or the minimum squared distance to the three clamped edge feet before a
//! single `sqrt`.
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these atoms into complex shapes,
//! the mesh baker, and any acceleration structure all stay on the host; the
//! device sees only the stateless, fixed-width distance evaluation, one query
//! at a time, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Both outputs thread through products, quotients and a `sqrt`, so the `CPU`
//! and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units
//! in the last place from the scalar reference. The parity test asserts each
//! distance within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` with a relative
//! floor of `1e-6`, tight enough to catch a genuinely wrong port yet loose
//! enough to admit a legal last-place difference. The triangle distance is
//! continuous across the face/edge classification boundary (the perpendicular
//! foot lands on an edge there, where both branches agree), so a branch
//! disagreement near that boundary cannot produce a distance cliff; the test
//! keeps triangles non-degenerate so the face branch's divide by the squared
//! normal stays well-conditioned.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `dot`, `cross`, `length`, `sqrt`, `select`, `+ - * /` and unsigned
//! index arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no
//! inverse trigonometry, no `round` and no `ceil`, and no `f64`/`u64`/`u16`/
//! `i64`/`i16`. It runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene/sdf_primitives.rs`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` box/triangle signed-distance kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// `prism_render_architecture::ray_scene::sdf_primitives::box_sdf` and
/// `triangle_sdf` closed forms; see the module documentation for the algorithm.
const SDF_BOX_TRIANGLE_WGSL: &str = r#"
// Box/triangle signed-distance twin: one thread computes one query's exact
// axis-aligned box distance and the exact unsigned triangle distance, mirroring
// the CPU golden `ray_scene::sdf_primitives::{box_sdf, triangle_sdf}` with only
// abs, min, max, clamp, dot, cross, length, select and a final sqrt. The CSG
// operators and the mesh baker stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene/sdf_primitives.rs；无
// 第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point.
    px: f32,
    py: f32,
    pz: f32,
    // Box half-extent (centred at the origin).
    hx: f32,
    hy: f32,
    hz: f32,
    // Triangle vertex a.
    ax: f32,
    ay: f32,
    az: f32,
    // Triangle vertex b.
    bx: f32,
    by: f32,
    bz: f32,
    // Triangle vertex c.
    cx: f32,
    cy: f32,
    cz: f32,
    pad0: f32,
}

struct Distances {
    // Signed distance to the axis-aligned box (negative inside).
    box_distance: f32,
    // Unsigned distance to the triangular patch.
    triangle_distance: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Distances>;

// Rust `f32::signum` reimplemented for the branch test: +1 for a non-negative
// argument, -1 otherwise. The reference's edge/vertex classification only cares
// about the sign of each edge test, and the per-test argument is kept clear of
// zero by the fixtures, so the behaviour at an exact zero never governs.
fn sign_of(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

// Squared length of a 3-vector.
fn dot2(v: vec3<f32>) -> f32 {
    return dot(v, v);
}

// Exact signed distance from `point` to an axis-aligned box of `half_extent`,
// centred at the origin: the positive exterior overshoot length plus the
// clamped interior term.
fn box_sdf(point: vec3<f32>, half_extent: vec3<f32>) -> f32 {
    let q = abs(point) - half_extent;
    let outside = length(max(q, vec3<f32>(0.0, 0.0, 0.0)));
    let inside = min(max(q.x, max(q.y, q.z)), 0.0);
    return outside + inside;
}

// Exact unsigned distance from `point` to the triangle (a, b, c): Inigo
// Quilez's udTriangle. Three edge-sign tests classify the query into the
// edge/vertex region (minimum distance to the clamped foot on each edge) versus
// the face-interior region (perpendicular projection onto the plane).
fn triangle_sdf(point: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let ba = b - a;
    let pa = point - a;
    let cb = c - b;
    let pb = point - b;
    let ac = a - c;
    let pc = point - c;
    let nor = cross(ba, ac);

    let edge_sum = sign_of(dot(cross(ba, nor), pa))
        + sign_of(dot(cross(cb, nor), pb))
        + sign_of(dot(cross(ac, nor), pc));

    var squared: f32;
    if (edge_sum < 2.0) {
        let t0 = clamp(dot(ba, pa) / dot2(ba), 0.0, 1.0);
        let e0 = dot2(ba * t0 - pa);
        let t1 = clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0);
        let e1 = dot2(cb * t1 - pb);
        let t2 = clamp(dot(ac, pc) / dot2(ac), 0.0, 1.0);
        let e2 = dot2(ac * t2 - pc);
        squared = min(e0, min(e1, e2));
    } else {
        let np = dot(nor, pa);
        squared = np * np / dot2(nor);
    }
    return sqrt(squared);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let point = vec3<f32>(q.px, q.py, q.pz);
    let half_extent = vec3<f32>(q.hx, q.hy, q.hz);
    let a = vec3<f32>(q.ax, q.ay, q.az);
    let b = vec3<f32>(q.bx, q.by, q.bz);
    let c = vec3<f32>(q.cx, q.cy, q.cz);

    var out: Distances;
    out.box_distance = box_sdf(point, half_extent);
    out.triangle_distance = triangle_sdf(point, a, b, c);
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_BOX_TRIANGLE_WGSL`].
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
/// the query point, the box half-extent and the three triangle vertices
/// flattened to scalar `f32` lanes (avoiding `vec3` alignment padding), plus one
/// pad word to a `64`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Query point `z`.
    pz: f32,
    /// Box half-extent `x`.
    hx: f32,
    /// Box half-extent `y`.
    hy: f32,
    /// Box half-extent `z`.
    hz: f32,
    /// Triangle vertex `a` component `x`.
    ax: f32,
    /// Triangle vertex `a` component `y`.
    ay: f32,
    /// Triangle vertex `a` component `z`.
    az: f32,
    /// Triangle vertex `b` component `x`.
    bx: f32,
    /// Triangle vertex `b` component `y`.
    by: f32,
    /// Triangle vertex `b` component `z`.
    bz: f32,
    /// Triangle vertex `c` component `x`.
    cx: f32,
    /// Triangle vertex `c` component `y`.
    cy: f32,
    /// Triangle vertex `c` component `z`.
    cz: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: the signed box distance and the unsigned triangle distance, plus two
/// pad words to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed distance to the axis-aligned box.
    box_distance: f32,
    /// Unsigned distance to the triangular patch.
    triangle_distance: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One query for the box/triangle distance twin: the query point, the box
/// half-extent and the three triangle vertices.
///
/// `point` is the position whose distance is sought; `half_extent` is the box's
/// half-size per axis, centred at the origin; `a`, `b`, `c` are the triangle
/// vertices. The host owns the `CSG` composition and enqueues one
/// [`SdfBoxTriangleQuery`] per distance evaluation it needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfBoxTriangleQuery {
    /// Query point.
    pub point: [f32; 3],
    /// Box half-extent, centred at the origin.
    pub half_extent: [f32; 3],
    /// Triangle vertex `a`.
    pub a: [f32; 3],
    /// Triangle vertex `b`.
    pub b: [f32; 3],
    /// Triangle vertex `c`.
    pub c: [f32; 3],
}

impl SdfBoxTriangleQuery {
    /// Builds a query from the point, box half-extent and triangle vertices.
    #[must_use]
    pub const fn new(
        point: [f32; 3],
        half_extent: [f32; 3],
        a: [f32; 3],
        b: [f32; 3],
        c: [f32; 3],
    ) -> SdfBoxTriangleQuery {
        SdfBoxTriangleQuery {
            point,
            half_extent,
            a,
            b,
            c,
        }
    }
}

/// One resolved query of the box/triangle distance twin: the signed box
/// distance and the unsigned triangle distance.
///
/// `box_distance` is
/// `prism_render_architecture::ray_scene::sdf_primitives::box_sdf` (negative
/// inside, positive outside); `triangle_distance` is `triangle_sdf` (always
/// non-negative, zero on the patch).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfBoxTriangleResult {
    /// Signed distance to the axis-aligned box.
    pub box_distance: f32,
    /// Unsigned distance to the triangular patch.
    pub triangle_distance: f32,
}

/// Encodes one [`SdfBoxTriangleQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfBoxTriangleQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        hx: q.half_extent[0],
        hy: q.half_extent[1],
        hz: q.half_extent[2],
        ax: q.a[0],
        ay: q.a[1],
        az: q.a[2],
        bx: q.b[0],
        by: q.b[1],
        bz: q.b[2],
        cx: q.c[0],
        cy: q.c[1],
        cz: q.c[2],
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfBoxTriangleResult`].
fn decode_result(raw: &GpuResult) -> SdfBoxTriangleResult {
    SdfBoxTriangleResult {
        box_distance: raw.box_distance,
        triangle_distance: raw.triangle_distance,
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

/// A compiled, reusable box/triangle distance compute pipeline, twinning the
/// `CPU` golden
/// `prism_render_architecture::ray_scene::sdf_primitives::box_sdf` and
/// `triangle_sdf`.
pub struct GpuSdfBoxTriangle {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfBoxTriangle {
    /// Compiles the box/triangle distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfBoxTriangle {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_box_triangle"),
            source: ShaderSource::Wgsl(SDF_BOX_TRIANGLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_box_triangle_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_box_triangle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_box_triangle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfBoxTriangle {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SdfBoxTriangleResult`] per input, in order.
    ///
    /// The distances match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfBoxTriangleQuery],
    ) -> Vec<SdfBoxTriangleResult> {
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
            label: Some("prism_volumetric_sdf_box_triangle_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_box_triangle_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_box_triangle_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_box_triangle_bind_group"),
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
            label: Some("prism_volumetric_sdf_box_triangle_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_box_triangle_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_box_triangle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
