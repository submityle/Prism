//! `wgpu` compute twin of three analytic two-dimensional cross/polygon
//! signed-distance primitives from the `CPU` golden path
//! (`prism_render_architecture::ray_scene::sdf_primitives`): the Inigo Quilez
//! `rounded_x`, `rounded_cross_2d` and `polygon_2d` closed forms.
//!
//! Vector graphics, UI glyph masks, procedural signage and extruded profile
//! modelling evaluate these analytic outlines whose exact Euclidean distance is
//! known in closed form rather than sampling a baked field. This module is the
//! on-device twin of three of those atoms, each a true distance built from
//! folds, clamps, a sign test and a `sqrt`:
//!
//! - `rounded_x`: a rounded diagonal cross (an "X"); the query folds to the
//!   first quadrant, projects onto the `y = x` skeleton segment clamped to the
//!   arm half-length `w / 2`, and insets by the stroke radius `r`.
//! - `rounded_cross_2d`: a rounded plus/cross of vertical reach `h` whose
//!   horizontal arms reach `x = +-1`; the first-quadrant fold routes the point
//!   to the circular fillet arc below the tip line or to the nearer convex tip.
//! - `polygon_2d`: the exact signed distance to a filled simple polygon of up
//!   to eight vertices; every edge contributes its clamped point-to-segment
//!   distance and an even-odd crossing test flips the running sign, so convex
//!   and concave outlines resolve alike, winding-independent.
//!
//! [`GpuSdfMisc2d`] evaluates all three for one query per thread, reproducing
//! the reference closed forms with only `abs`, `min`, `max`, `clamp`, a sign
//! test and a final `sqrt` — no transcendental — so a passing real-device
//! parity test is direct evidence the ported kernel computes the same distances
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfMisc2dQuery`] — the shared query point, the
//! `rounded_x` arm width and stroke radius, the `rounded_cross_2d` vertical
//! reach, and the `polygon_2d` vertex array with its valid vertex count — and
//! writes one [`SdfMisc2dResult`] holding the three signed distances. The
//! polygon is carried as a fixed-width array of eight vertex slots plus a
//! `polygon_vertex_count`; only the first `polygon_vertex_count` slots are read,
//! matching the reference's variable-length iteration and modular edge index.
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these outlines, the `extrude`
//! lift to three dimensions, any dynamic polygon longer than eight vertices,
//! and any acceleration structure all stay on the host; the device sees only
//! the stateless, fixed-width distance evaluation, one query at a time, so a
//! storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Every output threads through products, quotients and a `sqrt`, so the `CPU`
//! and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units
//! in the last place from the scalar reference. The parity test asserts each
//! distance within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` with a relative
//! floor of `1e-6`. Where `polygon_2d` flips its running sign the governing
//! locus is a polygon edge where the distance passes through zero, and where
//! `rounded_cross_2d` switches between its fillet and tip branches the two
//! branches meet continuously, so the field stays continuous; the randomized
//! sweep still rejects samples near each branch, fold or boundary to keep the
//! comparison far from any such edge.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `select`, `sqrt`, `bitcast`, `+ - * /`, unsigned index arithmetic
//! and a bounded `for` loop — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`,
//! no inverse trigonometry, no `round` and no `ceil`, and no
//! `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。
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

/// Fixed upper bound on polygon vertices carried by one query. The reference
/// `polygon_2d` accepts an arbitrary slice; the device twin carries a
/// fixed-width array and reads only the first `polygon_vertex_count` slots.
const MAX_POLYGON_VERTS: usize = 8;

/// The portable core-`WGSL` kernel for the three two-dimensional cross/polygon
/// signed-distance outlines, embedded inline so the twin ships as a single
/// source file. The single entry point `solve` runs one thread per query.
const SDF_MISC2D_WGSL: &str = r#"
// 2D cross/polygon signed-distance twin: one thread computes one query's
// rounded-X, rounded-cross and polygon distances, mirroring the CPU golden
// ray_scene::sdf_primitives::{rounded_x, rounded_cross_2d, polygon_2d} with
// only abs, min, max, clamp, a sign test and a final sqrt. The CSG/extrude
// operators and any longer polygon stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_primitives；无
// 第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Shared query point.
    px: f32,
    py: f32,
    // Rounded X: arm width and stroke radius.
    rx_width: f32,
    rx_radius: f32,
    // Rounded cross: vertical reach h.
    rc_height: f32,
    // Polygon: number of valid vertex slots (0..=8).
    poly_count: u32,
    pad0: f32,
    pad1: f32,
    // Polygon vertices flattened as [x0, y0, x1, y1, ...] over eight slots.
    poly: array<f32, 16>,
}

struct Distances {
    // Signed distance to the rounded X.
    rounded_x: f32,
    // Signed distance to the rounded cross.
    rounded_cross: f32,
    // Signed distance to the filled polygon.
    polygon: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Distances>;

// Euclidean length of a 2-vector.
fn length2(v: vec2<f32>) -> f32 {
    return sqrt(v.x * v.x + v.y * v.y);
}

// Exact signed distance to a rounded diagonal cross (an "X") of arm width
// `width` and stroke radius `radius`.
fn rounded_x(point: vec2<f32>, width: f32, radius: f32) -> f32 {
    let p = vec2<f32>(abs(point.x), abs(point.y));
    let m = min(p.x + p.y, width) * 0.5;
    return length2(vec2<f32>(p.x - m, p.y - m)) - radius;
}

// Exact signed distance to a rounded plus/cross of vertical reach `height`
// whose horizontal arms reach x = +-1.
fn rounded_cross_2d(point: vec2<f32>, height: f32) -> f32 {
    let k = 0.5 * (height + 1.0 / height);
    let p = vec2<f32>(abs(point.x), abs(point.y));
    if (p.x < 1.0 && p.y < p.x * (k - height) + height) {
        return k - length2(vec2<f32>(p.x - 1.0, p.y - k));
    }
    return min(
        length2(vec2<f32>(p.x, p.y - height)),
        length2(vec2<f32>(p.x - 1.0, p.y)),
    );
}

// Exact signed distance to a filled simple polygon of `count` vertices stored
// flat in `verts` as [x0, y0, x1, y1, ...]. Even-odd crossing flips the sign.
fn polygon_2d(point: vec2<f32>, verts: array<f32, 16>, count: u32) -> f32 {
    var vv = verts;
    if (count == 0u) {
        // Fewer than one vertex bounds no interior; mirror the reference's
        // positive-infinity sentinel via a bit pattern (no transcendental).
        return bitcast<f32>(0x7f800000u);
    }
    let v0 = vec2<f32>(vv[0], vv[1]);
    let w0 = point - v0;
    var d = w0.x * w0.x + w0.y * w0.y;
    var s = 1.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let j = (i + count - 1u) % count;
        let vi = vec2<f32>(vv[2u * i], vv[2u * i + 1u]);
        let vj = vec2<f32>(vv[2u * j], vv[2u * j + 1u]);
        let e = vj - vi;
        let w = point - vi;
        let dot_ee = e.x * e.x + e.y * e.y;
        var t = 0.0;
        if (dot_ee > 0.0) {
            t = clamp((e.x * w.x + e.y * w.y) / dot_ee, 0.0, 1.0);
        }
        let b = vec2<f32>(w.x - e.x * t, w.y - e.y * t);
        d = min(d, b.x * b.x + b.y * b.y);
        let c0 = point.y >= vi.y;
        let c1 = point.y < vj.y;
        let c2 = e.x * w.y > e.y * w.x;
        if ((c0 && c1 && c2) || (!c0 && !c1 && !c2)) {
            s = -s;
        }
    }
    return s * sqrt(d);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let point = vec2<f32>(q.px, q.py);

    var out: Distances;
    out.rounded_x = rounded_x(point, q.rx_width, q.rx_radius);
    out.rounded_cross = rounded_cross_2d(point, q.rc_height);
    out.polygon = polygon_2d(point, q.poly, q.poly_count);
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_MISC2D_WGSL`].
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
/// the shared query point, the per-shape scalars and the flattened polygon
/// vertex array.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Rounded-X arm width.
    rx_width: f32,
    /// Rounded-X stroke radius.
    rx_radius: f32,
    /// Rounded-cross vertical reach.
    rc_height: f32,
    /// Number of valid polygon vertices.
    poly_count: u32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Polygon vertices flattened as `[x0, y0, x1, y1, ...]` over eight slots.
    poly: [f32; 16],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: the three signed distances to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed distance to the rounded X.
    rounded_x: f32,
    /// Signed distance to the rounded cross.
    rounded_cross: f32,
    /// Signed distance to the filled polygon.
    polygon: f32,
    /// Padding word.
    pad0: f32,
}

/// One query for the two-dimensional cross/polygon distance twin: the shared
/// query point and the per-shape parameters.
///
/// `point` is the position whose distances are sought. `rounded_x_width` and
/// `rounded_x_radius` are the `rounded_x` arm width and stroke radius.
/// `rounded_cross_height` is the `rounded_cross_2d` vertical reach `h`
/// (positive). `polygon_vertex_count` is the number of valid entries in
/// `polygon_vertices` (`0..=8`); the reference `polygon_2d` is evaluated over
/// exactly that many leading vertices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfMisc2dQuery {
    /// Shared query point.
    pub point: [f32; 2],
    /// Rounded-X arm width.
    pub rounded_x_width: f32,
    /// Rounded-X stroke radius.
    pub rounded_x_radius: f32,
    /// Rounded-cross vertical reach `h`.
    pub rounded_cross_height: f32,
    /// Number of valid polygon vertices (`0..=8`).
    pub polygon_vertex_count: u32,
    /// Polygon vertices; only the first `polygon_vertex_count` are read.
    pub polygon_vertices: [[f32; 2]; MAX_POLYGON_VERTS],
}

/// One resolved query of the two-dimensional cross/polygon distance twin: the
/// three signed distances.
///
/// `rounded_x` is
/// `prism_render_architecture::ray_scene::sdf_primitives::rounded_x`,
/// `rounded_cross` is `rounded_cross_2d` and `polygon` is `polygon_2d`; each is
/// negative inside the respective shape and positive outside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfMisc2dResult {
    /// Signed distance to the rounded X.
    pub rounded_x: f32,
    /// Signed distance to the rounded cross.
    pub rounded_cross: f32,
    /// Signed distance to the filled polygon.
    pub polygon: f32,
}

/// Encodes one [`SdfMisc2dQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfMisc2dQuery) -> GpuQuery {
    let mut poly = [0.0_f32; 16];
    for (slot, vertex) in q.polygon_vertices.iter().enumerate() {
        poly[2 * slot] = vertex[0];
        poly[2 * slot + 1] = vertex[1];
    }
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        rx_width: q.rounded_x_width,
        rx_radius: q.rounded_x_radius,
        rc_height: q.rounded_cross_height,
        poly_count: q.polygon_vertex_count,
        pad0: 0.0,
        pad1: 0.0,
        poly,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfMisc2dResult`].
fn decode_result(raw: &GpuResult) -> SdfMisc2dResult {
    SdfMisc2dResult {
        rounded_x: raw.rounded_x,
        rounded_cross: raw.rounded_cross,
        polygon: raw.polygon,
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

/// A compiled, reusable two-dimensional cross/polygon distance compute
/// pipeline, twinning the `CPU` golden
/// `prism_render_architecture::ray_scene::sdf_primitives` outlines `rounded_x`,
/// `rounded_cross_2d` and `polygon_2d`.
pub struct GpuSdfMisc2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfMisc2d {
    /// Compiles the two-dimensional cross/polygon distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfMisc2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_misc2d"),
            source: ShaderSource::Wgsl(SDF_MISC2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_misc2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_misc2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_misc2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfMisc2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfMisc2dResult`] per
    /// input, in order.
    ///
    /// The distances match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[SdfMisc2dQuery]) -> Vec<SdfMisc2dResult> {
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
            label: Some("prism_volumetric_sdf_misc2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_misc2d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_misc2d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_misc2d_bind_group"),
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
            label: Some("prism_volumetric_sdf_misc2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_misc2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_misc2d_pass"),
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
