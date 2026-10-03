//! `wgpu` compute twin of a cohesive bundle of two-dimensional signed-distance
//! primitives and domain operators from the `CPU` golden path
//! (`prism_render_architecture::ray_scene`): the Inigo Quilez `circle_2d` and
//! `rounded_box_2d` primitives together with the `elongate_2d` /
//! `elongate_2d_correction` domain pair.
//!
//! Vector graphics, UI glyph masks and procedural profile modelling evaluate
//! these analytic atoms whose exact Euclidean distance (or exact domain
//! displacement) is known in closed form rather than sampling a baked field.
//! This module is the on-device twin of four of those atoms, each built from
//! `abs`, `min`, `max`, `clamp`, an ordered `select` and a final `sqrt`:
//!
//! - `circle_2d`: the radial distance to a centred circle, `|point| - radius`.
//! - `rounded_box_2d`: the exact distance to an axis-aligned rectangle with
//!   four independent corner radii, the active radius chosen by the query's
//!   quadrant through ordered comparisons.
//! - `elongate_2d`: the 2D elongation domain displacement
//!   `point - clamp(point, -half_extent, half_extent)` per axis, which carves a
//!   `[-h, h]` core out of each axis before a profile is evaluated.
//! - `elongate_2d_correction`: the non-positive interior correction scalar
//!   `min(max(|p.x| - h.x, |p.y| - h.y), 0)` that restores an exact interior
//!   field when added to the elongated profile.
//!
//! [`GpuSdf2dOps`] evaluates all four for one query per thread, reproducing the
//! reference closed forms with no transcendental, so a passing real-device
//! parity test is direct evidence the ported kernel computes the same values
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`Sdf2dOpsQuery`] — the shared query point, the circle
//! radius, the rounded box's half-extent and four corner radii, and the
//! elongation half-extent — and writes one [`Sdf2dOpsResult`] holding the three
//! scalar outputs (`circle_2d`, `rounded_box_2d`, `elongate_2d_correction`) and
//! the one vector output (`elongate_2d`).
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these outlines, the `extrude`
//! and `revolution` lifts to three dimensions, and any acceleration structure
//! all stay on the host; the device sees only the stateless distance and
//! displacement evaluation, one query at a time, so a storage buffer is never
//! zero-sized.
//!
//! # Correctness model
//!
//! `circle_2d` and `rounded_box_2d` thread through a `sqrt`, so the `CPU` and
//! `GPU` are not bit-exact: a `GPU` `sqrt` may land a few units in the last
//! place from the scalar reference. The parity test asserts each output within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` with a relative floor of `1e-6`.
//! The `rounded_box_2d` quadrant radius selection and the `elongate` core
//! boundary are measure-zero creases where the compared branches agree
//! continuously, so no branch disagreement can produce a cliff; the randomized
//! sweep still rejects samples near the coordinate axes and any surface zero to
//! keep the comparison far from such loci.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `select`, `sqrt`, `+ - * /` — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, `tan`, no inverse trigonometry, no `round`, and no
//! `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene`（`sdf_primitives` 与 `sdf_domain`）；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` kernel for the four two-dimensional primitives and
/// domain operators, embedded inline so the twin ships as a single source file.
/// The single entry point `solve` runs one thread per query.
const SDF_2D_OPS_WGSL: &str = r#"
// 2D primitive/domain twin: one thread computes one query's circle_2d,
// rounded_box_2d, elongate_2d and elongate_2d_correction, mirroring the CPU
// golden ray_scene::{sdf_primitives, sdf_domain} with only abs, min, max,
// clamp, an ordered select and a final sqrt. The CSG/extrude/revolution
// operators stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene；无第三方引擎源码或衍生代码。

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
    // Circle radius.
    circle_radius: f32,
    // Rounded box half-extent.
    box_hx: f32,
    box_hy: f32,
    // Rounded box corner radii [top_right, bottom_right, top_left, bottom_left].
    r0: f32,
    r1: f32,
    r2: f32,
    r3: f32,
    // Elongation half-extent.
    elong_hx: f32,
    elong_hy: f32,
    pad0: f32,
}

struct Outputs {
    // Signed distance to the circle.
    circle: f32,
    // Signed distance to the rounded box.
    rounded_box: f32,
    // Interior elongation correction scalar.
    correction: f32,
    // Elongation displacement x.
    elongate_x: f32,
    // Elongation displacement y.
    elongate_y: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Outputs>;

// Euclidean length of a 2-vector.
fn length2(v: vec2<f32>) -> f32 {
    return sqrt(v.x * v.x + v.y * v.y);
}

// Exact signed distance to a centred circle of radius `radius`.
fn circle_2d(point: vec2<f32>, radius: f32) -> f32 {
    return length2(point) - radius;
}

// Exact signed distance to an axis-aligned rounded rectangle with four
// independent corner radii [top_right, bottom_right, top_left, bottom_left],
// the active radius chosen by the query's quadrant with ordered comparisons.
fn rounded_box_2d(point: vec2<f32>, half_extent: vec2<f32>, radii: vec4<f32>) -> f32 {
    // select(false_value, true_value, condition): point.x <= 0 keeps the left
    // (false) radii, matching the reference `if point[0] > 0.0` else branch.
    let rx = select(radii.z, radii.x, point.x > 0.0);
    let ry = select(radii.w, radii.y, point.x > 0.0);
    let r = select(ry, rx, point.y > 0.0);
    let qx = abs(point.x) - half_extent.x + r;
    let qy = abs(point.y) - half_extent.y + r;
    return min(max(qx, qy), 0.0) + length2(vec2<f32>(max(qx, 0.0), max(qy, 0.0))) - r;
}

// 2D elongation displacement: carve the [-h, h] core out of each axis.
fn elongate_2d(point: vec2<f32>, half_extent: vec2<f32>) -> vec2<f32> {
    return point - clamp(point, -half_extent, half_extent);
}

// Non-positive interior elongation correction scalar.
fn elongate_2d_correction(point: vec2<f32>, half_extent: vec2<f32>) -> f32 {
    let qx = abs(point.x) - half_extent.x;
    let qy = abs(point.y) - half_extent.y;
    return min(max(qx, qy), 0.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let point = vec2<f32>(q.px, q.py);
    let box_half = vec2<f32>(q.box_hx, q.box_hy);
    let radii = vec4<f32>(q.r0, q.r1, q.r2, q.r3);
    let elong_half = vec2<f32>(q.elong_hx, q.elong_hy);

    let disp = elongate_2d(point, elong_half);

    var out: Outputs;
    out.circle = circle_2d(point, q.circle_radius);
    out.rounded_box = rounded_box_2d(point, box_half, radii);
    out.correction = elongate_2d_correction(point, elong_half);
    out.elongate_x = disp.x;
    out.elongate_y = disp.y;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_2D_OPS_WGSL`].
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
/// the shared query point, the circle radius, the rounded box half-extent and
/// corner radii, and the elongation half-extent.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Circle radius.
    circle_radius: f32,
    /// Rounded box half-extent `x`.
    box_hx: f32,
    /// Rounded box half-extent `y`.
    box_hy: f32,
    /// Corner radius top-right.
    r0: f32,
    /// Corner radius bottom-right.
    r1: f32,
    /// Corner radius top-left.
    r2: f32,
    /// Corner radius bottom-left.
    r3: f32,
    /// Elongation half-extent `x`.
    elong_hx: f32,
    /// Elongation half-extent `y`.
    elong_hy: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Outputs`
/// struct: the three scalar outputs and the two elongation displacement lanes
/// to a `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed distance to the circle.
    circle: f32,
    /// Signed distance to the rounded box.
    rounded_box: f32,
    /// Interior elongation correction scalar.
    correction: f32,
    /// Elongation displacement `x`.
    elongate_x: f32,
    /// Elongation displacement `y`.
    elongate_y: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// One query for the two-dimensional primitive/domain twin: the shared query
/// point and the per-shape parameters.
///
/// `point` is the position whose outputs are sought. `circle_radius` is the
/// `circle_2d` radius. `box_half_extent` and `box_radii`
/// (`[top_right, bottom_right, top_left, bottom_left]`) are the `rounded_box_2d`
/// half-size and per-corner radii. `elongate_half_extent` is the `elongate_2d`
/// / `elongate_2d_correction` half-extent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sdf2dOpsQuery {
    /// Shared query point.
    pub point: [f32; 2],
    /// Circle radius.
    pub circle_radius: f32,
    /// Rounded box half-extent.
    pub box_half_extent: [f32; 2],
    /// Rounded box corner radii `[top_right, bottom_right, top_left, bottom_left]`.
    pub box_radii: [f32; 4],
    /// Elongation half-extent.
    pub elongate_half_extent: [f32; 2],
}

/// One resolved query of the two-dimensional primitive/domain twin.
///
/// `circle` is
/// `prism_render_architecture::ray_scene::sdf_primitives::circle_2d`,
/// `rounded_box` is `rounded_box_2d`, `correction` is
/// `prism_render_architecture::ray_scene::sdf_domain::elongate_2d_correction`
/// and `elongate` is `elongate_2d`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sdf2dOpsResult {
    /// Signed distance to the circle.
    pub circle: f32,
    /// Signed distance to the rounded box.
    pub rounded_box: f32,
    /// Interior elongation correction scalar.
    pub correction: f32,
    /// Elongation displacement.
    pub elongate: [f32; 2],
}

/// Encodes one [`Sdf2dOpsQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &Sdf2dOpsQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        circle_radius: q.circle_radius,
        box_hx: q.box_half_extent[0],
        box_hy: q.box_half_extent[1],
        r0: q.box_radii[0],
        r1: q.box_radii[1],
        r2: q.box_radii[2],
        r3: q.box_radii[3],
        elong_hx: q.elongate_half_extent[0],
        elong_hy: q.elongate_half_extent[1],
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`Sdf2dOpsResult`].
fn decode_result(raw: &GpuResult) -> Sdf2dOpsResult {
    Sdf2dOpsResult {
        circle: raw.circle,
        rounded_box: raw.rounded_box,
        correction: raw.correction,
        elongate: [raw.elongate_x, raw.elongate_y],
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

/// A compiled, reusable two-dimensional primitive/domain compute pipeline,
/// twinning the `CPU` golden `prism_render_architecture::ray_scene` atoms
/// `circle_2d`, `rounded_box_2d`, `elongate_2d` and `elongate_2d_correction`.
pub struct GpuSdf2dOps {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdf2dOps {
    /// Compiles the two-dimensional primitive/domain kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdf2dOps {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_2d_ops"),
            source: ShaderSource::Wgsl(SDF_2D_OPS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_2d_ops_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_2d_ops_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_2d_ops_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdf2dOps {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`Sdf2dOpsResult`] per
    /// input, in order.
    ///
    /// The outputs match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[Sdf2dOpsQuery]) -> Vec<Sdf2dOpsResult> {
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
            label: Some("prism_volumetric_sdf_2d_ops_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_2d_ops_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_2d_ops_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_2d_ops_bind_group"),
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
            label: Some("prism_volumetric_sdf_2d_ops_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_2d_ops_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_2d_ops_pass"),
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
