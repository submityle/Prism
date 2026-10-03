//! `wgpu` compute twin of three exact polygon / prism signed-distance fields
//! from the reference ray-scene `SDF` primitive library
//! (`prism_render_architecture::ray_scene::sdf_primitives`).
//!
//! Three closed-form distance evaluators are twinned, each the exact field of a
//! convex or star polygon (and one extruded prism), all transcendental-free:
//!
//! - `triangular_prism(point, size, half_depth)`: the exact distance to a solid
//!   equilateral triangular prism extruded along `z`. An `x` fold, a single
//!   slanted-edge fold against the baked `sqrt 3` constant, a base-edge clamp,
//!   then the standard interior / exterior extrusion split against the depth
//!   caps.
//! - `trapezoid_isosceles(point, bottom_half, top_half, half_height)`: the
//!   exact distance to a 2D isosceles trapezoid. An `x` fold, the smaller of the
//!   capped-horizontal-edge candidate and the slanted-side candidate (a
//!   projection onto the side direction clamped to the segment), with the
//!   interior sign set left-of-the-slant and below the top.
//! - `star5_2d(point, radius, inner_ratio)`: the exact distance to an upward
//!   regular five-pointed star. Two mirror reflections against the baked
//!   `cos 36 deg` / `sin 36 deg` constants fold the query into one `36`-degree
//!   wedge, then a single edge distance with a left/right interior sign.
//!
//! # What is twinned
//!
//! One thread resolves one query. Each [`SdfPolygonQuery`] carries the point and
//! every shape parameter; the kernel evaluates all three fields and writes one
//! [`SdfPolygonResult`] holding the three signed distances. The twin spells out
//! the same closed form with the same ordered folds, clamps and sign tests as
//! the reference, so a passing real-device parity test is direct evidence the
//! ported kernel evaluates the same distance the reference does, not merely that
//! the shader compiles.
//!
//! # What stays on the host
//!
//! Nothing of the per-query math stays on the host: each field is a fixed,
//! bounded sequence of folds, clamps, products and one or two `sqrt` calls that
//! runs entirely on device. The host only flattens the query batch into a
//! `std430` storage buffer and short-circuits an empty batch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! Each distance is a *continuous* quantity threaded through `sqrt`, products
//! and quotients, so the `CPU` and `GPU` are not bit-exact: a device `sqrt` or
//! divide may land a few units in the last place from the scalar reference. The
//! parity test compares with an absolute-or-relative tolerance (`abs <= 1e-4 ||
//! rel <= 1e-3`, relative floor `1e-6`). Sharp corners are a conditioning
//! hot-spot where the nearest-feature branch flips; named fixtures keep clear of
//! the exact apex and the exact edges, and the randomized sweep rejects points
//! within a small margin of any fold boundary.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `sqrt`, `+ - * /` and ordered comparisons — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round` and no
//! `u64` / `u16` / `i64` / `f64`. No optional device feature is required, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each
//! thread performs a fixed, bounded sequence of arithmetic, so the kernel
//! provably terminates.
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

/// Baked `sqrt 3`, the equilateral-triangle fold constant, matching the golden
/// `triangular_prism` constant exactly.
const SQRT3: f32 = 1.732_050_8;

/// Baked `cos 36 deg`, the first star-fold reflection constant.
const K1X: f32 = 0.809_017;

/// Baked `-sin 36 deg`, the second star-fold reflection constant.
const K1Y: f32 = -0.587_785_25;

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::triangular_prism`, reproduced without importing
/// the golden so the twin stays self-contained and free of any cross-crate
/// dependency.
///
/// Folds the `xy` section into one wedge of the equilateral triangle, clamps the
/// base edge, signs the planar distance, then extrudes exactly against the depth
/// caps with the standard interior / exterior split.
#[must_use]
pub fn triangular_prism_sdf(point: [f32; 3], size: f32, half_depth: f32) -> f32 {
    let mut x = point[0].abs() - size;
    let mut y = point[1] + size / SQRT3;
    if x + SQRT3 * y > 0.0 {
        let folded_x = (x - SQRT3 * y) * 0.5;
        let folded_y = (-SQRT3 * x - y) * 0.5;
        x = folded_x;
        y = folded_y;
    }
    x -= x.clamp(-2.0 * size, 0.0);
    let planar_sign = if y < 0.0 { -1.0 } else { 1.0 };
    let planar = -(x * x + y * y).sqrt() * planar_sign;
    let cap0 = planar;
    let cap1 = point[2].abs() - half_depth;
    let inside = cap0.max(cap1).min(0.0);
    let ox = cap0.max(0.0);
    let oy = cap1.max(0.0);
    let outside = (ox * ox + oy * oy).sqrt();
    inside + outside
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::trapezoid_isosceles`, reproduced without
/// importing the golden.
///
/// Folds `x` to its magnitude, takes the smaller of the capped-edge candidate
/// and the slanted-side candidate (a projection onto the side direction clamped
/// to the segment), and signs the result interior when the point lies left of
/// the slant and below the top.
#[must_use]
pub fn trapezoid_isosceles_sdf(
    point: [f32; 2],
    bottom_half: f32,
    top_half: f32,
    half_height: f32,
) -> f32 {
    let (r1, r2, he) = (bottom_half, top_half, half_height);
    let k1 = [r2, he];
    let k2 = [r2 - r1, 2.0 * he];
    let p = [point[0].abs(), point[1]];
    let edge = if p[1] < 0.0 { r1 } else { r2 };
    let ca = [p[0] - p[0].min(edge), p[1].abs() - he];
    let denom = k2[0] * k2[0] + k2[1] * k2[1];
    let t = (((k1[0] - p[0]) * k2[0] + (k1[1] - p[1]) * k2[1]) / denom).clamp(0.0, 1.0);
    let cb = [p[0] - k1[0] + k2[0] * t, p[1] - k1[1] + k2[1] * t];
    let s = if cb[0] < 0.0 && ca[1] < 0.0 {
        -1.0
    } else {
        1.0
    };
    s * (ca[0] * ca[0] + ca[1] * ca[1])
        .min(cb[0] * cb[0] + cb[1] * cb[1])
        .sqrt()
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::star5_2d`, reproduced without importing the
/// golden.
///
/// Two mirror reflections against the baked `cos 36 deg` / `sin 36 deg`
/// constants fold the query into one wedge, then one edge distance (projection
/// onto the tip-to-inner edge clamped to the span) carries the field with a
/// left/right interior sign. The sign uses a positive-at-zero branch so the
/// host oracle and the device agree bit-for-bit on the sign away from the
/// exact edge.
#[must_use]
pub fn star5_2d_sdf(point: [f32; 2], radius: f32, inner_ratio: f32) -> f32 {
    let mut px = point[0].abs();
    let mut py = point[1];
    let d1 = (K1X * px + K1Y * py).max(0.0);
    px -= 2.0 * d1 * K1X;
    py -= 2.0 * d1 * K1Y;
    let d2 = (-K1X * px + K1Y * py).max(0.0);
    px -= 2.0 * d2 * (-K1X);
    py -= 2.0 * d2 * K1Y;
    px = px.abs();
    py -= radius;
    let bax = inner_ratio * (-K1Y);
    let bay = inner_ratio * K1X - 1.0;
    let bb = bax * bax + bay * bay;
    let h = ((px * bax + py * bay) / bb).clamp(0.0, radius);
    let dx = px - bax * h;
    let dy = py - bay * h;
    let sign = if py * bax - px * bay < 0.0 { -1.0 } else { 1.0 };
    (dx * dx + dy * dy).sqrt() * sign
}

/// The portable core-`WGSL` polygon / prism `SDF` kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the three golden evaluators; see the module documentation.
const SDF_POLYGON_WGSL: &str = r#"
// Polygon / prism SDF twin: one thread evaluates the triangular prism,
// isosceles trapezoid and five-pointed-star fields for one query, mirroring the
// CPU golden `ray_scene::sdf_primitives::{triangular_prism, trapezoid_isosceles,
// star5_2d}` with only folds, clamps, products and sqrt. Any variable-length
// scene assembly stays on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_primitives；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point (prism uses all three; the 2D fields use px, py).
    px: f32,
    py: f32,
    pz: f32,
    // Triangular prism parameters.
    size: f32,
    half_depth: f32,
    // Trapezoid parameters.
    bottom_half: f32,
    top_half: f32,
    half_height: f32,
    // Star parameters.
    radius: f32,
    inner_ratio: f32,
    pad0: f32,
    pad1: f32,
}

struct SdfResult {
    dist_prism: f32,
    dist_trapezoid: f32,
    dist_star5: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<SdfResult>;

const SQRT3: f32 = 1.732050808;
const K1X: f32 = 0.809017;
const K1Y: f32 = -0.58778525;

fn length2(ax: f32, ay: f32) -> f32 {
    return sqrt(ax * ax + ay * ay);
}

// Positive-at-zero sign, matching the host oracle's branch so the CPU and GPU
// agree on the interior sign away from the exact edge.
fn signum_branch(value: f32) -> f32 {
    if (value < 0.0) {
        return -1.0;
    }
    return 1.0;
}

fn prism_sdf(px: f32, py: f32, pz: f32, size: f32, half_depth: f32) -> f32 {
    var x = abs(px) - size;
    var y = py + size / SQRT3;
    if (x + SQRT3 * y > 0.0) {
        let folded_x = (x - SQRT3 * y) * 0.5;
        let folded_y = (-SQRT3 * x - y) * 0.5;
        x = folded_x;
        y = folded_y;
    }
    x = x - clamp(x, -2.0 * size, 0.0);
    let planar_sign = signum_branch(y);
    let planar = -length2(x, y) * planar_sign;
    let cap0 = planar;
    let cap1 = abs(pz) - half_depth;
    let inside = min(max(cap0, cap1), 0.0);
    let outside = length2(max(cap0, 0.0), max(cap1, 0.0));
    return inside + outside;
}

fn trapezoid_sdf(px: f32, py: f32, r1: f32, r2: f32, he: f32) -> f32 {
    let k1x = r2;
    let k1y = he;
    let k2x = r2 - r1;
    let k2y = 2.0 * he;
    let qx = abs(px);
    let qy = py;
    var edge = r2;
    if (qy < 0.0) {
        edge = r1;
    }
    let cax = qx - min(qx, edge);
    let cay = abs(qy) - he;
    let denom = k2x * k2x + k2y * k2y;
    let tval = clamp(((k1x - qx) * k2x + (k1y - qy) * k2y) / denom, 0.0, 1.0);
    let cbx = qx - k1x + k2x * tval;
    let cby = qy - k1y + k2y * tval;
    var s = 1.0;
    if (cbx < 0.0 && cay < 0.0) {
        s = -1.0;
    }
    let dmin = min(cax * cax + cay * cay, cbx * cbx + cby * cby);
    return s * sqrt(dmin);
}

fn star5_sdf(input_x: f32, input_y: f32, radius: f32, inner_ratio: f32) -> f32 {
    var px = abs(input_x);
    var py = input_y;
    let d1 = max(K1X * px + K1Y * py, 0.0);
    px = px - 2.0 * d1 * K1X;
    py = py - 2.0 * d1 * K1Y;
    let d2 = max(-K1X * px + K1Y * py, 0.0);
    px = px - 2.0 * d2 * (-K1X);
    py = py - 2.0 * d2 * K1Y;
    px = abs(px);
    py = py - radius;
    let bax = inner_ratio * (-K1Y);
    let bay = inner_ratio * K1X - 1.0;
    let bb = bax * bax + bay * bay;
    let h = clamp((px * bax + py * bay) / bb, 0.0, radius);
    let dx = px - bax * h;
    let dy = py - bay * h;
    return sqrt(dx * dx + dy * dy) * signum_branch(py * bax - px * bay);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: SdfResult;
    out.dist_prism = prism_sdf(q.px, q.py, q.pz, q.size, q.half_depth);
    out.dist_trapezoid = trapezoid_sdf(q.px, q.py, q.bottom_half, q.top_half, q.half_height);
    out.dist_star5 = star5_sdf(q.px, q.py, q.radius, q.inner_ratio);
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in [`SDF_POLYGON_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the point and every shape parameter,
/// matching the `WGSL` `Query` struct's `48`-byte stride (ten `f32` plus two pad
/// words).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Query point `z` (prism depth axis).
    pz: f32,
    /// Triangular prism half-side `size`.
    size: f32,
    /// Triangular prism `half_depth` along `z`.
    half_depth: f32,
    /// Trapezoid bottom half-width.
    bottom_half: f32,
    /// Trapezoid top half-width.
    top_half: f32,
    /// Trapezoid half-height.
    half_height: f32,
    /// Star outer-tip `radius`.
    radius: f32,
    /// Star inner-vertex ratio in `(0, 1)`.
    inner_ratio: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `SdfResult`
/// struct: the three signed distances in a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed distance to the triangular prism.
    dist_prism: f32,
    /// Signed distance to the isosceles trapezoid.
    dist_trapezoid: f32,
    /// Signed distance to the five-pointed star.
    dist_star5: f32,
    /// Padding word.
    pad0: f32,
}

/// One query: the point plus every shape parameter for the three fields.
///
/// The prism uses all three point components; the two planar fields use `px`
/// and `py`. The host enqueues one query per evaluation, and an empty batch is
/// short-circuited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfPolygonQuery {
    /// Query point `x`.
    pub px: f32,
    /// Query point `y`.
    pub py: f32,
    /// Query point `z` (prism depth axis).
    pub pz: f32,
    /// Triangular prism half-side `size`.
    pub size: f32,
    /// Triangular prism `half_depth` along `z`.
    pub half_depth: f32,
    /// Trapezoid bottom half-width.
    pub bottom_half: f32,
    /// Trapezoid top half-width.
    pub top_half: f32,
    /// Trapezoid half-height.
    pub half_height: f32,
    /// Star outer-tip `radius`.
    pub radius: f32,
    /// Star inner-vertex ratio in `(0, 1)`.
    pub inner_ratio: f32,
}

impl SdfPolygonQuery {
    /// Builds a query from the point and the three shapes' parameters.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "one query carries a point plus three shapes' parameters as flat scalars by design"
    )]
    pub const fn new(
        px: f32,
        py: f32,
        pz: f32,
        size: f32,
        half_depth: f32,
        bottom_half: f32,
        top_half: f32,
        half_height: f32,
        radius: f32,
        inner_ratio: f32,
    ) -> SdfPolygonQuery {
        SdfPolygonQuery {
            px,
            py,
            pz,
            size,
            half_depth,
            bottom_half,
            top_half,
            half_height,
            radius,
            inner_ratio,
        }
    }
}

/// One resolved query: the three signed distances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfPolygonResult {
    /// Signed distance to the triangular prism.
    pub dist_prism: f32,
    /// Signed distance to the isosceles trapezoid.
    pub dist_trapezoid: f32,
    /// Signed distance to the five-pointed star.
    pub dist_star5: f32,
}

/// Encodes one [`SdfPolygonQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfPolygonQuery) -> GpuQuery {
    GpuQuery {
        px: q.px,
        py: q.py,
        pz: q.pz,
        size: q.size,
        half_depth: q.half_depth,
        bottom_half: q.bottom_half,
        top_half: q.top_half,
        half_height: q.half_height,
        radius: q.radius,
        inner_ratio: q.inner_ratio,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfPolygonResult`].
fn decode_result(raw: &GpuResult) -> SdfPolygonResult {
    SdfPolygonResult {
        dist_prism: raw.dist_prism,
        dist_trapezoid: raw.dist_trapezoid,
        dist_star5: raw.dist_star5,
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

/// A compiled, reusable polygon / prism `SDF` compute pipeline, twinning the
/// golden `ray_scene::sdf_primitives` evaluators `triangular_prism`,
/// `trapezoid_isosceles` and `star5_2d`.
pub struct GpuSdfPolygon {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfPolygon {
    /// Compiles the polygon / prism `SDF` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfPolygon {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_polygon"),
            source: ShaderSource::Wgsl(SDF_POLYGON_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_polygon_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_polygon_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_polygon_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfPolygon {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one [`SdfPolygonResult`]
    /// per input, in order.
    ///
    /// Each distance equals the reference within floating-point tolerance. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[SdfPolygonQuery]) -> Vec<SdfPolygonResult> {
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
            label: Some("prism_volumetric_sdf_polygon_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_polygon_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_polygon_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_polygon_bind_group"),
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
            label: Some("prism_volumetric_sdf_polygon_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_polygon_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_polygon_pass"),
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
