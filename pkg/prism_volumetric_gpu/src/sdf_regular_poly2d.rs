//! `wgpu` compute twin of four exact 2D regular-polygon signed-distance fields
//! from the reference ray-scene `SDF` primitive library
//! (`prism_render_architecture::ray_scene::sdf_primitives`).
//!
//! Four closed-form distance evaluators are twinned, each the exact field of a
//! filled, origin-centred regular polygon, all transcendental-free (every
//! trigonometric term is baked into a compile-time fold constant):
//!
//! - `regular_hexagon_2d(point, apothem)`: a flat-top regular hexagon. One
//!   reflection against the baked `(-cos 30deg, sin 30deg)` constant folds the
//!   query into one sextant, then a single clamped edge against the baked
//!   `tan 30deg` carries the field as `length(p) * sign(p.y)`.
//! - `regular_pentagon_2d(point, apothem)`: a flat-top regular pentagon. The
//!   query is mirrored into the right half-plane and two reflections against the
//!   baked `cos 36deg` / `sin 36deg` constants fold it into the top sector,
//!   reducing to one clamped edge against the baked `tan 36deg`.
//! - `regular_octagon_2d(point, apothem)`: a flat-top regular octagon. The query
//!   is folded into the first quadrant and two reflections against the baked
//!   `cos 22.5deg` / `sin 22.5deg` constants collapse it to the top sector, a
//!   single clamped edge against the baked `tan 22.5deg`.
//! - `equilateral_triangle_2d(point, half_width)`: an apex-up equilateral
//!   triangle. A reflection about `x = 0` plus one fold across the baked
//!   `sqrt 3` edge collapse the query into one wedge, a single clamped edge with
//!   signed distance `-length(p) * sign(p.y)`.
//!
//! # What is twinned
//!
//! One thread resolves one query. Each [`SdfRegularPoly2dQuery`] carries the
//! point and the shape parameters; the kernel evaluates all four fields and
//! writes one [`SdfRegularPoly2dResult`] holding the four signed distances. The
//! twin spells out the same closed form with the same ordered reflections,
//! clamps and sign tests as the reference, so a passing real-device parity test
//! is direct evidence the ported kernel evaluates the same distance the
//! reference does, not merely that the shader compiles.
//!
//! # What stays on the host
//!
//! Nothing of the per-query math stays on the host: each field is a fixed,
//! bounded sequence of reflections, clamps, products and one `sqrt` that runs
//! entirely on device. The host only flattens the query batch into a `std430`
//! storage buffer and short-circuits an empty batch, since a storage buffer
//! cannot be zero-sized.
//!
//! # Correctness model
//!
//! Each distance is a *continuous* quantity threaded through `sqrt` and
//! products, so the `CPU` and `GPU` are not bit-exact: a device `sqrt` may land
//! a few units in the last place from the scalar reference. The parity test
//! compares with an absolute-or-relative tolerance (`abs <= 1e-4 || rel <=
//! 1e-3`, relative floor `1e-6`). The reference closes each field with
//! `sign(p.y)`, whose Rust `signum` returns `+1` at zero while `WGSL` `sign`
//! returns `0`; the twin instead uses a positive-at-zero branch on both the
//! host oracle and the device so they agree on the interior sign away from the
//! exact edge. Sharp corners are a conditioning hot-spot where the fold and the
//! sign branch flip; named fixtures keep clear of the exact apex and edges, and
//! the randomized sweep rejects points within a small margin of the sign
//! boundary.
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

/// Hexagon fold constant `x`: baked `-cos 30deg`.
const HEX_KX: f32 = -0.866_025_4;
/// Hexagon fold constant `y`: baked `sin 30deg`.
const HEX_KY: f32 = 0.5;
/// Hexagon edge-clamp constant: baked `tan 30deg`.
const HEX_KZ: f32 = 0.577_350_26;

/// Equilateral-triangle fold constant: baked `sqrt 3`.
const TRI_K: f32 = 1.732_050_8;

/// Pentagon fold constant `x`: baked `cos 36deg`.
const PENT_KX: f32 = 0.809_017;
/// Pentagon fold constant `y`: baked `sin 36deg`.
const PENT_KY: f32 = 0.587_785_25;
/// Pentagon edge-clamp constant: baked `tan 36deg`.
const PENT_KZ: f32 = 0.726_542_5;

/// Octagon fold constant `x`: baked `-cos 22.5deg`.
const OCT_KX: f32 = -0.923_879_5;
/// Octagon fold constant `y`: baked `sin 22.5deg`.
const OCT_KY: f32 = 0.382_683_43;
/// Octagon edge-clamp constant: baked `tan 22.5deg`.
const OCT_KZ: f32 = 0.414_213_56;

/// Positive-at-zero sign, matching the device `signum_branch` so the host
/// oracle and the device agree on the interior sign away from the exact edge.
fn signum_branch(value: f32) -> f32 {
    if value < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Euclidean length of a 2D vector; one `sqrt`, which is a core arithmetic
/// primitive rather than a transcendental.
fn length2(ax: f32, ay: f32) -> f32 {
    (ax * ax + ay * ay).sqrt()
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::regular_hexagon_2d`, reproduced without importing
/// the golden so the twin stays self-contained.
///
/// Folds the query into one sextant with a single reflection, then measures the
/// clamped top edge as `length(p) * sign(p.y)`.
#[must_use]
pub fn regular_hexagon_2d_sdf(point: [f32; 2], apothem: f32) -> f32 {
    let mut px = point[0].abs();
    let mut py = point[1].abs();
    let fold = 2.0 * (HEX_KX * px + HEX_KY * py).min(0.0);
    px -= fold * HEX_KX;
    py -= fold * HEX_KY;
    let ex = px - px.clamp(-HEX_KZ * apothem, HEX_KZ * apothem);
    let ey = py - apothem;
    length2(ex, ey) * signum_branch(ey)
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::regular_pentagon_2d`, reproduced without
/// importing the golden.
///
/// Mirrors `x` into the right half-plane, folds across the two upper edges,
/// then measures the clamped top edge as `length(p) * sign(p.y)`.
#[must_use]
pub fn regular_pentagon_2d_sdf(point: [f32; 2], apothem: f32) -> f32 {
    let r = apothem;
    let mut px = point[0].abs();
    let mut py = point[1];
    let f1 = 2.0 * ((-PENT_KX) * px + PENT_KY * py).min(0.0);
    px -= f1 * (-PENT_KX);
    py -= f1 * PENT_KY;
    let f2 = 2.0 * (PENT_KX * px + PENT_KY * py).min(0.0);
    px -= f2 * PENT_KX;
    py -= f2 * PENT_KY;
    let ex = px - px.clamp(-r * PENT_KZ, r * PENT_KZ);
    let ey = py - r;
    length2(ex, ey) * signum_branch(ey)
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::regular_octagon_2d`, reproduced without importing
/// the golden.
///
/// Folds the query into the first quadrant, reflects across the two diagonal
/// edges, then measures the clamped top edge as `length(p) * sign(p.y)`.
#[must_use]
pub fn regular_octagon_2d_sdf(point: [f32; 2], apothem: f32) -> f32 {
    let r = apothem;
    let mut px = point[0].abs();
    let mut py = point[1].abs();
    let f1 = 2.0 * (OCT_KX * px + OCT_KY * py).min(0.0);
    px -= f1 * OCT_KX;
    py -= f1 * OCT_KY;
    let f2 = 2.0 * ((-OCT_KX) * px + OCT_KY * py).min(0.0);
    px -= f2 * (-OCT_KX);
    py -= f2 * OCT_KY;
    let ex = px - px.clamp(-OCT_KZ * r, OCT_KZ * r);
    let ey = py - r;
    length2(ex, ey) * signum_branch(ey)
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::equilateral_triangle_2d`, reproduced without
/// importing the golden.
///
/// A reflection about `x = 0` plus one fold across the `sqrt 3` edge collapse
/// the query into one wedge; the clamped edge gives `-length(p) * sign(p.y)`.
#[must_use]
pub fn equilateral_triangle_2d_sdf(point: [f32; 2], half_width: f32) -> f32 {
    let r = half_width;
    let mut px = point[0].abs() - r;
    let mut py = point[1] + r / TRI_K;
    if px + TRI_K * py > 0.0 {
        let folded_x = (px - TRI_K * py) * 0.5;
        let folded_y = (-TRI_K * px - py) * 0.5;
        px = folded_x;
        py = folded_y;
    }
    px -= px.clamp(-2.0 * r, 0.0);
    -length2(px, py) * signum_branch(py)
}

/// The portable core-`WGSL` regular-polygon `SDF` kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the four golden evaluators; see the module documentation.
const SDF_REGULAR_POLY2D_WGSL: &str = r#"
// Regular-polygon 2D SDF twin: one thread evaluates the hexagon, pentagon,
// octagon and equilateral-triangle fields for one query, mirroring the CPU
// golden `ray_scene::sdf_primitives::{regular_hexagon_2d, regular_pentagon_2d,
// regular_octagon_2d, equilateral_triangle_2d}` with only reflections, clamps,
// products and sqrt. Any variable-length scene assembly stays on the host.
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
    // Query point.
    px: f32,
    py: f32,
    // Shared apothem for hexagon / pentagon / octagon.
    apothem: f32,
    // Half-width for the equilateral triangle.
    half_width: f32,
}

struct SdfResult {
    dist_hexagon: f32,
    dist_pentagon: f32,
    dist_octagon: f32,
    dist_triangle: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<SdfResult>;

const HEX_KX: f32 = -0.8660254;
const HEX_KY: f32 = 0.5;
const HEX_KZ: f32 = 0.57735026;
const TRI_K: f32 = 1.7320508;
const PENT_KX: f32 = 0.809017;
const PENT_KY: f32 = 0.58778525;
const PENT_KZ: f32 = 0.7265425;
const OCT_KX: f32 = -0.9238795;
const OCT_KY: f32 = 0.38268343;
const OCT_KZ: f32 = 0.41421356;

fn length2(ax: f32, ay: f32) -> f32 {
    return sqrt(ax * ax + ay * ay);
}

// Positive-at-zero sign, matching the host oracle so the CPU and GPU agree on
// the interior sign away from the exact edge.
fn signum_branch(value: f32) -> f32 {
    if (value < 0.0) {
        return -1.0;
    }
    return 1.0;
}

fn hexagon_sdf(point_x: f32, point_y: f32, apothem: f32) -> f32 {
    var px = abs(point_x);
    var py = abs(point_y);
    let fold = 2.0 * min(HEX_KX * px + HEX_KY * py, 0.0);
    px = px - fold * HEX_KX;
    py = py - fold * HEX_KY;
    let ex = px - clamp(px, -HEX_KZ * apothem, HEX_KZ * apothem);
    let ey = py - apothem;
    return length2(ex, ey) * signum_branch(ey);
}

fn pentagon_sdf(point_x: f32, point_y: f32, apothem: f32) -> f32 {
    let r = apothem;
    var px = abs(point_x);
    var py = point_y;
    let f1 = 2.0 * min((-PENT_KX) * px + PENT_KY * py, 0.0);
    px = px - f1 * (-PENT_KX);
    py = py - f1 * PENT_KY;
    let f2 = 2.0 * min(PENT_KX * px + PENT_KY * py, 0.0);
    px = px - f2 * PENT_KX;
    py = py - f2 * PENT_KY;
    let ex = px - clamp(px, -r * PENT_KZ, r * PENT_KZ);
    let ey = py - r;
    return length2(ex, ey) * signum_branch(ey);
}

fn octagon_sdf(point_x: f32, point_y: f32, apothem: f32) -> f32 {
    let r = apothem;
    var px = abs(point_x);
    var py = abs(point_y);
    let f1 = 2.0 * min(OCT_KX * px + OCT_KY * py, 0.0);
    px = px - f1 * OCT_KX;
    py = py - f1 * OCT_KY;
    let f2 = 2.0 * min((-OCT_KX) * px + OCT_KY * py, 0.0);
    px = px - f2 * (-OCT_KX);
    py = py - f2 * OCT_KY;
    let ex = px - clamp(px, -OCT_KZ * r, OCT_KZ * r);
    let ey = py - r;
    return length2(ex, ey) * signum_branch(ey);
}

fn triangle_sdf(point_x: f32, point_y: f32, half_width: f32) -> f32 {
    let r = half_width;
    var px = abs(point_x) - r;
    var py = point_y + r / TRI_K;
    if (px + TRI_K * py > 0.0) {
        let folded_x = (px - TRI_K * py) * 0.5;
        let folded_y = (-TRI_K * px - py) * 0.5;
        px = folded_x;
        py = folded_y;
    }
    px = px - clamp(px, -2.0 * r, 0.0);
    return -length2(px, py) * signum_branch(py);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: SdfResult;
    out.dist_hexagon = hexagon_sdf(q.px, q.py, q.apothem);
    out.dist_pentagon = pentagon_sdf(q.px, q.py, q.apothem);
    out.dist_octagon = octagon_sdf(q.px, q.py, q.apothem);
    out.dist_triangle = triangle_sdf(q.px, q.py, q.half_width);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`SDF_REGULAR_POLY2D_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the point and the shape parameters,
/// matching the `WGSL` `Query` struct's `16`-byte stride (four `f32`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Shared apothem for the hexagon, pentagon and octagon.
    apothem: f32,
    /// Half-width for the equilateral triangle.
    half_width: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `SdfResult`
/// struct: the four signed distances in a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed distance to the regular hexagon.
    dist_hexagon: f32,
    /// Signed distance to the regular pentagon.
    dist_pentagon: f32,
    /// Signed distance to the regular octagon.
    dist_octagon: f32,
    /// Signed distance to the equilateral triangle.
    dist_triangle: f32,
}

/// One query: the point plus the shape parameters for the four fields.
///
/// The hexagon, pentagon and octagon share the `apothem`; the triangle uses
/// `half_width`. The host enqueues one query per evaluation, and an empty batch
/// is short-circuited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfRegularPoly2dQuery {
    /// Query point `x`.
    pub px: f32,
    /// Query point `y`.
    pub py: f32,
    /// Shared apothem (inradius) for the hexagon, pentagon and octagon.
    pub apothem: f32,
    /// Half-width of the equilateral triangle's base.
    pub half_width: f32,
}

impl SdfRegularPoly2dQuery {
    /// Builds a query from the point, the shared apothem and the triangle's
    /// half-width.
    #[must_use]
    pub const fn new(px: f32, py: f32, apothem: f32, half_width: f32) -> SdfRegularPoly2dQuery {
        SdfRegularPoly2dQuery {
            px,
            py,
            apothem,
            half_width,
        }
    }
}

/// One resolved query: the four signed distances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfRegularPoly2dResult {
    /// Signed distance to the regular hexagon.
    pub dist_hexagon: f32,
    /// Signed distance to the regular pentagon.
    pub dist_pentagon: f32,
    /// Signed distance to the regular octagon.
    pub dist_octagon: f32,
    /// Signed distance to the equilateral triangle.
    pub dist_triangle: f32,
}

/// Encodes one [`SdfRegularPoly2dQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfRegularPoly2dQuery) -> GpuQuery {
    GpuQuery {
        px: q.px,
        py: q.py,
        apothem: q.apothem,
        half_width: q.half_width,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfRegularPoly2dResult`].
fn decode_result(raw: &GpuResult) -> SdfRegularPoly2dResult {
    SdfRegularPoly2dResult {
        dist_hexagon: raw.dist_hexagon,
        dist_pentagon: raw.dist_pentagon,
        dist_octagon: raw.dist_octagon,
        dist_triangle: raw.dist_triangle,
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

/// A compiled, reusable regular-polygon `SDF` compute pipeline, twinning the
/// golden `ray_scene::sdf_primitives` evaluators `regular_hexagon_2d`,
/// `regular_pentagon_2d`, `regular_octagon_2d` and `equilateral_triangle_2d`.
pub struct GpuSdfRegularPoly2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfRegularPoly2d {
    /// Compiles the regular-polygon `SDF` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfRegularPoly2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_regular_poly2d"),
            source: ShaderSource::Wgsl(SDF_REGULAR_POLY2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_regular_poly2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_regular_poly2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_regular_poly2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfRegularPoly2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`SdfRegularPoly2dResult`] per input, in order.
    ///
    /// Each distance equals the reference within floating-point tolerance. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfRegularPoly2dQuery],
    ) -> Vec<SdfRegularPoly2dResult> {
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
            label: Some("prism_volumetric_sdf_regular_poly2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_regular_poly2d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_regular_poly2d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_regular_poly2d_bind_group"),
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
            label: Some("prism_volumetric_sdf_regular_poly2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_regular_poly2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_regular_poly2d_pass"),
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
