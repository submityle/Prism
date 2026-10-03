//! `wgpu` compute twin of three exact 2D star / parallelogram signed-distance
//! fields from the reference ray-scene `SDF` primitive library
//! (`prism_render_architecture::ray_scene::sdf_primitives`).
//!
//! Three closed-form distance evaluators are twinned, each the exact field of a
//! filled, origin-centred shape, all transcendental-free (every trigonometric
//! term is baked into a compile-time fold constant):
//!
//! - `hexagram_2d(point, r)`: a six-pointed hexagram (Star of David). The query
//!   is folded into the first quadrant, two reflections against the baked
//!   `cos 30deg` / `sin 30deg` constants collapse it into one of the twelve
//!   congruent wedges, then a single clamped edge against the baked `tan 30deg`
//!   / `sqrt 3` span carries the field as `length(p) * sign(p.y)`.
//! - `pentagram_2d(point, radius)`: an upward {5/2} pentagram. It is the exact
//!   five-pointed star field with the golden-ratio inner radius
//!   `(3 - sqrt 5) / 2`; two reflections against the baked `cos 36deg` /
//!   `sin 36deg` constants fold the query into one `36deg` wedge, reducing to a
//!   single edge distance signed by the edge half-plane test.
//! - `parallelogram(point, half_width, half_height, skew)`: an origin-centred
//!   parallelogram with a horizontal `skew`. The lower half is folded onto the
//!   upper half, the field is the smaller of the horizontal-edge and
//!   slanted-edge distances, and the interior sign is recovered from the signed
//!   areas accumulated in the field's second channel.
//!
//! # What is twinned
//!
//! One thread resolves one query. Each [`SdfStar2dQuery`] carries the point and
//! the per-shape parameters; the kernel evaluates all three fields and writes
//! one [`SdfStar2dResult`] holding the three signed distances. The twin spells
//! out the same closed form with the same ordered reflections, clamps and sign
//! tests as the reference, so a passing real-device parity test is direct
//! evidence the ported kernel evaluates the same distance the reference does,
//! not merely that the shader compiles.
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
//! `signum`, whose Rust result is `+1` at zero while `WGSL` `sign` returns `0`;
//! the twin instead uses a positive-at-zero branch on both the host oracle and
//! the device so they agree on the interior sign away from the exact edge. The
//! parallelogram's `s < 0` fold also pivots on a signed area; sharp star tips
//! and that fold are conditioning hot-spots where the branch flips, so named
//! fixtures keep clear of the exact feature and the randomized sweep rejects
//! points within a small margin of any sign boundary.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `sqrt`, `select`, `+ - * /` and ordered comparisons — with
//! no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round` and no `u64` / `u16` / `i64` / `f64`. No optional device feature is
//! required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no
//! loop: each thread performs a fixed, bounded sequence of arithmetic, so the
//! kernel provably terminates.
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

/// Hexagram fold constant `x`: baked `-cos 60deg`.
const HG_KX: f32 = -0.5;
/// Hexagram fold constant `y`: baked `cos 30deg`.
const HG_KY: f32 = 0.866_025_4;
/// Hexagram inner edge-clamp constant: baked `tan 30deg`.
const HG_KZ: f32 = 0.577_350_26;
/// Hexagram outer edge-clamp constant: baked `sqrt 3`.
const HG_KW: f32 = 1.732_050_8;

/// Pentagram fold constant `x`: baked `cos 36deg`.
const PENTA_K1X: f32 = 0.809_017;
/// Pentagram fold constant `y`: baked `-sin 36deg`.
const PENTA_K1Y: f32 = -0.587_785_25;
/// Pentagram inner/outer radius ratio: baked `(3 - sqrt 5) / 2`.
const PENTA_INNER: f32 = 0.381_966_02;

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
/// `ray_scene::sdf_primitives::hexagram_2d`, reproduced without importing the
/// golden so the twin stays self-contained.
///
/// Folds the query into one of the twelve congruent wedges with two
/// reflections, then measures the clamped edge as `length(p) * sign(p.y)`.
#[must_use]
pub fn hexagram_2d_sdf(point: [f32; 2], r: f32) -> f32 {
    let mut px = point[0].abs();
    let mut py = point[1].abs();
    let f1 = 2.0 * (HG_KX * px + HG_KY * py).min(0.0);
    px -= f1 * HG_KX;
    py -= f1 * HG_KY;
    // Second reflection uses the swapped constant pair (`k.yx`).
    let f2 = 2.0 * (HG_KY * px + HG_KX * py).min(0.0);
    px -= f2 * HG_KY;
    py -= f2 * HG_KX;
    px -= px.clamp(HG_KZ * r, HG_KW * r);
    py -= r;
    length2(px, py) * signum_branch(py)
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::pentagram_2d`, reproduced without importing the
/// golden.
///
/// A pentagram is the exact five-pointed star field with the golden-ratio inner
/// radius; two reflections fold the query into one `36deg` wedge, then one edge
/// distance signed by the edge half-plane test gives the field.
#[must_use]
pub fn pentagram_2d_sdf(point: [f32; 2], radius: f32) -> f32 {
    let mut px = point[0].abs();
    let mut py = point[1];
    let d1 = (PENTA_K1X * px + PENTA_K1Y * py).max(0.0);
    px -= 2.0 * d1 * PENTA_K1X;
    py -= 2.0 * d1 * PENTA_K1Y;
    let d2 = ((-PENTA_K1X) * px + PENTA_K1Y * py).max(0.0);
    px -= 2.0 * d2 * (-PENTA_K1X);
    py -= 2.0 * d2 * PENTA_K1Y;
    px = px.abs();
    py -= radius;
    let bax = PENTA_INNER * (-PENTA_K1Y);
    let bay = PENTA_INNER * PENTA_K1X - 1.0;
    let bb = bax * bax + bay * bay;
    let h = ((px * bax + py * bay) / bb).clamp(0.0, radius);
    let dx = px - bax * h;
    let dy = py - bay * h;
    length2(dx, dy) * signum_branch(py * bax - px * bay)
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::parallelogram`, reproduced without importing the
/// golden.
///
/// Folds the lower half onto the upper half, takes the smaller of the
/// horizontal-edge and slanted-edge distances, and recovers the interior sign
/// from the signed areas carried in the field's second channel.
#[must_use]
pub fn parallelogram_sdf(point: [f32; 2], half_width: f32, half_height: f32, skew: f32) -> f32 {
    let ex = skew;
    let ey = half_height;
    let (mut px, mut py) = if point[1] < 0.0 {
        (-point[0], -point[1])
    } else {
        (point[0], point[1])
    };
    // Distance to the horizontal (top) edge.
    let mut wx = px - ex;
    let wy = py - ey;
    wx -= wx.clamp(-half_width, half_width);
    let mut d0 = wx * wx + wy * wy;
    let mut d1 = -wy;
    // Signed area selects the near slanted edge; fold again across it.
    let s = px * ey - py * ex;
    if s < 0.0 {
        px = -px;
        py = -py;
    }
    let vx = px - half_width;
    let vy = py;
    let g = ((vx * ex + vy * ey) / (ex * ex + ey * ey)).clamp(-1.0, 1.0);
    let vx2 = vx - ex * g;
    let vy2 = vy - ey * g;
    d0 = d0.min(vx2 * vx2 + vy2 * vy2);
    d1 = d1.min(half_width * half_height - s.abs());
    d0.sqrt() * signum_branch(-d1)
}

/// The portable core-`WGSL` star / parallelogram `SDF` kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the three golden evaluators; see the module documentation.
const SDF_STAR2D_WGSL: &str = r#"
// Star / parallelogram 2D SDF twin: one thread evaluates the hexagram,
// pentagram and parallelogram fields for one query, mirroring the golden
// {hexagram_2d, pentagram_2d, parallelogram} with only reflections, clamps,
// products and one sqrt. Transcendental-free: every trig term is a baked
// compile-time constant.
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_primitives；无第三方引擎源码或衍生代码。

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
    // Hexagram mid-scale radius.
    hex_r: f32,
    // Pentagram outer-tip radius.
    penta_radius: f32,
    // Parallelogram half-width, half-height and horizontal skew.
    para_half_width: f32,
    para_half_height: f32,
    para_skew: f32,
}

struct SdfResult {
    dist_hexagram: f32,
    dist_pentagram: f32,
    dist_parallelogram: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<SdfResult>;

const HG_KX: f32 = -0.5;
const HG_KY: f32 = 0.8660254;
const HG_KZ: f32 = 0.57735026;
const HG_KW: f32 = 1.7320508;
const PENTA_K1X: f32 = 0.809017;
const PENTA_K1Y: f32 = -0.58778525;
const PENTA_INNER: f32 = 0.38196602;

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

fn hexagram_sdf(point_x: f32, point_y: f32, r: f32) -> f32 {
    var px = abs(point_x);
    var py = abs(point_y);
    let f1 = 2.0 * min(HG_KX * px + HG_KY * py, 0.0);
    px = px - f1 * HG_KX;
    py = py - f1 * HG_KY;
    let f2 = 2.0 * min(HG_KY * px + HG_KX * py, 0.0);
    px = px - f2 * HG_KY;
    py = py - f2 * HG_KX;
    px = px - clamp(px, HG_KZ * r, HG_KW * r);
    py = py - r;
    return length2(px, py) * signum_branch(py);
}

fn pentagram_sdf(point_x: f32, point_y: f32, radius: f32) -> f32 {
    var px = abs(point_x);
    var py = point_y;
    let d1 = max(PENTA_K1X * px + PENTA_K1Y * py, 0.0);
    px = px - 2.0 * d1 * PENTA_K1X;
    py = py - 2.0 * d1 * PENTA_K1Y;
    let d2 = max((-PENTA_K1X) * px + PENTA_K1Y * py, 0.0);
    px = px - 2.0 * d2 * (-PENTA_K1X);
    py = py - 2.0 * d2 * PENTA_K1Y;
    px = abs(px);
    py = py - radius;
    let bax = PENTA_INNER * (-PENTA_K1Y);
    let bay = PENTA_INNER * PENTA_K1X - 1.0;
    let bb = bax * bax + bay * bay;
    let h = clamp((px * bax + py * bay) / bb, 0.0, radius);
    let dx = px - bax * h;
    let dy = py - bay * h;
    return length2(dx, dy) * signum_branch(py * bax - px * bay);
}

fn parallelogram_sdf(point_x: f32, point_y: f32, half_width: f32, half_height: f32, skew: f32) -> f32 {
    let ex = skew;
    let ey = half_height;
    var px = select(point_x, -point_x, point_y < 0.0);
    var py = select(point_y, -point_y, point_y < 0.0);
    var wx = px - ex;
    let wy = py - ey;
    wx = wx - clamp(wx, -half_width, half_width);
    var d0 = wx * wx + wy * wy;
    var d1 = -wy;
    let s = px * ey - py * ex;
    let flip = s < 0.0;
    px = select(px, -px, flip);
    py = select(py, -py, flip);
    let vx = px - half_width;
    let vy = py;
    let g = clamp((vx * ex + vy * ey) / (ex * ex + ey * ey), -1.0, 1.0);
    let vx2 = vx - ex * g;
    let vy2 = vy - ey * g;
    d0 = min(d0, vx2 * vx2 + vy2 * vy2);
    d1 = min(d1, half_width * half_height - abs(s));
    return sqrt(d0) * signum_branch(-d1);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: SdfResult;
    out.dist_hexagram = hexagram_sdf(q.px, q.py, q.hex_r);
    out.dist_pentagram = pentagram_sdf(q.px, q.py, q.penta_radius);
    out.dist_parallelogram = parallelogram_sdf(
        q.px,
        q.py,
        q.para_half_width,
        q.para_half_height,
        q.para_skew,
    );
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in [`SDF_STAR2D_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the point and the per-shape
/// parameters, matching the `WGSL` `Query` struct (seven `f32`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Hexagram mid-scale radius.
    hex_r: f32,
    /// Pentagram outer-tip radius.
    penta_radius: f32,
    /// Parallelogram half-width.
    para_half_width: f32,
    /// Parallelogram half-height.
    para_half_height: f32,
    /// Parallelogram horizontal skew.
    para_skew: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `SdfResult`
/// struct: the three signed distances.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed distance to the hexagram.
    dist_hexagram: f32,
    /// Signed distance to the pentagram.
    dist_pentagram: f32,
    /// Signed distance to the parallelogram.
    dist_parallelogram: f32,
}

/// One query: the point plus the per-shape parameters for the three fields.
///
/// `hex_r` scales the hexagram, `penta_radius` the pentagram, and the
/// `para_*` triple the parallelogram. The host enqueues one query per
/// evaluation, and an empty batch is short-circuited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfStar2dQuery {
    /// Query point `x`.
    pub px: f32,
    /// Query point `y`.
    pub py: f32,
    /// Hexagram mid-scale radius.
    pub hex_r: f32,
    /// Pentagram outer-tip radius.
    pub penta_radius: f32,
    /// Parallelogram half-width.
    pub para_half_width: f32,
    /// Parallelogram half-height.
    pub para_half_height: f32,
    /// Parallelogram horizontal skew.
    pub para_skew: f32,
}

impl SdfStar2dQuery {
    /// Builds a query from the point and the per-shape parameters.
    #[must_use]
    pub const fn new(
        px: f32,
        py: f32,
        hex_r: f32,
        penta_radius: f32,
        para_half_width: f32,
        para_half_height: f32,
        para_skew: f32,
    ) -> SdfStar2dQuery {
        SdfStar2dQuery {
            px,
            py,
            hex_r,
            penta_radius,
            para_half_width,
            para_half_height,
            para_skew,
        }
    }
}

/// One resolved query: the three signed distances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfStar2dResult {
    /// Signed distance to the hexagram.
    pub dist_hexagram: f32,
    /// Signed distance to the pentagram.
    pub dist_pentagram: f32,
    /// Signed distance to the parallelogram.
    pub dist_parallelogram: f32,
}

/// Encodes one [`SdfStar2dQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfStar2dQuery) -> GpuQuery {
    GpuQuery {
        px: q.px,
        py: q.py,
        hex_r: q.hex_r,
        penta_radius: q.penta_radius,
        para_half_width: q.para_half_width,
        para_half_height: q.para_half_height,
        para_skew: q.para_skew,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfStar2dResult`].
fn decode_result(raw: &GpuResult) -> SdfStar2dResult {
    SdfStar2dResult {
        dist_hexagram: raw.dist_hexagram,
        dist_pentagram: raw.dist_pentagram,
        dist_parallelogram: raw.dist_parallelogram,
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

/// A compiled, reusable star / parallelogram `SDF` compute pipeline, twinning
/// the golden `ray_scene::sdf_primitives` evaluators `hexagram_2d`,
/// `pentagram_2d` and `parallelogram`.
pub struct GpuSdfStar2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfStar2d {
    /// Compiles the star / parallelogram `SDF` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfStar2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_star2d"),
            source: ShaderSource::Wgsl(SDF_STAR2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_star2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_star2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_star2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfStar2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one [`SdfStar2dResult`]
    /// per input, in order.
    ///
    /// Each distance equals the reference within floating-point tolerance. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[SdfStar2dQuery]) -> Vec<SdfStar2dResult> {
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
            label: Some("prism_volumetric_sdf_star2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_star2d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_star2d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_star2d_bind_group"),
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
            label: Some("prism_volumetric_sdf_star2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_star2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_star2d_pass"),
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
