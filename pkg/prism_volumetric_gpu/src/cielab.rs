//! `wgpu` compute twin of the `CIE` 1976 `L*a*b*` (`CIELAB`) color contract
//! ([`cielab`](prism_render_architecture::particle::cielab), particle color
//! pipeline).
//!
//! The `CPU` golden
//! [`cielab`](prism_render_architecture::particle::cielab) is the
//! perceptually-uniform bridge the particle color stages read: it converts
//! `CIE` `XYZ` to and from `CIELAB`
//! ([`xyz_to_lab`](prism_render_architecture::particle::cielab::xyz_to_lab),
//! [`lab_to_xyz`](prism_render_architecture::particle::cielab::lab_to_xyz)),
//! evaluates the lightness nonlinearity and its exact inverse
//! ([`lab_f`](prism_render_architecture::particle::cielab::lab_f),
//! [`lab_f_inv`](prism_render_architecture::particle::cielab::lab_f_inv)) around
//! the `δ = 6/29` knot, computes the `CIE76` and `CIE94` color differences
//! ([`delta_e_76`](prism_render_architecture::particle::cielab::delta_e_76),
//! [`delta_e_94`](prism_render_architecture::particle::cielab::delta_e_94)), and
//! blends two colors
//! ([`lab_lerp`](prism_render_architecture::particle::cielab::lab_lerp)). The
//! cube root is a self-contained Newton iteration
//! ([`cbrt_newton`](prism_render_architecture::particle::cielab::cbrt_newton))
//! so the curve never calls a transcendental intrinsic. [`GpuCielab`] is the
//! on-device twin: one thread solves one query, so a passing real-device parity
//! test is direct evidence the ported kernel computes the same colors and
//! color differences the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every continuous quantity the reference computes is reproduced per query:
//! the Newton cube root `cbrt_newton`, the forward and inverse lightness legs
//! `lab_f` / `lab_f_inv`, the `xyz_to_lab` and `lab_to_xyz` conversions, the
//! `delta_e_76` and `delta_e_94` color differences (with the `graphic`
//! application selector carried as a per-query input), and the `lab_lerp`
//! blend. The pure host-side byte utilities
//! [`to_std430`](prism_render_architecture::particle::cielab::to_std430) and
//! [`gpu_storage_bytes`](prism_render_architecture::particle::cielab::gpu_storage_bytes)
//! are *not* twinned: they describe a `std430` packing on the host and run no
//! device arithmetic, so there is nothing for a `GPU` kernel to reproduce.
//!
//! # Correctness model
//!
//! The `cbrt_newton` kernel is a strict branch-for-branch port of the golden:
//! the same sign reflection, the same divide/multiply-by-eight range reduction
//! into `[1, 8)`, the same fixed `1.5` seed, and the same bounded Newton budget
//! `y ← y − (y³ − v) / (3y²)`. The lightness knot comparisons and the
//! `graphic` selector are discrete, but every reported value is a float that
//! threads through multiplies, adds, one guarded division and `sqrt`, so `CPU`
//! and `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits. The parity test
//! therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`), tight enough to catch a wrong port yet loose enough to
//! admit legal fused multiply-add contraction.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, `abs`,
//! `min`, `max`, `select` and `sqrt` — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, `cbrt`, no inverse trigonometry and no optional device feature, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`. The only loops are the
//! cube-root range reduction and the fixed Newton budget, both provably
//! terminating.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::cielab`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `CIELAB` kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the `CPU`
/// golden [`cielab`](prism_render_architecture::particle::cielab) branch for
/// branch; see the module documentation for the algorithm.
const CIELAB_WGSL: &str = r#"
// CIELAB twin: one thread per query reproduces the Newton cube root, the
// forward/inverse lightness legs, the XYZ<->Lab conversions, the CIE76/CIE94
// color differences and the Lab blend. It mirrors the CPU golden
// `particle::cielab` branch for branch, uses only the portable core-WGSL subset
// (+ - * /, abs, min, max, select and sqrt), takes no optional feature and so
// runs unmodified on Metal, Vulkan and DX12. The only loops are the cube-root
// range reduction and the fixed Newton budget, both provably terminating.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::cielab；无第三方
// 引擎源码或衍生代码。

// Magnitude below which the Newton denominator is treated as zero, breaking the
// iteration instead of dividing by ~zero. Matches the reference `CMP_EPS`; the
// compare rule used instead of an f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

// The CIELAB knot delta = 6/29 and its derived constants, written as divisions
// of named integers exactly as the golden does.
const DELTA: f32 = 6.0 / 29.0;
const DELTA_CUBED: f32 = DELTA * DELTA * DELTA;
const THREE_DELTA_SQ: f32 = 3.0 * DELTA * DELTA;
const TOE_OFFSET: f32 = 4.0 / 29.0;

// Newton refinement budget and the [1, 8) range-reduction window bounds.
const CBRT_ITERATIONS: u32 = 16u;
const CBRT_WINDOW_HI: f32 = 8.0;
const CBRT_WINDOW_LO: f32 = 1.0;

// CIE D65 reference white tristimulus, normalized to Y = 1.
const XN: f32 = 0.950489;
const YN: f32 = 1.0;
const ZN: f32 = 0.888840;

// CIE94 weights: graphic-arts then textiles application constants.
const K1_GRAPHIC: f32 = 0.045;
const K2_GRAPHIC: f32 = 0.015;
const KL_GRAPHIC: f32 = 1.0;
const K1_TEXTILE: f32 = 0.048;
const K2_TEXTILE: f32 = 0.014;
const KL_TEXTILE: f32 = 2.0;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // XYZ triple fed to xyz_to_lab, with a trailing pad lane.
    xyz_in: vec3<f32>,
    pad_x: f32,
    // Lab triple fed to lab_to_xyz, with a trailing pad lane.
    lab_in: vec3<f32>,
    pad_l: f32,
    // Reference Lab for delta_e_76/delta_e_94 and endpoint `a` of lab_lerp.
    lab_a: vec3<f32>,
    pad_a: f32,
    // Sample Lab for delta_e_76/delta_e_94 and endpoint `b` of lab_lerp.
    lab_b: vec3<f32>,
    pad_b: f32,
    // Scalar inputs: cbrt_newton argument, lab_f argument, lab_f_inv argument,
    // and the lab_lerp parameter t.
    scalars: vec4<f32>,
    // 1 selects the graphic-arts CIE94 constants, 0 the textiles constants.
    graphic: u32,
    pad_g0: u32,
    pad_g1: u32,
    pad_g2: u32,
}

struct Result {
    // xyz_to_lab output Lab, with a trailing pad lane.
    lab_from_xyz: vec3<f32>,
    pad0: f32,
    // lab_to_xyz output XYZ, with a trailing pad lane.
    xyz_from_lab: vec3<f32>,
    pad1: f32,
    // lab_lerp output Lab, with a trailing pad lane.
    lab_lerp_out: vec3<f32>,
    pad2: f32,
    // Scalar outputs: cbrt_newton, lab_f, lab_f_inv, delta_e_76.
    scalars: vec4<f32>,
    // delta_e_94 output, with three pad lanes.
    de94: f32,
    pad3: f32,
    pad4: f32,
    pad5: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Real cube root via the golden's self-contained Newton iteration: sign
// reflection, divide/multiply-by-eight range reduction into [1, 8), a fixed 1.5
// seed and a bounded Newton budget. Mirrors `cbrt_newton`.
fn cbrt_newton(x: f32) -> f32 {
    let sign = select(1.0, -1.0, x < 0.0);
    let ax = abs(x);
    if (ax <= 0.0) {
        return 0.0;
    }
    var v = ax;
    var scale = 1.0;
    loop {
        if (v <= CBRT_WINDOW_HI) {
            break;
        }
        v = v / 8.0;
        scale = scale * 2.0;
    }
    loop {
        if (v >= CBRT_WINDOW_LO) {
            break;
        }
        v = v * 8.0;
        scale = scale / 2.0;
    }
    var y = 1.5;
    for (var i = 0u; i < CBRT_ITERATIONS; i = i + 1u) {
        let denom = 3.0 * y * y;
        if (denom < CMP_EPS) {
            break;
        }
        y = y - (y * y * y - v) / denom;
    }
    return sign * y * scale;
}

// Forward CIELAB nonlinearity: cube root above the knot, linear toe at or below
// it. Mirrors `lab_f`.
fn lab_f(t: f32) -> f32 {
    if (t > DELTA_CUBED) {
        return cbrt_newton(t);
    }
    return t / THREE_DELTA_SQ + TOE_OFFSET;
}

// Exact inverse of lab_f: an integer cube above the knot, inverse linear toe at
// or below it. Mirrors `lab_f_inv`.
fn lab_f_inv(t: f32) -> f32 {
    if (t > DELTA) {
        return t * t * t;
    }
    return THREE_DELTA_SQ * (t - TOE_OFFSET);
}

// XYZ (relative to D65) to CIELAB. Mirrors `xyz_to_lab`; the Lab triple is
// packed as (l, a, b).
fn xyz_to_lab(c: vec3<f32>) -> vec3<f32> {
    let fx = lab_f(c.x / XN);
    let fy = lab_f(c.y / YN);
    let fz = lab_f(c.z / ZN);
    return vec3<f32>(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz));
}

// CIELAB (l, a, b) back to XYZ relative to D65. Mirrors `lab_to_xyz`.
fn lab_to_xyz(c: vec3<f32>) -> vec3<f32> {
    let fy = (c.x + 16.0) / 116.0;
    let fx = fy + c.y / 500.0;
    let fz = fy - c.z / 200.0;
    return vec3<f32>(XN * lab_f_inv(fx), YN * lab_f_inv(fy), ZN * lab_f_inv(fz));
}

// CIE76 color difference: Euclidean distance between two Lab colors. Mirrors
// `delta_e_76`.
fn delta_e_76(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b;
    return sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
}

// CIE94 color difference weighting chroma and hue by the reference chroma.
// Mirrors `delta_e_94`; `graphic` selects the graphic-arts constants.
fn delta_e_94(reference: vec3<f32>, sample: vec3<f32>, graphic: bool) -> f32 {
    var kl = KL_TEXTILE;
    var k1 = K1_TEXTILE;
    var k2 = K2_TEXTILE;
    if (graphic) {
        kl = KL_GRAPHIC;
        k1 = K1_GRAPHIC;
        k2 = K2_GRAPHIC;
    }

    let dl = reference.x - sample.x;
    let da = reference.y - sample.y;
    let db = reference.z - sample.z;

    let c_ref = sqrt(reference.y * reference.y + reference.z * reference.z);
    let c_sample = sqrt(sample.y * sample.y + sample.z * sample.z);
    let dc = c_ref - c_sample;

    let dh_sq = max(da * da + db * db - dc * dc, 0.0);
    let dh = sqrt(dh_sq);

    let sl = 1.0;
    let sc = 1.0 + k1 * c_ref;
    let sh = 1.0 + k2 * c_ref;

    let term_l = dl / (kl * sl);
    let term_c = dc / sc;
    let term_h = dh / sh;

    return sqrt(term_l * term_l + term_c * term_c + term_h * term_h);
}

// Component-wise linear blend of two Lab colors. Mirrors `lab_lerp`.
fn lab_lerp(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let graphic = q.graphic != 0u;

    var out: Result;
    out.lab_from_xyz = xyz_to_lab(q.xyz_in);
    out.pad0 = 0.0;
    out.xyz_from_lab = lab_to_xyz(q.lab_in);
    out.pad1 = 0.0;
    out.lab_lerp_out = lab_lerp(q.lab_a, q.lab_b, q.scalars.w);
    out.pad2 = 0.0;
    out.scalars = vec4<f32>(
        cbrt_newton(q.scalars.x),
        lab_f(q.scalars.y),
        lab_f_inv(q.scalars.z),
        delta_e_76(q.lab_a, q.lab_b),
    );
    out.de94 = delta_e_94(q.lab_a, q.lab_b, graphic);
    out.pad3 = 0.0;
    out.pad4 = 0.0;
    out.pad5 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`CIELAB_WGSL`].
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
/// Every `vec3` lane carries a trailing pad word so each stays `16`-byte
/// aligned on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `XYZ` input for `xyz_to_lab`.
    xyz_in: [f32; 3],
    /// Pad lane after `xyz_in`.
    pad_x: f32,
    /// `Lab` input for `lab_to_xyz`.
    lab_in: [f32; 3],
    /// Pad lane after `lab_in`.
    pad_l: f32,
    /// Reference `Lab` for the color differences and `lab_lerp` endpoint `a`.
    lab_a: [f32; 3],
    /// Pad lane after `lab_a`.
    pad_a: f32,
    /// Sample `Lab` for the color differences and `lab_lerp` endpoint `b`.
    lab_b: [f32; 3],
    /// Pad lane after `lab_b`.
    pad_b: f32,
    /// Scalar inputs: `cbrt_newton`, `lab_f`, `lab_f_inv` arguments and the
    /// `lab_lerp` parameter `t`.
    scalars: [f32; 4],
    /// `1` selects the graphic-arts `CIE94` constants, `0` the textiles ones.
    graphic: u32,
    /// Padding word.
    pad_g0: u32,
    /// Padding word.
    pad_g1: u32,
    /// Padding word.
    pad_g2: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `xyz_to_lab` output `Lab`.
    lab_from_xyz: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// `lab_to_xyz` output `XYZ`.
    xyz_from_lab: [f32; 3],
    /// Padding lane.
    pad1: f32,
    /// `lab_lerp` output `Lab`.
    lab_lerp: [f32; 3],
    /// Padding lane.
    pad2: f32,
    /// Scalar outputs: `cbrt_newton`, `lab_f`, `lab_f_inv`, `delta_e_76`.
    scalars: [f32; 4],
    /// `delta_e_94` output.
    de94: f32,
    /// Padding lane.
    pad3: f32,
    /// Padding lane.
    pad4: f32,
    /// Padding lane.
    pad5: f32,
}

/// One query for the `CIELAB` twin: an `XYZ` triple to convert forward, a `Lab`
/// triple to convert back, a reference/sample `Lab` pair for the color
/// differences and the blend, three scalar arguments and the blend parameter,
/// plus the `CIE94` application selector.
///
/// The functions are independent per query, so a single query exercises every
/// twinned function at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CielabQuery {
    /// `XYZ` triple (relative to `D65`) fed to `xyz_to_lab`.
    pub xyz_in: [f32; 3],
    /// `Lab` triple (`l`, `a`, `b`) fed to `lab_to_xyz`.
    pub lab_in: [f32; 3],
    /// Reference `Lab` (`l`, `a`, `b`) for `delta_e_76` / `delta_e_94` and the
    /// first endpoint of `lab_lerp`.
    pub lab_a: [f32; 3],
    /// Sample `Lab` (`l`, `a`, `b`) for `delta_e_76` / `delta_e_94` and the
    /// second endpoint of `lab_lerp`.
    pub lab_b: [f32; 3],
    /// Argument to `cbrt_newton`.
    pub cbrt_x: f32,
    /// Argument to `lab_f`.
    pub lab_f_t: f32,
    /// Argument to `lab_f_inv`.
    pub lab_f_inv_t: f32,
    /// Blend parameter `t` for `lab_lerp`.
    pub lerp_t: f32,
    /// `true` selects the graphic-arts `CIE94` constants, `false` the textiles
    /// constants.
    pub graphic: bool,
}

/// One resolved answer for a single query, mirroring every continuous value the
/// reference reports across its twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CielabResult {
    /// `cbrt_newton` output, matching
    /// [`cbrt_newton`](prism_render_architecture::particle::cielab::cbrt_newton).
    pub cbrt: f32,
    /// `lab_f` output, matching
    /// [`lab_f`](prism_render_architecture::particle::cielab::lab_f).
    pub lab_f: f32,
    /// `lab_f_inv` output, matching
    /// [`lab_f_inv`](prism_render_architecture::particle::cielab::lab_f_inv).
    pub lab_f_inv: f32,
    /// `xyz_to_lab` output `Lab` (`l`, `a`, `b`), matching
    /// [`xyz_to_lab`](prism_render_architecture::particle::cielab::xyz_to_lab).
    pub lab_from_xyz: [f32; 3],
    /// `lab_to_xyz` output `XYZ`, matching
    /// [`lab_to_xyz`](prism_render_architecture::particle::cielab::lab_to_xyz).
    pub xyz_from_lab: [f32; 3],
    /// `delta_e_76` output, matching
    /// [`delta_e_76`](prism_render_architecture::particle::cielab::delta_e_76).
    pub delta_e_76: f32,
    /// `delta_e_94` output, matching
    /// [`delta_e_94`](prism_render_architecture::particle::cielab::delta_e_94).
    pub delta_e_94: f32,
    /// `lab_lerp` output `Lab` (`l`, `a`, `b`), matching
    /// [`lab_lerp`](prism_render_architecture::particle::cielab::lab_lerp).
    pub lab_lerp: [f32; 3],
}

/// Encodes one [`CielabQuery`] into its `std430` [`GpuQuery`] slot, mapping the
/// `graphic` selector to a `u32` the shader reads.
fn encode_query(q: &CielabQuery) -> GpuQuery {
    GpuQuery {
        xyz_in: q.xyz_in,
        pad_x: 0.0,
        lab_in: q.lab_in,
        pad_l: 0.0,
        lab_a: q.lab_a,
        pad_a: 0.0,
        lab_b: q.lab_b,
        pad_b: 0.0,
        scalars: [q.cbrt_x, q.lab_f_t, q.lab_f_inv_t, q.lerp_t],
        graphic: u32::from(q.graphic),
        pad_g0: 0,
        pad_g1: 0,
        pad_g2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`CielabResult`].
fn decode_result(raw: &GpuResult) -> CielabResult {
    CielabResult {
        cbrt: raw.scalars[0],
        lab_f: raw.scalars[1],
        lab_f_inv: raw.scalars[2],
        lab_from_xyz: raw.lab_from_xyz,
        xyz_from_lab: raw.xyz_from_lab,
        delta_e_76: raw.scalars[3],
        delta_e_94: raw.de94,
        lab_lerp: raw.lab_lerp,
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

/// A compiled, reusable `CIELAB` compute pipeline, twinning the `CPU` golden
/// [`cielab`](prism_render_architecture::particle::cielab).
pub struct GpuCielab {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCielab {
    /// Compiles the `CIELAB` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCielab {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cielab"),
            source: ShaderSource::Wgsl(CIELAB_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cielab_bind_group_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cielab_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cielab_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCielab {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`CielabResult`] per
    /// input, in order.
    ///
    /// Every reported value matches the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[CielabQuery]) -> Vec<CielabResult> {
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
            label: Some("prism_volumetric_cielab_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cielab_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cielab_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cielab_bind_group"),
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
            label: Some("prism_volumetric_cielab_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cielab_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cielab_pass"),
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
