//! `wgpu` compute twin of the `CIE` 1931 color-space contract
//! ([`cie_xyz`](prism_render_architecture::particle::cie_xyz), particle design
//! §29).
//!
//! The `CPU` golden
//! [`cie_xyz`](prism_render_architecture::particle::cie_xyz) owns the small,
//! deterministic color algebra the particle tint pipeline shares: linear
//! `sRGB`/`Rec.709` <-> `CIE` `XYZ` through the fixed `D65` primaries matrix and
//! its inverse
//! ([`linear_srgb_to_xyz`](prism_render_architecture::particle::cie_xyz::linear_srgb_to_xyz),
//! [`xyz_to_linear_srgb`](prism_render_architecture::particle::cie_xyz::xyz_to_linear_srgb)),
//! the `XYZ` <-> `xyY` split that separates chromaticity from luminance
//! ([`xyz_to_xyy`](prism_render_architecture::particle::cie_xyz::xyz_to_xyy),
//! [`xyy_to_xyz`](prism_render_architecture::particle::cie_xyz::xyy_to_xyz)), and
//! `Bradford` chromatic adaptation between reference whites
//! ([`bradford_adapt`](prism_render_architecture::particle::cie_xyz::bradford_adapt)).
//! [`GpuCieXyz`] is the on-device twin: one thread per color reproduces every
//! transform branch for branch, so a passing real-device parity test is direct
//! evidence the ported kernel evaluates the same linear algebra and classifies
//! the same degenerate case the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Every per-color answer the reference computes is reproduced for a batch of
//! independent colors packed into one query: the `XYZ` image of a linear
//! `sRGB` input, the linear `sRGB` image of an `XYZ` input, the `xyY` split of
//! an `XYZ` input, the `XYZ` reconstruction of an `xyY` input, and the
//! `Bradford`-adapted `XYZ` of a source color transported between two reference
//! whites. The host-only `std430` packing helpers the reference exposes
//! (`to_std430`, `gpu_storage_bytes`) are not kernel math and are left to the
//! `CPU` reference; only the arithmetic transforms are twinned on-device.
//!
//! # Correctness model
//!
//! Each transform threads through a fixed, non-reorderable sequence of
//! multiplies, adds and divides (no transcendental, no `sqrt`), so `CPU` and
//! `GPU` evaluate the same closed form in the same associativity. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on every continuous quantity, tight enough to catch a
//! genuinely wrong port (a dropped term, a swapped coefficient, a transposed
//! matrix) yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! The reference guards two divisions, and the kernel mirrors both exactly with
//! a `<= 0.0` test rather than an `f32` `==`. When the tristimulus sum
//! `X + Y + Z` is non-positive,
//! [`xyz_to_xyy`](prism_render_architecture::particle::cie_xyz::xyz_to_xyy)
//! falls back to the `D65` white chromaticity while preserving `Y`; when the
//! chromaticity `y` is non-positive,
//! [`xyy_to_xyz`](prism_render_architecture::particle::cie_xyz::xyy_to_xyz) maps
//! to the `XYZ` origin. `Bradford` adaptation leaves a channel unscaled when its
//! source white cone response is non-positive, guarding that division too. An
//! empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `dot`, `+ - * /` and
//! unsigned index arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `tan`, no inverse trigonometry, no `sqrt` and no optional device feature, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each
//! thread performs a fixed, bounded sequence of arithmetic, so the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_xyz`；
//! standard `sRGB`/`Rec.709` `D65` primaries and `Bradford` chromatic adaptation
//! plus `wgpu` compute dispatch; no third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::cie_xyz::{
    bradford_adapt, linear_srgb_to_xyz, xyy_to_xyz, xyz_to_linear_srgb, xyz_to_xyy, LinearSrgb,
    Xyy, Xyz,
};
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

/// The portable core-`WGSL` `CIE` color kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`cie_xyz`](prism_render_architecture::particle::cie_xyz) branch for branch;
/// see the module documentation for the algorithm.
const CIE_XYZ_WGSL: &str = r#"
// CIE 1931 color twin: one thread per color reproduces the linear sRGB <-> XYZ
// matrix conversions, the XYZ <-> xyY split with its divide-by-zero guards, and
// Bradford chromatic adaptation between reference whites. It mirrors the CPU
// golden particle::cie_xyz branch for branch, uses only the portable core-WGSL
// subset (dot and + - * / plus unsigned index math), needs no sqrt and no
// transcendental call and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12. There is no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::cie_xyz; no
// third-party engine source or derived code.

// x / y chromaticity of the CIE D65 reference white, the divide-by-zero
// fallback for xyz_to_xyy. Matches the reference D65_CHROMA_X / D65_CHROMA_Y.
const D65_CHROMA_X: f32 = 0.3127;
const D65_CHROMA_Y: f32 = 0.329;

// Linear sRGB/Rec.709 D65 primaries matrix mapping linear sRGB to CIE XYZ, held
// as its three rows so the matrix-times-vector mirrors the reference row order.
const SRGB_TO_XYZ_R0 = vec3<f32>(0.4123908, 0.35758433, 0.1804808);
const SRGB_TO_XYZ_R1 = vec3<f32>(0.212639, 0.71516865, 0.07219232);
const SRGB_TO_XYZ_R2 = vec3<f32>(0.019330818, 0.11919478, 0.95053215);

// Inverse of the primaries matrix, mapping CIE XYZ back to linear sRGB.
const XYZ_TO_SRGB_R0 = vec3<f32>(3.24097, -1.5373832, -0.49861076);
const XYZ_TO_SRGB_R1 = vec3<f32>(-0.96924365, 1.8759675, 0.04155506);
const XYZ_TO_SRGB_R2 = vec3<f32>(0.05563008, -0.20397696, 1.0569715);

// Bradford LMS cone-response matrix mapping CIE XYZ to the Bradford basis.
const BRADFORD_R0 = vec3<f32>(0.8951, 0.2664, -0.1614);
const BRADFORD_R1 = vec3<f32>(-0.7502, 1.7135, 0.0367);
const BRADFORD_R2 = vec3<f32>(0.0389, -0.0685, 1.0296);

// Inverse Bradford matrix, mapping the Bradford LMS basis back to CIE XYZ.
const BRADFORD_INV_R0 = vec3<f32>(0.9869929, -0.1470543, 0.1599627);
const BRADFORD_INV_R1 = vec3<f32>(0.4323053, 0.5183603, 0.0492912);
const BRADFORD_INV_R2 = vec3<f32>(-0.0085287, 0.0400428, 0.9684867);

struct Params {
    // Number of colors in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Linear sRGB input for the sRGB -> XYZ conversion; a pad lane follows.
    rgb: vec3<f32>,
    pad0: f32,
    // XYZ input for the XYZ -> sRGB conversion and the XYZ -> xyY split.
    xyz: vec3<f32>,
    pad1: f32,
    // xyY input (x, y, Y) for the xyY -> XYZ reconstruction.
    xyy: vec3<f32>,
    pad2: f32,
    // Source color for Bradford adaptation.
    bsrc: vec3<f32>,
    pad3: f32,
    // Source reference white for Bradford adaptation.
    bsw: vec3<f32>,
    pad4: f32,
    // Destination reference white for Bradford adaptation.
    bdw: vec3<f32>,
    pad5: f32,
}

struct Result {
    // XYZ image of the linear sRGB input.
    to_xyz: vec3<f32>,
    pad0: f32,
    // Linear sRGB image of the XYZ input.
    to_srgb: vec3<f32>,
    pad1: f32,
    // xyY (x, y, Y) split of the XYZ input.
    to_xyy: vec3<f32>,
    pad2: f32,
    // XYZ reconstruction of the xyY input.
    from_xyy: vec3<f32>,
    pad3: f32,
    // Bradford-adapted XYZ of the source color.
    bradford: vec3<f32>,
    pad4: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Multiplies the 3x3 matrix given by its three rows by the column vector v,
// mirroring the reference `mat3_mul_vec`: each output lane is the row dotted
// with v, so the per-lane term order matches the scalar golden.
fn mat3_mul_vec(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(r0, v), dot(r1, v), dot(r2, v));
}

// Splits a CIE XYZ color into chromaticity (x, y) and luminance Y, mirroring
// the reference `xyz_to_xyy`. A non-positive tristimulus sum has no defined
// chromaticity and falls back to the D65 white chromaticity while preserving Y,
// guarding the division by the sum.
fn xyz_to_xyy(c: vec3<f32>) -> vec3<f32> {
    let sum = c.x + c.y + c.z;
    if (sum <= 0.0) {
        return vec3<f32>(D65_CHROMA_X, D65_CHROMA_Y, c.y);
    }
    return vec3<f32>(c.x / sum, c.y / sum, c.y);
}

// Reconstructs a CIE XYZ color from chromaticity (c.x, c.y) and luminance
// (c.z), mirroring the reference `xyy_to_xyz`. A non-positive chromaticity y is
// treated as black and maps to the XYZ origin, guarding the division by y.
fn xyy_to_xyz(c: vec3<f32>) -> vec3<f32> {
    if (c.y <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let ratio = c.z / c.y;
    let x = c.x * ratio;
    let z = (1.0 - c.x - c.y) * ratio;
    return vec3<f32>(x, c.z, z);
}

// Scales one Bradford cone-response lane by the destination/source white ratio,
// mirroring the reference guard: a non-positive source cone response leaves the
// lane unscaled so the division is never taken.
fn bradford_scale(s: f32, sw: f32, dw: f32) -> f32 {
    if (sw <= 0.0) {
        return s;
    }
    return s * (dw / sw);
}

// Transports a CIE XYZ color measured under src_white to dst_white using
// Bradford chromatic adaptation, mirroring the reference `bradford_adapt`.
fn bradford_adapt(src: vec3<f32>, src_white: vec3<f32>, dst_white: vec3<f32>) -> vec3<f32> {
    let src_lms = mat3_mul_vec(BRADFORD_R0, BRADFORD_R1, BRADFORD_R2, src);
    let sw = mat3_mul_vec(BRADFORD_R0, BRADFORD_R1, BRADFORD_R2, src_white);
    let dw = mat3_mul_vec(BRADFORD_R0, BRADFORD_R1, BRADFORD_R2, dst_white);
    var scaled: vec3<f32>;
    scaled.x = bradford_scale(src_lms.x, sw.x, dw.x);
    scaled.y = bradford_scale(src_lms.y, sw.y, dw.y);
    scaled.z = bradford_scale(src_lms.z, sw.z, dw.z);
    return mat3_mul_vec(BRADFORD_INV_R0, BRADFORD_INV_R1, BRADFORD_INV_R2, scaled);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.to_xyz = mat3_mul_vec(SRGB_TO_XYZ_R0, SRGB_TO_XYZ_R1, SRGB_TO_XYZ_R2, q.rgb);
    out.to_srgb = mat3_mul_vec(XYZ_TO_SRGB_R0, XYZ_TO_SRGB_R1, XYZ_TO_SRGB_R2, q.xyz);
    out.to_xyy = xyz_to_xyy(q.xyz);
    out.from_xyy = xyy_to_xyz(q.xyy);
    out.bradford = bradford_adapt(q.bsrc, q.bsw, q.bdw);
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    out.pad3 = 0.0;
    out.pad4 = 0.0;
    results[idx] = out;
}
"#;

/// One `CIE` color query bundling every input the reference transforms consume
/// for a single color: the linear `sRGB` input, the `XYZ` input, the `xyY`
/// input and the source color plus two reference whites for `Bradford`
/// adaptation.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_xyz`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CieXyzQuery {
    /// Linear `sRGB` input for the `sRGB` -> `XYZ` conversion.
    pub rgb: LinearSrgb,
    /// `XYZ` input for the `XYZ` -> `sRGB` conversion and the `XYZ` -> `xyY`
    /// split.
    pub xyz: Xyz,
    /// `xyY` input for the `xyY` -> `XYZ` reconstruction.
    pub xyy: Xyy,
    /// Source color for `Bradford` adaptation.
    pub bradford_src: Xyz,
    /// Source reference white for `Bradford` adaptation.
    pub bradford_src_white: Xyz,
    /// Destination reference white for `Bradford` adaptation.
    pub bradford_dst_white: Xyz,
}

impl CieXyzQuery {
    /// Builds a query from the four color inputs and the two `Bradford`
    /// reference whites.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_xyz`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(
        rgb: LinearSrgb,
        xyz: Xyz,
        xyy: Xyy,
        bradford_src: Xyz,
        bradford_src_white: Xyz,
        bradford_dst_white: Xyz,
    ) -> CieXyzQuery {
        CieXyzQuery {
            rgb,
            xyz,
            xyy,
            bradford_src,
            bradford_src_white,
            bradford_dst_white,
        }
    }
}

/// The resolved answer for one color, mirroring every transform the reference
/// exposes.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_xyz`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CieXyzResult {
    /// `XYZ` image of the linear `sRGB` input, matching
    /// [`linear_srgb_to_xyz`](prism_render_architecture::particle::cie_xyz::linear_srgb_to_xyz).
    pub to_xyz: Xyz,
    /// Linear `sRGB` image of the `XYZ` input, matching
    /// [`xyz_to_linear_srgb`](prism_render_architecture::particle::cie_xyz::xyz_to_linear_srgb).
    pub to_srgb: LinearSrgb,
    /// `xyY` split of the `XYZ` input, matching
    /// [`xyz_to_xyy`](prism_render_architecture::particle::cie_xyz::xyz_to_xyy).
    pub to_xyy: Xyy,
    /// `XYZ` reconstruction of the `xyY` input, matching
    /// [`xyy_to_xyz`](prism_render_architecture::particle::cie_xyz::xyy_to_xyz).
    pub from_xyy: Xyz,
    /// `Bradford`-adapted `XYZ` of the source color, matching
    /// [`bradford_adapt`](prism_render_architecture::particle::cie_xyz::bradford_adapt).
    pub bradford: Xyz,
}

/// Evaluates the `CPU` golden for one query, producing every transform the
/// on-device twin reproduces.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_xyz`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &CieXyzQuery) -> CieXyzResult {
    CieXyzResult {
        to_xyz: linear_srgb_to_xyz(&query.rgb),
        to_srgb: xyz_to_linear_srgb(&query.xyz),
        to_xyy: xyz_to_xyy(&query.xyz),
        from_xyy: xyy_to_xyz(&query.xyy),
        bradford: bradford_adapt(
            &query.bradford_src,
            &query.bradford_src_white,
            &query.bradford_dst_white,
        ),
    }
}

/// `repr(C)` `std430` layout of one packed query: six `vec4` slots, each a
/// `vec3` triple on its `16`-byte-aligned slot with a padding lane, exactly as
/// the `WGSL` `Query` struct reads it — `96` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Linear `sRGB` input.
    rgb: [f32; 3],
    /// Padding lane after the `sRGB` input.
    pad0: f32,
    /// `XYZ` input.
    xyz: [f32; 3],
    /// Padding lane after the `XYZ` input.
    pad1: f32,
    /// `xyY` input.
    xyy: [f32; 3],
    /// Padding lane after the `xyY` input.
    pad2: f32,
    /// `Bradford` source color.
    bsrc: [f32; 3],
    /// Padding lane after the source color.
    pad3: f32,
    /// `Bradford` source reference white.
    bsw: [f32; 3],
    /// Padding lane after the source white.
    pad4: f32,
    /// `Bradford` destination reference white.
    bdw: [f32; 3],
    /// Padding lane after the destination white.
    pad5: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &CieXyzQuery) -> GpuQuery {
        GpuQuery {
            rgb: [query.rgb.r, query.rgb.g, query.rgb.b],
            pad0: 0.0,
            xyz: [query.xyz.x, query.xyz.y, query.xyz.z],
            pad1: 0.0,
            xyy: [query.xyy.x, query.xyy.y, query.xyy.big_y],
            pad2: 0.0,
            bsrc: [
                query.bradford_src.x,
                query.bradford_src.y,
                query.bradford_src.z,
            ],
            pad3: 0.0,
            bsw: [
                query.bradford_src_white.x,
                query.bradford_src_white.y,
                query.bradford_src_white.z,
            ],
            pad4: 0.0,
            bdw: [
                query.bradford_dst_white.x,
                query.bradford_dst_white.y,
                query.bradford_dst_white.z,
            ],
            pad5: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: five `vec4` slots, each a `vec3`
/// triple with a padding lane matching the `WGSL` `Result` struct — `80` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `XYZ` image of the linear `sRGB` input.
    to_xyz: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// Linear `sRGB` image of the `XYZ` input.
    to_srgb: [f32; 3],
    /// Padding lane.
    pad1: f32,
    /// `xyY` split of the `XYZ` input.
    to_xyy: [f32; 3],
    /// Padding lane.
    pad2: f32,
    /// `XYZ` reconstruction of the `xyY` input.
    from_xyy: [f32; 3],
    /// Padding lane.
    pad3: f32,
    /// `Bradford`-adapted `XYZ` of the source color.
    bradford: [f32; 3],
    /// Padding lane.
    pad4: f32,
}

/// Uniform parameters for one dispatch: the color count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of colors in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable `CIE` color-conversion compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_xyz`；
/// no third-party engine source or derived code.
pub struct GpuCieXyz {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCieXyz {
    /// Compiles the `CIE` color kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_xyz`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCieXyz {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cie_xyz"),
            source: ShaderSource::Wgsl(CIE_XYZ_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cie_xyz_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cie_xyz_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cie_xyz_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCieXyz {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every color on-device and returns one [`CieXyzResult`] per input,
    /// in order.
    ///
    /// Each result equals the reference transforms to within the tolerance
    /// documented on this module. An empty input returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_xyz`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[CieXyzQuery]) -> Vec<CieXyzResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cie_xyz_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cie_xyz_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cie_xyz_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cie_xyz_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cie_xyz_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cie_xyz_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cie_xyz_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per color, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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

/// Decodes one packed [`GpuResult`] into the public [`CieXyzResult`].
fn decode_result(raw: &GpuResult) -> CieXyzResult {
    CieXyzResult {
        to_xyz: Xyz::new(raw.to_xyz[0], raw.to_xyz[1], raw.to_xyz[2]),
        to_srgb: LinearSrgb::new(raw.to_srgb[0], raw.to_srgb[1], raw.to_srgb[2]),
        to_xyy: Xyy::new(raw.to_xyy[0], raw.to_xyy[1], raw.to_xyy[2]),
        from_xyy: Xyz::new(raw.from_xyy[0], raw.from_xyy[1], raw.from_xyy[2]),
        bradford: Xyz::new(raw.bradford[0], raw.bradford[1], raw.bradford[2]),
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
