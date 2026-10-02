//! `wgpu` compute twin of the anisotropic texture-footprint golden
//! ([`anisotropic_footprint`](prism_render_architecture::particle::anisotropic_footprint),
//! particle design §16, §17).
//!
//! The `CPU` golden
//! ([`anisotropic_footprint`](prism_render_architecture::particle::anisotropic_footprint))
//! recovers the texture-space sampling ellipse a screen pixel covers from the
//! screen-space partial derivatives `ddx` / `ddy` (the Jacobian columns, in
//! texels). It forms the symmetric metric `M = [[A, B], [B, C]]` with
//! `A = ddx·ddx`, `B = ddx·ddy`, `C = ddy·ddy`, diagonalizes it in closed form
//! (`λ = mean ± disc`, `mean = (A + C)/2`, `disc = sqrt(((A - C)/2)² + B²)`),
//! and takes the square roots of the eigenvalues as the ellipse half-axes. The
//! `anisotropy` ratio clamps to `[1, max_anisotropy]`, the trilinear `LOD`
//! ([`lod_trilinear`](prism_render_architecture::particle::anisotropic_footprint::lod_trilinear))
//! is the `log2` of the longest gradient length, the anisotropic `LOD`
//! ([`Footprint::lod_anisotropic`](prism_render_architecture::particle::anisotropic_footprint::Footprint::lod_anisotropic))
//! is the `log2` of the short-axis half-length, and the sample count
//! ([`Footprint::sample_count`](prism_render_architecture::particle::anisotropic_footprint::Footprint::sample_count))
//! quantizes `anisotropy` through a fixed-point `div_ceil`.
//!
//! # The `log2` is a bit-trick, not a transcendental
//!
//! The golden never calls `log2` / `ln`. Instead
//! ([`log2_via_bits`](prism_render_architecture::particle::anisotropic_footprint))
//! reads the `IEEE754` `f32` bit pattern: for a positive normal
//! `x = (1 + f) * 2^e` the biased exponent field yields `e` and the mantissa
//! fraction `f` in `[0, 1)` is closed with the quadratic
//! `log2(1 + f) ≈ f + LOG2_MANTISSA_K * f * (1 - f)`, so powers of two are
//! exact. This twin reproduces that computation **bit-identically** in `WGSL`
//! using `bitcast<u32>` / `bitcast<f32>` and the same `LOG2_MANTISSA_K`
//! constant; the kernel calls no `WGSL` `log2` builtin anywhere.
//!
//! [`GpuAnisotropicFootprint`] is the on-device twin: one thread per
//! [`FootprintQuery`] (a `ddx` / `ddy` gradient pair plus `max_anisotropy`)
//! reproduces the same closed form branch for branch and writes one
//! [`FootprintResult`] holding the two half-lengths, the `anisotropy`, both
//! `LOD` levels and the integer sample count.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `floor`, `+ - * /`, unsigned bit ops, one `sqrt` and `bitcast` —
//! with no `sin`, `cos`, `exp`, `log`, `log2`, `pow`, `tan` and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds,
//! divides, one `sqrt` and integer bit work, so `CPU` and `GPU` evaluate the
//! same closed form in the same associativity. They are not bit-exact on the
//! `f32` fields: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on the continuous fields, while the integer sample count
//! is compared for exact equality.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓
//! `prism_render_architecture::particle::anisotropic_footprint`；closed-form
//! `2x2` symmetric-eigenvalue footprint plus a bit-pattern `log2` and `wgpu`
//! compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::anisotropic_footprint::{
    lod_trilinear, Footprint, Gradients,
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

/// The portable core-`WGSL` anisotropic-footprint kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`anisotropic_footprint`](prism_render_architecture::particle::anisotropic_footprint)
/// branch for branch, including the bit-pattern `log2`; see the module
/// documentation for the algorithm.
const ANISOTROPIC_FOOTPRINT_WGSL: &str = r#"
// Anisotropic-footprint twin: one thread per (ddx, ddy, max_anisotropy) query
// recovers the sampling ellipse, both LOD levels and the sample count. It
// mirrors the CPU golden particle::anisotropic_footprint branch for branch,
// uses only the portable core-WGSL subset (min/max/clamp/floor, + - * /,
// unsigned bit ops, one sqrt and bitcast) and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12. The log2 is a verbatim replica of
// the golden log2_via_bits bit trick, not a log2 builtin.
//
// Provenance: twinned from this repository's particle::anisotropic_footprint;
// no third-party engine source or derived code.

// Epsilon guarding the degenerate branches (a vanishing gradient and a
// collapsed minor axis) so the kernel never writes an f32 == / != and never
// emits a NaN. Matches the reference CMP_EPS.
const CMP_EPS: f32 = 1.0e-6;

// Quadratic mantissa-correction coefficient: log2(1 + f) ~= f + K*f*(1 - f).
// Byte-identical to the golden LOG2_MANTISSA_K so the bit trick agrees.
const LOG2_MANTISSA_K: f32 = 0.3465736;

// 2^23, the f32 mantissa field width, as a float divisor.
const MANTISSA_SCALE: f32 = 8388608.0;

// log2 result for non-positive / subnormal inputs: far below any physical mip
// level, so a degenerate footprint pins to the coarsest sampling after clamping
// instead of yielding -inf / NaN.
const LOG2_ZERO_FLOOR: f32 = -1000.0;

// Fixed-point scale turning the fractional anisotropy into an integer numerator
// for the div_ceil sample count.
const SAMPLE_FIXED_SCALE: f32 = 256.0;

// Integer form of SAMPLE_FIXED_SCALE, the div_ceil denominator.
const SAMPLE_FIXED_SCALE_U32: u32 = 256u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // ddx.x, ddx.y: the UV derivative along screen x, in texels.
    ddx0: f32,
    ddx1: f32,
    // ddy.x, ddy.y: the UV derivative along screen y, in texels.
    ddy0: f32,
    ddy1: f32,
    // Upper clamp on the anisotropy ratio.
    max_anisotropy: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Ellipse long- and short-axis half-lengths, in texels.
    major_len: f32,
    minor_len: f32,
    // major_len / minor_len, clamped to [1, max_anisotropy].
    anisotropy: f32,
    // Isotropic (trilinear) LOD and sharpened anisotropic LOD.
    lod_trilinear: f32,
    lod_anisotropic: f32,
    // Samples taken along the major axis, always at least one.
    sample_count: u32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// log2(x) built purely from the IEEE754 f32 bit pattern, a verbatim replica of
// the golden log2_via_bits. Non-positive or subnormal inputs return
// LOG2_ZERO_FLOOR; no log2 builtin is called.
fn log2_via_bits(x: f32) -> f32 {
    if (x <= 0.0) {
        return LOG2_ZERO_FLOOR;
    }
    let bits = bitcast<u32>(x);
    let exp_field = (bits >> 23u) & 0xffu;
    if (exp_field == 0u) {
        return LOG2_ZERO_FLOOR;
    }
    let mantissa = bits & 0x007fffffu;
    let frac = f32(mantissa) / MANTISSA_SCALE;
    let log_mant = frac + LOG2_MANTISSA_K * frac * (1.0 - frac);
    let exponent = f32(i32(exp_field) - 127);
    return exponent + log_mant;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Symmetric metric M = J^T * J with J columns ddx, ddy.
    let a = q.ddx0 * q.ddx0 + q.ddx1 * q.ddx1;
    let c = q.ddy0 * q.ddy0 + q.ddy1 * q.ddy1;
    let b = q.ddx0 * q.ddy0 + q.ddx1 * q.ddy1;

    let mean = (a + c) * 0.5;
    let diff = (a - c) * 0.5;
    let disc = sqrt(diff * diff + b * b);
    let lam_major = max(mean + disc, 0.0);
    let lam_minor = max(mean - disc, 0.0);

    let major_raw = sqrt(lam_major);
    let minor_raw = sqrt(lam_minor);

    var major_len: f32 = 0.0;
    var minor_len: f32 = 0.0;
    var anisotropy: f32 = 1.0;
    if (major_raw < CMP_EPS) {
        // Vanishing gradient collapses to a centered, isotropic footprint.
        major_len = 0.0;
        minor_len = 0.0;
        anisotropy = 1.0;
    } else {
        major_len = major_raw;
        minor_len = minor_raw;
        let max_a = max(q.max_anisotropy, 1.0);
        if (minor_raw < CMP_EPS) {
            anisotropy = max_a;
        } else {
            anisotropy = clamp(major_raw / minor_raw, 1.0, max_a);
        }
    }

    // Trilinear LOD uses the longest raw gradient length; anisotropic LOD uses
    // the (possibly collapsed) minor half-length.
    let dx = sqrt(a);
    let dy = sqrt(c);
    let lod_tri = log2_via_bits(max(dx, dy));
    let lod_aniso = log2_via_bits(minor_len);

    // Sample count: ceil(anisotropy) via a fixed-point div_ceil, floored at one.
    let scaled = floor(anisotropy * SAMPLE_FIXED_SCALE);
    let fixed = u32(scaled);
    let dc = (fixed + SAMPLE_FIXED_SCALE_U32 - 1u) / SAMPLE_FIXED_SCALE_U32;
    let samples = max(dc, 1u);

    var out: Result;
    out.major_len = major_len;
    out.minor_len = minor_len;
    out.anisotropy = anisotropy;
    out.lod_trilinear = lod_tri;
    out.lod_anisotropic = lod_aniso;
    out.sample_count = samples;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// One footprint query: the screen-space `UV` gradients `ddx` / `ddy` (in
/// texels) and the upper `max_anisotropy` clamp — the same inputs the reference
/// [`Footprint::from_gradients`](prism_render_architecture::particle::anisotropic_footprint::Footprint::from_gradients)
/// and
/// [`lod_trilinear`](prism_render_architecture::particle::anisotropic_footprint::lod_trilinear)
/// consume.
///
/// Provenance: 孪生自本仓
/// `prism_render_architecture::particle::anisotropic_footprint`；no third-party
/// engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FootprintQuery {
    /// `UV` derivative with respect to screen x, in texels.
    pub ddx: [f32; 2],
    /// `UV` derivative with respect to screen y, in texels.
    pub ddy: [f32; 2],
    /// Upper clamp on the anisotropy ratio.
    pub max_anisotropy: f32,
}

impl FootprintQuery {
    /// Builds a query from the two screen-space gradients and the
    /// `max_anisotropy` clamp.
    ///
    /// Provenance: 孪生自本仓
    /// `prism_render_architecture::particle::anisotropic_footprint`；no
    /// third-party engine source or derived code.
    #[must_use]
    pub const fn new(ddx: [f32; 2], ddy: [f32; 2], max_anisotropy: f32) -> FootprintQuery {
        FootprintQuery {
            ddx,
            ddy,
            max_anisotropy,
        }
    }
}

/// The resolved footprint for one query: the ellipse half-lengths, the
/// `anisotropy` ratio, both `LOD` levels and the integer sample count the
/// reference exposes.
///
/// Provenance: 孪生自本仓
/// `prism_render_architecture::particle::anisotropic_footprint`；no third-party
/// engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FootprintResult {
    /// Half-length of the ellipse's long axis, in texels (`sqrt(λ_major)`).
    pub major_len: f32,
    /// Half-length of the ellipse's short axis, in texels (`sqrt(λ_minor)`).
    pub minor_len: f32,
    /// `major_len / minor_len`, clamped to `[1, max_anisotropy]`.
    pub anisotropy: f32,
    /// Isotropic (trilinear) `LOD`: the `log2` of the longest gradient length.
    pub lod_trilinear: f32,
    /// Anisotropic `LOD`: the `log2` of the short-axis half-length.
    pub lod_anisotropic: f32,
    /// Samples taken along the major axis, always at least one.
    pub sample_count: u32,
}

/// Evaluates the `CPU` golden for one query, delegating field for field to the
/// reference
/// [`Footprint::from_gradients`](prism_render_architecture::particle::anisotropic_footprint::Footprint::from_gradients),
/// [`lod_trilinear`](prism_render_architecture::particle::anisotropic_footprint::lod_trilinear),
/// [`Footprint::lod_anisotropic`](prism_render_architecture::particle::anisotropic_footprint::Footprint::lod_anisotropic)
/// and
/// [`Footprint::sample_count`](prism_render_architecture::particle::anisotropic_footprint::Footprint::sample_count)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓
/// `prism_render_architecture::particle::anisotropic_footprint`；no third-party
/// engine source or derived code.
#[must_use]
pub fn golden(query: &FootprintQuery) -> FootprintResult {
    let gradients = Gradients {
        ddx: query.ddx,
        ddy: query.ddy,
    };
    let footprint = Footprint::from_gradients(&gradients, query.max_anisotropy);
    FootprintResult {
        major_len: footprint.major_len,
        minor_len: footprint.minor_len,
        anisotropy: footprint.anisotropy,
        lod_trilinear: lod_trilinear(&gradients),
        lod_anisotropic: footprint.lod_anisotropic(),
        sample_count: footprint.sample_count(),
    }
}

/// `repr(C)` `std430` layout of one packed query: two `vec4` slots holding
/// `(ddx.x, ddx.y, ddy.x, ddy.y)` and `(max_anisotropy, pad, pad, pad)` — `32`
/// bytes matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `ddx.x`.
    ddx0: f32,
    /// `ddx.y`.
    ddx1: f32,
    /// `ddy.x`.
    ddy0: f32,
    /// `ddy.y`.
    ddy1: f32,
    /// Upper clamp on the anisotropy ratio.
    max_anisotropy: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// Padding lane.
    pad2: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &FootprintQuery) -> GpuQuery {
        GpuQuery {
            ddx0: query.ddx[0],
            ddx1: query.ddx[1],
            ddy0: query.ddy[0],
            ddy1: query.ddy[1],
            max_anisotropy: query.max_anisotropy,
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: two `vec4` slots holding
/// `(major_len, minor_len, anisotropy, lod_trilinear)` and
/// `(lod_anisotropic, sample_count, pad, pad)` — `32` bytes matching the `WGSL`
/// `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Long-axis half-length.
    major_len: f32,
    /// Short-axis half-length.
    minor_len: f32,
    /// Clamped anisotropy ratio.
    anisotropy: f32,
    /// Isotropic (trilinear) `LOD`.
    lod_trilinear: f32,
    /// Anisotropic `LOD`.
    lod_anisotropic: f32,
    /// Samples along the major axis.
    sample_count: u32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable anisotropic-footprint compute pipeline.
///
/// Provenance: 孪生自本仓
/// `prism_render_architecture::particle::anisotropic_footprint`；no third-party
/// engine source or derived code.
pub struct GpuAnisotropicFootprint {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAnisotropicFootprint {
    /// Compiles the anisotropic-footprint kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓
    /// `prism_render_architecture::particle::anisotropic_footprint`；no
    /// third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAnisotropicFootprint {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_anisotropic_footprint"),
            source: ShaderSource::Wgsl(ANISOTROPIC_FOOTPRINT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_anisotropic_footprint_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_anisotropic_footprint_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_anisotropic_footprint_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAnisotropicFootprint {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`FootprintResult`] per
    /// input, in order.
    ///
    /// Each continuous field equals the reference
    /// [`anisotropic_footprint`](prism_render_architecture::particle::anisotropic_footprint)
    /// answers to within the tolerance documented on this module, and the
    /// integer `sample_count` matches exactly. An empty input returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    ///
    /// Provenance: 孪生自本仓
    /// `prism_render_architecture::particle::anisotropic_footprint`；no
    /// third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[FootprintQuery]) -> Vec<FootprintResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_anisotropic_footprint_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_anisotropic_footprint_output"),
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
            label: Some("prism_volumetric_anisotropic_footprint_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_anisotropic_footprint_bind_group"),
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
            label: Some("prism_volumetric_anisotropic_footprint_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_anisotropic_footprint_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_anisotropic_footprint_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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

/// Decodes one packed [`GpuResult`] into the public [`FootprintResult`].
fn decode_result(raw: &GpuResult) -> FootprintResult {
    FootprintResult {
        major_len: raw.major_len,
        minor_len: raw.minor_len,
        anisotropy: raw.anisotropy,
        lod_trilinear: raw.lod_trilinear,
        lod_anisotropic: raw.lod_anisotropic,
        sample_count: raw.sample_count,
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
