//! `wgpu` compute twin of the screen-space `vignette` mask golden
//! ([`vignette_mask`](prism_render_architecture::particle::vignette_mask),
//! particle design §16, §21).
//!
//! The `CPU` golden
//! [`vignette_mask`](prism_render_architecture::particle::vignette_mask) owns
//! the per-pixel darkening *mask* a post pass multiplies into the composited
//! color: an aspect-corrected, `roundness`-blended distance from a `uv` to the
//! optical center is fed through a `smoothstep` band and a rational-polynomial
//! softening curve, then folded into `1 - intensity * shaped` and clamped to
//! `[1 - intensity, 1]`. [`VignetteParams::evaluate`] returns that scalar and
//! [`VignetteParams::apply`] multiplies it into an `RGB` color.
//!
//! [`GpuVignetteMask`] is the on-device twin: one thread per pixel
//! [`VignetteMaskQuery`] reproduces the same closed form branch for branch — the
//! integer-fold `powi`, the manually expanded `smoothstep` `3t^2 - 2t^3`, the
//! rational `soften` and the final `clamp` — so a passing real-device parity
//! test is direct evidence the ported kernel evaluates the same mask the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query returns the mask weight
//! ([`VignetteParams::evaluate`](prism_render_architecture::particle::vignette_mask::VignetteParams::evaluate))
//! and the masked color
//! ([`VignetteParams::apply`](prism_render_architecture::particle::vignette_mask::VignetteParams::apply)).
//! The reference's two degenerate branches are mirrored: a near-zero-width
//! `smoothstep` interval collapses to a hard step, and a near-zero rational
//! denominator falls back to the raw band position, so neither device divides by
//! (near) zero or emits a `NaN`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `abs`, `+ - * /`, one `sqrt` for the euclidean distance and an
//! integer `for`-loop `powi` in place of the forbidden `pow` builtin — with no
//! `sin`, `cos`, `exp`, `log`, `tan`, `smoothstep` or optional device feature,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields, tight enough
//! to catch a genuinely wrong port (a dropped branch, a swapped coefficient, a
//! wrong clamp) yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::vignette_mask`；
//! standard screen-space `vignette` falloff plus `wgpu` compute dispatch; no
//! third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::vignette_mask::VignetteParams;
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

/// The portable core-`WGSL` `vignette`-mask kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`vignette_mask`](prism_render_architecture::particle::vignette_mask) branch
/// for branch; see the module documentation for the algorithm.
const VIGNETTE_MASK_WGSL: &str = r#"
// Vignette-mask twin: one thread per pixel reproduces the CPU golden
// particle::vignette_mask evaluate + apply branch for branch. It uses only the
// portable core-WGSL subset (clamp/min/max/abs, + - * / and one sqrt for the
// euclidean distance) with an integer-fold powi in place of the forbidden pow
// builtin, takes no optional feature, and so runs unmodified on Metal, Vulkan
// and DX12.
//
// Provenance: twinned from this repository's particle::vignette_mask; no
// third-party engine source or derived code.

// Spans and denominators with magnitude below this are treated as zero so a
// degenerate interval falls back to a defined result instead of dividing by
// (near) zero. Matches the reference `MIN_SPAN`.
const MIN_SPAN: f32 = 1.0e-6;

struct Params {
    // Number of pixel queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Optical center in screen uv.
    center_x: f32,
    center_y: f32,
    // Radius inside which the mask is 1.0, and the radius beyond which it
    // saturates to 1 - intensity.
    inner_radius: f32,
    outer_radius: f32,
    // Darkening strength, shape blend and horizontal aspect scale.
    intensity: f32,
    roundness: f32,
    aspect: f32,
    // The pixel uv being evaluated.
    uv_x: f32,
    uv_y: f32,
    // The linear RGB color the mask is applied to.
    color_r: f32,
    color_g: f32,
    color_b: f32,
}

struct Result {
    // The mask weight and the masked RGB color: four scalars filling one vec4
    // slot.
    mask: f32,
    applied_r: f32,
    applied_g: f32,
    applied_b: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Integer power via a multiplicative fold, mirroring the reference private
// powi: an exponent of zero yields 1.0 and the forbidden pow builtin is never
// used.
fn powi(base: f32, exp: u32) -> f32 {
    var acc: f32 = 1.0;
    for (var i: u32 = 0u; i < exp; i = i + 1u) {
        acc = acc * base;
    }
    return acc;
}

// smoothstep across [edge0, edge1], manually expanded to the cubic
// 3t^2 - 2t^3 so no smoothstep builtin is relied on. A near-zero-width interval
// falls back to a hard step at edge0, mirroring the reference.
fn smooth_band(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (abs(span) < MIN_SPAN) {
        if (x < edge0) {
            return 0.0;
        }
        return 1.0;
    }
    let t = clamp((x - edge0) / span, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Rational-polynomial S-curve t^2 / (t^2 + (1 - t)^2). The denominator is at
// least 0.5 on [0, 1] so the division is safe; a near-zero denominator falls
// back to the raw band position, mirroring the reference.
fn rational_soften(t: f32) -> f32 {
    let a = powi(t, 2u);
    let b = powi(1.0 - t, 2u);
    let denom = a + b;
    if (denom < MIN_SPAN) {
        return t;
    }
    return a / denom;
}

// Aspect-corrected, roundness-blended distance from the query uv to the optical
// center: roundness 1 is the euclidean (round) distance, 0 the chebyshev (box)
// distance, values between interpolate.
fn shape_distance(q: Query) -> f32 {
    let dx = (q.uv_x - q.center_x) * q.aspect;
    let dy = q.uv_y - q.center_y;
    let euclid = sqrt(dx * dx + dy * dy);
    let cheby = max(abs(dx), abs(dy));
    let r = clamp(q.roundness, 0.0, 1.0);
    return cheby + (euclid - cheby) * r;
}

// The vignette mask weight at the query uv, in [1 - intensity, 1].
fn evaluate(q: Query) -> f32 {
    let dist = shape_distance(q);
    let band = smooth_band(q.inner_radius, q.outer_radius, dist);
    let shaped = rational_soften(band);
    let intensity = clamp(q.intensity, 0.0, 1.0);
    return clamp(1.0 - intensity * shaped, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let mask = evaluate(q);

    var out: Result;
    out.mask = mask;
    out.applied_r = q.color_r * mask;
    out.applied_g = q.color_g * mask;
    out.applied_b = q.color_b * mask;
    results[idx] = out;
}
"#;

/// One `vignette`-mask pixel query: the mask [`VignetteParams`] evaluated at a
/// screen `uv`, together with the linear `RGB` color the mask is applied to —
/// the same inputs the reference
/// [`VignetteParams::evaluate`](prism_render_architecture::particle::vignette_mask::VignetteParams::evaluate)
/// and
/// [`VignetteParams::apply`](prism_render_architecture::particle::vignette_mask::VignetteParams::apply)
/// consume.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::vignette_mask`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VignetteMaskQuery {
    /// The mask parameters (center, radii, intensity, roundness, aspect).
    pub params: VignetteParams,
    /// The screen `uv` in `[0, 1]^2` being evaluated.
    pub uv: [f32; 2],
    /// The linear `RGB` color the mask is multiplied into.
    pub color: [f32; 3],
}

impl VignetteMaskQuery {
    /// Builds a query from the mask parameters, the screen `uv` and the color
    /// the mask is applied to.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::vignette_mask`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(params: VignetteParams, uv: [f32; 2], color: [f32; 3]) -> VignetteMaskQuery {
        VignetteMaskQuery { params, uv, color }
    }
}

/// The resolved answer for one query: the mask weight and the masked color.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::vignette_mask`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VignetteMaskResult {
    /// The mask weight in `[1 - intensity, 1]`, matching
    /// [`VignetteParams::evaluate`](prism_render_architecture::particle::vignette_mask::VignetteParams::evaluate).
    pub mask: f32,
    /// The color after the mask is applied per channel, matching
    /// [`VignetteParams::apply`](prism_render_architecture::particle::vignette_mask::VignetteParams::apply).
    pub applied: [f32; 3],
}

/// Evaluates the `CPU` golden for one query, delegating to the reference
/// [`VignetteParams::evaluate`](prism_render_architecture::particle::vignette_mask::VignetteParams::evaluate)
/// and
/// [`VignetteParams::apply`](prism_render_architecture::particle::vignette_mask::VignetteParams::apply)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::vignette_mask`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &VignetteMaskQuery) -> VignetteMaskResult {
    VignetteMaskResult {
        mask: query.params.evaluate(query.uv),
        applied: query.params.apply(query.color, query.uv),
    }
}

/// `repr(C)` `std430` layout of one packed query: twelve `f32`s spanning three
/// `vec4` slots — the seven `VignetteParams` scalars in the reference
/// [`VignetteParams::to_std430`](prism_render_architecture::particle::vignette_mask::VignetteParams::to_std430)
/// order, then the two `uv` lanes and the three color lanes — `48` bytes,
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Optical center x.
    center_x: f32,
    /// Optical center y.
    center_y: f32,
    /// Inner radius (fully lit inside).
    inner_radius: f32,
    /// Outer radius (saturated beyond).
    outer_radius: f32,
    /// Darkening strength.
    intensity: f32,
    /// Shape blend between box and round.
    roundness: f32,
    /// Horizontal aspect scale.
    aspect: f32,
    /// Pixel `uv` x.
    uv_x: f32,
    /// Pixel `uv` y.
    uv_y: f32,
    /// Color red channel.
    color_r: f32,
    /// Color green channel.
    color_g: f32,
    /// Color blue channel.
    color_b: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &VignetteMaskQuery) -> GpuQuery {
        let p = query.params;
        GpuQuery {
            center_x: p.center[0],
            center_y: p.center[1],
            inner_radius: p.inner_radius,
            outer_radius: p.outer_radius,
            intensity: p.intensity,
            roundness: p.roundness,
            aspect: p.aspect,
            uv_x: query.uv[0],
            uv_y: query.uv[1],
            color_r: query.color[0],
            color_g: query.color[1],
            color_b: query.color[2],
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-scalar slot
/// `(mask, applied_r, applied_g, applied_b)` — `16` bytes matching the `WGSL`
/// `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Mask weight.
    mask: f32,
    /// Masked red channel.
    applied_r: f32,
    /// Masked green channel.
    applied_g: f32,
    /// Masked blue channel.
    applied_b: f32,
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

/// A compiled, reusable `vignette`-mask compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::vignette_mask`；
/// no third-party engine source or derived code.
pub struct GpuVignetteMask {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVignetteMask {
    /// Compiles the `vignette`-mask kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::vignette_mask`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVignetteMask {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_vignette_mask"),
            source: ShaderSource::Wgsl(VIGNETTE_MASK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_vignette_mask_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_vignette_mask_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_vignette_mask_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVignetteMask {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`VignetteMaskResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference
    /// [`VignetteParams::evaluate`](prism_render_architecture::particle::vignette_mask::VignetteParams::evaluate)
    /// and
    /// [`VignetteParams::apply`](prism_render_architecture::particle::vignette_mask::VignetteParams::apply)
    /// answers to within the tolerance documented on this module. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::vignette_mask`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[VignetteMaskQuery]) -> Vec<VignetteMaskResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vignette_mask_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_vignette_mask_output"),
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
            label: Some("prism_volumetric_vignette_mask_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_vignette_mask_bind_group"),
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
            label: Some("prism_volumetric_vignette_mask_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_vignette_mask_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_vignette_mask_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pixel query, flattened to a 1-D dispatch.
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

/// Decodes one packed [`GpuResult`] into the public [`VignetteMaskResult`].
fn decode_result(raw: &GpuResult) -> VignetteMaskResult {
    VignetteMaskResult {
        mask: raw.mask,
        applied: [raw.applied_r, raw.applied_g, raw.applied_b],
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
