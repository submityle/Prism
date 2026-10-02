//! `wgpu` compute twin of the depth-of-field circle-of-confusion (`CoC`) golden
//! ([`depth_of_field`](prism_render_architecture::particle::depth_of_field),
//! particle design §16, §21).
//!
//! The `CPU` golden
//! [`depth_of_field`](prism_render_architecture::particle::depth_of_field) owns
//! the thin-lens maths a bokeh gather pass drives from: given a `view-space`
//! depth and the lens parameters it returns the signed `CoC` diameter
//! ([`DofParams::coc_signed`](prism_render_architecture::particle::depth_of_field::DofParams::coc_signed)),
//! its magnitude
//! ([`DofParams::coc_diameter`](prism_render_architecture::particle::depth_of_field::DofParams::coc_diameter)),
//! the clamped gather radius
//! ([`DofParams::coc_radius_clamped`](prism_render_architecture::particle::depth_of_field::DofParams::coc_radius_clamped)),
//! the normalized bokeh-kernel scale
//! ([`DofParams::bokeh_scale`](prism_render_architecture::particle::depth_of_field::DofParams::bokeh_scale)),
//! the `smoothstep` blur-fade weight
//! ([`DofParams::blur_fade`](prism_render_architecture::particle::depth_of_field::DofParams::blur_fade))
//! and the per-sample energy attenuation
//! ([`DofParams::energy_attenuation`](prism_render_architecture::particle::depth_of_field::DofParams::energy_attenuation)).
//!
//! [`GpuDepthOfField`] is the on-device twin: one thread per depth reproduces
//! the whole chain the reference
//! [`DofParams::evaluate`](prism_render_architecture::particle::depth_of_field::DofParams::evaluate)
//! runs, so a passing real-device parity test is direct evidence the ported
//! kernel evaluates the same thin-lens algebra and takes the same guard
//! branches (a non-positive depth, a focus distance not exceeding the focal
//! length, a near-zero denominator, a closed aperture, a degenerate
//! `smoothstep` span) the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-depth quantity the reference computes is reproduced: the signed
//! `CoC` diameter, its unsigned magnitude, the clamped gather radius, the
//! normalized bokeh scale read off that radius, the `smoothstep` blur-fade read
//! off the clamped radius, the rational-polynomial energy attenuation, and the
//! near/far classification (`depth > 0` and `depth < focus`). The reference's
//! private `smoothstep` is reproduced by hand as the cubic `3t^2 - 2t^3` over a
//! clamped parameter, with the same near-zero-span guard that falls back to a
//! hard step at the lower edge.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `select` and `+ - * /` — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, `tan`, no `smoothstep` builtin (the cubic is expanded by hand) and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Correctness model
//!
//! Each depth is a fixed, non-reorderable sequence of multiplies, adds and
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields and a bit
//! equality on the integer near/far code, tight enough to catch a genuinely
//! wrong port (a dropped guard, a swapped coefficient, a wrong clamp) yet loose
//! enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::depth_of_field`；
//! standard thin-lens circle-of-confusion maths plus `wgpu` compute dispatch; no
//! third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::depth_of_field::DofParams;
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

/// The portable core-`WGSL` depth-of-field kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`depth_of_field`](prism_render_architecture::particle::depth_of_field)
/// branch for branch; see the module documentation for the algorithm.
const DEPTH_OF_FIELD_WGSL: &str = r#"
// Depth-of-field twin: one thread per depth reproduces the signed CoC diameter,
// its magnitude, the clamped gather radius, the normalized bokeh scale, the
// smoothstep blur-fade and the per-sample energy attenuation, plus the near/far
// classification. It mirrors the CPU golden particle::depth_of_field branch for
// branch, uses only the portable core-WGSL subset (min/max/clamp/abs/select and
// + - * /, with smoothstep expanded by hand as the cubic 3t^2 - 2t^3) and takes
// no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::depth_of_field; no
// third-party engine source or derived code.

// Denominators (and divisors) with magnitude below this are treated as zero so
// evaluation falls back to a defined result instead of dividing by (near) zero.
// Matches the reference `MIN_DENOM`.
const MIN_DENOM: f32 = 1.0e-6;

struct Params {
    // Number of depths in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Thin-lens parameters packed as a vec4
    // (focus_distance, focal_length, aperture_diameter, max_coc_radius).
    lens: vec4<f32>,
    // The view-space depth sampled; three pad lanes follow.
    depth: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Signed CoC diameter, its magnitude, the clamped gather radius, the bokeh
    // scale, the smoothstep blur-fade, the energy attenuation, the near/far
    // code and one pad lane: eight scalars filling two vec4 slots.
    signed_coc: f32,
    coc_diameter: f32,
    radius: f32,
    bokeh_scale: f32,
    blur_fade: f32,
    energy_attenuation: f32,
    is_near: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Signed CoC diameter at `depth`, mirroring the reference `coc_signed`. Guards a
// non-positive depth, a focus distance not exceeding the focal length, and a
// near-zero denominator by returning 0.
fn coc_signed(focus: f32, f: f32, aperture: f32, depth: f32) -> f32 {
    if (depth <= 0.0 || focus <= f) {
        return 0.0;
    }
    let denom = depth * (focus - f);
    if (abs(denom) < MIN_DENOM) {
        return 0.0;
    }
    return aperture * f * (depth - focus) / denom;
}

// Clamped gather radius at `depth`, mirroring the reference
// `coc_radius_clamped`: half the unsigned CoC diameter, clamped to
// [0, max_coc_radius].
fn coc_radius_clamped(focus: f32, f: f32, aperture: f32, max_coc: f32, depth: f32) -> f32 {
    let radius = abs(coc_signed(focus, f, aperture, depth)) * 0.5;
    let max_radius = max(max_coc, 0.0);
    return min(radius, max_radius);
}

// Normalized bokeh-kernel scale in [0, 1], mirroring the reference
// `bokeh_scale`. A near-zero maximum radius returns 0 (no blur).
fn bokeh_scale(max_coc: f32, coc_radius: f32) -> f32 {
    if (abs(max_coc) < MIN_DENOM) {
        return 0.0;
    }
    return clamp(coc_radius / max_coc, 0.0, 1.0);
}

// Smoothstep over [edge0, edge1], mirroring the reference private `smoothstep`:
// 0 at or below edge0, 1 at or above edge1, and the cubic 3t^2 - 2t^3 between.
// A degenerate (near-zero-width) span falls back to a hard step at edge0. The
// cubic is written out by hand because the WGSL smoothstep builtin is banned.
fn smooth_fade(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (abs(span) < MIN_DENOM) {
        return select(1.0, 0.0, x < edge0);
    }
    let t = clamp((x - edge0) / span, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Smoothstep blur-fade weight in [0, 1] for the clamped radius at `depth`,
// mirroring the reference `blur_fade`.
fn blur_fade(focus: f32, f: f32, aperture: f32, max_coc: f32, depth: f32) -> f32 {
    let max_radius = max(max_coc, 0.0);
    let radius = coc_radius_clamped(focus, f, aperture, max_coc, depth);
    return smooth_fade(0.0, max_radius, radius);
}

// Per-sample energy attenuation in (0, 1] as 1 / (1 + r^2) for the normalized
// radius r = coc_radius / max_coc_radius, mirroring the reference
// `energy_attenuation`. A near-zero maximum radius returns 1.
fn energy_attenuation(max_coc: f32, coc_radius: f32) -> f32 {
    if (abs(max_coc) < MIN_DENOM) {
        return 1.0;
    }
    let r = coc_radius / max_coc;
    return 1.0 / (1.0 + r * r);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let focus = q.lens.x;
    let f = q.lens.y;
    let aperture = q.lens.z;
    let max_coc = q.lens.w;
    let depth = q.depth;

    let signed = coc_signed(focus, f, aperture, depth);
    let radius = coc_radius_clamped(focus, f, aperture, max_coc, depth);

    var out: Result;
    out.signed_coc = signed;
    out.coc_diameter = abs(signed);
    out.radius = radius;
    out.bokeh_scale = bokeh_scale(max_coc, radius);
    out.blur_fade = blur_fade(focus, f, aperture, max_coc, depth);
    out.energy_attenuation = energy_attenuation(max_coc, radius);
    out.is_near = select(0u, 1u, depth > 0.0 && depth < focus);
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// One depth-of-field query: the thin-lens `params` evaluated at a single
/// `view-space` `depth` — the same inputs the reference
/// [`DofParams::evaluate`](prism_render_architecture::particle::depth_of_field::DofParams::evaluate)
/// consumes. Each query carries its own lens so a batch can mix apertures.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::depth_of_field`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthOfFieldQuery {
    /// The thin-lens parameters the depth is evaluated against.
    pub params: DofParams,
    /// The `view-space` depth sampled.
    pub depth: f32,
}

impl DepthOfFieldQuery {
    /// Builds a query from the thin-lens parameters and the sampled depth.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::depth_of_field`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(params: DofParams, depth: f32) -> DepthOfFieldQuery {
        DepthOfFieldQuery { params, depth }
    }
}

/// The resolved answer for one depth, mirroring the whole evaluation chain the
/// reference exposes (a superset of the golden
/// [`DofSample`](prism_render_architecture::particle::depth_of_field::DofSample)).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::depth_of_field`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthOfFieldSample {
    /// Signed `CoC` diameter: negative for near defocus (`depth < focus`),
    /// positive for far defocus (`depth > focus`), zero on the focus plane.
    pub signed_coc: f32,
    /// Unsigned `CoC` diameter (the magnitude of `signed_coc`).
    pub coc_diameter: f32,
    /// Gather radius in `CoC` units, non-negative and clamped to
    /// [`DofParams::max_coc_radius`](prism_render_architecture::particle::depth_of_field::DofParams).
    pub radius: f32,
    /// Normalized bokeh-kernel scale in `[0, 1]` read off `radius`.
    pub bokeh_scale: f32,
    /// `smoothstep` blur-fade weight in `[0, 1]` read off the clamped radius.
    pub blur_fade: f32,
    /// Per-sample energy attenuation in `(0, 1]` read off `radius`.
    pub energy_attenuation: f32,
    /// Whether the depth lies in front of the focus plane (near defocus).
    pub is_near: bool,
}

/// Evaluates the `CPU` golden for one query.
///
/// This routes through the reference
/// [`DofParams`](prism_render_architecture::particle::depth_of_field::DofParams)
/// methods (`coc_signed`, `coc_diameter`, `coc_radius_clamped`, `bokeh_scale`,
/// `blur_fade`, `energy_attenuation`) so the host side and the device twin are
/// checked against the same source of truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::depth_of_field`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &DepthOfFieldQuery) -> DepthOfFieldSample {
    let params = query.params;
    let depth = query.depth;
    let radius = params.coc_radius_clamped(depth);
    DepthOfFieldSample {
        signed_coc: params.coc_signed(depth),
        coc_diameter: params.coc_diameter(depth),
        radius,
        bokeh_scale: params.bokeh_scale(radius),
        blur_fade: params.blur_fade(depth),
        energy_attenuation: params.energy_attenuation(radius),
        is_near: depth > 0.0 && depth < params.focus_distance,
    }
}

/// `repr(C)` `std430` layout of one packed query: two `vec4` slots holding the
/// lens `(focus_distance, focal_length, aperture_diameter, max_coc_radius)` and
/// `(depth, pad, pad, pad)` — `32` bytes, the lens `vec4` on its `16`-byte
/// slot exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Thin-lens parameters packed as a `vec4`.
    lens: [f32; 4],
    /// Sampled depth, in the first lane of the second slot.
    depth: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// Padding lane.
    pad2: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &DepthOfFieldQuery) -> GpuQuery {
        GpuQuery {
            lens: query.params.to_std430(),
            depth: query.depth,
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: two `vec4` slots holding
/// `(signed_coc, coc_diameter, radius, bokeh_scale)` and
/// `(blur_fade, energy_attenuation, is_near, pad)` — `32` bytes matching the
/// `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed `CoC` diameter.
    signed_coc: f32,
    /// Unsigned `CoC` diameter.
    coc_diameter: f32,
    /// Clamped gather radius.
    radius: f32,
    /// Normalized bokeh scale.
    bokeh_scale: f32,
    /// `smoothstep` blur-fade weight.
    blur_fade: f32,
    /// Per-sample energy attenuation.
    energy_attenuation: f32,
    /// Near/far code: `1` for near defocus, `0` otherwise.
    is_near: u32,
    /// Padding lane.
    pad0: u32,
}

/// Uniform parameters for one dispatch: the depth count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of depths in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable depth-of-field compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::depth_of_field`；
/// no third-party engine source or derived code.
pub struct GpuDepthOfField {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDepthOfField {
    /// Compiles the depth-of-field kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::depth_of_field`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDepthOfField {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_depth_of_field"),
            source: ShaderSource::Wgsl(DEPTH_OF_FIELD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_depth_of_field_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_depth_of_field_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_depth_of_field_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDepthOfField {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`DepthOfFieldSample`] per
    /// input, in order.
    ///
    /// Each result equals the reference
    /// [`DofParams::evaluate`](prism_render_architecture::particle::depth_of_field::DofParams::evaluate)
    /// chain to within the tolerance documented on this module. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::depth_of_field`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[DepthOfFieldQuery]) -> Vec<DepthOfFieldSample> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_depth_of_field_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_depth_of_field_output"),
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
            label: Some("prism_volumetric_depth_of_field_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_depth_of_field_bind_group"),
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
            label: Some("prism_volumetric_depth_of_field_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_depth_of_field_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_depth_of_field_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per depth, flattened to a 1-D dispatch.
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

/// Decodes one packed [`GpuResult`] into the public [`DepthOfFieldSample`].
fn decode_result(raw: &GpuResult) -> DepthOfFieldSample {
    DepthOfFieldSample {
        signed_coc: raw.signed_coc,
        coc_diameter: raw.coc_diameter,
        radius: raw.radius,
        bokeh_scale: raw.bokeh_scale,
        blur_fade: raw.blur_fade,
        energy_attenuation: raw.energy_attenuation,
        is_near: raw.is_near == 1,
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
