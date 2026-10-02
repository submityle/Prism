//! `wgpu` compute twin of the `Fresnel` / edge-light rim golden
//! ([`fresnel_rim`](prism_render_architecture::particle::fresnel_rim),
//! particle design §16, §17).
//!
//! The `CPU` golden
//! [`fresnel_rim`](prism_render_architecture::particle::fresnel_rim) owns the
//! edge-light rim term: a surface is brightened where it turns away from the
//! viewer, tracing a luminous outline around silhouettes. The model composes
//! three transcendental-free pieces, all driven by the view-space geometry term
//! `n_dot_v` (the cosine between the surface normal and the view direction):
//! the generalized `Schlick` rim factor
//! ([`rim_factor`](prism_render_architecture::particle::fresnel_rim::rim_factor))
//! with an artist-chosen integer exponent evaluated by an integer multiply loop
//! ([`power_u32`](prism_render_architecture::particle::fresnel_rim::power_u32))
//! rather than a `powf`, and the `smoothstep` intensity band
//! ([`rim_intensity`](prism_render_architecture::particle::fresnel_rim::rim_intensity))
//! that brightens toward grazing angles.
//! [`RimParams::evaluate`](prism_render_architecture::particle::fresnel_rim::RimParams::evaluate)
//! wires them into a `RimSample` (a scalar factor plus a linear-`RGB`
//! contribution).
//!
//! [`GpuFresnelRim`] is the on-device twin: one thread per query reproduces
//! `RimParams::evaluate` step for step, so a passing real-device parity test is
//! direct evidence the ported kernel evaluates the same rim math and classifies
//! the same degenerate cases (a near-zero-length direction normalizing to the
//! zero vector, a collapsed `smoothstep` interval) the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query reproduces the full `RimSample` the reference `evaluate` returns:
//! the unitless `factor` (the generalized `Schlick` term times the `smoothstep`
//! band) and the `rgb` contribution `rim_color * factor * intensity`. The
//! standalone `Schlick`
//! ([`fresnel_schlick`](prism_render_architecture::particle::fresnel_rim::fresnel_schlick))
//! value at the same `n_dot_v` is reported alongside, so the fixed fifth-power
//! `Schlick` path and the integer-exponent `rim_factor` path are both pinned
//! against the reference. The derived `n_dot_v` is reported too so the host can
//! cross-check the shared cosine.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `abs`, `+ - * /`, the `dot` builtin, one `sqrt` for the genuine
//! Euclidean normalization and `bitcast` for the packed parameter image — with
//! no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry and no
//! `smoothstep` builtin (the `Hermite` band is expanded by hand as
//! `t * t * (3 - 2 * t)`), so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. The only loop is the bounded `power_u32` multiply, so the kernel
//! provably terminates.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the continuous `f32` fields,
//! tight enough to catch a genuinely wrong port (a dropped branch, a swapped
//! coefficient, a wrong clamp) yet loose enough to admit legal fused
//! multiply-add contraction. The integer rim exponent is carried as a raw `u32`
//! so the twinned `power_u32` loop runs the exact same number of iterations.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fresnel_rim`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `Fresnel`-rim kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `RimParams::evaluate` step for step; see the module
/// documentation for the algorithm.
const FRESNEL_RIM_WGSL: &str = r#"
// Fresnel / edge-light rim twin: one thread per query reproduces the CPU golden
// `RimParams::evaluate` step for step. The clamped n_dot_v cosine drives the
// generalized Schlick rim_factor and the smoothstep rim_intensity band; their
// product is the rim factor and rim_color * factor * intensity is the RGB
// contribution. The standalone Schlick fresnel_schlick is reproduced too and
// reported alongside, with every fifth / integer power evaluated by the
// power_u32 multiply loop rather than a transcendental pow.

// Minimum squared length below which a direction is degenerate and normalizes
// to the zero vector instead of dividing by zero. Matches the reference.
const MIN_LEN_SQ: f32 = 1.0e-12;

// Minimum smoothstep interval (and generic denominator guard) below which a
// soft edge collapses to a hard step. Matches the reference MIN_EDGE.
const MIN_EDGE: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Surface normal (unnormalized); a pad lane follows.
    normal: vec3<f32>,
    pad0: f32,
    // View direction (unnormalized); a pad lane follows.
    view_dir: vec3<f32>,
    pad1: f32,
    // RimParams to_std430 image: [rim_r, rim_g, rim_b, intensity, f0,
    // power_bits, inner, outer]. Floats are stored as their bit patterns and
    // power as a raw u32, exactly as the host packs them.
    params: array<u32, 8>,
}

struct Result {
    // RimSample factor and RGB contribution: four scalars in one vec4 slot.
    factor: f32,
    rgb_r: f32,
    rgb_g: f32,
    rgb_b: f32,
    // Standalone Schlick value at n_dot_v, the derived n_dot_v, then two pads.
    fresnel: f32,
    n_dot_v: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Clamps a scalar into the 0..=1 range, mirroring the reference clamp01.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Normalizes a 3-component vector, returning the zero vector for a degenerate
// (near-zero-length) input instead of producing a NaN. Mirrors normalize3.
fn normalize3(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq < MIN_LEN_SQ) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let inv_len = 1.0 / sqrt(len_sq);
    return v * inv_len;
}

// Hermite smoothstep from edge0 to edge1 evaluated at x, mirroring the
// reference smoothstep: a degenerate interval collapses to a hard step at edge1
// rather than dividing by zero. Expanded by hand as t * t * (3 - 2 * t) since
// the WGSL smoothstep builtin is outside this crate's portable subset.
fn smoothstep_ref(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (span < MIN_EDGE) {
        if (x < edge1) {
            return 0.0;
        }
        return 1.0;
    }
    let t = clamp01((x - edge0) / span);
    return t * t * (3.0 - 2.0 * t);
}

// Raises base to the integer power exp by repeated multiplication, matching the
// reference power_u32 (the empty product is 1.0). Replaces pow so the rim
// exponent stays transcendental-free and bit-reproducible.
fn power_u32(base: f32, exp: u32) -> f32 {
    var acc: f32 = 1.0;
    var remaining: u32 = exp;
    loop {
        if (remaining == 0u) {
            break;
        }
        acc = acc * base;
        remaining = remaining - 1u;
    }
    return acc;
}

// The Schlick Fresnel approximation f0 + (1 - f0) * (1 - cos)^5, mirroring the
// reference fresnel_schlick. The local cosine is named cos_t because cos is a
// reserved WGSL builtin.
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let cos_t = clamp01(cos_theta);
    let one_minus_cos = 1.0 - cos_t;
    return f0 + (1.0 - f0) * power_u32(one_minus_cos, 5u);
}

// Generalized Schlick rim f0 + (1 - f0) * (1 - n_dot_v)^power, mirroring the
// reference rim_factor.
fn rim_factor(n_dot_v: f32, power: u32, f0: f32) -> f32 {
    let n = clamp01(n_dot_v);
    return f0 + (1.0 - f0) * power_u32(1.0 - n, power);
}

// Smoothstep intensity band 1 - smoothstep(inner, outer, n_dot_v), mirroring
// the reference rim_intensity.
fn rim_intensity(n_dot_v: f32, inner: f32, outer: f32) -> f32 {
    let n = clamp01(n_dot_v);
    return 1.0 - smoothstep_ref(inner, outer, n);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Unpack the RimParams to_std430 image.
    let rim_color = vec3<f32>(
        bitcast<f32>(q.params[0]),
        bitcast<f32>(q.params[1]),
        bitcast<f32>(q.params[2])
    );
    let intensity = bitcast<f32>(q.params[3]);
    let f0 = bitcast<f32>(q.params[4]);
    let power = q.params[5];
    let inner = bitcast<f32>(q.params[6]);
    let outer = bitcast<f32>(q.params[7]);

    // RimParams::evaluate: normalize both inputs, then NdotV = clamp(dot, 0, 1).
    let n = normalize3(q.normal);
    let v = normalize3(q.view_dir);
    let n_dot_v = clamp01(dot(n, v));

    let factor = rim_factor(n_dot_v, power, f0) * rim_intensity(n_dot_v, inner, outer);
    let scaled = factor * intensity;
    let rgb = rim_color * scaled;

    // Standalone Schlick value at the same cosine, reproduced for parity.
    let fresnel = fresnel_schlick(n_dot_v, f0);

    var out: Result;
    out.factor = factor;
    out.rgb_r = rgb.x;
    out.rgb_g = rgb.y;
    out.rgb_b = rgb.z;
    out.fresnel = fresnel;
    out.n_dot_v = n_dot_v;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// One rim query: a surface `normal`, a `view_dir` and the `RimParams` fields.
///
/// The `normal` and `view_dir` are normalized on-device exactly as the
/// reference `RimParams::evaluate` normalizes them (a degenerate direction
/// collapses to the zero vector, i.e. grazing).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FresnelRimQuery {
    /// Surface normal (need not be unit length).
    pub normal: [f32; 3],
    /// View direction (need not be unit length).
    pub view_dir: [f32; 3],
    /// Reflectance at normal incidence for the `Schlick` rim (`0..=1`).
    pub f0: f32,
    /// Integer rim exponent; larger values tighten the rim.
    pub power: u32,
    /// Linear-`RGB` colour of the rim light.
    pub rim_color: [f32; 3],
    /// Scalar multiplier applied to the emitted rim colour (`HDR` gain).
    pub intensity: f32,
    /// Inner `n_dot_v` edge of the `smoothstep` intensity band.
    pub inner: f32,
    /// Outer `n_dot_v` edge of the `smoothstep` intensity band.
    pub outer: f32,
}

impl FresnelRimQuery {
    /// Builds a query from a `normal`, a `view_dir` and the rim parameters.
    #[must_use]
    pub const fn new(
        normal: [f32; 3],
        view_dir: [f32; 3],
        f0: f32,
        power: u32,
        rim_color: [f32; 3],
        intensity: f32,
        inner: f32,
        outer: f32,
    ) -> FresnelRimQuery {
        FresnelRimQuery {
            normal,
            view_dir,
            f0,
            power,
            rim_color,
            intensity,
            inner,
            outer,
        }
    }
}

/// The resolved rim answer for one query.
///
/// `factor` and `rgb` mirror the reference `RimSample`; `fresnel` is the
/// standalone `Schlick` value at the derived `n_dot_v`, reported so the fixed
/// fifth-power path is pinned too, and `n_dot_v` is the shared clamped cosine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FresnelRimSample {
    /// The unitless rim strength: `rim_factor` times `rim_intensity`.
    pub factor: f32,
    /// The linear-`RGB` rim contribution `rim_color * factor * intensity`.
    pub rgb: [f32; 3],
    /// The standalone `Schlick` `fresnel_schlick` value at `n_dot_v`.
    pub fresnel: f32,
    /// The clamped cosine `clamp(dot(normalize(normal), normalize(view_dir)))`.
    pub n_dot_v: f32,
}

/// `repr(C)` `std430` layout of one packed query: a `(normal.xyz, pad)` slot, a
/// `(view_dir.xyz, pad)` slot and the eight-word `RimParams` `to_std430` image —
/// `64` bytes, each `vec3` on its `16`-byte-aligned slot exactly as the `WGSL`
/// `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Surface normal.
    normal: [f32; 3],
    /// Padding lane after the normal.
    pad0: f32,
    /// View direction.
    view_dir: [f32; 3],
    /// Padding lane after the view direction.
    pad1: f32,
    /// The `RimParams` `to_std430` image: `[rim_r, rim_g, rim_b, intensity, f0,
    /// power_bits, inner, outer]`.
    params: [u32; 8],
}

impl GpuQuery {
    /// Packs one query into its `std430` image, copying the reference
    /// `RimParams::to_std430` word order so the mixed `f32` / `u32` block
    /// round-trips exactly without a lossy cast.
    fn new(query: &FresnelRimQuery) -> GpuQuery {
        GpuQuery {
            normal: query.normal,
            pad0: 0.0,
            view_dir: query.view_dir,
            pad1: 0.0,
            params: [
                query.rim_color[0].to_bits(),
                query.rim_color[1].to_bits(),
                query.rim_color[2].to_bits(),
                query.intensity.to_bits(),
                query.f0.to_bits(),
                query.power,
                query.inner.to_bits(),
                query.outer.to_bits(),
            ],
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-scalar `(factor, rgb)` slot
/// then a `(fresnel, n_dot_v, pad, pad)` slot — `32` bytes matching the `WGSL`
/// `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Rim factor.
    factor: f32,
    /// Rim `RGB` contribution.
    rgb: [f32; 3],
    /// Standalone `Schlick` value at `n_dot_v`.
    fresnel: f32,
    /// Derived clamped cosine.
    n_dot_v: f32,
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

/// A compiled, reusable `Fresnel`-rim compute pipeline.
pub struct GpuFresnelRim {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFresnelRim {
    /// Compiles the `Fresnel`-rim kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFresnelRim {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fresnel_rim"),
            source: ShaderSource::Wgsl(FRESNEL_RIM_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fresnel_rim_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fresnel_rim_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fresnel_rim_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFresnelRim {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query on-device and returns one [`FresnelRimSample`] per
    /// input, in order.
    ///
    /// Each result equals the reference `RimParams::evaluate` (plus the
    /// standalone `fresnel_schlick`) to within the tolerance documented on this
    /// module. An empty input returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[FresnelRimQuery]) -> Vec<FresnelRimSample> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fresnel_rim_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fresnel_rim_output"),
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
            label: Some("prism_volumetric_fresnel_rim_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fresnel_rim_bind_group"),
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
            label: Some("prism_volumetric_fresnel_rim_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fresnel_rim_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fresnel_rim_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`FresnelRimSample`].
fn decode_result(raw: &GpuResult) -> FresnelRimSample {
    FresnelRimSample {
        factor: raw.factor,
        rgb: raw.rgb,
        fresnel: raw.fresnel,
        n_dot_v: raw.n_dot_v,
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
