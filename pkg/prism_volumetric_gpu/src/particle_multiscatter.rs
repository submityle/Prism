//! `wgpu` compute twin of the particle-engine octave-summed multiple-scattering
//! `RGB` response
//! ([`MultiScatterParams::response_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response_cos),
//! design `docs/prism_particle_engine_design_zh.md` §20 "单次 + 多次散射",
//! feeding the §17 `PBR` volumetric closure's multiple-scattering term), so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same per-channel response as the `CPU` golden, not merely that
//! its shader compiles.
//!
//! # Scope and distinction from [`crate::octave`]
//!
//! This twin is **not** the sibling [`GpuOctaveScatter`](crate::GpuOctaveScatter).
//! That one ports the cloud-path
//! [`octave_scatter`](prism_render_architecture::volumetric::scatter::octave_scatter):
//! a per-octave geometric decay of the scalar `(sigma_s, sigma_t, g)` triple.
//! This one ports the particle-path
//! [`MultiScatterParams::response_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response_cos):
//! an octave *sum* that folds a fixed number of scattering orders into one
//! per-channel (`RGB`) phase response, where order `i` carries a running-product
//! throughput `(albedo · octave_decay)^i`, a phase lobe broadened toward
//! isotropic by `anisotropy_falloff^i`, and an added isotropic ambient floor so
//! a high-albedo interior never goes dead black. Each order reuses the
//! `sqrt`-only double-lobe Henyey-Greenstein phase
//! [`double_lobe_phase`](prism_render_architecture::particle::volumetrics::double_lobe_phase).
//! [`GpuParticleMultiScatter`] is the on-device twin that runs one thread per
//! scattering cosine and returns the same `RGB` response the reference does.
//!
//! # Shared schedule
//!
//! Every query in one [`eval`](GpuParticleMultiScatter::eval) dispatch shares
//! one
//! [`MultiScatterParams`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams)
//! (the common case: a whole medium evaluated on one authored preset) and
//! differs only by its scattering cosine `cos_theta`.
//!
//! # Portability
//!
//! The kernel is the portable core-`WGSL` subset: `sqrt`, `min`, `max`,
//! `clamp`, `+ − × ÷` and index comparisons only. The two integer powers
//! (`(albedo · octave_decay)^i` and `anisotropy_falloff^i`) are formed as
//! running products inside the same fixed integer loop the reference uses, not
//! with `pow`; the Henyey-Greenstein denominator `(1 + g² − 2·g·cosθ)^1.5`
//! factors as `d · sqrt(d)`, so no `exp`, `pow` or `acos` and no optional
//! device feature appears. The scattering cosine is a direction dot product the
//! caller supplies, so no inverse trig is ever needed. The twin therefore runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The twin mirrors
//! [`response_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response_cos)
//! step for step: it sanitizes the parameters (albedo, `anisotropy_falloff` and
//! `octave_decay` into `0..=1`, `ambient_lift` to non-negative, the octave count
//! into `1..=MAX_OCTAVES`), then runs the same sequential octave loop in the
//! same order, so the directional sum is accumulated in the identical order a
//! `GPU` cannot legally reorder. The functions contain no transcendental call,
//! but they are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few `ULP`.
//! The parity test therefore asserts a tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`), tight enough to catch a genuinely wrong port (a swapped
//! factor, a dropped clamp, a missing ambient term) yet loose enough to admit
//! legal fused-multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard Wrenninge/Hillaire-style octave multiple-scattering
//! energy compensation plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Workgroup size along `x`; one invocation evaluates one scattering cosine.
///
/// Matches the `@workgroup_size` in [`SHADER_SOURCE`]; the dispatch rounds the
/// query count up to a multiple of this.
const WORKGROUP_SIZE: u32 = 64;

/// One multiple-scattering result: the per-channel (`RGB`) response.
///
/// Mirrors the `Vec3` returned by
/// [`MultiScatterParams::response_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response_cos):
/// `red`, `green` and `blue` are the three channels of the octave-summed
/// directional response plus the isotropic ambient floor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiScatterResponse {
    /// Red-channel multiple-scattering response.
    pub red: f32,
    /// Green-channel multiple-scattering response.
    pub green: f32,
    /// Blue-channel multiple-scattering response.
    pub blue: f32,
}

impl MultiScatterResponse {
    /// The scalar luminance (mean of the three channels), mirroring
    /// [`MultiScatterParams::luminance_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::luminance_cos).
    ///
    /// The `3.0` divisor is the channel count of an `RGB` triple.
    #[must_use]
    pub fn luminance(self) -> f32 {
        (self.red + self.green + self.blue) / 3.0
    }
}

/// Uniform parameters for one multiple-scattering dispatch.
///
/// Layout matches `Params` in [`SHADER_SOURCE`]: the three albedo channels, the
/// base double-lobe phase (`g`, back-lobe weight, back `g`), the per-octave
/// anisotropy falloff and throughput decay, the ambient-lift strength, the
/// octave count and the query count. All fields are scalars (no `vec3`), so the
/// struct packs tightly to `48` bytes with no `std140` vector-alignment padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Red-channel single-scatter albedo (sanitized in-shader into `0..=1`).
    albedo_r: f32,
    /// Green-channel single-scatter albedo.
    albedo_g: f32,
    /// Blue-channel single-scatter albedo.
    albedo_b: f32,
    /// Base forward-lobe Henyey-Greenstein anisotropy `g`.
    phase_g: f32,
    /// Back-lobe blend weight (clamped into `0..=1` inside the phase).
    back_lobe_weight: f32,
    /// Back-lobe Henyey-Greenstein anisotropy.
    back_g: f32,
    /// Per-octave anisotropy bandwidth factor (sanitized into `0..=1`).
    anisotropy_falloff: f32,
    /// Per-octave throughput decay (sanitized into `0..=1`).
    octave_decay: f32,
    /// Isotropic ambient-floor strength (sanitized to non-negative).
    ambient_lift: f32,
    /// Requested octave count (sanitized into `1..=MAX_OCTAVES` in-shader).
    octaves: u32,
    /// Number of valid entries in `queries` / outputs in `results`.
    count: u32,
    /// Pad word so the uniform block is a multiple of `16` bytes.
    pad0: u32,
}

/// One `RGB` response as uploaded for readback. `16`-byte stride matching
/// `Response` in [`SHADER_SOURCE`]: the triple plus one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResponse {
    red: f32,
    green: f32,
    blue: f32,
    pad: f32,
}

/// A compiled, reusable particle multiple-scattering pipeline.
pub struct GpuParticleMultiScatter {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuParticleMultiScatter {
    /// Compiles the multiple-scattering kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuParticleMultiScatter {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_particle_multiscatter"),
            source: ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_particle_multiscatter_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_particle_multiscatter_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_particle_multiscatter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("multiscatter"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuParticleMultiScatter {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the multiple-scattering response for every scattering cosine in
    /// `cosines` against the shared `params`, returning one
    /// [`MultiScatterResponse`] per cosine in input order.
    ///
    /// The returned response for cosine `c` equals
    /// `params.response_cos(c)` (see
    /// [`MultiScatterParams::response_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response_cos))
    /// to within the tolerance documented on this module; the shader sanitizes
    /// `params` exactly as the reference does, so out-of-range albedo, falloff,
    /// decay, ambient lift and octave count all behave identically. An empty
    /// `cosines` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        params: MultiScatterParams,
        cosines: &[f32],
    ) -> Vec<MultiScatterResponse> {
        if cosines.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = Params {
            albedo_r: params.albedo.x,
            albedo_g: params.albedo.y,
            albedo_b: params.albedo.z,
            phase_g: params.phase.g,
            back_lobe_weight: params.phase.back_lobe_weight,
            back_g: params.phase.back_g,
            anisotropy_falloff: params.anisotropy_falloff,
            octave_decay: params.octave_decay,
            ambient_lift: params.ambient_lift,
            octaves: params.octaves,
            count: cosines.len() as u32,
            pad0: 0,
        };

        let out_bytes = (cosines.len() as u64) * (size_of::<GpuResponse>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_particle_multiscatter_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_particle_multiscatter_queries"),
            contents: bytemuck::cast_slice(cosines),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_particle_multiscatter_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_particle_multiscatter_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_particle_multiscatter_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_particle_multiscatter_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_particle_multiscatter_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (cosines.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_results = bytemuck::cast_slice::<u8, GpuResponse>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), cosines.len());
        gpu_results
            .into_iter()
            .map(|r| MultiScatterResponse {
                red: r.red,
                green: r.green,
                blue: r.blue,
            })
            .collect()
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

/// The portable core-`WGSL` multiple-scattering kernel, embedded as a string so
/// this twin lives entirely in these two files (no external shader asset).
const SHADER_SOURCE: &str = r#"
// Particle multiple-scattering twin: for one scattering cosine it computes the
// octave-summed per-channel (RGB) response, mirroring the CPU golden
// `MultiScatterParams::response_cos` (particle::volumetric_multiscatter).
//
// Order i carries a running-product throughput `(albedo * octave_decay)^i` and
// a double-lobe Henyey-Greenstein phase whose anisotropy is the base g scaled
// by `anisotropy_falloff^i`, so deeper orders are both dimmer and more
// isotropic. An added isotropic ambient term `ambient_lift * ISOTROPIC_PHASE *
// sum(throughput)` guarantees a high-albedo interior never goes dead black.
//
// Only the portable core-WGSL subset is used: sqrt / min / max / clamp and
// + - * /. The two integer powers are formed as running products inside the
// fixed octave loop (never `pow`), and the HG denominator (1 + g^2 - 2 g cos)^1.5
// factors as `d * sqrt(d)`, so no exp / pow / trig and no optional device
// feature appear. The twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Wrenninge/Hillaire-style octave multiple-scattering
// energy compensation; no Unreal Engine source or derived code.

// 4*pi: the Henyey-Greenstein normalization constant. Spelled as a literal to
// match the CPU golden `FOUR_PI` (4.0 * pi rounded to f32); the determinism
// contract forbids deriving it from a transcendental pi.
const FOUR_PI: f32 = 12.566371;

// The isotropic phase 1 / (4*pi): the floor every lobe shares and the value a
// broadened octave tends toward. Derived from FOUR_PI by division, matching the
// CPU golden `ISOTROPIC_PHASE`.
const ISOTROPIC_PHASE: f32 = 1.0 / FOUR_PI;

// Magnitude floor on the HG denominator base, mirroring the CPU golden `EPS`,
// so the grazing case stays large-but-finite instead of dividing by zero.
const EPS: f32 = 1e-6;

// Upper bound on the folded octave count, mirroring the CPU `MAX_OCTAVES`.
const MAX_OCTAVES: u32 = 64u;

struct Params {
    albedo_r: f32,
    albedo_g: f32,
    albedo_b: f32,
    phase_g: f32,
    back_lobe_weight: f32,
    back_g: f32,
    anisotropy_falloff: f32,
    octave_decay: f32,
    ambient_lift: f32,
    octaves: u32,
    count: u32,
    pad0: u32,
}

// One RGB response. `16`-byte stride: the triple plus one pad word.
struct Response {
    red: f32,
    green: f32,
    blue: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cosines: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<Response>;

// Single-lobe Henyey-Greenstein phase, mirroring the CPU golden
// `henyey_greenstein`: (1 - g^2) / (4*pi * (1 + g^2 - 2 g cos)^1.5), with the
// ^1.5 power evaluated as `d * sqrt(d)` and the base floored at EPS.
fn henyey_greenstein(g: f32, cos_theta: f32) -> f32 {
    let g2 = g * g;
    let base = 1.0 + g2 - 2.0 * g * cos_theta;
    var d = base;
    if (base > EPS) {
        d = base;
    } else {
        d = EPS;
    }
    let d15 = d * sqrt(d);
    return (1.0 - g2) / (FOUR_PI * d15);
}

// Double-lobe (front + back) phase, mirroring the CPU golden
// `double_lobe_phase`: blends a forward lobe at `g` with a back lobe at
// `back_g`, weighted by `weight` clamped into [0, 1].
fn double_lobe_phase(g: f32, weight: f32, back_g: f32, cos_theta: f32) -> f32 {
    var w = 0.0;
    if (weight > 1.0) {
        w = 1.0;
    } else if (weight > 0.0) {
        w = weight;
    } else {
        w = 0.0;
    }
    let front = henyey_greenstein(g, cos_theta);
    let back = henyey_greenstein(back_g, cos_theta);
    return (1.0 - w) * front + w * back;
}

@compute @workgroup_size(64)
fn multiscatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let cos_theta = cosines[idx];

    // Sanitize exactly as the CPU golden `sanitized` does: albedo, falloff and
    // decay into [0, 1], ambient lift to non-negative, octaves into
    // [1, MAX_OCTAVES]. The base phase angles are left untouched (the back-lobe
    // weight is clamped inside `double_lobe_phase`).
    let albedo = vec3<f32>(
        clamp(params.albedo_r, 0.0, 1.0),
        clamp(params.albedo_g, 0.0, 1.0),
        clamp(params.albedo_b, 0.0, 1.0),
    );
    let anisotropy_falloff = clamp(params.anisotropy_falloff, 0.0, 1.0);
    let octave_decay = clamp(params.octave_decay, 0.0, 1.0);
    let ambient_lift = max(params.ambient_lift, 0.0);
    var octaves = params.octaves;
    if (octaves < 1u) {
        octaves = 1u;
    }
    if (octaves > MAX_OCTAVES) {
        octaves = MAX_OCTAVES;
    }

    // Per-octave throughput ratio: albedo tinted by the extra decay.
    let decay_rgb = albedo * octave_decay;
    // Order-0 throughput is unit; order-0 anisotropy scale is 1 (base lobe).
    var throughput = vec3<f32>(1.0, 1.0, 1.0);
    var anisotropy_scale = 1.0;
    var directional = vec3<f32>(0.0, 0.0, 0.0);
    var throughput_sum = vec3<f32>(0.0, 0.0, 0.0);

    var i = 0u;
    while (i < octaves) {
        let octave_phase = double_lobe_phase(
            params.phase_g * anisotropy_scale,
            params.back_lobe_weight,
            params.back_g * anisotropy_scale,
            cos_theta,
        );
        directional = directional + throughput * octave_phase;
        throughput_sum = throughput_sum + throughput;
        // Advance the running products for the next octave (multiply only).
        throughput = throughput * decay_rgb;
        anisotropy_scale = anisotropy_scale * anisotropy_falloff;
        i = i + 1u;
    }

    let ambient = throughput_sum * (ambient_lift * ISOTROPIC_PHASE);
    let out_rgb = directional + ambient;

    var out: Response;
    out.red = out_rgb.x;
    out.green = out_rgb.y;
    out.blue = out_rgb.z;
    out.pad = 0.0;
    results[idx] = out;
}
"#;
