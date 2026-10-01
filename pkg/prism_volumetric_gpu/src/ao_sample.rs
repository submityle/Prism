//! `wgpu` compute twin of the screen-space `hemisphere` ambient-occlusion
//! (`AO`) sampling kernel
//! ([`particle::ao_sample`](prism_render_architecture::particle::ao_sample),
//! design §16-§21, `SSAO` / `HBAO` style).
//!
//! Production engines darken creases and contact points with a screen-space
//! ambient-occlusion pass: around each shaded point a small set of offset
//! directions is scattered over the upper `hemisphere` aligned with the surface
//! normal, the scene depth is fetched at each offset, and a point counts as
//! occluded when the sampled geometry sits *nearer* the camera than the offset
//! position. The `CPU` golden
//! [`particle::ao_sample`](prism_render_architecture::particle::ao_sample) owns
//! that math; [`GpuAoSample`] is the on-device twin, documented so a passing
//! real-device parity test is direct evidence the ported kernels compute the
//! same values as the reference, not merely that the shaders compile.
//!
//! # Two kernels, two parity surfaces
//!
//! The twin ships two compute entry points behind two self-contained shader
//! modules, each binding at `@group(0)`:
//!
//! * `build_kernel` runs one thread per sample index and reproduces
//!   [`sample_kernel`](prism_render_architecture::particle::ao_sample::sample_kernel):
//!   an integer `R2` low-discrepancy sequence scatters the index over the unit
//!   square, the trig-free *elliptical grid* map lifts it to the unit disk, and
//!   `z = sqrt(1 - x*x - y*y)` lifts that onto the upper `hemisphere`
//!   (`z >= 0`, the normal-aligned axis), scaled by the near-field radius
//!   `0.1 + 0.9 * t*t`. This is the per-sample parity surface.
//! * `evaluate_ao` runs one thread per shaded pixel and reproduces
//!   [`AoParams::evaluate`](prism_render_architecture::particle::ao_sample::AoParams::evaluate):
//!   the per-pixel bias subtraction, the `smoothstep` range-checked occlusion
//!   fold of
//!   [`ao_from_samples`](prism_render_architecture::particle::ao_sample::ao_from_samples),
//!   the integer-exponent contrast of
//!   [`ao_power`](prism_render_architecture::particle::ao_sample::ao_power), and
//!   the intensity blend. This is the per-pixel parity surface.
//!
//! Both reuse the `CPU` golden types
//! ([`AoParams`](prism_render_architecture::particle::ao_sample::AoParams)) and
//! the shared workgroup round-up
//! [`dispatch_groups`](prism_render_architecture::particle::ao_sample::dispatch_groups);
//! nothing is re-derived.
//!
//! # Portability
//!
//! Both kernels use only the portable core-`WGSL` subset — `sqrt`, `min`,
//! `max`, `clamp`, `floor`/`ceil`, the four arithmetic operators and unsigned
//! integer bit operations — with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`,
//! `atan2` or optional device feature, so they run unmodified on Metal, Vulkan
//! and `DX12`. The only non-integer primitive is `sqrt` (the disk and
//! `hemisphere` lift), exactly as in the reference; the `hemisphere`
//! distribution is pure `sqrt` plus algebra, never trigonometry.
//!
//! # Correctness model
//!
//! Both kernels are closed-form algebra and integer hashing with no
//! transcendental call and no reorderable reduction, so `CPU` and `GPU`
//! evaluate the same expression in the same order. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate, or round
//! the `u32`-to-`f32` hash normalization by one unit in the last place. The
//! parity test asserts a tolerance (`abs_diff <= 1e-5` or `rel_diff <= 1e-5`)
//! tight enough to catch a genuinely wrong port yet loose enough to admit legal
//! fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `SSAO` / `HBAO` `hemisphere` sampling plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::ao_sample::{dispatch_groups, AoParams};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Threads per workgroup for both kernels. One thread handles one sample index
/// (`build_kernel`) or one shaded pixel (`evaluate_ao`).
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` `hemisphere` kernel-generation shader, embedded
/// inline. Reproduces
/// [`sample_kernel`](prism_render_architecture::particle::ao_sample::sample_kernel):
/// an integer `R2` sequence, the trig-free elliptical-grid disk map and a
/// `sqrt` `hemisphere` lift. Binds at `@group(0)`; see the module documentation
/// for the algorithm.
///
/// Provenance: standard `SSAO` / `HBAO` `hemisphere` sampling; no Unreal Engine
/// source or derived code.
const KERNEL_WGSL: &str = r#"
// Hemisphere kernel-generation twin: one thread per sample index scatters the
// index onto the upper hemisphere with an integer R2 low-discrepancy sequence
// and the trig-free elliptical-grid disk map lifted by `z = sqrt(1 - x*x -
// y*y)`. Mirrors the CPU golden `sample_kernel`, uses only the portable
// core-WGSL subset (sqrt/min/max and + - * / plus integer bit ops), and takes
// no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard SSAO / HBAO hemisphere sampling; no Unreal Engine source
// or derived code.

struct Params {
    // Number of hemisphere samples to generate (`n`); one thread each.
    sample_count: u32,
    // Seed for the integer hash / R2 sequence.
    seed: u32,
    // Padding to a 16-byte std430 struct.
    pad0: u32,
    pad1: u32,
}

// One hemisphere direction. 16-byte std430 stride: the xyz triple plus one pad
// word, matching the host `GpuKernelSample`.
struct KernelSample {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> kernel_out: array<KernelSample>;

// `2^32` as f32: normalizes a u32 hash word into a 0..1 unit fraction. Matches
// the reference `U32_SPAN`.
const U32_SPAN: f32 = 4294967296.0;
// Fixed-point .32 increment for the first R2 axis (`round(2^32 / plastic)`),
// matching the reference `R2_INC_X`.
const R2_INC_X: u32 = 3242174889u;
// Fixed-point .32 increment for the second R2 axis
// (`round(2^32 / plastic^2)`), matching the reference `R2_INC_Y`.
const R2_INC_Y: u32 = 2447445413u;
// Golden-ratio odd constant decorrelating the two hash seeds, matching the
// reference `seed ^ 0x9e37_79b9`.
const SEED_SHIFT: u32 = 0x9e3779b9u;

// Integer avalanche hash: xor-shifts and odd-constant multiplies, identical to
// the reference `hash_u32`. The u32 multiplies wrap modulo 2^32, matching
// `wrapping_mul`.
fn hash_u32(seed: u32) -> u32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

// Normalizes a u32 hash word into a 0..1 unit fraction, matching the reference
// `to_unit`.
fn to_unit(x: u32) -> f32 {
    return f32(x) / U32_SPAN;
}

@compute @workgroup_size(64)
fn build_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.sample_count) {
        return;
    }
    let base_x = hash_u32(params.seed);
    let base_y = hash_u32(params.seed ^ SEED_SHIFT);
    // Integer R2 advance (wraps modulo 2^32, matching wrapping_add/mul).
    let u = to_unit(base_x + R2_INC_X * i);
    let v = to_unit(base_y + R2_INC_Y * i);
    // Unit square -> [-1, 1]^2.
    let p = 2.0 * u - 1.0;
    let q = 2.0 * v - 1.0;
    // Elliptical grid map: square -> unit disk using only sqrt.
    let x = p * sqrt(max(1.0 - 0.5 * q * q, 0.0));
    let y = q * sqrt(max(1.0 - 0.5 * p * p, 0.0));
    // Lift onto the upper hemisphere; the (x, y, z) triple is unit long.
    let z = sqrt(max(1.0 - x * x - y * y, 0.0));
    let t = f32(i) / f32(params.sample_count);
    // Near-field radius bias `0.1 + 0.9 * t*t`; the two magic weights match the
    // reference and keep the first sample at the 0.1 minimum radius.
    let scale = 0.1 + 0.9 * t * t;
    var result: KernelSample;
    result.x = x * scale;
    result.y = y * scale;
    result.z = z * scale;
    result.pad = 0.0;
    kernel_out[i] = result;
}
"#;

/// The portable core-`WGSL` per-pixel occlusion-fold shader, embedded inline.
/// Reproduces
/// [`AoParams::evaluate`](prism_render_architecture::particle::ao_sample::AoParams::evaluate):
/// the bias subtraction, the `smoothstep` range-checked occlusion mean, the
/// integer-exponent contrast and the intensity blend. Binds at `@group(0)`.
///
/// Provenance: standard `SSAO` / `HBAO` occlusion fold; no Unreal Engine source
/// or derived code.
const EVAL_WGSL: &str = r#"
// Per-pixel occlusion-fold twin: one thread per shaded pixel folds the scene
// depths fetched around it into an occlusion term with a smoothstep range
// check, integer-exponent contrast and an intensity blend. Mirrors the CPU
// golden `AoParams::evaluate`, uses only the portable core-WGSL subset
// (clamp/min/max and + - * /), and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard SSAO / HBAO occlusion fold; no Unreal Engine source or
// derived code.

struct Params {
    // view-space gather radius, matching `AoParams::radius`.
    radius: f32,
    // Depth bias suppressing self-occlusion, matching `AoParams::bias`.
    bias: f32,
    // Darkening scale, matching `AoParams::intensity`.
    intensity: f32,
    // Integer contrast exponent, matching `AoParams::power`.
    power: u32,
    // Mirrors `AoParams::sample_count` for std430 word-order parity (unused by
    // the fold itself, which iterates `samples_per_pixel` fetched depths).
    sample_count: u32,
    // Count of fetched scene depths per pixel in this dispatch.
    samples_per_pixel: u32,
    // Total shaded-pixel count (one thread each).
    pixel_count: u32,
    // Padding to a 32-byte std430 struct (two vec4 slots).
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> center_depths: array<f32>;
@group(0) @binding(2) var<storage, read> sampled_depths: array<f32>;
@group(0) @binding(3) var<storage, read_write> ao_out: array<f32>;

// Generic denominator / soft-edge guard below which a division collapses to a
// hard step, matching the reference `MIN_EDGE`.
const MIN_EDGE: f32 = 1.0e-6;

// Clamps a scalar into 0..=1, matching the reference `clamp01`.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Hermite smoothstep from `edge0` to `edge1` evaluated at `x`, matching the
// reference `smoothstep`. A degenerate interval collapses to a hard step at
// `edge1` rather than dividing by zero.
fn smoothstep_ref(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (span < MIN_EDGE) {
        if (x < edge1) {
            return 0.0;
        }
        return 1.0;
    }
    let t = clamp01((x - edge0) / span);
    // The constants 3 and 2 are the fixed Hermite basis coefficients.
    return t * t * (3.0 - 2.0 * t);
}

// Raises `base_in` to the integer power `exponent` by repeated multiplication,
// matching the reference `ao_power`. `exponent == 0` yields 1.0.
fn ao_power(base_in: f32, exponent: u32) -> f32 {
    let base = clamp01(base_in);
    var acc = 1.0;
    var e = 0u;
    loop {
        if (e >= exponent) {
            break;
        }
        acc = acc * base;
        e = e + 1u;
    }
    return acc;
}

@compute @workgroup_size(64)
fn evaluate_ao(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.x;
    if (pixel >= params.pixel_count) {
        return;
    }
    // The bias is subtracted from the shaded depth so an occluder must be nearer
    // by more than the bias to count, matching `AoParams::evaluate`.
    let center = center_depths[pixel] - params.bias;
    let spp = params.samples_per_pixel;
    // An empty sample set is unoccluded (`ao_from_samples` returns 1.0); the
    // division guard below is only reached when `spp > 0`.
    var raw = 1.0;
    if (spp > 0u) {
        var occlusion = 0.0;
        var j = 0u;
        loop {
            if (j >= spp) {
                break;
            }
            let d = sampled_depths[pixel * spp + j];
            let delta = center - d;
            if (delta > 0.0) {
                // Soft range check: full weight while `delta < radius`, fading
                // out beyond it. `radius / delta` is guarded by `delta > 0`.
                occlusion = occlusion + smoothstep_ref(0.0, 1.0, params.radius / delta);
            }
            j = j + 1u;
        }
        let mean = occlusion / f32(spp);
        raw = clamp01(1.0 - mean);
    }
    let powered = ao_power(raw, params.power);
    ao_out[pixel] = clamp01(1.0 - params.intensity * (1.0 - powered));
}
"#;

/// Uniform parameters for one `build_kernel` dispatch. `repr(C)` `std430`
/// layout matching `Params` in [`KERNEL_WGSL`]: the sample count and seed then
/// two pad words — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct KernelParams {
    /// Number of `hemisphere` samples to generate.
    sample_count: u32,
    /// Seed for the integer hash / `R2` sequence.
    seed: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One `hemisphere` sample direction as read back. `16`-byte `std430` stride
/// matching `KernelSample` in [`KERNEL_WGSL`]: the `xyz` triple plus a pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuKernelSample {
    /// Tangent-space `x` component.
    x: f32,
    /// Tangent-space `y` component.
    y: f32,
    /// Normal-aligned `z` component (`>= 0`, the upper `hemisphere` axis).
    z: f32,
    /// Padding word.
    pad: f32,
}

/// Uniform parameters for one `evaluate_ao` dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`EVAL_WGSL`]. The first five words share the word
/// order of
/// [`AoParams::to_std430`](prism_render_architecture::particle::ao_sample::AoParams::to_std430)
/// (`radius`, `bias`, `intensity`, `power`, `sample_count`); the remaining
/// words carry this dispatch's `samples_per_pixel` and `pixel_count` plus one
/// pad — `32` bytes (two `vec4` slots) with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EvalParams {
    /// `view-space` gather radius.
    radius: f32,
    /// Depth bias suppressing self-occlusion.
    bias: f32,
    /// Darkening scale.
    intensity: f32,
    /// Integer contrast exponent.
    power: u32,
    /// Mirrors `AoParams::sample_count` for `std430` word-order parity.
    sample_count: u32,
    /// Count of fetched scene depths per pixel in this dispatch.
    samples_per_pixel: u32,
    /// Total shaded-pixel count.
    pixel_count: u32,
    /// Padding word.
    pad0: u32,
}

/// A compiled, reusable ambient-occlusion twin exposing both the per-sample
/// `hemisphere` kernel and the per-pixel occlusion fold.
pub struct GpuAoSample {
    #[expect(
        dead_code,
        reason = "kept alive so the kernel pipeline it produced stays valid"
    )]
    kernel_module: ShaderModule,
    #[expect(
        dead_code,
        reason = "kept alive so the evaluation pipeline it produced stays valid"
    )]
    eval_module: ShaderModule,
    kernel_layout: BindGroupLayout,
    kernel_pipeline: ComputePipeline,
    eval_layout: BindGroupLayout,
    eval_pipeline: ComputePipeline,
}

impl GpuAoSample {
    /// Compiles the ambient-occlusion kernels on `ctx`.
    ///
    /// Both entry points use only the portable core-`WGSL` subset, so no
    /// optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAoSample {
        let device = ctx.device();

        let kernel_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ao_sample_kernel"),
            source: ShaderSource::Wgsl(KERNEL_WGSL.into()),
        });
        let kernel_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ao_sample_kernel_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let kernel_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ao_sample_kernel_pipeline_layout"),
            bind_group_layouts: &[Some(&kernel_layout)],
            immediate_size: 0,
        });
        let kernel_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ao_sample_kernel_pipeline"),
            layout: Some(&kernel_pipeline_layout),
            module: &kernel_module,
            entry_point: Some("build_kernel"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let eval_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ao_sample_eval"),
            source: ShaderSource::Wgsl(EVAL_WGSL.into()),
        });
        let eval_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ao_sample_eval_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let eval_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ao_sample_eval_pipeline_layout"),
            bind_group_layouts: &[Some(&eval_layout)],
            immediate_size: 0,
        });
        let eval_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ao_sample_eval_pipeline"),
            layout: Some(&eval_pipeline_layout),
            module: &eval_module,
            entry_point: Some("evaluate_ao"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuAoSample {
            kernel_module,
            eval_module,
            kernel_layout,
            kernel_pipeline,
            eval_layout,
            eval_pipeline,
        }
    }

    /// Builds the `sample_count`-direction upper-`hemisphere` sampling kernel
    /// seeded by `seed`, returning one `[x, y, z]` direction per sample in
    /// index order.
    ///
    /// The returned directions equal
    /// [`sample_kernel`](prism_render_architecture::particle::ao_sample::sample_kernel)
    /// to within the tolerance documented on this module. A `sample_count` of
    /// `0` yields an empty kernel — a storage buffer cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval_kernel(&self, ctx: &GpuContext, sample_count: u32, seed: u32) -> Vec<[f32; 3]> {
        if sample_count == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let count = sample_count as usize;

        let params = KernelParams {
            sample_count,
            seed,
            pad0: 0,
            pad1: 0,
        };
        let out_bytes = (count as u64) * (size_of::<GpuKernelSample>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ao_sample_kernel_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ao_sample_kernel_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ao_sample_kernel_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ao_sample_kernel_bind_group"),
            layout: &self.kernel_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ao_sample_kernel_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ao_sample_kernel_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kernel_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per sample, in workgroups of `WORKGROUP_SIZE`, via the
            // shared reference round-up.
            pass.dispatch_workgroups(dispatch_groups(sample_count, WORKGROUP_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage_buf, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage_buf.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage_buf
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_samples = bytemuck::cast_slice::<u8, GpuKernelSample>(&view).to_vec();
        drop(view);
        stage_buf.unmap();
        debug_assert_eq!(gpu_samples.len(), count);

        gpu_samples.into_iter().map(|s| [s.x, s.y, s.z]).collect()
    }

    /// Evaluates the ambient-occlusion term in `0..=1` for every shaded pixel.
    ///
    /// `center_depths` holds one shaded-point depth per pixel.
    /// `sampled_depths` holds the scene depths fetched around each pixel, laid
    /// out pixel-major as `samples_per_pixel` contiguous depths per pixel
    /// (length `center_depths.len() * samples_per_pixel`). The returned term
    /// for pixel `i` equals
    /// [`AoParams::evaluate`](prism_render_architecture::particle::ao_sample::AoParams::evaluate)
    /// over that pixel's depth run to within the tolerance documented on this
    /// module, following the same `1.0` = fully lit convention.
    ///
    /// An empty `center_depths` yields an empty result (early return). A
    /// `samples_per_pixel` of `0` leaves every pixel fully lit, matching the
    /// empty-set rule of
    /// [`ao_from_samples`](prism_render_architecture::particle::ao_sample::ao_from_samples);
    /// the depth storage buffer is padded to one element so it is never
    /// zero-sized.
    #[must_use]
    pub fn eval_ao(
        &self,
        ctx: &GpuContext,
        params: AoParams,
        samples_per_pixel: u32,
        center_depths: &[f32],
        sampled_depths: &[f32],
    ) -> Vec<f32> {
        let pixel_count = center_depths.len();
        if pixel_count == 0 {
            return Vec::new();
        }
        debug_assert_eq!(
            sampled_depths.len(),
            pixel_count * (samples_per_pixel as usize),
            "sampled_depths must hold samples_per_pixel contiguous depths per pixel"
        );
        let device = ctx.device();

        let gpu_params = EvalParams {
            radius: params.radius,
            bias: params.bias,
            intensity: params.intensity,
            power: params.power,
            sample_count: params.sample_count,
            samples_per_pixel,
            pixel_count: pixel_count as u32,
            pad0: 0,
        };

        // A WebGPU storage binding may not be zero-sized; pad the depth buffer
        // to one element when a dispatch fetches no depths per pixel.
        let depth_fallback = [0.0f32];
        let depth_src: &[f32] = if sampled_depths.is_empty() {
            &depth_fallback
        } else {
            sampled_depths
        };
        let out_bytes = (pixel_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ao_sample_eval_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let center_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ao_sample_eval_center"),
            contents: bytemuck::cast_slice(center_depths),
            usage: BufferUsages::STORAGE,
        });
        let depths_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ao_sample_eval_depths"),
            contents: bytemuck::cast_slice(depth_src),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ao_sample_eval_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ao_sample_eval_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ao_sample_eval_bind_group"),
            layout: &self.eval_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: center_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: depths_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ao_sample_eval_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ao_sample_eval_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.eval_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pixel, in workgroups of `WORKGROUP_SIZE`.
            pass.dispatch_workgroups(dispatch_groups(pixel_count as u32, WORKGROUP_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage_buf, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage_buf.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage_buf
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_ao = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage_buf.unmap();
        debug_assert_eq!(gpu_ao.len(), pixel_count);

        gpu_ao
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
