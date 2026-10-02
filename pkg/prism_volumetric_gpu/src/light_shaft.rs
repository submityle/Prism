//! `wgpu` compute twin of the screen-space light-shaft (god-ray) radial-blur
//! contract
//! ([`light_shaft`](prism_render_architecture::particle::light_shaft), particle
//! design §16-§21).
//!
//! The `CPU` golden
//! [`light_shaft`](prism_render_architecture::particle::light_shaft) owns the
//! radial, light-centred smear a post-process pass applies to a single-channel
//! occlusion/brightness `mask`: for every destination pixel it marches a short
//! chain of taps from the pixel toward the light's on-screen `UV`, fetches the
//! `mask` with a half-texel-centred, clamp-to-edge bilinear filter, and
//! accumulates the taps under a per-step `illuminationDecay`
//! ([`LightShaftParams::radial_blur_pixel`](prism_render_architecture::particle::light_shaft::LightShaftParams::radial_blur_pixel)
//! and its full-image driver
//! [`LightShaftParams::radial_blur`](prism_render_architecture::particle::light_shaft::LightShaftParams::radial_blur)).
//! [`GpuLightShaft`] is the on-device twin: one shared read-only `mask` is
//! uploaded once and every thread resolves one destination pixel, so a passing
//! real-device parity test is direct evidence the ported kernel reproduces the
//! same marched, bilinearly-filtered shaft the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reproduces, for its pixel, the exact branch structure of
//! [`LightShaftParams::radial_blur_pixel`](prism_render_architecture::particle::light_shaft::LightShaftParams::radial_blur_pixel):
//! the pixel-centre `UV` `((px + 0.5) / w, (py + 0.5) / h)`, the `num_samples ==
//! 0` identity short-circuit returning the bilinear `mask` value, the
//! per-step offset `(light_uv - uv) * density / num_samples`, the seeded
//! accumulator, the `num_samples` taps each scaled by the running
//! `illuminationDecay` (seeded at `1` and multiplied by `decay` after every
//! step) and by `weight`, and the final `exposure` gain. The private
//! half-texel-centred bilinear fetch with clamp-to-edge addressing is twinned
//! too, including its empty-image guard.
//!
//! # Thread mapping
//!
//! The dispatch is one-dimensional with `@workgroup_size(64)`; the group count
//! is `div_ceil(width * height, 64)`. Thread `gid` resolves pixel `px = gid %
//! width`, `py = gid / width`, and a `gid` past `width * height`
//! short-circuits. An empty image or a `mask` shorter than `width * height`
//! short-circuits on the host with no dispatch, matching the empty return of
//! [`LightShaftParams::radial_blur`](prism_render_architecture::particle::light_shaft::LightShaftParams::radial_blur)
//! and respecting the rule that a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! The shaft value threads through multiplies, adds and one guarded division,
//! so it is not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every shaft pixel, tight enough to catch a dropped term, a swapped axis or a
//! mis-ordered decay yet loose enough to admit legal fused multiply-add
//! contraction. The integer texel addressing of the bilinear fetch is exact.
//!
//! # Degenerate inputs
//!
//! An empty image (`width == 0` or `height == 0`) or a `mask` shorter than
//! `width * height` yields an empty [`LightShaftResult`], mirroring the
//! reference `radial_blur`. A `dim = 1` axis clamps every texel read to index
//! `0`. An off-screen `light_uv` or an extreme `density` / `decay` stays finite
//! because the bilinear fetch clamps to the edge and the kernel contains no
//! transcendental call.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `min`,
//! `max`, `clamp`, `abs` and `+ - * /` plus unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `smoothstep` and no `sqrt`, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. The only loop runs `num_samples` times, a dispatch-uniform bound, so
//! the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::light_shaft`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` light-shaft kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`light_shaft`](prism_render_architecture::particle::light_shaft) branch for
/// branch; see the module documentation for the algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::light_shaft`。
const LIGHT_SHAFT_WGSL: &str = r#"
// Light-shaft twin: one shared read-only mask, one thread per destination
// pixel. Each thread reproduces the pixel-centre UV, the num_samples == 0
// identity short-circuit, the per-step march toward the light's screen UV, the
// half-texel-centred clamp-to-edge bilinear mask fetch and the decay-weighted
// accumulation finally scaled by exposure. It mirrors the CPU golden
// `particle::light_shaft` branch for branch, uses only the portable core-WGSL
// subset (floor/min/max/clamp/abs and + - * / plus index arithmetic), needs no
// sqrt and no transcendental call and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12. The only loop runs num_samples times, a
// dispatch-uniform bound, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::light_shaft；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Light's on-screen position in [0, 1] UV; may lie outside the frame.
    light_uv: vec2<f32>,
    // Fraction of the pixel-to-light vector the tap chain spans.
    density: f32,
    // Per-step multiplier applied to illuminationDecay.
    decay: f32,
    // Gain applied to each accumulated tap.
    weight: f32,
    // Global gain applied to the final accumulated shaft.
    exposure: f32,
    // Number of taps marched from each pixel toward the light.
    num_samples: u32,
    // Mask width in texels (at least 1 when a dispatch is issued).
    width: u32,
    // Mask height in texels (at least 1 when a dispatch is issued).
    height: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> shaft: array<f32>;

// Clamps a (possibly negative or oversized) texel coordinate into [0, dim - 1];
// mirrors the reference `clamp_coord`. `dim` is non-zero. In the interior
// branch the coordinate lies strictly inside the open interval (0, dim - 1), so
// it is non-negative and `u32(floor(v))` reproduces the host `v as usize`
// truncation toward zero bit for bit.
fn clamp_coord(v: f32, dim: u32) -> u32 {
    if (v <= 0.0) {
        return 0u;
    }
    let last = dim - 1u;
    if (v >= f32(last)) {
        return last;
    }
    return u32(floor(v));
}

// Hand-rolled bilinear fetch of the single-channel mask at a texture-space uv,
// using the half-texel-centred convention and clamp-to-edge addressing; mirrors
// the reference `sample_bilinear`. An empty image samples as 0.
fn sample_bilinear(uv: vec2<f32>) -> f32 {
    let w = params.width;
    let h = params.height;
    if (w == 0u || h == 0u) {
        return 0.0;
    }
    let fx = uv.x * f32(w) - 0.5;
    let fy = uv.y * f32(h) - 0.5;
    let x0f = floor(fx);
    let y0f = floor(fy);
    let tx = fx - x0f;
    let ty = fy - y0f;
    let ix0 = clamp_coord(x0f, w);
    let ix1 = clamp_coord(x0f + 1.0, w);
    let iy0 = clamp_coord(y0f, h);
    let iy1 = clamp_coord(y0f + 1.0, h);
    let c00 = mask[iy0 * w + ix0];
    let c10 = mask[iy0 * w + ix1];
    let c01 = mask[iy1 * w + ix0];
    let c11 = mask[iy1 * w + ix1];
    let top = c00 + (c10 - c00) * tx;
    let bot = c01 + (c11 - c01) * tx;
    return top + (bot - top) * ty;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let w = params.width;
    let h = params.height;
    let total = w * h;
    if (idx >= total) {
        return;
    }
    let px = idx % w;
    let py = idx / w;
    let uv = vec2<f32>(
        (f32(px) + 0.5) / f32(w),
        (f32(py) + 0.5) / f32(h),
    );
    if (params.num_samples == 0u) {
        shaft[idx] = sample_bilinear(uv);
        return;
    }
    let inv_n = 1.0 / f32(params.num_samples);
    let step = vec2<f32>(
        (params.light_uv.x - uv.x) * params.density * inv_n,
        (params.light_uv.y - uv.y) * params.density * inv_n,
    );
    var pos = uv;
    var color = sample_bilinear(pos);
    var illumination = 1.0;
    for (var i = 0u; i < params.num_samples; i = i + 1u) {
        pos = pos + step;
        let tap = sample_bilinear(pos) * illumination;
        color = color + tap * params.weight;
        illumination = illumination * params.decay;
    }
    shaft[idx] = color * params.exposure;
}
"#;

/// The six radial-blur knobs the twin feeds into one dispatch, mirroring the
/// field set of the `CPU` golden
/// [`LightShaftParams`](prism_render_architecture::particle::light_shaft::LightShaftParams).
///
/// `light_uv` is the light's on-screen position in `[0, 1]` texture space (it
/// may lie outside the frame); `num_samples` is the tap count marched from each
/// pixel toward the light; `density` scales the marched distance; `decay` is
/// the per-step `illuminationDecay` multiplier; `weight` is the per-tap gain;
/// and `exposure` is the global gain applied to the final shaft. No clamping or
/// normalisation is performed, so extreme values can be exercised directly.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::light_shaft`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuLightShaftParams {
    /// Light's on-screen position in `[0, 1]` texture-space `UV`.
    pub light_uv: [f32; 2],
    /// Number of taps marched from each pixel toward the light.
    pub num_samples: u32,
    /// Fraction of the pixel-to-light vector the tap chain spans.
    pub density: f32,
    /// Per-step multiplier applied to `illuminationDecay`.
    pub decay: f32,
    /// Gain applied to each accumulated tap.
    pub weight: f32,
    /// Global gain applied to the final accumulated shaft.
    pub exposure: f32,
}

impl GpuLightShaftParams {
    /// Builds a parameter set from its raw fields verbatim, in the same order
    /// as the `CPU` golden
    /// [`LightShaftParams::new`](prism_render_architecture::particle::light_shaft::LightShaftParams::new).
    #[must_use]
    pub const fn new(
        light_uv: [f32; 2],
        num_samples: u32,
        density: f32,
        decay: f32,
        weight: f32,
        exposure: f32,
    ) -> GpuLightShaftParams {
        GpuLightShaftParams {
            light_uv,
            num_samples,
            density,
            decay,
            weight,
            exposure,
        }
    }
}

/// Uniform parameters for one dispatch: the six radial-blur knobs plus the mask
/// dimensions, padded to the `std140` `16`-byte alignment matching `Params` in
/// [`LIGHT_SHAFT_WGSL`]. The leading `light_uv`/`density`/`decay`/`weight`/
/// `exposure`/`num_samples` layout reproduces the `std430` packing of
/// [`LightShaftParams::to_std430`](prism_render_architecture::particle::light_shaft::LightShaftParams::to_std430).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Light's on-screen position in `[0, 1]` texture-space `UV`.
    light_uv: [f32; 2],
    /// Fraction of the pixel-to-light vector the tap chain spans.
    density: f32,
    /// Per-step multiplier applied to `illuminationDecay`.
    decay: f32,
    /// Gain applied to each accumulated tap.
    weight: f32,
    /// Global gain applied to the final accumulated shaft.
    exposure: f32,
    /// Number of taps marched from each pixel toward the light.
    num_samples: u32,
    /// Mask width in texels.
    width: u32,
    /// Mask height in texels.
    height: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// The shaft image produced for one `mask`, in row-major (`X`-fastest) order,
/// `width * height` long. An empty image or an undersized `mask` yields an
/// empty vector, mirroring
/// [`LightShaftParams::radial_blur`](prism_render_architecture::particle::light_shaft::LightShaftParams::radial_blur).
#[derive(Clone, Debug, PartialEq)]
pub struct LightShaftResult {
    /// Row-major shaft values, one per destination pixel.
    pub shaft: Vec<f32>,
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

/// A compiled, reusable light-shaft compute pipeline, twinning the `CPU` golden
/// [`light_shaft`](prism_render_architecture::particle::light_shaft).
pub struct GpuLightShaft {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuLightShaft {
    /// Compiles the light-shaft kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuLightShaft {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_light_shaft"),
            source: ShaderSource::Wgsl(LIGHT_SHAFT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_light_shaft_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_light_shaft_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_light_shaft_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuLightShaft {
            module,
            layout,
            pipeline,
        }
    }

    /// Applies the radial blur described by `params` to a `width * height`
    /// single-channel `mask`, returning the row-major shaft image.
    ///
    /// `mask` must be at least `width * height` long; only the first `width *
    /// height` values are read, matching the reference. Every shaft pixel
    /// matches
    /// [`LightShaftParams::radial_blur_pixel`](prism_render_architecture::particle::light_shaft::LightShaftParams::radial_blur_pixel)
    /// to within the tolerance documented on this module. An empty image
    /// (`width == 0` or `height == 0`) or a `mask` shorter than `width *
    /// height` returns an empty [`LightShaftResult`] with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn blur(
        &self,
        ctx: &GpuContext,
        params: GpuLightShaftParams,
        mask: &[f32],
        width: u32,
        height: u32,
    ) -> LightShaftResult {
        let total = (width as usize) * (height as usize);
        if width == 0 || height == 0 || mask.len() < total {
            return LightShaftResult { shaft: Vec::new() };
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            light_uv: params.light_uv,
            density: params.density,
            decay: params.decay,
            weight: params.weight,
            exposure: params.exposure,
            num_samples: params.num_samples,
            width,
            height,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_light_shaft_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        // Only the first `total` mask texels participate, matching the
        // reference which reads `mask[0..w*h]`.
        let mask_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_light_shaft_mask"),
            contents: bytemuck::cast_slice(&mask[..total]),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (total * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_light_shaft_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_light_shaft_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: mask_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_light_shaft_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_light_shaft_encoder"),
        });
        {
            let groups = (total as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_light_shaft_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per destination pixel, flattened to a 1-D dispatch.
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
        let shaft = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        LightShaftResult { shaft }
    }
}
