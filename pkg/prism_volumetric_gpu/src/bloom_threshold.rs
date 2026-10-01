//! `wgpu` compute twin of the `bloom` bright-pass threshold extraction
//! ([`bloom_threshold`](prism_render_architecture::particle::bloom_threshold),
//! design §16-§21, "`Bloom` 阈值提取").
//!
//! `Bloom` sells the illusion of a surface emitting more light than the display
//! can show by bleeding the brightest regions of the `HDR` framebuffer into a
//! soft halo. The first stage of that effect is always the *bright-pass*:
//! isolate the pixels above a `luminance` `threshold`, keep only their
//! thresholded contribution, and scale the result by a global `intensity` gain.
//! The `CPU` golden
//! [`BloomThresholdParams`](prism_render_architecture::particle::bloom_threshold::BloomThresholdParams)
//! owns that math; [`GpuBloomThreshold`] is the on-device twin that runs one
//! thread per *input* `HDR` color and reproduces the batch form
//! [`threshold_batch`](prism_render_architecture::particle::bloom_threshold::BloomThresholdParams::threshold_batch)
//! element for element. A passing real-device parity test is therefore direct
//! evidence the ported kernel weights the same `Rec. 709` `luminance`, walks the
//! same soft-`knee` rational curve and preserves the same hue the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The whole per-pixel bright-pass contract is reproduced in one kernel: the
//! `Rec. 709`
//! [`luminance`](prism_render_architecture::particle::bloom_threshold::luminance)
//! dot product, the soft-`knee`
//! [`knee_response`](prism_render_architecture::particle::bloom_threshold::BloomThresholdParams::knee_response)
//! (the quadratic rational curve `soft^2 / (4 knee + eps)` floored by the hard
//! response `lum - threshold`), the normalized
//! [`contribution`](prism_render_architecture::particle::bloom_threshold::BloomThresholdParams::contribution)
//! `knee_response / max(lum, eps)` and the hue-preserving
//! [`threshold_color`](prism_render_architecture::particle::bloom_threshold::BloomThresholdParams::threshold_color)
//! scale by that `contribution` times `intensity`. Every scalar is evaluated in
//! the reference's own order so the low mantissa bits agree.
//!
//! The anti-firefly
//! [`karis_average`](prism_render_architecture::particle::bloom_threshold::BloomThresholdParams::karis_average)
//! is a *reduction* over a tap set, not a per-pixel map, so it is deliberately
//! out of scope for this one-thread-per-element twin; its per-tap
//! [`karis_weight`](prism_render_architecture::particle::bloom_threshold::BloomThresholdParams::karis_weight)
//! is a rational `1 / (1 + luma)` already exercised wherever a future downsample
//! twin binds it.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `max`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `sqrt` or optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. There is no transcendental call on this path:
//! the soft `knee` is a quadratic rational curve whose only divides are by the
//! guarded denominators `4 knee + eps` and `max(lum, eps)`, which are never
//! zero.
//!
//! # Correctness model
//!
//! Each output color is a fixed, non-reorderable sequence of multiplies, adds,
//! one `clamp`, two `max` and two guarded divides, so `CPU` and `GPU` evaluate
//! the same closed form in the same order. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough
//! to catch a genuinely wrong port (a swapped `luminance` weight, a dropped
//! `knee` term, a missing `intensity` gain, a lost hue factor) yet loose enough
//! to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `bloom` bright-pass prefilter with a soft `knee`
//! (`Jimenez`, "Next Generation Post Processing in Call of Duty: Advanced
//! Warfare", `SIGGRAPH` 2014) plus `wgpu` compute dispatch; mirrors the `CPU`
//! golden `prism_render_architecture::particle::bloom_threshold`; no
//! third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::bloom_threshold::BloomThresholdParams;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// the sibling twins use; a one-dimensional dispatch covers the flat color
/// array.
const WORKGROUP_SIZE: u32 = 64;

/// `f32` channels stored per color. The host flattens each `RGB` triple into a
/// `count * 3` `f32` array so a storage buffer needs no `vec3` alignment
/// padding, exactly as the reference keeps its colors.
const CHANNELS: usize = 3;

/// The portable core-`WGSL` bright-pass kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `threshold` mirrors
/// the `CPU` golden
/// [`threshold_color`](prism_render_architecture::particle::bloom_threshold::BloomThresholdParams::threshold_color)
/// scalar for scalar; see the module documentation for the algorithm.
const BLOOM_THRESHOLD_WGSL: &str = r#"
// Bloom bright-pass twin: one thread per input HDR color computes the Rec. 709
// luminance, the soft-knee threshold response, the normalized contribution and
// the hue-preserving intensity scale, mirroring the CPU golden
// `particle::bloom_threshold`. It uses only the portable core-WGSL subset
// (clamp/max and + - * / plus unsigned index math) with no transcendental and
// no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard bloom bright-pass prefilter (Jimenez, SIGGRAPH 2014);
// no third-party engine source or derived code.

struct Params {
    // Luminance threshold where a pixel starts contributing to bloom.
    threshold: f32,
    // Half-width of the soft knee transition band around the threshold.
    knee: f32,
    // Global gain applied to the extracted bloom source color.
    intensity: f32,
    // Number of input colors (one thread each); the flat buffers hold 3 each.
    count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

// Rec. 709 luminance weights, matching the reference `luminance` constants.
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

// Denominators with magnitude below this are treated as (near) zero so the
// guarded divides fall back to a defined result instead of dividing by zero,
// mirroring the reference `MIN_DENOM`.
const MIN_DENOM: f32 = 1e-6;

// The soft-knee numerator `max(soft, lum - threshold)`, where the `soft` branch
// is the quadratic rational `soft^2 / (4 knee + eps)` over the transition band,
// written in the reference `knee_response` order so the low mantissa bits agree.
fn knee_response(lum: f32) -> f32 {
    let over = lum - params.threshold;
    var soft = clamp(over + params.knee, 0.0, 2.0 * params.knee);
    soft = soft * soft / (4.0 * params.knee + MIN_DENOM);
    return max(soft, over);
}

@compute @workgroup_size(64)
fn threshold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let base = idx * 3u;
    let r = src[base];
    let g = src[base + 1u];
    let b = src[base + 2u];
    // Hand-rolled Rec. 709 dot product in the reference `luminance` order.
    let lum = r * LUMA_R + g * LUMA_G + b * LUMA_B;
    // contribution = knee_response(lum) / max(lum, eps); factor folds in gain.
    let contribution = knee_response(lum) / max(lum, MIN_DENOM);
    let factor = contribution * params.intensity;
    dst[base] = r * factor;
    dst[base + 1u] = g * factor;
    dst[base + 2u] = b * factor;
}
"#;

/// One bright-pass request: the batch of linear `HDR` `RGB` colors and the
/// bright-pass parameters.
///
/// Mirrors the `(colors, params)` pair the reference
/// [`threshold_batch`](prism_render_architecture::particle::bloom_threshold::BloomThresholdParams::threshold_batch)
/// consumes, carrying the `threshold`, `knee` and `intensity` in the reference
/// [`BloomThresholdParams`](prism_render_architecture::particle::bloom_threshold::BloomThresholdParams)
/// block. Derives only [`PartialEq`] (no `Eq`/`Hash`) because the colors hold
/// `f32` channels.
#[derive(Clone, Debug, PartialEq)]
pub struct BloomThresholdQuery {
    /// The source linear `HDR` `RGB` colors to run the bright-pass over.
    pub colors: Vec<[f32; 3]>,
    /// The bright-pass parameters: `threshold`, `knee` and `intensity`.
    pub params: BloomThresholdParams,
}

/// Uniform parameters for the bright-pass dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`BLOOM_THRESHOLD_WGSL`]: the three bright-pass scalars
/// and the color `count` — `16` bytes, a single `vec4` slot at the `std140`
/// uniform offsets the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// `Luminance` threshold where a pixel starts contributing to bloom.
    threshold: f32,
    /// Half-width of the soft `knee` transition band around the threshold.
    knee: f32,
    /// Global gain applied to the extracted bloom source color.
    intensity: f32,
    /// Number of input colors (one thread each).
    count: u32,
}

impl GpuParams {
    /// Packs the bright-pass parameters and the color `count` for one dispatch.
    fn new(params: &BloomThresholdParams, count: usize) -> GpuParams {
        GpuParams {
            threshold: params.threshold,
            knee: params.knee,
            intensity: params.intensity,
            count: count as u32,
        }
    }
}

/// A compiled, reusable `bloom` bright-pass pipeline.
pub struct GpuBloomThreshold {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBloomThreshold {
    /// Compiles the bright-pass kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBloomThreshold {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bloom_threshold"),
            source: ShaderSource::Wgsl(BLOOM_THRESHOLD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bloom_threshold_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bloom_threshold_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bloom_threshold_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("threshold"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBloomThreshold {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the bright-pass over `query.colors`, returning the thresholded bloom
    /// source colors in input order.
    ///
    /// The result equals
    /// [`query.params.threshold_batch(&query.colors)`](prism_render_architecture::particle::bloom_threshold::BloomThresholdParams::threshold_batch)
    /// to within the tolerance documented on this module. An empty color batch
    /// returns an empty vector and issues no dispatch (a storage buffer cannot
    /// be zero-sized), exactly as the reference returns an empty vector.
    /// Otherwise a single one-dimensional dispatch covers the flat color array,
    /// one thread per color.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &BloomThresholdQuery) -> Vec<[f32; 3]> {
        let colors = &query.colors;
        if colors.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let flat = flatten_colors(colors);
        let src = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bloom_threshold_source"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let out_bytes = (colors.len() * CHANNELS * size_of::<f32>()) as u64;
        let dst = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bloom_threshold_dst"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams::new(&query.params, colors.len());
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bloom_threshold_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bloom_threshold_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: src.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: dst.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bloom_threshold_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bloom_threshold_encoder"),
        });
        {
            let groups = (colors.len() as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bloom_threshold_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per input color, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&dst, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        let mut result: Vec<[f32; 3]> = Vec::with_capacity(colors.len());
        for chunk in out.chunks_exact(CHANNELS) {
            result.push([chunk[0], chunk[1], chunk[2]]);
        }
        debug_assert_eq!(result.len(), colors.len());
        result
    }
}

/// Flattens a slice of `RGB` colors into the row-major, 3-per-color `f32`
/// layout the device storage buffer expects.
fn flatten_colors(colors: &[[f32; 3]]) -> Vec<f32> {
    let mut data = Vec::with_capacity(colors.len() * CHANNELS);
    for color in colors {
        data.push(color[0]);
        data.push(color[1]);
        data.push(color[2]);
    }
    data
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
