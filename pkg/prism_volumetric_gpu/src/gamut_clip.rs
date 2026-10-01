//! `wgpu` compute twin of the `sRGB` `gamut` clip strategies
//! ([`gamut_clip`](prism_render_architecture::particle::gamut_clip), design
//! section 16, "色域裁剪").
//!
//! The final particle color must land inside the displayable unit cube
//! `0..=1`, but shading, tone mapping and grading all produce channels that may
//! sit outside it. The `CPU` golden
//! [`gamut_clip`](prism_render_architecture::particle::gamut_clip) owns that
//! last-mile clip in three strategies; [`GpuGamutClip`] is the on-device twin
//! that runs one thread per color and reproduces the same mapped triple, so a
//! passing real-device parity test is direct evidence the ported kernels take
//! the same branches, divide the same denominators and clamp the same walls the
//! reference does, not merely that the shaders compile.
//!
//! # What is twinned
//!
//! Three per-color strategies share one storage layout and one dispatch shape,
//! selected by [`GamutClipMode`] into one of three entry points:
//!
//! 1. [`GamutClipMode::Naive`] mirrors
//!    [`clip_naive`](prism_render_architecture::particle::gamut_clip::clip_naive):
//!    an independent per-channel `clamp` into `0..=1`.
//! 2. [`GamutClipMode::PreserveLuma`] mirrors
//!    [`clip_preserve_luma`](prism_render_architecture::particle::gamut_clip::clip_preserve_luma):
//!    it desaturates toward the equal-`Rec. 709`-luma gray, choosing the
//!    largest blend fraction that re-enters the cube per channel, and falls back
//!    to the naive clip when the input luma itself is undisplayable.
//! 3. [`GamutClipMode::Soft`] mirrors
//!    [`soft_clip`](prism_render_architecture::particle::gamut_clip::soft_clip):
//!    a monotone rational knee roll-off of half-width `knee` near each wall.
//!
//! The `Rec. 709` luma weights, the `channel_limit` divide guards, the
//! in-`gamut` membership epsilon and the knee clamp are all reproduced exactly
//! so `CPU` and `GPU` evaluate the same closed form in the same order.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `+ - * /` and unsigned index math — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `sqrt`, `tan` or built-in `smoothstep`, and no optional device
//! feature, so they run unmodified on `Metal`, `Vulkan` and `DX12`. The only
//! divisions are the two `channel_limit` ratios and the knee roll-off ratio,
//! each guarded against a zero denominator exactly as the reference guards them.
//!
//! # Correctness model
//!
//! Each color is a fixed, non-reorderable sequence of comparisons, clamps,
//! weighted adds and at most one divide, so `CPU` and `GPU` evaluate the same
//! algebra. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a genuinely
//! wrong port (a dropped luma weight, a wrong knee sign, a missing wall clamp)
//! yet loose enough to admit a legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `sRGB` `gamut` clip (per-channel clamp, equal-luma
//! desaturation and a monotone rational soft knee) plus `wgpu` compute dispatch;
//! no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::gamut_clip::Rgb;
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
/// the sibling twins in this crate share; one thread maps to one color.
const WORKGROUP_SIZE: u32 = 64;

/// Which `gamut`-clip strategy a dispatch runs, selecting the kernel entry
/// point. Each variant mirrors the identically named `CPU` golden routine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GamutClipMode {
    /// Per-channel hard clamp into `0..=1`
    /// ([`clip_naive`](prism_render_architecture::particle::gamut_clip::clip_naive)).
    Naive,
    /// Equal-luma desaturation toward the neutral gray
    /// ([`clip_preserve_luma`](prism_render_architecture::particle::gamut_clip::clip_preserve_luma)).
    PreserveLuma,
    /// Monotone rational knee roll-off near each wall
    /// ([`soft_clip`](prism_render_architecture::particle::gamut_clip::soft_clip)).
    Soft,
}

/// One `gamut`-clip request: the colors to map, the strategy and the soft knee.
///
/// The `knee` field is consumed only by [`GamutClipMode::Soft`]; the other two
/// strategies ignore it, exactly as the reference routines do.
#[derive(Clone, Debug, PartialEq)]
pub struct GamutClipQuery {
    /// The linear `sRGB` colors to clip; channels may fall outside `0..=1`.
    pub colors: Vec<Rgb>,
    /// Which clip strategy to run.
    pub mode: GamutClipMode,
    /// Soft-knee half-width, clamped into `[1e-6, 0.5]` by the kernel; used only
    /// by [`GamutClipMode::Soft`].
    pub knee: f32,
}

/// One color as uploaded. `16`-byte `repr(C)` `std430` record matching
/// `array<vec4<f32>>` in [`GAMUT_CLIP_WGSL`]: the three channels fill the first
/// three scalar slots and the fourth is zero padding, mirroring the reference
/// `to_std430` `vec4` layout.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuColor {
    /// Red channel (linear `sRGB`).
    r: f32,
    /// Green channel (linear `sRGB`).
    g: f32,
    /// Blue channel (linear `sRGB`).
    b: f32,
    /// Zero padding so the record is a whole `vec4`.
    pad: f32,
}

/// Uniform parameters for one dispatch. `32`-byte `repr(C)` layout matching
/// `Params` in [`GAMUT_CLIP_WGSL`]: the color count, three pad words, the soft
/// `knee` and three more pad words, so `knee` sits at the `16`-byte offset the
/// shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of colors in the dispatch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Soft-knee half-width (used only by the soft entry point).
    knee: f32,
    /// Padding word.
    pad3: f32,
    /// Padding word.
    pad4: f32,
    /// Padding word.
    pad5: f32,
}

/// The portable core-`WGSL` `gamut`-clip kernels, embedded inline so the twin
/// ships as a single source file. The three entry points `clip_naive_main`,
/// `clip_preserve_luma_main` and `soft_clip_main` mirror the `CPU` golden
/// `particle::gamut_clip` routines line for line; see the module documentation
/// for the algorithm.
const GAMUT_CLIP_WGSL: &str = r#"
// sRGB gamut clip twin: one thread per color runs one of three strategies.
// `clip_naive_main` clamps each channel into 0..=1; `clip_preserve_luma_main`
// desaturates toward the equal-Rec.709-luma gray and falls back to the naive
// clip when the luma is undisplayable; `soft_clip_main` applies a monotone
// rational knee roll-off near each wall. They mirror the CPU golden
// `particle::gamut_clip`, use only the portable core-WGSL subset (min/max/clamp
// and + - * / plus unsigned index math) with no transcendental or smoothstep
// built-in, and take no optional feature, so they run unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: standard sRGB gamut clip (per-channel clamp, equal-luma
// desaturation, monotone rational soft knee); no third-party engine source or
// derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    knee: f32,
    pad3: f32,
    pad4: f32,
    pad5: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> colors_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> colors_out: array<vec4<f32>>;

// In-gamut membership tolerance, matching the reference `CMP_EPS`.
const CMP_EPS: f32 = 1e-6;
// Smallest knee half-width and divide guard, matching the reference `MIN_KNEE`.
const MIN_KNEE: f32 = 1e-6;
// Rec. 709 luma weights; they sum to one so an equal-luma gray shares the luma.
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

// Clamps a scalar into 0..=1 without branching on floating equality.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Rec. 709 relative luminance of a linear color.
fn luma_rec709(c: vec3<f32>) -> f32 {
    return LUMA_R * c.x + LUMA_G * c.y + LUMA_B * c.z;
}

// True when every channel lies within 0..=1 up to CMP_EPS.
fn is_in_gamut(c: vec3<f32>) -> bool {
    let lo = -CMP_EPS;
    let hi = 1.0 + CMP_EPS;
    return c.x >= lo && c.x <= hi && c.y >= lo && c.y <= hi && c.z >= lo && c.z <= hi;
}

// Per-channel hard clip into 0..=1.
fn clip_naive(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(clamp01(c.x), clamp01(c.y), clamp01(c.z));
}

// Largest fraction of the excursion `channel - gray` that keeps
// `gray + t * (channel - gray)` inside 0..=1, returning 1.0 when the channel is
// already in range. The divide is guarded exactly as the reference guards it.
fn channel_limit(channel: f32, gray: f32) -> f32 {
    let d = channel - gray;
    if (channel > 1.0) {
        if (d > MIN_KNEE) {
            return (1.0 - gray) / d;
        }
        return 0.0;
    }
    if (channel < 0.0) {
        if (d < -MIN_KNEE) {
            return (0.0 - gray) / d;
        }
        return 0.0;
    }
    return 1.0;
}

// Desaturates toward the equal-luma gray until the color re-enters 0..=1,
// falling back to the naive clip when the input luma is itself undisplayable.
fn clip_preserve_luma(c: vec3<f32>) -> vec3<f32> {
    if (is_in_gamut(c)) {
        return c;
    }
    let l = luma_rec709(c);
    if (l < -CMP_EPS || l > 1.0 + CMP_EPS) {
        return clip_naive(c);
    }
    let gray = clamp01(l);
    let tx = channel_limit(c.x, gray);
    let ty = channel_limit(c.y, gray);
    let tz = channel_limit(c.z, gray);
    let t = clamp(min(tx, min(ty, tz)), 0.0, 1.0);
    return vec3<f32>(
        clamp01(gray + t * (c.x - gray)),
        clamp01(gray + t * (c.y - gray)),
        clamp01(gray + t * (c.z - gray))
    );
}

// Monotone rational soft clip of one channel toward 0..=1; identity inside
// [knee, 1 - knee], a saturating rational roll-off outside it.
fn soft_clip_channel(x: f32, knee: f32) -> f32 {
    let k = clamp(knee, MIN_KNEE, 0.5);
    let upper = 1.0 - k;
    let lower = k;
    if (x > upper) {
        let e = x - upper;
        return upper + k * (e / (e + k));
    }
    if (x < lower) {
        let e = lower - x;
        return lower - k * (e / (e + k));
    }
    return x;
}

// Soft-clips all three channels with a smooth knee of half-width `knee`.
fn soft_clip(c: vec3<f32>, knee: f32) -> vec3<f32> {
    return vec3<f32>(
        soft_clip_channel(c.x, knee),
        soft_clip_channel(c.y, knee),
        soft_clip_channel(c.z, knee)
    );
}

@compute @workgroup_size(64)
fn clip_naive_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let c = colors_in[idx].xyz;
    colors_out[idx] = vec4<f32>(clip_naive(c), 0.0);
}

@compute @workgroup_size(64)
fn clip_preserve_luma_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let c = colors_in[idx].xyz;
    colors_out[idx] = vec4<f32>(clip_preserve_luma(c), 0.0);
}

@compute @workgroup_size(64)
fn soft_clip_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let c = colors_in[idx].xyz;
    colors_out[idx] = vec4<f32>(soft_clip(c, params.knee), 0.0);
}
"#;

/// A compiled, reusable `gamut`-clip pipeline trio (naive, preserve-luma, soft).
pub struct GpuGamutClip {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_naive: ComputePipeline,
    pipeline_preserve_luma: ComputePipeline,
    pipeline_soft: ComputePipeline,
}

impl GpuGamutClip {
    /// Compiles the three `gamut`-clip kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGamutClip {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gamut_clip"),
            source: ShaderSource::Wgsl(GAMUT_CLIP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gamut_clip_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gamut_clip_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry: &str, label: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipeline_naive = make(
            "clip_naive_main",
            "prism_volumetric_gamut_clip_naive_pipeline",
        );
        let pipeline_preserve_luma = make(
            "clip_preserve_luma_main",
            "prism_volumetric_gamut_clip_preserve_luma_pipeline",
        );
        let pipeline_soft = make(
            "soft_clip_main",
            "prism_volumetric_gamut_clip_soft_pipeline",
        );
        GpuGamutClip {
            module,
            layout,
            pipeline_naive,
            pipeline_preserve_luma,
            pipeline_soft,
        }
    }

    /// Clips every color in `query.colors` with the selected strategy, returning
    /// one mapped color per input in order.
    ///
    /// The returned color for input `c` equals the reference routine for
    /// `query.mode` applied to `c` (with `query.knee` for
    /// [`GamutClipMode::Soft`]) to within the tolerance documented on this
    /// module. An empty `colors` slice yields an empty result with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &GamutClipQuery) -> Vec<Rgb> {
        if query.colors.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_colors: Vec<GpuColor> = query
            .colors
            .iter()
            .map(|c| GpuColor {
                r: c.r,
                g: c.g,
                b: c.b,
                pad: 0.0,
            })
            .collect();

        let gpu_params = Params {
            count: query.colors.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            knee: query.knee,
            pad3: 0.0,
            pad4: 0.0,
            pad5: 0.0,
        };

        let elem = size_of::<GpuColor>() as u64;
        let out_bytes = (query.colors.len() as u64) * elem;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gamut_clip_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let colors_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gamut_clip_colors"),
            contents: bytemuck::cast_slice(&gpu_colors),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gamut_clip_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gamut_clip_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gamut_clip_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: colors_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline = match query.mode {
            GamutClipMode::Naive => &self.pipeline_naive,
            GamutClipMode::PreserveLuma => &self.pipeline_preserve_luma,
            GamutClipMode::Soft => &self.pipeline_soft,
        };

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_gamut_clip_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gamut_clip_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per color, flattened to a 1-D dispatch.
            let groups = (query.colors.len() as u32).div_ceil(WORKGROUP_SIZE);
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
        let packed = bytemuck::cast_slice::<u8, GpuColor>(&view).to_vec();
        drop(view);
        results_stage.unmap();

        let colors: Vec<Rgb> = packed.iter().map(|p| Rgb::new(p.r, p.g, p.b)).collect();
        debug_assert_eq!(colors.len(), query.colors.len());
        colors
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
