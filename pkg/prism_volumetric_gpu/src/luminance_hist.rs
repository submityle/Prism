//! `wgpu` compute twin of the `luminance` `histogram` builder
//! ([`build_histogram`](prism_render_architecture::particle::luminance_hist::build_histogram),
//! design sections 16-21, "自动曝光直方图").
//!
//! Automatic exposure drives from a *`luminance` `histogram`*: each sample's
//! `luminance` is placed on a `log2`-`EV` axis, floored into one of `bins`
//! buckets across `[min_ev, max_ev]`, and the per-bucket counts feed a trimmed
//! mean. The `CPU` golden
//! [`build_histogram`](prism_render_architecture::particle::luminance_hist::build_histogram)
//! owns that math; [`GpuLuminanceHist`] is the on-device twin that runs one
//! thread per *sample* and `atomicAdd`s each sample into its bucket, so a
//! passing real-device parity test is direct evidence the ported kernel bins
//! samples into the same buckets the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! The bucketing contract is reproduced exactly. The axis placement uses the
//! golden's hand-rolled
//! [`approx_log2`](prism_render_architecture::particle::luminance_hist::approx_log2):
//! a bit-pattern `log2` that reads the biased exponent field for the integer
//! `EV` and closes the mantissa fraction with the quadratic
//! `log2(1 + f) ≈ f + LOG2_CORRECTION * f * (1 - f)`. The `WGSL` reproduces that
//! bit-for-bit — `bitcast<u32>`, a `>> 23` field extract, a `& 0x007fffff`
//! mantissa mask, an integer-to-`f32` divide by `2^23` and the same quadratic —
//! and it **never calls the `WGSL` built-in `log2`**, which would use a
//! different polynomial and desync the bucketing. Powers of two (mantissa
//! fraction `0`) therefore map to the identical integer `EV` on both paths,
//! bit-exactly. Non-positive and subnormal inputs return the golden's
//! `NEG_EV_FLOOR` sentinel so they bin as "darkest". After the axis placement
//! the twin reproduces the same normalize-against-`[min_ev, max_ev]`,
//! `floor`-into-`bins` and clamp-to-`[0, bins)` the reference's
//! [`bin_index`](prism_render_architecture::particle::luminance_hist::bin_index)
//! applies.
//!
//! # `atomicAdd` versus `saturating_add`
//!
//! The `CPU` golden accumulates with `saturating_add` so a pathologically large
//! input can never wrap a bucket count; the `WGSL` kernel accumulates with
//! `atomicAdd`, which wraps on `u32` overflow. This is a deliberate, documented
//! difference: it is observable only when a single bucket would exceed `2^32`
//! samples, far above any real-device parity fixture (and above any plausible
//! frame's pixel count), so the two accumulators are equivalent at test scale.
//! The parity test keeps every bucket count well under `u32::MAX`, so the wrap
//! and the saturate never diverge.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `bitcast`, integer
//! shifts and masks, `min`, `max`, `clamp`, `floor`, `+ - * /` and
//! `atomicAdd` — with no `log2`, `exp2`, `pow`, `sin`, `cos` or optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! A histogram bucket count is a `u32` integer, so the parity test asserts
//! **exact per-bucket equality**, not a float tolerance — precisely the check
//! that catches an `approx_log2` ported one bit wrong (an off-by-one bucket).
//! The one place `CPU` and `GPU` may differ is a sample whose `EV` sits within
//! a `ULP` of a bucket boundary, where a legal fused multiply-add in the
//! mantissa quadratic can floor it to the neighbouring bucket; the parity
//! fixtures therefore keep every sample's `EV` well away from a boundary (or
//! use exact powers of two, for which the quadratic term is identically zero),
//! so the exact-equality check stays meaningful.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `luminance`-`histogram` auto-exposure (`Unreal`
//! eye-adaptation / `Frostbite` auto-exposure structure) plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::luminance_hist::LumHistConfig;
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
/// that divides evenly across `Metal`, `Vulkan` and `DX12`.
const WORKGROUP_SIZE: u32 = 64;

/// Core-`WGSL` histogram kernel, inlined so this twin lives entirely in the two
/// twin files. Mirrors
/// [`build_histogram`](prism_render_architecture::particle::luminance_hist::build_histogram)
/// / [`bin_index`](prism_render_architecture::particle::luminance_hist::bin_index)
/// / [`approx_log2`](prism_render_architecture::particle::luminance_hist::approx_log2);
/// the constants are copied verbatim from the golden module.
const SHADER_SOURCE: &str = r#"
// Luminance-histogram twin: bins each luminance into one of `bins` buckets on a
// log2-EV axis and atomicAdds it, mirroring the CPU goldens `build_histogram`,
// `bin_index` and `approx_log2` in
// `prism_render_architecture::particle::luminance_hist`.
//
// `approx_log2` is reproduced bit-for-bit from the f32 field layout — never the
// WGSL built-in `log2` — so powers of two map to the identical integer EV and
// the bucketing matches exactly. Accumulation uses `atomicAdd` (wraps on
// overflow) where the CPU uses `saturating_add`; equivalent below 2^32 counts.
//
// Provenance: standard luminance-histogram auto-exposure; no third-party engine
// source or derived code.

// Quadratic correction coefficient for the log2 mantissa term (ln(2) / 2).
// Copied verbatim from the golden `LOG2_CORRECTION`.
const LOG2_CORRECTION: f32 = 0.3465736;
// Sentinel EV for non-positive / subnormal inputs (golden `NEG_EV_FLOOR`).
const NEG_EV_FLOOR: f32 = -1000.0;
// 2^23, the f32 mantissa-field width as a divisor (golden `MANTISSA_SCALE`).
const MANTISSA_SCALE: f32 = 8388608.0;
// Degenerate-span guard (golden `EPS`).
const EPS: f32 = 1.0e-6;

struct Params {
    count: u32,
    bins: u32,
    min_ev: f32,
    max_ev: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> luminances: array<f32>;
@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;

// Bit-pattern log2, reproducing the golden `approx_log2` exactly: the biased
// exponent field gives the integer EV and the mantissa fraction is closed with
// the quadratic `log2(1 + f) ~= f + LOG2_CORRECTION * f * (1 - f)`.
fn approx_log2(x: f32) -> f32 {
    if (x <= 0.0) {
        return NEG_EV_FLOOR;
    }
    let bits = bitcast<u32>(x);
    let exp_field = i32((bits >> 23u) & 0xffu);
    if (exp_field == 0) {
        // Subnormal: below the smallest normal luminance we ever bin.
        return NEG_EV_FLOOR;
    }
    let mantissa = bits & 0x007fffffu;
    let frac = f32(mantissa) / MANTISSA_SCALE;
    let log_mant = frac + LOG2_CORRECTION * frac * (1.0 - frac);
    let exponent = f32(exp_field - 127);
    return exponent + log_mant;
}

// Maps a luminance onto its histogram bucket in [0, bins), reproducing the
// golden `bin_index`.
fn bin_index(lum: f32) -> u32 {
    let bin_count = max(params.bins, 1u);
    let span = params.max_ev - params.min_ev;
    if (span <= EPS) {
        return 0u;
    }
    let ev = approx_log2(lum);
    let t = clamp((ev - params.min_ev) / span, 0.0, 1.0);
    let bins_f = f32(bin_count);
    let scaled = floor(t * bins_f);
    let raw = u32(scaled);
    let last = bin_count - 1u;
    if (raw > last) {
        return last;
    }
    return raw;
}

@compute @workgroup_size(64)
fn luminance_hist_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let bucket = bin_index(luminances[idx]);
    atomicAdd(&hist[bucket], 1u);
}
"#;

/// One histogram request: the batch of `luminance` samples and the
/// [`LumHistConfig`] describing the `EV` axis.
///
/// The returned histogram has length
/// [`LumHistConfig::effective_bins`](prism_render_architecture::particle::luminance_hist::LumHistConfig::effective_bins)
/// and each bucket counts the samples whose `EV` fell in it, matching
/// [`build_histogram`](prism_render_architecture::particle::luminance_hist::build_histogram).
#[derive(Clone, Debug, PartialEq)]
pub struct LuminanceHistQuery {
    /// The `luminance` samples to bin, in any order (binning is commutative).
    pub luminances: Vec<f32>,
    /// The `EV`-axis configuration (range, bin count, trim percentiles).
    pub config: LumHistConfig,
}

/// Uniform parameters for one dispatch. `16`-byte `repr(C)` matching `Params`
/// in the inlined shader: the sample count, the effective bin count and the
/// `EV` range endpoints.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    count: u32,
    bins: u32,
    min_ev: f32,
    max_ev: f32,
}

/// A compiled, reusable `luminance`-`histogram` pipeline.
pub struct GpuLuminanceHist {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuLuminanceHist {
    /// Compiles the `luminance`-`histogram` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuLuminanceHist {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_luminance_hist_shader"),
            source: ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_luminance_hist_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_luminance_hist_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_luminance_hist_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("luminance_hist_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuLuminanceHist {
            module,
            layout,
            pipeline,
        }
    }

    /// Builds the `luminance` `histogram` for `query` on-device.
    ///
    /// Returns a `Vec<u32>` of length
    /// [`LumHistConfig::effective_bins`](prism_render_architecture::particle::luminance_hist::LumHistConfig::effective_bins)
    /// whose entry `k` equals the count
    /// [`build_histogram`](prism_render_architecture::particle::luminance_hist::build_histogram)
    /// produces for bucket `k`, exactly (see the module-level correctness
    /// model). An empty sample batch issues **no dispatch** — a storage buffer
    /// may not be zero-sized, and an empty input simply yields the all-zero
    /// histogram of the correct length — so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &LuminanceHistQuery) -> Vec<u32> {
        let bins = query.config.effective_bins();
        let len = usize::try_from(bins).unwrap_or(usize::MAX);

        // Empty input: no sample to dispatch, and a storage buffer cannot be
        // zero-sized, so return the correctly sized zero histogram directly.
        if query.luminances.is_empty() {
            return alloc_zeros(len);
        }

        let device = ctx.device();

        let gpu_params = GpuParams {
            count: query.luminances.len() as u32,
            bins,
            min_ev: query.config.min_ev,
            max_ev: query.config.max_ev,
        };

        let out_bytes = (len as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_luminance_hist_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let luminances_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_luminance_hist_luminances"),
            contents: bytemuck::cast_slice(&query.luminances),
            usage: BufferUsages::STORAGE,
        });
        // Zero-initialized explicitly so every `atomicAdd` accumulates from `0`
        // regardless of the backend's buffer-clearing policy.
        let zeros = alloc_zeros(len);
        let hist_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_luminance_hist_hist"),
            contents: bytemuck::cast_slice(&zeros),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let hist_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_luminance_hist_hist_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_luminance_hist_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: luminances_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: hist_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_luminance_hist_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_luminance_hist_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per sample, flattened to a 1-D dispatch.
            let groups = (query.luminances.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&hist_buf, 0, &hist_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        hist_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = hist_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let counts = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        hist_stage.unmap();
        debug_assert_eq!(counts.len(), len);
        counts
    }
}

/// Allocates a zero-filled histogram of `len` buckets.
fn alloc_zeros(len: usize) -> Vec<u32> {
    vec![0u32; len]
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
