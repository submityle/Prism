//! `wgpu` compute twin of the two per-element pure maps in the `histogram`
//! equalization contract
//! ([`histogram_equalize`](prism_render_architecture::particle::histogram_equalize),
//! particle design §16-§21).
//!
//! The `CPU` golden
//! [`histogram_equalize`](prism_render_architecture::particle::histogram_equalize)
//! owns the whole equalization pipeline: bin the samples into a `histogram`,
//! prefix-sum it into a cumulative distribution function (`CDF`), normalize that
//! `CDF` into a lookup table (`LUT`), and finally read the `LUT` per sample.
//! Only the two embarrassingly parallel, per-element *pure maps* at the ends of
//! that pipeline are twinned here, one thread per element:
//!
//! * the `CDF`-entry to `LUT`-level map inside
//!   [`equalization_lut`](prism_render_architecture::particle::histogram_equalize::equalization_lut)
//!   — subtract `cdf_min`, divide by the saturated denominator, and (when more
//!   than one output level is requested) snap to the nearest of `out_levels`
//!   evenly spaced levels with a `floor(x + 0.5)` round;
//! * the sample to equalized-value map inside
//!   [`apply_equalization`](prism_render_architecture::particle::histogram_equalize::apply_equalization)
//!   — bin the sample into `lut.len()` buckets, read the normalized level, and
//!   expand it back across `[range_min, range_max]`.
//!
//! # What is not twinned
//!
//! The two *reductions* in the middle of the pipeline are deliberately left on
//! the host: the atomic `histogram` scatter
//! ([`build_histogram`](prism_render_architecture::particle::histogram_equalize::build_histogram))
//! and the sequential prefix scan
//! ([`cumulative_distribution`](prism_render_architecture::particle::histogram_equalize::cumulative_distribution)),
//! together with the `CLAHE` clip-and-redistribute pass
//! ([`clip_histogram`](prism_render_architecture::particle::histogram_equalize::clip_histogram)).
//! The host computes the `CDF`, the `cdf_min` scalar and the full `LUT` with the
//! golden reductions and feeds them in as plain inputs; the kernel only
//! reproduces the two per-element maps above.
//!
//! # Correctness model
//!
//! The integer stages — the saturated subtractions, the quantized level index
//! `floor(raw * (out_levels - 1) + 0.5)`, and the floored bin index — are exact
//! `u32` arithmetic, so for inputs clear of a `floor` tie the `CPU` and `GPU`
//! agree bit for bit on them. The surrounding normalized levels and equalized
//! values thread through a guarded divide, a multiply and an add, so they are
//! not bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on the continuous `f32` outputs,
//! tight enough to catch a genuinely wrong port (a dropped `cdf_min`, a swapped
//! denominator, a wrong clamp) yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A flat or empty distribution collapses the saturated denominator
//! `total - cdf_min` to zero; the level map then returns `0.0`, matching the
//! reference all-zero `LUT`. An empty `LUT` makes the apply map a pass-through
//! that returns the sample unchanged, and a zero-width range
//! (`range_max - range_min <= CMP_EPS`) returns `range_min`. An empty query
//! batch short-circuits on the host with no dispatch, since a storage buffer
//! cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `floor`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round` (the
//! reference's round is reproduced as `floor(x + 0.5)`), no `sqrt` and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. There is no loop: each thread performs a fixed, bounded sequence of
//! arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`histogram_equalize`](prism_render_architecture::particle::histogram_equalize);
//! no third-party engine source or derived code.
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

/// Query tag selecting the `CDF`-entry to `LUT`-level map.
const TAG_LUT_LEVEL: u32 = 0;

/// Query tag selecting the sample to equalized-value map.
const TAG_APPLY: u32 = 1;

/// The portable core-`WGSL` histogram-equalization kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `equalize`
/// dispatches on a per-query `tag` to one of the two per-element maps and
/// mirrors the `CPU` golden
/// [`histogram_equalize`](prism_render_architecture::particle::histogram_equalize)
/// branch for branch; see the module documentation for the algorithm.
const HISTOGRAM_EQUALIZE_WGSL: &str = r#"
// Histogram-equalization twin: one thread per query reproduces either the
// CDF-entry -> LUT-level map (tag 0) or the sample -> equalized-value map
// (tag 1). It mirrors the CPU golden `particle::histogram_equalize` branch for
// branch, uses only the portable core-WGSL subset (min/max/clamp/floor and
// + - * / plus unsigned index math), needs no round (reproduced as
// floor(x + 0.5)), no sqrt and no transcendental call and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12. There is no loop,
// so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::histogram_equalize；无第三方
// 引擎源码或衍生代码。

// Magnitude at or below which a value span is treated as zero-width. Matches
// the reference `CMP_EPS`; the compare rule used instead of an f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

const TAG_LUT_LEVEL: u32 = 0u;
const TAG_APPLY: u32 = 1u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    // Length of the equalization LUT; zero means an empty LUT (apply passes the
    // sample through unchanged).
    lut_len: u32,
    pad0: u32,
    pad1: u32,
}

struct Query {
    // Tag selecting the map: TAG_LUT_LEVEL or TAG_APPLY.
    tag: u32,
    // LUT-level inputs (tag 0): one CDF entry, the smallest non-zero CDF value,
    // the saturated sample total, and the requested number of output levels.
    cdf: u32,
    cdf_min: u32,
    total: u32,
    out_levels: u32,
    // Apply inputs (tag 1): the sample and the working value range.
    in_sample: f32,
    range_min: f32,
    range_max: f32,
}

struct Result {
    // The LUT level (tag 0) or the equalized sample (tag 1).
    value: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read> lut: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;

// CDF-entry -> LUT-level map, mirroring the per-entry body of the reference
// `equalization_lut`. `total - cdf_min` and `cdf - cdf_min` use `min` to
// reproduce the reference saturating subtraction; a zero denominator yields 0.
fn lut_level(value: u32, cdf_min: u32, total: u32, out_levels: u32) -> f32 {
    let denom = total - min(total, cdf_min);
    if (denom == 0u) {
        return 0.0;
    }
    let denom_f = f32(denom);
    let numer = value - min(value, cdf_min);
    let raw = clamp(f32(numer) / denom_f, 0.0, 1.0);
    if (out_levels > 1u) {
        let max_level_f = f32(out_levels - 1u);
        // Reference `round` reproduced as floor(x + 0.5), then clamped to the
        // valid level range before normalizing back into [0, 1].
        let level = clamp(floor(raw * max_level_f + 0.5), 0.0, max_level_f);
        return level / max_level_f;
    }
    return raw;
}

// Places a sample into one of `bins` uniform buckets spanning the range,
// mirroring the reference `bin_index`.
fn bin_index(s: f32, bins: u32, range_min: f32, range_max: f32) -> u32 {
    let last = bins - 1u;
    let span = range_max - range_min;
    if (span <= CMP_EPS) {
        return 0u;
    }
    if (s <= range_min) {
        return 0u;
    }
    if (s >= range_max) {
        return last;
    }
    let normalized = (s - range_min) / span;
    let scaled = normalized * f32(bins);
    let idx = u32(floor(scaled));
    return min(idx, last);
}

// Sample -> equalized-value map, mirroring the reference `apply_equalization`.
// An empty LUT returns the sample unchanged and a zero-width range returns
// `range_min`.
fn apply_sample(s: f32, range_min: f32, range_max: f32) -> f32 {
    if (params.lut_len == 0u) {
        return s;
    }
    let span = range_max - range_min;
    if (span <= CMP_EPS) {
        return range_min;
    }
    let idx = bin_index(s, params.lut_len, range_min, range_max);
    let normalized = lut[idx];
    return range_min + normalized * span;
}

@compute @workgroup_size(64)
fn equalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var value: f32 = 0.0;
    if (q.tag == TAG_LUT_LEVEL) {
        value = lut_level(q.cdf, q.cdf_min, q.total, q.out_levels);
    } else {
        value = apply_sample(q.in_sample, q.range_min, q.range_max);
    }

    var out: Result;
    out.value = value;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count and the `LUT` length
/// plus two pad words to fill a `16`-byte, `std140`-aligned uniform struct
/// matching `Params` in [`HISTOGRAM_EQUALIZE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Length of the equalization `LUT`; `0` means an empty `LUT`.
    lut_len: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// All lanes are scalar `u32` / `f32`, so the struct needs no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Tag selecting the map: [`TAG_LUT_LEVEL`] or [`TAG_APPLY`].
    tag: u32,
    /// One `CDF` entry (`LUT`-level map).
    cdf: u32,
    /// Smallest non-zero `CDF` value (`LUT`-level map).
    cdf_min: u32,
    /// Saturated sample total (`LUT`-level map).
    total: u32,
    /// Requested number of output levels (`LUT`-level map).
    out_levels: u32,
    /// Sample to equalize (apply map).
    in_sample: f32,
    /// Lower bound of the working value range (apply map).
    range_min: f32,
    /// Upper bound of the working value range (apply map).
    range_max: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// The `LUT` level or the equalized sample.
    value: f32,
}

/// One query for the histogram-equalization twin: either a single `CDF` entry
/// to map to a `LUT` level, or a single sample to map through a `LUT`.
///
/// The host precomputes the `CDF`, the `cdf_min` scalar and the `LUT` with the
/// golden reductions; this twin reproduces only the two per-element maps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HistogramEqualizeQuery {
    /// Maps one `CDF` entry to its normalized `LUT` level, mirroring the
    /// per-entry body of
    /// [`equalization_lut`](prism_render_architecture::particle::histogram_equalize::equalization_lut).
    LutLevel {
        /// The `CDF` entry to normalize.
        cdf: u32,
        /// The smallest non-zero `CDF` value, subtracted before normalizing.
        cdf_min: u32,
        /// The saturated sample total (the `CDF`'s last entry).
        total: u32,
        /// The number of discrete output levels to quantize onto.
        out_levels: u32,
    },
    /// Maps one sample through the `LUT` back into `[range_min, range_max]`,
    /// mirroring
    /// [`apply_equalization`](prism_render_architecture::particle::histogram_equalize::apply_equalization).
    Apply {
        /// The sample to equalize.
        sample: f32,
        /// Lower bound of the working value range.
        range_min: f32,
        /// Upper bound of the working value range.
        range_max: f32,
    },
}

/// One resolved answer for a single query: the normalized `LUT` level for a
/// [`HistogramEqualizeQuery::LutLevel`] or the equalized sample for a
/// [`HistogramEqualizeQuery::Apply`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistogramEqualizeResult {
    /// The `LUT` level (in `[0, 1]`) or the equalized sample (in the query's
    /// range).
    pub value: f32,
}

/// Encodes one [`HistogramEqualizeQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &HistogramEqualizeQuery) -> GpuQuery {
    match *q {
        HistogramEqualizeQuery::LutLevel {
            cdf,
            cdf_min,
            total,
            out_levels,
        } => GpuQuery {
            tag: TAG_LUT_LEVEL,
            cdf,
            cdf_min,
            total,
            out_levels,
            in_sample: 0.0,
            range_min: 0.0,
            range_max: 0.0,
        },
        HistogramEqualizeQuery::Apply {
            sample,
            range_min,
            range_max,
        } => GpuQuery {
            tag: TAG_APPLY,
            cdf: 0,
            cdf_min: 0,
            total: 0,
            out_levels: 0,
            in_sample: sample,
            range_min,
            range_max,
        },
    }
}

/// Decodes one packed [`GpuResult`] into the public [`HistogramEqualizeResult`].
fn decode_result(raw: &GpuResult) -> HistogramEqualizeResult {
    HistogramEqualizeResult { value: raw.value }
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

/// A compiled, reusable histogram-equalization compute pipeline, twinning the
/// two per-element pure maps of the `CPU` golden
/// [`histogram_equalize`](prism_render_architecture::particle::histogram_equalize).
pub struct GpuHistogramEqualize {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHistogramEqualize {
    /// Compiles the histogram-equalization kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHistogramEqualize {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_histogram_equalize"),
            source: ShaderSource::Wgsl(HISTOGRAM_EQUALIZE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_histogram_equalize_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_histogram_equalize_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_histogram_equalize_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("equalize"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHistogramEqualize {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` against the equalization `lut` and
    /// returns one [`HistogramEqualizeResult`] per input, in order.
    ///
    /// The `lut` is shared across the batch: [`HistogramEqualizeQuery::Apply`]
    /// queries read it, while [`HistogramEqualizeQuery::LutLevel`] queries
    /// ignore it. An empty `lut` makes every apply query a pass-through, exactly
    /// as the reference does. The integer stages match the reference exactly;
    /// the continuous outputs match to within the tolerance documented on this
    /// module. An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        lut: &[f32],
        queries: &[HistogramEqualizeQuery],
    ) -> Vec<HistogramEqualizeResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            lut_len: lut.len() as u32,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_histogram_equalize_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_histogram_equalize_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        // A storage binding may not be zero-sized, so an empty LUT is padded to a
        // single element the kernel never reads (lut_len stays 0).
        let lut_storage: Vec<f32> = if lut.is_empty() {
            alloc_single_zero()
        } else {
            lut.to_vec()
        };
        let lut_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_histogram_equalize_lut"),
            contents: bytemuck::cast_slice(&lut_storage),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_histogram_equalize_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_histogram_equalize_bind_group"),
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
                    resource: lut_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_histogram_equalize_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_histogram_equalize_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_histogram_equalize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}

/// Returns a single-element `f32` buffer used to back an empty `LUT`, since a
/// `WebGPU` storage binding may not be zero-sized.
fn alloc_single_zero() -> Vec<f32> {
    vec![0.0]
}
