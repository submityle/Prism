//! `wgpu` compute twin of the weighted reservoir sampling (`WRS` / `RIS`)
//! primitive
//! ([`reservoir_sample`](prism_render_architecture::particle::reservoir_sample)).
//!
//! A *reservoir* holds a single running sample drawn from a stream of weighted
//! candidates: each candidate is folded in with
//! [`update`](prism_render_architecture::particle::reservoir_sample::Reservoir::update),
//! which adds `weight` to the running `w_sum`, increments the count `m`, and
//! replaces the held sample with probability `weight / w_sum`. The acceptance
//! test is the division-free `rand_u01 * w_sum < weight`, so a zero `w_sum`
//! never replaces. The `ReSTIR` extensions
//! [`merge`](prism_render_architecture::particle::reservoir_sample::Reservoir::merge)
//! (fuse two reservoirs, summing `m`) and
//! [`finalize_w`](prism_render_architecture::particle::reservoir_sample::Reservoir::finalize_w)
//! (the unbiased contribution weight `UCW`, `W = w_sum / (m * target_pdf)`,
//! clamped to zero when the `pdf` vanishes) layer on top of it.
//!
//! # What is twinned
//!
//! Only the *per-element* kernels are ported: one thread owns one candidate
//! stream (one reservoir) and folds its whole stream in sequence, so `N`
//! independent streams map to `N` parallel threads. The host-side serial
//! orchestration of a cross-stream tree reduction is deliberately **not**
//! moved to the `GPU`; a `merge` is twinned only in its per-element form, where
//! one thread fuses exactly two reservoirs.
//!
//! * [`GpuReservoirSample::sample_streams`] runs the full `WRS` loop over each
//!   stream and finalizes `W`, mirroring an
//!   [`empty`](prism_render_architecture::particle::reservoir_sample::Reservoir::empty)
//!   reservoir driven by a sequence of
//!   [`update`](prism_render_architecture::particle::reservoir_sample::Reservoir::update)
//!   calls then
//!   [`finalize_w`](prism_render_architecture::particle::reservoir_sample::Reservoir::finalize_w).
//! * [`GpuReservoirSample::merge_pairs`] fuses two reservoirs per thread and
//!   finalizes `W`, mirroring
//!   [`merge`](prism_render_architecture::particle::reservoir_sample::Reservoir::merge)
//!   then
//!   [`finalize_w`](prism_render_architecture::particle::reservoir_sample::Reservoir::finalize_w).
//!
//! # `RNG`
//!
//! Each thread drives its own acceptance decisions with a `splitmix32` integer
//! hash seeded per stream, reproducing the golden
//! [`Rng`](prism_render_architecture::particle::reservoir_sample::Rng) bit for
//! bit: the state advances by the golden constant each draw and the mixed word
//! is scaled by `1 / 2^24` from its top `24` bits, exactly as the reference
//! does. Because both sides draw the same words in the same order and fold the
//! same terms in the same order, the branch decisions coincide, so the held
//! sample index and the count match exactly.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — integer `+ - * >> ^`
//! with wrapping `u32` arithmetic, `f32` `+ * /` and `<` comparison, and the
//! exact `u32`-to-`f32` conversion of a `24`-bit word. There is no
//! transcendental call, no intrinsic, no optional device feature and no `u64`,
//! so they run unmodified on `Metal`, `Vulkan` and `DX12`. The stream loop is
//! bounded by the runtime candidate count supplied per thread.
//!
//! # Correctness model
//!
//! `WGSL` `u32` arithmetic wraps on overflow, matching the reference
//! `wrapping_add` / `wrapping_mul`, so the `RNG` word stream is bit-identical.
//! The held sample index (`y`) and the count (`m`) are therefore compared with
//! an exact `==`; the `f32` accumulators `w_sum` and `W` are compared with a
//! small tolerance because an independent `GPU` may fuse or round the `f32`
//! adds and the final divide differently.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::reservoir_sample`；无第三方引擎源码或衍生代码。
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
/// that divides evenly on every target backend.
const WORKGROUP_SIZE: u32 = 64;

/// The weighted reservoir sampling kernels, mirroring the `CPU` golden
/// [`reservoir_sample`](prism_render_architecture::particle::reservoir_sample)
/// field for field. One source hosts two per-element entry points: a full
/// stream sampler and a two-reservoir merge. The `splitmix32` `RNG` is shared
/// by both so acceptance decisions match the reference word for word.
const RESERVOIR_SAMPLE_WGSL: &str = r#"
// Weighted reservoir sampling twin: one thread per element (one reservoir).
// `sample_stream` folds a whole candidate stream in order then finalizes W;
// `merge_pair` fuses two reservoirs then finalizes W. The splitmix32 RNG
// matches the CPU golden `particle::reservoir_sample::Rng` bit for bit: state
// advances by the golden constant each draw and the mixed word is scaled by
// 1 / 2^24 from its top 24 bits. Pure wrapping u32 integer algebra plus f32
// + * / and <; no transcendental, no intrinsic, no u64, portable on Metal,
// Vulkan and DX12. The stream loop is bounded by the runtime candidate count.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::reservoir_sample；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of valid elements (streams or pairs); threads past this stop.
    count: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// A reservoir as packed for upload / readback: 16 bytes, naturally aligned.
struct Reservoir {
    sample: u32,
    w_sum: f32,
    m: u32,
    w: f32,
}

@group(0) @binding(0) var<uniform> params: Params;

// --- sample_stream bindings ---
// Per-stream RNG seed, flat-array offset, candidate count and target pdf, then
// the flattened candidate / weight streams and the per-stream output.
@group(0) @binding(1) var<storage, read> in_seed: array<u32>;
@group(0) @binding(2) var<storage, read> in_offset: array<u32>;
@group(0) @binding(3) var<storage, read> in_length: array<u32>;
@group(0) @binding(4) var<storage, read> in_target_pdf: array<f32>;
@group(0) @binding(5) var<storage, read> in_candidate: array<u32>;
@group(0) @binding(6) var<storage, read> in_weight: array<f32>;
@group(0) @binding(7) var<storage, read_write> dst_stream: array<Reservoir>;

// --- merge_pair bindings ---
// The two input reservoirs, the per-pair RNG seed and target pdf, and output.
@group(0) @binding(8) var<storage, read> merge_in_a: array<Reservoir>;
@group(0) @binding(9) var<storage, read> merge_in_b: array<Reservoir>;
@group(0) @binding(10) var<storage, read> merge_seed: array<u32>;
@group(0) @binding(11) var<storage, read> merge_target_pdf: array<f32>;
@group(0) @binding(12) var<storage, read_write> dst_merge: array<Reservoir>;

// Below this the target pdf is treated as zero, forcing W to zero rather than
// dividing by a vanishing density. Matches FINALIZE_PDF_EPS in the reference.
const FINALIZE_PDF_EPS: f32 = 1e-6;

// 1 / 2^24, the reciprocal that normalizes a 24-bit mantissa word into [0, 1).
const INV_2_POW_24: f32 = 1.0 / 16777216.0;

// The golden ratio increment splitmix32 adds to the state each draw.
const GOLDEN: u32 = 0x9E3779B9u;

// Advance the splitmix32 state by one draw: state = state + GOLDEN. Wrapping
// u32 addition matches the reference `wrapping_add`.
fn rng_advance(state: u32) -> u32 {
    return state + GOLDEN;
}

// Mix an advanced state into the output word, exactly as the reference folds
// the state after adding GOLDEN. Wrapping u32 multiply matches `wrapping_mul`.
fn rng_mix(state: u32) -> u32 {
    var z = state;
    z = (z ^ (z >> 16u)) * 0x21F0AAADu;
    z = (z ^ (z >> 15u)) * 0x735A2D97u;
    return z ^ (z >> 15u);
}

// Map the top 24 bits of the mixed word to [0, 1), matching the reference
// `next_u01`: a 24-bit word is exact in the f32 mantissa so this never rounds
// up to 1.0.
fn rng_u01(mixed: u32) -> f32 {
    let bits = mixed >> 8u;
    return f32(bits) * INV_2_POW_24;
}

// Compute the unbiased contribution weight W = w_sum / (m * target_pdf),
// forcing zero when target_pdf vanishes or the reservoir is empty. Mirrors the
// reference `finalize_w`.
fn finalize_w(r: ptr<function, Reservoir>, target_pdf: f32) {
    if (target_pdf <= FINALIZE_PDF_EPS || (*r).m == 0u) {
        (*r).w = 0.0;
        return;
    }
    let denom = f32((*r).m) * target_pdf;
    (*r).w = (*r).w_sum / denom;
}

@compute @workgroup_size(64)
fn sample_stream(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    var r: Reservoir;
    r.sample = 0u;
    r.w_sum = 0.0;
    r.m = 0u;
    r.w = 0.0;

    var state = in_seed[idx];
    let off = in_offset[idx];
    let len = in_length[idx];
    for (var j = 0u; j < len; j = j + 1u) {
        let candidate = in_candidate[off + j];
        let weight = in_weight[off + j];
        // Draw one uniform in [0, 1), advancing the RNG exactly as the golden
        // caller would before each update.
        state = rng_advance(state);
        let rand_u01 = rng_u01(rng_mix(state));
        // Reservoir::update, inlined: fold one weighted candidate.
        r.w_sum = r.w_sum + weight;
        r.m = r.m + 1u;
        if (rand_u01 * r.w_sum < weight) {
            r.sample = candidate;
        }
    }
    finalize_w(&r, in_target_pdf[idx]);
    dst_stream[idx] = r;
}

@compute @workgroup_size(64)
fn merge_pair(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    var r = merge_in_a[idx];
    let other = merge_in_b[idx];
    // One RNG draw drives the merge acceptance, seeded per pair.
    var state = merge_seed[idx];
    state = rng_advance(state);
    let rand_u01 = rng_u01(rng_mix(state));
    // Reservoir::merge, inlined: treat `other` as a single super-candidate.
    let combined_m = r.m + other.m;
    r.w_sum = r.w_sum + other.w_sum;
    if (rand_u01 * r.w_sum < other.w_sum) {
        r.sample = other.sample;
    }
    r.m = combined_m;
    finalize_w(&r, merge_target_pdf[idx]);
    dst_merge[idx] = r;
}
"#;

/// A reservoir value as uploaded or read back: the held sample index, the
/// running weight sum, the candidate count and the unbiased contribution
/// weight. Mirrors the golden
/// [`Reservoir`](prism_render_architecture::particle::reservoir_sample::Reservoir),
/// matching its field order and `16`-byte `repr(C)` so it packs directly into a
/// `std430` storage buffer with no padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ReservoirValue {
    /// The identifier of the currently held candidate (the sample `y`).
    pub sample: u32,
    /// The running sum of all candidate weights observed so far.
    pub w_sum: f32,
    /// The number of candidates folded into this reservoir.
    pub m: u32,
    /// The unbiased contribution weight `W` from `finalize_w`.
    pub w: f32,
}

impl ReservoirValue {
    /// Returns an empty reservoir: no sample, zero weight sum, zero count.
    #[must_use]
    pub const fn empty() -> ReservoirValue {
        ReservoirValue {
            sample: 0,
            w_sum: 0.0,
            m: 0,
            w: 0.0,
        }
    }
}

/// Uniform parameters for one dispatch: the element `count` plus padding to a
/// `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RESERVOIR_SAMPLE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid elements (streams or pairs) in this dispatch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable set of weighted reservoir sampling kernels: a full
/// stream sampler and a two-reservoir merge.
pub struct GpuReservoirSample {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    stream_layout: BindGroupLayout,
    merge_layout: BindGroupLayout,
    pipeline_sample_stream: ComputePipeline,
    pipeline_merge_pair: ComputePipeline,
}

impl GpuReservoirSample {
    /// Compiles the reservoir sampling kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuReservoirSample {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_reservoir_sample"),
            source: ShaderSource::Wgsl(RESERVOIR_SAMPLE_WGSL.into()),
        });
        let stream_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_reservoir_sample_stream_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: true }),
                buffer_entry(7, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let merge_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_reservoir_sample_merge_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(8, BufferBindingType::Storage { read_only: true }),
                buffer_entry(9, BufferBindingType::Storage { read_only: true }),
                buffer_entry(10, BufferBindingType::Storage { read_only: true }),
                buffer_entry(11, BufferBindingType::Storage { read_only: true }),
                buffer_entry(12, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let stream_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_reservoir_sample_stream_pipeline_layout"),
            bind_group_layouts: &[Some(&stream_layout)],
            immediate_size: 0,
        });
        let merge_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_reservoir_sample_merge_pipeline_layout"),
            bind_group_layouts: &[Some(&merge_layout)],
            immediate_size: 0,
        });
        let pipeline_sample_stream = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_reservoir_sample_stream_pipeline"),
            layout: Some(&stream_pipeline_layout),
            module: &module,
            entry_point: Some("sample_stream"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_merge_pair = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_reservoir_sample_merge_pipeline"),
            layout: Some(&merge_pipeline_layout),
            module: &module,
            entry_point: Some("merge_pair"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuReservoirSample {
            module,
            stream_layout,
            merge_layout,
            pipeline_sample_stream,
            pipeline_merge_pair,
        }
    }

    /// Runs weighted reservoir sampling over each candidate stream and
    /// finalizes the unbiased contribution weight `W`.
    ///
    /// One thread owns one stream. Thread `i` seeds its `splitmix32`
    /// [`Rng`](prism_render_architecture::particle::reservoir_sample::Rng) with
    /// `seeds[i]`, folds the `lengths[i]` candidates starting at
    /// `offsets[i]` in the flattened `candidates` / `weights` arrays (drawing
    /// one uniform per candidate), then finalizes `W` with `target_pdfs[i]`.
    /// This mirrors an
    /// [`empty`](prism_render_architecture::particle::reservoir_sample::Reservoir::empty)
    /// reservoir driven by a sequence of
    /// [`update`](prism_render_architecture::particle::reservoir_sample::Reservoir::update)
    /// calls followed by
    /// [`finalize_w`](prism_render_architecture::particle::reservoir_sample::Reservoir::finalize_w).
    ///
    /// Returns one reservoir per stream, in order. An empty stream (`length`
    /// `0`) yields the degenerate empty reservoir (`m == 0`, `w == 0`). An
    /// empty input returns an empty vector with no dispatch issued (a storage
    /// buffer cannot be zero-sized).
    ///
    /// # Panics
    ///
    /// Panics if `seeds`, `offsets`, `lengths` and `target_pdfs` differ in
    /// length, or if `candidates` and `weights` differ in length.
    #[must_use]
    pub fn sample_streams(
        &self,
        ctx: &GpuContext,
        seeds: &[u32],
        offsets: &[u32],
        lengths: &[u32],
        target_pdfs: &[f32],
        candidates: &[u32],
        weights: &[f32],
    ) -> Vec<ReservoirValue> {
        assert_eq!(
            seeds.len(),
            offsets.len(),
            "seeds and offsets must be the same length"
        );
        assert_eq!(
            seeds.len(),
            lengths.len(),
            "seeds and lengths must be the same length"
        );
        assert_eq!(
            seeds.len(),
            target_pdfs.len(),
            "seeds and target_pdfs must be the same length"
        );
        assert_eq!(
            candidates.len(),
            weights.len(),
            "candidates and weights must be the same length"
        );
        let count = seeds.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_reservoir_sample_stream_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let seed_buf = storage_buffer(device, "stream_seed", bytemuck::cast_slice(seeds));
        let offset_buf = storage_buffer(device, "stream_offset", bytemuck::cast_slice(offsets));
        let length_buf = storage_buffer(device, "stream_length", bytemuck::cast_slice(lengths));
        let target_buf = storage_buffer(device, "stream_target", bytemuck::cast_slice(target_pdfs));
        // Storage buffers cannot be zero-sized, so pad an all-empty-stream run
        // with one never-read element.
        let candidate_pad: Vec<u32> = pad_u32(candidates);
        let weight_pad: Vec<f32> = pad_f32(weights);
        let candidate_buf = storage_buffer(
            device,
            "stream_candidate",
            bytemuck::cast_slice(&candidate_pad),
        );
        let weight_buf = storage_buffer(device, "stream_weight", bytemuck::cast_slice(&weight_pad));

        let out_bytes = (count as u64) * (size_of::<ReservoirValue>() as u64);
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_reservoir_sample_stream_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_reservoir_sample_stream_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_reservoir_sample_stream_bind_group"),
            layout: &self.stream_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: seed_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: offset_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: length_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: target_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: candidate_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: weight_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_reservoir_sample_stream_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_reservoir_sample_stream_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_sample_stream);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per stream, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        read_reservoirs(ctx, &stage, count)
    }

    /// Fuses two reservoirs per thread and finalizes the unbiased contribution
    /// weight `W`.
    ///
    /// Thread `i` adopts `b[i]` into `a[i]` with probability
    /// `b.w_sum / (a.w_sum + b.w_sum)` using one `splitmix32` draw seeded with
    /// `seeds[i]`, sums the counts `m`, then finalizes `W` with
    /// `target_pdfs[i]`. This mirrors
    /// [`merge`](prism_render_architecture::particle::reservoir_sample::Reservoir::merge)
    /// followed by
    /// [`finalize_w`](prism_render_architecture::particle::reservoir_sample::Reservoir::finalize_w)
    /// on each pair.
    ///
    /// Returns one merged reservoir per pair, in order. An empty input returns
    /// an empty vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `a`, `b`, `seeds` and `target_pdfs` differ in length.
    #[must_use]
    pub fn merge_pairs(
        &self,
        ctx: &GpuContext,
        a: &[ReservoirValue],
        b: &[ReservoirValue],
        seeds: &[u32],
        target_pdfs: &[f32],
    ) -> Vec<ReservoirValue> {
        assert_eq!(a.len(), b.len(), "a and b must be the same length");
        assert_eq!(a.len(), seeds.len(), "a and seeds must be the same length");
        assert_eq!(
            a.len(),
            target_pdfs.len(),
            "a and target_pdfs must be the same length"
        );
        let count = a.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_reservoir_sample_merge_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let a_buf = storage_buffer(device, "merge_a", bytemuck::cast_slice(a));
        let b_buf = storage_buffer(device, "merge_b", bytemuck::cast_slice(b));
        let seed_buf = storage_buffer(device, "merge_seed", bytemuck::cast_slice(seeds));
        let target_buf = storage_buffer(device, "merge_target", bytemuck::cast_slice(target_pdfs));

        let out_bytes = (count as u64) * (size_of::<ReservoirValue>() as u64);
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_reservoir_sample_merge_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_reservoir_sample_merge_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_reservoir_sample_merge_bind_group"),
            layout: &self.merge_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: a_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: b_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: seed_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: target_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_reservoir_sample_merge_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_reservoir_sample_merge_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_merge_pair);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pair, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        read_reservoirs(ctx, &stage, count)
    }
}

/// Returns a copy of `src` guaranteed to be at least one element long, so an
/// otherwise empty storage buffer never has a zero size.
fn pad_u32(src: &[u32]) -> Vec<u32> {
    if src.is_empty() {
        return vec![0];
    }
    src.to_vec()
}

/// Returns a copy of `src` guaranteed to be at least one element long, so an
/// otherwise empty storage buffer never has a zero size.
fn pad_f32(src: &[f32]) -> Vec<f32> {
    if src.is_empty() {
        return vec![0.0];
    }
    src.to_vec()
}

/// Creates a read-only / read-write storage buffer initialized with `contents`.
fn storage_buffer(device: &wgpu::Device, name: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(name),
        contents,
        usage: BufferUsages::STORAGE,
    })
}

/// Maps `stage`, reads back `count` reservoirs and unmaps.
fn read_reservoirs(ctx: &GpuContext, stage: &wgpu::Buffer, count: usize) -> Vec<ReservoirValue> {
    stage.slice(..).map_async(MapMode::Read, |_| {});
    ctx.wait();
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let result = bytemuck::cast_slice::<u8, ReservoirValue>(&view).to_vec();
    drop(view);
    stage.unmap();
    debug_assert_eq!(result.len(), count);
    result
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
