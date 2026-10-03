//! `wgpu` compute twin of the transparent-draw binner from the transparency
//! routing module
//! ([`routing`](prism_render_architecture::transparency::routing)).
//!
//! A frame's transparent draws cannot be composited with one universal rule:
//! sortable glass wants back-to-front alpha blending, overlapping foliage needs
//! an order-independent (`OIT`) resolve, water and hair have bespoke passes, and
//! participating media are marched as volumes. The golden
//! [`bin_transparent_draws`](prism_render_architecture::transparency::routing::bin_transparent_draws)
//! walks a frame's draw list, routes each draw through
//! [`select_transparency_path`](prism_render_architecture::transparency::routing::select_transparency_path),
//! and fans the draw indices into one bucket per
//! [`TransparencyPath`](prism_render_architecture::transparency::TransparencyPath),
//! preserving first-seen order inside every bucket. A draw whose slot has no
//! matching surface is skipped, so a stale draw list cannot crash submission.
//!
//! [`GpuTransparencyBinDraws`] ports that pass exactly: one thread owns one
//! query (a whole draw list plus its parallel surface descriptors), serially
//! scans the slots, reproduces the `select_transparency_path` `match` ladder to
//! pick a path discriminant, and appends the draw index into the matching
//! bucket. Because a single thread performs the sequential appends for its
//! query, intra-bucket order is reproduced bit-for-bit. A passing real-device
//! parity test is direct evidence the ported kernel reproduces the reference
//! binning, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one query carrying up to [`MAX_DRAWS`] draw indices plus the parallel
//! per-slot content class and flags, the kernel reproduces:
//! - the slot skip when `slot >= surface_count` (the golden `surfaces.get(slot)`
//!   returning `None`);
//! - `select_transparency_path` per surviving slot: `Water` to
//!   `SingleLayerWater`, `Hair` to `HairVisibility`, `Volume` to `Volumetric`,
//!   `Glass` to `LayeredGlass` when `layer_count > 1` else `Sorted`, and
//!   `General` to `MomentOit` when it is order-independent, high-fidelity and
//!   the backend supports moment `OIT`, to `WeightedOit` when order-independent
//!   otherwise, and to `Sorted` when depth-sortable; and
//! - the per-bucket append preserving first-seen order.
//!
//! # Discriminant encoding
//!
//! The twin encodes [`TransparencyPath`](prism_render_architecture::transparency::TransparencyPath)
//! exactly as the golden enum's `as u32`, which follows the declaration order:
//! `Sorted = 0`, `WeightedOit = 1`, `MomentOit = 2`, `LayeredGlass = 3`,
//! `SingleLayerWater = 4`, `HairVisibility = 5`, `Volumetric = 6`. The content
//! class `TransparentKind` is encoded likewise: `General = 0`, `Water = 1`,
//! `Hair = 2`, `Glass = 3`, `Volume = 4`. The parity test pins the kernel's
//! codes against the golden enums' `as u32`, so the two share one source of
//! truth.
//!
//! # Correctness model
//!
//! Every step is integer comparison, selection and indexing with no
//! floating-point arithmetic, so the `GPU` and `CPU` produce bit-identical
//! bucket counts and index sequences; parity is asserted with exact equality.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `u32` comparisons,
//! branches and array indexing — with no transcendental call and no `64`-bit
//! integers or floats. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::transparency::routing`；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup. One thread owns one query and serially
/// scans that query's slots, so a group holds a single invocation.
const WORKGROUP_SIZE: u32 = 1;

/// Number of transparency paths, matching the golden
/// [`TransparencyPath`](prism_render_architecture::transparency::TransparencyPath)
/// variant count.
const NUM_PATHS: usize = 7;

/// Upper bound on the number of draws (and parallel surfaces) one query carries.
/// Each query stages its inputs and bucket outputs against this fixed cap so the
/// `std430` layout is a plain fixed-size record.
pub const MAX_DRAWS: usize = 256;

/// The portable core-`WGSL` transparent-draw binner kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `bin`
/// resolves one query (one draw list) per thread.
const TRANSPARENCY_BIN_DRAWS_WGSL: &str = r#"
// Transparent-draw binner twin: one thread owns one query (a whole draw list
// plus its parallel surface descriptors). It serially scans the slots, skips a
// slot with no surface, reproduces the golden select_transparency_path match
// ladder to pick a path discriminant, and appends the draw index into the
// matching bucket, preserving first-seen order. All integer comparison,
// selection and indexing; no floating-point.
//
// Path codes match the golden enum `as u32`: Sorted=0, WeightedOit=1,
// MomentOit=2, LayeredGlass=3, SingleLayerWater=4, HairVisibility=5,
// Volumetric=6. Kind codes: General=0, Water=1, Hair=2, Glass=3, Volume=4.
//
// Provenance: 孪生自本仓 prism_render_architecture::transparency::routing；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the batch; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Number of draws (valid slots) in this query, clamped to MAX_DRAWS.
    draw_count: u32,
    // Number of parallel surfaces; a slot >= this has no surface and is skipped.
    surface_count: u32,
    // 1 when the backend can run the moment-based OIT resolve.
    moment_oit: u32,
    pad: u32,
    // Draw indices, one per slot.
    draws: array<u32, 256>,
    // Per-slot content-class discriminant.
    kinds: array<u32, 256>,
    // Per-slot order-independent flag (0 or 1).
    order_independent: array<u32, 256>,
    // Per-slot high-fidelity flag (0 or 1).
    high_fidelity: array<u32, 256>,
    // Per-slot glass layer count.
    layer_count: array<u32, 256>,
}

struct Bins {
    // Per-path bucket lengths; counts[7] is padding.
    counts: array<u32, 8>,
    // Flattened buckets: path p occupies [p*256, p*256+256).
    buckets: array<u32, 1792>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Bins>;

@compute @workgroup_size(1)
fn bin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= params.count) {
        return;
    }

    var counts: array<u32, 7>;
    for (var b: u32 = 0u; b < 7u; b = b + 1u) {
        counts[b] = 0u;
    }

    let dc = queries[qi].draw_count;
    let sc = queries[qi].surface_count;
    for (var slot: u32 = 0u; slot < dc; slot = slot + 1u) {
        // A draw whose slot has no surface is skipped (golden surfaces.get None).
        if (slot >= sc) {
            continue;
        }
        let kind = queries[qi].kinds[slot];

        // select_transparency_path: dedicated kinds take their paths; glass
        // splits on layer_count; general sorts unless order-independent, then
        // moment OIT only with both the request and the capability.
        var path: u32 = 0u;
        if (kind == 1u) {
            path = 4u;
        } else if (kind == 2u) {
            path = 5u;
        } else if (kind == 4u) {
            path = 6u;
        } else if (kind == 3u) {
            if (queries[qi].layer_count[slot] > 1u) {
                path = 3u;
            } else {
                path = 0u;
            }
        } else {
            if (queries[qi].order_independent[slot] == 1u) {
                if (queries[qi].high_fidelity[slot] == 1u && queries[qi].moment_oit == 1u) {
                    path = 2u;
                } else {
                    path = 1u;
                }
            } else {
                path = 0u;
            }
        }

        // Append the draw index into the matching bucket, preserving order.
        let pos = counts[path];
        results[qi].buckets[path * 256u + pos] = queries[qi].draws[slot];
        counts[path] = pos + 1u;
    }

    for (var b: u32 = 0u; b < 7u; b = b + 1u) {
        results[qi].counts[b] = counts[b];
    }
    results[qi].counts[7] = 0u;
}
"#;

/// Uniform parameters for the dispatch: the query count and three pad words,
/// filling a `16`-byte uniform struct matching `Params` in
/// [`TRANSPARENCY_BIN_DRAWS_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the batch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one binning query, matching the `WGSL` `Query`
/// struct. Each per-slot array is host-padded to [`MAX_DRAWS`] entries.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Number of draws (valid slots), clamped to [`MAX_DRAWS`].
    draw_count: u32,
    /// Number of parallel surfaces.
    surface_count: u32,
    /// `1` when the backend supports moment `OIT`.
    moment_oit: u32,
    /// Padding word.
    pad: u32,
    /// Draw indices, one per slot.
    draws: [u32; MAX_DRAWS],
    /// Per-slot content-class discriminant.
    kinds: [u32; MAX_DRAWS],
    /// Per-slot order-independent flag (`0` or `1`).
    order_independent: [u32; MAX_DRAWS],
    /// Per-slot high-fidelity flag (`0` or `1`).
    high_fidelity: [u32; MAX_DRAWS],
    /// Per-slot glass layer count.
    layer_count: [u32; MAX_DRAWS],
}

/// `repr(C)` `std430` layout of one query's binned result, matching the `WGSL`
/// `Bins` struct. `counts[7]` is padding; the flattened buckets place path `p`
/// at `[p * MAX_DRAWS, p * MAX_DRAWS + MAX_DRAWS)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBins {
    /// Per-path bucket lengths; `counts[7]` is padding.
    counts: [u32; 8],
    /// Flattened per-path buckets.
    buckets: [u32; NUM_PATHS * MAX_DRAWS],
}

/// One binning query for the twin: a draw list plus the parallel per-slot
/// surface descriptors and the backend moment-`OIT` capability.
///
/// The private fixed-size arrays are populated through [`new`]; longer inputs
/// are clamped to [`MAX_DRAWS`] to match the fixed `std430` record.
///
/// Provenance: mirrors the `draws`/`surfaces`/`capability` inputs of the golden
/// [`bin_transparent_draws`](prism_render_architecture::transparency::routing::bin_transparent_draws).
#[derive(Clone, Copy)]
pub struct TransparencyBinDrawsQuery {
    /// Number of valid slots (draws), clamped to [`MAX_DRAWS`].
    draw_count: u32,
    /// Number of parallel surfaces.
    surface_count: u32,
    /// `1` when the backend supports moment `OIT`.
    moment_oit: u32,
    /// Draw indices, one per slot.
    draws: [u32; MAX_DRAWS],
    /// Per-slot content-class discriminant.
    kinds: [u32; MAX_DRAWS],
    /// Per-slot order-independent flag (`0` or `1`).
    order_independent: [u32; MAX_DRAWS],
    /// Per-slot high-fidelity flag (`0` or `1`).
    high_fidelity: [u32; MAX_DRAWS],
    /// Per-slot glass layer count.
    layer_count: [u32; MAX_DRAWS],
}

impl TransparencyBinDrawsQuery {
    /// Builds a query from a draw list and parallel per-slot surface fields.
    ///
    /// `draws` holds the draw indices; `kinds`, `order_independent`,
    /// `high_fidelity` and `layer_count` describe `surfaces[i]` for draw `i`.
    /// `draw_count` is `draws.len()` clamped to [`MAX_DRAWS`]; `surface_count`
    /// is `kinds.len()` clamped to [`MAX_DRAWS`], mirroring the golden parallel
    /// slices where a slot past the surfaces is skipped. The booleans are
    /// mapped to `0`/`1` words.
    #[must_use]
    pub fn new(
        draws: &[u32],
        kinds: &[u32],
        order_independent: &[bool],
        high_fidelity: &[bool],
        layer_count: &[u32],
        moment_oit: bool,
    ) -> TransparencyBinDrawsQuery {
        let mut draws_buf = [0u32; MAX_DRAWS];
        let mut kinds_buf = [0u32; MAX_DRAWS];
        let mut oi_buf = [0u32; MAX_DRAWS];
        let mut hf_buf = [0u32; MAX_DRAWS];
        let mut lc_buf = [0u32; MAX_DRAWS];

        let draw_count = draws.len().min(MAX_DRAWS);
        let surface_count = kinds.len().min(MAX_DRAWS);

        for (dst, &src) in draws_buf.iter_mut().zip(draws.iter()).take(draw_count) {
            *dst = src;
        }
        for (dst, &src) in kinds_buf.iter_mut().zip(kinds.iter()).take(surface_count) {
            *dst = src;
        }
        for (dst, &src) in oi_buf
            .iter_mut()
            .zip(order_independent.iter())
            .take(surface_count)
        {
            *dst = u32::from(src);
        }
        for (dst, &src) in hf_buf
            .iter_mut()
            .zip(high_fidelity.iter())
            .take(surface_count)
        {
            *dst = u32::from(src);
        }
        for (dst, &src) in lc_buf
            .iter_mut()
            .zip(layer_count.iter())
            .take(surface_count)
        {
            *dst = src;
        }

        TransparencyBinDrawsQuery {
            draw_count: draw_count as u32,
            surface_count: surface_count as u32,
            moment_oit: u32::from(moment_oit),
            draws: draws_buf,
            kinds: kinds_buf,
            order_independent: oi_buf,
            high_fidelity: hf_buf,
            layer_count: lc_buf,
        }
    }
}

/// One query's binned result: the seven per-path buckets of draw indices,
/// mirroring the golden
/// [`TransparencyBins`](prism_render_architecture::transparency::routing::TransparencyBins).
#[derive(Clone)]
pub struct TransparencyBinDrawsResult {
    /// Per-path bucket lengths, indexed by the path discriminant (`0..7`).
    counts: [u32; NUM_PATHS],
    /// Per-path buckets of draw indices, each padded to [`MAX_DRAWS`] entries;
    /// only the first `counts[p]` entries of bucket `p` are meaningful.
    buckets: [[u32; MAX_DRAWS]; NUM_PATHS],
}

impl TransparencyBinDrawsResult {
    /// Number of draws routed into the bucket for path discriminant `path`.
    #[must_use]
    pub fn count(&self, path: usize) -> u32 {
        self.counts[path]
    }

    /// First-seen-order view of the draw indices routed into the bucket for
    /// path discriminant `path`.
    #[must_use]
    pub fn bucket(&self, path: usize) -> &[u32] {
        &self.buckets[path][..self.counts[path] as usize]
    }
}

/// Encodes one [`TransparencyBinDrawsQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &TransparencyBinDrawsQuery) -> GpuQuery {
    GpuQuery {
        draw_count: q.draw_count,
        surface_count: q.surface_count,
        moment_oit: q.moment_oit,
        pad: 0,
        draws: q.draws,
        kinds: q.kinds,
        order_independent: q.order_independent,
        high_fidelity: q.high_fidelity,
        layer_count: q.layer_count,
    }
}

/// Decodes one `std430` [`GpuBins`] into a [`TransparencyBinDrawsResult`],
/// copying the seven live bucket lengths and their flattened index runs.
fn decode_bins(b: &GpuBins) -> TransparencyBinDrawsResult {
    let mut counts = [0u32; NUM_PATHS];
    let mut buckets = [[0u32; MAX_DRAWS]; NUM_PATHS];
    for (p, count) in counts.iter_mut().enumerate() {
        *count = b.counts[p];
        let base = p * MAX_DRAWS;
        buckets[p].copy_from_slice(&b.buckets[base..base + MAX_DRAWS]);
    }
    TransparencyBinDrawsResult { counts, buckets }
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

/// A compiled, reusable transparent-draw binner compute pipeline, twinning the
/// `CPU` golden `bin_transparent_draws` from
/// [`routing`](prism_render_architecture::transparency::routing).
pub struct GpuTransparencyBinDraws {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTransparencyBinDraws {
    /// Compiles the binner kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTransparencyBinDraws {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_transparency_bin_draws"),
            source: ShaderSource::Wgsl(TRANSPARENCY_BIN_DRAWS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_transparency_bin_draws_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_transparency_bin_draws_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_transparency_bin_draws_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("bin"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTransparencyBinDraws {
            module,
            layout,
            pipeline,
        }
    }

    /// Bins every query in `queries` and returns one
    /// [`TransparencyBinDrawsResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TransparencyBinDrawsQuery],
    ) -> Vec<TransparencyBinDrawsResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_transparency_bin_draws_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let result_bytes = (count * size_of::<GpuBins>()) as u64;
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_transparency_bin_draws_results"),
            size: result_bytes,
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
            label: Some("prism_volumetric_transparency_bin_draws_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_transparency_bin_draws_bind_group"),
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

        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_transparency_bin_draws_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_transparency_bin_draws_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_transparency_bin_draws_stage"),
            size: result_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&results_buf, 0, &stage, 0, result_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let bins = bytemuck::cast_slice::<u8, GpuBins>(&view).to_vec();
        drop(view);
        stage.unmap();

        bins.iter().map(decode_bins).collect()
    }
}
