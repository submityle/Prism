//! `wgpu` compute twin of the particle-subsystem *warp-level prefix `scan`*
//! primitives
//! ([`gpu_scan_warp`](prism_render_architecture::particle::gpu_scan_warp),
//! particle design §5.2 compaction, §11 counters, §12 sort offsets).
//!
//! A *warp* (also called a *subgroup* or *wave*) is the fixed bundle of `SIMD`
//! lanes a `GPU` executes in lockstep — typically `32` lanes on `NVIDIA`, `32`
//! or `64` on `AMD`. The `CPU` golden
//! [`gpu_scan_warp`](prism_render_architecture::particle::gpu_scan_warp) owns
//! the math: a `Hillis-Steele` log-step shuffle-up sweep yields the inclusive
//! prefix of a warp's lanes, the exclusive prefix is that sweep shifted one
//! lane, the warp reduction is the terminal inclusive value, and a
//! `warp`-aggregated allocation turns per-lane counts into global destination
//! slots by exclusive-scanning each warp, reducing it to one warp-local sum,
//! and letting a single cross-warp `scan` promote those sums to per-warp base
//! offsets.
//!
//! # What is twinned
//!
//! [`GpuScanWarp`] is the on-device twin. Rather than depend on an optional
//! `subgroup` extension (which `WGSL` core does not guarantee), the kernel
//! replicates the shuffle sweep with a `workgroup`-shared array and
//! [`workgroupBarrier`] — one `workgroup` per warp, one thread per lane. The
//! kernel runs the `Hillis-Steele` inclusive sweep in shared memory, then emits
//! both the inclusive prefix and the exclusive prefix (the inclusive value
//! minus the lane's own value, a wrapping subtraction that is bit-exact with
//! the golden shift). The per-warp local sums are the tiny coarse `scan` the
//! host precomputes and uploads as per-warp base offsets — exactly the
//! `gpu_compact` cross-block paradigm — so every per-lane answer is still
//! reproduced on the device:
//!
//! - [`GpuWarpScanResult::inclusive`] equals the golden
//!   [`inclusive_scan_warp`](prism_render_architecture::particle::gpu_scan_warp::inclusive_scan_warp)
//!   over the whole input.
//! - [`GpuWarpScanResult::exclusive`] equals the golden
//!   [`exclusive_scan_warp`](prism_render_architecture::particle::gpu_scan_warp::exclusive_scan_warp).
//! - [`GpuWarpScanResult::total`] equals the golden
//!   [`warp_reduce`](prism_render_architecture::particle::gpu_scan_warp::warp_reduce).
//! - [`GpuScanWarp::aggregate_alloc`] reproduces every field of the golden
//!   [`warp_aggregate_alloc`](prism_render_architecture::particle::gpu_scan_warp::warp_aggregate_alloc):
//!   `lane_slots`, `warp_local_sums`, `warp_base_offsets` and `total`.
//!
//! # Correctness model
//!
//! Every value is a `u32`: a lane count, a prefix, a warp sum, a base offset or
//! a destination slot, all pure integer additions with `wrapping_add`
//! semantics. The device `u32` `+` and `-` wrap on overflow exactly as the
//! golden `wrapping_add` does, so `CPU` and `GPU` compute identical bit
//! patterns and the parity test asserts an exact `==` with no tolerance. There
//! is no float math anywhere, so there is no `ULP`-boundary degenerate region to
//! avoid: the comparison is bit-exact by construction.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned `+`, `-`,
//! `*`, `/`, the `max` built-in, shifts, unsigned comparison and
//! [`workgroupBarrier`] — with no `subgroup` built-in, no `sqrt`, no
//! transcendental call, no `u64` and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. The warp width is a `host`
//! parameter bounded by the shared array's [`WARP_LANE_LIMIT`] (`64`, the widest
//! real-hardware warp); the loop bound is the fixed workgroup size, so the
//! kernel provably terminates. Inputs longer than one warp are partitioned into
//! warps whose per-warp sums the host coarse-`scan`s into base offsets, so an
//! arbitrarily long input is still a single global prefix `scan`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup, equal to the widest supported warp. `64` is
/// the portable, warp-friendly default used across this crate's
/// one-thread-per-element kernels and is also the widest real-hardware warp
/// (`AMD`), so one workgroup covers one warp.
const WORKGROUP_SIZE: u32 = 64;

/// The largest warp lane count the shared-memory sweep supports, equal to
/// [`WORKGROUP_SIZE`]. It bounds the `host`-supplied warp width so the shared
/// array never overflows; `32` (`NVIDIA`) and `64` (`AMD`) both fit.
pub const WARP_LANE_LIMIT: u32 = WORKGROUP_SIZE;

/// The canonical warp width shared by most current `GPU` architectures, matching
/// the golden
/// [`WarpConfig::DEFAULT_LANE_COUNT`](prism_render_architecture::particle::gpu_scan_warp::WarpConfig::DEFAULT_LANE_COUNT).
pub const DEFAULT_LANE_COUNT: u32 = 32;

/// The warp-`scan` kernel, mirroring the `CPU` golden
/// [`gpu_scan_warp`](prism_render_architecture::particle::gpu_scan_warp). One
/// workgroup resolves one warp; thread `lane` loads its value into shared
/// memory, the workgroup runs the `Hillis-Steele` inclusive sweep with
/// [`workgroupBarrier`], and each lane writes its global inclusive and exclusive
/// prefix by adding the `host`-supplied per-warp base.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`。
const GPU_SCAN_WARP_WGSL: &str = r#"
// gpu_scan_warp twin: one workgroup per warp, one thread per lane, reproducing
// the CPU golden `particle::gpu_scan_warp`. The host uploads the coarse
// per-warp base offsets (the exclusive prefix sum of per-warp reductions); each
// workgroup runs the Hillis-Steele inclusive sweep over its warp's lanes in a
// workgroup-shared array guarded by workgroupBarrier, then every lane writes its
// global inclusive and exclusive prefix by adding its warp base. All arithmetic
// is unsigned integer, so the device `+` and `-` wrap exactly as the golden
// `wrapping_add`, agreeing bit for bit. WGSL core has no guaranteed subgroup
// op, so shared memory replaces the lane shuffle.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::gpu_scan_warp;
// 无第三方引擎源码或衍生代码。

// Dispatch parameters. 16-byte uniform block: the lane count, the warp width,
// and two pad words, matching the host `Params`.
struct Params {
    count: u32,
    warp_width: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lanes: array<u32>;
@group(0) @binding(2) var<storage, read> warp_bases: array<u32>;
@group(0) @binding(3) var<storage, read_write> inclusive: array<u32>;
@group(0) @binding(4) var<storage, read_write> exclusive: array<u32>;

// One warp's lanes, shared across the workgroup for the shuffle-up sweep.
var<workgroup> shared_lanes: array<u32, 64u>;

@compute @workgroup_size(64)
fn solve(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let warp_index = wid.x;
    let lane = lid.x;
    // Width is clamped to at least one so a degenerate zero never divides or
    // loops forever; it is uniform across the workgroup.
    let width = max(params.warp_width, 1u);
    let warp_start = warp_index * width;
    let global_index = warp_start + lane;

    // Load this lane's value; lanes beyond the warp width or beyond the input
    // contribute the neutral element 0.
    var value: u32 = 0u;
    if (lane < width && global_index < params.count) {
        value = lanes[global_index];
    }
    shared_lanes[lane] = value;
    workgroupBarrier();

    // Hillis-Steele inclusive sweep: offset doubles from 1 until it covers the
    // warp width. Every barrier is in uniform control flow (width is uniform),
    // so the workgroup never deadlocks. The addend is read before the write so
    // no lane observes a half-updated neighbour.
    var offset: u32 = 1u;
    loop {
        if (offset >= width) {
            break;
        }
        var addend: u32 = 0u;
        if (lane >= offset && lane < width) {
            addend = shared_lanes[lane - offset];
        }
        workgroupBarrier();
        if (lane >= offset && lane < width) {
            shared_lanes[lane] = shared_lanes[lane] + addend;
        }
        workgroupBarrier();
        offset = offset << 1u;
    }

    // Write the global inclusive and exclusive prefixes. The exclusive prefix
    // is the inclusive value minus this lane's own value: a wrapping u32
    // subtraction that reproduces the golden one-lane shift bit for bit, since
    // (prefix + value) - value == prefix modulo 2^32.
    if (lane < width && global_index < params.count) {
        let base = warp_bases[warp_index];
        let incl_within = shared_lanes[lane];
        inclusive[global_index] = base + incl_within;
        exclusive[global_index] = base + (incl_within - value);
    }
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`GPU_SCAN_WARP_WGSL`]: the lane `count`, the `warp_width` and
/// two pad words — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of input lanes (valid threads across all warps).
    count: u32,
    /// Lanes per warp (one `workgroup`'s domain).
    warp_width: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// Configuration for the `warp`/`subgroup` `scan`: the lane count of one warp,
/// mirroring the golden
/// [`WarpConfig`](prism_render_architecture::particle::gpu_scan_warp::WarpConfig).
///
/// Any positive `lane_count` is valid; it is clamped to at least one and at most
/// [`WARP_LANE_LIMIT`] on dispatch so the shared-memory sweep never overflows
/// and a degenerate zero can never divide by zero.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuWarpConfig {
    /// Lanes per warp. Clamped to `1..=WARP_LANE_LIMIT` on dispatch.
    pub lane_count: u32,
}

impl Default for GpuWarpConfig {
    /// The canonical [`DEFAULT_LANE_COUNT`]-lane warp width.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`。
    fn default() -> Self {
        Self {
            lane_count: DEFAULT_LANE_COUNT,
        }
    }
}

impl GpuWarpConfig {
    /// Builds a config with the given `lane_count`.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`。
    #[must_use]
    pub fn new(lane_count: u32) -> GpuWarpConfig {
        GpuWarpConfig { lane_count }
    }

    /// The effective warp width, clamped to `1..=WARP_LANE_LIMIT`.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`。
    #[must_use]
    pub fn width(self) -> u32 {
        self.lane_count.clamp(1, WARP_LANE_LIMIT)
    }
}

/// One `warp`-`scan` request: the `lanes` to scan. The whole input is treated as
/// a single global prefix `scan` (partitioned internally into warps of
/// [`WARP_LANE_LIMIT`] lanes whose per-warp sums the `host` coarse-`scan`s),
/// matching the golden
/// [`inclusive_scan_warp`](prism_render_architecture::particle::gpu_scan_warp::inclusive_scan_warp)
/// over the full slice.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuWarpScanQuery {
    /// The lanes to scan, in input order.
    pub lanes: Vec<u32>,
}

/// The resolved `scan` of one [`GpuWarpScanQuery`]: the inclusive prefix, the
/// exclusive prefix and the whole-input reduction.
///
/// `inclusive` matches the golden
/// [`inclusive_scan_warp`](prism_render_architecture::particle::gpu_scan_warp::inclusive_scan_warp),
/// `exclusive` matches
/// [`exclusive_scan_warp`](prism_render_architecture::particle::gpu_scan_warp::exclusive_scan_warp),
/// and `total` matches
/// [`warp_reduce`](prism_render_architecture::particle::gpu_scan_warp::warp_reduce),
/// each bit for bit.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuWarpScanResult {
    /// Per-lane inclusive prefix sum, in input order.
    pub inclusive: Vec<u32>,
    /// Per-lane exclusive prefix sum, in input order.
    pub exclusive: Vec<u32>,
    /// The whole-input `wrapping_add` reduction; `0` for an empty input.
    pub total: u32,
}

/// One `warp`-aggregated allocation request: the per-lane `counts` and the warp
/// `config` partitioning them.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuWarpAggregateQuery {
    /// Per-lane counts, in input order.
    pub counts: Vec<u32>,
    /// The warp width partitioning the counts.
    pub config: GpuWarpConfig,
}

/// Result of a [`warp`-aggregated allocation](GpuScanWarp::aggregate_alloc),
/// mirroring the golden
/// [`WarpAppend`](prism_render_architecture::particle::gpu_scan_warp::WarpAppend)
/// field for field.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuWarpAppend {
    /// Per-lane global base slot, in input order; equal to the plain global
    /// exclusive prefix sum of the input counts.
    pub lane_slots: Vec<u32>,
    /// Per-warp local sum: the `wrapping_add` reduction of each warp's lane
    /// counts, one entry per warp.
    pub warp_local_sums: Vec<u32>,
    /// Per-warp global base offset: the exclusive `scan` of `warp_local_sums`,
    /// one entry per warp.
    pub warp_base_offsets: Vec<u32>,
    /// Grand total reserved by the append: the `wrapping_add` sum of every
    /// warp-local sum.
    pub total: u32,
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

/// Per-warp `wrapping_add` reduction: one word per warp of `width` lanes,
/// mirroring the golden `warp_reduce` over each `counts.chunks(width)`.
fn warp_local_sums(values: &[u32], width: usize) -> Vec<u32> {
    let width = width.max(1);
    values
        .chunks(width)
        .map(|warp| warp.iter().fold(0u32, |acc, &lane| acc.wrapping_add(lane)))
        .collect()
}

/// Host-side coarse `scan`: the exclusive prefix sum of `values`, the per-warp
/// base offsets uploaded to the device. Accumulation uses `wrapping_add`,
/// matching the golden `exclusive_scan_warp`, so the uploaded bases agree with
/// the reference bit for bit.
fn exclusive_prefix(values: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(values.len());
    let mut running: u32 = 0;
    for &value in values {
        out.push(running);
        running = running.wrapping_add(value);
    }
    out
}

/// A compiled, reusable warp-`scan` compute pipeline, twinning the `CPU` golden
/// [`gpu_scan_warp`](prism_render_architecture::particle::gpu_scan_warp).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`。
pub struct GpuScanWarp {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuScanWarp {
    /// Compiles the warp-`scan` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset plus
    /// [`workgroupBarrier`], so no optional device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuScanWarp {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gpu_scan_warp_module"),
            source: ShaderSource::Wgsl(GPU_SCAN_WARP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gpu_scan_warp_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gpu_scan_warp_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gpu_scan_warp_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuScanWarp {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the one dispatch over `lanes` partitioned into warps of `width`,
    /// returning the global inclusive and exclusive prefixes. The `host`
    /// supplies the per-warp bases (the exclusive prefix sum of the per-warp
    /// reductions); the device runs the shared-memory `Hillis-Steele` sweep and
    /// adds its warp base. An empty input issues **no dispatch** — a storage
    /// buffer may not be zero-sized — and returns two empty vectors.
    fn dispatch(&self, ctx: &GpuContext, lanes: &[u32], width: u32) -> (Vec<u32>, Vec<u32>) {
        let count = lanes.len();
        if count == 0 {
            return (Vec::new(), Vec::new());
        }
        let device = ctx.device();

        let width = width.clamp(1, WARP_LANE_LIMIT);
        let sums = warp_local_sums(lanes, width as usize);
        let bases = exclusive_prefix(&sums);

        let params = Params {
            count: count as u32,
            warp_width: width,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_scan_warp_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let lanes_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_scan_warp_lanes"),
            contents: bytemuck::cast_slice(lanes),
            usage: BufferUsages::STORAGE,
        });
        let bases_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_scan_warp_warp_bases"),
            contents: bytemuck::cast_slice(&bases),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count as u64) * (size_of::<u32>() as u64);
        let inclusive_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_scan_warp_inclusive"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let inclusive_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_scan_warp_inclusive_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exclusive_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_scan_warp_exclusive"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let exclusive_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_scan_warp_exclusive_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gpu_scan_warp_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: lanes_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: bases_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: inclusive_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: exclusive_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_gpu_scan_warp_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gpu_scan_warp_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per warp, flattened to a 1-D dispatch.
            let groups = (count as u32).div_ceil(width);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&inclusive_buf, 0, &inclusive_stage, 0, out_bytes);
        encoder.copy_buffer_to_buffer(&exclusive_buf, 0, &exclusive_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        inclusive_stage.slice(..).map_async(MapMode::Read, |_| {});
        exclusive_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let inclusive_view = inclusive_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let inclusive = bytemuck::cast_slice::<u8, u32>(&inclusive_view).to_vec();
        drop(inclusive_view);
        inclusive_stage.unmap();

        let exclusive_view = exclusive_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let exclusive = bytemuck::cast_slice::<u8, u32>(&exclusive_view).to_vec();
        drop(exclusive_view);
        exclusive_stage.unmap();

        debug_assert_eq!(inclusive.len(), count);
        debug_assert_eq!(exclusive.len(), count);
        (inclusive, exclusive)
    }

    /// Scans `query.lanes` as one global prefix `scan`, returning the inclusive
    /// and exclusive prefixes and the whole-input reduction.
    ///
    /// `inclusive` equals the golden
    /// [`inclusive_scan_warp`](prism_render_architecture::particle::gpu_scan_warp::inclusive_scan_warp),
    /// `exclusive` equals
    /// [`exclusive_scan_warp`](prism_render_architecture::particle::gpu_scan_warp::exclusive_scan_warp),
    /// and `total` equals
    /// [`warp_reduce`](prism_render_architecture::particle::gpu_scan_warp::warp_reduce),
    /// each bit for bit. An empty input returns empty prefixes and a `total` of
    /// `0` with no dispatch issued.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`。
    #[must_use]
    pub fn scan(&self, ctx: &GpuContext, query: &GpuWarpScanQuery) -> GpuWarpScanResult {
        let (inclusive, exclusive) = self.dispatch(ctx, &query.lanes, WARP_LANE_LIMIT);
        let total = inclusive.last().copied().unwrap_or(0);
        GpuWarpScanResult {
            inclusive,
            exclusive,
            total,
        }
    }

    /// Runs the `warp`-aggregated allocation over `query.counts`, partitioned
    /// into warps of `query.config` width.
    ///
    /// Every field of the returned [`GpuWarpAppend`] matches the golden
    /// [`warp_aggregate_alloc`](prism_render_architecture::particle::gpu_scan_warp::warp_aggregate_alloc):
    /// `lane_slots` is the device per-warp exclusive `scan` plus the uploaded
    /// base (equal to the global exclusive prefix sum), while
    /// `warp_local_sums`, `warp_base_offsets` and `total` are the `host` coarse
    /// `scan` the device bases are built from. An empty input returns all-empty
    /// vectors and a `total` of `0` with no dispatch issued.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_warp`。
    #[must_use]
    pub fn aggregate_alloc(
        &self,
        ctx: &GpuContext,
        query: &GpuWarpAggregateQuery,
    ) -> GpuWarpAppend {
        let width = query.config.width();
        let (_, lane_slots) = self.dispatch(ctx, &query.counts, width);
        let sums = warp_local_sums(&query.counts, width as usize);
        let bases = exclusive_prefix(&sums);
        let total = sums.iter().fold(0u32, |acc, &sum| acc.wrapping_add(sum));
        GpuWarpAppend {
            lane_slots,
            warp_local_sums: sums,
            warp_base_offsets: bases,
            total,
        }
    }
}
