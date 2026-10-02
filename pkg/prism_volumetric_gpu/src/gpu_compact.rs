//! `wgpu` compute twin of the particle-subsystem *stream compaction* primitive
//! ([`gpu_compact`](prism_render_architecture::particle::gpu_compact),
//! particle design §5.2 compaction, §9 `Compaction` pass, §11 counters).
//!
//! *Stream compaction* takes a per-element predicate mask (`keep = 1` /
//! `drop = 0`) and packs every kept element down into a dense prefix of the
//! output. The `CPU` golden
//! [`gpu_compact`](prism_render_architecture::particle::gpu_compact) owns the
//! math as the classic two-pass `scatter`: a block-local *exclusive prefix sum*
//! of the keep bits yields each kept element's slot *within its block*, the
//! per-block survivor totals are themselves scanned into per-block base offsets
//! ([`scatter_offsets`](prism_render_architecture::particle::gpu_compact::scatter_offsets)),
//! and every kept element's global destination is its block base plus its
//! block-local offset. The surviving original index is scattered there, with no
//! holes ([`compact_indices`](prism_render_architecture::particle::gpu_compact::compact_indices)).
//!
//! [`GpuCompact`] is the on-device twin. The inter-block base offsets are a tiny
//! coarse `scan` the host precomputes and uploads (one word per block); every
//! per-element answer is reproduced *on the device*. One thread per element
//! runs the block-local exclusive prefix sum of keep bits in a bounded loop
//! over its own block, adds back its block base, and both records the keep bit
//! and the global `scatter` destination *and* scatters its surviving original
//! index into the dense compact slot. A passing real-device parity test is
//! therefore direct evidence the ported kernel reproduces the same destination
//! offsets and compacted index list the reference publishes, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces the golden per-element `scatter`:
//! [`GpuCompactResult::destination`] equals
//! [`scatter_offsets`](prism_render_architecture::particle::gpu_compact::scatter_offsets)
//! element for element, and the device-scattered dense list equals
//! [`compact_indices`](prism_render_architecture::particle::gpu_compact::compact_indices)
//! element for element. The host supplies only the per-block base offsets (the
//! exclusive prefix sum of the per-block survivor totals); the block-local
//! `scan`, the base addition and the final `scatter` write are all done on the
//! device. A non-zero flag counts as `keep`, mirroring the golden `keep_bit`.
//!
//! # Correctness model
//!
//! Every value is a `u32`: a keep bit, a block-local prefix, a block base and a
//! destination offset, all pure integer additions. The device `u32` add wraps
//! on overflow exactly as the golden `wrapping_add` does, so `CPU` and `GPU`
//! compute identical bit patterns and the parity test asserts an exact `==` on
//! every destination and every compacted index with no tolerance. There is no
//! float math anywhere, so there is no `ULP`-boundary degenerate region to
//! avoid: the comparison is bit-exact by construction.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `+`, `*`, `/`,
//! the `max` built-in, unsigned comparisons and index arithmetic. There is no
//! `sqrt`, no transcendental call, no `u64` and no optional device feature, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. The only loop is the
//! block-local `scan`, whose bound is the element index within its block and
//! thus at most `block_size - 1` iterations, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_compact`；无第三方引擎源码或衍生代码。
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

/// The stream-compaction kernel, mirroring the `CPU` golden
/// [`gpu_compact`](prism_render_architecture::particle::gpu_compact). The single
/// entry point `solve` resolves one element per thread, embedded inline so the
/// twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_compact`。
const GPU_COMPACT_WGSL: &str = r#"
// gpu_compact twin: one thread per element reproduces the CPU golden
// `particle::gpu_compact`. The host uploads the coarse per-block base offsets
// (the exclusive prefix sum of per-block survivor totals); each thread runs the
// block-local exclusive prefix sum of keep bits over its own block, adds its
// block base, records the keep bit and global scatter destination, and scatters
// its surviving original index into the dense compact slot. All arithmetic is
// unsigned integer, so the device `+` wraps exactly as the golden
// `wrapping_add`, agreeing bit for bit.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::gpu_compact;
// 无第三方引擎源码或衍生代码。

// Dispatch parameters. 16-byte uniform block: the element count, the block
// size, and two pad words, matching the host `Params`.
struct Params {
    count: u32,
    block_size: u32,
    pad0: u32,
    pad1: u32,
}

// One result. 8-byte std430 stride of two scalar words, matching the host
// `GpuCompactResult`: the keep bit and the global scatter destination.
struct CompactResult {
    keep: u32,
    destination: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> block_bases: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<CompactResult>;
@group(0) @binding(4) var<storage, read_write> compacted: array<u32>;

// Normalizes one predicate flag to a single keep bit (0 or 1); any non-zero
// value counts as keep, mirroring the golden `keep_bit`.
fn keep_bit(flag: u32) -> u32 {
    if (flag != 0u) {
        return 1u;
    }
    return 0u;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let block_size = max(params.block_size, 1u);
    let block = idx / block_size;
    let block_start = block * block_size;

    // Block-local exclusive prefix sum of keep bits: the count of survivors
    // strictly before `idx` within its own block. The loop runs from the block
    // start up to (but excluding) `idx`, so it executes at most block_size - 1
    // iterations and is bounded by the block size — the kernel provably
    // terminates.
    var local: u32 = 0u;
    for (var k: u32 = block_start; k < idx; k = k + 1u) {
        local = local + keep_bit(flags[k]);
    }

    let keep = keep_bit(flags[idx]);
    // Global scatter destination = block base + block-local offset. The u32 add
    // wraps on overflow, matching the golden `wrapping_add`.
    let destination = block_bases[block] + local;

    var out: CompactResult;
    out.keep = keep;
    out.destination = destination;
    results[idx] = out;

    // Scatter the surviving original index into its dense compact slot. Dropped
    // elements write nothing; every kept element owns a unique destination, so
    // no two threads ever race on the same slot.
    if (keep == 1u) {
        compacted[destination] = idx;
    }
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`GPU_COMPACT_WGSL`]: the element `count`, the `block_size` and
/// two pad words — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of elements (valid threads).
    count: u32,
    /// Block size each simulated `workgroup` compacts.
    block_size: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One stream-compaction request: the per-element predicate `flags` (`keep` is
/// any non-zero value, `drop` is `0`) and the `block_size` each simulated
/// `workgroup` compacts.
///
/// The `block_size` is clamped to at least one on dispatch so a degenerate zero
/// can never cause a divide-by-zero, exactly as the golden
/// [`CompactConfig`](prism_render_architecture::particle::gpu_compact::CompactConfig)
/// clamps it.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_compact`。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuCompactQuery {
    /// Per-element predicate flags: any non-zero value keeps the element.
    pub flags: Vec<u32>,
    /// Elements compacted per block (per simulated `workgroup`).
    pub block_size: u32,
}

/// One resolved answer for a single element, mirroring the golden per-element
/// `scatter`: the keep bit and the global `scatter` destination offset.
///
/// The `repr(C)` layout — two `u32` words, `8` bytes — matches the `WGSL`
/// `CompactResult` struct exactly, so device results are read back without a
/// separate decode step. `destination` equals the matching
/// [`scatter_offsets`](prism_render_architecture::particle::gpu_compact::scatter_offsets)
/// entry: for a kept element it is the final compact slot; for a dropped
/// element it is the slot the next survivor would take.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_compact`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuCompactResult {
    /// The keep bit: `1` when the element's flag is non-zero, else `0`.
    pub keep: u32,
    /// The global `scatter` destination offset (block base plus block-local
    /// exclusive prefix).
    pub destination: u32,
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

/// Normalizes one predicate flag to a single keep bit (`0` or `1`), mirroring
/// the golden `keep_bit`.
fn keep_bit(flag: u32) -> u32 {
    u32::from(flag != 0)
}

/// Host-side coarse `scan`: the per-block base offsets uploaded to the device.
///
/// Entry `b` is the exclusive prefix sum of the per-block survivor totals — the
/// first dense compact slot block `b` owns. Accumulation uses `wrapping_add`,
/// matching the golden `exclusive_prefix_sum`, so the uploaded bases agree with
/// the reference bit for bit. There is always at least one block for a
/// non-empty input, so the returned vector is never zero-length.
fn block_bases(flags: &[u32], block_size: usize) -> Vec<u32> {
    let block_size = block_size.max(1);
    let num_blocks = flags.len().div_ceil(block_size);
    let mut bases = Vec::with_capacity(num_blocks);
    let mut running: u32 = 0;
    for block in flags.chunks(block_size) {
        bases.push(running);
        let survivors = block
            .iter()
            .fold(0u32, |acc, &flag| acc.wrapping_add(keep_bit(flag)));
        running = running.wrapping_add(survivors);
    }
    bases
}

/// Number of survivors the predicate keeps (the population count of its keep
/// bits), which is exactly the length of the dense compacted list.
fn survivor_count(flags: &[u32]) -> usize {
    flags.iter().filter(|&&flag| flag != 0).count()
}

/// A compiled, reusable stream-compaction compute pipeline, twinning the `CPU`
/// golden [`gpu_compact`](prism_render_architecture::particle::gpu_compact).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_compact`。
pub struct GpuCompact {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCompact {
    /// Compiles the stream-compaction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_compact`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCompact {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gpu_compact_module"),
            source: ShaderSource::Wgsl(GPU_COMPACT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gpu_compact_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gpu_compact_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gpu_compact_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCompact {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the one dispatch and reads back both the per-element results and the
    /// dense compacted index list, returning `(results, compacted)`.
    ///
    /// An empty `flags` batch issues **no dispatch** — a storage buffer may not
    /// be zero-sized — and returns two empty vectors.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        query: &GpuCompactQuery,
    ) -> (Vec<GpuCompactResult>, Vec<u32>) {
        let count = query.flags.len();
        if count == 0 {
            return (Vec::new(), Vec::new());
        }
        let device = ctx.device();

        let block_size = query.block_size.max(1);
        let bases = block_bases(&query.flags, block_size as usize);
        let keep_total = survivor_count(&query.flags);

        let params = Params {
            count: count as u32,
            block_size,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_compact_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let flags_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_compact_flags"),
            contents: bytemuck::cast_slice(&query.flags),
            usage: BufferUsages::STORAGE,
        });
        let bases_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_compact_block_bases"),
            contents: bytemuck::cast_slice(&bases),
            usage: BufferUsages::STORAGE,
        });

        let results_bytes = (count as u64) * (size_of::<GpuCompactResult>() as u64);
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_compact_results"),
            size: results_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_compact_results_stage"),
            size: results_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // The compacted buffer spans every element (never zero-sized, since the
        // input is non-empty); only the first `keep_total` slots are survivors.
        // Zero-initialized so untouched tail slots read back as zero.
        let compacted_zeros = vec![0u32; count];
        let compacted_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_compact_compacted"),
            contents: bytemuck::cast_slice(&compacted_zeros),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let compacted_bytes = (count as u64) * (size_of::<u32>() as u64);
        let compacted_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_compact_compacted_stage"),
            size: compacted_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gpu_compact_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: flags_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: bases_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: results_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: compacted_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_gpu_compact_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gpu_compact_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per element, flattened to a 1-D dispatch.
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, results_bytes);
        encoder.copy_buffer_to_buffer(&compacted_buf, 0, &compacted_stage, 0, compacted_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        compacted_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let results_view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let results = bytemuck::cast_slice::<u8, GpuCompactResult>(&results_view).to_vec();
        drop(results_view);
        results_stage.unmap();

        let compacted_view = compacted_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let mut compacted = bytemuck::cast_slice::<u8, u32>(&compacted_view).to_vec();
        drop(compacted_view);
        compacted_stage.unmap();
        compacted.truncate(keep_total);

        debug_assert_eq!(results.len(), count);
        debug_assert_eq!(compacted.len(), keep_total);
        (results, compacted)
    }

    /// Resolves every element's keep bit and global `scatter` destination,
    /// returning one [`GpuCompactResult`] per input element, in order.
    ///
    /// Each result's `destination` equals the matching
    /// [`scatter_offsets`](prism_render_architecture::particle::gpu_compact::scatter_offsets)
    /// entry exactly, because the whole path is integer bit algebra. An empty
    /// `flags` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_compact`。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, query: &GpuCompactQuery) -> Vec<GpuCompactResult> {
        self.dispatch(ctx, query).0
    }

    /// Runs the full two-pass compaction on the device and reads back the dense
    /// list of surviving original indices, in ascending input order.
    ///
    /// The returned list equals the golden
    /// [`compact_indices`](prism_render_architecture::particle::gpu_compact::compact_indices)
    /// element for element, because every survivor is scattered on the device
    /// to its unique [`scatter_offsets`](prism_render_architecture::particle::gpu_compact::scatter_offsets)
    /// slot. An empty `flags` batch returns an empty vector with no dispatch
    /// issued.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_compact`。
    #[must_use]
    pub fn compact(&self, ctx: &GpuContext, query: &GpuCompactQuery) -> Vec<u32> {
        self.dispatch(ctx, query).1
    }
}
