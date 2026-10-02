//! `wgpu` compute twin of the particle-subsystem *hierarchical prefix scan*
//! primitive
//! ([`gpu_prefix_scan`](prism_render_architecture::particle::gpu_prefix_scan),
//! particle design §5.2 compaction, §11 counters, §12 sort offsets).
//!
//! A *prefix scan* turns a per-element count array into per-element start
//! offsets. The `CPU` golden
//! [`gpu_prefix_scan`](prism_render_architecture::particle::gpu_prefix_scan)
//! owns the math as a layered device scheme: the array is split into
//! `block_size`-element blocks, each block is scanned locally (the golden runs
//! the work-efficient Blelloch `up-sweep`/`down-sweep`), the per-block totals
//! are gathered into a `block_sums` array, that array is itself scanned into
//! per-block base offsets, and every block's exclusive offset is added back to
//! its elements. The result is a work-efficient scan that composes across
//! arbitrarily many `workgroup`s.
//!
//! [`GpuPrefixScan`] is the on-device twin. It follows the same two-level
//! host-aggregated pattern this crate's
//! [`gpu_compact`](crate::gpu_compact) twin uses: the inter-block base offsets
//! are a tiny coarse `scan` the host precomputes and uploads (one word per
//! block), while every per-element answer is reproduced *on the device*. One
//! thread per element runs the block-local exclusive prefix sum of the input
//! values in a bounded loop over its own block, adds back its block base, and
//! writes the scanned offset; one thread per block additionally reduces that
//! block's values into the device-written `block_sums` buffer. A passing
//! real-device parity test is therefore direct evidence the ported kernel
//! reproduces the same scanned offsets and per-block totals the reference
//! publishes, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces the golden
//! [`exclusive_scan_blocked`](prism_render_architecture::particle::gpu_prefix_scan::exclusive_scan_blocked):
//! [`GpuScanResult::scanned`] equals its `scanned` field element for element,
//! [`GpuScanResult::block_sums`] equals its `block_sums` field (the device
//! reduces each block's values), and [`GpuScanResult::total`] equals its
//! wrapping grand total. The inclusive mode reproduces
//! [`inclusive_scan`](prism_render_architecture::particle::gpu_prefix_scan::inclusive_scan),
//! which is the exclusive offset plus the element itself. The exclusive mode
//! also reproduces
//! [`naive_exclusive`](prism_render_architecture::particle::gpu_prefix_scan::naive_exclusive),
//! since the full exclusive prefix is independent of the block size. The host
//! supplies only the per-block base offsets (the exclusive prefix sum of the
//! per-block totals); the block-local `scan`, the base addition, and the
//! per-block reduction are all done on the device.
//!
//! # Correctness model
//!
//! Every value is a `u32`: a block-local prefix, a block base, a block total
//! and a scanned offset, all pure integer additions. Two's-complement addition
//! is associative, so a serial left-to-right sum and the golden block-tree sum
//! fold to the same bit pattern; the device `u32` add wraps on overflow exactly
//! as the golden `wrapping_add` does. `CPU` and `GPU` therefore compute
//! identical bit patterns and the parity test asserts an exact `==` on every
//! scanned offset, every block total and the grand total with no tolerance.
//! There is no float math anywhere, so there is no `ULP`-boundary degenerate
//! region to avoid: the comparison is bit-exact by construction.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `+`, `/`, the
//! `max` and `min` built-ins, unsigned comparisons and index arithmetic. There
//! is no `sqrt`, no transcendental call, no `u64` and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Both loops
//! are bounded by the block size (the per-element prefix runs at most
//! `block_size - 1` iterations; the per-block reduction at most `block_size`),
//! so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_prefix_scan`；无第三方引擎源码或衍生代码。
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

/// The hierarchical prefix-scan kernel, mirroring the `CPU` golden
/// [`gpu_prefix_scan`](prism_render_architecture::particle::gpu_prefix_scan).
/// The single entry point `solve` resolves one element per thread, embedded
/// inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_prefix_scan`。
const GPU_PREFIX_SCAN_WGSL: &str = r#"
// gpu_prefix_scan twin: one thread per element reproduces the CPU golden
// `particle::gpu_prefix_scan`. The host uploads the coarse per-block base
// offsets (the exclusive prefix sum of per-block totals); each thread runs the
// block-local exclusive prefix sum of its block's values, adds its block base,
// and writes the scanned offset. In inclusive mode it additionally adds its own
// element. One thread per block (the block-start thread) reduces that block's
// values into the block_sums buffer. All arithmetic is unsigned integer, so the
// device `+` wraps exactly as the golden `wrapping_add`, agreeing bit for bit.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::gpu_prefix_scan；
// 无第三方引擎源码或衍生代码。

// Dispatch parameters. 16-byte uniform block: the element count, the block
// size, the inclusive-mode flag, and one pad word, matching the host `Params`.
struct Params {
    count: u32,
    block_size: u32,
    inclusive: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read> block_bases: array<u32>;
@group(0) @binding(3) var<storage, read_write> scanned: array<u32>;
@group(0) @binding(4) var<storage, read_write> block_sums: array<u32>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let block_size = max(params.block_size, 1u);
    let block = idx / block_size;
    let block_start = block * block_size;

    // Block-local exclusive prefix sum of the input values strictly before
    // `idx` within its own block. The loop runs from the block start up to (but
    // excluding) `idx`, so it executes at most block_size - 1 iterations and is
    // bounded by the block size — the kernel provably terminates.
    var local: u32 = 0u;
    for (var k: u32 = block_start; k < idx; k = k + 1u) {
        local = local + values[k];
    }

    // Scanned offset = block base + block-local exclusive prefix. The u32 add
    // wraps on overflow, matching the golden `wrapping_add`.
    var result = block_bases[block] + local;
    // Inclusive scan adds the element itself, matching the golden
    // `inclusive_scan`.
    if (params.inclusive != 0u) {
        result = result + values[idx];
    }
    scanned[idx] = result;

    // The single block-start thread reduces its whole block into the per-block
    // total, matching the golden `block_sums`. The loop is bounded by the block
    // size, so the kernel provably terminates. No two threads write the same
    // block slot, so there is no race.
    if (idx == block_start) {
        let block_end = min(block_start + block_size, params.count);
        var sum: u32 = 0u;
        for (var k: u32 = block_start; k < block_end; k = k + 1u) {
            sum = sum + values[k];
        }
        block_sums[block] = sum;
    }
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`GPU_PREFIX_SCAN_WGSL`]: the element `count`, the `block_size`,
/// the `inclusive` flag and one pad word — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of elements (valid threads).
    count: u32,
    /// Block size each simulated `workgroup` scans.
    block_size: u32,
    /// Inclusive-mode flag: `1` adds each element to its own exclusive offset.
    inclusive: u32,
    /// Padding word.
    pad0: u32,
}

/// One hierarchical-scan request: the per-element input `values` and the
/// `block_size` each simulated `workgroup` scans.
///
/// The `block_size` is clamped to at least one on dispatch so a degenerate zero
/// can never cause a divide-by-zero, exactly as the golden
/// [`ScanConfig`](prism_render_architecture::particle::gpu_prefix_scan::ScanConfig)
/// clamps it. The full exclusive prefix is independent of the block size, so
/// any value reproduces the same [`GpuScanResult::scanned`]; only the
/// [`GpuScanResult::block_sums`] tiling changes.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_prefix_scan`。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuPrefixScanQuery {
    /// Per-element input values to scan.
    pub values: Vec<u32>,
    /// Elements scanned per block (per simulated `workgroup`).
    pub block_size: u32,
}

/// Result of a hierarchical multi-block exclusive scan on the device, mirroring
/// the golden
/// [`ScanResult`](prism_render_architecture::particle::gpu_prefix_scan::ScanResult)
/// field for field.
///
/// The three fields mirror the device storage buffers: the scanned per-element
/// offsets read back from the device, the per-block totals the device reduces,
/// and the wrapping grand total. Every field is compared with an exact `==`
/// against the golden, since all arithmetic is integer.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_prefix_scan`。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuScanResult {
    /// Exclusive prefix offset of every input element, in input order.
    pub scanned: Vec<u32>,
    /// Per-block totals, one entry per block, as the device reduces them.
    pub block_sums: Vec<u32>,
    /// Wrapping grand total of all input values (the exclusive offset just past
    /// the final element).
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

/// Host-side coarse `scan`: the per-block base offsets uploaded to the device,
/// paired with the wrapping grand total.
///
/// Entry `b` of the returned bases is the exclusive prefix sum of the per-block
/// totals — the first scanned offset block `b` owns. Accumulation uses
/// `wrapping_add`, matching the golden block-sum `scan`, so the uploaded bases
/// agree with the reference bit for bit. The returned total is the running
/// accumulator after the final block, which equals the golden grand total
/// (`offsets.last() + block_sums.last()`). There is always at least one block
/// for a non-empty input, so the returned vector is never zero-length.
fn block_bases(values: &[u32], block_size: usize) -> (Vec<u32>, u32) {
    let block_size = block_size.max(1);
    let num_blocks = values.len().div_ceil(block_size);
    let mut bases = Vec::with_capacity(num_blocks);
    let mut running: u32 = 0;
    for block in values.chunks(block_size) {
        bases.push(running);
        let total = block.iter().fold(0u32, |acc, &x| acc.wrapping_add(x));
        running = running.wrapping_add(total);
    }
    (bases, running)
}

/// A compiled, reusable hierarchical prefix-scan compute pipeline, twinning the
/// `CPU` golden
/// [`gpu_prefix_scan`](prism_render_architecture::particle::gpu_prefix_scan).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_prefix_scan`。
pub struct GpuPrefixScan {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPrefixScan {
    /// Compiles the hierarchical prefix-scan kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_prefix_scan`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPrefixScan {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gpu_prefix_scan_module"),
            source: ShaderSource::Wgsl(GPU_PREFIX_SCAN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gpu_prefix_scan_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gpu_prefix_scan_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gpu_prefix_scan_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPrefixScan {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the one dispatch for the given mode and reads back both the scanned
    /// offsets and the per-block totals, returning `(scanned, block_sums,
    /// total)`.
    ///
    /// An empty `values` batch issues **no dispatch** — a storage buffer may
    /// not be zero-sized — and returns two empty vectors with a zero total.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        query: &GpuPrefixScanQuery,
        inclusive: bool,
    ) -> (Vec<u32>, Vec<u32>, u32) {
        let count = query.values.len();
        if count == 0 {
            return (Vec::new(), Vec::new(), 0);
        }
        let device = ctx.device();

        let block_size = query.block_size.max(1);
        let (bases, total) = block_bases(&query.values, block_size as usize);
        let num_blocks = bases.len();

        let params = Params {
            count: count as u32,
            block_size,
            inclusive: u32::from(inclusive),
            pad0: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_prefix_scan_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let values_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_prefix_scan_values"),
            contents: bytemuck::cast_slice(&query.values),
            usage: BufferUsages::STORAGE,
        });
        let bases_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_prefix_scan_block_bases"),
            contents: bytemuck::cast_slice(&bases),
            usage: BufferUsages::STORAGE,
        });

        let scanned_bytes = (count as u64) * (size_of::<u32>() as u64);
        let scanned_zeros = vec![0u32; count];
        let scanned_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_prefix_scan_scanned"),
            contents: bytemuck::cast_slice(&scanned_zeros),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let scanned_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_prefix_scan_scanned_stage"),
            size: scanned_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let block_sums_bytes = (num_blocks as u64) * (size_of::<u32>() as u64);
        let block_sums_zeros = vec![0u32; num_blocks];
        let block_sums_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_prefix_scan_block_sums"),
            contents: bytemuck::cast_slice(&block_sums_zeros),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let block_sums_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_prefix_scan_block_sums_stage"),
            size: block_sums_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gpu_prefix_scan_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: values_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: bases_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: scanned_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: block_sums_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_gpu_prefix_scan_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gpu_prefix_scan_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per element, flattened to a 1-D dispatch.
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&scanned_buf, 0, &scanned_stage, 0, scanned_bytes);
        encoder.copy_buffer_to_buffer(&block_sums_buf, 0, &block_sums_stage, 0, block_sums_bytes);
        ctx.queue().submit([encoder.finish()]);

        scanned_stage.slice(..).map_async(MapMode::Read, |_| {});
        block_sums_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let scanned_view = scanned_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let scanned = bytemuck::cast_slice::<u8, u32>(&scanned_view).to_vec();
        drop(scanned_view);
        scanned_stage.unmap();

        let block_sums_view = block_sums_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let block_sums = bytemuck::cast_slice::<u8, u32>(&block_sums_view).to_vec();
        drop(block_sums_view);
        block_sums_stage.unmap();

        debug_assert_eq!(scanned.len(), count);
        debug_assert_eq!(block_sums.len(), num_blocks);
        (scanned, block_sums, total)
    }

    /// Runs the hierarchical exclusive scan on the device, returning the scanned
    /// offsets, the per-block totals and the wrapping grand total.
    ///
    /// Every field equals the matching golden
    /// [`exclusive_scan_blocked`](prism_render_architecture::particle::gpu_prefix_scan::exclusive_scan_blocked)
    /// field exactly, because the whole path is integer bit algebra. An empty
    /// `values` batch returns an empty result with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_prefix_scan`。
    #[must_use]
    pub fn scan(&self, ctx: &GpuContext, query: &GpuPrefixScanQuery) -> GpuScanResult {
        let (scanned, block_sums, total) = self.dispatch(ctx, query, false);
        GpuScanResult {
            scanned,
            block_sums,
            total,
        }
    }

    /// Runs the inclusive scan on the device, returning each element's wrapping
    /// prefix sum of `values[0..=i]`.
    ///
    /// The returned vector equals the golden
    /// [`inclusive_scan`](prism_render_architecture::particle::gpu_prefix_scan::inclusive_scan)
    /// element for element, because the full prefix is block-size-independent
    /// and the device adds each element to its exclusive offset with the same
    /// wrapping add. An empty `values` batch returns an empty vector with no
    /// dispatch issued.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_prefix_scan`。
    #[must_use]
    pub fn inclusive(&self, ctx: &GpuContext, query: &GpuPrefixScanQuery) -> Vec<u32> {
        self.dispatch(ctx, query, true).0
    }
}
