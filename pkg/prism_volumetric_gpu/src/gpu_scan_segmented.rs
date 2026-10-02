//! `wgpu` compute twin of the particle-subsystem *segmented* prefix-scan
//! primitive
//! ([`gpu_scan_segmented`](prism_render_architecture::particle::gpu_scan_segmented),
//! particle design §5.2 compaction, §11 counters, §12 sort offsets).
//!
//! A *segmented* scan is the per-segment cousin of the plain scan: alongside the
//! value array it consumes a *head-flag* array, where a set flag marks the first
//! element of a new segment, and wherever a head flag is set the running
//! accumulator resets to the neutral element `0` so the scan restarts
//! independently inside every segment. The `CPU` golden
//! [`gpu_scan_segmented`](prism_render_architecture::particle::gpu_scan_segmented)
//! owns the math as a three-pass block-decomposed scheme: pass one
//! segment-scans each fixed-size block independently and records whether the
//! block holds any head flag plus its trailing *open* sum, pass two threads a
//! per-block carry across the blocks (a head-bearing block restarts the carry
//! from its own open sum, a head-free block extends the inherited open segment),
//! and pass three folds each block's carry-in back into only the elements ahead
//! of that block's first head flag, so a head flag severs the carry and no sum
//! ever leaks across a segment boundary.
//!
//! [`GpuSegmentedScan`] is the on-device twin, built on the exact two-level
//! split [`GpuCompact`](crate::gpu_compact::GpuCompact) uses. The inter-block
//! carry-ins are the small coarse `scan` the host precomputes and uploads (one
//! word per block, pass two); every per-element answer is reproduced *on the
//! device*. One thread per element runs the block-local segment scan in a
//! bounded loop over its own block up to its index, decides from that same loop
//! whether a head flag sits at or before it (which severs the carry), folds in
//! its block carry-in only when no head severed it, and finally adds its own
//! value for the inclusive variant. A passing real-device parity test is
//! therefore direct evidence the ported kernel reproduces the same per-segment
//! offsets the reference publishes, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces the golden per-element segmented prefix:
//! [`GpuSegmentedScanResult::scanned`] equals the golden `scanned` element for
//! element, for both the exclusive and inclusive variants, and
//! [`GpuSegmentedScanResult::block_carry_ins`] equals the golden
//! `block_carry_ins` block for block. The host supplies only the per-block
//! carry-ins (pass two, the coarse open-segment `scan`); the block-local segment
//! scan, the carry fold and the inclusive self-add are all done on the device. A
//! non-zero flag counts as a segment head, mirroring the golden `is_head`, and a
//! flag array shorter than the value array is padded with non-head zeros on the
//! host so ragged input behaves exactly as the reference.
//!
//! # Correctness model
//!
//! Every value is a `u32`: a value, a head flag, a block carry and a running
//! prefix, all pure integer additions. The device `u32` add wraps on overflow
//! exactly as the golden `wrapping_add` does, so `CPU` and `GPU` compute
//! identical bit patterns and the parity test asserts an exact `==` on every
//! scanned offset and every block carry-in with no tolerance. There is no float
//! math anywhere, so there is no `ULP`-boundary degenerate region to avoid: the
//! comparison is bit-exact by construction.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `+`, `/`, the
//! `max` built-in, unsigned comparisons and index arithmetic. There is no
//! `sqrt`, no transcendental call, no `u64` and no optional device feature, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. The only loop is the
//! block-local segment scan, whose bound is the element's offset within its
//! block and thus at most `block_size` iterations, so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_segmented`；无第三方引擎源码或衍生代码。
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

/// Byte stride of a scalar `u32` `std430` storage element, mirroring the golden
/// `U32_STRIDE`.
const U32_STRIDE: usize = 4;

/// Total byte size of a `std430` storage buffer holding `count` `u32` elements,
/// clamped up to a single element because a `WebGPU` storage binding may not be
/// zero-sized. Mirrors the golden `storage_bytes` with `U32_STRIDE`.
fn storage_bytes(count: usize) -> usize {
    U32_STRIDE.saturating_mul(count.max(1))
}

/// The segmented-scan kernel, mirroring the `CPU` golden
/// [`gpu_scan_segmented`](prism_render_architecture::particle::gpu_scan_segmented).
/// The single entry point `solve` resolves one element per thread, embedded
/// inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_segmented`。
const GPU_SCAN_SEGMENTED_WGSL: &str = r#"
// gpu_scan_segmented twin: one thread per element reproduces the CPU golden
// `particle::gpu_scan_segmented`. The host uploads the coarse per-block
// carry-ins (pass two, the open-segment scan of per-block trailing sums); each
// thread runs the block-local segment scan over its own block up to its index,
// learns from that same loop whether a head flag sits at or before it, folds in
// its block carry-in only when no head severed it, and finally adds its own
// value for the inclusive variant. All arithmetic is unsigned integer, so the
// device `+` wraps exactly as the golden `wrapping_add`, agreeing bit for bit.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::gpu_scan_segmented;
// 无第三方引擎源码或衍生代码。

// Dispatch parameters. 16-byte uniform block: the element count, the block
// size, the inclusive flag, and one pad word, matching the host `Params`.
struct Params {
    count: u32,
    block_size: u32,
    inclusive: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> carry_ins: array<u32>;
@group(0) @binding(4) var<storage, read_write> scanned: array<u32>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let block_size = max(params.block_size, 1u);
    let block = idx / block_size;
    let block_start = block * block_size;

    // Block-local exclusive segment scan: starting from the neutral element at
    // the block start, accumulate values up to (but excluding) `idx`, resetting
    // the accumulator at every head flag. `head_seen` records whether any head
    // flag sits in [block_start, idx] — if so, that head severed the carry. The
    // loop runs at most block_size iterations and is bounded, so the kernel
    // provably terminates.
    var acc: u32 = 0u;
    var head_seen: bool = false;
    for (var k: u32 = block_start; k <= idx; k = k + 1u) {
        if (flags[k] != 0u) {
            acc = 0u;
            head_seen = true;
        }
        if (k == idx) {
            break;
        }
        acc = acc + values[k];
    }

    // Fold the block carry-in into this element only when no head flag at or
    // before it severed the carry (pass three), then add the element itself for
    // the inclusive variant. Both adds wrap exactly like the golden.
    var folded: u32 = acc;
    if (!head_seen) {
        folded = folded + carry_ins[block];
    }
    if (params.inclusive != 0u) {
        folded = folded + values[idx];
    }
    scanned[idx] = folded;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`GPU_SCAN_SEGMENTED_WGSL`]: the element `count`, the
/// `block_size`, the `inclusive` flag and one pad word — `16` bytes with no
/// interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of elements (valid threads).
    count: u32,
    /// Block size each simulated `workgroup` segment-scans.
    block_size: u32,
    /// `1` for the inclusive variant, `0` for the exclusive variant.
    inclusive: u32,
    /// Padding word.
    pad0: u32,
}

/// One segmented-scan request: the per-element `values`, the per-element head
/// `flags` (any non-zero value opens a new segment, `0` continues the current
/// one), the `block_size` each simulated `workgroup` segment-scans, and whether
/// the scan is `inclusive`.
///
/// The `flags` array may be shorter than `values`; the missing tail reads as
/// non-head, exactly as the golden `is_head` tolerates ragged input. The
/// `block_size` is clamped to at least one on dispatch so a degenerate zero can
/// never cause a divide-by-zero, exactly as the golden `SegmentedScanConfig`
/// clamps it.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_segmented`。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuSegmentedScanQuery {
    /// Per-element values to scan.
    pub values: Vec<u32>,
    /// Per-element head flags: any non-zero value opens a new segment.
    pub flags: Vec<u32>,
    /// Elements segment-scanned per block (per simulated `workgroup`).
    pub block_size: u32,
    /// `true` for an inclusive scan, `false` for an exclusive scan.
    pub inclusive: bool,
}

/// Result of a block-decomposed segmented scan, mirroring the golden
/// `SegmentedScanResult`: the per-element scanned offsets (`GPU`) and the
/// per-block carry-ins (host coarse `scan`).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_segmented`。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuSegmentedScanResult {
    /// Per-element prefix within its segment, in input order (exclusive or
    /// inclusive depending on the query's `inclusive` flag).
    pub scanned: Vec<u32>,
    /// Per-block carry-in: the open-segment sum flowing into each block from the
    /// blocks before it, one entry per block. A block whose first element is a
    /// head flag ignores its carry-in.
    pub block_carry_ins: Vec<u32>,
}

/// Configuration for a block-decomposed segmented scan: the block size each
/// simulated `workgroup` segment-scans, with the shared `std430` byte helpers.
///
/// Mirrors the golden `SegmentedScanConfig`: `new` clamps the block size to at
/// least one, and the byte helpers follow the shared stride rule so an empty
/// pool still reserves a single element.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_segmented`。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuSegmentedScanConfig {
    /// Elements segment-scanned per block.
    pub block_size: u32,
}

impl GpuSegmentedScanConfig {
    /// Builds a config with the given block size, clamped to at least one so a
    /// degenerate zero can never cause a divide-by-zero.
    #[must_use]
    pub fn new(block_size: u32) -> Self {
        Self {
            block_size: block_size.max(1),
        }
    }

    /// Whether the configured block size is a power of two (the natural
    /// `workgroup` width for a shared-memory segmented scan).
    #[must_use]
    pub fn is_power_of_two_block(self) -> bool {
        self.block_size.is_power_of_two()
    }

    /// Number of blocks needed to cover `len` elements at this block size.
    #[must_use]
    pub fn num_blocks(self, len: usize) -> usize {
        len.div_ceil(self.block_size.max(1) as usize)
    }

    /// Byte size of the `std430` storage buffer holding `len` scanned `u32`
    /// offsets.
    #[must_use]
    pub fn scanned_bytes(self, len: usize) -> usize {
        storage_bytes(len)
    }

    /// Byte size of the `std430` storage buffer holding the `len` head flags
    /// (one `u32` each) the segmented scan reads alongside its values.
    #[must_use]
    pub fn flag_bytes(self, len: usize) -> usize {
        storage_bytes(len)
    }

    /// Byte size of the `std430` storage buffer holding the per-block carry-in
    /// (`u32` each) for `len` input elements.
    #[must_use]
    pub fn block_carry_bytes(self, len: usize) -> usize {
        storage_bytes(self.num_blocks(len))
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

/// Pads `flags` to `len` with non-head zeros, mirroring the golden `is_head`
/// reading a ragged flag array's missing tail as "not a head".
fn padded_flags(flags: &[u32], len: usize) -> Vec<u32> {
    let mut out = vec![0u32; len];
    let copy = flags.len().min(len);
    out[..copy].copy_from_slice(&flags[..copy]);
    out
}

/// Host-side coarse `scan` (pass two): the per-block carry-ins uploaded to the
/// device.
///
/// Entry `b` is the open-segment sum entering block `b` from the blocks before
/// it. A block that holds a head flag restarts the carry from its own trailing
/// open-sum (the tail after its last head flag); a block with none extends the
/// inherited open segment. Accumulation uses `wrapping_add`, matching the golden
/// pass two, so the uploaded carries agree with the reference bit for bit.
fn block_carry_ins(values: &[u32], flags: &[u32], block_size: usize) -> Vec<u32> {
    let block = block_size.max(1);
    let len = values.len();
    let num_blocks = len.div_ceil(block);
    let mut out = Vec::with_capacity(num_blocks);
    let mut carry = 0u32;
    for block_idx in 0..num_blocks {
        let start = block_idx * block;
        let end = (start + block).min(len);
        let mut acc = 0u32;
        let mut has_head = false;
        for (k, &value) in values[start..end].iter().enumerate() {
            if flags.get(start + k).copied().unwrap_or(0) != 0 {
                acc = 0;
                has_head = true;
            }
            acc = acc.wrapping_add(value);
        }
        out.push(carry);
        if has_head {
            carry = acc;
        } else {
            carry = carry.wrapping_add(acc);
        }
    }
    out
}

/// A compiled, reusable segmented-scan compute pipeline, twinning the `CPU`
/// golden [`gpu_scan_segmented`](prism_render_architecture::particle::gpu_scan_segmented).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_segmented`。
pub struct GpuSegmentedScan {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSegmentedScan {
    /// Compiles the segmented-scan kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_segmented`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSegmentedScan {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gpu_scan_segmented_module"),
            source: ShaderSource::Wgsl(GPU_SCAN_SEGMENTED_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gpu_scan_segmented_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gpu_scan_segmented_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gpu_scan_segmented_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSegmentedScan {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the block-decomposed segmented scan on the device and reads back the
    /// per-element scanned offsets plus the per-block carry-ins.
    ///
    /// The returned [`GpuSegmentedScanResult::scanned`] equals the golden
    /// `scanned` element for element and
    /// [`GpuSegmentedScanResult::block_carry_ins`] equals the golden
    /// `block_carry_ins` block for block, because every per-element answer is
    /// pure integer bit algebra. An empty `values` batch returns an empty result
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_segmented`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        query: &GpuSegmentedScanQuery,
    ) -> GpuSegmentedScanResult {
        let count = query.values.len();
        if count == 0 {
            return GpuSegmentedScanResult {
                scanned: Vec::new(),
                block_carry_ins: Vec::new(),
            };
        }
        let device = ctx.device();

        let block_size = query.block_size.max(1);
        let flags = padded_flags(&query.flags, count);
        let carries = block_carry_ins(&query.values, &query.flags, block_size as usize);

        let params = Params {
            count: count as u32,
            block_size,
            inclusive: u32::from(query.inclusive),
            pad0: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_scan_segmented_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let values_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_scan_segmented_values"),
            contents: bytemuck::cast_slice(&query.values),
            usage: BufferUsages::STORAGE,
        });
        let flags_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_scan_segmented_flags"),
            contents: bytemuck::cast_slice(&flags),
            usage: BufferUsages::STORAGE,
        });
        let carries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_scan_segmented_carries"),
            contents: bytemuck::cast_slice(&carries),
            usage: BufferUsages::STORAGE,
        });

        let scanned_bytes = (count as u64) * (U32_STRIDE as u64);
        let scanned_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_scan_segmented_scanned"),
            size: scanned_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scanned_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_scan_segmented_scanned_stage"),
            size: scanned_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gpu_scan_segmented_bind_group"),
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
                    resource: flags_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: carries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: scanned_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_gpu_scan_segmented_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gpu_scan_segmented_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per element, flattened to a 1-D dispatch.
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&scanned_buf, 0, &scanned_stage, 0, scanned_bytes);
        ctx.queue().submit([encoder.finish()]);

        scanned_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let scanned_view = scanned_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let scanned = bytemuck::cast_slice::<u8, u32>(&scanned_view).to_vec();
        drop(scanned_view);
        scanned_stage.unmap();

        debug_assert_eq!(scanned.len(), count);
        GpuSegmentedScanResult {
            scanned,
            block_carry_ins: carries,
        }
    }
}
