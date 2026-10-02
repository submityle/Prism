//! `wgpu` compute twin of the device-free snapshot *byte-layout contract*
//! ([`serialization`](prism_render_architecture::particle::serialization),
//! particle design §5, §11, §29).
//!
//! The `CPU` golden
//! [`serialization`](prism_render_architecture::particle::serialization) owns
//! the `CPU`-verifiable layout contract of an Ember snapshot: the magic /
//! version words, the per-attribute `std430` channel strides, the fixed-size
//! header, the per-channel and total byte accounting, and the version-migration
//! decision. [`GpuSerialization`] is the on-device twin: one thread resolves
//! one [`GpuSerializationQuery`] into one [`GpuSerializationResult`], so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same channel size, total size and guard classification the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries an `attr_code` (a
//! [`SnapshotAttribute`](prism_render_architecture::particle::serialization::SnapshotAttribute)
//! variant in the golden `ALL` order), a channel `capacity`, an
//! `attribute_mask` bit pattern, and a header `magic` / `version` pair. The
//! kernel reproduces three golden answers in one pass:
//!
//! * `channel_bytes` mirrors
//!   [`attribute_channel_bytes`](prism_render_architecture::particle::serialization::attribute_channel_bytes)
//!   as `stride(attr) * max(capacity, 1)` — the same clamp-to-one-element rule a
//!   non-empty `WebGPU` storage binding needs.
//! * `total_bytes` mirrors
//!   [`snapshot_total_bytes`](prism_render_architecture::particle::serialization::snapshot_total_bytes)
//!   by summing the fixed `32`-byte header with every set attribute channel,
//!   walking a fixed, bounded loop over the known
//!   [`SnapshotAttribute::ALL`](prism_render_architecture::particle::serialization::SnapshotAttribute::ALL)
//!   bits exactly as the golden `iter_set` walk does.
//! * `guard_code` / `guard_from` / `guard_to` mirror
//!   [`guard`](prism_render_architecture::particle::serialization::guard) as a
//!   discrete classification code ([`CODE_COMPATIBLE`],
//!   [`CODE_NEEDS_MIGRATION`], [`CODE_INCOMPATIBLE`]) plus the migration range.
//!
//! The variable-length `iter_set` collector is intentionally not twinned; the
//! total-bytes kernel instead re-derives the same sum from the mask with a
//! fixed-count loop that a shader can run unconditionally.
//!
//! # Correctness model
//!
//! Every value is a `u32` byte size, a classification code or a migration-range
//! word: the accounting is pure unsigned multiply-add-and-max with no rounding,
//! and the guard is a discrete classification. `CPU` and `GPU` therefore
//! compute identical bit patterns, and the parity test asserts an exact `==` on
//! every field with no tolerance. Fixtures keep `header + sum(stride *
//! capacity)` well below `2^31`, so the device `u32` arithmetic never wraps
//! where the golden `saturating_add` / `saturating_mul` would otherwise clamp.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `*`, `+`, the
//! `max` built-in, bit `AND`, a left shift, and unsigned comparisons. There is
//! no `sqrt`, no divide, no transcendental call, no `u64` and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The single
//! loop runs a fixed `8` iterations, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::serialization`；无第三方引擎源码或衍生代码。
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

/// Guard code for a directly restorable snapshot (same magic and version),
/// mirroring
/// [`VersionGuard::Compatible`](prism_render_architecture::particle::serialization::VersionGuard::Compatible).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::serialization`。
pub const CODE_COMPATIBLE: u32 = 0;

/// Guard code for an older snapshot that must run the migration path, mirroring
/// [`VersionGuard::NeedsMigration`](prism_render_architecture::particle::serialization::VersionGuard::NeedsMigration);
/// the result's `guard_from` / `guard_to` carry the migration range.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::serialization`。
pub const CODE_NEEDS_MIGRATION: u32 = 1;

/// Guard code for a snapshot this build refuses — wrong magic or a future
/// version — mirroring
/// [`VersionGuard::Incompatible`](prism_render_architecture::particle::serialization::VersionGuard::Incompatible).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::serialization`。
pub const CODE_INCOMPATIBLE: u32 = 2;

/// The snapshot layout kernel, mirroring the `CPU` golden
/// [`serialization`](prism_render_architecture::particle::serialization) field
/// for field. The single entry point `solve` resolves one query per thread,
/// embedded inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::serialization`。
const GPU_SERIALIZATION_WGSL: &str = r#"
// serialization twin: one thread per query reproduces the CPU golden
// `particle::serialization`. `channel_bytes` is `stride(attr) *
// max(capacity, 1u)` — the clamp-to-one-element rule a non-empty WebGPU storage
// binding needs. `total_bytes` is the 32-byte header plus every set attribute
// channel, summed with a fixed 8-iteration loop over the known
// SnapshotAttribute bits (ascending bit == ALL order), matching the golden
// `iter_set` walk over known bits only. `guard_code` classifies the header
// magic/version into 0u (Compatible), 1u (NeedsMigration, with from/to range)
// or 2u (Incompatible). Pure u32 arithmetic: multiplies, adds, a `max`, a shift
// and unsigned compares. There is no sqrt, no divide, no transcendental call
// and no u64, so the kernel runs unmodified on Metal, Vulkan and DX12. The one
// loop runs a fixed 8 iterations, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::serialization；无第三方
// 引擎源码或衍生代码。

// std430 per-attribute strides, matching the golden `U32_STRIDE`,
// `VEC2_STRIDE` and `VEC4_STRIDE`.
const U32_STRIDE: u32 = 4u;
const VEC2_STRIDE: u32 = 8u;
const VEC4_STRIDE: u32 = 16u;

// The fixed header size: `HEADER_WORD_COUNT` (8) scalar words of U32_STRIDE.
const HEADER_BYTES: u32 = 32u;

// Number of known SnapshotAttribute channels (the golden `ALL` length).
const ATTRIBUTE_COUNT: u32 = 8u;

// Snapshot format identity, matching the golden `SNAPSHOT_MAGIC` (ASCII `PRSN`)
// and `SNAPSHOT_VERSION`.
const SNAPSHOT_MAGIC: u32 = 0x5052534Eu;
const SNAPSHOT_VERSION: u32 = 1u;

// Guard classification codes, matching the host constants.
const GUARD_COMPATIBLE: u32 = 0u;
const GUARD_NEEDS_MIGRATION: u32 = 1u;
const GUARD_INCOMPATIBLE: u32 = 2u;

// Dispatch parameters. 16-byte uniform block: the valid query count plus three
// pad words, matching the host `Params`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 20-byte std430 stride of five scalar words, matching the host
// `GpuSerializationQuery`.
struct SnapshotQuery {
    attr_code: u32,
    capacity: u32,
    attribute_mask: u32,
    header_magic: u32,
    header_version: u32,
}

// One result. 20-byte std430 stride of five scalar words, matching the host
// `GpuSerializationResult`.
struct SnapshotResult {
    channel_bytes: u32,
    total_bytes: u32,
    guard_code: u32,
    guard_from: u32,
    guard_to: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<SnapshotQuery>;
@group(0) @binding(2) var<storage, read_write> results: array<SnapshotResult>;

// The std430 per-particle stride of attribute `attr_code`, in the golden `ALL`
// order: Position(0u), Velocity(1u) and Color(4u) are vec4; Size(5u) is vec2;
// Age(2u), Lifetime(3u), Rotation(6u) and Custom(7u) are scalar u32.
fn stride_of(attr_code: u32) -> u32 {
    if (attr_code == 0u || attr_code == 1u || attr_code == 4u) {
        return VEC4_STRIDE;
    }
    if (attr_code == 5u) {
        return VEC2_STRIDE;
    }
    return U32_STRIDE;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // A non-empty storage binding reserves at least one element, so the
    // capacity is clamped up to one before any multiply.
    let effective_capacity = max(q.capacity, 1u);

    var out: SnapshotResult;

    // attribute_channel_bytes: this attribute's channel size.
    out.channel_bytes = stride_of(q.attr_code) * effective_capacity;

    // snapshot_total_bytes: the fixed header plus every set attribute channel.
    // The loop runs a fixed ATTRIBUTE_COUNT iterations over the known attribute
    // bits (ascending bit == ALL order), so it provably terminates and matches
    // the golden `iter_set` walk over the known bits only.
    var total: u32 = HEADER_BYTES;
    for (var i: u32 = 0u; i < ATTRIBUTE_COUNT; i = i + 1u) {
        let bit = 1u << i;
        if ((q.attribute_mask & bit) != 0u) {
            total = total + stride_of(i) * effective_capacity;
        }
    }
    out.total_bytes = total;

    // guard: classify the header magic/version against the current format. A
    // mismatched magic is Incompatible; an older version NeedsMigration (with
    // the from/to range); the current version Compatible; a future version
    // Incompatible.
    var code: u32 = GUARD_INCOMPATIBLE;
    var from_version: u32 = 0u;
    var to_version: u32 = 0u;
    if (q.header_magic == SNAPSHOT_MAGIC) {
        if (q.header_version < SNAPSHOT_VERSION) {
            code = GUARD_NEEDS_MIGRATION;
            from_version = q.header_version;
            to_version = SNAPSHOT_VERSION;
        } else if (q.header_version == SNAPSHOT_VERSION) {
            code = GUARD_COMPATIBLE;
        } else {
            code = GUARD_INCOMPATIBLE;
        }
    }
    out.guard_code = code;
    out.guard_from = from_version;
    out.guard_to = to_version;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`GPU_SERIALIZATION_WGSL`]: the valid query count plus three pad
/// words — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query for the snapshot layout twin: a
/// [`SnapshotAttribute`](prism_render_architecture::particle::serialization::SnapshotAttribute)
/// `attr_code`, a channel `capacity`, an `attribute_mask` bit pattern, and a
/// header `magic` / `version` pair.
///
/// The `attr_code` encodes a
/// [`SnapshotAttribute`](prism_render_architecture::particle::serialization::SnapshotAttribute)
/// variant in the golden
/// [`SnapshotAttribute::ALL`](prism_render_architecture::particle::serialization::SnapshotAttribute::ALL)
/// order (`0` for `Position` through `7` for `Custom`), which is also the bit
/// index that attribute occupies in `attribute_mask`. The `repr(C)` layout —
/// five `u32` words, `20` bytes with no padding — matches the `WGSL`
/// `SnapshotQuery` struct exactly, so it is uploaded to the device without a
/// separate encode step.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::serialization`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSerializationQuery {
    /// Attribute index in the golden `ALL` order (`0` = `Position`, …, `7` =
    /// `Custom`); also the bit index in `attribute_mask`.
    pub attr_code: u32,
    /// Channel length in particles; clamped up to one element for sizing.
    pub capacity: u32,
    /// Bit pattern selecting which attribute channels the snapshot captures.
    pub attribute_mask: u32,
    /// The snapshot header's format magic word.
    pub header_magic: u32,
    /// The snapshot header's layout version word.
    pub header_version: u32,
}

/// One resolved answer for a single [`GpuSerializationQuery`], mirroring the
/// golden `attribute_channel_bytes`, `snapshot_total_bytes` and `guard`
/// outputs.
///
/// The `repr(C)` layout — five `u32` words, `20` bytes — matches the `WGSL`
/// `SnapshotResult` struct exactly, so device results are read back without a
/// separate decode step. `guard_code` is one of [`CODE_COMPATIBLE`],
/// [`CODE_NEEDS_MIGRATION`] or [`CODE_INCOMPATIBLE`]; `guard_from` / `guard_to`
/// carry the migration range and are `0` unless `guard_code` is
/// [`CODE_NEEDS_MIGRATION`].
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::serialization`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSerializationResult {
    /// This attribute's channel size, matching the golden
    /// [`attribute_channel_bytes`](prism_render_architecture::particle::serialization::attribute_channel_bytes).
    pub channel_bytes: u32,
    /// The whole snapshot's byte size, matching the golden
    /// [`snapshot_total_bytes`](prism_render_architecture::particle::serialization::snapshot_total_bytes).
    pub total_bytes: u32,
    /// The guard classification code (see [`CODE_COMPATIBLE`]).
    pub guard_code: u32,
    /// Migration source version, non-zero only for [`CODE_NEEDS_MIGRATION`].
    pub guard_from: u32,
    /// Migration target version, non-zero only for [`CODE_NEEDS_MIGRATION`].
    pub guard_to: u32,
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

/// A compiled, reusable snapshot layout compute pipeline, twinning the `CPU`
/// golden [`serialization`](prism_render_architecture::particle::serialization).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::serialization`。
pub struct GpuSerialization {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSerialization {
    /// Compiles the snapshot layout kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::serialization`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSerialization {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_serialization"),
            source: ShaderSource::Wgsl(GPU_SERIALIZATION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_serialization_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_serialization_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_serialization_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSerialization {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`GpuSerializationResult`] per input, in order.
    ///
    /// Each result equals the matching golden triple exactly — `channel_bytes`
    /// mirrors
    /// [`attribute_channel_bytes`](prism_render_architecture::particle::serialization::attribute_channel_bytes),
    /// `total_bytes` mirrors
    /// [`snapshot_total_bytes`](prism_render_architecture::particle::serialization::snapshot_total_bytes),
    /// and `guard_code` mirrors
    /// [`guard`](prism_render_architecture::particle::serialization::guard) —
    /// because the whole path is integer bit algebra. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::serialization`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GpuSerializationQuery],
    ) -> Vec<GpuSerializationResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_serialization_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_serialization_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (queries.len() as u64) * (size_of::<GpuSerializationResult>() as u64);
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_serialization_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_serialization_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_serialization_bind_group"),
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

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_serialization_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_serialization_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuSerializationResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
    }
}
