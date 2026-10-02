//! `wgpu` compute twin of the device-free indirect compute-dispatch argument
//! packing
//! ([`indirect_dispatch`](prism_render_architecture::particle::indirect_dispatch),
//! particle design §9, §15).
//!
//! The `CPU` golden
//! [`indirect_dispatch`](prism_render_architecture::particle::indirect_dispatch)
//! owns the `std430` layout a `GPU` fill kernel writes before a dependent stage
//! dispatches from it with `dispatchWorkgroupsIndirect`: the three tightly
//! packed `u32` workgroup counts of a
//! [`DispatchIndirectCommand`](prism_render_architecture::particle::indirect_dispatch::DispatchIndirectCommand)
//! (`stride` `12` bytes) and the
//! [`from_element_count`](prism_render_architecture::particle::indirect_dispatch::DispatchIndirectCommand::from_element_count)
//! rule that turns a resolved element count into a 1-D launch size.
//! [`GpuIndirectDispatch`] is the on-device twin: one thread resolves one
//! [`GpuIndirectDispatchQuery`] — an (`element_count`, `workgroup_size`) pair —
//! into one [`GpuIndirectDispatchResult`] carrying the three workgroup counts
//! `x`, `y`, `z` plus the `is_empty` flag, so a passing real-device parity test
//! is direct evidence the ported kernel reproduces the exact `ceil`-division and
//! degenerate guards the reference publishes, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Each query carries an `element_count` and a `workgroup_size`. The kernel
//! reproduces the golden
//! [`from_element_count`](prism_render_architecture::particle::indirect_dispatch::DispatchIndirectCommand::from_element_count):
//! `x = ceil(element_count / workgroup_size)` with `y = z = 1`, guarded so a
//! zero `workgroup_size` writes `x = 0` instead of dividing by zero. It also
//! reproduces the golden
//! [`is_empty`](prism_render_architecture::particle::indirect_dispatch::DispatchIndirectCommand::is_empty)
//! predicate as the discrete `1`/`0` flag `x == 0 || y == 0 || z == 0` yields.
//! These three counts are exactly the words a device writes into an indirect
//! argument buffer, so the twin covers the real on-device computation.
//!
//! The `GPU` kernel deliberately stops at the per-record `u32` words. The
//! golden
//! [`total_workgroups`](prism_render_architecture::particle::indirect_dispatch::DispatchIndirectCommand::total_workgroups)
//! is an `x * y * z` product computed in `u64` to never wrap a `u32`; `WGSL`
//! has no `u64`, so that host convenience aggregate is **not** twinned on the
//! device. Instead [`GpuIndirectDispatch::evaluate`] folds the device `u32`
//! words into the `u64` product on the host, mirroring how the occlusion twin
//! keeps a `u64` roll-up on the `CPU` while only the per-thread `u32` core runs
//! on the `GPU`.
//!
//! # Correctness model
//!
//! Every twinned value is a `u32` count or a discrete `1`/`0` flag: the launch
//! size is an unsigned overflow-safe `ceil`-division (`quotient` plus a
//! `remainder != 0` bump) and the empty flag is an unsigned comparison. `CPU`
//! and `GPU` therefore compute identical bit patterns, and the parity test
//! asserts an exact `==` on every field with no tolerance. Fixtures keep every
//! `element_count` and `workgroup_size` well inside `u32` and away from the
//! `ceil`-division tie, covering exact-multiple and remainder cases with their
//! own deterministic expectations.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `/`, `%`,
//! `+`, an unsigned compare and index arithmetic. There is no `sqrt`, no
//! transcendental call, no `u64` and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread
//! performs a fixed, bounded sequence of integer work, so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_dispatch`；无第三方引擎源码或衍生代码。
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

/// The indirect-dispatch argument kernel, mirroring the `CPU` golden
/// [`indirect_dispatch`](prism_render_architecture::particle::indirect_dispatch)
/// record for record. The single entry point `solve` resolves one query per
/// thread, embedded inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_dispatch`。
const INDIRECT_DISPATCH_WGSL: &str = r#"
// indirect_dispatch twin: one thread per query reproduces the CPU golden
// `particle::indirect_dispatch`. A query is an (element_count, workgroup_size)
// pair; the kernel reproduces `DispatchIndirectCommand::from_element_count`:
// x = ceil(element_count / workgroup_size) with y = z = 1, guarded so a zero
// workgroup_size writes x = 0 instead of dividing by zero. It also reproduces
// `is_empty` as the 1u/0u flag `x == 0 || y == 0 || z == 0` yields. The u64
// `total_workgroups` product the golden exposes as a host convenience is NOT
// computed here: WGSL has no u64, so the host folds the per-record u32 words
// into that product after readback.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::indirect_dispatch；无第三方
// 引擎源码或衍生代码。

// Dispatch parameters. 16-byte uniform block: the valid query count plus three
// pad words, matching the host `Params`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 8-byte std430 stride of two scalar words, matching the host
// `GpuIndirectDispatchQuery`: the resolved element count and the dependent
// stage's threads-per-workgroup.
struct DispatchQuery {
    element_count: u32,
    workgroup_size: u32,
}

// One record. 16-byte std430 stride of four scalar words, matching the host
// `IndirectRecord`: the three workgroup counts and the empty flag.
struct DispatchRecord {
    x: u32,
    y: u32,
    z: u32,
    is_empty: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<DispatchQuery>;
@group(0) @binding(2) var<storage, read_write> records: array<DispatchRecord>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // ceil-division, guarded against a zero workgroup size and written in the
    // overflow-safe quotient/remainder form so it matches u32::div_ceil exactly
    // even near u32::MAX.
    var x: u32 = 0u;
    if (q.workgroup_size != 0u) {
        let quotient = q.element_count / q.workgroup_size;
        let remainder = q.element_count % q.workgroup_size;
        if (remainder != 0u) {
            x = quotient + 1u;
        } else {
            x = quotient;
        }
    }
    let y: u32 = 1u;
    let z: u32 = 1u;

    var is_empty: u32 = 0u;
    if (x == 0u || y == 0u || z == 0u) {
        is_empty = 1u;
    }

    var out: DispatchRecord;
    out.x = x;
    out.y = y;
    out.z = z;
    out.is_empty = is_empty;
    records[idx] = out;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`INDIRECT_DISPATCH_WGSL`]: the valid query count plus three pad
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

/// One raw device record read back from the kernel. `repr(C)` `std430` layout
/// matching the `WGSL` `DispatchRecord`: the three `u32` workgroup counts and
/// the `is_empty` flag — `16` bytes with no interior padding. This stays
/// private; the public [`GpuIndirectDispatchResult`] wraps it and adds the
/// host-folded `u64` product.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_dispatch`。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IndirectRecord {
    /// Workgroups launched along `x`.
    x: u32,
    /// Workgroups launched along `y` (always `1` for these 1-D launches).
    y: u32,
    /// Workgroups launched along `z` (always `1` for these 1-D launches).
    z: u32,
    /// Empty flag: `1` when any axis is zero, `0` otherwise.
    is_empty: u32,
}

/// One query for the indirect-dispatch twin: a resolved `element_count` and the
/// dependent stage's `workgroup_size`.
///
/// `element_count` is the number of domain elements the launch must cover
/// (typically a `GPU` append counter resolved by an earlier pass), and
/// `workgroup_size` is the threads-per-workgroup the dependent stage launches
/// with. The twin reproduces the golden
/// [`from_element_count`](prism_render_architecture::particle::indirect_dispatch::DispatchIndirectCommand::from_element_count):
/// `x = ceil(element_count / workgroup_size)`, with a zero `workgroup_size`
/// guarded to `x = 0`. The `repr(C)` layout — two `u32` words, `8` bytes with
/// no padding — matches the `WGSL` `DispatchQuery` struct exactly, so it is
/// uploaded to the device without a separate encode step.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_dispatch`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuIndirectDispatchQuery {
    /// Number of domain elements the launch must cover.
    pub element_count: u32,
    /// Threads per workgroup the dependent stage launches with.
    pub workgroup_size: u32,
}

impl GpuIndirectDispatchQuery {
    /// A query covering `element_count` elements at `workgroup_size` threads per
    /// group.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_dispatch`。
    #[must_use]
    pub const fn new(element_count: u32, workgroup_size: u32) -> GpuIndirectDispatchQuery {
        GpuIndirectDispatchQuery {
            element_count,
            workgroup_size,
        }
    }
}

/// One resolved answer for a single [`GpuIndirectDispatchQuery`], mirroring the
/// golden `x`, `y`, `z` workgroup counts and the `is_empty` flag.
///
/// The three counts `x`, `y`, `z` are read straight from the device record; the
/// `is_empty` flag is the device's discrete `1`/`0` output decoded to a `bool`.
/// `total_workgroups` is **not** a device output: the kernel stops at the
/// per-record `u32` words because `WGSL` has no `u64`, so this `x * y * z`
/// product is folded on the host in `u64` exactly as the golden
/// [`total_workgroups`](prism_render_architecture::particle::indirect_dispatch::DispatchIndirectCommand::total_workgroups)
/// does, where it can never wrap a `u32`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_dispatch`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuIndirectDispatchResult {
    /// Workgroups launched along `x`: `ceil(element_count / workgroup_size)`.
    pub x: u32,
    /// Workgroups launched along `y` (always `1` for these 1-D launches).
    pub y: u32,
    /// Workgroups launched along `z` (always `1` for these 1-D launches).
    pub z: u32,
    /// `true` when any axis is zero, i.e. the dispatch launches no workgroups.
    pub is_empty: bool,
    /// Total workgroups `x * y * z`, folded on the host in `u64` so the product
    /// can never overflow a `u32`.
    pub total_workgroups: u64,
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

/// A compiled, reusable indirect-dispatch argument compute pipeline, twinning
/// the `CPU` golden
/// [`indirect_dispatch`](prism_render_architecture::particle::indirect_dispatch).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_dispatch`。
pub struct GpuIndirectDispatch {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuIndirectDispatch {
    /// Compiles the indirect-dispatch argument kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_dispatch`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuIndirectDispatch {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_indirect_dispatch_module"),
            source: ShaderSource::Wgsl(INDIRECT_DISPATCH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_indirect_dispatch_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_indirect_dispatch_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_indirect_dispatch_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuIndirectDispatch {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`GpuIndirectDispatchResult`] per input, in order.
    ///
    /// Each result's `x`, `y`, `z` and `is_empty` equal the matching golden
    /// [`DispatchIndirectCommand`](prism_render_architecture::particle::indirect_dispatch::DispatchIndirectCommand)
    /// fields exactly, because the whole on-device path is unsigned integer
    /// algebra. `total_workgroups` is the host-folded `u64` product of the
    /// device `x`, `y`, `z` words, matching the golden
    /// [`total_workgroups`](prism_render_architecture::particle::indirect_dispatch::DispatchIndirectCommand::total_workgroups).
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_dispatch`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GpuIndirectDispatchQuery],
    ) -> Vec<GpuIndirectDispatchResult> {
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
            label: Some("prism_volumetric_indirect_dispatch_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_indirect_dispatch_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (queries.len() as u64) * (size_of::<IndirectRecord>() as u64);
        let records_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_indirect_dispatch_records"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let records_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_indirect_dispatch_records_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_indirect_dispatch_bind_group"),
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
                    resource: records_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_indirect_dispatch_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_indirect_dispatch_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&records_buf, 0, &records_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        records_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = records_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let records = bytemuck::cast_slice::<u8, IndirectRecord>(&view).to_vec();
        drop(view);
        records_stage.unmap();
        debug_assert_eq!(records.len(), queries.len());

        records
            .into_iter()
            .map(|r| GpuIndirectDispatchResult {
                x: r.x,
                y: r.y,
                z: r.z,
                is_empty: r.is_empty != 0,
                // u64 fold the device u32 words; the product can never wrap.
                total_workgroups: u64::from(r.x) * u64::from(r.y) * u64::from(r.z),
            })
            .collect()
    }
}
