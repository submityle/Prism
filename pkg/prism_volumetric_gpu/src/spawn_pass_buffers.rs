//! `wgpu` compute twin of the device-free `std430`/`std140` `Spawn`/`Emit`-pass
//! bind-group contract
//! ([`spawn_pass_buffers`](prism_render_architecture::particle::spawn_pass_buffers),
//! particle design §9 step 2).
//!
//! The `CPU` golden
//! [`spawn_pass_buffers`](prism_render_architecture::particle::spawn_pass_buffers)
//! owns one storage/uniform buffer enum for the `Spawn` pass and reports, for
//! each `@group(0)` binding, its `@binding` index, element stride, binding
//! kind (`storage` versus `uniform`), access mode, extent-sized element count
//! and clamped total byte size against a
//! [`SpawnPassExtent`](prism_render_architecture::particle::spawn_pass_buffers::SpawnPassExtent),
//! plus the pass-level
//! [`total_storage_bytes`](prism_render_architecture::particle::spawn_pass_buffers::total_storage_bytes)
//! of the `std430` pools. [`GpuSpawnPassBuffers`] is the on-device twin: one
//! thread resolves one [`GpuSpawnPassBuffersQuery`] — a `variant_code` plus the
//! three `SpawnPassExtent` counts — into one [`GpuSpawnPassBuffersResult`]
//! carrying all seven answers, so a passing real-device parity test is direct
//! evidence the ported kernel reproduces the same `ABI` the reference
//! publishes, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries a `variant_code` (the index into
//! [`SpawnPassBuffer::ALL`](prism_render_architecture::particle::spawn_pass_buffers::SpawnPassBuffer),
//! which equals the `@binding` index) and the extent's `capacity`,
//! `spawn_count` and `emitter_count`. The kernel reproduces the golden `match`
//! tables for `binding`, `stride`, `kind` (as the `0`/`1` code
//! [`SpawnBindingKind`](prism_render_architecture::particle::spawn_pass_buffers::SpawnBindingKind)
//! declares), `access` (as the `0`/`1` code
//! [`ParticleBufferAccess`](prism_render_architecture::particle::gpu_layout::ParticleBufferAccess)
//! uses), `element_count` and `byte_size` (the clamp-to-one
//! `stride * max(count, 1)` rule). The free-list and both attribute pools scale
//! with `capacity`; the counter block is the fixed
//! [`SPAWN_COUNTER_COUNT`](prism_render_architecture::particle::spawn_pass_buffers::SPAWN_COUNTER_COUNT);
//! the spawn-index queue scales with `spawn_count`; the uniform scales with
//! `emitter_count`. The pass-level `total_storage_bytes` sums the five `std430`
//! storage buffers, excluding the `std140` uniform.
//!
//! # Correctness model
//!
//! Every value is a `u32`, a classification code or a `bool`-flavoured flag:
//! the byte size is pure unsigned multiply-and-`max` with no rounding, and the
//! binding, stride, kind code and access code are discrete classifications.
//! `CPU` and `GPU` therefore compute identical bit patterns, and the parity
//! test asserts an exact `==` on every field with no tolerance. Fixtures keep
//! `stride * capacity` and the summed `total_storage_bytes` well below `2^31`,
//! so the device `u32` multiply and add never wrap where the golden
//! `saturating_mul`/`saturating_add` would otherwise clamp.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `*`, `+`, the
//! `max` built-in, unsigned comparisons and index arithmetic. There is no
//! `sqrt`, no divide, no transcendental call, no `u64` and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no
//! loop: each thread performs a fixed, bounded sequence of integer work, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::spawn_pass_buffers`；无第三方引擎源码或衍生代码。
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
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spawn_pass_buffers`。
const WORKGROUP_SIZE: u32 = 64;

/// The `Spawn`/`Emit`-pass bind-group kernel, mirroring the `CPU` golden
/// [`spawn_pass_buffers`](prism_render_architecture::particle::spawn_pass_buffers)
/// field for field. The single entry point `solve` resolves one query per
/// thread, embedded inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spawn_pass_buffers`。
const SPAWN_PASS_BUFFERS_WGSL: &str = r#"
// spawn_pass_buffers twin: one thread per query reproduces the CPU golden
// `particle::spawn_pass_buffers`. For each Spawn/Emit @group(0) binding
// (variant_code 0..6, the @binding index) the kernel reports the binding, the
// std430/std140 stride, the kind code (0u storage / 1u uniform), the access
// code (0u Read / 1u ReadWrite), the extent-sized element count, the
// clamp-to-one byte size `stride * max(count, 1u)`, and the pass-level storage
// total (the sum of the five std430 storage buffers, excluding the uniform).
// Pure u32 arithmetic: constants, a handful of unsigned compares, multiplies,
// adds and `max`. There is no sqrt, no divide, no transcendental call and no
// u64, so the kernel runs unmodified on Metal, Vulkan and DX12. There is no
// loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::spawn_pass_buffers；无第三方
// 引擎源码或衍生代码。

// Binding codes in @binding order, mirroring `SpawnPassBuffer::ALL`.
const FREE_LIST: u32 = 0u;
const COUNTERS: u32 = 1u;
const SPAWN_INDICES: u32 = 2u;
const SPAWN_PARAMS: u32 = 3u;
const POSITION_POOL: u32 = 4u;
const VELOCITY_POOL: u32 = 5u;

// std430 / std140 strides, mirroring the golden constants: U32_STRIDE for the
// free-list, counters and spawn-index arrays, SPAWN_PARAMS_STRIDE for the
// per-emitter uniform block, and VEC4_STRIDE for the position/velocity pools.
const U32_STRIDE: u32 = 4u;
const SPAWN_PARAMS_STRIDE: u32 = 32u;
const VEC4_STRIDE: u32 = 16u;

// Fixed number of atomic u32 counters the Spawn pass binds (SPAWN_COUNTER_COUNT).
const SPAWN_COUNTER_COUNT: u32 = 4u;

// Binding-kind codes, matching the golden `SpawnBindingKind` declaration order
// (Storage == 0u, Uniform == 1u).
const KIND_STORAGE: u32 = 0u;
const KIND_UNIFORM: u32 = 1u;

// Access codes, matching the golden `ParticleBufferAccess` declaration order
// (Read == 0u, ReadWrite == 1u).
const ACCESS_READ: u32 = 0u;
const ACCESS_READ_WRITE: u32 = 1u;

// Dispatch parameters. 16-byte uniform block: the valid query count plus three
// pad words, matching the host `Params`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 16-byte std430 stride of four scalar words, matching the host
// `GpuSpawnPassBuffersQuery`: the variant code and the three `SpawnPassExtent`
// field counts.
struct SpawnQuery {
    variant_code: u32,
    capacity: u32,
    spawn_count: u32,
    emitter_count: u32,
}

// One result. 28-byte std430 stride of seven scalar words, matching the host
// `GpuSpawnPassBuffersResult`.
struct SpawnResult {
    binding: u32,
    stride: u32,
    kind_code: u32,
    access_code: u32,
    element_count: u32,
    byte_size: u32,
    total_storage_bytes: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<SpawnQuery>;
@group(0) @binding(2) var<storage, read_write> results: array<SpawnResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let variant = q.variant_code;

    // binding: variant_code is the @binding index (ALL is dense and ordered).
    let binding = variant;

    // stride: U32 for the free-list, counters and spawn-index arrays; the
    // std140 uniform block stride for SpawnParams; VEC4 for the attribute pools.
    var stride: u32 = U32_STRIDE;
    if (variant == SPAWN_PARAMS) {
        stride = SPAWN_PARAMS_STRIDE;
    } else if (variant == POSITION_POOL || variant == VELOCITY_POOL) {
        stride = VEC4_STRIDE;
    }

    // kind_code / access_code: only SpawnParams is a read-only std140 uniform;
    // every std430 storage buffer is mutated in place by the pass.
    var kind_code: u32 = KIND_STORAGE;
    var access_code: u32 = ACCESS_READ_WRITE;
    if (variant == SPAWN_PARAMS) {
        kind_code = KIND_UNIFORM;
        access_code = ACCESS_READ;
    }

    // element_count: the free-list and both attribute pools span capacity; the
    // counter block is fixed at SPAWN_COUNTER_COUNT; the spawn-index queue holds
    // spawn_count; the uniform holds one block per emitter. Not clamped.
    var element_count: u32 = q.capacity;
    if (variant == COUNTERS) {
        element_count = SPAWN_COUNTER_COUNT;
    } else if (variant == SPAWN_INDICES) {
        element_count = q.spawn_count;
    } else if (variant == SPAWN_PARAMS) {
        element_count = q.emitter_count;
    }

    // byte_size: clamp the count up to one element before the multiply, so an
    // empty pool still yields a valid non-empty WebGPU binding.
    let byte_size = stride * max(element_count, 1u);

    // total_storage_bytes: the sum of the five std430 storage buffers (the
    // free-list, counters, spawn-index queue and the two attribute pools), each
    // clamped to one element, excluding the std140 SpawnParams uniform. Mirrors
    // the golden `total_storage_bytes`.
    var total: u32 = U32_STRIDE * max(q.capacity, 1u);
    total = total + U32_STRIDE * SPAWN_COUNTER_COUNT;
    total = total + U32_STRIDE * max(q.spawn_count, 1u);
    total = total + VEC4_STRIDE * max(q.capacity, 1u);
    total = total + VEC4_STRIDE * max(q.capacity, 1u);

    var out: SpawnResult;
    out.binding = binding;
    out.stride = stride;
    out.kind_code = kind_code;
    out.access_code = access_code;
    out.element_count = element_count;
    out.byte_size = byte_size;
    out.total_storage_bytes = total;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`SPAWN_PASS_BUFFERS_WGSL`]: the valid query count plus three pad
/// words — `16` bytes with no interior padding.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spawn_pass_buffers`。
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

/// One query for the `Spawn`/`Emit`-pass buffer-contract twin: a `variant_code`
/// and the three `SpawnPassExtent` field counts.
///
/// The `variant_code` is the index into
/// [`SpawnPassBuffer::ALL`](prism_render_architecture::particle::spawn_pass_buffers::SpawnPassBuffer)
/// in binding order `0..6` (which equals the `@binding` index). The three
/// counts mirror the golden
/// [`SpawnPassExtent`](prism_render_architecture::particle::spawn_pass_buffers::SpawnPassExtent)
/// fields `capacity`, `spawn_count` and `emitter_count`. The `repr(C)` layout —
/// four `u32` words, `16` bytes with no padding — matches the `WGSL`
/// `SpawnQuery` struct exactly, so it is uploaded to the device without a
/// separate encode step.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spawn_pass_buffers`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSpawnPassBuffersQuery {
    /// Buffer classification code, the `@binding` index in `0..6`.
    pub variant_code: u32,
    /// Emitter pool capacity (`SpawnPassExtent::capacity`).
    pub capacity: u32,
    /// Particles spawned this dispatch (`SpawnPassExtent::spawn_count`).
    pub spawn_count: u32,
    /// Per-emitter `SpawnParams` block count (`SpawnPassExtent::emitter_count`).
    pub emitter_count: u32,
}

/// One resolved answer for a single [`GpuSpawnPassBuffersQuery`], mirroring the
/// golden per-buffer contract outputs and the pass-level `total_storage_bytes`.
///
/// Every field is an exact integer twin of its golden counterpart: `binding`,
/// `stride`, `element_count`, `byte_size` and `total_storage_bytes` are `u32`
/// counts; `kind_code` is `0` for `Storage` and `1` for `Uniform`; and
/// `access_code` is `0` for `Read` and `1` for `ReadWrite`. The `repr(C)`
/// layout — seven `u32` words, `28` bytes — matches the `WGSL` `SpawnResult`
/// struct exactly, so device results are read back without a separate decode
/// step. `kind_code` encodes a
/// [`SpawnBindingKind`](prism_render_architecture::particle::spawn_pass_buffers::SpawnBindingKind)
/// variant and `access_code` encodes a
/// [`ParticleBufferAccess`](prism_render_architecture::particle::gpu_layout::ParticleBufferAccess)
/// variant.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spawn_pass_buffers`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSpawnPassBuffersResult {
    /// The `@group(0)` `@binding` index of the buffer.
    pub binding: u32,
    /// Byte stride of one element, matching the `WESL` scalar / block / struct
    /// layout.
    pub stride: u32,
    /// Binding-kind code: `0` for `Storage`, `1` for `Uniform`.
    pub kind_code: u32,
    /// Access-mode code: `0` for `Read`, `1` for `ReadWrite`.
    pub access_code: u32,
    /// Element count for the query's extent; `1` for the single-element records
    /// and the extent-driven count otherwise.
    pub element_count: u32,
    /// Total clamp-to-one-element byte size `stride * max(count, 1)`, matching
    /// the golden `byte_size`.
    pub byte_size: u32,
    /// Pass-level `std430` storage total for the query's extent, excluding the
    /// `std140` uniform, matching the golden `total_storage_bytes`.
    pub total_storage_bytes: u32,
}

/// Builds a compute-visible buffer binding layout entry.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spawn_pass_buffers`。
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

/// A compiled, reusable `Spawn`/`Emit`-pass buffer-contract compute pipeline,
/// twinning the `CPU` golden
/// [`spawn_pass_buffers`](prism_render_architecture::particle::spawn_pass_buffers).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spawn_pass_buffers`。
pub struct GpuSpawnPassBuffers {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSpawnPassBuffers {
    /// Compiles the `Spawn`/`Emit`-pass bind-group kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::spawn_pass_buffers`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSpawnPassBuffers {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_spawn_pass_buffers_module"),
            source: ShaderSource::Wgsl(SPAWN_PASS_BUFFERS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_spawn_pass_buffers_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_spawn_pass_buffers_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_spawn_pass_buffers_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSpawnPassBuffers {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`GpuSpawnPassBuffersResult`] per input, in order.
    ///
    /// Each result equals the matching golden tuple exactly — `binding`,
    /// `stride`, `kind_code`, `access_code`, `element_count`, `byte_size` and
    /// `total_storage_bytes` all mirror the `CPU`
    /// [`spawn_pass_buffers`](prism_render_architecture::particle::spawn_pass_buffers)
    /// reference — because the whole path is integer bit algebra. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::spawn_pass_buffers`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GpuSpawnPassBuffersQuery],
    ) -> Vec<GpuSpawnPassBuffersResult> {
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
            label: Some("prism_volumetric_spawn_pass_buffers_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spawn_pass_buffers_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (queries.len() as u64) * (size_of::<GpuSpawnPassBuffersResult>() as u64);
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_spawn_pass_buffers_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_spawn_pass_buffers_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_spawn_pass_buffers_bind_group"),
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
            label: Some("prism_volumetric_spawn_pass_buffers_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_spawn_pass_buffers_pass"),
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuSpawnPassBuffersResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
    }
}
