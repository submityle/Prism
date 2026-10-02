//! `wgpu` compute twin of the device-free per-pass `std430` buffer contract
//! ([`sim_pass_buffers`](prism_render_architecture::particle::sim_pass_buffers),
//! particle design §5, §7, §9).
//!
//! The `CPU` golden
//! [`sim_pass_buffers`](prism_render_architecture::particle::sim_pass_buffers)
//! publishes *what the `SimulationStages` kernel binds* at `@group(0)`: for each
//! buffer slot in binding order `0..8` it names the `@binding` index
//! ([`SimPassBuffer::binding`](prism_render_architecture::particle::sim_pass_buffers::SimPassBuffer::binding)),
//! the element `stride`
//! ([`SimPassBuffer::stride`](prism_render_architecture::particle::sim_pass_buffers::SimPassBuffer::stride)),
//! the access mode
//! ([`SimPassBuffer::access`](prism_render_architecture::particle::sim_pass_buffers::SimPassBuffer::access))
//! and its writability
//! ([`SimPassBuffer::is_output`](prism_render_architecture::particle::sim_pass_buffers::SimPassBuffer::is_output)),
//! the two persistence predicates
//! ([`SimPassBuffer::persists_across_frames`](prism_render_architecture::particle::sim_pass_buffers::SimPassBuffer::persists_across_frames)
//! and
//! [`SimPassBuffer::aliases_persistent_pool`](prism_render_architecture::particle::sim_pass_buffers::SimPassBuffer::aliases_persistent_pool)),
//! and the extent-sized
//! [`SimPassBuffer::element_count`](prism_render_architecture::particle::sim_pass_buffers::SimPassBuffer::element_count)
//! and
//! [`SimPassBuffer::byte_size`](prism_render_architecture::particle::sim_pass_buffers::SimPassBuffer::byte_size).
//! It also exposes the pass-level
//! [`transient_scratch_bytes`](prism_render_architecture::particle::sim_pass_buffers::transient_scratch_bytes),
//! the sum of the freshly allocated (non-aliasing) buffers.
//!
//! [`GpuSimPassBuffers`] is the on-device twin: one thread resolves one
//! [`GpuSimPassBuffersQuery`] (a slot classification code plus the three
//! `SimPassExtent` field counts) into one [`GpuSimPassBuffersResult`] carrying
//! all nine golden answers, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same binding, stride, access,
//! persistence, element count, byte size and transient-scratch total the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries a `slot_code` in `0..8` (the index into
//! [`SimPassBuffer::ALL`](prism_render_architecture::particle::sim_pass_buffers::SimPassBuffer::ALL),
//! which equals the `@binding` index) and the three `SimPassExtent` counts
//! (`particle_capacity`, `grid_cell_count`, `constraint_count`). The kernel
//! reproduces every per-slot query field for field, encoding the
//! [`ParticleBufferAccess`](prism_render_architecture::particle::gpu_layout::ParticleBufferAccess)
//! as `0` for `Read` and `1` for `ReadWrite`, and the four `bool` predicates as
//! discrete `1`/`0` flags. `byte_size` reproduces the clamp-to-one-element
//! [`storage_bytes`](prism_render_architecture::particle::gpu_layout::storage_bytes)
//! rule `stride * max(count, 1)`, and `transient_scratch_bytes` sums the five
//! non-persistent slots the same way.
//!
//! # Correctness model
//!
//! Every value is a `u32`, a classification code or a `bool`-flavoured flag:
//! strides and bindings are constants, byte sizes are pure unsigned
//! multiply-and-max with no rounding, and the access / persistence flags are
//! discrete classifications. `CPU` and `GPU` therefore compute identical bit
//! patterns, and the parity test asserts an exact `==` on every field with no
//! tolerance. Fixtures keep each extent count well below `2^18`, so with the
//! widest stride of `64` no device `u32` multiply (nor the summed transient
//! total) wraps where the golden `saturating_mul` would otherwise clamp.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `*` and `+`,
//! the `max` built-in, unsigned comparisons and index arithmetic. There is no
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
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sim_pass_buffers`；无第三方引擎源码或衍生代码。
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

/// The per-pass buffer-contract kernel, mirroring the `CPU` golden
/// [`sim_pass_buffers`](prism_render_architecture::particle::sim_pass_buffers)
/// field for field. The single entry point `solve` resolves one query per
/// thread, embedded inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sim_pass_buffers`。
const SIM_PASS_BUFFERS_WGSL: &str = r#"
// sim_pass_buffers twin: one thread per query reproduces the CPU golden
// `particle::sim_pass_buffers`. For each buffer slot (slot_code 0..8, the
// SimulationStages @binding index) the kernel reports the binding, std430
// stride, access code (0u Read / 1u ReadWrite), the writable / persistence /
// aliasing flags, the extent-sized element count, the clamp-to-one byte size
// `stride * max(count, 1u)`, and the pass-level transient-scratch total (the
// sum of the five non-persistent slots). Pure u32 arithmetic: constants, a
// handful of unsigned compares, multiplies, adds and `max`. There is no sqrt,
// no divide, no transcendental call and no u64, so the kernel runs unmodified
// on Metal, Vulkan and DX12. There is no loop, so the kernel provably
// terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::sim_pass_buffers；无第三方
// 引擎源码或衍生代码。

// std430 strides, mirroring the golden constants: VEC4_STRIDE (per-particle
// pools), the Simulate uniform block size, U32_STRIDE (cell table / lambdas),
// VEC2_STRIDE (grid entries) and the XPBD constraint descriptor stride.
const VEC4_STRIDE: u32 = 16u;
const SIM_UNIFORM_SIZE: u32 = 64u;
const U32_STRIDE: u32 = 4u;
const VEC2_STRIDE: u32 = 8u;
const CONSTRAINT_STRIDE: u32 = 32u;

// Access code for the ReadWrite variant, matching the golden enum's declaration
// order (Read == 0u, ReadWrite == 1u). It is the only writable access.
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
// `GpuSimPassBuffersQuery`: the slot classification code and the three
// `SimPassExtent` field counts.
struct SlotQuery {
    slot_code: u32,
    particle_capacity: u32,
    grid_cell_count: u32,
    constraint_count: u32,
}

// One result. 36-byte std430 stride of nine scalar words, matching the host
// `GpuSimPassBuffersResult`.
struct SlotResult {
    binding: u32,
    stride: u32,
    access_code: u32,
    is_output: u32,
    persists_across_frames: u32,
    aliases_persistent_pool: u32,
    element_count: u32,
    byte_size: u32,
    transient_scratch_bytes: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<SlotQuery>;
@group(0) @binding(2) var<storage, read_write> results: array<SlotResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let slot = q.slot_code;

    // binding: slot_code is the SimulationStages @binding index (ALL is dense
    // and ordered 0..8).
    let binding = slot;

    // stride: VEC4 for the per-particle pools (slots 0..2), the uniform block
    // size for SimParams (slot 3), VEC2 for GridEntries (slot 5), the
    // constraint descriptor stride for Constraints (slot 6), and U32 for the
    // cell-offset table (slot 4) and ConstraintLambdas (slot 7).
    var stride: u32 = U32_STRIDE;
    if (slot <= 2u) {
        stride = VEC4_STRIDE;
    } else if (slot == 3u) {
        stride = SIM_UNIFORM_SIZE;
    } else if (slot == 5u) {
        stride = VEC2_STRIDE;
    } else if (slot == 6u) {
        stride = CONSTRAINT_STRIDE;
    }

    // access_code: ReadWrite (1u) for the per-particle pools (slots 0..2) and
    // ConstraintLambdas (slot 7); Read (0u) otherwise.
    var access_code: u32 = 0u;
    if (slot <= 2u || slot == 7u) {
        access_code = ACCESS_READ_WRITE;
    }
    // is_output: the pass writes iff the access is ReadWrite.
    let is_output = access_code;

    // persistence / aliasing: only the per-particle Structure-of-Arrays pools
    // (slots 0..2) survive across frames and alias the §5 persistent pool.
    var persists: u32 = 0u;
    if (slot <= 2u) {
        persists = 1u;
    }
    let aliases = persists;

    // element_count: the per-particle pools and the neighbourhood entry list
    // (slots 0..2 and 5) are particle-sized; SimParams (slot 3) is a single
    // element; the cell-offset table (slot 4) is grid-sized; the constraint and
    // lambda buffers (slots 6, 7) are constraint-sized.
    var element_count: u32 = q.constraint_count;
    if (slot <= 2u || slot == 5u) {
        element_count = q.particle_capacity;
    } else if (slot == 3u) {
        element_count = 1u;
    } else if (slot == 4u) {
        element_count = q.grid_cell_count;
    }

    // byte_size: clamp the count up to one element before the multiply, so an
    // empty emitter still yields a valid non-empty WebGPU binding.
    let byte_size = stride * max(element_count, 1u);

    // transient_scratch_bytes: the sum of the five freshly allocated (non
    // aliasing) slots — SimParams, GridCellStart, GridEntries, Constraints and
    // ConstraintLambdas — each clamped to one element.
    var transient: u32 = SIM_UNIFORM_SIZE * 1u;
    transient = transient + U32_STRIDE * max(q.grid_cell_count, 1u);
    transient = transient + VEC2_STRIDE * max(q.particle_capacity, 1u);
    transient = transient + CONSTRAINT_STRIDE * max(q.constraint_count, 1u);
    transient = transient + U32_STRIDE * max(q.constraint_count, 1u);

    var out: SlotResult;
    out.binding = binding;
    out.stride = stride;
    out.access_code = access_code;
    out.is_output = is_output;
    out.persists_across_frames = persists;
    out.aliases_persistent_pool = aliases;
    out.element_count = element_count;
    out.byte_size = byte_size;
    out.transient_scratch_bytes = transient;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`SIM_PASS_BUFFERS_WGSL`]: the valid query count plus three pad
/// words — `16` bytes with no interior padding.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sim_pass_buffers`。
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

/// One query for the per-pass buffer-contract twin: a `slot_code` and the three
/// `SimPassExtent` field counts.
///
/// The `slot_code` is the index into
/// [`SimPassBuffer::ALL`](prism_render_architecture::particle::sim_pass_buffers::SimPassBuffer::ALL)
/// in binding order `0..8` (which equals the `@binding` index). The three
/// counts mirror the golden `SimPassExtent` fields `particle_capacity`,
/// `grid_cell_count` and `constraint_count`. The `repr(C)` layout — four `u32`
/// words, `16` bytes with no padding — matches the `WGSL` `SlotQuery` struct
/// exactly, so it is uploaded to the device without a separate encode step.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sim_pass_buffers`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSimPassBuffersQuery {
    /// Buffer slot classification code, the `@binding` index in `0..8`.
    pub slot_code: u32,
    /// Pooled particle capacity (`SimPassExtent::particle_capacity`).
    pub particle_capacity: u32,
    /// Spatial-hash grid cell count (`SimPassExtent::grid_cell_count`).
    pub grid_cell_count: u32,
    /// `XPBD` constraint count (`SimPassExtent::constraint_count`).
    pub constraint_count: u32,
}

/// One resolved answer for a single [`GpuSimPassBuffersQuery`], mirroring the
/// golden per-slot buffer-contract outputs and the pass-level transient-scratch
/// total.
///
/// Every field is an exact integer twin of its golden counterpart: `binding`,
/// `stride`, `element_count` and `byte_size` are `u32` counts; `access_code` is
/// `0` for `Read` and `1` for `ReadWrite`; and `is_output`,
/// `persists_across_frames` and `aliases_persistent_pool` are discrete `1`/`0`
/// flags. The `repr(C)` layout — nine `u32` words, `36` bytes — matches the
/// `WGSL` `SlotResult` struct exactly, so device results are read back without a
/// separate decode step.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sim_pass_buffers`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSimPassBuffersResult {
    /// The `@group(0)` `@binding` index of the slot.
    pub binding: u32,
    /// `std430` byte stride of one element.
    pub stride: u32,
    /// Access-mode code: `0` for `Read`, `1` for `ReadWrite`.
    pub access_code: u32,
    /// Writability flag, `1` when the pass writes the buffer and `0` otherwise.
    pub is_output: u32,
    /// Persistence flag, `1` when the buffer survives between frames.
    pub persists_across_frames: u32,
    /// Aliasing flag, `1` when the binding aliases a persistent pool.
    pub aliases_persistent_pool: u32,
    /// Element count for the query's extent.
    pub element_count: u32,
    /// Total clamp-to-one-element byte size `stride * max(count, 1)`.
    pub byte_size: u32,
    /// Pass-level transient-scratch total for the query's extent.
    pub transient_scratch_bytes: u32,
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

/// A compiled, reusable per-pass buffer-contract compute pipeline, twinning the
/// `CPU` golden
/// [`sim_pass_buffers`](prism_render_architecture::particle::sim_pass_buffers).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sim_pass_buffers`。
pub struct GpuSimPassBuffers {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSimPassBuffers {
    /// Compiles the per-pass buffer-contract kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::sim_pass_buffers`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSimPassBuffers {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sim_pass_buffers"),
            source: ShaderSource::Wgsl(SIM_PASS_BUFFERS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sim_pass_buffers_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sim_pass_buffers_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sim_pass_buffers_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSimPassBuffers {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`GpuSimPassBuffersResult`] per input, in order.
    ///
    /// Each result equals the matching golden slot answers exactly — binding,
    /// stride, access, writability, persistence, aliasing, element count and
    /// byte size from
    /// [`SimPassBuffer`](prism_render_architecture::particle::sim_pass_buffers::SimPassBuffer),
    /// plus the pass-level
    /// [`transient_scratch_bytes`](prism_render_architecture::particle::sim_pass_buffers::transient_scratch_bytes)
    /// — because the whole path is integer bit algebra. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::sim_pass_buffers`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GpuSimPassBuffersQuery],
    ) -> Vec<GpuSimPassBuffersResult> {
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
            label: Some("prism_volumetric_sim_pass_buffers_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sim_pass_buffers_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (queries.len() as u64) * (size_of::<GpuSimPassBuffersResult>() as u64);
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sim_pass_buffers_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sim_pass_buffers_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sim_pass_buffers_bind_group"),
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
            label: Some("prism_volumetric_sim_pass_buffers_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sim_pass_buffers_pass"),
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuSimPassBuffersResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
    }
}
