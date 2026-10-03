//! `wgpu` compute twin of the generational material-handle resolution core
//! ([`registry`](prism_render_architecture::material::registry)).
//!
//! GPU scene instances reference a material by a stable
//! [`MaterialHandle`](prism_render_architecture::material::MaterialHandle): a
//! `(index, generation)` pair into the renderer's
//! [`MaterialRegistry`](prism_render_architecture::material::registry::MaterialRegistry).
//! Because freed slots are recycled and bump their generation on reuse, a handle
//! to a removed material must *not* silently resolve to whatever material later
//! took its slot. The `CPU` golden
//! [`get`](prism_render_architecture::material::registry::MaterialRegistry::get)
//! enforces that with a single rule per handle: look the slot up by index, keep
//! it only when the slot's generation matches the handle's and the slot is
//! occupied, otherwise reject it. [`GpuMaterialRegistryResolve`] is the
//! on-device twin of exactly that rule, so a passing real-device parity test is
//! direct evidence the ported resolution reproduces the reference's accept and
//! reject decisions, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread resolves one handle against a dense slot table. Each slot carries
//! its current generation, an occupied flag and the execution-path code of its
//! live record. Given a handle `(index, generation)` the thread reproduces the
//! golden's decision:
//!
//! - an `index >= slot_count` is rejected (`valid == false`), the device mirror
//!   of `self.slots.get(index)` returning `None`;
//! - an in-range slot whose generation differs from the handle's is rejected,
//!   the device mirror of the golden's `slot.generation != handle.generation`
//!   guard;
//! - an in-range, generation-matched but *empty* slot is rejected, the device
//!   mirror of `slot.record` being `None`;
//! - otherwise the handle resolves (`valid == true`) and the slot's
//!   execution-path code is returned, the device mirror of
//!   `slot.record.as_ref()` yielding the record and the resolve stage reading
//!   its [`MaterialExecutionPath`](prism_render_architecture::material::MaterialExecutionPath).
//!
//! The execution-path code is the discriminant order of
//! [`MaterialExecutionPath`](prism_render_architecture::material::MaterialExecutionPath):
//! `0` = `FixedPbr`, `1` = `FixedNpr`, `2` = `ClosureTable`,
//! `3` = `DiagnosticFallback`. Every decision is an unsigned comparison and an
//! indexed load, so it is exact: there is no floating-point arithmetic in the
//! kernel at all.
//!
//! # What stays on the host
//!
//! The generational arena itself — the growable slot vector, the free list, the
//! generation bump on reuse and the live counter driven by
//! [`insert`](prism_render_architecture::material::registry::MaterialRegistry::insert)
//! and [`remove`](prism_render_architecture::material::registry::MaterialRegistry::remove)
//! — stays on the host: it is variable-length, stateful allocation, not a fixed
//! per-element numeric transform. The host also owns marshalling the live arena
//! into the dense `(generation, occupied, path_code)` slot table the device
//! consumes. The device returns the per-handle `(valid, path_code)` resolution;
//! the host uses it exactly as it would use `get(handle).map(|r| r.execution)`.
//! An empty handle batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! Every output is a chain of unsigned comparisons and an indexed load — no
//! divide, no transcendental, no floating point — so the `CPU` and `GPU` agree
//! exactly and the parity test asserts the validity flag and the execution-path
//! code with `==` against the golden
//! [`get`](prism_render_architecture::material::registry::MaterialRegistry::get)
//! and [`contains`](prism_render_architecture::material::registry::MaterialRegistry::contains).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned compares and
//! array loads — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse
//! trigonometry, no `sqrt`, no `round`, and no `u64`. No optional device feature
//! is required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is
//! no loop: each thread performs a fixed, bounded sequence of work, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::material::registry::MaterialRegistry::get`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` material-handle resolution kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`get`](prism_render_architecture::material::registry::MaterialRegistry::get)
/// per-handle decision; see the module documentation for the algorithm.
const MATERIAL_REGISTRY_RESOLVE_WGSL: &str = r#"
// Material-handle resolution twin: one thread resolves one generational handle
// against a dense slot table, mirroring the CPU golden
// `material::registry::MaterialRegistry::get`. A handle is accepted only when
// its index is in range, the slot generation matches and the slot is occupied;
// otherwise it is rejected (valid == 0). Only unsigned compares and array
// loads; no floating point, no u64, no transcendental. The stateful arena
// (slot vector, free list, generation bump) stays on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::material::registry::MaterialRegistry::get；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of slots in the dense table; indices at or past it are rejected.
    slot_count: u32,
    // Number of handle queries in the storage array; threads past this return.
    count: u32,
    pad0: u32,
    pad1: u32,
}

struct Slot {
    // Current generation of this slot.
    generation: u32,
    // 1 when the slot holds a live record, 0 when it is free.
    occupied: u32,
    // Execution-path code of the live record (meaningful only when occupied).
    path_code: u32,
    pad0: u32,
}

struct Query {
    // Handle index into the slot table.
    index: u32,
    // Handle generation, matched against the slot's generation.
    generation: u32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // 1 when the handle resolves (in range, generation match, occupied), else 0.
    valid: u32,
    // Execution-path code of the resolved record (meaningful only when valid).
    path_code: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> slots: array<Slot>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.count) {
        return;
    }
    let q = queries[tid];

    var out: Result;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.valid = 0u;
    out.path_code = 0u;

    // Mirrors `self.slots.get(index)`: an out-of-range index yields `None`.
    if (q.index < params.slot_count) {
        let s = slots[q.index];
        // Mirrors `slot.generation != handle.generation` rejecting a stale
        // handle, and `slot.record` being `None` rejecting a freed slot.
        if (s.generation == q.generation) {
            if (s.occupied == 1u) {
                out.valid = 1u;
                out.path_code = s.path_code;
            }
        }
    }
    results[tid] = out;
}
"#;

/// Uniform parameters for one dispatch: the slot-table length, the query count
/// and two pad words to fill a `16`-byte, `std140`-aligned uniform struct
/// matching `Params` in [`MATERIAL_REGISTRY_RESOLVE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of slots in the dense table.
    slot_count: u32,
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one dense slot: its generation, occupancy flag
/// and execution-path code plus one pad word to a `16`-byte stride matching the
/// `WGSL` `Slot` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSlot {
    /// Current generation of the slot.
    generation: u32,
    /// `1` when the slot holds a live record, `0` when free.
    occupied: u32,
    /// Execution-path code of the live record.
    path_code: u32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one handle query: the handle index and
/// generation plus two pad words to a `16`-byte stride matching the `WGSL`
/// `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Handle index into the slot table.
    index: u32,
    /// Handle generation.
    generation: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one resolution result: the validity flag and
/// the execution-path code plus two pad words to a `16`-byte stride matching the
/// `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` when the handle resolved, `0` when rejected.
    valid: u32,
    /// Execution-path code of the resolved record.
    path_code: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One dense slot of the material arena as seen by the resolution twin: the
/// current generation, whether a live record occupies it, and that record's
/// execution-path code.
///
/// The host marshals the live
/// [`MaterialRegistry`](prism_render_architecture::material::registry::MaterialRegistry)
/// into a slice of these, one per slot, before dispatch. `path_code` is ignored
/// when `occupied` is `false`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialRegistryResolveSlot {
    /// Current generation of the slot.
    pub generation: u32,
    /// Whether a live record occupies the slot.
    pub occupied: bool,
    /// Execution-path code of the live record (only when `occupied`).
    pub path_code: u32,
}

impl MaterialRegistryResolveSlot {
    /// Builds a slot descriptor from its generation, occupancy and path code.
    #[must_use]
    pub const fn new(
        generation: u32,
        occupied: bool,
        path_code: u32,
    ) -> MaterialRegistryResolveSlot {
        MaterialRegistryResolveSlot {
            generation,
            occupied,
            path_code,
        }
    }

    /// Builds the descriptor for a free (empty) slot carrying `generation`.
    #[must_use]
    pub const fn empty(generation: u32) -> MaterialRegistryResolveSlot {
        MaterialRegistryResolveSlot {
            generation,
            occupied: false,
            path_code: 0,
        }
    }
}

/// One handle-resolution query for the twin: a generational handle `(index,
/// generation)`, exactly one
/// [`MaterialHandle`](prism_render_architecture::material::MaterialHandle) the
/// golden would pass to
/// [`get`](prism_render_architecture::material::registry::MaterialRegistry::get).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialRegistryResolveQuery {
    /// Handle index into the slot table.
    pub index: u32,
    /// Handle generation.
    pub generation: u32,
}

impl MaterialRegistryResolveQuery {
    /// Builds a query from a handle index and generation.
    #[must_use]
    pub const fn new(index: u32, generation: u32) -> MaterialRegistryResolveQuery {
        MaterialRegistryResolveQuery { index, generation }
    }
}

/// One resolution result, mirroring the per-handle decision the golden
/// [`get`](prism_render_architecture::material::registry::MaterialRegistry::get)
/// makes.
///
/// `valid` is `false` when the handle was rejected (out of range, stale
/// generation, or empty slot); `path_code` is the execution-path code of the
/// resolved record, meaningful only when `valid` is `true`. The code is the
/// discriminant order of
/// [`MaterialExecutionPath`](prism_render_architecture::material::MaterialExecutionPath):
/// `0` = `FixedPbr`, `1` = `FixedNpr`, `2` = `ClosureTable`,
/// `3` = `DiagnosticFallback`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialRegistryResolveResult {
    /// Whether the handle resolved to a live record.
    pub valid: bool,
    /// Execution-path code of the resolved record (only when `valid`).
    pub path_code: u32,
}

/// Encodes one [`MaterialRegistryResolveSlot`] into its `std430` [`GpuSlot`].
fn encode_slot(s: &MaterialRegistryResolveSlot) -> GpuSlot {
    GpuSlot {
        generation: s.generation,
        occupied: u32::from(s.occupied),
        path_code: s.path_code,
        pad0: 0,
    }
}

/// Encodes one [`MaterialRegistryResolveQuery`] into its `std430` [`GpuQuery`].
fn encode_query(q: &MaterialRegistryResolveQuery) -> GpuQuery {
    GpuQuery {
        index: q.index,
        generation: q.generation,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`MaterialRegistryResolveResult`].
fn decode_result(raw: &GpuResult) -> MaterialRegistryResolveResult {
    MaterialRegistryResolveResult {
        valid: raw.valid != 0,
        path_code: raw.path_code,
    }
}

/// Builds a read-only or read-write storage/uniform binding layout entry.
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

/// A compiled, reusable material-handle resolution compute pipeline, twinning
/// the `CPU` golden
/// [`get`](prism_render_architecture::material::registry::MaterialRegistry::get).
pub struct GpuMaterialRegistryResolve {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMaterialRegistryResolve {
    /// Compiles the resolution kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMaterialRegistryResolve {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_material_registry_resolve"),
            source: ShaderSource::Wgsl(MATERIAL_REGISTRY_RESOLVE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_material_registry_resolve_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_material_registry_resolve_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_material_registry_resolve_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMaterialRegistryResolve {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every handle against the dense slot table and returns one
    /// [`MaterialRegistryResolveResult`] per input, in order.
    ///
    /// `slots` is the dense slot table: `slots[i]` describes slot `i`'s
    /// generation, occupancy and execution-path code. Each result equals the
    /// reference's per-handle resolution exactly. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        slots: &[MaterialRegistryResolveSlot],
        queries: &[MaterialRegistryResolveQuery],
    ) -> Vec<MaterialRegistryResolveResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            slot_count: slots.len() as u32,
            count: count as u32,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_material_registry_resolve_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        // A storage buffer cannot be zero-sized; when the slot table is empty
        // bind a single sentinel slot. `slot_count` stays `0`, so every handle
        // is rejected as out of range and the sentinel is never read.
        let encoded_slots: Vec<GpuSlot> = if slots.is_empty() {
            vec![GpuSlot::zeroed()]
        } else {
            slots.iter().map(encode_slot).collect()
        };
        let slots_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_material_registry_resolve_slots"),
            contents: bytemuck::cast_slice(&encoded_slots),
            usage: BufferUsages::STORAGE,
        });

        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_material_registry_resolve_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_material_registry_resolve_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_material_registry_resolve_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: slots_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_material_registry_resolve_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_material_registry_resolve_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_material_registry_resolve_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per handle query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
