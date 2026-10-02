//! `wgpu` compute twin of the particle-subsystem *atomic stream append*
//! primitive
//! ([`gpu_stream_append`](prism_render_architecture::particle::gpu_stream_append),
//! particle design §5.2 pooling counters, §9 `Spawn` / `Event Scatter` writes,
//! §11 atomic counters).
//!
//! An *append buffer* is the write side of the produce/consume pattern every
//! production `GPU` VFX stack leans on: `Direct3D`'s `AppendStructuredBuffer`
//! (`.Append()` atomically increments a hidden counter and returns the reserved
//! slot) and `Vulkan`'s `atomicAdd` on a dedicated atomic-counter binding are
//! the same idea — many concurrent invocations race to reserve a monotonically
//! increasing slot index, the reservations never collide, and a run that
//! overflows the fixed `capacity` is dropped rather than corrupting a
//! neighbour. The `CPU` golden
//! [`gpu_stream_append`](prism_render_architecture::particle::gpu_stream_append)
//! owns the deterministic reference as its
//! [`AppendCounter`](prism_render_architecture::particle::gpu_stream_append::AppendCounter):
//! a monotonic slot index (the length of a compact backing store), a fixed
//! `capacity`, and a running `overflow` tally.
//!
//! [`GpuStreamAppend`] is the on-device twin of the per-element
//! [`atomic_append`](prism_render_architecture::particle::gpu_stream_append::AppendCounter::atomic_append):
//! one thread per input value runs a single `atomicAdd(&counter, 1u)` to
//! reserve its slot, writes the value into the dense backing store when the
//! reserved slot is below `capacity`, and otherwise bumps an atomic `overflow`
//! word. The reservation counter ends at exactly the input count and the
//! `overflow` word ends at exactly `count - min(count, capacity)` regardless of
//! which lane won any particular atomic race, so the batch's final `std430`
//! counter record `[count, overflow, capacity, reserved]` (see
//! [`COUNTER_WORDS`]) is deterministic and parity-checked with exact `==`.
//!
//! # What is twinned, and the ordering contract
//!
//! The kernel twins the atomic reservation, not a stable write order. `GPU`
//! `atomicAdd` hands out unique slot indices but does **not** guarantee *which*
//! lane receives *which* index, so the mapping from an input value to its final
//! slot is not stable across runs or against the sequential golden. The twin's
//! determinism therefore lives in the counter record and in the invariant that
//! the written slots form a dense prefix `[0, min(count, capacity))` with each
//! slot filled exactly once by exactly one input value. The real-device parity
//! test asserts the deterministic facts — the terminal counter words with exact
//! `==`, and set-membership of the written slot values — rather than assuming a
//! stable order the hardware never promises. When `capacity >= count` nothing
//! overflows, so the written multiset equals the whole input multiset and the
//! sorted readback equals the sorted golden store element for element.
//!
//! Deliberately **not** twinned: the sequential
//! [`batch_append`](prism_render_architecture::particle::gpu_stream_append::AppendCounter::batch_append)
//! two-pass base allocation (a per-group prefix scan outside the
//! one-thread-per-element contract) and the tail
//! [`consume`](prism_render_architecture::particle::gpu_stream_append::AppendCounter::consume).
//!
//! # Correctness model
//!
//! Every value is a `u32`: the reserved slot index, the reservation counter and
//! the `overflow` word, all pure integer atomics. There is no float math
//! anywhere, so there is no `ULP`-boundary degenerate region to avoid; the
//! counter comparison is bit-exact by construction.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned comparisons,
//! `atomicAdd` and index arithmetic. There is no `sqrt`, no transcendental
//! call, no `u64` and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. There is no loop: each thread performs a fixed
//! sequence of operations and provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_stream_append`；无第三方引擎源码或衍生代码。
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

/// Number of `u32` words in the packed `std430` append-counter record,
/// mirroring the golden
/// [`COUNTER_WORDS`](prism_render_architecture::particle::gpu_stream_append::COUNTER_WORDS).
///
/// Layout is one `vec4`-aligned word block `[count, overflow, capacity, reserved]`,
/// matching the hidden atomic counter a `GPU` `AppendStructuredBuffer` keeps
/// beside its data buffer (padded up to a natural four-word slot).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_stream_append`。
pub const COUNTER_WORDS: usize = 4;

/// The atomic stream-append kernel, mirroring the `CPU` golden
/// [`gpu_stream_append`](prism_render_architecture::particle::gpu_stream_append).
/// The single entry point `solve` reserves one slot per thread via `atomicAdd`,
/// embedded inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_stream_append`。
const GPU_STREAM_APPEND_WGSL: &str = r#"
// gpu_stream_append twin: one thread per input value reproduces the CPU golden
// `particle::gpu_stream_append` atomic append. Each thread runs a single
// `atomicAdd(&counter[0], 1u)` to reserve a unique slot; if the slot is below
// capacity it writes the value into the dense store, otherwise it bumps the
// atomic overflow word `counter[1]`. The reservation counter ends at the input
// count and the overflow word ends at `count - min(count, capacity)` no matter
// which lane wins any atomic race, so the terminal counter record is
// deterministic. The per-slot value mapping is NOT order-stable (GPU atomics do
// not promise which lane gets which index); the deterministic contract is the
// counter record plus a dense, collision-free `[0, appended)` write front.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::gpu_stream_append;
// 无第三方引擎源码或衍生代码。

// Dispatch parameters. 16-byte uniform block: the element count, the slot
// capacity, and two pad words, matching the host `Params`.
struct Params {
    count: u32,
    capacity: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> slots: array<u32>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let value = values[idx];
    // Reserve the next slot. The atomic add returns a unique index to every
    // thread; the final value of `counter[0]` is exactly `params.count`.
    let slot = atomicAdd(&counter[0], 1u);
    if (slot < params.capacity) {
        // The reserved slot owns a unique dense position below capacity, so no
        // two threads ever race on the same store slot.
        slots[slot] = value;
    } else {
        // Capacity is exhausted: drop the value and tally the overflow. The
        // final value of `counter[1]` is exactly `count - min(count, cap)`.
        atomicAdd(&counter[1], 1u);
    }
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`GPU_STREAM_APPEND_WGSL`]: the element `count`, the slot
/// `capacity` and two pad words — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of input values (valid threads).
    count: u32,
    /// Fixed slot capacity; reservations at or beyond it overflow.
    capacity: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One atomic stream-append request: the ordered `values` to append and the
/// fixed slot `capacity` of the append buffer.
///
/// A `capacity` of zero is a degenerate buffer that overflows every value; the
/// device storage binding still clamps up to one element so the dispatch stays
/// valid, mirroring the golden
/// [`AppendConfig`](prism_render_architecture::particle::gpu_stream_append::AppendConfig)
/// clamp-to-one rule.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_stream_append`。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuStreamAppendQuery {
    /// The ordered values to append; each is reserved by one thread.
    pub values: Vec<u32>,
    /// Fixed slot capacity; reservations at or beyond it overflow.
    pub capacity: u32,
}

/// The decoded terminal append counter read back from the device: the packed
/// `std430` word block `[count, overflow, capacity, reserved]`, mirroring the
/// golden
/// [`AppendCounter::to_std430`](prism_render_architecture::particle::gpu_stream_append::AppendCounter::to_std430).
///
/// Every word is a `u32`, so the whole record is compared with exact `==`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_stream_append`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuStreamAppendCounter {
    /// The packed `std430` counter words `[count, overflow, capacity, reserved]`.
    words: [u32; COUNTER_WORDS],
}

impl GpuStreamAppendCounter {
    /// Wraps the raw `std430` counter `words` read back from the device.
    #[must_use]
    pub fn from_words(words: [u32; COUNTER_WORDS]) -> GpuStreamAppendCounter {
        GpuStreamAppendCounter { words }
    }

    /// The packed `std430` counter words `[count, overflow, capacity, reserved]`.
    #[must_use]
    pub fn words(&self) -> [u32; COUNTER_WORDS] {
        self.words
    }

    /// The appended slot count `min(input_count, capacity)`, mirroring the
    /// golden `AppendCounter::count`.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.words[0]
    }

    /// The number of values dropped because `capacity` was exhausted.
    #[must_use]
    pub fn overflow(&self) -> u32 {
        self.words[1]
    }

    /// The fixed slot `capacity` the request ran against.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.words[2]
    }
}

/// The resolved outcome of one atomic stream-append dispatch, mirroring the
/// terminal state of the golden
/// [`AppendCounter`](prism_render_architecture::particle::gpu_stream_append::AppendCounter).
///
/// The `counter` holds the deterministic terminal word block; `appended` and
/// `overflow` restate its `count` and `overflow` words as named scalars.
/// `slots` is the dense backing store read back from the device, truncated to
/// the `appended` written prefix. Its *multiset* of values is deterministic
/// only when nothing overflowed (`capacity >= count`), since `GPU` atomics do
/// not promise which lane claimed which slot; the parity test asserts the
/// deterministic counter words plus slot set-membership rather than a stable
/// order.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_stream_append`。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuStreamAppendResult {
    /// The decoded terminal `std430` append counter.
    pub counter: GpuStreamAppendCounter,
    /// The dense backing store, truncated to the `appended` written prefix.
    pub slots: Vec<u32>,
    /// The appended slot count `min(input_count, capacity)`.
    pub appended: u32,
    /// The number of values dropped because `capacity` was exhausted.
    pub overflow: u32,
}

/// Configuration for an append buffer: the fixed slot `capacity` and the byte
/// `stride` of one stored element, mirroring the golden
/// [`AppendConfig`](prism_render_architecture::particle::gpu_stream_append::AppendConfig).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_stream_append`。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuStreamAppendConfig {
    /// Maximum number of live slots the append buffer can hold. May be zero (a
    /// degenerate buffer that overflows every append); the `std430` byte size
    /// still clamps up to one element for a valid `GPU` binding.
    pub capacity: usize,
    /// Byte stride of one stored element, clamped to at least one so a
    /// degenerate zero can never collapse the data buffer size.
    pub stride: usize,
}

impl GpuStreamAppendConfig {
    /// Builds a config with the given `capacity` and element `stride`,
    /// clamping the stride to at least one byte.
    #[must_use]
    pub fn new(capacity: usize, stride: usize) -> GpuStreamAppendConfig {
        GpuStreamAppendConfig {
            capacity,
            stride: stride.max(1),
        }
    }

    /// Byte size of the `std430` data buffer holding all `capacity` slots at
    /// this element `stride`, clamped up to one element for a valid binding.
    #[must_use]
    pub fn buffer_bytes(self) -> usize {
        self.stride.saturating_mul(self.capacity.max(1))
    }

    /// Byte size of the `std430` hidden atomic-counter buffer: the packed
    /// [`COUNTER_WORDS`] `u32` record beside the data buffer.
    #[must_use]
    pub fn counter_bytes(self) -> usize {
        COUNTER_WORDS * size_of::<u32>()
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

/// A compiled, reusable atomic stream-append compute pipeline, twinning the
/// `CPU` golden
/// [`gpu_stream_append`](prism_render_architecture::particle::gpu_stream_append).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_stream_append`。
pub struct GpuStreamAppend {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuStreamAppend {
    /// Compiles the atomic stream-append kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset plus `atomicAdd`,
    /// so no optional device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_stream_append`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuStreamAppend {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gpu_stream_append_module"),
            source: ShaderSource::Wgsl(GPU_STREAM_APPEND_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gpu_stream_append_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gpu_stream_append_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gpu_stream_append_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuStreamAppend {
            module,
            layout,
            pipeline,
        }
    }

    /// Appends every value in `query` against its fixed `capacity` on the
    /// device, returning the terminal counter record and the dense written
    /// slot prefix.
    ///
    /// The returned [`GpuStreamAppendResult::counter`] equals the golden
    /// [`AppendCounter::to_std430`](prism_render_architecture::particle::gpu_stream_append::AppendCounter::to_std430)
    /// word for word, because the reservation counter and overflow word are
    /// order-independent integer atomics. The written `slots` form a dense,
    /// collision-free `[0, appended)` prefix; their value order is not stable,
    /// so callers compare them as a set. An empty `values` batch issues **no
    /// dispatch** — a storage buffer may not be zero-sized — and returns the
    /// empty terminal state directly.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_stream_append`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        query: &GpuStreamAppendQuery,
    ) -> GpuStreamAppendResult {
        let count = query.values.len();
        let capacity = query.capacity;
        let appended = (count as u64).min(u64::from(capacity)) as u32;
        if count == 0 {
            // Host short-circuit: no thread to dispatch. The terminal state is
            // an empty store with a zero count and no overflow.
            return GpuStreamAppendResult {
                counter: GpuStreamAppendCounter::from_words([0, 0, capacity, 0]),
                slots: Vec::new(),
                appended: 0,
                overflow: 0,
            };
        }
        let device = ctx.device();

        let params = Params {
            count: count as u32,
            capacity,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_stream_append_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let values_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_stream_append_values"),
            contents: bytemuck::cast_slice(&query.values),
            usage: BufferUsages::STORAGE,
        });

        // The counter buffer starts as `[0, 0, capacity, 0]`: the device only
        // ever atomic-adds the first two words, leaving the capacity and the
        // reserved pad word as the host set them.
        let counter_init: [u32; COUNTER_WORDS] = [0, 0, capacity, 0];
        let counter_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_stream_append_counter"),
            contents: bytemuck::cast_slice(&counter_init),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let counter_bytes = (COUNTER_WORDS as u64) * (size_of::<u32>() as u64);
        let counter_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_stream_append_counter_stage"),
            size: counter_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // The data store spans `max(capacity, 1)` slots (never zero-sized);
        // only the first `appended` slots are written. Zero-initialized so an
        // untouched tail reads back as zero.
        let store_len = (capacity.max(1)) as usize;
        let store_zeros = vec![0u32; store_len];
        let slots_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_stream_append_slots"),
            contents: bytemuck::cast_slice(&store_zeros),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let slots_bytes = (store_len as u64) * (size_of::<u32>() as u64);
        let slots_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_stream_append_slots_stage"),
            size: slots_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gpu_stream_append_bind_group"),
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
                    resource: counter_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: slots_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_gpu_stream_append_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gpu_stream_append_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per input value, flattened to a 1-D dispatch.
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&counter_buf, 0, &counter_stage, 0, counter_bytes);
        encoder.copy_buffer_to_buffer(&slots_buf, 0, &slots_stage, 0, slots_bytes);
        ctx.queue().submit([encoder.finish()]);

        counter_stage.slice(..).map_async(MapMode::Read, |_| {});
        slots_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let counter_view = counter_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let counter_words = bytemuck::cast_slice::<u8, u32>(&counter_view).to_vec();
        drop(counter_view);
        counter_stage.unmap();

        let slots_view = slots_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let mut slots = bytemuck::cast_slice::<u8, u32>(&slots_view).to_vec();
        drop(slots_view);
        slots_stage.unmap();
        slots.truncate(appended as usize);

        // The raw reservation counter overshoots to the full input count; the
        // golden `to_std430` reports the clamped slot count, so restate word 0
        // as the appended total while taking the overflow straight from the
        // device's atomic tally.
        let overflow = counter_words[1];
        let words: [u32; COUNTER_WORDS] = [appended, overflow, capacity, 0];

        debug_assert_eq!(slots.len(), appended as usize);
        GpuStreamAppendResult {
            counter: GpuStreamAppendCounter::from_words(words),
            slots,
            appended,
            overflow,
        }
    }
}
