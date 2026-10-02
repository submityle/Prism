//! `wgpu` compute twin of the particle-subsystem *statistics readback
//! reduction*
//! ([`readback`](prism_render_architecture::particle::readback), particle
//! design §9 pipeline stats, §13 culling counters).
//!
//! A `GPU`-driven particle engine never trusts the `CPU` to know how many
//! particles survived a frame: the spawn, simulation and cull passes each emit
//! a small per-`workgroup` *partial* record of `u32` counters, and those
//! partials must be folded into one reduced record before the counts are
//! reported. The `CPU` golden
//! [`reduce_partials`](prism_render_architecture::particle::readback::reduce_partials)
//! owns that fold: each of the [`STAT_FIELD_COUNT`] fields is summed across the
//! partials with `saturating_add`, so a pathological partial set clamps at
//! `u32::MAX` instead of wrapping, and an empty partial list reduces to all
//! zeroes.
//!
//! [`GpuReadbackReduce`] is the on-device twin. The partials are uploaded as a
//! single flattened, row-major `u32` storage buffer — partial `i`'s field `j`
//! lives at `i * STAT_FIELD_COUNT + j` — and the kernel runs one thread per
//! field slot (so [`STAT_FIELD_COUNT`] threads total). Each thread walks every
//! partial and folds its own field with a hand-rolled `saturating_add`, because
//! the raw `WGSL` `u32 + u32` *wraps* on overflow and must be clamped by hand
//! (a sum less than either addend means the add overflowed, so it is pinned to
//! `0xffffffffu`). The reduced record is read back and the two derived
//! accessors the render graph reports —
//! [`GpuReadbackReduceResult::total_culled`] and
//! [`GpuReadbackReduceResult::alive_after_cull`] — are computed host-side with
//! the identical `saturating_add` / `saturating_sub` the golden
//! [`StatSnapshot`](prism_render_architecture::particle::readback::StatSnapshot)
//! applies.
//!
//! # What is twinned
//!
//! The kernel reproduces the golden
//! [`reduce_partials`](prism_render_architecture::particle::readback::reduce_partials)
//! fold exactly: the device reduced record equals the golden reduced record,
//! field for field. The host then derives `total_culled` as the saturating sum
//! of the three cull counters (`CulledFrustum` + `CulledDistance` +
//! `CulledHzb`) and `alive_after_cull` as `AliveCount` saturating-minus
//! `total_culled`, mirroring the golden
//! [`StatSnapshot`](prism_render_architecture::particle::readback::StatSnapshot)
//! accessors.
//!
//! # What is not twinned
//!
//! The device-free scheduling and byte-budget half of the golden stays on the
//! `CPU` and is intentionally *not* reproduced here:
//!
//! - [`ReadbackRing`](prism_render_architecture::particle::readback::ReadbackRing)
//!   maps a `u64` `frame_index` to its write / read slot by modulo over the
//!   frames-in-flight count; `WGSL` has no `u64`, so the frame-latency ring is a
//!   pure host concept with no device analogue.
//! - [`stat_record_bytes`](prism_render_architecture::particle::readback::stat_record_bytes)
//!   and
//!   [`ReadbackRing::staging_bytes`](prism_render_architecture::particle::readback::ReadbackRing::staging_bytes)
//!   are `usize` / `u64` byte budgets for the staging allocation, not counter
//!   algebra, so they live entirely on the host.
//! - There is no `f32` quantity anywhere in this contract: every counter is a
//!   `u32`, so the whole twin is exact-integer and uses no epsilon.
//!
//! # Correctness model
//!
//! Every accumulation is a `u32` `saturating_add`: the device folds each field
//! with the hand-rolled clamp and the golden folds with
//! `u32::saturating_add`, which agree bit for bit. The reduction is a sum, so
//! regrouping the per-partial walk across one thread per field changes nothing.
//! The `GPU` reduced record therefore equals the golden reduced record exactly
//! and the parity test asserts an exact `==` with no tolerance.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `+`,
//! unsigned comparison and index arithmetic. There is no `sqrt`, no
//! transcendental call, no `f32`, no `f64` and no `u64`, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. The only loop walks the partial list,
//! bounded by `partial_count`, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::readback`；无第三方引擎源码或衍生代码。
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

use prism_render_architecture::particle::readback::{ParticleStatField, STAT_FIELD_COUNT};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::readback`。
const WORKGROUP_SIZE: u32 = 64;

/// The partial-reduction kernel, mirroring the `CPU` golden
/// [`reduce_partials`](prism_render_architecture::particle::readback::reduce_partials).
/// The single entry point `solve` folds one field slot per thread, embedded
/// inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::readback`。
const READBACK_REDUCE_WGSL: &str = r#"
// readback_reduce twin: one thread per field slot reproduces the CPU golden
// `particle::readback::reduce_partials`. The partials are a flattened,
// row-major u32 buffer (partial i's field j at i*field_count + j); each thread
// walks every partial and folds its own field with a hand-rolled saturating
// add. The raw WGSL `u32 + u32` wraps on overflow, so a sum less than either
// addend means it overflowed and is clamped to 0xffffffffu, matching the golden
// `u32::saturating_add` bit for bit. An empty partial list is short-circuited
// on the host, so this kernel always sees partial_count >= 1.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::readback;
// 无第三方引擎源码或衍生代码。

// Dispatch parameters. 16-byte uniform block: the partial count, the field
// count (STAT_FIELD_COUNT), and two pad words, matching the host `Params`.
struct Params {
    partial_count: u32,
    field_count: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> partials: array<u32>;
@group(0) @binding(2) var<storage, read_write> reduced: array<u32>;

// Saturating u32 add: the raw add wraps, so a sum below either addend signals
// overflow and pins the result at u32::MAX, matching the golden
// `u32::saturating_add`.
fn saturating_add_u32(a: u32, b: u32) -> u32 {
    let sum = a + b;
    if (sum < a) {
        return 0xffffffffu;
    }
    return sum;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let field = gid.x;
    if (field >= params.field_count) {
        return;
    }
    var acc: u32 = 0u;
    for (var p: u32 = 0u; p < params.partial_count; p = p + 1u) {
        let value = partials[p * params.field_count + field];
        acc = saturating_add_u32(acc, value);
    }
    reduced[field] = acc;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`READBACK_REDUCE_WGSL`]: the `partial_count`, the `field_count`
/// and two pad words — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of partial records to fold.
    partial_count: u32,
    /// Fields per record, always [`STAT_FIELD_COUNT`].
    field_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One reduction request: the per-`workgroup` partial counter records the `GPU`
/// cull / scatter passes emitted, each a dense `[u32; STAT_FIELD_COUNT]` array
/// in [`ParticleStatField`] record order.
///
/// An empty `partials` list is short-circuited on the host (a storage buffer
/// may not be zero-sized) and reduces to an all-zero record, exactly as the
/// golden
/// [`reduce_partials`](prism_render_architecture::particle::readback::reduce_partials)
/// yields on an empty slice.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::readback`。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuReadbackReduceQuery {
    /// The per-`workgroup` partial records to fold, in emission order.
    pub partials: Vec<[u32; STAT_FIELD_COUNT]>,
}

/// The device-computed answer of one reduction: the reduced counter record plus
/// the two derived accessors the render graph reports.
///
/// `reduced` is the field-wise saturating sum of the partials, matching the
/// golden
/// [`reduce_partials`](prism_render_architecture::particle::readback::reduce_partials).
/// `total_culled` and `alive_after_cull` mirror the golden
/// [`StatSnapshot`](prism_render_architecture::particle::readback::StatSnapshot)
/// accessors, derived host-side from `reduced` with the identical saturating
/// algebra.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::readback`。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuReadbackReduceResult {
    /// One reduced `u32` per [`ParticleStatField`], in record order.
    pub reduced: [u32; STAT_FIELD_COUNT],
    /// Saturating sum of the three culling counters (`CulledFrustum` +
    /// `CulledDistance` + `CulledHzb`).
    pub total_culled: u32,
    /// `AliveCount` saturating-minus `total_culled`, clamped at zero on
    /// underflow.
    pub alive_after_cull: u32,
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

/// Saturating sum of the three culling counters, mirroring the golden
/// [`StatSnapshot::total_culled`](prism_render_architecture::particle::readback::StatSnapshot::total_culled).
fn derive_total_culled(reduced: &[u32; STAT_FIELD_COUNT]) -> u32 {
    reduced[ParticleStatField::CulledFrustum.index()]
        .saturating_add(reduced[ParticleStatField::CulledDistance.index()])
        .saturating_add(reduced[ParticleStatField::CulledHzb.index()])
}

/// `AliveCount` saturating-minus `total_culled`, mirroring the golden
/// [`StatSnapshot::alive_after_cull`](prism_render_architecture::particle::readback::StatSnapshot::alive_after_cull).
fn derive_alive_after_cull(reduced: &[u32; STAT_FIELD_COUNT], total_culled: u32) -> u32 {
    reduced[ParticleStatField::AliveCount.index()].saturating_sub(total_culled)
}

/// A compiled, reusable statistics-readback reduction compute pipeline, twinning
/// the `CPU` golden
/// [`readback`](prism_render_architecture::particle::readback).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::readback`。
pub struct GpuReadbackReduce {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuReadbackReduce {
    /// Compiles the partial-reduction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::readback`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuReadbackReduce {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_readback_reduce_module"),
            source: ShaderSource::Wgsl(READBACK_REDUCE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_readback_reduce_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_readback_reduce_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_readback_reduce_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuReadbackReduce {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the full field-wise partial reduction on the device and returns the
    /// reduced record plus the two derived accessors.
    ///
    /// An empty `partials` list issues **no dispatch** — a storage buffer may
    /// not be zero-sized — and returns an all-zero reduced record, exactly as
    /// the golden
    /// [`reduce_partials`](prism_render_architecture::particle::readback::reduce_partials)
    /// yields on an empty slice. The derived accessors are then computed
    /// host-side with the golden saturating algebra.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::readback`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        query: &GpuReadbackReduceQuery,
    ) -> GpuReadbackReduceResult {
        let reduced = if query.partials.is_empty() {
            [0u32; STAT_FIELD_COUNT]
        } else {
            self.dispatch(ctx, &query.partials)
        };
        let total_culled = derive_total_culled(&reduced);
        let alive_after_cull = derive_alive_after_cull(&reduced, total_culled);
        GpuReadbackReduceResult {
            reduced,
            total_culled,
            alive_after_cull,
        }
    }

    /// Issues the one reduction dispatch and reads back the reduced record.
    /// Callers must guarantee a non-empty partial list; the public
    /// [`GpuReadbackReduce::evaluate`] short-circuits the empty case before
    /// calling in.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        partials: &[[u32; STAT_FIELD_COUNT]],
    ) -> [u32; STAT_FIELD_COUNT] {
        let device = ctx.device();

        let params = Params {
            partial_count: partials.len() as u32,
            field_count: STAT_FIELD_COUNT as u32,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_readback_reduce_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        // Flatten the partials row-major: partial i's field j at
        // i*STAT_FIELD_COUNT + j, matching the kernel's indexing.
        let mut flat: Vec<u32> = Vec::with_capacity(partials.len() * STAT_FIELD_COUNT);
        for partial in partials {
            flat.extend_from_slice(partial);
        }
        let partials_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_readback_reduce_partials"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });

        let reduced_bytes = (STAT_FIELD_COUNT as u64) * (size_of::<u32>() as u64);
        let reduced_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_readback_reduce_reduced"),
            size: reduced_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let reduced_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_readback_reduce_reduced_stage"),
            size: reduced_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_readback_reduce_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: partials_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: reduced_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_readback_reduce_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_readback_reduce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per field slot, flattened to a 1-D dispatch.
            let groups = (STAT_FIELD_COUNT as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&reduced_buf, 0, &reduced_stage, 0, reduced_bytes);
        ctx.queue().submit([encoder.finish()]);

        reduced_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let view = reduced_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let words = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        reduced_stage.unmap();

        debug_assert_eq!(words.len(), STAT_FIELD_COUNT);
        let mut reduced = [0u32; STAT_FIELD_COUNT];
        reduced.copy_from_slice(&words);
        reduced
    }
}
