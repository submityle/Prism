//! `wgpu` compute twin of the per-particle *trail ring-buffer ordering* golden
//! [`iter_ordered_indices`](prism_render_architecture::particle::ribbon_trail::iter_ordered_indices)
//! (design §15, "飘带/轨迹历史环形缓冲").
//!
//! The `CPU` golden
//! [`ribbon_trail`](prism_render_architecture::particle::ribbon_trail) owns a
//! fixed-capacity per-trail ring buffer and, through
//! [`iter_ordered_indices`](prism_render_architecture::particle::ribbon_trail::iter_ordered_indices),
//! reports the ring-slot indices of a trail's samples from oldest to newest.
//! Before the ring fills (`count < capacity`) the samples occupy slots
//! `0..count` in capture order; once full the oldest sample sits at `head` (the
//! next-to-overwrite slot) and iteration wraps around the ring, so the `i`-th
//! ordered sample lives in slot `(head + i) % capacity`.
//!
//! [`GpuRibbonTrail`] is the on-device twin: one thread resolves one
//! [`GpuRibbonTrailQuery`] — a ring `(head, count, capacity)` plus the ordered
//! element index `element` — into one [`GpuRibbonTrailResult`] carrying the
//! resolved ring `slot`, the ordered-run length `ordered_len` and a `valid`
//! flag. A passing real-device parity test is therefore direct evidence the
//! ported kernel reproduces the same ordered slot sequence the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries the ring `head`, the capture `count`, the ring
//! `capacity`, and the ordered sample index `element`. The kernel reproduces
//! the golden ordering: `ordered_len = min(count, capacity)` (and `0` for a
//! zero-capacity ring), and for an in-range `element` the ring slot is
//! `element` while the ring is filling (`count < capacity`) or
//! `(head + element) % capacity` once it has saturated. An `element` at or past
//! `ordered_len` is reported as degenerate (`slot = 0`, `valid = 0`).
//!
//! # Correctness model
//!
//! Every value is a `u32` index, length or `bool`-flavoured flag — pure integer
//! ring arithmetic with no rounding — so `CPU` and `GPU` compute identical bit
//! patterns and the parity test asserts an exact `==` on every field with no
//! tolerance. `WGSL` has no `u64`, so where the golden evaluates the modulo in
//! `u64` the kernel replaces it with a single *conditional subtract*: with
//! `head < capacity` and `element < capacity` the sum `head + element` is below
//! `2 * capacity`, so one subtraction of `capacity` reduces it exactly like
//! `(head + element) % capacity`. Fixtures use rejection sampling to keep
//! `head < capacity` and `count <= capacity` (so the "full" branch is the
//! saturated ring), and keep `head + count` far below [`u32::MAX`] so the
//! device `u32` add never wraps where the golden `u64` add cannot.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `+` and `-`,
//! the `min` built-in, unsigned comparisons and index arithmetic. There is no
//! `sqrt`, no divide, no modulo, no transcendental call, no `u64` and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. There is no loop: each thread performs a fixed, bounded sequence of
//! integer work, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ribbon_trail`；无第三方引擎源码或衍生代码。
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

/// The trail ring-ordering kernel, mirroring the `CPU` golden
/// [`iter_ordered_indices`](prism_render_architecture::particle::ribbon_trail::iter_ordered_indices)
/// element for element. The single entry point `solve` resolves one query per
/// thread, embedded inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::ribbon_trail`。
const RIBBON_TRAIL_WGSL: &str = r#"
// ribbon_trail twin: one thread per query reproduces the CPU golden
// `particle::ribbon_trail::iter_ordered_indices`. A query is a ring
// (head, count, capacity) plus the ordered element index `element`; the kernel
// returns the ring slot of that ordered sample, the ordered-run length and a
// validity flag. Before the ring fills (count < capacity) the i-th ordered
// sample lives in slot `i`; once full the oldest sample sits at `head` and
// iteration wraps, so slot = (head + i) % capacity. WGSL has no u64, so the
// modulo is replaced by a single conditional subtract, exact while head <
// capacity and element < capacity (both guaranteed by the fixture domain).
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::ribbon_trail；无第三方
// 引擎源码或衍生代码。

// Dispatch parameters. 16-byte uniform block: the valid query count plus three
// pad words, matching the host `Params`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 16-byte std430 stride of four scalar words, matching the host
// `GpuRibbonTrailQuery`: the ring head, capture count, ring capacity and the
// ordered element index.
struct RibbonTrailQuery {
    head: u32,
    count: u32,
    capacity: u32,
    element: u32,
}

// One result. 16-byte std430 stride of four scalar words, matching the host
// `GpuRibbonTrailResult`: the resolved ring slot, the validity flag, the
// ordered-run length and one pad word.
struct RibbonTrailResult {
    slot: u32,
    valid: u32,
    ordered_len: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<RibbonTrailQuery>;
@group(0) @binding(2) var<storage, read_write> results: array<RibbonTrailResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: RibbonTrailResult;
    out.slot = 0u;
    out.valid = 0u;
    out.ordered_len = 0u;
    out.pad0 = 0u;

    // A zero-capacity ring holds no samples: the degenerate guard keeps the
    // kernel clear of a divide/modulo-by-zero and yields an empty ordering.
    if (q.capacity == 0u) {
        results[idx] = out;
        return;
    }

    // The ordered run is `min(count, capacity)` samples long: it grows with the
    // capture count until the ring saturates at its capacity.
    let ordered_len = min(q.count, q.capacity);
    out.ordered_len = ordered_len;

    // Only element indices inside the ordered run map to a real slot; indices
    // at or past it stay degenerate (slot 0, valid 0).
    if (q.element >= ordered_len) {
        results[idx] = out;
        return;
    }

    var slot: u32;
    if (q.count < q.capacity) {
        // Before the ring fills, ordered sample `i` is simply slot `i`.
        slot = q.element;
    } else {
        // Once full the oldest sample sits at `head` and iteration wraps.
        // head < capacity and element < capacity, so head + element is below
        // 2 * capacity and a single conditional subtract reproduces the golden
        // (head + element) % capacity without a u64 modulo.
        var s = q.head + q.element;
        if (s >= q.capacity) {
            s = s - q.capacity;
        }
        slot = s;
    }

    out.slot = slot;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`RIBBON_TRAIL_WGSL`]: the valid query count plus three pad
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

/// One query for the trail ring-ordering twin: a ring `(head, count, capacity)`
/// plus the ordered element index `element`.
///
/// `head` is the ring write head (the oldest, next-to-overwrite slot once the
/// ring is full), `count` the number of samples captured so far, `capacity` the
/// ring capacity, and `element` the position in the oldest-to-newest ordering
/// whose ring slot is requested. The twin's exact domain keeps `head` below
/// `capacity` and `count` at most `capacity`, matching the golden
/// [`iter_ordered_indices`](prism_render_architecture::particle::ribbon_trail::iter_ordered_indices)
/// invariant that the write head stays inside the ring. The `repr(C)` layout —
/// four `u32` words, `16` bytes with no padding — matches the `WGSL`
/// `RibbonTrailQuery` struct exactly, so it is uploaded to the device without a
/// separate encode step.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::ribbon_trail`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuRibbonTrailQuery {
    /// Ring write head: the oldest, next-to-overwrite slot once the ring fills.
    pub head: u32,
    /// Number of samples captured so far.
    pub count: u32,
    /// Ring capacity (points retained before overwrite).
    pub capacity: u32,
    /// Ordered element index, counting from the oldest retained sample.
    pub element: u32,
}

/// One resolved answer for a single [`GpuRibbonTrailQuery`], mirroring the ring
/// slot the golden
/// [`iter_ordered_indices`](prism_render_architecture::particle::ribbon_trail::iter_ordered_indices)
/// assigns to the queried ordered element.
///
/// `slot` is the ring-buffer slot of the ordered element (meaningful only when
/// `valid` is `1`), `valid` the `1`/`0` flag reporting whether `element` fell
/// inside the ordered run, and `ordered_len` the ordered-run length
/// `min(count, capacity)` (`0` for a zero-capacity ring). The `repr(C)` layout —
/// four `u32` words, `16` bytes — matches the `WGSL` `RibbonTrailResult` struct
/// exactly, so device results are read back without a separate decode step.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::ribbon_trail`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuRibbonTrailResult {
    /// Ring-buffer slot of the ordered element; meaningful only when `valid`.
    pub slot: u32,
    /// Validity flag, `1` when `element` is inside the ordered run and `0`
    /// otherwise (including the empty / zero-capacity ring).
    pub valid: u32,
    /// Ordered-run length `min(count, capacity)`, or `0` for a zero capacity.
    pub ordered_len: u32,
    /// Padding word, always `0`.
    pub pad0: u32,
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

/// A compiled, reusable trail ring-ordering compute pipeline, twinning the
/// `CPU` golden
/// [`iter_ordered_indices`](prism_render_architecture::particle::ribbon_trail::iter_ordered_indices).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::ribbon_trail`。
pub struct GpuRibbonTrail {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRibbonTrail {
    /// Compiles the trail ring-ordering kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::ribbon_trail`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRibbonTrail {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ribbon_trail_module"),
            source: ShaderSource::Wgsl(RIBBON_TRAIL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ribbon_trail_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ribbon_trail_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ribbon_trail_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRibbonTrail {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`GpuRibbonTrailResult`] per input, in order.
    ///
    /// Each result equals the matching golden answer exactly — the ring `slot`,
    /// the `valid` flag and the `ordered_len` all mirror the `CPU`
    /// [`iter_ordered_indices`](prism_render_architecture::particle::ribbon_trail::iter_ordered_indices)
    /// reference — because the whole path is integer ring algebra. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::ribbon_trail`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GpuRibbonTrailQuery],
    ) -> Vec<GpuRibbonTrailResult> {
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
            label: Some("prism_volumetric_ribbon_trail_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ribbon_trail_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (queries.len() as u64) * (size_of::<GpuRibbonTrailResult>() as u64);
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ribbon_trail_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ribbon_trail_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ribbon_trail_bind_group"),
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
            label: Some("prism_volumetric_ribbon_trail_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ribbon_trail_pass"),
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuRibbonTrailResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
    }
}
