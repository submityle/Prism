//! `wgpu` compute twin of the integer midpoint-circle boundary rasterizer
//! ([`midpoint_circle`](prism_render_architecture::particle::midpoint_circle),
//! particle design §12-§13, §16).
//!
//! The `CPU` golden
//! [`rasterize`](prism_render_architecture::particle::midpoint_circle::rasterize)
//! turns a center `(cx, cy)` and an integer radius `r` into the set of lattice
//! points closest to the ideal circle, using the classic *midpoint circle*
//! walk: it steps `x` from `0` while `x <= y` (with `y` starting at `r`), keeps a
//! single integer decision variable `d = 1 - r`, and at each step either stays
//! on the current row (`d += 2*x + 3`) or drops one row inward
//! (`d += 2*(x - y) + 5`, `y -= 1`). Every visited octant point `(x, y)` is
//! reflected through the circle's eight symmetries. [`GpuMidpointCircle`] is the
//! on-device twin: one thread walks one circle, writing the eight symmetric
//! points of each iteration into a fixed per-query slot plus a running `count`,
//! so a passing real-device parity test is direct evidence the ported integer
//! walk visits the same lattice points the reference does.
//!
//! # What is twinned
//!
//! Only the geometric *generation* of the boundary runs on the `GPU`: the
//! per-iteration decision-variable recurrence and the eight-way symmetry push.
//! The kernel emits the raw point stream exactly as the octant walk produces it
//! — unsorted and with the coincident axis/diagonal reflections still present.
//!
//! The golden finishes with `sort_unstable` followed by `dedup`; this twin
//! reproduces that final step as a **host-side post-process**, not on the
//! `GPU`. After reading back the raw point stream and its count,
//! [`GpuMidpointCircle::rasterize`] applies the identical `sort_unstable` +
//! `dedup` on the host before comparing. This is an honest split: the lattice
//! geometry is computed on device, and the sort-and-dedup contract is a `CPU`
//! post-process, so the twin faithfully reproduces the reference point set
//! without pretending the `GPU` performed the ordering.
//!
//! # Degenerate radii
//!
//! The two special cases match the reference exactly and are handled on device:
//! `r < 0` is not a valid radius and yields an empty point set (`count == 0`);
//! `r == 0` yields the single center point `(cx, cy)` (`count == 1`), not the
//! eight-way push.
//!
//! # Integer range and capacity
//!
//! The whole walk is pure `i32` arithmetic: additions, an integer multiply by
//! two and comparisons on the decision variable, with no `f32`, no `sqrt` and no
//! transcendental call anywhere. The intermediate terms `2*x + 3` and
//! `2*(x - y) + 5`, and the accumulated `d`, all stay well within `i32` for the
//! supported radius subset `0 <= r <= `[`MAX_RADIUS`]. The twin does **not**
//! guarantee the full `i32` radius range: a radius near [`i32::MAX`] would
//! overflow those products, so callers must stay within the supported subset.
//!
//! Each thread writes into a fixed `std430` slot of [`MAX_POINTS`] lattice
//! points. The raw octant walk emits eight points per iteration and runs for
//! about `r / sqrt(2)` iterations, so [`MAX_POINTS`] is chosen large enough to
//! hold every raw point for any `r` up to [`MAX_RADIUS`] with margin; the kernel
//! also guards each eight-point block against the capacity so a malformed count
//! can never index past the fixed array.
//!
//! # What is *not* twinned
//!
//! The reference's `i64`-valued helpers are deliberately left on the `CPU`,
//! because the portable core-`WGSL` subset this crate targets has no 64-bit
//! integer type:
//!
//! * [`dist_sq`](prism_render_architecture::particle::midpoint_circle::dist_sq)
//!   returns an [`i64`] squared length so large offsets cannot overflow; a
//!   core-`WGSL` kernel has only `i32` / `u32` and cannot represent that result.
//! * [`filled_disk`](prism_render_architecture::particle::midpoint_circle::filled_disk)
//!   scans each row with an `i64` squared-distance inequality (`r_sq` and
//!   `dist_sq` both widen to [`i64`]); that same 64-bit comparison is outside the
//!   core-`WGSL` subset, so the solid-disk fill is not ported here.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `i32` additions, a
//! multiply by two, comparisons and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `sqrt` and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. The single loop is bounded by the radius (and the capacity guard), so
//! the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::midpoint_circle`；无第三方引擎源码或衍生代码。
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

/// Fixed per-circle capacity: the maximum number of *raw* lattice points one
/// query's octant walk may emit before the host sort-and-dedup.
///
/// The walk pushes eight points per iteration and runs for about `r / sqrt(2)`
/// iterations, so this budget comfortably holds every raw point for any radius
/// up to [`MAX_RADIUS`]. [`GpuMidpointCircle::rasterize`] rejects any larger
/// radius so the fixed slot can never overflow.
pub const MAX_POINTS: usize = 4096;

/// Largest radius the twin accepts. Within `0 <= r <= MAX_RADIUS` every
/// intermediate `i32` term (`2*x + 3`, `2*(x - y) + 5`, the accumulated `d`)
/// stays within `i32`, and the raw point stream fits inside [`MAX_POINTS`]. The
/// twin makes no promise for radii beyond this subset, up to [`i32::MAX`].
pub const MAX_RADIUS: i32 = 512;

/// The portable core-`WGSL` midpoint-circle kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `rasterize` mirrors the
/// `CPU` golden
/// [`midpoint_circle`](prism_render_architecture::particle::midpoint_circle)
/// walk step for step; see the module documentation for the algorithm.
const MIDPOINT_CIRCLE_WGSL: &str = r#"
// Midpoint-circle twin: one thread rasterizes one circle's integer boundary. It
// mirrors the CPU golden `particle::midpoint_circle::rasterize` walk step for
// step — the `d = 1 - r` decision-variable recurrence and the eight-way symmetry
// push — emitting the raw point stream (unsorted, with coincident reflections)
// plus a running count. The final sort_unstable + dedup is a host post-process,
// not performed here. The kernel uses only the portable core-WGSL subset (i32
// additions, a multiply by two, comparisons and unsigned index math), needs no
// sqrt and no transcendental call and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12. The single loop is bounded by the radius
// and the capacity guard, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::midpoint_circle；无第三方
// 引擎源码或衍生代码。

// Fixed per-circle raw-point capacity, mirroring the host `MAX_POINTS`. Each
// eight-point block checks this bound so a malformed count can never index past
// the fixed array.
const MAX_POINTS: u32 = 4096u;

struct Params {
    // Number of circle queries in the storage arrays; threads past this
    // short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Circle center and radius; a pad lane keeps the 16-byte std430 stride.
    cx: i32,
    cy: i32,
    r: i32,
    pad: i32,
}

struct Circle {
    // Number of raw lattice points written to `points`; three pad lanes lift the
    // fixed array to its vec2<i32>-aligned offset. Lanes at or past `count` are
    // never read.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    points: array<vec2<i32>, 4096>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Circle>;

@compute @workgroup_size(64)
fn rasterize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let cx = queries[idx].cx;
    let cy = queries[idx].cy;
    let r = queries[idx].r;

    var n: u32 = 0u;
    // r < 0 is not a valid radius: empty point set, mirroring the reference.
    if (r < 0) {
        results[idx].count = 0u;
        return;
    }
    // r == 0 is the single center point, not the eight-way push.
    if (r == 0) {
        results[idx].points[0] = vec2<i32>(cx, cy);
        results[idx].count = 1u;
        return;
    }

    var x: i32 = 0;
    var y: i32 = r;
    // Midpoint decision variable, initialized to 1 - r. When it is negative the
    // ideal circle passes above the midpoint, so the next pixel keeps the
    // current row; otherwise it drops one row inward.
    var d: i32 = 1 - r;
    loop {
        if (x > y) {
            break;
        }
        // Push the eight symmetric reflections of (x, y) about (cx, cy) in the
        // exact order the reference `push_octant_symmetry` emits them. The host
        // sort-and-dedup collapses the coincident axis/diagonal reflections.
        if (n + 8u <= MAX_POINTS) {
            results[idx].points[n + 0u] = vec2<i32>(cx + x, cy + y);
            results[idx].points[n + 1u] = vec2<i32>(cx - x, cy + y);
            results[idx].points[n + 2u] = vec2<i32>(cx + x, cy - y);
            results[idx].points[n + 3u] = vec2<i32>(cx - x, cy - y);
            results[idx].points[n + 4u] = vec2<i32>(cx + y, cy + x);
            results[idx].points[n + 5u] = vec2<i32>(cx - y, cy + x);
            results[idx].points[n + 6u] = vec2<i32>(cx + y, cy - x);
            results[idx].points[n + 7u] = vec2<i32>(cx - y, cy - x);
            n = n + 8u;
        }
        if (d < 0) {
            // Stay on the same row: advance x only.
            d = d + 2 * x + 3;
        } else {
            // Step inward: advance x and drop y by one row.
            d = d + 2 * (x - y) + 5;
            y = y - 1;
        }
        x = x + 1;
    }
    results[idx].count = n;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MIDPOINT_CIRCLE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid circle queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one circle query, matching the `WGSL` `Query`
/// struct: the center, the radius and one pad lane for a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Circle center `x`.
    cx: i32,
    /// Circle center `y`.
    cy: i32,
    /// Circle radius.
    r: i32,
    /// Padding lane.
    pad: i32,
}

/// `repr(C)` `std430` layout of one rasterized circle, matching the `WGSL`
/// `Circle` struct: the raw point count, three pad lanes that lift the fixed
/// array to its `vec2<i32>`-aligned offset, and the fixed [`MAX_POINTS`] slot of
/// `(x, y)` lanes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCircleOut {
    /// Number of raw lattice points written to `points`.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// The raw octant-walk point stream; lanes at or past `count` are unused.
    points: [[i32; 2]; MAX_POINTS],
}

/// One circle-rasterization query: a center `(cx, cy)` and an integer radius.
///
/// Mirrors a single reference
/// [`rasterize`](prism_render_architecture::particle::midpoint_circle::rasterize)
/// call. The radius must satisfy `r <= `[`MAX_RADIUS`] (any negative radius is
/// accepted and yields an empty point set). Integer-only, so it derives [`Eq`]
/// and [`Hash`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuMidpointCircleQuery {
    /// Circle center `x`.
    pub cx: i32,
    /// Circle center `y`.
    pub cy: i32,
    /// Circle radius; `r < 0` yields an empty point set, `r == 0` the single
    /// center point, and `0 < r <= `[`MAX_RADIUS`] the full boundary.
    pub r: i32,
}

/// The rasterized boundary of one circle, read back from the kernel and sorted
/// and deduplicated on the host.
///
/// `points` is the final lattice-point set after the host `sort_unstable` +
/// `dedup`, so it matches the reference
/// [`rasterize`](prism_render_architecture::particle::midpoint_circle::rasterize)
/// output exactly — ascending by `x`, then by `y`, with no duplicates. Its
/// length is the deduplicated point count. Integer-only, so it derives [`Eq`]
/// and [`Hash`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct GpuMidpointCircleResult {
    /// The sorted, deduplicated boundary lattice points `(x, y)`.
    pub points: Vec<(i32, i32)>,
}

impl GpuQuery {
    /// Packs a [`GpuMidpointCircleQuery`] into the `std430` upload layout.
    ///
    /// # Panics
    ///
    /// Panics when `query.r` exceeds [`MAX_RADIUS`], since the raw octant walk
    /// could then overflow the fixed [`MAX_POINTS`] slot.
    fn from_query(query: &GpuMidpointCircleQuery) -> GpuQuery {
        assert!(
            query.r <= MAX_RADIUS,
            "GpuMidpointCircleQuery radius {} exceeds MAX_RADIUS ({MAX_RADIUS})",
            query.r
        );
        GpuQuery {
            cx: query.cx,
            cy: query.cy,
            r: query.r,
            pad: 0,
        }
    }
}

/// Decodes one packed [`GpuCircleOut`] into the public
/// [`GpuMidpointCircleResult`], applying the reference's `sort_unstable` +
/// `dedup` on the host so the point set matches the golden exactly.
fn decode_result(raw: &GpuCircleOut) -> GpuMidpointCircleResult {
    let count = (raw.count as usize).min(MAX_POINTS);
    let mut points: Vec<(i32, i32)> = raw.points[..count].iter().map(|&[x, y]| (x, y)).collect();
    points.sort_unstable();
    points.dedup();
    GpuMidpointCircleResult { points }
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

/// A compiled, reusable midpoint-circle compute pipeline, twinning the `CPU`
/// golden
/// [`midpoint_circle`](prism_render_architecture::particle::midpoint_circle).
pub struct GpuMidpointCircle {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMidpointCircle {
    /// Compiles the midpoint-circle kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMidpointCircle {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_midpoint_circle"),
            source: ShaderSource::Wgsl(MIDPOINT_CIRCLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_midpoint_circle_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_midpoint_circle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_midpoint_circle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("rasterize"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMidpointCircle {
            module,
            layout,
            pipeline,
        }
    }

    /// Rasterizes every circle in `queries` and returns one
    /// [`GpuMidpointCircleResult`] per input, in order.
    ///
    /// Each result's `points` is the raw octant-walk stream read back from the
    /// `GPU` after the host `sort_unstable` + `dedup`, so it equals the reference
    /// [`rasterize`](prism_render_architecture::particle::midpoint_circle::rasterize)
    /// point set exactly. An empty `queries` batch returns an empty vector with
    /// no dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// # Panics
    ///
    /// Panics when any query's radius exceeds [`MAX_RADIUS`].
    #[must_use]
    pub fn rasterize(
        &self,
        ctx: &GpuContext,
        queries: &[GpuMidpointCircleQuery],
    ) -> Vec<GpuMidpointCircleResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_midpoint_circle_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_midpoint_circle_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuCircleOut>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_midpoint_circle_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_midpoint_circle_bind_group"),
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
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_midpoint_circle_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_midpoint_circle_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_midpoint_circle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per circle, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuCircleOut>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
