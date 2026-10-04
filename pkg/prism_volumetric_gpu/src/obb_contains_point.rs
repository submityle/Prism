//! `wgpu` compute twin of the oriented-bounding-box point-containment test from
//! the `CPU` golden `prism_physics_core::collider::obb::Obb::contains_point`.
//!
//! An `OBB` is a box with an arbitrary orthonormal frame. A point lies inside
//! (surface counts as inside) when each of its three signed projections onto the
//! box axes stays within the matching half-extent, widened by a small adaptive
//! tolerance that absorbs the round-off left by the box-fitting projection. One
//! thread resolves one independent query, writing the boolean containment flag
//! and a validity flag.
//!
//! [`GpuObbContainsPoint`] is the on-device twin; a passing real-device parity
//! test is direct evidence the ported kernel reproduces the same adaptive
//! tolerance and the same three-axis projection test the reference computes, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, independently of the golden crate and
//! of `glam`: the offset `d = point - center`; the adaptive tolerance
//! `tol = 1e-4 + 1e-4 * max_element(half_extents)`; and the conjunction of the
//! three ordered projection tests `|dot(d, axis_i)| <= half_extents_i + tol`.
//! There is no loop and no division: each thread runs a fixed, bounded sequence
//! of multiply-adds and comparisons, so the kernel provably terminates.
//!
//! # Correctness model
//!
//! The output is purely discrete: `contains` is `1` when the point is inside the
//! toleranced box and `0` otherwise, and `valid` is always `1` (the test is
//! total — every frame yields a definite answer). Both flags are compared
//! exactly in the parity test. The intermediate projections thread only through
//! dot products and multiply-adds, so `CPU` and `GPU` evaluate the same closed
//! form; the parity sweep reject-samples any query within `1e-2` of a face knee
//! so a fused multiply-add on the device can never flip the discrete verdict.
//!
//! # Degenerate inputs
//!
//! There is no division and no square root, so there is no degenerate guard: a
//! zero half-extent simply yields a thin slab, and the adaptive tolerance keeps
//! the test well-defined. The caller is expected to pass an orthonormal axis
//! frame, matching the golden `Obb` invariant. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `dot`, `abs`, `max`,
//! `select`, `+ - *` — with no `sin`, `cos`, `exp`, `log`, `pow`, no `round`, no
//! float `%`, and no `u64` / `i64` / `f64`, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. Every branch is an ordered comparison feeding `select`,
//! which is robust under `Metal`'s fast-math (an `x == x` test would be folded
//! to `true`).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::obb`；无第三方引擎
//! 源码或衍生代码。
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

/// The portable core-`WGSL` oriented-bounding-box containment kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden `contains_point`; see the module docs for
/// the closed form it reproduces.
const OBB_CONTAINS_POINT_WGSL: &str = r#"
// Oriented-bounding-box point-containment twin: one thread resolves one
// independent query. It offsets the point by the box centre, builds the
// adaptive tolerance from the largest half-extent, and tests the three signed
// axis projections against the toleranced half-extents. It mirrors the CPU
// golden exactly and uses only the portable core subset.

struct Params {
    // Number of valid queries in the input and output buffers.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Box centre in world space.
    cx: f32,
    cy: f32,
    cz: f32,
    // Box axis 0 (unit).
    a0x: f32,
    a0y: f32,
    a0z: f32,
    // Box axis 1 (unit).
    a1x: f32,
    a1y: f32,
    a1z: f32,
    // Box axis 2 (unit).
    a2x: f32,
    a2y: f32,
    a2z: f32,
    // Half-extents along the three axes.
    hx: f32,
    hy: f32,
    hz: f32,
    // Query point in world space.
    px: f32,
    py: f32,
    pz: f32,
}

struct Verdict {
    // 1 when the point lies inside the toleranced box, 0 otherwise.
    contains: u32,
    // 1 always: the containment test is total.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Verdict>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let center = vec3<f32>(q.cx, q.cy, q.cz);
    let axis0 = vec3<f32>(q.a0x, q.a0y, q.a0z);
    let axis1 = vec3<f32>(q.a1x, q.a1y, q.a1z);
    let axis2 = vec3<f32>(q.a2x, q.a2y, q.a2z);
    let point = vec3<f32>(q.px, q.py, q.pz);

    let d = point - center;
    // Adaptive tolerance from the largest half-extent, matching the golden.
    let max_el = max(max(q.hx, q.hy), q.hz);
    let tol = 1e-4 + 1e-4 * max_el;

    // Three signed axis projections tested against the toleranced half-extents.
    let in0 = abs(dot(d, axis0)) <= q.hx + tol;
    let in1 = abs(dot(d, axis1)) <= q.hy + tol;
    let in2 = abs(dot(d, axis2)) <= q.hz + tol;
    let contains = in0 && in1 && in2;

    var out: Verdict;
    out.contains = select(0u, 1u, contains);
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// Eighteen `f32` give a fixed `72`-byte stride with no trailing pad, since the
/// struct alignment is `4` and `72` is already a multiple of it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    cx: f32,
    cy: f32,
    cz: f32,
    a0x: f32,
    a0y: f32,
    a0z: f32,
    a1x: f32,
    a1y: f32,
    a1z: f32,
    a2x: f32,
    a2y: f32,
    a2z: f32,
    hx: f32,
    hy: f32,
    hz: f32,
    px: f32,
    py: f32,
    pz: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Verdict`
/// struct. Two `u32` give a fixed `8`-byte stride with no pad, since the struct
/// alignment is `4` and `8` is already a multiple of it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    contains: u32,
    valid: u32,
}

/// One query for the oriented-bounding-box containment twin: the box centre, its
/// three orthonormal axes, its half-extents and the query point, all in world
/// space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbContainsPointQuery {
    /// `x` of the box centre.
    pub cx: f32,
    /// `y` of the box centre.
    pub cy: f32,
    /// `z` of the box centre.
    pub cz: f32,
    /// `x` of box axis 0.
    pub a0x: f32,
    /// `y` of box axis 0.
    pub a0y: f32,
    /// `z` of box axis 0.
    pub a0z: f32,
    /// `x` of box axis 1.
    pub a1x: f32,
    /// `y` of box axis 1.
    pub a1y: f32,
    /// `z` of box axis 1.
    pub a1z: f32,
    /// `x` of box axis 2.
    pub a2x: f32,
    /// `y` of box axis 2.
    pub a2y: f32,
    /// `z` of box axis 2.
    pub a2z: f32,
    /// Half-extent along axis 0.
    pub hx: f32,
    /// Half-extent along axis 1.
    pub hy: f32,
    /// Half-extent along axis 2.
    pub hz: f32,
    /// `x` of the query point.
    pub px: f32,
    /// `y` of the query point.
    pub py: f32,
    /// `z` of the query point.
    pub pz: f32,
}

impl ObbContainsPointQuery {
    /// Builds a query from the box frame (centre, three axes, half-extents) and
    /// the query point.
    ///
    /// The vectors are grouped into fixed-length arrays so the constructor stays
    /// within a small, readable argument count.
    #[must_use]
    pub fn new(
        center: [f32; 3],
        axis0: [f32; 3],
        axis1: [f32; 3],
        axis2: [f32; 3],
        half_extents: [f32; 3],
        point: [f32; 3],
    ) -> ObbContainsPointQuery {
        ObbContainsPointQuery {
            cx: center[0],
            cy: center[1],
            cz: center[2],
            a0x: axis0[0],
            a0y: axis0[1],
            a0z: axis0[2],
            a1x: axis1[0],
            a1y: axis1[1],
            a1z: axis1[2],
            a2x: axis2[0],
            a2y: axis2[1],
            a2z: axis2[2],
            hx: half_extents[0],
            hy: half_extents[1],
            hz: half_extents[2],
            px: point[0],
            py: point[1],
            pz: point[2],
        }
    }
}

/// One resolved verdict for a single query: the containment flag and the
/// validity flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbContainsPointResult {
    /// `1` when the point lies inside the toleranced box, `0` otherwise.
    pub contains: u32,
    /// `1` always: the containment test is total.
    pub valid: u32,
}

/// Encodes one [`ObbContainsPointQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ObbContainsPointQuery) -> GpuQuery {
    GpuQuery {
        cx: q.cx,
        cy: q.cy,
        cz: q.cz,
        a0x: q.a0x,
        a0y: q.a0y,
        a0z: q.a0z,
        a1x: q.a1x,
        a1y: q.a1y,
        a1z: q.a1z,
        a2x: q.a2x,
        a2y: q.a2y,
        a2z: q.a2z,
        hx: q.hx,
        hy: q.hy,
        hz: q.hz,
        px: q.px,
        py: q.py,
        pz: q.pz,
    }
}

/// Decodes one `std430` [`GpuResult`] slot into a public
/// [`ObbContainsPointResult`].
fn decode_result(r: &GpuResult) -> ObbContainsPointResult {
    ObbContainsPointResult {
        contains: r.contains,
        valid: r.valid,
    }
}

/// Builds a read-only or read-write storage-buffer bind-group-layout entry at
/// `binding`.
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

/// The on-device twin of the `CPU` golden `contains_point`: a compiled compute
/// pipeline that resolves a batch of oriented-bounding-box containment queries.
pub struct GpuObbContainsPoint {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuObbContainsPoint {
    /// Compiles the inline kernel and builds the compute pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuObbContainsPoint {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_obb_contains_point_module"),
            source: ShaderSource::Wgsl(OBB_CONTAINS_POINT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_obb_contains_point_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_obb_contains_point_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_obb_contains_point_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuObbContainsPoint {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`ObbContainsPointResult`] per input, in order.
    ///
    /// The `contains` and `valid` flags match the reference exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ObbContainsPointQuery],
    ) -> Vec<ObbContainsPointResult> {
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
            label: Some("prism_volumetric_obb_contains_point_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_obb_contains_point_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_obb_contains_point_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_obb_contains_point_bind_group"),
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
            label: Some("prism_volumetric_obb_contains_point_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_obb_contains_point_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_obb_contains_point_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
