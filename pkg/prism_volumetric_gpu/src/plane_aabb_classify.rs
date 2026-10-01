//! `wgpu` compute twin of the single-plane `AABB` classifier
//! ([`plane_aabb_classify`](prism_render_architecture::particle::plane_aabb_classify),
//! design sections 12-13).
//!
//! The `CPU` golden answers one narrow geometric question: on which side of an
//! oriented plane `n·x + d = 0` does an axis-aligned box lie? It measures the
//! box center's signed distance `s = n·center + d` and the box's projected
//! radius `r = |n_x|·h_x + |n_y|·h_y + |n_z|·h_z` (an `L1`-weighted sum of the
//! half-extents), then reports [`Side::Positive`] when `s - r > CMP_EPS`,
//! [`Side::Negative`] when `s + r < -CMP_EPS`, and [`Side::Intersecting`]
//! otherwise.
//!
//! The `CPU` golden
//! [`plane_aabb_classify`](prism_render_architecture::particle::plane_aabb_classify)
//! owns that math; [`GpuPlaneAabbClassify`] is the on-device twin that runs one
//! thread per query and returns the same signed distance, projected radius and
//! side the reference does. A passing real-device parity test is therefore
//! direct evidence the ported kernel folds the same dot product, takes the same
//! absolute-value weighting and compares against the same [`CMP_EPS`] band the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! All four reference primitives are reproduced in the kernel: the three-term
//! dot product `v_dot`, the center signed distance `signed_distance_center`,
//! the `L1` projected radius `projected_radius`, and the three-way `classify`
//! decision. The accumulation order matches the reference lane order `x, y, z`
//! so the floating-point result agrees to the last few units in the last place.
//! The side decision is encoded as a `u32` discriminant in the shader and
//! mapped back to [`Side`] on the host, where it is compared exactly.
//!
//! # Correctness model
//!
//! The signed distance and projected radius are short, fixed-order sums of
//! products, so `CPU` and `GPU` evaluate the same closed form. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts the two `f32` outputs within a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) and the side discriminant exactly.
//! Fixtures that drive a side decision keep `|s - r|` and `|s + r|` well clear
//! of the [`CMP_EPS`] band so a legal fused multiply-add can never flip the
//! reported side.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - * /`,
//! unsigned index arithmetic and scalar comparisons — with no `sqrt`, `sin`,
//! `exp`, `pow`, `smoothstep` or optional device feature, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. There is no transcendental call on this
//! path.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::plane_aabb_classify`；
//! 无第三方引擎源码或衍生代码。

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::plane_aabb_classify::{Aabb, Plane, Side};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// used across this crate's kernels; the queries are flattened to a single
/// linear index so the dispatch stays one-dimensional.
const WORKGROUP_SIZE: u32 = 64;

/// Side discriminant for a box strictly on the `n·x + d > 0` half-space.
const SIDE_POSITIVE: u32 = 0;
/// Side discriminant for a box strictly on the `n·x + d < 0` half-space.
const SIDE_NEGATIVE: u32 = 1;

/// The portable core-`WGSL` single-plane classifier, embedded inline so the
/// twin ships as a single source file. The entry point `plane_aabb_classify_main`
/// mirrors the `CPU` golden
/// [`classify`](prism_render_architecture::particle::plane_aabb_classify::classify)
/// term for term; see the module documentation for the algorithm.
const PLANE_AABB_CLASSIFY_WGSL: &str = r#"
// Single-plane AABB classifier twin: one thread per query computes the center
// signed distance s = n·center + d, the L1 projected radius
// r = |n_x|·h_x + |n_y|·h_y + |n_z|·h_z, and the three-way side decision. It
// mirrors the CPU golden `particle::plane_aabb_classify`, uses only the
// portable core-WGSL subset (abs and + - * / plus unsigned index math) and
// takes no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::plane_aabb_classify;
// no third-party engine source or derived code.

struct Query {
    // Plane normal (need not be unit length) and constant d.
    nx: f32,
    ny: f32,
    nz: f32,
    d: f32,
    // Box center.
    cx: f32,
    cy: f32,
    cz: f32,
    // Box per-axis half-extents.
    hx: f32,
    hy: f32,
    hz: f32,
}

struct Res {
    // Center signed distance s = n·center + d.
    signed_distance: f32,
    // L1 projected radius r = |n|·|half| folded over the three axes.
    projected_radius: f32,
    // Side discriminant: 0 positive, 1 negative, 2 intersecting.
    side: u32,
}

struct Params {
    // Number of queries dispatched; threads past it return early.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// Absolute tolerance guarding the side decision, mirroring the reference
// `CMP_EPS` so a box that merely grazes the plane is reported as intersecting.
const CMP_EPS: f32 = 1.0e-6;

// Side discriminants, matching the host-side `SIDE_*` constants.
const SIDE_POSITIVE: u32 = 0u;
const SIDE_NEGATIVE: u32 = 1u;
const SIDE_INTERSECTING: u32 = 2u;

// Three-term dot product in the reference lane order x, y, z.
fn v_dot(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

@compute @workgroup_size(64)
fn plane_aabb_classify_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let normal = vec3<f32>(q.nx, q.ny, q.nz);
    let center = vec3<f32>(q.cx, q.cy, q.cz);
    let half = vec3<f32>(q.hx, q.hy, q.hz);

    // s = n·center + d, in the reference accumulation order.
    let s = v_dot(normal, center) + q.d;
    // r = |n_x|·|h_x| + |n_y|·|h_y| + |n_z|·|h_z|, lane order x, y, z.
    let r = abs(normal.x) * abs(half.x)
        + abs(normal.y) * abs(half.y)
        + abs(normal.z) * abs(half.z);

    var side: u32 = SIDE_INTERSECTING;
    if (s - r > CMP_EPS) {
        side = SIDE_POSITIVE;
    } else if (s + r < -CMP_EPS) {
        side = SIDE_NEGATIVE;
    }

    results[idx].signed_distance = s;
    results[idx].projected_radius = r;
    results[idx].side = side;
}
"#;

/// One classification query: an oriented [`Plane`] and an [`Aabb`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneAabbClassifyQuery {
    /// The oriented plane `n·x + d = 0`; the normal need not be unit length.
    pub plane: Plane,
    /// The axis-aligned box being classified.
    pub aabb: Aabb,
}

impl PlaneAabbClassifyQuery {
    /// Builds a query from a plane and a box.
    #[must_use]
    pub const fn new(plane: Plane, aabb: Aabb) -> PlaneAabbClassifyQuery {
        PlaneAabbClassifyQuery { plane, aabb }
    }
}

/// One classification result: the center signed distance, the projected radius
/// and the decided [`Side`], mirroring the reference outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneAabbClassification {
    /// Center signed distance `s = n·center + d`.
    pub signed_distance: f32,
    /// `L1` projected radius `r = |n_x|·h_x + |n_y|·h_y + |n_z|·h_z`.
    pub projected_radius: f32,
    /// The side the box occupies relative to the plane.
    pub side: Side,
}

/// One query as uploaded. `40`-byte `repr(C)` matching `Query` in
/// [`PLANE_AABB_CLASSIFY_WGSL`]: the plane normal, the constant `d`, the box
/// center and the box half-extents, all as scalar `f32` so the `std430` storage
/// stride needs no `vec3` padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    nx: f32,
    ny: f32,
    nz: f32,
    d: f32,
    cx: f32,
    cy: f32,
    cz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
}

/// One result as read back. `12`-byte `repr(C)` matching `Res` in
/// [`PLANE_AABB_CLASSIFY_WGSL`]: the signed distance, the projected radius and
/// the side discriminant.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    signed_distance: f32,
    projected_radius: f32,
    side: u32,
}

/// Uniform parameters for one dispatch. `16`-byte `repr(C)` matching `Params`
/// in [`PLANE_AABB_CLASSIFY_WGSL`]: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Maps a side discriminant read back from the kernel to a [`Side`]. The kernel
/// emits `0` for positive and `1` for negative; every other code (the kernel's
/// `2` intersecting discriminant included) is the conservative
/// [`Side::Intersecting`], matching the reference fall-through.
fn side_from_code(code: u32) -> Side {
    match code {
        SIDE_POSITIVE => Side::Positive,
        SIDE_NEGATIVE => Side::Negative,
        _ => Side::Intersecting,
    }
}

/// A compiled, reusable single-plane `AABB` classification pipeline.
pub struct GpuPlaneAabbClassify {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPlaneAabbClassify {
    /// Compiles the single-plane classification kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPlaneAabbClassify {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_plane_aabb_classify"),
            source: ShaderSource::Wgsl(PLANE_AABB_CLASSIFY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_plane_aabb_classify_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_plane_aabb_classify_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_plane_aabb_classify_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("plane_aabb_classify_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPlaneAabbClassify {
            module,
            layout,
            pipeline,
        }
    }

    /// Classifies every query in `queries`, returning one
    /// [`PlaneAabbClassification`] per query in input order.
    ///
    /// The returned value for query `q` equals the reference
    /// [`signed_distance_center`](prism_render_architecture::particle::plane_aabb_classify::signed_distance_center),
    /// [`projected_radius`](prism_render_architecture::particle::plane_aabb_classify::projected_radius)
    /// and
    /// [`classify`](prism_render_architecture::particle::plane_aabb_classify::classify)
    /// applied to the same inputs: the two `f32` outputs within a tight
    /// floating-point tolerance and the side exactly. An empty `queries` slice
    /// yields an empty result — storage buffers cannot be zero-sized, so it is
    /// handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[PlaneAabbClassifyQuery],
    ) -> Vec<PlaneAabbClassification> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                nx: q.plane.normal[0],
                ny: q.plane.normal[1],
                nz: q.plane.normal[2],
                d: q.plane.d,
                cx: q.aabb.center[0],
                cy: q.aabb.center[1],
                cz: q.aabb.center[2],
                hx: q.aabb.half[0],
                hy: q.aabb.half[1],
                hz: q.aabb.half[2],
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_plane_aabb_classify_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_plane_aabb_classify_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_plane_aabb_classify_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_plane_aabb_classify_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_plane_aabb_classify_bind_group"),
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
            label: Some("prism_volumetric_plane_aabb_classify_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_plane_aabb_classify_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            // One thread per query, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());
        raw.into_iter()
            .map(|r| PlaneAabbClassification {
                signed_distance: r.signed_distance,
                projected_radius: r.projected_radius,
                side: side_from_code(r.side),
            })
            .collect()
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
