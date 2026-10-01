//! `wgpu` compute twin of the analytic *closest point on an oriented bounding
//! box* (`OBB`) golden
//! ([`closest_point_obb`](prism_render_architecture::particle::closest_point_obb),
//! design §8.2, §10, §14).
//!
//! Many particle stages need the nearest point of a rotated box to an arbitrary
//! point: clamping a spawn or collision proxy back onto an oriented volume,
//! resolving a penetration against a rotated bound, snapping a decal footprint
//! onto an oriented brick, or answering a proximity query whose bounds are
//! rotated. The `CPU` golden
//! [`Obb::closest_point`](prism_render_architecture::particle::closest_point_obb::Obb::closest_point)
//! owns that math; [`GpuClosestPointObb`] is the on-device twin that runs one
//! thread per query and reproduces every lane. A passing real-device parity test
//! is therefore direct evidence the ported kernel projects the query into the
//! box local frame and folds the same three per-axis interval clamps the
//! reference does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! An `OBB` is an axis-aligned box rotated into world space: a `center`, three
//! mutually orthogonal **unit** axes and a non-negative half-extent along each.
//! Because an orthonormal basis is its own inverse, the signed coordinate of a
//! point `p` in the box frame along axis `i` is simply the dot product
//! `(p - center) . axes[i]`. Clamping that coordinate to `[-half[i], half[i]]`
//! and rebuilding a world point from the clamped coordinates yields the nearest
//! point of the box: each local axis is solved independently, so no search or
//! iteration is required. The point is *inside* exactly when every clamp was a
//! no-op, i.e. every projection's magnitude is within its half-extent. The
//! kernel reproduces the reference clamp-for-clamp and reads back the nearest
//! world point, the Euclidean distance and squared distance, and the inside
//! flag.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `abs`,
//! `min`, `max`, `dot`, `+ - * /` and one `sqrt` for the single genuine
//! Euclidean length — with no `sin`, `cos`, `exp`, `log`, `pow` or optional
//! device feature (the box faces are flat, so there is no quadratic). It
//! therefore runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of three per-axis clamps and
//! one length, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the point and distances yet an
//! *exact* match on the discrete inside flag, which is routed through the same
//! `<=` half-extent band the reference uses so a query placed clear of a face
//! boundary folds the identical boolean verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::closest_point_obb`；
//! standard per-axis interval clamp of a point against an `OBB` plus `wgpu`
//! compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::closest_point_obb::{Obb, Vec3};
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
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Discrete inside code written by the kernel for a lane whose query lies within
/// (or on the surface of) the box: matches the host `== 1` decode in
/// [`decode_result`]. A direct `f32` equality is forbidden, so the kernel emits
/// an integer flag rather than a sentinel float.
const CODE_INSIDE: u32 = 1;

/// The portable core-`WGSL` point-to-`OBB` nearest-point kernel, embedded inline
/// so the twin ships as a single source file. Mirrors the `CPU` golden
/// [`Obb::closest_point`](prism_render_architecture::particle::closest_point_obb::Obb::closest_point)
/// clamp-for-clamp; see the module documentation for the algorithm.
const CLOSEST_POINT_OBB_WGSL: &str = r#"
// Point vs OBB nearest-point twin: one thread per (point, box) query projects
// the query into the box local frame by dotting the origin-offset against the
// three unit axes, clamps each projection to [-half, half], rebuilds the world
// nearest point from the clamped projections, and writes that point, the
// Euclidean distance and squared distance, and the inside flag (true only when
// every projection magnitude is within its half-extent). It mirrors the CPU
// golden `particle::closest_point_obb` clamp for clamp, uses only the portable
// core-WGSL subset (clamp/abs/min/max, dot, + - * / and one sqrt), and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard per-axis interval clamp of a point against an OBB; no
// third-party engine source or derived code.

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 96-byte std430 stride matching the host `GpuQuery`: the query
// point, the box center, its three local unit axes and its half-extents, each
// padded to a vec4 so the storage array needs no manual vec3 alignment
// arithmetic.
struct Query {
    point: vec4<f32>,
    center: vec4<f32>,
    axis0: vec4<f32>,
    axis1: vec4<f32>,
    axis2: vec4<f32>,
    half_extents: vec4<f32>,
}

// One result. 32-byte std430 stride matching the host `GpuResult`: the nearest
// world point (in a vec4 lane), the Euclidean distance and squared distance,
// the inside flag as 0u/1u and one pad word.
struct Result {
    point: vec4<f32>,
    distance: f32,
    distance_squared: f32,
    inside: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Clamps one projected coordinate onto a single axis, mirroring one iteration of
// the reference loop: `proj` is the origin-offset projected onto the (unit)
// axis, clamped to [-half, half]. Returns the clamped projection; the caller
// scales the axis by it and accumulates the world point.
fn clamp_axis(proj: f32, half: f32) -> f32 {
    return clamp(proj, -half, half);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let query = queries[idx].point.xyz;
    let center = queries[idx].center.xyz;
    let axis0 = queries[idx].axis0.xyz;
    let axis1 = queries[idx].axis1.xyz;
    let axis2 = queries[idx].axis2.xyz;
    let half_extents = queries[idx].half_extents.xyz;

    let rel = query - center;
    var point = center;
    var inside = 1u;

    // Axis 0.
    let p0 = dot(rel, axis0);
    // `abs(proj) > half` is a `>` test, never an f32 equality: a projection
    // whose magnitude exceeds the half-extent was clamped, so the point is
    // outside along this axis.
    if (abs(p0) > half_extents.x) {
        inside = 0u;
    }
    point = point + axis0 * clamp_axis(p0, half_extents.x);

    // Axis 1.
    let p1 = dot(rel, axis1);
    if (abs(p1) > half_extents.y) {
        inside = 0u;
    }
    point = point + axis1 * clamp_axis(p1, half_extents.y);

    // Axis 2.
    let p2 = dot(rel, axis2);
    if (abs(p2) > half_extents.z) {
        inside = 0u;
    }
    point = point + axis2 * clamp_axis(p2, half_extents.z);

    let diff = point - query;
    let distance_squared = dot(diff, diff);

    var res: Result;
    res.point = vec4<f32>(point, 0.0);
    res.distance = sqrt(distance_squared);
    res.distance_squared = distance_squared;
    res.inside = inside;
    res.pad0 = 0u;

    results[idx] = res;
}
"#;

/// One point vs `OBB` nearest-point query: the query point and the oriented box
/// to clamp it onto.
///
/// Mirrors a single reference
/// [`Obb::closest_point`](prism_render_architecture::particle::closest_point_obb::Obb::closest_point)
/// call. Carrying the box per query lets one dispatch mix points against many
/// distinct boxes. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds
/// `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestPointObbQuery {
    /// The query point whose nearest box point is sought.
    pub point: Vec3,
    /// The oriented box the point is clamped onto.
    pub obb: Obb,
}

/// The resolved nearest-point reading for one query, the host-side mirror of the
/// kernel's `Result` lane.
///
/// `point` is the nearest point of the closed, filled box to the query;
/// `distance` and `distance_squared` are the Euclidean and squared Euclidean
/// distances to it; `inside` is `true` when the query lies inside (or on the
/// surface of) the box. Mirrors
/// [`ClosestPoint`](prism_render_architecture::particle::closest_point_obb::ClosestPoint).
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32`
/// parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestPointObbResult {
    /// The nearest point of the box to the query, in world space.
    pub point: Vec3,
    /// The Euclidean distance from the query point to [`Self::point`].
    pub distance: f32,
    /// The squared Euclidean distance (no `sqrt`), handy for comparisons.
    pub distance_squared: f32,
    /// Whether the query point lies inside (or on the surface of) the box.
    pub inside: bool,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`CLOSEST_POINT_OBB_WGSL`]: the query count and three pad words —
/// `16` bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `96`-byte `std430` stride matching `Query` in the
/// shader: the query point, the box center, its three local unit axes and its
/// half-extents, each padded to a `vec4` lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point in `xyz`; the `w` lane is unused padding.
    point: [f32; 4],
    /// Box center in `xyz`; the `w` lane is unused padding.
    center: [f32; 4],
    /// First local unit axis in `xyz`; the `w` lane is unused padding.
    axis0: [f32; 4],
    /// Second local unit axis in `xyz`; the `w` lane is unused padding.
    axis1: [f32; 4],
    /// Third local unit axis in `xyz`; the `w` lane is unused padding.
    axis2: [f32; 4],
    /// Per-axis half-extents in `xyz`; the `w` lane is unused padding.
    half_extents: [f32; 4],
}

impl GpuQuery {
    /// Packs a [`ClosestPointObbQuery`] into the `std430` upload layout.
    fn from_query(query: &ClosestPointObbQuery) -> GpuQuery {
        let p = query.point;
        let c = query.obb.center;
        let a0 = query.obb.axes[0];
        let a1 = query.obb.axes[1];
        let a2 = query.obb.axes[2];
        let h = query.obb.half;
        GpuQuery {
            point: [p.x, p.y, p.z, 0.0],
            center: [c.x, c.y, c.z, 0.0],
            axis0: [a0.x, a0.y, a0.z, 0.0],
            axis1: [a1.x, a1.y, a1.z, 0.0],
            axis2: [a2.x, a2.y, a2.z, 0.0],
            half_extents: [h[0], h[1], h[2], 0.0],
        }
    }
}

/// One result as read back. `32`-byte `std430` stride matching `Result` in the
/// shader: the nearest world point (in a `vec4` lane), the Euclidean distance
/// and squared distance, the inside flag as `0`/`1` and one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Nearest world point in `xyz`; the `w` lane is unused padding.
    point: [f32; 4],
    /// Euclidean distance from the query to the nearest point.
    distance: f32,
    /// Squared Euclidean distance from the query to the nearest point.
    distance_squared: f32,
    /// Inside flag (`1` = inside or on the surface).
    inside: u32,
    /// Padding word.
    pad0: u32,
}

/// Maps one kernel `Result` lane back to the host [`ClosestPointObbResult`].
fn decode_result(raw: &GpuResult) -> ClosestPointObbResult {
    ClosestPointObbResult {
        point: Vec3::new(raw.point[0], raw.point[1], raw.point[2]),
        distance: raw.distance,
        distance_squared: raw.distance_squared,
        inside: raw.inside == CODE_INSIDE,
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

/// A compiled, reusable point-to-`OBB` nearest-point pipeline.
pub struct GpuClosestPointObb {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClosestPointObb {
    /// Compiles the point-to-`OBB` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClosestPointObb {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_closest_point_obb"),
            source: ShaderSource::Wgsl(CLOSEST_POINT_OBB_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_closest_point_obb_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_closest_point_obb_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_closest_point_obb_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClosestPointObb {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries`, returning one [`ClosestPointObbResult`]
    /// per query in input order.
    ///
    /// The returned result for query `q` mirrors
    /// [`Obb::closest_point`](prism_render_architecture::particle::closest_point_obb::Obb::closest_point)
    /// evaluated on `q.obb` and `q.point`. An empty `queries` slice yields an
    /// empty result — storage buffers cannot be zero-sized, so it is handled by
    /// an early return before any dispatch.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[ClosestPointObbQuery],
    ) -> Vec<ClosestPointObbResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_closest_point_obb_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_closest_point_obb_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_closest_point_obb_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_closest_point_obb_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_closest_point_obb_bind_group"),
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
            label: Some("prism_volumetric_closest_point_obb_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_closest_point_obb_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
    }
}
