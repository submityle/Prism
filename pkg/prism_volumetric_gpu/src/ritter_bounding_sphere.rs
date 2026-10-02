//! `wgpu` compute twin of the Ritter approximate smallest-enclosing-sphere
//! golden
//! ([`ritter_bounding_sphere`](prism_render_architecture::particle::ritter_bounding_sphere),
//! particle design §12, §13).
//!
//! The `CPU` golden
//! [`ritter_bounding_sphere`](prism_render_architecture::particle::ritter_bounding_sphere::ritter_bounding_sphere)
//! owns the broadphase/culling bounding-volume contract: given a cloud of
//! [`Vec3`](prism_render_architecture::particle::ritter_bounding_sphere::Vec3)
//! particle positions it returns a single
//! [`Sphere`](prism_render_architecture::particle::ritter_bounding_sphere::Sphere)
//! that provably contains every point, built by Jack Ritter's classic two-pass
//! construction. [`GpuRitterBoundingSphere`] is the on-device twin: one thread
//! solves one independent point-set, so a passing real-device parity test is
//! direct evidence the ported kernel seeds on the same axis-extremal pair and
//! grows in the same per-point order the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! Each thread reproduces the golden lane for lane. The **seed pass** tracks the
//! six axis-extremal points (`min`/`max` along `x`, `y`, `z`) with the identical
//! strict `<` / `>` updates, picks the extremal *pair* with the largest squared
//! separation under the identical `>=` tie-break (`x` then `y` then `z`), and
//! seeds the sphere on the midpoint of that segment with radius half the
//! separation. The **grow pass** then walks every point once more in input
//! order; whenever a point falls outside the current sphere (`dist > radius`) it
//! expands the sphere the minimal amount that swallows the stray point and keeps
//! the old sphere enclosed, exactly as the reference does. A final relative
//! cushion (`RADIUS_REL_EPS`) is applied so rounding cannot let a boundary point
//! escape. The empty-set [`Option`] is mirrored by a `valid` flag: a point-set
//! with zero points reports `valid = 0` and a zeroed sphere, matching the
//! reference [`None`].
//!
//! # Correctness model
//!
//! The `valid` flag is a discrete classification on the integer point count, so
//! `CPU` and `GPU` agree on it exactly and the parity test asserts an exact
//! `==`. The center and radius thread through adds, multiplies, one guarded
//! divide and `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits. Because the grow pass is order-dependent, both sides walk the
//! points in the identical index order so the branch decisions stay
//! deterministic; the parity test then asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the center and radius, tight
//! enough to catch a genuinely wrong port (a swapped axis pair, a dropped grow
//! step) yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! An empty point-set reports `valid = 0`, mirroring the reference [`None`]. A
//! single point (or many coincident points) seeds a zero-radius sphere centered
//! on the point and never enters the grow branch, so the result is the point
//! itself plus the relative cushion. The grow-pass divide is reached only inside
//! the `dist > radius` branch, where `dist > radius >= 0` guarantees
//! `dist > 0`, so it never divides by zero. An empty query *batch*
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `sqrt`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry and no optional device feature,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Each per-set loop is
//! bounded by the point count (itself capped at [`MAX_POINTS`]), so the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ritter_bounding_sphere`；
//! Jack Ritter's *Graphics Gems* bounding-sphere construction plus `wgpu`
//! compute dispatch; no third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::ritter_bounding_sphere::{
    ritter_bounding_sphere, Sphere, Vec3,
};
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

/// Fixed per-set capacity: the maximum number of points one query may carry.
/// Each uploaded [`GpuQuery`] reserves this many `vec4` lanes, and the host
/// rejects any [`RitterQuery`] with more points than this.
pub const MAX_POINTS: usize = 32;

/// The portable core-`WGSL` Ritter bounding-sphere kernel, embedded inline so
/// the twin ships as a single source file. Mirrors the `CPU` golden
/// [`ritter_bounding_sphere`](prism_render_architecture::particle::ritter_bounding_sphere::ritter_bounding_sphere)
/// seed-and-grow construction.
const RITTER_BOUNDING_SPHERE_WGSL: &str = r#"
// Ritter approximate smallest-enclosing-sphere twin: one thread per point-set
// seeds the sphere on the axis-extremal pair with the largest squared span,
// then grows it over every point in input order, writing the center, radius and
// a 0u/1u valid flag (0u for an empty set). It mirrors the CPU golden
// `particle::ritter_bounding_sphere` branch for branch, uses only the portable
// core-WGSL subset (min/max, sqrt, + - * / and unsigned index arithmetic), and
// takes no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: Jack Ritter's Graphics Gems bounding-sphere construction; no
// third-party engine source or derived code.

struct Params {
    // Number of valid point-sets in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One point-set. 528-byte std430 stride matching the host `GpuQuery`: the actual
// point count, three pad words, then a fixed array of up to MAX_POINTS points,
// each padded to a vec4 so the storage array needs no manual vec3 alignment
// arithmetic. Lanes at or past `point_count` are never read.
struct Query {
    point_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    points: array<vec4<f32>, 32>,
}

// One result. 32-byte std430 stride matching the host `GpuResult`: the sphere
// center, its radius, a 0u/1u valid flag and three pad words.
struct Result {
    center: vec3<f32>,
    radius: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Fixed per-set capacity, mirroring the host `MAX_POINTS`. The grow and seed
// loops clamp their bound to this so a malformed count can never index past the
// fixed `points` array.
const MAX_POINTS: u32 = 32u;

// Relative cushion added to the final radius so floating-point rounding in the
// grow pass can never leave a boundary point marginally outside the sphere,
// mirroring the reference `RADIUS_REL_EPS`. It is relative (a multiply, not an
// add) so the construction stays exactly linear under uniform scaling.
const RADIUS_REL_EPS: f32 = 1.0e-6;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    var res: Result;
    res.center = vec3<f32>(0.0, 0.0, 0.0);
    res.radius = 0.0;
    res.valid = 0u;
    res.pad0 = 0u;
    res.pad1 = 0u;
    res.pad2 = 0u;

    // An empty point-set mirrors the reference `None`: valid stays 0u.
    let n = min(queries[idx].point_count, MAX_POINTS);
    if (n == 0u) {
        results[idx] = res;
        return;
    }

    // --- Seed pass: track the six axis-extremal points, exactly as the golden
    // does, with strict `<` / `>` updates seeded from the first point. ---
    let first = queries[idx].points[0].xyz;
    var min_x = first;
    var max_x = first;
    var min_y = first;
    var max_y = first;
    var min_z = first;
    var max_z = first;
    for (var i = 0u; i < n; i = i + 1u) {
        let p = queries[idx].points[i].xyz;
        if (p.x < min_x.x) {
            min_x = p;
        }
        if (p.x > max_x.x) {
            max_x = p;
        }
        if (p.y < min_y.y) {
            min_y = p;
        }
        if (p.y > max_y.y) {
            max_y = p;
        }
        if (p.z < min_z.z) {
            min_z = p;
        }
        if (p.z > max_z.z) {
            max_z = p;
        }
    }

    // Pick the axis-extremal pair with the largest squared separation, using the
    // identical `>=` tie-break (x then y then z) so the seed matches the golden.
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let dz = max_z - min_z;
    let span_x = dx.x * dx.x + dx.y * dx.y + dx.z * dx.z;
    let span_y = dy.x * dy.x + dy.y * dy.y + dy.z * dy.z;
    let span_z = dz.x * dz.x + dz.y * dz.y + dz.z * dz.z;

    var lo = min_z;
    var hi = max_z;
    if (span_x >= span_y && span_x >= span_z) {
        lo = min_x;
        hi = max_x;
    } else if (span_y >= span_z) {
        lo = min_y;
        hi = max_y;
    }

    // Seed the sphere on the chosen segment: center at the midpoint, radius at
    // half the separation.
    var center = (lo + hi) * 0.5;
    let seed = hi - lo;
    var radius = sqrt(seed.x * seed.x + seed.y * seed.y + seed.z * seed.z) * 0.5;

    // --- Grow pass: expand to swallow every stray point, in input order. ---
    for (var i = 0u; i < n; i = i + 1u) {
        let p = queries[idx].points[i].xyz;
        let offset = p - center;
        let dist = sqrt(offset.x * offset.x + offset.y * offset.y + offset.z * offset.z);
        if (dist > radius) {
            // The new sphere is the smallest one tangent to both the old sphere
            // and the stray point: its diameter spans from the far side of the
            // old sphere to the stray point.
            let new_radius = (radius + dist) * 0.5;
            // dist > radius >= 0 implies dist > 0, so this divide is guarded by
            // the branch and never divides by zero.
            let t = (new_radius - radius) / dist;
            center = center + offset * t;
            radius = new_radius;
        }
    }

    // Relative cushion: keeps scaling exactly linear and translation invariant.
    radius = radius + radius * RADIUS_REL_EPS;

    res.center = center;
    res.radius = radius;
    res.valid = 1u;
    results[idx] = res;
}
"#;

/// One Ritter query: an independent point-set whose approximate
/// smallest-enclosing sphere is solved by a single thread.
///
/// Mirrors a single reference
/// [`ritter_bounding_sphere`](prism_render_architecture::particle::ritter_bounding_sphere::ritter_bounding_sphere)
/// call on `points`. The slice may hold up to [`MAX_POINTS`] points; an empty
/// slice yields an invalid sphere, mirroring the reference [`None`]. Holds
/// `f32` geometry, so it is not hashable.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RitterQuery {
    /// The point-set to enclose; at most [`MAX_POINTS`] points.
    pub points: Vec<Vec3>,
}

/// The resolved bounding sphere for one query, read back from the kernel.
///
/// `valid` is the host mirror of the reference [`Option`]: `1` for a non-empty
/// point-set (a sphere was produced) and `0` for an empty one (the reference
/// [`None`]), in which case `center` and `radius` are both zero. `center` is the
/// sphere center in `xyz` and `radius` its non-negative radius. Holds `f32`
/// geometry, so it derives only [`PartialEq`] (no `Eq`/`Hash`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuRitterSphere {
    /// The sphere center in `xyz`; meaningful only when `valid == 1`.
    pub center: [f32; 3],
    /// The sphere radius; meaningful only when `valid == 1`.
    pub radius: f32,
    /// Validity flag: `1` when a sphere was produced, `0` for an empty set.
    pub valid: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`RITTER_BOUNDING_SPHERE_WGSL`]: the query count and three pad
/// words — `16` bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid point-sets.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One point-set as uploaded. `528`-byte `std430` stride matching `Query` in the
/// shader: the actual point count, three pad words and a fixed array of up to
/// [`MAX_POINTS`] points, each padded to a `vec4` lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Number of valid points in `points`.
    point_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// The point-set, `xyz` per lane; lanes at or past `point_count` are unused.
    points: [[f32; 4]; MAX_POINTS],
}

impl GpuQuery {
    /// Packs a [`RitterQuery`] into the `std430` upload layout.
    ///
    /// # Panics
    ///
    /// Panics when `query.points` holds more than [`MAX_POINTS`] points, since
    /// the fixed per-set upload lane budget cannot carry them.
    fn from_query(query: &RitterQuery) -> GpuQuery {
        assert!(
            query.points.len() <= MAX_POINTS,
            "RitterQuery holds {} points, exceeding MAX_POINTS ({MAX_POINTS})",
            query.points.len()
        );
        let mut points = [[0.0_f32; 4]; MAX_POINTS];
        for (lane, &p) in query.points.iter().enumerate() {
            points[lane] = [p.x, p.y, p.z, 0.0];
        }
        GpuQuery {
            point_count: query.points.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            points,
        }
    }
}

/// The `std430` readback lane. Identical byte layout to the public
/// [`GpuRitterSphere`], read back from the kernel before being handed to the
/// caller.
type GpuResult = GpuRitterSphere;

/// Maps one kernel `Result` lane back to the host [`GpuRitterSphere`]. The byte
/// layout already matches, so this only copies the raw readback lane.
fn decode_result(raw: &GpuResult) -> GpuRitterSphere {
    *raw
}

/// A compiled, reusable Ritter bounding-sphere pipeline.
pub struct GpuRitterBoundingSphere {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRitterBoundingSphere {
    /// Compiles the Ritter bounding-sphere kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRitterBoundingSphere {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ritter_bounding_sphere"),
            source: ShaderSource::Wgsl(RITTER_BOUNDING_SPHERE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ritter_bounding_sphere_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ritter_bounding_sphere_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ritter_bounding_sphere_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRitterBoundingSphere {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every point-set in `queries`, returning one [`GpuRitterSphere`]
    /// per query in input order.
    ///
    /// The returned sphere for query `q` mirrors
    /// [`ritter_bounding_sphere`](prism_render_architecture::particle::ritter_bounding_sphere::ritter_bounding_sphere)
    /// evaluated on `q.points`: `valid == 1` with the center and radius when the
    /// set is non-empty, and `valid == 0` with a zeroed sphere when it is empty
    /// (the reference [`None`]). An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// # Panics
    ///
    /// Panics when any query holds more than [`MAX_POINTS`] points.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[RitterQuery]) -> Vec<GpuRitterSphere> {
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
            label: Some("prism_volumetric_ritter_bounding_sphere_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ritter_bounding_sphere_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ritter_bounding_sphere_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ritter_bounding_sphere_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ritter_bounding_sphere_bind_group"),
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
            label: Some("prism_volumetric_ritter_bounding_sphere_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ritter_bounding_sphere_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per point-set, flattened to a 1-D dispatch.
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

/// The `CPU` golden sphere for one query, dispatching to the reference entry
/// point so callers (and the parity test) can pin the twin lane for lane.
///
/// Returns [`None`] for an empty point-set, mirroring the reference.
#[must_use]
pub fn cpu_reference(query: &RitterQuery) -> Option<Sphere> {
    ritter_bounding_sphere(&query.points)
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
