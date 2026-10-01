//! `wgpu` compute twin of the sphere-versus-axis-aligned-bounding-box (`AABB`)
//! proximity golden
//! ([`sphere_aabb`](prism_render_architecture::particle::sphere_aabb),
//! design §10, §14).
//!
//! The particle subsystem answers one narrow, purely geometric question per
//! sphere/box pair: what is the closest point on the box to the sphere centre,
//! do the two overlap, how deep is the penetration, and along which separation
//! normal should the sphere be pushed out? The canonical real-time-rendering
//! test clamps the centre into `[min, max]` for the closest point and compares
//! the squared distance against `r²` to decide intersection.
//!
//! The `CPU` golden
//! [`query`](prism_render_architecture::particle::sphere_aabb::query) owns that
//! math; [`GpuSphereAabb`] is the on-device twin that runs one thread per
//! sphere/box pair and returns the same
//! [`Proximity`](prism_render_architecture::particle::sphere_aabb::Proximity)
//! per query, in input order. Because the result carries both the
//! `intersecting` flag and the `center_inside_box` flag, this single kernel is
//! also the twin of the convenience boolean
//! [`intersects`](prism_render_architecture::particle::sphere_aabb::intersects)
//! and of
//! [`Aabb::contains_point`](prism_render_architecture::particle::sphere_aabb::Aabb::contains_point)
//! evaluated at the sphere centre. A passing real-device parity test is
//! therefore direct evidence the ported kernel clamps the same closest point,
//! folds the same squared-distance decision, picks the same nearest face and
//! writes the same penetration/separation the reference does, not merely that
//! its shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces the reference branch for branch:
//!
//! * **Centre strictly outside the box** (`outside_distance > CMP_EPS`) — the
//!   closest point is the clamped centre, the separation normal is the unit
//!   vector from the closest point toward the centre (`delta * (1 /
//!   outside_distance)`, the same reciprocal-scale order the reference uses),
//!   and the penetration depth is `radius - outside_distance` clamped to zero
//!   when the pair does not overlap.
//! * **Centre on the surface or inside the box** — the closest point coincides
//!   with the centre, so the outward normal is taken from the nearest box face
//!   (ties resolved to the first candidate in axis order `(-x, +x, -y, +y, -z,
//!   +z)` by a strict `<`, exactly as the reference) and the penetration depth
//!   is `face_distance + radius`.
//!
//! A negative radius is treated as a point (`max(radius, 0)`) and the
//! `center_inside_box` flag mirrors the inclusive
//! [`Aabb::contains_point`](prism_render_architecture::particle::sphere_aabb::Aabb::contains_point)
//! on both paths, matching the reference exactly.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `sqrt`, `+ - * /` and unsigned integer index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `smoothstep` or optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The only
//! floating-point primitive beyond ordinary arithmetic is `sqrt` (for the
//! closest-point distance), matching the reference, which also uses only
//! `f32::sqrt` plus `abs`/`min`/`max`/`clamp`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of clamps, subtractions, one
//! `sqrt`, a squared-distance compare and a branch, so `CPU` and `GPU` evaluate
//! the same closed form in the same order. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance on the continuous fields (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) and an exact match on the discrete booleans, and keeps
//! every fixture well clear of the two decision boundaries (the `CMP_EPS`
//! inside/outside branch split and the `dist_sq == radius_sq` tangent) so a
//! legal fused multiply-add cannot flip a branch or a flag.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sphere_aabb`;
//! canonical closest-point sphere-`AABB` test plus `wgpu` compute dispatch; no
//! third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::sphere_aabb::{Aabb, Proximity, Sphere, Vec3};
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
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` sphere-`AABB` proximity kernel, embedded inline so
/// the twin ships as a single source file. Mirrors the `CPU` golden
/// [`query`](prism_render_architecture::particle::sphere_aabb::query) branch for
/// branch; see the module documentation for the algorithm.
const SPHERE_AABB_WGSL: &str = r#"
// Sphere-AABB proximity twin: one thread per sphere/box pair clamps the centre
// into [min, max] for the closest point, compares squared distance to r^2 for
// intersection, and writes the full Proximity (closest point, outside distance,
// separation, intersecting flag, centre-inside flag, penetration depth and
// separation normal). It mirrors the CPU golden
// `particle::sphere_aabb::query` branch for branch, uses only the portable
// core-WGSL subset (min/max/clamp/abs/sqrt and + - * / plus unsigned compares,
// no transcendental and no smoothstep), and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::sphere_aabb;
// canonical closest-point sphere-AABB test; no third-party engine source or
// derived code.

struct Params {
    // Number of valid sphere/box queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 48-byte std430 stride matching the host `GpuQuery`: the sphere
// centre with its radius in `w`, the box min corner, and the box max corner,
// each triple padded to a 16-byte lane.
struct Query {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    radius: f32,
    min_x: f32,
    min_y: f32,
    min_z: f32,
    pad0: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
    pad1: f32,
}

// One result. 48-byte std430 stride matching the host `GpuProximity`: the
// closest point plus the outside distance, then separation / penetration /
// two discrete flags, then the separation normal plus one pad word.
struct Proximity {
    closest_x: f32,
    closest_y: f32,
    closest_z: f32,
    outside_distance: f32,
    separation: f32,
    penetration_depth: f32,
    intersecting: u32,
    center_inside_box: u32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Proximity>;

// Absolute tolerance guarding the inside/outside branch split, matching the
// reference `CMP_EPS`. A direct f32 `==`/`!=` is forbidden, so the branch
// compares the magnitude against this band instead of exact zero.
const CMP_EPS: f32 = 1.0e-6;

// Inclusive box containment at the sphere centre, mirroring the reference
// `Aabb::contains_point`. Uses only ordering compares, never an f32 equality.
fn contains_point(center: vec3<f32>, mn: vec3<f32>, mx: vec3<f32>) -> bool {
    return center.x >= mn.x && center.x <= mx.x
        && center.y >= mn.y && center.y <= mx.y
        && center.z >= mn.z && center.z <= mx.z;
}

@compute @workgroup_size(64)
fn proximity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    let center = vec3<f32>(q.center_x, q.center_y, q.center_z);
    let mn = vec3<f32>(q.min_x, q.min_y, q.min_z);
    let mx = vec3<f32>(q.max_x, q.max_y, q.max_z);
    // A negative radius is meaningless; treat it as a point, exactly as the
    // reference `sphere.radius.max(0.0)`.
    let radius = max(q.radius, 0.0);

    let closest = clamp(center, mn, mx);
    let delta = center - closest;
    let dist_sq = dot(delta, delta);
    let radius_sq = radius * radius;

    let outside_distance = sqrt(dist_sq);
    // Ordering compare (not equality): a tangent `dist_sq == radius_sq` counts
    // as intersecting, matching the reference `dist_sq <= radius_sq`.
    let intersecting = dist_sq <= radius_sq;
    let separation = outside_distance - radius;
    let hit = select(0u, 1u, intersecting);

    var out: Proximity;
    out.closest_x = closest.x;
    out.closest_y = closest.y;
    out.closest_z = closest.z;
    out.intersecting = hit;
    out.pad0 = 0.0;

    if (outside_distance > CMP_EPS) {
        // Centre strictly outside the box: the separation direction is the unit
        // vector from the closest surface point toward the centre. The
        // reciprocal-scale order matches the reference `delta.scale(1 / d)`.
        let inv = 1.0 / outside_distance;
        let normal = delta * inv;
        var pen = 0.0;
        if (intersecting) {
            pen = max(radius - outside_distance, 0.0);
        }
        out.outside_distance = outside_distance;
        out.separation = separation;
        out.penetration_depth = pen;
        out.center_inside_box = 0u;
        out.normal_x = normal.x;
        out.normal_y = normal.y;
        out.normal_z = normal.z;
    } else {
        // Centre on the surface or inside the box: pick the nearest face for
        // the outward separation normal. Candidates are folded in axis order
        // (-x, +x, -y, +y, -z, +z) with a strict `<`, so ties resolve to the
        // first candidate exactly as the reference `nearest_face`.
        var best_d = center.x - mn.x;
        var best_n = vec3<f32>(-1.0, 0.0, 0.0);
        let d_px = mx.x - center.x;
        if (d_px < best_d) {
            best_d = d_px;
            best_n = vec3<f32>(1.0, 0.0, 0.0);
        }
        let d_ny = center.y - mn.y;
        if (d_ny < best_d) {
            best_d = d_ny;
            best_n = vec3<f32>(0.0, -1.0, 0.0);
        }
        let d_py = mx.y - center.y;
        if (d_py < best_d) {
            best_d = d_py;
            best_n = vec3<f32>(0.0, 1.0, 0.0);
        }
        let d_nz = center.z - mn.z;
        if (d_nz < best_d) {
            best_d = d_nz;
            best_n = vec3<f32>(0.0, 0.0, -1.0);
        }
        let d_pz = mx.z - center.z;
        if (d_pz < best_d) {
            best_d = d_pz;
            best_n = vec3<f32>(0.0, 0.0, 1.0);
        }
        var pen = 0.0;
        if (intersecting) {
            pen = best_d + radius;
        }
        let inside = select(0u, 1u, contains_point(center, mn, mx));
        out.outside_distance = 0.0;
        out.separation = -radius;
        out.penetration_depth = pen;
        out.center_inside_box = inside;
        out.normal_x = best_n.x;
        out.normal_y = best_n.y;
        out.normal_z = best_n.z;
    }

    results[idx] = out;
}
"#;

/// One sphere-`AABB` proximity query: a sphere tested against an axis-aligned
/// box.
///
/// Mirrors a single reference
/// [`query`](prism_render_architecture::particle::sphere_aabb::query)`(sphere,
/// aabb)` call. Carrying both operands per query lets one dispatch mix
/// unrelated sphere/box pairs. Derives only [`PartialEq`] (no `Eq`/`Hash`)
/// because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereAabbQuery {
    /// The sphere (a particle modelled at its collision radius).
    pub sphere: Sphere,
    /// The axis-aligned box tested against `sphere`.
    pub aabb: Aabb,
}

/// Uniform parameters for one proximity dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`SPHERE_AABB_WGSL`]: the query count and three pad
/// words — `16` bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One query as uploaded. `48`-byte `std430` stride matching `Query` in the
/// shader: the sphere centre with its radius, the box min corner plus a pad
/// word, and the box max corner plus a pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    radius: f32,
    min_x: f32,
    min_y: f32,
    min_z: f32,
    pad0: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
    pad1: f32,
}

impl GpuQuery {
    /// Packs a [`SphereAabbQuery`] into the `std430` upload layout.
    fn from_query(query: &SphereAabbQuery) -> GpuQuery {
        let c = query.sphere.center;
        let mn = query.aabb.min;
        let mx = query.aabb.max;
        GpuQuery {
            center_x: c.x,
            center_y: c.y,
            center_z: c.z,
            radius: query.sphere.radius,
            min_x: mn.x,
            min_y: mn.y,
            min_z: mn.z,
            pad0: 0.0,
            max_x: mx.x,
            max_y: mx.y,
            max_z: mx.z,
            pad1: 0.0,
        }
    }
}

/// One query result as read back. `48`-byte `std430` stride matching
/// `Proximity` in the shader: the closest point plus the outside distance, then
/// separation / penetration / the two discrete flags, then the separation
/// normal plus one pad word. The two flags are carried as `u32` (`0` or `1`) so
/// the readback decodes them with an exact integer compare, never an `f32`
/// equality.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuProximity {
    closest_x: f32,
    closest_y: f32,
    closest_z: f32,
    outside_distance: f32,
    separation: f32,
    penetration_depth: f32,
    intersecting: u32,
    center_inside_box: u32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    pad0: f32,
}

impl GpuProximity {
    /// Decodes the raw device record into the reference
    /// [`Proximity`](prism_render_architecture::particle::sphere_aabb::Proximity).
    /// The `u32` flags are compared with an exact integer `== 1`, never an
    /// `f32` equality.
    fn into_proximity(self) -> Proximity {
        Proximity {
            closest_point: Vec3::new(self.closest_x, self.closest_y, self.closest_z),
            outside_distance: self.outside_distance,
            separation: self.separation,
            intersecting: self.intersecting == 1,
            center_inside_box: self.center_inside_box == 1,
            penetration_depth: self.penetration_depth,
            normal: Vec3::new(self.normal_x, self.normal_y, self.normal_z),
        }
    }
}

/// A compiled, reusable sphere-`AABB` proximity pipeline.
pub struct GpuSphereAabb {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSphereAabb {
    /// Compiles the sphere-`AABB` proximity kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSphereAabb {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sphere_aabb"),
            source: ShaderSource::Wgsl(SPHERE_AABB_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sphere_aabb_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sphere_aabb_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sphere_aabb_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("proximity"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSphereAabb {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the proximity query for every pair in `queries`, returning one
    /// [`Proximity`](prism_render_architecture::particle::sphere_aabb::Proximity)
    /// per query in input order.
    ///
    /// The returned proximity for query `q` equals
    /// [`query`](prism_render_architecture::particle::sphere_aabb::query)`(q.sphere,
    /// q.aabb)`; its `intersecting` flag equals
    /// [`intersects`](prism_render_architecture::particle::sphere_aabb::intersects)`(q.sphere,
    /// q.aabb)` and its `center_inside_box` flag equals
    /// [`Aabb::contains_point`](prism_render_architecture::particle::sphere_aabb::Aabb::contains_point)`(q.sphere.center)`.
    /// An empty `queries` slice yields an empty result — storage buffers cannot
    /// be zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[SphereAabbQuery]) -> Vec<Proximity> {
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

        let out_bytes = (queries.len() as u64) * (size_of::<GpuProximity>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sphere_aabb_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sphere_aabb_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sphere_aabb_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sphere_aabb_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sphere_aabb_bind_group"),
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
            label: Some("prism_volumetric_sphere_aabb_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sphere_aabb_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch of `64`-wide
            // workgroups.
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuProximity>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
            .into_iter()
            .map(GpuProximity::into_proximity)
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
