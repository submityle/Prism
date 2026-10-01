//! `wgpu` compute twin of the six-plane view-frustum culling golden
//! ([`frustum_aabb_cull`](prism_render_architecture::particle::frustum_aabb_cull),
//! design §12, §13).
//!
//! The particle culling stage holds the six oriented planes bounding the
//! visible volume and must answer, per emitter bounds primitive, the three-way
//! question: is this `AABB` or bounding sphere wholly [`Visibility::Inside`] the
//! frustum, wholly [`Visibility::Outside`] it, or [`Visibility::Intersecting`]
//! its boundary? That verdict gates whether a whole emitter's particles are
//! simulated, sorted and rasterized this frame, so the test must be cheap,
//! branch-light and give the *same discrete verdict* on the reference `CPU`
//! path and this `GPU` kernel.
//!
//! The `CPU` golden
//! [`cull_aabb`](prism_render_architecture::particle::frustum_aabb_cull::cull_aabb)
//! and
//! [`cull_sphere`](prism_render_architecture::particle::frustum_aabb_cull::cull_sphere)
//! own that classification; [`GpuFrustumCull`] is the on-device twin that runs
//! one thread per bounds primitive and returns the same [`Visibility`] verdict
//! per query, in input order. A passing real-device parity test is therefore
//! direct evidence the ported kernel folds the same per-plane decision the
//! reference does, not merely that its shader compiles.
//!
//! # Symmetric projected-radius test
//!
//! The kernel mirrors the reference p-vertex / n-vertex shortcut exactly. For a
//! plane with unit inward normal `n` and offset `d` (inside half-space
//! `n·p + d >= 0`):
//!
//! * the primitive center's signed distance is `s = n·center + d`, matching
//!   [`Plane::signed_distance`](prism_render_architecture::particle::frustum_aabb_cull::Plane::signed_distance);
//! * the extent projected onto `n` is the radius `r`: for an `AABB`,
//!   `r = |n_x|·h_x + |n_y|·h_y + |n_z|·h_z` (the reference
//!   [`projected_radius`](prism_render_architecture::particle::frustum_aabb_cull::projected_radius));
//!   for a sphere, `r` is simply the scalar radius, so no square root is ever
//!   formed.
//!
//! The p-vertex sits at `s + r` and the n-vertex at `s - r`. For each plane the
//! kernel folds the same decision the reference loop does: `s + r < -eps`
//! rejects the primitive (it short-circuits to [`Visibility::Outside`]);
//! `s - r < -eps` records an intersection vote; otherwise the primitive is
//! fully inside that plane. A primitive is [`Visibility::Outside`] as soon as
//! *any* plane rejects it, [`Visibility::Intersecting`] when every plane admits
//! the p-vertex but some plane cuts it, and [`Visibility::Inside`] when every
//! plane fully contains it.
//!
//! # Discrete-verdict parity
//!
//! The verdict is a discrete enum, not a continuous value, so the twin targets
//! an *exact* match rather than a tolerance band. The arithmetic is byte-for-
//! byte the same associativity as the reference (`v_dot` then `+ d`; the three
//! absolute-value terms summed left to right), and every sign decision goes
//! through the same tolerant [`CULL_EPS`](prism_render_architecture::particle::frustum_aabb_cull::CULL_EPS)
//! band (`-1e-6`) the reference uses, so a primitive that merely touches a
//! plane is admitted on both paths rather than flickering between verdicts. A
//! legal fused multiply-add on a `GPU` perturbs `s` or `r` by at most a few
//! units in the last place, far inside that `1e-6` band for any primitive not
//! deliberately placed within one `ULP` of a plane, so the folded discrete
//! verdict is reproduced exactly for every realistic bounds primitive.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`/`max`
//! via comparison, `+ - * /` and unsigned integer compares — with no `sin`,
//! `cos`, `exp`, `log`, `pow` or optional device feature. It forms no square
//! root at all (the sphere radius is scalar and the box radius is a sum of
//! absolute terms), matching the reference, which also uses only `f32`
//! add/sub/mul and [`f32::abs`]. The kernel therefore runs unmodified on Metal,
//! Vulkan and DX12 and can never produce a `NaN` from its own math.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: classic p-vertex / n-vertex symmetric projected-radius frustum
//! test plus `wgpu` compute dispatch; this module contains no third-party
//! engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::frustum_aabb_cull::{Aabb, Plane, Sphere, Visibility};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Discrete visibility code the kernel writes for a fully-contained primitive.
/// Matches the host [`Visibility::Inside`] mapping in [`decode_visibility`].
const CODE_INSIDE: u32 = 0;
/// Discrete visibility code for a primitive rejected by at least one plane.
/// Matches the host [`Visibility::Outside`] mapping in [`decode_visibility`].
const CODE_OUTSIDE: u32 = 1;
/// Discrete visibility code for a primitive cut by at least one plane. Matches
/// the host [`Visibility::Intersecting`] mapping in [`decode_visibility`].
const CODE_INTERSECTING: u32 = 2;

/// The portable core-`WGSL` frustum-cull kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` golden
/// [`cull_aabb`](prism_render_architecture::particle::frustum_aabb_cull::cull_aabb)
/// and
/// [`cull_sphere`](prism_render_architecture::particle::frustum_aabb_cull::cull_sphere)
/// plane-for-plane; see the module documentation for the algorithm.
const FRUSTUM_CULL_WGSL: &str = r#"
// Frustum-cull twin: one thread per bounds primitive folds the six-plane
// p-vertex / n-vertex test and writes a discrete visibility code
// (0 = Inside, 1 = Outside, 2 = Intersecting). It mirrors the CPU golden
// `particle::frustum_aabb_cull::{cull_aabb, cull_sphere}` plane-for-plane,
// uses only the portable core-WGSL subset (abs and + - * / plus unsigned
// compares, no sqrt or transcendental), and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: classic p-vertex / n-vertex symmetric projected-radius frustum
// test; no third-party engine source or derived code.

struct Params {
    // Number of valid bounds primitives in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One bounds query. 128-byte std430 stride matching the host `GpuQuery`: six
// inward frustum planes packed as vec4(nx, ny, nz, d), the primitive center
// with the primitive kind in `w` (0.0 = AABB, 1.0 = sphere), and the AABB
// half-extents with the sphere radius in `w`.
struct Query {
    planes: array<vec4<f32>, 6>,
    center_kind: vec4<f32>,
    half_radius: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

// Discrete visibility codes, matching the host `Visibility` mapping.
const INSIDE: u32 = 0u;
const OUTSIDE: u32 = 1u;
const INTERSECTING: u32 = 2u;

// Tolerant epsilon for sign decisions, matching the reference `CULL_EPS`. A
// direct f32 `==`/`!=` is forbidden, so every inside/outside test compares
// against this band instead of exact zero.
const CULL_EPS: f32 = 1.0e-6;

// Signed distance of `center` to the inward plane (n = p.xyz, d = p.w),
// matching the reference `Plane::signed_distance`: `v_dot(n, center) + d`,
// evaluated in the same left-to-right order (nx*cx + ny*cy + nz*cz) + d.
fn signed_distance(p: vec4<f32>, center: vec3<f32>) -> f32 {
    return p.x * center.x + p.y * center.y + p.z * center.z + p.w;
}

// Half-width of the box's projection onto the axis `n`, matching the reference
// `projected_radius`: |n_x|*h_x + |n_y|*h_y + |n_z|*h_z, summed in the same
// left-to-right order.
fn projected_radius(n: vec3<f32>, half: vec3<f32>) -> f32 {
    return abs(n.x) * half.x + abs(n.y) * half.y + abs(n.z) * half.z;
}

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let center = queries[idx].center_kind.xyz;
    // The kind flag is a discrete 0.0/1.0 marker; the 0.5 split avoids an f32
    // equality compare while classifying every well-formed query exactly.
    let is_sphere = queries[idx].center_kind.w > 0.5;
    let half = queries[idx].half_radius.xyz;
    let radius = queries[idx].half_radius.w;

    // Fold the per-plane decision across all six planes. A rejecting plane
    // (p-vertex outside) forces Outside regardless of later votes, exactly as
    // the reference early-returns Outside on the first rejecting plane; a
    // cutting plane (n-vertex outside) records an intersection vote.
    var outside = false;
    var intersecting = false;
    for (var i = 0u; i < 6u; i = i + 1u) {
        let p = queries[idx].planes[i];
        let s = signed_distance(p, center);
        var r: f32;
        if (is_sphere) {
            r = radius;
        } else {
            r = projected_radius(p.xyz, half);
        }
        if (s + r < -CULL_EPS) {
            outside = true;
        }
        if (s - r < -CULL_EPS) {
            intersecting = true;
        }
    }

    var verdict = INSIDE;
    if (outside) {
        verdict = OUTSIDE;
    } else if (intersecting) {
        verdict = INTERSECTING;
    }
    results[idx] = verdict;
}
"#;

/// A bounds primitive to classify against a frustum: either an axis-aligned box
/// or a bounding sphere.
///
/// Mirrors the two reference entry points
/// [`cull_aabb`](prism_render_architecture::particle::frustum_aabb_cull::cull_aabb)
/// and
/// [`cull_sphere`](prism_render_architecture::particle::frustum_aabb_cull::cull_sphere):
/// an [`FrustumCullPrimitive::Aabb`] is classified by the per-plane projected
/// radius, a [`FrustumCullPrimitive::Sphere`] by its scalar radius. Derives only
/// [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrustumCullPrimitive {
    /// An axis-aligned box, classified by its per-plane projected radius.
    Aabb(Aabb),
    /// A bounding sphere, classified by its scalar radius.
    Sphere(Sphere),
}

/// One frustum-cull query: the six inward bounding planes plus the bounds
/// primitive to classify against them.
///
/// Mirrors a single reference
/// [`cull_aabb`](prism_render_architecture::particle::frustum_aabb_cull::cull_aabb)`(&planes, &aabb)`
/// or
/// [`cull_sphere`](prism_render_architecture::particle::frustum_aabb_cull::cull_sphere)`(&planes, &sphere)`
/// call. Carrying the planes per query lets one dispatch mix primitives drawn
/// from several frustums. The plane normals must point **into** the frustum and
/// are assumed unit length, exactly as the reference requires. Derives only
/// [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrustumCullQuery {
    /// The six inward-facing frustum planes (left, right, bottom, top, near,
    /// far). Normals point into the frustum and are assumed unit length.
    pub planes: [Plane; 6],
    /// The bounds primitive classified against `planes`.
    pub primitive: FrustumCullPrimitive,
}

/// Uniform parameters for one frustum-cull dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`FRUSTUM_CULL_WGSL`]: the primitive count and three pad
/// words — `16` bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One bounds query as uploaded. `128`-byte `std430` stride matching `Query` in
/// the shader: six planes packed as `vec4(nx, ny, nz, d)`, the primitive center
/// with the kind flag in `w` (`0.0` = `AABB`, `1.0` = sphere), and the `AABB`
/// half-extents with the sphere radius in `w`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    planes: [[f32; 4]; 6],
    center_kind: [f32; 4],
    half_radius: [f32; 4],
}

impl GpuQuery {
    /// Packs a [`FrustumCullQuery`] into the `std430` upload layout. For an
    /// `AABB` the half-extents are carried and the radius lane is zero; for a
    /// sphere the radius is carried, the half-extent lanes are zero and the
    /// kind flag is `1.0`.
    fn from_query(query: &FrustumCullQuery) -> GpuQuery {
        let mut planes = [[0.0_f32; 4]; 6];
        for (slot, plane) in planes.iter_mut().zip(query.planes.iter()) {
            *slot = [plane.normal[0], plane.normal[1], plane.normal[2], plane.d];
        }
        let (center, kind, half, radius) = match query.primitive {
            FrustumCullPrimitive::Aabb(aabb) => (aabb.center, 0.0_f32, aabb.half, 0.0_f32),
            FrustumCullPrimitive::Sphere(sphere) => {
                (sphere.center, 1.0_f32, [0.0_f32; 3], sphere.radius)
            }
        };
        GpuQuery {
            planes,
            center_kind: [center[0], center[1], center[2], kind],
            half_radius: [half[0], half[1], half[2], radius],
        }
    }
}

/// Maps a discrete kernel visibility code back to the reference [`Visibility`].
///
/// # Panics
///
/// Panics if the code is outside `0..=2`, which cannot happen for a correct
/// kernel: the shader only ever writes [`CODE_INSIDE`], [`CODE_OUTSIDE`] or
/// [`CODE_INTERSECTING`].
fn decode_visibility(code: u32) -> Visibility {
    match code {
        CODE_INSIDE => Visibility::Inside,
        CODE_OUTSIDE => Visibility::Outside,
        CODE_INTERSECTING => Visibility::Intersecting,
        other => unreachable!("kernel only emits visibility codes 0..=2, got {other}"),
    }
}

/// A compiled, reusable six-plane frustum-cull pipeline.
pub struct GpuFrustumCull {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFrustumCull {
    /// Compiles the frustum-cull kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFrustumCull {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_frustum_cull"),
            source: ShaderSource::Wgsl(FRUSTUM_CULL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_frustum_cull_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_frustum_cull_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_frustum_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cull"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFrustumCull {
            module,
            layout,
            pipeline,
        }
    }

    /// Classifies every primitive in `queries` against its own frustum,
    /// returning one [`Visibility`] verdict per query in input order.
    ///
    /// The returned verdict for query `q` equals
    /// [`cull_aabb`](prism_render_architecture::particle::frustum_aabb_cull::cull_aabb)`(&q.planes, &aabb)`
    /// when `q.primitive` is an [`FrustumCullPrimitive::Aabb`], or
    /// [`cull_sphere`](prism_render_architecture::particle::frustum_aabb_cull::cull_sphere)`(&q.planes, &sphere)`
    /// when it is a [`FrustumCullPrimitive::Sphere`]. An empty `queries` slice
    /// yields an empty result — storage buffers cannot be zero-sized, so it is
    /// handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[FrustumCullQuery]) -> Vec<Visibility> {
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

        // One `u32` visibility code per primitive.
        let out_bytes = (queries.len() as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_frustum_cull_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_frustum_cull_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_frustum_cull_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_frustum_cull_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_frustum_cull_bind_group"),
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
            label: Some("prism_volumetric_frustum_cull_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_frustum_cull_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per primitive, in workgroups of 64 (the kernel's size).
            let groups = (queries.len() as u32).div_ceil(64);
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
        let codes = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(codes.len(), queries.len());

        codes.iter().map(|&code| decode_visibility(code)).collect()
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
