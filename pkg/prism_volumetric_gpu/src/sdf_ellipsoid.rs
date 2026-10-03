//! `wgpu` compute twin of three analytic ellipsoid/carved signed-distance
//! primitives of the `CPU` golden path
//! ([`ellipsoid_sdf`](prism_render_architecture::ray_scene::sdf_primitives::ellipsoid_sdf),
//! [`death_star`](prism_render_architecture::ray_scene::sdf_primitives::death_star)
//! and
//! [`quad_sdf`](prism_render_architecture::ray_scene::sdf_primitives::quad_sdf)).
//!
//! Implicit modelling needs *analytic* primitives whose signed distance is
//! known in closed form rather than sampled on a grid. The reference derives
//! three shapes: the gradient-corrected [`ellipsoid_sdf`] (an axis-aligned
//! ellipsoid centred at the origin), the carved [`death_star`] (a large sphere
//! minus a smaller biting sphere, leaving a crescent crater), and the planar
//! convex [`quad_sdf`] (the unsigned distance to a four-sided patch). Each
//! stays transcendental-free, using only `sqrt`, `abs`, `min`, `max`, `clamp`,
//! products and quotients. [`GpuSdfEllipsoid`] is the on-device twin: each
//! thread reads one point plus all three shapes' parameters and writes all
//! three distances, reproducing the reference closed forms operation for
//! operation.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfEllipsoidQuery`] — a query `point` plus the
//! ellipsoid `radii`, the death-star (`large_radius`, `small_radius`,
//! `bite_distance`) and the quad vertices (`quad_a`, `quad_b`, `quad_c`,
//! `quad_d`) — and writes one [`SdfEllipsoidResult`] holding the three
//! distances. The ellipsoid uses Inigo Quilez's gradient bound
//! `k0 * (k0 - 1) / k1`; the death star reduces to the meridian half-plane and
//! selects the crater-rim circle or the sphere-intersection body by a single
//! half-plane test; the quad classifies the point into the face-interior or an
//! edge/vertex region by the sum of four edge-sign tests, then takes the
//! perpendicular projection or the minimum over the four clamped edge feet.
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these atoms into complex shapes,
//! the ray-marcher that evaluates them along a ray, and the surface-normal
//! estimation all stay on the host; the device sees only the three stateless,
//! fixed-width distance evaluations, one query at a time, so a storage buffer
//! is never zero-sized.
//!
//! # Correctness model
//!
//! All three distances thread through `sqrt`, products and quotients, so the
//! `CPU` and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few
//! units in the last place from the scalar reference. The parity test asserts
//! each distance within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The
//! death-star half-plane test and the quad edge-region classification flip on a
//! seam where the two branches meet; fixtures stay a safe margin from those
//! seams so a last-place difference never picks a different branch, and the
//! ellipsoid's near-centre guard matches the reference's smallest-radius
//! collapse.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`, `min`,
//! `max`, `clamp`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round` and
//! no `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. The sign of a nonzero value
//! is recovered with an ordered comparison rather than the built-in `sign` so
//! it matches the reference `f32::signum` on the tested domain. It runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` ellipsoid/carved signed-distance kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`ellipsoid_sdf`](prism_render_architecture::ray_scene::sdf_primitives::ellipsoid_sdf),
/// [`death_star`](prism_render_architecture::ray_scene::sdf_primitives::death_star)
/// and
/// [`quad_sdf`](prism_render_architecture::ray_scene::sdf_primitives::quad_sdf).
const SDF_ELLIPSOID_WGSL: &str = r#"
// Ellipsoid/carved signed-distance twin: one thread computes one query point's
// ellipsoid bound, death-star and planar-quad distances, mirroring the CPU
// golden `ray_scene::sdf_primitives::{ellipsoid_sdf, death_star, quad_sdf}`
// with only sqrt, abs, min, max, clamp, products and quotients. The domain/CSG
// operators and the ray-marcher stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_primitives；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point components.
    px: f32,
    py: f32,
    pz: f32,
    // Ellipsoid per-axis radii.
    rx: f32,
    ry: f32,
    rz: f32,
    // Death star: large radius, small (biting) radius, bite distance.
    ds_large: f32,
    ds_small: f32,
    ds_bite: f32,
    // Quad vertex a.
    ax: f32,
    ay: f32,
    az: f32,
    // Quad vertex b.
    bx: f32,
    by: f32,
    bz: f32,
    // Quad vertex c.
    cx: f32,
    cy: f32,
    cz: f32,
    // Quad vertex d.
    dx: f32,
    dy: f32,
    dz: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Distances {
    // Ellipsoid gradient-bound signed distance.
    ellipsoid_sd: f32,
    // Death-star signed distance.
    death_star_sd: f32,
    // Planar-quad unsigned distance.
    quad_sd: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Distances>;

// Euclidean length of a 3-vector.
fn length3(x: f32, y: f32, z: f32) -> f32 {
    return sqrt(x * x + y * y + z * z);
}

// Euclidean length of a 2-vector.
fn length2(x: f32, y: f32) -> f32 {
    return sqrt(x * x + y * y);
}

// Dot product of two 3-vectors, summed left to right to match the reference.
fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Squared length of a 3-vector (dot with itself).
fn dot2_3(v: vec3<f32>) -> f32 {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// Cross product of two 3-vectors, component order matching the reference.
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    );
}

// Sign of a nonzero value, matching the reference `f32::signum` on the tested
// domain (negatives return -1, everything else returns +1). Fixtures stay away
// from the exact zero where the sign bit would decide.
fn sign_of(x: f32) -> f32 {
    if (x < 0.0) {
        return -1.0;
    }
    return 1.0;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Distances;

    // --- Ellipsoid (Inigo Quilez gradient bound k0 * (k0 - 1) / k1) ---
    let sx = q.px / q.rx;
    let sy = q.py / q.ry;
    let sz = q.pz / q.rz;
    let k0 = length3(sx, sy, sz);
    let k1 = length3(sx / q.rx, sy / q.ry, sz / q.rz);
    // Smallest positive normal f32, matching the reference `f32::MIN_POSITIVE`
    // guard at the exact centre where k1 collapses to zero.
    if (k1 <= 1.17549435e-38) {
        out.ellipsoid_sd = -min(min(q.rx, q.ry), q.rz);
    } else {
        out.ellipsoid_sd = k0 * (k0 - 1.0) / k1;
    }

    // --- Death star (large sphere minus a biting sphere) ---
    let dra = q.ds_large;
    let drb = q.ds_small;
    let dd = q.ds_bite;
    let da = (dra * dra - drb * drb + dd * dd) / (2.0 * dd);
    let db = sqrt(max(dra * dra - da * da, 0.0));
    let dp0 = q.px;
    let dp1 = length2(q.py, q.pz);
    if (dp0 * db - dp1 * da > dd * max(db - dp1, 0.0)) {
        out.death_star_sd = length2(dp0 - da, dp1 - db);
    } else {
        out.death_star_sd = max(
            length2(dp0, dp1) - dra,
            -(length2(dp0 - dd, dp1) - drb),
        );
    }

    // --- Planar convex quad (unsigned distance to the four-sided patch) ---
    let pt = vec3<f32>(q.px, q.py, q.pz);
    let qa = vec3<f32>(q.ax, q.ay, q.az);
    let qb = vec3<f32>(q.bx, q.by, q.bz);
    let qc = vec3<f32>(q.cx, q.cy, q.cz);
    let qd = vec3<f32>(q.dx, q.dy, q.dz);
    let ba = qb - qa;
    let pa = pt - qa;
    let cb = qc - qb;
    let pb = pt - qb;
    let dc = qd - qc;
    let pc = pt - qc;
    let ad = qa - qd;
    let pd = pt - qd;
    let nor = cross3(ba, ad);
    let edge_sum = sign_of(dot3(cross3(ba, nor), pa))
        + sign_of(dot3(cross3(cb, nor), pb))
        + sign_of(dot3(cross3(dc, nor), pc))
        + sign_of(dot3(cross3(ad, nor), pd));
    var squared: f32;
    if (edge_sum < 3.0) {
        let t0 = clamp(dot3(ba, pa) / dot2_3(ba), 0.0, 1.0);
        let e0 = dot2_3(ba * t0 - pa);
        let t1 = clamp(dot3(cb, pb) / dot2_3(cb), 0.0, 1.0);
        let e1 = dot2_3(cb * t1 - pb);
        let t2 = clamp(dot3(dc, pc) / dot2_3(dc), 0.0, 1.0);
        let e2 = dot2_3(dc * t2 - pc);
        let t3 = clamp(dot3(ad, pd) / dot2_3(ad), 0.0, 1.0);
        let e3 = dot2_3(ad * t3 - pd);
        squared = min(min(e0, e1), min(e2, e3));
    } else {
        let np = dot3(nor, pa);
        squared = np * np / dot2_3(nor);
    }
    out.quad_sd = sqrt(squared);

    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_ELLIPSOID_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the point components plus all three shapes' parameters and three pad words
/// to a `96`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Query point `z`.
    pz: f32,
    /// Ellipsoid radius along `x`.
    rx: f32,
    /// Ellipsoid radius along `y`.
    ry: f32,
    /// Ellipsoid radius along `z`.
    rz: f32,
    /// Death-star large radius.
    ds_large: f32,
    /// Death-star small (biting) radius.
    ds_small: f32,
    /// Death-star bite distance.
    ds_bite: f32,
    /// Quad vertex `a` component `x`.
    ax: f32,
    /// Quad vertex `a` component `y`.
    ay: f32,
    /// Quad vertex `a` component `z`.
    az: f32,
    /// Quad vertex `b` component `x`.
    bx: f32,
    /// Quad vertex `b` component `y`.
    by: f32,
    /// Quad vertex `b` component `z`.
    bz: f32,
    /// Quad vertex `c` component `x`.
    cx: f32,
    /// Quad vertex `c` component `y`.
    cy: f32,
    /// Quad vertex `c` component `z`.
    cz: f32,
    /// Quad vertex `d` component `x`.
    dx: f32,
    /// Quad vertex `d` component `y`.
    dy: f32,
    /// Quad vertex `d` component `z`.
    dz: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: the three distances plus one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Ellipsoid gradient-bound signed distance.
    ellipsoid_sd: f32,
    /// Death-star signed distance.
    death_star_sd: f32,
    /// Planar-quad unsigned distance.
    quad_sd: f32,
    /// Padding word.
    pad0: f32,
}

/// One query for the ellipsoid/carved signed-distance twin: the query `point`
/// plus the ellipsoid, death-star and quad shape parameters.
///
/// `point` is the evaluation position; `radii` describes the
/// [`ellipsoid_sdf`](prism_render_architecture::ray_scene::sdf_primitives::ellipsoid_sdf);
/// `large_radius`/`small_radius`/`bite_distance` describe the
/// [`death_star`](prism_render_architecture::ray_scene::sdf_primitives::death_star);
/// `quad_a`/`quad_b`/`quad_c`/`quad_d` describe the
/// [`quad_sdf`](prism_render_architecture::ray_scene::sdf_primitives::quad_sdf).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfEllipsoidQuery {
    /// Query point `[x, y, z]`.
    pub point: [f32; 3],
    /// Ellipsoid per-axis radii.
    pub radii: [f32; 3],
    /// Death-star large radius.
    pub large_radius: f32,
    /// Death-star small (biting) radius.
    pub small_radius: f32,
    /// Death-star bite distance.
    pub bite_distance: f32,
    /// Quad vertex `a`.
    pub quad_a: [f32; 3],
    /// Quad vertex `b`.
    pub quad_b: [f32; 3],
    /// Quad vertex `c`.
    pub quad_c: [f32; 3],
    /// Quad vertex `d`.
    pub quad_d: [f32; 3],
}

impl SdfEllipsoidQuery {
    /// Builds a query from the point and all three shapes' parameters.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the twin packs three independent primitives' parameters into one query"
    )]
    pub const fn new(
        point: [f32; 3],
        radii: [f32; 3],
        large_radius: f32,
        small_radius: f32,
        bite_distance: f32,
        quad_a: [f32; 3],
        quad_b: [f32; 3],
        quad_c: [f32; 3],
        quad_d: [f32; 3],
    ) -> SdfEllipsoidQuery {
        SdfEllipsoidQuery {
            point,
            radii,
            large_radius,
            small_radius,
            bite_distance,
            quad_a,
            quad_b,
            quad_c,
            quad_d,
        }
    }
}

/// One resolved query of the ellipsoid/carved signed-distance twin: the
/// ellipsoid, death-star and quad distances at the query point.
///
/// `ellipsoid_sd` is
/// [`ellipsoid_sdf`](prism_render_architecture::ray_scene::sdf_primitives::ellipsoid_sdf)
/// (negative inside, positive outside, zero on the surface); `death_star_sd` is
/// [`death_star`](prism_render_architecture::ray_scene::sdf_primitives::death_star)
/// (same sign convention); `quad_sd` is
/// [`quad_sdf`](prism_render_architecture::ray_scene::sdf_primitives::quad_sdf)
/// (always non-negative, zero exactly on the patch).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfEllipsoidResult {
    /// Ellipsoid gradient-bound signed distance.
    pub ellipsoid_sd: f32,
    /// Death-star signed distance.
    pub death_star_sd: f32,
    /// Planar-quad unsigned distance.
    pub quad_sd: f32,
}

/// Encodes one [`SdfEllipsoidQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfEllipsoidQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        rx: q.radii[0],
        ry: q.radii[1],
        rz: q.radii[2],
        ds_large: q.large_radius,
        ds_small: q.small_radius,
        ds_bite: q.bite_distance,
        ax: q.quad_a[0],
        ay: q.quad_a[1],
        az: q.quad_a[2],
        bx: q.quad_b[0],
        by: q.quad_b[1],
        bz: q.quad_b[2],
        cx: q.quad_c[0],
        cy: q.quad_c[1],
        cz: q.quad_c[2],
        dx: q.quad_d[0],
        dy: q.quad_d[1],
        dz: q.quad_d[2],
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfEllipsoidResult`].
fn decode_result(raw: &GpuResult) -> SdfEllipsoidResult {
    SdfEllipsoidResult {
        ellipsoid_sd: raw.ellipsoid_sd,
        death_star_sd: raw.death_star_sd,
        quad_sd: raw.quad_sd,
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

/// A compiled, reusable ellipsoid/carved signed-distance compute pipeline,
/// twinning the `CPU` golden
/// [`ellipsoid_sdf`](prism_render_architecture::ray_scene::sdf_primitives::ellipsoid_sdf),
/// [`death_star`](prism_render_architecture::ray_scene::sdf_primitives::death_star)
/// and
/// [`quad_sdf`](prism_render_architecture::ray_scene::sdf_primitives::quad_sdf).
pub struct GpuSdfEllipsoid {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfEllipsoid {
    /// Compiles the kernel and builds the reusable pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfEllipsoid {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_ellipsoid_module"),
            source: ShaderSource::Wgsl(SDF_ELLIPSOID_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_ellipsoid_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_ellipsoid_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_ellipsoid_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfEllipsoid {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfEllipsoidResult`]
    /// per input, in order.
    ///
    /// The distances match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfEllipsoidQuery],
    ) -> Vec<SdfEllipsoidResult> {
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
            label: Some("prism_volumetric_sdf_ellipsoid_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_ellipsoid_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_ellipsoid_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_ellipsoid_bind_group"),
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
            label: Some("prism_volumetric_sdf_ellipsoid_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_ellipsoid_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_ellipsoid_pass"),
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
