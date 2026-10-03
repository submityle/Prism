//! `wgpu` compute twin of three arbitrary-orientation cone/rhombus analytic
//! signed-distance primitives of the `CPU` golden path
//! ([`capped_cone_segment`](prism_render_architecture::ray_scene::sdf_primitives::capped_cone_segment),
//! [`round_cone_segment`](prism_render_architecture::ray_scene::sdf_primitives::round_cone_segment)
//! and [`rhombus`](prism_render_architecture::ray_scene::sdf_primitives::rhombus)).
//!
//! Implicit modelling needs *analytic* primitives whose exact signed distance
//! is known in closed form rather than sampled on a grid. The reference derives
//! three solids: a `capped_cone_segment` (a truncated cone — frustum — between
//! two arbitrary endpoints with a flat cap radius at each end), a
//! `round_cone_segment` (the convex hull of two spheres at arbitrary endpoints,
//! a tapered capsule), and a 3D `rhombus` (a rhombic bipyramid cross-section
//! extruded along `y` with a rounding radius). [`GpuSdfConeSegment`] is the
//! on-device twin: each thread reads one point plus every shape's parameters
//! and writes all three signed distances, reproducing the reference closed
//! forms with only `sqrt`, `abs`, `min`, `max`, `clamp`, `select`, products and
//! quotients.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfConeSegmentQuery`] — a query `point` plus the
//! capped-cone endpoints/radii (`capped_cone_a`, `capped_cone_b`,
//! `capped_cone_ra`, `capped_cone_rb`), the round-cone endpoints/radii
//! (`round_cone_a`, `round_cone_b`, `round_cone_r1`, `round_cone_r2`) and the
//! rhombus parameters (`rhombus_half_diag_x`, `rhombus_half_diag_z`,
//! `rhombus_half_height`, `rhombus_rounding`) — and writes one
//! [`SdfConeSegmentResult`] holding the three signed distances
//! `capped_cone_segment_value`, `round_cone_segment_value` and `rhombus_value`.
//!
//! The `capped_cone_segment` kernel projects the query onto the `a`->`b` axis,
//! takes the guarded radial distance, and resolves the nearer of the cap-rim
//! feature and the slanted lateral feature, flipping the interior sign when
//! both lateral and axial residuals are negative. The `round_cone_segment`
//! kernel splits the query into axial, beyond-far-cap and squared-radial
//! components scaled by the squared axis length, then a single comparison
//! against the slope term selects whether the near sphere, the far sphere or
//! the exact tangent flank governs. The `rhombus` kernel folds the point into
//! the first octant, projects onto the clamped edge to measure the planar edge
//! distance signed by the interior side test, and combines it with the vertical
//! cap via the rounded-box interior/exterior split.
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these atoms into complex shapes,
//! the ray-marcher that evaluates them along a ray, and the surface-normal
//! estimation all stay on the host; the device sees only the three stateless,
//! fixed-width signed-distance evaluations, one query at a time, so a storage
//! buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Every distance threads through `sqrt`, products and quotients, so the `CPU`
//! and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units
//! in the last place from the scalar reference. The parity test asserts each
//! distance within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, loose enough to
//! admit a legal last-place difference yet tight enough to catch a wrong port.
//! All three closed forms are continuous signed distances, so a last-place
//! difference in a branch or sign select only ever multiplies a near-zero
//! quantity; fixtures and the randomized sweep still keep a safe margin off
//! each surface via rejection sampling so the reported magnitudes stay well
//! clear of the sign-flip locus.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`, `min`,
//! `max`, `clamp`, `select`, `+ - * /` and unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round` and no `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs
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

/// The portable core-`WGSL` cone/rhombus signed-distance kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`capped_cone_segment`](prism_render_architecture::ray_scene::sdf_primitives::capped_cone_segment),
/// [`round_cone_segment`](prism_render_architecture::ray_scene::sdf_primitives::round_cone_segment)
/// and [`rhombus`](prism_render_architecture::ray_scene::sdf_primitives::rhombus)
/// closed forms; see the module documentation for the algorithm.
const SDF_CONE_SEGMENT_WGSL: &str = r#"
// Cone/rhombus signed-distance twin: one thread computes one query point's
// capped-cone-segment, round-cone-segment and rhombus signed distances,
// mirroring the CPU golden
// `ray_scene::sdf_primitives::{capped_cone_segment, round_cone_segment, rhombus}`
// with only sqrt, abs, min, max, clamp, select, products and quotients. The
// domain/CSG operators and the ray-marcher stay on the host.
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
    // Capped-cone segment: endpoint a, endpoint b, cap radii at a and b.
    cc_ax: f32,
    cc_ay: f32,
    cc_az: f32,
    cc_bx: f32,
    cc_by: f32,
    cc_bz: f32,
    cc_ra: f32,
    cc_rb: f32,
    // Round-cone segment: endpoint a, endpoint b, sphere radii at a and b.
    rc_ax: f32,
    rc_ay: f32,
    rc_az: f32,
    rc_bx: f32,
    rc_by: f32,
    rc_bz: f32,
    rc_r1: f32,
    rc_r2: f32,
    // Rhombus: half-diagonals along x and z, half-height along y, rounding.
    rh_half_diag_x: f32,
    rh_half_diag_z: f32,
    rh_half_height: f32,
    rh_rounding: f32,
    pad0: f32,
}

struct Distances {
    // Capped-cone-segment signed distance.
    capped_cone_segment_value: f32,
    // Round-cone-segment signed distance.
    round_cone_segment_value: f32,
    // Rhombus signed distance.
    rhombus_value: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Distances>;

// Dot product of two 3-vectors in the golden operation order.
fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Euclidean length of a 2-vector, matching the golden `length2` operation order
// `sqrt(x*x + y*y)` exactly.
fn len2(v: vec2<f32>) -> f32 {
    return sqrt(v.x * v.x + v.y * v.y);
}

// Rust `f32::signum` for the non-zero domain: `+1` for non-negative inputs,
// `-1` for negative inputs. Fixtures stay off the exact zero so the two-way
// select never straddles the convention boundary.
fn signum_rs(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

// Capped cone (frustum) with arbitrary endpoints: project onto the a->b axis,
// take the guarded radial distance, resolve the nearer of the cap-rim and the
// slanted lateral feature, flip the interior sign when both residuals negative.
fn capped_cone_segment_sd(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, ra: f32, rb: f32) -> f32 {
    let rba = rb - ra;
    let ba = b - a;
    let pa = p - a;
    let baba = dot3(ba, ba);
    let papa = dot3(pa, pa);
    let paba = dot3(pa, ba) / baba;
    let x = sqrt(max(papa - paba * paba * baba, 0.0));
    let cap_r = select(rb, ra, paba < 0.5);
    let cax = max(x - cap_r, 0.0);
    let cay = abs(paba - 0.5) - 0.5;
    let k = rba * rba + baba;
    let f = clamp((rba * (x - ra) + paba * baba) / k, 0.0, 1.0);
    let cbx = x - ra - f * rba;
    let cby = paba - f;
    let s = select(1.0, -1.0, cbx < 0.0 && cay < 0.0);
    return s * sqrt(min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));
}

// Round cone with arbitrary endpoints: split into axial, beyond-far-cap and
// squared-radial components scaled by the squared axis length, then one slope
// comparison selects near sphere, far sphere, or the exact tangent flank.
fn round_cone_segment_sd(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r1: f32, r2: f32) -> f32 {
    let ba = b - a;
    let l2 = dot3(ba, ba);
    let rr = r1 - r2;
    let a2 = l2 - rr * rr;
    let il2 = 1.0 / l2;
    let pa = p - a;
    let y = dot3(pa, ba);
    let z = y - l2;
    let perp = vec3<f32>(pa.x * l2 - ba.x * y, pa.y * l2 - ba.y * y, pa.z * l2 - ba.z * y);
    let x2 = dot3(perp, perp);
    let y2 = y * y * l2;
    let z2 = z * z * l2;
    let k = signum_rs(rr) * rr * rr * x2;
    if (signum_rs(z) * a2 * z2 > k) {
        return sqrt(x2 + z2) * il2 - r2;
    }
    if (signum_rs(y) * a2 * y2 < k) {
        return sqrt(x2 + y2) * il2 - r1;
    }
    return sqrt(x2 * a2 * il2) * il2 + y * rr * il2 - r1;
}

// Rhombic bipyramid cross-section extruded along y with a rounding radius: fold
// into the first octant, project onto the clamped edge for the planar edge
// distance signed by the side test, combine with the vertical cap.
fn rhombus_sd(point: vec3<f32>, bx: f32, bz: f32, half_height: f32, rounding: f32) -> f32 {
    let px = abs(point.x);
    let py = abs(point.y);
    let pz = abs(point.z);
    let ndot = bx * (bx - 2.0 * px) - bz * (bz - 2.0 * pz);
    let denom = bx * bx + bz * bz;
    let f = clamp(ndot / denom, -1.0, 1.0);
    let foot_x = 0.5 * bx * (1.0 - f);
    let foot_z = 0.5 * bz * (1.0 + f);
    let edge = len2(vec2<f32>(px - foot_x, pz - foot_z));
    let side = signum_rs(px * bz + pz * bx - bx * bz);
    let qx = edge * side - rounding;
    let qy = py - half_height;
    return min(max(qx, qy), 0.0) + len2(vec2<f32>(max(qx, 0.0), max(qy, 0.0)));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let point = vec3<f32>(q.px, q.py, q.pz);

    var out: Distances;
    out.capped_cone_segment_value = capped_cone_segment_sd(
        point,
        vec3<f32>(q.cc_ax, q.cc_ay, q.cc_az),
        vec3<f32>(q.cc_bx, q.cc_by, q.cc_bz),
        q.cc_ra,
        q.cc_rb,
    );
    out.round_cone_segment_value = round_cone_segment_sd(
        point,
        vec3<f32>(q.rc_ax, q.rc_ay, q.rc_az),
        vec3<f32>(q.rc_bx, q.rc_by, q.rc_bz),
        q.rc_r1,
        q.rc_r2,
    );
    out.rhombus_value = rhombus_sd(
        point,
        q.rh_half_diag_x,
        q.rh_half_diag_z,
        q.rh_half_height,
        q.rh_rounding,
    );
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching the `WGSL` `Params`.
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
/// the query point plus every shape's parameters, padded to a `96`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Query point `z`.
    pz: f32,
    /// Capped-cone endpoint `a` component `x`.
    cc_ax: f32,
    /// Capped-cone endpoint `a` component `y`.
    cc_ay: f32,
    /// Capped-cone endpoint `a` component `z`.
    cc_az: f32,
    /// Capped-cone endpoint `b` component `x`.
    cc_bx: f32,
    /// Capped-cone endpoint `b` component `y`.
    cc_by: f32,
    /// Capped-cone endpoint `b` component `z`.
    cc_bz: f32,
    /// Capped-cone cap radius at `a`.
    cc_ra: f32,
    /// Capped-cone cap radius at `b`.
    cc_rb: f32,
    /// Round-cone endpoint `a` component `x`.
    rc_ax: f32,
    /// Round-cone endpoint `a` component `y`.
    rc_ay: f32,
    /// Round-cone endpoint `a` component `z`.
    rc_az: f32,
    /// Round-cone endpoint `b` component `x`.
    rc_bx: f32,
    /// Round-cone endpoint `b` component `y`.
    rc_by: f32,
    /// Round-cone endpoint `b` component `z`.
    rc_bz: f32,
    /// Round-cone sphere radius at `a`.
    rc_r1: f32,
    /// Round-cone sphere radius at `b`.
    rc_r2: f32,
    /// Rhombus half-diagonal along `x`.
    rh_half_diag_x: f32,
    /// Rhombus half-diagonal along `z`.
    rh_half_diag_z: f32,
    /// Rhombus half-height along `y`.
    rh_half_height: f32,
    /// Rhombus rounding radius.
    rh_rounding: f32,
    /// Padding word to a `96`-byte stride.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: the three signed distances plus one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Capped-cone-segment signed distance.
    capped_cone_segment_value: f32,
    /// Round-cone-segment signed distance.
    round_cone_segment_value: f32,
    /// Rhombus signed distance.
    rhombus_value: f32,
    /// Padding word.
    pad0: f32,
}

/// One query for the cone/rhombus signed-distance twin: the query `point` plus
/// the capped-cone, round-cone and rhombus shape parameters.
///
/// `point` is the evaluation position; `capped_cone_a`/`capped_cone_b` are the
/// [`capped_cone_segment`](prism_render_architecture::ray_scene::sdf_primitives::capped_cone_segment)
/// endpoints with cap radii `capped_cone_ra`/`capped_cone_rb`;
/// `round_cone_a`/`round_cone_b` are the
/// [`round_cone_segment`](prism_render_architecture::ray_scene::sdf_primitives::round_cone_segment)
/// endpoints with sphere radii `round_cone_r1`/`round_cone_r2`;
/// `rhombus_half_diag_x`/`rhombus_half_diag_z`/`rhombus_half_height`/`rhombus_rounding`
/// are the
/// [`rhombus`](prism_render_architecture::ray_scene::sdf_primitives::rhombus)
/// half-diagonals, half-height and rounding radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfConeSegmentQuery {
    /// Query point `[x, y, z]`.
    pub point: [f32; 3],
    /// Capped-cone endpoint `a`.
    pub capped_cone_a: [f32; 3],
    /// Capped-cone endpoint `b`.
    pub capped_cone_b: [f32; 3],
    /// Capped-cone cap radius at `a`.
    pub capped_cone_ra: f32,
    /// Capped-cone cap radius at `b`.
    pub capped_cone_rb: f32,
    /// Round-cone endpoint `a`.
    pub round_cone_a: [f32; 3],
    /// Round-cone endpoint `b`.
    pub round_cone_b: [f32; 3],
    /// Round-cone sphere radius at `a`.
    pub round_cone_r1: f32,
    /// Round-cone sphere radius at `b`.
    pub round_cone_r2: f32,
    /// Rhombus half-diagonal along `x`.
    pub rhombus_half_diag_x: f32,
    /// Rhombus half-diagonal along `z`.
    pub rhombus_half_diag_z: f32,
    /// Rhombus half-height along `y`.
    pub rhombus_half_height: f32,
    /// Rhombus rounding radius.
    pub rhombus_rounding: f32,
}

impl SdfConeSegmentQuery {
    /// Builds a query from the point and every shape's parameters.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the three golden signatures packed into one query slot"
    )]
    pub const fn new(
        point: [f32; 3],
        capped_cone_a: [f32; 3],
        capped_cone_b: [f32; 3],
        capped_cone_ra: f32,
        capped_cone_rb: f32,
        round_cone_a: [f32; 3],
        round_cone_b: [f32; 3],
        round_cone_r1: f32,
        round_cone_r2: f32,
        rhombus_half_diag_x: f32,
        rhombus_half_diag_z: f32,
        rhombus_half_height: f32,
        rhombus_rounding: f32,
    ) -> SdfConeSegmentQuery {
        SdfConeSegmentQuery {
            point,
            capped_cone_a,
            capped_cone_b,
            capped_cone_ra,
            capped_cone_rb,
            round_cone_a,
            round_cone_b,
            round_cone_r1,
            round_cone_r2,
            rhombus_half_diag_x,
            rhombus_half_diag_z,
            rhombus_half_height,
            rhombus_rounding,
        }
    }
}

/// One resolved query of the cone/rhombus signed-distance twin: the capped-cone,
/// round-cone and rhombus signed distances at the query point.
///
/// `capped_cone_segment_value` is
/// [`capped_cone_segment`](prism_render_architecture::ray_scene::sdf_primitives::capped_cone_segment);
/// `round_cone_segment_value` is
/// [`round_cone_segment`](prism_render_architecture::ray_scene::sdf_primitives::round_cone_segment);
/// `rhombus_value` is
/// [`rhombus`](prism_render_architecture::ray_scene::sdf_primitives::rhombus).
/// Each is negative inside the solid, positive outside, zero on the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfConeSegmentResult {
    /// Capped-cone-segment signed distance.
    pub capped_cone_segment_value: f32,
    /// Round-cone-segment signed distance.
    pub round_cone_segment_value: f32,
    /// Rhombus signed distance.
    pub rhombus_value: f32,
}

/// Encodes one [`SdfConeSegmentQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfConeSegmentQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        cc_ax: q.capped_cone_a[0],
        cc_ay: q.capped_cone_a[1],
        cc_az: q.capped_cone_a[2],
        cc_bx: q.capped_cone_b[0],
        cc_by: q.capped_cone_b[1],
        cc_bz: q.capped_cone_b[2],
        cc_ra: q.capped_cone_ra,
        cc_rb: q.capped_cone_rb,
        rc_ax: q.round_cone_a[0],
        rc_ay: q.round_cone_a[1],
        rc_az: q.round_cone_a[2],
        rc_bx: q.round_cone_b[0],
        rc_by: q.round_cone_b[1],
        rc_bz: q.round_cone_b[2],
        rc_r1: q.round_cone_r1,
        rc_r2: q.round_cone_r2,
        rh_half_diag_x: q.rhombus_half_diag_x,
        rh_half_diag_z: q.rhombus_half_diag_z,
        rh_half_height: q.rhombus_half_height,
        rh_rounding: q.rhombus_rounding,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfConeSegmentResult`].
fn decode_result(raw: &GpuResult) -> SdfConeSegmentResult {
    SdfConeSegmentResult {
        capped_cone_segment_value: raw.capped_cone_segment_value,
        round_cone_segment_value: raw.round_cone_segment_value,
        rhombus_value: raw.rhombus_value,
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

/// A compiled, reusable cone/rhombus signed-distance compute pipeline, twinning
/// the `CPU` golden
/// [`capped_cone_segment`](prism_render_architecture::ray_scene::sdf_primitives::capped_cone_segment),
/// [`round_cone_segment`](prism_render_architecture::ray_scene::sdf_primitives::round_cone_segment)
/// and [`rhombus`](prism_render_architecture::ray_scene::sdf_primitives::rhombus).
pub struct GpuSdfConeSegment {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfConeSegment {
    /// Compiles the cone/rhombus signed-distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfConeSegment {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_cone_segment"),
            source: ShaderSource::Wgsl(SDF_CONE_SEGMENT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_cone_segment_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_cone_segment_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_cone_segment_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfConeSegment {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfConeSegmentResult`]
    /// per input, in order.
    ///
    /// The signed distances match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfConeSegmentQuery],
    ) -> Vec<SdfConeSegmentResult> {
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
            label: Some("prism_volumetric_sdf_cone_segment_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_cone_segment_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_cone_segment_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_cone_segment_bind_group"),
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
            label: Some("prism_volumetric_sdf_cone_segment_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_cone_segment_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_cone_segment_pass"),
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
