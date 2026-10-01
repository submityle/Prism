//! `wgpu` compute twin of the triangle barycentric-coordinate, attribute
//! interpolation and inside-test contract
//! ([`barycentric_coord`](prism_render_architecture::particle::barycentric_coord),
//! particle design §8.2).
//!
//! The `CPU` golden
//! [`barycentric_coord`](prism_render_architecture::particle::barycentric_coord)
//! owns the small, verifiable geometry several particle stages share: expressing
//! a point as a blend of a triangle's three corners
//! ([`Triangle2::barycentric2`](prism_render_architecture::particle::barycentric_coord::Triangle2::barycentric2)
//! in the plane and
//! [`Triangle3::barycentric3`](prism_render_architecture::particle::barycentric_coord::Triangle3::barycentric3)
//! in space), deciding whether those weights place the point inside the triangle
//! ([`point_in_triangle`](prism_render_architecture::particle::barycentric_coord::point_in_triangle)),
//! blending per-vertex attributes by the weights
//! ([`interpolate_scalar`](prism_render_architecture::particle::barycentric_coord::interpolate_scalar)
//! and its `vec2` / `vec3` / `vec4` siblings), and applying the perspective
//! divide a screen-space sample needs
//! ([`perspective_correct`](prism_render_architecture::particle::barycentric_coord::perspective_correct)).
//! [`GpuBarycentricCoord`] is the on-device twin: one thread solves one query,
//! so a passing real-device parity test is direct evidence the ported kernel
//! computes the same weights and classifies the same degenerate cases the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced for a batch of
//! independent queries: the 2D plane weights and the 3D Gram-matrix weights
//! (each with a validity flag mirroring the reference [`Option`]), the inside
//! test on a supplied weight triple, the scalar and `vec4` attribute blends, and
//! the perspective-correct reweighting. The `vec2` and `vec3` attribute blends
//! are not stored separately: barycentric interpolation is component-wise and
//! independent per lane, so the first two or three lanes of the `vec4` blend are
//! exactly [`interpolate_vec2`](prism_render_architecture::particle::barycentric_coord::interpolate_vec2)
//! and [`interpolate_vec3`](prism_render_architecture::particle::barycentric_coord::interpolate_vec3)
//! of the same per-lane attributes, and the parity test checks them that way.
//!
//! # Correctness model
//!
//! The validity flags and the inside flag are discrete classifications built
//! from `f32` magnitude comparisons, so for inputs clear of the degeneracy and
//! boundary thresholds the `CPU` and `GPU` agree exactly and the parity test
//! asserts an exact `==` on both. The weights and the blended attributes thread
//! through multiplies, adds and one guarded division, so `CPU` and `GPU` are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits. The parity test therefore asserts
//! a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every continuous
//! quantity, tight enough to catch a genuinely wrong port (a dropped term, a
//! swapped corner, a wrong clamp) yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A degenerate (zero-area) triangle would divide the weights by a near-zero
//! signed area or Gram determinant; the kernel checks the denominator against
//! [`CMP_EPS`] and reports an invalid flag (weights left at zero) instead,
//! matching the reference [`None`]. The perspective divide likewise falls back
//! to the affine weights when the reweighted sum is within [`CMP_EPS`] of zero,
//! so the result is never `NaN`. An empty query batch short-circuits on the host
//! with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `dot`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `sqrt` and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There
//! is no loop: each thread performs a fixed, bounded sequence of arithmetic, so
//! the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::barycentric_coord`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` barycentric-coordinate kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`barycentric_coord`](prism_render_architecture::particle::barycentric_coord)
/// branch for branch; see the module documentation for the algorithm.
const BARYCENTRIC_COORD_WGSL: &str = r#"
// Barycentric-coordinate twin: one thread per query reproduces the 2D plane
// weights, the 3D Gram-matrix weights, the inside test, the scalar and vec4
// attribute blends, and the perspective-correct reweighting. It mirrors the CPU
// golden `particle::barycentric_coord` branch for branch, uses only the portable
// core-WGSL subset (abs/dot and + - * / plus unsigned index math), needs no
// sqrt and no transcendental call and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12. There is no loop, so the kernel
// provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::barycentric_coord；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a signed area, a Gram determinant, a perspective sum or
// a weight is treated as zero. Matches the reference `CMP_EPS`; the compare rule
// used instead of an f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // 2D triangle corners a, b, c packed as three vec2, and the 2D query point.
    tri2_a: vec2<f32>,
    tri2_b: vec2<f32>,
    tri2_c: vec2<f32>,
    point2: vec2<f32>,
    // 3D triangle corners a, b, c (each a vec3 with a trailing pad lane), and the
    // 3D query point.
    tri3_a: vec3<f32>,
    pad_a: f32,
    tri3_b: vec3<f32>,
    pad_b: f32,
    tri3_c: vec3<f32>,
    pad_c: f32,
    point3: vec3<f32>,
    pad_p: f32,
    // Supplied weight triple for the inside test, the attribute blends and the
    // perspective divide; a pad lane follows.
    weights: vec3<f32>,
    pad_w: f32,
    // Per-corner scalar attributes for interpolate_scalar; a pad lane follows.
    scalar_attrs: vec3<f32>,
    pad_s: f32,
    // Per-corner 1/w clip-space values for perspective_correct; a pad lane
    // follows.
    inv_w: vec3<f32>,
    pad_i: f32,
    // Per-corner vec4 attributes for interpolate_vec4 (the vec2 / vec3 blends are
    // its leading lanes).
    attr_a: vec4<f32>,
    attr_b: vec4<f32>,
    attr_c: vec4<f32>,
}

struct Result {
    // barycentric2 weights, with the validity flag packed into w's lane.
    bary2: vec3<f32>,
    valid2: u32,
    // barycentric3 weights, with the validity flag packed into w's lane.
    bary3: vec3<f32>,
    valid3: u32,
    // perspective_correct weights, with the inside flag packed into w's lane.
    persp: vec3<f32>,
    inside: u32,
    // interpolate_vec4 blend.
    interp4: vec4<f32>,
    // interpolate_scalar blend, with three pad lanes.
    interp_scalar: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Scalar 2D cross product (the z component of the 3D cross), i.e. twice the
// signed area spanned by `lhs` and `rhs`; mirrors the reference `cross2`.
fn cross2(lhs: vec2<f32>, rhs: vec2<f32>) -> f32 {
    return lhs.x * rhs.y - lhs.y * rhs.x;
}

// Blends three per-corner scalars by the barycentric `w`; mirrors the reference
// `interpolate_scalar`.
fn blend_scalar(w: vec3<f32>, c0: f32, c1: f32, c2: f32) -> f32 {
    return w.x * c0 + w.y * c1 + w.z * c2;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // barycentric2: ratio of signed sub-areas. A degenerate (near-zero area)
    // triangle sets valid2 = 0 and leaves the weights at zero rather than
    // dividing by ~zero, mirroring the reference `None`.
    var bary2: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var valid2: u32 = 0u;
    let denom2 = cross2(q.tri2_b - q.tri2_a, q.tri2_c - q.tri2_a);
    if (abs(denom2) >= CMP_EPS) {
        let inv2 = 1.0 / denom2;
        let wa = cross2(q.tri2_c - q.tri2_b, q.point2 - q.tri2_b) * inv2;
        let wb = cross2(q.tri2_a - q.tri2_c, q.point2 - q.tri2_c) * inv2;
        let wc = cross2(q.tri2_b - q.tri2_a, q.point2 - q.tri2_a) * inv2;
        bary2 = vec3<f32>(wa, wb, wc);
        valid2 = 1u;
    }

    // barycentric3: Christer Ericson's Gram-matrix solve in the triangle's own
    // plane. A degenerate triangle sets valid3 = 0, mirroring the reference.
    var bary3: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var valid3: u32 = 0u;
    let v0 = q.tri3_b - q.tri3_a;
    let v1 = q.tri3_c - q.tri3_a;
    let v2 = q.point3 - q.tri3_a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom3 = d00 * d11 - d01 * d01;
    if (abs(denom3) >= CMP_EPS) {
        let inv3 = 1.0 / denom3;
        let v = (d11 * d20 - d01 * d21) * inv3;
        let w = (d00 * d21 - d01 * d20) * inv3;
        let u = 1.0 - v - w;
        bary3 = vec3<f32>(u, v, w);
        valid3 = 1u;
    }

    // point_in_triangle on the supplied weight triple: every weight at least
    // -CMP_EPS keeps an on-edge point classified as inside.
    let wgt = q.weights;
    var inside: u32 = 0u;
    if (wgt.x >= -CMP_EPS && wgt.y >= -CMP_EPS && wgt.z >= -CMP_EPS) {
        inside = 1u;
    }

    // interpolate_scalar and interpolate_vec4 blend the per-corner attributes by
    // the supplied weights; the vec2 / vec3 blends are the leading lanes of the
    // vec4 result.
    let interp_scalar = blend_scalar(wgt, q.scalar_attrs.x, q.scalar_attrs.y, q.scalar_attrs.z);
    let interp4 = vec4<f32>(
        blend_scalar(wgt, q.attr_a.x, q.attr_b.x, q.attr_c.x),
        blend_scalar(wgt, q.attr_a.y, q.attr_b.y, q.attr_c.y),
        blend_scalar(wgt, q.attr_a.z, q.attr_b.z, q.attr_c.z),
        blend_scalar(wgt, q.attr_a.w, q.attr_b.w, q.attr_c.w),
    );

    // perspective_correct: reweight by 1/w and renormalize. A reweighted sum
    // within CMP_EPS of zero falls back to the affine weights so the result is
    // never NaN.
    let n = vec3<f32>(wgt.x * q.inv_w.x, wgt.y * q.inv_w.y, wgt.z * q.inv_w.z);
    let sum = n.x + n.y + n.z;
    var persp: vec3<f32> = wgt;
    if (abs(sum) >= CMP_EPS) {
        let inv_sum = 1.0 / sum;
        persp = vec3<f32>(n.x * inv_sum, n.y * inv_sum, n.z * inv_sum);
    }

    var out: Result;
    out.bary2 = bary2;
    out.valid2 = valid2;
    out.bary3 = bary3;
    out.valid3 = valid3;
    out.persp = persp;
    out.inside = inside;
    out.interp4 = interp4;
    out.interp_scalar = interp_scalar;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`BARYCENTRIC_COORD_WGSL`].
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
/// Every `vec3` lane carries a trailing pad word so each stays `16`-byte
/// aligned on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// 2D triangle corner `a`.
    tri2_a: [f32; 2],
    /// 2D triangle corner `b`.
    tri2_b: [f32; 2],
    /// 2D triangle corner `c`.
    tri2_c: [f32; 2],
    /// 2D query point.
    point2: [f32; 2],
    /// 3D triangle corner `a`.
    tri3_a: [f32; 3],
    /// Pad lane after `tri3_a`.
    pad_a: f32,
    /// 3D triangle corner `b`.
    tri3_b: [f32; 3],
    /// Pad lane after `tri3_b`.
    pad_b: f32,
    /// 3D triangle corner `c`.
    tri3_c: [f32; 3],
    /// Pad lane after `tri3_c`.
    pad_c: f32,
    /// 3D query point.
    point3: [f32; 3],
    /// Pad lane after `point3`.
    pad_p: f32,
    /// Supplied weight triple for the blends, inside test and perspective divide.
    weights: [f32; 3],
    /// Pad lane after `weights`.
    pad_w: f32,
    /// Per-corner scalar attributes.
    scalar_attrs: [f32; 3],
    /// Pad lane after `scalar_attrs`.
    pad_s: f32,
    /// Per-corner `1/w` clip-space values.
    inv_w: [f32; 3],
    /// Pad lane after `inv_w`.
    pad_i: f32,
    /// Corner `a` `vec4` attribute.
    attr_a: [f32; 4],
    /// Corner `b` `vec4` attribute.
    attr_b: [f32; 4],
    /// Corner `c` `vec4` attribute.
    attr_c: [f32; 4],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `barycentric2` weights (zero when invalid).
    bary2: [f32; 3],
    /// `1` when the 2D triangle was non-degenerate, `0` otherwise.
    valid2: u32,
    /// `barycentric3` weights (zero when invalid).
    bary3: [f32; 3],
    /// `1` when the 3D triangle was non-degenerate, `0` otherwise.
    valid3: u32,
    /// `perspective_correct` weights.
    persp: [f32; 3],
    /// `1` when the supplied weights place the point inside, `0` otherwise.
    inside: u32,
    /// `interpolate_vec4` blend.
    interp4: [f32; 4],
    /// `interpolate_scalar` blend.
    interp_scalar: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// Padding lane.
    pad2: f32,
}

/// One query for the barycentric twin: a 2D and a 3D triangle with their query
/// points, a supplied weight triple, and the per-corner attributes the blends
/// and perspective divide consume.
///
/// The 2D and 3D solves are independent, as are the weight-driven blends, so a
/// single query exercises every twinned function at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarycentricQuery {
    /// 2D triangle corners in `a`, `b`, `c` order for `barycentric2`.
    pub tri2: [[f32; 2]; 3],
    /// Query point for the 2D solve.
    pub point2: [f32; 2],
    /// 3D triangle corners in `a`, `b`, `c` order for `barycentric3`.
    pub tri3: [[f32; 3]; 3],
    /// Query point for the 3D solve.
    pub point3: [f32; 3],
    /// Weight triple fed to the inside test, the attribute blends and the
    /// perspective divide.
    pub weights: [f32; 3],
    /// Per-corner scalar attributes for `interpolate_scalar`.
    pub scalar_attrs: [f32; 3],
    /// Per-corner `vec4` attributes for `interpolate_vec4`; its leading two and
    /// three lanes are the `interpolate_vec2` and `interpolate_vec3` blends.
    pub vec4_attrs: [[f32; 4]; 3],
    /// Per-corner `1/w` clip-space values for `perspective_correct`.
    pub inv_w: [f32; 3],
}

/// One resolved answer for a single query, mirroring every value the reference
/// reports across its twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarycentricResult {
    /// `barycentric2` weights, or [`None`] for a degenerate 2D triangle, matching
    /// [`Triangle2::barycentric2`](prism_render_architecture::particle::barycentric_coord::Triangle2::barycentric2).
    pub bary2: Option<[f32; 3]>,
    /// `barycentric3` weights, or [`None`] for a degenerate 3D triangle, matching
    /// [`Triangle3::barycentric3`](prism_render_architecture::particle::barycentric_coord::Triangle3::barycentric3).
    pub bary3: Option<[f32; 3]>,
    /// Inside test on the supplied weights, matching
    /// [`point_in_triangle`](prism_render_architecture::particle::barycentric_coord::point_in_triangle).
    pub inside: bool,
    /// Scalar attribute blend, matching
    /// [`interpolate_scalar`](prism_render_architecture::particle::barycentric_coord::interpolate_scalar).
    pub interp_scalar: f32,
    /// `vec4` attribute blend, matching
    /// [`interpolate_vec4`](prism_render_architecture::particle::barycentric_coord::interpolate_vec4);
    /// its leading two and three lanes match the `vec2` and `vec3` blends.
    pub interp_vec4: [f32; 4],
    /// Perspective-correct weights, matching
    /// [`perspective_correct`](prism_render_architecture::particle::barycentric_coord::perspective_correct).
    pub perspective: [f32; 3],
}

/// Encodes one [`BarycentricQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &BarycentricQuery) -> GpuQuery {
    GpuQuery {
        tri2_a: q.tri2[0],
        tri2_b: q.tri2[1],
        tri2_c: q.tri2[2],
        point2: q.point2,
        tri3_a: q.tri3[0],
        pad_a: 0.0,
        tri3_b: q.tri3[1],
        pad_b: 0.0,
        tri3_c: q.tri3[2],
        pad_c: 0.0,
        point3: q.point3,
        pad_p: 0.0,
        weights: q.weights,
        pad_w: 0.0,
        scalar_attrs: q.scalar_attrs,
        pad_s: 0.0,
        inv_w: q.inv_w,
        pad_i: 0.0,
        attr_a: q.vec4_attrs[0],
        attr_b: q.vec4_attrs[1],
        attr_c: q.vec4_attrs[2],
    }
}

/// Decodes one packed [`GpuResult`] into the public [`BarycentricResult`],
/// turning each validity flag back into an [`Option`].
fn decode_result(raw: &GpuResult) -> BarycentricResult {
    BarycentricResult {
        bary2: if raw.valid2 == 0 {
            None
        } else {
            Some(raw.bary2)
        },
        bary3: if raw.valid3 == 0 {
            None
        } else {
            Some(raw.bary3)
        },
        inside: raw.inside != 0,
        interp_scalar: raw.interp_scalar,
        interp_vec4: raw.interp4,
        perspective: raw.persp,
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

/// A compiled, reusable barycentric-coordinate compute pipeline, twinning the
/// `CPU` golden
/// [`barycentric_coord`](prism_render_architecture::particle::barycentric_coord).
pub struct GpuBarycentricCoord {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBarycentricCoord {
    /// Compiles the barycentric-coordinate kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBarycentricCoord {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_barycentric_coord"),
            source: ShaderSource::Wgsl(BARYCENTRIC_COORD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_barycentric_coord_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_barycentric_coord_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_barycentric_coord_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBarycentricCoord {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`BarycentricResult`] per
    /// input, in order.
    ///
    /// The validity and inside flags equal the reference exactly for inputs
    /// clear of the degeneracy and boundary thresholds; the weights and blended
    /// attributes match to within the tolerance documented on this module. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[BarycentricQuery],
    ) -> Vec<BarycentricResult> {
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
            label: Some("prism_volumetric_barycentric_coord_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_barycentric_coord_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_barycentric_coord_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_barycentric_coord_bind_group"),
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
            label: Some("prism_volumetric_barycentric_coord_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_barycentric_coord_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_barycentric_coord_pass"),
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
