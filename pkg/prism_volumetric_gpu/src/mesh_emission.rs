//! `wgpu` compute twin of the mesh-emission surface mathematics
//! ([`mesh_emission`](prism_render_architecture::particle::mesh_emission),
//! particle design §8).
//!
//! The `CPU` golden
//! [`mesh_emission`](prism_render_architecture::particle::mesh_emission) owns
//! the small, verifiable linear algebra a mesh / skeletal-mesh emitter needs
//! once the host has resolved which triangle and which bones a sample touches:
//! linear-blend skinning of a position
//! ([`skin_position`](prism_render_architecture::particle::mesh_emission::skin_position)),
//! linear-blend skinning of a renormalized normal
//! ([`skin_normal`](prism_render_architecture::particle::mesh_emission::skin_normal)),
//! the per-bone affine point / vector transforms
//! ([`BoneTransform::transform_point`](prism_render_architecture::particle::mesh_emission::BoneTransform::transform_point)
//! and
//! [`BoneTransform::transform_vector`](prism_render_architecture::particle::mesh_emission::BoneTransform::transform_vector)),
//! the triangle surface area
//! ([`MeshTriangle::area`](prism_render_architecture::particle::mesh_emission::MeshTriangle::area)),
//! and the barycentric interpolation of a surface point and shading normal
//! ([`MeshTriangle::position_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::position_at)
//! and
//! [`MeshTriangle::normal_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::normal_at)).
//! [`GpuMeshEmission`] is the on-device twin: one thread answers one
//! [`MeshEmissionQuery`], so a passing real-device parity test is direct
//! evidence the ported kernel computes the same skinned positions, skinned
//! normals, areas and barycentric samples the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! A single batched kernel evaluates a tagged union of independent queries,
//! one per thread, mirroring the reference tap for tap:
//!
//! - [`MeshEmissionQuery::SkinPosition`] reproduces
//!   [`skin_position`](prism_render_architecture::particle::mesh_emission::skin_position):
//!   each influence applies its [`BoneTransform`] to the bind-pose position
//!   (`basis_x·p.x + basis_y·p.y + basis_z·p.z + translation`) and accumulates
//!   it scaled by the weight, skipping near-zero weights (`|w| < EPS`).
//! - [`MeshEmissionQuery::SkinNormal`] reproduces
//!   [`skin_normal`](prism_render_architecture::particle::mesh_emission::skin_normal):
//!   the same blend applies the basis only (no translation) and the accumulated
//!   normal is renormalized with the guarded division
//!   [`Vec3::normalize_or_zero`](prism_render_architecture::particle::Vec3::normalize_or_zero).
//! - [`MeshEmissionQuery::TriangleArea`] reproduces
//!   [`MeshTriangle::area`](prism_render_architecture::particle::mesh_emission::MeshTriangle::area):
//!   `0.5 * |(b - a) × (c - a)|`, one cross product and one `sqrt`.
//! - [`MeshEmissionQuery::BarycentricSample`] reproduces
//!   [`MeshTriangle::position_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::position_at)
//!   and
//!   [`MeshTriangle::normal_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::normal_at):
//!   the barycentric blend of the three corner positions, and the renormalized
//!   blend of the three corner shading normals with the geometric-normal
//!   fallback when the blended normal is (numerically) zero.
//!
//! # What stays on the host (not twinned)
//!
//! The reference's `usize` slice traversal and index resolution of
//! [`Mesh`](prism_render_architecture::particle::mesh_emission::Mesh) /
//! [`SkeletalMesh`](prism_render_architecture::particle::mesh_emission::SkeletalMesh),
//! the unit-interval random draws of
//! [`UnitCursor`](prism_render_architecture::particle::emitter::UnitCursor), and
//! the area-`CDF` / alias-table triangle pick of
//! [`Mesh::select_triangle`](prism_render_architecture::particle::mesh_emission::Mesh::select_triangle)
//! remain on the host, which resolves the chosen triangle, the folded
//! barycentric weights, and the per-influence [`BoneTransform`] palette entries,
//! then hands the device the fixed-width, already-selected data each query
//! needs. An out-of-range influence in the reference is simply dropped on the
//! host by zeroing that influence weight, matching the reference `continue`.
//!
//! # Correctness model
//!
//! The skinned positions, skinned normals, areas and samples all thread through
//! multiplies, adds, a cross product and one guarded reciprocal `sqrt`, so `CPU`
//! and `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits. The parity test
//! therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on
//! every continuous quantity, tight enough to catch a genuinely wrong port (a
//! dropped influence, a swapped basis column, a missing translation, a swapped
//! corner) yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A skinned normal that blends to (numerically) zero, or a collinear triangle
//! whose cross product has a squared length at or below [`EPS_LEN_SQ`], yields
//! the zero vector instead of dividing by a near-zero length, matching the
//! reference `normalize_or_zero`. [`MeshEmissionQuery::BarycentricSample`] falls
//! back to the geometric normal when the blended shading normal is zero,
//! mirroring the reference `normal_at`. An empty batch short-circuits on the
//! host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `dot`,
//! `cross`, `sqrt`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! only loop is a fixed, bounded sweep over the four influences, so the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::mesh_emission`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::mesh_emission::{BoneTransform, MAX_BONE_INFLUENCES};
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

/// Operation tag selecting [`MeshEmissionQuery::SkinPosition`] on device.
const OP_SKIN_POSITION: u32 = 0;
/// Operation tag selecting [`MeshEmissionQuery::SkinNormal`] on device.
const OP_SKIN_NORMAL: u32 = 1;
/// Operation tag selecting [`MeshEmissionQuery::TriangleArea`] on device.
const OP_TRIANGLE_AREA: u32 = 2;
/// Operation tag selecting [`MeshEmissionQuery::BarycentricSample`] on device.
const OP_BARYCENTRIC_SAMPLE: u32 = 3;

/// The portable core-`WGSL` mesh-emission kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`mesh_emission`](prism_render_architecture::particle::mesh_emission)
/// function for function; see the module documentation for the algorithm.
const MESH_EMISSION_WGSL: &str = r#"
// Mesh-emission twin: one thread per query reproduces linear-blend skinning of a
// position (`skin_position`), linear-blend skinning of a renormalized normal
// (`skin_normal`), the triangle surface area (`MeshTriangle::area`) and the
// barycentric sample of a position and shading normal (`MeshTriangle::position_at`
// and `MeshTriangle::normal_at`). It mirrors the CPU golden
// `particle::mesh_emission` function for function, uses only the portable
// core-WGSL subset (abs/dot/cross/sqrt and + - * / plus unsigned index math),
// needs no transcendental call and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12. The only loop is a bounded four-step
// influence sweep, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::mesh_emission；无第三方
// 引擎源码或衍生代码。

// Weight magnitude below which an influence is skipped, matching the reference
// `EPS`; the compare rule used instead of an f32 `==`.
const EPS: f32 = 1.0e-6;

// Squared-length magnitude below which a vector is treated as zero, so a
// normalization yields the zero vector rather than dividing by a near-zero
// length. Matches the reference `EPS_LEN_SQ`; the compare rule used instead of
// an f32 `==`.
const EPS_LEN_SQ: f32 = 1.0e-12;

// Operation tags, matching the host op codes.
const OP_SKIN_POSITION: u32 = 0u;
const OP_SKIN_NORMAL: u32 = 1u;
const OP_TRIANGLE_AREA: u32 = 2u;
const OP_BARYCENTRIC_SAMPLE: u32 = 3u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation tag selecting which twinned function this thread evaluates.
    op: u32,
    pad_op0: u32,
    pad_op1: u32,
    pad_op2: u32,
    // Bind-pose position (skin position) or bind-pose normal (skin normal); a
    // pad lane follows.
    skin_vec: vec3<f32>,
    pad_sv: f32,
    // Per-influence blend weights for the skin ops.
    weights: vec4<f32>,
    // Four bone transforms, four vec4 per bone (basis_x, basis_y, basis_z,
    // translation; each xyz with a trailing pad lane), already resolved on the
    // host from the bone palette and the per-vertex influence indices.
    bones: array<vec4<f32>, 16>,
    // Triangle corner a position; a pad lane follows.
    tri_a: vec3<f32>,
    pad_ta: f32,
    // Triangle corner b position; a pad lane follows.
    tri_b: vec3<f32>,
    pad_tb: f32,
    // Triangle corner c position; a pad lane follows.
    tri_c: vec3<f32>,
    pad_tc: f32,
    // Triangle corner a shading normal; a pad lane follows.
    nrm_a: vec3<f32>,
    pad_na: f32,
    // Triangle corner b shading normal; a pad lane follows.
    nrm_b: vec3<f32>,
    pad_nb: f32,
    // Triangle corner c shading normal; a pad lane follows.
    nrm_c: vec3<f32>,
    pad_nc: f32,
    // Barycentric weight triple (w_a, w_b, w_c) for the sample op; a pad lane
    // follows.
    bary: vec3<f32>,
    pad_bary: f32,
}

struct Result {
    // Operation tag echoed back so the host decodes the right union variant.
    op: u32,
    pad_op0: u32,
    pad_op1: u32,
    pad_op2: u32,
    // Skinned or sampled position; a pad lane follows.
    position: vec3<f32>,
    pad_p: f32,
    // Skinned or sampled normal; a pad lane follows.
    normal: vec3<f32>,
    pad_n: f32,
    // Triangle area, with three pad lanes filling the 16-byte slot.
    area: f32,
    pad_a0: f32,
    pad_a1: f32,
    pad_a2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Unit vector along `v`, or the zero vector when `v` is (numerically) zero, so
// normalization never yields NaN. Mirrors the reference `Vec3::normalize_or_zero`
// using the squared-length guard `EPS_LEN_SQ`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let op = q.op;

    var out_pos: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var out_nrm: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var out_area: f32 = 0.0;

    if (op == OP_SKIN_POSITION || op == OP_SKIN_NORMAL) {
        // Linear-blend skinning: accumulate each influence's transformed source
        // scaled by its weight, skipping near-zero weights exactly as the
        // reference does. Positions apply the translation column; normals apply
        // the basis only.
        let src = q.skin_vec;
        var acc: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let w = q.weights[i];
            if (abs(w) < EPS) {
                continue;
            }
            let base = i * 4u;
            let basis_x = q.bones[base].xyz;
            let basis_y = q.bones[base + 1u].xyz;
            let basis_z = q.bones[base + 2u].xyz;
            let translation = q.bones[base + 3u].xyz;
            var mapped = basis_x * src.x + basis_y * src.y + basis_z * src.z;
            if (op == OP_SKIN_POSITION) {
                mapped = mapped + translation;
            }
            acc = acc + mapped * w;
        }
        if (op == OP_SKIN_POSITION) {
            out_pos = acc;
        } else {
            out_nrm = normalize_or_zero(acc);
        }
    } else if (op == OP_TRIANGLE_AREA) {
        // area = 0.5 * |(b - a) x (c - a)|; one cross and one sqrt.
        let e1 = q.tri_b - q.tri_a;
        let e2 = q.tri_c - q.tri_a;
        let cr = cross(e1, e2);
        out_area = sqrt(dot(cr, cr)) * 0.5;
    } else {
        // BarycentricSample: blend the corner positions, and blend / renormalize
        // the corner shading normals, falling back to the geometric normal when
        // the blended normal is (numerically) zero.
        out_pos = q.tri_a * q.bary.x + q.tri_b * q.bary.y + q.tri_c * q.bary.z;
        let blended = q.nrm_a * q.bary.x + q.nrm_b * q.bary.y + q.nrm_c * q.bary.z;
        let blended_len_sq = dot(blended, blended);
        if (blended_len_sq > EPS_LEN_SQ) {
            out_nrm = blended * (1.0 / sqrt(blended_len_sq));
        } else {
            let e1 = q.tri_b - q.tri_a;
            let e2 = q.tri_c - q.tri_a;
            out_nrm = normalize_or_zero(cross(e1, e2));
        }
    }

    var out: Result;
    out.op = op;
    out.pad_op0 = 0u;
    out.pad_op1 = 0u;
    out.pad_op2 = 0u;
    out.position = out_pos;
    out.pad_p = 0.0;
    out.normal = out_nrm;
    out.pad_n = 0.0;
    out.area = out_area;
    out.pad_a0 = 0.0;
    out.pad_a1 = 0.0;
    out.pad_a2 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MESH_EMISSION_WGSL`].
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
/// Every `vec3` lane carries a trailing pad word and the four-bone block is a
/// `16`-entry `vec4` array, so each member stays `16`-byte aligned on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Operation tag selecting the twinned function.
    op: u32,
    /// Padding word after `op`.
    pad_op0: u32,
    /// Padding word after `op`.
    pad_op1: u32,
    /// Padding word after `op`.
    pad_op2: u32,
    /// Bind-pose position or normal for the skin ops.
    skin_vec: [f32; 3],
    /// Pad lane after `skin_vec`.
    pad_sv: f32,
    /// Per-influence blend weights.
    weights: [f32; 4],
    /// Four bone transforms, four `vec4` columns per bone (`basis_x`, `basis_y`,
    /// `basis_z`, `translation`; each `xyz` with a trailing pad lane).
    bones: [[f32; 4]; 16],
    /// Triangle corner `a` position.
    tri_a: [f32; 3],
    /// Pad lane after `tri_a`.
    pad_ta: f32,
    /// Triangle corner `b` position.
    tri_b: [f32; 3],
    /// Pad lane after `tri_b`.
    pad_tb: f32,
    /// Triangle corner `c` position.
    tri_c: [f32; 3],
    /// Pad lane after `tri_c`.
    pad_tc: f32,
    /// Triangle corner `a` shading normal.
    nrm_a: [f32; 3],
    /// Pad lane after `nrm_a`.
    pad_na: f32,
    /// Triangle corner `b` shading normal.
    nrm_b: [f32; 3],
    /// Pad lane after `nrm_b`.
    pad_nb: f32,
    /// Triangle corner `c` shading normal.
    nrm_c: [f32; 3],
    /// Pad lane after `nrm_c`.
    pad_nc: f32,
    /// Barycentric weight triple for the sample op.
    bary: [f32; 3],
    /// Pad lane after `bary`.
    pad_bary: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Operation tag echoed back for decoding.
    op: u32,
    /// Padding word after `op`.
    pad_op0: u32,
    /// Padding word after `op`.
    pad_op1: u32,
    /// Padding word after `op`.
    pad_op2: u32,
    /// Skinned or sampled position.
    position: [f32; 3],
    /// Pad lane after `position`.
    pad_p: f32,
    /// Skinned or sampled normal.
    normal: [f32; 3],
    /// Pad lane after `normal`.
    pad_n: f32,
    /// Triangle area.
    area: f32,
    /// Padding lane.
    pad_a0: f32,
    /// Padding lane.
    pad_a1: f32,
    /// Padding lane.
    pad_a2: f32,
}

/// One query for the mesh-emission twin: a tagged union selecting which of the
/// twinned reference functions this element evaluates.
///
/// The skinning variants carry the fixed-width, host-resolved per-influence
/// weights and [`BoneTransform`] palette entries (`MAX_BONE_INFLUENCES` of
/// each); an out-of-range influence in the reference is dropped on the host by
/// zeroing that weight. The triangle variants carry the already-selected,
/// host-resolved triangle corners (and, for the sample, the folded barycentric
/// weights).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeshEmissionQuery {
    /// Linear-blend skinning of a bind-pose position, twinning
    /// [`skin_position`](prism_render_architecture::particle::mesh_emission::skin_position).
    SkinPosition {
        /// Bind-pose position.
        position: [f32; 3],
        /// Per-influence blend weights.
        weights: [f32; MAX_BONE_INFLUENCES],
        /// Per-influence bone transforms, host-resolved from the palette.
        bones: [BoneTransform; MAX_BONE_INFLUENCES],
    },
    /// Linear-blend skinning of a bind-pose normal, twinning
    /// [`skin_normal`](prism_render_architecture::particle::mesh_emission::skin_normal).
    SkinNormal {
        /// Bind-pose shading normal.
        normal: [f32; 3],
        /// Per-influence blend weights.
        weights: [f32; MAX_BONE_INFLUENCES],
        /// Per-influence bone transforms, host-resolved from the palette.
        bones: [BoneTransform; MAX_BONE_INFLUENCES],
    },
    /// Surface area of a triangle, twinning
    /// [`MeshTriangle::area`](prism_render_architecture::particle::mesh_emission::MeshTriangle::area).
    TriangleArea {
        /// Triangle corner positions in `a`, `b`, `c` order.
        positions: [[f32; 3]; 3],
    },
    /// Barycentric sample of a surface point and shading normal, twinning
    /// [`MeshTriangle::position_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::position_at)
    /// and
    /// [`MeshTriangle::normal_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::normal_at).
    BarycentricSample {
        /// Triangle corner positions in `a`, `b`, `c` order.
        positions: [[f32; 3]; 3],
        /// Triangle corner shading normals in `a`, `b`, `c` order.
        normals: [[f32; 3]; 3],
        /// Barycentric weight triple `(w_a, w_b, w_c)`.
        barycentric: [f32; 3],
    },
}

/// One resolved answer for a single [`MeshEmissionQuery`], mirroring the value
/// the reference reports for the corresponding function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeshEmissionResult {
    /// Skinned position, matching
    /// [`skin_position`](prism_render_architecture::particle::mesh_emission::skin_position).
    SkinPosition {
        /// Skinned position.
        position: [f32; 3],
    },
    /// Skinned, renormalized normal, matching
    /// [`skin_normal`](prism_render_architecture::particle::mesh_emission::skin_normal).
    SkinNormal {
        /// Skinned, renormalized normal (zero when the blend is numerically
        /// zero).
        normal: [f32; 3],
    },
    /// Surface area, matching
    /// [`MeshTriangle::area`](prism_render_architecture::particle::mesh_emission::MeshTriangle::area).
    TriangleArea {
        /// Triangle surface area.
        area: f32,
    },
    /// Sampled position and shading normal, matching
    /// [`MeshTriangle::position_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::position_at)
    /// and
    /// [`MeshTriangle::normal_at`](prism_render_architecture::particle::mesh_emission::MeshTriangle::normal_at).
    BarycentricSample {
        /// Interpolated position.
        position: [f32; 3],
        /// Renormalized shading normal with the geometric-normal fallback.
        normal: [f32; 3],
    },
}

/// Packs one [`BoneTransform`] into the four padded device columns.
fn pack_bone(b: &BoneTransform) -> [[f32; 4]; 4] {
    [
        [b.basis_x.x, b.basis_x.y, b.basis_x.z, 0.0],
        [b.basis_y.x, b.basis_y.y, b.basis_y.z, 0.0],
        [b.basis_z.x, b.basis_z.y, b.basis_z.z, 0.0],
        [b.translation.x, b.translation.y, b.translation.z, 0.0],
    ]
}

/// Packs the `MAX_BONE_INFLUENCES` host-resolved bones into the `16`-entry
/// `vec4` block, four columns per bone.
fn pack_bones(bones: &[BoneTransform; MAX_BONE_INFLUENCES]) -> [[f32; 4]; 16] {
    let mut out = [[0.0_f32; 4]; 16];
    for (i, bone) in bones.iter().enumerate() {
        let cols = pack_bone(bone);
        let base = i * 4;
        out[base] = cols[0];
        out[base + 1] = cols[1];
        out[base + 2] = cols[2];
        out[base + 3] = cols[3];
    }
    out
}

/// Encodes one [`MeshEmissionQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &MeshEmissionQuery) -> GpuQuery {
    let mut slot = GpuQuery::zeroed();
    match q {
        MeshEmissionQuery::SkinPosition {
            position,
            weights,
            bones,
        } => {
            slot.op = OP_SKIN_POSITION;
            slot.skin_vec = *position;
            slot.weights = *weights;
            slot.bones = pack_bones(bones);
        }
        MeshEmissionQuery::SkinNormal {
            normal,
            weights,
            bones,
        } => {
            slot.op = OP_SKIN_NORMAL;
            slot.skin_vec = *normal;
            slot.weights = *weights;
            slot.bones = pack_bones(bones);
        }
        MeshEmissionQuery::TriangleArea { positions } => {
            slot.op = OP_TRIANGLE_AREA;
            slot.tri_a = positions[0];
            slot.tri_b = positions[1];
            slot.tri_c = positions[2];
        }
        MeshEmissionQuery::BarycentricSample {
            positions,
            normals,
            barycentric,
        } => {
            slot.op = OP_BARYCENTRIC_SAMPLE;
            slot.tri_a = positions[0];
            slot.tri_b = positions[1];
            slot.tri_c = positions[2];
            slot.nrm_a = normals[0];
            slot.nrm_b = normals[1];
            slot.nrm_c = normals[2];
            slot.bary = *barycentric;
        }
    }
    slot
}

/// Decodes one packed [`GpuResult`] into the public [`MeshEmissionResult`],
/// selecting the union variant from the echoed operation tag.
fn decode_result(raw: &GpuResult) -> MeshEmissionResult {
    match raw.op {
        OP_SKIN_POSITION => MeshEmissionResult::SkinPosition {
            position: raw.position,
        },
        OP_SKIN_NORMAL => MeshEmissionResult::SkinNormal { normal: raw.normal },
        OP_TRIANGLE_AREA => MeshEmissionResult::TriangleArea { area: raw.area },
        _ => MeshEmissionResult::BarycentricSample {
            position: raw.position,
            normal: raw.normal,
        },
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

/// A compiled, reusable mesh-emission compute pipeline, twinning the `CPU`
/// golden
/// [`mesh_emission`](prism_render_architecture::particle::mesh_emission).
pub struct GpuMeshEmission {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMeshEmission {
    /// Compiles the mesh-emission kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMeshEmission {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_emission"),
            source: ShaderSource::Wgsl(MESH_EMISSION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_emission_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_emission_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_emission_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMeshEmission {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one [`MeshEmissionResult`]
    /// per input, in order.
    ///
    /// The skinned positions, skinned normals, areas and samples match the
    /// reference to within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MeshEmissionQuery],
    ) -> Vec<MeshEmissionResult> {
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
            label: Some("prism_volumetric_mesh_emission_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_emission_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_emission_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_emission_bind_group"),
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
            label: Some("prism_volumetric_mesh_emission_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_emission_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_emission_pass"),
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
