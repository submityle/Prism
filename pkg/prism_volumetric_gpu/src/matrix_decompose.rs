//! `wgpu` compute twin of the affine `matrix` / `quaternion` decomposition and
//! composition contract
//! ([`matrix_decompose`](prism_render_architecture::particle::matrix_decompose),
//! particle design §16, §26).
//!
//! The `CPU` golden
//! [`matrix_decompose`](prism_render_architecture::particle::matrix_decompose)
//! owns the small, verifiable linear algebra a particle emitter's world
//! placement needs: the column-major `3x3`
//! ([`Mat3`](prism_render_architecture::particle::matrix_decompose::Mat3)) and
//! `4x4`
//! ([`Mat4`](prism_render_architecture::particle::matrix_decompose::Mat4))
//! operators, the `quaternion` round trip
//! ([`mat3_from_quat`](prism_render_architecture::particle::matrix_decompose::mat3_from_quat)
//! and
//! [`quat_from_mat3`](prism_render_architecture::particle::matrix_decompose::quat_from_mat3)),
//! and the *translate / rotate / scale* split and merge
//! ([`decompose_affine`](prism_render_architecture::particle::matrix_decompose::decompose_affine)
//! and
//! [`compose_trs`](prism_render_architecture::particle::matrix_decompose::compose_trs)).
//! [`GpuMatrixDecompose`] is the on-device twin: one thread solves one query, so
//! a passing real-device parity test is direct evidence the ported kernel
//! computes the same matrices, `quaternion`s and
//! [`Trs`](prism_render_architecture::particle::matrix_decompose::Trs) triples
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query selects one routine by a `u32` tag and the kernel reproduces it
//! branch for branch: the `3x3` matrix-times-vector, matrix-times-matrix,
//! transpose and determinant; the `4x4` affine point transform; the unit
//! `quaternion`-to-rotation polynomial and the trace / `Shepperd` inverse; and
//! the affine decompose / compose round trip (including the negative-determinant
//! reflection fold). The iterative
//! [`polar_decompose`](prism_render_architecture::particle::matrix_decompose::polar_decompose)
//! is a fixed-count Newton loop in the reference and is deliberately *not*
//! twinned here.
//!
//! # No transcendental math
//!
//! Every routine is polynomial plus at most one `sqrt` per branch: the
//! `quaternion` recovery picks the trace / pivot branch with the largest `sqrt`
//! argument, and the scale extraction takes one `sqrt` per basis column via the
//! length helper. The kernel uses no `sin`, `cos`, `tan`, `exp`, `log`, `pow`,
//! no inverse trigonometry and no `smoothstep` or `round`; the `Shepperd`
//! comment mentions `atan2` only for intuition and no such call is emitted.
//!
//! # Correctness model
//!
//! The dispatch tag is an integer classification, so the kernel runs exactly the
//! branch the host requested. The matrix and `quaternion` entries thread through
//! multiplies, adds, one guarded division and at most one `sqrt`, so `CPU` and
//! `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`) on every continuous quantity, and because a `quaternion`
//! and its negation encode the same rotation it compares `quaternion`s under the
//! double-cover "same sign or wholly negated" rule the golden tests use.
//!
//! # Degenerate inputs
//!
//! A near-zero basis column would divide the de-scaled rotation by a near-zero
//! length; the kernel's `normalized_or_axis` guard falls back to the matching
//! identity axis instead, matching the reference and never producing a `NaN`. An
//! empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `sqrt`,
//! `+ - * /` and unsigned index arithmetic — with no optional device feature, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each
//! thread performs a fixed, bounded sequence of arithmetic, so the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::matrix_decompose`；无第三方引擎源码或衍生代码。
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

use prism_render_architecture::particle::matrix_decompose::{Mat3, Mat4, Trs};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` matrix-decomposition kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`matrix_decompose`](prism_render_architecture::particle::matrix_decompose)
/// branch for branch; see the module documentation for the algorithm.
const MATRIX_DECOMPOSE_WGSL: &str = r#"
// Matrix-decomposition twin: one thread per query runs the routine its `tag`
// selects, reproducing the CPU golden `particle::matrix_decompose` branch for
// branch. It uses only the portable core-WGSL subset (abs/sqrt and + - * / plus
// unsigned index math), needs no transcendental call and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12. There is no loop, so
// the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::matrix_decompose；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a basis-column length is treated as degenerate and the
// matching identity axis is used instead. Matches the reference `CMP_EPS`; the
// compare rule used instead of an f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

// Routine tags; the host casts its op enum straight to these codes.
const TAG_MUL_VEC3: u32 = 0u;
const TAG_MUL_MAT3: u32 = 1u;
const TAG_TRANSPOSE: u32 = 2u;
const TAG_DETERMINANT: u32 = 3u;
const TAG_MAT3_FROM_QUAT: u32 = 4u;
const TAG_QUAT_FROM_MAT3: u32 = 5u;
const TAG_MUL_POINT: u32 = 6u;
const TAG_DECOMPOSE_AFFINE: u32 = 7u;
const TAG_COMPOSE_TRS: u32 = 8u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Routine selector plus three pad words to fill the std430 16-byte slot.
    tag: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Primary matrix A as four column vec4; the upper-left 3x3 (xyz lanes of the
    // first three columns) feeds the Mat3 routines, the full 4x4 feeds mul_point.
    ma0: vec4<f32>,
    ma1: vec4<f32>,
    ma2: vec4<f32>,
    ma3: vec4<f32>,
    // Secondary 3x3 matrix B columns (xyz lanes) for the matrix-times-matrix op.
    mb0: vec4<f32>,
    mb1: vec4<f32>,
    mb2: vec4<f32>,
    mb3: vec4<f32>,
    // Quaternion [x, y, z, w] for mat3_from_quat.
    quat: vec4<f32>,
    // Operand vector / point (xyz; w pad) for mul_vec3 and mul_point.
    vec: vec4<f32>,
    // Trs translation (xyz), rotation quaternion, and scale (xyz) for compose_trs.
    trs_t: vec4<f32>,
    trs_r: vec4<f32>,
    trs_s: vec4<f32>,
}

struct Result {
    // Matrix output as four column vec4: a 3x3 result uses the xyz lanes of the
    // first three columns, a 4x4 result uses all four columns.
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    m3: vec4<f32>,
    // Quaternion output [x, y, z, w].
    quat: vec4<f32>,
    // Vector / scalar output: mul_vec3 and mul_point use xyz; determinant uses x.
    vec: vec4<f32>,
    // Trs output for decompose_affine: translation, rotation, scale.
    trs_t: vec4<f32>,
    trs_r: vec4<f32>,
    trs_s: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// A column-major 3x3 held as three column 3-vectors; `cols[c][r]` is `c<idx>.<r>`.
struct Mat3Cols {
    c0: vec3<f32>,
    c1: vec3<f32>,
    c2: vec3<f32>,
}

// The matrix-times-vector product; mirrors the reference `Mat3::mul_vec3`.
fn mul_vec3(m: Mat3Cols, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        m.c0.x * v.x + m.c1.x * v.y + m.c2.x * v.z,
        m.c0.y * v.x + m.c1.y * v.y + m.c2.y * v.z,
        m.c0.z * v.x + m.c1.z * v.y + m.c2.z * v.z,
    );
}

// The scalar determinant via cofactor expansion along the first row; mirrors the
// reference `Mat3::determinant`.
fn determinant3(m: Mat3Cols) -> f32 {
    let m00 = m.c0.x;
    let m01 = m.c1.x;
    let m02 = m.c2.x;
    let m10 = m.c0.y;
    let m11 = m.c1.y;
    let m12 = m.c2.y;
    let m20 = m.c0.z;
    let m21 = m.c1.z;
    let m22 = m.c2.z;
    return m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20)
        + m02 * (m10 * m21 - m11 * m20);
}

// The squared length of a 3-vector; mirrors the reference `length_squared3`.
fn length_squared3(v: vec3<f32>) -> f32 {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// The Euclidean length of a 3-vector; mirrors the reference `length3`.
fn length3(v: vec3<f32>) -> f32 {
    return sqrt(length_squared3(v));
}

// Divides `v` by `len`, falling back to `axis` when `len` is degenerate so the
// result stays a valid unit vector; mirrors the reference `normalized_or_axis`.
fn normalized_or_axis(v: vec3<f32>, len: f32, axis: vec3<f32>) -> vec3<f32> {
    if (abs(len) < CMP_EPS) {
        return axis;
    }
    let inv = 1.0 / len;
    return vec3<f32>(v.x * inv, v.y * inv, v.z * inv);
}

// Builds a column-major rotation matrix from a unit quaternion [x, y, z, w]
// using the standard polynomial identity; mirrors the reference `mat3_from_quat`.
fn mat3_from_quat(q: vec4<f32>) -> Mat3Cols {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    var m: Mat3Cols;
    m.c0 = vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy));
    m.c1 = vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx));
    m.c2 = vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy));
    return m;
}

// Recovers a unit quaternion [x, y, z, w] from a column-major rotation matrix via
// the trace / Shepperd method: the branch with the largest pivot is chosen so the
// `sqrt` argument stays away from zero; mirrors the reference `quat_from_mat3`.
fn quat_from_mat3(m: Mat3Cols) -> vec4<f32> {
    let m00 = m.c0.x;
    let m01 = m.c1.x;
    let m02 = m.c2.x;
    let m10 = m.c0.y;
    let m11 = m.c1.y;
    let m12 = m.c2.y;
    let m20 = m.c0.z;
    let m21 = m.c1.z;
    let m22 = m.c2.z;
    let trace = m00 + m11 + m22;
    if (trace > 0.0) {
        let s = sqrt(trace + 1.0) * 2.0;
        let inv = 1.0 / s;
        return vec4<f32>((m21 - m12) * inv, (m02 - m20) * inv, (m10 - m01) * inv, 0.25 * s);
    } else if (m00 > m11 && m00 > m22) {
        let s = sqrt(1.0 + m00 - m11 - m22) * 2.0;
        let inv = 1.0 / s;
        return vec4<f32>(0.25 * s, (m01 + m10) * inv, (m02 + m20) * inv, (m21 - m12) * inv);
    } else if (m11 > m22) {
        let s = sqrt(1.0 + m11 - m00 - m22) * 2.0;
        let inv = 1.0 / s;
        return vec4<f32>((m01 + m10) * inv, 0.25 * s, (m12 + m21) * inv, (m02 - m20) * inv);
    } else {
        let s = sqrt(1.0 + m22 - m00 - m11) * 2.0;
        let inv = 1.0 / s;
        return vec4<f32>((m02 + m20) * inv, (m12 + m21) * inv, 0.25 * s, (m10 - m01) * inv);
    }
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.m0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.m1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.m2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.m3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.quat = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.vec = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.trs_t = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.trs_r = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.trs_s = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    var a: Mat3Cols;
    a.c0 = q.ma0.xyz;
    a.c1 = q.ma1.xyz;
    a.c2 = q.ma2.xyz;

    if (q.tag == TAG_MUL_VEC3) {
        out.vec = vec4<f32>(mul_vec3(a, q.vec.xyz), 0.0);
    } else if (q.tag == TAG_MUL_MAT3) {
        var b: Mat3Cols;
        b.c0 = q.mb0.xyz;
        b.c1 = q.mb1.xyz;
        b.c2 = q.mb2.xyz;
        // Column k of the product is `a` applied to column k of `b`.
        out.m0 = vec4<f32>(mul_vec3(a, b.c0), 0.0);
        out.m1 = vec4<f32>(mul_vec3(a, b.c1), 0.0);
        out.m2 = vec4<f32>(mul_vec3(a, b.c2), 0.0);
    } else if (q.tag == TAG_TRANSPOSE) {
        out.m0 = vec4<f32>(a.c0.x, a.c1.x, a.c2.x, 0.0);
        out.m1 = vec4<f32>(a.c0.y, a.c1.y, a.c2.y, 0.0);
        out.m2 = vec4<f32>(a.c0.z, a.c1.z, a.c2.z, 0.0);
    } else if (q.tag == TAG_DETERMINANT) {
        out.vec = vec4<f32>(determinant3(a), 0.0, 0.0, 0.0);
    } else if (q.tag == TAG_MAT3_FROM_QUAT) {
        let r = mat3_from_quat(q.quat);
        out.m0 = vec4<f32>(r.c0, 0.0);
        out.m1 = vec4<f32>(r.c1, 0.0);
        out.m2 = vec4<f32>(r.c2, 0.0);
    } else if (q.tag == TAG_QUAT_FROM_MAT3) {
        out.quat = quat_from_mat3(a);
    } else if (q.tag == TAG_MUL_POINT) {
        // Affine point transform (implicit w = 1): 3x3 block plus translation.
        out.vec = vec4<f32>(
            q.ma0.x * q.vec.x + q.ma1.x * q.vec.y + q.ma2.x * q.vec.z + q.ma3.x,
            q.ma0.y * q.vec.x + q.ma1.y * q.vec.y + q.ma2.y * q.vec.z + q.ma3.y,
            q.ma0.z * q.vec.x + q.ma1.z * q.vec.y + q.ma2.z * q.vec.z + q.ma3.z,
            0.0,
        );
    } else if (q.tag == TAG_DECOMPOSE_AFFINE) {
        let translation = q.ma3.xyz;
        let c0 = q.ma0.xyz;
        let c1 = q.ma1.xyz;
        let c2 = q.ma2.xyz;
        var sx = length3(c0);
        let sy = length3(c1);
        let sz = length3(c2);
        // A negative determinant is a reflection; fold it into the first axis so
        // the recovered rotation stays a proper (right-handed) rotation.
        if (determinant3(a) < 0.0) {
            sx = -sx;
        }
        var r: Mat3Cols;
        r.c0 = normalized_or_axis(c0, sx, vec3<f32>(1.0, 0.0, 0.0));
        r.c1 = normalized_or_axis(c1, sy, vec3<f32>(0.0, 1.0, 0.0));
        r.c2 = normalized_or_axis(c2, sz, vec3<f32>(0.0, 0.0, 1.0));
        out.trs_t = vec4<f32>(translation, 0.0);
        out.trs_r = quat_from_mat3(r);
        out.trs_s = vec4<f32>(sx, sy, sz, 0.0);
    } else if (q.tag == TAG_COMPOSE_TRS) {
        let r = mat3_from_quat(q.trs_r);
        let sx = q.trs_s.x;
        let sy = q.trs_s.y;
        let sz = q.trs_s.z;
        // Upper-left block is rotation * diag(scale); the fourth column is the
        // translation.
        out.m0 = vec4<f32>(r.c0.x * sx, r.c0.y * sx, r.c0.z * sx, 0.0);
        out.m1 = vec4<f32>(r.c1.x * sy, r.c1.y * sy, r.c1.z * sy, 0.0);
        out.m2 = vec4<f32>(r.c2.x * sz, r.c2.y * sz, r.c2.z * sz, 0.0);
        out.m3 = vec4<f32>(q.trs_t.x, q.trs_t.y, q.trs_t.z, 1.0);
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MATRIX_DECOMPOSE_WGSL`].
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
/// Each matrix column is a padded `vec4` so every slot stays `16`-byte aligned on
/// device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Routine selector matching the `WGSL` `TAG_*` codes.
    tag: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Matrix `A` column `0`.
    ma0: [f32; 4],
    /// Matrix `A` column `1`.
    ma1: [f32; 4],
    /// Matrix `A` column `2`.
    ma2: [f32; 4],
    /// Matrix `A` column `3`.
    ma3: [f32; 4],
    /// Matrix `B` column `0`.
    mb0: [f32; 4],
    /// Matrix `B` column `1`.
    mb1: [f32; 4],
    /// Matrix `B` column `2`.
    mb2: [f32; 4],
    /// Matrix `B` column `3`.
    mb3: [f32; 4],
    /// Quaternion `[x, y, z, w]`.
    quat: [f32; 4],
    /// Operand vector or point (`xyz`; `w` pad).
    vec: [f32; 4],
    /// `Trs` translation (`xyz`; `w` pad).
    trs_t: [f32; 4],
    /// `Trs` rotation `quaternion`.
    trs_r: [f32; 4],
    /// `Trs` scale (`xyz`; `w` pad).
    trs_s: [f32; 4],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Matrix output column `0`.
    m0: [f32; 4],
    /// Matrix output column `1`.
    m1: [f32; 4],
    /// Matrix output column `2`.
    m2: [f32; 4],
    /// Matrix output column `3`.
    m3: [f32; 4],
    /// Quaternion output `[x, y, z, w]`.
    quat: [f32; 4],
    /// Vector output (`xyz`), or a scalar in `x` for the determinant op.
    vec: [f32; 4],
    /// `Trs` translation output (`xyz`; `w` pad).
    trs_t: [f32; 4],
    /// `Trs` rotation `quaternion` output.
    trs_r: [f32; 4],
    /// `Trs` scale output (`xyz`; `w` pad).
    trs_s: [f32; 4],
}

/// One query for the matrix-decomposition twin: a tagged union selecting which
/// golden routine to run with its typed inputs.
///
/// Each variant twins exactly one `CPU` golden function. The golden
/// [`Mat3`](prism_render_architecture::particle::matrix_decompose::Mat3),
/// [`Mat4`](prism_render_architecture::particle::matrix_decompose::Mat4) and
/// [`Trs`](prism_render_architecture::particle::matrix_decompose::Trs) contract
/// types are reused directly rather than re-declared.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::matrix_decompose`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MatrixDecomposeQuery {
    /// The `3x3` matrix-times-vector product, twinning
    /// [`Mat3::mul_vec3`](prism_render_architecture::particle::matrix_decompose::Mat3::mul_vec3).
    MulVec3 {
        /// Left matrix.
        matrix: Mat3,
        /// Right vector.
        vector: [f32; 3],
    },
    /// The `3x3` matrix-times-matrix product, twinning
    /// [`Mat3::mul_mat3`](prism_render_architecture::particle::matrix_decompose::Mat3::mul_mat3).
    MulMat3 {
        /// Left matrix.
        lhs: Mat3,
        /// Right matrix.
        rhs: Mat3,
    },
    /// The `3x3` transpose, twinning
    /// [`Mat3::transpose`](prism_render_architecture::particle::matrix_decompose::Mat3::transpose).
    Transpose {
        /// Matrix to transpose.
        matrix: Mat3,
    },
    /// The `3x3` determinant, twinning
    /// [`Mat3::determinant`](prism_render_architecture::particle::matrix_decompose::Mat3::determinant).
    Determinant {
        /// Matrix whose determinant is taken.
        matrix: Mat3,
    },
    /// The `quaternion`-to-rotation polynomial, twinning
    /// [`mat3_from_quat`](prism_render_architecture::particle::matrix_decompose::mat3_from_quat).
    Mat3FromQuat {
        /// Unit `quaternion` `[x, y, z, w]`.
        quat: [f32; 4],
    },
    /// The trace / `Shepperd` `quaternion` recovery, twinning
    /// [`quat_from_mat3`](prism_render_architecture::particle::matrix_decompose::quat_from_mat3).
    QuatFromMat3 {
        /// Rotation matrix to recover a `quaternion` from.
        matrix: Mat3,
    },
    /// The `4x4` affine point transform, twinning
    /// [`Mat4::mul_point`](prism_render_architecture::particle::matrix_decompose::Mat4::mul_point).
    MulPoint {
        /// Affine matrix.
        matrix: Mat4,
        /// Point (implicit `w = 1`).
        point: [f32; 3],
    },
    /// The affine decomposition, twinning
    /// [`decompose_affine`](prism_render_architecture::particle::matrix_decompose::decompose_affine).
    DecomposeAffine {
        /// Affine matrix to decompose.
        matrix: Mat4,
    },
    /// The affine composition, twinning
    /// [`compose_trs`](prism_render_architecture::particle::matrix_decompose::compose_trs).
    ComposeTrs {
        /// `Trs` triple to compose.
        trs: Trs,
    },
}

impl MatrixDecomposeQuery {
    /// Returns the `u32` tag the kernel branches on for this routine.
    #[must_use]
    const fn tag(&self) -> u32 {
        match self {
            MatrixDecomposeQuery::MulVec3 { .. } => 0,
            MatrixDecomposeQuery::MulMat3 { .. } => 1,
            MatrixDecomposeQuery::Transpose { .. } => 2,
            MatrixDecomposeQuery::Determinant { .. } => 3,
            MatrixDecomposeQuery::Mat3FromQuat { .. } => 4,
            MatrixDecomposeQuery::QuatFromMat3 { .. } => 5,
            MatrixDecomposeQuery::MulPoint { .. } => 6,
            MatrixDecomposeQuery::DecomposeAffine { .. } => 7,
            MatrixDecomposeQuery::ComposeTrs { .. } => 8,
        }
    }
}

/// One resolved answer for a single query: a tagged union whose variant matches
/// the routine the corresponding [`MatrixDecomposeQuery`] selected.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::matrix_decompose`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MatrixDecomposeResult {
    /// A `3`-vector result (`mul_vec3` or `mul_point`).
    Vec3([f32; 3]),
    /// A `3x3` matrix result (`mul_mat3`, `transpose` or `mat3_from_quat`).
    Mat3(Mat3),
    /// A scalar result (`determinant`).
    Scalar(f32),
    /// A `quaternion` result (`quat_from_mat3`).
    Quat([f32; 4]),
    /// A `4x4` matrix result (`compose_trs`).
    Mat4(Mat4),
    /// A `Trs` triple result (`decompose_affine`).
    Trs(Trs),
}

/// Packs a `3x3` matrix's three columns into padded `vec4` lanes.
fn mat3_cols(m: &Mat3) -> ([f32; 4], [f32; 4], [f32; 4]) {
    (
        [m.cols[0][0], m.cols[0][1], m.cols[0][2], 0.0],
        [m.cols[1][0], m.cols[1][1], m.cols[1][2], 0.0],
        [m.cols[2][0], m.cols[2][1], m.cols[2][2], 0.0],
    )
}

/// Encodes one [`MatrixDecomposeQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &MatrixDecomposeQuery) -> GpuQuery {
    let mut g = GpuQuery {
        tag: q.tag(),
        pad0: 0,
        pad1: 0,
        pad2: 0,
        ma0: [0.0; 4],
        ma1: [0.0; 4],
        ma2: [0.0; 4],
        ma3: [0.0; 4],
        mb0: [0.0; 4],
        mb1: [0.0; 4],
        mb2: [0.0; 4],
        mb3: [0.0; 4],
        quat: [0.0; 4],
        vec: [0.0; 4],
        trs_t: [0.0; 4],
        trs_r: [0.0; 4],
        trs_s: [0.0; 4],
    };
    match q {
        MatrixDecomposeQuery::MulVec3 { matrix, vector } => {
            (g.ma0, g.ma1, g.ma2) = mat3_cols(matrix);
            g.vec = [vector[0], vector[1], vector[2], 0.0];
        }
        MatrixDecomposeQuery::MulMat3 { lhs, rhs } => {
            (g.ma0, g.ma1, g.ma2) = mat3_cols(lhs);
            (g.mb0, g.mb1, g.mb2) = mat3_cols(rhs);
        }
        MatrixDecomposeQuery::Transpose { matrix }
        | MatrixDecomposeQuery::Determinant { matrix }
        | MatrixDecomposeQuery::QuatFromMat3 { matrix } => {
            (g.ma0, g.ma1, g.ma2) = mat3_cols(matrix);
        }
        MatrixDecomposeQuery::Mat3FromQuat { quat } => {
            g.quat = *quat;
        }
        MatrixDecomposeQuery::MulPoint { matrix, point } => {
            g.ma0 = matrix.cols[0];
            g.ma1 = matrix.cols[1];
            g.ma2 = matrix.cols[2];
            g.ma3 = matrix.cols[3];
            g.vec = [point[0], point[1], point[2], 0.0];
        }
        MatrixDecomposeQuery::DecomposeAffine { matrix } => {
            g.ma0 = matrix.cols[0];
            g.ma1 = matrix.cols[1];
            g.ma2 = matrix.cols[2];
            g.ma3 = matrix.cols[3];
        }
        MatrixDecomposeQuery::ComposeTrs { trs } => {
            g.trs_t = [
                trs.translation[0],
                trs.translation[1],
                trs.translation[2],
                0.0,
            ];
            g.trs_r = trs.rotation;
            g.trs_s = [trs.scale[0], trs.scale[1], trs.scale[2], 0.0];
        }
    }
    g
}

/// Rebuilds a `3x3` matrix from the first three padded result columns.
fn decode_mat3(raw: &GpuResult) -> Mat3 {
    Mat3::from_cols(
        [raw.m0[0], raw.m0[1], raw.m0[2]],
        [raw.m1[0], raw.m1[1], raw.m1[2]],
        [raw.m2[0], raw.m2[1], raw.m2[2]],
    )
}

/// Decodes one packed [`GpuResult`] into the public [`MatrixDecomposeResult`],
/// selecting the variant from the query's routine.
fn decode_result(q: &MatrixDecomposeQuery, raw: &GpuResult) -> MatrixDecomposeResult {
    match q {
        MatrixDecomposeQuery::MulVec3 { .. } | MatrixDecomposeQuery::MulPoint { .. } => {
            MatrixDecomposeResult::Vec3([raw.vec[0], raw.vec[1], raw.vec[2]])
        }
        MatrixDecomposeQuery::MulMat3 { .. }
        | MatrixDecomposeQuery::Transpose { .. }
        | MatrixDecomposeQuery::Mat3FromQuat { .. } => {
            MatrixDecomposeResult::Mat3(decode_mat3(raw))
        }
        MatrixDecomposeQuery::Determinant { .. } => MatrixDecomposeResult::Scalar(raw.vec[0]),
        MatrixDecomposeQuery::QuatFromMat3 { .. } => MatrixDecomposeResult::Quat(raw.quat),
        MatrixDecomposeQuery::DecomposeAffine { .. } => MatrixDecomposeResult::Trs(Trs {
            translation: [raw.trs_t[0], raw.trs_t[1], raw.trs_t[2]],
            rotation: raw.trs_r,
            scale: [raw.trs_s[0], raw.trs_s[1], raw.trs_s[2]],
        }),
        MatrixDecomposeQuery::ComposeTrs { .. } => MatrixDecomposeResult::Mat4(Mat4 {
            cols: [raw.m0, raw.m1, raw.m2, raw.m3],
        }),
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

/// A compiled, reusable matrix-decomposition compute pipeline, twinning the
/// `CPU` golden
/// [`matrix_decompose`](prism_render_architecture::particle::matrix_decompose).
pub struct GpuMatrixDecompose {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMatrixDecompose {
    /// Compiles the matrix-decomposition kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMatrixDecompose {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_matrix_decompose"),
            source: ShaderSource::Wgsl(MATRIX_DECOMPOSE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_matrix_decompose_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_matrix_decompose_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_matrix_decompose_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMatrixDecompose {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`MatrixDecomposeResult`]
    /// per input, in order.
    ///
    /// The result variant matches the routine each query selected, matching the
    /// reference to within the tolerance documented on this module (a
    /// `quaternion` under the double-cover rule). An empty `queries` batch returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MatrixDecomposeQuery],
    ) -> Vec<MatrixDecomposeResult> {
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
            label: Some("prism_volumetric_matrix_decompose_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_matrix_decompose_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_matrix_decompose_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_matrix_decompose_bind_group"),
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
            label: Some("prism_volumetric_matrix_decompose_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_matrix_decompose_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_matrix_decompose_pass"),
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(q, r)| decode_result(q, r))
            .collect()
    }
}
