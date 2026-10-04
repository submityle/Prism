//! CPU/GPU math consistency contract (§24.1): the **shader mirror** layer.
//!
//! The same geometry/culling/transform math runs on the CPU (spatial queries,
//! broad-phase, CPU culling) and on the GPU (compute culling, GPU-driven
//! rendering, skinning). When the two disagree the result is shimmering,
//! wrongly culled geometry, or physics/render misalignment. This module owns
//! the **portable, device-free half** of the contract:
//!
//! 1. **Byte-layout contract** — functions that pack [`Mat4`], [`Mat3`],
//!    [`Vec4`], [`Vec3`], and [`Quat`] into the exact little-endian,
//!    column-major, `std140`-aligned bytes a WGSL uniform/storage buffer
//!    expects. These are the single source of truth a driver uploads; a shader
//!    reads them back with the matching layout. Layout sizes are exported as
//!    constants so a consumer can assert its buffer strides.
//! 2. **Mirrored CPU reference ops** — CPU functions written with the *same
//!    algorithm and constants* as their WGSL twin, so a GPU readback can be
//!    diffed against the CPU value within tolerance. [`quat_rotate_vec3`] is
//!    the first such op; it is parity-checked against [`Quat::mul_vec3`].
//! 3. **Single-sourced WGSL fragments** — the authoritative shader text lives
//!    here as `&'static str` constants ([`WGSL_QUAT_ROTATE`]) so the GPU side
//!    cannot drift from the CPU reference silently; a consumer embeds the
//!    constant verbatim.
//!
//! The real WGSL *dispatch*, GPU buffer binding, and the CI round-trip that
//! reads results back from a device live in the consuming render/driver crate
//! (`prism_render_driver` RHI; see §24.10 honest boundary). This module never
//! touches a device. It is `no_std`; packing returns fixed-size arrays with no
//! allocation.

use crate::mat::{Mat3, Mat4};
use crate::quat::Quat;
use crate::vec::{Vec3, Vec4};

/// Bytes a `std140`/`std430` column-major `mat4x4<f32>` occupies: four 16-byte
/// columns.
pub const MAT4_STD140_SIZE: usize = 64;
/// Bytes a `std140` `mat3x3<f32>` occupies: three columns, each a `vec3`
/// padded to 16 bytes.
pub const MAT3_STD140_SIZE: usize = 48;
/// Bytes a `vec4<f32>` occupies.
pub const VEC4_SIZE: usize = 16;
/// Bytes a standalone `std140` `vec3<f32>` occupies (12 used + 4 pad): a `vec3`
/// has 16-byte alignment, so it reserves a full 16 bytes.
pub const VEC3_STD140_SIZE: usize = 16;
/// Bytes a quaternion (`vec4<f32>` as `xyzw`) occupies.
pub const QUAT_SIZE: usize = 16;

/// The clip-space depth range Prism mirrors on both sides: wgpu/WebGPU style
/// `z in [0, 1]` (not OpenGL's `[-1, 1]`). Projection matrices and any manual
/// depth math must honor this on CPU and GPU alike.
pub const NDC_DEPTH_RANGE: (f32, f32) = (0.0, 1.0);

/// Pack a [`Vec4`] as four little-endian `f32`s (16 bytes).
#[inline]
pub fn pack_vec4(v: Vec4) -> [u8; VEC4_SIZE] {
    let mut out = [0u8; VEC4_SIZE];
    write_f32s(&[v.x, v.y, v.z, v.w], &mut out);
    out
}

/// Pack a [`Vec3`] with `std140` padding: `xyz` then 4 pad bytes (16 total).
#[inline]
pub fn pack_vec3_std140(v: Vec3) -> [u8; VEC3_STD140_SIZE] {
    let mut out = [0u8; VEC3_STD140_SIZE];
    write_f32s(&[v.x, v.y, v.z], &mut out[..12]);
    out
}

/// Pack a [`Quat`] as `xyzw` little-endian (16 bytes), matching a WGSL
/// `vec4<f32>` with the scalar in `.w`.
#[inline]
pub fn pack_quat(q: Quat) -> [u8; QUAT_SIZE] {
    let mut out = [0u8; QUAT_SIZE];
    write_f32s(&[q.x, q.y, q.z, q.w], &mut out);
    out
}

/// Pack a [`Mat4`] **column-major** (WGSL/`std140` convention): columns
/// `x_axis, y_axis, z_axis, w_axis`, each a 16-byte `vec4` (64 bytes total).
#[inline]
pub fn pack_mat4(m: Mat4) -> [u8; MAT4_STD140_SIZE] {
    let mut out = [0u8; MAT4_STD140_SIZE];
    for (col, chunk) in [m.x_axis, m.y_axis, m.z_axis, m.w_axis]
        .iter()
        .zip(out.chunks_exact_mut(16))
    {
        write_f32s(&[col.x, col.y, col.z, col.w], chunk);
    }
    out
}

/// Pack a [`Mat3`] in `std140`: three columns, each a `vec3` padded to 16 bytes
/// (48 bytes total). This matches how a WGSL `mat3x3<f32>` is laid out in a
/// uniform block.
#[inline]
pub fn pack_mat3_std140(m: Mat3) -> [u8; MAT3_STD140_SIZE] {
    let mut out = [0u8; MAT3_STD140_SIZE];
    for (col, chunk) in [m.x_axis, m.y_axis, m.z_axis]
        .iter()
        .zip(out.chunks_exact_mut(16))
    {
        write_f32s(&[col.x, col.y, col.z], &mut chunk[..12]);
    }
    out
}

/// Rotate `v` by unit quaternion `q`, written in the exact form the mirrored
/// WGSL fragment [`WGSL_QUAT_ROTATE`] uses:
///
/// ```text
/// t = 2 * cross(q.xyz, v)
/// v' = v + q.w * t + cross(q.xyz, t)
/// ```
///
/// This is algebraically equivalent to `q * v * q^-1` for a unit quaternion and
/// to [`Quat::mul_vec3`]; keeping an explicit copy here guarantees the CPU
/// reference and the shader share one algorithm and operand order, so a GPU
/// readback can be diffed against this value.
#[inline]
pub fn quat_rotate_vec3(q: Quat, v: Vec3) -> Vec3 {
    let u = Vec3::new(q.x, q.y, q.z);
    let t = cross(u, v) * 2.0;
    v + t * q.w + cross(u, t)
}

/// The authoritative WGSL source for [`quat_rotate_vec3`]. A consumer embeds
/// this verbatim so the GPU twin cannot drift from the CPU reference.
pub const WGSL_QUAT_ROTATE: &str = "\
fn prism_quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {\n\
    let t = 2.0 * cross(q.xyz, v);\n\
    return v + q.w * t + cross(q.xyz, t);\n\
}\n";

#[inline]
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

/// Write `vals` little-endian into `out` (`out.len() == 4 * vals.len()`).
#[inline]
fn write_f32s(vals: &[f32], out: &mut [u8]) {
    for (v, chunk) in vals.iter().zip(out.chunks_exact_mut(4)) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}

/// The authoritative WGSL source for the right-handed projection-matrix
/// builders, mirroring [`crate::projection::perspective_rh`],
/// [`crate::projection::perspective_reverse_z_rh`], and
/// [`crate::projection::orthographic_rh`] **term for term**.
///
/// A consumer embeds this verbatim so a GPU-built projection matrix cannot
/// drift from the CPU constructors: the formulas, operand order, and the
/// column-major `mat4x4<f32>` layout are identical to the Rust source. The
/// clip-space depth range is [`NDC_DEPTH_RANGE`] (`z in [0, 1]`), matching the
/// reverse-Z and `[0, 1]` perspective constructors used across Prism.
///
/// Each function returns a `mat4x4<f32>` whose four constructor arguments are
/// the matrix **columns** `x_axis, y_axis, z_axis, w_axis`, exactly as
/// [`crate::mat::Mat4::from_cols`] expects.
pub const WGSL_PROJECTION_RH: &str = "\
fn prism_perspective_rh(fovy: f32, aspect: f32, z_near: f32, z_far: f32) -> mat4x4<f32> {\n\
    let f = 1.0 / tan(fovy * 0.5);\n\
    let r = z_far / (z_near - z_far);\n\
    return mat4x4<f32>(\n\
        vec4<f32>(f / aspect, 0.0, 0.0, 0.0),\n\
        vec4<f32>(0.0, f, 0.0, 0.0),\n\
        vec4<f32>(0.0, 0.0, r, -1.0),\n\
        vec4<f32>(0.0, 0.0, r * z_near, 0.0)\n\
    );\n\
}\n\
\n\
fn prism_perspective_reverse_z_rh(fovy: f32, aspect: f32, z_near: f32, z_far: f32) -> mat4x4<f32> {\n\
    let f = 1.0 / tan(fovy * 0.5);\n\
    let inv = 1.0 / (z_far - z_near);\n\
    return mat4x4<f32>(\n\
        vec4<f32>(f / aspect, 0.0, 0.0, 0.0),\n\
        vec4<f32>(0.0, f, 0.0, 0.0),\n\
        vec4<f32>(0.0, 0.0, z_near * inv, -1.0),\n\
        vec4<f32>(0.0, 0.0, z_far * z_near * inv, 0.0)\n\
    );\n\
}\n\
\n\
fn prism_orthographic_rh(left: f32, right: f32, bottom: f32, top: f32, z_near: f32, z_far: f32) -> mat4x4<f32> {\n\
    let rcp_w = 1.0 / (right - left);\n\
    let rcp_h = 1.0 / (top - bottom);\n\
    let rcp_d = 1.0 / (z_near - z_far);\n\
    return mat4x4<f32>(\n\
        vec4<f32>(2.0 * rcp_w, 0.0, 0.0, 0.0),\n\
        vec4<f32>(0.0, 2.0 * rcp_h, 0.0, 0.0),\n\
        vec4<f32>(0.0, 0.0, rcp_d, 0.0),\n\
        vec4<f32>(-(right + left) * rcp_w, -(top + bottom) * rcp_h, z_near * rcp_d, 1.0)\n\
    );\n\
}\n";

/// The authoritative WGSL source for the right-handed **view** (look-at)
/// matrix builders, mirroring [`crate::projection::look_at_rh`] and
/// [`crate::projection::look_to_rh`] **term for term**.
///
/// A consumer embeds this verbatim so a GPU-built view matrix cannot drift from
/// the CPU constructors: the orthonormal-basis derivation (`normalize`,
/// `cross`), operand order, and the column-major `mat4x4<f32>` layout are
/// identical to the Rust source. The camera forward is `-Z` (right-handed).
///
/// `prism_look_to_rh` takes an explicit (unnormalized) forward `dir`;
/// `prism_look_at_rh` derives it as `focus - eye` and delegates, exactly as
/// the CPU pair does.
pub const WGSL_LOOK_AT_RH: &str = "\
fn prism_look_to_rh(eye: vec3<f32>, dir: vec3<f32>, up: vec3<f32>) -> mat4x4<f32> {\n\
    let f = normalize(dir);\n\
    let s = normalize(cross(f, up));\n\
    let u = cross(s, f);\n\
    return mat4x4<f32>(\n\
        vec4<f32>(s.x, u.x, -f.x, 0.0),\n\
        vec4<f32>(s.y, u.y, -f.y, 0.0),\n\
        vec4<f32>(s.z, u.z, -f.z, 0.0),\n\
        vec4<f32>(-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0)\n\
    );\n\
}\n\
\n\
fn prism_look_at_rh(eye: vec3<f32>, focus: vec3<f32>, up: vec3<f32>) -> mat4x4<f32> {\n\
    return prism_look_to_rh(eye, focus - eye, up);\n\
}\n";
/// The authoritative WGSL source for a 4-influence **dual-quaternion skinning**
/// (`DLB` / dual-quaternion linear blend) vertex transform, mirroring the CPU
/// path [`crate::DualQuat::blend_weighted`] followed by
/// [`crate::DualQuat::transform_point3`] **step for step**: the first
/// non-zero-weight bone fixes the hemisphere pivot, later bones are flipped into
/// it before the weighted accumulation, the sum is renormalized to the unit
/// invariants (`|real| = 1`, `dot(real, dual) = 0`), and the point is finally
/// mapped by `rotation * p + translation`.
///
/// `prism_dq_skin4` depends on `prism_quat_rotate` from [`WGSL_QUAT_ROTATE`];
/// a consumer composes the two fragments (and never re-types them) so the GPU
/// skinning twin cannot drift from the CPU reference. Four bone influences is
/// the standard AAA skinning fan-in; a zero weight contributes nothing (matching
/// the CPU `w == 0` skip), so narrower fan-ins pad with zero weights.
pub const WGSL_DUAL_QUAT_SKIN: &str = "\
fn prism_quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {\n\
    return vec4<f32>(\n\
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,\n\
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,\n\
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,\n\
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z\n\
    );\n\
}\n\
\n\
fn prism_quat_conj(q: vec4<f32>) -> vec4<f32> {\n\
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);\n\
}\n\
\n\
fn prism_dq_skin4(\n\
    r0: vec4<f32>, d0: vec4<f32>,\n\
    r1: vec4<f32>, d1: vec4<f32>,\n\
    r2: vec4<f32>, d2: vec4<f32>,\n\
    r3: vec4<f32>, d3: vec4<f32>,\n\
    w: vec4<f32>, p: vec3<f32>) -> vec3<f32> {\n\
    var reals = array<vec4<f32>, 4>(r0, r1, r2, r3);\n\
    var duals = array<vec4<f32>, 4>(d0, d1, d2, d3);\n\
    var acc_real = vec4<f32>(0.0, 0.0, 0.0, 0.0);\n\
    var acc_dual = vec4<f32>(0.0, 0.0, 0.0, 0.0);\n\
    var has_pivot = false;\n\
    var pivot = vec4<f32>(0.0, 0.0, 0.0, 0.0);\n\
    for (var i = 0u; i < 4u; i = i + 1u) {\n\
        let wi = w[i];\n\
        if (wi == 0.0) { continue; }\n\
        var r = reals[i];\n\
        var d = duals[i];\n\
        if (has_pivot) {\n\
            if (dot(pivot, r) < 0.0) {\n\
                r = -r;\n\
                d = -d;\n\
            }\n\
        } else {\n\
            pivot = r;\n\
            has_pivot = true;\n\
        }\n\
        acc_real = acc_real + r * wi;\n\
        acc_dual = acc_dual + d * wi;\n\
    }\n\
    let inv = 1.0 / length(acc_real);\n\
    let real = acc_real * inv;\n\
    var dual = acc_dual * inv;\n\
    let dd = dot(real, dual);\n\
    dual = dual - real * dd;\n\
    let t4 = 2.0 * prism_quat_mul(dual, prism_quat_conj(real));\n\
    return prism_quat_rotate(real, p) + t4.xyz;\n\
}\n";
/// The authoritative WGSL source for order-3 (16-coefficient) real spherical
/// harmonics **evaluation**, mirroring [`crate::spherical::basis3`] and
/// [`crate::spherical::Sh3::eval`] **term for term**: the same Condon–Shortley
/// folded constants, the same direction-cosine polynomials, and the same
/// sequential 16-term accumulation. This is the GPU side of GI probe
/// reconstruction — a fragment or compute shader samples a probe's `SH` vector
/// and evaluates it per shading direction.
///
/// `dir` must be unit length (the polynomials assume `x^2 + y^2 + z^2 = 1`). A
/// consumer embeds this verbatim so the GPU probe evaluation cannot drift from
/// the CPU bake/reference.
pub const WGSL_SH3_EVAL: &str = "\
fn prism_sh3_basis(dir: vec3<f32>) -> array<f32, 16> {\n\
    let x = dir.x;\n\
    let y = dir.y;\n\
    let z = dir.z;\n\
    let x2 = x * x;\n\
    let y2 = y * y;\n\
    let z2 = z * z;\n\
    return array<f32, 16>(\n\
        0.28209479,\n\
        0.48860251 * y,\n\
        0.48860251 * z,\n\
        0.48860251 * x,\n\
        1.0925484 * x * y,\n\
        1.0925484 * y * z,\n\
        0.31539157 * (3.0 * z2 - 1.0),\n\
        1.0925484 * x * z,\n\
        0.5462742 * (x2 - y2),\n\
        0.5900436 * y * (3.0 * x2 - y2),\n\
        2.8906114 * x * y * z,\n\
        0.4570458 * y * (5.0 * z2 - 1.0),\n\
        0.3731763 * z * (5.0 * z2 - 3.0),\n\
        0.4570458 * x * (5.0 * z2 - 1.0),\n\
        0.5 * 2.8906114 * z * (x2 - y2),\n\
        0.5900436 * x * (x2 - 3.0 * y2)\n\
    );\n\
}\n\
\n\
fn prism_sh3_eval(coeffs: array<f32, 16>, dir: vec3<f32>) -> f32 {\n\
    var c = coeffs;\n\
    var basis = prism_sh3_basis(dir);\n\
    var acc = 0.0;\n\
    for (var i = 0u; i < 16u; i = i + 1u) {\n\
        acc = acc + c[i] * basis[i];\n\
    }\n\
    return acc;\n\
}\n";
/// Single-sourced WGSL for Morton (Z-order) spatial-locality keys, the GPU
/// radix-sort primitive mirrored from [`crate::spatial`].
///
/// WGSL has no 64-bit integer type, so these kernels operate at the
/// GPU-representable key widths whose results still fit a `u32`: the 2D encoder
/// consumes 16 bits per axis (32-bit key) and the 3D encoder 10 bits per axis
/// (30-bit key). Those are exactly the widths used for on-device BVH/radix-sort
/// keys. The bit-spread/compact masks are the standard 16-bit (`part1by1`) and
/// 10-bit (`part1by2`) sequences, and because interleaving is bit-local they
/// reproduce the *low bits* of the wider CPU encoders
/// ([`morton_encode2`](crate::spatial::morton_encode2) /
/// [`morton_encode3`](crate::spatial::morton_encode3)) exactly: for inputs
/// masked to the respective axis width the GPU key equals the CPU key
/// bit-for-bit (integer math, so the parity here is exact, not a tolerance).
pub const WGSL_MORTON: &str = "\
fn prism_morton_part1by1(value: u32) -> u32 {\n\
    var x = value & 0x0000ffffu;\n\
    x = (x | (x << 8u)) & 0x00ff00ffu;\n\
    x = (x | (x << 4u)) & 0x0f0f0f0fu;\n\
    x = (x | (x << 2u)) & 0x33333333u;\n\
    x = (x | (x << 1u)) & 0x55555555u;\n\
    return x;\n\
}\n\
\n\
fn prism_morton_compact1by1(value: u32) -> u32 {\n\
    var x = value & 0x55555555u;\n\
    x = (x ^ (x >> 1u)) & 0x33333333u;\n\
    x = (x ^ (x >> 2u)) & 0x0f0f0f0fu;\n\
    x = (x ^ (x >> 4u)) & 0x00ff00ffu;\n\
    x = (x ^ (x >> 8u)) & 0x0000ffffu;\n\
    return x;\n\
}\n\
\n\
fn prism_morton_part1by2(value: u32) -> u32 {\n\
    var x = value & 0x000003ffu;\n\
    x = (x | (x << 16u)) & 0x030000ffu;\n\
    x = (x | (x << 8u)) & 0x0300f00fu;\n\
    x = (x | (x << 4u)) & 0x030c30c3u;\n\
    x = (x | (x << 2u)) & 0x09249249u;\n\
    return x;\n\
}\n\
\n\
fn prism_morton_compact1by2(value: u32) -> u32 {\n\
    var x = value & 0x09249249u;\n\
    x = (x ^ (x >> 2u)) & 0x030c30c3u;\n\
    x = (x ^ (x >> 4u)) & 0x0300f00fu;\n\
    x = (x ^ (x >> 8u)) & 0x030000ffu;\n\
    x = (x ^ (x >> 16u)) & 0x000003ffu;\n\
    return x;\n\
}\n\
\n\
fn prism_morton_encode2(x: u32, y: u32) -> u32 {\n\
    return prism_morton_part1by1(x) | (prism_morton_part1by1(y) << 1u);\n\
}\n\
\n\
fn prism_morton_decode2(code: u32) -> vec2<u32> {\n\
    return vec2<u32>(prism_morton_compact1by1(code), prism_morton_compact1by1(code >> 1u));\n\
}\n\
\n\
fn prism_morton_encode3(x: u32, y: u32, z: u32) -> u32 {\n\
    return prism_morton_part1by2(x) | (prism_morton_part1by2(y) << 1u) | (prism_morton_part1by2(z) << 2u);\n\
}\n\
\n\
fn prism_morton_decode3(code: u32) -> vec3<u32> {\n\
    return vec3<u32>(prism_morton_compact1by2(code), prism_morton_compact1by2(code >> 1u), prism_morton_compact1by2(code >> 2u));\n\
}\n";
/// Single-sourced WGSL for GPU-driven view-frustum culling, mirroring the CPU
/// classifiers [`crate::intersect::frustum_sphere`] /
/// [`crate::intersect::frustum_aabb`]. Each plane is passed as a `vec4<f32>`
/// `(normal.xyz, d)` with an inward-facing unit normal, so the signed distance
/// is `dot(normal, p) + d`; the returned `u32` matches the
/// [`Containment`](crate::intersect::Containment) discriminants exactly
/// (`0 = Outside`, `1 = Intersecting`, `2 = Inside`). The arithmetic is the
/// identical plane/p-vertex test, so the discrete classification agrees with
/// the CPU for any geometry not within fast-math rounding of a plane boundary
/// (the standard conservative-culling caveat).
pub const WGSL_FRUSTUM_CULL: &str = "\
fn prism_frustum_classify_sphere(planes: array<vec4<f32>, 6>, center: vec3<f32>, radius: f32) -> u32 {\n\
    var result = 2u;\n\
    for (var i = 0u; i < 6u; i = i + 1u) {\n\
        let pl = planes[i];\n\
        let dist = dot(pl.xyz, center) + pl.w;\n\
        if (dist < -radius) { return 0u; }\n\
        if (dist < radius) { result = 1u; }\n\
    }\n\
    return result;\n\
}\n\
\n\
fn prism_frustum_classify_aabb(planes: array<vec4<f32>, 6>, center: vec3<f32>, extent: vec3<f32>) -> u32 {\n\
    var result = 2u;\n\
    for (var i = 0u; i < 6u; i = i + 1u) {\n\
        let pl = planes[i];\n\
        let n = pl.xyz;\n\
        let r = extent.x * abs(n.x) + extent.y * abs(n.y) + extent.z * abs(n.z);\n\
        let s = dot(n, center) + pl.w;\n\
        if (s < -r) { return 0u; }\n\
        if (s < r) { result = 1u; }\n\
    }\n\
    return result;\n\
}\n";

/// Single-sourced WGSL for octahedral unit-normal (de)compression, mirroring
/// the CPU codec [`crate::octahedral::encode`] / [`decode`] / [`pack_snorm`] /
/// [`unpack_snorm`]. Octahedral mapping is the standard compact `GBuffer` normal
/// encoding: it stores a unit direction in two numbers (or, snorm-packed, one
/// `u32`) with negligible angular error. The folding/sign convention and the
/// `round(c * 32767)`-ties-away snorm quantization are reproduced exactly, so
/// the GPU g-buffer write/read agrees with the CPU codec within fast-math
/// rounding (the encode L1-normalize divide is the only fast-math-sensitive
/// step, hence a tolerance rather than bit-exactness).
pub const WGSL_OCTAHEDRAL: &str = "\
fn prism_oct_sign_nonzero(v: f32) -> f32 {\n\
    if (v >= 0.0) { return 1.0; }\n\
    return -1.0;\n\
}\n\
\n\
fn prism_oct_encode(n: vec3<f32>) -> vec2<f32> {\n\
    let inv_l1 = 1.0 / (abs(n.x) + abs(n.y) + abs(n.z));\n\
    let p = vec2<f32>(n.x * inv_l1, n.y * inv_l1);\n\
    if (n.z >= 0.0) { return p; }\n\
    return vec2<f32>(\n\
        (1.0 - abs(p.y)) * prism_oct_sign_nonzero(p.x),\n\
        (1.0 - abs(p.x)) * prism_oct_sign_nonzero(p.y)\n\
    );\n\
}\n\
\n\
fn prism_oct_decode(e: vec2<f32>) -> vec3<f32> {\n\
    var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));\n\
    let t = max(-n.z, 0.0);\n\
    if (n.x >= 0.0) { n.x = n.x - t; } else { n.x = n.x + t; }\n\
    if (n.y >= 0.0) { n.y = n.y - t; } else { n.y = n.y + t; }\n\
    return normalize(n);\n\
}\n\
\n\
fn prism_oct_snorm16(v: f32) -> u32 {\n\
    let c = clamp(v, -1.0, 1.0);\n\
    let scaled = c * 32767.0;\n\
    let r = i32(floor(abs(scaled) + 0.5));\n\
    let signed = select(r, -r, scaled < 0.0);\n\
    return bitcast<u32>(signed) & 0xffffu;\n\
}\n\
\n\
fn prism_oct_pack_snorm(n: vec3<f32>) -> u32 {\n\
    let e = prism_oct_encode(n);\n\
    return prism_oct_snorm16(e.x) | (prism_oct_snorm16(e.y) << 16u);\n\
}\n\
\n\
fn prism_oct_unsnorm16(bits: u32) -> f32 {\n\
    let lo = i32(bits & 0xffffu);\n\
    let signed = select(lo, lo - 65536, lo >= 32768);\n\
    return clamp(f32(signed) / 32767.0, -1.0, 1.0);\n\
}\n\
\n\
fn prism_oct_unpack_snorm(bits: u32) -> vec3<f32> {\n\
    let x = prism_oct_unsnorm16(bits & 0xffffu);\n\
    let y = prism_oct_unsnorm16((bits >> 16u) & 0xffffu);\n\
    return prism_oct_decode(vec2<f32>(x, y));\n\
}\n";
/// Single-sourced WGSL for ray/primitive intersection, mirroring the CPU
/// queries [`crate::intersect::ray_sphere`] / [`crate::intersect::ray_aabb`].
/// Each kernel returns a [`PrismRayHit`]-shaped result (hit flag, ray parameter
/// `t`, world hit point, and surface normal oriented against the ray) — the
/// GPU side of picking / spatial queries / batched sphere-casts feeding
/// GPU-driven selection and collision pre-passes. The analytic quadratic
/// (sphere) and slab method (AABB) are reproduced exactly, so the device
/// agrees with the CPU within fast-math rounding on `t`/point and matches the
/// discrete hit flag and axis-aligned normal for geometry with a comfortable
/// margin from a tangent/edge grazing case (the standard conservative caveat;
/// the AABB slab bounds use a large finite sentinel in place of CPU infinity,
/// identical for finite well-separated geometry).
pub const WGSL_RAYCAST: &str = "\
struct PrismRayHit {\n\
    hit: f32,\n\
    t: f32,\n\
    point: vec3<f32>,\n\
    normal: vec3<f32>,\n\
};\n\
\n\
fn prism_ray_sphere(origin: vec3<f32>, dir: vec3<f32>, center: vec3<f32>, radius: f32) -> PrismRayHit {\n\
    var out: PrismRayHit;\n\
    out.hit = 0.0;\n\
    out.t = 0.0;\n\
    out.point = vec3<f32>(0.0, 0.0, 0.0);\n\
    out.normal = vec3<f32>(0.0, 0.0, 0.0);\n\
    let oc = origin - center;\n\
    let a = dot(dir, dir);\n\
    if (a <= 0.0) { return out; }\n\
    let b = 2.0 * dot(oc, dir);\n\
    let c = dot(oc, oc) - radius * radius;\n\
    let disc = b * b - 4.0 * a * c;\n\
    if (disc < 0.0) { return out; }\n\
    let sqrt_disc = sqrt(disc);\n\
    let inv2a = 1.0 / (2.0 * a);\n\
    let t0 = (-b - sqrt_disc) * inv2a;\n\
    let t1 = (-b + sqrt_disc) * inv2a;\n\
    var t = 0.0;\n\
    var inside = false;\n\
    if (t0 >= 0.0) {\n\
        t = t0;\n\
        inside = false;\n\
    } else if (t1 >= 0.0) {\n\
        t = t1;\n\
        inside = true;\n\
    } else {\n\
        return out;\n\
    }\n\
    let point = origin + dir * t;\n\
    var normal = (point - center) * (1.0 / radius);\n\
    if (inside) { normal = -normal; }\n\
    out.hit = 1.0;\n\
    out.t = t;\n\
    out.point = point;\n\
    out.normal = normal;\n\
    return out;\n\
}\n\
\n\
fn prism_ray_aabb(origin: vec3<f32>, dir: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> PrismRayHit {\n\
    var out: PrismRayHit;\n\
    out.hit = 0.0;\n\
    out.t = 0.0;\n\
    out.point = vec3<f32>(0.0, 0.0, 0.0);\n\
    out.normal = vec3<f32>(0.0, 0.0, 0.0);\n\
    var t_enter = -1e30;\n\
    var t_exit = 1e30;\n\
    var enter_axis = 0u;\n\
    var enter_sign = 1.0;\n\
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {\n\
        let d = dir[axis];\n\
        if (abs(d) <= 1.0e-20) {\n\
            if (origin[axis] < lo[axis] || origin[axis] > hi[axis]) { return out; }\n\
            continue;\n\
        }\n\
        let inv = 1.0 / d;\n\
        var ta = (lo[axis] - origin[axis]) * inv;\n\
        var tb = (hi[axis] - origin[axis]) * inv;\n\
        var sign = -1.0;\n\
        if (ta > tb) {\n\
            let tmp = ta;\n\
            ta = tb;\n\
            tb = tmp;\n\
            sign = 1.0;\n\
        }\n\
        if (ta > t_enter) {\n\
            t_enter = ta;\n\
            enter_axis = axis;\n\
            enter_sign = sign;\n\
        }\n\
        if (tb < t_exit) { t_exit = tb; }\n\
        if (t_enter > t_exit) { return out; }\n\
    }\n\
    if (t_exit < 0.0) { return out; }\n\
    var t = 0.0;\n\
    var axis_out = 0u;\n\
    var sign_out = 1.0;\n\
    if (t_enter >= 0.0) {\n\
        t = t_enter;\n\
        axis_out = enter_axis;\n\
        sign_out = enter_sign;\n\
    } else {\n\
        var exit_axis = 0u;\n\
        var exit_t = 1e30;\n\
        var exit_sign = 1.0;\n\
        for (var axis = 0u; axis < 3u; axis = axis + 1u) {\n\
            let d = dir[axis];\n\
            if (abs(d) <= 1.0e-20) { continue; }\n\
            let inv = 1.0 / d;\n\
            let ta = (lo[axis] - origin[axis]) * inv;\n\
            let tb = (hi[axis] - origin[axis]) * inv;\n\
            var far = tb;\n\
            var s = 1.0;\n\
            if (ta > tb) {\n\
                far = ta;\n\
                s = -1.0;\n\
            }\n\
            if (far < exit_t) {\n\
                exit_t = far;\n\
                exit_axis = axis;\n\
                exit_sign = s;\n\
            }\n\
        }\n\
        t = exit_t;\n\
        axis_out = exit_axis;\n\
        sign_out = exit_sign;\n\
    }\n\
    var normal = vec3<f32>(0.0, 0.0, 0.0);\n\
    if (axis_out == 0u) {\n\
        normal.x = sign_out;\n\
    } else if (axis_out == 1u) {\n\
        normal.y = sign_out;\n\
    } else {\n\
        normal.z = sign_out;\n\
    }\n\
    out.hit = 1.0;\n\
    out.t = t;\n\
    out.point = origin + dir * t;\n\
    out.normal = normal;\n\
    return out;\n\
}\n";
