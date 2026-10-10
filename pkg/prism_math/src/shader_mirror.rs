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

/// Single-sourced WGSL for GPU 2D Hilbert-curve encode/decode, mirroring the
/// CPU path [`crate::spatial::hilbert_encode2`] / [`crate::spatial::hilbert_decode2`].
///
/// WGSL has no 64-bit integer type, so the kernel runs the 16-bit-per-axis
/// Hilbert order (a 32-bit `u32` key) — exactly the width used for on-device
/// 2D tile / quadtree sort keys. Because the top 16 levels of the CPU's
/// 32-bit-per-axis curve are pure no-ops for inputs masked to 16 bits (they
/// contribute `0` to the index and apply an even number of axis swaps that
/// cancel), the CPU `hilbert_encode2` output for any `(x, y)` with
/// `x, y < 2^16` equals this order-16 key bit-for-bit, so the §24.1 parity
/// here is **exact integer equality**, not a floating-point tolerance.
pub const WGSL_HILBERT2: &str = "\
const PRISM_HILBERT2_N: u32 = 65536u;\n\
\n\
fn prism_hilbert_encode2(ix: u32, iy: u32) -> u32 {\n\
    var x = ix;\n\
    var y = iy;\n\
    var d: u32 = 0u;\n\
    var s: u32 = PRISM_HILBERT2_N >> 1u;\n\
    loop {\n\
        if (s == 0u) { break; }\n\
        var rx: u32 = 0u;\n\
        if ((x & s) > 0u) { rx = 1u; }\n\
        var ry: u32 = 0u;\n\
        if ((y & s) > 0u) { ry = 1u; }\n\
        d = d + s * s * ((3u * rx) ^ ry);\n\
        if (ry == 0u) {\n\
            if (rx == 1u) {\n\
                x = (PRISM_HILBERT2_N - 1u) - x;\n\
                y = (PRISM_HILBERT2_N - 1u) - y;\n\
            }\n\
            let t = x;\n\
            x = y;\n\
            y = t;\n\
        }\n\
        s = s >> 1u;\n\
    }\n\
    return d;\n\
}\n\
\n\
fn prism_hilbert_decode2(index: u32) -> vec2<u32> {\n\
    var t = index;\n\
    var x: u32 = 0u;\n\
    var y: u32 = 0u;\n\
    var s: u32 = 1u;\n\
    loop {\n\
        if (s >= PRISM_HILBERT2_N) { break; }\n\
        let rx = 1u & (t >> 1u);\n\
        let ry = 1u & (t ^ rx);\n\
        if (ry == 0u) {\n\
            if (rx == 1u) {\n\
                x = (s - 1u) - x;\n\
                y = (s - 1u) - y;\n\
            }\n\
            let tmp = x;\n\
            x = y;\n\
            y = tmp;\n\
        }\n\
        x = x + s * rx;\n\
        y = y + s * ry;\n\
        t = t >> 2u;\n\
        s = s << 1u;\n\
    }\n\
    return vec2<u32>(x, y);\n\
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

/// Single-sourced WGSL for quaternion interpolation (`slerp` / `nlerp`).
///
/// Mirrors the CPU references [`prism_math::Quat::slerp`] and
/// [`prism_math::Quat::nlerp`] step for step: both take the shortest arc by
/// flipping `b` when `dot(a, b) < 0`; `nlerp` is a normalized component-wise
/// lerp; `slerp` falls back to `nlerp` when the endpoints are nearly colinear
/// (`dot > 0.9995`, the CPU `DOT_THRESHOLD`) and otherwise blends by
/// `sin((1-t)θ)/sinθ` and `sin(tθ)/sinθ` with `θ = acos(clamp(dot,-1,1))`.
///
/// Parity, not bit-exactness: the device evaluates `acos`/`sin`/`normalize`
/// with Metal fast-math rounding (and the CPU routes the same transcendentals
/// through `libm` for determinism), so the twin agrees within tolerance, not
/// bit-for-bit. The quaternions `vec4<f32>` are laid out `xyzw`, matching
/// [`pack_quat`].
pub const WGSL_QUAT_INTERP: &str = "\
fn prism_quat_dot(a: vec4<f32>, b: vec4<f32>) -> f32 {\n\
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;\n\
}\n\
\n\
fn prism_quat_nlerp(a: vec4<f32>, b_in: vec4<f32>, t: f32) -> vec4<f32> {\n\
    var b = b_in;\n\
    if (prism_quat_dot(a, b) < 0.0) {\n\
        b = -b;\n\
    }\n\
    let r = a + (b - a) * t;\n\
    return normalize(r);\n\
}\n\
\n\
fn prism_quat_slerp(a: vec4<f32>, b_in: vec4<f32>, t: f32) -> vec4<f32> {\n\
    var b = b_in;\n\
    var d = prism_quat_dot(a, b);\n\
    if (d < 0.0) {\n\
        b = -b;\n\
        d = -d;\n\
    }\n\
    if (d > 0.9995) {\n\
        return prism_quat_nlerp(a, b, t);\n\
    }\n\
    let theta = acos(clamp(d, -1.0, 1.0));\n\
    let sin_theta = sin(theta);\n\
    let s0 = sin((1.0 - t) * theta) / sin_theta;\n\
    let s1 = sin(t * theta) / sin_theta;\n\
    return a * s0 + b * s1;\n\
}\n";

/// Single-sourced WGSL for ray-triangle intersection (Möller-Trumbore).
///
/// Mirrors the CPU reference [`prism_math::intersect::ray_triangle_bary`] step
/// for step: the `edge1`/`edge2`/`pvec`/`det` setup, the `|det| < 1e-8`
/// degenerate/parallel reject (the CPU `EPS`), the `u`/`v` barycentric
/// in-triangle tests, and the `t >= 0` forward reject. Both faces are hittable.
/// `prism_ray_triangle` additionally returns a geometric normal oriented
/// against the ray, matching [`prism_math::intersect::ray_triangle`].
///
/// Parity, not bit-exactness: the barycentric divides, cross/dot products, and
/// the normal `normalize` are evaluated under Metal fast-math (FMA contraction
/// and reassociation), so `t`/`u`/`v`/point/normal agree within a small
/// tolerance while the discrete hit flag matches for geometry with a
/// comfortable margin from the triangle edges / a grazing (near-parallel) ray.
pub const WGSL_RAYTRI: &str = "\
struct PrismTriHit {\n\
    hit: f32,\n\
    t: f32,\n\
    u: f32,\n\
    v: f32,\n\
    normal: vec3<f32>,\n\
};\n\
\n\
fn prism_ray_triangle(origin: vec3<f32>, dir: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> PrismTriHit {\n\
    var out: PrismTriHit;\n\
    out.hit = 0.0;\n\
    out.t = 0.0;\n\
    out.u = 0.0;\n\
    out.v = 0.0;\n\
    out.normal = vec3<f32>(0.0, 0.0, 0.0);\n\
    let edge1 = b - a;\n\
    let edge2 = c - a;\n\
    let pvec = cross(dir, edge2);\n\
    let det = dot(edge1, pvec);\n\
    if (abs(det) < 1.0e-8) { return out; }\n\
    let inv_det = 1.0 / det;\n\
    let tvec = origin - a;\n\
    let u = dot(tvec, pvec) * inv_det;\n\
    if (u < 0.0 || u > 1.0) { return out; }\n\
    let qvec = cross(tvec, edge1);\n\
    let v = dot(dir, qvec) * inv_det;\n\
    if (v < 0.0 || u + v > 1.0) { return out; }\n\
    let t = dot(edge2, qvec) * inv_det;\n\
    if (t < 0.0) { return out; }\n\
    var normal = normalize(cross(edge1, edge2));\n\
    if (dot(normal, dir) > 0.0) { normal = -normal; }\n\
    out.hit = 1.0;\n\
    out.t = t;\n\
    out.u = u;\n\
    out.v = v;\n\
    out.normal = normal;\n\
    return out;\n\
}\n";

/// Single-sourced WGSL for the half-precision (`binary16`) pack/unpack twin
/// (§24.1 / §24.3 f16 bandwidth path).
///
/// `prism_f16_pack2` folds two `f32` lanes into one `u32` holding two
/// `binary16` values (low half = `a`, high half = `b`), and `prism_f16_unpack2`
/// widens that `u32` back to two `f32`s. The body is the WGSL builtins
/// `pack2x16float` / `unpack2x16float`, which the spec defines as
/// round-to-nearest-even — the same rounding as the CPU reference
/// [`prism_math::f16::F16::from_f32`]. Wrapping them in named helpers keeps the
/// device entry point identical in shape to the other twins and gives the host
/// a single call site to compose.
///
/// Parity contract (honest boundary): for finite values inside the `f16`
/// **normal** range (`|x|` in `[2^-14, 65504]`) the packed 16 bits match the
/// CPU `F16::from_f32` bit-for-bit, because both round to nearest even. Two
/// cases are deliberately *not* asserted bit-exact: **subnormals** (`|x| <
/// 2^-14`), which Metal and other GPUs may flush to zero, and **overflow**
/// (`|x| > 65504`) plus `NaN`, which the WGSL spec leaves implementation-defined
/// for `pack2x16float`. Those are documented as a reconstruction-tolerance
/// boundary rather than a bit contract. The `f16 -> f32` unpack direction is
/// exact on both sides.
pub const WGSL_F16: &str = "\
fn prism_f16_pack2(a: f32, b: f32) -> u32 {\n\
    return pack2x16float(vec2<f32>(a, b));\n\
}\n\
\n\
fn prism_f16_unpack2(bits: u32) -> vec2<f32> {\n\
    return unpack2x16float(bits);\n\
}\n";

/// Single-sourced WGSL for the sRGB electro-optical transfer functions
/// (gamma encode/decode), mirroring the CPU exact piecewise IEC 61966-2-1
/// definition in [`crate::color::transfer`].
///
/// `prism_srgb_to_linear` decodes one non-linear sRGB component to linear
/// light; `prism_linear_to_srgb` encodes the inverse. Both use the exact
/// piecewise curve with the identical breakpoint literals as the CPU path, so
/// for the same input both sides take the same branch. The linear segment is a
/// bare multiply/divide; the power segment calls the WGSL `pow` builtin, which
/// Metal compiles under fast-math, so the §24.1 contract on the power segment
/// is a small tolerance rather than a bit contract (`libm::powf` versus device
/// `pow`). Alpha never passes through these curves.
pub const WGSL_SRGB: &str = "\
fn prism_srgb_to_linear(c: f32) -> f32 {\n\
    if (c <= 0.040448237) {\n\
        return c / 12.92;\n\
    }\n\
    return pow((c + 0.055) / 1.055, 2.4);\n\
}\n\
\n\
fn prism_linear_to_srgb(c: f32) -> f32 {\n\
    if (c <= 0.0031308) {\n\
        return c * 12.92;\n\
    }\n\
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;\n\
}\n";

/// Single-sourced WGSL for 8-bit-per-channel vertex-attribute packing,
/// mirroring the CPU codec [`crate::pack8`].
///
/// `prism_pack_unorm4x8` / `prism_pack_snorm4x8` fold a `vec4<f32>` into one
/// `u32` of four 8-bit channels (component 0 in the low byte) and the `unpack`
/// inverses widen them back, wrapping the WGSL built-ins `pack4x8unorm` /
/// `pack4x8snorm` / `unpack4x8unorm` / `unpack4x8snorm`. The WGSL spec defines
/// the quantizers as `⌊0.5 + N·clamp(c)⌋` (`N` = 255 unorm / 127 snorm), the
/// same rounding the CPU reference uses, so for exactly-representable quantized
/// inputs the packed bytes match the CPU path **byte-for-byte**; arbitrary
/// inputs may differ by at most one code at a rounding tie (a documented honest
/// boundary). The widening (`unpack`) direction is exact on both sides.
pub const WGSL_PACK8: &str = "\
fn prism_pack_unorm4x8(v: vec4<f32>) -> u32 {\n\
    return pack4x8unorm(v);\n\
}\n\
\n\
fn prism_unpack_unorm4x8(bits: u32) -> vec4<f32> {\n\
    return unpack4x8unorm(bits);\n\
}\n\
\n\
fn prism_pack_snorm4x8(v: vec4<f32>) -> u32 {\n\
    return pack4x8snorm(v);\n\
}\n\
\n\
fn prism_unpack_snorm4x8(bits: u32) -> vec4<f32> {\n\
    return unpack4x8snorm(bits);\n\
}\n";

/// Single-sourced WGSL for the 16-bit-per-channel vertex-attribute pack/unpack
/// helpers mirrored by [`crate::pack16`].
///
/// The `pack` helpers fold a `vec2<f32>` into one `u32` of two 16-bit channels
/// (component 0 in the low half-word) and the `unpack` inverses widen them
/// back, wrapping the WGSL built-ins `pack2x16unorm` / `pack2x16snorm` /
/// `unpack2x16unorm` / `unpack2x16snorm`. The WGSL spec defines the quantizers
/// as `⌊0.5 + N·clamp(c)⌋` (`N` = 65535 unorm / 32767 snorm), the same rounding
/// the CPU reference uses, so for exactly-representable quantized inputs the
/// packed half-words match the CPU path **bit-for-bit**; arbitrary inputs may
/// differ by at most one code at a rounding tie (a documented honest boundary).
/// The widening (`unpack`) direction is exact on both sides.
pub const WGSL_PACK16: &str = "\
fn prism_pack_unorm2x16(v: vec2<f32>) -> u32 {\n\
    return pack2x16unorm(v);\n\
}\n\
\n\
fn prism_unpack_unorm2x16(bits: u32) -> vec2<f32> {\n\
    return unpack2x16unorm(bits);\n\
}\n\
\n\
fn prism_pack_snorm2x16(v: vec2<f32>) -> u32 {\n\
    return pack2x16snorm(v);\n\
}\n\
\n\
fn prism_unpack_snorm2x16(bits: u32) -> vec2<f32> {\n\
    return unpack2x16snorm(bits);\n\
}\n";

/// Single-sourced WGSL for the linear-sRGB <-> `OkLab` perceptual color
/// conversions mirrored by [`crate::color::oklab::Oklaba`].
///
/// `OkLab` (Björn Ottosson, 2020) is the modern perceptually-uniform space used
/// for color grading and gradient mixing. The forward helper applies the LMS
/// analysis matrix, a per-channel cube root, and the `OkLab` matrix; the inverse
/// undoes them. Both carry the fourth lane (alpha) through unchanged. The 3x3
/// matrix literals match the CPU reference [`Oklaba::from_linear`] /
/// [`Oklaba::to_linear`] component-for-component.
///
/// WGSL has no `cbrt` built-in, so `prism_cbrt` is composed as
/// `sign(x) * pow(abs(x), 1/3)`. The CPU reference uses `libm::cbrt`
/// (near-correctly-rounded), so the two differ by a small `pow` rounding error:
/// this is a documented honest boundary, verified with a tolerance rather than
/// bit-exactly. The matrix/cube inverse direction is ordinary FMA arithmetic.
pub const WGSL_OKLAB: &str = "\
fn prism_cbrt(x: f32) -> f32 {\n\
    return sign(x) * pow(abs(x), 0.3333333333333333);\n\
}\n\
\n\
fn prism_linear_to_oklab(c: vec4<f32>) -> vec4<f32> {\n\
    let r = c.x;\n\
    let g = c.y;\n\
    let b = c.z;\n\
    let l = 0.41222147 * r + 0.53633255 * g + 0.051445995 * b;\n\
    let m = 0.2119035 * r + 0.6806995 * g + 0.10739696 * b;\n\
    let s = 0.08830246 * r + 0.28171885 * g + 0.6299787 * b;\n\
    let l_ = prism_cbrt(l);\n\
    let m_ = prism_cbrt(m);\n\
    let s_ = prism_cbrt(s);\n\
    return vec4<f32>(\n\
        0.21045426 * l_ + 0.7936178 * m_ - 0.004072047 * s_,\n\
        1.9779985 * l_ - 2.4285922 * m_ + 0.4505937 * s_,\n\
        0.025904037 * l_ + 0.78277177 * m_ - 0.80867577 * s_,\n\
        c.w,\n\
    );\n\
}\n\
\n\
fn prism_oklab_to_linear(c: vec4<f32>) -> vec4<f32> {\n\
    let l_ = c.x + 0.39633778 * c.y + 0.21580376 * c.z;\n\
    let m_ = c.x - 0.105561346 * c.y - 0.06385417 * c.z;\n\
    let s_ = c.x - 0.08948418 * c.y - 1.2914855 * c.z;\n\
    let l = l_ * l_ * l_;\n\
    let m = m_ * m_ * m_;\n\
    let s = s_ * s_ * s_;\n\
    return vec4<f32>(\n\
        4.0767417 * l - 3.3077116 * m + 0.23096994 * s,\n\
        -1.268438 * l + 2.6097574 * m - 0.34131938 * s,\n\
        -0.0041960863 * l - 0.7034186 * m + 1.7076147 * s,\n\
        c.w,\n\
    );\n\
}\n";

/// Single-sourced WGSL for the linear-sRGB <-> CIE 1931 XYZ (D65) conversion.
///
/// `prism_linear_to_xyz` applies the sRGB-primaries-to-XYZ matrix and
/// `prism_xyz_to_linear` its inverse; both carry the fourth lane (alpha)
/// through unchanged. The 3x3 matrix literals match the CPU reference
/// [`LinearRgba::to_xyz`](crate::color::LinearRgba::to_xyz) /
/// [`LinearRgba::from_xyz`](crate::color::LinearRgba::from_xyz)
/// component-for-component. This path is ordinary FMA arithmetic (no
/// transcendental), so parity is verified with a tight tolerance that only
/// absorbs Metal fast-math last-ULP rounding, not bit-exactly.
pub const WGSL_XYZ: &str = "\
fn prism_linear_to_xyz(c: vec4<f32>) -> vec4<f32> {\n\
    let r = c.x;\n\
    let g = c.y;\n\
    let b = c.z;\n\
    return vec4<f32>(\n\
        0.4124564 * r + 0.3575761 * g + 0.1804375 * b,\n\
        0.2126729 * r + 0.7151522 * g + 0.072175 * b,\n\
        0.0193339 * r + 0.119192 * g + 0.9503041 * b,\n\
        c.w,\n\
    );\n\
}\n\
\n\
fn prism_xyz_to_linear(c: vec4<f32>) -> vec4<f32> {\n\
    let x = c.x;\n\
    let y = c.y;\n\
    let z = c.z;\n\
    return vec4<f32>(\n\
        3.2404542 * x - 1.5371385 * y - 0.4985314 * z,\n\
        -0.969266 * x + 1.8760108 * y + 0.041556 * z,\n\
        0.0556434 * x - 0.2040259 * y + 1.0572252 * z,\n\
        c.w,\n\
    );\n\
}\n";

/// Single-sourced WGSL for the non-linear sRGB <-> HSL / HSV cylindrical color
/// conversions.
///
/// These mirror the CPU reference
/// [`Hsla`](crate::color::Hsla) / [`Hsva`](crate::color::Hsva): both models are
/// defined over **non-linear** sRGB components (the usual color-picker
/// convention), so the kernels operate directly on the stored sRGB quad with no
/// gamma step. Hue is carried in degrees `[0, 360)`.
///
/// WGSL has no `rem_euclid` built-in, so `prism_rem_euclid` reconstructs it as
/// `a - b*floor(a/b)` (equal to Rust `f32::rem_euclid` for the positive moduli
/// used here). The hue decomposition selects its branch on `max == r` /
/// `max == g`, and because `max`/`min` return one of their operands bit-for-bit
/// the GPU and CPU pick the same branch. The sector index uses `u32(h)` to
/// match the CPU's `h as u32` truncation. Everything else is ordinary FMA
/// arithmetic, so parity is a tight tolerance (hue is numerically unstable near
/// gray where chroma -> 0, hence round-trip is the strong check), not
/// bit-exact. The fourth lane (alpha) is carried through unchanged.
pub const WGSL_HSL: &str = "\
fn prism_rem_euclid(a: f32, b: f32) -> f32 {\n\
    return a - b * floor(a / b);\n\
}\n\
\n\
fn prism_rgb_to_hue(c: vec4<f32>) -> vec4<f32> {\n\
    let r = c.x;\n\
    let g = c.y;\n\
    let b = c.z;\n\
    let mx = max(max(r, g), b);\n\
    let mn = min(min(r, g), b);\n\
    let chroma = mx - mn;\n\
    var hue = 0.0;\n\
    if (chroma == 0.0) {\n\
        hue = 0.0;\n\
    } else if (mx == r) {\n\
        hue = 60.0 * prism_rem_euclid((g - b) / chroma, 6.0);\n\
    } else if (mx == g) {\n\
        hue = 60.0 * ((b - r) / chroma + 2.0);\n\
    } else {\n\
        hue = 60.0 * ((r - g) / chroma + 4.0);\n\
    }\n\
    return vec4<f32>(mx, mn, chroma, hue);\n\
}\n\
\n\
fn prism_hue_to_rgb(hue: f32, chroma: f32, m: f32, alpha: f32) -> vec4<f32> {\n\
    let h = prism_rem_euclid(hue, 360.0) / 60.0;\n\
    let x = chroma * (1.0 - abs(prism_rem_euclid(h, 2.0) - 1.0));\n\
    let sector = u32(h);\n\
    var r1 = 0.0;\n\
    var g1 = 0.0;\n\
    var b1 = 0.0;\n\
    if (sector == 0u) {\n\
        r1 = chroma; g1 = x; b1 = 0.0;\n\
    } else if (sector == 1u) {\n\
        r1 = x; g1 = chroma; b1 = 0.0;\n\
    } else if (sector == 2u) {\n\
        r1 = 0.0; g1 = chroma; b1 = x;\n\
    } else if (sector == 3u) {\n\
        r1 = 0.0; g1 = x; b1 = chroma;\n\
    } else if (sector == 4u) {\n\
        r1 = x; g1 = 0.0; b1 = chroma;\n\
    } else {\n\
        r1 = chroma; g1 = 0.0; b1 = x;\n\
    }\n\
    return vec4<f32>(r1 + m, g1 + m, b1 + m, alpha);\n\
}\n\
\n\
fn prism_hsl_from_srgb(c: vec4<f32>) -> vec4<f32> {\n\
    let d = prism_rgb_to_hue(c);\n\
    let mx = d.x;\n\
    let mn = d.y;\n\
    let chroma = d.z;\n\
    let hue = d.w;\n\
    let lightness = 0.5 * (mx + mn);\n\
    var saturation = 0.0;\n\
    if (lightness <= 0.0 || lightness >= 1.0) {\n\
        saturation = 0.0;\n\
    } else {\n\
        saturation = chroma / (1.0 - abs(2.0 * lightness - 1.0));\n\
    }\n\
    return vec4<f32>(hue, saturation, lightness, c.w);\n\
}\n\
\n\
fn prism_hsl_to_srgb(c: vec4<f32>) -> vec4<f32> {\n\
    let hue = c.x;\n\
    let saturation = c.y;\n\
    let lightness = c.z;\n\
    let chroma = (1.0 - abs(2.0 * lightness - 1.0)) * saturation;\n\
    let m = lightness - 0.5 * chroma;\n\
    return prism_hue_to_rgb(hue, chroma, m, c.w);\n\
}\n\
\n\
fn prism_hsv_from_srgb(c: vec4<f32>) -> vec4<f32> {\n\
    let d = prism_rgb_to_hue(c);\n\
    let mx = d.x;\n\
    let chroma = d.z;\n\
    let hue = d.w;\n\
    let value = mx;\n\
    var saturation = 0.0;\n\
    if (value <= 0.0) {\n\
        saturation = 0.0;\n\
    } else {\n\
        saturation = chroma / value;\n\
    }\n\
    return vec4<f32>(hue, saturation, value, c.w);\n\
}\n\
\n\
fn prism_hsv_to_srgb(c: vec4<f32>) -> vec4<f32> {\n\
    let hue = c.x;\n\
    let saturation = c.y;\n\
    let value = c.z;\n\
    let chroma = value * saturation;\n\
    let m = value - chroma;\n\
    return prism_hue_to_rgb(hue, chroma, m, c.w);\n\
}\n";

/// Single-sourced WGSL for the correlated-color-temperature -> linear-sRGB
/// conversion (Planckian locus).
///
/// `prism_temperature_to_linear` mirrors the CPU reference
/// [`LinearRgba::from_temperature`](crate::color::LinearRgba::from_temperature):
/// it clamps the Kelvin input to `[1667, 25000]`, evaluates the Kim et al.
/// (2002) piecewise-cubic chromaticity `(x, y)` on the Planckian locus, lifts it
/// to XYZ at unit luminance (`X = x/y`, `Y = 1`, `Z = (1-x-y)/y`), applies the
/// XYZ->linear-sRGB matrix (identical literals to
/// [`WGSL_XYZ`]'s `prism_xyz_to_linear`), and clamps negative out-of-gamut
/// components to `0`. The branch cutoffs (`t <= 2222`, `t <= 4000`) are on the
/// exact clamped input, so the GPU and CPU always select the same spline
/// segment. This is ordinary FMA plus two divides and clamps (no
/// transcendental), so parity is verified with a tight tolerance that only
/// absorbs Metal fast-math last-ULP rounding, not bit-exactly. The output alpha
/// is a constant `1.0` matching the CPU.
pub const WGSL_TEMPERATURE: &str = "\
fn prism_planckian_locus_xy(kelvin: f32) -> vec2<f32> {\n\
    let t = clamp(kelvin, 1667.0, 25000.0);\n\
    let inv = 1.0 / t;\n\
    let inv2 = inv * inv;\n\
    let inv3 = inv2 * inv;\n\
    var x = 0.0;\n\
    if (t <= 4000.0) {\n\
        x = -0.2661239e9 * inv3 - 0.2343589e6 * inv2 + 0.8776956e3 * inv + 0.179910;\n\
    } else {\n\
        x = -3.0258469e9 * inv3 + 2.107038e6 * inv2 + 0.2226347e3 * inv + 0.240390;\n\
    }\n\
    let x2 = x * x;\n\
    let x3 = x2 * x;\n\
    var y = 0.0;\n\
    if (t <= 2222.0) {\n\
        y = -1.1063814 * x3 - 1.3481102 * x2 + 2.1855583 * x - 0.20219683;\n\
    } else if (t <= 4000.0) {\n\
        y = -0.9549476 * x3 - 1.3741859 * x2 + 2.09137 * x - 0.16748867;\n\
    } else {\n\
        y = 3.081758 * x3 - 5.873387 * x2 + 3.7511299 * x - 0.37001483;\n\
    }\n\
    return vec2<f32>(x, y);\n\
}\n\
\n\
fn prism_temperature_to_linear(kelvin: f32) -> vec4<f32> {\n\
    let xy = prism_planckian_locus_xy(kelvin);\n\
    let x = xy.x;\n\
    let y = xy.y;\n\
    let big_x = x / y;\n\
    let big_y = 1.0;\n\
    let big_z = (1.0 - x - y) / y;\n\
    let r = 3.2404542 * big_x - 1.5371385 * big_y - 0.4985314 * big_z;\n\
    let g = -0.969266 * big_x + 1.8760108 * big_y + 0.041556 * big_z;\n\
    let b = 0.0556434 * big_x - 0.2040259 * big_y + 1.0572252 * big_z;\n\
    return vec4<f32>(max(r, 0.0), max(g, 0.0), max(b, 0.0), 1.0);\n\
}\n";

/// Single-sourced Perlin "improved" gradient-noise fragment (2D/3D), mirroring
/// the CPU reference [`crate::noise::Perlin::get2`]/[`get3`](crate::noise::Perlin::get3).
///
/// The permutation table is **uploaded, not rebuilt** on the device: the twin
/// binds the exact `[u32; 512]` from
/// [`Perlin::permutation_table`](crate::noise::Perlin::permutation_table) at
/// `@binding(1)`, so every integer hash lookup is bit-exact and the two sides
/// always dot the identical gradient. Only the quintic fade, the gradient dot
/// products, and the lerps are floating point, so parity is a small tolerance
/// (fast-math last-ULP), not bit-exact.
///
/// The fragment declares the perm storage binding (needed by
/// `prism_perm_hash`); the compute wrapper supplies the count uniform
/// (`@binding(0)`), the sample input (`@binding(2)`), the output
/// (`@binding(3)`), and `main`. Both the get2 and get3 kernels share this
/// fragment verbatim.
pub const WGSL_PERLIN: &str = "\
@group(0) @binding(1) var<storage, read> prism_perm: array<u32>;\n\
\n\
fn prism_perm_hash(i: i32) -> u32 {\n\
    return prism_perm[u32(i & 511)];\n\
}\n\
\n\
fn prism_perlin_fade(t: f32) -> f32 {\n\
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);\n\
}\n\
\n\
fn prism_perlin_lerp(a: f32, b: f32, t: f32) -> f32 {\n\
    return a + t * (b - a);\n\
}\n\
\n\
fn prism_perlin_grad2(hash: u32, x: f32, y: f32) -> f32 {\n\
    let h = hash & 7u;\n\
    let d = 0.7071067811865476;\n\
    var gx = -d;\n\
    var gy = -d;\n\
    switch (h) {\n\
        case 0u: { gx = 1.0; gy = 0.0; }\n\
        case 1u: { gx = -1.0; gy = 0.0; }\n\
        case 2u: { gx = 0.0; gy = 1.0; }\n\
        case 3u: { gx = 0.0; gy = -1.0; }\n\
        case 4u: { gx = d; gy = d; }\n\
        case 5u: { gx = -d; gy = d; }\n\
        case 6u: { gx = d; gy = -d; }\n\
        default: { gx = -d; gy = -d; }\n\
    }\n\
    return gx * x + gy * y;\n\
}\n\
\n\
fn prism_perlin_grad3(hash: u32, x: f32, y: f32, z: f32) -> f32 {\n\
    let h = hash & 15u;\n\
    var u = y;\n\
    if (h < 8u) { u = x; }\n\
    var v = z;\n\
    if (h < 4u) { v = y; } else if (h == 12u || h == 14u) { v = x; }\n\
    var uu = u;\n\
    if ((h & 1u) != 0u) { uu = -u; }\n\
    var vv = v;\n\
    if ((h & 2u) != 0u) { vv = -v; }\n\
    return uu + vv;\n\
}\n\
\n\
fn prism_perlin_get2(x: f32, y: f32) -> f32 {\n\
    let xi_f = floor(x);\n\
    let yi_f = floor(y);\n\
    let xf = x - xi_f;\n\
    let yf = y - yi_f;\n\
    let xi = i32(xi_f);\n\
    let yi = i32(yi_f);\n\
    let a = i32(prism_perm_hash(xi)) + yi;\n\
    let b = i32(prism_perm_hash(xi + 1)) + yi;\n\
    let u = prism_perlin_fade(xf);\n\
    let v = prism_perlin_fade(yf);\n\
    let aa = prism_perm_hash(a);\n\
    let ab = prism_perm_hash(a + 1);\n\
    let ba = prism_perm_hash(b);\n\
    let bb = prism_perm_hash(b + 1);\n\
    let x1 = prism_perlin_lerp(prism_perlin_grad2(aa, xf, yf), prism_perlin_grad2(ba, xf - 1.0, yf), u);\n\
    let x2 = prism_perlin_lerp(prism_perlin_grad2(ab, xf, yf - 1.0), prism_perlin_grad2(bb, xf - 1.0, yf - 1.0), u);\n\
    return prism_perlin_lerp(x1, x2, v) * 1.4142135623730951;\n\
}\n\
\n\
fn prism_perlin_get3(x: f32, y: f32, z: f32) -> f32 {\n\
    let xi_f = floor(x);\n\
    let yi_f = floor(y);\n\
    let zi_f = floor(z);\n\
    let xf = x - xi_f;\n\
    let yf = y - yi_f;\n\
    let zf = z - zi_f;\n\
    let xi = i32(xi_f);\n\
    let yi = i32(yi_f);\n\
    let zi = i32(zi_f);\n\
    let u = prism_perlin_fade(xf);\n\
    let v = prism_perlin_fade(yf);\n\
    let w = prism_perlin_fade(zf);\n\
    let a = i32(prism_perm_hash(xi)) + yi;\n\
    let b = i32(prism_perm_hash(xi + 1)) + yi;\n\
    let aa = i32(prism_perm_hash(a)) + zi;\n\
    let ab = i32(prism_perm_hash(a + 1)) + zi;\n\
    let ba = i32(prism_perm_hash(b)) + zi;\n\
    let bb = i32(prism_perm_hash(b + 1)) + zi;\n\
    let x1 = prism_perlin_lerp(prism_perlin_grad3(prism_perm_hash(aa), xf, yf, zf), prism_perlin_grad3(prism_perm_hash(ba), xf - 1.0, yf, zf), u);\n\
    let x2 = prism_perlin_lerp(prism_perlin_grad3(prism_perm_hash(ab), xf, yf - 1.0, zf), prism_perlin_grad3(prism_perm_hash(bb), xf - 1.0, yf - 1.0, zf), u);\n\
    let y1 = prism_perlin_lerp(x1, x2, v);\n\
    let x3 = prism_perlin_lerp(prism_perlin_grad3(prism_perm_hash(aa + 1), xf, yf, zf - 1.0), prism_perlin_grad3(prism_perm_hash(ba + 1), xf - 1.0, yf, zf - 1.0), u);\n\
    let x4 = prism_perlin_lerp(prism_perlin_grad3(prism_perm_hash(ab + 1), xf, yf - 1.0, zf - 1.0), prism_perlin_grad3(prism_perm_hash(bb + 1), xf - 1.0, yf - 1.0, zf - 1.0), u);\n\
    let y2 = prism_perlin_lerp(x3, x4, v);\n\
    return prism_perlin_lerp(y1, y2, w) * 1.0;\n\
}\n";

/// Single-sourced Simplex gradient-noise fragment (2D/3D), mirroring the CPU
/// reference [`crate::noise::Simplex::get2`]/[`get3`](crate::noise::Simplex::get3)
/// (Gustavson's public-domain formulation).
///
/// Like [`WGSL_PERLIN`], the seeded 512-entry permutation table is **uploaded,
/// not rebuilt** on the device at `@binding(1)`, so the integer hash path
/// (including the `% 12` gradient index) is bit-exact and both sides pick the
/// identical corner gradient. The 12 edge gradients are a fixed constant table
/// embedded in the shader. Only the skew/unskew, the `(0.5 - r^2)^4` corner
/// falloff, and the dot products are floating point.
///
/// # Parity note (branch sensitivity)
///
/// Which simplex (triangle/tetrahedron) a sample falls in is chosen by
/// floating-point comparisons (`x0 > y0`, the 3D cascade) and by `floor` of the
/// skewed coordinate. Exactly on a simplex boundary a last-ULP fast-math
/// difference could flip the branch and change the value by more than a
/// tolerance, but that boundary set is measure-zero; ordinary samples agree
/// within a small tolerance. Parity tests therefore sample off the boundaries.
///
/// The fragment declares the perm storage binding (needed by
/// `prism_perm_hash`); the compute wrapper supplies the count uniform
/// (`@binding(0)`), the sample input (`@binding(2)`), the output
/// (`@binding(3)`), and `main`. Both the get2 and get3 kernels share this
/// fragment verbatim.
pub const WGSL_SIMPLEX: &str = "\
@group(0) @binding(1) var<storage, read> prism_perm: array<u32>;\n\
\n\
fn prism_perm_hash(i: i32) -> u32 {\n\
    return prism_perm[u32(i & 511)];\n\
}\n\
\n\
fn prism_simplex_grad(idx: u32) -> vec3<f32> {\n\
    switch (idx) {\n\
        case 0u: { return vec3<f32>(1.0, 1.0, 0.0); }\n\
        case 1u: { return vec3<f32>(-1.0, 1.0, 0.0); }\n\
        case 2u: { return vec3<f32>(1.0, -1.0, 0.0); }\n\
        case 3u: { return vec3<f32>(-1.0, -1.0, 0.0); }\n\
        case 4u: { return vec3<f32>(1.0, 0.0, 1.0); }\n\
        case 5u: { return vec3<f32>(-1.0, 0.0, 1.0); }\n\
        case 6u: { return vec3<f32>(1.0, 0.0, -1.0); }\n\
        case 7u: { return vec3<f32>(-1.0, 0.0, -1.0); }\n\
        case 8u: { return vec3<f32>(0.0, 1.0, 1.0); }\n\
        case 9u: { return vec3<f32>(0.0, -1.0, 1.0); }\n\
        case 10u: { return vec3<f32>(0.0, 1.0, -1.0); }\n\
        default: { return vec3<f32>(0.0, -1.0, -1.0); }\n\
    }\n\
}\n\
\n\
fn prism_simplex_grad_index(i: i32) -> u32 {\n\
    return prism_perm_hash(i) % 12u;\n\
}\n\
\n\
fn prism_simplex_corner2(x: f32, y: f32, g: vec3<f32>) -> f32 {\n\
    let t = 0.5 - x * x - y * y;\n\
    if (t < 0.0) { return 0.0; }\n\
    let t2 = t * t;\n\
    return t2 * t2 * (g.x * x + g.y * y);\n\
}\n\
\n\
fn prism_simplex_corner3(x: f32, y: f32, z: f32, g: vec3<f32>) -> f32 {\n\
    let t = 0.6 - x * x - y * y - z * z;\n\
    if (t < 0.0) { return 0.0; }\n\
    let t2 = t * t;\n\
    return t2 * t2 * (g.x * x + g.y * y + g.z * z);\n\
}\n\
\n\
fn prism_simplex_hash3(a: i32, b: i32, c: i32) -> u32 {\n\
    return prism_perm_hash(a + i32(prism_perm_hash(b + i32(prism_perm_hash(c)))));\n\
}\n\
\n\
fn prism_simplex_get2(xin: f32, yin: f32) -> f32 {\n\
    let F2 = 0.36602542;\n\
    let G2 = 0.21132487;\n\
    let s = (xin + yin) * F2;\n\
    let i = floor(xin + s);\n\
    let j = floor(yin + s);\n\
    let t = (i + j) * G2;\n\
    let x0 = xin - (i - t);\n\
    let y0 = yin - (j - t);\n\
    var i1 = 0;\n\
    var j1 = 1;\n\
    if (x0 > y0) { i1 = 1; j1 = 0; }\n\
    let x1 = x0 - f32(i1) + G2;\n\
    let y1 = y0 - f32(j1) + G2;\n\
    let x2 = x0 - 1.0 + 2.0 * G2;\n\
    let y2 = y0 - 1.0 + 2.0 * G2;\n\
    let ii = i32(i);\n\
    let jj = i32(j);\n\
    let gi0 = prism_simplex_grad_index(ii + i32(prism_perm_hash(jj)));\n\
    let gi1 = prism_simplex_grad_index(ii + i1 + i32(prism_perm_hash(jj + j1)));\n\
    let gi2 = prism_simplex_grad_index(ii + 1 + i32(prism_perm_hash(jj + 1)));\n\
    let n0 = prism_simplex_corner2(x0, y0, prism_simplex_grad(gi0));\n\
    let n1 = prism_simplex_corner2(x1, y1, prism_simplex_grad(gi1));\n\
    let n2 = prism_simplex_corner2(x2, y2, prism_simplex_grad(gi2));\n\
    return 70.0 * (n0 + n1 + n2);\n\
}\n\
\n\
fn prism_simplex_get3(xin: f32, yin: f32, zin: f32) -> f32 {\n\
    let F3 = 1.0 / 3.0;\n\
    let G3 = 1.0 / 6.0;\n\
    let s = (xin + yin + zin) * F3;\n\
    let i = floor(xin + s);\n\
    let j = floor(yin + s);\n\
    let k = floor(zin + s);\n\
    let t = (i + j + k) * G3;\n\
    let x0 = xin - (i - t);\n\
    let y0 = yin - (j - t);\n\
    let z0 = zin - (k - t);\n\
    var i1 = 0;\n\
    var j1 = 0;\n\
    var k1 = 0;\n\
    var i2 = 0;\n\
    var j2 = 0;\n\
    var k2 = 0;\n\
    if (x0 >= y0) {\n\
        if (y0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }\n\
        else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }\n\
        else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }\n\
    } else {\n\
        if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }\n\
        else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }\n\
        else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }\n\
    }\n\
    let x1 = x0 - f32(i1) + G3;\n\
    let y1 = y0 - f32(j1) + G3;\n\
    let z1 = z0 - f32(k1) + G3;\n\
    let x2 = x0 - f32(i2) + 2.0 * G3;\n\
    let y2 = y0 - f32(j2) + 2.0 * G3;\n\
    let z2 = z0 - f32(k2) + 2.0 * G3;\n\
    let x3 = x0 - 1.0 + 3.0 * G3;\n\
    let y3 = y0 - 1.0 + 3.0 * G3;\n\
    let z3 = z0 - 1.0 + 3.0 * G3;\n\
    let ii = i32(i);\n\
    let jj = i32(j);\n\
    let kk = i32(k);\n\
    let gi0 = prism_simplex_hash3(ii, jj, kk) % 12u;\n\
    let gi1 = prism_simplex_hash3(ii + i1, jj + j1, kk + k1) % 12u;\n\
    let gi2 = prism_simplex_hash3(ii + i2, jj + j2, kk + k2) % 12u;\n\
    let gi3 = prism_simplex_hash3(ii + 1, jj + 1, kk + 1) % 12u;\n\
    let n0 = prism_simplex_corner3(x0, y0, z0, prism_simplex_grad(gi0));\n\
    let n1 = prism_simplex_corner3(x1, y1, z1, prism_simplex_grad(gi1));\n\
    let n2 = prism_simplex_corner3(x2, y2, z2, prism_simplex_grad(gi2));\n\
    let n3 = prism_simplex_corner3(x3, y3, z3, prism_simplex_grad(gi3));\n\
    return 32.0 * (n0 + n1 + n2 + n3);\n\
}\n";

/// Single-sourced fractal (multi-octave) WGSL fragment: fBm, turbulence, and
/// ridged multifractal, mirroring [`prism_math::noise::Fractal`]'s `fbm2` /
/// `fbm3` / `turbulence2` / `ridged2` operator-for-operator (amplitude-weighted
/// octave sum with `freq *= lacunarity`, `amp *= gain`, normalized by the
/// accumulated amplitude).
///
/// These octave loops call `prism_base_sample2` / `prism_base_sample3`, which
/// are **not** defined here: the host (`prism_math_gpu::fractal`) prepends one
/// base-noise fragment ([`WGSL_PERLIN`] or [`WGSL_SIMPLEX`]) plus a two-line
/// alias that forwards `prism_base_sample*` to that source's `get2` / `get3`.
/// Only one base fragment is prepended per kernel, because both declare the
/// same `@binding(1)` permutation storage array and prepending both would
/// collide. The fractal math is therefore single-sourced and cannot drift from
/// the CPU reference.
pub const WGSL_FRACTAL: &str = "\
fn prism_fbm2(x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32, frequency: f32) -> f32 {\n\
    var freq = frequency;\n\
    var amp = 1.0;\n\
    var sum = 0.0;\n\
    var norm = 0.0;\n\
    let oct = max(octaves, 1u);\n\
    for (var o = 0u; o < oct; o = o + 1u) {\n\
        sum = sum + amp * prism_base_sample2(x * freq, y * freq);\n\
        norm = norm + amp;\n\
        freq = freq * lacunarity;\n\
        amp = amp * gain;\n\
    }\n\
    return sum / norm;\n\
}\n\
\n\
fn prism_fbm3(x: f32, y: f32, z: f32, octaves: u32, lacunarity: f32, gain: f32, frequency: f32) -> f32 {\n\
    var freq = frequency;\n\
    var amp = 1.0;\n\
    var sum = 0.0;\n\
    var norm = 0.0;\n\
    let oct = max(octaves, 1u);\n\
    for (var o = 0u; o < oct; o = o + 1u) {\n\
        sum = sum + amp * prism_base_sample3(x * freq, y * freq, z * freq);\n\
        norm = norm + amp;\n\
        freq = freq * lacunarity;\n\
        amp = amp * gain;\n\
    }\n\
    return sum / norm;\n\
}\n\
\n\
fn prism_turbulence2(x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32, frequency: f32) -> f32 {\n\
    var freq = frequency;\n\
    var amp = 1.0;\n\
    var sum = 0.0;\n\
    var norm = 0.0;\n\
    let oct = max(octaves, 1u);\n\
    for (var o = 0u; o < oct; o = o + 1u) {\n\
        sum = sum + amp * abs(prism_base_sample2(x * freq, y * freq));\n\
        norm = norm + amp;\n\
        freq = freq * lacunarity;\n\
        amp = amp * gain;\n\
    }\n\
    return sum / norm;\n\
}\n\
\n\
fn prism_ridged2(x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32, frequency: f32) -> f32 {\n\
    var freq = frequency;\n\
    var amp = 1.0;\n\
    var sum = 0.0;\n\
    var norm = 0.0;\n\
    let oct = max(octaves, 1u);\n\
    for (var o = 0u; o < oct; o = o + 1u) {\n\
        let n = 1.0 - abs(prism_base_sample2(x * freq, y * freq));\n\
        sum = sum + amp * n * n;\n\
        norm = norm + amp;\n\
        freq = freq * lacunarity;\n\
        amp = amp * gain;\n\
    }\n\
    return sum / norm;\n\
}\n\
";

/// Single-sourced WGSL for the scalar easing-function family, mirroring the
/// CPU reference [`crate::curve::easing`].
///
/// Each `prism_ease_*` helper remaps a scalar parameter `t` and the
/// `prism_ease(op, t)` dispatcher selects one by op code (the same ordering as
/// the host `Ease` enum). The polynomial easings (smooth/smoother/quad/cubic)
/// are bare multiply/add arithmetic and match the CPU path to a tight FMA
/// tolerance; the sinusoidal and exponential easings call the WGSL `cos`/`sin`
/// / `pow` builtins, which Metal compiles under fast-math, whereas the CPU
/// reference uses deterministic `libm`. The §24.1 contract on those is
/// therefore a small absolute+relative tolerance rather than a bit contract.
/// The clamp and the `t <= 0`/`t >= 1`/`t < 0.5` branch literals are identical
/// on both sides, so a given input takes the same branch.
pub const WGSL_EASING: &str = "\
fn prism_ease_smoothstep(t0: f32) -> f32 {\n\
    let t = clamp(t0, 0.0, 1.0);\n\
    return t * t * (3.0 - 2.0 * t);\n\
}\n\
\n\
fn prism_ease_smootherstep(t0: f32) -> f32 {\n\
    let t = clamp(t0, 0.0, 1.0);\n\
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);\n\
}\n\
\n\
fn prism_ease_quad_in(t: f32) -> f32 {\n\
    return t * t;\n\
}\n\
\n\
fn prism_ease_quad_out(t: f32) -> f32 {\n\
    return t * (2.0 - t);\n\
}\n\
\n\
fn prism_ease_quad_in_out(t: f32) -> f32 {\n\
    if (t < 0.5) {\n\
        return 2.0 * t * t;\n\
    }\n\
    let u = -2.0 * t + 2.0;\n\
    return 1.0 - u * u * 0.5;\n\
}\n\
\n\
fn prism_ease_cubic_in(t: f32) -> f32 {\n\
    return t * t * t;\n\
}\n\
\n\
fn prism_ease_cubic_out(t: f32) -> f32 {\n\
    let u = 1.0 - t;\n\
    return 1.0 - u * u * u;\n\
}\n\
\n\
fn prism_ease_cubic_in_out(t: f32) -> f32 {\n\
    if (t < 0.5) {\n\
        return 4.0 * t * t * t;\n\
    }\n\
    let u = -2.0 * t + 2.0;\n\
    return 1.0 - u * u * u * 0.5;\n\
}\n\
\n\
fn prism_ease_sine_in(t: f32) -> f32 {\n\
    return 1.0 - cos(t * 1.5707964);\n\
}\n\
\n\
fn prism_ease_sine_out(t: f32) -> f32 {\n\
    return sin(t * 1.5707964);\n\
}\n\
\n\
fn prism_ease_sine_in_out(t: f32) -> f32 {\n\
    return -0.5 * (cos(3.1415927 * t) - 1.0);\n\
}\n\
\n\
fn prism_ease_expo_in(t: f32) -> f32 {\n\
    if (t <= 0.0) {\n\
        return 0.0;\n\
    }\n\
    return pow(2.0, 10.0 * (t - 1.0));\n\
}\n\
\n\
fn prism_ease_expo_out(t: f32) -> f32 {\n\
    if (t >= 1.0) {\n\
        return 1.0;\n\
    }\n\
    return 1.0 - pow(2.0, -10.0 * t);\n\
}\n\
\n\
fn prism_ease_expo_in_out(t: f32) -> f32 {\n\
    if (t <= 0.0) {\n\
        return 0.0;\n\
    }\n\
    if (t >= 1.0) {\n\
        return 1.0;\n\
    }\n\
    if (t < 0.5) {\n\
        return 0.5 * pow(2.0, 20.0 * t - 10.0);\n\
    }\n\
    return 1.0 - 0.5 * pow(2.0, -20.0 * t + 10.0);\n\
}\n\
\n\
fn prism_ease(op: u32, t: f32) -> f32 {\n\
    switch (op) {\n\
        case 0u: { return prism_ease_smoothstep(t); }\n\
        case 1u: { return prism_ease_smootherstep(t); }\n\
        case 2u: { return prism_ease_quad_in(t); }\n\
        case 3u: { return prism_ease_quad_out(t); }\n\
        case 4u: { return prism_ease_quad_in_out(t); }\n\
        case 5u: { return prism_ease_cubic_in(t); }\n\
        case 6u: { return prism_ease_cubic_out(t); }\n\
        case 7u: { return prism_ease_cubic_in_out(t); }\n\
        case 8u: { return prism_ease_sine_in(t); }\n\
        case 9u: { return prism_ease_sine_out(t); }\n\
        case 10u: { return prism_ease_sine_in_out(t); }\n\
        case 11u: { return prism_ease_expo_in(t); }\n\
        case 12u: { return prism_ease_expo_out(t); }\n\
        case 13u: { return prism_ease_expo_in_out(t); }\n\
        default: { return t; }\n\
    }\n\
}\n";

/// Cubic-spline segment evaluators mirroring [`crate::curve::spline`], acting on
/// `vec3<f32>` control values (the AAA use case being 3D position/velocity
/// curves). Each positional evaluator has a matching `*_tangent` returning the
/// derivative w.r.t. `t`. The dispatcher `prism_spline(op, a, b, c, d, t)`
/// selects a function by op code; `a..d` are the four control vectors in the
/// order each CPU function takes them. Pure multiply/add polynomials, so the
/// parity contract is a tight FMA tolerance with no transcendental calls.
pub const WGSL_SPLINE: &str = "\
fn prism_spline_hermite(p0: vec3<f32>, m0: vec3<f32>, p1: vec3<f32>, m1: vec3<f32>, t: f32) -> vec3<f32> {\n\
    let t2 = t * t;\n\
    let t3 = t2 * t;\n\
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;\n\
    let h10 = t3 - 2.0 * t2 + t;\n\
    let h01 = -2.0 * t3 + 3.0 * t2;\n\
    let h11 = t3 - t2;\n\
    return p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11;\n\
}\n\
\n\
fn prism_spline_hermite_tangent(p0: vec3<f32>, m0: vec3<f32>, p1: vec3<f32>, m1: vec3<f32>, t: f32) -> vec3<f32> {\n\
    let t2 = t * t;\n\
    let h00 = 6.0 * t2 - 6.0 * t;\n\
    let h10 = 3.0 * t2 - 4.0 * t + 1.0;\n\
    let h01 = -6.0 * t2 + 6.0 * t;\n\
    let h11 = 3.0 * t2 - 2.0 * t;\n\
    return p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11;\n\
}\n\
\n\
fn prism_spline_catmull_rom(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {\n\
    let m1 = (p2 - p0) * 0.5;\n\
    let m2 = (p3 - p1) * 0.5;\n\
    return prism_spline_hermite(p1, m1, p2, m2, t);\n\
}\n\
\n\
fn prism_spline_catmull_rom_tangent(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {\n\
    let m1 = (p2 - p0) * 0.5;\n\
    let m2 = (p3 - p1) * 0.5;\n\
    return prism_spline_hermite_tangent(p1, m1, p2, m2, t);\n\
}\n\
\n\
fn prism_spline_bezier_cubic(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {\n\
    let u = 1.0 - t;\n\
    let uu = u * u;\n\
    let tt = t * t;\n\
    let b0 = uu * u;\n\
    let b1 = 3.0 * uu * t;\n\
    let b2 = 3.0 * u * tt;\n\
    let b3 = tt * t;\n\
    return p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;\n\
}\n\
\n\
fn prism_spline_bezier_cubic_tangent(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {\n\
    let u = 1.0 - t;\n\
    let c0 = 3.0 * u * u;\n\
    let c1 = 6.0 * u * t;\n\
    let c2 = 3.0 * t * t;\n\
    return (p1 - p0) * c0 + (p2 - p1) * c1 + (p3 - p2) * c2;\n\
}\n\
\n\
fn prism_spline(op: u32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, t: f32) -> vec3<f32> {\n\
    switch (op) {\n\
        case 0u: { return prism_spline_hermite(a, b, c, d, t); }\n\
        case 1u: { return prism_spline_hermite_tangent(a, b, c, d, t); }\n\
        case 2u: { return prism_spline_catmull_rom(a, b, c, d, t); }\n\
        case 3u: { return prism_spline_catmull_rom_tangent(a, b, c, d, t); }\n\
        case 4u: { return prism_spline_bezier_cubic(a, b, c, d, t); }\n\
        case 5u: { return prism_spline_bezier_cubic_tangent(a, b, c, d, t); }\n\
        default: { return a; }\n\
    }\n\
}\n";

/// Tensor-product spline **surface** evaluators mirroring
/// [`crate::curve::surface`], acting on a 4x4 grid of `vec3<f32>` control
/// points (packed as 16 `vec4` with the value in `xyz`), indexed
/// `g[row*4 + col]` for `[u_row][v_col]`. Provides bicubic Bézier and uniform
/// cubic B-spline patches, each with `sample`, `tangent_u`, `tangent_v`, and
/// `normal`. The Bézier basis reuses `prism_spline_bezier_cubic`/`*_tangent`
/// from [`WGSL_SPLINE`] (compose both fragments), exactly as the CPU surface
/// reuses `curve::spline`; the B-spline basis is defined here. Pure
/// multiply/add polynomials (the `normal` query additionally normalizes with
/// the same `length > 1e-20` guard as the CPU `normalize_or_zero`), so the
/// parity contract is a tight FMA tolerance.
pub const WGSL_SURFACE: &str = "\
fn prism_bspline_cubic(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {\n\
    let t2 = t * t;\n\
    let t3 = t2 * t;\n\
    let b0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0;\n\
    let b1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;\n\
    let b2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;\n\
    let b3 = t3 / 6.0;\n\
    return p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;\n\
}\n\
\n\
fn prism_bspline_cubic_tangent(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {\n\
    let t2 = t * t;\n\
    let b0 = (-3.0 + 6.0 * t - 3.0 * t2) / 6.0;\n\
    let b1 = (-12.0 * t + 9.0 * t2) / 6.0;\n\
    let b2 = (3.0 + 6.0 * t - 9.0 * t2) / 6.0;\n\
    let b3 = (3.0 * t2) / 6.0;\n\
    return p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;\n\
}\n\
\n\
fn prism_surface_normal_from(tu: vec3<f32>, tv: vec3<f32>) -> vec3<f32> {\n\
    let c = cross(tu, tv);\n\
    let len = length(c);\n\
    if (len > 1.0e-20) {\n\
        return c * (1.0 / len);\n\
    }\n\
    return vec3<f32>(0.0, 0.0, 0.0);\n\
}\n\
\n\
fn prism_bezier_patch_sample(g: array<vec4<f32>, 16>, u: f32, v: f32) -> vec3<f32> {\n\
    let q0 = prism_spline_bezier_cubic(g[0].xyz, g[1].xyz, g[2].xyz, g[3].xyz, v);\n\
    let q1 = prism_spline_bezier_cubic(g[4].xyz, g[5].xyz, g[6].xyz, g[7].xyz, v);\n\
    let q2 = prism_spline_bezier_cubic(g[8].xyz, g[9].xyz, g[10].xyz, g[11].xyz, v);\n\
    let q3 = prism_spline_bezier_cubic(g[12].xyz, g[13].xyz, g[14].xyz, g[15].xyz, v);\n\
    return prism_spline_bezier_cubic(q0, q1, q2, q3, u);\n\
}\n\
\n\
fn prism_bezier_patch_tangent_u(g: array<vec4<f32>, 16>, u: f32, v: f32) -> vec3<f32> {\n\
    let q0 = prism_spline_bezier_cubic(g[0].xyz, g[1].xyz, g[2].xyz, g[3].xyz, v);\n\
    let q1 = prism_spline_bezier_cubic(g[4].xyz, g[5].xyz, g[6].xyz, g[7].xyz, v);\n\
    let q2 = prism_spline_bezier_cubic(g[8].xyz, g[9].xyz, g[10].xyz, g[11].xyz, v);\n\
    let q3 = prism_spline_bezier_cubic(g[12].xyz, g[13].xyz, g[14].xyz, g[15].xyz, v);\n\
    return prism_spline_bezier_cubic_tangent(q0, q1, q2, q3, u);\n\
}\n\
\n\
fn prism_bezier_patch_tangent_v(g: array<vec4<f32>, 16>, u: f32, v: f32) -> vec3<f32> {\n\
    let q0 = prism_spline_bezier_cubic_tangent(g[0].xyz, g[1].xyz, g[2].xyz, g[3].xyz, v);\n\
    let q1 = prism_spline_bezier_cubic_tangent(g[4].xyz, g[5].xyz, g[6].xyz, g[7].xyz, v);\n\
    let q2 = prism_spline_bezier_cubic_tangent(g[8].xyz, g[9].xyz, g[10].xyz, g[11].xyz, v);\n\
    let q3 = prism_spline_bezier_cubic_tangent(g[12].xyz, g[13].xyz, g[14].xyz, g[15].xyz, v);\n\
    return prism_spline_bezier_cubic(q0, q1, q2, q3, u);\n\
}\n\
\n\
fn prism_bezier_patch_normal(g: array<vec4<f32>, 16>, u: f32, v: f32) -> vec3<f32> {\n\
    return prism_surface_normal_from(prism_bezier_patch_tangent_u(g, u, v), prism_bezier_patch_tangent_v(g, u, v));\n\
}\n\
\n\
fn prism_bspline_patch_sample(g: array<vec4<f32>, 16>, u: f32, v: f32) -> vec3<f32> {\n\
    let q0 = prism_bspline_cubic(g[0].xyz, g[1].xyz, g[2].xyz, g[3].xyz, v);\n\
    let q1 = prism_bspline_cubic(g[4].xyz, g[5].xyz, g[6].xyz, g[7].xyz, v);\n\
    let q2 = prism_bspline_cubic(g[8].xyz, g[9].xyz, g[10].xyz, g[11].xyz, v);\n\
    let q3 = prism_bspline_cubic(g[12].xyz, g[13].xyz, g[14].xyz, g[15].xyz, v);\n\
    return prism_bspline_cubic(q0, q1, q2, q3, u);\n\
}\n\
\n\
fn prism_bspline_patch_tangent_u(g: array<vec4<f32>, 16>, u: f32, v: f32) -> vec3<f32> {\n\
    let q0 = prism_bspline_cubic(g[0].xyz, g[1].xyz, g[2].xyz, g[3].xyz, v);\n\
    let q1 = prism_bspline_cubic(g[4].xyz, g[5].xyz, g[6].xyz, g[7].xyz, v);\n\
    let q2 = prism_bspline_cubic(g[8].xyz, g[9].xyz, g[10].xyz, g[11].xyz, v);\n\
    let q3 = prism_bspline_cubic(g[12].xyz, g[13].xyz, g[14].xyz, g[15].xyz, v);\n\
    return prism_bspline_cubic_tangent(q0, q1, q2, q3, u);\n\
}\n\
\n\
fn prism_bspline_patch_tangent_v(g: array<vec4<f32>, 16>, u: f32, v: f32) -> vec3<f32> {\n\
    let q0 = prism_bspline_cubic_tangent(g[0].xyz, g[1].xyz, g[2].xyz, g[3].xyz, v);\n\
    let q1 = prism_bspline_cubic_tangent(g[4].xyz, g[5].xyz, g[6].xyz, g[7].xyz, v);\n\
    let q2 = prism_bspline_cubic_tangent(g[8].xyz, g[9].xyz, g[10].xyz, g[11].xyz, v);\n\
    let q3 = prism_bspline_cubic_tangent(g[12].xyz, g[13].xyz, g[14].xyz, g[15].xyz, v);\n\
    return prism_bspline_cubic(q0, q1, q2, q3, u);\n\
}\n\
\n\
fn prism_bspline_patch_normal(g: array<vec4<f32>, 16>, u: f32, v: f32) -> vec3<f32> {\n\
    return prism_surface_normal_from(prism_bspline_patch_tangent_u(g, u, v), prism_bspline_patch_tangent_v(g, u, v));\n\
}\n\
\n\
fn prism_surface(op: u32, g: array<vec4<f32>, 16>, u: f32, v: f32) -> vec3<f32> {\n\
    switch (op) {\n\
        case 0u: { return prism_bezier_patch_sample(g, u, v); }\n\
        case 1u: { return prism_bezier_patch_tangent_u(g, u, v); }\n\
        case 2u: { return prism_bezier_patch_tangent_v(g, u, v); }\n\
        case 3u: { return prism_bezier_patch_normal(g, u, v); }\n\
        case 4u: { return prism_bspline_patch_sample(g, u, v); }\n\
        case 5u: { return prism_bspline_patch_tangent_u(g, u, v); }\n\
        case 6u: { return prism_bspline_patch_tangent_v(g, u, v); }\n\
        case 7u: { return prism_bspline_patch_normal(g, u, v); }\n\
        default: { return vec3<f32>(0.0, 0.0, 0.0); }\n\
    }\n\
}\n";

/// Single-sourced WGSL for broad-phase primitive overlap tests, mirroring the
/// CPU boolean queries [`crate::intersect::aabb_aabb`] /
/// [`sphere_sphere`](crate::intersect::sphere_sphere) /
/// [`sphere_aabb`](crate::intersect::sphere_aabb). These are the classic
/// collision/culling broad-phase predicates — the GPU twin lets a compute pass
/// prefilter candidate pairs before a narrow phase. Each predicate returns
/// `1u` on overlap and `0u` on disjoint. The math is pure comparisons and dot
/// products on the identical inputs (no transcendental, no normalize), so away
/// from the exact tangency boundary the discrete result agrees with the CPU
/// reference exactly; only a pair whose separation lies within fast-math
/// rounding of touching could flip, which is the standard broad-phase
/// conservative tolerance real engines accept.
pub const WGSL_OVERLAP: &str = "\
fn prism_overlap_aabb_aabb(a_min: vec3<f32>, a_max: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> u32 {\n\
    if (a_min.x <= b_max.x && a_max.x >= b_min.x &&\n\
        a_min.y <= b_max.y && a_max.y >= b_min.y &&\n\
        a_min.z <= b_max.z && a_max.z >= b_min.z) {\n\
        return 1u;\n\
    }\n\
    return 0u;\n\
}\n\
\n\
fn prism_overlap_sphere_sphere(a_center: vec3<f32>, a_radius: f32, b_center: vec3<f32>, b_radius: f32) -> u32 {\n\
    let r = a_radius + b_radius;\n\
    let d = a_center - b_center;\n\
    if (dot(d, d) <= r * r) { return 1u; }\n\
    return 0u;\n\
}\n\
\n\
fn prism_overlap_sphere_aabb(center: vec3<f32>, radius: f32, b_min: vec3<f32>, b_max: vec3<f32>) -> u32 {\n\
    let closest = min(max(center, b_min), b_max);\n\
    let d = center - closest;\n\
    if (dot(d, d) <= radius * radius) { return 1u; }\n\
    return 0u;\n\
}\n";
