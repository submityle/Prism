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
