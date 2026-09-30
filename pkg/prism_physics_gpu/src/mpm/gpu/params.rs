//! Host-side packing helpers that lay `glam` vectors and matrices out to match
//! the `std430` uploads the MLS-MPM WGSL kernels expect, and read them back.
//!
//! A WGSL `mat3x3<f32>` is stored as three 16-byte-aligned columns (each a
//! `vec3` padded to a `vec4`), so a `glam::Mat3` is uploaded as `[[f32; 4]; 3]`
//! and read back the same way. A `glam::Vec3` binding is padded to a `vec4`.
//!
//! # Provenance
//!
//! Plain data marshalling; no Unreal Engine source or derived code.

use glam::{Mat3, Vec3};

/// Packs a [`Vec3`] into a padded `vec4` upload element.
#[must_use]
pub fn vec3_to_vec4(v: &Vec3) -> [f32; 4] {
    [v.x, v.y, v.z, 0.0]
}

/// Packs a [`Mat3`] into three padded column vectors, matching the
/// `mat3x3<f32>` `std430` layout (each column is a 16-byte-aligned `vec3`).
#[must_use]
pub fn mat3_to_cols(m: &Mat3) -> [[f32; 4]; 3] {
    [
        [m.x_axis.x, m.x_axis.y, m.x_axis.z, 0.0],
        [m.y_axis.x, m.y_axis.y, m.y_axis.z, 0.0],
        [m.z_axis.x, m.z_axis.y, m.z_axis.z, 0.0],
    ]
}

/// Rebuilds a [`Mat3`] from the three padded columns read back from a
/// `mat3x3<f32>` storage buffer.
#[must_use]
pub fn cols_to_mat3(cols: &[[f32; 4]; 3]) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(cols[0][0], cols[0][1], cols[0][2]),
        Vec3::new(cols[1][0], cols[1][1], cols[1][2]),
        Vec3::new(cols[2][0], cols[2][1], cols[2][2]),
    )
}
