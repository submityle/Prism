//! Oracle tests for the §24.1 shader-mirror byte-layout contract and the
//! mirrored CPU reference op.

use crate::mat::{Mat3, Mat4};
use crate::quat::Quat;
use crate::shader_mirror::{
    pack_mat3_std140, pack_mat4, pack_quat, pack_vec3_std140, pack_vec4, quat_rotate_vec3,
    MAT3_STD140_SIZE, MAT4_STD140_SIZE, NDC_DEPTH_RANGE, QUAT_SIZE, VEC3_STD140_SIZE, VEC4_SIZE,
};
use crate::vec::{Vec3, Vec4};

fn read_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn layout_sizes_match_std140_contract() {
    assert_eq!(MAT4_STD140_SIZE, 64);
    assert_eq!(MAT3_STD140_SIZE, 48);
    assert_eq!(VEC4_SIZE, 16);
    assert_eq!(VEC3_STD140_SIZE, 16);
    assert_eq!(QUAT_SIZE, 16);
    assert_eq!(NDC_DEPTH_RANGE, (0.0, 1.0));
}

#[test]
fn vec4_packs_xyzw_little_endian() {
    let bytes = pack_vec4(Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(read_f32s(&bytes), alloc::vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn vec3_std140_pads_to_sixteen_bytes() {
    let bytes = pack_vec3_std140(Vec3::new(5.0, 6.0, 7.0));
    assert_eq!(bytes.len(), 16);
    assert_eq!(read_f32s(&bytes[..12]), alloc::vec![5.0, 6.0, 7.0]);
    // Trailing 4 bytes are explicit padding.
    assert_eq!(&bytes[12..16], &[0u8; 4]);
}

#[test]
fn quat_packs_scalar_in_w() {
    let bytes = pack_quat(Quat::from_xyzw(0.1, 0.2, 0.3, 0.4));
    let f = read_f32s(&bytes);
    assert_eq!(f, alloc::vec![0.1, 0.2, 0.3, 0.4]);
}

#[test]
fn mat4_is_column_major() {
    // Columns are the axes; a translation matrix carries the translation in
    // the 4th column (w_axis = [tx, ty, tz, 1]).
    let m = Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0));
    let bytes = pack_mat4(m);
    assert_eq!(bytes.len(), 64);
    let f = read_f32s(&bytes);
    // Column 0 (x_axis) = [1,0,0,0]; column 3 (w_axis) = [10,20,30,1].
    assert_eq!(&f[0..4], &[1.0, 0.0, 0.0, 0.0]);
    assert_eq!(&f[4..8], &[0.0, 1.0, 0.0, 0.0]);
    assert_eq!(&f[8..12], &[0.0, 0.0, 1.0, 0.0]);
    assert_eq!(&f[12..16], &[10.0, 20.0, 30.0, 1.0]);
}

#[test]
fn mat3_std140_pads_each_column() {
    let m = Mat3::IDENTITY;
    let bytes = pack_mat3_std140(m);
    assert_eq!(bytes.len(), 48);
    let f = read_f32s(&bytes);
    // Each 16-byte column: [c.x, c.y, c.z, pad].
    assert_eq!(&f[0..4], &[1.0, 0.0, 0.0, 0.0]);
    assert_eq!(&f[4..8], &[0.0, 1.0, 0.0, 0.0]);
    assert_eq!(&f[8..12], &[0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn quat_rotate_matches_quat_mul_vec3() {
    // The mirrored reference must agree with the crate's own quaternion rotate
    // across several axes/angles (CPU parity for the shader twin).
    let cases = [
        (Quat::from_rotation_x(0.7), Vec3::new(0.0, 1.0, 0.0)),
        (Quat::from_rotation_y(1.3), Vec3::new(1.0, 0.0, 0.0)),
        (Quat::from_rotation_z(-0.9), Vec3::new(0.0, 0.0, 1.0)),
        (
            Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 2.1),
            Vec3::new(-2.0, 0.5, 4.0),
        ),
    ];
    for (q, v) in cases {
        let mirror = quat_rotate_vec3(q, v);
        let reference = q.mul_vec3(v);
        let d = mirror - reference;
        assert!(
            d.length() < 1e-5,
            "mirror {mirror:?} vs reference {reference:?}"
        );
    }
}

#[test]
fn identity_quat_rotate_is_passthrough() {
    let v = Vec3::new(3.0, -4.0, 5.0);
    assert_eq!(quat_rotate_vec3(Quat::IDENTITY, v), v);
}
