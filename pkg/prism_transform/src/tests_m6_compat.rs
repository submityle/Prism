//! M6 compat-bevy tests: the glam-named aliases resolve to the Prism types,
//! the Bevy-shaped extension methods agree with the native spellings, and the
//! prelude surfaces everything a ported call site reaches for.

use crate::compat_bevy::prelude::*;
use prism_math::{Affine3, vec3};

#[test]
fn affine3a_alias_is_prism_affine3() {
    // If the alias were a different type this assignment would not compile.
    let a: Affine3A = Affine3::IDENTITY;
    assert_eq!(a, Affine3::IDENTITY);
}

#[test]
fn glam_named_math_is_reexported() {
    let v: Vec3 = vec3(1.0, 2.0, 3.0);
    let _m: Mat4 = Mat4::IDENTITY;
    let _m2: Mat2 = Mat2::IDENTITY;
    let _q: Quat = Quat::IDENTITY;
    assert_eq!(v.x, 1.0);
}

#[test]
fn bevy_method_spellings_match_native() {
    let t = Transform::from_xyz(3.0, 4.0, 5.0).with_scale(vec3(2.0, 2.0, 2.0));
    let g = GlobalTransform::from_transform(&t);

    assert_eq!(g.compute_matrix(), g.to_matrix());
    assert_eq!(g.compute_affine(), g.affine());

    // compute_transform round-trips the translation (no shear here).
    let back = g.compute_transform();
    assert!((back.translation.x - 3.0).abs() < 1e-5);
    assert!((back.translation.y - 4.0).abs() < 1e-5);
    assert!((back.translation.z - 5.0).abs() < 1e-5);
}

#[test]
fn prelude_surfaces_the_2d_types() {
    let t = Transform2d::from_xy(1.0, 2.0);
    let g = GlobalTransform2d::from_transform(&t);
    assert_eq!(g.translation().x, 1.0);
    assert_eq!(g.translation().y, 2.0);
}
