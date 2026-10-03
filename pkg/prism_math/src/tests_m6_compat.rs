//! M6 compat tests: the `compat-bevy` glam-shaped aliases.
//!
//! These are gated on the `compat-bevy` feature. They verify that the aliases
//! name the exact Prism types (so values are interchangeable), that the
//! glam-only spelling `Affine3A` works as the aligned affine transform, and
//! that the glam-style prelude import compiles and constructs usable values.

use crate::compat_bevy as cb;

fn approx(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn aliases_are_same_types() {
    // Each alias must be the very same type as the Prism original: a value of
    // the alias type is assignable to the original without conversion.
    let v2: cb::Vec2 = crate::Vec2::new(1.0, 2.0);
    let _: crate::Vec2 = v2;
    let v3: cb::Vec3 = crate::Vec3::new(1.0, 2.0, 3.0);
    let _: crate::Vec3 = v3;
    let v3a: cb::Vec3A = crate::Vec3A::new(1.0, 2.0, 3.0);
    let _: crate::Vec3A = v3a;
    let v4: cb::Vec4 = crate::Vec4::new(1.0, 2.0, 3.0, 4.0);
    let _: crate::Vec4 = v4;

    let q: cb::Quat = crate::Quat::IDENTITY;
    let _: crate::Quat = q;

    // The glam-only spelling of the aligned affine transform.
    let a: cb::Affine3A = crate::Affine3::IDENTITY;
    let _: crate::Affine3 = a;

    let dv: cb::DVec3 = crate::DVec3::new(1.0, 2.0, 3.0);
    let _: crate::DVec3 = dv;
    let da: cb::DAffine3 = crate::DAffine3::IDENTITY;
    let _: crate::DAffine3 = da;
}

#[test]
fn affine3a_alias_transforms_like_affine3() {
    // Build a rotation+translation via the alias and transform a point: the
    // alias must behave exactly like the underlying Affine3.
    let rot = crate::Quat::from_rotation_z(core::f32::consts::FRAC_PI_2);
    let a: cb::Affine3A = crate::Affine3::from_scale_rotation_translation(
        crate::Vec3::ONE,
        rot,
        crate::Vec3::new(10.0, 0.0, 0.0),
    );
    let p = a.transform_point3(crate::Vec3::new(1.0, 0.0, 0.0));
    // +X rotated 90deg about +Z -> +Y, then translated by (10,0,0).
    assert!(approx(p.x, 10.0, 1.0e-5));
    assert!(approx(p.y, 1.0, 1.0e-5));
    assert!(approx(p.z, 0.0, 1.0e-5));
}

#[test]
fn prelude_import_constructs_values() {
    use cb::prelude::*;

    let v = vec3(1.0, 2.0, 3.0);
    let w: Vec3 = v;
    assert!(approx(w.x, 1.0, 0.0));

    let a: Vec3A = vec3a(4.0, 5.0, 6.0);
    assert!(approx(a.y, 5.0, 0.0));

    let d: DVec3 = dvec3(7.0, 8.0, 9.0);
    assert!((d.z - 9.0).abs() <= 0.0);

    let q = Quat::IDENTITY;
    let r = q.mul_vec3(v);
    assert!(approx(r.x, v.x, 1.0e-6));

    let m: Mat4 = Mat4::IDENTITY;
    let _ = m;
    let affine: Affine3A = Affine3A::IDENTITY;
    let _ = affine;
}
