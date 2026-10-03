//! M6 2D tests: hierarchy propagation equals manual affine composition over
//! multi-level trees (including non-uniform scale + rotation), roots pass
//! through unchanged, and the facade matches the free function.

use crate::hierarchy::{Hierarchy, HierarchyError};
use crate::transform_2d::{
    Affine2, GlobalTransform2d, Transform2d, TransformGraph2d, propagate_2d,
};
use prism_math::Vec2;

fn v2_approx(a: Vec2, b: Vec2, eps: f32) -> bool {
    (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps
}

fn root() -> Transform2d {
    Transform2d {
        translation: Vec2::new(10.0, -5.0),
        rotation: 0.5,
        scale: Vec2::new(2.0, 3.0),
    }
}
fn child() -> Transform2d {
    Transform2d {
        translation: Vec2::new(1.0, 2.0),
        rotation: -0.3,
        scale: Vec2::new(1.5, 0.5),
    }
}
fn grandchild() -> Transform2d {
    Transform2d {
        translation: Vec2::new(0.5, 0.5),
        rotation: 0.2,
        scale: Vec2::new(1.0, 4.0),
    }
}

#[test]
fn propagate_2d_matches_manual_composition() {
    let mut h = Hierarchy::new();
    let n0 = h.spawn_root();
    let n1 = h.spawn_child(n0);
    let n2 = h.spawn_child(n1);

    let locals = [root(), child(), grandchild()];
    let mut globals = [GlobalTransform2d::IDENTITY; 3];
    propagate_2d(&h, &locals, &mut globals).unwrap();

    // Independent reference: nest local point maps instead of composing affines.
    let p = Vec2::new(0.3, -0.7);
    let manual_root = root().transform_point(p);
    let manual_child = root().transform_point(child().transform_point(p));
    let manual_gc = root()
        .transform_point(child().transform_point(grandchild().transform_point(p)));

    assert!(v2_approx(globals[n0.index()].transform_point(p), manual_root, 1e-4));
    assert!(v2_approx(globals[n1.index()].transform_point(p), manual_child, 1e-4));
    assert!(v2_approx(globals[n2.index()].transform_point(p), manual_gc, 1e-4));
}

#[test]
fn root_world_equals_its_local_affine() {
    let mut h = Hierarchy::new();
    h.spawn_root();
    let locals = [root()];
    let mut globals = [GlobalTransform2d::IDENTITY; 1];
    propagate_2d(&h, &locals, &mut globals).unwrap();

    let expected = root().to_affine2();
    let got = globals[0].affine();
    assert!(v2_approx(got.matrix2.x_axis, expected.matrix2.x_axis, 1e-6));
    assert!(v2_approx(got.matrix2.y_axis, expected.matrix2.y_axis, 1e-6));
    assert!(v2_approx(got.translation, expected.translation, 1e-6));
}

#[test]
fn identity_transform_is_a_no_op() {
    let a = Affine2::IDENTITY;
    let p = Vec2::new(3.0, -4.0);
    assert!(v2_approx(a.transform_point(p), p, 0.0));

    // Compose identity on either side leaves the other operand unchanged.
    let t = child().to_affine2();
    assert!(v2_approx((Affine2::IDENTITY * t).translation, t.translation, 1e-6));
    assert!(v2_approx((t * Affine2::IDENTITY).translation, t.translation, 1e-6));
}

#[test]
fn affine2_compose_is_associative_on_points() {
    let a = root().to_affine2();
    let b = child().to_affine2();
    let c = grandchild().to_affine2();
    let p = Vec2::new(1.1, -2.2);
    let left = ((a * b) * c).transform_point(p);
    let right = (a * (b * c)).transform_point(p);
    assert!(v2_approx(left, right, 1e-4));
}

#[test]
fn transform_graph_2d_facade_matches_free_function() {
    let mut g = TransformGraph2d::new();
    let r = g.spawn_root(root());
    let c = g.spawn_child(r, child());
    let gc = g.spawn_child(c, grandchild());
    g.propagate();

    let p = Vec2::new(0.3, -0.7);
    let manual_gc = root()
        .transform_point(child().transform_point(grandchild().transform_point(p)));
    assert!(v2_approx(g.global(gc).transform_point(p), manual_gc, 1e-4));

    // Editing the middle node and re-propagating stays correct.
    let new_child = Transform2d::from_xy(4.0, 4.0);
    g.set_local(c, new_child);
    g.propagate();
    let manual_gc2 = root()
        .transform_point(new_child.transform_point(grandchild().transform_point(p)));
    assert!(v2_approx(g.global(gc).transform_point(p), manual_gc2, 1e-4));
}

#[test]
fn propagate_2d_rejects_length_mismatch() {
    let mut h = Hierarchy::new();
    h.spawn_root();
    h.spawn_root();
    let locals = [root()]; // too short on purpose
    let mut globals = [GlobalTransform2d::IDENTITY; 2];
    assert_eq!(
        propagate_2d(&h, &locals, &mut globals),
        Err(HierarchyError::LengthMismatch),
    );
}
