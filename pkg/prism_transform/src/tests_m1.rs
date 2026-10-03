//! M1 tests: hierarchy propagation, change ticks, acyclicity, and reparenting.

use crate::change::ChangeTicks;
use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};
use crate::propagation::propagate;
use crate::{GlobalTransform, Transform, TransformGraph};
use prism_math::{Affine3, Quat, Vec3, vec3};

fn v_approx(a: Vec3, b: Vec3, eps: f32) -> bool {
    (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps && (a.z - b.z).abs() <= eps
}

fn affine_approx(a: Affine3, b: Affine3, eps: f32) -> bool {
    v_approx(a.matrix3.x_axis, b.matrix3.x_axis, eps)
        && v_approx(a.matrix3.y_axis, b.matrix3.y_axis, eps)
        && v_approx(a.matrix3.z_axis, b.matrix3.z_axis, eps)
        && v_approx(a.translation, b.translation, eps)
}

// ---- deep parent-child world-pose correctness -----------------------------

#[test]
fn deep_translation_chain_four_levels() {
    let mut g = TransformGraph::new();
    let root = g.spawn_root(Transform::from_xyz(10.0, 0.0, 0.0));
    let c1 = g.spawn_child(root, Transform::from_xyz(0.0, 5.0, 0.0));
    let c2 = g.spawn_child(c1, Transform::from_xyz(0.0, 0.0, 3.0));
    let c3 = g.spawn_child(c2, Transform::from_xyz(1.0, 0.0, 0.0));
    g.propagate();

    assert!(v_approx(g.global(root).translation(), vec3(10.0, 0.0, 0.0), 1e-5));
    assert!(v_approx(g.global(c1).translation(), vec3(10.0, 5.0, 0.0), 1e-5));
    assert!(v_approx(g.global(c2).translation(), vec3(10.0, 5.0, 3.0), 1e-5));
    assert!(v_approx(g.global(c3).translation(), vec3(11.0, 5.0, 3.0), 1e-5));
}

#[test]
fn deep_rotation_chain_matches_affine_chain() {
    // Root rotates +90deg about Z; children carry offsets and spins. Compare
    // the propagated worlds against an explicit affine composition chain.
    let a = Transform::from_xyz(1.0, 2.0, 0.0)
        .with_rotation(Quat::from_rotation_z(core::f32::consts::FRAC_PI_2));
    let b = Transform::from_xyz(2.0, 0.0, 0.0).with_rotation(Quat::from_rotation_x(0.5));
    let c = Transform::from_xyz(0.0, 1.0, 4.0).with_rotation(Quat::from_rotation_y(0.7));

    let mut g = TransformGraph::new();
    let na = g.spawn_root(a);
    let nb = g.spawn_child(na, b);
    let nc = g.spawn_child(nb, c);
    g.propagate();

    let wa = a.to_affine();
    let wb = wa * b.to_affine();
    let wc = wb * c.to_affine();

    assert!(affine_approx(g.global(na).affine(), wa, 1e-5));
    assert!(affine_approx(g.global(nb).affine(), wb, 1e-5));
    assert!(affine_approx(g.global(nc).affine(), wc, 1e-5));

    // Spot-check a point transformed through the deepest node.
    let p = vec3(1.0, -2.0, 0.5);
    assert!(v_approx(g.global(nc).transform_point(p), wc.transform_point3(p), 1e-4));
}

#[test]
fn root_world_equals_local_affine() {
    let local = Transform {
        translation: vec3(3.0, -1.0, 2.0),
        rotation: Quat::from_axis_angle(vec3(0.2, 1.0, 0.4).normalize(), 0.9),
        scale: vec3(2.0, 0.5, 1.5),
    };
    let mut g = TransformGraph::new();
    let root = g.spawn_root(local);
    g.propagate();
    assert!(affine_approx(g.global(root).affine(), local.to_affine(), 1e-6));
}

// ---- non-uniform scale composing into a child world matrix ----------------

#[test]
fn non_uniform_parent_scale_composes_into_child() {
    // A non-uniform parent scale combined with a child rotation produces shear
    // in the child's world matrix that a pure TRS cannot hold. The affine
    // propagation must match the exact affine chain, and the recovered TRS must
    // NOT (proving we never do a lossy TRS writeback).
    let parent = Transform::from_scale(vec3(3.0, 1.0, 1.0));
    let child = Transform::from_xyz(1.0, 1.0, 0.0)
        .with_rotation(Quat::from_rotation_z(core::f32::consts::FRAC_PI_4));

    let mut g = TransformGraph::new();
    let p = g.spawn_root(parent);
    let c = g.spawn_child(p, child);
    g.propagate();

    let expected = parent.to_affine() * child.to_affine();
    assert!(affine_approx(g.global(c).affine(), expected, 1e-5));

    // The world translation picks up the parent's non-uniform scale: child
    // local translation (1,1,0) scaled by (3,1,1) -> (3,1,0).
    assert!(v_approx(g.global(c).translation(), vec3(3.0, 1.0, 0.0), 1e-5));

    // Confirm shear is actually present: re-composing the recovered TRS loses
    // information, so it would differ from the true affine (hence no writeback).
    let recovered = g.global(c).compute_transform();
    assert!(!affine_approx(recovered.to_affine(), expected, 1e-3));
}

// ---- change-tick / dirty behavior -----------------------------------------

#[test]
fn change_ticks_track_edits_and_passes() {
    let mut g = TransformGraph::new();
    let root = g.spawn_root(Transform::IDENTITY);
    let child = g.spawn_child(root, Transform::from_xyz(1.0, 0.0, 0.0));

    // Freshly spawned nodes are dirty until the first pass computes them.
    assert!(g.is_changed(root));
    assert!(g.is_changed(child));

    g.propagate();
    assert!(!g.is_changed(root));
    assert!(!g.is_changed(child));

    // Editing one local marks only that node.
    g.set_local(child, Transform::from_xyz(2.0, 0.0, 0.0));
    assert!(g.is_changed(child));
    assert!(!g.is_changed(root));

    g.propagate();
    assert!(!g.is_changed(child));
    assert!(v_approx(g.global(child).translation(), vec3(2.0, 0.0, 0.0), 1e-6));
}

#[test]
fn change_ticks_are_monotonic_across_passes() {
    let mut t = ChangeTicks::new();
    t.push();
    let n = NodeId::new(0);
    assert!(t.is_changed(n));
    let before = t.current_tick();
    t.end_pass();
    assert!(!t.is_changed(n));
    assert!(t.current_tick().get() > before.get());
    t.mark(n);
    assert!(t.is_changed(n));
}

// ---- reparenting -----------------------------------------------------------

#[test]
fn reparent_marks_child_changed() {
    let mut g = TransformGraph::new();
    let a = g.spawn_root(Transform::from_xyz(10.0, 0.0, 0.0));
    let b = g.spawn_root(Transform::from_xyz(0.0, 20.0, 0.0));
    let c = g.spawn_child(a, Transform::from_xyz(1.0, 0.0, 0.0));
    g.propagate();
    assert!(!g.is_changed(c));

    g.reparent(c, Some(b)).unwrap();
    assert!(g.is_changed(c));
    assert_eq!(g.hierarchy().parent(c), Some(b));

    // Local unchanged -> world moves under the new parent.
    g.propagate();
    assert!(v_approx(g.global(c).translation(), vec3(1.0, 20.0, 0.0), 1e-5));
}

#[test]
fn reparent_keeping_world_preserves_pose() {
    let mut g = TransformGraph::new();
    let a = g.spawn_root(Transform::from_xyz(10.0, 0.0, 0.0));
    let b = g.spawn_root(
        Transform::from_xyz(0.0, 20.0, 0.0)
            .with_rotation(Quat::from_rotation_z(core::f32::consts::FRAC_PI_2))
            .with_scale(Vec3::splat(2.0)), // uniform scale keeps TRS recovery exact
    );
    let c = g.spawn_child(a, Transform::from_xyz(1.0, 0.0, 0.0));
    g.propagate();
    let world_before = g.global(c).translation();

    g.reparent_keeping_world(c, Some(b)).unwrap();
    assert_eq!(g.hierarchy().parent(c), Some(b));
    let world_after = g.global(c).translation();
    assert!(v_approx(world_before, world_after, 1e-4));

    // Detaching back to a root likewise preserves the world pose.
    g.reparent_keeping_world(c, None).unwrap();
    assert_eq!(g.hierarchy().parent(c), None);
    assert!(v_approx(g.global(c).translation(), world_before, 1e-4));
}

// ---- acyclicity validation -------------------------------------------------

#[test]
fn from_parents_rejects_cycles() {
    let a = NodeId::new(0);
    let b = NodeId::new(1);
    // 0's parent is 1 and 1's parent is 0 -> pure cycle, no roots.
    let err = Hierarchy::from_parents(&[Some(b), Some(a)]).unwrap_err();
    assert_eq!(err, HierarchyError::Cycle);

    // Self-parent is also a cycle.
    let err = Hierarchy::from_parents(&[Some(a)]).unwrap_err();
    assert_eq!(err, HierarchyError::Cycle);
}

#[test]
fn from_parents_rejects_out_of_range_parent() {
    let bogus = NodeId::new(5);
    let err = Hierarchy::from_parents(&[None, Some(bogus)]).unwrap_err();
    assert_eq!(err, HierarchyError::InvalidParent);
}

#[test]
fn from_parents_accepts_forest_with_stable_order() {
    // 0 root; 1,2 children of 0; 3 child of 1.
    let r = NodeId::new(0);
    let n1 = NodeId::new(1);
    let h = Hierarchy::from_parents(&[None, Some(r), Some(r), Some(n1)]).unwrap();
    let order = h.compute_order().unwrap();
    assert_eq!(order.len(), 4);
    // Every parent precedes its children.
    for &node in &order {
        if let Some(parent) = h.parent(node) {
            let pi = order.iter().position(|&x| x == parent).unwrap();
            let ci = order.iter().position(|&x| x == node).unwrap();
            assert!(pi < ci, "parent must precede child");
        }
    }
}

#[test]
fn set_parent_rejects_cycle_and_invalid() {
    let mut h = Hierarchy::new();
    let root = h.spawn_root();
    let child = h.spawn_child(root);
    let grand = h.spawn_child(child);

    // Making the root a child of its own grandchild is a cycle.
    assert_eq!(h.set_parent(root, Some(grand)), Err(HierarchyError::Cycle));
    // A node cannot parent itself.
    assert_eq!(h.set_parent(child, Some(child)), Err(HierarchyError::Cycle));
    // Out-of-bounds ids are rejected.
    assert_eq!(h.set_parent(NodeId::new(99), None), Err(HierarchyError::InvalidNode));
    assert_eq!(h.set_parent(child, Some(NodeId::new(99))), Err(HierarchyError::InvalidNode));

    // A legal re-parent (grand -> root) still works and stays acyclic.
    assert!(h.set_parent(grand, Some(root)).is_ok());
    assert!(h.validate().is_ok());
}

// ---- free-function propagation + length checks ----------------------------

#[test]
fn propagate_reports_length_mismatch() {
    let mut h = Hierarchy::new();
    let _ = h.spawn_root();
    let locals = [Transform::IDENTITY];
    let mut globals: [GlobalTransform; 0] = [];
    assert_eq!(
        propagate(&h, &locals, &mut globals),
        Err(HierarchyError::LengthMismatch),
    );
}
