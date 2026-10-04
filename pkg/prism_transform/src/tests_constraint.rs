//! §24.6 tests: deterministic lightweight constraints with geometric oracles.

use crate::constraint::{
    look_at_rotation, rotation_arc, solve_chain, Aim, Constraint, LookAt, ParentBlend,
    PositionLimit,
};
use crate::{GlobalTransform, Transform};
use prism_math::{vec3, Quat, Vec3};

fn approx_vec(a: Vec3, b: Vec3, eps: f32) {
    assert!(
        (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps && (a.z - b.z).abs() <= eps,
        "expected {a:?} ≈ {b:?} (eps {eps})"
    );
}

fn at(translation: Vec3) -> GlobalTransform {
    GlobalTransform::from_transform(&Transform::from_translation(translation))
}

/// World forward (`-Z`) direction of a solved world transform.
fn world_forward(g: GlobalTransform) -> Vec3 {
    g.compute_transform().forward()
}

// ---- look-at ---------------------------------------------------------------

#[test]
fn look_at_points_forward_at_target() {
    // Node at origin, target on +X: forward (-Z) must rotate onto +X.
    let out = LookAt::new(vec3(5.0, 0.0, 0.0), Vec3::Y).solve(at(Vec3::ZERO));
    approx_vec(world_forward(out), vec3(1.0, 0.0, 0.0), 1.0e-5);
    // Translation and (unit) scale are preserved.
    approx_vec(out.translation(), Vec3::ZERO, 1.0e-6);
}

#[test]
fn look_at_from_offset_origin() {
    let node = at(vec3(1.0, 2.0, 3.0));
    let target = vec3(1.0, 2.0, -7.0); // straight along -Z from the node
    let out = LookAt::new(target, Vec3::Y).solve(node);
    approx_vec(world_forward(out), vec3(0.0, 0.0, -1.0), 1.0e-5);
}

#[test]
fn look_at_on_target_keeps_current_pose() {
    let node = at(vec3(4.0, 0.0, 0.0));
    // Target coincides with the node: zero view direction, pose unchanged.
    let out = LookAt::new(vec3(4.0, 0.0, 0.0), Vec3::Y).solve(node);
    assert_eq!(out, node);
}

#[test]
fn look_at_rotation_handles_parallel_up() {
    // forward parallel to the up hint must still yield a well-formed basis
    // whose forward matches the request.
    let r = look_at_rotation(vec3(0.0, 1.0, 0.0), Vec3::Y);
    let fwd = -(r * Vec3::Z);
    approx_vec(fwd, vec3(0.0, 1.0, 0.0), 1.0e-5);
}

// ---- aim -------------------------------------------------------------------

#[test]
fn aim_points_local_axis_at_target() {
    // Local +Z should point at a +X target with the minimal rotation.
    let out = Aim::new(Vec3::Z, vec3(5.0, 0.0, 0.0)).solve(at(Vec3::ZERO));
    let axis_world = out.compute_transform().rotation * Vec3::Z;
    approx_vec(axis_world, vec3(1.0, 0.0, 0.0), 1.0e-5);
}

#[test]
fn aim_already_aligned_is_near_identity() {
    // Local +Z already points at +Z target => no net rotation.
    let out = Aim::new(Vec3::Z, vec3(0.0, 0.0, 5.0)).solve(at(Vec3::ZERO));
    let axis_world = out.compute_transform().rotation * Vec3::Z;
    approx_vec(axis_world, vec3(0.0, 0.0, 1.0), 1.0e-5);
}

#[test]
fn rotation_arc_opposite_vectors_flips_180() {
    let r = rotation_arc(Vec3::X, vec3(-1.0, 0.0, 0.0));
    approx_vec(r * Vec3::X, vec3(-1.0, 0.0, 0.0), 1.0e-5);
}

#[test]
fn rotation_arc_identical_vectors_is_identity() {
    let r = rotation_arc(Vec3::X, Vec3::X);
    approx_vec(r * Vec3::X, Vec3::X, 1.0e-6);
    approx_vec(r * Vec3::Y, Vec3::Y, 1.0e-6);
}

// ---- parent-blend ----------------------------------------------------------

#[test]
fn parent_blend_averages_translation_by_weight() {
    let mut blend = ParentBlend::new();
    blend.push(at(Vec3::ZERO), 1.0);
    blend.push(at(vec3(10.0, 0.0, 0.0)), 1.0);
    // Equal weights -> midpoint.
    approx_vec(blend.blend().translation(), vec3(5.0, 0.0, 0.0), 1.0e-5);
}

#[test]
fn parent_blend_weight_bias() {
    let mut blend = ParentBlend::new();
    blend.push(at(Vec3::ZERO), 3.0);
    blend.push(at(vec3(10.0, 0.0, 0.0)), 1.0);
    // Weighted mean: (3*0 + 1*10) / 4 = 2.5.
    approx_vec(blend.blend().translation(), vec3(2.5, 0.0, 0.0), 1.0e-5);
}

#[test]
fn parent_blend_empty_is_identity() {
    assert_eq!(ParentBlend::new().blend(), GlobalTransform::IDENTITY);
}

// ---- position-limit --------------------------------------------------------

#[test]
fn position_limit_clamps_into_box() {
    let limit = PositionLimit::new(vec3(-10.0, -10.0, -10.0), vec3(10.0, 10.0, 10.0));
    let out = limit.solve(at(vec3(100.0, -100.0, 5.0)));
    approx_vec(out.translation(), vec3(10.0, -10.0, 5.0), 1.0e-6);
}

#[test]
fn position_limit_inside_box_is_unchanged() {
    let limit = PositionLimit::new(vec3(-10.0, -10.0, -10.0), vec3(10.0, 10.0, 10.0));
    let node = at(vec3(1.0, 2.0, 3.0));
    assert_eq!(limit.solve(node), node);
}

// ---- chain -----------------------------------------------------------------

#[test]
fn solve_chain_applies_in_order() {
    // First look at +X, then clamp translation into a tiny box around origin.
    let chain = [
        Constraint::LookAt(LookAt::new(vec3(100.0, 0.0, 0.0), Vec3::Y)),
        Constraint::PositionLimit(PositionLimit::new(
            vec3(-1.0, -1.0, -1.0),
            vec3(1.0, 1.0, 1.0),
        )),
    ];
    let out = solve_chain(&chain, at(vec3(50.0, 0.0, 0.0)));
    // Clamp wins on translation; look-at set the forward.
    approx_vec(out.translation(), vec3(1.0, 0.0, 0.0), 1.0e-6);
    approx_vec(world_forward(out), vec3(1.0, 0.0, 0.0), 1.0e-5);
}

#[test]
fn solve_chain_empty_is_identity_on_input() {
    let node = at(vec3(7.0, 8.0, 9.0));
    let chain: [Constraint; 0] = [];
    assert_eq!(solve_chain(&chain, node), node);
}

#[test]
fn constraint_enum_dispatches_like_inner_solver() {
    let node = at(vec3(50.0, 0.0, 0.0));
    let inner = PositionLimit::new(vec3(-1.0, -1.0, -1.0), vec3(1.0, 1.0, 1.0));
    let via_enum = Constraint::PositionLimit(inner).solve(node);
    assert_eq!(via_enum, inner.solve(node));
}

#[test]
fn look_at_rotation_degenerate_direction_is_identity() {
    assert_eq!(look_at_rotation(Vec3::ZERO, Vec3::Y), Quat::IDENTITY);
}
