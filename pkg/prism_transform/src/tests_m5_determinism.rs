//! M5 tests: deterministic fixed-point propagation (bit-exact, double-run equal).

use prism_math::{Fixed, FxVec3};

use crate::determinism::{
    hash_globals, identity_globals_fixed, propagate_fixed, propagate_fixed_in_order, FxAffine3,
    FxMat3,
};
use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};

fn fx(n: i64) -> Fixed {
    Fixed::from_int(n)
}

fn fxv(x: i64, y: i64, z: i64) -> FxVec3 {
    FxVec3::new(fx(x), fx(y), fx(z))
}

fn approx(a: Fixed, b: Fixed, eps: f64) -> bool {
    (a.to_f64() - b.to_f64()).abs() <= eps
}

#[test]
fn identity_is_neutral() {
    let a = FxAffine3::from_translation(fxv(3, 4, 5));
    assert_eq!(FxAffine3::IDENTITY.mul_affine(a), a);
    assert_eq!(a.mul_affine(FxAffine3::IDENTITY), a);
}

#[test]
fn mat3_identity_and_scale_exact() {
    assert_eq!(FxMat3::IDENTITY.mul_vec3(fxv(2, 3, 4)), fxv(2, 3, 4));
    let s = FxMat3::from_scale(fxv(2, 3, 4));
    // Scaling the unit cube corner is exact in integer fixed-point.
    assert_eq!(s.mul_vec3(fxv(1, 1, 1)), fxv(2, 3, 4));
}

#[test]
fn translation_composes_exactly() {
    // Pure translations add with zero error in fixed-point.
    let parent = FxAffine3::from_translation(fxv(10, 0, 0));
    let child = FxAffine3::from_translation(fxv(0, 5, 0));
    let world = parent.mul_affine(child);
    assert_eq!(world.translation, fxv(10, 5, 0));
}

#[test]
fn rotation_z_quarter_turn() {
    // Rotating +X by +90° about Z yields +Y to within the fixed-trig tolerance.
    let m = FxMat3::from_rotation_z(Fixed::FRAC_PI_2);
    let r = m.mul_vec3(FxVec3::X);
    assert!(approx(r.x, Fixed::ZERO, 1e-4), "x={}", r.x.to_f64());
    assert!(approx(r.y, Fixed::ONE, 1e-4), "y={}", r.y.to_f64());
    assert!(approx(r.z, Fixed::ZERO, 1e-4), "z={}", r.z.to_f64());
}

#[test]
fn trs_order_scale_then_rotate_then_translate() {
    // scale 2x, rotate +90° about Z, translate by (100,0,0); apply to +X.
    let a = FxAffine3::from_scale_rotation_z_translation(
        fxv(2, 2, 2),
        Fixed::FRAC_PI_2,
        fxv(100, 0, 0),
    );
    let p = a.transform_point3(FxVec3::X);
    // +X -> scale -> (2,0,0) -> rotate 90°Z -> (0,2,0) -> translate -> (100,2,0)
    assert!(approx(p.x, fx(100), 1e-3), "x={}", p.x.to_f64());
    assert!(approx(p.y, fx(2), 1e-3), "y={}", p.y.to_f64());
    assert!(approx(p.z, Fixed::ZERO, 1e-3), "z={}", p.z.to_f64());
}

fn build_scene() -> (Hierarchy, [FxAffine3; 4]) {
    // root -> a -> b, and root -> c
    let mut hier = Hierarchy::new();
    let root = hier.spawn_root();
    let a = hier.spawn_child(root);
    let _b = hier.spawn_child(a);
    let _c = hier.spawn_child(root);
    let locals = [
        FxAffine3::from_scale_rotation_z_translation(fxv(1, 1, 1), Fixed::FRAC_PI_4, fxv(10, 0, 0)),
        FxAffine3::from_translation(fxv(0, 5, 0)),
        FxAffine3::from_rotation_z(Fixed::FRAC_PI_4),
        FxAffine3::from_translation(fxv(-3, -3, 0)),
    ];
    (hier, locals)
}

#[test]
fn propagation_double_run_is_bit_identical() {
    let (hier, locals) = build_scene();

    let mut g1 = identity_globals_fixed(&hier);
    propagate_fixed(&hier, &locals, &mut g1).unwrap();

    let mut g2 = identity_globals_fixed(&hier);
    propagate_fixed(&hier, &locals, &mut g2).unwrap();

    // Bit-for-bit equal across runs — the determinism guarantee.
    for (x, y) in g1.iter().zip(g2.iter()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
    assert_eq!(hash_globals(&g1), hash_globals(&g2));
}

#[test]
fn hash_detects_single_bit_desync() {
    let (hier, locals) = build_scene();
    let mut g = identity_globals_fixed(&hier);
    propagate_fixed(&hier, &locals, &mut g).unwrap();
    let digest = hash_globals(&g);

    // Perturb one node's translation by a single ULP and re-hash.
    let mut desynced = g.clone();
    let t = desynced[2].translation;
    desynced[2].translation = FxVec3::new(Fixed::from_bits(t.x.to_bits() + 1), t.y, t.z);
    assert_ne!(hash_globals(&desynced), digest);
}

#[test]
fn child_world_is_parent_times_local() {
    let (hier, locals) = build_scene();
    let mut g = identity_globals_fixed(&hier);
    propagate_fixed(&hier, &locals, &mut g).unwrap();
    // node 1 (a) is a child of node 0 (root): world[a] == world[root] * local[a]
    let expected = g[0].mul_affine(locals[1]);
    assert_eq!(g[1].to_bits(), expected.to_bits());
    // node 2 (b) is a child of node 1 (a).
    let expected_b = g[1].mul_affine(locals[2]);
    assert_eq!(g[2].to_bits(), expected_b.to_bits());
}

#[test]
fn roots_pass_through_their_local() {
    let (hier, locals) = build_scene();
    let mut g = identity_globals_fixed(&hier);
    propagate_fixed(&hier, &locals, &mut g).unwrap();
    assert_eq!(g[0].to_bits(), locals[0].to_bits());
}

#[test]
fn in_order_matches_full_pass() {
    let (hier, locals) = build_scene();
    let order = hier.compute_order().unwrap();

    let mut full = identity_globals_fixed(&hier);
    propagate_fixed(&hier, &locals, &mut full).unwrap();

    let mut ordered = identity_globals_fixed(&hier);
    propagate_fixed_in_order(&hier, &order, &locals, &mut ordered);

    for (x, y) in full.iter().zip(ordered.iter()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

#[test]
fn length_mismatch_errors() {
    let mut hier = Hierarchy::new();
    hier.spawn_root();
    let _ = NodeId::new(0);
    let locals: [FxAffine3; 0] = [];
    let mut globals = identity_globals_fixed(&hier);
    assert_eq!(
        propagate_fixed(&hier, &locals, &mut globals),
        Err(HierarchyError::LengthMismatch)
    );
}

#[test]
fn mat3_mul_matches_sequential_vec_transform() {
    // (A*B) applied to v equals A applied to (B applied to v).
    let a = FxMat3::from_rotation_z(Fixed::FRAC_PI_4);
    let b = FxMat3::from_scale(fxv(2, 2, 2));
    let v = fxv(1, 1, 0);
    let combined = a.mul_mat3(b).mul_vec3(v);
    let sequential = a.mul_vec3(b.mul_vec3(v));
    assert_eq!(combined.to_bits(), sequential.to_bits());
}
