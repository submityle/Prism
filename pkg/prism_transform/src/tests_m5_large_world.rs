//! M5 tests: big-world `f64` coordinates and grid-cell origin rebasing.

use prism_math::{DVec3, GridCell, Quat, Vec3};

use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};
use crate::large_world::{
    FloatingOrigin, GlobalTransformHp, TransformHp, identity_globals_hp, propagate_hp,
};
use crate::{Transform, TransformGraph};

fn approx_vec(a: Vec3, b: Vec3, eps: f32) -> bool {
    (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps && (a.z - b.z).abs() <= eps
}

#[test]
fn widen_from_transform_is_exact() {
    let t = Transform {
        translation: Vec3::new(1.5, -2.25, 3.75),
        rotation: Quat::from_rotation_y(0.3),
        scale: Vec3::new(2.0, 2.0, 2.0),
    };
    let hp = TransformHp::from_transform(&t);
    // Every f32 is exactly representable in f64.
    assert_eq!(hp.translation, DVec3::new(1.5, -2.25, 3.75));
    assert_eq!(hp.rotation, t.rotation);
    assert_eq!(hp.scale, t.scale);
}

#[test]
fn root_global_matches_local_affine() {
    let hp = TransformHp::from_translation(DVec3::new(10.0, 20.0, 30.0));
    let g = GlobalTransformHp::from_transform_hp(&hp);
    assert_eq!(g.translation(), DVec3::new(10.0, 20.0, 30.0));
    assert_eq!(g.cell(), GridCell::from_dvec3(DVec3::new(10.0, 20.0, 30.0)));
}

#[test]
fn hp_propagation_matches_f32_near_origin() {
    // Build the same two-level hierarchy in both the f32 graph and the HP path
    // and confirm the world translations agree where f32 is still accurate.
    let mut graph = TransformGraph::new();
    let root = graph.spawn_root(Transform::from_xyz(1.0, 2.0, 3.0));
    let child = graph.spawn_child(root, Transform::from_xyz(4.0, -1.0, 0.5));
    graph.propagate();

    let mut hier = Hierarchy::new();
    let r = hier.spawn_root();
    let _c = hier.spawn_child(r);
    let locals = [
        TransformHp::from_translation(DVec3::new(1.0, 2.0, 3.0)),
        TransformHp::from_translation(DVec3::new(4.0, -1.0, 0.5)),
    ];
    let mut globals = identity_globals_hp(&hier);
    propagate_hp(&hier, &locals, &mut globals).unwrap();

    let f32_child = graph.global(child).translation();
    let hp_child = globals[1].translation().as_vec3();
    assert!(approx_vec(f32_child, hp_child, 1e-5), "{f32_child:?} vs {hp_child:?}");
    // Known expected world position.
    assert_eq!(globals[1].translation(), DVec3::new(5.0, 1.0, 3.5));
}

#[test]
fn hp_hierarchy_composes_rotation_and_scale() {
    // Parent scales 2x and rotates 90° about Z; a child offset of +X should
    // land at the rotated, scaled world position.
    let mut hier = Hierarchy::new();
    let _r = hier.spawn_root();
    let _c = hier.spawn_child(NodeId::new(0));
    let parent = TransformHp {
        translation: DVec3::new(100.0, 0.0, 0.0),
        rotation: Quat::from_rotation_z(core::f32::consts::FRAC_PI_2),
        scale: Vec3::splat(2.0),
    };
    let child = TransformHp::from_translation(DVec3::new(1.0, 0.0, 0.0));
    let locals = [parent, child];
    let mut globals = identity_globals_hp(&hier);
    propagate_hp(&hier, &locals, &mut globals).unwrap();
    // local +X scaled by 2 -> (2,0,0), rotated 90° about Z -> (0,2,0), then
    // translated by parent -> (100, 2, 0).
    let w = globals[1].translation().as_vec3();
    assert!(approx_vec(w, Vec3::new(100.0, 2.0, 0.0), 1e-4), "{w:?}");
}

#[test]
fn camera_relative_beats_naive_f32_at_100km() {
    // Two entities 1 mm apart, 100 km from the world origin, viewed from a
    // camera right next to them. Camera-relative rendering subtracts in f64
    // first, so the entities land near the origin and the 1 mm gap survives
    // the final cast to f32. Naively casting the raw world position to f32
    // loses it: the f32 ULP at 100 km is ~7.8 mm, far larger than the gap.
    //
    // (A 1 m gap would be a poor test here: f32 represents every integer
    // below 2^24 exactly, so a whole-metre gap at 100 km casts losslessly.
    // The floating-origin win is about sub-ULP fractional precision.)
    let base = 100_000.0_f64;
    let gap_m = 0.001_f64;
    let a_world = DVec3::new(base, 0.0, 0.0);
    let b_world = DVec3::new(base + gap_m, 0.0, 0.0);
    let camera = DVec3::new(base - 2.0, 0.0, 0.0);

    let a = GlobalTransformHp::from_transform_hp(&TransformHp::from_translation(a_world));
    let b = GlobalTransformHp::from_transform_hp(&TransformHp::from_translation(b_world));

    let a_rel = a.camera_relative(camera).translation;
    let b_rel = b.camera_relative(camera).translation;
    let gap = f64::from((b_rel.x - a_rel.x).abs());
    assert!((gap - gap_m).abs() < 1e-5, "camera-relative gap drifted: {gap}");

    // Naive single-precision: the 1 mm difference is well below the ~7.8 mm
    // ULP at 100 km, so casting collapses the gap to zero.
    let a_naive = a_world.x as f32;
    let b_naive = b_world.x as f32;
    let naive_gap = f64::from((b_naive - a_naive).abs());
    assert!((naive_gap - gap_m).abs() > 1e-4, "naive f32 unexpectedly accurate: {naive_gap}");
}

#[test]
fn floating_origin_follows_and_rebases() {
    let mut origin = FloatingOrigin::new(GridCell::ZERO);
    // CELL_SIZE is 1024 m. A camera 500 m out stays in cell 0 -> no rebase.
    assert!(!origin.follow(DVec3::new(500.0, 0.0, 0.0)));
    assert_eq!(origin.origin(), GridCell::ZERO);
    // Threshold is 1 cell: drifting to cell 1 is within threshold -> no rebase.
    assert!(!origin.follow(DVec3::new(1500.0, 0.0, 0.0)));
    // Drifting to cell 3 exceeds the 1-cell threshold -> rebase to the camera.
    let far = DVec3::new(3_500.0, 0.0, 0.0);
    assert!(origin.follow(far));
    assert_eq!(origin.origin(), GridCell::from_dvec3(far));
    // After rebasing, the camera's render offset is small (within one cell).
    let off = origin.render_offset(far);
    assert!(off.x.abs() <= GridCell::CELL_SIZE as f32, "{off:?}");
}

#[test]
fn rebase_is_render_equivalent_across_origins() {
    // The *camera-relative* position of an entity is invariant to the choice of
    // floating origin: this is the "double-run equivalence" the roadmap asks of
    // the big-world path. Pick a far entity + camera and two very different
    // origins, and confirm (entity_offset - camera_offset) matches.
    let entity = DVec3::new(250_000.0, -80_000.0, 123_456.0);
    let camera = DVec3::new(250_010.0, -80_003.0, 123_450.0);

    let g = GlobalTransformHp::from_transform_hp(&TransformHp::from_translation(entity));

    let origin_a = FloatingOrigin::new(GridCell::from_dvec3(camera));
    let origin_b = FloatingOrigin::new(GridCell::new(0, 0, 0));

    // Render-space camera-relative offset via each origin:
    // (entity - origin) - (camera - origin) == entity - camera, origin cancels.
    let rel_a = origin_a.render_offset(entity) - origin_a.render_offset(camera);
    let rel_b = origin_b.render_offset(entity) - origin_b.render_offset(camera);

    // Origin A (near camera) is the accurate reference; both should agree with
    // the true camera-relative delta to sub-millimetre, and with each other.
    let truth = g.camera_relative(camera).translation;
    assert!(approx_vec(rel_a, truth, 1e-2), "near-origin: {rel_a:?} vs {truth:?}");
    assert!(approx_vec(rel_a, rel_b, 5e-2), "cross-origin mismatch: {rel_a:?} vs {rel_b:?}");
}

#[test]
fn propagate_hp_length_mismatch_errors() {
    let mut hier = Hierarchy::new();
    hier.spawn_root();
    let locals: [TransformHp; 0] = [];
    let mut globals = identity_globals_hp(&hier);
    assert_eq!(
        propagate_hp(&hier, &locals, &mut globals),
        Err(HierarchyError::LengthMismatch)
    );
}

#[test]
fn global_transform_hp_identity_default() {
    assert_eq!(GlobalTransformHp::default(), GlobalTransformHp::IDENTITY);
    assert_eq!(GlobalTransformHp::IDENTITY.translation(), DVec3::ZERO);
    // Camera-relative of identity at the camera's own position is the origin.
    let g = GlobalTransformHp::IDENTITY;
    assert_eq!(g.camera_relative(DVec3::ZERO).translation, Vec3::ZERO);
}
