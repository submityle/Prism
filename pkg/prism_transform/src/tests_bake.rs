//! §24.1 tests: static baking equals a level-by-level propagation oracle,
//! batch merging groups and world-bakes correctly, and invalidate/unfreeze
//! behave as documented, plus boundary cases.

use crate::bake::{
    merge_instances, merge_meshes, BatchKey, MeshSource, StaticBaker,
};
use crate::hierarchy::{HierarchyError, NodeId};
use crate::propagation::identity_globals;
use crate::{GlobalTransform, Transform, TransformGraph};
use prism_math::{vec3, Quat, Vec3};

// ---- helpers ---------------------------------------------------------------

fn v_approx(a: Vec3, b: Vec3, eps: f32) -> bool {
    (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps && (a.z - b.z).abs() <= eps
}

fn g_approx(a: GlobalTransform, b: GlobalTransform, eps: f32) -> bool {
    let a = a.affine();
    let b = b.affine();
    v_approx(a.matrix3.x_axis, b.matrix3.x_axis, eps)
        && v_approx(a.matrix3.y_axis, b.matrix3.y_axis, eps)
        && v_approx(a.matrix3.z_axis, b.matrix3.z_axis, eps)
        && v_approx(a.translation, b.translation, eps)
}

fn rot_z(deg: f32) -> Quat {
    Quat::from_rotation_z(deg.to_radians())
}

/// Build a graph with scale + rotation + translation at several levels so the
/// baked affine actually exercises composition (not just translation adds).
/// Returns the graph and the node ids `[root, a, b, a0, a1, b0]`.
fn build_graph() -> (TransformGraph, [NodeId; 6]) {
    let mut g = TransformGraph::new();
    let root = g.spawn_root(
        Transform::from_xyz(1.0, 2.0, 3.0)
            .with_rotation(rot_z(30.0))
            .with_scale(vec3(2.0, 2.0, 2.0)),
    );
    let a = g.spawn_child(root, Transform::from_xyz(1.0, 0.0, 0.0).with_rotation(rot_z(15.0)));
    let b = g.spawn_child(root, Transform::from_xyz(0.0, 1.0, 0.0).with_scale(vec3(0.5, 1.0, 1.0)));
    let a0 = g.spawn_child(a, Transform::from_xyz(0.0, 0.5, 0.0));
    let a1 = g.spawn_child(a, Transform::from_xyz(0.0, 0.0, 0.5).with_rotation(rot_z(45.0)));
    let b0 = g.spawn_child(b, Transform::from_xyz(0.3, 0.0, 0.0));
    g.propagate();
    (g, [root, a, b, a0, a1, b0])
}

// ---- baking correctness ----------------------------------------------------

#[test]
fn bake_whole_tree_matches_full_propagation_oracle() {
    let (g, ids) = build_graph();

    // Mark the entire tree static.
    let mut baker = StaticBaker::with_len(g.len());
    for &id in &ids {
        baker.mark_static(id);
    }

    // Fully static scene: all static roots are world roots, so identity seeds.
    let seeds = identity_globals(g.hierarchy());
    let locals: Vec<Transform> =
        (0..g.len()).map(|i| g.local(NodeId::new(i as u32))).collect();
    let stats = baker.bake(g.hierarchy(), &locals, &seeds).unwrap();

    assert_eq!(stats.baked, g.len());
    assert_eq!(stats.roots, 1, "a single-rooted tree has one static root");

    // Oracle: the graph's own full propagation.
    for &id in &ids {
        assert!(
            g_approx(baker.baked(id), g.global(id), 1e-4),
            "baked world transform for node {} disagrees with propagation",
            id.index()
        );
    }
}

#[test]
fn bake_static_subtree_hanging_off_dynamic_parent_seeds_from_parent_world() {
    let (g, ids) = build_graph();
    let [root, a, _b, a0, a1, _b0] = ids;

    // Only the subtree rooted at `a` (a, a0, a1) is static; root/b/b0 dynamic.
    let mut baker = StaticBaker::with_len(g.len());
    baker.mark_static(a);
    baker.mark_static(a0);
    baker.mark_static(a1);

    // The static root `a` has a *dynamic* parent (root); seed from its world.
    let locals: Vec<Transform> =
        (0..g.len()).map(|i| g.local(NodeId::new(i as u32))).collect();
    let stats = baker.bake(g.hierarchy(), &locals, g.globals()).unwrap();

    assert_eq!(stats.roots, 1);
    assert_eq!(stats.baked, 3);
    assert_eq!(baker.static_roots(g.hierarchy()), alloc::vec![a]);

    for &id in &[a, a0, a1] {
        assert!(g_approx(baker.baked(id), g.global(id), 1e-4));
    }
    // Dynamic nodes were never baked.
    assert!(baker.try_baked(root).is_none());
}

#[test]
fn two_disjoint_static_components_bake_independently() {
    let (g, ids) = build_graph();
    let [_root, a, b, a0, a1, b0] = ids;

    // Static set {a, a0, a1} and {b, b0}: two components, both under dynamic root.
    let mut baker = StaticBaker::with_len(g.len());
    for id in [a, a0, a1, b, b0] {
        baker.mark_static(id);
    }
    let locals: Vec<Transform> =
        (0..g.len()).map(|i| g.local(NodeId::new(i as u32))).collect();
    let stats = baker.bake(g.hierarchy(), &locals, g.globals()).unwrap();

    assert_eq!(stats.roots, 2);
    assert_eq!(stats.baked, 5);
    let mut roots = baker.static_roots(g.hierarchy());
    roots.sort_by_key(|n| n.index());
    assert_eq!(roots, alloc::vec![a, b]);
    for id in [a, a0, a1, b, b0] {
        assert!(g_approx(baker.baked(id), g.global(id), 1e-4));
    }
}

// ---- invalidate / unfreeze -------------------------------------------------

#[test]
fn invalidate_subtree_stales_only_that_component_and_rebakes() {
    let (g, ids) = build_graph();
    let [_root, a, b, a0, a1, b0] = ids;

    let mut baker = StaticBaker::with_len(g.len());
    for id in [a, a0, a1, b, b0] {
        baker.mark_static(id);
    }
    let locals: Vec<Transform> =
        (0..g.len()).map(|i| g.local(NodeId::new(i as u32))).collect();
    baker.bake(g.hierarchy(), &locals, g.globals()).unwrap();

    let gen_before = baker.generation();
    let invalidated = baker.invalidate_subtree(g.hierarchy(), a);
    assert_eq!(invalidated, 3, "a, a0, a1");
    assert!(baker.generation() > gen_before, "invalidation bumps generation");

    // The `a` component is now stale; the `b` component is untouched.
    assert!(baker.try_baked(a).is_none());
    assert!(baker.try_baked(a0).is_none());
    assert!(baker.is_baked(b));
    assert!(baker.is_baked(b0));

    // Re-baking refreshes exactly the stale component.
    let stats = baker.bake(g.hierarchy(), &locals, g.globals()).unwrap();
    assert_eq!(stats.baked, 3);
    assert!(baker.is_baked(a) && baker.is_baked(a1));
}

#[test]
fn unfreeze_subtree_returns_nodes_to_dynamic_set() {
    let (g, ids) = build_graph();
    let [_root, a, _b, a0, a1, _b0] = ids;

    let mut baker = StaticBaker::with_len(g.len());
    for id in [a, a0, a1] {
        baker.mark_static(id);
    }
    let locals: Vec<Transform> =
        (0..g.len()).map(|i| g.local(NodeId::new(i as u32))).collect();
    baker.bake(g.hierarchy(), &locals, g.globals()).unwrap();

    let unfrozen = baker.unfreeze_subtree(g.hierarchy(), a);
    // Parent-before-child order with `a` first.
    assert_eq!(unfrozen.first(), Some(&a));
    assert_eq!(unfrozen.len(), 3);

    for id in [a, a0, a1] {
        assert!(!baker.is_static(id));
        assert!(baker.try_baked(id).is_none());
    }
    assert!(baker.static_roots(g.hierarchy()).is_empty());
}

#[test]
fn rebaking_an_unchanged_scene_is_idempotent() {
    let (g, ids) = build_graph();
    let mut baker = StaticBaker::with_len(g.len());
    for &id in &ids {
        baker.mark_static(id);
    }
    let seeds = identity_globals(g.hierarchy());
    let locals: Vec<Transform> =
        (0..g.len()).map(|i| g.local(NodeId::new(i as u32))).collect();
    let first = baker.bake(g.hierarchy(), &locals, &seeds).unwrap();
    assert_eq!(first.baked, g.len(), "first bake freezes everything");
    // Re-baking an unchanged (still-valid) scene recomputes nothing.
    let second = baker.bake(g.hierarchy(), &locals, &seeds).unwrap();
    assert_eq!(second.baked, 0, "nothing stale => zero compositions");
    assert_eq!(second.roots, first.roots);
    for &id in &ids {
        assert!(g_approx(baker.baked(id), g.global(id), 1e-4));
    }
}

// ---- batch merge: instances ------------------------------------------------

#[test]
fn merge_instances_groups_by_key_and_world_bakes() {
    let (g, ids) = build_graph();
    let [_root, a, b, a0, a1, b0] = ids;

    let mut baker = StaticBaker::with_len(g.len());
    for id in [a, a0, a1, b, b0] {
        baker.mark_static(id);
    }
    let locals: Vec<Transform> =
        (0..g.len()).map(|i| g.local(NodeId::new(i as u32))).collect();
    baker.bake(g.hierarchy(), &locals, g.globals()).unwrap();

    // Key 7: {a0, a1}, key 3: {b0}; `b` offered but with a *dynamic*-only key 3.
    let keyed = [
        (a0, BatchKey(7)),
        (b0, BatchKey(3)),
        (a1, BatchKey(7)),
    ];
    let batches = merge_instances(&baker, &keyed);

    // Sorted by ascending key: 3 then 7.
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].key, BatchKey(3));
    assert_eq!(batches[0].sources, alloc::vec![b0]);
    assert_eq!(batches[1].key, BatchKey(7));
    // Within a key, input order preserved.
    assert_eq!(batches[1].sources, alloc::vec![a0, a1]);

    // Each instance matrix equals the baked world affine.
    assert!(g_approx(
        GlobalTransform(batches[1].instances[0]),
        baker.baked(a0),
        1e-5
    ));
    assert!(g_approx(
        GlobalTransform(batches[1].instances[1]),
        baker.baked(a1),
        1e-5
    ));
}

#[test]
fn merge_instances_skips_unbaked_sources() {
    let (g, ids) = build_graph();
    let [root, a, _b, a0, _a1, _b0] = ids;

    let mut baker = StaticBaker::with_len(g.len());
    baker.mark_static(a);
    baker.mark_static(a0);
    let locals: Vec<Transform> =
        (0..g.len()).map(|i| g.local(NodeId::new(i as u32))).collect();
    baker.bake(g.hierarchy(), &locals, g.globals()).unwrap();

    // `root` is dynamic (never baked) and must be dropped from the batch.
    let keyed = [(root, BatchKey(1)), (a, BatchKey(1)), (a0, BatchKey(1))];
    let batches = merge_instances(&baker, &keyed);
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].sources, alloc::vec![a, a0]);
}

// ---- batch merge: merged world-space meshes --------------------------------

#[test]
fn merge_meshes_bakes_vertices_to_world_and_records_ranges() {
    let (g, ids) = build_graph();
    let [_root, a, _b, a0, a1, _b0] = ids;

    let mut baker = StaticBaker::with_len(g.len());
    for id in [a, a0, a1] {
        baker.mark_static(id);
    }
    let locals: Vec<Transform> =
        (0..g.len()).map(|i| g.local(NodeId::new(i as u32))).collect();
    baker.bake(g.hierarchy(), &locals, g.globals()).unwrap();

    let a0_verts = [vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)];
    let a1_verts = [vec3(0.0, 1.0, 0.0)];
    let sources = [
        MeshSource { node: a0, key: BatchKey(5), vertices: &a0_verts },
        MeshSource { node: a1, key: BatchKey(5), vertices: &a1_verts },
    ];
    let merged = merge_meshes(&baker, &sources);

    assert_eq!(merged.len(), 1);
    let m = &merged[0];
    assert_eq!(m.key, BatchKey(5));
    assert_eq!(m.vertices.len(), 3);
    // Source ranges: a0 -> 0..2, a1 -> 2..3.
    assert_eq!(m.sources[0], (a0, 0..2));
    assert_eq!(m.sources[1], (a1, 2..3));

    // Oracle: each merged vertex equals the source's baked world * local vertex.
    for (vi, &lv) in a0_verts.iter().enumerate() {
        let expected = baker.baked(a0).transform_point(lv);
        assert!(v_approx(m.vertices[vi], expected, 1e-5));
    }
    let expected = baker.baked(a1).transform_point(a1_verts[0]);
    assert!(v_approx(m.vertices[2], expected, 1e-5));
}

// ---- boundaries ------------------------------------------------------------

#[test]
fn bake_rejects_length_mismatch() {
    let (g, _ids) = build_graph();
    let mut baker = StaticBaker::with_len(g.len());
    let short_locals = alloc::vec![Transform::IDENTITY; g.len() - 1];
    let seeds = identity_globals(g.hierarchy());
    assert_eq!(
        baker.bake(g.hierarchy(), &short_locals, &seeds),
        Err(HierarchyError::LengthMismatch)
    );
}

#[test]
fn empty_graph_bakes_nothing() {
    let g = TransformGraph::new();
    let mut baker = StaticBaker::new();
    let stats = baker.bake(g.hierarchy(), &[], &[]).unwrap();
    assert_eq!(stats, crate::bake::BakeStats::default());
    assert!(baker.is_empty());
}

#[test]
fn marking_and_clearing_static_moves_generation() {
    let (g, ids) = build_graph();
    let mut baker = StaticBaker::with_len(g.len());
    let g0 = baker.generation();
    baker.mark_static(ids[0]);
    let g1 = baker.generation();
    assert!(g1 > g0);
    // Marking an already-static node is a no-op for the generation.
    baker.mark_static(ids[0]);
    assert_eq!(baker.generation(), g1);
    assert!(baker.clear_static(ids[0]));
    assert!(baker.generation() > g1);
    assert!(!baker.clear_static(ids[0]), "already cleared");
}
