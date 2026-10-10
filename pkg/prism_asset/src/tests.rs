//! Unit tests for the asset kernel.
//!
//! These exercise identity, path parsing, generational storage, handle
//! lifetime, change events, dependency ordering, and load-state folding.

use crate::{
    AssetError, AssetErrorId, AssetEvent, AssetId, AssetIndex, AssetPath, Assets, DependencyError,
    DependencyGraph, ErrorRegistry, LoadState, RecursiveDependencyLoadState, UntypedAssetId,
};
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// A deliberately non-`Clone`, non-`Default` payload: storage only requires
/// the lightweight [`Asset`] marker (needed to mint type-tagged ids), never
/// `Clone`/`Default`/`Send` on the stored value.
struct Mesh {
    verts: u32,
}
impl Asset for Mesh {
    const TYPE_NAME: &'static str = "prism_asset::tests::Mesh";
}

#[test]
fn asset_index_round_trips_parts() {
    let index = AssetIndex::from_parts(7, 3);
    assert_eq!(index.index(), 7);
    assert_eq!(index.generation(), 3);
}

#[test]
fn asset_id_untyped_round_trip() {
    let index = AssetIndex::from_parts(2, 5);
    let typed: AssetId<Mesh> = AssetId::new(index);
    let untyped = typed.untyped();
    assert_eq!(untyped.index(), index);
    let back: AssetId<Mesh> = untyped.typed().expect("same type round-trips");
    assert_eq!(back, typed);
    assert_eq!(UntypedAssetId::from(typed), untyped);
}

#[test]
fn asset_path_parses_label() {
    let path = AssetPath::parse("models/hero.gltf#Mesh0");
    assert_eq!(path.path(), "models/hero.gltf");
    assert_eq!(path.label(), Some("Mesh0"));
    assert!(path.has_label());
    assert_eq!(path.to_string(), "models/hero.gltf#Mesh0");
}

#[test]
fn asset_path_without_label() {
    let path = AssetPath::parse("textures/stone.png");
    assert_eq!(path.path(), "textures/stone.png");
    assert_eq!(path.label(), None);
    assert!(!path.has_label());

    let empty_label = AssetPath::parse("file#");
    assert_eq!(empty_label.path(), "file");
    assert_eq!(empty_label.label(), None);

    let relabeled = path.with_label("mip0");
    assert_eq!(relabeled.label(), Some("mip0"));
    assert_eq!(relabeled.without_label().label(), None);
}

#[test]
fn insert_get_and_contains() {
    let mut assets = Assets::<Mesh>::new();
    assert!(assets.is_empty());
    let handle = assets.insert(Mesh { verts: 12 });
    assert_eq!(assets.len(), 1);
    assert!(assets.contains(handle.id()));
    assert_eq!(assets.get(handle.id()).map(|m| m.verts), Some(12));
}

#[test]
fn get_mut_mutates_and_emits_modified() {
    let mut assets = Assets::<Mesh>::new();
    let handle = assets.insert(Mesh { verts: 1 });
    let _ = assets.drain_events();
    if let Some(mesh) = assets.get_mut(handle.id()) {
        mesh.verts = 99;
    }
    assert_eq!(assets.get(handle.id()).map(|m| m.verts), Some(99));
    let events = assets.drain_events();
    assert_eq!(events, vec![AssetEvent::Modified { id: handle.id() }]);
}

#[test]
fn remove_returns_value_and_frees_slot() {
    let mut assets = Assets::<Mesh>::new();
    let handle = assets.insert(Mesh { verts: 4 });
    let id = handle.id();
    let removed = assets.remove(id);
    assert!(removed.is_some_and(|m| m.verts == 4));
    assert!(!assets.contains(id));
    assert!(assets.get(id).is_none());
    assert!(assets.is_empty());
}

#[test]
fn stale_id_does_not_alias_recycled_slot() {
    let mut assets = Assets::<Mesh>::new();
    let first = assets.insert(Mesh { verts: 1 });
    let stale = first.id();
    assets.remove(stale);
    // Reusing the freed slot bumps its generation.
    let second = assets.insert(Mesh { verts: 2 });
    assert_eq!(second.id().index().index(), stale.index().index());
    assert_ne!(second.id().index().generation(), stale.index().generation());
    assert!(assets.get(stale).is_none());
    assert_eq!(assets.get(second.id()).map(|m| m.verts), Some(2));
}

#[test]
fn insert_emits_added_and_remove_emits_removed() {
    let mut assets = Assets::<Mesh>::new();
    let handle = assets.insert(Mesh { verts: 1 });
    let added = assets.drain_events();
    assert_eq!(added, vec![AssetEvent::Added { id: handle.id() }]);
    // Draining clears the queue.
    assert_eq!(assets.pending_event_count(), 0);

    let id = handle.id();
    assets.remove(id);
    let removed = assets.drain_events();
    assert_eq!(removed, vec![AssetEvent::Removed { id }]);
}

#[test]
fn remove_unused_reclaims_only_abandoned_assets() {
    let mut assets = Assets::<Mesh>::new();
    let kept = assets.insert(Mesh { verts: 1 });
    let dropped = assets.insert(Mesh { verts: 2 });
    let dropped_id = dropped.id();
    let _ = assets.drain_events();

    // While a strong handle lives, nothing is reclaimed.
    drop(dropped);
    assert_eq!(assets.remove_unused(), 1);
    assert!(!assets.contains(dropped_id));
    assert!(assets.contains(kept.id()));
    assert_eq!(assets.len(), 1);

    let events = assets.drain_events();
    assert_eq!(events, vec![AssetEvent::Removed { id: dropped_id }]);

    // Dropping the last handle makes the final asset reclaimable too.
    let kept_id = kept.id();
    drop(kept);
    assert_eq!(assets.remove_unused(), 1);
    assert!(assets.is_empty());
    assert_eq!(
        assets.drain_events(),
        vec![AssetEvent::Removed { id: kept_id }]
    );
}

#[test]
fn iter_yields_live_pairs() {
    let mut assets = Assets::<Mesh>::new();
    let a = assets.insert(Mesh { verts: 1 });
    let b = assets.insert(Mesh { verts: 2 });
    let mut seen: Vec<(AssetId<Mesh>, u32)> =
        assets.iter().map(|(id, mesh)| (id, mesh.verts)).collect();
    seen.sort_by_key(|(id, _)| *id);
    assert_eq!(seen, vec![(a.id(), 1), (b.id(), 2)]);
}

#[test]
fn handle_clone_shares_identity() {
    let mut assets = Assets::<Mesh>::new();
    let handle = assets.insert(Mesh { verts: 1 });
    let clone = handle.clone();
    assert_eq!(handle, clone);
    assert_eq!(handle.id(), clone.id());
    assert_eq!(handle.handle_id(), clone.handle_id());
    assert_eq!(handle.strong_count(), 2);
}

#[test]
fn independent_handles_share_asset_but_differ_in_handle_id() {
    let mut assets = Assets::<Mesh>::new();
    let a = assets.insert(Mesh { verts: 1 });
    let b = assets.insert(Mesh { verts: 2 });
    assert_ne!(a.id(), b.id());
    assert_ne!(a.handle_id(), b.handle_id());
}

#[test]
fn downgrade_and_upgrade_track_strong_count() {
    let mut assets = Assets::<Mesh>::new();
    let handle = assets.insert(Mesh { verts: 1 });
    let weak = handle.downgrade();
    assert!(weak.upgrade().is_some());
    assert_eq!(weak.strong_count(), 1);
    drop(handle);
    assert_eq!(weak.strong_count(), 0);
    assert!(weak.upgrade().is_none());
}

#[test]
fn untyped_handle_round_trips() {
    let mut assets = Assets::<Mesh>::new();
    let handle = assets.insert(Mesh { verts: 1 });
    let untyped = handle.untyped();
    assert_eq!(untyped.id(), handle.untyped_id());
    assert_eq!(untyped.handle_id(), handle.handle_id());
    let typed: crate::Handle<Mesh> = untyped.typed::<Mesh>().expect("same type round-trips");
    assert_eq!(typed.id(), handle.id());
}

#[test]
fn reserve_is_pending_then_fulfill_makes_ready() {
    let mut assets = Assets::<Mesh>::new();
    let handle = assets.reserve();
    let id = handle.id();
    // Reserved: occupied and counted, but not readable and not loaded.
    assert!(assets.contains(id));
    assert_eq!(assets.len(), 1);
    assert!(!assets.is_ready(id));
    assert!(assets.get(id).is_none());
    assert_eq!(assets.load_state(id), LoadState::Loading);
    // Reserve emits no event.
    assert_eq!(assets.pending_event_count(), 0);

    assert!(assets.fulfill(id, Mesh { verts: 7 }));
    assert!(assets.is_ready(id));
    assert_eq!(assets.get(id).map(|m| m.verts), Some(7));
    assert_eq!(assets.load_state(id), LoadState::Loaded);
    // Fulfilling a pending slot surfaces as Added (the asset became present).
    assert_eq!(assets.drain_events(), vec![AssetEvent::Added { id }]);
}

#[test]
fn fulfill_existing_ready_slot_emits_modified() {
    let mut assets = Assets::<Mesh>::new();
    let handle = assets.insert(Mesh { verts: 1 });
    let id = handle.id();
    let _ = assets.drain_events();
    // Re-fulfilling an already-ready slot is a reload: Modified, new value.
    assert!(assets.fulfill(id, Mesh { verts: 2 }));
    assert_eq!(assets.get(id).map(|m| m.verts), Some(2));
    assert_eq!(assets.drain_events(), vec![AssetEvent::Modified { id }]);
}

#[test]
fn fail_marks_slot_failed_and_emits_failed() {
    let mut assets = Assets::<Mesh>::new();
    let mut reg = ErrorRegistry::new();
    let err = reg.record(AssetError::new("mesh/a.bin", "truncated"));

    let handle = assets.reserve();
    let id = handle.id();
    let _ = assets.drain_events();

    assert!(assets.fail(id, err));
    // Failed slots stay occupied but are not readable.
    assert!(assets.contains(id));
    assert!(!assets.is_ready(id));
    assert!(assets.get(id).is_none());
    assert_eq!(assets.load_state(id), LoadState::Failed(err));
    assert_eq!(
        assets.drain_events(),
        vec![AssetEvent::Failed { id, error: err }]
    );

    // A failed load can still recover via fulfill.
    assert!(assets.fulfill(id, Mesh { verts: 3 }));
    assert_eq!(assets.get(id).map(|m| m.verts), Some(3));
    assert_eq!(assets.load_state(id), LoadState::Loaded);
    assert_eq!(assets.drain_events(), vec![AssetEvent::Added { id }]);
}

#[test]
fn fulfill_and_fail_reject_stale_ids() {
    let mut assets = Assets::<Mesh>::new();
    let mut reg = ErrorRegistry::new();
    let err = reg.record(AssetError::new("x", "y"));
    let handle = assets.insert(Mesh { verts: 1 });
    let stale = handle.id();
    assets.remove(stale);
    assert!(!assets.fulfill(stale, Mesh { verts: 9 }));
    assert!(!assets.fail(stale, err));
    assert_eq!(assets.load_state(stale), LoadState::NotLoaded);
}

#[test]
fn pending_and_failed_slots_are_skipped_by_iter() {
    let mut assets = Assets::<Mesh>::new();
    let mut reg = ErrorRegistry::new();
    let err = reg.record(AssetError::new("x", "y"));
    let ready = assets.insert(Mesh { verts: 5 });
    let pending = assets.reserve();
    let failing = assets.reserve();
    assets.fail(failing.id(), err);

    let ids: Vec<_> = assets.iter().map(|(id, _)| id).collect();
    assert_eq!(ids, vec![ready.id()]);
    // All three slots are occupied even though only one is ready.
    assert_eq!(assets.len(), 3);
    let _ = (pending, failing);
}

#[test]
fn remove_unused_reclaims_abandoned_pending_slot() {
    let mut assets = Assets::<Mesh>::new();
    let pending = assets.reserve();
    let id = pending.id();
    drop(pending);
    assert_eq!(assets.remove_unused(), 1);
    assert!(!assets.contains(id));
    assert!(assets.is_empty());
}

#[test]
fn collect_releases_fast_path_skips_scan_without_drops() {
    let mut assets = Assets::<Mesh>::new();
    let _kept = assets.insert(Mesh { verts: 1 });
    let _ = assets.drain_events();
    // No handle has been abandoned, so each reclaim point is a no-op and the
    // frame counter still advances.
    assert_eq!(assets.collect_releases(), 0);
    assert_eq!(assets.collect_releases(), 0);
    assert_eq!(assets.frame(), 2);
    assert!(assets.drain_events().is_empty());
    assert_eq!(assets.len(), 1);
}

#[test]
fn collect_releases_emits_unused_then_removed_at_zero_grace() {
    let mut assets = Assets::<Mesh>::new();
    let handle = assets.insert(Mesh { verts: 7 });
    let id = handle.id();
    let _ = assets.drain_events();

    drop(handle);
    // With the default zero grace window, the first reclaim point both marks
    // the slot unused and frees it, in that order.
    assert_eq!(assets.collect_releases(), 1);
    assert!(assets.is_empty());
    assert_eq!(
        assets.drain_events(),
        vec![AssetEvent::Unused { id }, AssetEvent::Removed { id },]
    );
    // Nothing left to do: the signal fast path reports zero afterwards.
    assert_eq!(assets.collect_releases(), 0);
}

#[test]
fn collect_releases_honors_grace_window() {
    let mut assets = Assets::<Mesh>::with_grace_frames(2);
    assert_eq!(assets.grace_frames(), 2);
    let handle = assets.insert(Mesh { verts: 9 });
    let id = handle.id();
    let _ = assets.drain_events();

    drop(handle);
    // Frame 1: observed abandoned -> Unused, but not yet matured.
    assert_eq!(assets.collect_releases(), 0);
    assert_eq!(assets.drain_events(), vec![AssetEvent::Unused { id }]);
    assert!(assets.contains(id));
    // Frame 2: still inside the window (2 - 1 = 1 < 2).
    assert_eq!(assets.collect_releases(), 0);
    assert!(assets.contains(id));
    assert!(assets.drain_events().is_empty());
    // Frame 3: 3 - 1 = 2 >= 2 -> freed.
    assert_eq!(assets.collect_releases(), 1);
    assert!(!assets.contains(id));
    assert_eq!(assets.drain_events(), vec![AssetEvent::Removed { id }]);
}

#[test]
fn collect_releases_defers_abandoned_in_flight_load() {
    let mut assets = Assets::<Mesh>::new();
    let pending = assets.reserve();
    let id = pending.id();
    let _ = assets.drain_events();

    drop(pending);
    // The handle is gone but the load is still pending: the slot is marked
    // unused yet never freed while in flight, so the loader can still land.
    assert_eq!(assets.collect_releases(), 0);
    assert!(assets.contains(id));
    assert_eq!(assets.drain_events(), vec![AssetEvent::Unused { id }]);
    // A later resolve succeeds against the still-valid id...
    assert!(assets.fail(id, mk_error()));
    let _ = assets.drain_events();
    // ...and now the resolved-but-abandoned slot is reclaimed.
    assert_eq!(assets.collect_releases(), 1);
    assert!(!assets.contains(id));
    assert_eq!(assets.drain_events(), vec![AssetEvent::Removed { id }]);
}

#[test]
fn load_state_is_copy_and_small() {
    // `LoadState` must stay `Copy` so it can live by value in components and
    // cross the main/render split without cloning. A `Failed` id costs 4 bytes.
    let s = LoadState::Failed(mk_error());
    let copied = s; // move-by-copy; `s` is still usable afterwards.
    assert_eq!(s, copied);
    assert!(size_of::<LoadState>() <= 8);
}

/// Records one error in a throwaway registry and returns its id.
fn mk_error() -> AssetErrorId {
    let mut reg = ErrorRegistry::new();
    reg.record(AssetError::new("tex/a.png", "io"))
}

#[test]
fn load_state_helpers() {
    let err = mk_error();
    assert!(LoadState::Loaded.is_loaded());
    assert!(LoadState::Loading.is_loading());
    assert!(LoadState::Failed(err).is_failed());
    assert_eq!(LoadState::Failed(err).error(), Some(err));
    assert_eq!(LoadState::Loaded.error(), None);
    assert_eq!(LoadState::default(), LoadState::NotLoaded);
}

#[test]
fn recursive_load_state_combine_precedence() {
    use RecursiveDependencyLoadState as R;
    let err = mk_error();
    // Failed dominates everything.
    assert!(R::Loaded.combine(R::Failed(err)).is_failed());
    assert!(R::Failed(err).combine(R::Loading).is_failed());
    // Then Loading beats NotLoaded and Loaded.
    assert_eq!(R::Loaded.combine(R::Loading), R::Loading);
    // Then NotLoaded beats Loaded.
    assert_eq!(R::Loaded.combine(R::NotLoaded), R::NotLoaded);
    // All-loaded stays loaded.
    assert_eq!(R::Loaded.combine(R::Loaded), R::Loaded);
}

#[test]
fn recursive_load_state_combine_keeps_first_failure() {
    use RecursiveDependencyLoadState as R;
    let mut reg = ErrorRegistry::new();
    let first = reg.record(AssetError::new("a", "a-reason"));
    let second = reg.record(AssetError::new("b", "b-reason"));
    assert_ne!(first, second);
    // When both sides failed, the left-hand failure is preserved.
    assert_eq!(
        R::Failed(first).combine(R::Failed(second)).error(),
        Some(first)
    );
}

#[test]
fn recursive_load_state_from_load_state() {
    use RecursiveDependencyLoadState as R;
    let err = mk_error();
    assert_eq!(R::from(LoadState::NotLoaded), R::NotLoaded);
    assert_eq!(R::from(LoadState::Loading), R::Loading);
    assert_eq!(R::from(LoadState::Loaded), R::Loaded);
    assert_eq!(R::from(LoadState::Failed(err)), R::Failed(err));
}

#[test]
fn asset_event_failed_carries_error() {
    let mut reg = ErrorRegistry::new();
    let err = reg.record(AssetError::new("tex/a.png", "io"));
    let other = reg.record(AssetError::new("x", "y"));
    assert_ne!(err, other);
    let id = AssetId::<()>::new(AssetIndex::from_parts(3, 0));
    let ev = AssetEvent::Failed { id, error: err };
    assert!(ev.is_failed());
    assert_eq!(ev.error(), Some(err));
    assert_eq!(ev.id(), id);
    // Two failures of the same asset with different errors are not equal.
    let ev2 = AssetEvent::Failed { id, error: other };
    assert_ne!(ev, ev2);
    // Non-failure events report no error.
    let added: AssetEvent<()> = AssetEvent::Added { id };
    assert_eq!(added.error(), None);
    assert!(!added.is_failed());
}

/// Builds an [`UntypedAssetId`] for graph tests from a raw index.
fn node(index: u32) -> UntypedAssetId {
    UntypedAssetId::new(AssetIndex::from_parts(index, 0), AssetTypeId::of::<Mesh>())
}

#[test]
fn topological_order_places_dependencies_first() {
    let mut graph = DependencyGraph::new();
    let (material, texture, shader) = (node(0), node(1), node(2));
    graph.add_dependency(material, texture);
    graph.add_dependency(material, shader);

    let order = graph.topological_order().expect("acyclic");
    let pos = |id: UntypedAssetId| order.iter().position(|&n| n == id).unwrap();
    assert_eq!(order.len(), 3);
    assert!(pos(texture) < pos(material));
    assert!(pos(shader) < pos(material));
}

#[test]
fn topological_order_is_deterministic() {
    let mut graph = DependencyGraph::new();
    for i in 0..5 {
        graph.add_asset(node(i));
    }
    let first = graph.topological_order().expect("acyclic");
    let second = graph.topological_order().expect("acyclic");
    assert_eq!(first, second);
    // Independent nodes come out in ascending id order.
    assert_eq!(first, vec![node(0), node(1), node(2), node(3), node(4)]);
}

#[test]
fn dependencies_and_dependents_are_sorted() {
    let mut graph = DependencyGraph::new();
    let root = node(10);
    graph.add_dependency(root, node(3));
    graph.add_dependency(root, node(1));
    graph.add_dependency(root, node(2));
    assert_eq!(graph.dependencies(root), vec![node(1), node(2), node(3)]);
    assert_eq!(graph.dependents(node(1)), vec![root]);
    assert_eq!(graph.len(), 4);
}

#[test]
fn remove_asset_cleans_edges() {
    let mut graph = DependencyGraph::new();
    let (a, b, c) = (node(0), node(1), node(2));
    graph.add_dependency(a, b);
    graph.add_dependency(b, c);
    graph.remove_asset(b);
    assert_eq!(graph.len(), 2);
    assert!(graph.dependencies(a).is_empty());
    assert!(graph.dependents(c).is_empty());
}

#[test]
fn self_edge_is_ignored() {
    let mut graph = DependencyGraph::new();
    let a = node(0);
    graph.add_dependency(a, a);
    assert_eq!(graph.len(), 1);
    assert!(graph.dependencies(a).is_empty());
    assert_eq!(graph.topological_order().expect("acyclic"), vec![a]);
}

#[test]
fn cycle_is_detected() {
    let mut graph = DependencyGraph::new();
    let (a, b, c) = (node(0), node(1), node(2));
    graph.add_dependency(a, b);
    graph.add_dependency(b, c);
    graph.add_dependency(c, a);
    match graph.topological_order() {
        Err(DependencyError::Cycle { participants }) => {
            assert_eq!(participants, vec![a, b, c]);
        }
        Ok(order) => panic!("expected cycle, got order {order:?}"),
    }
}

#[test]
fn dependency_error_display_is_readable() {
    let err = DependencyError::Cycle {
        participants: vec![node(0), node(1)],
    };
    let text: String = err.to_string();
    assert!(text.contains('2'));
}

#[test]
fn incremental_readiness_marks_dependent_loaded() {
    // material -> {texture, shader}: material is recursive-Loaded only once
    // both of its dependencies finish.
    let mut graph = DependencyGraph::new();
    let (material, texture, shader) = (node(0), node(1), node(2));
    graph.add_dependency(material, texture);
    graph.add_dependency(material, shader);

    // Nothing loaded yet.
    assert_eq!(
        graph.recursive_state(material),
        RecursiveDependencyLoadState::NotLoaded
    );

    // Both dependencies are in flight; material's own data has loaded. The
    // closure is still Loading until every dependency finishes.
    graph.set_self_state(material, LoadState::Loaded);
    graph.set_self_state(texture, LoadState::Loading);
    graph.set_self_state(shader, LoadState::Loading);
    assert_eq!(
        graph.recursive_state(material),
        RecursiveDependencyLoadState::Loading
    );

    // Finishing one dependency keeps material Loading (worst-wins over the
    // other still-loading dependency).
    let after_tex = graph.set_self_state(texture, LoadState::Loaded);
    assert!(after_tex.contains(&texture));
    assert_eq!(
        graph.recursive_state(material),
        RecursiveDependencyLoadState::Loading
    );

    let after_shader = graph.set_self_state(shader, LoadState::Loaded);
    // Finishing the last dependency flips both shader and material to Loaded.
    assert!(after_shader.contains(&shader));
    assert!(after_shader.contains(&material));
    assert_eq!(
        graph.recursive_state(material),
        RecursiveDependencyLoadState::Loaded
    );
    assert_eq!(graph.self_state(material), LoadState::Loaded);
}

#[test]
fn incremental_readiness_is_worst_wins() {
    // One failed dependency makes the whole closure Failed, carrying its id.
    let mut graph = DependencyGraph::new();
    let (material, texture, shader) = (node(0), node(1), node(2));
    graph.add_dependency(material, texture);
    graph.add_dependency(material, shader);

    let err = mk_error();
    graph.set_self_state(material, LoadState::Loaded);
    graph.set_self_state(texture, LoadState::Loaded);
    let changed = graph.set_self_state(shader, LoadState::Failed(err));

    assert!(changed.contains(&material));
    assert_eq!(
        graph.recursive_state(material),
        RecursiveDependencyLoadState::Failed(err)
    );
}

#[test]
fn incremental_readiness_propagates_through_chain() {
    // a -> b -> c: readiness ripples all the way up as leaves complete.
    let mut graph = DependencyGraph::new();
    let (a, b, c) = (node(0), node(1), node(2));
    graph.add_dependency(a, b);
    graph.add_dependency(b, c);

    graph.set_self_state(a, LoadState::Loaded);
    graph.set_self_state(b, LoadState::Loaded);
    assert_eq!(
        graph.recursive_state(a),
        RecursiveDependencyLoadState::NotLoaded
    );

    let changed = graph.set_self_state(c, LoadState::Loaded);
    // Completing the deepest leaf readies c, then b, then a.
    assert_eq!(changed, [a, b, c].into_iter().collect());
    assert_eq!(
        graph.recursive_state(a),
        RecursiveDependencyLoadState::Loaded
    );
}

#[test]
fn try_add_dependency_rejects_cycle_with_participants() {
    let mut graph = DependencyGraph::new();
    let (a, b, c) = (node(0), node(1), node(2));
    graph.try_add_dependency(a, b).expect("acyclic");
    graph.try_add_dependency(b, c).expect("acyclic");
    // Closing c -> a would form a cycle a->b->c->a.
    match graph.try_add_dependency(c, a) {
        Err(DependencyError::Cycle { participants }) => {
            assert_eq!(participants, vec![a, b, c]);
        }
        other => panic!("expected cycle rejection, got {other:?}"),
    }
    // The rejected edge must not have been added.
    assert!(graph.dependencies(c).is_empty());
    assert!(graph.topological_order().is_ok());
}

#[test]
fn invalidate_marks_asset_and_transitive_dependents() {
    // c is depended on by b, which is depended on by a. Changing c should mark
    // c, b and a stale (their derived data depended on c).
    let mut graph = DependencyGraph::new();
    let (a, b, c, unrelated) = (node(0), node(1), node(2), node(3));
    graph.add_dependency(a, b);
    graph.add_dependency(b, c);
    graph.add_asset(unrelated);

    let affected = graph.invalidate(c);
    assert_eq!(affected, vec![a, b, c]);
    assert!(graph.is_stale(a));
    assert!(graph.is_stale(b));
    assert!(graph.is_stale(c));
    assert!(!graph.is_stale(unrelated));
    assert_eq!(graph.stale_count(), 3);
}

#[test]
fn take_stale_drains_sorted_and_clear_stale_works() {
    let mut graph = DependencyGraph::new();
    let (a, b) = (node(5), node(2));
    graph.add_asset(a);
    graph.add_asset(b);
    graph.invalidate(a);
    graph.invalidate(b);
    assert!(graph.clear_stale(a));
    assert!(!graph.clear_stale(a)); // already cleared
    assert_eq!(graph.take_stale(), vec![b]);
    assert_eq!(graph.stale_count(), 0);
    assert!(graph.take_stale().is_empty());
}

#[test]
fn remove_asset_repairs_dependent_readiness() {
    // material depends on texture (Loading) and shader (Loaded). Removing the
    // still-loading texture should let material's closure become Loaded.
    let mut graph = DependencyGraph::new();
    let (material, texture, shader) = (node(0), node(1), node(2));
    graph.add_dependency(material, texture);
    graph.add_dependency(material, shader);
    graph.set_self_state(material, LoadState::Loaded);
    graph.set_self_state(shader, LoadState::Loaded);
    graph.set_self_state(texture, LoadState::Loading);
    assert_eq!(
        graph.recursive_state(material),
        RecursiveDependencyLoadState::Loading
    );

    graph.remove_asset(texture);
    assert_eq!(
        graph.recursive_state(material),
        RecursiveDependencyLoadState::Loaded
    );
}

// --- M1: stable identity, type ids, assets, errors, soft handles, schemes ---

use crate::{direct_dependencies, normalize_path, Asset, AssetTypeId, SoftHandle, StableGuid};

/// A leaf asset with no dependencies, for identity/type tests.
struct Image;
impl Asset for Image {
    const TYPE_NAME: &'static str = "prism_asset::tests::Image";
}

/// A composite asset that references other assets, to exercise dependency
/// disclosure.
struct Material {
    textures: Vec<UntypedAssetId>,
}
impl Asset for Material {
    const TYPE_NAME: &'static str = "prism_asset::tests::Material";
    fn visit_dependencies(&self, visit: &mut dyn FnMut(UntypedAssetId)) {
        for id in &self.textures {
            visit(*id);
        }
    }
}

#[test]
fn stable_guid_from_path_is_deterministic() {
    assert_eq!(
        StableGuid::from_path("models/hero.gltf"),
        StableGuid::from_path("models/hero.gltf")
    );
}

#[test]
fn stable_guid_normalizes_before_hashing() {
    // Redundant separators, `.` and `..` segments, and backslashes all
    // collapse to the same canonical form, so these are the *same* asset.
    assert_eq!(
        StableGuid::from_path("a//b/../c.png"),
        StableGuid::from_path("a/c.png")
    );
    assert_eq!(
        StableGuid::from_path("a\\b\\c.png"),
        StableGuid::from_path("a/b/c.png")
    );
    assert_eq!(
        StableGuid::from_path("./a/./c.png"),
        StableGuid::from_path("a/c.png")
    );
}

#[test]
fn stable_guid_distinguishes_distinct_paths_and_domains() {
    assert_ne!(
        StableGuid::from_path("a/b.png"),
        StableGuid::from_path("a/c.png")
    );
    // A path guid and a content guid of the same bytes live in disjoint
    // domains, so they must differ.
    assert_ne!(
        StableGuid::from_path("abc"),
        StableGuid::from_content(b"abc")
    );
    // Case is preserved (not folded), so these stay distinct.
    assert_ne!(
        StableGuid::from_path("Hero.png"),
        StableGuid::from_path("hero.png")
    );
}

#[test]
fn stable_guid_round_trips_u128_and_nil() {
    let g = StableGuid::from_path("a/b/c.png");
    assert_eq!(StableGuid::from_u128(g.to_u128()), g);
    assert!(StableGuid::NIL.is_nil());
    assert!(!g.is_nil());
    assert_eq!(StableGuid::from_u128(0), StableGuid::NIL);
}

#[test]
fn stable_guid_sub_assets_are_stable_and_distinct() {
    let parent = StableGuid::from_path("scene.gltf");
    let mesh0 = StableGuid::derive_sub(parent, "Mesh0");
    let mesh1 = StableGuid::derive_sub(parent, "Mesh1");
    // Reproducible for the same (parent, label).
    assert_eq!(mesh0, StableGuid::derive_sub(parent, "Mesh0"));
    // Siblings differ, and a child differs from its parent.
    assert_ne!(mesh0, mesh1);
    assert_ne!(mesh0, parent);
    // The same label under a different parent is a different sub-asset.
    let other_parent = StableGuid::from_path("other.gltf");
    assert_ne!(mesh0, StableGuid::derive_sub(other_parent, "Mesh0"));
}

#[test]
fn stable_guid_display_is_32_hex_digits() {
    use alloc::format;
    let g = StableGuid::from_u128(0x1234);
    assert_eq!(format!("{g}"), "00000000000000000000000000001234");
}

#[test]
fn normalize_path_edge_cases() {
    assert_eq!(normalize_path("a/b/c"), "a/b/c");
    assert_eq!(normalize_path("a//b///c"), "a/b/c");
    assert_eq!(normalize_path("a/b/"), "a/b");
    assert_eq!(normalize_path("/abs/path/"), "/abs/path");
    assert_eq!(normalize_path("/"), "/");
    assert_eq!(normalize_path(""), "");
    assert_eq!(normalize_path("a/./b"), "a/b");
    assert_eq!(normalize_path("a/b/../c"), "a/c");
    // Relative `..` that cannot pop is preserved; absolute `..` is dropped.
    assert_eq!(normalize_path("../a"), "../a");
    assert_eq!(normalize_path("a/../../b"), "../b");
    assert_eq!(normalize_path("/../a"), "/a");
}

#[test]
fn fnv_primitives_match_reference_vectors() {
    // Canonical FNV-1a reference: hashing the empty input yields the offset
    // basis unchanged; this guards against an accidental constant drift.
    assert_eq!(StableGuid::from_content(b"").to_u128(), {
        // offset, then fold the content-domain tag byte 0x02.
        let mut h = crate::hash::FNV128_OFFSET;
        h ^= 0x02u128;
        h = h.wrapping_mul(crate::hash::FNV128_PRIME);
        h
    });
    assert_eq!(crate::hash::fnv1a_64(b""), crate::hash::FNV64_OFFSET);
    assert_eq!(crate::hash::fnv1a_128(b""), crate::hash::FNV128_OFFSET);
}

#[test]
fn asset_type_id_is_stable_and_type_specific() {
    assert_eq!(Image::asset_type(), AssetTypeId::of::<Image>());
    assert_eq!(
        AssetTypeId::of::<Image>(),
        AssetTypeId::of_name("prism_asset::tests::Image")
    );
    assert_ne!(AssetTypeId::of::<Image>(), AssetTypeId::of::<Material>());
    let id = AssetTypeId::of::<Material>();
    assert_eq!(AssetTypeId::from_u64(id.to_u64()), id);
}

#[test]
fn asset_visit_dependencies_reports_every_reference() {
    let a = UntypedAssetId::new(AssetIndex::from_parts(1, 0), AssetTypeId::of::<Image>());
    let b = UntypedAssetId::new(AssetIndex::from_parts(2, 0), AssetTypeId::of::<Image>());
    let material = Material {
        textures: vec![a, b],
    };
    assert_eq!(direct_dependencies(&material), vec![a, b]);
    // A leaf asset reports nothing by default.
    assert!(direct_dependencies(&Image).is_empty());
}

#[test]
fn error_registry_records_and_resolves() {
    let mut registry = ErrorRegistry::new();
    assert!(registry.is_empty());
    let dependent = UntypedAssetId::new(AssetIndex::from_parts(9, 1), AssetTypeId::of::<Image>());
    let id0 = registry.record(AssetError::new("a.png", "file not found"));
    let id1 = registry.record(AssetError::new("b.png", "decode failed").with_dependent(dependent));
    assert_eq!(registry.len(), 2);
    assert_ne!(id0, id1);

    let e0 = registry.get(id0).expect("id0 resolves");
    assert_eq!(e0.path, "a.png");
    assert_eq!(e0.reason, "file not found");
    assert_eq!(e0.dependent, None);

    let e1 = registry.get(id1).expect("id1 resolves");
    assert_eq!(e1.dependent, Some(dependent));

    // An id minted by a *different*, larger registry is out of range here and
    // resolves to None rather than silently aliasing another record.
    let mut scratch = ErrorRegistry::new();
    let mut high = id0;
    for _ in 0..8 {
        high = scratch.record(AssetError::new("scratch", "overshoot"));
    }
    assert!(registry.get(high).is_none());

    let collected: Vec<_> = registry.iter().map(|(_, e)| e.path.clone()).collect();
    assert_eq!(collected, vec!["a.png".to_string(), "b.png".to_string()]);
}

#[test]
fn soft_handle_is_copy_identity_only() {
    let h: SoftHandle<Image> = SoftHandle::from_path("textures/albedo.png");
    assert_eq!(h.guid(), StableGuid::from_path("textures/albedo.png"));
    assert_eq!(h.type_id(), AssetTypeId::of::<Image>());
    assert!(!h.is_null());

    // Copy semantics: a soft handle is a plain value, not an owning ref.
    let copy = h;
    assert_eq!(h, copy);

    // Null soft handle.
    let null: SoftHandle<Image> = SoftHandle::null(AssetTypeId::of::<Image>());
    assert!(null.is_null());
    assert_ne!(h, null);

    // Same guid but different type tag compares unequal (cross-type guard).
    let as_material = SoftHandle::<Material>::new(h.guid(), AssetTypeId::of::<Material>());
    assert_ne!(h.guid(), StableGuid::NIL);
    assert_ne!(as_material.type_id(), h.type_id());
}

#[test]
fn asset_path_parses_scheme_and_label() {
    let p = AssetPath::parse("source://models/hero.gltf#Mesh0");
    assert_eq!(p.scheme(), Some("source"));
    assert_eq!(p.path(), "models/hero.gltf");
    assert_eq!(p.label(), Some("Mesh0"));
    assert!(p.has_scheme() && p.has_label());
    // Round-trips through Display.
    assert_eq!(p.to_string(), "source://models/hero.gltf#Mesh0");
}

#[test]
fn asset_path_scheme_optional_and_robust() {
    let plain = AssetPath::parse("models/hero.gltf");
    assert_eq!(plain.scheme(), None);
    assert_eq!(plain.path(), "models/hero.gltf");

    // Empty scheme is not recognised; `://` stays part of the path.
    let weird = AssetPath::parse("://thing");
    assert_eq!(weird.scheme(), None);
    assert_eq!(weird.path(), "://thing");

    // A `#` with empty remainder yields no label.
    let no_label = AssetPath::parse("a.gltf#");
    assert_eq!(no_label.label(), None);
    assert_eq!(no_label.path(), "a.gltf");

    // Builder round-trip.
    let built = AssetPath::new("a/b.png")
        .with_scheme("dlc")
        .with_label("Lod0");
    assert_eq!(built.to_string(), "dlc://a/b.png#Lod0");
    assert_eq!(built.clone().without_scheme().to_string(), "a/b.png#Lod0");
    assert_eq!(built.without_label().to_string(), "dlc://a/b.png");
}

// --- M1: deterministic loader-selection policy (design §9.1) ---

use crate::{LoaderRegistry, SuffixConflict};

#[test]
fn loader_one_loader_many_extensions() {
    let mut reg = LoaderRegistry::new();
    let (id, conflicts) = reg.register(AssetTypeId::of::<Image>(), ["png", "ktx2", "basis"], 0);
    assert!(conflicts.is_empty());
    for ext in ["img.png", "img.ktx2", "img.basis"] {
        assert_eq!(reg.resolve_untyped(&AssetPath::parse(ext)), Some(id));
    }
    // Unknown suffix matches nothing.
    assert_eq!(reg.resolve_untyped(&AssetPath::parse("img.exr")), None);
}

#[test]
fn loader_longest_suffix_wins() {
    let mut reg = LoaderRegistry::new();
    // A generic `.zst` loader and a specific `.tar.zst` loader.
    let (zst, _) = reg.register(AssetTypeId::of::<Mesh>(), ["zst"], 0);
    let (tarzst, _) = reg.register(AssetTypeId::of::<Image>(), ["tar.zst"], 0);
    // The compound suffix must beat the shorter tail.
    assert_eq!(
        reg.resolve_untyped(&AssetPath::parse("archive.tar.zst")),
        Some(tarzst)
    );
    // A plain `.zst` still goes to the generic loader.
    assert_eq!(
        reg.resolve_untyped(&AssetPath::parse("blob.zst")),
        Some(zst)
    );
}

#[test]
fn loader_case_insensitive() {
    let mut reg = LoaderRegistry::new();
    let (id, _) = reg.register(AssetTypeId::of::<Image>(), ["png"], 0);
    assert_eq!(reg.resolve_untyped(&AssetPath::parse("HERO.PNG")), Some(id));
    // Registering an upper-cased suffix normalizes to the same canonical form.
    let mut reg2 = LoaderRegistry::new();
    let (id2, _) = reg2.register(AssetTypeId::of::<Image>(), [".PNG"], 0);
    assert_eq!(reg2.resolve_untyped(&AssetPath::parse("a.png")), Some(id2));
}

#[test]
fn loader_alias_remaps_to_canonical() {
    let mut reg = LoaderRegistry::new();
    let (jpg, _) = reg.register(AssetTypeId::of::<Image>(), ["jpg"], 0);
    reg.add_alias("jpeg", "jpg");
    // A `.jpeg` filename resolves to the `jpg` loader via the alias.
    assert_eq!(
        reg.resolve_untyped(&AssetPath::parse("photo.jpeg")),
        Some(jpg)
    );
    // The longer alias beats a shorter real suffix.
    reg.add_alias("JPE", "jpg");
    assert_eq!(
        reg.resolve_untyped(&AssetPath::parse("photo.jpe")),
        Some(jpg)
    );
}

#[test]
fn loader_typed_disambiguation() {
    let mut reg = LoaderRegistry::new();
    // Two loaders claim `.asset` but produce different types.
    let (as_image, _) = reg.register(AssetTypeId::of::<Image>(), ["asset"], 0);
    let (as_mesh, _) = reg.register(AssetTypeId::of::<Mesh>(), ["asset"], 0);
    assert_eq!(
        reg.resolve_for_type(&AssetPath::parse("x.asset"), AssetTypeId::of::<Image>()),
        Some(as_image)
    );
    assert_eq!(
        reg.resolve_for_type(&AssetPath::parse("x.asset"), AssetTypeId::of::<Mesh>()),
        Some(as_mesh)
    );
    // A type nobody produces resolves to nothing even though the suffix matches.
    assert_eq!(
        reg.resolve_for_type(&AssetPath::parse("x.asset"), AssetTypeId::of::<Material>()),
        None
    );
}

#[test]
fn loader_untyped_priority_and_override() {
    let mut reg = LoaderRegistry::new();
    let (_low, _) = reg.register(AssetTypeId::of::<Image>(), ["png"], 0);
    let (high, conflicts) = reg.register(AssetTypeId::of::<Mesh>(), ["png"], 10);
    // The second registration conflicts with the first on `png`.
    assert_eq!(
        conflicts,
        vec![SuffixConflict {
            suffix: "png".into(),
            existing: _low,
            incoming: high,
        }]
    );
    // Higher priority wins the untyped resolution.
    assert_eq!(reg.resolve_untyped(&AssetPath::parse("a.png")), Some(high));

    // Equal priority: the later registration (higher seq) wins the tie.
    let mut reg2 = LoaderRegistry::new();
    let (_first, _) = reg2.register(AssetTypeId::of::<Image>(), ["dds"], 5);
    let (second, _) = reg2.register(AssetTypeId::of::<Mesh>(), ["dds"], 5);
    assert_eq!(
        reg2.resolve_untyped(&AssetPath::parse("a.dds")),
        Some(second)
    );
}

#[test]
fn loader_dotfile_without_stem_does_not_match() {
    let mut reg = LoaderRegistry::new();
    reg.register(AssetTypeId::of::<Image>(), ["png"], 0);
    // A bare dotfile `.png` has no stem and must not resolve.
    assert_eq!(reg.resolve_untyped(&AssetPath::parse(".png")), None);
    // But `a.png` (stem `a`) does.
    assert!(reg.resolve_untyped(&AssetPath::parse("a.png")).is_some());
}

#[test]
fn loader_ignores_scheme_and_label() {
    let mut reg = LoaderRegistry::new();
    let (id, _) = reg.register(AssetTypeId::of::<Image>(), ["png"], 0);
    // Scheme and `#label` are identity, not suffix hints — the suffix of the
    // path segment still resolves.
    let p = AssetPath::parse("source://textures/hero.png#Lod0");
    assert_eq!(reg.resolve_untyped(&p), Some(id));
}

#[test]
fn loader_registry_bookkeeping() {
    let mut reg = LoaderRegistry::new();
    assert!(reg.is_empty());
    let (id, _) = reg.register(AssetTypeId::of::<Image>(), ["png"], 0);
    assert_eq!(reg.len(), 1);
    assert!(!reg.is_empty());
    assert_eq!(reg.produced_type(id), Some(AssetTypeId::of::<Image>()));
    // Deeper path segments: directories are ignored, only the filename counts.
    assert_eq!(
        reg.resolve_untyped(&AssetPath::parse("a/b/c/hero.png")),
        Some(id)
    );
}

// --- std VFS: AssetReader / MemSource / FsSource / AssetSources (§10) ---

#[cfg(feature = "std")]
mod vfs {
    use crate::{AssetPath, AssetReader, AssetSources, FsSource, MemSource, ReadError};
    use alloc::string::ToString;
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;

    fn mem() -> MemSource {
        let mut m = MemSource::new();
        m.insert("textures/hero.png", vec![1u8, 2, 3, 4, 5]);
        m.insert("textures/sky.png", vec![9u8, 8, 7]);
        m.insert("meshes/hero.gltf", vec![0u8; 16]);
        m
    }

    #[test]
    fn mem_read_and_range() {
        let m = mem();
        assert_eq!(m.read("textures/hero.png").unwrap(), vec![1, 2, 3, 4, 5]);
        assert_eq!(
            m.read_range("textures/hero.png", 1, 3).unwrap(),
            vec![2, 3, 4]
        );
        assert_eq!(
            m.read_range("textures/hero.png", 3, 10),
            Err(ReadError::OutOfRange {
                offset: 3,
                len: 10,
                size: 5
            })
        );
        assert_eq!(m.read("nope.png"), Err(ReadError::NotFound));
    }

    #[test]
    fn mem_metadata_and_list() {
        let m = mem();
        let meta = m.metadata("textures/hero.png").unwrap();
        assert_eq!(meta.size, 5);
        assert!(!meta.is_dir);
        assert!(m.metadata("textures").unwrap().is_dir);
        let mut top = m.list("").unwrap();
        top.sort();
        assert_eq!(top, vec!["meshes".to_string(), "textures".to_string()]);
        let mut tex = m.list("textures").unwrap();
        tex.sort();
        assert_eq!(tex, vec!["hero.png".to_string(), "sky.png".to_string()]);
    }

    #[test]
    fn path_traversal_is_rejected() {
        let m = mem();
        assert_eq!(m.read("../secret"), Err(ReadError::InvalidPath));
        assert_eq!(m.read("/etc/passwd"), Err(ReadError::InvalidPath));
        assert_eq!(m.read("a/../../b"), Err(ReadError::InvalidPath));
    }

    #[test]
    fn overlay_priority_and_scheme_targeting() {
        let mut base = MemSource::new();
        base.insert("config.txt", vec![0u8]);
        base.insert("only_base.txt", vec![42u8]);
        let mut patch = MemSource::new();
        patch.insert("config.txt", vec![1u8]); // shadows base

        let mut sources = AssetSources::new();
        sources.mount("base", Arc::new(base), 0);
        sources.mount("patch", Arc::new(patch), 10); // higher priority wins

        // No scheme: overlay order means the patch shadows the base.
        assert_eq!(
            sources.read(&AssetPath::parse("config.txt")).unwrap(),
            vec![1]
        );
        // Falls through to base for assets the patch lacks.
        assert_eq!(
            sources.read(&AssetPath::parse("only_base.txt")).unwrap(),
            vec![42]
        );
        // Explicit scheme targets exactly one mount (even the shadowed one).
        assert_eq!(
            sources
                .read(&AssetPath::parse("base://config.txt"))
                .unwrap(),
            vec![0]
        );
        // Unknown mount name resolves to NotFound.
        assert_eq!(
            sources.read(&AssetPath::parse("ghost://x.txt")),
            Err(ReadError::NotFound)
        );
        assert_eq!(sources.overlay_order(), vec!["patch", "base"]);
    }

    #[test]
    fn hard_error_not_masked_by_lower_mount() {
        // A mount that returns a hard error must not be silently skipped in
        // favor of a lower mount; only NotFound falls through.
        struct Corrupt;
        impl AssetReader for Corrupt {
            fn read(&self, _path: &str) -> Result<Vec<u8>, ReadError> {
                Err(ReadError::Io("disk fault".into()))
            }
            fn metadata(&self, _path: &str) -> Result<crate::AssetMeta, ReadError> {
                Err(ReadError::Io("disk fault".into()))
            }
        }
        let mut base = MemSource::new();
        base.insert("x.txt", vec![7u8]);
        let mut sources = AssetSources::new();
        sources.mount("base", Arc::new(base), 0);
        sources.mount("corrupt", Arc::new(Corrupt), 10);
        assert_eq!(
            sources.read(&AssetPath::parse("x.txt")),
            Err(ReadError::Io("disk fault".into()))
        );
    }

    #[test]
    fn fs_source_reads_and_confines_to_root() {
        let dir = std::env::temp_dir().join(alloc::format!(
            "prism_asset_fs_{}_{}",
            std::process::id(),
            now_nanos()
        ));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("a.txt"), b"hello").unwrap();
        std::fs::write(dir.join("sub/b.txt"), b"world!!").unwrap();

        let src = FsSource::new(&dir);
        assert_eq!(src.read("a.txt").unwrap(), b"hello");
        assert_eq!(src.read_range("sub/b.txt", 1, 4).unwrap(), b"orld");
        assert_eq!(src.metadata("a.txt").unwrap().size, 5);
        assert!(src.metadata("sub").unwrap().is_dir);
        let mut listed = src.list("").unwrap();
        listed.sort();
        assert_eq!(listed, vec!["a.txt".to_string(), "sub".to_string()]);
        // Traversal is refused before any syscall.
        assert_eq!(src.read("../a.txt"), Err(ReadError::InvalidPath));
        assert_eq!(src.read("/etc/hosts"), Err(ReadError::InvalidPath));
        assert_eq!(src.read("missing.txt"), Err(ReadError::NotFound));

        std::fs::remove_dir_all(&dir).ok();
    }

    fn now_nanos() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    }
}
