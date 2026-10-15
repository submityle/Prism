//! Real anchor/delta history, reconstruction, rebasing and failure atomicity.

use prism_ecs::prelude::*;
use prism_ecs::relation::RelationKind;
use prism_ecs::world::snapshot::{SnapshotRing, WorldSnapshot};
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::Arc;

#[derive(Component, Clone, Debug, PartialEq, Eq, Hash)]
struct Value {
    number: u64,
    text: String,
    samples: Vec<u32>,
}

fn value(number: u64) -> Value {
    Value {
        number,
        text: format!("owning-{number}"),
        samples: vec![number as u32, 3, 5],
    }
}

#[derive(Component, Clone, Debug, PartialEq, Eq, Hash)]
#[component(storage = "SparseSet")]
struct Sparse(Vec<String>);

#[derive(Component, Clone, Debug, PartialEq, Eq, Hash)]
#[component(storage = "Shared")]
struct Shared(String);

#[derive(Component, Clone, Debug, PartialEq, Eq, Hash)]
struct Tag;

#[derive(Component)]
struct Link;

#[derive(Resource, Clone, Debug, PartialEq, Eq, Hash)]
struct Score(Vec<String>);

#[derive(Resource)]
struct External(u64);

fn world() -> World {
    let mut world = World::new();
    world.register_snapshot_component_equatable::<Value>();
    world.register_snapshot_component_hashable::<Value>();
    world.register_snapshot_component_equatable::<Sparse>();
    world.register_snapshot_component_equatable::<Shared>();
    world.register_snapshot_component_equatable::<Tag>();
    world.register_snapshot_resource_equatable::<Score>();
    world.register_relation::<Link>(RelationKind::new());
    world
}

fn cells(snapshot: &WorldSnapshot) -> usize {
    let delta = snapshot.diff(snapshot);
    delta.changed_cell_count() + delta.reused_cell_count()
}

fn verify(ring: &SnapshotRing, expected: &BTreeMap<u64, WorldSnapshot>) {
    assert_eq!(
        ring.frames().collect::<Vec<_>>(),
        expected.keys().copied().collect::<Vec<_>>()
    );
    assert_eq!(ring.len(), expected.len());
    assert!(ring.len() <= ring.capacity());
    assert_eq!(ring.is_empty(), expected.is_empty());
    assert_eq!(
        ring.full_snapshot_count(),
        usize::from(!expected.is_empty())
    );
    assert_eq!(ring.delta_count(), expected.len().saturating_sub(1));
    assert_eq!(
        ring.oldest().map(|(f, _)| f),
        expected.keys().next().copied()
    );
    assert_eq!(ring.latest_frame(), expected.keys().next_back().copied());
    let mut previous: Option<&WorldSnapshot> = None;
    let mut owned = 0;
    let mut reused = 0;
    for (&frame, full) in expected {
        assert!(ring.contains(frame));
        let rebuilt = ring.get(frame).expect("retained frame reconstructs");
        assert!(rebuilt.structurally_eq(full), "frame {frame}");
        assert_eq!(rebuilt.state_hash(), full.state_hash(), "frame {frame}");
        if let Some(base) = previous {
            let delta = base.diff(full);
            owned += delta.changed_cell_count();
            reused += delta.reused_cell_count();
        } else {
            owned += cells(full);
        }
        previous = Some(full);
    }
    assert_eq!(ring.stored_component_cell_count(), owned);
    assert_eq!(ring.reused_component_cell_count(), reused);
    assert_eq!(
        ring.stored_resource_count(),
        expected.values().map(|s| s.resource_count()).sum::<usize>()
    );
    assert_eq!(
        ring.stored_relation_count(),
        expected.values().map(|s| s.relation_count()).sum::<usize>()
    );
}

fn record(
    ring: &mut SnapshotRing,
    expected: &mut BTreeMap<u64, WorldSnapshot>,
    frame: u64,
    world: &World,
) {
    expected.insert(frame, world.snapshot());
    while expected.len() > ring.capacity() {
        expected.pop_first();
    }
    ring.push(frame, world.snapshot());
    verify(ring, expected);
}

#[test]
fn generated_frames_match_full_snapshots_through_eviction_replacement_and_restore() {
    let mut world = world();
    world.insert_resource(External(99));
    let root = world.spawn(());
    let mut live = (0..8)
        .map(|i| world.spawn((value(i), Shared("pool".into()))))
        .collect::<Vec<_>>();
    let mut ring = SnapshotRing::new(7);
    let mut expected = BTreeMap::new();
    for step in 0..96u64 {
        world.increment_change_tick();
        if step % 4 == 0 {
            live.push(world.spawn((value(100 + step), Tag)));
        }
        if step % 5 == 0 && live.len() > 5 {
            assert!(world.despawn(live.remove(1)));
            // Exercise a reused allocator slot with a different generation.
            live.push(world.spawn((value(200 + step),)));
        }
        let entity = live[step as usize % live.len()];
        world
            .get_mut::<Value>(entity)
            .unwrap()
            .text
            .push_str(" edited");
        if step % 2 == 0 {
            world.insert(
                entity,
                (
                    Sparse(vec![format!("sparse-{step}")]),
                    Shared(format!("pool-{}", step % 3)),
                ),
            );
            world.insert_resource(Score(vec![format!("resource-{step}")]));
        } else {
            world.remove::<Sparse>(entity);
            world.remove_resource::<Score>();
        }
        if step % 3 == 0 {
            world.remove::<Tag>(entity);
            world.remove_relation::<Link>(entity, root);
        } else {
            world.insert(entity, Tag);
            world.add_relation::<Link>(entity, root);
        }
        world.register_relation::<Link>(
            RelationKind::new()
                .with_transitive(step % 3 == 0)
                .with_exclusive(step % 2 == 0),
        );
        let frame = if step < 32 { step } else { step * 17 % 48 };
        let source = world.snapshot();
        record(&mut ring, &mut expected, frame, &world);
        let (&selected, full) = expected.iter().nth(step as usize % expected.len()).unwrap();
        world.restore(&ring.get(selected).unwrap());
        assert!(
            full.structurally_eq(&world.snapshot()),
            "restore frame {selected}"
        );
        assert_eq!(world.resource::<External>().0, 99);
        world.restore(&source);
    }
}

#[test]
fn replaces_anchor_middle_and_tail_without_rewriting_other_frames() {
    let mut world = world();
    let entity = world.spawn(value(0));
    let mut ring = SnapshotRing::new(4);
    let mut expected = BTreeMap::new();
    for frame in [10, 20, 30, 40] {
        world.get_mut::<Value>(entity).unwrap().number = frame;
        record(&mut ring, &mut expected, frame, &world);
    }
    for frame in [10, 30, 40, 20, 10] {
        world.increment_change_tick();
        world.get_mut::<Value>(entity).unwrap().number = frame + 100;
        world.insert_resource(Score(vec![format!("replacement-{frame}")]));
        record(&mut ring, &mut expected, frame, &world);
    }
    // Rebase after all replacement positions, then continue the surviving chain.
    record(&mut ring, &mut expected, 50, &world);
    record(&mut ring, &mut expected, 60, &world);
}

#[test]
fn out_of_order_keys_are_numeric_and_full_history_discards_only_smallest_key() {
    let mut world = world();
    let entity = world.spawn(value(0));
    let mut ring = SnapshotRing::new(4);
    let mut expected = BTreeMap::new();
    // Prepend, middle insert, full middle insert (both promotion cases), ignored
    // stale input, and the boundary keys all have explicit numeric semantics.
    for frame in [30, 10, 20, 40, 15, 35, 0, 25, u64::MAX, u64::MAX - 1] {
        world.get_mut::<Value>(entity).unwrap().number = frame;
        record(&mut ring, &mut expected, frame, &world);
    }
    ring.truncate_after(39);
    expected.retain(|&frame, _| frame <= 39);
    verify(&ring, &expected);
    record(&mut ring, &mut expected, 38, &world);
    ring.truncate_after(0);
    expected.clear();
    verify(&ring, &expected);
    record(&mut ring, &mut expected, 0, &world);
    ring.clear();
    expected.clear();
    verify(&ring, &expected);
    record(&mut ring, &mut expected, u64::MAX, &world);
}

#[test]
fn capacity_one_empty_and_missing_frames() {
    let mut world = world();
    let entity = world.spawn(value(0));
    let mut ring = SnapshotRing::new(1);
    let mut expected = BTreeMap::new();
    verify(&ring, &expected);
    assert!(ring.get(0).is_none());
    assert!(ring.latest().is_none());
    for frame in [2, 1, 2, 4, u64::MAX] {
        world.get_mut::<Value>(entity).unwrap().number = frame;
        record(&mut ring, &mut expected, frame, &world);
        let (latest, snapshot) = ring.latest().unwrap();
        assert!(snapshot.structurally_eq(&expected[&latest]));
    }
    assert!(!ring.contains(2));
    assert!(ring.get(2).is_none());
}

#[test]
#[should_panic(expected = "SnapshotRing capacity must be non-zero")]
fn zero_capacity_is_rejected() {
    SnapshotRing::new(0);
}

#[test]
fn resource_presence_and_ordered_relation_indexes_survive_anchor_replacement_and_eviction() {
    let mut world = world();
    let a = world.spawn(());
    let b = world.spawn(());
    let c = world.spawn(());
    let mut ring = SnapshotRing::new(3);
    let mut expected = BTreeMap::new();
    world.add_relation::<Link>(a, c);
    world.add_relation::<Link>(b, c);
    record(&mut ring, &mut expected, 1, &world);
    world.insert_resource(Score(vec!["present".into()]));
    world.add_relation::<Link>(c, a);
    record(&mut ring, &mut expected, 2, &world);
    world.remove_relation::<Link>(a, c);
    world.add_relation::<Link>(a, c);
    world.register_relation::<Link>(RelationKind::new().with_exclusive(true));
    record(&mut ring, &mut expected, 3, &world);
    world.remove_resource::<Score>();
    world.remove_relation::<Link>(c, a);
    record(&mut ring, &mut expected, 1, &world);
    record(&mut ring, &mut expected, 4, &world);
    world.restore(&ring.get(3).unwrap());
    assert_eq!(world.relation_sources::<Link>(c), &[b, a]);
    assert_eq!(world.relation_targets::<Link>(c), &[a]);
    assert_eq!(world.resource::<Score>().0, ["present"]);
    let kind = world.components().id_of::<Link>().unwrap();
    assert_eq!(
        world.relations().kind(kind),
        Some(&RelationKind::new().with_exclusive(true))
    );
    world.restore(&ring.get(4).unwrap());
    assert!(world.get_resource::<Score>().is_none());
    assert!(world.relation_targets::<Link>(c).is_empty());
}

#[derive(Component, Clone, PartialEq, Eq)]
struct Collision(u64);

impl Hash for Collision {
    fn hash<H: Hasher>(&self, state: &mut H) {
        0u8.hash(state);
    }
}

#[test]
fn colliding_hashes_never_substitute_for_value_comparison() {
    let mut world = World::new();
    world.register_snapshot_component_hashable::<Collision>();
    world.register_snapshot_component_equatable::<Collision>();
    let entity = world.spawn(Collision(1));
    world.get_mut::<Collision>(entity).unwrap().0 = 1;
    let mut ring = SnapshotRing::new(2);
    let before = world.snapshot();
    ring.push(1, world.snapshot());
    // Same world/cell ticks, but a different semantic value with the same hash.
    world.get_mut::<Collision>(entity).unwrap().0 = 2;
    let after = world.snapshot();
    assert_eq!(before.state_hash(), after.state_hash());
    ring.push(2, world.snapshot());
    assert_eq!(ring.stored_component_cell_count(), 2);
    assert_eq!(ring.reused_component_cell_count(), 0);
    assert!(ring.get(1).unwrap().structurally_eq(&before));
    assert!(ring.get(2).unwrap().structurally_eq(&after));
    ring.push(3, world.snapshot());
    world.restore(&ring.get(2).unwrap());
    assert_eq!(world.get::<Collision>(entity).unwrap().0, 2);
}

#[derive(Component, Clone)]
struct Opaque(Vec<String>);

#[test]
fn clone_only_values_remain_restorable_without_false_compression_claims() {
    let mut world = World::new();
    world.register_snapshot_component::<Opaque>();
    let entity = world.spawn(Opaque(vec!["initial".into()]));
    let mut ring = SnapshotRing::new(3);
    for frame in 0..8 {
        world.get_mut::<Opaque>(entity).unwrap().0 = vec![format!("frame-{frame}")];
        ring.push(frame, world.snapshot());
    }
    assert_eq!(ring.stored_component_cell_count(), 3);
    assert_eq!(ring.reused_component_cell_count(), 0);
    for frame in 5..8 {
        world.restore(&ring.get(frame).unwrap());
        assert_eq!(
            world.get::<Opaque>(entity).unwrap().0,
            [format!("frame-{frame}")]
        );
    }
}

struct Faults {
    clone_budget: AtomicIsize,
    equality_budget: AtomicIsize,
    live: AtomicIsize,
}

impl Faults {
    fn new() -> Self {
        Self {
            clone_budget: AtomicIsize::new(-1),
            equality_budget: AtomicIsize::new(-1),
            live: AtomicIsize::new(0),
        }
    }

    fn disable(&self) {
        self.clone_budget.store(-1, Ordering::SeqCst);
        self.equality_budget.store(-1, Ordering::SeqCst);
    }
}

fn fault(budget: &AtomicIsize) {
    if budget.load(Ordering::SeqCst) >= 0 {
        assert_ne!(
            budget.fetch_sub(1, Ordering::SeqCst),
            0,
            "injected snapshot value panic"
        );
    }
}

#[derive(Component)]
struct Fragile {
    value: u64,
    faults: Arc<Faults>,
}

impl Fragile {
    fn new(value: u64, faults: &Arc<Faults>) -> Self {
        faults.live.fetch_add(1, Ordering::SeqCst);
        Self {
            value,
            faults: faults.clone(),
        }
    }
}

impl Clone for Fragile {
    fn clone(&self) -> Self {
        fault(&self.faults.clone_budget);
        Self::new(self.value, &self.faults)
    }
}

impl PartialEq for Fragile {
    fn eq(&self, other: &Self) -> bool {
        fault(&self.faults.equality_budget);
        self.value == other.value
    }
}

impl Drop for Fragile {
    fn drop(&mut self) {
        self.faults.live.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Resource, Clone, PartialEq)]
struct FragileResource(Fragile);

#[test]
fn storage_retains_one_anchor_and_changed_cells_not_read_caches_or_full_frames() {
    let faults = Arc::new(Faults::new());
    {
        let mut world = World::new();
        world.register_snapshot_component_equatable::<Fragile>();
        let entities = (0..32)
            .map(|i| world.spawn(Fragile::new(i, &faults)))
            .collect::<Vec<_>>();
        let mut ring = SnapshotRing::new(8);
        for frame in 0..24 {
            world.increment_change_tick();
            world
                .get_mut::<Fragile>(entities[frame as usize % entities.len()])
                .unwrap()
                .value += 100;
            ring.push(frame, world.snapshot());
            let deltas = ring.len() - 1;
            assert_eq!(ring.full_snapshot_count(), 1);
            assert_eq!(ring.stored_component_cell_count(), 32 + deltas);
            assert_eq!(ring.reused_component_cell_count(), 31 * deltas);
            let retained = 32 + ring.stored_component_cell_count() as isize;
            assert_eq!(faults.live.load(Ordering::SeqCst), retained);
            for key in ring.frames() {
                let rebuilt = ring.get(key).unwrap();
                assert_eq!(faults.live.load(Ordering::SeqCst), retained + 32);
                drop(rebuilt);
                assert_eq!(faults.live.load(Ordering::SeqCst), retained);
            }
        }
        let detached = ring.get(20).unwrap();
        ring.clear();
        assert_eq!(faults.live.load(Ordering::SeqCst), 64);
        world.restore(&detached);
        drop(detached);
        assert_eq!(faults.live.load(Ordering::SeqCst), 32);
    }
    assert_eq!(faults.live.load(Ordering::SeqCst), 0);
}

#[test]
fn every_clone_or_equality_failure_before_commit_leaves_original_history_intact() {
    // All positional mutation paths, including a full insertion whose input
    // becomes the new anchor and a full append that promotes an old delta.
    for (capacity, frame) in [
        (5, 5),
        (5, 15),
        (3, 10),
        (3, 20),
        (3, 30),
        (3, 15),
        (3, 25),
        (3, 40),
    ] {
        for comparison_failure in [false, true] {
            let mut reached_success = false;
            let mut failures = 0;
            for budget in 0..128 {
                let faults = Arc::new(Faults::new());
                {
                    let mut world = World::new();
                    world.register_snapshot_component_equatable::<Fragile>();
                    world.register_snapshot_resource_equatable::<FragileResource>();
                    world.insert_resource(FragileResource(Fragile::new(100, &faults)));
                    let entities = (0..3)
                        .map(|i| world.spawn(Fragile::new(i, &faults)))
                        .collect::<Vec<_>>();
                    let mut ring = SnapshotRing::new(capacity);
                    let mut expected = BTreeMap::new();
                    for key in [10, 20, 30] {
                        world.increment_change_tick();
                        world.get_mut::<Fragile>(entities[0]).unwrap().value = key;
                        record(&mut ring, &mut expected, key, &world);
                    }
                    world.increment_change_tick();
                    world.get_mut::<Fragile>(entities[0]).unwrap().value = 999;
                    world.resource_mut::<FragileResource>().0.value = 777;
                    let target = world.snapshot();
                    let baseline_live = faults.live.load(Ordering::SeqCst);
                    let input = world.snapshot();
                    if comparison_failure {
                        faults.equality_budget.store(budget, Ordering::SeqCst);
                    } else {
                        faults.clone_budget.store(budget, Ordering::SeqCst);
                    }
                    let result = catch_unwind(AssertUnwindSafe(|| ring.push(frame, input)));
                    faults.disable();
                    if result.is_err() {
                        failures += 1;
                        assert_eq!(faults.live.load(Ordering::SeqCst), baseline_live, "leaked preparation: capacity {capacity}, frame {frame}, budget {budget}");
                    } else {
                        reached_success = true;
                        expected.insert(frame, target);
                        while expected.len() > capacity {
                            expected.pop_first();
                        }
                    }
                    verify(&ring, &expected);
                }
                assert_eq!(
                    faults.live.load(Ordering::SeqCst),
                    0,
                    "leaked a component/resource after unwind"
                );
                if reached_success {
                    break;
                }
            }
            assert!(reached_success, "did not exhaust all fallible operations");
            assert!(failures > 0, "failure injection never exercised this path");
        }
    }
}

#[test]
fn failed_read_and_ignored_stale_push_do_not_mutate_or_cache_history() {
    let faults = Arc::new(Faults::new());
    {
        let mut world = World::new();
        world.register_snapshot_component_equatable::<Fragile>();
        world.spawn(Fragile::new(1, &faults));
        let mut ring = SnapshotRing::new(2);
        let mut expected = BTreeMap::new();
        record(&mut ring, &mut expected, 10, &world);
        record(&mut ring, &mut expected, 20, &world);
        for key in [10, 20] {
            let baseline = faults.live.load(Ordering::SeqCst);
            faults.clone_budget.store(0, Ordering::SeqCst);
            assert!(catch_unwind(AssertUnwindSafe(|| ring.get(key))).is_err());
            faults.disable();
            assert_eq!(faults.live.load(Ordering::SeqCst), baseline);
            verify(&ring, &expected);
        }
        let input = world.snapshot();
        faults.clone_budget.store(0, Ordering::SeqCst);
        faults.equality_budget.store(0, Ordering::SeqCst);
        ring.push(0, input);
        assert_eq!(faults.clone_budget.load(Ordering::SeqCst), 0);
        assert_eq!(faults.equality_budget.load(Ordering::SeqCst), 0);
        assert!(!ring.contains(0));
        assert!(ring.get(0).is_none());
        faults.disable();
        verify(&ring, &expected);
    }
    assert_eq!(faults.live.load(Ordering::SeqCst), 0);
}

#[test]
fn different_world_registries_are_fresh_edges_and_survive_rebase() {
    let mut first = world();
    first.spawn(value(1));
    let mut second = world();
    let entity = second.spawn(value(2));
    let mut ring = SnapshotRing::new(2);
    let mut expected = BTreeMap::new();
    record(&mut ring, &mut expected, 1, &first);
    record(&mut ring, &mut expected, 2, &second);
    assert_eq!(ring.stored_component_cell_count(), 2);
    assert_eq!(ring.reused_component_cell_count(), 0);
    record(&mut ring, &mut expected, 3, &second);
    second.restore(&ring.get(2).unwrap());
    assert_eq!(second.get::<Value>(entity).unwrap().number, 2);
}
