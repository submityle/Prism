//! The UI adapter's real delta storage, navigation and World restoration.
#![cfg(feature = "ecs")]

use prism_ecs::{
    component::{Component, StorageType},
    relation::RelationKind,
    resource::Resource,
    world::{snapshot::WorldSnapshot, World},
};
use prism_ui::Element;
use prism_ui_timetravel::{EcsFrame, EcsTimeline, Frame};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicIsize, Ordering},
        Arc,
    },
};

#[derive(Clone, PartialEq, Eq)]
struct Value(String);
impl Component for Value {}

#[derive(Clone, PartialEq, Eq)]
struct Sparse(Vec<i32>);
impl Component for Sparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[derive(prism_ecs::prelude::Component, Clone, PartialEq, Eq, Hash)]
#[component(storage = "Shared")]
struct Shared(String);

struct Link;
impl Component for Link {}

#[derive(Clone, PartialEq, Eq)]
struct Score(Vec<String>);
impl Resource for Score {}

struct External(i32);
impl Resource for External {}

fn world() -> World {
    let mut world = World::new();
    world.register_snapshot_component_equatable::<Value>();
    world.register_snapshot_component_equatable::<Sparse>();
    world.register_snapshot_component_equatable::<Shared>();
    world.register_snapshot_resource_equatable::<Score>();
    world.register_relation::<Link>(RelationKind::new());
    world
}

fn assert_history(history: &EcsTimeline, expected: &[(String, WorldSnapshot)]) {
    assert_eq!(history.len(), expected.len());
    assert_eq!(history.is_empty(), expected.is_empty());
    let storage = history.storage();
    assert_eq!(storage.full_snapshots, usize::from(!expected.is_empty()));
    assert_eq!(storage.deltas, expected.len().saturating_sub(1));
    let mut owned = 0;
    let mut reused = 0;
    for (index, (label, full)) in expected.iter().enumerate() {
        let frame = &history.frames()[index];
        assert_eq!(frame.label(), label);
        let rebuilt = frame.snapshot();
        // Typed semantic comparison, NOT a hash-equality approximation.
        assert!(rebuilt.structurally_eq(full), "checkpoint {index}: {label}");
        if index == 0 {
            let cells = full.diff(full);
            owned += cells.changed_cell_count() + cells.reused_cell_count();
        } else {
            let delta = expected[index - 1].1.diff(full);
            owned += delta.changed_cell_count();
            reused += delta.reused_cell_count();
        }
    }
    assert_eq!(storage.component_cells, owned);
    assert_eq!(storage.reused_component_cells, reused);
    assert_eq!(
        storage.resources,
        expected
            .iter()
            .map(|(_, s)| s.resource_count())
            .sum::<usize>()
    );
    assert_eq!(
        storage.relation_edges,
        expected
            .iter()
            .map(|(_, s)| s.relation_count())
            .sum::<usize>()
    );
}

#[test]
fn generated_branching_and_eviction_restore_exact_full_snapshot_oracles() {
    for capacity in [0, 1, 4] {
        let mut world = world();
        let root = world.spawn(());
        let a = world.spawn((Value("a".into()), Shared("initial".into())));
        let b = world.spawn(Value("b".into()));
        world.insert_resource(External(0));
        let mut history = EcsTimeline::new(&world, capacity);
        let mut expected: Vec<(String, WorldSnapshot)> = Vec::new();
        let mut ephemeral = None;
        for step in 0..36 {
            // Rewind at multiple retained positions and then record to fork.
            if step % 7 == 6 && history.len() > 1 {
                let index = step as usize % (history.len() - 1);
                history.seek(&mut world, index).unwrap();
                assert!(world.snapshot().structurally_eq(&expected[index].1));
                assert!(history.can_redo());
                expected.truncate(index + 1);
            }
            world.increment_change_tick();
            world.get_mut::<Value>(a).unwrap().0 = format!("value-{step}");
            world.insert(a, Shared(format!("batch-{}", step % 3)));
            if step % 2 == 0 {
                world.insert(a, Sparse(vec![step, step + 1]));
                world.insert_resource(Score(vec![format!("resource-{step}")]));
            } else {
                world.remove::<Sparse>(a);
                world.remove_resource::<Score>();
            }
            if let Some(previous) = ephemeral.take() {
                world.despawn(previous);
            }
            if step % 3 != 0 {
                ephemeral = Some(world.spawn(Value(format!("ephemeral-{step}"))));
            }
            world.register_relation::<Link>(RelationKind::new().with_transitive(step % 2 == 0));
            world.add_relation::<Link>(a, root);
            world.add_relation::<Link>(b, root);
            // Exercise both forward and reverse adjacency order, not just counts.
            if step % 3 == 0 {
                world.remove_relation::<Link>(a, root);
                world.add_relation::<Link>(a, root);
                world.add_relation::<Link>(a, b);
            } else {
                world.remove_relation::<Link>(a, b);
            }
            world.resource_mut::<External>().0 = step;
            let label = format!("frame-{step}");
            let ui = (step % 3 == 0).then(|| Frame::capture(&label, &Element::text(&label)));
            expected.push((label.clone(), world.snapshot()));
            let recorded = history.record_with_ui(&world, label.clone(), ui).unwrap();
            if capacity != 0 && expected.len() > capacity {
                expected.remove(0);
            }
            assert_eq!(recorded, expected.len() - 1);
            assert_eq!(history.cursor(), Some(recorded));
            assert!(!history.can_redo());
            assert_history(&history, &expected);
            if step % 3 == 0 {
                assert_eq!(history.current().unwrap().ui().unwrap().label(), label);
            } else {
                assert!(history.current().unwrap().ui().is_none());
            }

            // Every generated retained state must restore ticks, allocator,
            // table/sparse/shared values, resources and ordered relations.
            for (index, (_, full)) in expected.iter().enumerate() {
                history.seek(&mut world, index).unwrap();
                assert!(world.snapshot().structurally_eq(full));
                assert_eq!(world.resource::<External>().0, step);
            }
            let latest = expected.len() - 1;
            history.seek(&mut world, latest).unwrap();
            if latest > 0 {
                history.undo(&mut world).unwrap().unwrap();
                assert!(world.snapshot().structurally_eq(&expected[latest - 1].1));
                history.redo(&mut world).unwrap().unwrap();
                assert!(world.snapshot().structurally_eq(&expected[latest].1));
            } else {
                assert!(history.undo(&mut world).unwrap().is_none());
            }
            assert!(history.redo(&mut world).unwrap().is_none());
        }
        // Retain an owned read across clear and restart the same history.
        let detached = history.current().unwrap().snapshot();
        history.clear();
        assert_history(&history, &[]);
        assert_eq!(history.cursor(), None);
        assert_eq!(history.capacity(), (capacity != 0).then_some(capacity));
        world.restore(&detached);
        assert!(world.snapshot().structurally_eq(&detached));
        assert_eq!(history.record(&world, "restart").unwrap(), 0);
        assert!(history
            .current()
            .unwrap()
            .snapshot()
            .structurally_eq(&detached));
    }
}

#[test]
fn zero_capacity_repeated_growth_retains_every_frame_and_only_branch_discards_redo() {
    let mut world = world();
    let entity = world.spawn(Value("initial".into()));
    let mut history = EcsTimeline::new(&world, 0);
    let mut expected: Vec<(String, WorldSnapshot)> = Vec::new();
    for step in 0..40 {
        world.increment_change_tick();
        world.get_mut::<Value>(entity).unwrap().0 = format!("frame-{step}");
        let label = format!("frame-{step}");
        expected.push((label.clone(), world.snapshot()));
        assert_eq!(history.record(&world, label).unwrap(), step);
    }
    assert_eq!(history.capacity(), None);
    assert_history(&history, &expected);
    history.seek(&mut world, 7).unwrap();
    assert_eq!(history.len(), 40);
    world.get_mut::<Value>(entity).unwrap().0 = "fork".into();
    expected.truncate(8);
    expected.push(("fork".into(), world.snapshot()));
    assert_eq!(history.record(&world, "fork").unwrap(), 8);
    assert_history(&history, &expected);
    for step in 0..32 {
        let label = format!("fork-{step}");
        world.get_mut::<Value>(entity).unwrap().0 = label.clone();
        expected.push((label.clone(), world.snapshot()));
        history.record(&world, label).unwrap();
    }
    assert_history(&history, &expected);
    for (index, (_, full)) in expected.iter().enumerate() {
        history.seek(&mut world, index).unwrap();
        assert!(world.snapshot().structurally_eq(full));
    }
}

struct Counted {
    value: usize,
    live: Arc<AtomicIsize>,
    panic_eq: Arc<AtomicBool>,
}
impl Component for Counted {}

impl Counted {
    fn new(value: usize, live: &Arc<AtomicIsize>, panic_eq: &Arc<AtomicBool>) -> Self {
        live.fetch_add(1, Ordering::SeqCst);
        Self {
            value,
            live: live.clone(),
            panic_eq: panic_eq.clone(),
        }
    }
}

impl Clone for Counted {
    fn clone(&self) -> Self {
        Self::new(self.value, &self.live, &self.panic_eq)
    }
}

impl PartialEq for Counted {
    fn eq(&self, other: &Self) -> bool {
        assert!(
            !self.panic_eq.load(Ordering::SeqCst),
            "injected delta comparison panic"
        );
        self.value == other.value
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

#[test]
fn retained_values_are_one_anchor_plus_dirty_cells_and_reads_never_cache_full_frames() {
    for capacity in [0, 1, 5] {
        let live = Arc::new(AtomicIsize::new(0));
        let panic_eq = Arc::new(AtomicBool::new(false));
        {
            let mut world = World::new();
            world.register_snapshot_component_equatable::<Counted>();
            let entities = (0..12)
                .map(|i| world.spawn(Counted::new(i, &live, &panic_eq)))
                .collect::<Vec<_>>();
            let mut history = EcsTimeline::new(&world, capacity);
            for step in 0..40 {
                world.increment_change_tick();
                world
                    .get_mut::<Counted>(entities[step % entities.len()])
                    .unwrap()
                    .value += 100;
                history.record(&world, format!("frame-{step}")).unwrap();
                let storage = history.storage();
                assert_eq!(storage.full_snapshots, 1);
                assert_eq!(storage.component_cells, 12 + history.len() - 1);
                assert_eq!(storage.reused_component_cells, 11 * (history.len() - 1));
                let retained = 12 + storage.component_cells as isize;
                assert_eq!(live.load(Ordering::SeqCst), retained);
                for frame in history.frames() {
                    let owned = frame.snapshot();
                    assert_eq!(live.load(Ordering::SeqCst), retained + 12);
                    drop(owned);
                    assert_eq!(live.load(Ordering::SeqCst), retained);
                }
            }
            let detached = history.current().unwrap().snapshot();
            history.clear();
            assert_eq!(live.load(Ordering::SeqCst), 24);
            world.restore(&detached);
            drop(detached);
            assert_eq!(live.load(Ordering::SeqCst), 12);
        }
        assert_eq!(live.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn delta_comparison_panics_preserve_redo_eviction_window_and_unbounded_growth() {
    // Ordinary eviction, replacement of a redo frame, and geometric expansion.
    for (capacity, rewind) in [(3, None), (4, Some(0)), (0, None)] {
        let live = Arc::new(AtomicIsize::new(0));
        let panic_eq = Arc::new(AtomicBool::new(false));
        {
            let mut world = World::new();
            world.register_snapshot_component_equatable::<Counted>();
            let entity = world.spawn(Counted::new(0, &live, &panic_eq));
            let mut history = EcsTimeline::new(&world, capacity);
            let mut expected: Vec<(String, WorldSnapshot)> = Vec::new();
            let count = if capacity == 0 { 2 } else { 3 };
            for step in 0..count {
                world.get_mut::<Counted>(entity).unwrap().value = step;
                let label = format!("frame-{step}");
                expected.push((label.clone(), world.snapshot()));
                history.record(&world, label).unwrap();
            }
            if let Some(index) = rewind {
                history.seek(&mut world, index).unwrap();
            }
            let cursor = history.cursor();
            let storage = history.storage();
            let retained = live.load(Ordering::SeqCst);
            panic_eq.store(true, Ordering::SeqCst);
            let result = catch_unwind(AssertUnwindSafe(|| history.record(&world, "rejected")));
            panic_eq.store(false, Ordering::SeqCst);
            assert!(result.is_err());
            assert_eq!(history.cursor(), cursor);
            assert_eq!(history.storage(), storage);
            assert_eq!(history.can_redo(), rewind.is_some());
            assert_eq!(live.load(Ordering::SeqCst), retained);
            assert_history(&history, &expected);
            // Recover the poisoned write lock and keep using the same history.
            history.record(&world, "accepted").unwrap();
        }
        assert_eq!(live.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn delta_backing_preserves_send_and_sync_of_the_public_history_types() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<EcsTimeline>();
    assert_send_sync::<EcsFrame>();
}

#[test]
fn clone_only_glue_keeps_values_without_claiming_delta_cell_reuse() {
    let mut world = World::new();
    world.register_snapshot_component::<Value>();
    let entity = world.spawn(Value("unchanged".into()));
    let mut history = EcsTimeline::new(&world, 3);
    for _ in 0..6 {
        history.record(&world, "clone only").unwrap();
    }
    assert_eq!(history.storage().full_snapshots, 1);
    assert_eq!(history.storage().deltas, 2);
    assert_eq!(history.storage().component_cells, 3);
    assert_eq!(history.storage().reused_component_cells, 0);
    for index in 0..history.len() {
        history.seek(&mut world, index).unwrap();
        assert_eq!(world.get::<Value>(entity).unwrap().0, "unchanged");
    }
}
