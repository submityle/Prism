//! Correctness and operation-count regressions for unchanged flushes.

use super::*;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone)]
struct Value(u32);
impl Component for Value {}

struct Sparse(u32);
impl Component for Sparse {
    const STORAGE: crate::component::StorageType = crate::component::StorageType::SparseSet;
}

fn count(counter: &Arc<AtomicUsize>) -> impl FnMut(&mut ChangeContext<'_>) + Send + Sync + 'static {
    let counter = counter.clone();
    move |ctx| {
        counter.fetch_add(ctx.len(), Ordering::Relaxed);
    }
}

fn assert_idle(world: &mut World) {
    let tick = world.change_tick();
    assert_eq!(world.flush_changed_observers(), 0);
    assert_eq!(world.change_tick().get(), tick.get().wrapping_add(1));
    assert_eq!(
        world.change_observers.stats,
        FlushStats {
            revision_checks: 1,
            idle_fast_path: true,
            ..FlushStats::default()
        }
    );
}

#[test]
fn idle_cost_does_not_visit_active_or_removed_historical_slots() {
    for slots in [1, 32, 1_024] {
        let mut world = World::new();
        let calls = Arc::new(AtomicUsize::new(0));
        for _ in 0..slots {
            let id = world.observe_changed::<Value, _>(count(&calls));
            assert!(world.remove_change_observer(id));
        }
        for _ in 0..slots {
            world.observe_changed::<Value, _>(count(&calls));
        }
        assert_eq!(world.flush_changed_observers(), 0);
        assert_eq!(world.change_observers.stats.collections, 0);
        assert_eq!(world.change_observers.stats.candidate_slots, 0);
        assert_eq!(world.change_observers.stats.slot_visits, 0);
        for _ in 0..4 {
            assert_idle(&mut world);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn nonempty_flush_visits_only_active_candidates_after_heavy_churn() {
    let mut world = World::new();
    let calls = Arc::new(AtomicUsize::new(0));
    for _ in 0..2_048 {
        let id = world.observe_changed::<Value, _>(count(&calls));
        assert!(world.remove_change_observer(id));
    }
    let entity = world.spawn((Value(0), Sparse(0)));
    world.observe_changed::<Value, _>(count(&calls));
    world.observe_changed::<Sparse, _>(count(&calls));
    assert_eq!(world.flush_changed_observers(), 0);

    world.get_mut::<Value>(entity).unwrap().0 += 1;
    world.get_mut::<Sparse>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(world.change_observers.stats.candidate_slots, 2);
    assert_eq!(world.change_observers.stats.slot_visits, 4);
    assert_eq!(world.change_observers.stats.collections, 2);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn nonempty_flush_visits_only_observers_for_dirty_components() {
    let mut world = World::new();
    let entity = world.spawn((Value(0), Sparse(0)));
    let value_calls = Arc::new(AtomicUsize::new(0));
    let sparse_calls = Arc::new(AtomicUsize::new(0));
    for _ in 0..1_024 {
        world.observe_changed::<Value, _>(count(&value_calls));
    }
    world.observe_changed::<Sparse, _>(count(&sparse_calls));
    assert_eq!(world.flush_changed_observers(), 0);

    world.get_mut::<Sparse>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(world.change_observers.stats.candidate_slots, 1);
    assert_eq!(world.change_observers.stats.slot_visits, 2);
    assert_eq!(world.change_observers.stats.collections, 1);
    assert_eq!(value_calls.load(Ordering::Relaxed), 0);
    assert_eq!(sparse_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn immediate_table_and_sparse_writes_after_idle_are_not_skipped() {
    let mut world = World::new();
    let entity = world.spawn((Value(0), Sparse(0)));
    let calls = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&calls));
    world.observe_changed::<Sparse, _>(count(&calls));
    assert_eq!(world.flush_changed_observers(), 0);
    assert_idle(&mut world);
    world.get_mut::<Value>(entity).unwrap().0 += 1;
    world.get_mut::<Sparse>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_idle(&mut world);
}

#[test]
fn registration_after_idle_keeps_its_own_baseline() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let old = Arc::new(AtomicUsize::new(0));
    let new = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&old));
    assert_eq!(world.flush_changed_observers(), 0);
    assert_idle(&mut world);
    world.get_mut::<Value>(entity).unwrap().0 = 1;
    world.observe_changed::<Value, _>(count(&new));
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(old.load(Ordering::Relaxed), 1);
    assert_eq!(new.load(Ordering::Relaxed), 0);
    assert_idle(&mut world);
    world.get_mut::<Value>(entity).unwrap().0 = 2;
    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(new.load(Ordering::Relaxed), 1);
}

#[test]
fn callback_writes_invalidate_clean_revision_for_next_flush() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let calls = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>({
        let calls = calls.clone();
        move |ctx| {
            if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                ctx.world.get_mut::<Value>(entity).unwrap().0 += 1;
            }
        }
    });
    world.get_mut::<Value>(entity).unwrap().0 = 1;
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_idle(&mut world);
}

#[test]
fn idle_flush_still_settles_reactions_and_keeps_their_writes_pending() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let calls = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&calls));
    let source = world.reaction_source::<Sparse>();
    world.add_reaction(&[source], move |world| {
        world.get_mut::<Value>(entity).unwrap().0 += 1;
    });
    world.flush_changed_observers();
    // Settle any initial effect and establish an exact clean baseline.
    world.flush_changed_observers();
    let before = calls.load(Ordering::Relaxed);
    world.mark_reaction_dirty(source);
    assert_eq!(world.flush_changed_observers(), 0);
    assert!(world.change_observers.stats.idle_fast_path);
    assert_eq!(world.change_observers.stats.slot_visits, 0);
    assert_eq!(world.change_observers.stats.collections, 0);
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), before + 1);
    assert_idle(&mut world);
}

#[test]
fn restore_invalidates_idle_revision_and_accepts_same_tick_write() {
    let mut world = World::new();
    world.register_snapshot_component::<Value>();
    let entity = world.spawn(Value(0));
    let calls = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&calls));
    let snapshot = world.snapshot();
    world.flush_changed_observers();
    assert_idle(&mut world);
    world.get_mut::<Value>(entity).unwrap().0 = 10;
    world.restore(&snapshot);
    assert_eq!(world.flush_changed_observers(), 0);
    assert_idle(&mut world);
    world.restore(&snapshot);
    let tick = world.change_tick();
    world.get_mut::<Value>(entity).unwrap().0 = 1;
    assert_eq!(world.change_tick(), tick);
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_idle(&mut world);
}

#[test]
fn removal_despawn_and_generation_reuse_do_not_replay_stale_changes() {
    let mut world = World::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let id = world.observe_changed::<Value, _>(count(&calls));
    let entity = world.spawn(Value(0));
    world.flush_changed_observers();
    assert_idle(&mut world);
    world.get_mut::<Value>(entity).unwrap().0 = 1;
    assert!(world.despawn(entity));
    assert_eq!(world.flush_changed_observers(), 0);
    assert_idle(&mut world);
    let replacement = world.spawn(Value(2));
    assert_eq!(world.flush_changed_observers(), 1);
    assert_ne!(entity, replacement);
    assert!(world.remove_change_observer(id));
    let after = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&after));
    assert_eq!(world.flush_changed_observers(), 0);
    assert_idle(&mut world);
    world.get_mut::<Value>(replacement).unwrap().0 = 3;
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(after.load(Ordering::Relaxed), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[cfg(feature = "std")]
#[test]
fn active_index_preserves_order_across_remove_and_readd() {
    use std::sync::Mutex;

    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let order = Arc::new(Mutex::new(Vec::new()));
    let observe = |label, order: &Arc<Mutex<Vec<u32>>>| {
        let order = order.clone();
        move |_: &mut ChangeContext<'_>| order.lock().unwrap().push(label)
    };

    world.observe_changed::<Value, _>(observe(1, &order));
    let removed = world.observe_changed::<Value, _>(observe(2, &order));
    world.observe_changed::<Value, _>(observe(3, &order));
    assert_eq!(world.flush_changed_observers(), 0);
    assert!(world.remove_change_observer(removed));
    world.observe_changed::<Value, _>(observe(4, &order));

    world.get_mut::<Value>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_eq!(*order.lock().unwrap(), [1, 3, 4]);
}

#[cfg(feature = "std")]
#[test]
fn callback_registration_and_removal_do_not_change_current_candidates() {
    use std::sync::Mutex;

    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let order = Arc::new(Mutex::new(Vec::new()));
    let removed_id = Arc::new(std::sync::OnceLock::<ObserverId>::new());
    let registered = Arc::new(core::sync::atomic::AtomicBool::new(false));

    let order_a = order.clone();
    let removed_id_a = removed_id.clone();
    let registered_a = registered.clone();
    world.observe_changed::<Value, _>(move |ctx| {
        order_a.lock().unwrap().push(1);
        if !registered_a.swap(true, Ordering::Relaxed) {
            let id = *removed_id_a.get().expect("later observer id");
            assert!(ctx.world.remove_change_observer(id));
            let order_d = order_a.clone();
            ctx.world.observe_changed::<Value, _>(move |_| {
                order_d.lock().unwrap().push(4);
            });
        }
    });
    let removed = world.observe_changed::<Value, _>({
        let order = order.clone();
        move |_| order.lock().unwrap().push(2)
    });
    assert!(removed_id.set(removed).is_ok());
    world.observe_changed::<Value, _>({
        let order = order.clone();
        move |_| order.lock().unwrap().push(3)
    });
    assert_eq!(world.flush_changed_observers(), 0);

    world.get_mut::<Value>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(*order.lock().unwrap(), [1, 3]);

    world.get_mut::<Value>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_eq!(*order.lock().unwrap(), [1, 3, 1, 3, 4]);
}

#[cfg(feature = "std")]
#[test]
fn panic_after_idle_retries_pending_without_new_writes_or_replaying_successes() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let success = Arc::new(AtomicUsize::new(0));
    let attempts = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&success));
    world.observe_changed::<Value, _>({
        let attempts = attempts.clone();
        move |_| {
            assert_ne!(attempts.fetch_add(1, Ordering::Relaxed), 0, "retry");
        }
    });
    world.flush_changed_observers();
    assert_idle(&mut world);
    world.get_mut::<Value>(entity).unwrap().0 = 1;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.flush_changed_observers();
    }))
    .is_err());
    assert!(!world.change_observers.flushing);
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(world.change_observers.stats.candidate_slots, 1);
    assert_eq!(success.load(Ordering::Relaxed), 1);
    assert_eq!(attempts.load(Ordering::Relaxed), 2);
    assert_idle(&mut world);
}

#[cfg(feature = "std")]
#[test]
fn world_replacement_does_not_clear_replacement_panic_retry_index() {
    let mut replacement = World::new();
    let replacement_entity = replacement.spawn(Value(0));
    let replacement_attempts = Arc::new(AtomicUsize::new(0));
    replacement.observe_changed::<Value, _>({
        let replacement_attempts = replacement_attempts.clone();
        move |_| {
            if replacement_attempts.fetch_add(1, Ordering::Relaxed) == 0 {
                panic!("replacement observer retry");
            }
        }
    });
    replacement.get_mut::<Value>(replacement_entity).unwrap().0 = 1;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        replacement.flush_changed_observers();
    }))
    .is_err());
    assert_eq!(replacement.change_observers.pending_indices, [0]);

    let replacement_id = replacement.id();
    let mut replacement = Some(replacement);
    let mut world = World::new();
    let old_entity = world.spawn(Value(0));
    world.observe_changed::<Value, _>(move |ctx| {
        *ctx.world = replacement.take().expect("old observer must run once");
    });
    world.get_mut::<Value>(old_entity).unwrap().0 = 1;
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(world.id(), replacement_id);

    // No new write occurs in the replacement World. Its old panic batch must
    // still be selected solely through the preserved pending index.
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(replacement_attempts.load(Ordering::Relaxed), 2);
    assert_eq!(world.flush_changed_observers(), 0);
}
