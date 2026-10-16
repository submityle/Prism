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
        assert_eq!(world.change_observers.stats.collections, slots);
        assert_eq!(world.change_observers.stats.slot_visits, 4 * slots);
        for _ in 0..4 {
            assert_idle(&mut world);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
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
    assert_eq!(success.load(Ordering::Relaxed), 1);
    assert_eq!(attempts.load(Ordering::Relaxed), 2);
    assert_idle(&mut world);
}
