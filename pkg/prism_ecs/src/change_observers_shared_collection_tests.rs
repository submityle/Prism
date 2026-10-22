//! Value-oracle and actual tick-check regressions for shared collection.

use super::*;
use crate::resource::Resource;

#[derive(Clone)]
struct Value<const SPARSE: bool>(u32);

impl<const SPARSE: bool> Component for Value<SPARSE> {
    const STORAGE: StorageType = if SPARSE {
        StorageType::SparseSet
    } else {
        StorageType::Table
    };
}

#[derive(Debug, PartialEq, Eq)]
struct Delivery {
    label: usize,
    rows: Vec<(Entity, u32)>,
}

#[derive(Default)]
struct Log(Vec<Delivery>);
impl Resource for Log {}

fn record<const S: bool>(ctx: &mut ChangeContext<'_>, label: usize) {
    let rows = ctx
        .changed()
        .iter()
        .map(|&entity| (entity, ctx.world.get::<Value<S>>(entity).unwrap().0))
        .collect();
    ctx.world
        .resource_mut::<Log>()
        .0
        .push(Delivery { label, rows });
}

fn subscribe<const S: bool>(world: &mut World, label: usize) -> ObserverId {
    world.observe_changed::<Value<S>, _>(move |ctx| record::<S>(ctx, label))
}

fn take_log(world: &mut World) -> Vec<Delivery> {
    core::mem::take(&mut world.resource_mut::<Log>().0)
}

fn delivery(label: usize, rows: &[(Entity, u32)]) -> Delivery {
    Delivery {
        label,
        rows: rows.to_vec(),
    }
}

fn assert_shared(world: &World, observers: usize, tick_checks: usize) {
    let stats = world.change_observers.stats;
    assert_eq!(stats.collections, 1, "{stats:?}");
    assert_eq!(stats.collection_cache_hits, observers - 1, "{stats:?}");
    assert_eq!(stats.tick_checks, tick_checks, "{stats:?}");
}

fn warm<const S: bool>(world: &mut World, entity: Entity, observers: usize) {
    world.get_mut::<Value<S>>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), observers);
    assert_eq!(take_log(world).len(), observers);
}

fn scaling_matrix<const S: bool>() {
    // U stays at three, with unrelated live rows proving this is not a scan.
    // W counts actual mutable writes; M counts independent subscriptions.
    for observers in [1, 8, 64] {
        for writes_per_entity in [1, 8, 128_u32] {
            let mut world = World::new();
            world.insert_resource(Log::default());
            let entities: Vec<_> = (0..32).map(|i| world.spawn(Value::<S>(i))).collect();
            for label in 0..observers {
                subscribe::<S>(&mut world, label);
            }
            // Initial registrations have different boundaries. Only a real
            // collection aligns their cursors/windows; do not fake cache hits.
            warm::<S>(&mut world, entities[0], observers);

            for _ in 0..writes_per_entity {
                for &entity in entities[..3].iter().rev() {
                    world.get_mut::<Value<S>>(entity).unwrap().0 += 1;
                }
            }
            let expected = [
                (entities[0], 1 + writes_per_entity),
                (entities[1], 1 + writes_per_entity),
                (entities[2], 2 + writes_per_entity),
            ];
            assert_eq!(world.flush_changed_observers(), observers);
            assert_eq!(
                take_log(&mut world),
                (0..observers)
                    .map(|label| delivery(label, &expected))
                    .collect::<Vec<_>>()
            );
            assert_shared(&world, observers, 3);
            assert_eq!(
                world.change_observers.stats.log_events,
                3 * writes_per_entity as usize
            );

            // Every subscriber must own the advanced cursor, not just the
            // subscriber whose callback populated the shared collection.
            world.get_mut::<Value<S>>(entities[2]).unwrap().0 = 777;
            assert_eq!(world.flush_changed_observers(), observers);
            assert_shared(&world, observers, 1);
            assert_eq!(
                take_log(&mut world),
                (0..observers)
                    .map(|label| delivery(label, &[(entities[2], 777)]))
                    .collect::<Vec<_>>()
            );
            assert_eq!(world.flush_changed_observers(), 0);
            assert_eq!(world.change_observers.stats.tick_checks, 0);
            assert_eq!(world.change_observers.stats.collection_cache_hits, 0);
            assert!(take_log(&mut world).is_empty());
        }
    }
}

#[test]
fn shared_table_tick_checks_scale_with_u_not_w_or_m() {
    scaling_matrix::<false>();
}

#[test]
fn shared_sparse_tick_checks_scale_with_u_not_w_or_m() {
    scaling_matrix::<true>();
}

fn late_registration<const S: bool>() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let a = world.spawn(Value::<S>(0));
    let b = world.spawn(Value::<S>(0));
    for label in 0..2 {
        subscribe::<S>(&mut world, label);
    }
    warm::<S>(&mut world, a, 2);
    // Exercise the lazy window rebase as well as the registration boundary.
    assert_eq!(world.flush_changed_observers(), 0);
    assert_eq!(world.flush_changed_observers(), 0);
    world.get_mut::<Value<S>>(a).unwrap().0 = 10;
    subscribe::<S>(&mut world, 2);
    world.get_mut::<Value<S>>(b).unwrap().0 = 20;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_eq!(
        take_log(&mut world),
        [
            delivery(0, &[(a, 10), (b, 20)]),
            delivery(1, &[(a, 10), (b, 20)]),
            delivery(2, &[(b, 20)]),
        ]
    );
    let stats = world.change_observers.stats;
    assert_eq!(
        (
            stats.collections,
            stats.collection_cache_hits,
            stats.tick_checks
        ),
        (2, 1, 3)
    );
    world.get_mut::<Value<S>>(a).unwrap().0 = 11;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_shared(&world, 3, 1);
    assert_eq!(
        take_log(&mut world),
        (0..3)
            .map(|label| delivery(label, &[(a, 11)]))
            .collect::<Vec<_>>()
    );
}

#[test]
fn shared_collection_preserves_late_registration_and_idle_boundaries() {
    late_registration::<false>();
    late_registration::<true>();
}

#[test]
fn shared_collection_is_component_local() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let entity = world.spawn((Value::<false>(0), Value::<true>(0)));
    subscribe::<false>(&mut world, 0);
    subscribe::<true>(&mut world, 1);
    subscribe::<false>(&mut world, 2);
    subscribe::<true>(&mut world, 3);
    world.get_mut::<Value<false>>(entity).unwrap().0 = 1;
    world.get_mut::<Value<true>>(entity).unwrap().0 = 2;
    assert_eq!(world.flush_changed_observers(), 4);
    take_log(&mut world);
    world.get_mut::<Value<false>>(entity).unwrap().0 = 10;
    world.get_mut::<Value<true>>(entity).unwrap().0 = 20;
    assert_eq!(world.flush_changed_observers(), 4);
    let stats = world.change_observers.stats;
    assert_eq!(
        (
            stats.collections,
            stats.collection_cache_hits,
            stats.tick_checks
        ),
        (2, 2, 2)
    );
    assert_eq!(
        take_log(&mut world),
        [
            delivery(0, &[(entity, 10)]),
            delivery(1, &[(entity, 20)]),
            delivery(2, &[(entity, 10)]),
            delivery(3, &[(entity, 20)]),
        ]
    );
}

fn gap_and_registration<const S: bool>() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let entities: Vec<_> = (0..12).map(|_| world.spawn(Value::<S>(0))).collect();
    for label in 0..4 {
        subscribe::<S>(&mut world, label);
    }
    warm::<S>(&mut world, entities[0], 4);
    // Real overflow from authoritative writes, not synthetic gap diagnostics.
    for _ in 0..=crate::change_index::MAX_RETAINED_EVENTS {
        world.get_mut::<Value<S>>(entities[0]).unwrap().0 += 1;
    }
    let a_value = crate::change_index::MAX_RETAINED_EVENTS as u32 + 2;
    subscribe::<S>(&mut world, 4);
    world.get_mut::<Value<S>>(entities[1]).unwrap().0 = 71;
    assert_eq!(world.flush_changed_observers(), 5);
    let mut expected: Vec<_> = (0..4)
        .map(|label| delivery(label, &[(entities[0], a_value), (entities[1], 71)]))
        .collect();
    expected.push(delivery(4, &[(entities[1], 71)]));
    assert_eq!(take_log(&mut world), expected);
    let stats = world.change_observers.stats;
    assert_eq!((stats.collections, stats.collection_cache_hits), (2, 3));
    assert_eq!(stats.tick_checks, entities.len() + 1);
    assert_eq!(stats.gap_rows_checked, if S { 0 } else { entities.len() });
    assert_eq!(
        stats.gap_sparse_entities_checked,
        if S { entities.len() } else { 0 }
    );

    // The late subscription and gap-recovered old subscriptions now converge.
    world.get_mut::<Value<S>>(entities[2]).unwrap().0 = 72;
    assert_eq!(world.flush_changed_observers(), 5);
    assert_shared(&world, 5, 1);
    assert_eq!(
        take_log(&mut world),
        (0..5)
            .map(|label| delivery(label, &[(entities[2], 72)]))
            .collect::<Vec<_>>()
    );
}

#[test]
fn shared_table_gap_scans_once_without_crossing_late_registration() {
    gap_and_registration::<false>();
}

#[test]
fn shared_sparse_gap_scans_once_without_crossing_late_registration() {
    gap_and_registration::<true>();
}

#[derive(Default)]
struct Actions {
    write_and_register: bool,
    panic_once: bool,
    restore: bool,
}
impl Resource for Actions {}

#[cfg(feature = "std")]
#[test]
fn shared_collection_defers_callback_writes_and_registration_across_panic() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    world.insert_resource(Actions::default());
    let a = world.spawn(Value::<false>(0));
    let b = world.spawn(Value::<false>(0));
    let c = world.spawn(Value::<false>(0));
    world.observe_changed::<Value<false>, _>(move |ctx| {
        record::<false>(ctx, 0);
        if core::mem::take(&mut ctx.world.resource_mut::<Actions>().write_and_register) {
            ctx.world.get_mut::<Value<false>>(b).unwrap().0 = 20;
            subscribe::<false>(ctx.world, 3);
            ctx.world.get_mut::<Value<false>>(c).unwrap().0 = 30;
        }
    });
    world.observe_changed::<Value<false>, _>(|ctx| {
        record::<false>(ctx, 1);
        if core::mem::take(&mut ctx.world.resource_mut::<Actions>().panic_once) {
            panic!("retry this delivery");
        }
    });
    subscribe::<false>(&mut world, 2);
    warm::<false>(&mut world, a, 3);
    world.resource_mut::<Actions>().write_and_register = true;
    world.resource_mut::<Actions>().panic_once = true;
    world.get_mut::<Value<false>>(a).unwrap().0 = 10;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.flush_changed_observers();
    }))
    .is_err());
    assert_shared(&world, 3, 1);
    assert_eq!(
        take_log(&mut world),
        [delivery(0, &[(a, 10)]), delivery(1, &[(a, 10)])]
    );

    assert_eq!(world.flush_changed_observers(), 4);
    assert_eq!(
        take_log(&mut world),
        [
            delivery(0, &[(b, 20), (c, 30)]),
            delivery(1, &[(a, 10), (b, 20), (c, 30)]),
            delivery(2, &[(a, 10), (b, 20), (c, 30)]),
            delivery(3, &[(c, 30)]),
        ]
    );
    let stats = world.change_observers.stats;
    assert_eq!(
        (
            stats.collections,
            stats.collection_cache_hits,
            stats.tick_checks
        ),
        (2, 2, 3)
    );
    assert_eq!(world.flush_changed_observers(), 0);
    world.get_mut::<Value<false>>(a).unwrap().0 = 11;
    assert_eq!(world.flush_changed_observers(), 4);
    assert_shared(&world, 4, 1);
    assert_eq!(
        take_log(&mut world),
        (0..4)
            .map(|label| delivery(label, &[(a, 11)]))
            .collect::<Vec<_>>()
    );
}

#[cfg(feature = "std")]
fn retry_world() -> (World, Entity) {
    let mut world = World::new();
    world.register_snapshot_component::<Value<false>>();
    world.insert_resource(Log::default());
    world.insert_resource(Actions::default());
    let entity = world.spawn(Value::<false>(0));
    subscribe::<false>(&mut world, 0);
    world.observe_changed::<Value<false>, _>(|ctx| {
        record::<false>(ctx, 1);
        if core::mem::take(&mut ctx.world.resource_mut::<Actions>().panic_once) {
            panic!("retry this delivery");
        }
    });
    subscribe::<false>(&mut world, 2);
    warm::<false>(&mut world, entity, 3);
    (world, entity)
}

#[cfg(feature = "std")]
#[test]
fn shared_collection_retries_pending_without_new_writes_or_successful_replay() {
    let (mut world, entity) = retry_world();
    world.resource_mut::<Actions>().panic_once = true;
    world.get_mut::<Value<false>>(entity).unwrap().0 = 10;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.flush_changed_observers();
    }))
    .is_err());
    assert_shared(&world, 3, 1);
    assert_eq!(
        take_log(&mut world),
        [delivery(0, &[(entity, 10)]), delivery(1, &[(entity, 10)])]
    );
    assert_eq!(world.flush_changed_observers(), 2);
    assert_shared(&world, 2, 0);
    assert_eq!(
        take_log(&mut world),
        [delivery(1, &[(entity, 10)]), delivery(2, &[(entity, 10)])]
    );
    assert_eq!(world.flush_changed_observers(), 0);
    assert!(take_log(&mut world).is_empty());
}

#[cfg(feature = "std")]
#[test]
fn shared_collection_restore_discards_panicked_pending_and_accepts_same_tick_write() {
    let (mut world, entity) = retry_world();
    let snapshot = world.snapshot();
    world.resource_mut::<Actions>().panic_once = true;
    world.get_mut::<Value<false>>(entity).unwrap().0 = 99;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.flush_changed_observers();
    }))
    .is_err());
    assert_shared(&world, 3, 1);
    take_log(&mut world);
    world.restore(&snapshot);
    assert_eq!(world.get::<Value<false>>(entity).unwrap().0, 1);
    world.get_mut::<Value<false>>(entity).unwrap().0 = 7;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_eq!(world.change_observers.stats.collection_cache_hits, 0);
    assert_eq!(
        take_log(&mut world),
        (0..3)
            .map(|label| delivery(label, &[(entity, 7)]))
            .collect::<Vec<_>>()
    );
    world.get_mut::<Value<false>>(entity).unwrap().0 = 8;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_shared(&world, 3, 1);
    assert_eq!(
        take_log(&mut world),
        (0..3)
            .map(|label| delivery(label, &[(entity, 8)]))
            .collect::<Vec<_>>()
    );
    assert_eq!(world.flush_changed_observers(), 0);
}

#[cfg(feature = "std")]
#[test]
fn shared_collection_restore_in_callback_rebases_running_and_unrun_slots() {
    use alloc::sync::Arc;
    use std::sync::Mutex;

    let mut world = World::new();
    world.register_snapshot_component::<Value<false>>();
    world.insert_resource(Log::default());
    world.insert_resource(Actions::default());
    let a = world.spawn(Value::<false>(0));
    let b = world.spawn(Value::<false>(0));
    let snapshot = Arc::new(Mutex::new(None));
    world.observe_changed::<Value<false>, _>({
        let snapshot = snapshot.clone();
        move |ctx| {
            record::<false>(ctx, 0);
            if core::mem::take(&mut ctx.world.resource_mut::<Actions>().restore) {
                ctx.world
                    .restore(snapshot.lock().unwrap().as_ref().unwrap());
                ctx.world.get_mut::<Value<false>>(b).unwrap().0 = 7;
            }
        }
    });
    subscribe::<false>(&mut world, 1);
    subscribe::<false>(&mut world, 2);
    warm::<false>(&mut world, a, 3);
    *snapshot.lock().unwrap() = Some(world.snapshot());
    world.resource_mut::<Actions>().restore = true;
    world.get_mut::<Value<false>>(a).unwrap().0 = 99;
    assert_eq!(world.flush_changed_observers(), 1);
    assert_shared(&world, 3, 1);
    assert_eq!(take_log(&mut world), [delivery(0, &[(a, 99)])]);
    assert_eq!(world.get::<Value<false>>(a).unwrap().0, 1);
    for &index in &world.change_observers.active_indices {
        let slot = world.change_observers.slots[index].slot.as_ref().unwrap();
        assert!(slot.cursor_cohort.is_none());
        assert!(slot.pending.is_empty());
    }
    assert_eq!(world.flush_changed_observers(), 3);
    assert_eq!(world.change_observers.stats.collection_cache_hits, 0);
    assert_eq!(
        take_log(&mut world),
        (0..3)
            .map(|label| delivery(label, &[(b, 7)]))
            .collect::<Vec<_>>()
    );
    world.get_mut::<Value<false>>(a).unwrap().0 = 8;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_shared(&world, 3, 1);
    assert_eq!(
        take_log(&mut world),
        (0..3)
            .map(|label| delivery(label, &[(a, 8)]))
            .collect::<Vec<_>>()
    );
}

fn recycled_entities<const S: bool>() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let a = world.spawn(Value::<S>(0));
    let removed = world.spawn(Value::<S>(0));
    let dead = world.spawn(Value::<S>(0));
    for label in 0..4 {
        subscribe::<S>(&mut world, label);
    }
    warm::<S>(&mut world, a, 4);
    for _ in 0..8 {
        world.get_mut::<Value<S>>(a).unwrap().0 = 10;
        world.get_mut::<Value<S>>(removed).unwrap().0 = 20;
        world.get_mut::<Value<S>>(dead).unwrap().0 = 30;
    }
    world.remove::<Value<S>>(removed);
    assert!(world.despawn(dead));
    let replacement = world.spawn(Value::<S>(40));
    assert_eq!(replacement.index(), dead.index());
    assert_ne!(replacement.generation(), dead.generation());
    assert_eq!(world.flush_changed_observers(), 4);
    // Four unique candidate handles, two currently readable components. A
    // generation mismatch must not map the stale handle to the replacement.
    assert_shared(&world, 4, 4);
    assert_eq!(
        take_log(&mut world),
        (0..4)
            .map(|label| delivery(label, &[(a, 10), (replacement, 40)]))
            .collect::<Vec<_>>()
    );
}

#[test]
fn shared_collection_rechecks_removed_and_recycled_entities() {
    recycled_entities::<false>();
    recycled_entities::<true>();
}

#[test]
fn shared_collection_never_aliases_exhausted_cohort_ids() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let entity = world.spawn(Value::<false>(0));
    for label in 0..3 {
        subscribe::<false>(&mut world, label);
    }
    warm::<false>(&mut world, entity, 3);
    world.change_observers.collection_cohort = Some(u64::MAX - 1);
    world.get_mut::<Value<false>>(entity).unwrap().0 = 10;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_shared(&world, 3, 1);
    take_log(&mut world);
    assert_eq!(world.change_observers.collection_cohort, Some(u64::MAX));
    for value in [11, 12] {
        world.get_mut::<Value<false>>(entity).unwrap().0 = value;
        assert_eq!(world.flush_changed_observers(), 3);
        let stats = world.change_observers.stats;
        assert_eq!(world.change_observers.collection_cohort, None);
        assert_eq!(
            (
                stats.collections,
                stats.collection_cache_hits,
                stats.tick_checks
            ),
            (3, 0, 3)
        );
        assert_eq!(
            take_log(&mut world),
            (0..3)
                .map(|label| delivery(label, &[(entity, value)]))
                .collect::<Vec<_>>()
        );
    }
}

fn different_windows<const S: bool>(replayed_wrap: bool) {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let a = world.spawn(Value::<S>(0));
    let b = world.spawn(Value::<S>(0));
    subscribe::<S>(&mut world, 0);
    let later = subscribe::<S>(&mut world, 1);
    warm::<S>(&mut world, a, 2);
    let old_baseline = Tick::new(world.change_tick().get().wrapping_sub(1));
    let later_baseline;
    if replayed_wrap {
        // Explicit replay stamps exercise wrapping-window predicates, not a
        // claim that the public World clock actually rolled over in this test.
        let mut value = world.get_mut::<Value<S>>(a).unwrap();
        value.bypass_change_detection().0 = 10;
        value.set_last_changed(Tick::MAX);
        let mut value = world.get_mut::<Value<S>>(b).unwrap();
        value.bypass_change_detection().0 = 20;
        value.set_last_changed(Tick::ZERO);
        let first = world.change_observers.active_indices[0];
        world.change_observers.slots[first]
            .slot
            .as_mut()
            .unwrap()
            .last_run = Tick::new(u32::MAX - 1);
        later_baseline = Tick::MAX;
    } else {
        world.get_mut::<Value<S>>(a).unwrap().0 = 10;
        later_baseline = world.change_tick();
        world.increment_change_tick();
        world.get_mut::<Value<S>>(b).unwrap().0 = 20;
        assert_eq!(
            world.change_observers.slots[0]
                .slot
                .as_ref()
                .unwrap()
                .last_run,
            old_baseline
        );
    }
    // White-box setup isolates the effective-window part of the cache key:
    // cursor/provenance are identical, only this subscriber's window differs.
    let later_index = world.change_observers.slot_by_id[&later];
    world.change_observers.slots[later_index]
        .slot
        .as_mut()
        .unwrap()
        .last_run = later_baseline;
    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(
        take_log(&mut world),
        [delivery(0, &[(a, 10), (b, 20)]), delivery(1, &[(b, 20)])]
    );
    let stats = world.change_observers.stats;
    assert_eq!(
        (
            stats.collections,
            stats.collection_cache_hits,
            stats.tick_checks
        ),
        (2, 0, 3)
    );
    world.get_mut::<Value<S>>(a).unwrap().0 = 11;
    assert_eq!(world.flush_changed_observers(), 2);
    assert_shared(&world, 2, 1);
    assert_eq!(
        take_log(&mut world),
        [delivery(0, &[(a, 11)]), delivery(1, &[(a, 11)])]
    );
}

#[test]
fn shared_collection_keeps_different_effective_windows_separate() {
    different_windows::<false>(false);
    different_windows::<true>(false);
}

#[test]
fn shared_collection_keeps_replayed_wrap_windows_separate() {
    different_windows::<false>(true);
    different_windows::<true>(true);
}

#[test]
fn shared_collection_requires_cursor_provenance_not_just_equal_windows() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let a = world.spawn(Value::<false>(0));
    let b = world.spawn(Value::<false>(0));
    subscribe::<false>(&mut world, 0);
    subscribe::<false>(&mut world, 1);
    warm::<false>(&mut world, a, 2);
    // The old slots missed an overflow, but a newly activated cursor already
    // acknowledges that loss revision. The opaque cursor fields must not be
    // guessed equal merely because the delivery windows can be made equal.
    for _ in 0..=crate::change_index::MAX_RETAINED_EVENTS {
        world.get_mut::<Value<false>>(a).unwrap().0 = 10;
    }
    let late = subscribe::<false>(&mut world, 2);
    let first = world.change_observers.active_indices[0];
    let baseline = world.change_observers.slots[first]
        .slot
        .as_ref()
        .unwrap()
        .last_run;
    let late_index = world.change_observers.slot_by_id[&late];
    world.change_observers.slots[late_index]
        .slot
        .as_mut()
        .unwrap()
        .last_run = baseline;
    world.get_mut::<Value<false>>(b).unwrap().0 = 20;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_eq!(
        take_log(&mut world),
        [
            delivery(0, &[(a, 10), (b, 20)]),
            delivery(1, &[(a, 10), (b, 20)]),
            delivery(2, &[(b, 20)]),
        ]
    );
    let stats = world.change_observers.stats;
    assert_eq!(
        (
            stats.collections,
            stats.collection_cache_hits,
            stats.tick_checks
        ),
        (2, 1, 3)
    );
    world.get_mut::<Value<false>>(b).unwrap().0 = 21;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_shared(&world, 3, 1);
    assert_eq!(
        take_log(&mut world),
        (0..3)
            .map(|label| delivery(label, &[(b, 21)]))
            .collect::<Vec<_>>()
    );
}

#[test]
fn shared_collection_keeps_removal_and_callback_registration_order() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    world.insert_resource(Actions::default());
    let a = world.spawn(Value::<false>(0));
    let b = world.spawn(Value::<false>(0));
    // Register the remover first, but publish the target id only afterwards.
    struct Removal(Option<ObserverId>);
    impl Resource for Removal {}
    world.insert_resource(Removal(None));
    world.observe_changed::<Value<false>, _>(move |ctx| {
        record::<false>(ctx, 0);
        if core::mem::take(&mut ctx.world.resource_mut::<Actions>().write_and_register) {
            let removed = ctx.world.resource::<Removal>().0.unwrap();
            assert!(ctx.world.remove_change_observer(removed));
            subscribe::<false>(ctx.world, 3);
            ctx.world.get_mut::<Value<false>>(b).unwrap().0 = 20;
        }
    });
    world.resource_mut::<Removal>().0 = Some(subscribe::<false>(&mut world, 1));
    subscribe::<false>(&mut world, 2);
    warm::<false>(&mut world, a, 3);
    world.resource_mut::<Actions>().write_and_register = true;
    world.get_mut::<Value<false>>(a).unwrap().0 = 10;
    assert_eq!(world.flush_changed_observers(), 2);
    assert_shared(&world, 3, 1);
    assert_eq!(
        take_log(&mut world),
        [delivery(0, &[(a, 10)]), delivery(2, &[(a, 10)])]
    );
    assert_eq!(world.flush_changed_observers(), 3);
    assert_eq!(
        take_log(&mut world),
        [
            delivery(0, &[(b, 20)]),
            delivery(2, &[(b, 20)]),
            delivery(3, &[(b, 20)]),
        ]
    );
    let stats = world.change_observers.stats;
    assert_eq!(
        (
            stats.collections,
            stats.collection_cache_hits,
            stats.tick_checks
        ),
        (2, 1, 2)
    );
    world.get_mut::<Value<false>>(a).unwrap().0 = 11;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_shared(&world, 3, 1);
    assert_eq!(
        take_log(&mut world),
        [
            delivery(0, &[(a, 11)]),
            delivery(2, &[(a, 11)]),
            delivery(3, &[(a, 11)]),
        ]
    );
}
