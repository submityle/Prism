//! Observable field projections, rather than bridge-side counters, verify that
//! automatic synchronization really avoids reading clean ECS bindings.

use std::cell::{Cell, RefCell};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use prism_ecs::component::StorageType;
use prism_ecs::prelude::{Component, Schedule, World};
use prism_ui_ecs::{
    add_loom_sync_systems, cleanup_bridge_observers, insert_bridge, loom_pull_system,
    loom_push_system, remove_bridge, BridgeWorldMismatch, EcsBridge,
};
use prism_ui_reactive::Runtime;

#[derive(Component, Clone)]
struct Value(i32);

#[derive(Clone)]
struct Sparse(i32);
impl Component for Sparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[test]
fn quiet_frames_read_no_fields_and_repeated_writes_refresh_only_the_target() {
    let mut world = World::new();
    let runtime = Runtime::new();
    let mut bridge = EcsBridge::new();
    let reads = Rc::new(Cell::new(0));
    let mut entities = Vec::new();
    let mut signals = Vec::new();
    for index in 0..256 {
        let entity = world.spawn(Value(index));
        let signal = runtime.signal(-1);
        let reads = reads.clone();
        bridge.bind::<Value, _>(entity, signal.clone(), move |value| {
            reads.set(reads.get() + 1);
            value.0
        });
        entities.push(entity);
        signals.push(signal);
    }
    assert_eq!(bridge.pull_changed(&mut world), Ok(256));
    assert_eq!(reads.replace(0), 256);
    for _ in 0..5 {
        assert_eq!(bridge.pull_changed(&mut world), Ok(0));
    }
    assert_eq!(reads.get(), 0);
    for value in 300..310 {
        world.get_mut::<Value>(entities[42]).unwrap().0 = value;
        // Even multiple deliveries before a UI frame coalesce into one read.
        world.flush_changed_observers();
    }
    // An unbound entity of the observed type must not wake any binding.
    let unbound = world.spawn(Value(0));
    world.get_mut::<Value>(unbound).unwrap().0 = 10;
    assert_eq!(bridge.pull_changed(&mut world), Ok(1));
    assert_eq!(reads.get(), 1);
    assert_eq!(signals[42].get_untracked(), 309);
    assert_eq!(signals[43].get_untracked(), 43);
}

#[test]
fn scheduled_two_way_sync_skips_clean_fields_and_preserves_ui_edits() {
    let mut world = World::new();
    let entity = world.spawn(Value(1));
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let reads = Rc::new(Cell::new(0));
    let counter = reads.clone();
    let mut bridge = EcsBridge::new();
    bridge.bind_two_way::<Value, _>(
        entity,
        signal.clone(),
        move |value| {
            counter.set(counter.get() + 1);
            value.0
        },
        |value, signal| value.0 = *signal,
    );
    insert_bridge(&mut world, bridge);
    let mut schedule = Schedule::default();
    add_loom_sync_systems(&mut schedule);
    schedule.run(&mut world);
    assert_eq!(signal.get_untracked(), 1);
    reads.set(0);
    for _ in 0..3 {
        schedule.run(&mut world);
    }
    assert_eq!(reads.get(), 0, "neither pull nor push reads clean fields");
    signal.set(2);
    schedule.run(&mut world);
    assert_eq!(world.get::<Value>(entity).unwrap().0, 2);
    schedule.run(&mut world); // consume our own ECS write notification
    reads.set(0);
    schedule.run(&mut world);
    assert_eq!(reads.get(), 0);
    signal.set(3);
    world.get_mut::<Value>(entity).unwrap().0 = 4;
    schedule.run(&mut world);
    assert_eq!(signal.get_untracked(), 4, "ECS wins simultaneous edits");
    assert_eq!(world.get::<Value>(entity).unwrap().0, 4);
}

#[test]
fn table_and_sparse_same_tick_and_query_writes_are_delivered() {
    let mut world = World::new();
    let entity = world.spawn((Value(1), Sparse(2)));
    let runtime = Runtime::new();
    let table = runtime.signal(0);
    let sparse = runtime.signal(0);
    let mut bridge = EcsBridge::new();
    bridge.bind::<Value, _>(entity, table.clone(), |value| value.0);
    bridge.bind::<Sparse, _>(entity, sparse.clone(), |value| value.0);
    bridge.pull_changed(&mut world).unwrap();
    // No explicit tick advance between these direct writes.
    world.get_mut::<Value>(entity).unwrap().0 = 3;
    world.get_mut::<Value>(entity).unwrap().0 = 4;
    world.get_mut::<Sparse>(entity).unwrap().0 = 5;
    assert_eq!(bridge.pull_changed(&mut world), Ok(2));
    assert_eq!((table.get_untracked(), sparse.get_untracked()), (4, 5));
    let mut query = world.query::<(&mut Value, &mut Sparse)>();
    for (mut a, mut b) in query.iter_mut(&mut world) {
        a.0 = 6;
        b.0 = 7;
    }
    assert_eq!(bridge.pull_changed(&mut world), Ok(2));
    assert_eq!((table.get_untracked(), sparse.get_untracked()), (6, 7));
}

#[test]
fn missing_insert_remove_reinsert_and_recycled_entities() {
    let mut world = World::new();
    let entity = world.spawn(());
    let runtime = Runtime::new();
    let signal = runtime.signal(-1);
    let mut bridge = EcsBridge::new();
    bridge.bind::<Value, _>(entity, signal.clone(), |value| value.0);
    assert_eq!(bridge.pull_changed(&mut world), Ok(0));
    assert!(world.insert(entity, Value(1)));
    assert_eq!(bridge.pull_changed(&mut world), Ok(1));
    assert!(world.remove::<Value>(entity));
    assert_eq!(bridge.pull_changed(&mut world), Ok(0));
    assert_eq!(signal.get_untracked(), 1);
    // Several replacements in the same tick still require a forced refresh.
    assert!(world.insert(entity, Value(2)));
    assert!(world.remove::<Value>(entity));
    assert!(world.insert(entity, Value(3)));
    assert_eq!(bridge.pull_changed(&mut world), Ok(1));
    assert_eq!(signal.get_untracked(), 3);
    assert!(world.despawn(entity));
    let other = world.spawn(Value(99));
    assert_ne!(entity, other);
    assert_eq!(bridge.pull_changed(&mut world), Ok(0));
    assert_eq!(signal.get_untracked(), 3);
}

#[test]
fn sparse_lifecycle_replacements_refresh_even_without_a_newer_tick() {
    let mut world = World::new();
    let entity = world.spawn(Sparse(1));
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let mut bridge = EcsBridge::new();
    bridge.bind::<Sparse, _>(entity, signal.clone(), |value| value.0);
    bridge.pull_changed(&mut world).unwrap();
    assert!(world.remove::<Sparse>(entity));
    assert!(world.insert(entity, Sparse(2)));
    assert!(world.insert(entity, Sparse(3)));
    assert_eq!(bridge.pull_changed(&mut world), Ok(1));
    assert_eq!(signal.get_untracked(), 3);
    assert!(world.remove::<Sparse>(entity));
    assert_eq!(bridge.pull_changed(&mut world), Ok(0));
    assert_eq!(signal.get_untracked(), 3);
}

#[test]
fn restore_refreshes_older_ticks_and_revives_bindings_then_resumes_observation() {
    let mut world = World::new();
    world.register_snapshot_component::<Value>();
    world.register_snapshot_component::<Sparse>();
    let entity = world.spawn((Value(1), Sparse(2)));
    let runtime = Runtime::new();
    let a = runtime.signal(0);
    let b = runtime.signal(0);
    let mut bridge = EcsBridge::new();
    bridge.bind::<Value, _>(entity, a.clone(), |value| value.0);
    bridge.bind::<Sparse, _>(entity, b.clone(), |value| value.0);
    bridge.pull_changed(&mut world).unwrap();
    let snapshot = world.snapshot();
    world.get_mut::<Value>(entity).unwrap().0 = 10;
    world.get_mut::<Sparse>(entity).unwrap().0 = 20;
    bridge.pull_changed(&mut world).unwrap();
    assert!(world.despawn(entity));
    bridge.pull_changed(&mut world).unwrap();
    world.restore(&snapshot);
    assert_eq!(bridge.pull_changed(&mut world), Ok(2));
    assert_eq!((a.get_untracked(), b.get_untracked()), (1, 2));
    world.get_mut::<Value>(entity).unwrap().0 = 30;
    world.get_mut::<Sparse>(entity).unwrap().0 = 40;
    assert_eq!(bridge.pull_changed(&mut world), Ok(2));
    assert_eq!((a.get_untracked(), b.get_untracked()), (30, 40));
    assert_eq!(bridge.pull_changed(&mut world), Ok(0));
}

#[test]
fn late_binding_and_unbind_are_independent_of_other_bridges() {
    let mut world = World::new();
    let entity = world.spawn(Value(1));
    let runtime = Runtime::new();
    let mut first = EcsBridge::new();
    let a = runtime.signal(0);
    let b = runtime.signal(0);
    let c = runtime.signal(0);
    let id = first.bind::<Value, _>(entity, a.clone(), |value| value.0);
    first.pull_changed(&mut world).unwrap();
    let second_id = first.bind::<Value, _>(entity, b.clone(), |value| value.0 + 1);
    assert_eq!(first.pull_changed(&mut world), Ok(1));
    let mut second = EcsBridge::new();
    let foreign = second.bind::<Value, _>(entity, c.clone(), |value| value.0);
    second.pull_changed(&mut world).unwrap();
    assert!(!first.unbind(&mut world, foreign));
    assert!(first.unbind(&mut world, id));
    world.get_mut::<Value>(entity).unwrap().0 = 10;
    assert_eq!(first.pull_changed(&mut world), Ok(1));
    assert_eq!(second.pull_changed(&mut world), Ok(1));
    assert_eq!(
        (a.get_untracked(), b.get_untracked(), c.get_untracked()),
        (1, 11, 10)
    );
    assert!(first.unbind(&mut world, second_id));
    assert!(first.is_empty());
    assert!(!first.unbind(&mut world, id));
    world.get_mut::<Value>(entity).unwrap().0 = 20;
    assert_eq!(second.pull_changed(&mut world), Ok(1));
    assert_eq!(c.get_untracked(), 20);
}

#[test]
fn world_identity_requires_explicit_detach() {
    let mut original = World::new();
    let entity = original.spawn(Value(1));
    let mut replacement = World::new();
    let aliased = replacement.spawn(Value(99));
    assert_eq!(entity, aliased);
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let mut bridge = EcsBridge::new();
    bridge.bind::<Value, _>(entity, signal.clone(), |value| value.0);
    bridge.pull_changed(&mut original).unwrap();
    assert_eq!(
        bridge.pull_changed(&mut replacement),
        Err(BridgeWorldMismatch)
    );
    assert_eq!(
        bridge.push_changed(&mut replacement),
        Err(BridgeWorldMismatch)
    );
    assert_eq!(signal.get_untracked(), 1);
    bridge.detach(&mut original);
    assert_eq!(bridge.pull_changed(&mut replacement), Ok(1));
    assert_eq!(signal.get_untracked(), 99);
}

#[test]
fn reader_panic_restores_installed_bridge_and_retries_pending_binding() {
    let mut world = World::new();
    let entity = world.spawn(Value(1));
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let panic_once = Rc::new(Cell::new(true));
    let flag = panic_once.clone();
    let mut bridge = EcsBridge::new();
    bridge.bind::<Value, _>(entity, signal.clone(), move |value| {
        assert!(!flag.replace(false), "reader fails once");
        value.0
    });
    insert_bridge(&mut world, bridge);
    assert!(catch_unwind(AssertUnwindSafe(|| loom_pull_system(&mut world))).is_err());
    assert!(world.contains_non_send::<EcsBridge>());
    loom_pull_system(&mut world);
    assert_eq!(signal.get_untracked(), 1);
}

#[test]
fn refreshes_keep_registration_order_and_retry_only_unfinished_bindings() {
    let mut world = World::new();
    let entity = world.spawn(Value(1));
    let runtime = Runtime::new();
    let log = Rc::new(RefCell::new(Vec::new()));
    let fail = Rc::new(Cell::new(false));
    let mut bridge = EcsBridge::new();
    for index in 0..3 {
        let log = log.clone();
        let fail = fail.clone();
        bridge.bind::<Value, _>(entity, runtime.signal(0), move |value| {
            log.borrow_mut().push(index);
            if index == 1 {
                assert!(!fail.replace(false), "middle projection fails once");
            }
            value.0
        });
    }
    bridge.pull_changed(&mut world).unwrap();
    log.borrow_mut().clear();
    fail.set(true);
    world.get_mut::<Value>(entity).unwrap().0 = 2;
    assert!(catch_unwind(AssertUnwindSafe(|| bridge.pull_changed(&mut world))).is_err());
    assert_eq!(*log.borrow(), [0, 1]);
    log.borrow_mut().clear();
    bridge.pull_changed(&mut world).unwrap();
    assert_eq!(*log.borrow(), [1, 2]);
    log.borrow_mut().clear();
    bridge.pull_changed(&mut world).unwrap();
    assert!(log.borrow().is_empty());
}

#[test]
fn panicking_external_observer_preserves_the_ui_delivery_for_retry() {
    let mut world = World::new();
    let entity = world.spawn(Value(1));
    let mut fail = true;
    world.observe_changed::<Value, _>(move |_| {
        assert!(
            !std::mem::replace(&mut fail, false),
            "source observer fails once"
        );
    });
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let mut bridge = EcsBridge::new();
    bridge.bind::<Value, _>(entity, signal.clone(), |value| value.0);
    insert_bridge(&mut world, bridge);
    loom_pull_system(&mut world);
    world.get_mut::<Value>(entity).unwrap().0 = 2;
    assert!(catch_unwind(AssertUnwindSafe(|| loom_pull_system(&mut world))).is_err());
    assert_eq!(signal.get_untracked(), 1);
    loom_pull_system(&mut world);
    assert_eq!(signal.get_untracked(), 2);
}

#[test]
fn callback_installing_a_bridge_in_the_same_world_keeps_the_new_bridge() {
    let mut world = World::new();
    let entity = world.spawn(Value(1));
    let runtime = Runtime::new();
    let mut bridge = EcsBridge::new();
    bridge.bind::<Value, _>(entity, runtime.signal(0), |value| value.0);
    insert_bridge(&mut world, bridge);
    loom_pull_system(&mut world);
    world.observe_changed::<Value, _>(|ctx| {
        insert_bridge(ctx.world, EcsBridge::new());
    });
    world.get_mut::<Value>(entity).unwrap().0 = 2;
    loom_pull_system(&mut world);
    assert!(remove_bridge(&mut world).unwrap().is_empty());
}

#[test]
fn callback_world_replacement_never_restores_old_bridge_even_on_panic() {
    for panic in [false, true] {
        for install_new_bridge in [false, true] {
            let mut world = World::new();
            let entity = world.spawn(Value(1));
            let runtime = Runtime::new();
            let signal = runtime.signal(0);
            let mut bridge = EcsBridge::new();
            bridge.bind::<Value, _>(entity, signal.clone(), |value| value.0);
            insert_bridge(&mut world, bridge);
            loom_pull_system(&mut world);
            world.observe_changed::<Value, _>(move |ctx| {
                *ctx.world = World::new();
                ctx.world.spawn(Value(99));
                if install_new_bridge {
                    insert_bridge(ctx.world, EcsBridge::new());
                }
                assert!(!panic, "replacement then panic");
            });
            world.get_mut::<Value>(entity).unwrap().0 = 2;
            let result = catch_unwind(AssertUnwindSafe(|| loom_pull_system(&mut world)));
            assert_eq!(result.is_err(), panic);
            assert_eq!(
                signal.get_untracked(),
                1,
                "never read aliased replacement entity"
            );
            assert_eq!(world.contains_non_send::<EcsBridge>(), install_new_bridge);
            if let Some(bridge) = remove_bridge(&mut world) {
                assert!(bridge.is_empty());
            }
        }
    }
}

#[test]
fn dropping_or_replacing_bridge_releases_projection_captures() {
    let mut world = World::new();
    let entity = world.spawn(Value(1));
    let runtime = Runtime::new();
    let capture = Rc::new(());
    let weak = Rc::downgrade(&capture);
    let mut bridge = EcsBridge::new();
    bridge.bind::<Value, _>(entity, runtime.signal(0), move |value| {
        let _keep = &capture;
        value.0
    });
    bridge.pull_changed(&mut world).unwrap();
    drop(bridge);
    assert!(
        weak.upgrade().is_none(),
        "observers must not own projections or signals"
    );
    cleanup_bridge_observers(&mut world);
    let mut bridge = EcsBridge::new();
    bridge.bind::<Value, _>(entity, runtime.signal(0), |value| value.0);
    insert_bridge(&mut world, bridge);
    loom_pull_system(&mut world);
    insert_bridge(&mut world, EcsBridge::new());
    loom_push_system(&mut world);
    assert!(remove_bridge(&mut world).unwrap().is_empty());
}
