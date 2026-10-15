//! Observe value operations, not bridge implementation counters, to enforce
//! notification-driven push and the pull-before-push reconciliation contract.

use std::cell::{Cell, RefCell};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use prism_ecs::prelude::{Component, Schedule, World};
use prism_ui_ecs::{
    add_loom_sync_systems, insert_bridge, loom_pull_system, loom_push_system, BridgeWorldMismatch,
    EcsBridge,
};
use prism_ui_reactive::Runtime;

#[derive(Component, Clone)]
struct Value(i32);

#[derive(Default)]
struct Operations {
    clones: Cell<usize>,
    comparisons: Cell<usize>,
    projections: Cell<usize>,
    writes: Cell<usize>,
}

impl Operations {
    fn take(&self) -> (usize, usize, usize, usize) {
        (
            self.clones.replace(0),
            self.comparisons.replace(0),
            self.projections.replace(0),
            self.writes.replace(0),
        )
    }
}

struct Counted {
    value: i32,
    operations: Rc<Operations>,
}

impl Clone for Counted {
    fn clone(&self) -> Self {
        self.operations.clones.set(self.operations.clones.get() + 1);
        Self {
            value: self.value,
            operations: self.operations.clone(),
        }
    }
}

impl PartialEq for Counted {
    fn eq(&self, other: &Self) -> bool {
        self.operations
            .comparisons
            .set(self.operations.comparisons.get() + 1);
        self.value == other.value
    }
}

#[test]
fn static_scheduled_frames_do_not_read_clone_or_compare_binding_values() {
    let mut world = World::new();
    let runtime = Runtime::new();
    let operations = Rc::new(Operations::default());
    let mut bridge = EcsBridge::new();
    let mut signals = Vec::new();
    let mut entities = Vec::new();
    for index in 0..192 {
        let entity = world.spawn(Value(index));
        let signal = runtime.signal(Counted {
            value: -1,
            operations: operations.clone(),
        });
        bridge.bind_two_way::<Value, _>(
            entity,
            signal.clone(),
            {
                let operations = operations.clone();
                move |value| {
                    operations.projections.set(operations.projections.get() + 1);
                    Counted {
                        value: value.0,
                        operations: operations.clone(),
                    }
                }
            },
            {
                let operations = operations.clone();
                move |value, signal| {
                    operations.writes.set(operations.writes.get() + 1);
                    value.0 = signal.value;
                }
            },
        );
        signals.push(signal);
        entities.push(entity);
    }
    let readonly = runtime.signal(Counted {
        value: 0,
        operations: operations.clone(),
    });
    bridge.bind::<Value, _>(entities[0], readonly.clone(), {
        let operations = operations.clone();
        move |value| {
            operations.projections.set(operations.projections.get() + 1);
            Counted {
                value: value.0,
                operations: operations.clone(),
            }
        }
    });
    insert_bridge(&mut world, bridge);
    let mut schedule = Schedule::default();
    add_loom_sync_systems(&mut schedule);
    schedule.run(&mut world);
    operations.take();
    for _ in 0..5 {
        schedule.run(&mut world);
    }
    assert_eq!(operations.take(), (0, 0, 0, 0));

    for value in 200..210 {
        signals[42].set(Counted {
            value,
            operations: operations.clone(),
        });
    }
    readonly.set(Counted {
        value: 999,
        operations: operations.clone(),
    });
    assert_eq!(operations.take(), (0, 0, 0, 0), "enqueue is value-free");
    schedule.run(&mut world);
    assert_eq!(
        operations.take(),
        (1, 2, 1, 1),
        "one coalesced two-way binding"
    );
    assert_eq!(world.get::<Value>(entities[42]).unwrap().0, 209);
    schedule.run(&mut world); // consume the ECS echo once
    operations.take();
    for _ in 0..5 {
        schedule.run(&mut world);
    }
    assert_eq!(operations.take(), (0, 0, 0, 0));
}

#[test]
fn unbind_cancels_pending_work_and_rebind_never_reuses_the_old_route() {
    let mut world = World::new();
    let first = world.spawn(Value(0));
    let second = world.spawn(Value(0));
    let runtime = Runtime::new();
    let signal = runtime.signal(1);
    let mut bridge = EcsBridge::new();
    let old = bridge.bind_two_way::<Value, _>(first, signal.clone(), |v| v.0, |v, s| v.0 = *s);
    signal.set(2);
    assert!(bridge.unbind(&mut world, old));
    let new = bridge.bind_two_way::<Value, _>(second, signal.clone(), |v| v.0, |v, s| v.0 = *s);
    assert_ne!(old, new);
    assert_eq!(bridge.push_changed(&mut world), Ok(1));
    assert_eq!(world.get::<Value>(first).unwrap().0, 0);
    assert_eq!(world.get::<Value>(second).unwrap().0, 2);
    signal.set(3);
    assert!(bridge.unbind(&mut world, new));
    assert_eq!(bridge.push_changed(&mut world), Ok(0));
    assert_eq!(world.get::<Value>(second).unwrap().0, 2);
}

#[test]
fn independent_bridges_keep_independent_leases_and_reject_foreign_worlds() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let mut foreign = World::new();
    let alias = foreign.spawn(Value(100));
    assert_eq!(entity, alias);
    let runtime = Runtime::new();
    let signal = runtime.signal(1);
    let mut a = EcsBridge::new();
    let mut b = EcsBridge::new();
    let id = a.bind_two_way::<Value, _>(entity, signal.clone(), |v| v.0, |v, s| v.0 = *s);
    b.bind_two_way::<Value, _>(entity, signal.clone(), |v| v.0, |v, s| v.0 = *s);
    assert_eq!(a.push_changed(&mut world), Ok(1));
    signal.set(2);
    assert_eq!(a.push_changed(&mut foreign), Err(BridgeWorldMismatch));
    assert_eq!(foreign.get::<Value>(alias).unwrap().0, 100);
    assert_eq!(
        a.push_changed(&mut world),
        Ok(1),
        "wrong World did not consume work"
    );
    assert!(a.unbind(&mut world, id));
    signal.set(3);
    assert_eq!(a.push_changed(&mut world), Ok(0));
    assert_eq!(b.push_changed(&mut world), Ok(1));
    b.detach(&mut world);
    assert_eq!(b.pull_changed(&mut foreign), Ok(1));
    assert_eq!(signal.get_untracked(), 100);
    assert_eq!(b.push_changed(&mut foreign), Ok(0));
}

#[test]
fn scheduled_push_panic_restores_bridge_and_retries_only_unfinished_entries() {
    let mut world = World::new();
    let runtime = Runtime::new();
    let fail = Rc::new(Cell::new(false));
    let log = Rc::new(RefCell::new(Vec::new()));
    let mut bridge = EcsBridge::new();
    let mut signals = Vec::new();
    let mut entities = Vec::new();
    for index in 0..3 {
        let entity = world.spawn(Value(0));
        let signal = runtime.signal(0);
        bridge.bind_two_way::<Value, _>(entity, signal.clone(), |v| v.0, {
            let fail = fail.clone();
            let log = log.clone();
            move |v, s| {
                log.borrow_mut().push(index);
                if index == 1 {
                    assert!(!fail.replace(false), "middle writer fails once");
                }
                v.0 = *s;
            }
        });
        signals.push(signal);
        entities.push(entity);
    }
    insert_bridge(&mut world, bridge);
    loom_pull_system(&mut world);
    loom_push_system(&mut world);
    for signal in &signals {
        signal.set(10);
    }
    fail.set(true);
    assert!(catch_unwind(AssertUnwindSafe(|| loom_push_system(&mut world))).is_err());
    assert!(world.contains_non_send::<EcsBridge>());
    assert_eq!(*log.borrow(), [0, 1]);
    assert_eq!(world.get::<Value>(entities[0]).unwrap().0, 10);
    assert_eq!(world.get::<Value>(entities[1]).unwrap().0, 0);
    assert_eq!(world.get::<Value>(entities[2]).unwrap().0, 0);
    log.borrow_mut().clear();
    loom_push_system(&mut world);
    assert_eq!(*log.borrow(), [1, 2]);
    log.borrow_mut().clear();
    loom_push_system(&mut world);
    assert!(log.borrow().is_empty());
}

#[test]
fn partial_writer_panic_reconciles_without_rolling_back_or_repeating_side_effects() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let calls = Rc::new(Cell::new(0));
    let mut bridge = EcsBridge::new();
    bridge.bind_two_way::<Value, _>(entity, signal.clone(), |v| v.0, {
        let calls = calls.clone();
        move |v, s| {
            calls.set(calls.get() + 1);
            v.0 = *s;
            panic!("writer failed after mutation");
        }
    });
    bridge.pull_changed(&mut world).unwrap();
    bridge.push_changed(&mut world).unwrap();
    signal.set(7);
    assert!(catch_unwind(AssertUnwindSafe(|| bridge.push_changed(&mut world))).is_err());
    assert_eq!(world.get::<Value>(entity).unwrap().0, 7);
    assert_eq!(bridge.push_changed(&mut world), Ok(0));
    assert_eq!(
        calls.get(),
        1,
        "matching partial write is reconciled, not replayed"
    );
}

#[test]
fn notification_created_by_writer_survives_acknowledging_its_current_batch() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let mut bridge = EcsBridge::new();
    bridge.bind_two_way::<Value, _>(entity, signal.clone(), |v| v.0, {
        let signal = signal.clone();
        move |v, s| {
            v.0 = *s;
            if *s == 1 {
                signal.set(2);
            }
        }
    });
    bridge.pull_changed(&mut world).unwrap();
    bridge.push_changed(&mut world).unwrap();
    signal.set(1);
    assert_eq!(bridge.push_changed(&mut world), Ok(1));
    assert_eq!(world.get::<Value>(entity).unwrap().0, 1);
    assert_eq!(bridge.push_changed(&mut world), Ok(1));
    assert_eq!(world.get::<Value>(entity).unwrap().0, 2);
    assert_eq!(bridge.push_changed(&mut world), Ok(0));
}

#[test]
fn restore_discards_queued_ui_edits_and_echoes_without_advancing_component_tick() {
    let mut world = World::new();
    world.register_snapshot_component::<Value>();
    let entity = world.spawn(Value(1));
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let writes = Rc::new(Cell::new(0));
    let mut bridge = EcsBridge::new();
    bridge.bind_two_way::<Value, _>(entity, signal.clone(), |v| v.0, {
        let writes = writes.clone();
        move |v, s| {
            writes.set(writes.get() + 1);
            v.0 = *s;
        }
    });
    insert_bridge(&mut world, bridge);
    loom_pull_system(&mut world);
    loom_push_system(&mut world);
    let snapshot = world.snapshot();
    world.increment_change_tick();
    world.get_mut::<Value>(entity).unwrap().0 = 10;
    loom_pull_system(&mut world);
    loom_push_system(&mut world);
    signal.set(99); // pending user edit from the abandoned timeline
    world.restore(&snapshot);
    loom_pull_system(&mut world);
    let tick = world
        .get_entity(entity)
        .unwrap()
        .get_ref::<Value>()
        .unwrap()
        .last_changed();
    loom_push_system(&mut world);
    assert_eq!(signal.get_untracked(), 1);
    assert_eq!(world.get::<Value>(entity).unwrap().0, 1);
    assert_eq!(writes.get(), 0);
    assert_eq!(
        world
            .get_entity(entity)
            .unwrap()
            .get_ref::<Value>()
            .unwrap()
            .last_changed(),
        tick
    );
    signal.set(2);
    loom_push_system(&mut world);
    assert_eq!(world.get::<Value>(entity).unwrap().0, 2);
    assert_eq!(writes.get(), 1);
}

#[test]
fn effect_panic_during_pull_keeps_bridge_and_notification_work_recoverable() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let runtime = Runtime::new();
    let signal = runtime.signal(0);
    let fail = Rc::new(Cell::new(false));
    let seen = Rc::new(RefCell::new(Vec::new()));
    let effect = runtime.effect({
        let signal = signal.clone();
        let fail = fail.clone();
        let seen = seen.clone();
        move || {
            assert!(!fail.replace(false), "signal effect fails before reading");
            seen.borrow_mut().push(signal.get());
        }
    });
    let mut bridge = EcsBridge::new();
    bridge.bind_two_way::<Value, _>(entity, signal.clone(), |v| v.0, |v, s| v.0 = *s);
    insert_bridge(&mut world, bridge);
    loom_pull_system(&mut world);
    loom_push_system(&mut world);
    fail.set(true);
    world.get_mut::<Value>(entity).unwrap().0 = 1;
    assert!(catch_unwind(AssertUnwindSafe(|| loom_pull_system(&mut world))).is_err());
    assert!(world.contains_non_send::<EcsBridge>());
    loom_pull_system(&mut world);
    loom_push_system(&mut world);
    runtime.batch(|| ());
    assert_eq!(*seen.borrow(), [0, 1]);
    signal.set(2);
    loom_push_system(&mut world);
    assert_eq!(world.get::<Value>(entity).unwrap().0, 2);
    assert_eq!(*seen.borrow(), [0, 1, 2]);
    effect.dispose();
}
