//! Lifecycle/event callbacks must not resume an abandoned world's operation.

use prism_ecs::component_hooks::{ComponentHooks, HookContext};
use prism_ecs::observer::{LifecycleEvent, ObserverId};
use prism_ecs::prelude::*;
use prism_ecs::relation::RelationKind;
use prism_ecs::world::snapshot::WorldSnapshot;
use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Component, Clone, Debug, PartialEq)]
struct Value(i32);
#[derive(Component)]
struct Ping;
#[derive(Component)]
struct Parent;

thread_local! {
    static SNAPSHOT: RefCell<Option<WorldSnapshot>> = const { RefCell::new(None) };
}

fn restore(world: &mut World) {
    SNAPSHOT.with(|snapshot| world.restore(snapshot.borrow().as_ref().unwrap()));
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Insert,
    Remove,
    Despawn,
}

fn apply(world: &mut World, entity: Entity, operation: Operation) -> bool {
    match operation {
        Operation::Insert => world.insert(entity, Value(20)),
        Operation::Remove => world.remove::<Value>(entity),
        Operation::Despawn => world.despawn(entity),
    }
}

#[test]
fn lifecycle_replacement_keeps_empty_and_colliding_populated_registries_even_on_panic() {
    for populated in [false, true] {
        for panic_after in [false, true] {
            let mut world = World::new();
            let old_calls = Arc::new(AtomicUsize::new(0));
            let new_calls = Arc::new(AtomicUsize::new(0));
            world.observe::<Value, _>(LifecycleEvent::Add, {
                let old_calls = old_calls.clone();
                let new_calls = new_calls.clone();
                move |ctx| {
                    old_calls.fetch_add(1, Ordering::Relaxed);
                    *ctx.world = World::new();
                    if populated {
                        // The new world's first registration has the same id
                        // and slot number as this detached old callback.
                        let new_calls = new_calls.clone();
                        ctx.world
                            .observe::<Value, _>(LifecycleEvent::Add, move |_| {
                                new_calls.fetch_add(1, Ordering::Relaxed);
                            });
                    }
                    assert!(!panic_after, "intentional callback panic");
                }
            });
            world.observe::<Value, _>(LifecycleEvent::Add, |_| panic!("old later callback"));
            world.observe::<Value, _>(LifecycleEvent::Insert, |_| panic!("old next phase"));
            let original = world.id();
            let result = catch_unwind(AssertUnwindSafe(|| world.spawn(Value(1))));
            assert_eq!(result.is_err(), panic_after);
            assert_ne!(world.id(), original);
            assert_eq!(new_calls.load(Ordering::Relaxed), 0);
            let entity = world.spawn(Value(2));
            assert_eq!(world.get::<Value>(entity), Some(&Value(2)));
            assert_eq!(old_calls.load(Ordering::Relaxed), 1);
            assert_eq!(new_calls.load(Ordering::Relaxed), usize::from(populated));
        }
    }
}

#[test]
fn pre_transition_replacement_never_mutates_same_numbered_entity_in_new_world() {
    for operation in [Operation::Insert, Operation::Remove, Operation::Despawn] {
        for event in [LifecycleEvent::Replace, LifecycleEvent::Remove] {
            if matches!(operation, Operation::Insert) && event == LifecycleEvent::Remove {
                continue;
            }
            for panic_after in [false, true] {
                let mut world = World::new();
                let entity = world.spawn(Value(1));
                world.observe::<Value, _>(event, move |ctx| {
                    *ctx.world = World::new();
                    let replacement = ctx.world.spawn(Value(99));
                    assert_eq!(replacement, ctx.entity);
                    assert!(!panic_after, "intentional callback panic");
                });
                world.observe::<Value, _>(event, |_| panic!("old later callback"));
                let result =
                    catch_unwind(AssertUnwindSafe(|| apply(&mut world, entity, operation)));
                assert_eq!(result.is_err(), panic_after, "{operation:?} {event:?}");
                assert_eq!(world.get::<Value>(entity), Some(&Value(99)));
                assert!(world.insert(entity, Value(100)));
                assert_eq!(world.get::<Value>(entity), Some(&Value(100)));
            }
        }
    }
}

#[test]
fn restore_aborts_pre_transition_and_preserves_callback_on_success_and_unwind() {
    for operation in [Operation::Insert, Operation::Remove, Operation::Despawn] {
        for event in [LifecycleEvent::Replace, LifecycleEvent::Remove] {
            if matches!(operation, Operation::Insert) && event == LifecycleEvent::Remove {
                continue;
            }
            for panic_after in [false, true] {
                let mut world = World::new();
                world.register_snapshot_component::<Value>();
                let entity = world.spawn(Value(1));
                SNAPSHOT.with(|slot| *slot.borrow_mut() = Some(world.snapshot()));
                world.get_mut::<Value>(entity).unwrap().0 = 9;
                let calls = Arc::new(AtomicUsize::new(0));
                let id = world.observe::<Value, _>(event, {
                    let calls = calls.clone();
                    move |ctx| {
                        if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                            restore(ctx.world);
                            assert!(!panic_after, "intentional callback panic");
                        }
                    }
                });
                let generation = world.archetype_generation();
                let result =
                    catch_unwind(AssertUnwindSafe(|| apply(&mut world, entity, operation)));
                assert_eq!(result.is_err(), panic_after);
                assert_ne!(generation, world.archetype_generation());
                assert_eq!(world.get::<Value>(entity), Some(&Value(1)));
                assert!(apply(&mut world, entity, operation));
                assert_eq!(calls.load(Ordering::Relaxed), 2);
                assert!(world.remove_observer(id));
                SNAPSHOT.with(|slot| *slot.borrow_mut() = None);
            }
        }
    }
}

#[test]
fn restore_during_add_stops_old_insert_phase_but_observer_survives() {
    for panic_after in [false, true] {
        let mut world = World::new();
        world.register_snapshot_component::<Value>();
        SNAPSHOT.with(|slot| *slot.borrow_mut() = Some(world.snapshot()));
        let mut first = true;
        world.observe::<Value, _>(LifecycleEvent::Add, move |ctx| {
            if first {
                first = false;
                restore(ctx.world);
                assert!(!panic_after, "intentional callback panic");
            }
        });
        let inserts = Arc::new(AtomicUsize::new(0));
        world.observe::<Value, _>(LifecycleEvent::Insert, {
            let inserts = inserts.clone();
            move |_| {
                inserts.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(
            catch_unwind(AssertUnwindSafe(|| world.spawn(Value(1)))).is_err(),
            panic_after
        );
        assert_eq!(inserts.load(Ordering::Relaxed), 0);
        world.spawn(Value(2));
        assert_eq!(inserts.load(Ordering::Relaxed), 1);
        SNAPSHOT.with(|slot| *slot.borrow_mut() = None);
    }
}

#[test]
fn spawn_batch_replans_remaining_bundles_after_world_replacement() {
    let mut world = World::new();
    world.observe::<Value, _>(LifecycleEvent::Add, |ctx| {
        *ctx.world = World::new();
        // Force a different component-id/layout plan for remaining bundles.
        ctx.world.spawn(Ping);
    });
    let entities = world.spawn_batch([Value(1), Value(2), Value(3)]);
    assert_eq!(entities.len(), 3);
    assert_eq!(world.get::<Value>(entities[1]), Some(&Value(2)));
    assert_eq!(world.get::<Value>(entities[2]), Some(&Value(3)));
}

#[test]
fn running_lifecycle_observer_can_remove_itself_even_when_panicking() {
    for panic_after in [false, true] {
        let mut world = World::new();
        let id = Arc::new(Mutex::new(None::<ObserverId>));
        let calls = Arc::new(AtomicUsize::new(0));
        let registration = world.observe::<Value, _>(LifecycleEvent::Add, {
            let id = id.clone();
            let calls = calls.clone();
            move |ctx| {
                calls.fetch_add(1, Ordering::Relaxed);
                assert!(ctx.world.remove_observer(id.lock().unwrap().unwrap()));
                assert!(!panic_after, "intentional callback panic");
            }
        });
        *id.lock().unwrap() = Some(registration);
        assert_eq!(
            catch_unwind(AssertUnwindSafe(|| world.spawn(Value(1)))).is_err(),
            panic_after
        );
        assert!(!world.remove_observer(registration));
        world.spawn(Value(2));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn nested_lifecycle_dispatch_skips_running_slot_and_defers_new_registration() {
    let mut world = World::new();
    let log = Arc::new(Mutex::new(Vec::new()));
    world.observe::<Value, _>(LifecycleEvent::Add, {
        let log = log.clone();
        move |ctx| {
            log.lock().unwrap().push("outer");
            ctx.world.spawn(Value(2));
            let log = log.clone();
            ctx.world
                .observe::<Value, _>(LifecycleEvent::Add, move |_| {
                    log.lock().unwrap().push("new");
                });
        }
    });
    world.observe::<Value, _>(LifecycleEvent::Add, {
        let log = log.clone();
        move |_| {
            log.lock().unwrap().push("other");
        }
    });
    world.spawn(Value(1));
    assert_eq!(*log.lock().unwrap(), ["outer", "other", "other"]);
    log.lock().unwrap().clear();
    world.spawn(Value(3));
    assert_eq!(
        *log.lock().unwrap(),
        ["outer", "other", "new", "other", "new"]
    );
}

#[test]
fn event_replacement_does_not_restore_old_slot_or_continue_bubbling_even_on_panic() {
    for panic_after in [false, true] {
        for populated in [false, true] {
            let mut world = World::new();
            let target = world.spawn(());
            let calls = Arc::new(AtomicUsize::new(0));
            world.observe_event::<Ping, _>({
                let calls = calls.clone();
                move |ctx| {
                    *ctx.world = World::new();
                    assert_eq!(ctx.world.spawn(()), ctx.target);
                    if populated {
                        let calls = calls.clone();
                        ctx.world.observe_event::<Ping, _>(move |_| {
                            calls.fetch_add(1, Ordering::Relaxed);
                        });
                    }
                    assert!(!panic_after, "intentional callback panic");
                }
            });
            world.observe_event::<Ping, _>(|_| panic!("old event callback"));
            assert_eq!(
                catch_unwind(AssertUnwindSafe(
                    || world.trigger::<Ping, Parent>(target, true)
                ))
                .is_err(),
                panic_after
            );
            assert_eq!(calls.load(Ordering::Relaxed), 0);
            world.trigger::<Ping, Parent>(target, true);
            assert_eq!(calls.load(Ordering::Relaxed), usize::from(populated));
        }
    }
}

#[test]
fn event_unwind_restores_observer_and_nested_trigger_cannot_reenter_it() {
    let mut world = World::new();
    let entity = world.spawn(());
    let calls = Arc::new(AtomicUsize::new(0));
    world.observe_event::<Ping, _>({
        let calls = calls.clone();
        move |ctx| {
            ctx.world.trigger::<Ping, Parent>(ctx.target, false);
            assert_ne!(
                calls.fetch_add(1, Ordering::Relaxed),
                0,
                "first delivery panics"
            );
        }
    });
    let others = Arc::new(AtomicUsize::new(0));
    world.observe_event::<Ping, _>({
        let others = others.clone();
        move |_| {
            others.fetch_add(1, Ordering::Relaxed);
        }
    });
    assert!(catch_unwind(AssertUnwindSafe(
        || world.trigger::<Ping, Parent>(entity, false)
    ))
    .is_err());
    assert_eq!(others.load(Ordering::Relaxed), 1);
    world.trigger::<Ping, Parent>(entity, false);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(others.load(Ordering::Relaxed), 3);
}

#[test]
fn event_registration_is_deferred_across_ancestors_and_stop_only_stops_bubbling() {
    let mut world = World::new();
    world.register_relation::<Parent>(RelationKind::new());
    let parent = world.spawn(());
    let child = world.spawn(());
    world.add_relation::<Parent>(child, parent);
    let new_calls = Arc::new(AtomicUsize::new(0));
    let mut first = true;
    world.observe_event::<Ping, _>({
        let new_calls = new_calls.clone();
        move |ctx| {
            if first {
                first = false;
                let new_calls = new_calls.clone();
                ctx.world.observe_event::<Ping, _>(move |_| {
                    new_calls.fetch_add(1, Ordering::Relaxed);
                });
            }
        }
    });
    world.trigger::<Ping, Parent>(child, true);
    assert_eq!(new_calls.load(Ordering::Relaxed), 0);
    world.trigger::<Ping, Parent>(child, true);
    assert_eq!(new_calls.load(Ordering::Relaxed), 2);
    world.observe_event::<Ping, _>(|ctx| ctx.stop_propagation());
    let last_calls = Arc::new(AtomicUsize::new(0));
    world.observe_event::<Ping, _>({
        let last_calls = last_calls.clone();
        move |_| {
            last_calls.fetch_add(1, Ordering::Relaxed);
        }
    });
    world.trigger::<Ping, Parent>(child, true);
    assert_eq!(last_calls.load(Ordering::Relaxed), 1);
    assert_eq!(new_calls.load(Ordering::Relaxed), 3);
}

#[test]
fn hook_world_replacement_abandons_remaining_hooks_and_observers() {
    fn replace(ctx: HookContext<'_>) {
        *ctx.world = World::new();
    }
    fn stale(_: HookContext<'_>) {
        panic!("stale insert hook");
    }
    let mut world = World::new();
    world.register_component_hooks::<Value>(
        ComponentHooks::new()
            .with_on_add(replace)
            .with_on_insert(stale),
    );
    world.observe::<Value, _>(LifecycleEvent::Add, |_| panic!("stale add observer"));
    world.spawn(Value(1));
    let entity = world.spawn(());
    assert_eq!(world.get::<Value>(entity), None);
}

#[test]
fn event_restore_stops_old_delivery_and_restores_running_slot_even_on_panic() {
    for panic_after in [false, true] {
        let mut world = World::new();
        world.register_snapshot_component::<Value>();
        let entity = world.spawn(Value(1));
        SNAPSHOT.with(|slot| *slot.borrow_mut() = Some(world.snapshot()));
        world.get_mut::<Value>(entity).unwrap().0 = 2;
        let mut first = true;
        world.observe_event::<Ping, _>(move |ctx| {
            if first {
                first = false;
                restore(ctx.world);
                assert!(!panic_after, "intentional callback panic");
            }
        });
        let calls = Arc::new(AtomicUsize::new(0));
        world.observe_event::<Ping, _>({
            let calls = calls.clone();
            move |_| {
                calls.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(
            catch_unwind(AssertUnwindSafe(
                || world.trigger::<Ping, Parent>(entity, false)
            ))
            .is_err(),
            panic_after
        );
        assert_eq!(world.get::<Value>(entity), Some(&Value(1)));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        world.trigger::<Ping, Parent>(entity, false);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        SNAPSHOT.with(|slot| *slot.borrow_mut() = None);
    }
}

#[test]
fn running_event_can_remove_itself_even_when_panicking() {
    for panic_after in [false, true] {
        let mut world = World::new();
        let entity = world.spawn(());
        let id = Arc::new(Mutex::new(None::<ObserverId>));
        let registration = world.observe_event::<Ping, _>({
            let id = id.clone();
            move |ctx| {
                assert!(ctx.world.remove_observer(id.lock().unwrap().unwrap()));
                assert!(!panic_after, "intentional callback panic");
            }
        });
        *id.lock().unwrap() = Some(registration);
        assert_eq!(
            catch_unwind(AssertUnwindSafe(
                || world.trigger::<Ping, Parent>(entity, false)
            ))
            .is_err(),
            panic_after
        );
        assert!(!world.remove_observer(registration));
        world.trigger::<Ping, Parent>(entity, false);
    }
}

#[test]
fn lifecycle_panic_keeps_owning_groups_consistent_with_committed_storage() {
    for operation in [Operation::Insert, Operation::Remove, Operation::Despawn] {
        let mut world = World::new();
        let component = world.register_component::<Value>();
        let group = world.register_owning_group(&[component]).unwrap();
        let entity = if matches!(operation, Operation::Insert) {
            world.spawn(())
        } else {
            world.spawn(Value(1))
        };
        let event = if matches!(operation, Operation::Insert) {
            LifecycleEvent::Add
        } else {
            LifecycleEvent::Remove
        };
        world.observe::<Value, _>(event, |_| panic!("intentional callback panic"));
        assert!(catch_unwind(AssertUnwindSafe(|| apply(&mut world, entity, operation))).is_err());
        assert!(world.contains(entity));
        assert!(world.has::<Value>(entity));
        assert_eq!(
            world
                .owning_group(group)
                .unwrap()
                .iter()
                .map(|(entity, _)| entity)
                .collect::<Vec<_>>(),
            [entity],
            "{operation:?}"
        );
    }
    let mut world = World::new();
    let component = world.register_component::<Value>();
    let group = world.register_owning_group(&[component]).unwrap();
    world.observe::<Value, _>(LifecycleEvent::Add, |_| {
        panic!("intentional callback panic")
    });
    assert!(catch_unwind(AssertUnwindSafe(|| world.spawn(Value(1)))).is_err());
    let entities = world.query::<Entity>().iter(&world).collect::<Vec<_>>();
    assert_eq!(entities.len(), 1);
    assert_eq!(
        world
            .owning_group(group)
            .unwrap()
            .iter()
            .map(|(entity, _)| entity)
            .collect::<Vec<_>>(),
        entities
    );
}
