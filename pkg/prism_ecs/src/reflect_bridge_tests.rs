//! Behavioural tests for the reflection bridge (design §16.5 / §24.3).

use prism_reflect::Reflect;

use crate::component::Component;
use crate::entity::Entity;
use crate::query::{Changed, With};
use crate::world::World;

/// A reflected component (integer payload keeps the test bit-exact).
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq)]
struct Health(i32);
impl Component for Health {}

/// A second reflected component, to prove ids are per-type.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq)]
struct Speed(i32);
impl Component for Speed {}

/// Reflected, but deliberately never registered in the world under test.
#[derive(Reflect, Debug, Clone, Copy, PartialEq, Eq)]
struct Opaque(i32);
impl Component for Opaque {}

#[test]
fn register_then_get_apply_and_remove_round_trips() {
    let mut world = World::new();
    let health = world.register_reflect_component::<Health>();
    let speed = world.register_reflect_component::<Speed>();
    assert_ne!(health, speed, "each type gets its own stable id");
    assert_eq!(world.reflected_component_count(), 2);

    let e = world.spawn(Health(10));

    // Dynamic read: no Rust type named at the call site.
    let read = world
        .get_reflect(e, health)
        .expect("entity carries Health")
        .downcast_ref::<Health>()
        .copied()
        .expect("the concrete type matches");
    assert_eq!(read, Health(10));

    // The other registered type does not match this entity.
    assert!(world.get_reflect(e, speed).is_none());

    // Dynamic overwrite by reflected value.
    assert!(world.insert_reflect(e, &Health(42)));
    assert_eq!(world.get::<Health>(e), Some(&Health(42)));

    // Dynamic remove, then it is gone for real.
    assert!(world.remove_reflect(e, health));
    assert!(world.get::<Health>(e).is_none());
    assert!(!world.remove_reflect(e, health), "second remove is a no-op");
}

#[test]
fn unregistered_or_mismatched_values_are_rejected() {
    let mut world = World::new();
    let health = world.register_reflect_component::<Health>();
    let e = world.spawn(Health(1));

    // A type that was never registered has no stable id in this world.
    let opaque = prism_reflect::StableTypeId::of_type::<Opaque>();
    assert!(world.get_reflect(e, opaque).is_none());
    assert!(!world.remove_reflect(e, opaque));
    assert!(!world.insert_reflect(e, &Opaque(9)));

    // A registered *other* type also cannot write through `Health`'s id: the
    // value selects the component, so `Speed` would need its own registration.
    assert!(!world.insert_reflect(e, &Speed(9)));
    assert_eq!(world.get::<Health>(e), Some(&Health(1)));
    // ... and the original component is untouched.
    assert!(world.get_reflect(e, health).is_some());
}

#[test]
fn stable_and_local_ids_map_both_ways() {
    let mut world = World::new();
    let stable = world.register_reflect_component::<Health>();
    let local = world
        .reflect_component_id(stable)
        .expect("registered types resolve to a component id");
    assert_eq!(world.reflect_stable_id(local), Some(stable));

    // Registering again is idempotent and keeps the same mapping.
    assert_eq!(world.register_reflect_component::<Health>(), stable);
    assert_eq!(world.reflect_component_id(stable), Some(local));
}

#[test]
fn get_reflect_mut_marks_the_component_changed() {
    let mut world = World::new();
    let health = world.register_reflect_component::<Health>();
    let e = world.spawn(Health(5));

    // Advance a frame so the spawn tick is in the past.
    let now = world.change_tick();
    world.set_last_change_tick(now);
    world.increment_change_tick();

    {
        let value = world
            .get_reflect_mut(e, health)
            .expect("entity carries Health");
        // Patch a field through the reflected view.
        value
            .downcast_mut::<Health>()
            .expect("concrete type")
            .0 += 37;
    }
    assert_eq!(world.get::<Health>(e), Some(&Health(42)));

    // A modification through reflection must be visible to change detection,
    // exactly like `world.get_mut::<Health>`.
    let changed = world.query_filtered::<Entity, Changed<Health>>();
    assert_eq!(changed.get(&world, e), Ok(e));
}

#[test]
fn dynamic_access_composes_with_filters_and_other_components() {
    let mut world = World::new();
    let health = world.register_reflect_component::<Health>();
    let e = world.spawn((Health(7), Speed(3)));

    // The bridge does not disturb ordinary typed queries.
    let with_speed = world.query_filtered::<&Health, With<Speed>>();
    assert_eq!(with_speed.get(&world, e), Ok(&Health(7)));

    // Overwriting through the bridge keeps the rest of the entity intact.
    assert!(world.insert_reflect(e, &Health(8)));
    assert_eq!(world.get::<Speed>(e), Some(&Speed(3)));
    assert_eq!(
        world
            .get_reflect(e, health)
            .and_then(|v| v.downcast_ref::<Health>())
            .copied(),
        Some(Health(8))
    );
}

/// A shared reflected read must *not* mark the component changed — only the
/// mutable view does.
#[test]
fn a_shared_reflect_read_does_not_mark_changed() {
    let mut world = World::new();
    let health = world.register_reflect_component::<Health>();
    let e = world.spawn(Health(1));

    let now = world.change_tick();
    world.set_last_change_tick(now);
    world.increment_change_tick();

    let _ = world.get_reflect(e, health).expect("live");

    let changed = world.query_filtered::<Entity, Changed<Health>>();
    assert_eq!(
        changed.get(&world, e),
        Err(crate::query::QueryEntityError::QueryDoesNotMatch(e))
    );
}
