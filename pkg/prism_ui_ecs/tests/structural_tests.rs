//! Integration tests for signal-driven structural bindings (`Show` / `For`).
//!
//! These exercise the public API against a real Bevy [`World`] and a real
//! reactive [`Runtime`], covering conditional mounting, keyed reuse across
//! reorders, and batched flushing through a [`StructuralScope`].

use bevy_ecs::prelude::{Component, Entity, World};
use prism_ui_ecs::{ForBinding, ShowBinding, StructuralBinding, StructuralScope};
use prism_ui_reactive::Runtime;

#[derive(Component)]
struct Label {
    id: i32,
}

// `World::new()` seeds an internal entity for the default query filters,
// so user-spawned entities are counted relative to that baseline.
fn count_entities(world: &mut World) -> usize {
    let baseline = World::new().iter_entities().count();
    world.iter_entities().count() - baseline
}

#[test]
fn show_full_lifecycle_true_false_true() {
    let mut world = World::new();
    let rt = Runtime::new();
    let visible = rt.signal(false);

    let mut show = ShowBinding::new(visible.clone(), |w| w.spawn(Label { id: 0 }).id());

    // Hidden initially: nothing spawned.
    show.reconcile(&world);
    assert_eq!(show.flush(&mut world).total(), 0);
    assert!(show.current_entity().is_none());
    assert_eq!(count_entities(&mut world), 0);

    // Turn on.
    visible.set(true);
    show.reconcile(&world);
    assert_eq!(show.flush(&mut world).spawned, 1);
    assert!(show.is_mounted());
    let mounted = show.current_entity().unwrap();
    assert_eq!(count_entities(&mut world), 1);

    // Turn on again (idempotent): no new spawn, same entity.
    visible.set(true);
    show.reconcile(&world);
    assert_eq!(show.flush(&mut world).total(), 0);
    assert_eq!(show.current_entity(), Some(mounted));

    // Turn off.
    visible.set(false);
    show.reconcile(&world);
    assert_eq!(show.flush(&mut world).despawned, 1);
    assert!(!show.is_mounted());
    assert_eq!(count_entities(&mut world), 0);

    // Turn on once more: fresh entity.
    visible.set(true);
    show.reconcile(&world);
    assert_eq!(show.flush(&mut world).spawned, 1);
    assert_eq!(count_entities(&mut world), 1);
}

#[test]
fn for_reorder_reuses_entities_minimally() {
    let mut world = World::new();
    let rt = Runtime::new();
    let items = rt.signal(vec![1i32, 2, 3]);

    let mut binding = ForBinding::new(
        items.clone(),
        |value: &i32| *value,
        |w, value: &i32| w.spawn(Label { id: *value }).id(),
    );

    binding.reconcile(&world);
    assert_eq!(binding.flush(&mut world).spawned, 3);

    let e1 = binding.entity_for(&1).unwrap();
    let e2 = binding.entity_for(&2).unwrap();
    let e3 = binding.entity_for(&3).unwrap();

    // Reorder the same keys: no spawn, no despawn, ids preserved.
    items.set(vec![3i32, 1, 2]);
    binding.reconcile(&world);
    let stats = binding.flush(&mut world);
    assert_eq!(stats.spawned, 0);
    assert_eq!(stats.despawned, 0);

    assert_eq!(binding.entity_for(&1), Some(e1));
    assert_eq!(binding.entity_for(&2), Some(e2));
    assert_eq!(binding.entity_for(&3), Some(e3));

    // Entities ordered by the new key order.
    assert_eq!(binding.entities(), vec![e3, e1, e2]);
    assert_eq!(count_entities(&mut world), 3);
}

#[test]
fn for_add_and_remove_minimal_set() {
    let mut world = World::new();
    let rt = Runtime::new();
    let items = rt.signal(vec![10i32, 20, 30]);

    let mut binding = ForBinding::new(
        items.clone(),
        |value: &i32| *value,
        |w, value: &i32| w.spawn(Label { id: *value }).id(),
    );
    binding.reconcile(&world);
    binding.flush(&mut world);
    let e10 = binding.entity_for(&10).unwrap();
    let e30 = binding.entity_for(&30).unwrap();

    // Drop 20, add 40: exactly one despawn and one spawn; survivors keep ids.
    items.set(vec![10i32, 30, 40]);
    binding.reconcile(&world);
    let stats = binding.flush(&mut world);
    assert_eq!(stats.spawned, 1);
    assert_eq!(stats.despawned, 1);

    assert_eq!(binding.entity_for(&10), Some(e10));
    assert_eq!(binding.entity_for(&30), Some(e30));
    assert!(binding.entity_for(&20).is_none());
    assert!(binding.entity_for(&40).is_some());
    assert_eq!(count_entities(&mut world), 3);
}

#[test]
fn scope_batches_until_flush_and_counts() {
    let mut world = World::new();
    let rt = Runtime::new();
    let visible = rt.signal(true);
    let items = rt.signal(vec![1i32, 2]);

    let mut scope = StructuralScope::new();
    scope.add(ShowBinding::new(visible.clone(), |w| {
        w.spawn(Label { id: -1 }).id()
    }));
    scope.add(ForBinding::new(
        items.clone(),
        |value: &i32| *value,
        |w, value: &i32| w.spawn(Label { id: *value }).id(),
    ));

    // Reconcile alone must not touch the world.
    scope.reconcile_all(&world);
    assert_eq!(count_entities(&mut world), 0);

    // Flush applies everything: 1 (show) + 2 (for) spawns.
    let stats = scope.flush(&mut world);
    assert_eq!(stats.spawned, 3);
    assert_eq!(stats.despawned, 0);
    assert_eq!(count_entities(&mut world), 3);

    // Next frame: hide the show entity, grow the list by one.
    visible.set(false);
    items.set(vec![1i32, 2, 3]);
    let stats = scope.run(&mut world);
    assert_eq!(stats.spawned, 1);
    assert_eq!(stats.despawned, 1);
    assert_eq!(count_entities(&mut world), 3);
}

#[test]
fn for_with_update_refreshes_reused_entities() {
    let mut world = World::new();
    let rt = Runtime::new();
    // (key, payload); key is stable, payload changes in place.
    let items = rt.signal(vec![(1i32, 100i32), (2, 200)]);

    let mut binding = ForBinding::with_update(
        items.clone(),
        |pair: &(i32, i32)| pair.0,
        |w, pair: &(i32, i32)| w.spawn(Label { id: pair.1 }).id(),
        |w, entity: Entity, pair: &(i32, i32)| {
            if let Some(mut label) = w.get_mut::<Label>(entity) {
                label.id = pair.1;
            }
        },
    );
    binding.reconcile(&world);
    binding.flush(&mut world);
    let e1 = binding.entity_for(&1).unwrap();
    assert_eq!(world.get::<Label>(e1).unwrap().id, 100);

    // Same keys, new payloads: no structural change, fields refreshed.
    items.set(vec![(1i32, 111i32), (2, 222)]);
    binding.reconcile(&world);
    let stats = binding.flush(&mut world);
    assert_eq!(stats.total(), 0);
    assert_eq!(binding.entity_for(&1), Some(e1));
    assert_eq!(world.get::<Label>(e1).unwrap().id, 111);
}
