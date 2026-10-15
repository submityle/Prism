//! Behavioural tests for the extraction contract (design §23.5).

use alloc::vec::Vec;

use crate::component::Component;
use crate::entity::Entity;
use crate::extract::{
    extract_components, extract_resource, ComponentExtraction, ExtractComponent, ExtractResource,
    ExtractionStats,
};
use crate::query::{Changed, With};
use crate::resource::Resource;
use crate::world::World;

/// Simulation-side resource.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SunAngle(f32);
impl Resource for SunAngle {}

/// Secondary-side projection of [`SunAngle`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct SunDirection([f32; 3]);
impl Resource for SunDirection {}

impl ExtractResource for SunAngle {
    type Out = SunDirection;

    fn extract(&self, _main: &World) -> Option<Self::Out> {
        // A projection, not a physical model: exact arithmetic keeps the test
        // free of platform-dependent float differences (and the workspace lint
        // bar disallows transcendental calls in this crate).
        Some(SunDirection([self.0, 1.0 - self.0, 0.0]))
    }
}

/// A resource whose extractor can decline a frame.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Paused(bool);
impl Resource for Paused {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PausedView(bool);
impl Resource for PausedView {}

impl ExtractResource for Paused {
    type Out = PausedView;

    fn extract(&self, _main: &World) -> Option<Self::Out> {
        (!self.0).then_some(PausedView(self.0))
    }
}

/// Simulation-side component.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Health {
    hp: i32,
    max: i32,
}
impl Component for Health {}

/// Secondary-side projection of [`Health`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct HealthBar(f32);
impl Component for HealthBar {}

/// Marker on the simulation side that narrows what gets extracted.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Visible;
impl Component for Visible {}

impl ExtractComponent for Health {
    type Out = HealthBar;
    type QueryData = (&'static Health, &'static Visible);
    type QueryFilter = With<Visible>;

    fn extract((health, _visible): (&Health, &Visible)) -> Option<Self::Out> {
        (health.max > 0).then(|| HealthBar(health.hp as f32 / health.max as f32))
    }
}

#[test]
fn resources_are_published_as_a_projection() {
    let mut main = World::new();
    let mut secondary = World::new();
    main.insert_resource(SunAngle(0.25));

    assert!(extract_resource::<SunAngle>(&main, &mut secondary));
    assert_eq!(
        secondary.get_resource::<SunDirection>(),
        Some(&SunDirection([0.25, 0.75, 0.0]))
    );

    // Re-extracting overwrites with the new frame's value.
    main.insert_resource(SunAngle(0.5));
    assert!(extract_resource::<SunAngle>(&main, &mut secondary));
    assert_eq!(
        secondary.get_resource::<SunDirection>(),
        Some(&SunDirection([0.5, 0.5, 0.0]))
    );
}

#[test]
fn a_missing_or_declining_resource_leaves_the_secondary_world_alone() {
    let mut main = World::new();
    let mut secondary = World::new();

    // Absent on the main side.
    assert!(!extract_resource::<SunAngle>(&main, &mut secondary));
    assert!(secondary.get_resource::<SunDirection>().is_none());

    // Present, but the extractor declines this frame.
    main.insert_resource(Paused(true));
    assert!(!extract_resource::<Paused>(&main, &mut secondary));
    assert!(secondary.get_resource::<PausedView>().is_none());

    // And publishes once it stops declining.
    main.insert_resource(Paused(false));
    assert!(extract_resource::<Paused>(&main, &mut secondary));
    assert_eq!(
        secondary.get_resource::<PausedView>(),
        Some(&PausedView(false))
    );
}

#[test]
fn components_spawn_one_secondary_entity_per_extracted_row() {
    let mut main = World::new();
    let mut secondary = World::new();
    let full = main.spawn((Health { hp: 5, max: 10 }, Visible));
    let empty = main.spawn((Health { hp: 0, max: 0 }, Visible)); // extractor declines
    let hidden = main.spawn(Health { hp: 1, max: 2 }); // filtered out

    let mapping = extract_components::<Health>(&mut main, &mut secondary);

    assert_eq!(
        mapping.len(),
        1,
        "only the extracted row produces an entity"
    );
    assert_eq!(mapping[0].0, full);
    assert_eq!(
        secondary.get::<HealthBar>(mapping[0].1),
        Some(&HealthBar(0.5))
    );
    // The declined and the filtered entities produced nothing, and the secondary
    // entity really lives in the secondary world (its own allocator decides the
    // id, so it may or may not coincide with the main one).
    assert!(secondary.contains(mapping[0].1));
    assert_eq!(secondary.entity_count(), 1);
    let _ = (empty, hidden);
}

#[test]
fn extraction_never_mutates_or_dirty_marks_the_simulation() {
    let mut main = World::new();
    let mut secondary = World::new();
    let e = main.spawn((Health { hp: 3, max: 4 }, Visible));

    // Advance a frame so the spawn tick is in the past.
    let now = main.change_tick();
    main.set_last_change_tick(now);
    main.increment_change_tick();

    let before = main.get::<Health>(e).copied();
    let _ = extract_components::<Health>(&mut main, &mut secondary);

    assert_eq!(
        main.get::<Health>(e).copied(),
        before,
        "values are untouched"
    );
    // A read-only extraction must not look like a write to change detection.
    let changed = main.query_filtered::<&Health, Changed<Health>>();
    assert_eq!(
        changed.iter(&main).count(),
        0,
        "extraction must not dirty rows"
    );
}

#[test]
fn repeated_extraction_rebuilds_the_secondary_entities() {
    let mut main = World::new();
    let mut secondary = World::new();
    for i in 1..=3 {
        main.spawn((Health { hp: i, max: 4 }, Visible));
    }

    let first: Vec<Entity> = extract_components::<Health>(&mut main, &mut secondary)
        .into_iter()
        .map(|(_, secondary)| secondary)
        .collect();
    assert_eq!(first.len(), 3);

    // A second pass spawns a fresh set (the caller owns the mapping, and is the
    // one that despawns the previous frame's entities if it wants to).
    let second: Vec<Entity> = extract_components::<Health>(&mut main, &mut secondary)
        .into_iter()
        .map(|(_, secondary)| secondary)
        .collect();
    assert_eq!(second.len(), 3);
    assert!(second.iter().all(|e| !first.contains(e)));
}

#[test]
fn continuing_extraction_updates_and_retires_without_accumulating_entities() {
    let mut main = World::new();
    let mut secondary = World::new();
    let visible = main.spawn((Health { hp: 5, max: 10 }, Visible));
    let declined = main.spawn((Health { hp: 0, max: 0 }, Visible));
    let unrelated = secondary.spawn(HealthBar(42.0));
    let mut stream = ComponentExtraction::<Health>::new(&mut main, &mut secondary);
    assert_eq!(stream.sync(&main, &mut secondary).spawned, 1);
    let target = stream.entity(visible).unwrap();
    for hp in 1..=32 {
        main.get_mut::<Health>(visible).unwrap().hp = hp;
        let stats = stream.sync(&main, &mut secondary);
        assert_eq!(stats.updated, 1);
        assert_eq!(stats.spawned, 0);
        assert_eq!(stream.entity(visible), Some(target));
        assert_eq!(
            secondary.get::<HealthBar>(target),
            Some(&HealthBar(hp as f32 / 10.0))
        );
        assert_eq!(secondary.entity_count(), 2);
    }
    assert!(stream.entity(declined).is_none());
    main.remove::<Visible>(visible);
    assert_eq!(stream.sync(&main, &mut secondary).despawned, 1);
    assert!(!secondary.contains(target));
    assert!(secondary.contains(unrelated));
    main.insert(visible, Visible);
    assert_eq!(stream.sync(&main, &mut secondary).spawned, 1);
    main.get_mut::<Health>(visible).unwrap().max = 0;
    assert_eq!(stream.sync(&main, &mut secondary).despawned, 1);
    assert_eq!(secondary.entity_count(), 1);
}

#[derive(Clone, Debug, PartialEq)]
struct DeltaValue {
    value: i32,
    fail: bool,
}
impl Component for DeltaValue {}

#[derive(Clone, Debug, PartialEq)]
struct DeltaView(i32);
impl Component for DeltaView {}

impl ExtractComponent for DeltaValue {
    type Out = DeltaView;
    type QueryData = &'static DeltaValue;
    type QueryFilter = Changed<DeltaValue>;
    fn extract(value: &DeltaValue) -> Option<DeltaView> {
        assert!(!value.fail, "extractor failure");
        Some(DeltaView(value.value))
    }
}

#[test]
fn temporal_extraction_retains_idle_members_and_uses_its_own_tick_cursor() {
    let mut main = World::new();
    let mut secondary = World::new();
    let source = main.spawn((
        DeltaValue {
            value: 1,
            fail: false,
        },
        Visible,
    ));
    let mut stream = ComponentExtraction::<DeltaValue, With<Visible>>::with_membership(
        &mut main,
        &mut secondary,
    );
    assert_eq!(stream.sync(&main, &mut secondary).spawned, 1);
    let target = stream.entity(source).unwrap();
    main.increment_change_tick();
    assert_eq!(
        stream.sync(&main, &mut secondary),
        ExtractionStats {
            retained: 1,
            ..Default::default()
        }
    );
    main.increment_change_tick();
    main.get_mut::<DeltaValue>(source).unwrap().value = 7;
    // Another consumer already read the frame; this stream must still see it.
    main.set_last_change_tick(main.change_tick());
    assert_eq!(stream.sync(&main, &mut secondary).updated, 1);
    assert_eq!(stream.entity(source), Some(target));
    assert_eq!(secondary.get::<DeltaView>(target), Some(&DeltaView(7)));
    main.increment_change_tick();
    main.remove::<Visible>(source);
    assert_eq!(stream.sync(&main, &mut secondary).despawned, 1);
    // The value is old, but rejoining membership must bootstrap a projection.
    main.increment_change_tick();
    main.insert(source, Visible);
    assert_eq!(stream.sync(&main, &mut secondary).spawned, 1);
    assert_eq!(
        secondary.get::<DeltaView>(stream.entity(source).unwrap()),
        Some(&DeltaView(7))
    );
}

#[test]
fn continuing_extraction_repairs_deletion_and_never_touches_recycled_slots() {
    let mut main = World::new();
    let mut secondary = World::new();
    let source = main.spawn((Health { hp: 1, max: 2 }, Visible));
    let mut stream = ComponentExtraction::<Health>::new(&mut main, &mut secondary);
    stream.sync(&main, &mut secondary);
    let old = stream.entity(source).unwrap();
    secondary.despawn(old);
    let unrelated = secondary.spawn(HealthBar(99.0));
    assert_eq!(old.index(), unrelated.index());
    assert_eq!(stream.sync(&main, &mut secondary).spawned, 1);
    assert_ne!(stream.entity(source), Some(unrelated));
    assert_eq!(
        secondary.get::<HealthBar>(unrelated),
        Some(&HealthBar(99.0))
    );
    assert_eq!(stream.clear(&mut secondary), 1);
    assert_eq!(secondary.entity_count(), 1);
    assert_eq!(stream.clear(&mut secondary), 0);
    assert_eq!(stream.sync(&main, &mut secondary).spawned, 1);
}

#[test]
fn streams_keep_independent_ownership_in_the_same_secondary_world() {
    let mut main = World::new();
    let mut secondary = World::new();
    let source = main.spawn((Health { hp: 3, max: 4 }, Visible));
    let mut first = ComponentExtraction::<Health>::new(&mut main, &mut secondary);
    let mut second = ComponentExtraction::<Health>::new(&mut main, &mut secondary);
    first.sync(&main, &mut secondary);
    second.sync(&main, &mut secondary);
    assert_ne!(first.entity(source), second.entity(source));
    first.clear(&mut secondary);
    assert!(secondary.contains(second.entity(source).unwrap()));
    assert_eq!(secondary.entity_count(), 1);
}

#[test]
fn extraction_recovers_secondary_rollback_after_the_mapping_was_cleared() {
    let mut main = World::new();
    let mut secondary = World::new();
    secondary.register_snapshot_component::<HealthBar>();
    let source = main.spawn((Health { hp: 3, max: 4 }, Visible));
    let mut stream = ComponentExtraction::<Health>::new(&mut main, &mut secondary);
    stream.sync(&main, &mut secondary);
    let target = stream.entity(source).unwrap();
    let saved = secondary.snapshot();
    stream.clear(&mut secondary);
    secondary.restore(&saved);
    assert_eq!(stream.sync(&main, &mut secondary).updated, 1);
    assert_eq!(stream.entity(source), Some(target));
    assert_eq!(secondary.entity_count(), 1);
    main.despawn(source);
    assert_eq!(stream.sync(&main, &mut secondary).despawned, 1);
}

#[test]
fn source_rollback_forces_projection_even_when_ticks_rewind() {
    let mut main = World::new();
    let mut secondary = World::new();
    main.register_snapshot_component::<DeltaValue>();
    main.register_snapshot_component::<Visible>();
    let source = main.spawn((
        DeltaValue {
            value: 1,
            fail: false,
        },
        Visible,
    ));
    let saved = main.snapshot();
    let mut stream = ComponentExtraction::<DeltaValue, With<Visible>>::with_membership(
        &mut main,
        &mut secondary,
    );
    stream.sync(&main, &mut secondary);
    let target = stream.entity(source).unwrap();
    main.increment_change_tick();
    main.get_mut::<DeltaValue>(source).unwrap().value = 50;
    stream.sync(&main, &mut secondary);
    assert_eq!(secondary.get::<DeltaView>(target), Some(&DeltaView(50)));
    main.restore(&saved);
    assert_eq!(stream.sync(&main, &mut secondary).updated, 1);
    assert_eq!(secondary.get::<DeltaView>(target), Some(&DeltaView(1)));
}

#[cfg(feature = "std")]
#[test]
fn extractor_panic_does_not_publish_or_advance_the_successful_tick() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let mut main = World::new();
    let mut secondary = World::new();
    let first = main.spawn((
        DeltaValue {
            value: 1,
            fail: false,
        },
        Visible,
    ));
    let second = main.spawn((
        DeltaValue {
            value: 2,
            fail: false,
        },
        Visible,
    ));
    let mut stream = ComponentExtraction::<DeltaValue, With<Visible>>::with_membership(
        &mut main,
        &mut secondary,
    );
    stream.sync(&main, &mut secondary);
    let first_view = stream.entity(first).unwrap();
    let second_view = stream.entity(second).unwrap();
    main.increment_change_tick();
    main.get_mut::<DeltaValue>(first).unwrap().value = 10;
    main.get_mut::<DeltaValue>(second).unwrap().fail = true;
    assert!(catch_unwind(AssertUnwindSafe(|| stream.sync(&main, &mut secondary))).is_err());
    assert_eq!(secondary.get::<DeltaView>(first_view), Some(&DeltaView(1)));
    assert_eq!(secondary.get::<DeltaView>(second_view), Some(&DeltaView(2)));
    main.get_mut::<DeltaValue>(second).unwrap().fail = false;
    assert_eq!(stream.sync(&main, &mut secondary).updated, 2);
    assert_eq!(secondary.get::<DeltaView>(first_view), Some(&DeltaView(10)));
}

#[cfg(feature = "std")]
#[test]
fn extraction_rejects_cross_world_reuse_before_publication() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let mut main = World::new();
    let mut secondary = World::new();
    main.spawn((Health { hp: 1, max: 2 }, Visible));
    let mut stream = ComponentExtraction::<Health>::new(&mut main, &mut secondary);
    let wrong_source = World::new();
    assert!(catch_unwind(AssertUnwindSafe(
        || stream.sync(&wrong_source, &mut secondary)
    ))
    .is_err());
    assert_eq!(secondary.entity_count(), 0);
    let mut wrong_secondary = World::new();
    assert!(catch_unwind(AssertUnwindSafe(|| stream.sync(&main, &mut wrong_secondary))).is_err());
    assert_eq!(wrong_secondary.entity_count(), 0);
    assert_eq!(stream.sync(&main, &mut secondary).spawned, 1);
}
