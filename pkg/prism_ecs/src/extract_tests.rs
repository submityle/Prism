//! Behavioural tests for the extraction contract (design §23.5).

use alloc::vec::Vec;

use crate::component::Component;
use crate::entity::Entity;
use crate::extract::{extract_components, extract_resource, ExtractComponent, ExtractResource};
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
