//! End-to-end World history through the UI time-travel adapter.
#![cfg(feature = "ecs")]

use prism_ecs::{
    component::{Component, StorageType},
    resource::Resource,
    world::World,
};
use prism_ui::Element;
use prism_ui_timetravel::{EcsHistoryError, EcsTimeline, Frame};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Position(i32);
impl Component for Position {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Sparse(i32);
impl Component for Sparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

struct NoClone;
impl Component for NoClone {}

#[derive(Clone)]
struct Score(i32);
impl Resource for Score {}

struct External(i32);
impl Resource for External {}

#[test]
fn seek_restores_real_entities_sparse_values_ticks_and_opt_in_resources() {
    let mut world = World::new();
    world.register_snapshot_component::<Position>();
    world.register_snapshot_component::<Sparse>();
    world.register_snapshot_resource::<Score>();
    world.insert_resource(Score(1));
    world.insert_resource(External(10));
    let entity = world.spawn((Position(3), Sparse(5)));
    let tick = world.change_tick();
    let changed = world.get_ref::<Position>(entity).unwrap().changed_tick();
    let mut timeline = EcsTimeline::new(&world, 4);
    timeline
        .record_with_ui(
            &world,
            "before",
            Some(Frame::capture("before", &Element::text("3"))),
        )
        .unwrap();

    world.increment_change_tick();
    world.despawn(entity);
    let replacement = world.spawn((Position(99), Sparse(100)));
    world.resource_mut::<Score>().0 = 2;
    world.resource_mut::<External>().0 = 20;
    timeline.record(&world, "after").unwrap();

    let frame = timeline.undo(&mut world).unwrap().unwrap();
    assert_eq!(frame.label(), "before");
    assert_eq!(
        frame.ui().unwrap().snapshot().root.text.as_deref(),
        Some("3")
    );
    assert!(world.contains(entity));
    assert!(!world.contains(replacement));
    assert_eq!(world.get::<Position>(entity), Some(&Position(3)));
    assert_eq!(world.get::<Sparse>(entity), Some(&Sparse(5)));
    assert_eq!(world.change_tick(), tick);
    assert_eq!(
        world.get_ref::<Position>(entity).unwrap().changed_tick(),
        changed
    );
    assert_eq!(world.resource::<Score>().0, 1);
    assert_eq!(world.resource::<External>().0, 20);

    assert_eq!(timeline.redo(&mut world).unwrap().unwrap().label(), "after");
    assert!(!world.contains(entity));
    assert!(world.contains(replacement));
    assert_eq!(world.resource::<Score>().0, 2);
}

#[test]
fn failed_capture_keeps_redo_and_successful_record_forks_and_evicts() {
    let mut world = World::new();
    world.register_snapshot_component::<Position>();
    let entity = world.spawn(Position(1));
    let mut history = EcsTimeline::new(&world, 2);
    history.record(&world, "one").unwrap();
    world.get_mut::<Position>(entity).unwrap().0 = 2;
    history.record(&world, "two").unwrap();
    history.undo(&mut world).unwrap();

    let opaque = world.spawn(NoClone);
    assert!(matches!(
        history.record(&world, "bad"),
        Err(EcsHistoryError::MissingCloneGlue(_))
    ));
    assert_eq!(history.cursor(), Some(0));
    assert_eq!(history.len(), 2);
    assert!(history.can_redo());
    world.despawn(opaque);
    world.get_mut::<Position>(entity).unwrap().0 = 3;
    history.record(&world, "branch").unwrap();
    assert!(!history.can_redo());
    assert_eq!(history.frames()[1].label(), "branch");
    history.record(&world, "latest").unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history.frames()[0].label(), "branch");
    assert_eq!(history.cursor(), Some(1));
    history.undo(&mut world).unwrap();
    assert_eq!(world.get::<Position>(entity), Some(&Position(3)));
}

#[test]
fn identity_and_invalid_index_errors_do_not_touch_world_or_cursor() {
    let world = World::new();
    let mut history = EcsTimeline::new(&world, 0);
    history.record(&world, "source").unwrap();
    let mut other = World::new();
    let entity = other.spawn(Position(9));
    assert!(matches!(
        history.seek(&mut other, 0),
        Err(EcsHistoryError::WrongWorld)
    ));
    assert!(matches!(
        history.undo(&mut other),
        Err(EcsHistoryError::WrongWorld)
    ));
    assert!(matches!(
        history.record(&other, "foreign"),
        Err(EcsHistoryError::WrongWorld)
    ));
    assert_eq!(other.get::<Position>(entity), Some(&Position(9)));
    assert_eq!(history.cursor(), Some(0));
    let mut world = world;
    assert!(matches!(
        history.seek(&mut world, 55),
        Err(EcsHistoryError::MissingFrame(55))
    ));
    assert_eq!(history.cursor(), Some(0));
    assert!(history.undo(&mut world).unwrap().is_none());
    assert!(history.redo(&mut world).unwrap().is_none());
    history.clear();
    assert!(history.is_empty());
    assert_eq!(history.capacity(), None);
    assert_eq!(history.cursor(), None);
}
