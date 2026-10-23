//! External-consumer journal contracts, independent of query internals.

use prism_ecs::archetype::{ColumnChunkChange, COLUMN_CHANGE_LOG_CAPACITY};
use prism_ecs::prelude::*;

#[derive(Component, Clone)]
struct Value(u32);

#[derive(Component)]
struct Other(u32);

#[test]
fn component_subscription_skips_unrelated_overflow_and_preserves_raw_work_counts() {
    let mut world = World::new();
    let entity = world.spawn((Value(0), Other(0)));
    let component = world.components().id_of::<Value>().unwrap();
    let initial = world.archetypes().column_chunks_since(None, component);
    assert!(initial.gap);
    assert_eq!(initial.records_scanned, 0);
    assert_eq!(initial.iter().len(), 0);

    // Overflow only another component's history after a complete baseline.
    for write in 0..COLUMN_CHANGE_LOG_CAPACITY * 2 {
        world.get_mut::<Other>(entity).unwrap().0 = write as u32;
    }
    let quiet = world
        .archetypes()
        .column_chunks_since(Some(initial.cursor), component);
    assert!(!quiet.gap);
    assert_eq!(quiet.records_scanned, 0);
    assert_eq!(quiet.iter().len(), 0);

    for value in 1..=17 {
        world.get_mut::<Value>(entity).unwrap().0 = value;
    }
    let changes = world
        .archetypes()
        .column_chunks_since(Some(quiet.cursor), component);
    assert!(!changes.gap);
    assert_eq!(changes.records_scanned, 17);
    assert_eq!(
        changes.iter().collect::<Vec<_>>(),
        vec![ColumnChunkChange {
            archetype: world.entities().location(entity).unwrap().archetype_id,
            chunk: Some(0),
            event_count: 17,
        }]
    );
    // Reads never acknowledge or destroy another consumer's pending writes.
    let retry = world
        .archetypes()
        .column_chunks_since(Some(quiet.cursor), component);
    assert_eq!(retry.cursor, changes.cursor);
    assert_eq!(
        retry.iter().collect::<Vec<_>>(),
        changes.iter().collect::<Vec<_>>()
    );
}

#[test]
fn retained_unrelated_noise_does_not_increase_component_scan_work() {
    let mut world = World::new();
    let entity = world.spawn((Value(0), Other(0)));
    let value_id = world.components().id_of::<Value>().unwrap();
    let cursor = world
        .archetypes()
        .column_chunks_since(None, value_id)
        .cursor;

    // Stay strictly within the global retention window: this measures index
    // selectivity, not the unrelated-overflow fallback behavior above.
    for value in 0..6_000 {
        world.get_mut::<Other>(entity).unwrap().0 = value;
    }
    for _ in 0..3 {
        world.get_mut::<Value>(entity).unwrap().0 += 1;
    }

    let changes = world
        .archetypes()
        .column_chunks_since(Some(cursor), value_id);
    assert!(!changes.gap);
    assert_eq!(changes.records_scanned, 3);
    assert_eq!(
        changes.iter().collect::<Vec<_>>(),
        vec![ColumnChunkChange {
            archetype: world.entities().location(entity).unwrap().archetype_id,
            chunk: Some(0),
            event_count: 3,
        }]
    );
}

#[test]
fn lost_selected_or_structural_history_requires_baseline_not_partial_delta() {
    for structural in [false, true] {
        let mut world = World::new();
        let entity = world.spawn((Value(0), Other(0)));
        let component = world.components().id_of::<Value>().unwrap();
        let baseline = world.archetypes().column_chunks_since(None, component);
        if structural {
            let transient = world.spawn(Value(5));
            assert!(world.despawn(transient));
        } else {
            world.get_mut::<Value>(entity).unwrap().0 = 5;
        }
        for write in 0..COLUMN_CHANGE_LOG_CAPACITY {
            world.get_mut::<Other>(entity).unwrap().0 = write as u32;
        }
        let gap = world
            .archetypes()
            .column_chunks_since(Some(baseline.cursor), component);
        assert!(gap.gap);
        assert_eq!(gap.records_scanned, 0);
        assert_eq!(gap.iter().len(), 0);
        // Once the consumer has rebuilt, the returned position can resume.
        world.get_mut::<Value>(entity).unwrap().0 = 6;
        let resumed = world
            .archetypes()
            .column_chunks_since(Some(gap.cursor), component);
        assert!(!resumed.gap);
        assert_eq!(resumed.records_scanned, 1);
        assert_eq!(resumed.iter().len(), 1);
    }
}

#[test]
fn same_tick_and_backdated_writes_are_independent_of_tick_windows() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let component = world.components().id_of::<Value>().unwrap();
    let baseline = world.archetypes().column_chunks_since(None, component);
    world.get_mut::<Value>(entity).unwrap().0 = 1;
    let first = world
        .archetypes()
        .column_chunks_since(Some(baseline.cursor), component);
    assert!(!first.gap);
    assert_eq!(first.records_scanned, 1);
    world.get_mut::<Value>(entity).unwrap().0 = 2;
    let second = world
        .archetypes()
        .column_chunks_since(Some(first.cursor), component);
    assert!(!second.gap);
    assert_eq!(second.records_scanned, 1);
    assert_ne!(first.cursor, second.cursor);
    let now = world.increment_change_tick();
    let historical = Tick::new(u32::MAX);
    {
        let mut value = world.get_mut::<Value>(entity).unwrap();
        value.bypass_change_detection().0 = 3;
        value.set_last_changed(historical);
        assert_eq!(value.changed_tick(), historical);
    }
    let backdated = world
        .archetypes()
        .column_chunks_since(Some(second.cursor), component);
    assert!(!backdated.gap);
    assert_eq!(backdated.records_scanned, 1);
    assert_eq!(world.change_tick(), now);
    assert_eq!(
        world.get_ticks::<Value>(entity).unwrap().changed,
        historical
    );
    assert_eq!(world.get::<Value>(entity).unwrap().0, 3);
}

#[test]
fn foreign_registry_cursor_requires_fresh_baseline() {
    let mut first = World::new();
    first.spawn(Value(0));
    let component = first.components().id_of::<Value>().unwrap();
    let old = first.archetypes().column_chunks_since(None, component);
    let mut second = World::new();
    second.spawn(Value(1));
    let component = second.components().id_of::<Value>().unwrap();
    let changes = second
        .archetypes()
        .column_chunks_since(Some(old.cursor), component);
    assert!(changes.gap);
    assert_eq!(changes.iter().len(), 0);
    assert_ne!(old.cursor, changes.cursor);
}

#[test]
fn filtered_component_stream_matches_full_journal_model_for_mixed_world_edits() {
    let mut world = World::new();
    let entities: Vec<_> = (0..12)
        .map(|value| world.spawn((Value(value), Other(value))))
        .collect();
    let value_id = world.components().id_of::<Value>().unwrap();
    let other_id = world.components().id_of::<Other>().unwrap();
    let full_baseline = world.archetypes().column_changes_since(None);
    let filtered_baseline = world.archetypes().column_chunks_since(None, value_id);
    assert!(full_baseline.gap && filtered_baseline.gap);

    // Mix same-chunk repetitions, scattered writes, unrelated writes, and
    // structural creation/destruction. The full journal provides an oracle
    // for exact selected revision order and consecutive-run compression.
    let transient = world.spawn((Value(100), Other(100)));
    for (round, &entity) in entities.iter().enumerate() {
        if round % 3 == 0 {
            for value in 0..4 {
                world.get_mut::<Value>(entity).unwrap().0 += value;
            }
        } else if round % 3 == 1 {
            world.get_mut::<Other>(entity).unwrap().0 += round as u32;
        } else {
            world.get_mut::<Value>(entity).unwrap().0 += round as u32;
            world.get_mut::<Other>(entity).unwrap().0 += 1;
        }
    }
    assert!(world.despawn(transient));
    let later = world.spawn(Value(200));
    world.get_mut::<Value>(later).unwrap().0 += 1;

    let full = world
        .archetypes()
        .column_changes_since(Some(full_baseline.cursor));
    let filtered = world
        .archetypes()
        .column_chunks_since(Some(filtered_baseline.cursor), value_id);
    assert!(!full.gap && !filtered.gap);

    let selected: Vec<_> = full
        .events
        .iter()
        .copied()
        .filter(|event| {
            event
                .component
                .is_none_or(|component| component == value_id)
        })
        .collect();
    let mut expected: Vec<ColumnChunkChange> = Vec::new();
    for event in selected.iter().copied() {
        if event.chunk.is_some()
            && let Some(previous) = expected.last_mut()
            && previous.archetype == event.archetype
            && previous.chunk == event.chunk
        {
            previous.event_count += 1;
        } else {
            expected.push(ColumnChunkChange {
                archetype: event.archetype,
                chunk: event.chunk,
                event_count: 1,
            });
        }
    }
    assert_eq!(filtered.records_scanned, selected.len());
    assert_eq!(filtered.iter().collect::<Vec<_>>(), expected);

    // A non-selected component subscription sees no component writes from the
    // same revision range, but still observes structural edits.
    let unrelated_baseline = world
        .archetypes()
        .column_chunks_since(Some(filtered_baseline.cursor), other_id);
    assert!(!unrelated_baseline.gap);
    assert!(unrelated_baseline.records_scanned > 0);
}
