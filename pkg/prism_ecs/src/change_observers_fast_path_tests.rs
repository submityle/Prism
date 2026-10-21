//! Correctness and operation-count regressions for unchanged flushes.

use super::*;
use crate::query::DirtyChunk;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Clone)]
struct Value(u32);
impl Component for Value {}

struct Sparse(u32);
impl Component for Sparse {
    const STORAGE: crate::component::StorageType = crate::component::StorageType::SparseSet;
}

#[derive(Clone)]
struct Unobserved(u32);
impl Component for Unobserved {}

#[derive(Clone)]
struct Noise;
impl Component for Noise {}

#[derive(Clone)]
struct ExtraA;
impl Component for ExtraA {}

#[derive(Clone)]
struct ExtraB;
impl Component for ExtraB {}

struct GapNoise<const N: usize>;
impl<const N: usize> Component for GapNoise<N> {}

fn populate_unrelated_gap_archetypes(world: &mut World) {
    // Every subset is a distinct archetype, none containing Value or Sparse.
    for mask in 0..64 {
        for _ in 0..16 {
            let entity = world.spawn(Noise);
            if mask & 1 != 0 {
                world.insert(entity, GapNoise::<0>);
            }
            if mask & 2 != 0 {
                world.insert(entity, GapNoise::<1>);
            }
            if mask & 4 != 0 {
                world.insert(entity, GapNoise::<2>);
            }
            if mask & 8 != 0 {
                world.insert(entity, GapNoise::<3>);
            }
            if mask & 16 != 0 {
                world.insert(entity, GapNoise::<4>);
            }
            if mask & 32 != 0 {
                world.insert(entity, GapNoise::<5>);
            }
        }
    }
}

fn count(counter: &Arc<AtomicUsize>) -> impl FnMut(&mut ChangeContext<'_>) + Send + Sync + 'static {
    let counter = counter.clone();
    move |ctx| {
        counter.fetch_add(ctx.len(), Ordering::Relaxed);
    }
}

fn assert_idle(world: &mut World) {
    let tick = world.change_tick();
    assert_eq!(world.flush_changed_observers(), 0);
    assert_eq!(world.change_tick().get(), tick.get().wrapping_add(1));
    assert_eq!(
        world.change_observers.stats,
        FlushStats {
            revision_checks: 1,
            idle_fast_path: true,
            ..FlushStats::default()
        }
    );
}

fn force_change_index_gap(
    world: &mut World,
    component: crate::component::ComponentId,
    entity: Entity,
) {
    let tick = world.change_tick();
    for _ in 0..=crate::change_index::MAX_RETAINED_EVENTS {
        world.change_tracker.record(component, entity, tick);
    }
}

#[test]
fn idle_cost_does_not_visit_active_or_removed_historical_slots() {
    for slots in [1, 32, 1_024] {
        let mut world = World::new();
        let calls = Arc::new(AtomicUsize::new(0));
        for _ in 0..slots {
            let id = world.observe_changed::<Value, _>(count(&calls));
            assert!(world.remove_change_observer(id));
        }
        for _ in 0..slots {
            world.observe_changed::<Value, _>(count(&calls));
        }
        assert_eq!(world.flush_changed_observers(), 0);
        assert_eq!(world.change_observers.stats.collections, 0);
        assert_eq!(world.change_observers.stats.candidate_slots, 0);
        assert_eq!(world.change_observers.stats.slot_visits, 0);
        for _ in 0..4 {
            assert_idle(&mut world);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn nonempty_flush_visits_only_active_candidates_after_heavy_churn() {
    let mut world = World::new();
    let calls = Arc::new(AtomicUsize::new(0));
    for _ in 0..2_048 {
        let id = world.observe_changed::<Value, _>(count(&calls));
        assert!(world.remove_change_observer(id));
    }
    let entity = world.spawn((Value(0), Sparse(0)));
    world.observe_changed::<Value, _>(count(&calls));
    world.observe_changed::<Sparse, _>(count(&calls));
    assert_eq!(world.flush_changed_observers(), 0);

    world.get_mut::<Value>(entity).unwrap().0 += 1;
    world.get_mut::<Sparse>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(world.change_observers.stats.candidate_slots, 2);
    assert_eq!(world.change_observers.stats.slot_visits, 4);
    assert_eq!(world.change_observers.stats.collections, 2);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn repeated_observer_churn_reuses_slots_instead_of_growing_registry() {
    let mut world = World::new();
    let calls = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&calls));

    for _ in 0..10_000 {
        let id = world.observe_changed::<Value, _>(count(&calls));
        assert!(world.remove_change_observer(id));
    }

    assert_eq!(world.change_observers.active, 1);
    assert_eq!(world.change_observers.slots.len(), 2);
    assert_eq!(world.change_observers.free_indices.len(), 1);
}

#[test]
fn observer_rebuilds_candidates_after_a_log_gap() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let calls = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&calls));
    assert_eq!(world.flush_changed_observers(), 0);

    world.get_mut::<Value>(entity).unwrap().0 += 1;
    let component = world.components().id_of::<Value>().unwrap();
    force_change_index_gap(&mut world, component, entity);

    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(world.flush_changed_observers(), 0);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn table_gap_fallback_rebuilds_only_live_changed_component_rows() {
    let mut world = World::new();
    populate_unrelated_gap_archetypes(&mut world);
    let value_only = world.spawn(Value(0));
    let value_with_noise = world.spawn((Value(0), Noise));
    let noise_only = world.spawn(Noise);
    let calls = Arc::new(AtomicUsize::new(0));
    let value_only_seen = Arc::new(AtomicBool::new(false));
    let value_with_noise_seen = Arc::new(AtomicBool::new(false));
    let noise_only_seen = Arc::new(AtomicBool::new(false));

    world.observe_changed::<Value, _>({
        let calls = calls.clone();
        let value_only_seen = value_only_seen.clone();
        let value_with_noise_seen = value_with_noise_seen.clone();
        let noise_only_seen = noise_only_seen.clone();
        move |ctx| {
            calls.fetch_add(ctx.len(), Ordering::Relaxed);
            for &entity in ctx.changed() {
                if entity == value_only {
                    value_only_seen.store(true, Ordering::Relaxed);
                }
                if entity == value_with_noise {
                    value_with_noise_seen.store(true, Ordering::Relaxed);
                }
                if entity == noise_only {
                    noise_only_seen.store(true, Ordering::Relaxed);
                }
            }
        }
    });
    assert_eq!(world.flush_changed_observers(), 0);

    world.get_mut::<Value>(value_only).unwrap().0 += 1;
    world.get_mut::<Value>(value_with_noise).unwrap().0 += 1;
    let component = world.components().id_of::<Value>().unwrap();
    force_change_index_gap(&mut world, component, value_only);

    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert!(value_only_seen.load(Ordering::Relaxed));
    assert!(value_with_noise_seen.load(Ordering::Relaxed));
    assert!(!noise_only_seen.load(Ordering::Relaxed));
    assert_eq!(world.change_observers.stats.gap_archetypes_scanned, 2);
    assert_eq!(world.change_observers.stats.gap_rows_checked, 2);
    assert_eq!(world.change_observers.stats.gap_sparse_entities_checked, 0);
    assert_idle(&mut world);
}

#[test]
fn sparse_gap_fallback_rebuilds_only_sparse_set_members() {
    let mut world = World::new();
    populate_unrelated_gap_archetypes(&mut world);
    let changed = world.spawn(Sparse(0));
    let unchanged = world.spawn((Sparse(0), Noise));
    let unrelated = world.spawn((Value(0), Noise));
    let calls = Arc::new(AtomicUsize::new(0));
    let changed_seen = Arc::new(AtomicBool::new(false));
    let unchanged_seen = Arc::new(AtomicBool::new(false));
    let unrelated_seen = Arc::new(AtomicBool::new(false));

    world.observe_changed::<Sparse, _>({
        let calls = calls.clone();
        let changed_seen = changed_seen.clone();
        let unchanged_seen = unchanged_seen.clone();
        let unrelated_seen = unrelated_seen.clone();
        move |ctx| {
            calls.fetch_add(ctx.len(), Ordering::Relaxed);
            for &entity in ctx.changed() {
                if entity == changed {
                    changed_seen.store(true, Ordering::Relaxed);
                }
                if entity == unchanged {
                    unchanged_seen.store(true, Ordering::Relaxed);
                }
                if entity == unrelated {
                    unrelated_seen.store(true, Ordering::Relaxed);
                }
            }
        }
    });
    assert_eq!(world.flush_changed_observers(), 0);

    world.get_mut::<Sparse>(changed).unwrap().0 += 1;
    let component = world.components().id_of::<Sparse>().unwrap();
    force_change_index_gap(&mut world, component, changed);

    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(changed_seen.load(Ordering::Relaxed));
    assert!(!unchanged_seen.load(Ordering::Relaxed));
    assert!(!unrelated_seen.load(Ordering::Relaxed));
    assert_eq!(world.change_observers.stats.gap_archetypes_scanned, 0);
    assert_eq!(world.change_observers.stats.gap_rows_checked, 0);
    assert_eq!(world.change_observers.stats.gap_sparse_entities_checked, 2);
    assert_idle(&mut world);
}

#[test]
fn gap_fallback_does_not_replay_despawned_generation() {
    let mut world = World::new();
    let old = world.spawn(Value(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let old_seen = Arc::new(AtomicBool::new(false));
    let replacement_seen = Arc::new(AtomicBool::new(false));

    world.observe_changed::<Value, _>({
        let calls = calls.clone();
        let old_seen = old_seen.clone();
        let replacement_seen = replacement_seen.clone();
        move |ctx| {
            calls.fetch_add(ctx.len(), Ordering::Relaxed);
            for &entity in ctx.changed() {
                if entity == old {
                    old_seen.store(true, Ordering::Relaxed);
                }
                if entity.index() == old.index() && entity != old {
                    replacement_seen.store(true, Ordering::Relaxed);
                }
            }
        }
    });
    assert_eq!(world.flush_changed_observers(), 0);

    world.get_mut::<Value>(old).unwrap().0 += 1;
    assert!(world.despawn(old));
    let replacement = world.spawn(Value(2));
    let component = world.components().id_of::<Value>().unwrap();
    force_change_index_gap(&mut world, component, old);

    // A stale log record for `old` may survive until the gap, but the
    // authoritative table scan must only return the live replacement.
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(!old_seen.load(Ordering::Relaxed));
    assert!(replacement_seen.load(Ordering::Relaxed));
    assert_eq!(old.index(), replacement.index());
    assert_ne!(old, replacement);
}

#[cfg(feature = "std")]
#[test]
fn sparse_registry_compaction_preserves_order_and_pending_delivery() {
    use std::sync::Mutex;

    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let order = Arc::new(Mutex::new(Vec::new()));
    let ids: Vec<_> = (0..256)
        .map(|label| {
            let order = order.clone();
            world.observe_changed::<Value, _>(move |_| order.lock().unwrap().push(label))
        })
        .collect();
    assert_eq!(world.flush_changed_observers(), 0);

    for id in ids.iter().take(240) {
        assert!(world.remove_change_observer(*id));
    }
    assert_eq!(world.change_observers.active, 16);
    assert!(
        world.change_observers.slots.len() <= world.change_observers.active.saturating_mul(2) + 64
    );

    world.get_mut::<Value>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 16);
    assert_eq!(*order.lock().unwrap(), (240..256).collect::<Vec<_>>());
}

#[test]
fn nonempty_flush_visits_only_observers_for_dirty_components() {
    let mut world = World::new();
    let entity = world.spawn((Value(0), Sparse(0)));
    let value_calls = Arc::new(AtomicUsize::new(0));
    let sparse_calls = Arc::new(AtomicUsize::new(0));
    for _ in 0..1_024 {
        world.observe_changed::<Value, _>(count(&value_calls));
    }
    world.observe_changed::<Sparse, _>(count(&sparse_calls));
    assert_eq!(world.flush_changed_observers(), 0);

    world.get_mut::<Sparse>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(world.change_observers.stats.candidate_slots, 1);
    assert_eq!(world.change_observers.stats.slot_visits, 2);
    assert_eq!(world.change_observers.stats.collections, 1);
    assert_eq!(value_calls.load(Ordering::Relaxed), 0);
    assert_eq!(sparse_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn observer_to_reaction_sparse_flush_ignores_unaffected_registry_entries() {
    const ENTITIES: usize = 4_096;
    const QUERY_REGISTRATIONS: usize = 64;
    const NOISE_OBSERVERS: usize = 512;
    const CLEAN_REACTIONS: usize = 2_048;

    let mut world = World::new();
    let entities: Vec<_> = (0..ENTITIES)
        .map(|index| world.spawn((Value(index as u32), Unobserved(index as u32), Noise)))
        .collect();

    // Populate many reusable query states and verify their initial full-set
    // view. The measured Observer -> Reaction push path below does not execute
    // these queries, so this is not evidence for the full Query -> Observer
    // chain's asymptotic cost.
    let queries: Vec<_> = (0..QUERY_REGISTRATIONS)
        .map(|_| world.query::<&Value>())
        .collect();
    for query in &queries {
        assert_eq!(query.iter(&world).count(), ENTITIES);
    }

    let noise_calls = Arc::new(AtomicUsize::new(0));
    for _ in 0..NOISE_OBSERVERS {
        world.observe_changed::<Noise, _>(count(&noise_calls));
    }

    let source = world.reaction_source::<Value>();
    let effect_calls = Arc::new(AtomicUsize::new(0));
    world.add_reaction(&[source], {
        let effect_calls = effect_calls.clone();
        move |_| {
            effect_calls.fetch_add(1, Ordering::Relaxed);
        }
    });
    for _ in 0..CLEAN_REACTIONS {
        world.add_reaction(&[], |_| {});
    }

    // Establish a clean revision after registration. Neither entity count nor
    // clean registry cardinality should enter the subsequent sparse flush.
    assert_eq!(world.flush_changed_observers(), 0);
    assert_idle(&mut world);

    // A write to a component with no active subscription must retain the
    // observer tracker's idle fast path, independently of the dirty-query
    // coverage below.
    world
        .get_mut::<Unobserved>(entities[ENTITIES / 2])
        .unwrap()
        .0 += 1;
    assert_eq!(world.flush_changed_observers(), 0);
    assert!(world.change_observers.stats.idle_fast_path);
    assert_eq!(world.change_observers.stats.slot_visits, 0);
    assert_eq!(world.change_observers.stats.candidate_slots, 0);

    // The first query pass establishes its journal cursor after setup. Its
    // conservative fallback must see a clean same-tick window.
    let value_query = world.query::<&Value>();
    let baseline_tick = world.change_tick();
    let baseline = value_query.dirty_chunks_with_stats(&world, baseline_tick, baseline_tick);
    assert!(baseline.chunks.is_empty());
    assert!(baseline.stats.journal_gap);

    // In one forward tick, mutate one unobserved row and one Value row. The
    // dirty-query journal path must discard the unrelated component event and
    // report only Value's touched table chunk before observers are flushed.
    let last_run = world.change_tick();
    world.increment_change_tick();
    let this_run = world.change_tick();
    world
        .get_mut::<Unobserved>(entities[ENTITIES / 2])
        .unwrap()
        .0 += 1;
    let value_entity = entities[ENTITIES - 1];
    let value_location = world.entities().location(value_entity).unwrap();
    world.get_mut::<Value>(value_entity).unwrap().0 += 1;

    let value_archetype = world.archetypes().get(value_location.archetype_id).unwrap();
    let rows_per_chunk = value_archetype.table().rows_per_chunk();
    let chunk_start = value_location.row as usize / rows_per_chunk * rows_per_chunk;
    let chunk_end = (chunk_start + rows_per_chunk).min(value_archetype.table().len());
    let dirty = value_query.dirty_chunks_with_stats(&world, last_run, this_run);
    assert_eq!(
        dirty.chunks,
        [DirtyChunk {
            archetype: value_location.archetype_id,
            start: chunk_start,
            end: chunk_end,
        }]
    );
    assert!(!dirty.stats.journal_gap);
    assert_eq!(dirty.stats.journal_events, 1);
    assert_eq!(dirty.stats.journal_records_scanned, 1);
    assert_eq!(dirty.stats.candidate_archetype_checks, 1);
    assert_eq!(dirty.stats.merged_candidates, 1);
    assert_eq!(dirty.stats.summaries_tested, 0);
    assert_eq!(dirty.stats.column_candidates, 0);

    // The same sparse pair of writes then flows through only the Value source
    // observer and its affected reaction, despite the large clean registries.
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(effect_calls.load(Ordering::Relaxed), 1);
    assert_eq!(noise_calls.load(Ordering::Relaxed), 0);
    assert_eq!(world.change_observers.stats.candidate_slots, 1);
    assert_eq!(world.change_observers.stats.slot_visits, 2);
    assert_eq!(world.change_observers.stats.collections, 1);
    assert_eq!(world.change_observers.stats.log_batches, 1);
    assert_eq!(world.change_observers.stats.log_events, 1);
    assert_eq!(
        world.change_observers.stats.retain_tick_lookup_upper_bound,
        1
    );
    let reaction_stats = world.reactions.reaction_stats();
    assert_eq!(reaction_stats.ordering_nodes, 2);
    assert_eq!(reaction_stats.ordering_edges, 2);
    assert_eq!(reaction_stats.executed_nodes, 1);
}

fn sparse_pipeline_work_counts(
    entity_count: usize,
    noise_observers: usize,
    clean_reactions: usize,
    unrelated_writes: usize,
    grow_archetypes: bool,
) -> [usize; 14] {
    let mut world = World::new();
    let mut entities = Vec::with_capacity(entity_count);
    for index in 0..entity_count {
        let value = Value(index as u32);
        let unobserved = Unobserved(index as u32);
        let entity = if grow_archetypes {
            match index % 4 {
                0 => world.spawn((value, unobserved, Noise)),
                1 => world.spawn((value, unobserved, Noise, ExtraA)),
                2 => world.spawn((value, unobserved, Noise, ExtraB)),
                _ => world.spawn((value, unobserved, Noise, ExtraA, ExtraB)),
            }
        } else {
            world.spawn((value, unobserved, Noise))
        };
        entities.push(entity);
    }

    let noise_calls = Arc::new(AtomicUsize::new(0));
    for _ in 0..noise_observers {
        world.observe_changed::<Noise, _>(count(&noise_calls));
    }
    let source = world.reaction_source::<Value>();
    let effect_calls = Arc::new(AtomicUsize::new(0));
    world.add_reaction(&[source], {
        let effect_calls = effect_calls.clone();
        move |_| {
            effect_calls.fetch_add(1, Ordering::Relaxed);
        }
    });
    for _ in 0..clean_reactions {
        world.add_reaction(&[], |_| {});
    }
    assert_eq!(world.flush_changed_observers(), 0);

    let query = world.query::<&Value>();
    let baseline_tick = world.change_tick();
    assert!(query
        .dirty_chunks_with_stats(&world, baseline_tick, baseline_tick)
        .chunks
        .is_empty());

    let last_run = world.change_tick();
    world.increment_change_tick();
    let this_run = world.change_tick();
    assert!(unrelated_writes < entity_count);
    for &entity in entities.iter().take(unrelated_writes) {
        world.get_mut::<Unobserved>(entity).unwrap().0 += 1;
    }
    world.get_mut::<Value>(*entities.last().unwrap()).unwrap().0 += 1;

    let dirty = query.dirty_chunks_with_stats(&world, last_run, this_run);
    assert!(!dirty.stats.journal_gap);
    assert_eq!(dirty.chunks.len(), 1);
    assert_eq!(dirty.stats.candidate_archetype_checks, 1);
    assert_eq!(dirty.stats.merged_candidates, 1);
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(effect_calls.load(Ordering::Relaxed), 1);
    assert_eq!(noise_calls.load(Ordering::Relaxed), 0);

    let observer = world.change_observers.stats;
    let reaction = world.reactions.reaction_stats();
    [
        dirty.chunks.len(),
        dirty.stats.journal_events,
        dirty.stats.journal_records_scanned,
        dirty.stats.candidate_archetype_checks,
        dirty.stats.merged_candidates,
        dirty.stats.summaries_tested,
        dirty.stats.column_candidates,
        observer.candidate_slots,
        observer.slot_visits,
        observer.collections,
        reaction.ordering_nodes,
        reaction.ordering_edges,
        reaction.executed_nodes,
        world.archetypes().len(),
    ]
}

#[test]
fn sparse_pipeline_work_counts_ignore_clean_registry_and_entity_growth() {
    // Increase entities, matching archetypes, unrelated observer slots,
    // clean reaction nodes, and unrelated journal writes while keeping the
    // relevant write fixed at one.
    let small = sparse_pipeline_work_counts(256, 16, 64, 1, false);
    let large = sparse_pipeline_work_counts(8_192, 512, 2_048, 4_096, true);
    assert_eq!(
        &large[..13],
        &small[..13],
        "clean scale must not add sparse-path work"
    );
    assert!(large[13] > small[13], "the large case adds archetypes");
    assert_eq!(large[0], 1, "one touched table chunk is reported");
    assert_eq!(large[1], 1, "one relevant journal event is consumed");
    assert_eq!(large[2], 1, "only the relevant journal record is scanned");
    assert_eq!(&large[7..10], &[1, 2, 1]);
    assert_eq!(&large[10..13], &[2, 2, 1]);
}

#[test]
fn immediate_table_and_sparse_writes_after_idle_are_not_skipped() {
    let mut world = World::new();
    let entity = world.spawn((Value(0), Sparse(0)));
    let calls = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&calls));
    world.observe_changed::<Sparse, _>(count(&calls));
    assert_eq!(world.flush_changed_observers(), 0);
    assert_idle(&mut world);
    world.get_mut::<Value>(entity).unwrap().0 += 1;
    world.get_mut::<Sparse>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_idle(&mut world);
}

#[test]
fn registration_after_idle_keeps_its_own_baseline() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let old = Arc::new(AtomicUsize::new(0));
    let new = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&old));
    assert_eq!(world.flush_changed_observers(), 0);
    assert_idle(&mut world);
    world.get_mut::<Value>(entity).unwrap().0 = 1;
    world.observe_changed::<Value, _>(count(&new));
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(old.load(Ordering::Relaxed), 1);
    assert_eq!(new.load(Ordering::Relaxed), 0);
    assert_idle(&mut world);
    world.get_mut::<Value>(entity).unwrap().0 = 2;
    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(new.load(Ordering::Relaxed), 1);
}

#[test]
fn callback_writes_invalidate_clean_revision_for_next_flush() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let calls = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>({
        let calls = calls.clone();
        move |ctx| {
            if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                ctx.world.get_mut::<Value>(entity).unwrap().0 += 1;
            }
        }
    });
    world.get_mut::<Value>(entity).unwrap().0 = 1;
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_idle(&mut world);
}

#[test]
fn idle_flush_still_settles_reactions_and_keeps_their_writes_pending() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let calls = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&calls));
    let source = world.reaction_source::<Sparse>();
    world.add_reaction(&[source], move |world| {
        world.get_mut::<Value>(entity).unwrap().0 += 1;
    });
    world.flush_changed_observers();
    // Settle any initial effect and establish an exact clean baseline.
    world.flush_changed_observers();
    let before = calls.load(Ordering::Relaxed);
    world.mark_reaction_dirty(source);
    assert_eq!(world.flush_changed_observers(), 0);
    assert!(world.change_observers.stats.idle_fast_path);
    assert_eq!(world.change_observers.stats.slot_visits, 0);
    assert_eq!(world.change_observers.stats.collections, 0);
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), before + 1);
    assert_idle(&mut world);
}

#[test]
fn restore_invalidates_idle_revision_and_accepts_same_tick_write() {
    let mut world = World::new();
    world.register_snapshot_component::<Value>();
    let entity = world.spawn(Value(0));
    let calls = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&calls));
    let snapshot = world.snapshot();
    world.flush_changed_observers();
    assert_idle(&mut world);
    world.get_mut::<Value>(entity).unwrap().0 = 10;
    world.restore(&snapshot);
    assert_eq!(world.flush_changed_observers(), 0);
    assert_idle(&mut world);
    world.restore(&snapshot);
    let tick = world.change_tick();
    world.get_mut::<Value>(entity).unwrap().0 = 1;
    assert_eq!(world.change_tick(), tick);
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_idle(&mut world);
}

#[test]
fn removal_despawn_and_generation_reuse_do_not_replay_stale_changes() {
    let mut world = World::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let id = world.observe_changed::<Value, _>(count(&calls));
    let entity = world.spawn(Value(0));
    world.flush_changed_observers();
    assert_idle(&mut world);
    world.get_mut::<Value>(entity).unwrap().0 = 1;
    assert!(world.despawn(entity));
    assert_eq!(world.flush_changed_observers(), 0);
    assert_idle(&mut world);
    let replacement = world.spawn(Value(2));
    assert_eq!(world.flush_changed_observers(), 1);
    assert_ne!(entity, replacement);
    assert!(world.remove_change_observer(id));
    let after = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&after));
    assert_eq!(world.flush_changed_observers(), 0);
    assert_idle(&mut world);
    world.get_mut::<Value>(replacement).unwrap().0 = 3;
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(after.load(Ordering::Relaxed), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[cfg(feature = "std")]
#[test]
fn active_index_preserves_order_across_remove_and_readd() {
    use std::sync::Mutex;

    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let order = Arc::new(Mutex::new(Vec::new()));
    let observe = |label, order: &Arc<Mutex<Vec<u32>>>| {
        let order = order.clone();
        move |_: &mut ChangeContext<'_>| order.lock().unwrap().push(label)
    };

    world.observe_changed::<Value, _>(observe(1, &order));
    let removed = world.observe_changed::<Value, _>(observe(2, &order));
    world.observe_changed::<Value, _>(observe(3, &order));
    assert_eq!(world.flush_changed_observers(), 0);
    assert!(world.remove_change_observer(removed));
    let readded = world.observe_changed::<Value, _>(observe(4, &order));

    world.get_mut::<Value>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_eq!(*order.lock().unwrap(), [1, 3, 4]);
    assert!(world.remove_change_observer(readded));
}

#[cfg(feature = "std")]
#[test]
fn callback_churn_then_panic_keeps_candidates_and_deferred_slots_valid() {
    use std::sync::{Mutex, OnceLock};

    let mut world = World::new();
    let entity = world.spawn(Value(0));
    for _ in 0..4 {
        let id = world.observe_changed::<Value, _>(|_| {});
        assert!(world.remove_change_observer(id));
    }

    let order = Arc::new(Mutex::new(Vec::new()));
    let future_id = Arc::new(OnceLock::<ObserverId>::new());
    let panic_once = Arc::new(core::sync::atomic::AtomicBool::new(true));
    let first_order = order.clone();
    let future_id_in_first = future_id.clone();
    let panic_once_in_first = panic_once.clone();
    world.observe_changed::<Value, _>(move |ctx| {
        first_order.lock().unwrap().push(1);
        if panic_once_in_first.swap(false, Ordering::Relaxed) {
            assert!(ctx
                .world
                .remove_change_observer(*future_id_in_first.get().unwrap()));
            let replacement_order = first_order.clone();
            ctx.world.observe_changed::<Value, _>(move |_| {
                replacement_order.lock().unwrap().push(4);
            });
            panic!("exercise deferred slot reuse during unwind");
        }
    });
    let removed = world.observe_changed::<Value, _>({
        let order = order.clone();
        move |_| order.lock().unwrap().push(2)
    });
    assert!(future_id.set(removed).is_ok());
    world.observe_changed::<Value, _>({
        let order = order.clone();
        move |_| order.lock().unwrap().push(3)
    });
    assert_eq!(world.flush_changed_observers(), 0);

    world.get_mut::<Value>(entity).unwrap().0 += 1;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.flush_changed_observers();
    }))
    .is_err());
    assert_eq!(*order.lock().unwrap(), [1]);

    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(*order.lock().unwrap(), [1, 1, 3]);

    world.get_mut::<Value>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_eq!(*order.lock().unwrap(), [1, 1, 3, 1, 3, 4]);
}

#[cfg(feature = "std")]
#[test]
fn callback_registration_and_removal_do_not_change_current_candidates() {
    use std::sync::Mutex;

    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let order = Arc::new(Mutex::new(Vec::new()));
    let removed_id = Arc::new(std::sync::OnceLock::<ObserverId>::new());
    let registered = Arc::new(core::sync::atomic::AtomicBool::new(false));

    let order_a = order.clone();
    let removed_id_a = removed_id.clone();
    let registered_a = registered.clone();
    world.observe_changed::<Value, _>(move |ctx| {
        order_a.lock().unwrap().push(1);
        if !registered_a.swap(true, Ordering::Relaxed) {
            let id = *removed_id_a.get().expect("later observer id");
            assert!(ctx.world.remove_change_observer(id));
            let order_d = order_a.clone();
            ctx.world.observe_changed::<Value, _>(move |_| {
                order_d.lock().unwrap().push(4);
            });
        }
    });
    let removed = world.observe_changed::<Value, _>({
        let order = order.clone();
        move |_| order.lock().unwrap().push(2)
    });
    assert!(removed_id.set(removed).is_ok());
    world.observe_changed::<Value, _>({
        let order = order.clone();
        move |_| order.lock().unwrap().push(3)
    });
    assert_eq!(world.flush_changed_observers(), 0);

    world.get_mut::<Value>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(*order.lock().unwrap(), [1, 3]);

    world.get_mut::<Value>(entity).unwrap().0 += 1;
    assert_eq!(world.flush_changed_observers(), 3);
    assert_eq!(*order.lock().unwrap(), [1, 3, 1, 3, 4]);
}

#[cfg(feature = "std")]
#[test]
fn panic_after_idle_retries_pending_without_new_writes_or_replaying_successes() {
    let mut world = World::new();
    let entity = world.spawn(Value(0));
    let success = Arc::new(AtomicUsize::new(0));
    let attempts = Arc::new(AtomicUsize::new(0));
    world.observe_changed::<Value, _>(count(&success));
    world.observe_changed::<Value, _>({
        let attempts = attempts.clone();
        move |_| {
            assert_ne!(attempts.fetch_add(1, Ordering::Relaxed), 0, "retry");
        }
    });
    world.flush_changed_observers();
    assert_idle(&mut world);
    world.get_mut::<Value>(entity).unwrap().0 = 1;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.flush_changed_observers();
    }))
    .is_err());
    assert!(!world.change_observers.flushing);
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(world.change_observers.stats.candidate_slots, 1);
    assert_eq!(success.load(Ordering::Relaxed), 1);
    assert_eq!(attempts.load(Ordering::Relaxed), 2);
    assert_idle(&mut world);
}

#[cfg(feature = "std")]
#[test]
fn world_replacement_does_not_clear_replacement_panic_retry_index() {
    let mut replacement = World::new();
    let replacement_entity = replacement.spawn(Value(0));
    let replacement_attempts = Arc::new(AtomicUsize::new(0));
    replacement.observe_changed::<Value, _>({
        let replacement_attempts = replacement_attempts.clone();
        move |_| {
            if replacement_attempts.fetch_add(1, Ordering::Relaxed) == 0 {
                panic!("replacement observer retry");
            }
        }
    });
    replacement.get_mut::<Value>(replacement_entity).unwrap().0 = 1;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        replacement.flush_changed_observers();
    }))
    .is_err());
    assert_eq!(replacement.change_observers.pending_indices, [0]);

    let replacement_id = replacement.id();
    let mut replacement = Some(replacement);
    let mut world = World::new();
    let old_entity = world.spawn(Value(0));
    world.observe_changed::<Value, _>(move |ctx| {
        *ctx.world = replacement.take().expect("old observer must run once");
    });
    world.get_mut::<Value>(old_entity).unwrap().0 = 1;
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(world.id(), replacement_id);

    // No new write occurs in the replacement World. Its old panic batch must
    // still be selected solely through the preserved pending index.
    assert_eq!(world.flush_changed_observers(), 1);
    assert_eq!(replacement_attempts.load(Ordering::Relaxed), 2);
    assert_eq!(world.flush_changed_observers(), 0);
}
