use super::*;
use alloc::vec;
use prism_ecs::component::Component;
use prism_ecs::policy::{EcsCapabilities, EcsProfile, LodBandProfile, LodProfile};

#[derive(Debug, PartialEq)]
struct Counter(u32);
impl Component for Counter {}

fn profile() -> EcsProfile {
    EcsProfile {
        lod: Some(LodProfile {
            bands: vec![
                LodBandProfile {
                    max_distance_sq: 4.0,
                    tick_period: 1,
                    quality: 0,
                },
                LodBandProfile {
                    max_distance_sq: 100.0,
                    tick_period: 4,
                    quality: 7,
                },
            ],
            dormant_beyond_last: true,
        }),
        streaming: Some(StreamingProfile {
            cell_size: 10.0,
            load_radius: 0,
            unload_radius: 1,
            max_cells_per_interest: 1,
        }),
        ..EcsProfile::default()
    }
}

fn configured() -> App {
    let mut app = App::empty();
    app.configure_ecs_policy(profile(), EcsCapabilities::default())
        .unwrap();
    app
}

fn pos(x: f64) -> WorldPos {
    WorldPos::new(x, 0.0, 0.0)
}

fn origin_load(app: &mut App) -> EcsCellLoad {
    let result = app.advance_ecs_streaming(pos(0.0), &[]).unwrap();
    assert_eq!(result.to_load.len(), 1);
    let ticket = result.to_load[0].clone();
    assert_eq!(ticket.cell(), CellCoord::ORIGIN);
    assert_eq!(ticket.world_id(), app.world().id());
    ticket
}

#[test]
fn lod_callbacks_apply_cadence_quality_and_dormancy_to_real_values() {
    let mut app = configured();
    let near = app.try_spawn_with_policy((Counter(0),)).unwrap();
    let middle = app.try_spawn_with_policy((Counter(0),)).unwrap();
    let far = app.try_spawn_with_policy((Counter(0),)).unwrap();
    let population = [
        (far, [11.0, 0.0, 0.0]),
        (middle, [10.0, 0.0, 0.0]),
        (near, [2.0, 0.0, 0.0]),
    ];
    let mut calls = Vec::new();
    let mut tick = |world: &mut World, entity: Entity, decision: LodDecision| {
        world.get_mut::<Counter>(entity).unwrap().0 += 1;
        calls.push((entity, decision.quality.0));
    };
    let first = app
        .drive_ecs_lod([0.0; 3], &population, 1, &mut tick)
        .unwrap();
    assert_eq!(first.to_tick, vec![near]);
    assert_eq!(first.slept, vec![far]);
    app.drive_ecs_lod([0.0; 3], &population, 4, &mut tick)
        .unwrap();
    assert_eq!(calls, vec![(near, 0), (near, 0), (middle, 7)]);
    assert_eq!(app.world().get::<Counter>(near), Some(&Counter(2)));
    assert_eq!(app.world().get::<Counter>(middle), Some(&Counter(1)));
    assert_eq!(app.world().get::<Counter>(far), Some(&Counter(0)));
    assert!(app.is_ecs_dormant(far).unwrap());
    let wake = app
        .drive_ecs_lod([0.0; 3], &[(far, [0.0; 3])], 5, |world, entity, _| {
            world.get_mut::<Counter>(entity).unwrap().0 += 1;
        })
        .unwrap();
    assert_eq!(wake.woken, vec![far]);
    assert!(!app.is_ecs_dormant(far).unwrap());
    assert_eq!(app.world().get::<Counter>(far), Some(&Counter(1)));
}

#[test]
fn per_entity_lod_phase_distributes_actual_callback_work() {
    let mut app = configured();
    app.set_ecs_lod_phase(PhasePolicy::PerEntity).unwrap();
    let population: Vec<_> = (0..4)
        .map(|_| {
            (
                app.try_spawn_with_policy((Counter(0),)).unwrap(),
                [5.0, 0.0, 0.0],
            )
        })
        .collect();
    for frame in 0..4 {
        let result = app
            .drive_ecs_lod([0.0; 3], &population, frame, |world, entity, _| {
                world.get_mut::<Counter>(entity).unwrap().0 += 1;
            })
            .unwrap();
        assert_eq!(result.to_tick.len(), 1);
        assert!((frame + u64::from(result.to_tick[0].index())).is_multiple_of(4));
    }
    for (entity, _) in population {
        assert_eq!(app.world().get::<Counter>(entity), Some(&Counter(1)));
    }
}

#[test]
fn lod_rejects_invalid_populations_before_callbacks_or_sleep_transitions() {
    let mut app = configured();
    let live = app.try_spawn_with_policy(()).unwrap();
    let dead = app.try_spawn_with_policy(()).unwrap();
    app.world_mut().despawn(dead);
    let mut calls = 0;
    assert_eq!(
        app.drive_ecs_lod(
            [0.0; 3],
            &[(live, [20.0, 0.0, 0.0]), (live, [0.0; 3])],
            0,
            |_, _, _| calls += 1
        )
        .err(),
        Some(EcsPolicyError::DuplicateEntity(live))
    );
    assert!(!app.is_ecs_dormant(live).unwrap());
    assert_eq!(
        app.drive_ecs_lod(
            [0.0; 3],
            &[(live, [20.0, 0.0, 0.0]), (dead, [0.0; 3])],
            0,
            |_, _, _| calls += 1
        )
        .err(),
        Some(EcsPolicyError::EntityNotLive(dead))
    );
    assert_eq!(
        app.drive_ecs_lod([f32::NAN; 3], &[], 0, |_, _, _| calls += 1)
            .err(),
        Some(EcsPolicyError::InvalidPosition)
    );
    assert_eq!(
        app.drive_ecs_lod([f32::MAX; 3], &[(live, [-f32::MAX; 3])], 0, |_, _, _| {
            calls += 1;
        })
        .err(),
        Some(EcsPolicyError::InvalidPosition)
    );
    assert_eq!(calls, 0);
    assert!(!app.is_ecs_dormant(live).unwrap());
}

#[test]
fn lod_callback_panic_retains_runtime_and_dead_entries_are_pruned() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let mut app = configured();
    let live = app.try_spawn_with_policy((Counter(0),)).unwrap();
    let sleeping = app.try_spawn_with_policy(()).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        app.drive_ecs_lod(
            [0.0; 3],
            &[(live, [0.0; 3]), (sleeping, [20.0; 3])],
            0,
            |world, entity, _| {
                world.get_mut::<Counter>(entity).unwrap().0 += 1;
                panic!("host callback");
            },
        )
        .unwrap();
    }))
    .is_err());
    assert!(app.is_ecs_dormant(sleeping).unwrap());
    assert_eq!(app.world().get::<Counter>(live), Some(&Counter(1)));
    app.world_mut().despawn(sleeping);
    app.drive_ecs_lod([0.0; 3], &[(live, [0.0; 3])], 1, |world, entity, _| {
        world.get_mut::<Counter>(entity).unwrap().0 += 1;
    })
    .unwrap();
    assert_eq!(app.world().get::<Counter>(live), Some(&Counter(2)));
    let runtime = &app.ecs_policy.as_ref().unwrap().partition;
    assert!(runtime.dormancy.is_empty());
    assert_eq!(runtime.dormancy.pending_wake_count(), 0);
}

#[test]
fn lod_callback_can_despawn_a_later_selected_entity() {
    let mut app = configured();
    let first = app.try_spawn_with_policy(()).unwrap();
    let second = app.try_spawn_with_policy(()).unwrap();
    let mut ran = Vec::new();
    let result = app
        .drive_ecs_lod(
            [0.0; 3],
            &[(second, [0.0; 3]), (first, [0.0; 3])],
            0,
            |world, entity, _| {
                ran.push(entity);
                world.despawn(second);
            },
        )
        .unwrap();
    assert_eq!(ran, vec![first]);
    assert_eq!(result.to_tick, vec![first, second]);
}

#[test]
fn streaming_consumes_loads_hysteresis_interests_and_real_world_eviction() {
    let mut app = configured();
    let ticket = origin_load(&mut app);
    let resident = app.try_spawn_with_policy((Counter(0),)).unwrap();
    app.settle_ecs_cell(&ticket, &[resident]).unwrap();
    let grid = app.ecs_world_partition().unwrap();
    assert_eq!(grid.index().cell_of(resident), Some(CellCoord::ORIGIN));
    assert_eq!(
        grid.driver().streamer().state(CellCoord::ORIGIN),
        CellState::Loaded
    );
    assert!(app
        .advance_ecs_streaming(pos(10.0), &[])
        .unwrap()
        .despawned
        .is_empty());
    // A second host interest keeps the origin even far from the camera.
    assert!(app
        .advance_ecs_streaming(pos(30.0), &[pos(0.0)])
        .unwrap()
        .despawned
        .is_empty());
    assert!(app.world().contains(resident));
    let evicted = app.advance_ecs_streaming(pos(30.0), &[]).unwrap();
    assert_eq!(evicted.despawned, vec![resident]);
    assert!(evicted.unloaded_cells.contains(&CellCoord::ORIGIN));
    assert!(!app.world().contains(resident));
    assert_eq!(
        app.ecs_world_partition().unwrap().index().cell_of(resident),
        None
    );
    // Real subsequent App work observes the eviction, not a parallel mock index.
    app.add_systems(crate::Update, |world: &mut World| {
        assert_eq!(world.entity_count(), 0);
    });
    app.update();
}

#[test]
fn pending_tickets_are_stable_but_evicted_completed_and_foreign_tickets_fail() {
    let mut app = configured();
    let first = origin_load(&mut app);
    let repeated = origin_load(&mut app);
    assert!(Arc::ptr_eq(&first.identity, &repeated.identity));
    let mut other = configured();
    let foreign = origin_load(&mut other);
    assert_eq!(
        app.settle_ecs_cell(&foreign, &[]),
        Err(EcsPolicyError::InvalidLoad)
    );
    app.advance_ecs_streaming(pos(30.0), &[]).unwrap();
    assert_eq!(
        app.settle_ecs_cell(&first, &[]),
        Err(EcsPolicyError::InvalidLoad)
    );
    let replacement = origin_load(&mut app);
    assert!(!Arc::ptr_eq(&first.identity, &replacement.identity));
    assert_eq!(
        app.settle_ecs_cell(&first, &[]),
        Err(EcsPolicyError::InvalidLoad)
    );
    app.settle_ecs_cell(&replacement, &[]).unwrap();
    assert_eq!(
        app.settle_ecs_cell(&replacement, &[]),
        Err(EcsPolicyError::InvalidLoad)
    );
}

#[test]
fn cell_settlement_validates_whole_payload_and_preserves_host_ownership_on_error() {
    let mut app = configured();
    let ticket = origin_load(&mut app);
    let resident = app.try_spawn_with_policy(()).unwrap();
    let dead = app.try_spawn_with_policy(()).unwrap();
    app.world_mut().despawn(dead);
    assert_eq!(
        app.settle_ecs_cell(&ticket, &[resident, resident]),
        Err(EcsPolicyError::DuplicateEntity(resident))
    );
    assert_eq!(
        app.settle_ecs_cell(&ticket, &[resident, dead]),
        Err(EcsPolicyError::EntityNotLive(dead))
    );
    assert!(app.ecs_world_partition().unwrap().index().is_empty());
    assert!(app.world().contains(resident));
    app.settle_ecs_cell(&ticket, &[resident]).unwrap();
    let next = app
        .advance_ecs_streaming(pos(10.0), &[])
        .unwrap()
        .to_load
        .remove(0);
    assert_eq!(
        app.settle_ecs_cell(&next, &[resident]),
        Err(EcsPolicyError::EntityAlreadyStreamed(resident))
    );
    assert_eq!(
        app.ecs_world_partition().unwrap().index().cell_of(resident),
        Some(CellCoord::ORIGIN)
    );
    app.settle_ecs_cell(&next, &[]).unwrap();
}

#[test]
fn admission_reservations_block_settlement_without_losing_pending_request() {
    let mut app = configured();
    let ticket = origin_load(&mut app);
    let reserved = app.world().entities().reserve_entity();
    assert_eq!(
        app.settle_ecs_cell(&ticket, &[]),
        Err(EcsPolicyError::Admission(
            prism_ecs::policy::AdmissionError::OutstandingReservations
        ))
    );
    assert_eq!(
        app.ecs_world_partition()
            .unwrap()
            .driver()
            .streamer()
            .state(ticket.cell()),
        CellState::Loading
    );
    app.world_mut().flush_reserved();
    app.settle_ecs_cell(&ticket, &[reserved]).unwrap();
}

#[test]
fn unchanged_streaming_reconfiguration_preserves_residency_and_pending_identity() {
    let mut app = configured();
    let ticket = origin_load(&mut app);
    app.configure_ecs_policy(profile(), EcsCapabilities::default())
        .unwrap();
    let repeated = origin_load(&mut app);
    assert!(Arc::ptr_eq(&ticket.identity, &repeated.identity));
    let resident = app.try_spawn_with_policy(()).unwrap();
    app.settle_ecs_cell(&ticket, &[resident]).unwrap();
    let mut replacement = profile();
    replacement.streaming.as_mut().unwrap().cell_size = 20.0;
    let original = app.ecs_policy().unwrap().clone();
    assert_eq!(
        app.configure_ecs_policy(replacement.clone(), EcsCapabilities::default())
            .err(),
        Some(EcsPolicyError::StreamingBusy)
    );
    assert_eq!(app.ecs_policy(), Some(&original));
    assert!(app.world().contains(resident));
    let drained = app.drain_ecs_streaming().unwrap();
    assert_eq!(drained.despawned, vec![resident]);
    assert!(app.ecs_world_partition().unwrap().index().is_empty());
    app.configure_ecs_policy(replacement, EcsCapabilities::default())
        .unwrap();
    assert_eq!(origin_load(&mut app).cell(), CellCoord::ORIGIN);
}

#[test]
fn invalid_streaming_positions_are_rejected_before_mutation() {
    let mut app = configured();
    let ticket = origin_load(&mut app);
    let resident = app.try_spawn_with_policy(()).unwrap();
    app.settle_ecs_cell(&ticket, &[resident]).unwrap();
    for camera in [
        pos(f64::NAN),
        pos(f64::INFINITY),
        pos(f64::MAX),
        pos(f64::from(CELL_BOUND) * 20.0),
    ] {
        assert_eq!(
            app.advance_ecs_streaming(camera, &[]).err(),
            Some(EcsPolicyError::InvalidPosition)
        );
    }
    assert_eq!(
        app.advance_ecs_streaming(pos(30.0), &[pos(f64::NAN)]).err(),
        Some(EcsPolicyError::InvalidPosition)
    );
    assert!(app.world().contains(resident));
    assert_eq!(
        app.ecs_world_partition().unwrap().index().cell_of(resident),
        Some(CellCoord::ORIGIN)
    );
    assert!(app
        .advance_ecs_streaming(pos(0.0), &[])
        .unwrap()
        .to_load
        .is_empty());
    // Both ends of the checked interval can safely coexist as interests.
    let edge = f64::from(CELL_BOUND) * 10.0;
    let result = app.advance_ecs_streaming(pos(edge), &[pos(-edge)]).unwrap();
    assert_eq!(result.to_load.len(), 2);
    assert_eq!(result.to_load[0].cell().x, -CELL_BOUND);
    assert_eq!(result.to_load[1].cell().x, CELL_BOUND);
}

#[test]
fn external_despawn_and_eviction_hook_panic_keep_retryable_entity_index() {
    use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use prism_ecs::component_hooks::{ComponentHooks, HookContext};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    struct PanicOnce {
        armed: Arc<AtomicBool>,
        drops: Arc<AtomicUsize>,
    }
    impl Component for PanicOnce {}
    impl Drop for PanicOnce {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn panic_once(ctx: HookContext<'_>) {
        let value = ctx.world.get::<PanicOnce>(ctx.entity).unwrap();
        assert!(
            !value.armed.swap(false, Ordering::Relaxed),
            "one pre-removal hook panic"
        );
    }
    let mut app = configured();
    app.world_mut()
        .register_component_hooks::<PanicOnce>(ComponentHooks::new().with_on_remove(panic_once));
    let ticket = origin_load(&mut app);
    let armed = Arc::new(AtomicBool::new(true));
    let drops = Arc::new(AtomicUsize::new(0));
    let completed = app.try_spawn_with_policy((Counter(0),)).unwrap();
    let first = app
        .try_spawn_with_policy((PanicOnce {
            armed,
            drops: drops.clone(),
        },))
        .unwrap();
    let second = app.try_spawn_with_policy((Counter(1),)).unwrap();
    let external = app.try_spawn_with_policy(()).unwrap();
    app.settle_ecs_cell(&ticket, &[external, second, first, completed])
        .unwrap();
    app.world_mut().despawn(external);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        app.drain_ecs_streaming().unwrap();
    }))
    .is_err());
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    assert!(!app.world().contains(completed));
    assert!(app.world().contains(first));
    assert!(app.world().contains(second));
    // The failed cloned eviction must not orphan the remaining live payload.
    assert_eq!(
        app.ecs_world_partition().unwrap().index().cell_of(second),
        Some(CellCoord::ORIGIN)
    );
    let retry = app.drain_ecs_streaming().unwrap();
    assert_eq!(retry.despawned, vec![first, second]);
    assert!(retry.already_gone.contains(&completed));
    assert!(retry.already_gone.contains(&external));
    assert!(!app.world().contains(second));
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert!(app.ecs_world_partition().unwrap().index().is_empty());
}

#[test]
fn replaced_world_rejects_old_load_tickets_and_can_start_fresh() {
    let mut app = configured();
    let ticket = origin_load(&mut app);
    app.main_mut().world = World::new();
    assert_eq!(
        app.advance_ecs_streaming(pos(0.0), &[]).err(),
        Some(EcsPolicyError::WorldChanged)
    );
    app.configure_ecs_policy(profile(), EcsCapabilities::default())
        .unwrap();
    let fresh = origin_load(&mut app);
    assert_ne!(fresh.world_id(), ticket.world_id());
    assert_eq!(
        app.settle_ecs_cell(&ticket, &[]),
        Err(EcsPolicyError::InvalidLoad)
    );
    app.settle_ecs_cell(&fresh, &[]).unwrap();
}

#[test]
fn disabled_facilities_reject_execution_without_affecting_admission() {
    let mut app = App::empty();
    app.configure_ecs_policy(EcsProfile::default(), EcsCapabilities::default())
        .unwrap();
    let entity = app.try_spawn_with_policy(()).unwrap();
    assert_eq!(
        app.set_ecs_lod_phase(PhasePolicy::PerEntity).err(),
        Some(EcsPolicyError::LodDisabled)
    );
    assert_eq!(app.is_ecs_dormant(entity), Err(EcsPolicyError::LodDisabled));
    assert_eq!(
        app.drive_ecs_lod([0.0; 3], &[(entity, [0.0; 3])], 0, |_, _, _| {
            panic!("disabled LOD must not invoke host work");
        })
        .err(),
        Some(EcsPolicyError::LodDisabled)
    );
    assert_eq!(
        app.advance_ecs_streaming(pos(0.0), &[]).err(),
        Some(EcsPolicyError::StreamingDisabled)
    );
    assert_eq!(
        app.drain_ecs_streaming().err(),
        Some(EcsPolicyError::StreamingDisabled)
    );
    assert!(app.world().contains(entity));
    assert!(app.check_entity_admission(1).is_ok());
}

#[test]
fn pending_only_grid_must_be_drained_before_reconfiguration() {
    let mut app = configured();
    let ticket = origin_load(&mut app);
    let mut replacement = profile();
    replacement.streaming = None;
    assert_eq!(
        app.configure_ecs_policy(replacement.clone(), EcsCapabilities::default())
            .err(),
        Some(EcsPolicyError::StreamingBusy)
    );
    let drained = app.drain_ecs_streaming().unwrap();
    assert_eq!(drained.unloaded_cells, vec![ticket.cell()]);
    assert!(drained.to_load.is_empty());
    assert_eq!(
        app.settle_ecs_cell(&ticket, &[]),
        Err(EcsPolicyError::InvalidLoad)
    );
    app.configure_ecs_policy(replacement, EcsCapabilities::default())
        .unwrap();
    assert_eq!(
        app.ecs_world_partition().err(),
        Some(EcsPolicyError::StreamingDisabled)
    );
}

#[test]
fn settlement_cannot_hide_raw_spawns_exceeding_explicit_budget() {
    let mut app = App::empty();
    let mut limited = profile();
    limited.entities = prism_ecs::policy::EntityBudget {
        max_live: 1,
        max_slots: 2,
    };
    app.configure_ecs_policy(limited, EcsCapabilities::default())
        .unwrap();
    let ticket = origin_load(&mut app);
    let admitted = app.try_spawn_with_policy(()).unwrap();
    let bypass = app.world_mut().spawn(());
    assert!(matches!(
        app.settle_ecs_cell(&ticket, &[admitted, bypass]),
        Err(EcsPolicyError::Admission(_))
    ));
    assert!(app.ecs_world_partition().unwrap().index().is_empty());
    assert!(app.world().contains(admitted));
    assert!(app.world().contains(bypass));
    app.world_mut().despawn(bypass);
    app.settle_ecs_cell(&ticket, &[admitted]).unwrap();
    assert_eq!(app.drain_ecs_streaming().unwrap().despawned, vec![admitted]);
}
