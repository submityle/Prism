use super::*;
use prism_ecs::component::Component;
use prism_ecs::policy::{EntityBudget, EntityBudgetKind, QualityTier};
use prism_ecs::world::World;

#[derive(Debug, PartialEq)]
struct Value(u32);
impl Component for Value {}

fn limited(max_live: u32, max_slots: u32) -> EcsProfile {
    EcsProfile {
        entities: EntityBudget {
            max_live,
            max_slots,
        },
        ..EcsProfile::default()
    }
}

#[test]
fn policy_is_opt_in_and_preserves_independent_app_configuration() {
    use crate::capability::{Capabilities, QualityTier as AppTier};
    use crate::run_mode::RunMode;

    let mut app = App::empty();
    assert!(app.ecs_policy().is_none());
    assert_eq!(
        app.check_entity_admission(0),
        Err(EcsPolicyError::NotConfigured)
    );
    app.insert_resource(Capabilities::headless());
    app.insert_resource(AppTier::Mobile);
    app.insert_resource(RunMode::EditorEmbedded);
    let profile = EcsProfile {
        tier: QualityTier::Custom("host-selected".into()),
        max_workers: Some(1),
        ..EcsProfile::default()
    };
    let host = EcsCapabilities {
        worker_threads: 8,
        ..EcsCapabilities::default()
    };
    app.configure_ecs_policy(profile.clone(), host).unwrap();
    assert_eq!(app.ecs_policy().unwrap().profile(), &profile);
    assert_eq!(app.ecs_policy().unwrap().capabilities(), host);
    assert_eq!(app.ecs_policy().unwrap().worker_limit(), 1);
    assert_eq!(
        app.world().get_resource::<Capabilities>(),
        Some(&Capabilities::headless())
    );
    assert_eq!(
        app.world().get_resource::<AppTier>(),
        Some(&AppTier::Mobile)
    );
    assert_eq!(
        app.world().get_resource::<RunMode>(),
        Some(&RunMode::EditorEmbedded)
    );
    assert_eq!(app.world().get_resource::<EcsPolicy>(), app.ecs_policy());
}

#[test]
fn invalid_configuration_leaves_applied_policy_and_entities_unchanged() {
    let mut app = App::empty();
    app.configure_ecs_policy(limited(2, 2), EcsCapabilities::default())
        .unwrap();
    let entity = app.try_spawn_with_policy((Value(7),)).unwrap();
    let original = app.ecs_policy().unwrap().clone();
    assert!(matches!(
        app.configure_ecs_policy(
            EcsProfile::default(),
            EcsCapabilities {
                worker_threads: 0,
                ..EcsCapabilities::default()
            }
        )
        .err(),
        Some(EcsPolicyError::Policy(PolicyError::InvalidWorkerCount))
    ));
    assert!(matches!(
        app.configure_ecs_policy(limited(0, 2), EcsCapabilities::default())
            .err(),
        Some(EcsPolicyError::Admission(AdmissionError::Exceeded {
            kind: EntityBudgetKind::Live,
            ..
        }))
    ));
    let invalid_chunks = EcsProfile {
        chunks: prism_ecs::policy::ChunkPolicy {
            chunks_per_batch: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(matches!(
        app.configure_ecs_policy(invalid_chunks, EcsCapabilities::default())
            .err(),
        Some(EcsPolicyError::Policy(PolicyError::EmptyChunkBatch))
    ));
    assert_eq!(app.ecs_policy(), Some(&original));
    assert_eq!(app.world().get_resource::<EcsPolicy>(), Some(&original));
    assert_eq!(app.world().get::<Value>(entity), Some(&Value(7)));
}

#[test]
fn explicit_admission_reuses_slots_but_is_not_a_global_entity_limit() {
    let mut app = App::empty();
    app.configure_ecs_policy(limited(1, 1), EcsCapabilities::default())
        .unwrap();
    let first = app.try_spawn_with_policy((Value(1),)).unwrap();
    assert!(matches!(
        app.try_spawn_with_policy((Value(2),)),
        Err(EcsPolicyError::Admission(AdmissionError::Exceeded {
            kind: EntityBudgetKind::Live,
            requested: 2,
            limit: 1
        }))
    ));
    assert_eq!(app.world().entity_count(), 1);
    assert!(app.world_mut().despawn(first));
    let replacement = app.try_spawn_with_policy((Value(3),)).unwrap();
    assert_ne!(first, replacement);
    assert_eq!(first.index(), replacement.index());
    assert_eq!(app.world().entities().capacity(), 1);

    // Raw World operations remain usable; the adapter must not claim a quota
    // enforced on Commands/hooks/other callers that bypass explicit admission.
    let bypass = app.world_mut().spawn((Value(4),));
    assert!(app.world().contains(bypass));
    assert!(matches!(
        app.check_entity_admission(0),
        Err(EcsPolicyError::Admission(_))
    ));
}

#[test]
fn dead_metadata_highwater_and_outstanding_reservations_are_not_hidden() {
    let mut app = App::empty();
    let a = app.world_mut().spawn(());
    let b = app.world_mut().spawn(());
    app.world_mut().despawn(a);
    app.world_mut().despawn(b);
    assert!(matches!(
        app.configure_ecs_policy(limited(1, 1), EcsCapabilities::default())
            .err(),
        Some(EcsPolicyError::Admission(AdmissionError::Exceeded {
            kind: EntityBudgetKind::Slots,
            requested: 2,
            limit: 1
        }))
    ));
    app.configure_ecs_policy(limited(2, 2), EcsCapabilities::default())
        .unwrap();
    let reserved = app.world().entities().reserve_entity();
    assert_eq!(
        app.check_entity_admission(1),
        Err(EcsPolicyError::Admission(
            AdmissionError::OutstandingReservations
        ))
    );
    assert_eq!(
        app.try_spawn_with_policy(()),
        Err(EcsPolicyError::Admission(
            AdmissionError::OutstandingReservations
        ))
    );
    assert!(app
        .configure_ecs_policy(EcsProfile::default(), EcsCapabilities::default())
        .is_err());
    assert!(app.world().entities().has_outstanding_reservations());
    app.world_mut().flush_reserved();
    assert!(app.world().contains(reserved));
    assert!(app.try_spawn_with_policy(()).is_ok());
}

#[test]
fn main_world_binding_and_authority_do_not_leak_to_secondary_worlds() {
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Secondary;
    let mut app = App::empty();
    app.insert_sub_app(Secondary, crate::SubApp::new());
    app.configure_ecs_policy(limited(1, 1), EcsCapabilities::default())
        .unwrap();
    assert!(app
        .get_sub_app(Secondary)
        .unwrap()
        .world
        .get_resource::<EcsPolicy>()
        .is_none());
    #[cfg(feature = "multi_thread")]
    assert!(app
        .get_sub_app(Secondary)
        .unwrap()
        .world
        .get_resource::<crate::tasks::TaskPoolResource>()
        .is_none());
    // The resource is a system-readable copy, not the adapter's authority.
    app.world_mut().insert_resource(
        EcsPolicy::resolve(EcsProfile::default(), EcsCapabilities::default()).unwrap(),
    );
    app.try_spawn_with_policy(()).unwrap();
    assert!(app.try_spawn_with_policy(()).is_err());

    app.main_mut().world = World::new();
    assert!(app.ecs_policy().is_none());
    assert_eq!(
        app.check_entity_admission(0),
        Err(EcsPolicyError::WorldChanged)
    );
    app.configure_ecs_policy(limited(2, 2), EcsCapabilities::default())
        .unwrap();
    assert!(app.try_spawn_with_policy(()).is_ok());
}

#[test]
fn serial_adapter_rejects_parallel_feature_unification_without_pool_support() {
    #[cfg(not(feature = "multi_thread"))]
    if EcsFeature::MultiThread.is_compiled() {
        let policy = EcsPolicy::resolve(
            EcsProfile::default(),
            EcsCapabilities {
                worker_threads: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(matches!(
            App::empty().apply_ecs_policy(policy).err(),
            Some(EcsPolicyError::FeatureUnavailable(EcsFeature::MultiThread))
        ));
    }
}

#[test]
fn adapter_rejects_partition_feature_unification_without_app_support() {
    #[cfg(not(feature = "partition"))]
    if EcsFeature::Partition.is_compiled() {
        let profile = EcsProfile {
            streaming: Some(prism_ecs::policy::StreamingProfile {
                cell_size: 10.0,
                load_radius: 0,
                unload_radius: 1,
                max_cells_per_interest: 1,
            }),
            ..Default::default()
        };
        let policy = EcsPolicy::resolve(profile, EcsCapabilities::default()).unwrap();
        let mut app = App::empty();
        assert!(matches!(
            app.apply_ecs_policy(policy).err(),
            Some(EcsPolicyError::FeatureUnavailable(EcsFeature::Partition))
        ));
        assert!(app.ecs_policy().is_none());
        assert!(app.world().get_resource::<EcsPolicy>().is_none());
    }
}

#[cfg(feature = "multi_thread")]
mod parallel {
    use super::*;
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use prism_ecs::command::Commands;
    use prism_tasks::{CoreClassPolicy, Priority, TaskPoolConfig};

    #[test]
    fn policy_pool_dispatches_real_schedule_and_flushes_deferred_commands() {
        let mut app = App::empty();
        app.configure_ecs_policy(EcsProfile::default(), EcsCapabilities::default())
            .unwrap();
        assert_eq!(app.task_pool().unwrap().0.worker_count(), 0);
        let seen = Arc::new(AtomicUsize::new(0));
        let observer = seen.clone();
        app.add_systems(
            crate::Update,
            move |policy: prism_ecs::system::Res<EcsPolicy>, mut commands: Commands| {
                observer.store(policy.worker_limit(), Ordering::Relaxed);
                commands.spawn((Value(9),));
            },
        );
        app.main_mut().run_schedule(crate::Update);
        assert_eq!(seen.load(Ordering::Relaxed), 1);
        assert_eq!(app.world().entity_count(), 1);
        // The budget includes the caller, not an off-by-one extra worker.
        let profile = EcsProfile {
            max_workers: Some(3),
            ..Default::default()
        };
        let policy = EcsPolicy::resolve(
            profile,
            EcsCapabilities {
                worker_threads: 8,
                ..Default::default()
            },
        )
        .unwrap();
        app.apply_ecs_policy_with_pool_config(
            policy,
            TaskPoolConfig {
                threads: usize::MAX,
                pin_workers: false,
                core_class_policy: CoreClassPolicy::Flat,
            },
        )
        .unwrap();
        let workers = app.task_pool().unwrap().0.worker_count();
        assert!(
            workers == 0 || workers == 2,
            "serial task backend or exactly two background workers"
        );
        app.main_mut().run_schedule(crate::Update);
        assert_eq!(seen.load(Ordering::Relaxed), 3);
        assert_eq!(app.world().entity_count(), 2);
    }

    #[test]
    fn policy_pool_replacement_preserves_queued_and_follow_up_qos_work() {
        let mut app = App::empty();
        app.configure_ecs_policy(EcsProfile::default(), EcsCapabilities::default())
            .unwrap();
        let qos = app.task_qos().unwrap().clone();
        let hits = Arc::new(AtomicUsize::new(0));
        let first = hits.clone();
        let submitter = qos.clone();
        qos.submit_foreground(Priority::Critical, move || {
            first.fetch_add(1, Ordering::Relaxed);
            submitter.submit_foreground(Priority::Normal, move || {
                first.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert!(app
            .configure_ecs_policy(
                limited(0, 0),
                EcsCapabilities {
                    worker_threads: 0,
                    ..Default::default()
                }
            )
            .is_err());
        assert_eq!(qos.pending(), 1);
        app.configure_ecs_policy(EcsProfile::default(), EcsCapabilities::default())
            .unwrap();
        assert_eq!(app.task_qos().unwrap().pending(), 1);
        let extra = hits.clone();
        qos.submit_background(0, move || {
            extra.fetch_add(1, Ordering::Relaxed);
        });
        app.update();
        assert_eq!(hits.load(Ordering::Relaxed), 3);
        assert!(qos.is_empty());
        app.update();
        assert_eq!(hits.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn backend_worker_minimum_is_enforced_without_overwriting_previous_policy() {
        let mut app = App::empty();
        app.configure_ecs_policy(EcsProfile::default(), EcsCapabilities::default())
            .unwrap();
        let original = app.ecs_policy().unwrap().clone();
        let mut profile = EcsProfile::default();
        profile.requirements.min_workers = 2;
        let result = app.configure_ecs_policy(
            profile,
            EcsCapabilities {
                worker_threads: 2,
                ..Default::default()
            },
        );
        // `prism_tasks/single` may be selected through feature unification.
        if let Err(error) = result {
            assert_eq!(
                error,
                EcsPolicyError::BackendWorkers {
                    required: 2,
                    actual: 1
                }
            );
            assert_eq!(app.ecs_policy(), Some(&original));
            assert_eq!(app.task_pool().unwrap().0.worker_count(), 0);
        } else {
            assert_eq!(app.ecs_policy().unwrap().worker_limit(), 2);
            assert_eq!(app.task_pool().unwrap().0.worker_count(), 1);
        }
    }
}
