#![allow(clippy::field_reassign_with_default)]

use super::*;
use crate::component::Component;
use alloc::format;
use alloc::vec;

fn resolve(profile: EcsProfile) -> Result<EcsPolicy, PolicyError> {
    EcsPolicy::resolve(profile, EcsCapabilities::default())
}

fn lod() -> LodProfile {
    LodProfile {
        bands: vec![
            LodBandProfile {
                max_distance_sq: 100.0,
                tick_period: 1,
                quality: 0,
            },
            LodBandProfile {
                max_distance_sq: 10_000.0,
                tick_period: 4,
                quality: 7,
            },
        ],
        dormant_beyond_last: true,
    }
}

fn streaming() -> StreamingProfile {
    StreamingProfile {
        cell_size: 128.0,
        load_radius: 1,
        unload_radius: 2,
        max_cells_per_interest: 27,
    }
}

#[test]
fn default_profile_is_core_only_and_shareable_as_resource() {
    let policy = resolve(EcsProfile::default()).unwrap();
    assert_eq!(policy.worker_limit(), 1);
    assert_eq!(policy.profile().tier, QualityTier::Baseline);
    assert!(policy.profile().lod.is_none());
    assert!(policy.profile().streaming.is_none());
    let mut world = World::new();
    world.insert_resource(policy.clone());
    assert_eq!(world.resource::<EcsPolicy>(), &policy);
    assert_eq!(policy.check_world_admission(&world, 1), Ok(()));
    assert_eq!(
        policy.arena_strategy(),
        if cfg!(feature = "vm") {
            ArenaStrategy::VirtualMemoryWithGlobalFallback
        } else {
            ArenaStrategy::GlobalBlocks
        }
    );
}

#[test]
fn quality_labels_do_not_infer_hardware_or_change_numeric_decisions() {
    let mut profile = EcsProfile::default();
    profile.entities = EntityBudget {
        max_live: 17,
        max_slots: 23,
    };
    for tier in [
        QualityTier::Mobile,
        QualityTier::Console,
        QualityTier::HighEndDesktop,
        QualityTier::Custom("headless-batch".into()),
    ] {
        profile.tier = tier.clone();
        let policy = resolve(profile.clone()).unwrap();
        assert_eq!(policy.profile().tier, tier);
        assert_eq!(policy.capabilities(), EcsCapabilities::default());
        assert_eq!(policy.worker_limit(), 1);
        assert_eq!(policy.profile().entities.max_live, 17);
        assert_eq!(policy.profile().entities.max_slots, 23);
        assert_eq!(policy.profile().chunks.target_bytes, TARGET_CHUNK_BYTES);
    }
}

#[test]
fn invalid_worker_entity_and_chunk_configurations_are_rejected() {
    let mut profile = EcsProfile::default();
    profile.max_workers = Some(0);
    assert_eq!(
        resolve(profile).unwrap_err(),
        PolicyError::InvalidWorkerCount
    );
    let mut profile = EcsProfile::default();
    profile.requirements.min_workers = 0;
    assert_eq!(
        resolve(profile).unwrap_err(),
        PolicyError::InvalidWorkerCount
    );
    let mut profile = EcsProfile::default();
    profile.requirements.min_workers = 2;
    profile.max_workers = Some(1);
    assert_eq!(
        resolve(profile).unwrap_err(),
        PolicyError::InvalidWorkerCount
    );
    let capabilities = EcsCapabilities {
        worker_threads: 0,
        ..Default::default()
    };
    assert_eq!(
        EcsPolicy::resolve(EcsProfile::default(), capabilities).unwrap_err(),
        PolicyError::InvalidWorkerCount
    );
    let mut profile = EcsProfile::default();
    profile.entities = EntityBudget {
        max_live: 2,
        max_slots: 1,
    };
    assert_eq!(
        resolve(profile).unwrap_err(),
        PolicyError::InvalidEntityBudget
    );
    for bytes in [
        0,
        TARGET_CHUNK_BYTES / 2,
        TARGET_CHUNK_BYTES * 2,
        usize::MAX,
    ] {
        let mut profile = EcsProfile::default();
        profile.chunks.target_bytes = bytes;
        assert_eq!(
            resolve(profile).unwrap_err(),
            PolicyError::UnsupportedChunkSize
        );
    }
    let mut profile = EcsProfile::default();
    profile.chunks.chunks_per_batch = 0;
    assert_eq!(resolve(profile).unwrap_err(), PolicyError::EmptyChunkBatch);
}

#[test]
fn worker_selection_respects_compilation_host_and_project_ceiling() {
    let capabilities = EcsCapabilities {
        worker_threads: 8,
        ..Default::default()
    };
    let mut profile = EcsProfile::default();
    profile.max_workers = Some(3);
    let policy = EcsPolicy::resolve(profile.clone(), capabilities).unwrap();
    assert_eq!(
        policy.worker_limit(),
        if cfg!(feature = "multi_thread") { 3 } else { 1 }
    );
    profile.requirements.min_workers = 2;
    let result = EcsPolicy::resolve(profile.clone(), capabilities);
    if cfg!(feature = "multi_thread") {
        assert_eq!(result.unwrap().worker_limit(), 3);
        assert_eq!(
            resolve(profile).unwrap_err(),
            PolicyError::InsufficientWorkers
        );
    } else {
        assert_eq!(
            result.unwrap_err(),
            PolicyError::FeatureUnavailable(EcsFeature::MultiThread)
        );
    }
}

#[cfg(feature = "multi_thread")]
#[test]
fn task_pool_adapter_reserves_caller_and_preserves_explicit_affinity() {
    use prism_tasks::{CoreClassPolicy, TaskPoolConfig};

    let config = TaskPoolConfig {
        threads: 99,
        pin_workers: true,
        core_class_policy: CoreClassPolicy::PerformanceFirst,
    };
    let single = resolve(EcsProfile::default()).unwrap();
    let adapted = single.task_pool_config(config);
    assert_eq!(adapted.threads, 0);
    assert!(adapted.pin_workers);
    assert!(matches!(
        adapted.core_class_policy,
        CoreClassPolicy::PerformanceFirst
    ));

    let mut profile = EcsProfile::default();
    profile.max_workers = Some(3);
    let capabilities = EcsCapabilities {
        worker_threads: 8,
        ..Default::default()
    };
    let parallel = EcsPolicy::resolve(profile, capabilities).unwrap();
    let adapted = parallel.task_pool_config(config);
    assert_eq!(adapted.threads, 2);
    assert_eq!(adapted.threads + 1, parallel.worker_limit());
    assert_eq!(
        config.threads, 99,
        "the caller's configuration is unchanged"
    );
}

#[test]
fn simd_gpu_vm_and_determinism_require_real_compile_gates_and_host_reports() {
    let mut profile = EcsProfile::default();
    profile.requirements.min_simd_width_bits = 128;
    assert_eq!(
        resolve(profile.clone()).unwrap_err(),
        if cfg!(feature = "simd") {
            PolicyError::InsufficientSimd
        } else {
            PolicyError::FeatureUnavailable(EcsFeature::Simd)
        }
    );
    let capable = EcsCapabilities {
        worker_threads: 1,
        simd_width_bits: 128,
        gpu_resident_mapping: true,
        virtual_memory: true,
    };
    assert_eq!(
        EcsPolicy::resolve(profile, capable).is_ok(),
        cfg!(feature = "simd")
    );
    let mut profile = EcsProfile::default();
    profile.requirements.gpu_resident_mapping = true;
    assert_eq!(
        resolve(profile.clone()).unwrap_err(),
        if cfg!(feature = "gpu_resident") {
            PolicyError::GpuMappingUnavailable
        } else {
            PolicyError::FeatureUnavailable(EcsFeature::GpuResident)
        }
    );
    assert_eq!(
        EcsPolicy::resolve(profile, capable).is_ok(),
        cfg!(feature = "gpu_resident")
    );
    let mut profile = EcsProfile::default();
    profile.requirements.virtual_memory = true;
    assert_eq!(
        resolve(profile.clone()).unwrap_err(),
        if cfg!(feature = "vm") {
            PolicyError::VirtualMemoryUnavailable
        } else {
            PolicyError::FeatureUnavailable(EcsFeature::VirtualMemory)
        }
    );
    assert_eq!(
        EcsPolicy::resolve(profile, capable).is_ok(),
        cfg!(feature = "vm")
    );
    let mut profile = EcsProfile::default();
    profile.requirements.determinism = true;
    let result = resolve(profile);
    if cfg!(feature = "determinism") {
        assert!(result.is_ok());
    } else {
        assert_eq!(
            result.unwrap_err(),
            PolicyError::FeatureUnavailable(EcsFeature::Determinism)
        );
    }
}

#[test]
fn admission_honors_limits_reuses_slots_and_never_wraps() {
    let budget = EntityBudget {
        max_live: 6,
        max_slots: 8,
    };
    assert_eq!(budget.check(4, 8, 2), Ok(()));
    assert_eq!(budget.check(6, 8, 0), Ok(()));
    assert_eq!(
        budget.check(6, 8, 1),
        Err(AdmissionError::Exceeded {
            kind: EntityBudgetKind::Live,
            requested: 7,
            limit: 6,
        })
    );
    assert_eq!(
        budget.check(1, 9, 0),
        Err(AdmissionError::Exceeded {
            kind: EntityBudgetKind::Slots,
            requested: 9,
            limit: 8,
        })
    );
    assert_eq!(budget.check(9, 8, 0), Err(AdmissionError::InvalidCounts));
    assert_eq!(
        EntityBudget::default().check(u32::MAX, u32::MAX, 1),
        Err(AdmissionError::Exceeded {
            kind: EntityBudgetKind::Live,
            requested: u64::from(u32::MAX) + 1,
            limit: u32::MAX,
        })
    );
    assert_eq!(
        EntityBudget {
            max_live: 0,
            max_slots: 0
        }
        .check(0, 0, 0),
        Ok(())
    );
}

#[test]
fn world_admission_rejects_reservations_and_preserves_metadata_high_water() {
    let mut profile = EcsProfile::default();
    profile.entities = EntityBudget {
        max_live: 2,
        max_slots: 2,
    };
    let policy = resolve(profile).unwrap();
    let mut world = World::new();
    let first = world.spawn(());
    let second = world.spawn(());
    assert!(policy.check_world_admission(&world, 1).is_err());
    assert!(world.despawn(first));
    assert_eq!(policy.check_world_admission(&world, 1), Ok(()));
    let reserved = world.entities().reserve_entity();
    assert_eq!(
        policy.check_world_admission(&world, 0),
        Err(AdmissionError::OutstandingReservations)
    );
    world.flush_reserved();
    assert_eq!(policy.check_world_admission(&world, 0), Ok(()));
    assert!(world.despawn(reserved));
    assert!(world.despawn(second));
    let mut tighter = policy.profile().clone();
    tighter.entities = EntityBudget {
        max_live: 1,
        max_slots: 1,
    };
    assert_eq!(
        resolve(tighter).unwrap().check_world_admission(&world, 0),
        Err(AdmissionError::Exceeded {
            kind: EntityBudgetKind::Slots,
            requested: 2,
            limit: 1,
        })
    );
    assert_eq!(world.entity_count(), 0, "checks never spawn or flush");
}

struct Payload {
    _bytes: [u8; 2048],
}
impl Component for Payload {}

#[test]
fn chunk_batches_follow_actual_table_layout_and_cover_partial_tail_once() {
    let mut profile = EcsProfile::default();
    profile.chunks.chunks_per_batch = 2;
    let policy = resolve(profile).unwrap();
    let mut world = World::new();
    let first = world.spawn((Payload { _bytes: [1; 2048] },));
    let arch = world.entities().location(first).unwrap().archetype_id;
    let rows_per_chunk = world
        .archetypes()
        .get(arch)
        .unwrap()
        .table()
        .rows_per_chunk();
    let total = rows_per_chunk * 4 + 3;
    for _ in 1..total {
        world.spawn((Payload { _bytes: [2; 2048] },));
    }
    let table = world.archetypes().get(arch).unwrap().table();
    let mut batches = policy.chunk_batches(table);
    assert_eq!(batches.len(), 3);
    let first = batches.next().unwrap();
    assert_eq!(
        first,
        ChunkBatch {
            chunks: 0..2,
            rows: 0..rows_per_chunk * 2
        }
    );
    assert_eq!(batches.size_hint(), (2, Some(2)));
    let second = batches.next().unwrap();
    assert_eq!(
        second,
        ChunkBatch {
            chunks: 2..4,
            rows: rows_per_chunk * 2..rows_per_chunk * 4
        }
    );
    let tail = batches.next().unwrap();
    assert_eq!(
        tail,
        ChunkBatch {
            chunks: 4..5,
            rows: rows_per_chunk * 4..total
        }
    );
    assert_eq!(batches.len(), 0);
    assert_eq!(batches.next(), None);
    assert_eq!(batches.next(), None);
    assert_eq!(table.iter_chunks().map(|c| c.rows).sum::<usize>(), total);
}

#[test]
fn chunk_batch_extreme_granularity_handles_empty_zst_and_oversized_rows() {
    use crate::component::ComponentId;
    use core::alloc::Layout;

    let mut profile = EcsProfile::default();
    profile.chunks.chunks_per_batch = usize::MAX;
    let policy = resolve(profile).unwrap();
    let empty = Table::new(core::iter::empty());
    assert_eq!(policy.chunk_batches(&empty).len(), 0);
    assert_eq!(policy.chunk_batches(&empty).next(), None);
    struct Marker;
    impl Component for Marker {}
    #[repr(align(4096))]
    struct Aligned {
        _bytes: [u8; 32768],
    }
    impl Component for Aligned {}
    let mut world = World::new();
    for _ in 0..3 {
        let entity = world.spawn((Marker,));
        let table = world
            .archetypes()
            .get(world.entities().location(entity).unwrap().archetype_id)
            .unwrap()
            .table();
        assert_eq!(policy.chunk_batches(table).len(), 1);
        assert_eq!(
            policy.chunk_batches(table).next().unwrap().rows,
            0..table.len()
        );
    }
    let first = world.spawn((Aligned { _bytes: [0; 32768] },));
    world.spawn((Aligned { _bytes: [1; 32768] },));
    let table = world
        .archetypes()
        .get(world.entities().location(first).unwrap().archetype_id)
        .unwrap()
        .table();
    assert_eq!(table.rows_per_chunk(), 1);
    assert_eq!(
        policy.chunk_batches(table).next(),
        Some(ChunkBatch {
            chunks: 0..2,
            rows: 0..2
        })
    );
    let layout_only = Table::new([(ComponentId::new(0), Layout::new::<Aligned>(), None)]);
    assert_eq!(layout_only.rows_per_chunk(), 1);
    assert_eq!(
        policy.profile().chunks.target_bytes,
        16384,
        "payload target is not a strict allocation cap"
    );
}

#[test]
fn chunk_batches_debug_does_not_require_table_debug() {
    let policy = resolve(EcsProfile::default()).unwrap();
    let table = Table::new(core::iter::empty());
    let debug = format!("{:?}", policy.chunk_batches(&table));
    assert!(debug.starts_with("ChunkBatches {"));
    assert!(debug.contains("chunk_count: 0"));
    assert!(debug.contains("chunks_per_batch: 1"));
}

#[test]
fn malformed_lod_is_rejected_before_any_processor_is_built() {
    let invalid = [
        vec![],
        vec![LodBandProfile {
            max_distance_sq: f32::NAN,
            tick_period: 1,
            quality: 0,
        }],
        vec![LodBandProfile {
            max_distance_sq: f32::INFINITY,
            tick_period: 1,
            quality: 0,
        }],
        vec![LodBandProfile {
            max_distance_sq: -1.0,
            tick_period: 1,
            quality: 0,
        }],
        vec![LodBandProfile {
            max_distance_sq: 1.0,
            tick_period: 0,
            quality: 0,
        }],
        vec![
            LodBandProfile {
                max_distance_sq: 1.0,
                tick_period: 1,
                quality: 0
            };
            2
        ],
        vec![lod().bands[1], lod().bands[0]],
        (0..257)
            .map(|i| LodBandProfile {
                max_distance_sq: i as f32,
                tick_period: 1,
                quality: 0,
            })
            .collect(),
    ];
    for bands in invalid {
        let mut profile = EcsProfile::default();
        profile.lod = Some(LodProfile {
            bands,
            dormant_beyond_last: false,
        });
        assert_eq!(resolve(profile).unwrap_err(), PolicyError::InvalidLodBands);
    }
}

#[test]
fn malformed_or_explosive_streaming_profiles_are_rejected() {
    let base = streaming();
    for config in [
        StreamingProfile {
            cell_size: 0.0,
            ..base
        },
        StreamingProfile {
            cell_size: -1.0,
            ..base
        },
        StreamingProfile {
            cell_size: f64::INFINITY,
            ..base
        },
        StreamingProfile {
            cell_size: f64::NAN,
            ..base
        },
        StreamingProfile {
            unload_radius: 0,
            ..base
        },
        StreamingProfile {
            max_cells_per_interest: 26,
            ..base
        },
        StreamingProfile {
            unload_radius: u32::MAX,
            ..base
        },
        StreamingProfile {
            load_radius: i32::MAX as u32,
            unload_radius: i32::MAX as u32,
            max_cells_per_interest: u64::MAX,
            ..base
        },
    ] {
        let mut profile = EcsProfile::default();
        profile.streaming = Some(config);
        assert_eq!(resolve(profile).unwrap_err(), PolicyError::InvalidStreaming);
    }
}

#[test]
fn valid_partition_profiles_require_the_real_feature() {
    let mut profile = EcsProfile::default();
    profile.lod = Some(lod());
    profile.streaming = Some(streaming());
    let result = resolve(profile);
    if cfg!(feature = "partition") {
        assert!(result.is_ok());
    } else {
        assert_eq!(
            result.unwrap_err(),
            PolicyError::FeatureUnavailable(EcsFeature::Partition)
        );
    }
}

#[cfg(feature = "partition")]
#[test]
fn policy_builds_real_lod_processor_with_distance_cadence_quality_and_dormancy() {
    use crate::partition::lod::{LodLevel, LodQuality};
    use crate::partition::processor::PhasePolicy;
    let mut profile = EcsProfile::default();
    profile.lod = Some(lod());
    let policy = resolve(profile).unwrap();
    let processor = policy.lod_processor(PhasePolicy::PerEntity).unwrap();
    assert_eq!(processor.phase_policy(), PhasePolicy::PerEntity);
    let schedule = processor.schedule();
    assert_eq!(schedule.evaluate(100.0, 1).level, Some(LodLevel(0)));
    let far = schedule.evaluate(101.0, 4);
    assert_eq!(far.level, Some(LodLevel(1)));
    assert_eq!(far.quality, LodQuality(7));
    assert!(far.tick_this_frame);
    assert!(!schedule.evaluate(101.0, 3).tick_this_frame);
    assert_eq!(schedule.evaluate(10_001.0, 4).level, None);
    let mut profile = policy.profile().clone();
    profile.lod.as_mut().unwrap().dormant_beyond_last = false;
    let clamped = resolve(profile).unwrap().lod_schedule().unwrap();
    assert_eq!(clamped.evaluate(10_001.0, 4).level, Some(LodLevel(1)));
    assert_eq!(
        schedule.evaluate(10_001.0, 4).level,
        None,
        "existing resolved policy stays immutable"
    );
}

#[cfg(feature = "partition")]
#[test]
fn policy_builds_real_streaming_grid_and_hysteresis_without_world_side_effects() {
    use crate::partition::floating_origin::WorldPos;
    let mut profile = EcsProfile::default();
    profile.streaming = Some(streaming());
    let policy = resolve(profile).unwrap();
    let mut partition = policy.world_partition().unwrap();
    assert_eq!(partition.origin().cell_size(), 128.0);
    assert_eq!(partition.interest_grid().cell_size(), 128.0);
    assert_eq!(partition.driver().streamer().load_radius(), 1);
    assert_eq!(partition.driver().streamer().unload_radius(), 2);
    let mut world = World::new();
    let camera = WorldPos::new(0.0, 0.0, 0.0);
    let initial = partition.advance(&mut world, camera, &[]);
    assert_eq!(initial.to_load.len(), 27);
    for cell in initial.to_load {
        partition.settle_loaded(cell, &[]);
    }
    assert!(partition
        .advance(&mut world, camera, &[])
        .to_load
        .is_empty());
    let adjacent = partition.advance(&mut world, WorldPos::new(128.0, 0.0, 0.0), &[]);
    assert_eq!(adjacent.to_load.len(), 9);
    assert!(
        adjacent.unloaded_cells.is_empty(),
        "retention radius keeps the old load cube resident"
    );
    assert_eq!(
        world.entity_count(),
        0,
        "policy/streaming requests do not fabricate entities"
    );
    assert!(resolve(EcsProfile::default())
        .unwrap()
        .world_partition()
        .is_none());
    assert!(resolve(EcsProfile::default())
        .unwrap()
        .lod_processor(crate::partition::processor::PhasePolicy::Synchronized)
        .is_none());
}
