extern crate std;
use alloc::format;

use self::std::panic::{catch_unwind, AssertUnwindSafe};
use super::tests::{Marker, Provider, ReloadSettings};
use super::*;
use crate::{ApplyError, SceneHost, WorldBudget, WorldRuntime};
use prism_asset::StableGuid;
use prism_ecs::component_hooks::{ComponentHooks, HookContext};
use prism_world_persist::{
    ActorId, FieldId, FileJournal, LogicalClock, OverrideDelta, OverrideValue,
    PersistentId as WorldPersistentId, Transaction, TransactionId,
};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct JournalPath(PathBuf);
impl JournalPath {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(std::env::temp_dir().join(format!(
            "prism-world-scene-journal-{}-{}.pwrj",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for JournalPath {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
use prism_world_residency::{
    MaterializationBudget, ResidencyAction, ResidencyPolicy, ResidencyRecord, ResidencyScheduler,
    ResidencyState,
};
use prism_world_stream::{
    AabbI64, CellManifest, DemandSource, GlobalPos, StreamAction, StreamBudget,
};

fn make_runtime() -> WorldRuntime {
    WorldRuntime::new(ResidencyScheduler::new(
        ResidencyPolicy::new(1, 2, 3, 4, 0).unwrap(),
    ))
}

fn budget() -> WorldBudget {
    WorldBudget {
        stream: StreamBudget {
            io_bytes: 8,
            max_actions: 8,
        },
        materialization: MaterializationBudget::new(8, 8),
    }
}

fn manifest(sector: SectorId) -> CellManifest {
    CellManifest {
        sector,
        bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap(),
        scene_key: 1,
        content_version: 1,
        derived_hash: 2,
        io_bytes: 1,
        data_layers: prism_world_stream::DataLayerMask::EMPTY,
    }
}

fn load_bound_cell(
    runtime: &mut WorldRuntime,
    host: &mut SceneEcsHost<'_, Provider>,
    sector: SectorId,
    id: u64,
) -> Entity {
    let plan = runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[manifest(sector)],
        &[],
        1,
        budget(),
    );
    assert!(matches!(
        plan.stream.actions.as_slice(),
        [StreamAction::Load(_)]
    ));
    runtime.apply_scene_plan(&plan, host).unwrap();
    runtime.mark_resident_manifest(manifest(sector));
    host.bind_residency(
        id,
        ResidencyBinding {
            sector,
            local: LocalEntityId::new(0),
        },
    )
    .unwrap();
    host.loaded_cell(sector).unwrap().root.unwrap()
}

#[test]
fn world_persistence_delta_survives_scene_eviction_and_reload() {
    let mut world = World::new();
    let sector = SectorId::new(0, 0, 0, 0);
    let object = WorldPersistentId(0xfeed);
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(
        &mut world,
        Provider {
            persistent_object: Some(object),
            ..Provider::default()
        },
    );
    let first_root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    host.world_mut().get_mut::<Marker>(first_root).unwrap().0 = 42;

    let eviction = runtime.tick(&[], &[manifest(sector)], &[], 2, budget());
    runtime.apply_scene_plan(&eviction, &mut host).unwrap();
    runtime.mark_unloaded(sector);
    assert!(!host.world().contains(first_root));
    assert_eq!(runtime.persistence().len(), 1);
    assert_eq!(runtime.persistence().get(0).unwrap().object(), object);

    let reload = runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[manifest(sector)],
        &[],
        3,
        budget(),
    );
    runtime.apply_scene_plan(&reload, &mut host).unwrap();
    let reloaded_root = host.loaded_cell(sector).unwrap().root.unwrap();
    assert_ne!(reloaded_root, first_root);
    assert_eq!(host.world().get::<Marker>(reloaded_root), Some(&Marker(42)));
    assert_eq!(host.loaded_cell_count(), 1);
}

#[test]
fn failed_scene_reload_keeps_old_scene_and_bindings_authoritative() {
    let mut world = World::new();
    let sector = SectorId::new(0, 5, 2, 1);
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    SceneHost::load_cell(&mut host, sector).unwrap();
    let old_root = host.loaded_cell(sector).unwrap().root.unwrap();
    host.bind_residency(
        45,
        ResidencyBinding {
            sector,
            local: LocalEntityId::new(1),
        },
    )
    .unwrap();

    assert_eq!(
        host.reload_cell(sector),
        Err(SceneEcsHostError::ReloadBindingUnavailable {
            residency_id: 45,
            sector,
            local: LocalEntityId::new(1),
        })
    );
    assert_eq!(host.loaded_cell(sector).unwrap().root, Some(old_root));
    assert!(host.world().contains(old_root));
    assert_eq!(host.world().entity_count(), 1);
    assert_eq!(
        host.residency_binding(45),
        Some(ResidencyBinding {
            sector,
            local: LocalEntityId::new(1)
        })
    );
}

#[test]
fn scene_reload_stages_replacement_preserves_bindings_and_cleans_old_entities() {
    let mut world = World::new();
    let sector = SectorId::new(0, 5, 2, 1);
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    SceneHost::load_cell(&mut host, sector).unwrap();
    let old = host.loaded_cell(sector).unwrap().clone();
    let old_root = old.root.unwrap();
    host.bind_residency(
        44,
        ResidencyBinding {
            sector,
            local: LocalEntityId::new(0),
        },
    )
    .unwrap();
    let persistent = prism_scene::PersistentId::from_u128(0x44);
    host.world_mut()
        .get_mut::<SceneInstance>(old_root)
        .unwrap()
        .persistent = Some(persistent);
    host.world_mut()
        .get_mut::<SceneInstance>(old_root)
        .unwrap()
        .overrides_mut()
        .set_property(prism_scene::PropertyOverride::new(
            LocalEntityId::new(0),
            prism_scene::ComponentTypeId::of::<ReloadSettings>(),
            prism_scene::FieldPath::parse("value").unwrap(),
            prism_scene::OverrideValue::U64(123),
        ));

    host.reload_cell(sector).unwrap();
    let new_root = host.loaded_cell(sector).unwrap().root.unwrap();
    assert_ne!(new_root, old_root);
    assert!(!host.world().contains(old_root));
    assert_eq!(host.world().get::<Marker>(new_root), Some(&Marker(70)));
    assert_eq!(
        host.world().get::<ReloadSettings>(new_root),
        Some(&ReloadSettings { value: 123 })
    );
    let new_instance = host.world().get::<SceneInstance>(new_root).unwrap();
    assert_eq!(new_instance.persistent, Some(persistent));
    assert_eq!(new_instance.overrides().unwrap().property_count(), 1);
    assert_eq!(
        host.residency_binding(44),
        Some(ResidencyBinding {
            sector,
            local: LocalEntityId::new(0)
        })
    );
    assert_eq!(
        host.world()
            .get::<SceneInstance>(new_root)
            .unwrap()
            .template,
        StableGuid::from_u128(sector.morton_hint() as u128 + 100)
    );
}

#[test]
fn atomic_multi_cell_reload_rolls_back_all_staged_cells_on_later_provider_failure() {
    let first = SectorId::new(0, 11, 0, 0);
    let second = SectorId::new(0, 12, 0, 0);
    let mut world = World::new();
    let mut provider = Provider::default();
    provider.fail_stage_on = Some(2);
    let mut host = SceneEcsHost::new(&mut world, provider);
    for (id, sector) in [(41, first), (42, second)] {
        SceneHost::load_cell(&mut host, sector).unwrap();
        host.bind_residency(
            id,
            ResidencyBinding {
                sector,
                local: LocalEntityId::new(0),
            },
        )
        .unwrap();
    }
    let old_roots = [
        host.loaded_cell(first).unwrap().root.unwrap(),
        host.loaded_cell(second).unwrap().root.unwrap(),
    ];

    assert_eq!(
        host.reload_cells_atomic(&[second, first]),
        Err(SceneEcsHostError::Provider(()))
    );
    assert_eq!(host.loaded_cell(first).unwrap().root, Some(old_roots[0]));
    assert_eq!(host.loaded_cell(second).unwrap().root, Some(old_roots[1]));
    assert!(old_roots
        .into_iter()
        .all(|root| host.world().contains(root)));
    assert_eq!(host.world().entity_count(), 2);
    assert_eq!(host.residency_binding(41).unwrap().sector, first);
    assert_eq!(host.residency_binding(42).unwrap().sector, second);
}

#[test]
fn atomic_multi_cell_reload_rejects_duplicates_before_provider_work() {
    let sector = SectorId::new(0, 13, 0, 0);
    let mut world = World::new();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    SceneHost::load_cell(&mut host, sector).unwrap();
    let root = host.loaded_cell(sector).unwrap().root.unwrap();

    assert_eq!(
        host.reload_cells_atomic(&[sector, sector]),
        Err(SceneEcsHostError::DuplicateReloadSector)
    );
    assert_eq!(host.provider().stage_attempts, 0);
    assert_eq!(host.loaded_cell(sector).unwrap().root, Some(root));
}

#[test]
fn atomic_multi_cell_reload_publishes_every_replacement_together() {
    let first = SectorId::new(0, 21, 0, 0);
    let second = SectorId::new(0, 22, 0, 0);
    let mut world = World::new();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    for (id, sector) in [(51, first), (52, second)] {
        SceneHost::load_cell(&mut host, sector).unwrap();
        host.bind_residency(
            id,
            ResidencyBinding {
                sector,
                local: LocalEntityId::new(0),
            },
        )
        .unwrap();
    }
    let old_roots = [
        host.loaded_cell(first).unwrap().root.unwrap(),
        host.loaded_cell(second).unwrap().root.unwrap(),
    ];

    host.reload_cells_atomic(&[second, first]).unwrap();

    let new_roots = [
        host.loaded_cell(first).unwrap().root.unwrap(),
        host.loaded_cell(second).unwrap().root.unwrap(),
    ];
    assert!(new_roots.iter().all(|root| !old_roots.contains(root)));
    assert!(old_roots
        .into_iter()
        .all(|root| !host.world().contains(root)));
    assert!(new_roots
        .into_iter()
        .all(|root| host.world().contains(root)));
    assert_eq!(host.loaded_cell_count(), 2);
    assert_eq!(host.residency_binding(51).unwrap().sector, first);
    assert_eq!(host.residency_binding(52).unwrap().sector, second);
}

#[test]
fn durable_atomic_multi_cell_reload_replays_one_snapshot_before_publication() {
    let first = SectorId::new(0, 31, 0, 0);
    let second = SectorId::new(0, 32, 0, 0);
    let object = WorldPersistentId(0x3132);
    let mut world = World::new();
    let mut host = SceneEcsHost::new(
        &mut world,
        Provider {
            persistent_object: Some(object),
            ..Provider::default()
        },
    );
    for sector in [first, second] {
        SceneHost::load_cell(&mut host, sector).unwrap();
    }
    let old_roots = [
        host.loaded_cell(first).unwrap().root.unwrap(),
        host.loaded_cell(second).unwrap().root.unwrap(),
    ];
    let delta = OverrideDelta::single(FieldId(1), OverrideValue::Integer(55)).unwrap();
    let inverse = OverrideDelta::single(FieldId(1), OverrideValue::Integer(7)).unwrap();
    let transaction = Transaction::new(
        TransactionId(0x3132),
        ActorId(1),
        LogicalClock(1),
        0,
        object,
        delta,
        inverse,
    )
    .unwrap();
    let mut log = AppendOnlyLog::new();
    log.append(transaction).unwrap();

    host.reload_cells_atomic_with_persistence(&[second, first], &log)
        .unwrap();

    let new_roots = [
        host.loaded_cell(first).unwrap().root.unwrap(),
        host.loaded_cell(second).unwrap().root.unwrap(),
    ];
    assert!(new_roots.iter().all(|root| !old_roots.contains(root)));
    assert!(old_roots
        .into_iter()
        .all(|root| !host.world().contains(root)));
    assert!(new_roots
        .into_iter()
        .all(|root| host.world().contains(root)));
}

#[test]
fn durable_atomic_multi_cell_replay_failure_keeps_all_old_cells_authoritative() {
    let first = SectorId::new(0, 33, 0, 0);
    let second = SectorId::new(0, 34, 0, 0);
    let object = WorldPersistentId(0x3334);
    let mut world = World::new();
    let mut host = SceneEcsHost::new(
        &mut world,
        Provider {
            persistent_object: Some(object),
            ..Provider::default()
        },
    );
    for sector in [first, second] {
        SceneHost::load_cell(&mut host, sector).unwrap();
    }
    let old_roots = [
        host.loaded_cell(first).unwrap().root.unwrap(),
        host.loaded_cell(second).unwrap().root.unwrap(),
    ];
    let delta = OverrideDelta::single(FieldId(1), OverrideValue::Integer(55)).unwrap();
    let inverse = OverrideDelta::single(FieldId(1), OverrideValue::Integer(7)).unwrap();
    let transaction = Transaction::new(
        TransactionId(0x3334),
        ActorId(1),
        LogicalClock(1),
        0,
        object,
        delta,
        inverse,
    )
    .unwrap();
    let mut log = AppendOnlyLog::new();
    log.append(transaction).unwrap();
    host.provider_mut().fail_replay = true;

    assert_eq!(
        host.reload_cells_atomic_with_persistence(&[second, first], &log),
        Err(SceneEcsHostError::Provider(()))
    );

    assert_eq!(
        [
            host.loaded_cell(first).unwrap().root.unwrap(),
            host.loaded_cell(second).unwrap().root.unwrap(),
        ],
        old_roots
    );
    assert!(old_roots
        .into_iter()
        .all(|root| host.world().contains(root)));
    assert_eq!(host.loaded_cell_count(), 2);
    assert_eq!(host.world().entity_count(), 2);
}

#[test]
fn durable_scene_plan_groups_contiguous_reload_actions_into_one_batch() {
    let first = SectorId::new(0, 35, 0, 0);
    let second = SectorId::new(0, 36, 0, 0);
    let mut world = World::new();
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    for sector in [first, second] {
        SceneHost::load_cell(&mut host, sector).unwrap();
        runtime.mark_resident_manifest(manifest(sector));
    }
    let revision = |sector| CellManifest {
        content_version: 2,
        derived_hash: 3,
        ..manifest(sector)
    };
    let plan = runtime.tick(
        &[
            DemandSource::new(first, 0, 1, 10).unwrap(),
            DemandSource::new(second, 0, 1, 10).unwrap(),
        ],
        &[revision(first), revision(second)],
        &[],
        2,
        budget(),
    );
    assert_eq!(plan.stream.actions.len(), 2);
    assert!(plan
        .stream
        .actions
        .iter()
        .all(|action| matches!(action, StreamAction::Reload(_))));

    let path = JournalPath::new();
    let mut journal = FileJournal::open(&path.0).unwrap();
    runtime
        .apply_scene_plan_durable(&plan, &mut host, &mut journal)
        .unwrap();
    assert_eq!(host.provider().stage_attempts, 2);
    assert_eq!(host.loaded_cell_count(), 2);
}

#[test]
fn durable_scene_plan_batch_failure_keeps_every_old_cell_authoritative() {
    let first = SectorId::new(0, 37, 0, 0);
    let second = SectorId::new(0, 38, 0, 0);
    let mut world = World::new();
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    for sector in [first, second] {
        SceneHost::load_cell(&mut host, sector).unwrap();
        runtime.mark_resident_manifest(manifest(sector));
    }
    let old_roots = [
        host.loaded_cell(first).unwrap().root.unwrap(),
        host.loaded_cell(second).unwrap().root.unwrap(),
    ];
    let revision = |sector| CellManifest {
        content_version: 2,
        derived_hash: 3,
        ..manifest(sector)
    };
    let plan = runtime.tick(
        &[
            DemandSource::new(first, 0, 1, 10).unwrap(),
            DemandSource::new(second, 0, 1, 10).unwrap(),
        ],
        &[revision(first), revision(second)],
        &[],
        2,
        budget(),
    );
    assert_eq!(plan.stream.actions.len(), 2);
    host.provider_mut().fail_stage_on = Some(2);

    let path = JournalPath::new();
    let mut journal = FileJournal::open(&path.0).unwrap();
    assert!(runtime
        .apply_scene_plan_durable(&plan, &mut host, &mut journal)
        .is_err());

    assert_eq!(host.provider().stage_attempts, 2);
    assert_eq!(
        [
            host.loaded_cell(first).unwrap().root.unwrap(),
            host.loaded_cell(second).unwrap().root.unwrap(),
        ],
        old_roots
    );
    assert!(old_roots
        .into_iter()
        .all(|root| host.world().contains(root)));
    assert_eq!(host.loaded_cell_count(), 2);
}

#[test]
fn durable_scene_reload_replays_latest_cell_state_before_publishing() {
    let path = JournalPath::new();
    let sector = SectorId::new(0, 4, 1, 2);
    let object = WorldPersistentId(0x4455);
    let mut journal = FileJournal::open(&path.0).unwrap();
    journal
        .append(
            Transaction::new(
                TransactionId(0x4455),
                ActorId(1),
                LogicalClock(1),
                0,
                object,
                OverrideDelta::single(FieldId(1), OverrideValue::Integer(88)).unwrap(),
                OverrideDelta::single(FieldId(1), OverrideValue::Integer(7)).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();

    let mut world = World::new();
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(
        &mut world,
        Provider {
            persistent_object: Some(object),
            ..Provider::default()
        },
    );
    let first = runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[manifest(sector)],
        &[],
        1,
        budget(),
    );
    runtime
        .apply_scene_plan_durable(&first, &mut host, &mut journal)
        .unwrap();
    runtime.mark_resident_manifest(manifest(sector));
    host.bind_residency(
        441,
        ResidencyBinding {
            sector,
            local: LocalEntityId::new(0),
        },
    )
    .unwrap();
    let root = host.loaded_cell(sector).unwrap().root.unwrap();
    assert_eq!(host.world().get::<Marker>(root), Some(&Marker(88)));

    let revision = CellManifest {
        content_version: 2,
        derived_hash: 3,
        ..manifest(sector)
    };
    let reload = runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[revision],
        &[],
        2,
        budget(),
    );
    runtime.mark_resident_manifest(manifest(sector));
    let old_root = root;
    runtime
        .apply_scene_plan_durable(&reload, &mut host, &mut journal)
        .unwrap();
    let new_root = host.loaded_cell(sector).unwrap().root.unwrap();
    assert_ne!(old_root, new_root);
    assert!(!host.world().contains(old_root));
    assert_eq!(host.world().get::<Marker>(new_root), Some(&Marker(88)));
}

#[test]
fn repeated_stream_plan_load_dispatches_to_staged_scene_reload() {
    let mut world = World::new();
    let sector = SectorId::new(0, 8, 1, 1);
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let first = runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[manifest(sector)],
        &[],
        1,
        budget(),
    );
    runtime.apply_scene_plan(&first, &mut host).unwrap();
    runtime.mark_resident_manifest(manifest(sector));
    let old_root = host.loaded_cell(sector).unwrap().root.unwrap();

    let revision = CellManifest {
        content_version: 2,
        derived_hash: 3,
        ..manifest(sector)
    };
    let update = runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[revision],
        &[],
        2,
        budget(),
    );
    assert!(matches!(
        update.stream.actions.as_slice(),
        [StreamAction::Reload(_)]
    ));
    runtime.apply_scene_plan(&update, &mut host).unwrap();
    let new_root = host.loaded_cell(sector).unwrap().root.unwrap();
    assert_ne!(new_root, old_root);
    assert!(!host.world().contains(old_root));
    assert_eq!(host.world().get::<Marker>(new_root), Some(&Marker(70)));
}

#[test]
fn durable_scene_eviction_survives_runtime_and_journal_restart() {
    let path = JournalPath::new();
    let sector = SectorId::new(0, 0, 0, 0);
    let object = WorldPersistentId(0xabcd);
    let mut journal = FileJournal::open(&path.0).unwrap();
    let mut world = World::new();
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(
        &mut world,
        Provider {
            persistent_object: Some(object),
            ..Provider::default()
        },
    );
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    host.world_mut().get_mut::<Marker>(root).unwrap().0 = 51;
    let eviction = runtime.tick(&[], &[manifest(sector)], &[], 2, budget());
    runtime
        .apply_scene_plan_durable(&eviction, &mut host, &mut journal)
        .unwrap();
    assert!(!host.world().contains(root));
    assert_eq!(journal.log().len(), 1);
    drop(host);
    drop(runtime);
    drop(journal);

    let mut journal = FileJournal::open(&path.0).unwrap();
    let mut world = World::new();
    let mut second_runtime = make_runtime();
    let mut host = SceneEcsHost::new(
        &mut world,
        Provider {
            persistent_object: Some(object),
            ..Provider::default()
        },
    );
    let load = second_runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[manifest(sector)],
        &[],
        1,
        budget(),
    );
    second_runtime
        .apply_scene_plan_durable(&load, &mut host, &mut journal)
        .unwrap();
    let fresh = host.loaded_cell(sector).unwrap().root.unwrap();
    assert_eq!(host.world().get::<Marker>(fresh), Some(&Marker(51)));
}

#[test]
fn rejected_durable_eviction_keeps_scene_resident_and_log_unchanged() {
    let path = JournalPath::new();
    let sector = SectorId::new(0, 0, 0, 0);
    let object = WorldPersistentId(0xabce);
    let mut journal = FileJournal::open(&path.0).unwrap();
    let existing = Transaction::new(
        TransactionId(1000),
        ActorId(1),
        LogicalClock(0),
        0,
        object,
        OverrideDelta::single(FieldId(1), OverrideValue::Integer(1)).unwrap(),
        OverrideDelta::single(FieldId(1), OverrideValue::Integer(7)).unwrap(),
    )
    .unwrap();
    journal.append(existing.clone()).unwrap();

    let mut world = World::new();
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(
        &mut world,
        Provider {
            persistent_object: Some(object),
            ..Provider::default()
        },
    );
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    host.world_mut().get_mut::<Marker>(root).unwrap().0 = 51;
    let eviction = runtime.tick(&[], &[manifest(sector)], &[], 2, budget());

    let error = runtime
        .apply_scene_plan_durable(&eviction, &mut host, &mut journal)
        .expect_err("duplicate transaction must reject durable unload");
    assert!(matches!(
        error,
        ApplyError::Host(SceneEcsHostError::DurableAppend(_))
    ));
    assert!(host.world().contains(root));
    assert_eq!(host.loaded_cell_count(), 1);
    assert_eq!(journal.log().as_slice(), &[existing]);
}

#[test]
fn rejected_persistence_batch_keeps_cell_and_log_unchanged() {
    let mut world = World::new();
    let sector = SectorId::new(0, 0, 0, 0);
    let object = WorldPersistentId(0xbeef);
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(
        &mut world,
        Provider {
            persistent_object: Some(object),
            ..Provider::default()
        },
    );
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    let existing = Transaction::new(
        TransactionId(1000),
        ActorId(1),
        LogicalClock(0),
        0,
        object,
        OverrideDelta::single(FieldId(1), OverrideValue::Integer(8)).unwrap(),
        OverrideDelta::single(FieldId(1), OverrideValue::Integer(7)).unwrap(),
    )
    .unwrap();
    runtime.append(existing).unwrap();

    let eviction = runtime.tick(&[], &[manifest(sector)], &[], 2, budget());
    assert!(matches!(
        runtime.apply_scene_plan(&eviction, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::PersistenceAppend(_)))
    ));
    assert!(host.world().contains(root));
    assert_eq!(host.loaded_cell_count(), 1);
    assert_eq!(host.residency_binding(41).unwrap().sector, sector);
    assert_eq!(runtime.persistence().len(), 1);
}

#[test]
fn failed_world_state_replay_despawns_unpublished_scene() {
    let mut world = World::new();
    let sector = SectorId::new(0, 0, 0, 0);
    let object = WorldPersistentId(0xcafe);
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(
        &mut world,
        Provider {
            persistent_object: Some(object),
            ..Provider::default()
        },
    );
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    host.world_mut().get_mut::<Marker>(root).unwrap().0 = 13;
    let eviction = runtime.tick(&[], &[manifest(sector)], &[], 2, budget());
    runtime.apply_scene_plan(&eviction, &mut host).unwrap();
    runtime.mark_unloaded(sector);
    host.provider_mut().fail_replay = true;

    let reload = runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[manifest(sector)],
        &[],
        3,
        budget(),
    );
    assert_eq!(
        runtime.apply_scene_plan(&reload, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::Provider(())))
    );
    assert_eq!(host.loaded_cell_count(), 0);
    assert_eq!(host.residency_binding(41), None);
    assert_eq!(host.world().entity_count(), 0);
    assert_eq!(runtime.persistence().len(), 1);
}

#[test]
fn runtime_tick_unload_and_demote_apply_successfully() {
    let mut world = World::new();
    let sector = SectorId::new(0, 0, 0, 0);
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);

    let plan = runtime.tick(
        &[],
        &[manifest(sector)],
        &[ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1)],
        2,
        budget(),
    );
    assert_eq!(
        plan.stream.actions,
        alloc::vec![StreamAction::Unload(sector)]
    );
    assert!(matches!(
        plan.residency.actions.as_slice(),
        [ResidencyAction::Demote { id: 41, .. }]
    ));
    runtime.apply_scene_plan(&plan, &mut host).unwrap();
    assert_eq!(host.loaded_cell_count(), 0);
    assert_eq!(host.residency_binding(41), None);
    assert!(!host.world().contains(root));
    assert_eq!(host.world().entity_count(), 0);
    assert!(host.provider().events.is_empty());
}

#[test]
fn scene_plan_with_records_commits_provider_transition_after_scene_stream_work() {
    let mut world = World::new();
    let sector = SectorId::new(0, 0, 0, 0);
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    let mut records = alloc::vec![ResidencyRecord::new(
        41,
        ResidencyState::TierC,
        0,
        1,
        1,
        1,
    )];
    let plan = runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[manifest(sector)],
        &records,
        2,
        budget(),
    );
    assert!(plan.stream.actions.is_empty());
    assert!(matches!(
        plan.residency.actions.as_slice(),
        [ResidencyAction::Promote {
            id: 41,
            from: ResidencyState::TierC,
            to: ResidencyState::TierA,
            ..
        }]
    ));
    runtime
        .apply_scene_plan_with_records(&plan, &mut records, 2, &mut host)
        .unwrap();
    assert_eq!(records[0].state, ResidencyState::TierA);
    assert_eq!(records[0].last_transition_frame, Some(2));
    assert_eq!(host.world().get::<Marker>(root), Some(&Marker(9)));
    assert_eq!(
        host.provider().events,
        alloc::vec![(ResidencyState::TierA, 41, root)]
    );
}

#[test]
fn dropping_host_preserves_world_owned_cells_and_bindings_for_takeover() {
    let mut world = World::new();
    let sector = SectorId::new(0, 0, 0, 0);
    let mut runtime = make_runtime();
    let root = {
        let mut host = SceneEcsHost::new(&mut world, Provider::default());
        load_bound_cell(&mut runtime, &mut host, sector, 41)
    };
    assert!(world.contains(root));

    let mut replacement = SceneEcsHost::new(&mut world, Provider::default());
    assert_eq!(replacement.loaded_cell_count(), 1);
    assert_eq!(replacement.loaded_cell(sector).unwrap().root, Some(root));
    assert_eq!(replacement.residency_binding(41).unwrap().sector, sector);
    SceneHost::transition_residency(
        &mut replacement,
        41,
        ResidencyState::TierA,
        ResidencyState::TierC,
    )
    .unwrap();
    assert_eq!(replacement.world().get::<Marker>(root), Some(&Marker(3)));
    SceneHost::unload_cell(&mut replacement, sector).unwrap();
    assert_eq!(replacement.world().entity_count(), 0);
}

#[test]
fn same_plan_eviction_does_not_hide_an_originally_unknown_id() {
    let mut world = World::new();
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let sector = SectorId::new(0, 0, 0, 0);
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    let plan = runtime.tick(
        &[],
        &[],
        &[
            ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1),
            ResidencyRecord::new(99, ResidencyState::TierA, 9, 1, 1, 1),
        ],
        2,
        budget(),
    );
    assert_eq!(plan.residency.actions[0].id(), 41);
    assert_eq!(
        runtime.apply_scene_plan(&plan, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::ResidencyNotBound(99)))
    );
    // The failed application is intentionally not a transaction.
    assert!(!host.world().contains(root));
    assert_eq!(host.loaded_cell_count(), 0);
    assert!(host.provider().events.is_empty());
    // Even the previous eviction's id must fail in the next application.
    let next = runtime.tick(
        &[],
        &[],
        &[ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1)],
        3,
        budget(),
    );
    assert_eq!(
        runtime.apply_scene_plan(&next, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::ResidencyNotBound(41)))
    );
}

#[test]
fn successful_plan_does_not_leave_eviction_exemptions() {
    let mut world = World::new();
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let sector = SectorId::new(0, 0, 0, 0);
    load_bound_cell(&mut runtime, &mut host, sector, 41);
    let record = ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1);
    let plan = runtime.tick(&[], &[], &[record], 2, budget());
    runtime.apply_scene_plan(&plan, &mut host).unwrap();
    assert_eq!(
        SceneHost::transition_residency(
            &mut host,
            41,
            ResidencyState::TierA,
            ResidencyState::TierC,
        ),
        Err(SceneEcsHostError::ResidencyNotBound(41))
    );
    let next = runtime.tick(&[], &[], &[record], 3, budget());
    assert!(next.stream.actions.is_empty());
    assert_eq!(
        runtime.apply_scene_plan(&next, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::ResidencyNotBound(41)))
    );
    // After a real reload and explicit rebind, the same id is usable again.
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    let plan = runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[],
        &[record],
        4,
        budget(),
    );
    runtime.apply_scene_plan(&plan, &mut host).unwrap();
    assert_eq!(
        host.provider().events,
        alloc::vec![(ResidencyState::TierC, 41, root)]
    );
}

#[test]
fn eviction_does_not_skip_surviving_cells_or_provider_errors() {
    let mut world = World::new();
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let evicted = SectorId::new(0, 0, 0, 0);
    let surviving = SectorId::new(0, 10, 0, 0);
    let removed_root = load_bound_cell(&mut runtime, &mut host, evicted, 41);
    // Keep the first cell in range while loading the second one.
    let load = runtime.tick(
        &[
            DemandSource::new(evicted, 0, 1, 10).unwrap(),
            DemandSource::new(surviving, 0, 1, 10).unwrap(),
        ],
        &[manifest(surviving)],
        &[],
        2,
        budget(),
    );
    runtime.apply_scene_plan(&load, &mut host).unwrap();
    runtime.mark_resident(surviving);
    host.bind_residency(
        42,
        ResidencyBinding {
            sector: surviving,
            local: LocalEntityId::new(0),
        },
    )
    .unwrap();
    let surviving_root = host.loaded_cell(surviving).unwrap().root.unwrap();
    let demands = [DemandSource::new(surviving, 0, 1, 10).unwrap()];
    let plan = runtime.tick(
        &demands,
        &[],
        &[
            ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1),
            ResidencyRecord::new(42, ResidencyState::TierA, 9, 1, 1, 1),
        ],
        3,
        budget(),
    );
    host.provider_mut().fail_transition = true;
    assert_eq!(
        runtime.apply_scene_plan(&plan, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::Provider(())))
    );
    assert!(!host.world().contains(removed_root));
    assert_eq!(host.world().get::<Marker>(surviving_root), Some(&Marker(7)));
    host.provider_mut().fail_transition = false;
    let next = runtime.tick(
        &demands,
        &[],
        &[ResidencyRecord::new(42, ResidencyState::TierA, 9, 1, 1, 1)],
        4,
        budget(),
    );
    runtime.apply_scene_plan(&next, &mut host).unwrap();
    assert_eq!(
        host.provider().events,
        alloc::vec![(ResidencyState::TierC, 42, surviving_root)]
    );
    assert_eq!(host.world().get::<Marker>(surviving_root), Some(&Marker(3)));
}

#[test]
fn bindings_to_never_loaded_cells_and_missing_locals_still_fail() {
    let mut world = World::new();
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let sector = SectorId::new(0, 0, 0, 0);
    host.bind_residency(
        41,
        ResidencyBinding {
            sector,
            local: LocalEntityId::new(1),
        },
    )
    .unwrap();
    let record = ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1);
    let plan = runtime.tick(&[], &[], &[record], 1, budget());
    assert_eq!(
        runtime.apply_scene_plan(&plan, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::CellNotLoaded(sector)))
    );
    let plan = runtime.tick(
        &[DemandSource::new(sector, 0, 1, 10).unwrap()],
        &[manifest(sector)],
        &[record],
        2,
        budget(),
    );
    assert_eq!(
        runtime.apply_scene_plan(&plan, &mut host),
        Err(ApplyError::Host(
            SceneEcsHostError::LocalEntityUnavailable {
                residency_id: 41,
                sector,
                local: LocalEntityId::new(1),
            }
        ))
    );
}

#[test]
fn eviction_skips_promotions_but_rejects_unknown_promotions() {
    let mut world = World::new();
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let sector = SectorId::new(0, 0, 0, 0);
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    let plan = runtime.tick(
        &[],
        &[],
        &[
            ResidencyRecord::new(41, ResidencyState::TierC, 0, 1, 1, 1),
            ResidencyRecord::new(99, ResidencyState::TierC, 0, 1, 1, 1),
        ],
        2,
        budget(),
    );
    assert_eq!(
        plan.stream.actions,
        alloc::vec![StreamAction::Unload(sector)]
    );
    assert!(matches!(
        plan.residency.actions.as_slice(),
        [
            ResidencyAction::Promote { id: 41, .. },
            ResidencyAction::Promote { id: 99, .. }
        ]
    ));
    assert_eq!(
        runtime.apply_scene_plan(&plan, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::ResidencyNotBound(99)))
    );
    assert!(!host.world().contains(root));
    assert!(host.provider().events.is_empty());
}

#[test]
fn out_of_plan_unload_and_failed_unload_never_exempt_tier_work() {
    let mut world = World::new();
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let sector = SectorId::new(0, 0, 0, 0);
    load_bound_cell(&mut runtime, &mut host, sector, 41);
    let record = ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1);
    let plan = runtime.tick(&[], &[], &[record], 2, budget());
    SceneHost::unload_cell(&mut host, sector).unwrap();
    // This unload did not succeed inside the application: fail before tier work.
    assert_eq!(
        runtime.apply_scene_plan(&plan, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::CellNotLoaded(sector)))
    );
    let next = runtime.tick(&[], &[], &[record], 3, budget());
    assert!(next.stream.actions.is_empty());
    assert_eq!(
        runtime.apply_scene_plan(&next, &mut host),
        Err(ApplyError::Host(SceneEcsHostError::ResidencyNotBound(41)))
    );
    assert!(host.provider().events.is_empty());
}

#[test]
fn explicit_shutdown_preserves_unrelated_entities_and_allows_reuse() {
    let mut world = World::new();
    let unrelated = world.spawn(Marker(17));
    let sector = SectorId::new(0, 0, 0, 0);
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    // Shutdown also clears bindings without a corresponding loaded cell.
    host.bind_residency(
        99,
        ResidencyBinding {
            sector: SectorId::new(0, 1, 0, 0),
            local: LocalEntityId::new(0),
        },
    )
    .unwrap();
    assert_eq!(host.shutdown(), 1);
    assert_eq!(host.shutdown(), 0);
    assert_eq!(host.residency_binding(41), None);
    assert_eq!(host.residency_binding(99), None);
    assert!(!host.world().contains(root));
    assert_eq!(host.world().get::<Marker>(unrelated), Some(&Marker(17)));
    runtime.mark_unloaded(sector);
    let provider = host.into_provider();
    let mut replacement = SceneEcsHost::new(&mut world, provider);
    assert_eq!(replacement.loaded_cell_count(), 0);
    load_bound_cell(&mut runtime, &mut replacement, sector, 41);
    assert_eq!(replacement.shutdown(), 1);
    assert_eq!(replacement.world().entity_count(), 1);
}

#[test]
fn provider_handoff_preserves_state_without_unloading_or_overwriting_cells() {
    let mut world = World::new();
    let sector = SectorId::new(0, 0, 0, 0);
    let mut runtime = make_runtime();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    SceneHost::transition_residency(
        &mut host,
        41,
        ResidencyState::TierC,
        ResidencyState::TierA,
    )
    .unwrap();
    let provider = host.into_provider();
    let mut replacement = SceneEcsHost::new(&mut world, provider);
    assert_eq!(
        replacement.provider().events,
        alloc::vec![(ResidencyState::TierA, 41, root)]
    );
    assert_eq!(SceneHost::load_cell(&mut replacement, sector), Ok(()));
    assert_ne!(replacement.loaded_cell(sector).unwrap().root, Some(root));
    assert_eq!(replacement.shutdown(), 1);
}

struct DespawnPanicProbe {
    entity: Entity,
    id: u64,
    binding: ResidencyBinding,
    ownership_visible_in_hook: bool,
}

impl prism_ecs::resource::Resource for DespawnPanicProbe {}

fn panic_during_remove(ctx: HookContext<'_>) {
    let probe = ctx.world.resource::<DespawnPanicProbe>();
    if ctx.entity != probe.entity || probe.id == u64::MAX {
        return;
    }
    let ownership = ctx.world.resource::<SceneOwnership>();
    let visible = ownership.cells.contains_key(&probe.binding.sector)
        && ownership.bindings.get(&probe.id) == Some(&probe.binding);
    ctx.world
        .resource_mut::<DespawnPanicProbe>()
        .ownership_visible_in_hook = visible;
    panic!("scene despawn hook panic");
}

fn panic_once_during_remove(ctx: HookContext<'_>) {
    let probe = ctx.world.resource::<DespawnPanicProbe>();
    if ctx.entity != probe.entity || probe.id != u64::MAX {
        return;
    }
    ctx.world.resource_mut::<DespawnPanicProbe>().id = u64::MAX - 1;
    panic!("scene batch retirement hook panic");
}

fn install_despawn_panic(host: &mut SceneEcsHost<'_, Provider>, entity: Entity, id: u64) {
    let binding = host.residency_binding(id).unwrap();
    host.world_mut().insert_resource(DespawnPanicProbe {
        entity,
        id,
        binding,
        ownership_visible_in_hook: false,
    });
    host.world_mut().register_component_hooks::<Marker>(
        ComponentHooks::new().with_on_remove(panic_during_remove),
    );
}

#[test]
fn batch_reload_keeps_retired_scene_tokens_when_teardown_hook_panics() {
    let first = SectorId::new(0, 41, 0, 0);
    let second = SectorId::new(0, 42, 0, 0);
    let mut world = World::new();
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    for (id, sector) in [(61, first), (62, second)] {
        SceneHost::load_cell(&mut host, sector).unwrap();
        host.bind_residency(
            id,
            ResidencyBinding {
                sector,
                local: LocalEntityId::new(0),
            },
        )
        .unwrap();
    }
    let old_roots = [
        host.loaded_cell(first).unwrap().root.unwrap(),
        host.loaded_cell(second).unwrap().root.unwrap(),
    ];
    let binding = host.residency_binding(62).unwrap();
    let probe_entity = old_roots[1];
    host.world_mut().insert_resource(DespawnPanicProbe {
        entity: probe_entity,
        id: u64::MAX,
        binding,
        ownership_visible_in_hook: false,
    });
    host.world_mut().register_component_hooks::<Marker>(
        ComponentHooks::new().with_on_remove(panic_once_during_remove),
    );

    let panic = catch_unwind(AssertUnwindSafe(|| {
        host.reload_cells_atomic(&[first, second]).unwrap();
    }))
    .expect_err("retirement hook panic must propagate");
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"scene batch retirement hook panic")
    );
    assert_eq!(host.loaded_cell_count(), 2);
    assert!(old_roots.iter().all(|root| host.world().contains(*root)));
    assert_eq!(host.retired_cell_count(), 2);

    assert_eq!(host.retry_retired_cells(), 2);
    assert_eq!(host.retired_cell_count(), 0);
    assert!(old_roots.iter().all(|root| !host.world().contains(*root)));
}

#[test]
fn unload_hook_panic_preserves_ownership_for_handoff_and_plan_retry() {
    let mut world = World::new();
    let mut runtime = make_runtime();
    let sector = SectorId::new(0, 0, 0, 0);
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let root = load_bound_cell(&mut runtime, &mut host, sector, 41);
    let binding = host.residency_binding(41).unwrap();
    let plan = runtime.tick(
        &[],
        &[],
        &[ResidencyRecord::new(41, ResidencyState::TierA, 10, 1, 1, 1)],
        2,
        budget(),
    );
    assert_eq!(
        plan.stream.actions,
        alloc::vec![StreamAction::Unload(sector)]
    );
    assert!(matches!(
        plan.residency.actions.as_slice(),
        [ResidencyAction::Demote { id: 41, .. }]
    ));
    install_despawn_panic(&mut host, root, 41);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        runtime.apply_scene_plan(&plan, &mut host)
    }))
    .expect_err("the component hook panic must propagate");
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"scene despawn hook panic")
    );
    assert!(host.world().contains(root));
    assert!(host.loaded_cell(sector).is_some(), "unload lost its token");
    assert_eq!(host.residency_binding(41), Some(binding));
    assert!(
        host.world()
            .resource::<DespawnPanicProbe>()
            .ownership_visible_in_hook
    );

    let provider = host.into_provider();
    let mut replacement = SceneEcsHost::new(&mut world, provider);
    assert_eq!(replacement.loaded_cell(sector).unwrap().root, Some(root));
    assert_eq!(replacement.residency_binding(41), Some(binding));
    replacement
        .world_mut()
        .register_component_hooks::<Marker>(ComponentHooks::new());
    runtime.apply_scene_plan(&plan, &mut replacement).unwrap();
    assert!(!replacement.world().contains(root));
    assert_eq!(replacement.loaded_cell_count(), 0);
    assert_eq!(replacement.residency_binding(41), None);
    assert!(replacement.provider().events.is_empty());
}

#[test]
fn shutdown_hook_panic_preserves_failed_and_pending_cells_for_retry() {
    let mut world = World::new();
    let unrelated = world.spawn(Marker(17));
    let mut host = SceneEcsHost::new(&mut world, Provider::default());
    let mut sectors = [
        SectorId::new(0, 0, 0, 0),
        SectorId::new(0, 1, 0, 0),
        SectorId::new(0, 2, 0, 0),
    ];
    sectors.sort();
    let roots: Vec<_> = sectors
        .iter()
        .enumerate()
        .map(|(index, &sector)| {
            SceneHost::load_cell(&mut host, sector).unwrap();
            host.bind_residency(
                index as u64,
                ResidencyBinding {
                    sector,
                    local: LocalEntityId::new(0),
                },
            )
            .unwrap();
            host.loaded_cell(sector).unwrap().root.unwrap()
        })
        .collect();
    let orphan = ResidencyBinding {
        sector: SectorId::new(0, 99, 0, 0),
        local: LocalEntityId::new(0),
    };
    host.bind_residency(99, orphan).unwrap();
    install_despawn_panic(&mut host, roots[1], 1);
    let panic = catch_unwind(AssertUnwindSafe(|| host.shutdown()))
        .expect_err("the component hook panic must propagate");
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"scene despawn hook panic")
    );
    assert!(!host.world().contains(roots[0]));
    assert!(host.loaded_cell(sectors[0]).is_none());
    assert_eq!(host.residency_binding(0), None);
    for index in 1..3 {
        assert!(host.world().contains(roots[index]));
        assert!(
            host.loaded_cell(sectors[index]).is_some(),
            "shutdown lost pending token"
        );
        assert_eq!(
            host.residency_binding(index as u64).unwrap().sector,
            sectors[index]
        );
    }
    assert_eq!(host.loaded_cell_count(), 2);
    assert_eq!(host.residency_binding(99), Some(orphan));
    assert!(
        host.world()
            .resource::<DespawnPanicProbe>()
            .ownership_visible_in_hook
    );
    drop(host);

    let mut replacement = SceneEcsHost::new(&mut world, Provider::default());
    assert_eq!(replacement.loaded_cell_count(), 2);
    replacement
        .world_mut()
        .register_component_hooks::<Marker>(ComponentHooks::new());
    assert_eq!(replacement.shutdown(), 2);
    assert_eq!(replacement.shutdown(), 0);
    for id in [0, 1, 2, 99] {
        assert_eq!(replacement.residency_binding(id), None);
    }
    assert_eq!(replacement.world().entity_count(), 1);
    assert_eq!(
        replacement.world().get::<Marker>(unrelated),
        Some(&Marker(17))
    );
}
