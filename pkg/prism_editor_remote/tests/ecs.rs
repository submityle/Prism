//! Real Prism ECS integration; no substitute engine store.
#![cfg(feature = "ecs")]

use prism_ecs::{
    component::{Component, StorageType},
    world::World,
};
use prism_editor_remote::{
    ecs::{EcsSyncLimits, EcsSyncStore},
    Change, ComponentId, EntityId, PropertyPath, PropertySegment, Replica, Revision, SyncError,
    SyncStore, TargetId, Value,
};
use prism_reflect::Reflect;
use std::collections::BTreeMap;

#[derive(Clone, Reflect)]
struct Label {
    text: String,
    order: i32,
}
impl Component for Label {}

const TARGET: TargetId = TargetId::from_raw(0xe5);

#[test]
fn idle_incremental_queries_do_no_projection_work_after_real_remote_commit() {
    let mut world = World::new();
    let entity = world.spawn(label(1));
    for index in 2..4096 {
        world.spawn(label(index));
    }
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let component = store.register_component::<Label>();
    let id = store.entity_id(entity).unwrap();
    let base = store.synchronize().unwrap();
    let initial = store.projection_work();
    assert_eq!(initial.entities, 4095);
    assert_eq!(initial.components, 4095);
    for _ in 0..1000 {
        assert!(store.read_changes(TARGET, base).unwrap().changes.is_empty());
        assert_eq!(store.entity_id(entity).unwrap(), id);
    }
    assert_eq!(store.projection_work(), initial);
    let commit = store
        .apply_changes(
            TARGET,
            base,
            &[field(id, &component, "order", Value::Unsigned(42))],
        )
        .unwrap();
    assert_eq!(store.world().get::<Label>(entity).unwrap().order, 42);
    assert_eq!(store.projection_work(), initial);
    let delta = store.read_changes(TARGET, base).unwrap();
    assert_eq!(delta.target_revision, commit.revision);
    assert_eq!(delta.changes, commit.changes);
    assert!(store
        .read_changes(TARGET, commit.revision)
        .unwrap()
        .changes
        .is_empty());
    assert_eq!(store.projection_work(), initial);
    store
        .world_mut()
        .unwrap()
        .get_mut::<Label>(entity)
        .unwrap()
        .order = 43;
    let changed = store.read_changes(TARGET, commit.revision).unwrap();
    assert_eq!(changed.changes.len(), 1);
    assert_eq!(store.world().get::<Label>(entity).unwrap().order, 43);
    assert_eq!(store.projection_work().scans, initial.scans + 1);
    let refreshed = store.projection_work();
    assert!(store
        .read_changes(TARGET, changed.target_revision)
        .unwrap()
        .changes
        .is_empty());
    assert_eq!(store.projection_work(), refreshed);
}

#[test]
fn empty_initial_projection_and_failed_host_refresh_never_publish_stale_success() {
    let mut world = World::new();
    let limits = EcsSyncLimits {
        max_value_bytes: 256,
        ..EcsSyncLimits::default()
    };
    let mut store = EcsSyncStore::new(&mut world, TARGET, limits).unwrap();
    assert_eq!(store.synchronize().unwrap(), Revision::ZERO);
    let empty_work = store.projection_work();
    assert!(store.read_snapshot(TARGET).unwrap().entities.is_empty());
    assert_eq!(store.projection_work(), empty_work);
    let component = store.register_component::<Label>();
    let entity = store.world_mut().unwrap().spawn(label(1));
    let id = store.entity_id(entity).unwrap();
    let base = store.synchronize().unwrap();
    store
        .world_mut()
        .unwrap()
        .get_mut::<Label>(entity)
        .unwrap()
        .text = "x".repeat(1024);
    let before_failures = store.projection_work();
    assert!(store.read_changes(TARGET, base).is_err());
    assert!(store.read_snapshot(TARGET).is_err());
    assert!(store
        .apply_changes(
            TARGET,
            base,
            &[field(id, &component, "order", Value::Unsigned(8))]
        )
        .is_err());
    assert_eq!(store.projection_work().scans, before_failures.scans + 3);
    assert_eq!(store.world().get::<Label>(entity).unwrap().order, 1);
    store
        .world_mut()
        .unwrap()
        .get_mut::<Label>(entity)
        .unwrap()
        .text = "repaired".into();
    let repaired = store.read_snapshot(TARGET).unwrap();
    assert!(repaired.revision > base);
    assert_eq!(
        repaired.entities[&id].components[&component],
        Value::object([
            ("text", Value::String("repaired".into())),
            ("order", Value::Unsigned(1)),
        ])
        .unwrap()
    );
    let repaired_work = store.projection_work();
    assert!(store
        .read_changes(TARGET, repaired.revision)
        .unwrap()
        .changes
        .is_empty());
    assert_eq!(store.projection_work(), repaired_work);
}

#[test]
fn snapshot_reads_real_typed_columns_and_commit_writes_them() {
    let mut world = World::new();
    let entity = world.spawn(Label {
        text: "engine".into(),
        order: 3,
    });
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let component = store.register_component::<Label>();
    let id = store.entity_id(entity).unwrap();
    let before = store.read_snapshot(TARGET).unwrap();
    assert!(before.revision > Revision::ZERO);
    assert_eq!(
        before.entities[&id].components[&component],
        Value::object([
            ("text", Value::String("engine".into())),
            ("order", Value::Unsigned(3))
        ])
        .unwrap()
    );
    let result = store
        .apply_changes(
            TARGET,
            before.revision,
            &[Change::SetComponent {
                entity: id,
                component: component.clone(),
                value: Value::object([
                    ("text", Value::String("edited".into())),
                    ("order", Value::Unsigned(7)),
                ])
                .unwrap(),
            }],
        )
        .unwrap();
    assert!(result.revision > before.revision);
    assert_eq!(store.world().get::<Label>(entity).unwrap().text, "edited");
    assert_eq!(store.world().get::<Label>(entity).unwrap().order, 7);
    assert_eq!(
        store.read_changes(TARGET, before.revision).unwrap().changes,
        result.changes
    );
}

fn label(order: i32) -> Label {
    Label {
        text: "native 空白\n".into(),
        order,
    }
}
fn field(entity: EntityId, component: &ComponentId, name: &str, value: Value) -> Change {
    Change::SetProperty {
        entity,
        component: component.clone(),
        path: PropertyPath::new(vec![PropertySegment::Field(name.into())]).unwrap(),
        value,
    }
}

#[test]
fn late_conversion_failure_never_partially_writes_real_components() {
    let mut world = World::new();
    let a = world.spawn(label(1));
    let b = world.spawn(label(2));
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let component = store.register_component::<Label>();
    let aid = store.entity_id(a).unwrap();
    let bid = store.entity_id(b).unwrap();
    let before = store.read_snapshot(TARGET).unwrap();
    for invalid in [
        Value::Signed(i64::MAX),
        Value::Bool(true),
        Value::String("3".into()),
    ] {
        assert!(store
            .apply_changes(
                TARGET,
                before.revision,
                &[
                    field(aid, &component, "order", Value::Unsigned(99)),
                    field(bid, &component, "order", invalid),
                ]
            )
            .is_err());
        assert_eq!(store.world().get::<Label>(a).unwrap().order, 1);
        assert_eq!(store.world().get::<Label>(b).unwrap().order, 2);
        assert_eq!(store.read_snapshot(TARGET).unwrap(), before);
        assert!(store
            .read_changes(TARGET, before.revision)
            .unwrap()
            .changes
            .is_empty());
    }
    let result = store
        .apply_changes(
            TARGET,
            before.revision,
            &[
                field(aid, &component, "order", Value::Unsigned(8)),
                field(aid, &component, "order", Value::Unsigned(9)),
                field(bid, &component, "order", Value::Unsigned(10)),
            ],
        )
        .unwrap();
    assert_eq!(result.changes.len(), 2);
    assert_eq!(store.world().get::<Label>(a).unwrap().order, 9);
    assert_eq!(store.world().get::<Label>(b).unwrap().order, 10);
    let no_op = store
        .apply_changes(
            TARGET,
            result.revision,
            &[field(aid, &component, "order", Value::Unsigned(9))],
        )
        .unwrap();
    assert_eq!(no_op.revision, result.revision);
    assert!(no_op.changes.is_empty());
}

#[test]
fn structural_or_missing_component_failure_is_atomic() {
    let mut world = World::new();
    let a = world.spawn(label(1));
    let empty = world.spawn(());
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let component = store.register_component::<Label>();
    let aid = store.entity_id(a).unwrap();
    let empty = store.entity_id(empty).unwrap();
    let before = store.read_snapshot(TARGET).unwrap();
    let structural = [
        Change::DeleteEntity { entity: aid },
        Change::CreateEntity {
            entity: EntityId::from_raw(123),
        },
        Change::RemoveComponent {
            entity: aid,
            component: component.clone(),
        },
        Change::SetComponent {
            entity: empty,
            component: component.clone(),
            value: before.entities[&aid].components[&component].clone(),
        },
    ];
    for command in structural {
        assert!(store
            .apply_changes(
                TARGET,
                before.revision,
                &[field(aid, &component, "order", Value::Unsigned(7)), command]
            )
            .is_err());
        assert_eq!(store.read_snapshot(TARGET).unwrap(), before);
        assert_eq!(store.world().get::<Label>(a).unwrap().order, 1);
    }
}

#[derive(Clone, Reflect, Debug, PartialEq)]
enum Choice {
    Idle,
    Pair(i32, String),
    Named { active: bool },
}
#[derive(Clone, Reflect)]
struct Collection {
    list: Vec<i32>,
    array: [u16; 2],
    map: BTreeMap<String, String>,
    choice: Choice,
}
impl Component for Collection {}

#[test]
fn nested_containers_enum_switch_and_map_removal_are_exact() {
    let mut world = World::new();
    let e = world.spawn(Collection {
        list: vec![1, 2, 3],
        array: [4, 5],
        map: BTreeMap::from([("old".into(), "value".into())]),
        choice: Choice::Idle,
    });
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let component = store.register_component::<Collection>();
    store.register_value_type::<Vec<i32>>();
    store.register_value_type::<[u16; 2]>();
    store.register_value_type::<BTreeMap<String, String>>();
    store.register_value_type::<Choice>();
    let id = store.entity_id(e).unwrap();
    let base = store.synchronize().unwrap();
    let variant = |name: &str, fields| {
        Value::object([("variant", Value::String(name.into())), ("fields", fields)]).unwrap()
    };
    let result = store
        .apply_changes(
            TARGET,
            base,
            &[
                field(
                    id,
                    &component,
                    "list",
                    Value::Array(vec![Value::Unsigned(9)]),
                ),
                field(
                    id,
                    &component,
                    "choice",
                    variant(
                        "Pair",
                        Value::Array(vec![Value::Signed(-3), Value::String("新\n".into())]),
                    ),
                ),
                Change::RemoveProperty {
                    entity: id,
                    component: component.clone(),
                    path: PropertyPath::new(vec![
                        PropertySegment::Field("map".into()),
                        PropertySegment::Key("old".into()),
                    ])
                    .unwrap(),
                },
            ],
        )
        .unwrap();
    let actual = store.world().get::<Collection>(e).unwrap();
    assert_eq!(actual.list, [9]);
    assert!(actual.map.is_empty());
    assert_eq!(actual.choice, Choice::Pair(-3, "新\n".into()));
    store
        .apply_changes(
            TARGET,
            result.revision,
            &[
                field(
                    id,
                    &component,
                    "choice",
                    variant(
                        "Named",
                        Value::object([("active", Value::Bool(true))]).unwrap(),
                    ),
                ),
                field(
                    id,
                    &component,
                    "list",
                    Value::Array(vec![
                        Value::Unsigned(1),
                        Value::Unsigned(2),
                        Value::Unsigned(3),
                        Value::Unsigned(4),
                    ]),
                ),
            ],
        )
        .unwrap();
    assert_eq!(
        store.world().get::<Collection>(e).unwrap().choice,
        Choice::Named { active: true }
    );
    assert_eq!(
        store.world().get::<Collection>(e).unwrap().list,
        [1, 2, 3, 4]
    );
    let before = store.read_snapshot(TARGET).unwrap();
    for bad in [
        field(
            id,
            &component,
            "array",
            Value::Array(vec![Value::Unsigned(1)]),
        ),
        field(id, &component, "choice", variant("Missing", Value::Null)),
        field(
            id,
            &component,
            "choice",
            variant("Idle", Value::Array(vec![])),
        ),
        Change::RemoveProperty {
            entity: id,
            component: component.clone(),
            path: PropertyPath::new(vec![PropertySegment::Field("list".into())]).unwrap(),
        },
    ] {
        assert!(store
            .apply_changes(TARGET, before.revision, &[bad])
            .is_err());
        assert_eq!(store.read_snapshot(TARGET).unwrap(), before);
    }
}

#[derive(Clone, Reflect)]
struct Numbers {
    single: f32,
    double: f64,
    signed: i128,
    unsigned: u128,
    letter: char,
    small: u8,
}
impl Component for Numbers {}

#[test]
fn numeric_width_bits_unicode_and_ranges_are_preserved() {
    let mut world = World::new();
    let e = world.spawn(Numbers {
        single: f32::from_bits(0x7fc01234),
        double: f64::from_bits(0x7ff8000000001234),
        signed: i128::MIN,
        unsigned: u128::MAX,
        letter: '界',
        small: 255,
    });
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let component = store.register_component::<Numbers>();
    let id = store.entity_id(e).unwrap();
    let before = store.read_snapshot(TARGET).unwrap();
    let Value::Object(value) = &before.entities[&id].components[&component] else {
        panic!("numeric struct")
    };
    assert_eq!(value["single"], Value::Unsigned(0x7fc01234));
    assert_eq!(value["double"], Value::Float(0x7ff8000000001234));
    assert_eq!(
        value["signed"],
        Value::Bytes(i128::MIN.to_le_bytes().to_vec())
    );
    assert_eq!(
        value["unsigned"],
        Value::Bytes(u128::MAX.to_le_bytes().to_vec())
    );
    let commit = store
        .apply_changes(
            TARGET,
            before.revision,
            &[
                field(
                    id,
                    &component,
                    "single",
                    Value::Unsigned(u64::from((-0.0f32).to_bits())),
                ),
                field(id, &component, "double", Value::Float((-0.0f64).to_bits())),
                field(id, &component, "letter", Value::String("🦀".into())),
            ],
        )
        .unwrap();
    let actual = store.world().get::<Numbers>(e).unwrap();
    assert_eq!(actual.single.to_bits(), (-0.0f32).to_bits());
    assert_eq!(actual.double.to_bits(), (-0.0f64).to_bits());
    assert_eq!(actual.letter, '🦀');
    assert_eq!(actual.signed, i128::MIN);
    assert_eq!(actual.unsigned, u128::MAX);
    let before = store.read_snapshot(TARGET).unwrap();
    for (field_name, value) in [
        ("small", Value::Unsigned(256)),
        ("single", Value::Unsigned(u64::MAX)),
        ("letter", Value::String("ab".into())),
        ("signed", Value::Bytes(vec![0; 15])),
        ("double", Value::Unsigned(0)),
    ] {
        assert!(store
            .apply_changes(
                TARGET,
                commit.revision,
                &[field(id, &component, field_name, value)]
            )
            .is_err());
        assert_eq!(store.read_snapshot(TARGET).unwrap(), before);
    }
}

#[derive(Clone, Reflect)]
struct Sparse(i32);
impl Component for Sparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[test]
fn host_structural_edits_and_sparse_writes_replicate_actual_world() {
    let mut world = World::new();
    let e = world.spawn((label(1), Sparse(2)));
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let component = store.register_component::<Label>();
    let sparse = store.register_component::<Sparse>();
    let id = store.entity_id(e).unwrap();
    let before = store.read_snapshot(TARGET).unwrap();
    let mut replica = Replica::new(TARGET);
    replica.install_snapshot(before.clone()).unwrap();
    let result = store
        .apply_changes(
            TARGET,
            before.revision,
            &[Change::SetComponent {
                entity: id,
                component: sparse,
                value: Value::Array(vec![Value::Unsigned(5)]),
            }],
        )
        .unwrap();
    assert_eq!(store.world().get::<Sparse>(e).unwrap().0, 5);
    replica
        .apply_changes(&store.read_changes(TARGET, before.revision).unwrap())
        .unwrap();
    let world = store.world_mut().unwrap();
    world.remove::<Label>(e);
    let other = world.spawn(label(9));
    let delta = store.read_changes(TARGET, result.revision).unwrap();
    assert!(delta
        .changes
        .iter()
        .any(|c| matches!(c,Change::RemoveComponent { component:c,..} if c==&component)));
    replica.apply_changes(&delta).unwrap();
    assert_eq!(
        replica.snapshot().unwrap(),
        &store.read_snapshot(TARGET).unwrap()
    );
    store.world_mut().unwrap().despawn(other);
    let next = store.read_changes(TARGET, delta.target_revision).unwrap();
    assert!(next
        .changes
        .iter()
        .any(|c| matches!(c, Change::DeleteEntity { .. })));
    replica.apply_changes(&next).unwrap();
    assert_eq!(
        replica.snapshot().unwrap(),
        &store.read_snapshot(TARGET).unwrap()
    );
}

#[test]
fn host_aba_barrier_stale_revision_and_history_pruning_are_explicit() {
    let mut world = World::new();
    let e = world.spawn(label(1));
    let limits = EcsSyncLimits {
        history_batches: 1,
        ..EcsSyncLimits::default()
    };
    let mut store = EcsSyncStore::new(&mut world, TARGET, limits).unwrap();
    let c = store.register_component::<Label>();
    let id = store.entity_id(e).unwrap();
    let old = store.synchronize().unwrap();
    store
        .world_mut()
        .unwrap()
        .get_mut::<Label>(e)
        .unwrap()
        .order = 1;
    assert!(matches!(
        store.apply_changes(TARGET, old, &[field(id, &c, "order", Value::Unsigned(7))]),
        Err(SyncError::RevisionConflict { .. })
    ));
    assert!(matches!(
        store.read_changes(TARGET, old),
        Err(SyncError::RevisionUnavailable { .. })
    ));
    let base = store.synchronize().unwrap();
    let first = store
        .apply_changes(TARGET, base, &[field(id, &c, "order", Value::Unsigned(2))])
        .unwrap();
    let second = store
        .apply_changes(
            TARGET,
            first.revision,
            &[field(id, &c, "order", Value::Unsigned(3))],
        )
        .unwrap();
    assert!(matches!(
        store.read_changes(TARGET, base),
        Err(SyncError::RevisionUnavailable { .. })
    ));
    assert_eq!(
        store
            .read_changes(TARGET, first.revision)
            .unwrap()
            .target_revision,
        second.revision
    );
    assert!(matches!(
        store.read_changes(TARGET, Revision::from_raw(u64::MAX)),
        Err(SyncError::RevisionConflict { .. })
    ));
    assert!(matches!(
        store.read_snapshot(TargetId::from_raw(999)),
        Err(SyncError::UnknownTarget { .. })
    ));
}

#[test]
fn recycled_entity_restore_and_new_adapter_reject_old_wire_identities() {
    let mut world = World::new();
    world.register_snapshot_component::<Label>();
    let e = world.spawn(label(1));
    let saved = world.snapshot();
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let c = store.register_component::<Label>();
    let old = store.entity_id(e).unwrap();
    let host = store.world_mut().unwrap();
    host.despawn(e);
    let replacement = host.spawn(label(2));
    assert_eq!(replacement.index(), e.index());
    assert_ne!(replacement.generation(), e.generation());
    let current = store.synchronize().unwrap();
    assert!(matches!(
        store.apply_changes(
            TARGET,
            current,
            &[field(old, &c, "order", Value::Unsigned(99))]
        ),
        Err(SyncError::MissingEntity { .. })
    ));
    store.world_mut().unwrap().restore(&saved);
    let restored = store.entity_id(e).unwrap();
    assert_ne!(old, restored);
    let current = store.synchronize().unwrap();
    assert!(matches!(
        store.apply_changes(
            TARGET,
            current,
            &[field(old, &c, "order", Value::Unsigned(99))]
        ),
        Err(SyncError::MissingEntity { .. })
    ));
    assert_eq!(store.world().get::<Label>(e).unwrap().order, 1);
    drop(store);
    let mut new_store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    new_store.register_component::<Label>();
    assert_ne!(new_store.entity_id(e).unwrap(), restored);
}

#[test]
fn budgets_reject_requests_and_native_scan_without_stale_success() {
    let mut world = World::new();
    let e = world.spawn(label(1));
    let limits = EcsSyncLimits {
        max_entities: 1,
        max_changes: 1,
        max_value_bytes: 256,
        max_value_nodes: 16,
        max_depth: 4,
        ..EcsSyncLimits::default()
    };
    let mut store = EcsSyncStore::new(&mut world, TARGET, limits).unwrap();
    let c = store.register_component::<Label>();
    let id = store.entity_id(e).unwrap();
    let before = store.read_snapshot(TARGET).unwrap();
    for batch in [
        vec![field(id, &c, "text", Value::String("x".repeat(257)))],
        vec![field(id, &c, "order", Value::Unsigned(2)); 2],
        vec![field(id, &c, "order", Value::Array(vec![Value::Null; 17]))],
    ] {
        assert!(store
            .apply_changes(TARGET, before.revision, &batch)
            .is_err());
        assert_eq!(store.read_snapshot(TARGET).unwrap(), before);
    }
    let extra = store.world_mut().unwrap().spawn(());
    assert!(store.read_snapshot(TARGET).is_err());
    assert!(store.read_changes(TARGET, before.revision).is_err());
    assert!(store
        .apply_changes(
            TARGET,
            before.revision,
            &[field(id, &c, "order", Value::Unsigned(7))]
        )
        .is_err());
    store.world_mut().unwrap().despawn(extra);
    assert!(store.read_snapshot(TARGET).unwrap().revision > before.revision);
    assert_eq!(store.world().get::<Label>(e).unwrap().order, 1);
}

#[derive(Clone, Reflect, Hash, Eq, PartialEq)]
struct Shared(u32);
impl Component for Shared {
    const STORAGE: StorageType = StorageType::Shared;
    fn install_storage_glue(
        components: &mut prism_ecs::component::Components,
        id: prism_ecs::component::ComponentId,
    ) {
        components.set_shared_box(id, prism_ecs::component::shared_box_of::<Self>());
    }
}

#[test]
fn shared_storage_is_readable_and_never_partially_written() {
    let mut world = World::new();
    let e = world.spawn((label(1), Shared(7)));
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let label = store.register_component::<Label>();
    let shared = store.register_component::<Shared>();
    let id = store.entity_id(e).unwrap();
    let before = store.read_snapshot(TARGET).unwrap();
    assert_eq!(
        before.entities[&id].components[&shared],
        Value::Array(vec![Value::Unsigned(7)])
    );
    assert!(store
        .apply_changes(
            TARGET,
            before.revision,
            &[
                field(id, &label, "order", Value::Unsigned(9)),
                Change::SetComponent {
                    entity: id,
                    component: shared,
                    value: Value::Array(vec![Value::Unsigned(2)])
                }
            ]
        )
        .is_err());
    assert_eq!(store.world().get::<Label>(e).unwrap().order, 1);
    assert_eq!(store.world().get::<Shared>(e).unwrap().0, 7);
    assert_eq!(store.read_snapshot(TARGET).unwrap(), before);
}

#[test]
fn inplace_commit_marks_engine_ticks_without_structural_hooks() {
    use prism_ecs::component_hooks::ComponentHooks;
    let mut world = World::new();
    let e = world.spawn(label(1));
    world.register_component_hooks::<Label>(
        ComponentHooks::new()
            .with_on_insert(|_| panic!("unexpected structural insert"))
            .with_on_replace(|_| panic!("unexpected structural replace")),
    );
    let old = world.get_ticks::<Label>(e).unwrap();
    let tick = world.increment_change_tick();
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let c = store.register_component::<Label>();
    let id = store.entity_id(e).unwrap();
    let base = store.synchronize().unwrap();
    store
        .apply_changes(TARGET, base, &[field(id, &c, "order", Value::Unsigned(42))])
        .unwrap();
    let updated = store.world().get_ticks::<Label>(e).unwrap();
    assert_eq!(updated.added, old.added);
    assert_eq!(updated.changed, tick);
}

#[test]
fn pending_reservations_and_world_replacement_are_not_silently_projected() {
    let mut world = World::new();
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let reserved = store.world().entities().reserve_entity();
    assert!(store.read_snapshot(TARGET).is_err());
    store.world_mut().unwrap().flush_reserved();
    assert!(store.read_snapshot(TARGET).is_err());
    store.world_mut().unwrap().spawn_at(reserved, ());
    assert!(store
        .read_snapshot(TARGET)
        .unwrap()
        .entities
        .contains_key(&store.entity_id(reserved).unwrap()));
    *store.world_mut().unwrap() = World::new();
    assert!(store.read_snapshot(TARGET).is_err());
    assert!(store.world_mut().is_err());
}

#[derive(Clone, Reflect)]
struct NotMapped(std::collections::BTreeSet<i32>);
impl Component for NotMapped {}

#[test]
fn unsupported_shapes_and_unregistered_nested_schema_fail_explicitly() {
    let mut world = World::new();
    world.spawn(NotMapped(Default::default()));
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    store.register_component::<NotMapped>();
    assert!(store
        .read_snapshot(TARGET)
        .unwrap_err()
        .to_string()
        .contains("unsupported"));
    drop(store);
    let mut world = World::new();
    let e = world.spawn(Sparse(1));
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let c = store.register_component::<Sparse>();
    let id = store.entity_id(e).unwrap();
    let before = store.read_snapshot(TARGET).unwrap();
    let forged = ComponentId::from_static("project::Other");
    assert!(store
        .apply_changes(
            TARGET,
            before.revision,
            &[Change::SetComponent {
                entity: id,
                component: forged,
                value: Value::Unsigned(2)
            }]
        )
        .is_err());
    assert!(store
        .apply_changes(
            TARGET,
            before.revision,
            &[Change::SetComponent {
                entity: id,
                component: c,
                value: Value::Array(vec![Value::Unsigned(2), Value::Unsigned(3)])
            }]
        )
        .is_err());
    assert_eq!(store.read_snapshot(TARGET).unwrap(), before);
}

#[derive(Clone, Reflect)]
struct Nested(Vec<i32>);
impl Component for Nested {}

#[test]
fn nested_schema_registration_is_required_before_typed_writes() {
    let mut world = World::new();
    let e = world.spawn(Nested(vec![1, 2]));
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let component = store.register_component::<Nested>();
    let id = store.entity_id(e).unwrap();
    let before = store.read_snapshot(TARGET).unwrap();
    let changes = [Change::SetComponent {
        entity: id,
        component,
        value: Value::Array(vec![Value::Array(vec![Value::Unsigned(3)])]),
    }];
    let error = store
        .apply_changes(TARGET, before.revision, &changes)
        .unwrap_err();
    assert!(error.to_string().contains("unregistered nested ECS type"));
    assert_eq!(store.world().get::<Nested>(e).unwrap().0, [1, 2]);
    assert_eq!(store.read_snapshot(TARGET).unwrap(), before);
    store.register_value_type::<Vec<i32>>();
    store
        .apply_changes(TARGET, before.revision, &changes)
        .unwrap();
    assert_eq!(store.world().get::<Nested>(e).unwrap().0, [3]);
}

#[derive(Clone, Reflect)]
struct MapValue(BTreeMap<String, String>);
impl Component for MapValue {}

#[test]
fn reserved_json_tags_are_rejected_without_lossy_map_roundtrip() {
    let mut world = World::new();
    let e = world.spawn(MapValue(BTreeMap::new()));
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    let component = store.register_component::<MapValue>();
    store.register_value_type::<BTreeMap<String, String>>();
    let id = store.entity_id(e).unwrap();
    for key in ["$prism.bytes", "$prism.floatBits"] {
        let before = store.read_snapshot(TARGET).unwrap();
        let changes = [Change::SetComponent {
            entity: id,
            component: component.clone(),
            value: Value::Array(vec![
                Value::object([(key, Value::String("01".into()))]).unwrap()
            ]),
        }];
        assert!(store
            .apply_changes(TARGET, before.revision, &changes)
            .unwrap_err()
            .to_string()
            .contains("reserved JSON-RPC"));
        assert!(store.world().get::<MapValue>(e).unwrap().0.is_empty());
        assert_eq!(store.read_snapshot(TARGET).unwrap(), before);
        store
            .world_mut()
            .unwrap()
            .get_mut::<MapValue>(e)
            .unwrap()
            .0
            .insert(key.into(), "01".into());
        assert!(store
            .read_snapshot(TARGET)
            .unwrap_err()
            .to_string()
            .contains("reserved JSON-RPC"));
        store
            .world_mut()
            .unwrap()
            .get_mut::<MapValue>(e)
            .unwrap()
            .0
            .clear();
        store.synchronize().unwrap();
    }
}

#[test]
fn retained_deltas_obey_aggregate_node_and_byte_budgets() {
    for (max_value_nodes, max_value_bytes) in [(5, 1024), (128, 512)] {
        let mut world = World::new();
        let e = world.spawn(label(1));
        let limits = EcsSyncLimits {
            max_value_nodes,
            max_value_bytes,
            ..EcsSyncLimits::default()
        };
        let mut store = EcsSyncStore::new(&mut world, TARGET, limits).unwrap();
        let c = store.register_component::<Label>();
        let id = store.entity_id(e).unwrap();
        let base = store.synchronize().unwrap();
        let first = store
            .apply_changes(
                TARGET,
                base,
                &[field(id, &c, "text", Value::String("a".repeat(300)))],
            )
            .unwrap();
        let second = store
            .apply_changes(
                TARGET,
                first.revision,
                &[field(id, &c, "text", Value::String("b".repeat(300)))],
            )
            .unwrap();
        assert!(store
            .read_changes(TARGET, base)
            .unwrap_err()
            .to_string()
            .contains("budget"));
        assert_eq!(
            store
                .read_changes(TARGET, first.revision)
                .unwrap()
                .target_revision,
            second.revision
        );
        assert_eq!(
            store.read_snapshot(TARGET).unwrap().revision,
            second.revision
        );
        assert_eq!(store.world().get::<Label>(e).unwrap().text, "b".repeat(300));
    }
}

fn json(
    entries: impl IntoIterator<Item = (&'static str, prism_editor_remote::JsonValue)>,
) -> prism_editor_remote::JsonValue {
    prism_editor_remote::JsonValue::Object(
        entries.into_iter().map(|(k, v)| (k.into(), v)).collect(),
    )
}
fn json_fields(
    value: &prism_editor_remote::JsonValue,
) -> &BTreeMap<String, prism_editor_remote::JsonValue> {
    let prism_editor_remote::JsonValue::Object(object) = value else {
        panic!("JSON object: {value:?}")
    };
    object
}

struct ActualSession<'w>(
    prism_editor_remote::JsonRpcServer<prism_editor_remote::SyncEndpoint<EcsSyncStore<'w>>>,
);
impl prism_editor_remote::RuntimeSession for ActualSession<'_> {
    fn request(
        &mut self,
        message: prism_editor_remote::JsonRpcMessage,
        _: std::time::Duration,
    ) -> Result<prism_editor_remote::JsonRpcExchange, prism_editor_remote::RuntimeError> {
        let output = self
            .0
            .handle_bytes(&prism_editor_remote::encode_json_rpc(&message))
            .unwrap();
        let prism_editor_remote::JsonRpcMessage::Response(response) =
            prism_editor_remote::decode_json_rpc(&output[0]).unwrap()
        else {
            panic!("response")
        };
        Ok(prism_editor_remote::JsonRpcExchange {
            response,
            notifications: vec![],
        })
    }
}

#[test]
fn runtime_adapter_negotiates_real_ecs_and_rejects_unknown_target_and_zero_deadline() {
    use prism_editor_remote::{
        ecs::PrismEcsRuntimeAdapter, EngineRuntimeAdapter, JsonRpcServer, RuntimeError,
        SyncEndpoint,
    };
    use std::time::Duration;
    for target in [TARGET, TargetId::from_raw(123)] {
        let mut world = World::new();
        world.spawn(label(1));
        let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
        store.register_component::<Label>();
        let mut session = ActualSession(JsonRpcServer::new(SyncEndpoint::new(store)));
        let adapter = PrismEcsRuntimeAdapter::new(target);
        assert_eq!(adapter.engine_id(), "prism-ecs");
        assert!(matches!(
            adapter.initialize(&mut session, Duration::ZERO),
            Err(RuntimeError::Timeout { .. })
        ));
        assert_eq!(
            adapter
                .initialize(&mut session, Duration::from_secs(1))
                .is_ok(),
            target == TARGET
        );
    }
}

const ECS_CHILD_ADDRESS: &str = "PRISM_EDITOR_REMOTE_REAL_ECS_TEST_ADDRESS";

// This is the actual engine host entrypoint invoked by RuntimeBridge below.
// Without the private launch argument this test has no child work to perform.
#[test]
fn ecs_runtime_host() {
    use prism_editor_remote::{JsonRpcServer, JsonRpcServerSession, SyncEndpoint, TransportLimits};
    let Some(address) = std::env::var_os(ECS_CHILD_ADDRESS) else {
        return;
    };
    let listener = std::net::TcpListener::bind(address.to_str().unwrap()).unwrap();
    let (stream, _) = listener.accept().unwrap();
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    stream
        .set_write_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    let mut world = World::new();
    let e = world.spawn(label(1));
    let mut store = EcsSyncStore::new(&mut world, TARGET, EcsSyncLimits::default()).unwrap();
    store.register_component::<Label>();
    let mut session = JsonRpcServerSession::new(
        JsonRpcServer::new(SyncEndpoint::new(store)),
        stream,
        TransportLimits::default(),
    );
    let mut frames = 0;
    while session.serve_one().unwrap() {
        frames += 1;
        if frames == 4 {
            assert_eq!(
                session
                    .server()
                    .endpoint()
                    .store()
                    .world()
                    .get::<Label>(e)
                    .unwrap()
                    .order,
                42
            );
        }
        if frames == 5 {
            session
                .server_mut()
                .endpoint_mut()
                .store_mut()
                .world_mut()
                .unwrap()
                .get_mut::<Label>(e)
                .unwrap()
                .order = 43;
        }
    }
}

#[test]
#[cfg(unix)]
fn runtime_bridge_edits_real_typed_ecs_in_a_child_process() {
    use prism_editor_remote::{
        ecs::PrismEcsRuntimeAdapter, request, JsonNumber, JsonRpcMessage, JsonValue, RuntimeBridge,
        RuntimeLaunchSpec, RuntimeLimits, RuntimeState, RuntimeTransport,
    };
    use std::time::Duration;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let mut spec = RuntimeLaunchSpec::new("/bin/sh", RuntimeTransport::Tcp { address });
    spec.args = vec![
        "-c".into(),
        // TCP hosts must not write to the runtime's unused stdout pipe.
        // libtest writes a banner before running the real ECS host below.
        "exec \"$1\" --exact ecs_runtime_host --nocapture >/dev/null".into(),
        "prism-ecs-test-host".into(),
        std::env::current_exe().unwrap().into_os_string(),
    ];
    spec.env
        .insert(ECS_CHILD_ADDRESS.into(), address.to_string().into());
    let limits = RuntimeLimits {
        startup_timeout: Duration::from_secs(3),
        shutdown_timeout: Duration::from_secs(3),
        ..RuntimeLimits::default()
    };
    let mut bridge = RuntimeBridge::new(PrismEcsRuntimeAdapter::new(TARGET), spec, limits);
    assert_eq!(bridge.start().unwrap().state, RuntimeState::Running);
    let mut call = |message: JsonRpcMessage| {
        let result = bridge
            .request(message)
            .unwrap()
            .wait_timeout(Duration::from_secs(3))
            .unwrap()
            .response;
        assert!(result.error.is_none(), "{:?}", result.error);
        result.result.unwrap()
    };
    let target = || JsonValue::String(TARGET.raw().to_string());
    let snapshot = call(request(
        3,
        "prism/world/readSnapshot",
        Some(json([("target", target())])),
    ));
    let entities = json_fields(&json_fields(&snapshot)["entities"]);
    let (entity, components) = entities.first_key_value().unwrap();
    let component = core::any::type_name::<Label>();
    assert_eq!(
        json_fields(&json_fields(components)[component])["order"],
        JsonValue::Number(JsonNumber::Unsigned(1))
    );
    let base = json_fields(&snapshot)["revision"].clone();
    let commit = call(request(
        4,
        "prism/world/applyChanges",
        Some(json([
            ("target", target()),
            ("baseRevision", base.clone()),
            (
                "changes",
                JsonValue::Array(vec![json([
                    ("op", JsonValue::String("setProperty".into())),
                    ("entity", JsonValue::String(entity.clone())),
                    ("component", JsonValue::String(component.into())),
                    (
                        "path",
                        JsonValue::Array(vec![json([
                            ("kind", JsonValue::String("field".into())),
                            ("value", JsonValue::String("order".into())),
                        ])]),
                    ),
                    ("value", JsonValue::Number(JsonNumber::Unsigned(42))),
                ])]),
            ),
        ])),
    ));
    assert_ne!(json_fields(&commit)["revision"], base);
    let changes = call(request(
        5,
        "prism/world/readChanges",
        Some(json([("target", target()), ("fromRevision", base)])),
    ));
    assert!(
        matches!(&json_fields(&changes)["changes"],JsonValue::Array(changes) if !changes.is_empty())
    );
    let after = call(request(
        6,
        "prism/world/readSnapshot",
        Some(json([("target", target())])),
    ));
    let entities = json_fields(&json_fields(&after)["entities"]);
    assert_eq!(
        json_fields(&json_fields(&entities[entity])[component])["order"],
        JsonValue::Number(JsonNumber::Unsigned(43))
    );
    assert_eq!(bridge.stop().unwrap().state, RuntimeState::Stopped);
}
