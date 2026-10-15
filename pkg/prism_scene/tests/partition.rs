//! Real scene-owned persistence consumes the kernel's streaming driver.
#![cfg(feature = "partition")]

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use prism_ecs::{
    component::{Component, ComponentId, Components, StorageType},
    entity::Entity,
    observer::LifecycleEvent,
    partition::{
        cell::{CellCoord, CellState},
        data_layer::{DataLayerId, DataLayerState},
        floating_origin::WorldPos,
        streaming::WeakEntity,
    },
    reflect_archive::ReflectArchiveError,
    relation::{CleanupPolicy, RelationKind},
    resource::Resource,
    world::World,
};
use prism_reflect::{ParsedPath, Reflect, StableTypeId};
use prism_scene::{
    partition::{
        CellOperation, CellStore, FileCellStore, PartitionError, PersistentEntityRef,
        PersistentPartition,
    },
    PersistentId,
};

const A: CellCoord = CellCoord::ORIGIN;
const B: CellCoord = CellCoord::new(3, 0, 0);
const C: CellCoord = CellCoord::new(6, 0, 0);

#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
struct Value {
    number: i32,
    label: String,
}
impl Component for Value {}

#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
struct Sparse {
    number: u64,
}
impl Component for Sparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
struct Target {
    reference: PersistentEntityRef,
}
impl Component for Target {}

struct Parent;
impl Component for Parent {}

struct UnreflectedSparse;
impl Component for UnreflectedSparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[derive(Reflect, Clone, Debug, PartialEq, Eq, Hash)]
struct Shared {
    number: i32,
}
impl Component for Shared {
    const STORAGE: StorageType = StorageType::Shared;
    fn install_storage_glue(components: &mut Components, id: ComponentId) {
        components.set_shared_box(id, prism_ecs::component::shared_box_of::<Self>());
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Global(u64);
impl Resource for Global {}

// Every test owns a unique directory; no shared build cache or fixture is used.
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "prism-scene-partition-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create test directory: {error}"),
            }
        }
    }
    fn path(&self) -> &Path {
        &self.0
    }
    fn store(&self) -> FileCellStore {
        FileCellStore::new(self.path().join("cells"))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn id(value: u128) -> PersistentId {
    PersistentId::from_u128(value)
}

fn value(number: i32) -> Value {
    Value {
        number,
        label: format!("value-{number}"),
    }
}

fn kind(policy: CleanupPolicy) -> RelationKind {
    RelationKind::new()
        .with_fragmenting(true)
        .with_transitive(true)
        .with_exclusive(true)
        .with_on_delete_target(policy)
}

fn world() -> World {
    let mut world = World::new();
    world.register_reflect_component::<Value>();
    world.register_reflect_component::<Sparse>();
    world.register_reflect_component::<Target>();
    world.register_relation::<Parent>(kind(CleanupPolicy::Delete));
    world
}

fn owner(world: &World, scratch: &Scratch) -> PersistentPartition<FileCellStore> {
    PersistentPartition::new(world, scratch.store(), 100.0, 0, 0)
}

fn state(owner: &PersistentPartition<FileCellStore>, cell: CellCoord) -> CellState {
    owner.partition().driver().streamer().state(cell)
}

#[test]
fn file_roundtrip_preserves_table_sparse_empty_entities_and_world_resources() {
    let scratch = Scratch::new();
    let mut world = world();
    world.insert_resource(Global(771));
    let original = world.spawn((value(42), Sparse { number: 99 }));
    let empty = world.spawn(());
    let weak = WeakEntity::new(original);
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, original, id(1)).unwrap();
    owner.register(&world, A, empty, id(2)).unwrap();

    let despawned = owner.unload_cell(&mut world, A).unwrap();
    assert_eq!(despawned.len(), 2);
    assert!(!world.contains(empty));
    assert!(!weak.is_alive(&world));
    assert_eq!(owner.resolve(&world, id(1)), None);
    assert!(owner.partition().index().entities_in(A).is_empty());
    assert_eq!(state(&owner, A), CellState::Unloaded);
    assert!(owner.store().cell_path(A).is_file());
    assert_eq!(world.get_resource::<Global>(), Some(&Global(771)));

    let restored = owner.load_cell(&mut world, A).unwrap();
    assert_eq!(restored.len(), 2);
    let fresh = owner.resolve(&world, id(1)).unwrap();
    assert_ne!(fresh, original);
    assert_eq!(world.get::<Value>(fresh), Some(&value(42)));
    assert_eq!(world.get::<Sparse>(fresh), Some(&Sparse { number: 99 }));
    let fresh_empty = owner.resolve(&world, id(2)).unwrap();
    assert!(world.contains(fresh_empty));
    assert!(world.reflect_entity(fresh_empty).is_empty());
    assert_eq!(state(&owner, A), CellState::Loaded);
    assert_eq!(owner.partition().index().cell_of(fresh), Some(A));
    assert_eq!(world.get_resource::<Global>(), Some(&Global(771)));
    // File replacement is complete and leaves no orphan temporary files.
    owner.save_cell(&world, A).unwrap();
    assert_eq!(fs::read_dir(owner.store().root()).unwrap().count(), 1);
}

#[test]
fn loading_into_a_fresh_world_never_overwrites_occupied_runtime_handles() {
    let scratch = Scratch::new();
    let mut source = world();
    let archived = source.spawn(value(9));
    let mut writer = owner(&source, &scratch);
    writer.register(&source, A, archived, id(1)).unwrap();
    writer.save_cell(&source, A).unwrap();

    let mut destination = world();
    let occupied = destination.spawn(value(-9));
    assert_eq!(occupied, archived, "exercise source-handle collision");
    let mut reader = owner(&destination, &scratch);
    reader.load_cell(&mut destination, A).unwrap();
    let fresh = reader.resolve(&destination, id(1)).unwrap();
    assert_ne!(fresh, occupied);
    assert_eq!(destination.get::<Value>(occupied), Some(&value(-9)));
    assert_eq!(destination.get::<Value>(fresh), Some(&value(9)));
}

#[test]
fn production_advance_saves_evicts_loads_and_retries_missing_cells() {
    let scratch = Scratch::new();
    let mut world = world();
    let a = world.spawn(value(1));
    let b = world.spawn(value(2));
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    owner.register(&world, B, b, id(2)).unwrap();
    owner.unload_cell(&mut world, B).unwrap();
    let tick = owner
        .advance(&mut world, WorldPos::new(310.0, 0.0, 0.0), &[])
        .unwrap();
    assert!(tick.failed.is_empty());
    assert_eq!(tick.unloaded, vec![A]);
    assert_eq!(tick.despawned, vec![a]);
    assert_eq!(tick.loaded.len(), 1);
    assert_eq!(tick.loaded[0].0, B);
    assert_eq!(state(&owner, B), CellState::Loaded);
    assert_eq!(owner.partition().origin().origin().x, 3);
    assert_eq!(
        world.get::<Value>(owner.resolve(&world, id(2)).unwrap()),
        Some(&value(2))
    );

    for _ in 0..2 {
        let tick = owner
            .advance(&mut world, WorldPos::new(610.0, 0.0, 0.0), &[])
            .unwrap();
        assert_eq!(tick.failed.len(), 1);
        assert_eq!(tick.failed[0].cell, C);
        assert_eq!(tick.failed[0].operation, CellOperation::Load);
        assert!(matches!(tick.failed[0].error, PartitionError::Io(_)));
        assert_eq!(state(&owner, C), CellState::Unloaded);
    }
    // Missing is not empty. Explicit authoring creates a valid empty archive.
    owner.create_empty_cell(&world, C).unwrap();
    owner.unload_cell(&mut world, C).unwrap();
    let tick = owner
        .advance(&mut world, WorldPos::new(610.0, 0.0, 0.0), &[])
        .unwrap();
    assert!(tick.failed.is_empty());
    assert_eq!(tick.loaded, vec![(C, vec![])]);
    assert_eq!(state(&owner, C), CellState::Loaded);
}

#[test]
fn failed_save_keeps_entities_values_relations_index_and_loaded_state_then_retries() {
    let scratch = Scratch::new();
    let mut world = world();
    let a = world.spawn(value(11));
    let b = world.spawn(value(12));
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    owner.register(&world, B, b, id(2)).unwrap();
    owner.link::<Parent>(&mut world, id(1), id(2)).unwrap();
    owner.save_cell(&world, A).unwrap();
    let old_file = owner.store().read(A).unwrap();
    let root = owner.store().root().to_owned();
    let obstruction = scratch.path().join("not-a-directory");
    fs::write(&obstruction, b"obstruction").unwrap();
    *owner.store_mut() = FileCellStore::new(&obstruction);

    let tick = owner
        .advance(&mut world, WorldPos::new(310.0, 0.0, 0.0), &[])
        .unwrap();
    assert_eq!(tick.failed.len(), 1);
    assert_eq!(tick.failed[0].operation, CellOperation::Unload);
    assert!(tick.despawned.is_empty());
    assert!(world.contains(a) && world.contains(b));
    assert_eq!(world.get::<Value>(a), Some(&value(11)));
    assert_eq!(world.relation_targets::<Parent>(a), &[b]);
    assert_eq!(owner.partition().index().entities_in(A), vec![a]);
    assert_eq!(state(&owner, A), CellState::Loaded);

    *owner.store_mut() = FileCellStore::new(root);
    assert_eq!(owner.store().read(A).unwrap(), old_file);
    let tick = owner
        .advance(&mut world, WorldPos::new(310.0, 0.0, 0.0), &[])
        .unwrap();
    assert!(tick.failed.is_empty());
    assert_eq!(tick.unloaded, vec![A]);
    assert!(!world.contains(a));
    assert!(world.contains(b));
    owner.load_cell(&mut world, A).unwrap();
    let fresh = owner.resolve(&world, id(1)).unwrap();
    assert_eq!(world.relation_targets::<Parent>(fresh), &[b]);
}

#[test]
fn corrupt_archive_load_settles_nothing_and_retry_after_repair_succeeds() {
    let scratch = Scratch::new();
    let mut world = world();
    let a = world.spawn(value(5));
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    owner.unload_cell(&mut world, A).unwrap();
    let valid = owner.store().read(A).unwrap();
    let mut broken = valid.clone();
    broken[0] ^= 1;
    fs::write(owner.store().cell_path(A), broken).unwrap();
    let unrelated = world.spawn(value(91));
    let count = world.entities().len();
    let tick = owner
        .advance(&mut world, WorldPos::new(10.0, 0.0, 0.0), &[])
        .unwrap();
    assert_eq!(tick.failed.len(), 1);
    assert!(matches!(
        tick.failed[0].error,
        PartitionError::InvalidArchive(_)
    ));
    assert_eq!(world.entities().len(), count);
    assert_eq!(world.get::<Value>(unrelated), Some(&value(91)));
    assert!(owner.partition().index().entities_in(A).is_empty());
    assert_eq!(owner.resolve(&world, id(1)), None);
    assert_eq!(state(&owner, A), CellState::Unloaded);

    owner.store_mut().write_atomic(A, &valid).unwrap();
    let tick = owner
        .advance(&mut world, WorldPos::new(10.0, 0.0, 0.0), &[])
        .unwrap();
    assert!(tick.failed.is_empty());
    assert_eq!(state(&owner, A), CellState::Loaded);
    assert_eq!(world.entities().len(), count + 1);
}

#[test]
fn stable_cross_cell_values_and_source_owned_relations_survive_restart_in_either_order() {
    for target_first in [false, true] {
        let scratch = Scratch::new();
        let mut source = world();
        let a = source.spawn((
            value(1),
            Target {
                reference: PersistentEntityRef::new(id(2)),
            },
        ));
        let b = source.spawn(value(2));
        let mut writer = owner(&source, &scratch);
        writer.register(&source, A, a, id(1)).unwrap();
        writer.register(&source, B, b, id(2)).unwrap();
        writer.link::<Parent>(&mut source, id(1), id(2)).unwrap();
        writer.unload_cell(&mut source, B).unwrap();
        assert!(
            source.contains(a),
            "target unload must not cascade-delete source"
        );
        assert!(source.relation_targets::<Parent>(a).is_empty());
        assert_eq!(
            source
                .get::<Target>(a)
                .unwrap()
                .reference
                .resolve(&writer, &source),
            None
        );
        // Source archive must retain its outgoing edge even with target absent.
        writer.unload_cell(&mut source, A).unwrap();
        drop(writer);

        let mut destination = world();
        let mut reader = owner(&destination, &scratch);
        let order = if target_first { [B, A] } else { [A, B] };
        reader.load_cell(&mut destination, order[0]).unwrap();
        if !target_first {
            let a = reader.resolve(&destination, id(1)).unwrap();
            assert!(destination.relation_targets::<Parent>(a).is_empty());
            assert_eq!(
                destination
                    .get::<Target>(a)
                    .unwrap()
                    .reference
                    .resolve(&reader, &destination),
                None
            );
        }
        reader.load_cell(&mut destination, order[1]).unwrap();
        let a = reader.resolve(&destination, id(1)).unwrap();
        let b = reader.resolve(&destination, id(2)).unwrap();
        assert_eq!(destination.get::<Value>(a), Some(&value(1)));
        assert_eq!(destination.get::<Value>(b), Some(&value(2)));
        assert_eq!(
            destination
                .get::<Target>(a)
                .unwrap()
                .reference
                .resolve(&reader, &destination),
            Some(b)
        );
        assert_eq!(destination.relation_targets::<Parent>(a), &[b]);
    }
}

#[test]
fn pending_unlink_and_exclusive_reparent_do_not_resurrect_old_edges() {
    let scratch = Scratch::new();
    let mut world = world();
    let a = world.spawn(value(1));
    let b = world.spawn(value(2));
    let c = world.spawn(value(3));
    let mut owner = owner(&world, &scratch);
    for (cell, entity, identity) in [(A, a, id(1)), (B, b, id(2)), (C, c, id(3))] {
        owner.register(&world, cell, entity, identity).unwrap();
    }
    owner.link::<Parent>(&mut world, id(1), id(2)).unwrap();
    owner.unload_cell(&mut world, B).unwrap();
    assert!(owner.unlink::<Parent>(&mut world, id(1), id(2)).unwrap());
    assert!(!owner.unlink::<Parent>(&mut world, id(1), id(2)).unwrap());
    owner.load_cell(&mut world, B).unwrap();
    assert!(world.relation_targets::<Parent>(a).is_empty());

    owner.link::<Parent>(&mut world, id(1), id(2)).unwrap();
    owner.unload_cell(&mut world, B).unwrap();
    // Direct ECS reparent to a live target supersedes the old pending target.
    world.add_relation::<Parent>(a, c);
    owner.load_cell(&mut world, B).unwrap();
    assert_eq!(world.relation_targets::<Parent>(a), &[c]);
    // Direct deletion of a live edge remains deleted through both reloads.
    assert!(world.remove_relation::<Parent>(a, c));
    owner.unload_cell(&mut world, C).unwrap();
    owner.load_cell(&mut world, C).unwrap();
    assert!(world.relation_targets::<Parent>(a).is_empty());

    // Editing an exclusive edge while both potential targets are absent.
    owner.unload_cell(&mut world, B).unwrap();
    owner.unload_cell(&mut world, C).unwrap();
    owner.link::<Parent>(&mut world, id(1), id(2)).unwrap();
    owner.link::<Parent>(&mut world, id(1), id(3)).unwrap();
    owner.load_cell(&mut world, B).unwrap();
    assert!(world.relation_targets::<Parent>(a).is_empty());
    owner.load_cell(&mut world, C).unwrap();
    let c = owner.resolve(&world, id(3)).unwrap();
    assert_eq!(world.relation_targets::<Parent>(a), &[c]);
}

#[test]
fn panic_cleanup_policy_cannot_prevent_safe_target_eviction_and_rebinding() {
    let scratch = Scratch::new();
    let mut world = world();
    world.register_relation::<Parent>(kind(CleanupPolicy::Panic));
    let a = world.spawn(value(1));
    let b = world.spawn(value(2));
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    owner.register(&world, B, b, id(2)).unwrap();
    owner.link::<Parent>(&mut world, id(1), id(2)).unwrap();
    owner.unload_cell(&mut world, B).unwrap();
    assert!(world.contains(a));
    assert!(world.relation_targets::<Parent>(a).is_empty());
    owner.load_cell(&mut world, B).unwrap();
    assert_eq!(
        world.relation_targets::<Parent>(a),
        &[owner.resolve(&world, id(2)).unwrap()]
    );
}

#[test]
fn missing_or_mismatched_relation_registration_fails_before_allocation_then_retries() {
    let scratch = Scratch::new();
    let mut source = world();
    let a = source.spawn(value(1));
    let b = source.spawn(value(2));
    let mut writer = owner(&source, &scratch);
    writer.register(&source, A, a, id(1)).unwrap();
    writer.register(&source, B, b, id(2)).unwrap();
    writer.link::<Parent>(&mut source, id(1), id(2)).unwrap();
    writer.save_cell(&source, A).unwrap();

    let mut destination = World::new();
    destination.register_reflect_component::<Value>();
    let mut reader = owner(&destination, &scratch);
    assert!(matches!(
        reader.load_cell(&mut destination, A),
        Err(PartitionError::UnknownRelation(_))
    ));
    assert_eq!(destination.entities().len(), 0);
    destination.register_relation::<Parent>(RelationKind::new());
    assert!(matches!(
        reader.load_cell(&mut destination, A),
        Err(PartitionError::RelationPolicy(_))
    ));
    assert_eq!(destination.entities().len(), 0);
    assert_eq!(state(&reader, A), CellState::Unloaded);
    destination.register_relation::<Parent>(kind(CleanupPolicy::Delete));
    reader.load_cell(&mut destination, A).unwrap();
    assert_eq!(destination.entities().len(), 1);
}

#[test]
fn unreflected_sparse_component_rejection_preserves_previous_archive_and_live_cell() {
    let scratch = Scratch::new();
    let mut world = world();
    let a = world.spawn(value(1));
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    owner.save_cell(&world, A).unwrap();
    let original_file = owner.store().read(A).unwrap();
    world.insert(a, UnreflectedSparse);
    assert!(matches!(
        owner.unload_cell(&mut world, A),
        Err(PartitionError::UnsupportedComponent(_))
    ));
    assert!(world.get::<UnreflectedSparse>(a).is_some());
    assert_eq!(owner.store().read(A).unwrap(), original_file);
    assert_eq!(owner.partition().index().entities_in(A), vec![a]);
    assert_eq!(state(&owner, A), CellState::Loaded);
}

#[test]
fn shared_storage_is_rejected_instead_of_silently_omitted() {
    let scratch = Scratch::new();
    let mut world = world();
    world.register_reflect_component::<Shared>();
    let a = world.spawn((value(1), Shared { number: 7 }));
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    assert!(matches!(
        owner.unload_cell(&mut world, A),
        Err(PartitionError::UnsupportedComponent(_))
    ));
    assert!(world.contains(a));
    assert_eq!(owner.partition().index().entities_in(A), vec![a]);
    assert!(!owner.store().cell_path(A).exists());
}

#[test]
fn lifecycle_callbacks_are_rejected_before_persistence_or_world_mutation() {
    let scratch = Scratch::new();
    let mut world = world();
    let a = world.spawn(value(1));
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    world.observe::<Value, _>(LifecycleEvent::Remove, |_| panic!("must not execute"));
    assert!(matches!(
        owner.unload_cell(&mut world, A),
        Err(PartitionError::LifecycleCallbacks)
    ));
    assert!(world.contains(a));
    assert_eq!(owner.partition().index().entities_in(A), vec![a]);
    assert!(!owner.store().cell_path(A).exists());
}

#[test]
fn world_replacement_and_snapshot_restore_invalidate_owner_handles() {
    let scratch = Scratch::new();
    let mut world = world();
    let before = world.snapshot();
    let a = world.spawn(value(1));
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    let mut different = World::new();
    assert_eq!(owner.resolve(&different, id(1)), None);
    assert!(matches!(
        owner.unload_cell(&mut different, A),
        Err(PartitionError::WorldChanged)
    ));
    assert_eq!(state(&owner, A), CellState::Loaded);
    assert!(world.contains(a));

    world.restore(&before);
    assert_eq!(owner.resolve(&world, id(1)), None);
    assert!(matches!(
        owner.save_cell(&world, A),
        Err(PartitionError::WorldChanged)
    ));
    assert!(matches!(
        owner.load_cell(&mut world, B),
        Err(PartitionError::WorldChanged)
    ));
}

#[test]
fn internal_empty_prefab_ancestor_roundtrips_but_cross_cell_ancestry_is_rejected() {
    let scratch = Scratch::new();
    let mut world = world();
    let parent = world.spawn(());
    let instance = world.spawn(value(3));
    world.add_prefab(instance, parent);
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, parent, id(1)).unwrap();
    owner.register(&world, A, instance, id(2)).unwrap();
    owner.unload_cell(&mut world, A).unwrap();
    owner.load_cell(&mut world, A).unwrap();
    let parent = owner.resolve(&world, id(1)).unwrap();
    let instance = owner.resolve(&world, id(2)).unwrap();
    assert_eq!(world.prefabs_of(instance), &[parent]);
    assert_eq!(world.get::<Value>(instance), Some(&value(3)));
    owner.register(&world, B, parent, id(1)).unwrap();
    assert!(matches!(
        owner.save_cell(&world, A),
        Err(PartitionError::InvalidState(_))
    ));
    assert!(world.contains(parent) && world.contains(instance));
}

#[test]
fn data_layer_gates_drive_the_same_durable_unload_and_load_path() {
    let scratch = Scratch::new();
    let mut world = world();
    let a = world.spawn(value(1));
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    let layer = DataLayerId::new(77);
    owner.data_layers_mut().assign(A, layer);
    // Unregistered layers default to Unloaded even under the camera.
    let camera = WorldPos::new(10.0, 0.0, 0.0);
    let tick = owner.advance(&mut world, camera, &[]).unwrap();
    assert!(tick.failed.is_empty());
    assert_eq!(tick.unloaded, vec![A]);
    assert!(tick.loaded.is_empty());
    owner
        .data_layers_mut()
        .set_state(layer, DataLayerState::Activated);
    let tick = owner.advance(&mut world, camera, &[]).unwrap();
    assert!(tick.failed.is_empty());
    assert_eq!(tick.loaded.len(), 1);
    assert_eq!(tick.loaded[0].0, A);
    assert_eq!(
        world.get::<Value>(owner.resolve(&world, id(1)).unwrap()),
        Some(&value(1))
    );
}

#[test]
fn duplicate_identity_and_unowned_relation_endpoint_are_never_silently_saved() {
    let scratch = Scratch::new();
    let mut world = world();
    let a = world.spawn(value(1));
    let b = world.spawn(value(2));
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    assert!(matches!(
        owner.register(&world, B, b, id(1)),
        Err(PartitionError::DuplicateId(_))
    ));
    assert_eq!(owner.partition().index().cell_of(b), None);
    world.add_relation::<Parent>(a, b);
    assert!(matches!(
        owner.unload_cell(&mut world, A),
        Err(PartitionError::InvalidState(_))
    ));
    assert_eq!(world.relation_targets::<Parent>(a), &[b]);
    assert!(world.contains(a) && world.contains(b));
}

#[test]
fn valid_checksum_member_mismatch_rolls_back_native_archive_allocations() {
    let scratch = Scratch::new();
    let mut world = world();
    let a = world.spawn(value(1));
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    owner.unload_cell(&mut world, A).unwrap();
    let valid = owner.store().read(A).unwrap();
    let mut broken = valid.clone();
    // PSCL version + coordinates + member count + Entity + PersistentId.
    // Claim the reflected entity is empty, leaving its real archive intact.
    let flag = 5 + 3 * 4 + 4 + 8 + 16;
    assert_eq!(broken[flag], 1);
    broken[flag] = 0;
    let end = broken.len() - 8;
    let hash = broken[..end]
        .iter()
        .fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
    broken[end..].copy_from_slice(&hash.to_le_bytes());
    owner.store_mut().write_atomic(A, &broken).unwrap();
    let existing = world.spawn(value(99));
    let count = world.entities().len();
    assert!(matches!(
        owner.load_cell(&mut world, A),
        Err(PartitionError::InvalidArchive(
            "payload/member set mismatch"
        ))
    ));
    assert_eq!(world.entities().len(), count);
    assert_eq!(world.get::<Value>(existing), Some(&value(99)));
    assert!(owner.partition().index().entities_in(A).is_empty());
    assert_eq!(owner.resolve(&world, id(1)), None);
    assert_eq!(state(&owner, A), CellState::Unloaded);
    owner.store_mut().write_atomic(A, &valid).unwrap();
    owner.load_cell(&mut world, A).unwrap();
    assert_eq!(world.entities().len(), count + 1);
}

fn reject_remap(
    _: StableTypeId,
    _: Option<&ParsedPath>,
    _: &mut dyn Reflect,
    _: &BTreeMap<Entity, Entity>,
) -> Result<(), ReflectArchiveError> {
    Err(ReflectArchiveError::InvalidPrefabMetadata(
        "application rejected remap",
    ))
}

#[test]
fn remap_failure_leaks_no_entities_and_does_not_settle_then_can_retry() {
    let scratch = Scratch::new();
    let mut world = world();
    let a = world.spawn(value(1));
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    owner.unload_cell(&mut world, A).unwrap();
    let count = world.entities().len();
    owner.set_remap(Some(reject_remap));
    assert!(matches!(
        owner.load_cell(&mut world, A),
        Err(PartitionError::Archive(_))
    ));
    assert_eq!(world.entities().len(), count);
    assert!(owner.partition().index().entities_in(A).is_empty());
    assert_eq!(state(&owner, A), CellState::Unloaded);
    owner.set_remap(None);
    owner.load_cell(&mut world, A).unwrap();
    assert_eq!(world.entities().len(), count + 1);
}

#[test]
fn failed_atomic_rename_keeps_live_cell_and_cleans_only_its_temporary() {
    let scratch = Scratch::new();
    let mut world = world();
    let a = world.spawn(value(1));
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    let obstruction = owner.store().cell_path(A);
    fs::create_dir_all(&obstruction).unwrap();
    fs::write(obstruction.join("sentinel"), b"must survive").unwrap();
    assert!(matches!(
        owner.unload_cell(&mut world, A),
        Err(PartitionError::Io(_))
    ));
    assert_eq!(world.get::<Value>(a), Some(&value(1)));
    assert_eq!(owner.partition().index().entities_in(A), vec![a]);
    assert_eq!(state(&owner, A), CellState::Loaded);
    assert_eq!(
        fs::read(obstruction.join("sentinel")).unwrap(),
        b"must survive"
    );
    assert_eq!(fs::read_dir(owner.store().root()).unwrap().count(), 1);
}

#[test]
fn stable_refs_to_empty_standalone_targets_are_not_limited_by_native_remap_map() {
    let scratch = Scratch::new();
    let mut world = world();
    let a = world.spawn(Target {
        reference: PersistentEntityRef::new(id(2)),
    });
    let empty = world.spawn(());
    let mut owner = owner(&world, &scratch);
    owner.register(&world, A, a, id(1)).unwrap();
    owner.register(&world, A, empty, id(2)).unwrap();
    owner.unload_cell(&mut world, A).unwrap();
    owner.load_cell(&mut world, A).unwrap();
    let a = owner.resolve(&world, id(1)).unwrap();
    let empty = owner.resolve(&world, id(2)).unwrap();
    assert_eq!(
        world
            .get::<Target>(a)
            .unwrap()
            .reference
            .resolve(&owner, &world),
        Some(empty)
    );
    assert!(world.reflect_entity(empty).is_empty());
}
