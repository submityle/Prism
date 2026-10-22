//! Authored identities survive structural publication, not generational reuse.

use prism_ecs::{
    component_hooks::{ComponentHooks, HookContext},
    prelude::{ChildOf, Component, Entity, Resource, World},
    relation::{CleanupPolicy, RelationKind},
};
use prism_editor_app::{
    ControllerLimits, ControllerObject, DocumentCodec, DocumentLimits, DocumentSchemaCatalog,
    EditorController, EditorEcsDocumentBridge, EditorEcsDocumentError, EditorEcsStructureSync,
};
use prism_editor_core::SelectionItem;
use prism_editor_panels::{OutlinerId, OutlinerNode};
use prism_editor_ui::EditorAction;
use prism_reflect::{DynamicStruct, FromReflect, Reflect, TypeRegistry};

#[derive(Clone, Debug, PartialEq, Component, Reflect)]
struct Model {
    value: i32,
}

#[derive(Component)]
struct RuntimeOnly(u32);

#[derive(Component)]
struct HostRelation;

fn registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    registry.register::<Model>();
    registry
}

fn item(id: u128) -> SelectionItem {
    SelectionItem::entity(id)
}

fn row(id: u128, parent: Option<u128>, object: Option<u128>) -> OutlinerNode {
    let mut row = OutlinerNode::new(
        OutlinerId::from_raw(id),
        parent.map(OutlinerId::from_raw),
        format!("row {id}"),
    );
    if let Some(object) = object {
        row.set_selection(item(object));
    }
    row
}

fn object(id: u128, parent: Option<u128>) -> ControllerObject {
    ControllerObject {
        item: item(id),
        root: Box::new(Model { value: id as i32 }),
        registry: registry(),
        row: row(id, parent, Some(id)),
    }
}

fn fixture() -> (EditorController, World, EditorEcsDocumentBridge) {
    let mut controller = EditorController::new();
    controller
        .register_outliner_node(row(1, None, None))
        .unwrap();
    controller.create_objects(vec![object(2, Some(1))]).unwrap();
    let mut world = World::new();
    world.register_reflect_component::<Model>();
    let bridge = EditorEcsDocumentBridge::materialize(&mut world, &controller).unwrap();
    (controller, world, bridge)
}

fn roundtrip(
    world: &World,
    source: &EditorController,
    bridge: &EditorEcsDocumentBridge,
) -> EditorController {
    let mut schemas = DocumentSchemaCatalog::new();
    schemas
        .register(std::any::type_name::<Model>(), 1, registry)
        .unwrap();
    let captured = bridge
        .capture_controller(world, source, ControllerLimits::default())
        .unwrap();
    let bytes = DocumentCodec::encode(&captured, &schemas, DocumentLimits::default()).unwrap();
    DocumentCodec::decode(
        &bytes,
        &schemas,
        DocumentLimits::default(),
        ControllerLimits::default(),
    )
    .unwrap()
    .into_controller()
}

#[test]
fn create_delete_undo_redo_capture_reload_keeps_survivor_runtime_state() {
    let (mut controller, mut world, mut bridge) = fixture();
    let survivor = bridge.entity_for_object(&item(2)).unwrap();
    world.insert(survivor, RuntimeOnly(91));
    world.get_mut::<Model>(survivor).unwrap().value = 27;
    controller.create_objects(vec![object(3, Some(1))]).unwrap();
    let history = controller.document_history_lengths();
    assert_eq!(
        bridge
            .synchronize_structure(&mut world, &controller)
            .unwrap(),
        EditorEcsStructureSync {
            spawned: 1,
            despawned: 0,
            reparented: 1,
        }
    );
    assert_eq!(controller.document_history_lengths(), history);
    let created = bridge.entity_for_object(&item(3)).unwrap();
    assert_eq!(world.get::<Model>(created).unwrap().value, 3);
    assert_eq!(bridge.entity_for_object(&item(2)), Some(survivor));
    assert_eq!(world.get::<Model>(survivor).unwrap().value, 27);
    assert_eq!(world.get::<RuntimeOnly>(survivor).unwrap().0, 91);
    let loaded = roundtrip(&world, &controller, &bridge);
    assert_eq!(loaded.document_objects().len(), 2);
    // Document decoding produces a dynamic reflected tree, not a concrete
    // Rust Model allocation. Reconstruct it through the schema contract.
    assert_eq!(
        Model::from_reflect(loaded.object_root(&item(2)).unwrap())
            .unwrap()
            .value,
        27
    );
    let mut reloaded_world = World::new();
    reloaded_world.register_reflect_component::<Model>();
    let reloaded_bridge =
        EditorEcsDocumentBridge::materialize(&mut reloaded_world, &loaded).unwrap();
    let reloaded_survivor = reloaded_bridge.entity_for_object(&item(2)).unwrap();
    assert_eq!(
        reloaded_world
            .get::<Model>(reloaded_survivor)
            .unwrap()
            .value,
        27
    );
    assert!(reloaded_world
        .get::<RuntimeOnly>(reloaded_survivor)
        .is_none());
    assert_eq!(reloaded_bridge.despawn(&mut reloaded_world), 3);
    assert_eq!(reloaded_world.entity_count(), 0);

    controller.delete_objects(&[item(3)]).unwrap();
    assert_eq!(
        bridge
            .synchronize_structure(&mut world, &controller)
            .unwrap()
            .despawned,
        1
    );
    assert!(!world.contains(created));
    assert!(bridge.entity_for_object(&item(3)).is_none());
    controller.execute(EditorAction::Undo).unwrap();
    assert_eq!(
        bridge
            .synchronize_structure(&mut world, &controller)
            .unwrap()
            .spawned,
        1
    );
    let recreated = bridge.entity_for_object(&item(3)).unwrap();
    assert_ne!(recreated, created);
    assert!(!world.contains(created));
    controller.execute(EditorAction::Redo).unwrap();
    bridge
        .synchronize_structure(&mut world, &controller)
        .unwrap();
    assert!(!world.contains(recreated));
    let loaded = roundtrip(&world, &controller, &bridge);
    assert_eq!(loaded.document_objects().len(), 1);
    assert_eq!(bridge.despawn(&mut world), 2);
    assert_eq!(world.entity_count(), 0);
}

#[test]
fn noop_does_not_dirty_models_or_relations() {
    let (controller, mut world, mut bridge) = fixture();
    let actor = bridge.entity_for_object(&item(2)).unwrap();
    let before = world.get_ticks::<Model>(actor).unwrap().changed;
    let relation_revision = world.relations().index().change_cursor();
    world.increment_change_tick();
    assert_eq!(
        bridge
            .synchronize_structure(&mut world, &controller)
            .unwrap(),
        EditorEcsStructureSync::default()
    );
    assert_eq!(world.relations().index().change_cursor(), relation_revision);
    assert_eq!(world.get_ticks::<Model>(actor).unwrap().changed, before);
    assert_eq!(world.entity_count(), 2);
}

#[test]
fn detached_survivor_keeps_entity_when_old_parent_is_removed() {
    let (_, mut world, mut bridge) = fixture();
    let survivor = bridge.entity_for_object(&item(2)).unwrap();
    let group = bridge.entity_for_outliner(OutlinerId::from_raw(1)).unwrap();
    let mut detached = EditorController::new();
    detached
        .register_object(item(2), Box::new(Model { value: -100 }), registry())
        .unwrap();
    let report = bridge.synchronize_structure(&mut world, &detached).unwrap();
    assert_eq!(
        report,
        EditorEcsStructureSync {
            spawned: 0,
            despawned: 1,
            reparented: 1
        }
    );
    assert_eq!(bridge.entity_for_object(&item(2)), Some(survivor));
    assert!(world.contains(survivor));
    assert!(!world.contains(group));
    assert!(world.relation_targets::<ChildOf>(survivor).is_empty());
    assert_eq!(world.get::<Model>(survivor).unwrap().value, 2);
    roundtrip(&world, &detached, &bridge);
}

#[test]
fn moving_object_to_new_row_preserves_object_not_old_row_identity() {
    let (_, mut world, mut bridge) = fixture();
    let actor = bridge.entity_for_object(&item(2)).unwrap();
    let mut source = EditorController::new();
    source
        .register_object(item(2), Box::new(Model { value: 99 }), registry())
        .unwrap();
    source.register_outliner_node(row(2, None, None)).unwrap();
    source
        .register_outliner_node(row(9, Some(2), Some(2)))
        .unwrap();
    let report = bridge.synchronize_structure(&mut world, &source).unwrap();
    let group = bridge.entity_for_outliner(OutlinerId::from_raw(2)).unwrap();
    assert_ne!(group, actor);
    assert_eq!(
        bridge.entity_for_outliner(OutlinerId::from_raw(9)),
        Some(actor)
    );
    assert_eq!(world.relation_targets::<ChildOf>(actor), &[group]);
    assert!(world.get::<Model>(group).is_none());
    assert_eq!(report.spawned, 1);
    roundtrip(&world, &source, &bridge);
}

#[test]
fn removed_object_row_replaced_with_group_does_not_keep_model() {
    let (_, mut world, mut bridge) = fixture();
    let old = bridge.entity_for_object(&item(2)).unwrap();
    let mut source = EditorController::new();
    source.register_outliner_node(row(2, None, None)).unwrap();
    let report = bridge.synchronize_structure(&mut world, &source).unwrap();
    assert_eq!(report.spawned, 1);
    assert_eq!(report.despawned, 2);
    assert!(!world.contains(old));
    let group = bridge.entity_for_outliner(OutlinerId::from_raw(2)).unwrap();
    assert!(world.get::<Model>(group).is_none());
    roundtrip(&world, &source, &bridge);
}

#[test]
fn deletion_rejects_foreign_child_without_mutating_any_entities() {
    let (mut source, mut world, mut bridge) = fixture();
    let actor = bridge.entity_for_object(&item(2)).unwrap();
    let foreign = world.spawn(());
    world.add_relation::<ChildOf>(foreign, actor);
    source.delete_objects(&[item(2)]).unwrap();
    assert!(matches!(bridge.synchronize_structure(&mut world, &source),
        Err(EditorEcsDocumentError::DeletionConflict {target, source}) if target == actor && source == foreign));
    assert_eq!(bridge.entity_for_object(&item(2)), Some(actor));
    assert_eq!(world.entity_count(), 3);
    assert_eq!(world.relation_targets::<ChildOf>(foreign), &[actor]);
}

#[test]
fn deletion_rejects_custom_delete_and_panic_policies_before_publication() {
    for policy in [CleanupPolicy::Delete, CleanupPolicy::Panic] {
        let (mut source, mut world, mut bridge) = fixture();
        let actor = bridge.entity_for_object(&item(2)).unwrap();
        let foreign = world.spawn(());
        world.register_relation::<HostRelation>(RelationKind::new().with_on_delete_target(policy));
        world.add_relation::<HostRelation>(foreign, actor);
        source.delete_objects(&[item(2)]).unwrap();
        assert!(matches!(bridge.synchronize_structure(&mut world, &source),
            Err(EditorEcsDocumentError::DeletionConflict {target, source}) if target == actor && source == foreign));
        assert_eq!(world.entity_count(), 3);
        assert_eq!(world.relation_targets::<HostRelation>(foreign), &[actor]);
    }
}

#[test]
fn duplicate_binding_and_unregistered_addition_are_preflight_failures() {
    let (mut source, mut world, mut bridge) = fixture();
    source
        .register_outliner_node(row(4, None, Some(2)))
        .unwrap();
    assert!(matches!(
        bridge.synchronize_structure(&mut world, &source),
        Err(EditorEcsDocumentError::DuplicateObjectRow(_))
    ));
    assert_eq!(world.entity_count(), 2);
    #[derive(Reflect)]
    struct Unknown {
        value: i32,
    }
    let mut other = EditorController::new();
    let mut types = TypeRegistry::new();
    types.register::<Unknown>();
    other
        .register_object(item(9), Box::new(Unknown { value: 1 }), types)
        .unwrap();
    assert!(matches!(
        bridge.synchronize_structure(&mut world, &other),
        Err(EditorEcsDocumentError::UnregisteredComponent(_))
    ));
    assert_eq!(world.entity_count(), 2);
}

#[test]
fn stale_retained_entity_is_not_rebound_to_reused_slot() {
    let (source, mut world, mut bridge) = fixture();
    let actor = bridge.entity_for_object(&item(2)).unwrap();
    world.despawn(actor);
    let replacement = world.spawn(Model { value: 77 });
    assert!(
        matches!(bridge.synchronize_structure(&mut world, &source), Err(EditorEcsDocumentError::StaleEntity(entity)) if entity == actor)
    );
    assert_eq!(world.get::<Model>(replacement).unwrap().value, 77);
}

#[test]
fn different_world_with_colliding_handles_is_never_read_or_mutated() {
    let (source, world, mut bridge) = fixture();
    let mut other = World::new();
    other.register_reflect_component::<Model>();
    let _other_bridge = EditorEcsDocumentBridge::materialize(&mut other, &source).unwrap();
    assert_eq!(world.entity_count(), other.entity_count());
    assert!(matches!(
        bridge.capture_controller(&other, &source, ControllerLimits::default()),
        Err(EditorEcsDocumentError::WorldMismatch)
    ));
    assert!(matches!(
        bridge.synchronize_structure(&mut other, &source),
        Err(EditorEcsDocumentError::WorldMismatch)
    ));
    assert_eq!(bridge.despawn(&mut other), 0);
    assert_eq!(other.entity_count(), 2);
    assert_eq!(world.entity_count(), 2);
}

#[test]
fn capture_rejects_changed_row_bindings_even_when_identity_sets_match() {
    let (mut source, mut world, _) = fixture();
    source.create_objects(vec![object(3, Some(1))]).unwrap();
    let mut bridge = EditorEcsDocumentBridge::materialize(&mut world, &source).unwrap();
    let actor2 = bridge.entity_for_object(&item(2)).unwrap();
    let actor3 = bridge.entity_for_object(&item(3)).unwrap();

    // Build a second valid document with identical object and row ID sets but
    // intentionally different object-to-row bindings.
    let mut swapped = EditorController::new();
    swapped.register_outliner_node(row(1, None, None)).unwrap();
    for id in [2, 3] {
        swapped
            .register_object(item(id), Box::new(Model { value: id as i32 }), registry())
            .unwrap();
    }
    swapped
        .register_outliner_node(row(2, Some(1), Some(3)))
        .unwrap();
    swapped
        .register_outliner_node(row(3, Some(1), Some(2)))
        .unwrap();
    assert!(matches!(
        bridge.capture_controller(&world, &swapped, ControllerLimits::default()),
        Err(EditorEcsDocumentError::DocumentShapeChanged)
    ));
    let report = bridge.synchronize_structure(&mut world, &swapped).unwrap();
    assert_eq!(report.spawned, 0);
    assert_eq!(report.despawned, 0);
    assert_eq!(
        bridge.entity_for_outliner(OutlinerId::from_raw(2)),
        Some(actor3)
    );
    assert_eq!(
        bridge.entity_for_outliner(OutlinerId::from_raw(3)),
        Some(actor2)
    );
    roundtrip(&world, &swapped, &bridge);
}

#[derive(Resource)]
struct HookEdge {
    holder: Entity,
    target: Entity,
}

fn add_host_edge(ctx: HookContext<'_>) {
    let edge = ctx.world.resource::<HookEdge>();
    let (holder, target) = (edge.holder, edge.target);
    ctx.world.add_relation::<HostRelation>(holder, target);
}

#[test]
fn insertion_hook_cannot_bypass_deletion_preflight() {
    for policy in [CleanupPolicy::Delete, CleanupPolicy::Panic] {
        let (mut source, mut world, mut bridge) = fixture();
        let actor = bridge.entity_for_object(&item(2)).unwrap();
        let holder = world.spawn(RuntimeOnly(17));
        world.register_relation::<HostRelation>(RelationKind::new().with_on_delete_target(policy));
        world.insert_resource(HookEdge {
            holder,
            target: actor,
        });
        world
            .register_component_hooks::<Model>(ComponentHooks::new().with_on_insert(add_host_edge));
        source.delete_objects(&[item(2)]).unwrap();
        source.create_objects(vec![object(3, Some(1))]).unwrap();
        let relation_before = world.relation_targets::<ChildOf>(actor).to_vec();
        assert!(matches!(bridge.synchronize_structure(&mut world, &source),
            Err(EditorEcsDocumentError::DeletionConflict { target, source })
                if target == actor && source == holder));
        assert_eq!(world.entity_count(), 3);
        assert_eq!(bridge.entity_for_object(&item(2)), Some(actor));
        assert_eq!(bridge.entity_for_object(&item(3)), None);
        assert_eq!(world.relation_targets::<ChildOf>(actor), relation_before);
        // Host effects are intentionally not rolled back, but neither are
        // they allowed to delete a foreign entity during publication.
        assert_eq!(world.relation_targets::<HostRelation>(holder), &[actor]);
        assert_eq!(world.get::<RuntimeOnly>(holder).unwrap().0, 17);
    }
}

fn replace_world(ctx: HookContext<'_>) {
    *ctx.world = World::new();
    ctx.world.spawn(RuntimeOnly(333));
}

#[test]
fn world_replacement_in_hook_stops_synchronization_and_materialization() {
    let (mut source, mut world, mut bridge) = fixture();
    source
        .create_objects(vec![object(3, Some(1)), object(4, Some(1))])
        .unwrap();
    world.register_component_hooks::<Model>(ComponentHooks::new().with_on_insert(replace_world));
    assert!(matches!(
        bridge.synchronize_structure(&mut world, &source),
        Err(EditorEcsDocumentError::WorldMismatch)
    ));
    assert_eq!(world.entity_count(), 1);
    assert_eq!(bridge.entity_for_object(&item(3)), None);
    assert_eq!(bridge.entity_for_object(&item(4)), None);
    let survivors = world.query::<&RuntimeOnly>();
    assert_eq!(
        survivors
            .iter(&world)
            .map(|value| value.0)
            .collect::<Vec<_>>(),
        vec![333]
    );

    let mut world = World::new();
    world.register_reflect_component::<Model>();
    world.register_component_hooks::<Model>(ComponentHooks::new().with_on_insert(replace_world));
    assert!(matches!(
        EditorEcsDocumentBridge::materialize(&mut world, &source),
        Err(EditorEcsDocumentError::WorldMismatch)
    ));
    assert_eq!(world.entity_count(), 1);
}

#[test]
fn rejected_reflected_insertion_cleans_staging_and_can_be_retried() {
    let (_, mut world, mut bridge) = fixture();
    let mut malformed = DynamicStruct::new();
    malformed.set_represented_type_name(std::any::type_name::<Model>());
    malformed.insert("value", "not an integer".to_string());
    let mut source = EditorController::new();
    source
        .register_object(item(9), Box::new(malformed), registry())
        .unwrap();
    assert!(matches!(
        bridge.synchronize_structure(&mut world, &source),
        Err(EditorEcsDocumentError::InsertRejected { .. })
    ));
    assert_eq!(world.entity_count(), 2);
    assert_eq!(bridge.entity_count(), 2);
    assert!(bridge.entity_for_object(&item(9)).is_none());
    let mut valid = EditorController::new();
    valid.create_objects(vec![object(9, None)]).unwrap();
    let report = bridge.synchronize_structure(&mut world, &valid).unwrap();
    assert_eq!(report.spawned, 1);
    assert_eq!(report.despawned, 2);
    roundtrip(&world, &valid, &bridge);
}

#[test]
fn empty_document_removes_only_owned_entities_and_collapsed_undo_is_noop() {
    let (mut source, mut world, mut bridge) = fixture();
    let foreign = world.spawn(RuntimeOnly(18));
    source.create_objects(vec![object(3, Some(1))]).unwrap();
    source.execute(EditorAction::Undo).unwrap();
    assert_eq!(
        bridge.synchronize_structure(&mut world, &source).unwrap(),
        EditorEcsStructureSync::default()
    );
    let report = bridge
        .synchronize_structure(&mut world, &EditorController::new())
        .unwrap();
    assert_eq!(report.despawned, 2);
    assert_eq!(world.entity_count(), 1);
    assert_eq!(world.get::<RuntimeOnly>(foreign).unwrap().0, 18);
    assert_eq!(bridge.entity_count(), 0);
}
