//! PRISMDOC-to-ECS materialization and save/reload contract tests.

use prism_ecs::{
    prelude::{ChildOf, Component, World},
    relation::RelationKind,
};
use prism_editor_app::{
    ControllerLimits, DocumentCodec, DocumentLimits, DocumentSchemaCatalog, EditorController,
    EditorEcsDocumentBridge,
};
use prism_editor_core::SelectionItem;
use prism_editor_panels::{OutlinerId, OutlinerNode};
use prism_reflect::{FromReflect, Reflect, TypeRegistry};

#[derive(Clone, Debug, PartialEq, Reflect, Component)]
struct ActorModel {
    health: i32,
    label: String,
}

#[derive(Clone, Debug, PartialEq, Reflect, Component)]
struct EditorMarker(u32);

fn registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    registry.register::<ActorModel>();
    registry
}

fn schemas() -> DocumentSchemaCatalog {
    let mut schemas = DocumentSchemaCatalog::new();
    schemas
        .register(std::any::type_name::<ActorModel>(), 1, registry)
        .unwrap();
    schemas
}

fn authored_document() -> EditorController {
    let item = SelectionItem::entity(0x1234);
    let mut controller = EditorController::new();
    controller
        .register_object(
            item.clone(),
            Box::new(ActorModel {
                health: 80,
                label: "dock worker".into(),
            }),
            registry(),
        )
        .unwrap();
    let mut group = OutlinerNode::new(OutlinerId::from_raw(1), None, "Actors");
    group.set_expanded(false);
    controller.register_outliner_node(group).unwrap();
    let mut actor = OutlinerNode::new(
        OutlinerId::from_raw(2),
        Some(OutlinerId::from_raw(1)),
        "Worker",
    );
    actor.set_visible(false);
    actor.set_selection(item);
    controller.register_outliner_node(actor).unwrap();
    controller
        .register_outliner_node(OutlinerNode::new(OutlinerId::from_raw(3), None, "Workers"))
        .unwrap();
    controller
}

fn decode(bytes: &[u8]) -> EditorController {
    DocumentCodec::decode(
        bytes,
        &schemas(),
        DocumentLimits::default(),
        ControllerLimits::default(),
    )
    .unwrap()
    .into_controller()
}

#[test]
fn prismdoc_scene_ecs_edit_save_reload_roundtrip() {
    let authored = authored_document();
    let bytes = DocumentCodec::encode(&authored, &schemas(), DocumentLimits::default()).unwrap();
    let loaded = decode(&bytes);

    let mut world = World::new();
    world.register_reflect_component::<ActorModel>();
    let bridge = EditorEcsDocumentBridge::materialize(&mut world, &loaded).unwrap();

    let item = SelectionItem::entity(0x1234);
    let actor = bridge.entity_for_object(&item).unwrap();
    let group = bridge.entity_for_outliner(OutlinerId::from_raw(1)).unwrap();
    assert_eq!(
        bridge.entity_for_outliner(OutlinerId::from_raw(2)),
        Some(actor)
    );
    assert_eq!(world.relation_targets::<ChildOf>(actor), &[group]);
    assert_eq!(world.get::<ActorModel>(actor).unwrap().health, 80);
    let replacement_group = bridge.entity_for_outliner(OutlinerId::from_raw(3)).unwrap();
    assert_eq!(
        world.add_relation::<ChildOf>(actor, replacement_group),
        Some(group)
    );

    {
        let mut model = world.get_mut::<ActorModel>(actor).unwrap();
        model.health = 12;
        model.label = "edited in ECS".into();
    }
    let captured = bridge
        .capture_controller(&world, &loaded, ControllerLimits::default())
        .unwrap();
    let saved = DocumentCodec::encode(&captured, &schemas(), DocumentLimits::default()).unwrap();
    let reloaded = decode(&saved);
    let model = ActorModel::from_reflect(reloaded.object_root(&item).unwrap()).unwrap();
    assert_eq!(model.health, 12);
    assert_eq!(model.label, "edited in ECS");
    assert_eq!(
        saved,
        DocumentCodec::encode(&reloaded, &schemas(), DocumentLimits::default()).unwrap()
    );

    let rows = reloaded.document_outliner_nodes();
    assert_eq!(rows.len(), 3);
    let actor_row = rows
        .iter()
        .find(|row| row.id() == OutlinerId::from_raw(2))
        .unwrap();
    let group_row = rows
        .iter()
        .find(|row| row.id() == OutlinerId::from_raw(1))
        .unwrap();
    assert_eq!(actor_row.parent(), Some(OutlinerId::from_raw(3)));
    assert_eq!(actor_row.selection(), Some(&item));
    assert!(!group_row.is_expanded());
    assert!(!actor_row.is_visible());
    assert_eq!(bridge.despawn(&mut world), 3);
    assert_eq!(world.entity_count(), 0);
}

#[test]
fn bridge_preflight_rejects_unregistered_component_without_allocating() {
    let loaded = authored_document();
    let mut world = World::new();
    assert!(matches!(
        EditorEcsDocumentBridge::materialize(&mut world, &loaded),
        Err(prism_editor_app::EditorEcsDocumentError::UnregisteredComponent(_))
    ));
    assert_eq!(world.entity_count(), 0);
}

#[test]
fn bridge_preflight_rejects_ambiguous_outliner_binding_without_allocating() {
    let mut controller = authored_document();
    let mut duplicate = OutlinerNode::new(
        OutlinerId::from_raw(4),
        Some(OutlinerId::from_raw(1)),
        "duplicate",
    );
    duplicate.set_selection(SelectionItem::entity(0x1234));
    controller.register_outliner_node(duplicate).unwrap();

    let mut world = World::new();
    world.register_reflect_component::<ActorModel>();
    assert!(matches!(
        EditorEcsDocumentBridge::materialize(&mut world, &controller),
        Err(prism_editor_app::EditorEcsDocumentError::DuplicateObjectRow(_))
    ));
    assert_eq!(world.entity_count(), 0);
}

#[test]
fn bridge_preflight_preserves_custom_child_of_policy() {
    let source = authored_document();
    let mut world = World::new();
    world.register_reflect_component::<ActorModel>();
    world.register_relation::<ChildOf>(RelationKind::new());

    assert!(matches!(
        EditorEcsDocumentBridge::materialize(&mut world, &source),
        Err(prism_editor_app::EditorEcsDocumentError::IncompatibleChildOfRelation)
    ));
    assert_eq!(world.entity_count(), 0);
    let child_of = world.components().id_of::<ChildOf>().unwrap();
    assert_eq!(world.relations().kind(child_of), Some(&RelationKind::new()));
}

#[test]
fn capture_rejects_nonexclusive_multiple_outliner_parents() {
    let source = authored_document();
    let mut world = World::new();
    world.register_reflect_component::<ActorModel>();
    let bridge = EditorEcsDocumentBridge::materialize(&mut world, &source).unwrap();

    let actor = bridge.entity_for_outliner(OutlinerId::from_raw(2)).unwrap();
    let other_parent = bridge.entity_for_outliner(OutlinerId::from_raw(3)).unwrap();
    world.register_relation::<ChildOf>(RelationKind::new());
    world.add_relation::<ChildOf>(actor, other_parent);

    assert!(matches!(
        bridge.capture_controller(&world, &source, ControllerLimits::default()),
        Err(prism_editor_app::EditorEcsDocumentError::MultipleOutlinerParents(entity))
            if entity == actor
    ));
}

#[test]
fn capture_rejects_ecs_outliner_cycles() {
    let source = authored_document();
    let mut world = World::new();
    world.register_reflect_component::<ActorModel>();
    let bridge = EditorEcsDocumentBridge::materialize(&mut world, &source).unwrap();

    let group = bridge.entity_for_outliner(OutlinerId::from_raw(1)).unwrap();
    let actor = bridge.entity_for_outliner(OutlinerId::from_raw(2)).unwrap();
    let replacement_group = bridge.entity_for_outliner(OutlinerId::from_raw(3)).unwrap();
    world.add_relation::<ChildOf>(group, actor);
    world.add_relation::<ChildOf>(replacement_group, group);

    assert!(matches!(
        bridge.capture_controller(&world, &source, ControllerLimits::default()),
        Err(prism_editor_app::EditorEcsDocumentError::HierarchyCycle)
    ));
}

#[test]
fn capture_rejects_reflected_component_addition_instead_of_silently_dropping_it() {
    let source = authored_document();
    let mut world = World::new();
    world.register_reflect_component::<ActorModel>();
    world.register_reflect_component::<EditorMarker>();
    let bridge = EditorEcsDocumentBridge::materialize(&mut world, &source).unwrap();
    let actor = bridge
        .entity_for_object(&SelectionItem::entity(0x1234))
        .unwrap();
    world.insert(actor, EditorMarker(7));

    assert!(matches!(
        bridge.capture_controller(&world, &source, ControllerLimits::default()),
        Err(prism_editor_app::EditorEcsDocumentError::UnexpectedReflectedComponents {
            item,
            type_paths,
        }) if item == SelectionItem::entity(0x1234)
            && type_paths == vec![std::any::type_name::<EditorMarker>().to_string()]
    ));
}

#[test]
fn capture_rejects_removal_of_the_prismdoc_root_component() {
    let source = authored_document();
    let mut world = World::new();
    world.register_reflect_component::<ActorModel>();
    let bridge = EditorEcsDocumentBridge::materialize(&mut world, &source).unwrap();
    let actor = bridge
        .entity_for_object(&SelectionItem::entity(0x1234))
        .unwrap();
    assert!(world.remove::<ActorModel>(actor));

    assert!(matches!(
        bridge.capture_controller(&world, &source, ControllerLimits::default()),
        Err(prism_editor_app::EditorEcsDocumentError::MissingComponent { item, .. })
            if item == SelectionItem::entity(0x1234)
    ));
}
