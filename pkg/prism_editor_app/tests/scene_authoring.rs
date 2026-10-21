//! Opt-in authoring uses the real controller, document codec and CPU scene bridge.

#![cfg(feature = "desktop")]

use prism_editor_app::{
    editor_scene_mesh_catalog, editor_scene_mesh_registry, install_scene_authoring_commands,
    ControllerError, ControllerLimits, DocumentCodec, DocumentLimits, EditorController,
    EditorSceneMesh, EditorSceneSession, SCENE_ADD_TRIANGLE_COMMAND,
    SCENE_DELETE_SELECTION_COMMAND,
};
use prism_editor_core::SelectionItem;
use prism_editor_panels::{OutlinerId, OutlinerNode};
use prism_editor_ui::EditorAction;
use prism_editor_viewport::SCENE_ID_IDENTITY_MATRIX;
use prism_reflect::FromReflect;

fn execute(controller: &mut EditorController, command: &str) {
    let outcome = controller
        .execute(EditorAction::ExecuteCommand {
            command_id: command.into(),
        })
        .expect("execute authoring factory through normal command dispatch");
    assert!(outcome.changed);
    assert!(outcome.model_changed);
}

fn mesh(controller: &EditorController, item: &SelectionItem) -> EditorSceneMesh {
    EditorSceneMesh::from_reflect(controller.object_root(item).unwrap()).unwrap()
}

fn primary(controller: &EditorController) -> SelectionItem {
    controller.selection().primary().unwrap().clone()
}

#[test]
fn opt_in_is_clean_and_duplicate_install_does_not_touch_document() {
    let mut controller = EditorController::new();
    install_scene_authoring_commands(&mut controller).unwrap();
    let revision = controller.revision();
    assert!(controller.has_command(SCENE_ADD_TRIANGLE_COMMAND));
    assert!(controller.has_command(SCENE_DELETE_SELECTION_COMMAND));
    assert_eq!(controller.document_objects().len(), 0);
    assert!(controller.document_outliner_nodes().is_empty());
    assert!(controller.selection().items().is_empty());
    assert!(!controller.is_dirty());
    assert_eq!(controller.model_revision(), 0);
    assert_eq!(controller.document_history_lengths(), (0, 0));
    assert!(matches!(
        install_scene_authoring_commands(&mut controller),
        Err(ControllerError::DuplicateCommand(_))
    ));
    assert_eq!(controller.document_objects().len(), 0);
    assert_eq!(controller.revision(), revision);
}

#[test]
fn triangle_create_delete_undo_redo_updates_real_ecs_and_scene() {
    let mut controller = EditorController::new();
    install_scene_authoring_commands(&mut controller).unwrap();
    let mut session = EditorSceneSession::new(&controller).unwrap();
    assert!(session.scene().objects().is_empty());

    execute(&mut controller, SCENE_ADD_TRIANGLE_COMMAND);
    let item = primary(&controller);
    let expected = mesh(&controller, &item);
    expected.validate().unwrap();
    assert_eq!(expected.positions.len(), 3);
    assert_eq!(expected.indices, vec![0, 1, 2]);
    assert_eq!(expected.matrix, SCENE_ID_IDENTITY_MATRIX);
    assert!(expected.visible);
    assert_eq!(controller.document_objects().len(), 1);
    assert_eq!(controller.document_outliner_nodes().len(), 1);
    assert_eq!(
        controller.document_outliner_nodes()[0].selection(),
        Some(&item)
    );
    assert_eq!(controller.selection().items(), core::slice::from_ref(&item));
    assert_eq!(controller.document_history_lengths(), (1, 0));
    assert!(controller.is_dirty());
    assert!(session.synchronize(&controller).unwrap());
    assert_eq!(session.bridge().entity_count(), 1);
    let entity = session.bridge().entity_for_object(&item).unwrap();
    assert_eq!(
        session.world().get::<EditorSceneMesh>(entity),
        Some(&expected)
    );
    let scene_mesh = &session.scene().objects()[0];
    assert_eq!(scene_mesh.selection, item);
    assert_eq!(scene_mesh.positions, expected.positions);
    assert_eq!(scene_mesh.indices, expected.indices);
    assert_eq!(scene_mesh.model, expected.matrix);
    assert!(scene_mesh.visible);

    controller.execute(EditorAction::Undo).unwrap();
    assert!(controller.document_outliner_nodes().is_empty());
    assert!(controller.selection().items().is_empty());
    assert!(session.synchronize(&controller).unwrap());
    assert_eq!(session.bridge().entity_count(), 0);
    assert!(session.scene().objects().is_empty());
    controller.execute(EditorAction::Redo).unwrap();
    assert_eq!(primary(&controller), item);
    assert_eq!(mesh(&controller, &item), expected);
    assert!(session.synchronize(&controller).unwrap());

    execute(&mut controller, SCENE_DELETE_SELECTION_COMMAND);
    assert!(controller.selection().items().is_empty());
    assert!(controller.document_outliner_nodes().is_empty());
    assert_eq!(controller.document_objects().len(), 0);
    assert!(session.synchronize(&controller).unwrap());
    assert!(session.scene().objects().is_empty());
    controller.execute(EditorAction::Undo).unwrap();
    assert_eq!(primary(&controller), item);
    assert_eq!(mesh(&controller, &item), expected);
    assert!(session.synchronize(&controller).unwrap());
    assert_eq!(session.bridge().entity_count(), 1);
    controller.execute(EditorAction::Redo).unwrap();
    assert!(session.synchronize(&controller).unwrap());
    assert!(session.scene().objects().is_empty());
    assert!(controller.is_dirty());
}

#[test]
fn creation_restores_selection_and_never_reuses_discarded_redo_identity() {
    let mut controller = EditorController::new();
    install_scene_authoring_commands(&mut controller).unwrap();
    execute(&mut controller, SCENE_ADD_TRIANGLE_COMMAND);
    let first = primary(&controller);
    execute(&mut controller, SCENE_ADD_TRIANGLE_COMMAND);
    let second = primary(&controller);
    let second_row = controller.selected_outliner().unwrap();
    assert_ne!(first, second);
    controller.execute(EditorAction::Undo).unwrap();
    assert_eq!(primary(&controller), first);
    controller.execute(EditorAction::Redo).unwrap();
    assert_eq!(primary(&controller), second);
    assert_eq!(controller.selected_outliner(), Some(second_row));
    controller.execute(EditorAction::Undo).unwrap();
    execute(&mut controller, SCENE_ADD_TRIANGLE_COMMAND);
    let third = primary(&controller);
    assert_ne!(third, first);
    assert_ne!(third, second);
    assert_ne!(controller.selected_outliner(), Some(second_row));
    assert_eq!(controller.document_history_lengths(), (2, 0));
}

#[test]
fn bootstrap_identities_and_document_roundtrip_admit_further_authoring() {
    let mut controller = EditorController::new();
    install_scene_authoring_commands(&mut controller).unwrap();
    execute(&mut controller, SCENE_ADD_TRIANGLE_COMMAND);
    let expected = mesh(&controller, &primary(&controller));

    let mut loaded = EditorController::new();
    let bootstrap = SelectionItem::entity(700);
    loaded
        .register_object(
            bootstrap.clone(),
            Box::new(expected.clone()),
            editor_scene_mesh_registry(),
        )
        .unwrap();
    let mut row = OutlinerNode::new(OutlinerId::from_raw(900), None, "Persisted mesh");
    row.set_selection(bootstrap.clone());
    loaded.register_outliner_node(row).unwrap();
    assert!(!loaded.is_dirty());
    let next = loaded.next_entity_identity().unwrap();
    assert_ne!(next.0, bootstrap);
    assert!(next.1.raw() > 900);

    let catalog = editor_scene_mesh_catalog().unwrap();
    let bytes = DocumentCodec::encode(&loaded, &catalog, DocumentLimits::default()).unwrap();
    let mut decoded = DocumentCodec::decode(
        &bytes,
        &catalog,
        DocumentLimits::default(),
        ControllerLimits::default(),
    )
    .unwrap()
    .into_controller();
    assert!(!decoded.has_command(SCENE_ADD_TRIANGLE_COMMAND));
    assert_eq!(mesh(&decoded, &bootstrap), expected);
    install_scene_authoring_commands(&mut decoded).unwrap();
    execute(&mut decoded, SCENE_ADD_TRIANGLE_COMMAND);
    assert_eq!(primary(&decoded), next.0);
    assert_eq!(decoded.selected_outliner(), Some(next.1));
    assert_eq!(decoded.document_objects().len(), 2);
    let session = EditorSceneSession::new(&decoded).unwrap();
    assert_eq!(session.bridge().entity_count(), 2);
    assert_eq!(session.scene().objects().len(), 2);
}

#[test]
fn failed_creation_budgets_do_not_mutate_model_or_consume_identity() {
    let defaults = ControllerLimits::default();
    for limits in [
        ControllerLimits {
            max_objects: 0,
            ..defaults
        },
        ControllerLimits {
            max_outliner_nodes: 0,
            ..defaults
        },
        ControllerLimits {
            max_selection: 0,
            ..defaults
        },
        ControllerLimits {
            max_history: 0,
            ..defaults
        },
        ControllerLimits {
            max_history_per_object: 0,
            ..defaults
        },
    ] {
        let mut controller = EditorController::with_limits(limits);
        install_scene_authoring_commands(&mut controller).unwrap();
        let suggested = controller.next_entity_identity().unwrap();
        let revision = controller.revision();
        assert!(controller
            .execute(EditorAction::ExecuteCommand {
                command_id: SCENE_ADD_TRIANGLE_COMMAND.into(),
            })
            .is_err());
        assert_eq!(controller.next_entity_identity().unwrap(), suggested);
        assert_eq!(controller.document_objects().len(), 0);
        assert!(controller.document_outliner_nodes().is_empty());
        assert!(controller.selection().items().is_empty());
        assert_eq!(controller.document_history_lengths(), (0, 0));
        assert_eq!(controller.revision(), revision);
        assert_eq!(controller.model_revision(), 0);
        assert!(!controller.is_dirty());
    }
}

#[test]
fn command_budget_failure_is_atomic_without_creating_sample_objects() {
    let mut controller = EditorController::with_limits(ControllerLimits {
        max_commands: 1,
        ..ControllerLimits::default()
    });
    assert!(install_scene_authoring_commands(&mut controller).is_err());
    assert!(!controller.has_command(SCENE_ADD_TRIANGLE_COMMAND));
    assert!(!controller.has_command(SCENE_DELETE_SELECTION_COMMAND));
    assert_eq!(controller.document_objects().len(), 0);
    assert_eq!(controller.model_revision(), 0);
    assert_eq!(controller.revision(), 0);
    assert!(!controller.is_dirty());
}

#[test]
fn batch_registration_rejects_duplicate_invalid_and_over_budget_ids_atomically() {
    use prism_editor_app::ControllerAuthoringCommand;
    fn factory(_: &EditorController) -> Result<ControllerAuthoringCommand, ControllerError> {
        Ok(ControllerAuthoringCommand::Delete(Vec::new()))
    }
    let mut controller = EditorController::with_limits(ControllerLimits {
        max_commands: 2,
        ..ControllerLimits::default()
    });
    for ids in [["new", "new"], ["new", ""], ["new", " "]] {
        assert!(controller
            .register_authoring_commands(ids.map(|id| (id.into(), factory)))
            .is_err());
        assert!(!controller.has_command("new"));
        assert_eq!(controller.revision(), 0);
    }
    controller
        .register_command("existing", |_| Ok(EditorAction::Undo))
        .unwrap();
    let revision = controller.revision();
    for ids in [["new", "existing"], ["new", "second"]] {
        assert!(controller
            .register_authoring_commands(ids.map(|id| (id.into(), factory)))
            .is_err());
        assert!(controller.has_command("existing"));
        assert!(!controller.has_command("new"));
        assert_eq!(controller.revision(), revision);
    }
    controller
        .register_authoring_commands([("new".into(), factory)])
        .unwrap();
    assert!(controller.has_command("new"));
    assert_eq!(controller.revision(), revision + 1);
    assert_eq!(controller.model_revision(), 0);
    assert!(!controller.is_dirty());
}

#[test]
fn failed_second_add_keeps_existing_saved_mesh_selection_and_history() {
    let mut controller = EditorController::with_limits(ControllerLimits {
        max_objects: 1,
        ..ControllerLimits::default()
    });
    install_scene_authoring_commands(&mut controller).unwrap();
    execute(&mut controller, SCENE_ADD_TRIANGLE_COMMAND);
    let item = primary(&controller);
    let expected = mesh(&controller, &item);
    let rows = controller.document_outliner_nodes().to_vec();
    let model_revision = controller.model_revision();
    controller.mark_saved(model_revision).unwrap();
    let revision = controller.revision();
    let suggested = controller.next_entity_identity().unwrap();
    assert!(controller
        .execute(EditorAction::ExecuteCommand {
            command_id: SCENE_ADD_TRIANGLE_COMMAND.into(),
        })
        .is_err());
    assert_eq!(controller.document_objects().len(), 1);
    assert_eq!(mesh(&controller, &item), expected);
    assert_eq!(controller.document_outliner_nodes(), rows);
    assert_eq!(primary(&controller), item);
    assert_eq!(controller.document_history_lengths(), (1, 0));
    assert_eq!(controller.revision(), revision);
    assert_eq!(controller.model_revision(), model_revision);
    assert_eq!(controller.next_entity_identity().unwrap(), suggested);
    assert!(!controller.is_dirty());
}

#[test]
fn delete_selection_uses_recursive_alias_closure_and_undo_restores_it() {
    let mut controller = EditorController::new();
    install_scene_authoring_commands(&mut controller).unwrap();
    execute(&mut controller, SCENE_ADD_TRIANGLE_COMMAND);
    let parent = primary(&controller);
    let parent_row = controller.selected_outliner().unwrap();
    execute(&mut controller, SCENE_ADD_TRIANGLE_COMMAND);
    let child = primary(&controller);
    let child_mesh = mesh(&controller, &child);
    // A second alias of the child lives under the parent. Deleting the parent
    // must delete the descendant object and its root alias as one transaction.
    let mut alias = OutlinerNode::new(OutlinerId::from_raw(100), Some(parent_row), "Child alias");
    alias.set_selection(child.clone());
    controller.register_outliner_node(alias).unwrap();
    controller.select(parent.clone()).unwrap();
    let rows = controller.document_outliner_nodes().to_vec();
    execute(&mut controller, SCENE_DELETE_SELECTION_COMMAND);
    assert_eq!(controller.document_objects().len(), 0);
    assert!(controller.document_outliner_nodes().is_empty());
    assert!(controller.selection().items().is_empty());
    controller.execute(EditorAction::Undo).unwrap();
    assert_eq!(controller.document_objects().len(), 2);
    assert_eq!(controller.document_outliner_nodes(), rows);
    assert_eq!(primary(&controller), parent);
    assert_eq!(mesh(&controller, &child), child_mesh);
    controller.execute(EditorAction::Redo).unwrap();
    assert_eq!(controller.document_objects().len(), 0);
    assert!(controller.document_outliner_nodes().is_empty());
}

#[test]
fn exhausted_identity_rejects_add_without_wrapping_or_creating_geometry() {
    let mut controller = EditorController::new();
    controller
        .register_outliner_node(OutlinerNode::new(
            OutlinerId::from_raw(u128::MAX),
            None,
            "Identity boundary",
        ))
        .unwrap();
    install_scene_authoring_commands(&mut controller).unwrap();
    let revision = controller.revision();
    let rows = controller.document_outliner_nodes().to_vec();
    assert!(matches!(
        controller.execute(EditorAction::ExecuteCommand {
            command_id: SCENE_ADD_TRIANGLE_COMMAND.into(),
        }),
        Err(ControllerError::RevisionExhausted)
    ));
    assert_eq!(controller.document_objects().len(), 0);
    assert_eq!(controller.document_outliner_nodes(), rows);
    assert_eq!(controller.document_history_lengths(), (0, 0));
    assert_eq!(controller.revision(), revision);
    assert!(!controller.is_dirty());
}

#[test]
fn conflicting_delete_id_is_detected_before_installing_add_factory() {
    let mut controller = EditorController::new();
    controller
        .register_command(SCENE_DELETE_SELECTION_COMMAND, |_| Ok(EditorAction::Undo))
        .unwrap();
    let revision = controller.revision();
    assert!(matches!(
        install_scene_authoring_commands(&mut controller),
        Err(ControllerError::DuplicateCommand(_))
    ));
    assert!(!controller.has_command(SCENE_ADD_TRIANGLE_COMMAND));
    assert_eq!(controller.revision(), revision);
    assert_eq!(controller.document_objects().len(), 0);
    assert!(!controller.is_dirty());
}
