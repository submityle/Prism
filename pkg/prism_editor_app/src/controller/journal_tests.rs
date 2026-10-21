use super::*;

#[derive(Clone, Reflect)]
struct Model {
    matrix: [[f32; 2]; 2],
    visible: bool,
}

#[test]
fn field_journal_coalesces_nested_fields_and_tracks_history_navigation() {
    let mut controller = EditorController::new();
    let item = SelectionItem::entity(1);
    let mut registry = TypeRegistry::new();
    registry.register::<Model>();
    controller
        .register_object(
            item.clone(),
            Box::new(Model {
                matrix: [[0.0; 2]; 2],
                visible: true,
            }),
            registry,
        )
        .unwrap();
    let shape = controller.shape_revision();
    controller.select(item.clone()).unwrap();
    assert_eq!(controller.shape_revision(), shape);
    let prepared = controller
        .prepare_edits(vec![
            ControllerFieldEdit {
                item: item.clone(),
                path: FieldPath::parse("matrix[0][0]").unwrap(),
                value: ActionValue::float(1.0),
            },
            ControllerFieldEdit {
                item: item.clone(),
                path: FieldPath::parse("matrix[1][1]").unwrap(),
                value: ActionValue::float(1.0),
            },
        ])
        .unwrap();
    controller.commit_edits(prepared).unwrap();
    assert_eq!(
        controller.changed_fields_since(0).unwrap()[&item],
        BTreeSet::from(["matrix".to_owned()])
    );
    controller.execute(EditorAction::Undo).unwrap();
    assert_eq!(
        controller.changed_fields_since(1).unwrap()[&item],
        BTreeSet::from(["matrix".to_owned()])
    );
    controller.execute(EditorAction::Redo).unwrap();
    assert_eq!(
        controller.changed_objects_since(2),
        Some(vec![item.clone()])
    );
    controller
        .execute(EditorAction::EditInspectorField {
            path: FieldPath::parse("visible").unwrap(),
            value: ActionValue::bool(false),
        })
        .unwrap();
    assert_eq!(
        controller.changed_fields_since(0).unwrap()[&item],
        BTreeSet::from(["matrix".to_owned(), "visible".to_owned()])
    );
    assert_eq!(controller.shape_revision(), shape);
    assert!(controller.changed_fields_since(5).is_none());
}

#[test]
fn bounded_journal_requests_full_resync_after_history_gap() {
    let mut controller = EditorController::with_limits(ControllerLimits {
        max_history: 1,
        ..ControllerLimits::default()
    });
    let item = SelectionItem::entity(1);
    let mut registry = TypeRegistry::new();
    registry.register::<Model>();
    controller
        .register_object(
            item.clone(),
            Box::new(Model {
                matrix: [[0.0; 2]; 2],
                visible: true,
            }),
            registry,
        )
        .unwrap();
    controller.select(item).unwrap();
    controller
        .execute(EditorAction::EditInspectorField {
            path: FieldPath::parse("visible").unwrap(),
            value: ActionValue::bool(false),
        })
        .unwrap();
    controller.execute(EditorAction::Undo).unwrap();
    assert!(controller.changed_fields_since(0).is_none());
    assert!(controller.changed_fields_since(1).is_some());
}
