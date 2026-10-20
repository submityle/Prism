//! Real model, bound UI events, reversible commands, and writeback acceptance.

use prism_editor_app::{ControllerError, ControllerLimits, EditorController};
use prism_editor_core::SelectionItem;
use prism_editor_panels::{OutlinerId, OutlinerNode};
use prism_editor_reflect::{EditorError, FieldPath};
use prism_editor_ui::{
    inspector_view_bound, outliner_panel_bound, ActionBinding, ActionBindings, ActionTemplate,
    ActionValue, EditorAction, InspectorViewProps, OutlinerViewProps, SelectionMode, UiEvent,
};
use prism_reflect::{DynamicStruct, FieldMetadata, Reflect, TypeMetadata, TypeRegistry};

#[derive(Clone, Debug, PartialEq, Reflect)]
struct Model {
    speed: i8,
    count: u8,
    gain: f32,
    double: f64,
    label: String,
    enabled: bool,
    glyph: char,
    locked: u32,
    values: Vec<i32>,
}

fn model() -> Model {
    Model {
        speed: 10,
        count: 1,
        gain: 0.25,
        double: 0.5,
        label: "船".into(),
        enabled: true,
        glyph: '船',
        locked: 7,
        values: vec![1, 2],
    }
}

fn registry() -> TypeRegistry {
    let mut r = TypeRegistry::new();
    r.register::<Model>();
    r.register_type_data::<Model, _>(
        TypeMetadata::new()
            .with_field(FieldMetadata::new("locked").readonly(true))
            .with_field(FieldMetadata::new("gain").with_range(0.0, 1.0)),
    );
    r
}

fn loaded(limits: ControllerLimits) -> EditorController {
    let mut c = EditorController::with_limits(limits);
    for id in 1..=3 {
        let item = SelectionItem::entity(id);
        c.register_object(item.clone(), Box::new(model()), registry())
            .expect("load");
        let mut node = OutlinerNode::new(OutlinerId::from_raw(id), None, format!("Object {id}"));
        node.set_selection(item);
        c.register_outliner_node(node).expect("row");
    }
    c.select(SelectionItem::entity(1)).expect("select");
    c
}

fn snapshot(c: &EditorController) -> Model {
    snapshot_of(c, c.selection().primary().expect("primary"))
}

fn snapshot_of(c: &EditorController, item: &SelectionItem) -> Model {
    let mut actual = model();
    actual
        .apply(&*c.snapshot(item).expect("snapshot"))
        .expect("writeback");
    actual
}

fn document_models(c: &EditorController) -> Vec<(SelectionItem, Model)> {
    c.document_objects()
        .map(|(item, _, _)| (item.clone(), snapshot_of(c, item)))
        .collect()
}

fn edit(path: &str, value: ActionValue) -> EditorAction {
    EditorAction::EditInspectorField {
        path: FieldPath::parse(path).expect("path"),
        value,
    }
}

fn select(id: u128, mode: SelectionMode) -> EditorAction {
    EditorAction::SelectOutliner {
        node: OutlinerId::from_raw(id),
        mode,
    }
}

fn rejected(c: &mut EditorController, action: EditorAction) -> ControllerError {
    let before = document_models(c);
    let selection = c.selection().clone();
    let outliner = c.outliner().clone();
    let state = (c.revision(), c.model_revision(), c.is_dirty());
    let history = c.document_history_lengths();
    let object_histories: Vec<_> = c
        .document_objects()
        .map(|(item, _, _)| (item.clone(), c.history_lengths(item).expect("history")))
        .collect();
    let error = c.execute(action).expect_err("reject");
    assert_eq!(document_models(c), before);
    assert_eq!(c.selection(), &selection);
    assert_eq!(c.outliner(), &outliner);
    assert_eq!((c.revision(), c.model_revision(), c.is_dirty()), state);
    assert_eq!(c.document_history_lengths(), history);
    for (item, counts) in object_histories {
        assert_eq!(c.history_lengths(&item).expect("history"), counts);
    }
    error
}

#[test]
fn real_bound_panels_selection_field_history_and_model_writeback() {
    let mut c = loaded(ControllerLimits::default());
    c.clear_selection().expect("clear");
    let panel = outliner_panel_bound(&OutlinerViewProps {
        outliner: c.outliner(),
        filter: "",
        start: 0,
        count: 10,
        selected: None,
    })
    .expect("bound panel");
    c.dispatch(
        panel.bindings(),
        "outliner:2",
        &UiEvent::Select {
            mode: SelectionMode::Replace,
        },
    )
    .expect("select");
    assert_eq!(c.selected_outliner(), Some(OutlinerId::from_raw(2)));
    assert_eq!(c.selection().primary(), Some(&SelectionItem::entity(2)));
    let projection = c.inspector_projection().expect("projection");
    let panel = inspector_view_bound(&InspectorViewProps {
        projection: &projection,
        show_root: false,
    })
    .expect("inspector");
    let result = c
        .dispatch(
            panel.bindings(),
            "field:.speed",
            &UiEvent::Commit {
                value: ActionValue::text("42").expect("text"),
            },
        )
        .expect("edit");
    assert!(result.changed && result.model_changed && c.is_dirty());
    assert_eq!(snapshot(&c).speed, 42);
    let mut bindings = ActionBindings::new();
    bindings.bind_undo("undo").expect("bind");
    bindings.bind_redo("redo").expect("bind");
    c.dispatch(&bindings, "undo", &UiEvent::Activate)
        .expect("undo");
    assert_eq!(snapshot(&c).speed, 10);
    c.dispatch(&bindings, "redo", &UiEvent::Activate)
        .expect("redo");
    assert_eq!(snapshot(&c).speed, 42);
    assert_eq!(
        c.primary_projection()
            .expect("projection")
            .root
            .children
            .len(),
        9
    );
}

#[test]
fn conversion_failures_preserve_model_selection_revisions_and_redo() {
    let mut c = loaded(ControllerLimits::default());
    c.execute(edit("speed", ActionValue::integer(12)))
        .expect("edit");
    c.execute(EditorAction::Undo).expect("undo");
    for action in [
        edit("speed", ActionValue::integer(128)),
        edit("speed", ActionValue::unsigned(u64::MAX)),
        edit("count", ActionValue::integer(-1)),
        edit("count", ActionValue::unsigned(256)),
        edit("speed", ActionValue::float(2.0)),
        edit("speed", ActionValue::bool(true)),
        edit("speed", ActionValue::text("1.5").expect("text")),
        edit("gain", ActionValue::float(f64::MAX)),
        edit("gain", ActionValue::float(1e-100)),
        edit("double", ActionValue::float(f64::NAN)),
        edit("double", ActionValue::float(f64::INFINITY)),
        edit("double", ActionValue::float(f64::NEG_INFINITY)),
        edit("double", ActionValue::text("NaN").expect("text")),
        edit("double", ActionValue::text("1e999").expect("text")),
        edit("double", ActionValue::unsigned(9_007_199_254_740_993)),
        edit("gain", ActionValue::unsigned(16_777_217)),
        edit("label", ActionValue::integer(3)),
        edit("enabled", ActionValue::integer(1)),
        edit("glyph", ActionValue::text("ab").expect("text")),
        edit("values", ActionValue::text("[3]").expect("text")),
    ] {
        assert!(matches!(
            rejected(&mut c, action),
            ControllerError::ValueConversion { .. }
        ));
    }
    rejected(&mut c, edit("locked", ActionValue::unsigned(7)));
    assert!(matches!(
        rejected(&mut c, edit("gain", ActionValue::float(2.0))),
        ControllerError::Edit(EditorError::Validation(_))
    ));
    assert!(matches!(
        rejected(&mut c, edit("missing", ActionValue::integer(1))),
        ControllerError::PathNotFound(_)
    ));
    assert!(matches!(
        rejected(&mut c, select(999, SelectionMode::Replace)),
        ControllerError::Outliner(_)
    ));
    assert!(matches!(
        rejected(
            &mut c,
            EditorAction::ExecuteCommand {
                command_id: "unknown".into()
            }
        ),
        ControllerError::UnknownCommand(_)
    ));
    c.execute(EditorAction::Redo).expect("preserved redo");
    assert_eq!(snapshot(&c).speed, 12);
}

#[test]
fn accepted_scalars_list_paths_and_same_value_no_op() {
    let mut c = loaded(ControllerLimits::default());
    for action in [
        edit("speed", ActionValue::integer(-128)),
        edit("count", ActionValue::unsigned(255)),
        edit("gain", ActionValue::text("0.75").expect("text")),
        edit("double", ActionValue::unsigned(1_u64 << 63)),
        edit("enabled", ActionValue::text("false").expect("text")),
        edit("label", ActionValue::text("飞船🚀").expect("text")),
        edit("glyph", ActionValue::text("🚀").expect("text")),
        edit("values[1]", ActionValue::integer(99)),
    ] {
        c.execute(action).expect("edit");
    }
    let v = snapshot(&c);
    assert_eq!(
        (v.speed, v.count, v.gain, v.double),
        (-128, 255, 0.75, (1_u64 << 63) as f64)
    );
    assert_eq!(
        (v.enabled, v.label.as_str(), v.glyph, v.values[1]),
        (false, "飞船🚀", '🚀', 99)
    );
    let revision = c.revision();
    assert!(
        !c.execute(edit("speed", ActionValue::integer(-128)))
            .expect("same")
            .changed
    );
    assert_eq!(c.revision(), revision);
}

#[test]
fn document_history_undoes_last_command_across_selection_and_without_primary() {
    let mut c = loaded(ControllerLimits::default());
    let a = SelectionItem::entity(1);
    let b = SelectionItem::entity(2);
    c.execute(edit("speed", ActionValue::integer(11)))
        .expect("A edit");
    c.execute(select(2, SelectionMode::Replace))
        .expect("B select");
    c.execute(edit("speed", ActionValue::integer(22)))
        .expect("B edit");
    c.execute(select(1, SelectionMode::Replace))
        .expect("A select");
    assert_eq!(c.document_history_lengths(), (2, 0));
    let selection = c.selection().clone();
    let revision = c.model_revision();
    let outcome = c
        .execute(EditorAction::Undo)
        .expect("undo B while A selected");
    assert!(outcome.changed && outcome.model_changed);
    assert_eq!(c.model_revision(), revision + 1);
    assert_eq!(c.selection(), &selection);
    assert_eq!(snapshot_of(&c, &a).speed, 11);
    assert_eq!(snapshot_of(&c, &b).speed, 10);
    assert_eq!(c.history_lengths(&a).unwrap(), (1, 0));
    assert_eq!(c.history_lengths(&b).unwrap(), (0, 1));

    c.clear_selection().expect("clear");
    c.execute(EditorAction::Undo).expect("unselected undo A");
    assert_eq!(snapshot_of(&c, &a).speed, 10);
    assert_eq!(c.document_history_lengths(), (0, 2));
    let revision = c.revision();
    assert!(!c.execute(EditorAction::Undo).unwrap().changed);
    assert_eq!(c.revision(), revision);
    c.execute(EditorAction::Redo)
        .expect("unselected redo A first");
    assert_eq!(snapshot_of(&c, &a).speed, 11);
    assert_eq!(snapshot_of(&c, &b).speed, 10);
    c.execute(EditorAction::Redo)
        .expect("unselected redo B second");
    assert_eq!(snapshot_of(&c, &b).speed, 22);
    assert!(c.selection().primary().is_none());
    assert_eq!(c.document_history_lengths(), (2, 0));
    let revision = c.revision();
    assert!(!c.execute(EditorAction::Redo).unwrap().changed);
    assert_eq!(c.revision(), revision);
}

#[test]
fn new_document_branch_discards_other_objects_redo_only_after_success() {
    let mut c = loaded(ControllerLimits::default());
    let a = SelectionItem::entity(1);
    let b = SelectionItem::entity(2);
    c.execute(edit("speed", ActionValue::integer(11))).unwrap();
    c.select(b.clone()).unwrap();
    c.execute(edit("speed", ActionValue::integer(22))).unwrap();
    c.execute(EditorAction::Undo).unwrap();
    c.execute(EditorAction::Undo).unwrap();
    c.select(a.clone()).unwrap();
    rejected(&mut c, edit("gain", ActionValue::float(2.0)));
    let revision = (c.revision(), c.model_revision());
    assert!(
        !c.execute(edit("speed", ActionValue::integer(10)))
            .unwrap()
            .changed
    );
    assert_eq!((c.revision(), c.model_revision()), revision);
    assert_eq!(c.document_history_lengths(), (0, 2));
    c.execute(EditorAction::Redo)
        .expect("A redo survives failed/no-op commits");
    assert_eq!(snapshot_of(&c, &a).speed, 11);
    c.execute(edit("speed", ActionValue::integer(33))).unwrap();
    assert_eq!(c.document_history_lengths(), (2, 0));
    assert_eq!(c.history_lengths(&b).unwrap(), (0, 0));
    assert!(!c.execute(EditorAction::Redo).unwrap().changed);
    assert_eq!(snapshot_of(&c, &b).speed, 10);
    c.execute(EditorAction::Undo).unwrap();
    assert_eq!(snapshot_of(&c, &a).speed, 11);
    c.execute(EditorAction::Undo).unwrap();
    assert_eq!(snapshot_of(&c, &a).speed, 10);
}

#[test]
fn global_capacity_rejects_without_eviction_and_reuses_discarded_redo_capacity() {
    let mut c = loaded(ControllerLimits {
        max_history: 2,
        ..ControllerLimits::default()
    });
    c.execute(edit("speed", ActionValue::integer(11))).unwrap();
    c.select(SelectionItem::entity(2)).unwrap();
    c.execute(edit("speed", ActionValue::integer(22))).unwrap();
    c.select(SelectionItem::entity(3)).unwrap();
    assert!(matches!(
        rejected(&mut c, edit("speed", ActionValue::integer(33))),
        ControllerError::BudgetExceeded {
            resource: "document history",
            limit: 2
        }
    ));
    assert!(
        !c.execute(edit("speed", ActionValue::integer(10)))
            .unwrap()
            .changed
    );
    c.execute(EditorAction::Undo).unwrap();
    rejected(&mut c, edit("speed", ActionValue::integer(128)));
    assert_eq!(c.document_history_lengths(), (1, 1));
    c.execute(edit("speed", ActionValue::integer(33)))
        .expect("new branch fits");
    assert_eq!(c.document_history_lengths(), (2, 0));
    assert_eq!(
        c.history_lengths(&SelectionItem::entity(2)).unwrap(),
        (0, 0)
    );
    c.clear_selection().unwrap();
    c.execute(EditorAction::Undo).unwrap();
    c.execute(EditorAction::Undo).unwrap();
    assert_eq!(snapshot_of(&c, &SelectionItem::entity(1)).speed, 10);
    assert_eq!(snapshot_of(&c, &SelectionItem::entity(3)).speed, 10);
    c.execute(EditorAction::Redo).unwrap();
    c.execute(EditorAction::Redo).unwrap();
    assert_eq!(snapshot_of(&c, &SelectionItem::entity(1)).speed, 11);
    assert_eq!(snapshot_of(&c, &SelectionItem::entity(3)).speed, 33);
    assert_eq!(c.document_history_lengths(), (2, 0));
}

#[test]
fn field_submissions_are_not_merged_and_zero_capacity_does_not_disable_no_ops() {
    let mut c = loaded(ControllerLimits::default());
    c.execute(edit("speed", ActionValue::integer(11))).unwrap();
    c.execute(edit("speed", ActionValue::integer(12))).unwrap();
    assert_eq!(c.document_history_lengths(), (2, 0));
    c.execute(EditorAction::Undo).unwrap();
    assert_eq!(snapshot(&c).speed, 11);
    c.execute(EditorAction::Undo).unwrap();
    assert_eq!(snapshot(&c).speed, 10);

    let mut c = loaded(ControllerLimits {
        max_history: 0,
        ..ControllerLimits::default()
    });
    rejected(&mut c, edit("speed", ActionValue::integer(11)));
    assert!(
        !c.execute(edit("speed", ActionValue::integer(10)))
            .unwrap()
            .changed
    );
    assert!(!c.execute(EditorAction::Undo).unwrap().changed);
    assert!(!c.execute(EditorAction::Redo).unwrap().changed);
    assert!(!c.is_dirty());
}

#[test]
fn selection_modes_expansion_and_hidden_anchor_are_atomic() {
    let mut c = loaded(ControllerLimits::default());
    c.execute(select(3, SelectionMode::Extend)).expect("range");
    assert_eq!(c.selection().items().len(), 3);
    c.execute(select(2, SelectionMode::Toggle)).expect("toggle");
    c.execute(select(1, SelectionMode::Add)).expect("add");
    assert_eq!(c.selection().primary(), Some(&SelectionItem::entity(1)));
    assert!(!c.is_dirty());
    let parent = OutlinerId::from_raw(10);
    c.register_outliner_node(OutlinerNode::new(parent, None, "Group"))
        .expect("group");
    let mut child = OutlinerNode::new(OutlinerId::from_raw(11), Some(parent), "Child");
    child.set_selection(SelectionItem::entity(1));
    c.register_outliner_node(child).expect("child");
    let bound = outliner_panel_bound(&OutlinerViewProps {
        outliner: c.outliner(),
        filter: "",
        start: 0,
        count: 20,
        selected: None,
    })
    .expect("bound");
    c.execute(select(11, SelectionMode::Replace))
        .expect("anchor");
    c.dispatch(
        bound.bindings(),
        "outliner-disclosure:10",
        &UiEvent::Activate,
    )
    .expect("collapse");
    assert!(!c
        .outliner()
        .visible_window(0, 20)
        .contains(&OutlinerId::from_raw(11)));
    assert!(matches!(
        rejected(&mut c, select(3, SelectionMode::Extend)),
        ControllerError::InvisibleRangeEndpoint(_)
    ));
    assert!(matches!(
        rejected(&mut c, select(10, SelectionMode::Replace)),
        ControllerError::UnselectableNode(_)
    ));
    c.execute(EditorAction::ToggleOutlinerExpansion { node: parent })
        .expect("expand");
    assert_eq!(
        c.outliner().node(parent).expect("parent").children(),
        &[OutlinerId::from_raw(11)]
    );
}

#[test]
fn budgets_no_primary_and_registration_failures() {
    let mut c = loaded(ControllerLimits {
        max_history_per_object: 1,
        max_selection: 1,
        ..ControllerLimits::default()
    });
    c.execute(edit("speed", ActionValue::integer(11)))
        .expect("first");
    assert!(matches!(
        rejected(&mut c, edit("speed", ActionValue::integer(12))),
        ControllerError::BudgetExceeded { .. }
    ));
    rejected(&mut c, select(2, SelectionMode::Add));
    rejected(&mut c, select(3, SelectionMode::Extend));
    c.execute(EditorAction::Undo).expect("undo");
    c.execute(edit("speed", ActionValue::integer(13)))
        .expect("replace redo branch");
    c.clear_selection().expect("clear");
    c.execute(EditorAction::Undo).expect("undo without primary");
    assert_eq!(snapshot_of(&c, &SelectionItem::entity(1)).speed, 10);
    c.execute(EditorAction::Redo).expect("redo without primary");
    assert_eq!(snapshot_of(&c, &SelectionItem::entity(1)).speed, 13);
    assert!(c.selection().primary().is_none());
    let revision = c.revision();
    assert!(matches!(
        c.execute(edit("speed", ActionValue::integer(14))),
        Err(ControllerError::NoPrimary)
    ));
    assert!(matches!(
        c.inspector_projection(),
        Err(ControllerError::NoPrimary)
    ));
    assert!(matches!(
        c.primary_snapshot(),
        Err(ControllerError::NoPrimary)
    ));
    assert!(c.select(SelectionItem::entity(999)).is_err());
    assert_eq!(c.revision(), revision);

    let mut c = EditorController::with_limits(ControllerLimits {
        max_inspector_nodes: 2,
        max_inspector_depth: 1,
        max_objects: 1,
        max_outliner_nodes: 1,
        max_outliner_depth: 0,
        ..ControllerLimits::default()
    });
    assert!(c
        .register_object(SelectionItem::entity(1), Box::new(model()), registry())
        .is_err());
    let mut nested = DynamicStruct::new();
    nested.insert("a", vec![1_i32]);
    assert!(c
        .register_object(
            SelectionItem::entity(1),
            Box::new(nested),
            TypeRegistry::new()
        )
        .is_err());
    assert_eq!(c.revision(), 0);
    c.register_object(
        SelectionItem::entity(1),
        Box::new(1_i32),
        TypeRegistry::new(),
    )
    .expect("leaf");
    assert!(matches!(
        c.register_object(
            SelectionItem::entity(1),
            Box::new(1_i32),
            TypeRegistry::new()
        ),
        Err(ControllerError::DuplicateObject(_))
    ));
    assert!(c
        .register_object(
            SelectionItem::entity(2),
            Box::new(1_i32),
            TypeRegistry::new()
        )
        .is_err());
    let mut node = OutlinerNode::new(OutlinerId::from_raw(1), None, "one");
    node.set_selection(SelectionItem::entity(999));
    assert!(c.register_outliner_node(node).is_err());
    c.register_outliner_node(OutlinerNode::new(OutlinerId::from_raw(1), None, "one"))
        .expect("row");
    assert!(c
        .register_outliner_node(OutlinerNode::new(
            OutlinerId::from_raw(2),
            Some(OutlinerId::from_raw(1)),
            "two"
        ))
        .is_err());
    let mut c = EditorController::new();
    assert!(c
        .register_object(
            SelectionItem::entity(1),
            Box::new(f64::NAN),
            TypeRegistry::new()
        )
        .is_err());
    assert_eq!(c.revision(), 0);
}

#[test]
fn registered_command_execution_cycles_errors_and_binding_failures() {
    let mut c = loaded(ControllerLimits {
        max_commands: 3,
        max_command_depth: 2,
        ..ControllerLimits::default()
    });
    c.register_command("set-speed", |_| Ok(edit("speed", ActionValue::integer(44))))
        .expect("register");
    c.register_command("cycle", |_| {
        Ok(EditorAction::ExecuteCommand {
            command_id: "cycle".into(),
        })
    })
    .expect("cycle");
    c.register_command("invalid", |_| Err(ControllerError::NoPrimary))
        .expect("invalid");
    assert!(c
        .register_command("extra", |_| Ok(EditorAction::Undo))
        .is_err());
    assert!(matches!(
        c.register_command("set-speed", |_| Ok(EditorAction::Undo)),
        Err(ControllerError::DuplicateCommand(_))
    ));
    assert!(c.register_command("", |_| Ok(EditorAction::Undo)).is_err());
    let mut bindings = ActionBindings::new();
    bindings
        .bind(
            ActionBinding::activate(
                "button",
                ActionTemplate::execute_command("set-speed").expect("template"),
            )
            .expect("binding"),
        )
        .expect("bind");
    c.dispatch(&bindings, "button", &UiEvent::Activate)
        .expect("execute");
    assert_eq!(snapshot(&c).speed, 44);
    c.execute(EditorAction::Undo).expect("undo");
    assert_eq!(snapshot(&c).speed, 10);
    assert!(matches!(
        rejected(
            &mut c,
            EditorAction::ExecuteCommand {
                command_id: "cycle".into()
            }
        ),
        ControllerError::BudgetExceeded { .. }
    ));
    rejected(
        &mut c,
        EditorAction::ExecuteCommand {
            command_id: "invalid".into(),
        },
    );
    let revision = c.revision();
    assert!(c
        .dispatch(&bindings, "not-bound", &UiEvent::Activate)
        .is_err());
    assert_eq!(c.revision(), revision);
}

#[test]
fn save_acknowledgement_and_readonly_container_descendants() {
    let mut c = loaded(ControllerLimits::default());
    c.execute(edit("speed", ActionValue::integer(11)))
        .expect("edit");
    let saved = c.model_revision();
    c.execute(edit("speed", ActionValue::integer(12)))
        .expect("newer edit");
    let revision = c.revision();
    assert!(matches!(
        c.mark_saved(saved),
        Err(ControllerError::StaleRevision { .. })
    ));
    assert!(c.is_dirty());
    assert_eq!(c.revision(), revision);
    c.mark_saved(c.model_revision()).expect("current save");
    assert!(!c.is_dirty());
    c.execute(EditorAction::Undo).expect("undo saved change");
    assert!(c.is_dirty());
    let mut r = registry();
    r.register_type_data::<Model, _>(
        TypeMetadata::new().with_field(FieldMetadata::new("values").readonly(true)),
    );
    let mut c = EditorController::new();
    c.register_object(SelectionItem::entity(1), Box::new(model()), r)
        .expect("load");
    c.select(SelectionItem::entity(1)).expect("select");
    assert!(matches!(
        rejected(&mut c, edit("values[0]", ActionValue::integer(99))),
        ControllerError::Edit(EditorError::ReadOnly(_))
    ));
}

#[test]
fn document_borrows_actual_models_registries_and_all_outliner_rows() {
    let mut c = EditorController::new();
    for id in [2, 1] {
        c.register_object(SelectionItem::entity(id), Box::new(model()), registry())
            .unwrap();
    }
    let parent = OutlinerId::from_raw(10);
    let child = OutlinerId::from_raw(11);
    let mut row = OutlinerNode::new(parent, None, "collapsed");
    row.set_expanded(false);
    c.register_outliner_node(row).unwrap();
    let mut row = OutlinerNode::new(child, Some(parent), "hidden");
    row.set_visible(false);
    row.set_selection(SelectionItem::entity(1));
    c.register_outliner_node(row).unwrap();
    c.select(SelectionItem::entity(1)).unwrap();
    c.execute(edit("count", ActionValue::integer(23))).unwrap();
    let before = (c.revision(), c.model_revision(), c.is_dirty());

    let mut objects = c.document_objects();
    assert_eq!(objects.len(), 2);
    let (item, root, registry) = objects.next().unwrap();
    assert_eq!(item, &SelectionItem::entity(1));
    // A snapshot clone may be dynamic; this API must expose the actual root.
    assert_eq!(root.downcast_ref::<Model>().unwrap().count, 23);
    let metadata = registry
        .get_with_name(root.type_name())
        .unwrap()
        .data::<TypeMetadata>()
        .unwrap();
    assert!(metadata.field("locked").unwrap().is_readonly());
    assert_eq!(objects.len(), 1);
    assert_eq!(objects.next().unwrap().0, &SelectionItem::entity(2));
    assert_eq!(objects.len(), 0);
    assert!(objects.next().is_none());
    drop(objects);

    let rows = c.document_outliner_nodes();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].id(), parent);
    assert!(!rows[0].is_expanded());
    assert_eq!(rows[1].id(), child);
    assert_eq!(rows[1].parent(), Some(parent));
    assert!(!rows[1].is_visible());
    assert_eq!(rows[1].selection(), Some(&SelectionItem::entity(1)));
    assert!(!c.outliner().visible_window(0, usize::MAX).contains(&child));
    assert_eq!((c.revision(), c.model_revision(), c.is_dirty()), before);
    c.execute(EditorAction::ToggleOutlinerExpansion { node: parent })
        .unwrap();
    assert!(c.document_outliner_nodes()[0].is_expanded());
}

#[test]
fn multi_selection_commits_once_and_validates_all_items_before_mutation() {
    let first = SelectionItem::entity(1);
    let second = SelectionItem::entity(2);
    let missing = SelectionItem::entity(99);
    let mut c = EditorController::new();
    for (raw, item) in [(1, &first), (2, &second)] {
        c.register_object(item.clone(), Box::new(model()), registry())
            .unwrap();
        let mut row = OutlinerNode::new(OutlinerId::from_raw(raw), None, "object");
        row.set_selection(item.clone());
        c.register_outliner_node(row).unwrap();
    }
    let before = c.revision();
    assert!(
        c.select_items(&[first.clone(), second.clone()], SelectionMode::Replace)
            .expect("multi-select")
            .changed
    );
    assert_eq!(c.selection().items(), &[first.clone(), second.clone()]);
    assert_eq!(c.revision(), before + 1);
    let snapshot = c.selection().clone();
    assert!(matches!(
        c.select_items(&[first, missing], SelectionMode::Toggle),
        Err(ControllerError::MissingObject(_))
    ));
    assert_eq!(c.selection(), &snapshot);
}
