//! Generic aggregate admission, cached scalar history and profile transactions.

extern crate alloc;

use alloc::sync::Arc;
use core::any::type_name;
use std::sync::Mutex;

use prism_editor_app::{
    ControllerAuthoringCommand, ControllerDocumentBudget, ControllerError, ControllerFieldEdit,
    ControllerLimits, ControllerObject, EditorController,
};
use prism_editor_core::{Selection, SelectionItem};
use prism_editor_panels::{OutlinerId, OutlinerNode};
use prism_editor_reflect::FieldPath;
use prism_editor_ui::{ActionValue, EditorAction, SelectionMode};
use prism_reflect::{DynamicStruct, Reflect, ReflectRef, TypeRegistry};

const UNITS: &str = "test.units";
const SECOND: &str = "test.second";

#[derive(Reflect)]
struct Model {
    units: i64,
    extra: i64,
}

fn item(id: u128) -> SelectionItem {
    SelectionItem::entity(id)
}

// Do not downcast the root: prepared candidates and decoded documents can be
// DynamicStruct even when bootstrap used a derived concrete model.
fn field(root: &dyn Reflect, name: &str) -> Result<i64, String> {
    let ReflectRef::Struct(root) = root.reflect_ref() else {
        return Err("expected struct".into());
    };
    root.field(name)
        .and_then(|value| value.downcast_ref::<i64>())
        .copied()
        .ok_or_else(|| format!("missing i64 field {name}"))
}

fn contribution(_: &SelectionItem, root: &dyn Reflect) -> Result<u64, String> {
    let units = field(root, "units")?;
    let extra = field(root, "extra")?;
    let total = units.checked_add(extra).ok_or("cost overflow")?;
    u64::try_from(total).map_err(|_| "negative cost".into())
}

fn budget(limit: u64) -> ControllerDocumentBudget {
    ControllerDocumentBudget::new(UNITS, limit, contribution)
}

fn registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    registry.register::<Model>();
    registry.register::<i64>();
    registry
}

fn object(id: u128, units: i64, dynamic: bool) -> ControllerObject {
    let root: Box<dyn Reflect> = if dynamic {
        let mut root = DynamicStruct::new();
        root.set_represented_type_name(type_name::<Model>());
        root.insert("units", units);
        root.insert("extra", 0i64);
        Box::new(root)
    } else {
        Box::new(Model { units, extra: 0 })
    };
    let mut row = OutlinerNode::new(OutlinerId::from_raw(id), None, format!("row {id}"));
    row.set_selection(item(id));
    ControllerObject {
        item: item(id),
        root,
        registry: registry(),
        row,
    }
}

fn bootstrap(c: &mut EditorController, id: u128, units: i64, dynamic: bool) {
    let object = object(id, units, dynamic);
    c.register_object(object.item, object.root, object.registry)
        .unwrap();
    c.register_outliner_node(object.row).unwrap();
}

fn edit(id: u128, name: &str, value: i64) -> ControllerFieldEdit {
    ControllerFieldEdit {
        item: item(id),
        path: FieldPath::parse(name).unwrap(),
        value: ActionValue::integer(value),
    }
}

fn apply(c: &mut EditorController, edits: Vec<ControllerFieldEdit>) {
    let prepared = c.prepare_edits(edits).unwrap();
    c.commit_edits(prepared).unwrap();
}

#[derive(Debug, PartialEq)]
struct Snapshot {
    objects: Vec<(SelectionItem, i64, i64)>,
    rows: Vec<OutlinerNode>,
    selection: Selection,
    revisions: (u64, u64, u64),
    history: (usize, usize),
    dirty: bool,
    next_identity: Option<(SelectionItem, OutlinerId)>,
    usage: [Option<(u64, u64)>; 2],
}

fn snapshot(c: &EditorController) -> Snapshot {
    Snapshot {
        objects: c
            .document_objects()
            .map(|(item, root, _)| {
                (
                    item.clone(),
                    field(root, "units").unwrap(),
                    field(root, "extra").unwrap(),
                )
            })
            .collect(),
        rows: c.document_outliner_nodes().to_vec(),
        selection: c.selection().clone(),
        revisions: (c.revision(), c.model_revision(), c.shape_revision()),
        history: c.document_history_lengths(),
        dirty: c.is_dirty(),
        next_identity: c.next_entity_identity().ok(),
        usage: [
            c.document_budget_usage(UNITS),
            c.document_budget_usage(SECOND),
        ],
    }
}

fn assert_exceeded<T>(result: Result<T, ControllerError>, id: &'static str, limit: u64) {
    assert!(matches!(
        result,
        Err(ControllerError::DocumentBudgetExceeded { budget, limit: actual })
            if budget == id && actual == limit
    ));
}

fn empty_command(_: &EditorController) -> Result<ControllerAuthoringCommand, ControllerError> {
    Ok(ControllerAuthoringCommand::Delete(Vec::new()))
}

#[test]
fn bootstrap_over_budget_rejects_root_and_budget_installation_atomically() {
    for dynamic in [false, true] {
        let mut c = EditorController::new();
        c.register_document_budgets([budget(10)]).unwrap();
        bootstrap(&mut c, 1, 6, dynamic);
        c.select(item(1)).unwrap();
        let before = snapshot(&c);
        let root = object(99, 5, dynamic);
        assert_exceeded(
            c.register_object(root.item, root.root, root.registry),
            UNITS,
            10,
        );
        assert_eq!(snapshot(&c), before);
        assert!(c.object_root(&item(99)).is_err());
        bootstrap(&mut c, 2, 4, dynamic);
        assert_eq!(c.document_budget_usage(UNITS), Some((10, 10)));

        let mut loaded = EditorController::new();
        bootstrap(&mut loaded, 1, 6, dynamic);
        bootstrap(&mut loaded, 2, 5, dynamic);
        let before = snapshot(&loaded);
        assert_exceeded(loaded.register_document_budgets([budget(10)]), UNITS, 10);
        assert_eq!(snapshot(&loaded), before);
        assert_eq!(loaded.document_budget_usage(UNITS), None);
    }
}

#[test]
fn balanced_multi_root_final_candidate_is_admitted_and_rejection_preserves_redo() {
    for dynamic in [false, true] {
        let mut c = EditorController::new();
        c.register_document_budgets([budget(10)]).unwrap();
        bootstrap(&mut c, 1, 6, dynamic);
        bootstrap(&mut c, 2, 4, dynamic);
        c.select_items(&[item(2), item(1)], SelectionMode::Replace)
            .unwrap();
        // Grow first: evaluating an intermediate aggregate would exceed 10.
        apply(&mut c, vec![edit(1, "units", 9), edit(2, "units", 1)]);
        assert_eq!(field(c.object_root(&item(1)).unwrap(), "units"), Ok(9));
        assert_eq!(field(c.object_root(&item(2)).unwrap(), "units"), Ok(1));
        assert_eq!(c.document_budget_usage(UNITS), Some((10, 10)));
        assert_eq!(c.document_history_lengths(), (1, 0));
        c.execute(EditorAction::Undo).unwrap();
        c.mark_saved(c.model_revision()).unwrap();
        let before = snapshot(&c);
        assert_exceeded(
            c.prepare_edits(vec![edit(1, "units", 8), edit(2, "units", 3)]),
            UNITS,
            10,
        );
        assert_eq!(snapshot(&c), before);
        c.execute(EditorAction::Redo).unwrap();
        assert_eq!(field(c.object_root(&item(1)).unwrap(), "units"), Ok(9));
        assert_eq!(field(c.object_root(&item(2)).unwrap(), "units"), Ok(1));
        assert_eq!(c.document_budget_usage(UNITS), Some((10, 10)));
    }
}

#[test]
fn scalar_callbacks_are_sparse_once_per_changed_root_and_history_uses_cached_costs() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let recorded = calls.clone();
    let counted = ControllerDocumentBudget::new(UNITS, 100, move |item, root| {
        recorded.lock().unwrap().push(item.clone());
        contribution(item, root)
    });
    let mut c = EditorController::new();
    for id in 1..=16 {
        bootstrap(&mut c, id, 1, id % 2 == 0);
    }
    c.register_document_budgets([counted]).unwrap();
    assert_eq!(calls.lock().unwrap().len(), 16);
    calls.lock().unwrap().clear();
    let prepared = c
        .prepare_edits(vec![
            edit(2, "units", 2),
            edit(2, "extra", 1),
            edit(7, "units", 3),
            edit(8, "units", 1), // no-op root must not be re-costed
        ])
        .unwrap();
    let mut touched = calls.lock().unwrap().clone();
    touched.sort();
    assert_eq!(touched, vec![item(2), item(7)]);
    assert_eq!(c.document_budget_usage(UNITS), Some((16, 100)));
    c.commit_edits(prepared).unwrap();
    assert_eq!(calls.lock().unwrap().len(), 2);
    assert_eq!(c.document_budget_usage(UNITS), Some((20, 100)));
    for _ in 0..2 {
        c.execute(EditorAction::Undo).unwrap();
        assert_eq!(c.document_budget_usage(UNITS), Some((16, 100)));
        c.execute(EditorAction::Redo).unwrap();
        assert_eq!(c.document_budget_usage(UNITS), Some((20, 100)));
    }
    assert_eq!(calls.lock().unwrap().len(), 2);
    apply(&mut c, vec![edit(2, "units", 2)]);
    assert_eq!(calls.lock().unwrap().len(), 2);
    assert_eq!(c.document_history_lengths(), (1, 0));
}

#[test]
fn structural_capacity_tracks_create_delete_and_both_history_directions() {
    for dynamic in [false, true] {
        let mut c = EditorController::new();
        c.register_document_budgets([budget(10)]).unwrap();
        bootstrap(&mut c, 1, 6, dynamic);
        c.select(item(1)).unwrap();
        let before = snapshot(&c);
        assert_exceeded(
            c.create_objects(vec![object(2, 2, dynamic), object(3, 3, dynamic)]),
            UNITS,
            10,
        );
        assert_eq!(snapshot(&c), before);
        c.create_objects(vec![object(2, 4, dynamic)]).unwrap();
        assert_eq!(c.document_budget_usage(UNITS), Some((10, 10)));
        c.execute(EditorAction::Undo).unwrap();
        assert_eq!(c.document_budget_usage(UNITS), Some((6, 10)));
        c.execute(EditorAction::Redo).unwrap();
        assert_eq!(c.document_budget_usage(UNITS), Some((10, 10)));
        c.delete_objects(&[item(1)]).unwrap();
        assert_eq!(c.document_budget_usage(UNITS), Some((4, 10)));
        for _ in 0..2 {
            // Structural restores may re-cost the restored roots.
            c.execute(EditorAction::Undo).unwrap();
            assert_eq!(c.document_budget_usage(UNITS), Some((10, 10)));
            assert!(c.object_root(&item(1)).is_ok());
            c.execute(EditorAction::Redo).unwrap();
            assert_eq!(c.document_budget_usage(UNITS), Some((4, 10)));
            assert!(c.object_root(&item(1)).is_err());
        }
        let before = snapshot(&c);
        assert_exceeded(c.create_objects(vec![object(3, 7, dynamic)]), UNITS, 10);
        assert_eq!(snapshot(&c), before);
        c.create_objects(vec![object(3, 6, dynamic)]).unwrap();
        assert_eq!(c.document_budget_usage(UNITS), Some((10, 10)));
        c.delete_objects(&[item(2), item(3)]).unwrap();
        assert_eq!(c.document_budget_usage(UNITS), Some((0, 10)));
        c.execute(EditorAction::Undo).unwrap();
        assert_eq!(c.document_budget_usage(UNITS), Some((10, 10)));
        c.execute(EditorAction::Redo).unwrap();
        assert_eq!(c.document_budget_usage(UNITS), Some((0, 10)));
    }
}

#[test]
fn aggregate_u64_overflow_is_rejected_without_wrapping_or_publication() {
    let enormous = ControllerDocumentBudget::new(UNITS, u64::MAX, |_, root| {
        Ok(if field(root, "units")? == 1 {
            u64::MAX
        } else {
            1
        })
    });
    let mut c = EditorController::new();
    c.register_document_budgets([enormous.clone()]).unwrap();
    bootstrap(&mut c, 1, 1, true);
    assert_eq!(c.document_budget_usage(UNITS), Some((u64::MAX, u64::MAX)));
    let before = snapshot(&c);
    let second = object(2, 2, false);
    assert_exceeded(
        c.register_object(second.item, second.root, second.registry),
        UNITS,
        u64::MAX,
    );
    assert_eq!(snapshot(&c), before);
    assert_exceeded(c.create_objects(vec![object(2, 2, true)]), UNITS, u64::MAX);
    assert_eq!(snapshot(&c), before);

    let mut loaded = EditorController::new();
    bootstrap(&mut loaded, 1, 1, false);
    bootstrap(&mut loaded, 2, 2, true);
    let before = snapshot(&loaded);
    assert_exceeded(
        loaded.register_document_budgets([enormous]),
        UNITS,
        u64::MAX,
    );
    assert_eq!(snapshot(&loaded), before);

    let mut edited = EditorController::new();
    edited
        .register_document_budgets([ControllerDocumentBudget::new(UNITS, u64::MAX, |_, root| {
            Ok(if field(root, "units")? == 1 {
                u64::MAX
            } else {
                1
            })
        })])
        .unwrap();
    bootstrap(&mut edited, 1, 2, false);
    bootstrap(&mut edited, 2, 2, true);
    let before = snapshot(&edited);
    assert_exceeded(
        edited.prepare_edits(vec![edit(1, "units", 1)]),
        UNITS,
        u64::MAX,
    );
    assert_eq!(snapshot(&edited), before);
}

#[test]
fn contribution_errors_identify_budget_item_and_reason_without_partial_edits() {
    let mut c = EditorController::new();
    c.register_document_budgets([budget(10)]).unwrap();
    bootstrap(&mut c, 1, 2, false);
    bootstrap(&mut c, 2, 2, true);
    c.select(item(2)).unwrap();
    let before = snapshot(&c);
    assert!(matches!(
        c.prepare_edits(vec![edit(1, "units", 3), edit(2, "units", -1)]),
        Err(ControllerError::DocumentBudgetContribution { budget, item: failed, reason })
            if budget == UNITS && failed == item(2) && reason == "negative cost"
    ));
    assert_eq!(snapshot(&c), before);
    let rejected = object(3, -1, true);
    assert!(matches!(
        c.register_object(rejected.item, rejected.root, rejected.registry),
        Err(ControllerError::DocumentBudgetContribution { budget, item: failed, reason })
            if budget == UNITS && failed == item(3) && reason == "negative cost"
    ));
    assert_eq!(snapshot(&c), before);
}

#[test]
fn profile_command_capacity_failure_does_not_install_any_budget_or_command() {
    let mut c = EditorController::with_limits(ControllerLimits {
        max_commands: 1,
        ..ControllerLimits::default()
    });
    c.register_command("existing", |_| Ok(EditorAction::Undo))
        .unwrap();
    let before = snapshot(&c);
    assert!(c
        .register_authoring_profile([("new".into(), empty_command)], [budget(10)])
        .is_err());
    assert_eq!(snapshot(&c), before);
    assert!(!c.has_command("new"));
    assert!(c.has_command("existing"));
    assert_eq!(c.document_budget_usage(UNITS), None);
}

#[test]
fn profile_budget_failure_does_not_install_commands_or_sibling_budgets() {
    let mut c = EditorController::new();
    bootstrap(&mut c, 1, 11, true);
    let before = snapshot(&c);
    assert_exceeded(
        c.register_authoring_profile(
            [("new".into(), empty_command)],
            [
                ControllerDocumentBudget::new(SECOND, 100, contribution),
                budget(10),
            ],
        ),
        UNITS,
        10,
    );
    assert_eq!(snapshot(&c), before);
    assert!(!c.has_command("new"));
    c.register_authoring_profile([("new".into(), empty_command)], [budget(12)])
        .unwrap();
    assert!(c.has_command("new"));
    assert_eq!(c.document_budget_usage(UNITS), Some((11, 12)));
    assert!(
        !c.execute(EditorAction::ExecuteCommand {
            command_id: "new".into()
        })
        .unwrap()
        .changed
    );
    assert_eq!(c.document_history_lengths(), (0, 0));
}

#[test]
fn document_budget_count_limit_is_atomic_and_defaults_to_sixteen() {
    assert_eq!(ControllerLimits::default().max_document_budgets, 16);
    let mut c = EditorController::with_limits(ControllerLimits {
        max_document_budgets: 1,
        ..ControllerLimits::default()
    });
    let before = snapshot(&c);
    assert!(c
        .register_document_budgets([
            budget(10),
            ControllerDocumentBudget::new(SECOND, 10, contribution)
        ])
        .is_err());
    assert_eq!(snapshot(&c), before);
    c.register_document_budgets([budget(10)]).unwrap();
    let before = snapshot(&c);
    assert!(c
        .register_authoring_profile(
            [("new".into(), empty_command)],
            [ControllerDocumentBudget::new(SECOND, 10, contribution)],
        )
        .is_err());
    assert_eq!(snapshot(&c), before);
    assert!(!c.has_command("new"));
}

#[test]
fn identical_arc_and_limit_are_idempotent_but_same_id_new_callback_conflicts() {
    let mut c = EditorController::new();
    let existing = budget(10);
    c.register_document_budgets([existing.clone(), existing.clone()])
        .unwrap();
    let before = snapshot(&c);
    c.register_document_budgets([existing.clone()]).unwrap();
    assert_eq!(snapshot(&c), before);
    for conflict in [budget(10), budget(11)] {
        assert!(matches!(
            c.register_document_budgets([conflict]),
            Err(ControllerError::DuplicateDocumentBudget(UNITS))
        ));
        assert_eq!(snapshot(&c), before);
    }
    assert!(matches!(
        c.register_document_budgets([
            ControllerDocumentBudget::new(SECOND, 100, contribution),
            budget(10),
        ]),
        Err(ControllerError::DuplicateDocumentBudget(UNITS))
    ));
    assert_eq!(snapshot(&c), before);
}

#[test]
fn existing_budget_can_be_reinstalled_with_history_but_new_budget_is_edit_conflict() {
    let mut c = EditorController::new();
    let existing = budget(10);
    c.register_document_budgets([existing.clone()]).unwrap();
    bootstrap(&mut c, 1, 1, true);
    apply(&mut c, vec![edit(1, "units", 2)]);
    for redo_only in [false, true] {
        if redo_only {
            c.execute(EditorAction::Undo).unwrap();
            assert_eq!(c.document_history_lengths(), (0, 1));
        } else {
            assert_eq!(c.document_history_lengths(), (1, 0));
        }
        let before = snapshot(&c);
        c.register_document_budgets([existing.clone()]).unwrap();
        assert_eq!(snapshot(&c), before);
        assert!(matches!(
            c.register_document_budgets([
                existing.clone(),
                ControllerDocumentBudget::new(SECOND, 100, contribution),
            ]),
            Err(ControllerError::EditConflict)
        ));
        assert_eq!(snapshot(&c), before);
        assert!(matches!(
            c.register_authoring_profile(
                [("new".into(), empty_command)],
                [ControllerDocumentBudget::new(SECOND, 100, contribution)],
            ),
            Err(ControllerError::EditConflict)
        ));
        assert_eq!(snapshot(&c), before);
        assert!(!c.has_command("new"));
    }
}

#[test]
fn installing_new_budget_invalidates_prepared_edits_even_when_candidate_fits() {
    let mut c = EditorController::new();
    bootstrap(&mut c, 1, 1, true);
    let prepared = c.prepare_edits(vec![edit(1, "units", 2)]).unwrap();
    c.register_document_budgets([budget(10)]).unwrap();
    let before = snapshot(&c);
    assert!(matches!(
        c.commit_edits(prepared),
        Err(ControllerError::EditConflict)
    ));
    assert_eq!(snapshot(&c), before);
    apply(&mut c, vec![edit(1, "units", 2)]);
    assert_eq!(c.document_budget_usage(UNITS), Some((2, 10)));
}

#[test]
fn prepared_edits_cannot_cross_independent_controllers_with_matching_budget_and_revisions() {
    let shared = budget(10);
    let mut source = EditorController::new();
    let mut target = EditorController::new();
    for c in [&mut source, &mut target] {
        c.register_document_budgets([shared.clone()]).unwrap();
        bootstrap(c, 1, 1, true);
    }
    assert_eq!(snapshot(&source), snapshot(&target));
    let prepared = source.prepare_edits(vec![edit(1, "units", 2)]).unwrap();
    let before_source = snapshot(&source);
    let before_target = snapshot(&target);
    assert!(matches!(
        target.commit_edits(prepared),
        Err(ControllerError::EditConflict)
    ));
    assert_eq!(snapshot(&source), before_source);
    assert_eq!(snapshot(&target), before_target);
    apply(&mut source, vec![edit(1, "units", 2)]);
    assert_eq!(source.document_budget_usage(UNITS), Some((2, 10)));
    assert_eq!(target.document_budget_usage(UNITS), Some((1, 10)));
}

#[test]
fn scalar_history_preserves_bootstrap_contributions_added_on_either_history_branch() {
    for dynamic in [false, true] {
        let mut c = EditorController::new();
        c.register_document_budgets([budget(12)]).unwrap();
        bootstrap(&mut c, 1, 2, dynamic);
        apply(&mut c, vec![edit(1, "units", 5)]);
        bootstrap(&mut c, 2, 4, !dynamic);
        assert_eq!(c.document_budget_usage(UNITS), Some((9, 12)));
        assert_eq!(c.document_history_lengths(), (1, 0));

        c.execute(EditorAction::Undo).unwrap();
        assert_eq!(c.document_budget_usage(UNITS), Some((6, 12)));
        assert_eq!(field(c.object_root(&item(1)).unwrap(), "units"), Ok(2));
        assert_eq!(field(c.object_root(&item(2)).unwrap(), "units"), Ok(4));
        // Bootstrap also preserves the redo branch. Its new cost must remain
        // present when the old scalar patch is replayed in either direction.
        bootstrap(&mut c, 3, 3, dynamic);
        assert_eq!(c.document_budget_usage(UNITS), Some((9, 12)));
        assert_eq!(c.document_history_lengths(), (0, 1));
        let shape = c.shape_revision();
        for _ in 0..2 {
            c.execute(EditorAction::Redo).unwrap();
            assert_eq!(c.document_budget_usage(UNITS), Some((12, 12)));
            assert_eq!(field(c.object_root(&item(1)).unwrap(), "units"), Ok(5));
            assert_eq!(c.document_history_lengths(), (1, 0));
            c.execute(EditorAction::Undo).unwrap();
            assert_eq!(c.document_budget_usage(UNITS), Some((9, 12)));
            assert_eq!(field(c.object_root(&item(1)).unwrap(), "units"), Ok(2));
            assert_eq!(c.document_history_lengths(), (0, 1));
            assert_eq!(field(c.object_root(&item(2)).unwrap(), "units"), Ok(4));
            assert_eq!(field(c.object_root(&item(3)).unwrap(), "units"), Ok(3));
            assert_eq!(c.document_objects().len(), 3);
            assert_eq!(c.shape_revision(), shape);
        }
    }
}

#[test]
fn independent_budgets_exchange_together_and_either_rejection_preserves_all_state() {
    for dynamic in [false, true] {
        let mut c = EditorController::new();
        c.register_document_budgets([
            ControllerDocumentBudget::new(UNITS, 10, |_, root| {
                u64::try_from(field(root, "units")?).map_err(|_| "negative units".into())
            }),
            ControllerDocumentBudget::new(SECOND, 6, |_, root| {
                u64::try_from(field(root, "extra")?).map_err(|_| "negative extra".into())
            }),
        ])
        .unwrap();
        bootstrap(&mut c, 1, 3, dynamic);
        bootstrap(&mut c, 2, 2, !dynamic);
        c.select_items(&[item(2), item(1)], SelectionMode::Replace)
            .unwrap();
        let before = snapshot(&c);
        let prepared = c
            .prepare_edits(vec![
                edit(1, "units", 6),
                edit(1, "extra", 4),
                edit(2, "units", 2),
                edit(2, "extra", 2),
            ])
            .unwrap();
        assert_eq!(snapshot(&c), before);
        assert_eq!(c.document_budget_usage(UNITS), Some((5, 10)));
        assert_eq!(c.document_budget_usage(SECOND), Some((0, 6)));
        c.commit_edits(prepared).unwrap();
        let committed = snapshot(&c);
        assert_eq!(c.document_budget_usage(UNITS), Some((8, 10)));
        assert_eq!(c.document_budget_usage(SECOND), Some((6, 6)));
        assert_eq!(field(c.object_root(&item(1)).unwrap(), "units"), Ok(6));
        assert_eq!(field(c.object_root(&item(1)).unwrap(), "extra"), Ok(4));
        assert_eq!(field(c.object_root(&item(2)).unwrap(), "extra"), Ok(2));

        for _ in 0..2 {
            c.execute(EditorAction::Undo).unwrap();
            assert_eq!(snapshot(&c).objects, before.objects);
            assert_eq!(c.document_budget_usage(UNITS), Some((5, 10)));
            assert_eq!(c.document_budget_usage(SECOND), Some((0, 6)));
            assert_eq!(c.document_history_lengths(), (0, 1));
            c.mark_saved(c.model_revision()).unwrap();
            let unchanged = snapshot(&c);
            // Each batch is legal under one policy and exceeds the other.
            // In particular, SECOND rejects after UNITS has accepted its cost.
            for (id, limit, edits) in [
                (UNITS, 10, vec![edit(1, "units", 9), edit(2, "extra", 1)]),
                (SECOND, 6, vec![edit(1, "units", 4), edit(2, "extra", 7)]),
            ] {
                assert_exceeded(c.prepare_edits(edits), id, limit);
                assert_eq!(snapshot(&c), unchanged);
            }
            c.execute(EditorAction::Redo).unwrap();
            assert_eq!(snapshot(&c).objects, committed.objects);
            assert_eq!(c.document_budget_usage(UNITS), Some((8, 10)));
            assert_eq!(c.document_budget_usage(SECOND), Some((6, 6)));
            assert_eq!(c.document_history_lengths(), (1, 0));
        }
    }
}

#[test]
fn empty_prepared_edits_are_stale_after_installing_first_or_additional_policy() {
    for dynamic in [false, true] {
        for existing_policy in [false, true] {
            for no_op_field in [false, true] {
                let mut c = EditorController::new();
                bootstrap(&mut c, 1, 1, dynamic);
                if existing_policy {
                    c.register_document_budgets([budget(10)]).unwrap();
                }
                let before = snapshot(&c);
                let edits = if no_op_field {
                    vec![edit(1, "units", 1)]
                } else {
                    Vec::new()
                };
                let prepared = c.prepare_edits(edits).unwrap();
                assert_eq!(snapshot(&c), before);
                let policy = if existing_policy {
                    ControllerDocumentBudget::new(SECOND, 10, contribution)
                } else {
                    budget(10)
                };
                c.register_document_budgets([policy]).unwrap();
                let after_install = snapshot(&c);
                assert!(matches!(
                    c.commit_edits(prepared),
                    Err(ControllerError::EditConflict)
                ));
                assert_eq!(snapshot(&c), after_install);
                let fresh = c.prepare_edits(Vec::new()).unwrap();
                assert!(!c.commit_edits(fresh).unwrap().changed);
                assert_eq!(snapshot(&c), after_install);
            }
        }
    }
}
