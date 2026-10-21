//! Structural authoring is part of the same atomic document history.

use prism_editor_app::{
    ControllerError, ControllerFieldEdit, ControllerLimits, ControllerObject,
    ControllerObjectValidator, EditorController,
};
use prism_editor_core::{Selection, SelectionItem};
use prism_editor_panels::{OutlinerId, OutlinerNode};
use prism_editor_reflect::FieldPath;
use prism_editor_ui::{ActionValue, EditorAction, SelectionMode};
use prism_reflect::{Reflect, TypeRegistry};

#[derive(Reflect)]
struct Model {
    value: i32,
    payload: Vec<u8>,
}

fn item(id: u128) -> SelectionItem {
    SelectionItem::entity(id)
}
fn row(id: u128, parent: Option<u128>, selected: Option<u128>) -> OutlinerNode {
    let mut row = OutlinerNode::new(
        OutlinerId::from_raw(id),
        parent.map(OutlinerId::from_raw),
        format!("row {id}"),
    );
    if let Some(selected) = selected {
        row.set_selection(item(selected));
    }
    row
}
fn object(id: u128, parent: Option<u128>) -> ControllerObject {
    let mut registry = TypeRegistry::new();
    registry.register::<Model>();
    ControllerObject {
        item: item(id),
        root: Box::new(Model {
            value: 1,
            payload: vec![5; 64],
        }),
        registry,
        row: row(id, parent, Some(id)),
    }
}
fn ids(c: &EditorController) -> Vec<u128> {
    c.document_outliner_nodes()
        .iter()
        .map(|r| r.id().raw())
        .collect()
}
fn value(c: &EditorController, id: u128) -> i32 {
    c.object_root(&item(id))
        .unwrap()
        .downcast_ref::<Model>()
        .unwrap()
        .value
}
fn state(c: &EditorController) -> (u64, u64, u64, (usize, usize), Vec<u128>) {
    (
        c.revision(),
        c.model_revision(),
        c.shape_revision(),
        c.document_history_lengths(),
        ids(c),
    )
}

// Failed operations must preserve data, row flags/order, selection, allocation
// proposals and dirty state, not just history lengths and revision counters.
#[derive(Debug, PartialEq)]
struct Snapshot {
    state: (u64, u64, u64, (usize, usize), Vec<u128>),
    objects: Vec<(SelectionItem, i32, Vec<u8>, usize)>,
    rows: Vec<OutlinerNode>,
    selection: Selection,
    next_identity: Option<(SelectionItem, OutlinerId)>,
    dirty: bool,
}

fn snapshot(c: &EditorController) -> Snapshot {
    Snapshot {
        state: state(c),
        objects: c
            .document_objects()
            .map(|(item, root, _)| {
                let model = root.downcast_ref::<Model>().unwrap();
                (
                    item.clone(),
                    model.value,
                    model.payload.clone(),
                    model.payload.as_ptr() as usize,
                )
            })
            .collect(),
        rows: c.document_outliner_nodes().to_vec(),
        selection: c.selection().clone(),
        next_identity: c.next_entity_identity().ok(),
        dirty: c.is_dirty(),
    }
}

fn edit(id: u128, path: &str, value: i64) -> ControllerFieldEdit {
    ControllerFieldEdit {
        item: item(id),
        path: FieldPath::parse(path).unwrap(),
        value: ActionValue::integer(value),
    }
}

fn apply_edits(c: &mut EditorController, edits: Vec<ControllerFieldEdit>) {
    let prepared = c.prepare_edits(edits).unwrap();
    c.commit_edits(prepared).unwrap();
}

fn select_row(c: &mut EditorController, id: u128, mode: SelectionMode) {
    c.execute(EditorAction::SelectOutliner {
        node: OutlinerId::from_raw(id),
        mode,
    })
    .unwrap();
}

#[test]
fn delete_preserves_order_and_nonlast_primary_across_undo_redo() {
    let mut c = EditorController::new();
    c.create_objects((1..=256).map(|id| object(id, None)).collect())
        .unwrap();
    // Making an existing selected item primary must not reorder selection.
    c.select_items(&[item(2)], SelectionMode::Add).unwrap();
    assert_eq!(c.selection().primary(), Some(&item(2)));
    let before = c.selection().clone();
    let removed: Vec<_> = (3..=255).map(item).collect();
    c.delete_objects(&removed).unwrap();
    assert_eq!(c.selection().items(), &[item(1), item(2), item(256)]);
    assert_eq!(c.selection().primary(), Some(&item(2)));
    let after = c.selection().clone();
    c.execute(EditorAction::Undo).unwrap();
    assert_eq!(c.selection(), &before);
    c.execute(EditorAction::Redo).unwrap();
    assert_eq!(c.selection(), &after);
    // If the explicit primary is deleted, promote the last surviving item.
    c.delete_objects(&[item(2)]).unwrap();
    assert_eq!(c.selection().items(), &[item(1), item(256)]);
    assert_eq!(c.selection().primary(), Some(&item(256)));
    c.execute(EditorAction::Undo).unwrap();
    assert_eq!(c.selection(), &after);
}

#[test]
fn range_anchor_survives_unrelated_deletion_and_structural_undo_redo() {
    // Independent fixtures probe both the initial deletion and its replay;
    // range selection itself must not add a document history entry.
    for replay in [false, true] {
        let mut c = EditorController::new();
        c.create_objects((1..=5).map(|id| object(id, None)).collect())
            .unwrap();
        select_row(&mut c, 2, SelectionMode::Replace);
        select_row(&mut c, 4, SelectionMode::Extend);
        assert_eq!(c.selection().items(), &[item(2), item(3), item(4)]);
        assert_eq!(c.selection().primary(), Some(&item(4)));

        c.delete_objects(&[item(3)]).unwrap();
        assert_eq!(ids(&c), vec![1, 2, 4, 5]);
        assert_eq!(c.selection().items(), &[item(2), item(4)]);
        assert_eq!(c.selection().primary(), Some(&item(4)));
        if replay {
            c.execute(EditorAction::Undo).unwrap();
            assert_eq!(ids(&c), vec![1, 2, 3, 4, 5]);
            assert_eq!(c.selection().items(), &[item(2), item(3), item(4)]);
            assert_eq!(c.selection().primary(), Some(&item(4)));
            // Restored anchor is row 2, not the restored primary row 4.
            select_row(&mut c, 1, SelectionMode::Extend);
            assert_eq!(c.selection().items(), &[item(1), item(2)]);
            assert_eq!(c.selection().primary(), Some(&item(1)));
            // Restore the pre-delete selection without moving its anchor.
            select_row(&mut c, 4, SelectionMode::Extend);
            c.execute(EditorAction::Redo).unwrap();
            assert_eq!(ids(&c), vec![1, 2, 4, 5]);
            assert_eq!(c.selection().items(), &[item(2), item(4)]);
            assert_eq!(c.selection().primary(), Some(&item(4)));
        }

        let model_revision = c.model_revision();
        let history = c.document_history_lengths();
        select_row(&mut c, 5, SelectionMode::Extend);
        assert_eq!(c.selection().items(), &[item(2), item(4), item(5)]);
        assert_eq!(c.selection().primary(), Some(&item(5)));
        assert_eq!(c.model_revision(), model_revision);
        assert_eq!(c.document_history_lengths(), history);
    }
}

#[test]
fn deleted_alias_range_anchor_is_cleared_restored_by_undo_and_cleared_by_redo() {
    for replay in [false, true] {
        let mut c = EditorController::new();
        c.create_objects((1..=5).map(|id| object(id, None)).collect())
            .unwrap();
        c.register_outliner_node(row(20, None, Some(2))).unwrap();
        // Anchor the alias, not object 2's first row. The range ends at row 4,
        // so primary alone cannot recover which Outliner row was the anchor.
        select_row(&mut c, 20, SelectionMode::Replace);
        select_row(&mut c, 4, SelectionMode::Extend);
        assert_eq!(c.selection().items(), &[item(4), item(5), item(2)]);
        assert_eq!(c.selection().primary(), Some(&item(4)));

        c.delete_objects(&[item(2)]).unwrap();
        assert_eq!(ids(&c), vec![1, 3, 4, 5]);
        assert_eq!(c.selection().items(), &[item(4), item(5)]);
        assert_eq!(c.selection().primary(), Some(&item(4)));
        if replay {
            c.execute(EditorAction::Undo).unwrap();
            assert_eq!(ids(&c), vec![1, 2, 3, 4, 5, 20]);
            assert_eq!(c.selection().items(), &[item(4), item(5), item(2)]);
            assert_eq!(c.selection().primary(), Some(&item(4)));
            select_row(&mut c, 3, SelectionMode::Extend);
            assert_eq!(c.selection().items(), &[item(3), item(4), item(5), item(2)]);
            assert_eq!(c.selection().primary(), Some(&item(3)));
            // Keep alias row 20 as anchor while restoring the old selection.
            select_row(&mut c, 4, SelectionMode::Extend);
            c.execute(EditorAction::Redo).unwrap();
            assert_eq!(ids(&c), vec![1, 3, 4, 5]);
            assert_eq!(c.selection().items(), &[item(4), item(5)]);
            assert_eq!(c.selection().primary(), Some(&item(4)));
        }

        let model_revision = c.model_revision();
        let history = c.document_history_lengths();
        // With no anchor, Extend starts at its target. A stale alias would
        // error; falling back to primary row 4 would incorrectly include 3/4.
        select_row(&mut c, 1, SelectionMode::Extend);
        assert_eq!(c.selection().items(), &[item(1)]);
        assert_eq!(c.selection().primary(), Some(&item(1)));
        assert_eq!(c.model_revision(), model_revision);
        assert_eq!(c.document_history_lengths(), history);
    }
}

#[test]
fn structural_changes_require_full_resync_but_later_scalar_deltas_remain_incremental() {
    let mut c = EditorController::new();
    let initial = c.model_revision();
    c.create_objects(vec![object(1, None)]).unwrap();
    let created = c.model_revision();
    assert_eq!(c.changed_objects_since(initial), None);
    assert_eq!(c.changed_objects_since(created), Some(vec![]));

    apply_edits(&mut c, vec![edit(1, "value", 42)]);
    assert_eq!(c.changed_objects_since(initial), None);
    assert_eq!(c.changed_objects_since(created), Some(vec![item(1)]));
    let edited = c.model_revision();
    c.delete_objects(&[item(1)]).unwrap();
    assert_eq!(c.changed_objects_since(edited), None);

    let deleted = c.model_revision();
    c.execute(EditorAction::Undo).unwrap();
    assert_eq!(c.changed_objects_since(deleted), None);
    assert_eq!(value(&c, 1), 42);
    let restored = c.model_revision();
    c.execute(EditorAction::Redo).unwrap();
    assert_eq!(c.changed_objects_since(restored), None);
    assert_eq!(c.changed_objects_since(c.model_revision()), Some(vec![]));
}

#[test]
fn create_delete_move_original_roots_and_restore_selection_in_shared_history() {
    let mut c = EditorController::new();
    c.create_objects(vec![object(1, None), object(2, Some(1))])
        .unwrap();
    assert!(c.is_dirty());
    assert_eq!(c.document_history_lengths(), (1, 0));
    assert_eq!(c.selection().items(), &[item(1), item(2)]);
    let allocation = c
        .object_root(&item(2))
        .unwrap()
        .downcast_ref::<Model>()
        .unwrap()
        .payload
        .as_ptr();
    c.execute(EditorAction::EditInspectorField {
        path: FieldPath::parse("value").unwrap(),
        value: ActionValue::integer(9),
    })
    .unwrap();
    c.delete_objects(&[item(1)]).unwrap();
    assert_eq!(c.document_objects().len(), 0);
    assert!(c.selection().items().is_empty());
    assert_eq!(c.history_lengths(&item(2)).unwrap(), (3, 0));
    c.execute(EditorAction::Undo).unwrap();
    assert_eq!(ids(&c), vec![1, 2]);
    assert_eq!(value(&c, 2), 9);
    assert_eq!(c.selection().primary(), Some(&item(2)));
    // Scalar transactions exchange leaves, structural history moves roots.
    assert_eq!(
        c.object_root(&item(2))
            .unwrap()
            .downcast_ref::<Model>()
            .unwrap()
            .payload
            .as_ptr(),
        allocation
    );
    c.execute(EditorAction::Undo).unwrap();
    assert_eq!(value(&c, 2), 1);
    c.execute(EditorAction::Undo).unwrap();
    assert_eq!(c.document_objects().len(), 0);
    for _ in 0..3 {
        c.execute(EditorAction::Redo).unwrap();
    }
    assert_eq!(c.document_objects().len(), 0);
    assert!(c.is_dirty());
}

#[test]
fn deletion_closes_over_aliases_descendants_groups_and_restores_exact_rows() {
    let mut c = EditorController::new();
    c.create_objects(vec![
        object(1, None),
        object(2, Some(1)),
        object(3, None),
        object(4, None),
    ])
    .unwrap();
    c.register_outliner_node(row(10, Some(3), Some(2))).unwrap();
    c.register_outliner_node(row(11, Some(10), None)).unwrap();
    c.register_outliner_node(row(12, Some(11), Some(4)))
        .unwrap();
    c.register_outliner_node(row(13, None, Some(4))).unwrap();
    let rows = c.document_outliner_nodes().to_vec();
    c.select_items(&[item(1), item(3)], SelectionMode::Replace)
        .unwrap();
    let selection = c.selection().clone();
    c.delete_objects(&[item(1)]).unwrap();
    assert_eq!(ids(&c), vec![3]);
    assert_eq!(c.selection().items(), &[item(3)]);
    c.execute(EditorAction::Undo).unwrap();
    assert_eq!(c.document_outliner_nodes(), rows);
    assert_eq!(c.selection(), &selection);
    c.execute(EditorAction::Redo).unwrap();
    assert_eq!(ids(&c), vec![3]);
}

#[test]
fn invalid_batches_and_restore_collisions_preserve_both_history_branches() {
    let mut c = EditorController::new();
    c.create_objects(vec![object(1, None)]).unwrap();
    c.execute(EditorAction::Undo).unwrap();
    let before = state(&c);
    assert!(c
        .create_objects(vec![object(2, None), object(2, None)])
        .is_err());
    assert!(c.create_objects(vec![object(2, Some(999))]).is_err());
    let mut wrong = object(2, None);
    wrong.row = row(2, None, Some(3));
    assert!(c.create_objects(vec![wrong]).is_err());
    assert_eq!(state(&c), before);
    c.execute(EditorAction::Redo).unwrap();
    let before = state(&c);
    assert!(c.delete_objects(&[item(1), item(99)]).is_err());
    assert!(c.delete_objects(&[item(1), item(1)]).is_err());
    assert_eq!(state(&c), before);
    c.delete_objects(&[item(1)]).unwrap();
    let replacement = object(1, None);
    c.register_object(replacement.item, replacement.root, replacement.registry)
        .unwrap();
    let before = state(&c);
    assert!(matches!(
        c.execute(EditorAction::Undo),
        Err(ControllerError::DuplicateObject(_))
    ));
    assert_eq!(state(&c), before);
    assert_eq!(value(&c, 1), 1);
}

#[test]
fn identity_high_water_survives_undo_and_discarded_redo_and_checks_overflow() {
    let mut c = EditorController::new();
    c.create_objects(vec![object(12, None)]).unwrap();
    c.execute(EditorAction::Undo).unwrap();
    assert_eq!(c.next_entity_identity().unwrap().0, item(13));
    c.create_objects(vec![object(13, None)]).unwrap();
    assert_eq!(c.document_history_lengths(), (1, 0));
    c.delete_objects(&[item(13)]).unwrap();
    assert_eq!(c.next_entity_identity().unwrap().0, item(14));
    c.register_outliner_node(row(u128::MAX, None, None))
        .unwrap();
    assert!(matches!(
        c.next_entity_identity(),
        Err(ControllerError::RevisionExhausted)
    ));
}

#[test]
fn structural_budgets_and_stale_prepared_edits_reject_before_publication() {
    for limits in [
        ControllerLimits {
            max_objects: 0,
            ..ControllerLimits::default()
        },
        ControllerLimits {
            max_outliner_nodes: 0,
            ..ControllerLimits::default()
        },
        ControllerLimits {
            max_selection: 0,
            ..ControllerLimits::default()
        },
        ControllerLimits {
            max_history: 0,
            ..ControllerLimits::default()
        },
        ControllerLimits {
            max_history_per_object: 0,
            ..ControllerLimits::default()
        },
    ] {
        let mut c = EditorController::with_limits(limits);
        let before = state(&c);
        assert!(matches!(
            c.create_objects(vec![object(1, None)]),
            Err(ControllerError::BudgetExceeded { .. })
        ));
        assert_eq!(state(&c), before);
        assert!(!c.is_dirty());
        assert!(!c.create_objects(vec![]).unwrap().changed);
        assert!(!c.delete_objects(&[]).unwrap().changed);
    }
    let mut c = EditorController::with_limits(ControllerLimits {
        max_outliner_depth: 0,
        ..ControllerLimits::default()
    });
    assert!(c
        .create_objects(vec![object(1, None), object(2, Some(1))])
        .is_err());
    assert_eq!(c.document_objects().len(), 0);
    c.create_objects(vec![object(1, None)]).unwrap();
    c.select(item(1)).unwrap();
    let prepared = c
        .prepare_edits(vec![ControllerFieldEdit {
            item: item(1),
            path: FieldPath::parse("value").unwrap(),
            value: ActionValue::integer(5),
        }])
        .unwrap();
    c.register_outliner_node(row(99, None, None)).unwrap();
    assert!(matches!(
        c.commit_edits(prepared),
        Err(ControllerError::EditConflict)
    ));
    assert_eq!(value(&c, 1), 1);
}

#[test]
fn new_branch_history_budget_counts_only_retained_undo() {
    let mut c = EditorController::with_limits(ControllerLimits {
        max_history: 1,
        max_history_per_object: 1,
        ..ControllerLimits::default()
    });
    c.create_objects(vec![object(1, None)]).unwrap();
    let before = state(&c);
    assert!(c.delete_objects(&[item(1)]).is_err());
    assert_eq!(state(&c), before);
    c.execute(EditorAction::Undo).unwrap();
    c.create_objects(vec![object(2, None)]).unwrap();
    assert_eq!(c.document_history_lengths(), (1, 0));
}

#[test]
fn rejected_sibling_batches_preserve_data_redo_and_identity_proposals() {
    let mut c = EditorController::new();
    c.create_objects(vec![object(1, None)]).unwrap();
    c.execute(EditorAction::Undo).unwrap();
    c.mark_saved(c.model_revision()).unwrap();
    let before = snapshot(&c);

    let mut invalid = object(u128::MAX, None);
    invalid
        .registry
        .register_type_data::<Model, _>(ControllerObjectValidator::new(|_, _| {
            Err("schema rejects this object".into())
        }));
    let mut duplicate_row = object(3, None);
    duplicate_row.row = row(2, None, Some(3));
    let mut self_parent = object(4, Some(4));
    self_parent.row.set_visible(false);
    for batch in [
        vec![object(2, None), invalid],
        vec![object(2, None), duplicate_row],
        vec![object(2, None), object(3, Some(999))],
        vec![object(3, Some(4)), object(4, None)],
        vec![object(2, None), self_parent],
    ] {
        assert!(c.create_objects(batch).is_err());
        assert_eq!(snapshot(&c), before);
    }
    assert!(!c.create_objects(vec![]).unwrap().changed);
    assert!(!c.delete_objects(&[]).unwrap().changed);
    assert_eq!(snapshot(&c), before);
    c.execute(EditorAction::Redo).unwrap();
    assert_eq!(ids(&c), vec![1]);
    assert_eq!(c.next_entity_identity().unwrap().0, item(2));
}

#[test]
fn hidden_alias_chain_reaches_fixed_point_and_preserves_unrelated_roots() {
    let mut c = EditorController::new();
    let mut parent = object(1, None);
    parent.row.set_expanded(false);
    c.create_objects(vec![
        parent,
        object(2, None),
        object(3, None),
        object(4, None),
        object(5, None),
    ])
    .unwrap();
    let unrepresented = object(6, None);
    c.register_object(
        unrepresented.item,
        unrepresented.root,
        unrepresented.registry,
    )
    .unwrap();
    let mut hidden = row(10, Some(1), Some(2));
    hidden.set_visible(false);
    c.register_outliner_node(hidden).unwrap();
    c.register_outliner_node(row(11, Some(2), None)).unwrap();
    c.register_outliner_node(row(12, Some(11), Some(3)))
        .unwrap();
    c.register_outliner_node(row(13, Some(3), Some(4))).unwrap();
    // Alias graph cycles need not be Outliner parent cycles.
    c.register_outliner_node(row(14, Some(4), Some(1))).unwrap();
    c.select_items(&[item(1), item(5), item(6)], SelectionMode::Replace)
        .unwrap();
    let before = snapshot(&c);
    let tree: Vec<_> = before
        .rows
        .iter()
        .map(|row| c.outliner().node(row.id()).unwrap().clone())
        .collect();
    // Overlapping closure requests are valid (unlike duplicate input IDs).
    c.delete_objects(&[item(1), item(2)]).unwrap();
    for _ in 0..2 {
        assert_eq!(ids(&c), vec![5]);
        assert_eq!(c.document_objects().len(), 2);
        assert_eq!(c.selection().items(), &[item(5), item(6)]);
        c.execute(EditorAction::Undo).unwrap();
        let restored = snapshot(&c);
        assert_eq!(restored.objects, before.objects);
        assert_eq!(restored.rows, before.rows);
        assert_eq!(restored.selection, before.selection);
        for node in &tree {
            assert_eq!(c.outliner().node(node.id()), Some(node));
        }
        c.execute(EditorAction::Redo).unwrap();
    }
}

#[test]
fn replay_preflight_rejects_row_collisions_dependencies_and_object_budget() {
    for alias in [false, true] {
        let mut c = EditorController::new();
        c.create_objects(vec![object(1, None)]).unwrap();
        c.register_outliner_node(if alias {
            row(9, None, Some(1))
        } else {
            row(9, Some(1), None)
        })
        .unwrap();
        let before = snapshot(&c);
        for _ in 0..2 {
            assert!(c.execute(EditorAction::Undo).is_err());
            assert_eq!(snapshot(&c), before);
        }
    }

    let mut c = EditorController::new();
    c.create_objects(vec![object(1, None), object(2, Some(1))])
        .unwrap();
    c.execute(EditorAction::Undo).unwrap();
    c.register_outliner_node(row(2, None, None)).unwrap();
    let before = snapshot(&c);
    for _ in 0..2 {
        assert!(c.execute(EditorAction::Redo).is_err());
        assert_eq!(snapshot(&c), before);
    }

    let mut c = EditorController::with_limits(ControllerLimits {
        max_objects: 2,
        ..ControllerLimits::default()
    });
    c.create_objects(vec![object(1, None)]).unwrap();
    c.delete_objects(&[item(1)]).unwrap();
    for id in [2, 3] {
        let o = object(id, None);
        c.register_object(o.item, o.root, o.registry).unwrap();
    }
    let before = snapshot(&c);
    for _ in 0..2 {
        assert!(matches!(
            c.execute(EditorAction::Undo),
            Err(ControllerError::BudgetExceeded { .. })
        ));
        assert_eq!(snapshot(&c), before);
    }
}

#[test]
fn mixed_multifield_history_counts_each_object_once_and_replays_original_roots() {
    let mut c = EditorController::with_limits(ControllerLimits {
        max_history_per_object: 3,
        ..ControllerLimits::default()
    });
    c.create_objects(vec![object(1, None), object(2, Some(1))])
        .unwrap();
    let original = snapshot(&c).objects;
    apply_edits(
        &mut c,
        vec![
            edit(1, "value", 7),
            edit(2, "value", 9),
            edit(2, "payload[0]", 8),
        ],
    );
    let edited = snapshot(&c).objects;
    c.select(item(1)).unwrap();
    // Two scalar fields on object 2 are one entry, leaving room to delete.
    c.delete_objects(&[item(1)]).unwrap();
    assert_eq!(c.history_lengths(&item(2)).unwrap(), (3, 0));
    for _ in 0..2 {
        c.execute(EditorAction::Undo).unwrap();
        assert_eq!(snapshot(&c).objects, edited);
        c.execute(EditorAction::Undo).unwrap();
        assert_eq!(snapshot(&c).objects, original);
        c.execute(EditorAction::Undo).unwrap();
        assert_eq!(c.document_objects().len(), 0);
        c.execute(EditorAction::Redo).unwrap();
        assert_eq!(snapshot(&c).objects, original);
        c.execute(EditorAction::Redo).unwrap();
        assert_eq!(snapshot(&c).objects, edited);
        c.execute(EditorAction::Redo).unwrap();
        assert_eq!(c.document_objects().len(), 0);
    }
    c.execute(EditorAction::Undo).unwrap();
    // Branch across a structural redo. Object 2 reaches its retained budget,
    // whereas object 1 still has room; deleting their closure must be atomic.
    apply_edits(&mut c, vec![edit(2, "value", 10)]);
    assert_eq!(c.document_history_lengths(), (3, 0));
    let before = snapshot(&c);
    assert!(matches!(
        c.delete_objects(&[item(1)]),
        Err(ControllerError::BudgetExceeded { .. })
    ));
    assert_eq!(snapshot(&c), before);
}
