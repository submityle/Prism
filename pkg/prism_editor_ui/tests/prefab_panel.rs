//! Public API: real Prefab commands, exact fences, bounded retained Loom views.
use prism_editor_core::prefab::*;
use prism_editor_ui::{
    BoundPrefabPanel, PrefabPanelAction as A, PrefabPanelController as Controller,
    PrefabPanelError as Error, PrefabPanelLimits, SelectionMode, UiEvent,
};
use prism_ui::{
    layout::{AvailableSpace, Size},
    Element, Key, RecordingBackend, Ui,
};
use prism_ui_render_backend::{rasterize, RetainedScene};
use std::{collections::BTreeMap, sync::Arc};

fn asset(n: u128) -> PrefabOwner {
    PrefabOwner::Asset(PrefabId::from_raw(n))
}
fn instance(n: u128) -> PrefabOwner {
    PrefabOwner::Instance(PrefabInstanceId::from_raw(n))
}
fn property(name: &str) -> PrefabProperty {
    PrefabProperty {
        entity: PrefabEntityPath::local(PrefabEntityId::from_raw(1)),
        component: "project/schema@1".into(),
        fields: vec![name.into()],
    }
}
fn pin(owner: PrefabOwner, p: PrefabProperty, value: i64) -> PrefabEdit {
    PrefabEdit::Override {
        owner,
        property: p,
        value: PrefabOverride::Set(Arc::new(PrefabValue::Signed(value))),
    }
}
fn store(fields: usize) -> PrefabStore {
    let mut store = PrefabStore::new(PrefabLimits::default()).unwrap();
    let mut layer = PrefabLayer::default();
    let values = (0..fields)
        .map(|i| (format!("field{i:04}"), PrefabValue::Signed(i as i64)))
        .collect();
    layer.entities.insert(
        PrefabEntityId::from_raw(1),
        PrefabEntity {
            parent: None,
            components: BTreeMap::from([(
                "project/schema@1".into(),
                Arc::new(PrefabValue::Object(values)),
            )]),
        },
    );
    let mut edits = vec![PrefabEdit::InsertAsset(Arc::new(PrefabAsset {
        id: PrefabId::from_raw(1),
        base: None,
        layer,
    }))];
    for n in [2, 3] {
        edits.push(PrefabEdit::InsertAsset(Arc::new(PrefabAsset {
            id: PrefabId::from_raw(n),
            base: Some(PrefabId::from_raw(n - 1)),
            layer: PrefabLayer::default(),
        })));
    }
    for n in [1, 2, 3] {
        edits.push(PrefabEdit::InsertInstance(Arc::new(PrefabInstance {
            id: PrefabInstanceId::from_raw(n),
            prefab: PrefabId::from_raw(3),
            layer: PrefabLayer::default(),
        })));
    }
    store.transact(store.revision(), &edits).unwrap();
    store
}
fn controller(owner: PrefabOwner) -> Controller {
    let mut store = store(3);
    store
        .transact(
            store.revision(),
            &[
                pin(owner, property("field0000"), 80),
                pin(owner, property("field0001"), 90),
            ],
        )
        .unwrap();
    Controller::new(store, owner, PrefabPanelLimits::default()).unwrap()
}
fn read(c: &Controller, owner: PrefabOwner, p: &PrefabProperty) -> Option<PrefabValue> {
    c.store().compose(owner).unwrap().value(p).cloned()
}
fn find<'a>(e: &'a Element, key: &str) -> Option<&'a Element> {
    if matches!(e.explicit_key(),Some(Key::Str(k)) if k==key) {
        return Some(e);
    }
    e.child_elements().iter().find_map(|e| find(e, key))
}
fn send(c: &mut Controller, key: &str) -> Result<(), Error> {
    let v = c.build()?;
    assert!(find(v.element(), key).is_some(), "painted key {key}");
    c.handle_event(&v, key, &UiEvent::Activate)
}
fn select(c: &mut Controller, index: usize, mode: SelectionMode) {
    let v = c.build().unwrap();
    c.handle_event(&v, &v.rows()[index].key, &UiEvent::Select { mode })
        .unwrap();
}
fn all_bindings_painted(v: &BoundPrefabPanel) {
    for (key, _) in v.bindings() {
        assert!(find(v.element(), key).is_some(), "{key}");
    }
}
fn element_keys(e: &Element, out: &mut Vec<Option<String>>) {
    out.push(match e.explicit_key() {
        Some(Key::Str(k)) => Some(k.to_string()),
        _ => None,
    });
    for child in e.child_elements() {
        element_keys(child, out);
    }
}

#[test]
fn painted_revert_apply_base_and_history_write_real_model_atomically() {
    let mut c = controller(instance(1));
    let untouched = c
        .store()
        .instance(PrefabInstanceId::from_raw(3))
        .unwrap()
        .clone();
    c.configure_targets(&c.snapshot(), Some(PrefabId::from_raw(1)), vec![])
        .unwrap();
    select(&mut c, 0, SelectionMode::Replace);
    select(&mut c, 1, SelectionMode::Add);
    assert_eq!(c.selected().len(), 2);
    send(&mut c, "prefab:apply-base").unwrap();
    assert_eq!(c.history_depths(), (1, 0));
    assert!(c.store().layer(instance(1)).unwrap().overrides.is_empty());
    for owner in [instance(1), instance(2), asset(1), asset(3)] {
        assert_eq!(
            read(&c, owner, &property("field0000")),
            Some(PrefabValue::Signed(80))
        );
        assert_eq!(
            read(&c, owner, &property("field0001")),
            Some(PrefabValue::Signed(90))
        );
    }
    assert!(Arc::ptr_eq(
        &untouched,
        c.store().instance(PrefabInstanceId::from_raw(3)).unwrap()
    ));
    send(&mut c, "prefab:undo").unwrap();
    assert_eq!(
        read(&c, instance(2), &property("field0000")),
        Some(PrefabValue::Signed(0))
    );
    assert_eq!(c.store().layer(instance(1)).unwrap().overrides.len(), 2);
    send(&mut c, "prefab:redo").unwrap();
    send(&mut c, "prefab:undo").unwrap();
    select(&mut c, 0, SelectionMode::Replace);
    select(&mut c, 1, SelectionMode::Add);
    send(&mut c, "prefab:revert").unwrap();
    assert_eq!(
        read(&c, instance(1), &property("field0000")),
        Some(PrefabValue::Signed(0))
    );
    assert_eq!(c.history_depths(), (1, 0));
    send(&mut c, "prefab:undo").unwrap();
    assert_eq!(
        read(&c, instance(1), &property("field0001")),
        Some(PrefabValue::Signed(90))
    );
}

#[test]
fn checked_propagation_preserves_other_pins_and_uses_one_inverse_command() {
    let mut c = controller(asset(2));
    c.execute(
        &c.snapshot(),
        vec![
            pin(instance(2), property("field0000"), 333),
            pin(instance(3), property("field0000"), 444),
        ],
    )
    .unwrap();
    let unrelated = c
        .store()
        .instance(PrefabInstanceId::from_raw(3))
        .unwrap()
        .clone();
    c.configure_targets(&c.snapshot(), None, vec![instance(1), instance(2)])
        .unwrap();
    send(&mut c, "prefab:recipient:1").unwrap(); // Explicitly unchecked sibling.
    select(&mut c, 0, SelectionMode::Replace);
    select(&mut c, 1, SelectionMode::Toggle);
    let depth = c.history_depths().0;
    send(&mut c, "prefab:propagate").unwrap();
    assert_eq!(c.history_depths().0, depth + 1);
    assert_eq!(c.store().layer(instance(1)).unwrap().overrides.len(), 2);
    assert_eq!(
        read(&c, instance(1), &property("field0000")),
        Some(PrefabValue::Signed(80))
    );
    assert_eq!(
        read(&c, instance(2), &property("field0000")),
        Some(PrefabValue::Signed(333))
    );
    assert_eq!(
        read(&c, instance(3), &property("field0000")),
        Some(PrefabValue::Signed(444))
    );
    assert!(Arc::ptr_eq(
        &unrelated,
        c.store().instance(PrefabInstanceId::from_raw(3)).unwrap()
    ));
    send(&mut c, "prefab:undo").unwrap();
    assert!(c.store().layer(instance(1)).unwrap().overrides.is_empty());
    send(&mut c, "prefab:redo").unwrap();
    assert_eq!(c.store().layer(instance(1)).unwrap().overrides.len(), 2);
}

#[test]
fn conflict_in_later_selected_field_leaves_entire_batch_and_view_unchanged() {
    let mut c = controller(instance(1));
    c.execute(
        &c.snapshot(),
        vec![pin(asset(2), property("field0001"), 500)],
    )
    .unwrap();
    c.configure_targets(&c.snapshot(), Some(PrefabId::from_raw(1)), vec![])
        .unwrap();
    select(&mut c, 0, SelectionMode::Replace);
    select(&mut c, 1, SelectionMode::Add);
    let bytes = c.store().encode().unwrap();
    let history = c.history_depths();
    let token = c.snapshot();
    let view = c.build().unwrap();
    assert!(matches!(
        c.handle_event(&view, "prefab:apply-base", &UiEvent::Activate),
        Err(Error::Command(_))
    ));
    assert_eq!(c.store().encode().unwrap(), bytes);
    assert_eq!(c.history_depths(), history);
    assert!(c.matches_snapshot(&token));
    assert_eq!(c.selected().len(), 2);
    // Failed publication did not invalidate the still-current view.
    c.handle_event(&view, "prefab:revert", &UiEvent::Activate)
        .unwrap();
    assert!(c.store().layer(instance(1)).unwrap().overrides.is_empty());
}

#[test]
fn model_snapshot_and_controller_view_fences_reject_aba_and_foreign_views() {
    let mut c = controller(instance(1));
    let mut other = controller(instance(1));
    let original = c.build().unwrap();
    let token = c.snapshot();
    assert_eq!(other.store().revision(), c.store().revision());
    assert_eq!(
        other.handle_event(&original, &original.rows()[0].key, &UiEvent::Activate),
        Err(Error::StaleView)
    );
    assert_eq!(other.execute(&token, vec![]), Err(Error::StaleView));
    select(&mut c, 0, SelectionMode::Replace);
    assert!(c.matches_snapshot(&token)); // View-only mutation preserves model token.
    assert_eq!(
        c.handle_event(&original, &original.rows()[0].key, &UiEvent::Activate),
        Err(Error::StaleView)
    );
    let before_edit = c.build().unwrap();
    send(&mut c, "prefab:revert").unwrap();
    send(&mut c, "prefab:undo").unwrap(); // Equal values, different exact publication.
    assert!(!c.matches_snapshot(&token));
    assert_eq!(
        c.handle_event(&before_edit, "prefab:revert", &UiEvent::Activate),
        Err(Error::StaleView)
    );
    let current = c.snapshot();
    let same_revision =
        PrefabStore::decode(&c.store().encode().unwrap(), c.store().limits()).unwrap();
    let view = c.build().unwrap();
    c.replace_store(&current, same_revision, instance(1))
        .unwrap();
    assert!(!c.matches_snapshot(&current));
    assert_eq!(c.history_depths(), (0, 0));
    assert_eq!(
        c.handle_event(&view, "prefab:undo", &UiEvent::Activate),
        Err(Error::StaleView)
    );
}

#[test]
fn reviewed_exact_conflicts_cancel_and_resolve_with_one_real_inverse_history() {
    let mut c = controller(instance(1));
    c.execute(
        &c.snapshot(),
        vec![
            pin(asset(1), property("field0000"), 400),
            pin(asset(2), property("field0001"), 500),
            pin(asset(2), property("field0002"), 600),
            pin(instance(3), property("field0000"), 700),
        ],
    )
    .unwrap();
    c.configure_targets(&c.snapshot(), Some(PrefabId::from_raw(1)), vec![])
        .unwrap();
    select(&mut c, 0, SelectionMode::Replace);
    select(&mut c, 1, SelectionMode::Add);
    let bytes = c.store().encode().unwrap();
    let history = c.history_depths();
    let token = c.snapshot();
    let untouched = c
        .store()
        .instance(PrefabInstanceId::from_raw(3))
        .unwrap()
        .clone();
    send(&mut c, "prefab:review-apply").unwrap();
    let review = c.build().unwrap();
    all_bindings_painted(&review);
    let pins = review.apply_review().unwrap();
    assert_eq!(pins.len(), 2);
    assert_eq!(pins[0].owner, asset(1));
    assert_eq!(pins[1].owner, asset(2));
    assert!(!review.apply_review_blocked());
    assert!(find(review.element(), "prefab:confirm-apply").is_some());
    send(&mut c, "prefab:cancel-apply").unwrap();
    assert!(c.build().unwrap().apply_review().is_none());
    assert_eq!(c.store().encode().unwrap(), bytes);
    assert_eq!(c.history_depths(), history);
    assert!(c.matches_snapshot(&token));
    assert_eq!(
        c.handle_event(&review, "prefab:confirm-apply", &UiEvent::Activate),
        Err(Error::StaleView)
    );
    send(&mut c, "prefab:review-apply").unwrap();
    send(&mut c, "prefab:confirm-apply").unwrap();
    assert_eq!(c.history_depths().0, history.0 + 1);
    assert!(c.build().unwrap().apply_review().is_none());
    assert_eq!(c.store().layer(asset(2)).unwrap().overrides.len(), 1);
    assert_eq!(
        read(&c, asset(2), &property("field0002")),
        Some(PrefabValue::Signed(600))
    );
    assert_eq!(
        read(&c, instance(3), &property("field0000")),
        Some(PrefabValue::Signed(700))
    );
    assert!(Arc::ptr_eq(
        &untouched,
        c.store().instance(PrefabInstanceId::from_raw(3)).unwrap()
    ));
    assert!(c.store().layer(instance(1)).unwrap().overrides.is_empty());
    for owner in [asset(1), asset(2), instance(1), instance(2)] {
        assert_eq!(
            read(&c, owner, &property("field0000")),
            Some(PrefabValue::Signed(80))
        );
        assert_eq!(
            read(&c, owner, &property("field0001")),
            Some(PrefabValue::Signed(90))
        );
    }
    send(&mut c, "prefab:undo").unwrap();
    assert_eq!(
        read(&c, asset(1), &property("field0000")),
        Some(PrefabValue::Signed(400))
    );
    assert_eq!(
        read(&c, asset(2), &property("field0001")),
        Some(PrefabValue::Signed(500))
    );
    assert_eq!(c.store().layer(instance(1)).unwrap().overrides.len(), 2);
    send(&mut c, "prefab:redo").unwrap();
    assert_eq!(
        read(&c, instance(2), &property("field0001")),
        Some(PrefabValue::Signed(90))
    );
}

#[test]
fn review_preserves_broader_and_narrower_pins_and_enforces_materialization_budget() {
    for broad_ancestor in [true, false] {
        let mut s = store(2);
        let mut root = property("field0000");
        root.fields.clear();
        let whole = PrefabOverride::Set(Arc::new(PrefabValue::Object(BTreeMap::from([
            ("field0000".into(), PrefabValue::Signed(700)),
            ("field0001".into(), PrefabValue::Signed(800)),
        ]))));
        let edits = if broad_ancestor {
            vec![
                PrefabEdit::Override {
                    owner: asset(2),
                    property: root,
                    value: whole,
                },
                pin(instance(1), property("field0000"), 99),
            ]
        } else {
            vec![
                pin(asset(1), property("field0000"), 44),
                PrefabEdit::Override {
                    owner: instance(1),
                    property: root,
                    value: whole,
                },
            ]
        };
        s.transact(s.revision(), &edits).unwrap();
        let mut c = Controller::new(s, instance(1), PrefabPanelLimits::default()).unwrap();
        c.configure_targets(&c.snapshot(), Some(PrefabId::from_raw(1)), vec![])
            .unwrap();
        select(&mut c, 0, SelectionMode::Replace);
        let bytes = c.store().encode().unwrap();
        send(&mut c, "prefab:review-apply").unwrap();
        let review = c.build().unwrap();
        assert_eq!(review.apply_review().unwrap().len(), 1);
        assert!(review.apply_review_blocked());
        assert_eq!(
            c.handle_event(&review, "prefab:confirm-apply", &UiEvent::Activate),
            Err(Error::UnboundEvent)
        );
        assert_eq!(c.apply(A::ConfirmApplyReview), Err(Error::Target));
        assert_eq!(c.store().encode().unwrap(), bytes);
        assert_eq!(c.history_depths(), (0, 0));
    }
    let mut s = store(1);
    s.transact(
        s.revision(),
        &[
            pin(asset(1), property("field0000"), 20),
            pin(asset(2), property("field0000"), 30),
            pin(instance(1), property("field0000"), 40),
        ],
    )
    .unwrap();
    let mut c = Controller::new(
        s,
        instance(1),
        PrefabPanelLimits {
            selection: 1,
            ..Default::default()
        },
    )
    .unwrap();
    c.configure_targets(&c.snapshot(), Some(PrefabId::from_raw(1)), vec![])
        .unwrap();
    select(&mut c, 0, SelectionMode::Replace);
    let view = c.build().unwrap();
    assert_eq!(
        c.handle_event(&view, "prefab:review-apply", &UiEvent::Activate),
        Err(Error::Capacity)
    );
    assert!(c.build().unwrap().apply_review().is_none());
    assert_eq!(c.history_depths(), (0, 0));
    c.handle_event(&view, "prefab:revert", &UiEvent::Activate)
        .unwrap();
}

#[test]
fn review_expanded_edit_budget_rejects_without_model_history_or_view_publication() {
    let mut s = controller(instance(1)).into_store();
    s.transact(s.revision(), &[pin(asset(2), property("field0000"), 55)])
        .unwrap();
    let mut limits = s.limits();
    limits.max_edits = 2;
    let s = PrefabStore::decode(&s.encode().unwrap(), limits).unwrap();
    let mut c = Controller::new(s, instance(1), Default::default()).unwrap();
    c.configure_targets(&c.snapshot(), Some(PrefabId::from_raw(1)), vec![])
        .unwrap();
    select(&mut c, 0, SelectionMode::Replace);
    select(&mut c, 1, SelectionMode::Add);
    let token = c.snapshot();
    let bytes = c.store().encode().unwrap();
    let view = c.build().unwrap();
    // Two source applies plus one intermediate revert exceeds the real budget.
    assert_eq!(
        c.handle_event(&view, "prefab:review-apply", &UiEvent::Activate),
        Err(Error::Capacity)
    );
    assert_eq!(c.store().encode().unwrap(), bytes);
    assert_eq!(c.history_depths(), (0, 0));
    assert!(c.matches_snapshot(&token));
    assert!(c.build().unwrap().apply_review().is_none());
    c.handle_event(&view, "prefab:revert", &UiEvent::Activate)
        .unwrap();
}

#[test]
fn failed_review_commit_restores_intervening_pin_and_keeps_review_current() {
    let mut s = store(1);
    let p = property("introduced");
    s.transact(
        s.revision(),
        &[
            pin(asset(2), p.clone(), 500),
            PrefabEdit::Override {
                owner: instance(1),
                property: p.clone(),
                value: PrefabOverride::Remove,
            },
        ],
    )
    .unwrap();
    let mut c = Controller::new(s, instance(1), PrefabPanelLimits::default()).unwrap();
    c.configure_targets(&c.snapshot(), Some(PrefabId::from_raw(1)), vec![])
        .unwrap();
    select(&mut c, 0, SelectionMode::Replace);
    send(&mut c, "prefab:review-apply").unwrap();
    let view = c.build().unwrap();
    let bytes = c.store().encode().unwrap();
    let token = c.snapshot();
    assert!(matches!(
        c.handle_event(&view, "prefab:confirm-apply", &UiEvent::Activate),
        Err(Error::Command(_))
    ));
    // Revert of the intervening Set was staged before the invalid base Remove;
    // the actual kernel transaction must restore all authored records.
    assert_eq!(c.store().encode().unwrap(), bytes);
    assert_eq!(c.history_depths(), (0, 0));
    assert!(c.matches_snapshot(&token));
    assert_eq!(read(&c, asset(2), &p), Some(PrefabValue::Signed(500)));
    assert!(c.build().unwrap().apply_review().is_some());
    c.handle_event(&view, "prefab:cancel-apply", &UiEvent::Activate)
        .unwrap();
}

#[test]
fn review_does_not_survive_selection_configuration_or_model_changes() {
    let mut c = controller(instance(1));
    c.configure_targets(&c.snapshot(), Some(PrefabId::from_raw(1)), vec![])
        .unwrap();
    select(&mut c, 0, SelectionMode::Replace);
    send(&mut c, "prefab:review-apply").unwrap();
    assert!(c.build().unwrap().apply_review().unwrap().is_empty());
    let view = c.build().unwrap();
    select(&mut c, 1, SelectionMode::Replace);
    assert!(c.build().unwrap().apply_review().is_none());
    assert_eq!(
        c.handle_event(&view, "prefab:confirm-apply", &UiEvent::Activate),
        Err(Error::StaleView)
    );
    send(&mut c, "prefab:review-apply").unwrap();
    c.configure_targets(&c.snapshot(), Some(PrefabId::from_raw(2)), vec![])
        .unwrap();
    assert!(c.build().unwrap().apply_review().is_none());
    send(&mut c, "prefab:review-apply").unwrap();
    c.execute(
        &c.snapshot(),
        vec![pin(instance(2), property("field0001"), 700)],
    )
    .unwrap();
    assert!(c.build().unwrap().apply_review().is_none());
    assert_eq!(c.apply(A::ConfirmApplyReview), Err(Error::Target));
}

#[test]
fn off_page_review_proposals_are_shared_painted_and_confirmed_via_retained_controls() {
    let mut c = Controller::new(
        controller(instance(1)).into_store(),
        instance(1),
        PrefabPanelLimits {
            rows: 1,
            ..Default::default()
        },
    )
    .unwrap();
    c.configure_targets(&c.snapshot(), Some(PrefabId::from_raw(1)), vec![])
        .unwrap();
    select(&mut c, 0, SelectionMode::Replace);
    send(&mut c, "prefab:next").unwrap();
    select(&mut c, 0, SelectionMode::Add);
    send(&mut c, "prefab:review-apply").unwrap();
    let view = c.build().unwrap();
    assert_eq!(view.rows().len(), 1);
    let proposed = view.apply_proposals().unwrap();
    assert_eq!(proposed.len(), 2);
    for row in proposed {
        assert_eq!(row.owner, instance(1));
        assert!(find(view.element(), &row.key).is_some());
        let PrefabOverride::Set(value) = &row.value else {
            panic!("actual shared source Set");
        };
        let PrefabOverride::Set(stored) =
            &c.store().layer(instance(1)).unwrap().overrides[&row.property]
        else {
            panic!("stored source Set");
        };
        assert!(Arc::ptr_eq(value, stored));
    }
    let mut ui = Ui::new(RetainedScene::new());
    ui.mount(view.element());
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(1100.),
        AvailableSpace::MaxContent,
    ));
    let mut rects = Vec::new();
    ui.backend().visit_layout(|_, rect| rects.push(rect));
    let mut names = Vec::new();
    element_keys(view.element(), &mut names);
    assert_eq!(names.len(), rects.len());
    let confirm = names
        .iter()
        .position(|key| key.as_deref() == Some("prefab:confirm-apply"))
        .unwrap();
    let rect = rects[confirm];
    assert!(rect.size.width > 0. && rect.size.height >= 28.);
    let image = rasterize(&ui.backend().to_draw_list(), 1100, 1600);
    let pixel = image.pixel((rect.left() + 770.) as u32, (rect.top() + 20.) as u32);
    assert!((pixel[2] - 0.22).abs() < 0.01, "enabled confirm {pixel:?}");
    c.handle_event(&view, "prefab:confirm-apply", &UiEvent::Activate)
        .unwrap();
    assert_eq!(c.history_depths(), (1, 0));
    assert!(c.store().layer(instance(1)).unwrap().overrides.is_empty());
    assert_eq!(
        read(&c, asset(1), &property("field0000")),
        Some(PrefabValue::Signed(80))
    );
}

#[test]
fn keyset_pages_are_bounded_bidirectional_and_cross_page_selection_is_limited() {
    let mut s = store(101);
    let edits: Vec<_> = (0..101)
        .map(|i| pin(instance(1), property(&format!("field{i:04}")), 999))
        .collect();
    s.transact(s.revision(), &edits).unwrap();
    let mut c = Controller::new(
        s,
        instance(1),
        PrefabPanelLimits {
            rows: 7,
            selection: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let first = c.build().unwrap();
    assert_eq!(first.stats().examined, 8);
    assert!(!first.stats().has_previous);
    assert!(first.stats().has_next);
    all_bindings_painted(&first);
    assert_eq!(
        c.apply(A::Before(first.rows()[0].property.clone())),
        Err(Error::Target)
    );
    select(&mut c, 0, SelectionMode::Replace);
    send(&mut c, "prefab:next").unwrap();
    assert_eq!(c.build().unwrap().rows()[0].property, property("field0007"));
    select(&mut c, 0, SelectionMode::Add);
    let v = c.build().unwrap();
    assert_eq!(
        c.handle_event(
            &v,
            &v.rows()[1].key,
            &UiEvent::Select {
                mode: SelectionMode::Add
            }
        ),
        Err(Error::Capacity)
    );
    assert_eq!(c.selected().len(), 2);
    c.handle_event(&v, "prefab:next", &UiEvent::Activate)
        .unwrap(); // Failure preserves fence.
    send(&mut c, "prefab:previous").unwrap();
    assert_eq!(c.build().unwrap().rows()[0].property, property("field0007"));
    send(&mut c, "prefab:first").unwrap();
    assert_eq!(
        c.build().unwrap().rows()[0].property,
        first.rows()[0].property
    );
    let mut count = 0;
    loop {
        let v = c.build().unwrap();
        assert!(v.stats().examined <= 8);
        count += v.rows().len();
        if !v.stats().has_next {
            break;
        }
        c.handle_event(&v, "prefab:next", &UiEvent::Activate)
            .unwrap();
    }
    assert_eq!(count, 101);
    assert_eq!(c.build().unwrap().rows().len(), 3);
    let end = c.build().unwrap();
    assert_eq!(
        c.apply(A::After(end.rows().last().unwrap().property.clone())),
        Err(Error::Target)
    );
    assert_eq!(c.build().unwrap().rows()[0].key, end.rows()[0].key);
}

#[test]
fn semantic_paths_nested_slots_remove_null_and_shared_values_are_lossless() {
    let mut s = store(3);
    let mut host = PrefabAsset {
        id: PrefabId::from_raw(10),
        base: None,
        layer: PrefabLayer::default(),
    };
    host.layer.slots.insert(
        PrefabSlotId::from_raw(8),
        Some(PrefabSlot {
            prefab: PrefabId::from_raw(1),
            parent: None,
        }),
    );
    let mut p = property(" literal. / 空\n ");
    p.entity.slots.push(PrefabSlotId::from_raw(8));
    let shared = Arc::new(PrefabValue::Array(vec![PrefabValue::Bytes(vec![
        9;
        1024
            * 1024
    ])]));
    s.transact(
        s.revision(),
        &[
            PrefabEdit::InsertAsset(Arc::new(host)),
            PrefabEdit::Override {
                owner: asset(10),
                property: p.clone(),
                value: PrefabOverride::Set(shared.clone()),
            },
            PrefabEdit::Override {
                owner: asset(10),
                property: {
                    let mut q = p.clone();
                    q.fields = vec!["null".into()];
                    q
                },
                value: PrefabOverride::Set(Arc::new(PrefabValue::Null)),
            },
            PrefabEdit::Override {
                owner: asset(10),
                property: {
                    let mut q = p.clone();
                    q.fields = vec!["field0000".into()];
                    q
                },
                value: PrefabOverride::Remove,
            },
        ],
    )
    .unwrap();
    let mut c = Controller::new(s, asset(10), Default::default()).unwrap();
    let view = c.build().unwrap();
    let row = view.rows().iter().find(|r| r.property == p).unwrap();
    assert!(row.label.contains("空\\n"));
    let PrefabOverride::Set(value) = &row.value else {
        panic!("actual shared Set")
    };
    assert!(Arc::ptr_eq(value, &shared));
    assert!(view
        .rows()
        .iter()
        .any(|r| r.preview == "Set Null (explicit)"));
    assert!(view.rows().iter().any(|r| r.preview == "Remove (explicit)"));
    c.handle_event(&view, &row.key, &UiEvent::Activate).unwrap();
    assert!(c.selected().contains(&p));
    send(&mut c, "prefab:revert").unwrap();
    assert_eq!(read(&c, asset(10), &p), None);
    send(&mut c, "prefab:undo").unwrap();
    let restored = c.build().unwrap();
    let row = restored.rows().iter().find(|r| r.property == p).unwrap();
    let PrefabOverride::Set(value) = &row.value else {
        panic!("Set")
    };
    assert!(Arc::ptr_eq(value, &shared));
}

#[test]
fn selection_configuration_history_and_address_budgets_fail_explicitly() {
    let mut c = Controller::new(
        controller(instance(1)).into_store(),
        instance(1),
        PrefabPanelLimits {
            history: 1,
            recipients: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let snapshot = c.snapshot();
    assert_eq!(
        c.configure_targets(&snapshot, None, vec![instance(1), instance(2)]),
        Err(Error::Capacity)
    );
    assert!(c.matches_snapshot(&snapshot));
    select(&mut c, 0, SelectionMode::Replace);
    send(&mut c, "prefab:revert").unwrap();
    let token = c.snapshot();
    let bytes = c.store().encode().unwrap();
    assert_eq!(
        c.execute(&token, vec![pin(instance(1), property("field0000"), 10)]),
        Err(Error::Capacity)
    );
    assert_eq!(c.store().encode().unwrap(), bytes);
    send(&mut c, "prefab:undo").unwrap();
    c.execute(
        &c.snapshot(),
        vec![pin(instance(1), property("field0000"), 10)],
    )
    .unwrap(); // Branch replaces redo within ceiling.
    assert_eq!(c.history_depths(), (1, 0));
    let c = Controller::new(
        c.into_store(),
        instance(1),
        PrefabPanelLimits {
            address_bytes: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(c.build(), Err(Error::Capacity)));
}

#[test]
fn propagation_expansion_budget_rejects_before_any_write_or_history() {
    let c = controller(asset(2));
    let mut limits = c.store().limits();
    limits.max_edits = 3;
    let s = PrefabStore::decode(&c.store().encode().unwrap(), limits).unwrap();
    let mut c = Controller::new(s, asset(2), Default::default()).unwrap();
    c.configure_targets(&c.snapshot(), None, vec![instance(1), instance(2)])
        .unwrap();
    select(&mut c, 0, SelectionMode::Replace);
    select(&mut c, 1, SelectionMode::Add);
    let token = c.snapshot();
    let before = c.store().encode().unwrap();
    assert_eq!(send(&mut c, "prefab:propagate"), Err(Error::Capacity));
    assert_eq!(c.store().encode().unwrap(), before);
    assert_eq!(c.history_depths(), (0, 0));
    assert!(c.matches_snapshot(&token));
}

#[test]
fn owner_changes_disabled_controls_unbound_events_and_reload_failures_are_explicit() {
    let mut c = controller(instance(1));
    let token = c.snapshot();
    let view = c.build().unwrap();
    assert_eq!(
        c.handle_event(&view, "prefab:revert", &UiEvent::Activate),
        Err(Error::UnboundEvent)
    );
    assert_eq!(
        c.handle_event(&view, "prefab:redo", &UiEvent::Activate),
        Err(Error::UnboundEvent)
    );
    assert_eq!(
        c.handle_event(
            &view,
            &view.rows()[0].key,
            &UiEvent::Commit {
                value: prism_editor_ui::ActionValue::integer(2)
            }
        ),
        Err(Error::UnboundEvent)
    );
    c.configure_targets(&token, Some(PrefabId::from_raw(1)), vec![instance(2)])
        .unwrap();
    c.apply(A::Owner(asset(2))).unwrap();
    assert!(c.matches_snapshot(&token));
    assert!(c.selected().is_empty());
    assert!(c.build().unwrap().rows().is_empty());
    assert_eq!(
        c.handle_event(&view, "prefab:revert", &UiEvent::Activate),
        Err(Error::StaleView)
    );
    let next = c.build().unwrap();
    assert!(!next
        .bindings()
        .any(|(key, _)| key.starts_with("prefab:recipient:")));
    let before = c.store().encode().unwrap();
    assert!(c.replace_store(&token, store(1), asset(999)).is_err());
    assert_eq!(c.store().encode().unwrap(), before);
    assert!(c.matches_snapshot(&token));
    assert_eq!(c.owner(), asset(2));
}

#[test]
fn invalid_propagation_recipient_rolls_back_before_other_recipients_and_keeps_config_fence() {
    let mut c = controller(asset(2));
    c.configure_targets(&c.snapshot(), None, vec![instance(1), asset(1)])
        .unwrap();
    select(&mut c, 0, SelectionMode::Replace);
    let token = c.snapshot();
    let bytes = c.store().encode().unwrap();
    assert!(matches!(
        send(&mut c, "prefab:propagate"),
        Err(Error::Command(_))
    ));
    assert_eq!(c.store().encode().unwrap(), bytes);
    assert!(c.store().layer(instance(1)).unwrap().overrides.is_empty());
    assert_eq!(c.history_depths(), (0, 0));
    assert!(c.matches_snapshot(&token));
    let view = c.build().unwrap();
    assert_eq!(
        c.configure_targets(&token, None, vec![instance(1), instance(1)]),
        Err(Error::Target)
    );
    c.handle_event(&view, "prefab:recipient:1", &UiEvent::Activate)
        .unwrap();
    send(&mut c, "prefab:propagate").unwrap();
    assert_eq!(c.store().layer(instance(1)).unwrap().overrides.len(), 1);
}

#[test]
fn actual_retained_layout_raster_and_reconciliation_follow_real_selection() {
    let mut c = controller(instance(1));
    let original = c.build().unwrap();
    let mut recorder = Ui::new(RecordingBackend::new());
    recorder.mount(original.element());
    recorder.backend_mut().clear();
    recorder.update(c.build().unwrap().element());
    assert!(recorder.backend().ops().is_empty());
    select(&mut c, 0, SelectionMode::Replace);
    let view = c.build().unwrap();
    all_bindings_painted(&view);
    recorder.update(view.element());
    assert!(!recorder.backend().ops().is_empty());
    let mut ui = Ui::new(RetainedScene::new());
    ui.mount(view.element());
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(1100.),
        AvailableSpace::MaxContent,
    ));
    let mut rects = Vec::new();
    ui.backend().visit_layout(|_, rect| rects.push(rect));
    let mut names = Vec::new();
    element_keys(view.element(), &mut names);
    assert_eq!(names.len(), rects.len());
    let rects: BTreeMap<_, _> = names
        .into_iter()
        .zip(rects)
        .filter_map(|(k, r)| k.map(|k| (k, r)))
        .collect();
    let selected = rects[&view.rows()[0].key];
    let unselected = rects[&view.rows()[1].key];
    assert!((selected.left() - 260.).abs() < 0.1);
    assert!((unselected.top() - selected.top() - 48.).abs() < 0.1);
    let list = ui.backend().to_draw_list();
    assert!(!list.is_empty());
    let image = rasterize(&list, 1100, 600);
    let a = image.pixel(
        (selected.left() + 770.) as u32,
        (selected.top() + 40.) as u32,
    );
    let b = image.pixel(
        (unselected.left() + 770.) as u32,
        (unselected.top() + 40.) as u32,
    );
    assert!((a[2] - 0.52).abs() < 0.01, "selected {a:?}");
    assert!((b[2] - 0.13).abs() < 0.01, "unselected {b:?}");
    c.handle_event(&view, "prefab:revert", &UiEvent::Activate)
        .unwrap();
    assert_eq!(
        read(&c, instance(1), &property("field0000")),
        Some(PrefabValue::Signed(0))
    );
    recorder.backend_mut().clear();
    recorder.update(c.build().unwrap().element());
    assert!(!recorder.backend().ops().is_empty());
}
