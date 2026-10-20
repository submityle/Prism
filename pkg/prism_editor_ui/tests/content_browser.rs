//! Content browser events exercised through real Loom trees and catalog updates.

use std::collections::{BTreeMap, BTreeSet};

use prism_editor_content::{
    AssetId, AssetKind, AssetRecord, CatalogSnapshotLimits, ContentCatalog, ContentLimits,
    VcsRecord, VcsState,
};
use prism_editor_ui::{
    ActionValue, BoundContentBrowser, ContentBrowserAction as Action, ContentBrowserController,
    ContentBrowserError, SelectionMode, UiEvent, MAX_ACTION_TEXT_BYTES, MAX_CONTENT_PAGE_HISTORY,
    MAX_VISIBLE_ROWS,
};
use prism_ui::{BackendId, BackendOp, Element, Key, RecordingBackend, Ui};

fn asset(raw: u128, path: &str, kind: &str) -> AssetRecord {
    AssetRecord::new(AssetId::from_raw(raw), path, AssetKind::new(kind).unwrap()).unwrap()
}

fn find<'a>(element: &'a Element, key: &str) -> Option<&'a Element> {
    if matches!(element.explicit_key(), Some(Key::Str(value)) if value == key) {
        return Some(element);
    }
    element
        .child_elements()
        .iter()
        .find_map(|child| find(child, key))
}

fn elements<'a>(element: &'a Element, out: &mut Vec<&'a Element>) {
    out.push(element);
    for child in element.child_elements() {
        elements(child, out);
    }
}

fn row_ids(view: &BoundContentBrowser) -> Vec<u128> {
    find(view.element(), "content:list")
        .unwrap()
        .child_elements()
        .iter()
        .filter_map(|el| match el.explicit_key() {
            Some(Key::Str(key)) => key
                .strip_prefix("content:asset:")
                .map(|v| v.parse().unwrap()),
            _ => None,
        })
        .collect()
}

fn send(
    controller: &mut ContentBrowserController,
    catalog: &mut ContentCatalog,
    view: &BoundContentBrowser,
    key: &str,
    event: UiEvent,
) -> Action {
    // The host obtains the target from a real Element, never a detached metadata fixture.
    let target = match find(view.element(), key)
        .expect("interactive Element exists")
        .explicit_key()
        .unwrap()
    {
        Key::Str(target) => target,
        _ => unreachable!(),
    };
    controller
        .handle_event(catalog, view, target, &event)
        .unwrap()
}

fn commit(value: &str) -> UiEvent {
    UiEvent::Commit {
        value: ActionValue::text(value).unwrap(),
    }
}

#[test]
fn loom_events_mutate_catalog_and_reconcile_existing_nodes() {
    let mut catalog = ContentCatalog::new();
    let mut record = asset(71, "hello.png", "texture");
    record.add_tag("hero").unwrap();
    record.dependencies.insert(AssetId::from_raw(999));
    catalog.insert(record.clone()).unwrap();
    let mut controller = ContentBrowserController::default();
    let view = controller.build(&mut catalog).unwrap();
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    ui.compute_layout(prism_ui::layout::Size::new(
        prism_ui::layout::AvailableSpace::Definite(1000.0),
        prism_ui::layout::AvailableSpace::Definite(800.0),
    ));
    let mut nodes = Vec::new();
    elements(view.element(), &mut nodes);
    let backend_ids: Vec<_> = ui
        .backend()
        .ops()
        .iter()
        .filter_map(|op| {
            if let BackendOp::Create { id, .. } = op {
                Some(*id)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(nodes.len(), backend_ids.len());
    let row_index = nodes
        .iter()
        .position(
            |el| matches!(el.explicit_key(), Some(Key::Str(key)) if key == "content:asset:71"),
        )
        .unwrap();
    let row_id = backend_ids[row_index];
    let bindings: Vec<_> = view.bindings().collect();
    for (key, _) in &bindings {
        assert!(find(view.element(), key).is_some(), "phantom binding {key}");
    }
    assert_eq!(bindings.len(), 14);
    ui.backend_mut().clear();
    let unchanged = controller.build(&mut catalog).unwrap();
    ui.update(unchanged.element());
    assert!(
        ui.backend().is_empty(),
        "same state must produce zero Loom mutations"
    );

    let generation = catalog.generation();
    assert_eq!(
        send(
            &mut controller,
            &mut catalog,
            &view,
            "content:favorite:71",
            UiEvent::Activate
        ),
        Action::SetFavorite {
            asset: record.id,
            value: true
        }
    );
    assert!(catalog.generation() > generation);
    let next = controller.build(&mut catalog).unwrap();
    ui.update(next.element());
    assert!(ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::SetText { text, .. } if text == "Favorite: on")));
    assert!(!ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::Create { .. } | BackendOp::Remove { .. })));
    assert_eq!(catalog.get(record.id).unwrap().tags, record.tags);
    assert_eq!(
        catalog.get(record.id).unwrap().dependencies,
        record.dependencies
    );

    send(
        &mut controller,
        &mut catalog,
        &next,
        "content:pin:71",
        UiEvent::Activate,
    );
    let next = controller.build(&mut catalog).unwrap();
    ui.backend_mut().clear();
    ui.update(next.element());
    assert!(ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::SetText { text, .. } if text == "Pinned: on")));
    assert!(catalog.get(record.id).unwrap().pinned);
    send(
        &mut controller,
        &mut catalog,
        &next,
        "content:asset:71",
        UiEvent::Select {
            mode: SelectionMode::Replace,
        },
    );
    let selected = controller.build(&mut catalog).unwrap();
    ui.backend_mut().clear();
    ui.update(selected.element());
    assert!(ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::SetText { text, .. } if text == "[x]")));
    assert!(!ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::Remove { id } if *id == row_id)));

    let mut renamed = catalog.get(record.id).unwrap().clone();
    renamed.path = "renamed.png".into();
    catalog.update(renamed).unwrap();
    let renamed = controller.build(&mut catalog).unwrap();
    ui.backend_mut().clear();
    ui.update(renamed.element());
    assert!(ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::SetText { text, .. } if text == "renamed.png")));
    assert!(!ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::Create { .. } | BackendOp::Remove { .. })));
    assert!(controller.selection().contains(&record.id));
    assert_eq!(row_ids(&renamed), [71]);
    assert!(ui.node_count() < 100);
}

#[test]
fn all_filters_and_folder_navigation_flow_through_rendered_controls() {
    let mut catalog = ContentCatalog::new();
    let mut hero = asset(1, "艺术 folder/sub/hero.png", "texture");
    hero.add_tag("Hero").unwrap();
    hero.favorite = true;
    hero.pinned = true;
    catalog.insert(hero).unwrap();
    catalog
        .insert(asset(2, "艺术 folder/sub/wall.png", "texture"))
        .unwrap();
    catalog.insert(asset(3, "other.mesh", "mesh")).unwrap();
    catalog
        .set_vcs_status(
            "艺术 folder/sub/hero.png",
            VcsRecord {
                state: VcsState::Modified,
                staged: true,
                ..VcsRecord::default()
            },
        )
        .unwrap();
    let mut controller = ContentBrowserController::default();
    let mut ui = Ui::new(RecordingBackend::new());
    let mut view = controller.build(&mut catalog).unwrap();
    ui.mount(view.element());
    for (key, event) in [
        ("content:search", commit("HERO")),
        ("content:kind", commit("texture")),
        ("content:tag", commit("hero")),
        ("content:vcs", commit("modified")),
        ("content:favorites-only", UiEvent::Activate),
        ("content:pinned-only", UiEvent::Activate),
        ("content:folder:艺术 folder", UiEvent::Activate),
        ("content:folder:艺术 folder/sub", UiEvent::Activate),
    ] {
        send(&mut controller, &mut catalog, &view, key, event);
        view = controller.build(&mut catalog).unwrap();
        ui.update(view.element());
    }
    assert_eq!(row_ids(&view), [1]);
    assert_eq!(controller.query().root, "艺术 folder/sub");
    assert!(find(view.element(), "content:asset:1")
        .unwrap()
        .child_elements()
        .iter()
        .any(|el| el.text_content() == Some("VCS: modified (staged)")));
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:favorite:1",
        UiEvent::Activate,
    );
    view = controller.build(&mut catalog).unwrap();
    ui.backend_mut().clear();
    ui.update(view.element());
    assert!(row_ids(&view).is_empty());
    assert!(ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::Remove { .. })));
    assert!(!catalog.get(AssetId::from_raw(1)).unwrap().favorite);
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:up",
        UiEvent::Activate,
    );
    view = controller.build(&mut catalog).unwrap();
    assert_eq!(controller.query().root, "艺术 folder");
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:home",
        UiEvent::Activate,
    );
    view = controller.build(&mut catalog).unwrap();
    assert_eq!(controller.query().root, "");
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:vcs",
        commit(""),
    );
    assert_eq!(controller.query().vcs, None);
}

#[test]
fn paging_multiselection_and_refresh_are_bounded_and_use_stable_ids() {
    let mut catalog = ContentCatalog::new();
    for raw in 0..6 {
        catalog
            .insert(asset(raw, &format!("{raw}.png"), "texture"))
            .unwrap();
    }
    let mut controller = ContentBrowserController::default();
    controller.apply(&mut catalog, Action::PageSize(2)).unwrap();
    let mut view = controller.build(&mut catalog).unwrap();
    assert_eq!(row_ids(&view), [0, 1]);
    for (key, mode) in [
        ("content:asset:0", SelectionMode::Replace),
        ("content:asset:1", SelectionMode::Add),
    ] {
        send(
            &mut controller,
            &mut catalog,
            &view,
            key,
            UiEvent::Select { mode },
        );
        view = controller.build(&mut catalog).unwrap();
    }
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:next",
        UiEvent::Activate,
    );
    view = controller.build(&mut catalog).unwrap();
    assert_eq!(row_ids(&view), [2, 3]);
    assert_eq!(controller.page_offset(), 2);
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:asset:3",
        UiEvent::Select {
            mode: SelectionMode::Add,
        },
    );
    view = controller.build(&mut catalog).unwrap();
    assert_eq!(controller.selection().len(), 3);
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:asset:2",
        UiEvent::Select {
            mode: SelectionMode::Extend,
        },
    );
    view = controller.build(&mut catalog).unwrap();
    assert_eq!(
        controller
            .selection()
            .iter()
            .map(|id| id.raw())
            .collect::<Vec<_>>(),
        [2, 3]
    );
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:asset:2",
        UiEvent::Select {
            mode: SelectionMode::Toggle,
        },
    );
    view = controller.build(&mut catalog).unwrap();
    assert_eq!(
        controller
            .selection()
            .iter()
            .map(|id| id.raw())
            .collect::<Vec<_>>(),
        [3]
    );
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:previous",
        UiEvent::Activate,
    );
    view = controller.build(&mut catalog).unwrap();
    assert_eq!(row_ids(&view), [0, 1]);
    catalog.remove(AssetId::from_raw(3)).unwrap();
    assert_eq!(
        controller.handle_event(&mut catalog, &view, "content:next", &UiEvent::Activate),
        Err(ContentBrowserError::StaleView)
    );
    view = controller.build(&mut catalog).unwrap();
    assert!(controller.selection().is_empty());
    assert_eq!(controller.page_offset(), 0);
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:next",
        UiEvent::Activate,
    );
    view = controller.build(&mut catalog).unwrap();
    assert_eq!(row_ids(&view), [2, 4]);
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:next",
        UiEvent::Activate,
    );
    view = controller.build(&mut catalog).unwrap();
    assert_eq!(row_ids(&view), [5]);
    assert!(!view.bindings().any(|(key, _)| key == "content:next"));
}

#[test]
fn invalid_inputs_and_stale_views_do_not_write_catalog_or_query() {
    let mut catalog = ContentCatalog::new();
    catalog.insert(asset(1, "one.png", "texture")).unwrap();
    let mut controller = ContentBrowserController::default();
    let view = controller.build(&mut catalog).unwrap();
    let generation = catalog.generation();
    for (key, event, expected) in [
        (
            "content:search",
            UiEvent::Commit {
                value: ActionValue::integer(3),
            },
            ContentBrowserError::InvalidValue,
        ),
        (
            "content:vcs",
            commit("invented"),
            ContentBrowserError::InvalidValue,
        ),
        (
            "content:asset:1",
            UiEvent::Activate,
            ContentBrowserError::UnboundEvent,
        ),
        (
            "content:favorite:9",
            UiEvent::Activate,
            ContentBrowserError::UnboundEvent,
        ),
    ] {
        assert_eq!(
            controller.handle_event(&mut catalog, &view, key, &event),
            Err(expected)
        );
    }
    assert!(controller
        .apply(&mut catalog, Action::Navigate("../outside".into()))
        .is_err());
    assert!(controller
        .apply(&mut catalog, Action::Navigate("missing".into()))
        .is_err());
    assert_eq!(
        controller.apply(
            &mut catalog,
            Action::Search("x".repeat(MAX_ACTION_TEXT_BYTES + 1))
        ),
        Err(ContentBrowserError::TextLimit)
    );
    assert_eq!(
        controller.apply(&mut catalog, Action::PageSize(0)),
        Err(ContentBrowserError::InvalidPageSize)
    );
    assert_eq!(
        controller.apply(
            &mut catalog,
            Action::Select {
                asset: AssetId::from_raw(9),
                mode: SelectionMode::Replace
            }
        ),
        Err(ContentBrowserError::AssetNotVisible)
    );
    assert_eq!(catalog.generation(), generation);
    assert_eq!(
        controller.query(),
        &prism_editor_content::TreeQuery::default()
    );
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:favorite:1",
        UiEvent::Activate,
    );
    assert_eq!(
        controller.handle_event(
            &mut catalog,
            &view,
            "content:favorite:1",
            &UiEvent::Activate
        ),
        Err(ContentBrowserError::StaleView)
    );
    assert!(catalog.get(AssetId::from_raw(1)).unwrap().favorite);
}

#[test]
fn page_and_history_limits_empty_catalog_and_deleted_folder_are_real_views() {
    let mut catalog = ContentCatalog::with_limits(ContentLimits {
        max_page_size: 3,
        ..ContentLimits::default()
    })
    .unwrap();
    let mut controller = ContentBrowserController::default();
    let empty = controller.build(&mut catalog).unwrap();
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(empty.element());
    assert!(ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::SetText { text, .. } if text == "Catalog is empty")));
    for raw in 0..(MAX_CONTENT_PAGE_HISTORY as u128 + 5) {
        catalog
            .insert(asset(raw, &format!("{raw:04}.png"), "texture"))
            .unwrap();
    }
    controller
        .apply(&mut catalog, Action::PageSize(usize::MAX))
        .unwrap();
    let mut view = controller.build(&mut catalog).unwrap();
    assert_eq!(row_ids(&view).len(), 3);
    assert!(row_ids(&view).len() <= MAX_VISIBLE_ROWS);
    controller.apply(&mut catalog, Action::PageSize(1)).unwrap();
    for _ in 0..(MAX_CONTENT_PAGE_HISTORY + 4) {
        controller.apply(&mut catalog, Action::NextPage).unwrap();
    }
    for _ in 0..MAX_CONTENT_PAGE_HISTORY {
        controller
            .apply(&mut catalog, Action::PreviousPage)
            .unwrap();
    }
    view = controller.build(&mut catalog).unwrap();
    assert_eq!(controller.page_offset(), 4);
    assert!(!view.bindings().any(|(key, _)| key == "content:previous"));
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:home",
        UiEvent::Activate,
    );
    assert_eq!(controller.page_offset(), 0);

    catalog
        .insert(asset(1000, "gone/only.png", "texture"))
        .unwrap();
    controller
        .apply(&mut catalog, Action::Navigate("gone".into()))
        .unwrap();
    view = controller.build(&mut catalog).unwrap();
    ui.update(view.element());
    catalog.remove(AssetId::from_raw(1000)).unwrap();
    view = controller.build(&mut catalog).unwrap();
    ui.update(view.element());
    assert_eq!(controller.query().root, "");
    assert!(!row_ids(&view).is_empty());
}

#[test]
fn every_materialized_key_is_unique_and_bound_nodes_have_layout() {
    let mut catalog = ContentCatalog::new();
    catalog.insert(asset(1, "a.png", "texture")).unwrap();
    catalog.insert(asset(2, "nested/a.png", "texture")).unwrap();
    let view = ContentBrowserController::default()
        .build(&mut catalog)
        .unwrap();
    let mut nodes = Vec::new();
    elements(view.element(), &mut nodes);
    let mut keys = BTreeSet::new();
    for node in &nodes {
        if let Some(Key::Str(key)) = node.explicit_key() {
            assert!(keys.insert(key), "duplicate key {key}");
        }
    }
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    ui.compute_layout(prism_ui::layout::Size::new(
        prism_ui::layout::AvailableSpace::Definite(1200.0),
        prism_ui::layout::AvailableSpace::Definite(900.0),
    ));
    let ids: Vec<BackendId> = ui
        .backend()
        .ops()
        .iter()
        .filter_map(|op| match op {
            BackendOp::Create { id, .. } => Some(*id),
            _ => None,
        })
        .collect();
    let layouts: BTreeMap<_, _> = ui
        .backend()
        .ops()
        .iter()
        .filter_map(|op| match op {
            BackendOp::SetLayout { id, size, .. } => Some((*id, size)),
            _ => None,
        })
        .collect();
    for (target, _) in view.bindings() {
        let index = nodes
            .iter()
            .position(|el| matches!(el.explicit_key(), Some(Key::Str(key)) if key == target))
            .unwrap();
        let size = layouts[&ids[index]];
        assert!(
            size.width > 0.0 && size.height > 0.0,
            "unhittable target {target}: {size:?}"
        );
    }
}

#[test]
fn independent_filters_and_pin_removal_update_the_reconciled_list() {
    let mut catalog = ContentCatalog::new();
    let mut hero = asset(1, "hero.png", "texture");
    hero.add_tag("hero").unwrap();
    hero.favorite = true;
    hero.pinned = true;
    catalog.insert(hero).unwrap();
    catalog.insert(asset(2, "wall.mesh", "mesh")).unwrap();
    catalog
        .set_vcs_status(
            "hero.png",
            VcsRecord {
                state: VcsState::Modified,
                ..VcsRecord::default()
            },
        )
        .unwrap();
    let mut controller = ContentBrowserController::default();
    let mut view = controller.build(&mut catalog).unwrap();
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    for (key, event, expected) in [
        ("content:search", commit("wall"), vec![2]),
        ("content:search", commit(""), vec![1, 2]),
        ("content:kind", commit("mesh"), vec![2]),
        ("content:kind", commit(""), vec![1, 2]),
        ("content:tag", commit("HERO"), vec![1]),
        ("content:tag", commit(""), vec![1, 2]),
        ("content:vcs", commit("unknown"), vec![2]),
        ("content:vcs", commit("clean"), vec![]),
        ("content:vcs", commit("all"), vec![1, 2]),
        ("content:favorites-only", UiEvent::Activate, vec![1]),
        ("content:favorites-only", UiEvent::Activate, vec![1, 2]),
        ("content:pinned-only", UiEvent::Activate, vec![1]),
        ("content:pin:1", UiEvent::Activate, vec![]),
    ] {
        send(&mut controller, &mut catalog, &view, key, event);
        view = controller.build(&mut catalog).unwrap();
        ui.update(view.element());
        assert_eq!(row_ids(&view), expected, "filter {key}");
    }
    assert!(!catalog.get(AssetId::from_raw(1)).unwrap().pinned);
}

#[test]
fn atomic_catalog_publication_invalidates_events_but_preserves_surviving_selection() {
    let mut catalog = ContentCatalog::new();
    catalog.insert(asset(1, "old/hero.png", "texture")).unwrap();
    catalog
        .insert(asset(2, "old/remove.png", "texture"))
        .unwrap();
    let mut controller = ContentBrowserController::default();
    controller
        .apply(&mut catalog, Action::Navigate("old".into()))
        .unwrap();
    let mut view = controller.build(&mut catalog).unwrap();
    for key in ["content:asset:1", "content:asset:2"] {
        send(
            &mut controller,
            &mut catalog,
            &view,
            key,
            UiEvent::Select {
                mode: SelectionMode::Add,
            },
        );
        view = controller.build(&mut catalog).unwrap();
    }
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    let generation = catalog.generation();
    catalog
        .replace_assets(
            generation,
            [
                asset(1, "new/hero.png", "texture"),
                asset(3, "added.png", "texture"),
            ],
            CatalogSnapshotLimits::default(),
        )
        .unwrap();
    assert_eq!(
        controller.handle_event(
            &mut catalog,
            &view,
            "content:favorite:1",
            &UiEvent::Activate
        ),
        Err(ContentBrowserError::StaleView)
    );
    view = controller.build(&mut catalog).unwrap();
    ui.backend_mut().clear();
    ui.update(view.element());
    assert_eq!(
        controller
            .selection()
            .iter()
            .map(|id| id.raw())
            .collect::<Vec<_>>(),
        [1]
    );
    assert_eq!(controller.query().root, "");
    assert!(find(view.element(), "content:folder:new").is_some());
    assert!(find(view.element(), "content:folder:old").is_none());
    assert!(ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op, BackendOp::Remove { .. })));
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:folder:new",
        UiEvent::Activate,
    );
    view = controller.build(&mut catalog).unwrap();
    assert_eq!(row_ids(&view), [1]);
    assert!(find(view.element(), "content:asset:1")
        .unwrap()
        .class_names()
        .iter()
        .any(|class| class == "is-selected"));
    send(
        &mut controller,
        &mut catalog,
        &view,
        "content:clear-selection",
        UiEvent::Activate,
    );
    assert!(controller.selection().is_empty());
}
