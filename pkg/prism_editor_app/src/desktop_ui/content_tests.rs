use super::*;
use prism_editor_content::{AssetId, AssetKind, AssetRecord};
use prism_reflect::{Reflect, TypeRegistry};

#[derive(Reflect)]
struct Model {
    count: u32,
}

fn record(id: u128, path: &str) -> AssetRecord {
    AssetRecord::new(AssetId::from_raw(id), path, AssetKind::new("mesh").unwrap()).unwrap()
}

fn mounted(paths: &[&str]) -> EditorDesktopUi {
    let mut controller = EditorController::new();
    let mut registry = TypeRegistry::new();
    registry.register::<Model>();
    controller
        .register_object(
            SelectionItem::entity(1),
            Box::new(Model { count: 7 }),
            registry,
        )
        .unwrap();
    controller.select(SelectionItem::entity(1)).unwrap();
    let mut ui = EditorDesktopUi::new(controller).unwrap();
    let mut catalog = ContentCatalog::new();
    for (i, path) in paths.iter().enumerate() {
        catalog.insert(record(i as u128 + 1, path)).unwrap();
    }
    ui.mount_content_browser(catalog).unwrap();
    ui
}

fn hit(ui: &EditorDesktopUi, target: &str) -> Rect {
    ui.hits
        .iter()
        .find(|h| h.target == target)
        .unwrap_or_else(|| panic!("missing {target}"))
        .rect
}

fn click(ui: &mut EditorDesktopUi, target: &str, mode: SelectionMode) {
    let rect = hit(ui, target);
    assert!(
        rect.size.width > 0.0 && rect.size.height > 0.0,
        "clipped {target}: {rect:?}"
    );
    // Row labels precede nested favorite/pin buttons.
    ui.click(rect.left() + 3.0, rect.top() + rect.size.height * 0.5, mode)
        .unwrap();
}

fn text_visible(ui: &EditorDesktopUi, text: &str) -> bool {
    ui.draw_list().commands().iter().any(|cmd| {
        matches!(cmd,
        DrawCommand::Glyph(g) if ui.draw_list().text(g.text).contains(text)
        && g.rect.size.width > 0.0 && g.rect.size.height > 0.0)
    })
}

fn commit(ui: &mut EditorDesktopUi, target: &str, value: &str) {
    click(ui, target, SelectionMode::Replace);
    if value.is_empty() {
        ui.key(DesktopKey::Backspace).unwrap();
    } else {
        ui.text(value).unwrap();
    }
    ui.key(DesktopKey::Enter).unwrap();
}

#[test]
fn mounted_rows_nested_buttons_and_multiselect_are_independent_of_document() {
    let mut ui = mounted(&["alpha.mesh", "beta.mesh", "gamma.mesh"]);
    let revision = ui.controller().revision();
    let model_revision = ui.controller().model_revision();
    assert!(text_visible(&ui, "alpha.mesh"));
    for target in [
        "content:search",
        "content:kind",
        "content:tag",
        "content:vcs",
        "content:favorites-only",
        "content:pinned-only",
        "content:asset:1",
        "content:favorite:1",
        "content:pin:1",
    ] {
        let rect = hit(&ui, target);
        assert!(
            rect.size.width > 0.0 && rect.size.height > 0.0,
            "{target}: {rect:?}"
        );
        assert!(rect.top() >= ui.viewport.bottom());
    }
    click(&mut ui, "content:asset:1", SelectionMode::Replace);
    click(&mut ui, "content:asset:3", SelectionMode::Extend);
    assert_eq!(ui.content_browser().unwrap().selection().len(), 3);
    click(&mut ui, "content:favorite:1", SelectionMode::Replace);
    click(&mut ui, "content:pin:1", SelectionMode::Replace);
    let asset = ui
        .content_catalog()
        .unwrap()
        .get(AssetId::from_raw(1))
        .unwrap();
    assert!(asset.favorite && asset.pinned);
    assert_eq!(
        ui.content_browser().unwrap().selection().len(),
        3,
        "nested buttons must not select rows"
    );
    assert!(text_visible(&ui, "Favorite: on"));
    click(&mut ui, "content:clear-selection", SelectionMode::Replace);
    assert!(ui.content_browser().unwrap().selection().is_empty());
    assert_eq!(ui.controller().revision(), revision);
    assert_eq!(ui.controller().model_revision(), model_revision);
    assert_eq!(
        ui.controller().selection().primary(),
        Some(&SelectionItem::entity(1))
    );
    assert_eq!(ui.controller().document_history_lengths(), (0, 0));
    assert!(!ui.controller().is_dirty());
    assert!(!ui.refresh().unwrap());
}

#[test]
fn browser_text_ime_filters_and_failed_input_keep_their_own_draft() {
    let mut ui = mounted(&["船.mesh", "beta.mesh"]);
    click(&mut ui, "content:search", SelectionMode::Replace);
    assert!(ui.text_input_rect().is_some());
    ui.preedit("船").unwrap();
    assert!(text_visible(&ui, "[船]"));
    assert!(!ui.key(DesktopKey::Enter).unwrap());
    assert!(!ui.key(DesktopKey::Tab).unwrap());
    ui.text("船").unwrap();
    assert!(!ui.key(DesktopKey::Undo).unwrap());
    assert_eq!(ui.focus.as_ref().unwrap().buffer, "船");
    ui.key(DesktopKey::Enter).unwrap();
    assert_eq!(
        ui.content_browser().unwrap().query().text.as_deref(),
        Some("船")
    );
    assert!(!ui.hits.iter().any(|h| h.target == "content:asset:2"));
    commit(&mut ui, "content:search", "");
    commit(&mut ui, "content:kind", "mesh");
    commit(&mut ui, "content:tag", "missing");
    assert!(text_visible(&ui, "No matching content"));
    commit(&mut ui, "content:tag", "");
    click(&mut ui, "content:vcs", SelectionMode::Replace);
    ui.text("not-a-vcs-state").unwrap();
    let draft = ui.focus.clone();
    assert!(ui.key(DesktopKey::Enter).is_err());
    assert_eq!(ui.focus, draft);
    assert_eq!(ui.content_browser().unwrap().query().vcs, None);
    ui.key(DesktopKey::Escape).unwrap();
    commit(&mut ui, "content:vcs", "unknown");
    assert!(ui.error_message().is_none());
    assert!(ui.hits.iter().any(|h| h.target == "content:asset:2"));
    assert_eq!(ui.controller().document_history_lengths(), (0, 0));
}

#[test]
fn catalog_generation_refreshes_rows_rejects_old_clicks_and_preserves_inspector_draft() {
    let mut ui = mounted(&["alpha.mesh", "beta.mesh"]);
    let revision = ui.controller().revision();
    click(&mut ui, "content:asset:1", SelectionMode::Replace);
    let old = hit(&ui, "content:asset:1");
    ui.content_catalog_mut()
        .unwrap()
        .remove(AssetId::from_raw(1))
        .unwrap();
    ui.click(
        old.left() + 3.0,
        old.top() + old.size.height * 0.5,
        SelectionMode::Replace,
    )
    .unwrap();
    assert!(
        ui.content_browser().unwrap().selection().is_empty(),
        "must not replay click onto replacement row"
    );
    assert!(!ui.hits.iter().any(|h| h.target == "content:asset:1"));
    click(&mut ui, "field:.count", SelectionMode::Replace);
    ui.text("42").unwrap();
    let draft = ui.focus.clone();
    let mut renamed = ui
        .content_catalog()
        .unwrap()
        .get(AssetId::from_raw(2))
        .unwrap()
        .clone();
    renamed.path = "renamed.mesh".into();
    ui.content_catalog_mut().unwrap().update(renamed).unwrap();
    assert!(ui.refresh().unwrap());
    assert_eq!(ui.focus, draft);
    assert!(text_visible(&ui, "renamed.mesh"));
    assert_eq!(ui.controller().revision(), revision);
    assert!(!ui.refresh().unwrap());
    ui.key(DesktopKey::Escape).unwrap();
    click(&mut ui, "content:search", SelectionMode::Replace);
    ui.text("outdated draft").unwrap();
    ui.content_catalog_mut()
        .unwrap()
        .insert(record(3, "new.mesh"))
        .unwrap();
    assert!(ui.refresh().unwrap());
    assert!(ui.focus.is_none());
    assert_eq!(ui.content_browser().unwrap().query().text, None);
}

#[test]
fn native_dock_pages_are_reachable_and_directory_navigation_relayouts() {
    let paths: Vec<_> = (0..20).map(|n| format!("folder/item{n:02}.mesh")).collect();
    let mut ui = mounted(&paths.iter().map(String::as_str).collect::<Vec<_>>());
    for (width, height) in [(1280.0, 800.0), (960.0, 640.0)] {
        ui.resize(width, height).unwrap();
        click(&mut ui, "content:folder:folder", SelectionMode::Replace);
        assert_eq!(ui.content_browser().unwrap().query().root, "folder");
        assert!(
            ui.hits
                .iter()
                .filter(|h| h.target.starts_with("content:asset:"))
                .count()
                <= 6
        );
        let next = hit(&ui, "content:next");
        assert!(
            next.bottom() <= height - STATUS_HEIGHT,
            "unreachable page footer {next:?}"
        );
        click(&mut ui, "content:next", SelectionMode::Replace);
        assert!(ui.content_browser().unwrap().page_offset() > 0);
        click(&mut ui, "content:previous", SelectionMode::Replace);
        assert_eq!(ui.content_browser().unwrap().page_offset(), 0);
        click(&mut ui, "content:up", SelectionMode::Replace);
        assert!(ui.content_browser().unwrap().query().root.is_empty());
        assert!(!ui.refresh().unwrap());
    }
}
