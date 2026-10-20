//! Real Loom grid/collection events and real PNG request-to-visual data.

use prism_editor_content::{
    thumbnail::{ThumbnailError, ThumbnailKey, ThumbnailLimits, ThumbnailService},
    AssetId, AssetKind, AssetQuery, AssetRecord, ContentCatalog, PageRequest, VcsRecord, VcsState,
};
use prism_editor_ui::{
    ActionValue, BoundContentBrowser, ContentBrowserAction as Action,
    ContentBrowserController as Controller, ContentBrowserError as Error, ContentLayout,
    ContentThumbnailState, ContentThumbnails, SelectionMode, UiEvent, MAX_CONTENT_COLLECTIONS,
};
use prism_ui::{
    layout::{AvailableSpace, Size},
    BackendId, BackendOp, Element, ElementKind, Key, RecordingBackend, Ui,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

fn id(value: u128) -> AssetId {
    AssetId::from_raw(value)
}
fn asset(value: u128, path: &str) -> AssetRecord {
    AssetRecord::new(id(value), path, AssetKind::new("texture").unwrap()).unwrap()
}
fn nodes<'a>(element: &'a Element, out: &mut Vec<&'a Element>) {
    out.push(element);
    for child in element.child_elements() {
        nodes(child, out);
    }
}
fn find<'a>(element: &'a Element, key: &str) -> Option<&'a Element> {
    let mut all = Vec::new();
    nodes(element, &mut all);
    all.into_iter()
        .find(|node| matches!(node.explicit_key(), Some(Key::Str(value)) if value == key))
}
fn send(
    controller: &mut Controller,
    catalog: &mut ContentCatalog,
    ui: &mut Ui<RecordingBackend>,
    view: BoundContentBrowser,
    key: &str,
    event: UiEvent,
) -> BoundContentBrowser {
    let Some(Key::Str(target)) = find(view.element(), key)
        .expect("real event target")
        .explicit_key()
    else {
        panic!("missing key");
    };
    controller
        .handle_event(catalog, &view, target, &event)
        .unwrap();
    let next = controller.build(catalog).unwrap();
    ui.update(next.element());
    for (key, _) in next.bindings() {
        assert!(find(next.element(), key).is_some());
    }
    next
}
fn commit(value: &str) -> UiEvent {
    UiEvent::Commit {
        value: ActionValue::text(value).unwrap(),
    }
}
fn select(mode: SelectionMode) -> UiEvent {
    UiEvent::Select { mode }
}
fn ids(view: &BoundContentBrowser) -> Vec<u128> {
    let mut all = Vec::new();
    nodes(find(view.element(), "content:list").unwrap(), &mut all);
    all.into_iter()
        .filter_map(|node| match node.explicit_key() {
            Some(Key::Str(key)) => key
                .strip_prefix("content:asset:")
                .map(|value| value.parse().unwrap()),
            _ => None,
        })
        .collect()
}
fn mount(view: &BoundContentBrowser) -> Ui<RecordingBackend> {
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(view.element());
    ui
}

#[test]
fn collection_events_select_add_filter_page_rename_remove_and_delete() {
    let mut catalog = ContentCatalog::new();
    for i in 1..=5 {
        catalog.insert(asset(i, &format!("{i}.png"))).unwrap();
    }
    let mut controller = Controller::default();
    controller.apply(&mut catalog, Action::PageSize(3)).unwrap();
    let view = controller.build(&mut catalog).unwrap();
    let mut ui = mount(&view);
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:layout:grid",
        UiEvent::Activate,
    );
    assert_eq!(controller.layout(), ContentLayout::Grid { columns: 4 });
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:asset:1",
        select(SelectionMode::Replace),
    );
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:asset:3",
        select(SelectionMode::Extend),
    );
    assert_eq!(controller.selection().len(), 3);
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:collection:create",
        commit("英雄"),
    );
    assert!(ids(&view).is_empty());
    assert_eq!(controller.active_collection(), Some(1));
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:collection:add:1",
        UiEvent::Activate,
    );
    assert_eq!(ids(&view), [1, 2, 3]);
    assert_eq!(controller.collections()[0].members().len(), 3);
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:collection:folders",
        UiEvent::Activate,
    );
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:next",
        UiEvent::Activate,
    );
    assert_eq!(ids(&view), [4, 5]);
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:asset:5",
        select(SelectionMode::Replace),
    );
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:asset:4",
        select(SelectionMode::Add),
    );
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:collection:add:1",
        UiEvent::Activate,
    );
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:collection:open:1",
        UiEvent::Activate,
    );
    assert_eq!(ids(&view), [1, 2, 3]);
    let stale = view.clone();
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:next",
        UiEvent::Activate,
    );
    assert_eq!(ids(&view), [4, 5]);
    assert_eq!(controller.page_offset(), 3);
    assert_eq!(
        controller.handle_event(
            &mut catalog,
            &stale,
            "content:collection:delete:1",
            &UiEvent::Activate
        ),
        Err(Error::StaleView)
    );
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:previous",
        UiEvent::Activate,
    );
    assert_eq!(ids(&view), [1, 2, 3]);
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:collection:rename:1",
        commit("精选"),
    );
    assert_eq!(controller.collections()[0].name(), "精选");
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:search",
        commit("5.png"),
    );
    assert_eq!(ids(&view), [5]);
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:favorite:5",
        UiEvent::Activate,
    );
    assert!(catalog.get(id(5)).unwrap().favorite);
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:collection:remove:1",
        UiEvent::Activate,
    );
    assert!(ids(&view).is_empty());
    assert_eq!(
        controller.collections()[0].members(),
        &BTreeSet::from([id(1), id(2), id(3)])
    );
    let view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:collection:delete:1",
        UiEvent::Activate,
    );
    assert!(controller.collections().is_empty());
    assert_eq!(catalog.len(), 5);
    assert!(view.images().is_empty());
    controller
        .apply(&mut catalog, Action::CreateCollection("new".into()))
        .unwrap();
    assert_eq!(controller.collections()[0].id(), 2);
}

#[test]
fn collection_filters_match_catalog_policy_and_preserve_members_across_publication() {
    let mut catalog = ContentCatalog::new();
    let mut a = asset(1, "a.png");
    a.favorite = true;
    a.pinned = true;
    a.add_tag("Hero").unwrap();
    catalog.insert(a).unwrap();
    catalog.insert(asset(2, "b.png")).unwrap();
    catalog
        .set_vcs_status(
            "a.png",
            VcsRecord {
                state: VcsState::Modified,
                ..Default::default()
            },
        )
        .unwrap();
    let mut controller = Controller::default();
    controller.build(&mut catalog).unwrap();
    for (asset, mode) in [(id(1), SelectionMode::Replace), (id(2), SelectionMode::Add)] {
        controller
            .apply(&mut catalog, Action::Select { asset, mode })
            .unwrap();
    }
    controller
        .apply(&mut catalog, Action::CreateCollection("all".into()))
        .unwrap();
    controller
        .apply(&mut catalog, Action::AddSelectionToCollection(1))
        .unwrap();
    let filters = [
        Action::Search("HERO".into()),
        Action::Search("".into()),
        Action::Tag(Some(" HERO ".into())),
        Action::Tag(None),
        Action::Vcs(Some(VcsState::Unknown)),
        Action::Vcs(None),
        Action::FavoritesOnly(true),
        Action::FavoritesOnly(false),
        Action::PinnedOnly(true),
        Action::PinnedOnly(false),
        Action::Kind(Some(AssetKind::new("mesh").unwrap())),
        Action::Kind(None),
    ];
    for action in filters {
        controller.apply(&mut catalog, action).unwrap();
        let view = controller.build(&mut catalog).unwrap();
        let q = controller.query();
        let query = AssetQuery {
            root: None,
            text: q.text.clone(),
            tag: q.tag.clone(),
            kind: q.kind.clone(),
            favorites_only: q.favorites_only,
            pinned_only: q.pinned_only,
            vcs: q.vcs,
        };
        let expected = catalog
            .query_page(&query, PageRequest::new(64), None)
            .unwrap();
        assert_eq!(
            ids(&view),
            expected.items.iter().map(|id| id.raw()).collect::<Vec<_>>()
        );
    }
    let mut record = catalog.get(id(1)).unwrap().clone();
    record.path = "nested/renamed.png".into();
    catalog.update(record).unwrap();
    let view = controller.build(&mut catalog).unwrap();
    assert_eq!(ids(&view), [2, 1]);
    catalog.remove(id(1)).unwrap();
    let view = controller.build(&mut catalog).unwrap();
    assert_eq!(ids(&view), [2]);
    assert!(controller.collections()[0].members().contains(&id(1)));
    assert!(!controller.selection().contains(&id(1)));
}

#[test]
fn grid_has_real_bounded_layout_and_unchanged_reconciliation_is_empty() {
    let mut catalog = ContentCatalog::new();
    for i in 1..=5 {
        catalog.insert(asset(i, &format!("{i}.png"))).unwrap();
    }
    let mut controller = Controller::default();
    controller.apply(&mut catalog, Action::PageSize(5)).unwrap();
    controller
        .apply(
            &mut catalog,
            Action::Layout(ContentLayout::Grid { columns: 2 }),
        )
        .unwrap();
    let view = controller.build(&mut catalog).unwrap();
    let mut ui = mount(&view);
    let mut all = Vec::new();
    nodes(view.element(), &mut all);
    let created: Vec<_> = ui
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
    assert_eq!(all.len(), created.len());
    let targets: BTreeMap<_, _> = all
        .iter()
        .zip(created)
        .filter_map(|(node, id)| {
            if let Some(Key::Str(key)) = node.explicit_key() {
                Some((key.clone(), id))
            } else {
                None
            }
        })
        .collect();
    ui.backend_mut().clear();
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(1000.0),
        AvailableSpace::Definite(1800.0),
    ));
    let layouts: BTreeMap<BackendId, _> = ui
        .backend()
        .ops()
        .iter()
        .filter_map(|op| {
            if let BackendOp::SetLayout { id, location, size } = op {
                Some((*id, (*location, *size)))
            } else {
                None
            }
        })
        .collect();
    for key in ["content:asset:1", "content:asset:2", "content:asset:3"] {
        assert!(layouts[&targets[key]].1.width >= 159.0);
    }
    assert!(layouts[&targets["content:asset:2"]].0.x > layouts[&targets["content:asset:1"]].0.x);
    assert!(
        layouts[&targets["content:grid-row:1"]].0.y > layouts[&targets["content:grid-row:0"]].0.y
    );
    assert_eq!(
        find(view.element(), "content:list")
            .unwrap()
            .child_elements()
            .len(),
        3
    );
    ui.backend_mut().clear();
    let next = controller.build(&mut catalog).unwrap();
    ui.update(next.element());
    assert!(ui.backend().is_empty());
    for columns in [0, 9, usize::MAX] {
        assert_eq!(
            controller.apply(
                &mut catalog,
                Action::Layout(ContentLayout::Grid { columns })
            ),
            Err(Error::InvalidGridColumns)
        );
    }
    assert_eq!(controller.layout(), ContentLayout::Grid { columns: 2 });
}

#[test]
fn collection_limits_reject_without_overwriting_and_membership_work_is_bounded() {
    let mut catalog = ContentCatalog::new();
    let mut controller = Controller::default();
    assert_eq!(
        controller.apply(&mut catalog, Action::CreateCollection(" ".into())),
        Err(Error::InvalidCollection)
    );
    for _ in 0..MAX_CONTENT_COLLECTIONS {
        controller
            .apply(&mut catalog, Action::CreateCollection("set".into()))
            .unwrap();
    }
    let before = controller.collections().to_vec();
    assert_eq!(
        controller.apply(&mut catalog, Action::CreateCollection("extra".into())),
        Err(Error::CollectionLimit)
    );
    assert_eq!(controller.collections(), before);
    assert_eq!(
        controller.apply(
            &mut catalog,
            Action::RenameCollection {
                id: 1,
                name: " ".into()
            }
        ),
        Err(Error::InvalidCollection)
    );
    assert_eq!(
        controller.apply(&mut catalog, Action::OpenCollection(Some(0))),
        Err(Error::InvalidCollection)
    );
    assert_eq!(controller.collections(), before);
}

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "prism-ui-thumbnail-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        let this = Self(path);
        this.write();
        this
    }
    fn write(&self) {
        let mut encoder = png::Encoder::new(
            std::fs::File::create(self.0.join("image.png")).unwrap(),
            2,
            1,
        );
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&[255, 0, 0, 255, 0, 255, 0, 255])
            .unwrap();
    }
    fn key(&self, asset: u128, revision: u128) -> ThumbnailKey {
        ThumbnailKey {
            asset: id(asset),
            revision,
            source: self.0.join("image.png"),
            width: 128,
            height: 128,
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn finish(thumbnails: &mut ContentThumbnails) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while thumbnails.needs_poll() {
        assert!(Instant::now() < deadline);
        thumbnails.poll();
        std::thread::yield_now();
    }
}

#[test]
fn real_png_pixels_have_actual_loom_visual_layout_and_cache_revision_identity() {
    let fixture = Fixture::new();
    let service = ThumbnailService::new(ThumbnailLimits::default()).unwrap();
    let mut thumbnails = ContentThumbnails::new(service.clone());
    let mut catalog = ContentCatalog::new();
    catalog.insert(asset(1, "image.png")).unwrap();
    let mut controller = Controller::default();
    controller
        .apply(
            &mut catalog,
            Action::Layout(ContentLayout::Grid { columns: 2 }),
        )
        .unwrap();
    let view = controller.build(&mut catalog).unwrap();
    let mut ui = mount(&view);
    assert!(view.images().is_empty());
    thumbnails
        .sync_visible(&controller, [fixture.key(1, 1)])
        .unwrap();
    finish(&mut thumbnails);
    let ready = controller
        .build_with_thumbnails(&mut catalog, &thumbnails)
        .unwrap();
    assert_eq!(ready.images().len(), 1);
    let visual = &ready.images()[0];
    assert_eq!(visual.visual_key, "content:image:1");
    assert_eq!((visual.image.width(), visual.image.height()), (2, 1));
    assert_eq!(visual.image.rgba8(), &[255, 0, 0, 255, 0, 255, 0, 255]);
    assert_eq!(visual.request, fixture.key(1, 1));
    assert_eq!(
        find(ready.element(), &visual.visual_key).unwrap().kind(),
        &ElementKind::Custom("prism-editor-content-image".into())
    );
    ui.backend_mut().clear();
    ui.update(ready.element());
    let visual_id = ui
        .backend()
        .ops()
        .iter()
        .find_map(|op| {
            if let BackendOp::Create {
                id,
                kind: ElementKind::Custom(name),
                ..
            } = op
            {
                (name == "prism-editor-content-image").then_some(*id)
            } else {
                None
            }
        })
        .unwrap();
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(900.0),
        AvailableSpace::Definite(1400.0),
    ));
    assert!(ui.backend().ops().iter().any(|op| matches!(op,BackendOp::SetLayout {id,size,..} if *id == visual_id && size.width == 128.0 && size.height == 64.0)));
    let pixels = visual.image.clone();
    assert_eq!(thumbnails.retained_bytes(), 8);
    std::fs::remove_file(fixture.0.join("image.png")).unwrap();
    thumbnails.clear();
    thumbnails
        .sync_visible(&controller, [fixture.key(1, 1)])
        .unwrap();
    finish(&mut thumbnails);
    assert!(Arc::ptr_eq(&pixels, thumbnails.ready(id(1)).unwrap().1));
    thumbnails
        .sync_visible(&controller, [fixture.key(1, 2)])
        .unwrap();
    finish(&mut thumbnails);
    assert!(matches!(
        thumbnails.state(id(1)),
        Some(ContentThumbnailState::Failed(ThumbnailError::Io(_)))
    ));
    let failed = controller
        .build_with_thumbnails(&mut catalog, &thumbnails)
        .unwrap();
    assert!(failed.images().is_empty());
    ui.backend_mut().clear();
    ui.update(failed.element());
    assert!(ui
        .backend()
        .ops()
        .iter()
        .any(|op| matches!(op,BackendOp::Remove {id} if *id == visual_id)));
    fixture.write();
    assert!(thumbnails.retry(id(1)));
    finish(&mut thumbnails);
    assert!(thumbnails.ready(id(1)).is_some());
    assert_eq!(thumbnails.ready(id(1)).unwrap().0.revision, 2);
    thumbnails.clear();
    service.clear().unwrap();
    assert_eq!(thumbnails.retained_bytes(), 0);
}

#[test]
fn visible_thumbnail_keys_are_atomic_bounded_and_removed_pages_drop_pixels() {
    let fixture = Fixture::new();
    let service = ThumbnailService::new(ThumbnailLimits {
        max_in_flight: 1,
        ..Default::default()
    })
    .unwrap();
    let mut thumbnails = ContentThumbnails::new(service.clone());
    let mut catalog = ContentCatalog::new();
    catalog.insert(asset(1, "a.png")).unwrap();
    catalog.insert(asset(2, "b.png")).unwrap();
    let mut controller = Controller::default();
    controller.build(&mut catalog).unwrap();
    thumbnails
        .sync_visible(&controller, [fixture.key(1, 1), fixture.key(2, 1)])
        .unwrap();
    finish(&mut thumbnails);
    assert_eq!(thumbnails.retained_bytes(), 16);
    assert_eq!(
        thumbnails.sync_visible(&controller, [fixture.key(1, 1), fixture.key(1, 2)]),
        Err(ThumbnailError::InvalidConfiguration)
    );
    assert_eq!(thumbnails.retained_bytes(), 16);
    assert_eq!(
        thumbnails.sync_visible(&controller, [fixture.key(99, 1)]),
        Err(ThumbnailError::InvalidConfiguration)
    );
    controller.apply(&mut catalog, Action::PageSize(1)).unwrap();
    let view = controller.build(&mut catalog).unwrap();
    let mut ui = mount(&view);
    let _view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:next",
        UiEvent::Activate,
    );
    thumbnails
        .sync_visible(&controller, [fixture.key(2, 1)])
        .unwrap();
    assert!(thumbnails.ready(id(1)).is_none());
    assert_eq!(thumbnails.retained_bytes(), 8);
    let mut record = catalog.get(id(2)).unwrap().clone();
    record.favorite = true;
    catalog.update(record).unwrap();
    let view = controller
        .build_with_thumbnails(&mut catalog, &thumbnails)
        .unwrap();
    assert!(view.images().is_empty());
    assert_eq!(
        thumbnails.sync_visible(&controller, [fixture.key(2, 1)]),
        Err(ThumbnailError::InvalidConfiguration)
    );
    // A catalog publication resets pagination. Navigate through the rebuilt
    // tree before supplying keys for the second page again.
    ui.update(view.element());
    let _view = send(
        &mut controller,
        &mut catalog,
        &mut ui,
        view,
        "content:next",
        UiEvent::Activate,
    );
    thumbnails
        .sync_visible(&controller, [fixture.key(2, 1)])
        .unwrap();
    assert_eq!(
        controller
            .build_with_thumbnails(&mut catalog, &thumbnails)
            .unwrap()
            .images()
            .len(),
        1
    );
    thumbnails.sync_visible(&controller, []).unwrap();
    assert_eq!(thumbnails.retained_bytes(), 0);
    assert!(!thumbnails.needs_poll());
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn decoded_budget_rejects_extra_pixels_and_offscreen_workers_release_capacity() {
    let fixture = Fixture::new();
    let service = ThumbnailService::new(ThumbnailLimits {
        max_in_flight: 1,
        ..Default::default()
    })
    .unwrap();
    let mut thumbnails = ContentThumbnails::with_budget(service.clone(), 8).unwrap();
    let mut catalog = ContentCatalog::new();
    for i in 1..=2 {
        catalog.insert(asset(i, &format!("{i}.png"))).unwrap();
    }
    let mut controller = Controller::default();
    controller.build(&mut catalog).unwrap();
    thumbnails
        .sync_visible(&controller, [fixture.key(1, 1), fixture.key(2, 1)])
        .unwrap();
    assert!(matches!(
        thumbnails.state(id(1)),
        Some(ContentThumbnailState::Loading)
    ));
    assert!(matches!(
        thumbnails.state(id(2)),
        Some(ContentThumbnailState::Queued)
    ));
    finish(&mut thumbnails);
    assert_eq!(thumbnails.retained_bytes(), 8);
    assert!(matches!(
        thumbnails.state(id(2)),
        Some(ContentThumbnailState::Failed(ThumbnailError::Limit(_)))
    ));
    thumbnails.clear();
    thumbnails
        .sync_visible(&controller, [fixture.key(1, 2)])
        .unwrap();
    thumbnails.sync_visible(&controller, []).unwrap();
    assert!(!thumbnails.needs_poll());
    let deadline = Instant::now() + Duration::from_secs(10);
    while service.in_flight() != 0 {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(thumbnails.retained_bytes(), 0);
}

// Explicit opt-in; a GPU request failure is a failure, never a silent pass.
#[cfg(feature = "gpu-tests")]
#[test]
fn real_png_to_loom_layout_to_gpu_pixels_clip_opacity_and_texture_release() {
    use futures_lite::future::block_on;
    use prism_editor_viewport::image::{ImageLimits, ImageQuad, ImageRenderer};
    let fixture = Fixture::new();
    let service = ThumbnailService::new(ThumbnailLimits::default()).unwrap();
    let mut thumbnails = ContentThumbnails::new(service);
    let mut catalog = ContentCatalog::new();
    catalog.insert(asset(1, "image.png")).unwrap();
    let mut controller = Controller::default();
    controller
        .apply(
            &mut catalog,
            Action::Layout(ContentLayout::Grid { columns: 2 }),
        )
        .unwrap();
    controller.build(&mut catalog).unwrap();
    thumbnails
        .sync_visible(&controller, [fixture.key(1, 1)])
        .unwrap();
    finish(&mut thumbnails);
    let view = controller
        .build_with_thumbnails(&mut catalog, &thumbnails)
        .unwrap();
    let visual = &view.images()[0];
    let mut ui = mount(&view);
    let mut all = Vec::new();
    nodes(view.element(), &mut all);
    let created: Vec<_> = ui
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
    let anchor = all
        .iter()
        .zip(created)
        .find_map(|(node, id)| {
            matches!(node.explicit_key(),Some(Key::Str(key)) if key==&visual.visual_key)
                .then_some(id)
        })
        .unwrap();
    let parents: BTreeMap<_, _> = ui
        .backend()
        .ops()
        .iter()
        .filter_map(|op| {
            if let BackendOp::Create { id, parent, .. } = op {
                Some((*id, *parent))
            } else {
                None
            }
        })
        .collect();
    const WIDTH: u32 = 1024;
    const HEIGHT: u32 = 1600;
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(WIDTH as f32),
        AvailableSpace::Definite(HEIGHT as f32),
    ));
    let layouts: BTreeMap<_, _> = ui
        .backend()
        .ops()
        .iter()
        .filter_map(|op| {
            if let BackendOp::SetLayout { id, location, size } = op {
                Some((*id, (*location, *size)))
            } else {
                None
            }
        })
        .collect();
    let size = layouts[&anchor].1;
    let mut rect = [0.0, 0.0, size.width, size.height];
    let mut current = Some(anchor);
    while let Some(node) = current {
        let (point, _) = layouts[&node];
        rect[0] += point.x;
        rect[1] += point.y;
        current = parents[&node];
    }
    assert_eq!([rect[2], rect[3]], [128.0, 64.0]);
    assert!(
        rect[0] >= 0.0
            && rect[1] >= 0.0
            && rect[0] + rect[2] < WIDTH as f32
            && rect[1] + rect[3] < HEIGHT as f32
    );
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
        backend_options: wgpu::BackendOptions::default(),
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("required physical GPU adapter");
    let info = adapter.get_info();
    assert_ne!(info.device_type, wgpu::DeviceType::Cpu);
    eprintln!(
        "EXERCISED content grid PNG -> actual Loom key/layout -> GPU: {info:?}; rect={rect:?}"
    );
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut renderer = ImageRenderer::new(&device, &queue, format, ImageLimits::default()).unwrap();
    let handle = renderer
        .upload(
            visual.image.width(),
            visual.image.height(),
            visual.image.rgba8(),
        )
        .unwrap();
    assert_eq!(renderer.usage(), (1, 8));
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("content UI pixel proof"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());
    let render = |renderer: &ImageRenderer| {
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test-only content pixels"),
            size: u64::from(WIDTH) * u64::from(HEIGHT) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        }
        renderer.encode(&mut encoder, &target).unwrap();
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(WIDTH * 4),
                    rows_per_image: Some(HEIGHT),
                },
            },
            target.size(),
        );
        queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap();
        let bytes = readback.slice(..).get_mapped_range().unwrap().to_vec();
        readback.unmap();
        bytes
    };
    let full = [0.0, 0.0, WIDTH as f32, HEIGHT as f32];
    renderer
        .prepare(
            [WIDTH, HEIGHT],
            &[ImageQuad {
                image: handle.clone(),
                rect,
                clip: full,
                opacity: 1.0,
            }],
        )
        .unwrap();
    let pixels = render(&renderer);
    let sample = |pixels: &[u8], dx: f32| {
        let x = (rect[0] + dx) as usize;
        let y = (rect[1] + 32.0) as usize;
        let offset = (y * WIDTH as usize + x) * 4;
        pixels[offset..offset + 4].to_vec()
    };
    assert_eq!(sample(&pixels, 16.0), [255, 0, 0, 255]);
    assert_eq!(sample(&pixels, 112.0), [0, 255, 0, 255]);
    assert_eq!(&pixels[0..4], &[0, 0, 0, 255]);
    let uploads = renderer.geometry_uploads();
    renderer
        .prepare(
            [WIDTH, HEIGHT],
            &[ImageQuad {
                image: handle.clone(),
                rect,
                clip: full,
                opacity: 1.0,
            }],
        )
        .unwrap();
    assert_eq!(renderer.geometry_uploads(), uploads);
    renderer
        .prepare(
            [WIDTH, HEIGHT],
            &[ImageQuad {
                image: handle.clone(),
                rect,
                clip: [rect[0] + 64.0, rect[1], 64.0, 64.0],
                opacity: 0.5,
            }],
        )
        .unwrap();
    let clipped = render(&renderer);
    assert_eq!(sample(&clipped, 16.0), [0, 0, 0, 255]);
    let green = sample(&clipped, 112.0);
    assert!(green[1].abs_diff(188) <= 2, "{green:?}");
    assert_eq!([green[0], green[2], green[3]], [0, 0, 255]);
    renderer.remove(&handle).unwrap();
    assert_eq!(renderer.usage(), (0, 0));
}
