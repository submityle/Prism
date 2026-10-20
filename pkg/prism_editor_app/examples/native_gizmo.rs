//! Run with `--features desktop --example native_gizmo -- --frames 3`.
#[cfg(feature = "desktop")]
#[path = "native_gizmo/support.rs"]
mod support;

#[cfg(feature = "desktop")]
#[expect(clippy::print_stdout, reason = "native smoke evidence")]
fn main() {
    use prism_editor_app::{run_desktop, DesktopConfig, EditorDesktopUi};
    let args: Vec<String> = std::env::args().collect();
    let frames = args
        .windows(2)
        .find(|a| a[0] == "--frames")
        .and_then(|a| a[1].parse().ok());
    let (controller, scene) = support::scene().expect("example host bindings");
    let mut ui = EditorDesktopUi::new(controller).expect("example UI");
    ui.mount_content_browser(prism_editor_content::ContentCatalog::new())
        .expect("content dock");
    // Explicit CLI schema: one supplied PNG path and its host publication
    // revision, rather than inferring source paths or project field names.
    if let Some(path) = args.windows(2).find(|a| a[0] == "--png") {
        mount_png(&mut ui, &path[1], &args);
    }
    if let Some(root) = args.windows(2).find(|a| a[0] == "--project") {
        let root = std::fs::canonicalize(&root[1]).expect("existing project root");
        let store = prism_editor_ui::CollectionStore::for_project(&root).expect("collection store");
        ui.mount_content_collections(store)
            .expect("collection load; refusing to overwrite failed load");
        ui.mount_content_git(
            prism_editor_content::git::GitProvider::new(Default::default()).expect("Git provider"),
            root,
        )
        .expect("Git binding");
    }
    let outcome = run_desktop(
        DesktopConfig {
            frame_limit: frames,
            time_limit: frames.map(|_| std::time::Duration::from_secs(15)),
            ..Default::default()
        },
        ui,
        scene,
    );
    println!(
        "native_gizmo_frames={} adapter={} backend={} error={:?} report={:?}",
        outcome.report.presented_frames,
        outcome.report.adapter_name,
        outcome.report.backend,
        outcome.error,
        outcome.report
    );
    assert!(outcome.error.is_none(), "native Gizmo smoke failed");
    if let Some(expected) = frames {
        assert!(
            outcome.report.presented_frames >= expected,
            "native Gizmo presentation deadline"
        );
    }
}

#[cfg(feature = "desktop")]
fn mount_png(ui: &mut prism_editor_app::EditorDesktopUi, path: &str, args: &[String]) {
    use prism_editor_content::{
        thumbnail::{ThumbnailKey, ThumbnailLimits, ThumbnailService},
        AssetId, AssetKind, AssetRecord, ContentCatalog,
    };
    use prism_editor_ui::{ContentBrowserAction, ContentLayout};
    struct Source {
        path: std::path::PathBuf,
        revision: u128,
    }
    impl prism_editor_app::ContentThumbnailSource for Source {
        fn key(&self, asset: &AssetRecord, size: [u32; 2]) -> Option<ThumbnailKey> {
            Some(ThumbnailKey {
                asset: asset.id,
                revision: self.revision,
                source: self.path.clone(),
                width: size[0],
                height: size[1],
            })
        }
    }
    let revision = args
        .windows(2)
        .find(|a| a[0] == "--revision")
        .expect("--png requires the authoritative --revision <u128>")[1]
        .parse()
        .expect("revision must be u128");
    // Startup only: no synchronous filesystem operations in the native loop.
    let path = std::fs::canonicalize(path).expect("existing PNG source");
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .expect("UTF-8 PNG filename");
    let mut catalog = ContentCatalog::new();
    catalog
        .insert(
            AssetRecord::new(
                AssetId::from_raw(1),
                name,
                AssetKind::new("image").expect("kind"),
            )
            .expect("asset"),
        )
        .expect("catalog insert");
    ui.mount_content_browser(catalog).expect("catalog mount");
    ui.content_action(ContentBrowserAction::Layout(ContentLayout::Grid {
        columns: 4,
    }))
    .expect("grid");
    ui.mount_content_thumbnails(
        ThumbnailService::new(ThumbnailLimits::default()).expect("decoder"),
        Box::new(Source { path, revision }),
    )
    .expect("thumbnail binding");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while ui.content_thumbnails_pending() {
        assert!(std::time::Instant::now() < deadline, "PNG startup deadline");
        std::thread::sleep(std::time::Duration::from_millis(2));
        ui.refresh().expect("thumbnail poll");
    }
    assert!(!ui.content_images().is_empty(), "PNG decode failed");
    ui.scroll_at(10.0, ui.viewport_rect().bottom() + 10.0, i32::MAX)
        .expect("show grid image");
}

#[cfg(not(feature = "desktop"))]
fn main() {}
