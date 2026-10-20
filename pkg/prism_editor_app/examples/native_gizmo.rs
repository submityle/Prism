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
    let outcome = run_desktop(
        DesktopConfig {
            frame_limit: frames,
            time_limit: frames.map(|_| std::time::Duration::from_secs(15)),
            ..Default::default()
        },
        EditorDesktopUi::new(controller).expect("example UI"),
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

#[cfg(not(feature = "desktop"))]
fn main() {}
