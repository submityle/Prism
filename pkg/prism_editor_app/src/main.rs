//! Prism editor application entry point for headless and native smoke runs.

#![allow(
    clippy::print_stdout,
    reason = "headless CLI emits its deterministic report"
)]

use prism_editor_app::{EditorAppConfig, EditorApplication};
#[cfg(feature = "desktop")]
use std::time::Duration;

#[cfg(feature = "desktop")]
use prism_editor_app::{
    run_desktop, DesktopConfig, DesktopScene, EditorController, EditorDesktopUi,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let frames = option_u64(&args, "--frames").unwrap_or(1);
    #[cfg(feature = "desktop")]
    if args.iter().any(|arg| arg == "--desktop") {
        run_native(&args, frames);
        return;
    }
    let mut app =
        EditorApplication::new(EditorAppConfig::default()).expect("valid editor configuration");
    let report = app
        .run_headless(frames, |_| {})
        .expect("requested frame count is within the configured bound");
    println!(
        "frames={} documents={} selection={} consoles={} viewports={} assets={} checksum={}",
        report.frames,
        report.snapshot.document_count,
        report.snapshot.selection_count,
        report.snapshot.console_count,
        report.snapshot.viewport_count,
        report.snapshot.asset_count,
        report.snapshot.checksum,
    );
}

fn option_u64(args: &[String], name: &str) -> Option<u64> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse::<u64>().ok())
}

#[cfg(feature = "desktop")]
fn run_native(args: &[String], frames: u64) {
    let width = option_u64(args, "--width").unwrap_or(1280);
    let height = option_u64(args, "--height").unwrap_or(800);
    let logical_size = [
        u32::try_from(width).expect("--width must fit u32"),
        u32::try_from(height).expect("--height must fit u32"),
    ];
    let mut ui = EditorDesktopUi::new(EditorController::new()).expect("valid empty editor UI");
    ui.mount_content_browser(prism_editor_content::ContentCatalog::new())
        .expect("valid empty content catalog");
    let scene = DesktopScene::new(Vec::new()).expect("valid empty desktop scene");
    let outcome = run_desktop(
        DesktopConfig {
            logical_size,
            frame_limit: Some(frames.max(1)),
            time_limit: Some(Duration::from_secs(5)),
            ..DesktopConfig::default()
        },
        ui,
        scene,
    );
    println!(
        "native_frames={} native_input_events={} adapter={} backend={} error={:?}",
        outcome.report.presented_frames,
        outcome.report.native_input_events,
        outcome.report.adapter_name,
        outcome.report.backend,
        outcome.error,
    );
}
