//! Headless Prism editor application entry point for CI and smoke tests.

#![allow(
    clippy::print_stdout,
    reason = "headless CLI emits its deterministic report"
)]

use prism_editor_app::{EditorAppConfig, EditorApplication};

fn main() {
    let frames = std::env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .windows(2)
        .find(|pair| pair[0] == "--frames")
        .and_then(|pair| pair[1].parse::<u64>().ok())
        .unwrap_or(1);
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
