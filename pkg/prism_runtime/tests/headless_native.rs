//! Integration coverage for the Gate A native headless launcher.

use prism_runtime::{run_headless, HeadlessRuntimeConfig};

#[test]
fn gate_a_submits_the_requested_native_frames() {
    let report = run_headless(HeadlessRuntimeConfig::new(2, 16, 8))
        .expect("the null backend should assemble and run");

    assert_eq!(report.frames_completed, 2);
    assert_eq!(report.render_record.frame_index, 2);
    assert_eq!(report.render_record.viewport.width, 16);
    assert_eq!(report.render_record.viewport.height, 8);
    assert_eq!(report.render_stats.frames_submitted, 2);
    assert_eq!(report.render_stats.frames_failed, 0);
    assert_eq!(report.render_stats.frames_skipped, 0);
    assert!(report.is_successful());
}
