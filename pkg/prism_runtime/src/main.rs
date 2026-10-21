//! Command-line entry point for the Gate A headless native frame loop.

fn main() {
    let report = prism_runtime::run_headless(prism_runtime::HeadlessRuntimeConfig::default())
        .unwrap_or_else(|error| panic!("Prism headless launcher failed: {error}"));

    println!(
        "Prism headless native loop: requested={}, completed={}, submitted={}, failed={}, skipped={}",
        report.frames_requested,
        report.frames_completed,
        report.render_stats.frames_submitted,
        report.render_stats.frames_failed,
        report.render_stats.frames_skipped,
    );

    assert!(
        report.is_successful(),
        "Prism headless native loop did not submit every requested frame: {report:?}"
    );
}
