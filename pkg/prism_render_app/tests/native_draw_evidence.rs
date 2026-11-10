//! Bounded evidence checks for native draw acceptance.
//!
//! Render counters are scheduling/recording telemetry.  They do not prove that
//! a raster pass changed the target.  Keep this check in an integration-test
//! file so production draw recording stays untouched and acceptance tests must
//! provide an actual readback when they claim visible output.
#![forbid(unsafe_code)]

use prism_render_app::RenderFrameStats;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DrawEvidence {
    NotSubmitted,
    NoRecordedDraw,
    RecordedWithoutVisiblePixels,
    RecordedWithVisiblePixels,
}

fn classify(
    stats: &RenderFrameStats,
    pixels: &[[f32; 4]],
    clear: [f32; 4],
    tolerance: f32,
) -> DrawEvidence {
    if stats.frames_submitted == 0 {
        return DrawEvidence::NotSubmitted;
    }
    if stats.draws_recorded == 0 {
        return DrawEvidence::NoRecordedDraw;
    }

    let changed = pixels.iter().any(|pixel| {
        pixel
            .iter()
            .zip(clear)
            .any(|(actual, expected)| (actual - expected).abs() > tolerance)
    });
    if changed {
        DrawEvidence::RecordedWithVisiblePixels
    } else {
        DrawEvidence::RecordedWithoutVisiblePixels
    }
}

fn stats_with_recorded_draws(draws_recorded: u64) -> RenderFrameStats {
    let mut stats = RenderFrameStats::default();
    stats.frames_submitted = u64::from(draws_recorded > 0);
    stats.draws_recorded = draws_recorded;
    stats
}

#[test]
fn recorded_draws_without_readback_are_not_visual_evidence() {
    let stats = stats_with_recorded_draws(1);
    let pixels = [[0.0, 0.0, 0.0, 1.0]; 4];

    assert_eq!(
        classify(&stats, &pixels, [0.0, 0.0, 0.0, 1.0], 1.0e-5),
        DrawEvidence::RecordedWithoutVisiblePixels
    );
}

#[test]
fn an_unsubmitted_frame_cannot_borrow_pixels_as_its_evidence() {
    let stats = RenderFrameStats::default();
    let pixels = [[1.0, 0.0, 0.0, 1.0]; 4];

    assert_eq!(
        classify(&stats, &pixels, [0.0, 0.0, 0.0, 1.0], 1.0e-5),
        DrawEvidence::NotSubmitted
    );
}

#[test]
fn a_non_clear_readback_is_required_for_visible_draw_evidence() {
    let stats = stats_with_recorded_draws(1);
    let pixels = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    assert_eq!(
        classify(&stats, &pixels, [0.0, 0.0, 0.0, 1.0], 1.0e-5),
        DrawEvidence::RecordedWithVisiblePixels
    );
}

#[test]
fn no_recorded_draw_does_not_become_visible_evidence_from_clear_variation() {
    let mut stats = stats_with_recorded_draws(0);
    stats.frames_submitted = 1;
    let pixels = [[1.0, 0.0, 0.0, 1.0]; 4];

    assert_eq!(
        classify(&stats, &pixels, [0.0, 0.0, 0.0, 1.0], 1.0e-5),
        DrawEvidence::NoRecordedDraw
    );
}
