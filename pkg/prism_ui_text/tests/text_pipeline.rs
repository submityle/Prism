//! End-to-end integration tests for the `prism_ui_text` stack.

use prism_ui_text::bidi::{base_direction, resolve_levels, Direction};
use prism_ui_text::cache::ShapeCache;
use prism_ui_text::cursor::{caret_positions, hit_test, x_for_offset, Selection};
use prism_ui_text::line_break::wrap_by_width;
use prism_ui_text::rich_text::{RichText, Span, TextStyle};
use prism_ui_text::segmentation::{grapheme_count, graphemes};
use prism_ui_text::shaper::{MetricShaper, Shaper};

#[test]
fn shape_then_cursor_round_trip() {
    let shaper = MetricShaper::default();
    let style = TextStyle::default();
    let text = "hello world";
    let run = shaper.shape(text, &style);

    // One glyph per grapheme cluster.
    assert_eq!(run.glyph_count(), grapheme_count(text));

    // Caret positions bracket every cluster plus the end.
    let carets = caret_positions(&run);
    assert_eq!(carets.len(), run.glyph_count() + 1);
    assert_eq!(carets[0].x, 0.0);
    assert_eq!(carets.last().map(|c| c.x), Some(run.width()));

    // Hit-testing the extremes resolves to the start and end offsets.
    assert_eq!(hit_test(&run, 0.0), 0);
    assert_eq!(hit_test(&run, run.width()), text.len());

    // x_for_offset is monotonic across the first few offsets.
    assert!(x_for_offset(&run, 1) < x_for_offset(&run, 5));
}

#[test]
fn wrap_and_measure_lines() {
    let text = "aaa bbb ccc ddd";
    let lines = wrap_by_width(text, 8);
    assert!(lines.len() >= 2);

    // Every produced line is non-decreasing in start offset and in range.
    let mut prev_end = 0;
    for line in &lines {
        assert!(line.start <= line.end);
        assert!(line.start >= prev_end || line.start == prev_end);
        prev_end = line.end;
    }
    // The last line reaches the end of the text.
    assert_eq!(lines.last().map(|l| l.end), Some(text.len()));
}

#[test]
fn shape_cache_reuses_entries() {
    let shaper = MetricShaper::default();
    let style = TextStyle::default();
    let mut cache = ShapeCache::new();

    let w1 = cache.shape_cached(&shaper, "cached", &style, 120.0).width();
    assert_eq!(cache.len(), 1);
    let w2 = cache.shape_cached(&shaper, "cached", &style, 120.0).width();
    assert_eq!(cache.len(), 1);
    assert_eq!(w1, w2);

    // A different width is a different key.
    cache.shape_cached(&shaper, "cached", &style, 240.0);
    assert_eq!(cache.len(), 2);
}

#[test]
fn bidi_mixed_direction_runs() {
    // Latin + Hebrew + Latin produces three runs with alternating direction.
    let text = "ab\u{05D0}\u{05D1}cd";
    assert_eq!(base_direction(text), Direction::Ltr);
    let runs = resolve_levels(text, Direction::Ltr);
    assert_eq!(runs.len(), 3);
    assert_eq!(runs[0].direction(), Direction::Ltr);
    assert_eq!(runs[1].direction(), Direction::Rtl);
    assert_eq!(runs[2].direction(), Direction::Ltr);

    // Runs tile the text exactly.
    assert_eq!(runs[0].start, 0);
    assert_eq!(runs.last().map(|r| r.end), Some(text.len()));
}

#[test]
fn rich_text_styling_and_shaping_widths() {
    let shaper = MetricShaper::default();
    let bold = TextStyle::default().bold();
    let rt = RichText::new("Hi there").with_span(Span::new(0, 2, bold));

    // The first two bytes resolve to the bold span; the rest to the default.
    assert!(rt.style_at(0).weight.is_bold());
    assert!(!rt.style_at(5).weight.is_bold());

    // Shaping the bold prefix is wider than the same text at normal weight.
    let bold_run = shaper.shape("Hi", &bold);
    let normal_run = shaper.shape("Hi", &TextStyle::default());
    assert!(bold_run.width() > normal_run.width());
}

#[test]
fn selection_over_graphemes() {
    let text = "e\u{0301}llo";
    let segs = graphemes(text);
    // Select from the start of the second cluster to the end.
    let sel = Selection::new(segs[1].start, text.len());
    assert_eq!(sel.start(), segs[1].start);
    assert_eq!(sel.end(), text.len());
    assert!(!sel.is_collapsed());
}
