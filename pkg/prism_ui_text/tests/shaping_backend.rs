//! Integration coverage for the optional swash-backed shaper.
//!
//! These tests only compile and run when the `shaping` feature is enabled; the
//! default build sees an empty crate here.

#![cfg(feature = "shaping")]

use prism_ui_text::rich_text::TextStyle;
use prism_ui_text::shaper::{MetricShaper, Shaper, SwashShaper};

#[test]
fn swash_matches_metric_for_plain_ascii() {
    let swash = SwashShaper::default();
    let metric = MetricShaper::default();
    let style = TextStyle::default();
    let a = swash.shape("plain ascii", &style);
    let b = metric.shape("plain ascii", &style);
    assert_eq!(a.width(), b.width());
    assert_eq!(a.glyph_count(), b.glyph_count());
}

#[test]
fn swash_zero_advance_for_combining_mark() {
    let swash = SwashShaper::default();
    let style = TextStyle::default();
    let combined = swash.shape("a\u{0301}", &style);
    let base = swash.shape("a", &style);
    assert_eq!(combined.width(), base.width());
    assert_eq!(combined.glyph_count(), 2);
}
