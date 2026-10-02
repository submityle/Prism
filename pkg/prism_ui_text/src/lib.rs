#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
//! Loom text stack: segmentation, line breaking, rich text, shaping, cursor /
//! selection and a minimal bidi implementation.
//!
//! The crate is deliberately engine- and font-agnostic. All geometry is
//! produced by a pluggable [`shaper::Shaper`]; the default
//! [`shaper::MetricShaper`] computes deterministic advances using only the four
//! basic arithmetic operations, so measurements are fully reproducible and need
//! no transcendental functions. An optional `swash`-backed shaper
//! ([`shaper::SwashShaper`], behind the `shaping` feature) refines cluster
//! handling using the swash Unicode property database.
//!
//! Design background lives in `docs/prism_ui_loom_design_zh.md` §9.6.
//!
//! # Modules
//!
//! * [`segmentation`] — grapheme-cluster and word segmentation.
//! * [`line_break`] — UAX#14 break opportunities and greedy width wrapping.
//! * [`rich_text`] — span-level styling and paragraph properties.
//! * [`shaper`] — the [`shaper::Shaper`] trait and its deterministic default.
//! * [`cursor`] — caret / selection model and glyph hit-testing.
//! * [`bidi`] — a documented minimal subset of UAX#9.
//! * [`cache`] — shaping-result cache keyed by `(text, style, width)`.
//!
//! # Example
//!
//! ```
//! use prism_ui_text::rich_text::TextStyle;
//! use prism_ui_text::shaper::{MetricShaper, Shaper};
//! use prism_ui_text::segmentation::grapheme_count;
//! use prism_ui_text::cursor::{hit_test, x_for_offset};
//!
//! let shaper = MetricShaper::default();
//! let style = TextStyle::default();
//! let run = shaper.shape("hello", &style);
//!
//! // One glyph per grapheme cluster, positive total advance.
//! assert_eq!(run.glyph_count(), grapheme_count("hello"));
//! assert!(run.width() > 0.0);
//!
//! // Round-trip a caret: the x of offset 0 is 0, and hit-testing it returns 0.
//! assert_eq!(x_for_offset(&run, 0), 0.0);
//! assert_eq!(hit_test(&run, 0.0), 0);
//! ```

extern crate alloc;

pub mod bidi;
pub mod cache;
pub mod cursor;
pub mod line_break;
pub mod rich_text;
pub mod segmentation;
pub mod shaper;

pub use bidi::{base_direction, resolve_levels, Direction, Run};
pub use cache::{CacheKey, ShapeCache};
pub use cursor::{caret_positions, hit_test, x_for_offset, Caret, Composition, Selection};
pub use line_break::{break_opportunities, wrap_by_width, BreakKind, BreakPoint, WrappedLine};
pub use rich_text::{
    Align, Color, FontWeight, ParagraphStyle, RichText, Span, TextStyle, Truncate,
};
pub use segmentation::{
    grapheme_count, graphemes, is_grapheme_boundary, next_grapheme_boundary,
    prev_grapheme_boundary, word_bounds, words, Segment,
};
pub use shaper::{MetricShaper, ShapedGlyph, ShapedRun, Shaper};

#[cfg(feature = "shaping")]
pub use shaper::SwashShaper;
