//! The pluggable shaper abstraction and its deterministic default.
//!
//! Shaping turns a run of text plus a [`TextStyle`] into positioned glyphs.
//! The crate ships a font-independent default, [`MetricShaper`], whose advances
//! are computed with the four basic arithmetic operations only, so results are
//! fully reproducible across platforms and need no transcendental functions.
//!
//! When the `shaping` feature is enabled an additional [`SwashShaper`] becomes
//! available. It still produces deterministic *metric* advances but refines
//! cluster handling using the swash Unicode property database (for example by
//! folding combining marks onto their base glyph). It performs **no** font
//! rasterisation and loads no font files.

use crate::rich_text::TextStyle;
use crate::segmentation;
use alloc::vec::Vec;
use unicode_width::UnicodeWidthStr;

/// Line-height-independent vertical extent factor applied to the font size when
/// reporting a run's height.
const LINE_EXTENT_RATIO: f32 = 1.25;

/// A single positioned glyph produced by a [`Shaper`].
///
/// The default metric shaper emits exactly one glyph per grapheme cluster; a
/// more capable shaper may emit several glyphs sharing a `cluster` offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapedGlyph {
    /// Byte offset of the cluster this glyph belongs to within the source text.
    pub cluster: usize,
    /// Horizontal advance of the glyph in logical pixels.
    pub advance: f32,
}

impl ShapedGlyph {
    /// Builds a glyph for `cluster` with the given `advance`.
    #[must_use]
    pub fn new(cluster: usize, advance: f32) -> Self {
        Self { cluster, advance }
    }
}

/// The result of shaping a run of text: positioned glyphs plus overall metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapedRun {
    /// Glyphs in logical (left-to-right) order.
    pub glyphs: Vec<ShapedGlyph>,
    /// Sum of all glyph advances in logical pixels.
    pub width: f32,
    /// Vertical extent of the run in logical pixels.
    pub height: f32,
    /// Byte length of the shaped source text.
    pub text_len: usize,
}

impl ShapedRun {
    /// Returns the number of glyphs in the run.
    #[must_use]
    pub fn glyph_count(&self) -> usize {
        self.glyphs.len()
    }

    /// Returns `true` when the run contains no glyphs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    /// Returns the total advance width of the run.
    #[must_use]
    pub fn width(&self) -> f32 {
        self.width
    }

    /// Returns the vertical extent of the run.
    #[must_use]
    pub fn height(&self) -> f32 {
        self.height
    }

    /// Returns the byte length of the shaped source text.
    #[must_use]
    pub fn text_len(&self) -> usize {
        self.text_len
    }
}

/// A text shaper: converts `text` and a [`TextStyle`] into a [`ShapedRun`].
pub trait Shaper {
    /// Shapes `text` under `style`.
    fn shape(&self, text: &str, style: &TextStyle) -> ShapedRun;
}

/// A deterministic, font-independent shaper.
///
/// Each grapheme cluster is assigned an advance proportional to its display
/// column count (from [`unicode_width`]), the font size and a configurable
/// `advance_ratio`. Bold runs receive a small additional per-column factor.
/// Only `+ - * /` are used, so the output is stable and reproducible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricShaper {
    /// Logical pixels of advance per display column, per unit font size.
    pub advance_ratio: f32,
    /// Extra per-column advance ratio added for bold runs.
    pub bold_extra: f32,
}

impl Default for MetricShaper {
    fn default() -> Self {
        Self {
            advance_ratio: 0.5,
            bold_extra: 0.05,
        }
    }
}

impl MetricShaper {
    /// Builds a metric shaper with explicit ratios.
    #[must_use]
    pub fn new(advance_ratio: f32, bold_extra: f32) -> Self {
        Self {
            advance_ratio,
            bold_extra,
        }
    }

    /// Computes the advance for `columns` display columns under `style`.
    #[must_use]
    pub fn column_advance(&self, columns: f32, style: &TextStyle) -> f32 {
        let mut ratio = self.advance_ratio;
        if style.weight.is_bold() {
            ratio += self.bold_extra;
        }
        columns * style.font_size * ratio
    }
}

impl Shaper for MetricShaper {
    fn shape(&self, text: &str, style: &TextStyle) -> ShapedRun {
        let mut glyphs = Vec::new();
        let mut width = 0.0f32;
        for seg in segmentation::graphemes(text) {
            let columns = seg.text.width() as f32;
            let advance = self.column_advance(columns, style);
            glyphs.push(ShapedGlyph::new(seg.start, advance));
            width += advance;
        }
        ShapedRun {
            glyphs,
            width,
            height: style.font_size * LINE_EXTENT_RATIO,
            text_len: text.len(),
        }
    }
}

/// A swash-backed metric shaper (available with the `shaping` feature).
///
/// This shaper reuses [`MetricShaper`]'s deterministic arithmetic but consults
/// the swash Unicode property database to merge combining marks (canonical
/// combining class other than zero) onto the preceding base character, giving
/// them a zero advance. It performs no font loading or rasterisation.
#[cfg(feature = "shaping")]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SwashShaper {
    /// Underlying metric model used for base-character advances.
    pub metrics: MetricShaper,
}

#[cfg(feature = "shaping")]
impl SwashShaper {
    /// Builds a swash shaper from an explicit metric model.
    #[must_use]
    pub fn new(metrics: MetricShaper) -> Self {
        Self { metrics }
    }
}

#[cfg(feature = "shaping")]
impl Shaper for SwashShaper {
    fn shape(&self, text: &str, style: &TextStyle) -> ShapedRun {
        use swash::text::Codepoint;
        use unicode_width::UnicodeWidthChar;

        let mut glyphs: Vec<ShapedGlyph> = Vec::new();
        let mut width = 0.0f32;
        for (offset, ch) in text.char_indices() {
            if ch.combining_class() != 0 {
                // Combining mark: fold onto the current base cluster.
                if let Some(last) = glyphs.last() {
                    glyphs.push(ShapedGlyph::new(last.cluster, 0.0));
                } else {
                    glyphs.push(ShapedGlyph::new(offset, 0.0));
                }
                continue;
            }
            let columns = UnicodeWidthChar::width(ch).unwrap_or(0) as f32;
            let advance = self.metrics.column_advance(columns, style);
            glyphs.push(ShapedGlyph::new(offset, advance));
            width += advance;
        }
        ShapedRun {
            glyphs,
            width,
            height: style.font_size * LINE_EXTENT_RATIO,
            text_len: text.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_shaper_one_glyph_per_cluster() {
        let shaper = MetricShaper::default();
        let style = TextStyle::default();
        let run = shaper.shape("abc", &style);
        assert_eq!(run.glyph_count(), 3);
        assert_eq!(run.text_len(), 3);
        assert!(run.width() > 0.0);
        assert!(run.height() > 0.0);
    }

    #[test]
    fn combining_mark_is_one_cluster() {
        let shaper = MetricShaper::default();
        let style = TextStyle::default();
        // "e" + combining acute is a single grapheme cluster.
        let run = shaper.shape("e\u{0301}", &style);
        assert_eq!(run.glyph_count(), 1);
    }

    #[test]
    fn bold_is_wider_than_regular() {
        let shaper = MetricShaper::default();
        let regular = shaper.shape("mmmm", &TextStyle::default());
        let bold = shaper.shape("mmmm", &TextStyle::default().bold());
        assert!(bold.width() > regular.width());
    }

    #[test]
    fn larger_font_is_wider() {
        let shaper = MetricShaper::default();
        let small = shaper.shape("x", &TextStyle::default().with_size(10.0));
        let big = shaper.shape("x", &TextStyle::default().with_size(40.0));
        assert!(big.width() > small.width());
    }

    #[test]
    fn empty_text_has_no_glyphs() {
        let run = MetricShaper::default().shape("", &TextStyle::default());
        assert!(run.is_empty());
        assert_eq!(run.width(), 0.0);
    }

    #[test]
    fn column_advance_is_deterministic() {
        let shaper = MetricShaper::new(0.6, 0.1);
        let style = TextStyle::default().with_size(20.0);
        let a = shaper.column_advance(2.0, &style);
        let b = shaper.column_advance(2.0, &style);
        assert_eq!(a, b);
        assert_eq!(a, 2.0 * 20.0 * 0.6);
    }

    #[cfg(feature = "shaping")]
    #[test]
    fn swash_folds_combining_marks() {
        let shaper = SwashShaper::default();
        let style = TextStyle::default();
        // Base + combining mark: two chars, but the mark has zero advance.
        let run = shaper.shape("e\u{0301}", &style);
        assert_eq!(run.glyph_count(), 2);
        let base = shaper.shape("e", &style);
        assert_eq!(run.width(), base.width());
    }

    #[cfg(feature = "shaping")]
    #[test]
    fn swash_plain_matches_metric_width() {
        let swash = SwashShaper::default();
        let metric = MetricShaper::default();
        let style = TextStyle::default();
        assert_eq!(
            swash.shape("hello", &style).width(),
            metric.shape("hello", &style).width()
        );
    }
}
