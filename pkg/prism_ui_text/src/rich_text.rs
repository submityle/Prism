//! Rich-text model: colours, font styling, spans and paragraph properties.
//!
//! A [`RichText`] owns the backing string plus an ordered list of style
//! [`Span`]s and a [`ParagraphStyle`]. Spans are byte-ranged; where several
//! spans cover the same offset the later span in document order wins, matching
//! the usual "last declaration wins" cascade.

use alloc::string::String;
use alloc::vec::Vec;

/// A straight-alpha 8-bit-per-channel RGBA colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Color {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Alpha channel, where `255` is fully opaque.
    pub a: u8,
}

impl Color {
    /// Opaque black.
    pub const BLACK: Color = Color::rgb(0, 0, 0);
    /// Opaque white.
    pub const WHITE: Color = Color::rgb(255, 255, 255);
    /// Fully transparent.
    pub const TRANSPARENT: Color = Color::rgba(0, 0, 0, 0);

    /// Builds a colour from explicit channels.
    #[must_use]
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Builds an opaque colour from red, green and blue channels.
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Returns `true` when the colour is fully opaque.
    #[must_use]
    pub const fn is_opaque(self) -> bool {
        self.a == 255
    }
}

/// A CSS-style numeric font weight in the range `1..=1000`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FontWeight(pub u16);

impl FontWeight {
    /// Thin (100).
    pub const THIN: FontWeight = FontWeight(100);
    /// Normal / regular (400).
    pub const NORMAL: FontWeight = FontWeight(400);
    /// Medium (500).
    pub const MEDIUM: FontWeight = FontWeight(500);
    /// Bold (700).
    pub const BOLD: FontWeight = FontWeight(700);
    /// Black / heavy (900).
    pub const BLACK: FontWeight = FontWeight(900);

    /// Builds a weight from a raw value.
    #[must_use]
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    /// Returns `true` when the weight is bold or heavier (>= 600).
    #[must_use]
    pub const fn is_bold(self) -> bool {
        self.0 >= 600
    }
}

impl Default for FontWeight {
    fn default() -> Self {
        FontWeight::NORMAL
    }
}

/// Visual styling applied to a run of text.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextStyle {
    /// Font size in logical pixels.
    pub font_size: f32,
    /// Font weight.
    pub weight: FontWeight,
    /// Whether the run is rendered in an italic / oblique style.
    pub italic: bool,
    /// Whether the run is underlined.
    pub underline: bool,
    /// Fill colour of the glyphs.
    pub color: Color,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font_size: 16.0,
            weight: FontWeight::NORMAL,
            italic: false,
            underline: false,
            color: Color::BLACK,
        }
    }
}

impl TextStyle {
    /// Returns a copy with the given font size.
    #[must_use]
    pub fn with_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    /// Returns a copy with the given weight.
    #[must_use]
    pub fn with_weight(mut self, weight: FontWeight) -> Self {
        self.weight = weight;
        self
    }

    /// Returns a copy forced to bold weight.
    #[must_use]
    pub fn bold(mut self) -> Self {
        self.weight = FontWeight::BOLD;
        self
    }

    /// Returns a copy marked italic.
    #[must_use]
    pub fn italic(mut self) -> Self {
        self.italic = true;
        self
    }

    /// Returns a copy marked underlined.
    #[must_use]
    pub fn underlined(mut self) -> Self {
        self.underline = true;
        self
    }

    /// Returns a copy with the given fill colour.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
}

/// A styled byte range within a [`RichText`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    /// Inclusive start byte offset.
    pub start: usize,
    /// Exclusive end byte offset.
    pub end: usize,
    /// Style applied to the range.
    pub style: TextStyle,
}

impl Span {
    /// Builds a span over `[start, end)` with the given style.
    #[must_use]
    pub fn new(start: usize, end: usize, style: TextStyle) -> Self {
        Self { start, end, style }
    }

    /// Returns `true` when `offset` falls within `[start, end)`.
    #[must_use]
    pub fn contains(&self, offset: usize) -> bool {
        offset >= self.start && offset < self.end
    }

    /// Returns the span length in bytes.
    #[must_use]
    pub fn len_bytes(&self) -> usize {
        self.end - self.start
    }

    /// Returns `true` when the span covers no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Horizontal paragraph alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Align {
    /// Align to the leading edge (left for LTR).
    #[default]
    Start,
    /// Centre within the available width.
    Center,
    /// Align to the trailing edge (right for LTR).
    End,
    /// Stretch inter-word spacing to fill the width.
    Justify,
}

/// Paragraph truncation policy when content overflows the available area.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Truncate {
    /// Never truncate; overflow is clipped by the caller.
    #[default]
    None,
    /// Replace the overflowing tail with an ellipsis.
    Ellipsis,
}

/// Paragraph-level layout properties.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParagraphStyle {
    /// Horizontal alignment.
    pub align: Align,
    /// Line-height multiplier applied to the font size.
    pub line_height: f32,
    /// Truncation policy.
    pub truncate: Truncate,
}

impl Default for ParagraphStyle {
    fn default() -> Self {
        Self {
            align: Align::Start,
            line_height: 1.2,
            truncate: Truncate::None,
        }
    }
}

/// A string with per-range styling and paragraph properties.
#[derive(Clone, Debug, PartialEq)]
pub struct RichText {
    /// Backing text.
    text: String,
    /// Ordered style spans.
    spans: Vec<Span>,
    /// Paragraph properties.
    paragraph: ParagraphStyle,
}

impl RichText {
    /// Builds unstyled rich text from a plain string.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            spans: Vec::new(),
            paragraph: ParagraphStyle::default(),
        }
    }

    /// Builds rich text whose whole range carries `style`.
    #[must_use]
    pub fn plain(text: impl Into<String>, style: TextStyle) -> Self {
        let text = text.into();
        let span = Span::new(0, text.len(), style);
        Self {
            text,
            spans: alloc::vec![span],
            paragraph: ParagraphStyle::default(),
        }
    }

    /// Appends a style span and returns `self` for chaining.
    #[must_use]
    pub fn with_span(mut self, span: Span) -> Self {
        self.spans.push(span);
        self
    }

    /// Appends a style span in place.
    pub fn push_span(&mut self, span: Span) {
        self.spans.push(span);
    }

    /// Replaces the paragraph style and returns `self` for chaining.
    #[must_use]
    pub fn with_paragraph(mut self, paragraph: ParagraphStyle) -> Self {
        self.paragraph = paragraph;
        self
    }

    /// Returns the backing text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the ordered style spans.
    #[must_use]
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    /// Returns the paragraph style.
    #[must_use]
    pub fn paragraph(&self) -> &ParagraphStyle {
        &self.paragraph
    }

    /// Returns the length of the backing text in bytes.
    #[must_use]
    pub fn len_bytes(&self) -> usize {
        self.text.len()
    }

    /// Returns `true` when the backing text is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Resolves the effective style at `offset`, applying later spans over
    /// earlier ones and falling back to [`TextStyle::default`].
    #[must_use]
    pub fn style_at(&self, offset: usize) -> TextStyle {
        let mut style = TextStyle::default();
        for span in &self.spans {
            if span.contains(offset) {
                style = span.style;
            }
        }
        style
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_helpers() {
        assert_eq!(Color::rgb(1, 2, 3), Color::rgba(1, 2, 3, 255));
        assert!(Color::BLACK.is_opaque());
        assert!(!Color::TRANSPARENT.is_opaque());
    }

    #[test]
    fn font_weight_bold_threshold() {
        assert!(!FontWeight::NORMAL.is_bold());
        assert!(FontWeight::BOLD.is_bold());
        assert!(FontWeight::new(600).is_bold());
        assert_eq!(FontWeight::default(), FontWeight::NORMAL);
    }

    #[test]
    fn text_style_builders() {
        let s = TextStyle::default()
            .bold()
            .italic()
            .underlined()
            .with_size(24.0);
        assert!(s.weight.is_bold());
        assert!(s.italic);
        assert!(s.underline);
        assert_eq!(s.font_size, 24.0);
    }

    #[test]
    fn style_at_later_span_wins() {
        let rt = RichText::new("abcdef")
            .with_span(Span::new(0, 6, TextStyle::default().with_size(10.0)))
            .with_span(Span::new(2, 4, TextStyle::default().with_size(20.0)));
        assert_eq!(rt.style_at(0).font_size, 10.0);
        assert_eq!(rt.style_at(3).font_size, 20.0);
        assert_eq!(rt.style_at(5).font_size, 10.0);
    }

    #[test]
    fn style_at_default_without_spans() {
        let rt = RichText::new("x");
        assert_eq!(rt.style_at(0), TextStyle::default());
        assert_eq!(rt.len_bytes(), 1);
        assert!(!rt.is_empty());
    }

    #[test]
    fn plain_covers_whole_text() {
        let rt = RichText::plain("hey", TextStyle::default().bold());
        assert_eq!(rt.spans().len(), 1);
        assert!(rt.style_at(1).weight.is_bold());
        assert_eq!(rt.paragraph().align, Align::Start);
    }

    #[test]
    fn span_contains_bounds() {
        let span = Span::new(2, 5, TextStyle::default());
        assert!(!span.contains(1));
        assert!(span.contains(2));
        assert!(span.contains(4));
        assert!(!span.contains(5));
        assert_eq!(span.len_bytes(), 3);
    }
}
