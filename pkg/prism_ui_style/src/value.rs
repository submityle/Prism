//! Styleable properties and the values they can take.
//!
//! [`StyleProp`] enumerates every property the style layer understands, and
//! [`StyleValue`] is the pure-data value stored for a property. Values are
//! deliberately engine-agnostic so a style sheet can be authored, serialized
//! and hot-reloaded without pulling in any renderer types.

use alloc::string::String;

/// A styleable property.
///
/// Properties are physical/longhand (for example `PaddingLeft` rather than a
/// `padding` shorthand) so the cascade can merge them deterministically with a
/// simple last-writer-wins rule. Shorthand helpers that expand into these
/// longhands live on [`crate::class::Class`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum StyleProp {
    /// Box width.
    Width,
    /// Box height.
    Height,
    /// Minimum width.
    MinWidth,
    /// Minimum height.
    MinHeight,
    /// Maximum width.
    MaxWidth,
    /// Maximum height.
    MaxHeight,
    /// Top padding.
    PaddingTop,
    /// Right padding.
    PaddingRight,
    /// Bottom padding.
    PaddingBottom,
    /// Left padding.
    PaddingLeft,
    /// Top margin.
    MarginTop,
    /// Right margin.
    MarginRight,
    /// Bottom margin.
    MarginBottom,
    /// Left margin.
    MarginLeft,
    /// Flex grow factor.
    FlexGrow,
    /// Flex shrink factor.
    FlexShrink,
    /// Flex basis.
    FlexBasis,
    /// Gap between both rows and columns.
    Gap,
    /// Gap between rows.
    RowGap,
    /// Gap between columns.
    ColumnGap,
    /// Foreground (text) color.
    Color,
    /// Background color.
    BackgroundColor,
    /// Border color.
    BorderColor,
    /// Font size.
    FontSize,
    /// Font weight.
    FontWeight,
    /// Opacity in the range `0.0..=1.0`.
    Opacity,
    /// Border radius.
    BorderRadius,
    /// Border width.
    BorderWidth,
    /// Display mode.
    Display,
    /// Flex main-axis direction.
    FlexDirection,
    /// Main-axis content justification.
    JustifyContent,
    /// Cross-axis item alignment.
    AlignItems,
}

/// A length value, as used by box metrics such as width, padding and gaps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Length {
    /// A length in logical pixels.
    Px(f32),
    /// A percentage of the relevant parent dimension, where `100.0` is 100%.
    Percent(f32),
    /// The layout engine chooses the value automatically.
    Auto,
}

/// An `RGBA` color stored as four linear channels in the range `0.0..=1.0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
    /// Alpha channel, where `1.0` is fully opaque.
    pub a: f32,
}

impl Color {
    /// Builds a color from four float channels in the range `0.0..=1.0`.
    #[must_use]
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Builds a color from four 8-bit channels, normalizing to `0.0..=1.0`.
    #[must_use]
    pub fn rgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: f32::from(r) / 255.0,
            g: f32::from(g) / 255.0,
            b: f32::from(b) / 255.0,
            a: f32::from(a) / 255.0,
        }
    }
}

/// A keyword value for enumerated properties such as display and alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Keyword {
    // Display.
    /// `display: flex`.
    Flex,
    /// `display: grid`.
    Grid,
    /// `display: block`.
    Block,
    /// Absence of a box (`display: none`).
    None,
    // Flex direction.
    /// Lay children out in a row.
    Row,
    /// Lay children out in a column.
    Column,
    /// Row, reversed.
    RowReverse,
    /// Column, reversed.
    ColumnReverse,
    // Justify / align.
    /// Pack items at the start.
    Start,
    /// Pack items at the end.
    End,
    /// Center items.
    Center,
    /// Distribute items with space between them.
    SpaceBetween,
    /// Distribute items with space around them.
    SpaceAround,
    /// Distribute items with equal space around them.
    SpaceEvenly,
    /// Stretch items to fill the cross axis.
    Stretch,
    /// Align items to their text baseline.
    Baseline,
}

/// A concrete style value.
///
/// A value is either a resolved literal (length, number, color, keyword) or a
/// [`StyleValue::TokenRef`] that points at a named design token to be resolved
/// through a [`crate::token::TokenStore`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum StyleValue {
    /// A length (pixels, percentage or auto).
    Length(Length),
    /// A unitless number, used for factors like flex-grow and opacity.
    Number(f32),
    /// A color.
    Color(Color),
    /// A keyword for an enumerated property.
    Keyword(Keyword),
    /// A reference to a named design token.
    TokenRef(String),
}

impl StyleValue {
    /// Creates a pixel length value.
    #[must_use]
    pub const fn px(value: f32) -> Self {
        StyleValue::Length(Length::Px(value))
    }

    /// Creates a percentage length value (`100.0` means 100%).
    #[must_use]
    pub const fn percent(value: f32) -> Self {
        StyleValue::Length(Length::Percent(value))
    }

    /// Creates an automatic length value.
    #[must_use]
    pub const fn auto() -> Self {
        StyleValue::Length(Length::Auto)
    }

    /// Creates a unitless number value.
    #[must_use]
    pub const fn number(value: f32) -> Self {
        StyleValue::Number(value)
    }

    /// Creates a color value from float channels.
    #[must_use]
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        StyleValue::Color(Color::rgba(r, g, b, a))
    }

    /// Creates a color value from 8-bit channels.
    #[must_use]
    pub fn rgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        StyleValue::Color(Color::rgba8(r, g, b, a))
    }

    /// Creates a keyword value.
    #[must_use]
    pub const fn keyword(keyword: Keyword) -> Self {
        StyleValue::Keyword(keyword)
    }

    /// Creates a token reference value.
    #[must_use]
    pub fn token(name: impl Into<String>) -> Self {
        StyleValue::TokenRef(name.into())
    }

    /// Returns `true` if this value is an unresolved token reference.
    #[must_use]
    pub const fn is_token_ref(&self) -> bool {
        matches!(self, StyleValue::TokenRef(_))
    }
}
