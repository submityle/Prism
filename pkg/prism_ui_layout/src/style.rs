//! Style inputs that drive the layout algorithm.
//!
//! [`LayoutStyle`] mirrors the subset of CSS flexbox properties implemented
//! by this crate. Every field has a sensible default so callers only need to
//! set the properties they care about.

use crate::geometry::{Dimension, Edges, Size};

/// Controls whether and how a box participates in layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Display {
    /// The box lays its children out as a flex container.
    #[default]
    Flex,
    /// The box and its subtree are removed from layout and collapse to a
    /// zero size.
    None,
}

/// Controls how a box is positioned relative to its flex flow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Position {
    /// The box participates in normal flex flow.
    #[default]
    Relative,
    /// The box is taken out of flow and positioned against its container
    /// using `inset`.
    Absolute,
}

/// The direction in which flex items are placed along the main axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FlexDirection {
    /// Left to right.
    #[default]
    Row,
    /// Top to bottom.
    Column,
    /// Right to left.
    RowReverse,
    /// Bottom to top.
    ColumnReverse,
}

impl FlexDirection {
    /// Returns `true` when the main axis is horizontal.
    pub fn is_row(self) -> bool {
        matches!(self, FlexDirection::Row | FlexDirection::RowReverse)
    }

    /// Returns `true` when items are placed in reverse order along the main
    /// axis.
    pub fn is_reverse(self) -> bool {
        matches!(
            self,
            FlexDirection::RowReverse | FlexDirection::ColumnReverse
        )
    }
}

/// Controls whether flex items are forced onto a single line or allowed to
/// wrap onto multiple lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FlexWrap {
    /// All items are placed on a single line.
    #[default]
    NoWrap,
    /// Items wrap onto additional lines as needed.
    Wrap,
    /// Items wrap, with the cross-axis order of lines reversed.
    WrapReverse,
}

/// Distributes free space on the main axis of a flex line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum JustifyContent {
    /// Pack items toward the start of the line.
    #[default]
    Start,
    /// Pack items toward the end of the line.
    End,
    /// Center items within the line.
    Center,
    /// Distribute free space between items, with none at the edges.
    SpaceBetween,
    /// Distribute free space around items, with half-size gaps at the edges.
    SpaceAround,
    /// Distribute free space so gaps between and around items are equal.
    SpaceEvenly,
}

/// Aligns flex items along the cross axis of their line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlignItems {
    /// Align to the cross-start edge of the line.
    Start,
    /// Align to the cross-end edge of the line.
    End,
    /// Center on the cross axis.
    Center,
    /// Stretch to fill the line's cross size when the item's cross size is
    /// automatic.
    #[default]
    Stretch,
}

/// Distributes flex lines along the cross axis of a multi-line container.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlignContent {
    /// Pack lines toward the cross-start edge.
    Start,
    /// Pack lines toward the cross-end edge.
    End,
    /// Center lines on the cross axis.
    Center,
    /// Stretch lines to fill the container's cross size.
    #[default]
    Stretch,
    /// Distribute free space between lines, with none at the edges.
    SpaceBetween,
    /// Distribute free space around lines, with half-size gaps at the edges.
    SpaceAround,
    /// Distribute free space so gaps between and around lines are equal.
    SpaceEvenly,
}

/// The full set of style inputs consumed by the layout solver for a box.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutStyle {
    /// Whether and how the box participates in layout.
    pub display: Display,
    /// Whether the box is in flow or absolutely positioned.
    pub position: Position,
    /// Offsets used to position an absolutely positioned box.
    pub inset: Edges<Dimension>,

    /// Direction of the main axis for this flex container.
    pub flex_direction: FlexDirection,
    /// Whether items wrap onto multiple lines.
    pub flex_wrap: FlexWrap,

    /// Main-axis distribution of items within a line.
    pub justify_content: JustifyContent,
    /// Default cross-axis alignment applied to items.
    pub align_items: AlignItems,
    /// Per-box override of [`LayoutStyle::align_items`]; `None` means use the
    /// container's value.
    pub align_self: Option<AlignItems>,
    /// Cross-axis distribution of lines in a multi-line container.
    pub align_content: AlignContent,

    /// Spacing between items (`width`) and between lines (`height`).
    pub gap: Size<f32>,

    /// Growth factor applied to positive free space on the main axis.
    pub flex_grow: f32,
    /// Shrink factor applied to negative free space on the main axis.
    pub flex_shrink: f32,
    /// Initial main size of the item before free space is distributed.
    pub flex_basis: Dimension,

    /// Preferred size of the box.
    pub size: Size<Dimension>,
    /// Minimum size of the box.
    pub min_size: Size<Dimension>,
    /// Maximum size of the box.
    pub max_size: Size<Dimension>,

    /// Space outside the box's border.
    pub margin: Edges<Dimension>,
    /// Space between the box's border and its content.
    pub padding: Edges<Dimension>,
    /// Border thickness surrounding the padding.
    pub border: Edges<Dimension>,
}

impl Default for LayoutStyle {
    fn default() -> Self {
        Self {
            display: Display::default(),
            position: Position::default(),
            inset: Edges::splat(Dimension::Auto),

            flex_direction: FlexDirection::default(),
            flex_wrap: FlexWrap::default(),

            justify_content: JustifyContent::default(),
            align_items: AlignItems::default(),
            align_self: None,
            align_content: AlignContent::default(),

            gap: Size::new(0.0, 0.0),

            flex_grow: 0.0,
            flex_shrink: 1.0,
            flex_basis: Dimension::Auto,

            size: Size::new(Dimension::Auto, Dimension::Auto),
            min_size: Size::new(Dimension::Auto, Dimension::Auto),
            max_size: Size::new(Dimension::Auto, Dimension::Auto),

            margin: Edges::splat(Dimension::Points(0.0)),
            padding: Edges::splat(Dimension::Points(0.0)),
            border: Edges::splat(Dimension::Points(0.0)),
        }
    }
}
