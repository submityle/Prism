//! Direction-aware logical-property resolution (the `RTL` layer).
//!
//! Loom authors box spacing with *logical* sides — `inline-start`,
//! `inline-end`, `block-start`, `block-end` — rather than the physical
//! `left`/`right`/`top`/`bottom` understood by the style crate. Logical sides
//! are resolved against a writing [`Direction`]: in a left-to-right (`LTR`)
//! context `inline-start` is the left edge, while in a right-to-left (`RTL`)
//! context it mirrors to the right edge. Block sides never mirror.
//!
//! This module turns [`LogicalEdge`] spacing into concrete
//! [`prism_ui_style::StyleProp`] entries and can also mirror an already-physical
//! [`PropMap`], which is what lets a single authored style sheet flip correctly
//! between `LTR` and `RTL`.

use alloc::collections::BTreeMap;

use prism_ui_style::{PropMap, StyleProp, StyleValue};

/// The writing direction a logical side is resolved against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Direction {
    /// Left-to-right (the default for Latin scripts).
    #[default]
    Ltr,
    /// Right-to-left (for scripts such as Arabic or Hebrew).
    Rtl,
}

impl Direction {
    /// Returns `true` if this is the right-to-left direction.
    #[must_use]
    pub fn is_rtl(self) -> bool {
        matches!(self, Direction::Rtl)
    }

    /// Returns `true` if this is the left-to-right direction.
    #[must_use]
    pub fn is_ltr(self) -> bool {
        matches!(self, Direction::Ltr)
    }

    /// Returns the opposite direction.
    #[must_use]
    pub fn flipped(self) -> Self {
        match self {
            Direction::Ltr => Direction::Rtl,
            Direction::Rtl => Direction::Ltr,
        }
    }
}

/// Which box property family a logical edge belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoxEdge {
    /// The padding family (`PaddingTop`, `PaddingRight`, ...).
    Padding,
    /// The margin family (`MarginTop`, `MarginRight`, ...).
    Margin,
}

/// A direction-relative side of a box.
///
/// Inline sides run along the text flow and mirror with the [`Direction`];
/// block sides run across it and are invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogicalSide {
    /// The leading inline edge (left in `LTR`, right in `RTL`).
    InlineStart,
    /// The trailing inline edge (right in `LTR`, left in `RTL`).
    InlineEnd,
    /// The leading block edge (always the top).
    BlockStart,
    /// The trailing block edge (always the bottom).
    BlockEnd,
}

/// A physical side, used internally while resolving a [`LogicalSide`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PhysicalSide {
    Top,
    Right,
    Bottom,
    Left,
}

impl LogicalSide {
    /// Resolves this logical side to a physical side under `direction`.
    fn to_physical(self, direction: Direction) -> PhysicalSide {
        match self {
            LogicalSide::InlineStart => {
                if direction.is_rtl() {
                    PhysicalSide::Right
                } else {
                    PhysicalSide::Left
                }
            }
            LogicalSide::InlineEnd => {
                if direction.is_rtl() {
                    PhysicalSide::Left
                } else {
                    PhysicalSide::Right
                }
            }
            LogicalSide::BlockStart => PhysicalSide::Top,
            LogicalSide::BlockEnd => PhysicalSide::Bottom,
        }
    }

    /// Returns `true` if this side mirrors with the writing direction.
    #[must_use]
    pub fn is_inline(self) -> bool {
        matches!(self, LogicalSide::InlineStart | LogicalSide::InlineEnd)
    }
}

/// A logical box edge: a property family paired with a direction-relative side.
///
/// # Example
///
/// ```
/// use prism_ui_theme::{BoxEdge, Direction, LogicalEdge, LogicalSide};
/// use prism_ui_style::StyleProp;
///
/// let start = LogicalEdge::padding(LogicalSide::InlineStart);
/// assert_eq!(start.to_physical(Direction::Ltr), StyleProp::PaddingLeft);
/// assert_eq!(start.to_physical(Direction::Rtl), StyleProp::PaddingRight);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalEdge {
    /// The box property family this edge belongs to.
    pub edge: BoxEdge,
    /// The direction-relative side.
    pub side: LogicalSide,
}

impl LogicalEdge {
    /// Builds a logical edge from a family and a side.
    #[must_use]
    pub const fn new(edge: BoxEdge, side: LogicalSide) -> Self {
        Self { edge, side }
    }

    /// Builds a padding edge on `side`.
    #[must_use]
    pub const fn padding(side: LogicalSide) -> Self {
        Self::new(BoxEdge::Padding, side)
    }

    /// Builds a margin edge on `side`.
    #[must_use]
    pub const fn margin(side: LogicalSide) -> Self {
        Self::new(BoxEdge::Margin, side)
    }

    /// Resolves this logical edge to a physical [`StyleProp`] under `direction`.
    #[must_use]
    pub fn to_physical(self, direction: Direction) -> StyleProp {
        prop_for(self.edge, self.side.to_physical(direction))
    }
}

/// Combines a property family and a physical side into a concrete [`StyleProp`].
fn prop_for(edge: BoxEdge, side: PhysicalSide) -> StyleProp {
    match (edge, side) {
        (BoxEdge::Padding, PhysicalSide::Top) => StyleProp::PaddingTop,
        (BoxEdge::Padding, PhysicalSide::Right) => StyleProp::PaddingRight,
        (BoxEdge::Padding, PhysicalSide::Bottom) => StyleProp::PaddingBottom,
        (BoxEdge::Padding, PhysicalSide::Left) => StyleProp::PaddingLeft,
        (BoxEdge::Margin, PhysicalSide::Top) => StyleProp::MarginTop,
        (BoxEdge::Margin, PhysicalSide::Right) => StyleProp::MarginRight,
        (BoxEdge::Margin, PhysicalSide::Bottom) => StyleProp::MarginBottom,
        (BoxEdge::Margin, PhysicalSide::Left) => StyleProp::MarginLeft,
    }
}

/// Mirrors a physical property across the inline axis for `direction`.
///
/// Under [`Direction::Ltr`] the property is returned unchanged. Under
/// [`Direction::Rtl`] the left and right members of the padding and margin
/// families are swapped; every other property (including the block-axis
/// top/bottom members) is left untouched.
#[must_use]
pub fn mirror_physical(prop: StyleProp, direction: Direction) -> StyleProp {
    if direction.is_ltr() {
        return prop;
    }
    match prop {
        StyleProp::PaddingLeft => StyleProp::PaddingRight,
        StyleProp::PaddingRight => StyleProp::PaddingLeft,
        StyleProp::MarginLeft => StyleProp::MarginRight,
        StyleProp::MarginRight => StyleProp::MarginLeft,
        other => other,
    }
}

/// Mirrors every entry of a physical [`PropMap`] for `direction`.
///
/// This flips an already-resolved sheet from one writing direction to the
/// other. Under [`Direction::Ltr`] the result is an exact clone.
#[must_use]
pub fn mirror_prop_map(props: &PropMap, direction: Direction) -> PropMap {
    let mut out = PropMap::new();
    for (prop, value) in props {
        out.insert(mirror_physical(*prop, direction), value.clone());
    }
    out
}

/// A set of logical box-edge spacings awaiting direction resolution.
///
/// Components author spacing here once, in logical terms, then call
/// [`LogicalSpacing::resolve`] with the active [`Direction`] to obtain a
/// physical [`PropMap`] ready for the cascade.
///
/// # Example
///
/// ```
/// use prism_ui_theme::{Direction, LogicalSide, LogicalSpacing};
/// use prism_ui_style::{StyleProp, StyleValue};
///
/// let spacing = LogicalSpacing::new()
///     .pad(LogicalSide::InlineStart, StyleValue::px(12.0))
///     .pad(LogicalSide::BlockStart, StyleValue::px(4.0));
///
/// let rtl = spacing.resolve(Direction::Rtl);
/// // Inline-start padding mirrored to the right edge; block-start unchanged.
/// assert_eq!(rtl.get(&StyleProp::PaddingRight), Some(&StyleValue::px(12.0)));
/// assert_eq!(rtl.get(&StyleProp::PaddingTop), Some(&StyleValue::px(4.0)));
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LogicalSpacing {
    edges: BTreeMap<LogicalEdge, StyleValue>,
}

impl LogicalSpacing {
    /// Creates an empty logical spacing set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            edges: BTreeMap::new(),
        }
    }

    /// Sets the value for a logical edge, replacing any existing value.
    pub fn set(&mut self, edge: LogicalEdge, value: StyleValue) {
        self.edges.insert(edge, value);
    }

    /// Sets a logical edge and returns `self` for chaining.
    #[must_use]
    pub fn with(mut self, edge: LogicalEdge, value: StyleValue) -> Self {
        self.set(edge, value);
        self
    }

    /// Sets a padding edge on `side` and returns `self` for chaining.
    #[must_use]
    pub fn pad(self, side: LogicalSide, value: StyleValue) -> Self {
        self.with(LogicalEdge::padding(side), value)
    }

    /// Sets a margin edge on `side` and returns `self` for chaining.
    #[must_use]
    pub fn margin(self, side: LogicalSide, value: StyleValue) -> Self {
        self.with(LogicalEdge::margin(side), value)
    }

    /// Resolves every logical edge to a physical [`PropMap`] under `direction`.
    #[must_use]
    pub fn resolve(&self, direction: Direction) -> PropMap {
        let mut out = PropMap::new();
        for (edge, value) in &self.edges {
            out.insert(edge.to_physical(direction), value.clone());
        }
        out
    }

    /// Returns the number of logical edges held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Returns `true` if no logical edges are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_predicates_and_flip() {
        assert!(Direction::Ltr.is_ltr());
        assert!(Direction::Rtl.is_rtl());
        assert_eq!(Direction::Ltr.flipped(), Direction::Rtl);
        assert_eq!(Direction::Rtl.flipped(), Direction::Ltr);
        assert_eq!(Direction::default(), Direction::Ltr);
    }

    #[test]
    fn inline_start_mirrors_with_direction() {
        let edge = LogicalEdge::padding(LogicalSide::InlineStart);
        assert_eq!(edge.to_physical(Direction::Ltr), StyleProp::PaddingLeft);
        assert_eq!(edge.to_physical(Direction::Rtl), StyleProp::PaddingRight);
    }

    #[test]
    fn inline_end_mirrors_with_direction() {
        let edge = LogicalEdge::margin(LogicalSide::InlineEnd);
        assert_eq!(edge.to_physical(Direction::Ltr), StyleProp::MarginRight);
        assert_eq!(edge.to_physical(Direction::Rtl), StyleProp::MarginLeft);
    }

    #[test]
    fn block_sides_never_mirror() {
        let start = LogicalEdge::padding(LogicalSide::BlockStart);
        let end = LogicalEdge::margin(LogicalSide::BlockEnd);
        assert_eq!(start.to_physical(Direction::Ltr), StyleProp::PaddingTop);
        assert_eq!(start.to_physical(Direction::Rtl), StyleProp::PaddingTop);
        assert_eq!(end.to_physical(Direction::Ltr), StyleProp::MarginBottom);
        assert_eq!(end.to_physical(Direction::Rtl), StyleProp::MarginBottom);
    }

    #[test]
    fn is_inline_classifies_sides() {
        assert!(LogicalSide::InlineStart.is_inline());
        assert!(LogicalSide::InlineEnd.is_inline());
        assert!(!LogicalSide::BlockStart.is_inline());
        assert!(!LogicalSide::BlockEnd.is_inline());
    }

    #[test]
    fn mirror_physical_is_identity_in_ltr() {
        assert_eq!(
            mirror_physical(StyleProp::PaddingLeft, Direction::Ltr),
            StyleProp::PaddingLeft,
        );
    }

    #[test]
    fn mirror_physical_swaps_inline_in_rtl() {
        assert_eq!(
            mirror_physical(StyleProp::PaddingLeft, Direction::Rtl),
            StyleProp::PaddingRight,
        );
        assert_eq!(
            mirror_physical(StyleProp::MarginRight, Direction::Rtl),
            StyleProp::MarginLeft,
        );
        // Block-axis and non-box props are untouched.
        assert_eq!(
            mirror_physical(StyleProp::PaddingTop, Direction::Rtl),
            StyleProp::PaddingTop,
        );
        assert_eq!(
            mirror_physical(StyleProp::Width, Direction::Rtl),
            StyleProp::Width,
        );
    }

    #[test]
    fn mirror_prop_map_flips_inline_entries() {
        let mut props = PropMap::new();
        props.insert(StyleProp::PaddingLeft, StyleValue::px(10.0));
        props.insert(StyleProp::PaddingTop, StyleValue::px(2.0));

        let ltr = mirror_prop_map(&props, Direction::Ltr);
        assert_eq!(ltr, props);

        let rtl = mirror_prop_map(&props, Direction::Rtl);
        assert_eq!(
            rtl.get(&StyleProp::PaddingRight),
            Some(&StyleValue::px(10.0))
        );
        assert_eq!(rtl.get(&StyleProp::PaddingTop), Some(&StyleValue::px(2.0)));
        assert_eq!(rtl.get(&StyleProp::PaddingLeft), None);
    }

    #[test]
    fn logical_spacing_resolves_both_directions() {
        let spacing = LogicalSpacing::new()
            .pad(LogicalSide::InlineStart, StyleValue::px(12.0))
            .margin(LogicalSide::InlineEnd, StyleValue::px(6.0))
            .pad(LogicalSide::BlockStart, StyleValue::px(4.0));
        assert_eq!(spacing.len(), 3);
        assert!(!spacing.is_empty());

        let ltr = spacing.resolve(Direction::Ltr);
        assert_eq!(
            ltr.get(&StyleProp::PaddingLeft),
            Some(&StyleValue::px(12.0))
        );
        assert_eq!(ltr.get(&StyleProp::MarginRight), Some(&StyleValue::px(6.0)));
        assert_eq!(ltr.get(&StyleProp::PaddingTop), Some(&StyleValue::px(4.0)));

        let rtl = spacing.resolve(Direction::Rtl);
        assert_eq!(
            rtl.get(&StyleProp::PaddingRight),
            Some(&StyleValue::px(12.0))
        );
        assert_eq!(rtl.get(&StyleProp::MarginLeft), Some(&StyleValue::px(6.0)));
        assert_eq!(rtl.get(&StyleProp::PaddingTop), Some(&StyleValue::px(4.0)));
    }

    #[test]
    fn empty_spacing_resolves_to_empty_map() {
        let spacing = LogicalSpacing::new();
        assert!(spacing.is_empty());
        assert_eq!(spacing.resolve(Direction::Ltr).len(), 0);
    }

    #[test]
    fn set_replaces_existing_edge() {
        let mut spacing = LogicalSpacing::new();
        spacing.set(
            LogicalEdge::padding(LogicalSide::InlineStart),
            StyleValue::px(1.0),
        );
        spacing.set(
            LogicalEdge::padding(LogicalSide::InlineStart),
            StyleValue::px(2.0),
        );
        assert_eq!(spacing.len(), 1);
        let ltr = spacing.resolve(Direction::Ltr);
        assert_eq!(ltr.get(&StyleProp::PaddingLeft), Some(&StyleValue::px(2.0)));
    }
}
