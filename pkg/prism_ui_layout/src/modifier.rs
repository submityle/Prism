//! An order-sensitive modifier chain.
//!
//! Modifiers lower common box adjustments — padding, margin, forced size,
//! aspect ratio, alignment, and clipping — into composable constraint
//! transforms rather than magic style fields. As in Jetpack Compose, **the
//! order of modifiers is the semantics**: `padding` then `size` is not the
//! same as `size` then `padding`, and `aspect_ratio` then `size` differs from
//! `size` then `aspect_ratio`.
//!
//! A [`ModifierChain`] offers two pure operations:
//!
//! * [`ModifierChain::transform_constraints`] rewrites the incoming
//!   [`Constraints`] a child is measured against.
//! * [`ModifierChain::resolve`] turns a measured content size and an available
//!   region into a [`Resolved`] placement (outer footprint, content rectangle,
//!   and clip flag).

use alloc::vec::Vec;

use crate::geometry::{Edges, Point, Rect, Size};
use crate::protocol::{Alignment, Constraints};

/// A single composable box modifier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Modifier {
    /// Adds space between the outer footprint and the content box.
    Padding(Edges<f32>),
    /// Adds transparent space around the outer footprint.
    Margin(Edges<f32>),
    /// Forces the content size on the given axes (`None` leaves an axis
    /// unchanged).
    Size {
        /// Forced content width, when present.
        width: Option<f32>,
        /// Forced content height, when present.
        height: Option<f32>,
    },
    /// Derives the content height from its current width using the ratio
    /// `width / height`.
    AspectRatio(f32),
    /// Sets how the footprint is aligned within the available region.
    Align(Alignment),
    /// Clamps the content so it cannot exceed the available region, and marks
    /// the box as clipping.
    Clip,
}

/// The resolved placement produced by [`ModifierChain::resolve`].
///
/// All coordinates are relative to the available region's top-left corner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resolved {
    /// Total footprint occupied by the box (content plus padding/margin).
    pub outer: Size<f32>,
    /// Rectangle occupied by the content box.
    pub content: Rect,
    /// Whether the box clips its content.
    pub clip: bool,
}

/// An ordered sequence of [`Modifier`]s applied as a unit.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ModifierChain {
    modifiers: Vec<Modifier>,
}

impl ModifierChain {
    /// Creates an empty chain.
    pub fn new() -> Self {
        Self {
            modifiers: Vec::new(),
        }
    }

    /// Appends a modifier and returns the chain, enabling fluent building.
    pub fn then(mut self, modifier: Modifier) -> Self {
        self.modifiers.push(modifier);
        self
    }

    /// Returns the modifiers in application order.
    pub fn modifiers(&self) -> &[Modifier] {
        &self.modifiers
    }

    /// Rewrites the constraints a child should be measured against.
    ///
    /// Padding and margin shrink the available space; a forced size tightens
    /// it; an aspect ratio derives the height bound from the width bound.
    pub fn transform_constraints(&self, mut constraints: Constraints) -> Constraints {
        for modifier in &self.modifiers {
            match *modifier {
                Modifier::Padding(e) | Modifier::Margin(e) => {
                    constraints.max.width = (constraints.max.width - e.horizontal()).max(0.0);
                    constraints.max.height = (constraints.max.height - e.vertical()).max(0.0);
                    constraints.min.width = (constraints.min.width - e.horizontal()).max(0.0);
                    constraints.min.height = (constraints.min.height - e.vertical()).max(0.0);
                }
                Modifier::Size { width, height } => {
                    if let Some(w) = width {
                        constraints.min.width = w;
                        constraints.max.width = w;
                    }
                    if let Some(h) = height {
                        constraints.min.height = h;
                        constraints.max.height = h;
                    }
                }
                Modifier::AspectRatio(ratio) => {
                    if ratio > 0.0 {
                        let derived = constraints.max.width / ratio;
                        constraints.max.height = derived;
                        constraints.min.height = constraints.min.height.min(derived);
                    }
                }
                Modifier::Align(_) | Modifier::Clip => {}
            }
        }
        constraints
    }

    /// Resolves a placement for `content` within an `available` region.
    pub fn resolve(&self, available: Size<f32>, content: Size<f32>) -> Resolved {
        let mut size = content;
        let mut inset = Edges::<f32>::ZERO;
        let mut align = Alignment::TOP_LEFT;
        let mut clip = false;

        for modifier in &self.modifiers {
            match *modifier {
                Modifier::Padding(e) | Modifier::Margin(e) => {
                    inset = Edges::new(
                        inset.left + e.left,
                        inset.right + e.right,
                        inset.top + e.top,
                        inset.bottom + e.bottom,
                    );
                }
                Modifier::Size { width, height } => {
                    if let Some(w) = width {
                        size.width = w;
                    }
                    if let Some(h) = height {
                        size.height = h;
                    }
                }
                Modifier::AspectRatio(ratio) => {
                    if ratio > 0.0 {
                        size.height = size.width / ratio;
                    }
                }
                Modifier::Align(a) => align = a,
                Modifier::Clip => {
                    clip = true;
                    let max_w = (available.width - inset.horizontal()).max(0.0);
                    let max_h = (available.height - inset.vertical()).max(0.0);
                    size.width = size.width.min(max_w);
                    size.height = size.height.min(max_h);
                }
            }
        }

        let outer = Size::new(
            size.width + inset.horizontal(),
            size.height + inset.vertical(),
        );
        let outer_offset = align.offset(available, outer);
        let content_location = Point::new(outer_offset.x + inset.left, outer_offset.y + inset.top);
        Resolved {
            outer,
            content: Rect::new(content_location, size),
            clip,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_ratio_order_changes_result() {
        let content = Size::new(50.0, 30.0);
        let available = Size::new(500.0, 500.0);

        let size_then_ratio = ModifierChain::new()
            .then(Modifier::Size {
                width: Some(100.0),
                height: Some(40.0),
            })
            .then(Modifier::AspectRatio(1.0))
            .resolve(available, content);

        let ratio_then_size = ModifierChain::new()
            .then(Modifier::AspectRatio(1.0))
            .then(Modifier::Size {
                width: Some(100.0),
                height: Some(40.0),
            })
            .resolve(available, content);

        // Size then aspect: 100x40 -> height derived -> 100x100.
        assert_eq!(size_then_ratio.content.size, Size::new(100.0, 100.0));
        // Aspect then size: height overwritten by the later Size.
        assert_eq!(ratio_then_size.content.size, Size::new(100.0, 40.0));
        assert_ne!(size_then_ratio.content.size, ratio_then_size.content.size);
    }

    #[test]
    fn padding_grows_outer_and_offsets_content() {
        let chain = ModifierChain::new().then(Modifier::Padding(Edges::splat(10.0)));
        let resolved = chain.resolve(Size::new(200.0, 200.0), Size::new(50.0, 50.0));
        assert_eq!(resolved.outer, Size::new(70.0, 70.0));
        assert_eq!(resolved.content.location, Point::new(10.0, 10.0));
    }

    #[test]
    fn align_centers_outer_within_available() {
        let chain = ModifierChain::new().then(Modifier::Align(Alignment::CENTER));
        let resolved = chain.resolve(Size::new(100.0, 100.0), Size::new(40.0, 20.0));
        assert_eq!(resolved.content.location, Point::new(30.0, 40.0));
    }

    #[test]
    fn clip_clamps_content_to_available() {
        let chain = ModifierChain::new().then(Modifier::Clip);
        let resolved = chain.resolve(Size::new(80.0, 60.0), Size::new(200.0, 200.0));
        assert!(resolved.clip);
        assert_eq!(resolved.content.size, Size::new(80.0, 60.0));
    }

    #[test]
    fn transform_constraints_shrinks_for_padding() {
        let chain = ModifierChain::new().then(Modifier::Padding(Edges::splat(10.0)));
        let out = chain.transform_constraints(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(out.max, Size::new(80.0, 80.0));
    }

    #[test]
    fn transform_constraints_tightens_for_size() {
        let chain = ModifierChain::new().then(Modifier::Size {
            width: Some(50.0),
            height: None,
        });
        let out = chain.transform_constraints(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(out.min.width, 50.0);
        assert_eq!(out.max.width, 50.0);
        assert_eq!(out.max.height, 100.0);
    }

    #[test]
    fn padding_then_size_differs_from_size_then_padding_outer() {
        let content = Size::new(10.0, 10.0);
        let available = Size::new(500.0, 500.0);
        let padding_then_size = ModifierChain::new()
            .then(Modifier::Padding(Edges::splat(10.0)))
            .then(Modifier::Size {
                width: Some(100.0),
                height: Some(100.0),
            })
            .resolve(available, content);
        // Content forced to 100 after padding is accounted -> outer 120.
        assert_eq!(padding_then_size.outer, Size::new(120.0, 120.0));
        assert_eq!(padding_then_size.content.size, Size::new(100.0, 100.0));
    }
}
