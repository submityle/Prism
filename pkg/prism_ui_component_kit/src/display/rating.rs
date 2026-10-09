//! [`Rating`] — a read-only star rating.
//!
//! A rating renders `max` star children in a row; a star is *filled* when its
//! 1-based position is within `value`, otherwise *empty*. The `value` is
//! clamped to `0..=max` so out-of-range props still render a sensible row.
//! Colors come from theme tokens via [`crate::preset`]; the control attaches
//! only `pk-rating` class names.

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Rating`].
#[derive(Clone, Debug, PartialEq)]
pub struct RatingProps {
    /// The current rating value (number of filled stars).
    pub value: f32,
    /// The total number of stars to render.
    pub max: u8,
}

impl Default for RatingProps {
    fn default() -> Self {
        Self { value: 0.0, max: 5 }
    }
}

impl RatingProps {
    /// Creates props for a rating out of five stars.
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self {
            value,
            ..Self::default()
        }
    }

    /// Sets the maximum number of stars.
    #[must_use]
    pub fn max(mut self, max: u8) -> Self {
        self.max = max;
        self
    }
}

/// The rating control. Zero-sized; all configuration lives in [`RatingProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Rating;

impl Rating {
    /// The number of filled stars for `props`, clamped to `0..=max`.
    #[must_use]
    pub fn filled_count(props: &RatingProps) -> u8 {
        if props.value <= 0.0 {
            return 0;
        }
        // Round to the nearest whole star, then clamp to the available range.
        let rounded = round_to_nearest(props.value);
        if rounded >= f32::from(props.max) {
            props.max
        } else {
            rounded as u8
        }
    }
}

/// `no_std`-friendly round-half-up for non-negative inputs.
fn round_to_nearest(value: f32) -> f32 {
    // `f32::round` is unavailable in core; emulate for the non-negative domain
    // this control operates in.
    let truncated = value as u32 as f32;
    if value - truncated >= 0.5 {
        truncated + 1.0
    } else {
        truncated
    }
}

impl Component for Rating {
    type Props = RatingProps;

    fn render(&self, props: &Self::Props) -> Element {
        let filled = Rating::filled_count(props);
        let mut el = Element::box_().class("pk-rating");
        for i in 0..props.max {
            let modifier = if i < filled {
                "pk-rating__star--filled"
            } else {
                "pk-rating__star--empty"
            };
            let star = Element::text("★")
                .class("pk-rating__star")
                .class(modifier);
            el = el.child(star);
        }
        el
    }
}

/// Registers the `pk-rating` class family: container plus filled/empty stars.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Container: a tight horizontal row of stars.
    sheet.insert(
        Class::new("pk-rating")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xxs")),
    );

    // Star base: body-sized glyph.
    sheet.insert(
        Class::new("pk-rating__star").with(StyleProp::FontSize, tok("font.size.body")),
    );

    // Filled: the accent tint.
    sheet.insert(
        Class::new("pk-rating__star--filled").with(StyleProp::Color, tok("color.tint")),
    );

    // Empty: a quaternary/tertiary gray.
    sheet.insert(
        Class::new("pk-rating__star--empty").with(StyleProp::Color, tok("color.separator.opaque")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn render(props: RatingProps) -> Element {
        Rating.render(&props)
    }

    fn modifiers(el: &Element) -> Vec<bool> {
        el.child_elements()
            .iter()
            .map(|star| star.class_names().iter().any(|c| c == "pk-rating__star--filled"))
            .collect()
    }

    #[test]
    fn renders_max_stars() {
        let el = render(RatingProps::new(3.0).max(5));
        assert_eq!(el.class_names(), ["pk-rating"]);
        assert_eq!(el.child_elements().len(), 5);
    }

    #[test]
    fn fills_the_first_n_stars() {
        let el = render(RatingProps::new(3.0).max(5));
        assert_eq!(modifiers(&el), [true, true, true, false, false]);
    }

    #[test]
    fn rounds_to_nearest_star() {
        assert_eq!(Rating::filled_count(&RatingProps::new(3.5).max(5)), 4);
        assert_eq!(Rating::filled_count(&RatingProps::new(3.4).max(5)), 3);
    }

    #[test]
    fn clamps_out_of_range_values() {
        assert_eq!(Rating::filled_count(&RatingProps::new(-1.0).max(5)), 0);
        assert_eq!(Rating::filled_count(&RatingProps::new(9.0).max(5)), 5);
    }

    #[test]
    fn stars_carry_base_class() {
        let el = render(RatingProps::new(1.0).max(2));
        for star in el.child_elements() {
            assert!(star.class_names().iter().any(|c| c == "pk-rating__star"));
        }
    }
}
