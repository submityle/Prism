//! [`Carousel`] — a horizontal strip of slides with optional indicators.
//!
//! A carousel renders a `pk-carousel` container holding a `pk-carousel__track`
//! of `pk-carousel__item` slides (the active one tagged
//! `pk-carousel__item--active`) and, when `show_indicators` is set, a row of
//! `pk-carousel__dot` indicators (the active one tagged
//! `pk-carousel__dot--active`). The actual scroll/translation of the track is
//! driven by the runtime layer; this control only encodes which slide is
//! active. The control attaches only kit class names; values resolve from theme
//! tokens via [`crate::preset`].

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Carousel`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CarouselProps {
    /// The slide contents, in order.
    pub items: Vec<Element>,
    /// The zero-based index of the active slide (clamped to the item range).
    pub active: usize,
    /// Whether to render the indicator dots.
    pub show_indicators: bool,
}

impl CarouselProps {
    /// Creates empty carousel props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a slide.
    #[must_use]
    pub fn item(mut self, element: Element) -> Self {
        self.items.push(element);
        self
    }

    /// Replaces the slides.
    #[must_use]
    pub fn items<I: IntoIterator<Item = Element>>(mut self, items: I) -> Self {
        self.items = items.into_iter().collect();
        self
    }

    /// Sets the active slide index.
    #[must_use]
    pub fn active(mut self, active: usize) -> Self {
        self.active = active;
        self
    }

    /// Toggles the indicator dots.
    #[must_use]
    pub fn show_indicators(mut self, show: bool) -> Self {
        self.show_indicators = show;
        self
    }
}

/// The carousel control. Zero-sized; config lives in [`CarouselProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Carousel;

impl Carousel {
    /// The active slide index, clamped to `0..items.len()` (0 when empty).
    #[must_use]
    pub fn active_index(props: &CarouselProps) -> usize {
        if props.items.is_empty() {
            0
        } else if props.active >= props.items.len() {
            props.items.len() - 1
        } else {
            props.active
        }
    }

    /// The accessibility role a carousel approximates.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for Carousel {
    type Props = CarouselProps;

    fn render(&self, props: &Self::Props) -> Element {
        let active = Carousel::active_index(props);
        let mut el = Element::box_().class("pk-carousel");

        // Track: the slides laid out in a row; the active one is flagged.
        let mut track = Element::box_().class("pk-carousel__track");
        for (index, item) in props.items.iter().enumerate() {
            let mut slide = item.clone().class("pk-carousel__item");
            if index == active {
                slide = slide.class("pk-carousel__item--active");
            }
            track = track.child(slide);
        }
        el = el.child(track);

        // Indicators: one dot per slide, active one highlighted.
        if props.show_indicators {
            let mut dots = Element::box_().class("pk-carousel__indicators");
            for index in 0..props.items.len() {
                let mut dot = Element::box_().class("pk-carousel__dot");
                if index == active {
                    dot = dot.class("pk-carousel__dot--active");
                }
                dots = dots.child(dot);
            }
            el = el.child(dots);
        }
        el
    }
}

/// Registers the `pk-carousel` class family: container, track, item (+active)
/// and the indicator dots (+active).
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Container: a vertical stack of the track and its indicators.
    sheet.insert(
        Class::new("pk-carousel")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    // Track: the row of slides.
    sheet.insert(
        Class::new("pk-carousel__track")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Stretch))
            .with(StyleProp::Gap, tok("space.md"))
            .with(StyleProp::BorderRadius, tok("radius.lg")),
    );

    // Item: a full-basis slide that neither grows nor shrinks away.
    sheet.insert(
        Class::new("pk-carousel__item")
            .with(StyleProp::FlexGrow, StyleValue::number(0.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::FlexBasis, StyleValue::percent(100.0))
            .with(StyleProp::Opacity, StyleValue::number(0.5)),
    );

    // Active item: fully opaque.
    sheet.insert(
        Class::new("pk-carousel__item--active").with(StyleProp::Opacity, StyleValue::number(1.0)),
    );

    // Indicators: a centered row of dots.
    sheet.insert(
        Class::new("pk-carousel__indicators")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs")),
    );

    // Dot: a small inactive capsule.
    sheet.insert(
        Class::new("pk-carousel__dot")
            .with(StyleProp::Width, StyleValue::px(8.0))
            .with(StyleProp::Height, StyleValue::px(8.0))
            .with(StyleProp::MinWidth, StyleValue::px(8.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    // Active dot: the accent tint.
    sheet.insert(
        Class::new("pk-carousel__dot--active").with(StyleProp::BackgroundColor, tok("color.tint")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: CarouselProps) -> Element {
        Carousel.render(&props)
    }

    fn slides(el: &Element) -> &[Element] {
        el.child_elements()[0].child_elements()
    }

    #[test]
    fn empty_carousel_has_empty_track() {
        let el = render(CarouselProps::new());
        assert_eq!(el.class_names(), ["pk-carousel"]);
        assert_eq!(el.child_elements().len(), 1);
        assert!(slides(&el).is_empty());
    }

    #[test]
    fn active_slide_is_flagged() {
        let el = render(
            CarouselProps::new()
                .items([Element::box_(), Element::box_(), Element::box_()])
                .active(1),
        );
        let items = slides(&el);
        assert_eq!(items.len(), 3);
        assert!(!items[0].class_names().iter().any(|n| n == "pk-carousel__item--active"));
        assert!(items[1].class_names().iter().any(|n| n == "pk-carousel__item--active"));
        assert!(!items[2].class_names().iter().any(|n| n == "pk-carousel__item--active"));
    }

    #[test]
    fn active_index_clamps_out_of_range() {
        let props = CarouselProps::new()
            .items([Element::box_(), Element::box_()])
            .active(9);
        assert_eq!(Carousel::active_index(&props), 1);
        assert_eq!(Carousel::active_index(&CarouselProps::new()), 0);
    }

    #[test]
    fn indicators_render_one_dot_per_slide() {
        let el = render(
            CarouselProps::new()
                .items([Element::box_(), Element::box_()])
                .active(0)
                .show_indicators(true),
        );
        assert_eq!(el.child_elements().len(), 2);
        let dots = el.child_elements()[1].child_elements();
        assert_eq!(dots.len(), 2);
        assert!(dots[0].class_names().iter().any(|n| n == "pk-carousel__dot--active"));
        assert!(!dots[1].class_names().iter().any(|n| n == "pk-carousel__dot--active"));
    }

    #[test]
    fn indicators_hidden_by_default() {
        let el = render(CarouselProps::new().items([Element::box_()]));
        assert_eq!(el.child_elements().len(), 1);
    }

    #[test]
    fn role_is_group() {
        assert_eq!(Carousel::role(), Role::Group);
    }
}
