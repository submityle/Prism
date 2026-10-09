//! [`Tour`] — a guided onboarding coachmark on the overlay base.
//!
//! A tour walks through a list of [`TourStep`]s one at a time, highlighting the
//! `current` step's `title` and `body` and showing a `current / total`
//! progress footer. It composes the shared [`Popover`] base per architecture
//! invariant K7 and attaches only kit classes; [`crate::preset`] resolves every
//! value against the active theme. The [`Coachmark`] alias names the same
//! control.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::feedback::popover::{Popover, PopoverProps};
use crate::preset::StyleSheet;

/// A single step in a [`Tour`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TourStep {
    /// The step's heading.
    pub title: String,
    /// The step's explanatory body.
    pub body: String,
}

impl TourStep {
    /// Creates a step with `title` and `body`.
    #[must_use]
    pub fn new(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            body: body.into(),
        }
    }
}

/// Props for [`Tour`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TourProps {
    /// The ordered steps.
    pub steps: Vec<TourStep>,
    /// The zero-based index of the currently shown step.
    pub current: usize,
    /// Whether the tour is currently shown.
    pub open: bool,
}

impl TourProps {
    /// Creates empty, closed tour props at the first step.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a step.
    #[must_use]
    pub fn step(mut self, step: TourStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Replaces the steps with `steps`.
    #[must_use]
    pub fn steps<I: IntoIterator<Item = TourStep>>(mut self, steps: I) -> Self {
        self.steps = steps.into_iter().collect();
        self
    }

    /// Sets the zero-based current step index.
    #[must_use]
    pub fn current(mut self, current: usize) -> Self {
        self.current = current;
        self
    }

    /// Sets whether the tour is shown.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }
}

/// The tour control. Zero-sized; configuration lives in [`TourProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Tour;

/// Alias for [`Tour`] under its common "coachmark" name.
pub type Coachmark = Tour;

impl Component for Tour {
    type Props = TourProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut card = Element::box_().class("pk-tour");

        if let Some(step) = props.steps.get(props.current) {
            card = card.child(Element::text(step.title.clone()).class("pk-tour__title"));
            card = card.child(Element::text(step.body.clone()).class("pk-tour__body"));
        }

        // Footer: a human `current / total` progress indicator.
        let shown = if props.steps.is_empty() {
            0
        } else {
            props.current + 1
        };
        let mut progress = String::new();
        progress.push_str(&shown.to_string());
        progress.push_str(" / ");
        progress.push_str(&props.steps.len().to_string());
        card = card.child(Element::text(progress).class("pk-tour__footer"));

        Popover.render(&PopoverProps::new().open(props.open).child(card))
    }
}

/// Registers the `pk-tour` class family: the coachmark card, the step title
/// and body, and the progress footer.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Card: a vertical stack sized for one step.
    sheet.insert(
        Class::new("pk-tour")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::MinWidth, StyleValue::px(240.0))
            .with(StyleProp::MaxWidth, StyleValue::px(320.0)),
    );

    // Title: the step heading.
    sheet.insert(
        Class::new("pk-tour__title")
            .with(StyleProp::FontSize, tok("font.size.headline"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Body: the step's explanatory prose.
    sheet.insert(
        Class::new("pk-tour__body")
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Footer: a quiet progress indicator pinned to the trailing edge.
    sheet.insert(
        Class::new("pk-tour__footer")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::JustifyContent, kw(Keyword::End))
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::Color, tok("color.label.tertiary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: TourProps) -> Element {
        Tour.render(&props)
    }

    fn card(el: &Element) -> Element {
        let surface = el
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-popover__surface"))
            .expect("surface")
            .clone();
        surface.child_elements()[0].clone()
    }

    #[test]
    fn composes_popover_base() {
        let el = render(TourProps::new().open(true));
        assert!(el.class_names().iter().any(|c| c == "pk-popover"));
        assert!(el.class_names().iter().any(|c| c == "is-open"));
    }

    #[test]
    fn shows_the_current_step() {
        let el = render(
            TourProps::new()
                .step(TourStep::new("Step 1", "First"))
                .step(TourStep::new("Step 2", "Second"))
                .current(1),
        );
        let card = card(&el);
        let kids = card.child_elements();
        assert!(kids[0].class_names().iter().any(|c| c == "pk-tour__title"));
        assert_eq!(kids[0].text_content(), Some("Step 2"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-tour__body"));
        assert_eq!(kids[1].text_content(), Some("Second"));
    }

    #[test]
    fn footer_reports_progress() {
        let el = render(
            TourProps::new()
                .step(TourStep::new("a", "x"))
                .step(TourStep::new("b", "y"))
                .step(TourStep::new("c", "z"))
                .current(0),
        );
        let card = card(&el);
        let footer = card.child_elements();
        let footer = footer.last().unwrap();
        assert!(footer.class_names().iter().any(|c| c == "pk-tour__footer"));
        assert_eq!(footer.text_content(), Some("1 / 3"));
    }

    #[test]
    fn empty_tour_shows_zero_progress() {
        let el = render(TourProps::new());
        let card = card(&el);
        let footer = card.child_elements();
        let footer = footer.last().unwrap();
        assert_eq!(footer.text_content(), Some("0 / 0"));
    }

    #[test]
    fn coachmark_alias_is_the_same_control() {
        let el = Coachmark::default().render(&TourProps::new().open(true));
        assert!(el.class_names().iter().any(|c| c == "pk-popover"));
    }
}
