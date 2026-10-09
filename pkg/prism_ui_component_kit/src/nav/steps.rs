//! [`Steps`] — a horizontal progress indicator for multi-stage flows.
//!
//! A steps control renders a `pk-steps` row of `pk-steps__step` entries. A
//! completed step gains the shared `is-done` state class; the step at the
//! zero-based `current` index gains the shared `is-current` state class. Only
//! kit class names are attached; color and spacing resolve from theme tokens.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// A single step: a title plus whether it has been completed.
#[derive(Clone, Debug, PartialEq)]
pub struct StepItem {
    /// The step's title text.
    pub title: String,
    /// Whether this step is complete.
    pub done: bool,
}

impl StepItem {
    /// Creates an incomplete step with the given title.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            done: false,
        }
    }

    /// Sets whether the step is complete.
    #[must_use]
    pub fn done(mut self, done: bool) -> Self {
        self.done = done;
        self
    }
}

/// Props for [`Steps`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct StepsProps {
    /// The steps, rendered left-to-right.
    pub steps: Vec<StepItem>,
    /// The zero-based index of the current (active) step.
    pub current: usize,
}

impl StepsProps {
    /// Creates empty steps props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a step.
    #[must_use]
    pub fn step(mut self, step: StepItem) -> Self {
        self.steps.push(step);
        self
    }

    /// Replaces the steps with `steps`.
    #[must_use]
    pub fn steps<I: IntoIterator<Item = StepItem>>(mut self, steps: I) -> Self {
        self.steps = steps.into_iter().collect();
        self
    }

    /// Sets the current step index.
    #[must_use]
    pub fn current(mut self, current: usize) -> Self {
        self.current = current;
        self
    }
}

/// The steps control. Zero-sized; config lives in [`StepsProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Steps;

impl Component for Steps {
    type Props = StepsProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-steps");
        for (index, step) in props.steps.iter().enumerate() {
            let mut entry = Element::box_()
                .class("pk-steps__step")
                .child(Element::text(step.title.clone()).class("pk-steps__title"));
            if step.done {
                entry = entry.class("is-done");
            }
            if index == props.current {
                entry = entry.class("is-current");
            }
            el = el.child(entry);
        }
        el
    }
}

/// Registers the `pk-steps` class family: the row, the step entries and the
/// shared done / current states.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Row: a flex row spacing steps evenly.
    sheet.insert(
        Class::new("pk-steps")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.md")),
    );

    // Step: a muted entry by default.
    sheet.insert(
        Class::new("pk-steps__step")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::Color, tok("color.label.tertiary")),
    );

    // Title: body typography.
    sheet.insert(
        Class::new("pk-steps__title").with(StyleProp::FontSize, tok("font.size.subheadline")),
    );

    // Shared done state: completed steps read in the success tone.
    sheet.insert(
        Class::new("is-done").with(StyleProp::Color, tok("color.green")),
    );

    // Shared current state: an accent-tinted surface with an accent label.
    // Shared verbatim with the sibling control so the state class resolves
    // consistently no matter which registrar inserts it last.
    sheet.insert(
        Class::new("is-current")
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: StepsProps) -> Element {
        Steps.render(&props)
    }

    #[test]
    fn empty_steps_has_no_entries() {
        let el = render(StepsProps::new());
        assert_eq!(el.class_names(), ["pk-steps"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn done_and_current_states_attach() {
        let el = render(
            StepsProps::new()
                .step(StepItem::new("One").done(true))
                .step(StepItem::new("Two"))
                .current(1),
        );
        let entries = el.child_elements();
        assert_eq!(entries[0].class_names(), ["pk-steps__step", "is-done"]);
        assert_eq!(entries[1].class_names(), ["pk-steps__step", "is-current"]);
    }

    #[test]
    fn step_title_renders_as_text_child() {
        let el = render(StepsProps::new().step(StepItem::new("Review")));
        let entry = &el.child_elements()[0];
        assert!(entry.child_elements()[0]
            .class_names()
            .iter()
            .any(|c| c == "pk-steps__title"));
        assert_eq!(entry.child_elements()[0].text_content(), Some("Review"));
    }
}
