//! [`Mentions`] — a text input with an `@`-mention suggestion dropdown.
//!
//! A mentions control renders a `pk-mentions` surface pairing a `__input` slot
//! (the current value) with a suggestion dropdown of `__suggestion`s. The
//! dropdown is composed from the shared [`Popover`] overlay base rather than a
//! bespoke surface, per the kit's single-overlay invariant. All color comes
//! from theme tokens via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::feedback::{Popover, PopoverProps};
use crate::preset::StyleSheet;

/// Props for [`Mentions`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MentionsProps {
    /// The current text value.
    pub value: String,
    /// The mention suggestions offered in the dropdown.
    pub suggestions: Vec<String>,
    /// Whether the suggestion dropdown is open.
    pub open: bool,
}

impl MentionsProps {
    /// Creates empty, closed mentions props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the current value.
    #[must_use]
    pub fn value(mut self, value: impl Into<String>) -> Self {
        self.value = value.into();
        self
    }

    /// Replaces the suggestions with `suggestions`.
    #[must_use]
    pub fn suggestions<I, S>(mut self, suggestions: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.suggestions = suggestions.into_iter().map(Into::into).collect();
        self
    }

    /// Sets whether the dropdown is open.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }
}

/// The mentions control. Zero-sized; config lives in [`MentionsProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Mentions;

impl Mentions {
    /// The accessibility role a mentions control exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Textbox
    }
}

impl Component for Mentions {
    type Props = MentionsProps;

    fn render(&self, props: &Self::Props) -> Element {
        let input = Element::text(props.value.clone()).class("pk-mentions__input");

        let suggestions: Vec<Element> = props
            .suggestions
            .iter()
            .map(|s| Element::text(s.clone()).class("pk-mentions__suggestion"))
            .collect();
        let popover = Popover.render(&PopoverProps::new().open(props.open).children(suggestions));

        Element::box_().class("pk-mentions").child(input).child(popover)
    }
}

/// Registers the `pk-mentions` class family: base surface, input slot and the
/// suggestion rows.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-mentions")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::MinHeight, tok("size.control.height"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_glass(0.0, tok("color.surface.secondary"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Focus, StyleProp::BorderColor, tok("color.tint")),
    );

    sheet.insert(
        Class::new("pk-mentions__input")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );

    sheet.insert(
        Class::new("pk-mentions__suggestion")
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: MentionsProps) -> Element {
        Mentions.render(&props)
    }

    #[test]
    fn input_slot_carries_the_value() {
        let el = render(MentionsProps::new().value("@al"));
        let input = &el.child_elements()[0];
        assert_eq!(input.text_content(), Some("@al"));
        assert_eq!(input.class_names(), ["pk-mentions__input"]);
    }

    #[test]
    fn suggestions_render_in_dropdown() {
        let el = render(
            MentionsProps::new()
                .value("@a")
                .suggestions(["alice", "alex"])
                .open(true),
        );
        let popover = el.child_elements().last().expect("popover");
        let suggestions: Vec<_> = popover
            .child_elements()
            .iter()
            .flat_map(Element::child_elements)
            .filter(|c| c.class_names().iter().any(|n| n == "pk-mentions__suggestion"))
            .collect();
        assert_eq!(suggestions.len(), 2);
        assert_eq!(suggestions[0].text_content(), Some("alice"));
    }

    #[test]
    fn role_is_textbox() {
        assert_eq!(Mentions::role(), Role::Textbox);
    }
}
