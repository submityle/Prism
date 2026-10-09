//! [`Popconfirm`] — an inline confirm/cancel bubble on the overlay base.
//!
//! A popconfirm asks a short `message` and offers a confirm and a cancel
//! action, tinted by a semantic [`Tone`]. It composes the shared [`Popover`]
//! base per architecture invariant K7 and attaches only kit classes;
//! [`crate::preset`] resolves the tone accent and glass surface against the
//! active theme.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::feedback::popover::{Popover, PopoverProps};
use crate::kit::{classes, Tone};
use crate::preset::StyleSheet;

/// Props for [`Popconfirm`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PopconfirmProps {
    /// The confirmation prompt.
    pub message: String,
    /// The confirm button label.
    pub confirm_label: String,
    /// The cancel button label.
    pub cancel_label: String,
    /// The semantic tone selecting the confirm accent.
    pub tone: Tone,
    /// Whether the bubble is currently shown.
    pub open: bool,
}

impl PopconfirmProps {
    /// Creates popconfirm props with `message` and the default (accent) tone.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            ..Self::default()
        }
    }

    /// Sets the confirm button label.
    #[must_use]
    pub fn confirm_label(mut self, label: impl Into<String>) -> Self {
        self.confirm_label = label.into();
        self
    }

    /// Sets the cancel button label.
    #[must_use]
    pub fn cancel_label(mut self, label: impl Into<String>) -> Self {
        self.cancel_label = label.into();
        self
    }

    /// Sets the semantic tone.
    #[must_use]
    pub fn tone(mut self, tone: Tone) -> Self {
        self.tone = tone;
        self
    }

    /// Sets whether the bubble is shown.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }
}

/// The popconfirm control. Zero-sized; configuration lives in
/// [`PopconfirmProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Popconfirm;

impl Component for Popconfirm {
    type Props = PopconfirmProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut bubble = Element::box_();
        for name in classes("pk-popconfirm", &[props.tone.suffix()]) {
            bubble = bubble.class(name);
        }
        bubble = bubble.child(Element::text(props.message.clone()).class("pk-popconfirm__message"));

        let actions = Element::box_()
            .class("pk-popconfirm__actions")
            .child(Element::text(props.cancel_label.clone()).class("pk-popconfirm__cancel"))
            .child(Element::text(props.confirm_label.clone()).class("pk-popconfirm__confirm"));
        bubble = bubble.child(actions);

        Popover.render(&PopoverProps::new().open(props.open).child(bubble))
    }
}

/// Builds the `--tone` modifier name for a block, e.g. `pk-popconfirm--danger`.
fn tone_class(block: &str, tone: Tone) -> String {
    let mut name = String::with_capacity(block.len() + 2 + tone.suffix().len());
    name.push_str(block);
    name.push_str("--");
    name.push_str(tone.suffix());
    name
}

/// Every tone, for exhaustive style registration.
const TONES: [Tone; 5] = [
    Tone::Accent,
    Tone::Neutral,
    Tone::Success,
    Tone::Warning,
    Tone::Danger,
];

/// Registers the `pk-popconfirm` class family: the bubble, per-tone confirm
/// accent, the message run and the trailing actions row.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Bubble: a compact vertical stack sized for a short prompt.
    sheet.insert(
        Class::new("pk-popconfirm")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::MaxWidth, StyleValue::px(260.0)),
    );

    // Tones paint the confirm accent used by the confirm button.
    for tone in TONES {
        sheet.insert(
            Class::new(tone_class("pk-popconfirm", tone))
                .with(StyleProp::Color, tok(tone.color_token())),
        );
    }

    // Message: readable body prompt.
    sheet.insert(
        Class::new("pk-popconfirm__message")
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Actions: a trailing row of cancel/confirm controls.
    sheet.insert(
        Class::new("pk-popconfirm__actions")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::JustifyContent, kw(Keyword::End))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    // Cancel: a quiet, neutral label.
    sheet.insert(
        Class::new("pk-popconfirm__cancel")
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::FontWeight, tok("font.weight.regular"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Confirm: emphasised; inherits the tone accent from the bubble.
    sheet.insert(
        Class::new("pk-popconfirm__confirm")
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: PopconfirmProps) -> Element {
        Popconfirm.render(&props)
    }

    fn bubble(el: &Element) -> Element {
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
        let el = render(PopconfirmProps::new("Delete?").open(true));
        assert!(el.class_names().iter().any(|c| c == "pk-popover"));
        assert!(el.class_names().iter().any(|c| c == "is-open"));
    }

    #[test]
    fn attaches_tone_modifier() {
        let el = render(PopconfirmProps::new("Delete?").tone(Tone::Danger));
        let bubble = bubble(&el);
        assert!(bubble.class_names().iter().any(|c| c == "pk-popconfirm"));
        assert!(bubble.class_names().iter().any(|c| c == "pk-popconfirm--danger"));
    }

    #[test]
    fn renders_message_then_actions() {
        let el = render(
            PopconfirmProps::new("Delete this item?")
                .cancel_label("Cancel")
                .confirm_label("Delete"),
        );
        let bubble = bubble(&el);
        let kids = bubble.child_elements();
        assert!(kids[0].class_names().iter().any(|c| c == "pk-popconfirm__message"));
        assert_eq!(kids[0].text_content(), Some("Delete this item?"));
        let actions = &kids[1];
        assert!(actions.class_names().iter().any(|c| c == "pk-popconfirm__actions"));
        let buttons = actions.child_elements();
        assert!(buttons[0].class_names().iter().any(|c| c == "pk-popconfirm__cancel"));
        assert!(buttons[1].class_names().iter().any(|c| c == "pk-popconfirm__confirm"));
        assert_eq!(buttons[1].text_content(), Some("Delete"));
    }
}
