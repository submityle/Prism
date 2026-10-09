//! [`Toast`] — a brief, self-dismissing status pill.
//!
//! A toast is a single `message` plus a semantic [`Tone`]. It renders a
//! `pk-toast` glass pill carrying only kit classes; [`crate::preset`] resolves
//! the tone's accent and the glass surface against the active theme.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::{classes, Tone};
use crate::preset::StyleSheet;

/// Props for [`Toast`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ToastProps {
    /// The short status message.
    pub message: String,
    /// The semantic tone selecting the accent.
    pub tone: Tone,
}

impl ToastProps {
    /// Creates toast props with the default (accent) tone.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            ..Self::default()
        }
    }

    /// Sets the semantic tone.
    #[must_use]
    pub fn tone(mut self, tone: Tone) -> Self {
        self.tone = tone;
        self
    }
}

/// The toast control. Zero-sized; all configuration lives in [`ToastProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Toast;

impl Component for Toast {
    type Props = ToastProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-toast", &[props.tone.suffix()]) {
            el = el.class(name);
        }
        el.child(Element::text(props.message.clone()).class("pk-toast__message"))
    }
}

/// Builds the `--tone` modifier name for a block, e.g. `pk-toast--success`.
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

/// Registers the `pk-toast` class family: the glass pill base, per-tone accent
/// text, and the message run.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Base: a capsule glass pill.
    sheet.insert(
        Class::new("pk-toast")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::Color, tok("color.label"))
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with_shadow(0.0, 6.0, 18.0, tok("glass.shadow")),
    );

    // Tones paint the accent used by the message/icon.
    for tone in TONES {
        sheet.insert(
            Class::new(tone_class("pk-toast", tone))
                .with(StyleProp::Color, tok(tone.color_token())),
        );
    }

    // Message: body type that inherits the tone color.
    sheet.insert(
        Class::new("pk-toast__message").with(StyleProp::FontSize, tok("font.size.body")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: ToastProps) -> Element {
        Toast.render(&props)
    }

    #[test]
    fn attaches_base_and_tone_classes() {
        let el = render(ToastProps::new("Saved").tone(Tone::Success));
        assert_eq!(el.class_names(), ["pk-toast", "pk-toast--success"]);
    }

    #[test]
    fn default_tone_is_accent() {
        let el = render(ToastProps::new("Hi"));
        assert_eq!(el.class_names(), ["pk-toast", "pk-toast--accent"]);
    }

    #[test]
    fn renders_message_child() {
        let el = render(ToastProps::new("Copied"));
        let msg = &el.child_elements()[0];
        assert_eq!(msg.text_content(), Some("Copied"));
        assert!(msg.class_names().iter().any(|c| c == "pk-toast__message"));
    }
}
