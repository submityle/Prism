//! [`Alert`] — an inline banner stating an important, tone-colored message.
//!
//! An alert pairs an optional `title` with a required `message` and a semantic
//! [`Tone`]. It renders a `pk-alert` banner with `__title`/`__message` slots,
//! attaching only kit classes; [`crate::preset`] resolves the tone accent and
//! surface against the active theme.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::{classes, Tone};
use crate::preset::StyleSheet;

/// Props for [`Alert`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct AlertProps {
    /// Optional short heading.
    pub title: Option<String>,
    /// The alert body message.
    pub message: String,
    /// The semantic tone selecting the accent.
    pub tone: Tone,
}

impl AlertProps {
    /// Creates alert props with the default (accent) tone and no title.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            ..Self::default()
        }
    }

    /// Sets the heading.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Sets the semantic tone.
    #[must_use]
    pub fn tone(mut self, tone: Tone) -> Self {
        self.tone = tone;
        self
    }
}

/// The alert control. Zero-sized; all configuration lives in [`AlertProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Alert;

impl Component for Alert {
    type Props = AlertProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-alert", &[props.tone.suffix()]) {
            el = el.class(name);
        }
        if let Some(title) = props.title.clone() {
            el = el.child(Element::text(title).class("pk-alert__title"));
        }
        el.child(Element::text(props.message.clone()).class("pk-alert__message"))
    }
}

/// Builds the `--tone` modifier name for a block, e.g. `pk-alert--danger`.
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

/// Registers the `pk-alert` class family: the banner base, per-tone border
/// accent, and the title/message slots.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Base: a vertical banner with a quiet fill and a lead accent border.
    sheet.insert(
        Class::new("pk-alert")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    // Tones paint the border and title accent.
    for tone in TONES {
        sheet.insert(
            Class::new(tone_class("pk-alert", tone))
                .with(StyleProp::BorderColor, tok(tone.color_token())),
        );
    }

    // Title: emphasised body type.
    sheet.insert(
        Class::new("pk-alert__title")
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Message: secondary body type.
    sheet.insert(
        Class::new("pk-alert__message")
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: AlertProps) -> Element {
        Alert.render(&props)
    }

    #[test]
    fn attaches_base_and_tone_classes() {
        let el = render(AlertProps::new("Disk full").tone(Tone::Danger));
        assert_eq!(el.class_names(), ["pk-alert", "pk-alert--danger"]);
    }

    #[test]
    fn message_only_has_single_child() {
        let el = render(AlertProps::new("Heads up"));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0].text_content(), Some("Heads up"));
        assert!(kids[0].class_names().iter().any(|c| c == "pk-alert__message"));
    }

    #[test]
    fn title_precedes_message() {
        let el = render(AlertProps::new("Body").title("Heading"));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-alert__title"));
        assert_eq!(kids[0].text_content(), Some("Heading"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-alert__message"));
    }
}
