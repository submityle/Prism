//! [`Callout`] — a tinted info box drawing attention to supporting detail.
//!
//! A callout pairs an optional `title` with a `body` slot and a semantic
//! [`Tone`]. It renders a `pk-callout`(+`--tone`) box with a `__title` and a
//! `__body`, attaching only kit classes; [`crate::preset`] resolves the tinted
//! surface and accent against the active theme.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::{classes, Tone};
use crate::preset::StyleSheet;

/// Props for [`Callout`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CalloutProps {
    /// Optional short heading.
    pub title: Option<String>,
    /// The callout body content.
    pub body: Vec<Element>,
    /// The semantic tone selecting the tint/accent.
    pub tone: Tone,
}

impl CalloutProps {
    /// Creates empty callout props with the default (accent) tone.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the heading.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Appends a body child.
    #[must_use]
    pub fn body_child(mut self, child: Element) -> Self {
        self.body.push(child);
        self
    }

    /// Replaces the body content with `body`.
    #[must_use]
    pub fn body<I: IntoIterator<Item = Element>>(mut self, body: I) -> Self {
        self.body = body.into_iter().collect();
        self
    }

    /// Sets the semantic tone.
    #[must_use]
    pub fn tone(mut self, tone: Tone) -> Self {
        self.tone = tone;
        self
    }
}

/// The callout control. Zero-sized; all configuration lives in [`CalloutProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Callout;

impl Component for Callout {
    type Props = CalloutProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-callout", &[props.tone.suffix()]) {
            el = el.class(name);
        }
        if let Some(title) = props.title.clone() {
            el = el.child(Element::text(title).class("pk-callout__title"));
        }
        el.child(
            Element::box_()
                .class("pk-callout__body")
                .children(props.body.iter().cloned()),
        )
    }
}

/// Builds the `--tone` modifier name for a block, e.g. `pk-callout--success`.
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

/// Registers the `pk-callout` class family: the tinted box base, per-tone
/// border/title accent, and the title/body slots.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Base: a rounded box on a quiet fill with a leading accent border.
    sheet.insert(
        Class::new("pk-callout")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xs"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.md"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BackgroundColor, tok("color.fill.tertiary")),
    );

    // Tones paint the border accent.
    for tone in TONES {
        sheet.insert(
            Class::new(tone_class("pk-callout", tone))
                .with(StyleProp::BorderColor, tok(tone.color_token())),
        );
    }

    // Title: emphasised body type.
    sheet.insert(
        Class::new("pk-callout__title")
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Body: a vertical stack of secondary body content.
    sheet.insert(
        Class::new("pk-callout__body")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: CalloutProps) -> Element {
        Callout.render(&props)
    }

    #[test]
    fn attaches_base_and_tone_classes() {
        let el = render(CalloutProps::new().tone(Tone::Success));
        assert_eq!(el.class_names(), ["pk-callout", "pk-callout--success"]);
    }

    #[test]
    fn body_only_has_single_body_slot() {
        let el = render(CalloutProps::new().body_child(Element::text("note")));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-callout__body"));
        assert_eq!(kids[0].child_elements().len(), 1);
    }

    #[test]
    fn title_precedes_body() {
        let el = render(
            CalloutProps::new()
                .title("Tip")
                .body_child(Element::text("Use tokens.")),
        );
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-callout__title"));
        assert_eq!(kids[0].text_content(), Some("Tip"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-callout__body"));
    }
}
