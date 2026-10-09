//! [`Banner`] — a full-width, tone-colored notice strip.
//!
//! A banner carries a `message`, a semantic [`Tone`] and a `dismissible` flag.
//! It renders a `pk-banner`(+`--tone`, +`is-dismissible`) strip with a
//! `__message` and, when dismissible, a `__dismiss` affordance. It attaches only
//! kit classes; [`crate::preset`] resolves the tone accent and surface.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::{classes, Tone};
use crate::preset::StyleSheet;

/// Props for [`Banner`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct BannerProps {
    /// The notice message.
    pub message: String,
    /// The semantic tone selecting the accent.
    pub tone: Tone,
    /// Whether the banner shows a dismiss affordance.
    pub dismissible: bool,
}

impl BannerProps {
    /// Creates banner props with the default (accent) tone, not dismissible.
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

    /// Sets whether a dismiss affordance is shown.
    #[must_use]
    pub fn dismissible(mut self, dismissible: bool) -> Self {
        self.dismissible = dismissible;
        self
    }
}

/// The banner control. Zero-sized; all configuration lives in [`BannerProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Banner;

impl Component for Banner {
    type Props = BannerProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-banner", &[props.tone.suffix()]) {
            el = el.class(name);
        }
        if props.dismissible {
            el = el.class("is-dismissible");
        }

        el = el.child(Element::text(props.message.clone()).class("pk-banner__message"));
        if props.dismissible {
            el = el.child(Element::text("×").class("pk-banner__dismiss"));
        }
        el
    }
}

/// Builds the `--tone` modifier name for a block, e.g. `pk-banner--warning`.
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

/// Registers the `pk-banner` class family: the strip base, per-tone surface,
/// the message run and the dismiss affordance.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Base: a full-width row with the message leading and dismiss trailing.
    sheet.insert(
        Class::new("pk-banner")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Tones paint the strip surface.
    for tone in TONES {
        sheet.insert(
            Class::new(tone_class("pk-banner", tone))
                .with(StyleProp::BackgroundColor, tok(tone.color_token())),
        );
    }

    // Message: body type that fills the available width.
    sheet.insert(
        Class::new("pk-banner__message")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );

    // Dismiss: a small, tappable close glyph.
    sheet.insert(
        Class::new("pk-banner__dismiss")
            .with(StyleProp::FontSize, tok("font.size.headline"))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: BannerProps) -> Element {
        Banner.render(&props)
    }

    #[test]
    fn attaches_base_and_tone_classes() {
        let el = render(BannerProps::new("Heads up").tone(Tone::Warning));
        assert_eq!(el.class_names(), ["pk-banner", "pk-banner--warning"]);
    }

    #[test]
    fn non_dismissible_has_only_message() {
        let el = render(BannerProps::new("Note"));
        assert!(!el.class_names().iter().any(|c| c == "is-dismissible"));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-banner__message"));
    }

    #[test]
    fn dismissible_adds_marker_and_dismiss_child() {
        let el = render(BannerProps::new("Note").dismissible(true));
        assert!(el.class_names().iter().any(|c| c == "is-dismissible"));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert!(kids[1].class_names().iter().any(|c| c == "pk-banner__dismiss"));
    }
}
