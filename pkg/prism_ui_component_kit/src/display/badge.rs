//! [`Badge`] — a compact count / status indicator.
//!
//! A badge is a tiny capsule that paints a semantic [`Tone`] and either shows a
//! numeric `count` or collapses to a bare `dot`. Like every kit control it
//! carries no literal style: it attaches the `pk-badge` class family and lets
//! [`crate::preset`] resolve color and metrics against the active theme.

use alloc::string::{String, ToString};

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::{classes, Tone};
use crate::preset::StyleSheet;

/// Props for [`Badge`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct BadgeProps {
    /// The numeric count to display. Ignored when [`BadgeProps::dot`] is set.
    pub count: Option<u64>,
    /// When `true`, the badge collapses to a bare status dot (no label).
    pub dot: bool,
    /// Semantic tone selecting which accent token the badge paints with.
    pub tone: Tone,
}

impl BadgeProps {
    /// Creates props for a count badge with the default (accent) tone.
    #[must_use]
    pub fn count(value: u64) -> Self {
        Self {
            count: Some(value),
            ..Self::default()
        }
    }

    /// Creates props for a bare status dot with the default (accent) tone.
    #[must_use]
    pub fn dot() -> Self {
        Self {
            dot: true,
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

/// The badge control. Zero-sized; all configuration lives in [`BadgeProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Badge;

impl Component for Badge {
    type Props = BadgeProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-badge", &[props.tone.suffix()]) {
            el = el.class(name);
        }
        if props.dot {
            el = el.class("pk-badge--dot");
            return el;
        }
        if let Some(count) = props.count {
            el = el.child(Element::text(count.to_string()).class("pk-badge__count"));
        }
        el
    }
}

/// Builds the `--tone` modifier name for a block, e.g. `pk-badge--success`.
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

/// Registers the `pk-badge` class family: one base, five tones, one dot.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Base: a centered capsule with caption typography and a white label.
    sheet.insert(
        Class::new("pk-badge")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with_padding_x(tok("space.xs"))
            .with_padding_y(tok("space.xxs"))
            .with(StyleProp::MinWidth, StyleValue::px(18.0))
            .with(StyleProp::Height, StyleValue::px(18.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::FontSize, tok("font.size.caption2"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, StyleValue::rgba8(255, 255, 255, 255)),
    );

    // Tones paint the fill with the tone's semantic color token.
    for tone in TONES {
        sheet.insert(
            Class::new(tone_class("pk-badge", tone))
                .with(StyleProp::BackgroundColor, tok(tone.color_token())),
        );
    }

    // Dot: a fixed, label-less circle.
    sheet.insert(
        Class::new("pk-badge--dot")
            .with(StyleProp::Width, StyleValue::px(8.0))
            .with(StyleProp::Height, StyleValue::px(8.0))
            .with(StyleProp::MinWidth, StyleValue::px(8.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    fn render(props: BadgeProps) -> Element {
        Badge.render(&props)
    }

    #[test]
    fn attaches_base_and_tone_classes_in_order() {
        let el = render(BadgeProps::count(3).tone(Tone::Danger));
        assert_eq!(el.class_names(), ["pk-badge", "pk-badge--danger"]);
    }

    #[test]
    fn count_renders_as_text_child() {
        let el = render(BadgeProps::count(42));
        let label = el
            .child_elements()
            .iter()
            .find(|c| c.kind() == &ElementKind::Text)
            .expect("count child");
        assert_eq!(label.text_content(), Some("42"));
    }

    #[test]
    fn dot_adds_marker_and_has_no_children() {
        let el = render(BadgeProps::dot().tone(Tone::Success));
        assert!(el.class_names().iter().any(|c| c == "pk-badge--dot"));
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn default_tone_is_accent() {
        let el = render(BadgeProps::count(1));
        assert_eq!(el.class_names(), ["pk-badge", "pk-badge--accent"]);
    }
}
