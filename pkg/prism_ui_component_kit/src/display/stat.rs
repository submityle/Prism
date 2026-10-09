//! [`Stat`] — a KPI tile: a label, a prominent value, and an optional delta.
//!
//! A stat stacks a secondary `label` over a large `value`, with an optional
//! `delta` string (e.g. `+12%`) tinted by a semantic [`Tone`] to signal
//! direction. All color and typography come from theme tokens via
//! [`crate::preset`]; the control attaches only `pk-stat` class names.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::{classes, Tone};
use crate::preset::StyleSheet;

/// Props for [`Stat`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct StatProps {
    /// The descriptive label shown above the value.
    pub label: String,
    /// The prominent primary value.
    pub value: String,
    /// Optional change indicator (e.g. `+12%`), rendered below the value.
    pub delta: Option<String>,
    /// Semantic tone the delta is tinted with.
    pub delta_tone: Tone,
}

impl StatProps {
    /// Creates props for a `label` / `value` KPI tile.
    #[must_use]
    pub fn new(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            ..Self::default()
        }
    }

    /// Sets the delta string and its tone.
    #[must_use]
    pub fn delta(mut self, delta: impl Into<String>, tone: Tone) -> Self {
        self.delta = Some(delta.into());
        self.delta_tone = tone;
        self
    }
}

/// The stat control. Zero-sized; all configuration lives in [`StatProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Stat;

impl Component for Stat {
    type Props = StatProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-stat");
        el = el.child(Element::text(props.label.clone()).class("pk-stat__label"));
        el = el.child(Element::text(props.value.clone()).class("pk-stat__value"));
        if let Some(delta) = props.delta.clone() {
            let mut delta_el = Element::text(delta);
            for name in classes("pk-stat__delta", &[props.delta_tone.suffix()]) {
                delta_el = delta_el.class(name);
            }
            el = el.child(delta_el);
        }
        el
    }
}

/// Builds the `--tone` modifier name for a block, e.g. `pk-stat__delta--success`.
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

/// Registers the `pk-stat` class family: container, label, value, delta tones.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Container: a tight vertical stack.
    sheet.insert(
        Class::new("pk-stat")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs")),
    );

    // Label: small, secondary.
    sheet.insert(
        Class::new("pk-stat__label")
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::FontWeight, tok("font.weight.regular"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Value: the hero number.
    sheet.insert(
        Class::new("pk-stat__value")
            .with(StyleProp::FontSize, tok("font.size.title1"))
            .with(StyleProp::FontWeight, tok("font.weight.bold"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Delta: caption typography, tone-colored.
    sheet.insert(
        Class::new("pk-stat__delta")
            .with(StyleProp::FontSize, tok("font.size.caption1"))
            .with(StyleProp::FontWeight, tok("font.weight.medium")),
    );
    for tone in TONES {
        sheet.insert(
            Class::new(tone_class("pk-stat__delta", tone))
                .with(StyleProp::Color, tok(tone.color_token())),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use prism_ui::ElementKind;

    fn render(props: StatProps) -> Element {
        Stat.render(&props)
    }

    #[test]
    fn renders_label_and_value_text() {
        let el = render(StatProps::new("Revenue", "$1.2M"));
        let texts: Vec<_> = el
            .child_elements()
            .iter()
            .filter_map(Element::text_content)
            .collect();
        assert_eq!(texts, ["Revenue", "$1.2M"]);
    }

    #[test]
    fn no_delta_yields_two_children() {
        let el = render(StatProps::new("Users", "100"));
        assert_eq!(el.child_elements().len(), 2);
    }

    #[test]
    fn delta_adds_toned_child() {
        let el = render(StatProps::new("Users", "100").delta("+12%", Tone::Success));
        assert_eq!(el.child_elements().len(), 3);
        let delta = &el.child_elements()[2];
        assert_eq!(delta.kind(), &ElementKind::Text);
        assert_eq!(delta.text_content(), Some("+12%"));
        assert_eq!(
            delta.class_names(),
            ["pk-stat__delta", "pk-stat__delta--success"]
        );
    }

    #[test]
    fn container_has_stat_class() {
        let el = render(StatProps::new("A", "1"));
        assert_eq!(el.class_names(), ["pk-stat"]);
    }
}
