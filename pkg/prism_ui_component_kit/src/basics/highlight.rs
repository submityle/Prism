//! [`Highlight`] — inline marked/emphasised text, tinted by [`Tone`].

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::{classes, Tone};
use crate::preset::StyleSheet;

/// Props for [`Highlight`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct HighlightProps {
    /// The highlighted text.
    pub text: String,
    /// The semantic tone of the highlight wash.
    pub tone: Tone,
}

impl HighlightProps {
    /// Creates props for a highlight.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// Sets the tone.
    #[must_use]
    pub fn tone(mut self, tone: Tone) -> Self {
        self.tone = tone;
        self
    }
}

/// The highlight control. Zero-sized; configuration lives in [`HighlightProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Highlight;

impl Component for Highlight {
    type Props = HighlightProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::text(props.text.clone());
        for name in classes("pk-highlight", &[props.tone.suffix()]) {
            el = el.class(name);
        }
        el
    }
}

/// Registers the `pk-highlight` family.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp};

    use crate::preset::tok;

    sheet.insert(
        Class::new("pk-highlight")
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::BorderRadius, tok("radius.xs"))
            .with_padding_x(tok("space.xxs")),
    );
    for tone in [
        Tone::Accent,
        Tone::Neutral,
        Tone::Success,
        Tone::Warning,
        Tone::Danger,
    ] {
        let mut name = String::from("pk-highlight--");
        name.push_str(tone.suffix());
        sheet.insert(Class::new(name).with(StyleProp::Color, tok(tone.color_token())));
    }
}

/// Alias: `Mark` mirrors the HTML element name for highlighted text.
pub type Mark = Highlight;

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    #[test]
    fn renders_tone_class() {
        let el = Highlight.render(&HighlightProps::new("hi").tone(Tone::Warning));
        assert_eq!(el.kind(), &ElementKind::Text);
        assert_eq!(el.class_names(), ["pk-highlight", "pk-highlight--warning"]);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in ["pk-highlight", "pk-highlight--warning"] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
