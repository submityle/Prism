//! [`Tag`] — a small capsule label, optionally tinted by [`Tone`] and removable.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::{classes, Tone};
use crate::preset::StyleSheet;

/// Props for [`Tag`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TagProps {
    /// The tag text.
    pub text: String,
    /// The semantic tone.
    pub tone: Tone,
    /// Whether to render a trailing remove affordance.
    pub removable: bool,
}

impl TagProps {
    /// Creates props for a tag.
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

    /// Adds a trailing remove affordance.
    #[must_use]
    pub fn removable(mut self, removable: bool) -> Self {
        self.removable = removable;
        self
    }
}

/// The tag control. Zero-sized; all configuration lives in [`TagProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Tag;

impl Component for Tag {
    type Props = TagProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-tag", &[props.tone.suffix()]) {
            el = el.class(name);
        }
        el = el.child(Element::text(props.text.clone()).class("pk-tag__text"));
        if props.removable {
            el = el.child(Element::text("\u{00d7}").class("pk-tag__remove"));
        }
        el
    }
}

/// Registers the `pk-tag` family: base, per-tone tints, element parts.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-tag")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::FontSize, tok("font.size.caption1"))
            .with(StyleProp::FontWeight, tok("font.weight.medium"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.label"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xxs")),
    );
    for tone in [
        Tone::Accent,
        Tone::Neutral,
        Tone::Success,
        Tone::Warning,
        Tone::Danger,
    ] {
        let mut name = String::from("pk-tag--");
        name.push_str(tone.suffix());
        sheet.insert(Class::new(name).with(StyleProp::Color, tok(tone.color_token())));
    }
    sheet.insert(Class::new("pk-tag__remove").with(StyleProp::Color, tok("color.label.secondary")));
}

/// Alias: a chip is another name for a tag.
pub type Chip = Tag;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removable_adds_second_child() {
        let el = Tag.render(&TagProps::new("beta").tone(Tone::Success).removable(true));
        assert_eq!(el.class_names(), ["pk-tag", "pk-tag--success"]);
        assert_eq!(el.child_elements().len(), 2);
    }

    #[test]
    fn plain_tag_has_single_child() {
        let el = Tag.render(&TagProps::new("beta"));
        assert_eq!(el.child_elements().len(), 1);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in ["pk-tag", "pk-tag--success", "pk-tag__remove"] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
