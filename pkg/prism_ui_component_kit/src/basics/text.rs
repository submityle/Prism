//! [`Text`] — the kit's typographic primitive.
//!
//! A text run selects a semantic [`TextRole`] (which maps to a `font.size.*`
//! token) and an optional muted [`tone`](TextProps::tone). It carries no raw
//! font size: the preset layer resolves each role class against the active
//! theme, so a type-scale change is a token edit, not a control edit.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// The semantic size/weight step a [`Text`] run paints with.
///
/// Each role maps to a `font.size.*` (and matching weight) token in
/// [`register_styles`], mirroring the platform type ramp.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum TextRole {
    /// Standard reading copy (`font.size.body`).
    #[default]
    Body,
    /// Slightly emphasised body (`font.size.callout`).
    Callout,
    /// Secondary, smaller copy (`font.size.subheadline`).
    Subheadline,
    /// Fine print (`font.size.footnote`).
    Footnote,
    /// Smallest caption (`font.size.caption1`).
    Caption,
    /// A bold inline headline (`font.size.headline`).
    Headline,
}

impl TextRole {
    /// The modifier suffix used in class names (e.g. `pk-text--body`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            TextRole::Body => "body",
            TextRole::Callout => "callout",
            TextRole::Subheadline => "subheadline",
            TextRole::Footnote => "footnote",
            TextRole::Caption => "caption",
            TextRole::Headline => "headline",
        }
    }
}

/// How prominently a text run reads against its surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum TextTone {
    /// Primary label color.
    #[default]
    Primary,
    /// Secondary (de-emphasised) label color.
    Secondary,
    /// Tertiary (hint) label color.
    Tertiary,
}

impl TextTone {
    /// The modifier suffix used in class names (e.g. `pk-text--secondary`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            TextTone::Primary => "primary",
            TextTone::Secondary => "secondary",
            TextTone::Tertiary => "tertiary",
        }
    }
}

/// Props for [`Text`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TextProps {
    /// The literal string to render.
    pub content: String,
    /// The semantic type-scale role.
    pub role: TextRole,
    /// The emphasis/tone of the label color.
    pub tone: TextTone,
}

impl TextProps {
    /// Creates props for a text run with the default body role.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ..Self::default()
        }
    }

    /// Sets the type-scale role.
    #[must_use]
    pub fn role(mut self, role: TextRole) -> Self {
        self.role = role;
        self
    }

    /// Sets the emphasis tone.
    #[must_use]
    pub fn tone(mut self, tone: TextTone) -> Self {
        self.tone = tone;
        self
    }
}

/// The text control. Zero-sized; all configuration lives in [`TextProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Text;

impl Component for Text {
    type Props = TextProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::text(props.content.clone());
        for name in classes("pk-text", &[props.role.suffix(), props.tone.suffix()]) {
            el = el.class(name);
        }
        el
    }
}

/// Registers the `pk-text` family: one base plus a class per role and tone.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp};

    use crate::preset::tok;

    sheet.insert(
        Class::new("pk-text")
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::FontWeight, tok("font.weight.regular")),
    );
    for (role, size_token, weight_token) in [
        (TextRole::Body, "font.size.body", "font.weight.regular"),
        (TextRole::Callout, "font.size.callout", "font.weight.regular"),
        (
            TextRole::Subheadline,
            "font.size.subheadline",
            "font.weight.regular",
        ),
        (TextRole::Footnote, "font.size.footnote", "font.weight.regular"),
        (TextRole::Caption, "font.size.caption1", "font.weight.regular"),
        (TextRole::Headline, "font.size.headline", "font.weight.semibold"),
    ] {
        let mut name = String::from("pk-text--");
        name.push_str(role.suffix());
        sheet.insert(
            Class::new(name)
                .with(StyleProp::FontSize, tok(size_token))
                .with(StyleProp::FontWeight, tok(weight_token)),
        );
    }
    for (tone, color_token) in [
        (TextTone::Primary, "color.label"),
        (TextTone::Secondary, "color.label.secondary"),
        (TextTone::Tertiary, "color.label.tertiary"),
    ] {
        let mut name = String::from("pk-text--");
        name.push_str(tone.suffix());
        sheet.insert(Class::new(name).with(StyleProp::Color, tok(color_token)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    #[test]
    fn renders_text_with_role_and_tone_classes() {
        let el = Text.render(
            &TextProps::new("hello")
                .role(TextRole::Headline)
                .tone(TextTone::Secondary),
        );
        assert_eq!(el.kind(), &ElementKind::Text);
        assert_eq!(el.text_content(), Some("hello"));
        assert_eq!(
            el.class_names(),
            ["pk-text", "pk-text--headline", "pk-text--secondary"]
        );
    }

    #[test]
    fn register_adds_role_and_tone_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        assert!(sheet.get("pk-text").is_some());
        assert!(sheet.get("pk-text--headline").is_some());
        assert!(sheet.get("pk-text--secondary").is_some());
    }
}
