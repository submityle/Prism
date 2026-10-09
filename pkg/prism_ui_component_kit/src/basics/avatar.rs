//! [`Avatar`] — a circular user/entity representation (image, initials, or icon).

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::{classes, ControlSize};
use crate::preset::StyleSheet;

/// What an [`Avatar`] displays when it has no image.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum AvatarContent {
    /// Up to two initials.
    Initials(String),
    /// A backend-resolved image source.
    Image(String),
    /// A neutral placeholder silhouette.
    #[default]
    Placeholder,
}

/// Props for [`Avatar`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct AvatarProps {
    /// The content strategy.
    pub content: AvatarContent,
    /// The density step controlling diameter.
    pub size: ControlSize,
}

impl AvatarProps {
    /// Creates props for a placeholder avatar.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Shows initials.
    #[must_use]
    pub fn initials(mut self, initials: impl Into<String>) -> Self {
        self.content = AvatarContent::Initials(initials.into());
        self
    }

    /// Shows an image source.
    #[must_use]
    pub fn image(mut self, src: impl Into<String>) -> Self {
        self.content = AvatarContent::Image(src.into());
        self
    }

    /// Sets the size step.
    #[must_use]
    pub fn size(mut self, size: ControlSize) -> Self {
        self.size = size;
        self
    }
}

/// The avatar control. Zero-sized; all configuration lives in [`AvatarProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Avatar;

impl Component for Avatar {
    type Props = AvatarProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-avatar", &[props.size.suffix()]) {
            el = el.class(name);
        }
        match &props.content {
            AvatarContent::Initials(text) => {
                el.child(Element::text(text.clone()).class("pk-avatar__initials"))
            }
            AvatarContent::Image(src) => el
                .class("pk-avatar--image")
                .child(Element::custom("pk-avatar__image").child(Element::text(src.clone()))),
            AvatarContent::Placeholder => el.class("pk-avatar--placeholder"),
        }
    }
}

/// Registers the `pk-avatar` family: base, per-size diameters, variants.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-avatar")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );
    for (size, px) in [
        (ControlSize::Small, 24.0),
        (ControlSize::Medium, 36.0),
        (ControlSize::Large, 56.0),
    ] {
        let mut name = String::from("pk-avatar--");
        name.push_str(size.suffix());
        sheet.insert(
            Class::new(name)
                .with(StyleProp::Width, StyleValue::px(px))
                .with(StyleProp::Height, StyleValue::px(px)),
        );
    }
    sheet.insert(
        Class::new("pk-avatar--placeholder")
            .with(StyleProp::BackgroundColor, tok("color.fill.tertiary")),
    );
    sheet.insert(Class::new("pk-avatar--image"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initials_render_as_child_text() {
        let el = Avatar.render(&AvatarProps::new().initials("WK").size(ControlSize::Large));
        assert_eq!(el.class_names(), ["pk-avatar", "pk-avatar--lg"]);
        assert_eq!(el.child_elements().len(), 1);
    }

    #[test]
    fn placeholder_has_variant_class() {
        let el = Avatar.render(&AvatarProps::new());
        assert_eq!(
            el.class_names(),
            ["pk-avatar", "pk-avatar--md", "pk-avatar--placeholder"]
        );
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in ["pk-avatar", "pk-avatar--sm", "pk-avatar--placeholder"] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
