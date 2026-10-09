//! [`Image`] — a backend-resolved raster/vector image with a fit mode.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// How an [`Image`] fills its box.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ImageFit {
    /// Cover the box, cropping overflow.
    #[default]
    Cover,
    /// Fit entirely within the box, letterboxing.
    Contain,
    /// Stretch to fill both axes.
    Fill,
}

impl ImageFit {
    /// The modifier suffix (e.g. `pk-image--cover`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            ImageFit::Cover => "cover",
            ImageFit::Contain => "contain",
            ImageFit::Fill => "fill",
        }
    }
}

/// Props for [`Image`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ImageProps {
    /// The image source (resolved by the backend).
    pub src: String,
    /// Accessible alternative text.
    pub alt: String,
    /// How the image fills its box.
    pub fit: ImageFit,
}

impl ImageProps {
    /// Creates props for an image.
    #[must_use]
    pub fn new(src: impl Into<String>) -> Self {
        Self {
            src: src.into(),
            ..Self::default()
        }
    }

    /// Sets accessible alt text.
    #[must_use]
    pub fn alt(mut self, alt: impl Into<String>) -> Self {
        self.alt = alt.into();
        self
    }

    /// Sets the fit mode.
    #[must_use]
    pub fn fit(mut self, fit: ImageFit) -> Self {
        self.fit = fit;
        self
    }
}

/// The image control. Zero-sized; configuration lives in [`ImageProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Image;

impl Image {
    /// The accessibility role an image exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Image
    }
}

impl Component for Image {
    type Props = ImageProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut fit = String::from("pk-image--");
        fit.push_str(props.fit.suffix());
        Element::custom("pk-image")
            .class("pk-image")
            .class(fit)
            .child(Element::text(props.src.clone()).class("pk-image__src"))
    }
}

/// Registers the `pk-image` family.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp};

    use crate::preset::tok;

    sheet.insert(Class::new("pk-image").with(StyleProp::BorderRadius, tok("radius.md")));
    sheet.insert(Class::new("pk-image--cover"));
    sheet.insert(Class::new("pk-image--contain"));
    sheet.insert(Class::new("pk-image--fill"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_fit_class() {
        let el = Image.render(&ImageProps::new("a.png").alt("A").fit(ImageFit::Contain));
        assert_eq!(el.class_names(), ["pk-image", "pk-image--contain"]);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in ["pk-image", "pk-image--cover", "pk-image--fill"] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
