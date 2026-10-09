//! [`AsyncImage`] — an [`Image`](super::image::Image) with placeholder + lazy
//! load states.
//!
//! The control models three load phases via a modifier class so a backend can
//! swap a skeleton placeholder for the resolved image. No I/O happens here.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// The load phase of an [`AsyncImage`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum LoadPhase {
    /// Not yet loaded; shows a placeholder.
    #[default]
    Loading,
    /// Loaded successfully; shows the image.
    Loaded,
    /// Failed to load; shows a fallback.
    Failed,
}

impl LoadPhase {
    /// The modifier suffix (e.g. `pk-async-image--loading`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            LoadPhase::Loading => "loading",
            LoadPhase::Loaded => "loaded",
            LoadPhase::Failed => "failed",
        }
    }
}

/// Props for [`AsyncImage`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct AsyncImageProps {
    /// The image source (resolved by the backend).
    pub src: String,
    /// Accessible alternative text.
    pub alt: String,
    /// The current load phase.
    pub phase: LoadPhase,
}

impl AsyncImageProps {
    /// Creates props for an async image in the loading phase.
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

    /// Sets the load phase.
    #[must_use]
    pub fn phase(mut self, phase: LoadPhase) -> Self {
        self.phase = phase;
        self
    }
}

/// The async-image control. Zero-sized; configuration lives in
/// [`AsyncImageProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct AsyncImage;

impl Component for AsyncImage {
    type Props = AsyncImageProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut phase = String::from("pk-async-image--");
        phase.push_str(props.phase.suffix());
        let mut el = Element::custom("pk-async-image")
            .class("pk-async-image")
            .class(phase);
        el = match props.phase {
            LoadPhase::Loading => el.child(Element::box_().class("pk-async-image__placeholder")),
            LoadPhase::Loaded => {
                el.child(Element::text(props.src.clone()).class("pk-async-image__src"))
            }
            LoadPhase::Failed => el.child(Element::box_().class("pk-async-image__fallback")),
        };
        el
    }
}

/// Registers the `pk-async-image` family.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp};

    use crate::preset::tok;

    sheet.insert(Class::new("pk-async-image").with(StyleProp::BorderRadius, tok("radius.md")));
    sheet.insert(
        Class::new("pk-async-image__placeholder")
            .with(StyleProp::BackgroundColor, tok("color.fill.tertiary"))
            .with(StyleProp::BorderRadius, tok("radius.md")),
    );
    sheet.insert(
        Class::new("pk-async-image__fallback")
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::BorderRadius, tok("radius.md")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_shows_placeholder() {
        let el = AsyncImage.render(&AsyncImageProps::new("a.png"));
        assert_eq!(el.class_names(), ["pk-async-image", "pk-async-image--loading"]);
        assert_eq!(
            el.child_elements()[0].class_names(),
            ["pk-async-image__placeholder"]
        );
    }

    #[test]
    fn loaded_shows_src() {
        let el = AsyncImage.render(&AsyncImageProps::new("a.png").phase(LoadPhase::Loaded));
        assert_eq!(el.child_elements()[0].text_content(), Some("a.png"));
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in [
            "pk-async-image",
            "pk-async-image__placeholder",
            "pk-async-image__fallback",
        ] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
