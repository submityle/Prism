//! [`AspectRatio`] — a box that constrains its child to a fixed ratio.
//!
//! An aspect-ratio box renders a `pk-aspect-ratio` container wrapping a single
//! child. The target `ratio` (width / height) is carried in the props as a
//! backend layout hint: the kit's [`StyleProp`](prism_ui_style::StyleProp)
//! vocabulary has no aspect-ratio property, so a capable layout backend reads
//! `ratio` to size the box. The control attaches only kit class names.

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`AspectRatio`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct AspectRatioProps {
    /// The target width-to-height ratio (e.g. `16.0 / 9.0`). A backend hint.
    pub ratio: f32,
    /// The child constrained to the ratio.
    pub child: Option<Element>,
}

impl AspectRatioProps {
    /// Creates aspect-ratio props for the given `ratio` (width / height).
    #[must_use]
    pub fn new(ratio: f32) -> Self {
        Self {
            ratio,
            ..Self::default()
        }
    }

    /// Sets the target width-to-height ratio.
    #[must_use]
    pub fn ratio(mut self, ratio: f32) -> Self {
        self.ratio = ratio;
        self
    }

    /// Sets the child element.
    #[must_use]
    pub fn child(mut self, element: Element) -> Self {
        self.child = Some(element);
        self
    }
}

/// The aspect-ratio control. Zero-sized; config lives in [`AspectRatioProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct AspectRatio;

impl Component for AspectRatio {
    type Props = AspectRatioProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-aspect-ratio");
        if let Some(child) = props.child.clone() {
            el = el.child(child);
        }
        el
    }
}

/// Registers the `pk-aspect-ratio` class. Sizing is driven by the backend from
/// the `ratio` hint; the base class only provides a block display.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::kw;

    sheet.insert(Class::new("pk-aspect-ratio").with(StyleProp::Display, kw(Keyword::Block)));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: AspectRatioProps) -> Element {
        AspectRatio.render(&props)
    }

    #[test]
    fn empty_box_has_only_base_class_and_no_child() {
        let el = render(AspectRatioProps::new(1.0));
        assert_eq!(el.class_names(), ["pk-aspect-ratio"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn wraps_single_child() {
        let el = render(AspectRatioProps::new(16.0 / 9.0).child(Element::box_().class("inner")));
        assert_eq!(el.child_elements().len(), 1);
        assert!(el.child_elements()[0].class_names().iter().any(|c| c == "inner"));
    }

    #[test]
    fn ratio_is_carried_in_props() {
        let props = AspectRatioProps::new(4.0 / 3.0);
        assert!((props.ratio - 4.0 / 3.0).abs() < f32::EPSILON);
    }
}
