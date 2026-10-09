//! [`Fade`] — an opacity-only enter/leave transition.
//!
//! The thinnest possible motion control: it attaches `pk-fade` plus a
//! [`TransitionPhase`] modifier and wraps its children. Phase is driven
//! externally (see [`crate::motion::transition`]); a backend tweens
//! [`Opacity`](prism_ui_style::StyleProp::Opacity) between the start and end
//! keyframes.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::motion::transition::TransitionPhase;
use crate::preset::StyleSheet;

/// Props for [`Fade`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct FadeProps {
    /// The externally-driven lifecycle phase.
    pub phase: TransitionPhase,
    /// Content wrapped by the fade box.
    pub children: Vec<Element>,
}

impl FadeProps {
    /// Creates empty props at the default (`Enter`) phase.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the lifecycle phase.
    #[must_use]
    pub fn phase(mut self, phase: TransitionPhase) -> Self {
        self.phase = phase;
        self
    }

    /// Appends a single child.
    #[must_use]
    pub fn child(mut self, element: Element) -> Self {
        self.children.push(element);
        self
    }

    /// Replaces the children with the given iterator's items.
    #[must_use]
    pub fn children(mut self, children: impl IntoIterator<Item = Element>) -> Self {
        self.children = children.into_iter().collect();
        self
    }
}

/// The fade control. Zero-sized; all configuration lives in [`FadeProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Fade;

impl Component for Fade {
    type Props = FadeProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-fade", &[props.phase.suffix()]) {
            el = el.class(name);
        }
        el.children(props.children.iter().cloned())
    }
}

/// Registers the `pk-fade` family: a baseline plus one opacity keyframe per
/// [`TransitionPhase`] (hidden on `enter`/`left`, visible on `entered`/`leave`).
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp, StyleValue};

    sheet.insert(Class::new("pk-fade").with(StyleProp::Opacity, StyleValue::number(1.0)));
    sheet.insert(Class::new("pk-fade--enter").with(StyleProp::Opacity, StyleValue::number(0.0)));
    sheet.insert(Class::new("pk-fade--entered").with(StyleProp::Opacity, StyleValue::number(1.0)));
    sheet.insert(Class::new("pk-fade--leave").with(StyleProp::Opacity, StyleValue::number(1.0)));
    sheet.insert(Class::new("pk-fade--left").with(StyleProp::Opacity, StyleValue::number(0.0)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    fn render(props: FadeProps) -> Element {
        Fade.render(&props)
    }

    #[test]
    fn attaches_base_and_phase_modifier() {
        let el = render(FadeProps::new().phase(TransitionPhase::Leave));
        assert_eq!(el.class_names(), ["pk-fade", "pk-fade--leave"]);
    }

    #[test]
    fn default_phase_is_enter() {
        let el = render(FadeProps::new());
        assert_eq!(el.class_names(), ["pk-fade", "pk-fade--enter"]);
    }

    #[test]
    fn wraps_children() {
        let el = render(FadeProps::new().child(Element::text("hi")));
        assert_eq!(el.child_elements().len(), 1);
        assert_eq!(el.child_elements()[0].kind(), &ElementKind::Text);
    }
}
