//! [`Scale`] — a "pop" enter/leave transition, approximated via opacity.
//!
//! There is no transform/scale [`StyleProp`](prism_ui_style::StyleProp) in the
//! stack, so this control records the intent (`pk-scale` + a
//! [`TransitionPhase`] modifier) and expresses the visible part — opacity —
//! itself. A capable backend may additionally honor the class name with a real
//! scale transform, exactly as glass blur is a backend-only hint.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::motion::transition::TransitionPhase;
use crate::preset::StyleSheet;

/// Props for [`Scale`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ScaleProps {
    /// The externally-driven lifecycle phase.
    pub phase: TransitionPhase,
    /// Content wrapped by the scale box.
    pub children: Vec<Element>,
}

impl ScaleProps {
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

/// The scale control. Zero-sized; all configuration lives in [`ScaleProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Scale;

impl Component for Scale {
    type Props = ScaleProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-scale", &[props.phase.suffix()]) {
            el = el.class(name);
        }
        el.children(props.children.iter().cloned())
    }
}

/// Registers the `pk-scale` family. Without a transform prop the visible
/// effect is opacity (hidden on `enter`/`left`, visible on `entered`/`leave`);
/// the class name still carries the "scale" intent for a capable backend.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp, StyleValue};

    sheet.insert(Class::new("pk-scale").with(StyleProp::Opacity, StyleValue::number(1.0)));
    sheet.insert(Class::new("pk-scale--enter").with(StyleProp::Opacity, StyleValue::number(0.0)));
    sheet.insert(Class::new("pk-scale--entered").with(StyleProp::Opacity, StyleValue::number(1.0)));
    sheet.insert(Class::new("pk-scale--leave").with(StyleProp::Opacity, StyleValue::number(1.0)));
    sheet.insert(Class::new("pk-scale--left").with(StyleProp::Opacity, StyleValue::number(0.0)));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: ScaleProps) -> Element {
        Scale.render(&props)
    }

    #[test]
    fn attaches_base_and_phase_modifier() {
        let el = render(ScaleProps::new().phase(TransitionPhase::Entered));
        assert_eq!(el.class_names(), ["pk-scale", "pk-scale--entered"]);
    }

    #[test]
    fn default_phase_is_enter() {
        let el = render(ScaleProps::new());
        assert_eq!(el.class_names(), ["pk-scale", "pk-scale--enter"]);
    }

    #[test]
    fn wraps_children() {
        let el = render(ScaleProps::new().child(Element::box_()).child(Element::box_()));
        assert_eq!(el.child_elements().len(), 2);
    }
}
