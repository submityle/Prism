//! [`Slide`] — a directional enter/leave transition.
//!
//! Attaches `pk-slide`, a [`TransitionPhase`] modifier, and a [`SlideFrom`]
//! direction modifier, then wraps its children. The visible part the kit can
//! express today is opacity (there is no transform prop); the direction
//! modifier carries a token-backed edge offset as a layout hint, and names the
//! axis so a capable backend can translate along it.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::motion::transition::TransitionPhase;
use crate::preset::StyleSheet;

/// The edge a [`Slide`] originates from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SlideFrom {
    /// Enters downward from the top edge.
    #[default]
    Top,
    /// Enters upward from the bottom edge.
    Bottom,
    /// Enters rightward from the left edge.
    Left,
    /// Enters leftward from the right edge.
    Right,
}

impl SlideFrom {
    /// The modifier suffix used in class names (e.g. `pk-slide--from-top`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            SlideFrom::Top => "from-top",
            SlideFrom::Bottom => "from-bottom",
            SlideFrom::Left => "from-left",
            SlideFrom::Right => "from-right",
        }
    }
}

/// Props for [`Slide`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SlideProps {
    /// The externally-driven lifecycle phase.
    pub phase: TransitionPhase,
    /// The edge the content slides from.
    pub from: SlideFrom,
    /// Content wrapped by the slide box.
    pub children: Vec<Element>,
}

impl SlideProps {
    /// Creates empty props at the default (`Enter`) phase sliding from the top.
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

    /// Sets the origin edge.
    #[must_use]
    pub fn from(mut self, from: SlideFrom) -> Self {
        self.from = from;
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

/// The slide control. Zero-sized; all configuration lives in [`SlideProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Slide;

impl Component for Slide {
    type Props = SlideProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-slide", &[props.phase.suffix(), props.from.suffix()]) {
            el = el.class(name);
        }
        el.children(props.children.iter().cloned())
    }
}

/// Registers the `pk-slide` family: a baseline, one opacity keyframe per
/// [`TransitionPhase`], and one token-backed edge-offset hint per
/// [`SlideFrom`]. Real translation is a backend concern (mirroring the glass
/// blur hint); the offset uses a spacing token so it stays theme-driven.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp, StyleValue};

    use crate::preset::tok;

    sheet.insert(Class::new("pk-slide").with(StyleProp::Opacity, StyleValue::number(1.0)));
    sheet.insert(Class::new("pk-slide--enter").with(StyleProp::Opacity, StyleValue::number(0.0)));
    sheet.insert(Class::new("pk-slide--entered").with(StyleProp::Opacity, StyleValue::number(1.0)));
    sheet.insert(Class::new("pk-slide--leave").with(StyleProp::Opacity, StyleValue::number(1.0)));
    sheet.insert(Class::new("pk-slide--left").with(StyleProp::Opacity, StyleValue::number(0.0)));

    // Direction hints: a spacing-token offset on the origin edge. A capable
    // backend may replace this with a real translation along the named axis.
    sheet.insert(Class::new("pk-slide--from-top").with(StyleProp::MarginTop, tok("space.sm")));
    sheet.insert(Class::new("pk-slide--from-bottom").with(StyleProp::MarginBottom, tok("space.sm")));
    sheet.insert(Class::new("pk-slide--from-left").with(StyleProp::MarginLeft, tok("space.sm")));
    sheet.insert(Class::new("pk-slide--from-right").with(StyleProp::MarginRight, tok("space.sm")));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: SlideProps) -> Element {
        Slide.render(&props)
    }

    #[test]
    fn attaches_base_phase_and_direction() {
        let el = render(
            SlideProps::new()
                .phase(TransitionPhase::Entered)
                .from(SlideFrom::Left),
        );
        assert_eq!(
            el.class_names(),
            ["pk-slide", "pk-slide--entered", "pk-slide--from-left"]
        );
    }

    #[test]
    fn defaults_to_enter_from_top() {
        let el = render(SlideProps::new());
        assert_eq!(
            el.class_names(),
            ["pk-slide", "pk-slide--enter", "pk-slide--from-top"]
        );
    }

    #[test]
    fn wraps_children() {
        let el = render(SlideProps::new().child(Element::text("hi")));
        assert_eq!(el.child_elements().len(), 1);
    }

    #[test]
    fn direction_suffixes() {
        assert_eq!(SlideFrom::Top.suffix(), "from-top");
        assert_eq!(SlideFrom::Right.suffix(), "from-right");
    }
}
