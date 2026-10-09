//! [`Transition`] — the kit's reusable enter/leave scheduler primitive.
//!
//! A transition is a `(duration, easing)` plan plus a lifecycle
//! [`TransitionPhase`]. The plan is captured by [`TransitionSpec`] (pure data,
//! no timers); the phase is driven *externally* by whatever owns the animation
//! clock. The [`Transition`] control is a thin wrapper that attaches the
//! `pk-transition` class family (base + a phase modifier) and wraps its
//! children, so the active theme — not this control — owns every value.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// The easing curve a [`TransitionSpec`] animates along.
///
/// This selects *shape*, not duration; it is data consumed by a backend clock,
/// never a style value (there is no easing [`StyleProp`](prism_ui_style::StyleProp)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Easing {
    /// Constant velocity.
    Linear,
    /// Accelerates from rest.
    EaseIn,
    /// Decelerates to rest.
    EaseOut,
    /// Accelerates then decelerates — the default, natural feel.
    #[default]
    EaseInOut,
}

/// The lifecycle phase of an enter/leave transition.
///
/// `Enter`/`Leave` are the *start* keyframes (pre-animation state) and
/// `Entered`/`Left` are the *end* keyframes (post-animation state). The owner
/// advances this value; the control simply reflects it as a modifier class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum TransitionPhase {
    /// About to appear: the hidden start keyframe.
    #[default]
    Enter,
    /// Fully appeared: the visible end keyframe.
    Entered,
    /// About to disappear: the visible start keyframe.
    Leave,
    /// Fully gone: the hidden end keyframe.
    Left,
}

impl TransitionPhase {
    /// The modifier suffix used in class names (e.g. `pk-transition--enter`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            TransitionPhase::Enter => "enter",
            TransitionPhase::Entered => "entered",
            TransitionPhase::Leave => "leave",
            TransitionPhase::Left => "left",
        }
    }

    /// Whether this phase is a *visible* keyframe (end-of-enter / start-of-leave).
    #[must_use]
    pub const fn is_visible(self) -> bool {
        matches!(self, TransitionPhase::Entered | TransitionPhase::Leave)
    }
}

/// The reusable enter/leave scheduler primitive: a `(duration, easing)` plan.
///
/// Data-only by design — it holds no clock and runs no timer. A runtime reads
/// `duration_ms`/`easing` to drive the externally-owned [`TransitionPhase`].
///
/// Named distinctly from the [`Transition`] *control* so the plan (data) and
/// the renderer (component) never collide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransitionSpec {
    /// Total animation duration, in milliseconds.
    pub duration_ms: u32,
    /// The easing curve to animate along.
    pub easing: Easing,
}

impl Default for TransitionSpec {
    fn default() -> Self {
        // A 200ms ease-in-out is the kit's default "quick but legible" motion.
        Self {
            duration_ms: 200,
            easing: Easing::default(),
        }
    }
}

impl TransitionSpec {
    /// Creates a spec with the given duration (ms) and the default easing.
    #[must_use]
    pub fn new(duration_ms: u32) -> Self {
        Self {
            duration_ms,
            ..Self::default()
        }
    }

    /// Sets the duration, in milliseconds.
    #[must_use]
    pub fn duration_ms(mut self, duration_ms: u32) -> Self {
        self.duration_ms = duration_ms;
        self
    }

    /// Sets the easing curve.
    #[must_use]
    pub fn easing(mut self, easing: Easing) -> Self {
        self.easing = easing;
        self
    }
}

/// Props for [`Transition`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TransitionProps {
    /// The externally-driven lifecycle phase.
    pub phase: TransitionPhase,
    /// Content wrapped by the transition box.
    pub children: Vec<Element>,
}

impl TransitionProps {
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

/// The transition control. Zero-sized; all configuration lives in
/// [`TransitionProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Transition;

impl Component for Transition {
    type Props = TransitionProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-transition", &[props.phase.suffix()]) {
            el = el.class(name);
        }
        el.children(props.children.iter().cloned())
    }
}

/// Registers the `pk-transition` class family: a baseline plus one class per
/// [`TransitionPhase`]. The start keyframes (`enter`/`left`) are fully
/// transparent; the end keyframes (`entered`/`leave`) are fully opaque. A
/// backend tweens [`Opacity`](prism_ui_style::StyleProp::Opacity) between them
/// over the spec's duration.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp, StyleValue};

    sheet.insert(Class::new("pk-transition").with(StyleProp::Opacity, StyleValue::number(1.0)));
    sheet.insert(Class::new("pk-transition--enter").with(StyleProp::Opacity, StyleValue::number(0.0)));
    sheet.insert(Class::new("pk-transition--entered").with(StyleProp::Opacity, StyleValue::number(1.0)));
    sheet.insert(Class::new("pk-transition--leave").with(StyleProp::Opacity, StyleValue::number(1.0)));
    sheet.insert(Class::new("pk-transition--left").with(StyleProp::Opacity, StyleValue::number(0.0)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    fn render(props: TransitionProps) -> Element {
        Transition.render(&props)
    }

    #[test]
    fn attaches_base_and_phase_modifier() {
        let el = render(TransitionProps::new().phase(TransitionPhase::Entered));
        assert_eq!(el.class_names(), ["pk-transition", "pk-transition--entered"]);
    }

    #[test]
    fn default_phase_is_enter() {
        let el = render(TransitionProps::new());
        assert_eq!(el.class_names(), ["pk-transition", "pk-transition--enter"]);
    }

    #[test]
    fn wraps_children_in_order() {
        let el = render(
            TransitionProps::new()
                .child(Element::box_().class("a"))
                .child(Element::text("hi")),
        );
        let kinds: Vec<_> = el.child_elements().iter().map(Element::kind).collect();
        assert_eq!(kinds, [&ElementKind::Box, &ElementKind::Text]);
    }

    #[test]
    fn children_setter_replaces_contents() {
        let el = render(
            TransitionProps::new()
                .child(Element::box_())
                .children([Element::text("x"), Element::text("y")]),
        );
        assert_eq!(el.child_elements().len(), 2);
    }

    #[test]
    fn phase_suffix_and_visibility() {
        assert_eq!(TransitionPhase::Leave.suffix(), "leave");
        assert!(TransitionPhase::Entered.is_visible());
        assert!(!TransitionPhase::Enter.is_visible());
    }

    #[test]
    fn spec_defaults_and_builders() {
        assert_eq!(TransitionSpec::default().duration_ms, 200);
        let spec = TransitionSpec::new(500).easing(Easing::Linear);
        assert_eq!(spec.duration_ms, 500);
        assert_eq!(spec.easing, Easing::Linear);
    }
}
