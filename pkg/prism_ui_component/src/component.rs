//! The rendering context that ties props, context and slots together.
//!
//! [`ComponentCtx`] is the ambient environment handed to a component while it
//! renders: it borrows a [`ContextMap`] for dependency injection and may carry
//! an owned [`Slots`] collection describing the children the caller supplied.
//! [`render_with_context`] is a tiny helper that threads such a context into a
//! rendering closure, keeping call sites uniform.

use alloc::rc::Rc;

use prism_ui::Element;

use crate::context::ContextMap;
use crate::slots::Slots;

/// The ambient environment a component renders against.
///
/// A `ComponentCtx` borrows the active [`ContextMap`] so a component can
/// [`inject`](ComponentCtx::inject) its dependencies, and optionally owns the
/// [`Slots`] the caller passed in. It is cheap to build and intended to live
/// only for the duration of a render.
pub struct ComponentCtx<'a> {
    context: &'a ContextMap,
    slots: Option<Slots>,
}

impl<'a> ComponentCtx<'a> {
    /// Creates a context backed by `context` with no slots.
    #[must_use]
    pub fn new(context: &'a ContextMap) -> Self {
        Self {
            context,
            slots: None,
        }
    }

    /// Attaches a [`Slots`] collection to this context.
    #[must_use]
    pub fn with_slots(mut self, slots: Slots) -> Self {
        self.slots = Some(slots);
        self
    }

    /// Borrows the underlying [`ContextMap`].
    #[must_use]
    pub fn context(&self) -> &ContextMap {
        self.context
    }

    /// Injects a shared value of type `T` from the underlying [`ContextMap`].
    #[must_use]
    pub fn inject<T: 'static>(&self) -> Option<Rc<T>> {
        self.context.inject::<T>()
    }

    /// Borrows the attached [`Slots`], if any were supplied.
    #[must_use]
    pub fn slots(&self) -> Option<&Slots> {
        self.slots.as_ref()
    }
}

/// Renders an [`Element`] by handing `ctx` to the `render` closure.
///
/// This keeps the common "render against an ambient [`ComponentCtx`]" pattern
/// at a single call shape, so components can inject dependencies and read slots
/// without each site re-implementing the plumbing.
#[must_use]
pub fn render_with_context(
    ctx: &ComponentCtx<'_>,
    render: impl FnOnce(&ComponentCtx<'_>) -> Element,
) -> Element {
    render(ctx)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::std_instead_of_alloc, reason = "tests run under std")]

    use super::*;
    use prism_ui::ElementKind;

    struct Theme {
        label: String,
    }

    #[test]
    fn component_reads_context_value_into_text() {
        let mut ctx_map = ContextMap::new();
        ctx_map.provide(Theme {
            label: "midnight".to_string(),
        });

        let ctx = ComponentCtx::new(&ctx_map);
        let view = render_with_context(&ctx, |ctx| {
            let theme = ctx.inject::<Theme>().expect("theme provided");
            Element::box_().child(Element::text(theme.label.clone()))
        });

        assert_eq!(view.kind(), &ElementKind::Box);
        assert_eq!(view.child_elements()[0].text_content(), Some("midnight"));
    }

    #[test]
    fn context_carries_slots() {
        let ctx_map = ContextMap::new();
        let slots = Slots::new().with_default(Element::text("child"));
        let ctx = ComponentCtx::new(&ctx_map).with_slots(slots);

        let supplied = ctx.slots().expect("slots attached");
        assert_eq!(supplied.default_slot().len(), 1);
    }
}
