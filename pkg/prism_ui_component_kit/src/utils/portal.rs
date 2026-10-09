//! [`Portal`] — a teleport marker that relocates its subtree.
//!
//! A portal renders no visible chrome of its own. It emits a single
//! [`Element::custom`] node named `pk-portal` that carries the portal's
//! children. The name is a *backend-recognized teleport marker*: when the
//! runtime is wired to an overlay layer, the backend lifts the children out of
//! their normal position in the tree and re-parents them into the overlay root
//! (so popovers, tooltips and dialogs escape ancestor clipping and stacking).
//!
//! Until that wiring exists the node is inert data — the children simply sit
//! inside the custom element — so the control composes and unit-tests without a
//! running runtime, like every other control in the kit.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Portal`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PortalProps {
    /// The subtree to teleport. Rendered as children of the `pk-portal` marker.
    pub children: Vec<Element>,
}

impl PortalProps {
    /// Creates empty portal props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a single child to the teleported subtree.
    #[must_use]
    pub fn child(mut self, child: Element) -> Self {
        self.children.push(child);
        self
    }

    /// Appends many children to the teleported subtree.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children.extend(children);
        self
    }
}

/// The portal control. Zero-sized; all configuration lives in [`PortalProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Portal;

impl Component for Portal {
    type Props = PortalProps;

    fn render(&self, props: &Self::Props) -> Element {
        Element::custom("pk-portal").children(props.children.iter().cloned())
    }
}

/// The `pk-portal` marker carries no style of its own: it is a structural
/// teleport hint resolved by the backend, not a painted surface. This registrar
/// is intentionally empty so the module keeps a uniform `register_styles`
/// surface alongside its siblings.
pub(crate) fn register_styles(_sheet: &mut StyleSheet) {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use prism_ui::ElementKind;

    fn render(props: PortalProps) -> Element {
        Portal.render(&props)
    }

    #[test]
    fn renders_custom_teleport_marker() {
        let el = render(PortalProps::new());
        assert_eq!(el.kind(), &ElementKind::Custom("pk-portal".into()));
    }

    #[test]
    fn carries_children_in_order() {
        let el = render(
            PortalProps::new()
                .child(Element::box_().class("a"))
                .children(vec![Element::box_().class("b"), Element::box_().class("c")]),
        );
        let classes: Vec<_> = el
            .child_elements()
            .iter()
            .map(|c| c.class_names()[0].clone())
            .collect();
        assert_eq!(classes, ["a", "b", "c"]);
    }

    #[test]
    fn marker_has_no_classes_of_its_own() {
        let el = render(PortalProps::new().child(Element::text("x")));
        assert!(el.class_names().is_empty());
    }
}
