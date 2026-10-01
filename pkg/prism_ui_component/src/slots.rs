//! Named child slots via [`Slots`].
//!
//! Component composition often needs more than a single flat list of children:
//! a layout may expose a `header`, a `body` and a `footer`, each filled
//! independently by the caller. [`Slots`] models exactly that — a map of named
//! slots plus one unnamed *default* slot — and [`SlottedComponent`] is a small
//! helper that drops a chosen slot's children into a container [`Element`].

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;

/// A collection of named child [`Element`] lists plus a default slot.
///
/// Each named slot and the default slot hold an ordered list of elements.
/// Builder methods append to a slot, so a slot may be filled incrementally.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Slots {
    named: BTreeMap<String, Vec<Element>>,
    default: Vec<Element>,
}

impl Slots {
    /// Creates an empty slot collection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `element` to the named slot `name`, creating it if needed.
    #[must_use]
    pub fn with(mut self, name: impl Into<String>, element: Element) -> Self {
        self.named.entry(name.into()).or_default().push(element);
        self
    }

    /// Appends `element` to the unnamed default slot.
    #[must_use]
    pub fn with_default(mut self, element: Element) -> Self {
        self.default.push(element);
        self
    }

    /// Borrows the elements in the named slot `name`.
    ///
    /// Returns an empty slice when the slot was never filled.
    #[must_use]
    pub fn named(&self, name: &str) -> &[Element] {
        self.named.get(name).map(Vec::as_slice).unwrap_or_default()
    }

    /// Borrows the elements in the default slot.
    #[must_use]
    pub fn default_slot(&self) -> &[Element] {
        &self.default
    }

    /// Removes and returns the elements of the named slot `name`.
    ///
    /// Returns an empty vector when the slot was never filled.
    #[must_use]
    pub fn take(mut self, name: &str) -> Vec<Element> {
        self.named.remove(name).unwrap_or_default()
    }
}

/// A convenience that fills a container [`Element`] from a chosen slot.
///
/// The component keeps a container template and the name of the slot to pull
/// children from (or the default slot when unset). [`SlottedComponent::render`]
/// clones the container and appends the chosen slot's elements as its children.
#[derive(Clone, Debug, PartialEq)]
pub struct SlottedComponent {
    container: Element,
    slot: Option<String>,
}

impl SlottedComponent {
    /// Creates a component that fills `container` from the default slot.
    #[must_use]
    pub fn new(container: Element) -> Self {
        Self {
            container,
            slot: None,
        }
    }

    /// Selects a named slot to pull children from instead of the default slot.
    #[must_use]
    pub fn slot(mut self, name: impl Into<String>) -> Self {
        self.slot = Some(name.into());
        self
    }

    /// Renders the container with children drawn from the chosen slot.
    #[must_use]
    pub fn render(&self, slots: &Slots) -> Element {
        let children = match &self.slot {
            Some(name) => slots.named(name),
            None => slots.default_slot(),
        };
        self.container.clone().children(children.iter().cloned())
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::std_instead_of_alloc, reason = "tests run under std")]

    use super::*;
    use prism_ui::ElementKind;

    #[test]
    fn default_slot_collects_elements() {
        let slots = Slots::new()
            .with_default(Element::text("a"))
            .with_default(Element::text("b"));

        assert_eq!(slots.default_slot().len(), 2);
        assert_eq!(slots.default_slot()[0].text_content(), Some("a"));
    }

    #[test]
    fn named_slots_and_missing_slot() {
        let slots = Slots::new().with("header", Element::text("title"));

        assert_eq!(slots.named("header").len(), 1);
        assert_eq!(slots.named("header")[0].text_content(), Some("title"));
        assert!(slots.named("footer").is_empty());
    }

    #[test]
    fn take_moves_slot_out() {
        let slots = Slots::new()
            .with("body", Element::text("one"))
            .with("body", Element::text("two"));

        let body = slots.take("body");
        assert_eq!(body.len(), 2);
        assert_eq!(body[1].text_content(), Some("two"));
    }

    #[test]
    fn slotted_component_fills_container() {
        let slots = Slots::new()
            .with("main", Element::text("x"))
            .with("main", Element::text("y"));

        let view = SlottedComponent::new(Element::box_().class("panel"))
            .slot("main")
            .render(&slots);

        assert_eq!(view.kind(), &ElementKind::Box);
        assert_eq!(view.class_names(), &["panel".to_string()]);
        assert_eq!(view.child_elements().len(), 2);
        assert_eq!(view.child_elements()[0].text_content(), Some("x"));
    }
}
