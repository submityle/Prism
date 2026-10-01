//! Props and the [`Component`] abstraction.
//!
//! A [`Component`] is any value that can turn a typed `Props` value into a
//! [`Element`] subtree. Two flavours are provided:
//!
//! * implement [`Component`] on your own struct when the component needs to
//!   carry state or configuration, or
//! * wrap a plain closure in a [`FnComponent`] when a function is enough.
//!
//! [`mount_component`] is the single entry point that renders either flavour by
//! threading an owned `Props` value through [`Component::render`].

use alloc::string::String;
use alloc::vec::Vec;
use core::marker::PhantomData;

use prism_ui::Element;

/// A reusable builder bag of the properties shared by most components.
///
/// `Props` is a convenience value type: it accumulates style classes and child
/// [`Element`]s through chained builder methods and hands them back through
/// borrowing accessors. Components are free to use it directly as their
/// [`Component::Props`] type or to define their own richer props struct.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Props {
    classes: Vec<String>,
    children: Vec<Element>,
}

impl Props {
    /// Creates an empty `Props` bag.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a style class, preserving insertion order.
    #[must_use]
    pub fn class(mut self, name: impl Into<String>) -> Self {
        self.classes.push(name.into());
        self
    }

    /// Appends a single child [`Element`].
    #[must_use]
    pub fn child(mut self, child: Element) -> Self {
        self.children.push(child);
        self
    }

    /// Appends many child [`Element`]s.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children.extend(children);
        self
    }

    /// The accumulated class names, in insertion order.
    #[must_use]
    pub fn class_names(&self) -> &[String] {
        &self.classes
    }

    /// The accumulated child elements, in insertion order.
    #[must_use]
    pub fn child_elements(&self) -> &[Element] {
        &self.children
    }
}

/// A unit of UI that renders a typed `Props` value into an [`Element`] tree.
///
/// Implementors declare their input via the [`Component::Props`] associated
/// type and produce a view in [`Component::render`]. `render` takes `&self`, so
/// a component may hold configuration or shared state while remaining cheap to
/// call every frame.
pub trait Component {
    /// The typed input this component renders from.
    type Props;

    /// Renders the component into a data-only [`Element`] subtree.
    fn render(&self, props: &Self::Props) -> Element;
}

/// An adapter that turns a `Fn(&P) -> Element` closure into a [`Component`].
///
/// This lets a plain function or closure be used anywhere a [`Component`] is
/// expected without declaring a dedicated struct. The props type `P` is
/// recovered from the closure signature.
pub struct FnComponent<P, F> {
    render: F,
    _marker: PhantomData<fn(&P)>,
}

impl<P, F> FnComponent<P, F>
where
    F: Fn(&P) -> Element,
{
    /// Wraps `render` so the closure can be used as a [`Component`].
    #[must_use]
    pub fn new(render: F) -> Self {
        Self {
            render,
            _marker: PhantomData,
        }
    }
}

impl<P, F> Component for FnComponent<P, F>
where
    F: Fn(&P) -> Element,
{
    type Props = P;

    fn render(&self, props: &Self::Props) -> Element {
        (self.render)(props)
    }
}

/// Renders `component` with an owned `props` value.
///
/// This is the canonical way to invoke a [`Component`]: it borrows the props
/// for the duration of [`Component::render`] and returns the resulting
/// [`Element`] tree. The props are consumed so callers can build them inline.
#[must_use]
pub fn mount_component<C: Component>(component: &C, props: C::Props) -> Element {
    component.render(&props)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::std_instead_of_alloc, reason = "tests run under std")]

    use super::*;
    use prism_ui::ElementKind;

    #[test]
    fn fn_component_renders_expected_element() {
        let greeting =
            FnComponent::new(|name: &String| Element::box_().child(Element::text(name.clone())));
        let view = mount_component(&greeting, "world".to_string());

        assert_eq!(view.kind(), &ElementKind::Box);
        assert_eq!(view.child_elements()[0].text_content(), Some("world"));
    }

    struct Card;

    struct CardProps {
        label: String,
    }

    impl Component for Card {
        type Props = CardProps;

        fn render(&self, props: &Self::Props) -> Element {
            Element::box_()
                .class("card")
                .child(Element::text(props.label.clone()))
        }
    }

    #[test]
    fn component_struct_renders_and_passes_props() {
        let view = mount_component(
            &Card,
            CardProps {
                label: "hi".to_string(),
            },
        );

        assert_eq!(view.class_names(), &["card".to_string()]);
        assert_eq!(view.child_elements()[0].text_content(), Some("hi"));
    }

    #[test]
    fn props_builder_accumulates() {
        let props = Props::new()
            .class("a")
            .class("b")
            .child(Element::text("x"))
            .children([Element::text("y"), Element::text("z")]);

        assert_eq!(props.class_names(), &["a".to_string(), "b".to_string()]);
        assert_eq!(props.child_elements().len(), 3);
    }
}
