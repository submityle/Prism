//! [`EmptyState`] — the placeholder shown when a view has no content.
//!
//! An empty state centers an optional `icon`, a `title`, an optional
//! `description`, and an optional `action` slot (e.g. a button). It is a pure
//! layout composition: all spacing and typography come from theme tokens, and
//! it attaches only `pk-empty-state` class names.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`EmptyState`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct EmptyStateProps {
    /// Optional leading icon/illustration slot.
    pub icon: Option<Element>,
    /// The primary title line.
    pub title: String,
    /// Optional supporting description.
    pub description: Option<String>,
    /// Optional trailing action slot (e.g. a call-to-action button).
    pub action: Option<Element>,
}

impl EmptyStateProps {
    /// Creates props for an empty state with the given `title`.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Self::default()
        }
    }

    /// Sets the icon slot.
    #[must_use]
    pub fn icon(mut self, element: Element) -> Self {
        self.icon = Some(element);
        self
    }

    /// Sets the supporting description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the action slot.
    #[must_use]
    pub fn action(mut self, element: Element) -> Self {
        self.action = Some(element);
        self
    }
}

/// The empty-state control. Zero-sized; config lives in [`EmptyStateProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct EmptyState;

impl Component for EmptyState {
    type Props = EmptyStateProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-empty-state");
        if let Some(icon) = props.icon.clone() {
            el = el.child(icon.class("pk-empty-state__icon"));
        }
        el = el.child(Element::text(props.title.clone()).class("pk-empty-state__title"));
        if let Some(description) = props.description.clone() {
            el = el.child(
                Element::text(description).class("pk-empty-state__description"),
            );
        }
        if let Some(action) = props.action.clone() {
            el = el.child(action.class("pk-empty-state__action"));
        }
        el
    }
}

/// Registers the `pk-empty-state` class family: container plus four slots.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Container: a centered vertical stack.
    sheet.insert(
        Class::new("pk-empty-state")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.sm"))
            .with_padding_x(tok("space.xl"))
            .with_padding_y(tok("space.xl")),
    );

    // Icon: tertiary tint so it reads as a quiet affordance.
    sheet.insert(
        Class::new("pk-empty-state__icon").with(StyleProp::Color, tok("color.label.tertiary")),
    );

    // Title: headline weight, primary label.
    sheet.insert(
        Class::new("pk-empty-state__title")
            .with(StyleProp::FontSize, tok("font.size.headline"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Description: body text, secondary.
    sheet.insert(
        Class::new("pk-empty-state__description")
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Action: a little breathing room above the CTA.
    sheet.insert(
        Class::new("pk-empty-state__action").with(StyleProp::MarginTop, tok("space.sm")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    fn render(props: EmptyStateProps) -> Element {
        EmptyState.render(&props)
    }

    #[test]
    fn title_only_yields_single_text_child() {
        let el = render(EmptyStateProps::new("Nothing here"));
        assert_eq!(el.class_names(), ["pk-empty-state"]);
        assert_eq!(el.child_elements().len(), 1);
        let title = &el.child_elements()[0];
        assert_eq!(title.kind(), &ElementKind::Text);
        assert_eq!(title.text_content(), Some("Nothing here"));
    }

    #[test]
    fn full_slots_render_in_order() {
        let el = render(
            EmptyStateProps::new("Empty")
                .icon(Element::box_().class("icon"))
                .description("Add your first item")
                .action(Element::box_().class("cta")),
        );
        let children = el.child_elements();
        assert_eq!(children.len(), 4);
        assert!(children[0].class_names().iter().any(|c| c == "pk-empty-state__icon"));
        assert!(children[1].class_names().iter().any(|c| c == "pk-empty-state__title"));
        assert!(children[2]
            .class_names()
            .iter()
            .any(|c| c == "pk-empty-state__description"));
        assert!(children[3].class_names().iter().any(|c| c == "pk-empty-state__action"));
    }

    #[test]
    fn description_is_optional() {
        let el = render(EmptyStateProps::new("X").action(Element::box_()));
        assert_eq!(el.child_elements().len(), 2);
    }
}
