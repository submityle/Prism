//! [`Panel`] — a general titled container.
//!
//! A panel renders a `pk-panel` surface with an optional `pk-panel__title`
//! above a `pk-panel__body` that wraps its children. The `glass` flag swaps the
//! opaque surface for the frosted [`with_glass`](prism_ui_style::Class::with_glass)
//! treatment. Every value resolves from theme tokens via [`crate::preset`]; the
//! control attaches only kit class names.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Panel`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PanelProps {
    /// Optional title rendered above the body.
    pub title: Option<String>,
    /// The panel's body content.
    pub children: Vec<Element>,
    /// Whether to use the frosted-glass surface instead of the opaque one.
    pub glass: bool,
}

impl PanelProps {
    /// Creates empty panel props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the title text.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Appends a body child.
    #[must_use]
    pub fn child(mut self, element: Element) -> Self {
        self.children.push(element);
        self
    }

    /// Replaces the body children with `children`.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children = children.into_iter().collect();
        self
    }

    /// Selects the frosted-glass surface variant.
    #[must_use]
    pub fn glass(mut self, glass: bool) -> Self {
        self.glass = glass;
        self
    }
}

/// The panel control. Zero-sized; all configuration lives in [`PanelProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Panel;

impl Panel {
    /// The accessibility role a panel exposes (a generic grouping).
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for Panel {
    type Props = PanelProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-panel");
        if props.glass {
            el = el.class("pk-panel--glass");
        }
        if let Some(title) = props.title.clone() {
            el = el.child(Element::text(title).class("pk-panel__title"));
        }
        let body = Element::box_()
            .class("pk-panel__body")
            .children(props.children.iter().cloned());
        el.child(body)
    }
}

/// Registers the `pk-panel` class family: surface, glass variant, title, body.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Surface: a vertical stack with rounded corners, a hairline border and a
    // tertiary surface fill.
    sheet.insert(
        Class::new("pk-panel")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.md"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::BackgroundColor, tok("color.surface.tertiary")),
    );

    // Glass variant: frosted tint and lit rim.
    sheet.insert(
        Class::new("pk-panel--glass")
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight"))),
    );

    // Title: headline typography.
    sheet.insert(
        Class::new("pk-panel__title")
            .with(StyleProp::FontSize, tok("font.size.headline"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Body: a vertical stack of the panel's children.
    sheet.insert(
        Class::new("pk-panel__body")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::Color, tok("color.label")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui::ElementKind;

    fn render(props: PanelProps) -> Element {
        Panel.render(&props)
    }

    #[test]
    fn body_only_when_no_title() {
        let el = render(PanelProps::new().child(Element::box_()));
        assert_eq!(el.class_names(), ["pk-panel"]);
        assert_eq!(el.child_elements().len(), 1);
        assert_eq!(el.child_elements()[0].class_names(), ["pk-panel__body"]);
    }

    #[test]
    fn title_precedes_body() {
        let el = render(PanelProps::new().title("Settings").child(Element::box_()));
        let children = el.child_elements();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].kind(), &ElementKind::Text);
        assert_eq!(children[0].text_content(), Some("Settings"));
        assert!(children[0].class_names().iter().any(|c| c == "pk-panel__title"));
        assert_eq!(children[1].class_names(), ["pk-panel__body"]);
    }

    #[test]
    fn glass_adds_modifier_class() {
        let el = render(PanelProps::new().glass(true));
        assert_eq!(el.class_names(), ["pk-panel", "pk-panel--glass"]);
    }

    #[test]
    fn body_wraps_all_children() {
        let el = render(PanelProps::new().children([Element::box_(), Element::box_()]));
        let body = el
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-panel__body"))
            .expect("body");
        assert_eq!(body.child_elements().len(), 2);
    }

    #[test]
    fn role_is_group() {
        assert_eq!(Panel::role(), Role::Group);
    }
}
