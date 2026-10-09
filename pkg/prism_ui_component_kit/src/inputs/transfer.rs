//! [`Transfer`] — a dual-list "shuttle" for moving items between two panels.
//!
//! A transfer renders a `pk-transfer` row of two `__panel`s (source and target)
//! separated by a middle `__actions` column holding the move affordances. Each
//! panel lists its entries as `__item`s. It carries no style of its own: it
//! attaches the `pk-transfer` class family and lets [`crate::preset`] resolve
//! every value against the active theme. Move logic lives in `prism_ui_form`;
//! this control only draws the two columns its props describe.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Transfer`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TransferProps {
    /// Entries in the left (source) panel.
    pub source: Vec<String>,
    /// Entries in the right (target) panel.
    pub target: Vec<String>,
}

impl TransferProps {
    /// Creates empty transfer props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the source entries with `source`.
    #[must_use]
    pub fn source<I, S>(mut self, source: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.source = source.into_iter().map(Into::into).collect();
        self
    }

    /// Replaces the target entries with `target`.
    #[must_use]
    pub fn target<I, S>(mut self, target: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.target = target.into_iter().map(Into::into).collect();
        self
    }
}

/// Builds one `__panel` listing `entries` as `__item`s.
fn panel(entries: &[String]) -> Element {
    let mut p = Element::box_().class("pk-transfer__panel");
    for entry in entries {
        p = p.child(Element::text(entry.clone()).class("pk-transfer__item"));
    }
    p
}

/// The transfer control. Zero-sized; config lives in [`TransferProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Transfer;

impl Transfer {
    /// The accessibility role a transfer exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for Transfer {
    type Props = TransferProps;

    fn render(&self, props: &Self::Props) -> Element {
        let actions = Element::box_()
            .class("pk-transfer__actions")
            .child(Element::box_().class("pk-transfer__action"))
            .child(Element::box_().class("pk-transfer__action"));

        Element::box_()
            .class("pk-transfer")
            .child(panel(&props.source))
            .child(actions)
            .child(panel(&props.target))
    }
}

/// Registers the `pk-transfer` class family: base row, two panels, their items,
/// the middle actions column and its buttons.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-transfer")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Stretch))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    sheet.insert(
        Class::new("pk-transfer__panel")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::FlexBasis, StyleValue::px(0.0))
            .with(StyleProp::MinWidth, StyleValue::px(140.0))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_glass(0.0, tok("color.surface.secondary"), Some(tok("glass.highlight"))),
    );

    sheet.insert(
        Class::new("pk-transfer__item")
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    sheet.insert(
        Class::new("pk-transfer__actions")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs")),
    );

    sheet.insert(
        Class::new("pk-transfer__action")
            .with(StyleProp::Width, StyleValue::px(28.0))
            .with(StyleProp::Height, StyleValue::px(28.0))
            .with(StyleProp::MinWidth, StyleValue::px(28.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.tint"))
            .with_state(InteractionState::Disabled, StyleProp::Opacity, StyleValue::number(0.4)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: TransferProps) -> Element {
        Transfer.render(&props)
    }

    #[test]
    fn renders_two_panels_around_actions() {
        let el = render(TransferProps::new().source(["a", "b"]).target(["c"]));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 3);
        assert_eq!(kids[0].class_names(), ["pk-transfer__panel"]);
        assert_eq!(kids[1].class_names(), ["pk-transfer__actions"]);
        assert_eq!(kids[2].class_names(), ["pk-transfer__panel"]);
    }

    #[test]
    fn panels_list_their_entries() {
        let el = render(TransferProps::new().source(["a", "b"]).target(["c"]));
        let kids = el.child_elements();
        assert_eq!(kids[0].child_elements().len(), 2);
        assert_eq!(kids[0].child_elements()[1].text_content(), Some("b"));
        assert_eq!(kids[2].child_elements().len(), 1);
        assert_eq!(kids[2].child_elements()[0].text_content(), Some("c"));
    }

    #[test]
    fn actions_column_has_two_buttons() {
        let el = render(TransferProps::new());
        let actions = &el.child_elements()[1];
        assert_eq!(actions.child_elements().len(), 2);
        assert!(actions.child_elements()[0].class_names().iter().any(|c| c == "pk-transfer__action"));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(Transfer::role(), Role::Group);
    }
}
