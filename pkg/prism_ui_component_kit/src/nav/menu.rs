//! [`Menu`] — a vertical list of command entries.
//!
//! A menu renders a `pk-menu` container of `pk-menu__item` rows; a disabled
//! entry gains the shared `is-disabled` state class. The menu exposes the
//! [`Role::Menu`](prism_ui_a11y::Role::Menu) role and each item the
//! [`Role::MenuItem`](prism_ui_a11y::Role::MenuItem) role. Only kit class names
//! are attached; surface, spacing and typography resolve from theme tokens.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// A single menu entry: a label plus whether it is selectable.
#[derive(Clone, Debug, PartialEq)]
pub struct MenuEntry {
    /// The visible entry label.
    pub label: String,
    /// Whether the entry is non-interactive.
    pub disabled: bool,
}

impl MenuEntry {
    /// Creates an enabled entry with the given label.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            disabled: false,
        }
    }

    /// Sets whether the entry is disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// Props for [`Menu`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MenuProps {
    /// The entries, rendered top-to-bottom.
    pub items: Vec<MenuEntry>,
}

impl MenuProps {
    /// Creates empty menu props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an entry.
    #[must_use]
    pub fn item(mut self, item: MenuEntry) -> Self {
        self.items.push(item);
        self
    }

    /// Replaces the entries with `items`.
    #[must_use]
    pub fn items<I: IntoIterator<Item = MenuEntry>>(mut self, items: I) -> Self {
        self.items = items.into_iter().collect();
        self
    }
}

/// The menu control. Zero-sized; config lives in [`MenuProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Menu;

impl Menu {
    /// The accessibility role the menu container exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Menu
    }

    /// The accessibility role each entry exposes.
    #[must_use]
    pub const fn item_role() -> Role {
        Role::MenuItem
    }
}

impl Component for Menu {
    type Props = MenuProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-menu");
        for item in &props.items {
            let mut row = Element::box_()
                .class("pk-menu__item")
                .child(Element::text(item.label.clone()).class("pk-menu__label"));
            if item.disabled {
                row = row.class("is-disabled");
            }
            el = el.child(row);
        }
        el
    }
}

/// Registers the `pk-menu` class family: the container, the item rows and the
/// shared disabled state.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Container: a vertical glass surface with rounded corners and padding.
    sheet.insert(
        Class::new("pk-menu")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with_padding_x(tok("space.xs"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with_shadow(0.0, 8.0, 24.0, tok("glass.shadow")),
    );

    // Item: a padded row with a hover wash and body typography.
    sheet.insert(
        Class::new("pk-menu__item")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    // Label: inherits the item color.
    sheet.insert(
        Class::new("pk-menu__label").with(StyleProp::Color, tok("color.label")),
    );

    // Shared disabled state: dim and non-committal.
    sheet.insert(
        Class::new("is-disabled").with(StyleProp::Opacity, StyleValue::number(0.4)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: MenuProps) -> Element {
        Menu.render(&props)
    }

    #[test]
    fn empty_menu_has_no_items() {
        let el = render(MenuProps::new());
        assert_eq!(el.class_names(), ["pk-menu"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn disabled_entry_gets_is_disabled() {
        let el = render(
            MenuProps::new()
                .item(MenuEntry::new("Open"))
                .item(MenuEntry::new("Delete").disabled(true)),
        );
        let items = el.child_elements();
        assert_eq!(items[0].class_names(), ["pk-menu__item"]);
        assert_eq!(items[1].class_names(), ["pk-menu__item", "is-disabled"]);
    }

    #[test]
    fn item_renders_label_text() {
        let el = render(MenuProps::new().item(MenuEntry::new("Open")));
        let item = &el.child_elements()[0];
        assert_eq!(item.child_elements()[0].text_content(), Some("Open"));
    }

    #[test]
    fn roles_are_menu_and_menu_item() {
        assert_eq!(Menu::role(), Role::Menu);
        assert_eq!(Menu::item_role(), Role::MenuItem);
    }
}
