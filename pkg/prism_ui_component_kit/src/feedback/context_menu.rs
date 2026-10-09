//! [`ContextMenu`] — a floating list of actions built on the overlay base.
//!
//! A context menu is a list of [`ContextMenuItem`]s shown in a floating glass
//! surface. Each item is a label with optional `disabled` and `danger` states.
//! It composes the shared [`Popover`] base per architecture invariant K7,
//! exposes [`Role::Menu`] for assistive tech, and attaches only kit classes;
//! [`crate::preset`] resolves every value against the active theme.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::feedback::popover::{Popover, PopoverProps};
use crate::kit::classes;
use crate::preset::StyleSheet;

/// A single entry in a [`ContextMenu`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ContextMenuItem {
    /// The visible action label.
    pub label: String,
    /// Whether the item is non-interactive.
    pub disabled: bool,
    /// Whether the item is destructive (painted with the danger accent).
    pub danger: bool,
}

impl ContextMenuItem {
    /// Creates an enabled, non-destructive item with `label`.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ..Self::default()
        }
    }

    /// Marks the item disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Marks the item destructive.
    #[must_use]
    pub fn danger(mut self, danger: bool) -> Self {
        self.danger = danger;
        self
    }
}

/// Props for [`ContextMenu`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ContextMenuProps {
    /// The ordered menu entries.
    pub items: Vec<ContextMenuItem>,
    /// Whether the menu is currently shown.
    pub open: bool,
}

impl ContextMenuProps {
    /// Creates empty, closed context-menu props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a menu entry.
    #[must_use]
    pub fn item(mut self, item: ContextMenuItem) -> Self {
        self.items.push(item);
        self
    }

    /// Replaces the entries with `items`.
    #[must_use]
    pub fn items<I: IntoIterator<Item = ContextMenuItem>>(mut self, items: I) -> Self {
        self.items = items.into_iter().collect();
        self
    }

    /// Sets whether the menu is shown.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }
}

/// The context-menu control. Zero-sized; configuration lives in
/// [`ContextMenuProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ContextMenu;

impl ContextMenu {
    /// The accessibility role a context menu exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Menu
    }
}

impl Component for ContextMenu {
    type Props = ContextMenuProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut menu = Element::box_().class("pk-context-menu");
        for item in &props.items {
            let mut mods: Vec<&str> = Vec::new();
            if item.disabled {
                mods.push("disabled");
            }
            if item.danger {
                mods.push("danger");
            }
            let mut row = Element::text(item.label.clone());
            for name in classes("pk-context-menu__item", &mods) {
                row = row.class(name);
            }
            menu = menu.child(row);
        }

        Popover.render(&PopoverProps::new().open(props.open).child(menu))
    }
}

/// Registers the `pk-context-menu` class family: the vertical list, the item
/// row, and the `--disabled` / `--danger` modifiers.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // List: a tight vertical stack of rows.
    sheet.insert(
        Class::new("pk-context-menu")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::MinWidth, StyleValue::px(180.0)),
    );

    // Item: a padded, rounded row that highlights on hover.
    sheet.insert(
        Class::new("pk-context-menu__item")
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    // Disabled: dimmed and inert.
    sheet.insert(
        Class::new("pk-context-menu__item--disabled")
            .with(StyleProp::Color, tok("color.label.tertiary"))
            .with_state(InteractionState::Disabled, StyleProp::Opacity, StyleValue::number(0.4)),
    );

    // Danger: destructive red label.
    sheet.insert(
        Class::new("pk-context-menu__item--danger").with(StyleProp::Color, tok("color.red")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: ContextMenuProps) -> Element {
        ContextMenu.render(&props)
    }

    fn menu(el: &Element) -> Element {
        let surface = el
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-popover__surface"))
            .expect("surface")
            .clone();
        surface.child_elements()[0].clone()
    }

    #[test]
    fn role_is_menu() {
        assert_eq!(ContextMenu::role(), Role::Menu);
    }

    #[test]
    fn composes_popover_base() {
        let el = render(ContextMenuProps::new().open(true));
        assert!(el.class_names().iter().any(|c| c == "pk-popover"));
        assert!(el.class_names().iter().any(|c| c == "is-open"));
    }

    #[test]
    fn items_render_as_classed_rows() {
        let el = render(
            ContextMenuProps::new()
                .item(ContextMenuItem::new("Open"))
                .item(ContextMenuItem::new("Delete").danger(true))
                .item(ContextMenuItem::new("Rename").disabled(true)),
        );
        let menu = menu(&el);
        assert!(menu.class_names().iter().any(|c| c == "pk-context-menu"));
        let rows = menu.child_elements();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].text_content(), Some("Open"));
        assert!(rows[0].class_names().iter().any(|c| c == "pk-context-menu__item"));
        assert!(rows[1]
            .class_names()
            .iter()
            .any(|c| c == "pk-context-menu__item--danger"));
        assert!(rows[2]
            .class_names()
            .iter()
            .any(|c| c == "pk-context-menu__item--disabled"));
    }
}
