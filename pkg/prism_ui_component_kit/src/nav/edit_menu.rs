//! [`EditMenu`] — the horizontal "edit menu" command bar.
//!
//! This mirrors the system text-selection edit menu (the capsule bar that
//! surfaces `Cut | Copy | Paste | …`): a single glass pill laying out command
//! entries left-to-right, each separated from the next by a hairline rule, with
//! an optional trailing overflow chevron. Destructive entries (such as
//! `Delete`) paint in the system red; disabled entries dim via the shared
//! `is-disabled` state.
//!
//! The container exposes the [`Role::Menu`](prism_ui_a11y::Role::Menu) role and
//! each entry the [`Role::MenuItem`](prism_ui_a11y::Role::MenuItem) role. The
//! separators are purely decorative and carry no role. Only kit class names are
//! attached; surface, spacing, radius and typography resolve from theme tokens,
//! so the bar flips between light and dark with the active appearance.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// A single edit-menu entry: a label plus its emphasis and interactivity.
#[derive(Clone, Debug, PartialEq)]
pub struct EditMenuItem {
    /// The visible command label.
    pub label: String,
    /// Whether the entry is destructive (painted in the system red).
    pub destructive: bool,
    /// Whether the entry is non-interactive (dimmed).
    pub disabled: bool,
}

impl EditMenuItem {
    /// Creates an enabled, non-destructive entry with the given label.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            destructive: false,
            disabled: false,
        }
    }

    /// Marks the entry destructive, so it paints in the system red.
    #[must_use]
    pub fn destructive(mut self, destructive: bool) -> Self {
        self.destructive = destructive;
        self
    }

    /// Sets whether the entry is disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// Props for [`EditMenu`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct EditMenuProps {
    /// The command entries, laid out left-to-right.
    pub items: Vec<EditMenuItem>,
    /// Whether to append a trailing overflow chevron (`›`) indicating more
    /// commands are available behind a page turn.
    pub overflow: bool,
}

impl EditMenuProps {
    /// Creates empty edit-menu props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a command entry.
    #[must_use]
    pub fn item(mut self, item: EditMenuItem) -> Self {
        self.items.push(item);
        self
    }

    /// Replaces the entries with `items`.
    #[must_use]
    pub fn items<I: IntoIterator<Item = EditMenuItem>>(mut self, items: I) -> Self {
        self.items = items.into_iter().collect();
        self
    }

    /// Enables (or disables) the trailing overflow chevron.
    #[must_use]
    pub fn overflow(mut self, overflow: bool) -> Self {
        self.overflow = overflow;
        self
    }
}

/// The edit-menu control. Zero-sized; config lives in [`EditMenuProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct EditMenu;

impl EditMenu {
    /// The accessibility role the bar exposes.
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

impl Component for EditMenu {
    type Props = EditMenuProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-edit-menu");
        for (index, item) in props.items.iter().enumerate() {
            if index > 0 {
                el = el.child(Element::box_().class("pk-edit-menu__sep"));
            }
            let mut entry = Element::box_()
                .class("pk-edit-menu__item")
                .child(Element::text(item.label.clone()).class("pk-edit-menu__label"));
            if item.destructive {
                entry = entry.class("pk-edit-menu__item--destructive");
            }
            if item.disabled {
                entry = entry.class("is-disabled");
            }
            el = el.child(entry);
        }
        if props.overflow {
            el = el
                .child(Element::box_().class("pk-edit-menu__sep"))
                .child(
                    Element::box_()
                        .class("pk-edit-menu__item")
                        .class("pk-edit-menu__overflow")
                        .child(Element::text(String::from(">")).class("pk-edit-menu__label")),
                );
        }
        el
    }
}

/// Registers the `pk-edit-menu` class family: the capsule glass bar, the entry
/// rows, the destructive emphasis, the hairline separators and the overflow
/// chevron.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Container: a single horizontal glass capsule. `Stretch` lets the hairline
    // separators span the full bar height while each entry centers its own
    // label.
    sheet.insert(
        Class::new("pk-edit-menu")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Stretch))
            .with(StyleProp::BorderRadius, tok("radius.xl"))
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with_shadow(0.0, 8.0, 24.0, tok("glass.shadow")),
    );

    // Entry: a centered, padded command cell with subheadline typography and a
    // hover wash. Padding-y establishes the intrinsic bar height.
    sheet.insert(
        Class::new("pk-edit-menu__item")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with_state(
                InteractionState::Hover,
                StyleProp::BackgroundColor,
                tok("color.fill.secondary"),
            ),
    );

    // Destructive entry: system red label.
    sheet.insert(
        Class::new("pk-edit-menu__item--destructive").with(StyleProp::Color, tok("color.red")),
    );

    // Label carries no color of its own: it inherits the entry color so the
    // destructive red (set on the entry) cascades down to the glyphs.

    // Overflow chevron: slightly tighter so the `›` reads as a page turn.
    sheet.insert(
        Class::new("pk-edit-menu__overflow")
            .with_padding_x(tok("space.sm"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Hairline separator: a 1px full-height rule between entries.
    sheet.insert(
        Class::new("pk-edit-menu__sep")
            .with(StyleProp::Width, StyleValue::px(1.0))
            .with(StyleProp::BackgroundColor, tok("color.separator")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: EditMenuProps) -> Element {
        EditMenu.render(&props)
    }

    #[test]
    fn empty_edit_menu_has_no_children() {
        let el = render(EditMenuProps::new());
        assert_eq!(el.class_names(), ["pk-edit-menu"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn separators_interleave_entries() {
        let el = render(
            EditMenuProps::new()
                .item(EditMenuItem::new("Cut"))
                .item(EditMenuItem::new("Copy"))
                .item(EditMenuItem::new("Paste")),
        );
        let kids = el.child_elements();
        // item, sep, item, sep, item => 5 children.
        assert_eq!(kids.len(), 5);
        assert_eq!(kids[0].class_names(), ["pk-edit-menu__item"]);
        assert_eq!(kids[1].class_names(), ["pk-edit-menu__sep"]);
        assert_eq!(kids[3].class_names(), ["pk-edit-menu__sep"]);
    }

    #[test]
    fn destructive_entry_gets_modifier() {
        let el = render(EditMenuProps::new().item(EditMenuItem::new("Delete").destructive(true)));
        assert_eq!(
            el.child_elements()[0].class_names(),
            ["pk-edit-menu__item", "pk-edit-menu__item--destructive"]
        );
    }

    #[test]
    fn disabled_entry_gets_is_disabled() {
        let el = render(EditMenuProps::new().item(EditMenuItem::new("Paste").disabled(true)));
        assert_eq!(
            el.child_elements()[0].class_names(),
            ["pk-edit-menu__item", "is-disabled"]
        );
    }

    #[test]
    fn overflow_appends_chevron() {
        let el = render(
            EditMenuProps::new()
                .item(EditMenuItem::new("Cut"))
                .overflow(true),
        );
        let kids = el.child_elements();
        // item, sep, overflow => 3 children.
        assert_eq!(kids.len(), 3);
        let last = kids.last().unwrap();
        assert_eq!(
            last.class_names(),
            ["pk-edit-menu__item", "pk-edit-menu__overflow"]
        );
        assert_eq!(
            last.child_elements()[0].text_content(),
            Some(">")
        );
    }

    #[test]
    fn roles_are_menu_and_menu_item() {
        assert_eq!(EditMenu::role(), Role::Menu);
        assert_eq!(EditMenu::item_role(), Role::MenuItem);
    }
}
