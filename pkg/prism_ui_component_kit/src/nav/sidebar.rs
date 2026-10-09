//! [`Sidebar`] — a vertical navigation list (a.k.a. [`NavRail`]).
//!
//! A sidebar renders a `pk-sidebar` column of `pk-sidebar__item` rows, each
//! pairing an optional leading icon with a label. The selected item gains the
//! block-level `pk-sidebar__item--selected` modifier (not a shared `is-*`
//! state class). Setting `collapsed` swaps the full-width rail for the narrow
//! `pk-sidebar--collapsed` modifier. Only kit class names are attached; color
//! and spacing resolve from theme tokens.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// A single sidebar row: a label plus an optional leading icon.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SidebarItem {
    /// The visible row label.
    pub label: String,
    /// Optional icon rendered before the label.
    pub icon: Option<Element>,
    /// Whether this row is the selected one.
    pub selected: bool,
}

impl SidebarItem {
    /// Creates a labelled, unselected row with no icon.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            icon: None,
            selected: false,
        }
    }

    /// Sets the row's leading icon.
    #[must_use]
    pub fn icon(mut self, icon: Element) -> Self {
        self.icon = Some(icon);
        self
    }

    /// Marks the row selected.
    #[must_use]
    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }
}

/// Props for [`Sidebar`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SidebarProps {
    /// The rows, rendered top-to-bottom.
    pub items: Vec<SidebarItem>,
    /// Whether the rail is in its narrow, collapsed form.
    pub collapsed: bool,
}

impl SidebarProps {
    /// Creates empty sidebar props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a row.
    #[must_use]
    pub fn item(mut self, item: SidebarItem) -> Self {
        self.items.push(item);
        self
    }

    /// Replaces the rows with `items`.
    #[must_use]
    pub fn items<I: IntoIterator<Item = SidebarItem>>(mut self, items: I) -> Self {
        self.items = items.into_iter().collect();
        self
    }

    /// Sets the collapsed state.
    #[must_use]
    pub fn collapsed(mut self, collapsed: bool) -> Self {
        self.collapsed = collapsed;
        self
    }
}

/// The sidebar control. Zero-sized; config lives in [`SidebarProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Sidebar;

/// A navigation rail — the same control as [`Sidebar`], named for the
/// compact-first use case.
pub type NavRail = Sidebar;

impl Component for Sidebar {
    type Props = SidebarProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mods: &[&str] = if props.collapsed { &["collapsed"] } else { &[] };
        let mut el = Element::box_();
        for name in classes("pk-sidebar", mods) {
            el = el.class(name);
        }

        for item in &props.items {
            let item_mods: &[&str] = if item.selected { &["selected"] } else { &[] };
            let mut row = Element::box_();
            for name in classes("pk-sidebar__item", item_mods) {
                row = row.class(name);
            }
            if let Some(icon) = item.icon.clone() {
                row = row.child(icon.class("pk-sidebar__icon"));
            }
            row = row.child(Element::text(item.label.clone()).class("pk-sidebar__label"));
            el = el.child(row);
        }
        el
    }
}

/// Registers the `pk-sidebar` class family: the rail, the collapsed modifier,
/// the item row, its selected modifier, and the icon / label slots.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Rail: a padded vertical column over a secondary surface.
    sheet.insert(
        Class::new("pk-sidebar")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::Width, StyleValue::px(240.0))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::BackgroundColor, tok("color.surface.secondary"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Collapsed: a narrow icon-only rail.
    sheet.insert(
        Class::new("pk-sidebar--collapsed").with(StyleProp::Width, StyleValue::px(64.0)),
    );

    // Item: a centered icon + label row with a rounded hit area.
    sheet.insert(
        Class::new("pk-sidebar__item")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.sm"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Selected: accent treatment on a tinted fill.
    sheet.insert(
        Class::new("pk-sidebar__item--selected")
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );

    // Icon slot: never shrinks.
    sheet.insert(
        Class::new("pk-sidebar__icon").with(StyleProp::FlexShrink, StyleValue::number(0.0)),
    );

    // Label slot: grows to fill the row.
    sheet.insert(
        Class::new("pk-sidebar__label").with(StyleProp::FlexGrow, StyleValue::number(1.0)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: SidebarProps) -> Element {
        Sidebar.render(&props)
    }

    #[test]
    fn empty_rail_has_only_base_class() {
        let el = render(SidebarProps::new());
        assert_eq!(el.class_names(), ["pk-sidebar"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn collapsed_adds_modifier_class() {
        let el = render(SidebarProps::new().collapsed(true));
        assert_eq!(el.class_names(), ["pk-sidebar", "pk-sidebar--collapsed"]);
    }

    #[test]
    fn selected_item_gets_block_level_modifier() {
        let el = render(
            SidebarProps::new()
                .item(SidebarItem::new("Home"))
                .item(SidebarItem::new("Search").selected(true)),
        );
        let items = el.child_elements();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].class_names(), ["pk-sidebar__item"]);
        assert_eq!(
            items[1].class_names(),
            ["pk-sidebar__item", "pk-sidebar__item--selected"]
        );
    }

    #[test]
    fn item_renders_icon_then_label() {
        let el = render(
            SidebarProps::new().item(SidebarItem::new("Home").icon(Element::box_().class("ic"))),
        );
        let item = &el.child_elements()[0];
        let kids = item.child_elements();
        assert_eq!(kids.len(), 2);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-sidebar__icon"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-sidebar__label"));
        assert_eq!(kids[1].text_content(), Some("Home"));
    }

    #[test]
    fn nav_rail_is_sidebar_alias() {
        let el = NavRail::default().render(&SidebarProps::new());
        assert_eq!(el.class_names(), ["pk-sidebar"]);
    }
}
