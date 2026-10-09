//! [`TabBar`] — the bottom tab navigation strip.
//!
//! A tab bar renders a `pk-tab-bar` row of equal-weight `pk-tab-bar__item`
//! tabs, each pairing an optional icon with a label. The item at `selected`
//! gains the shared `is-selected` state class. Each tab exposes the
//! [`Role::Tab`](prism_ui_a11y::Role::Tab) accessibility role. Only kit class
//! names are attached; color and spacing resolve from theme tokens.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// A single tab: a label plus an optional leading icon.
#[derive(Clone, Debug, PartialEq)]
pub struct TabBarItem {
    /// The visible tab label.
    pub label: String,
    /// Optional icon rendered above/before the label.
    pub icon: Option<Element>,
}

impl TabBarItem {
    /// Creates a labelled tab with no icon.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            icon: None,
        }
    }

    /// Sets the tab's icon.
    #[must_use]
    pub fn icon(mut self, icon: Element) -> Self {
        self.icon = Some(icon);
        self
    }
}

/// Props for [`TabBar`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TabBarProps {
    /// The tabs, rendered left-to-right.
    pub items: Vec<TabBarItem>,
    /// The index of the selected tab.
    pub selected: usize,
}

impl TabBarProps {
    /// Creates empty tab-bar props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a tab.
    #[must_use]
    pub fn item(mut self, item: TabBarItem) -> Self {
        self.items.push(item);
        self
    }

    /// Replaces the tabs with `items`.
    #[must_use]
    pub fn items<I: IntoIterator<Item = TabBarItem>>(mut self, items: I) -> Self {
        self.items = items.into_iter().collect();
        self
    }

    /// Sets the selected tab index.
    #[must_use]
    pub fn selected(mut self, selected: usize) -> Self {
        self.selected = selected;
        self
    }
}

/// The tab-bar control. Zero-sized; config lives in [`TabBarProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TabBar;

impl TabBar {
    /// The accessibility role each tab exposes.
    #[must_use]
    pub const fn item_role() -> Role {
        Role::Tab
    }
}

impl Component for TabBar {
    type Props = TabBarProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-tab-bar");
        for (index, item) in props.items.iter().enumerate() {
            let mut tab = Element::box_().class("pk-tab-bar__item");
            if index == props.selected {
                tab = tab.class("is-selected");
            }
            if let Some(icon) = item.icon.clone() {
                tab = tab.child(icon.class("pk-tab-bar__icon"));
            }
            tab = tab.child(Element::text(item.label.clone()).class("pk-tab-bar__label"));
            el = el.child(tab);
        }
        el
    }
}

/// Registers the `pk-tab-bar` class family: bar, item, selected state, icon
/// and label slots.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Bar: a glass chrome strip whose tabs share the width evenly.
    sheet.insert(
        Class::new("pk-tab-bar")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceEvenly))
            .with_padding_y(tok("space.xs"))
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight")))
            .with_shadow(0.0, -1.0, 12.0, tok("glass.shadow")),
    );

    // Item: a centered icon-over-label stack that grows to an equal share.
    sheet.insert(
        Class::new("pk-tab-bar__item")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Shared selected state: accent treatment used across nav controls.
    sheet.insert(
        Class::new("is-selected")
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );

    // Icon slot: never shrinks.
    sheet.insert(
        Class::new("pk-tab-bar__icon").with(StyleProp::FlexShrink, StyleValue::number(0.0)),
    );

    // Label slot: footnote-sized caption.
    sheet.insert(
        Class::new("pk-tab-bar__label").with(StyleProp::FontSize, tok("font.size.footnote")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: TabBarProps) -> Element {
        TabBar.render(&props)
    }

    #[test]
    fn empty_bar_has_no_items() {
        let el = render(TabBarProps::new());
        assert_eq!(el.class_names(), ["pk-tab-bar"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn selected_item_gets_is_selected() {
        let el = render(
            TabBarProps::new()
                .item(TabBarItem::new("Home"))
                .item(TabBarItem::new("Search"))
                .selected(1),
        );
        let items = el.child_elements();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].class_names(), ["pk-tab-bar__item"]);
        assert_eq!(items[1].class_names(), ["pk-tab-bar__item", "is-selected"]);
    }

    #[test]
    fn item_renders_icon_then_label() {
        let el = render(
            TabBarProps::new().item(TabBarItem::new("Home").icon(Element::box_().class("ic"))),
        );
        let item = &el.child_elements()[0];
        let kids = item.child_elements();
        assert_eq!(kids.len(), 2);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-tab-bar__icon"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-tab-bar__label"));
        assert_eq!(kids[1].text_content(), Some("Home"));
    }

    #[test]
    fn item_role_is_tab() {
        assert_eq!(TabBar::item_role(), Role::Tab);
    }
}
