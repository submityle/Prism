//! [`Tabs`] — a tab bar plus the selected panel.
//!
//! [`Tabs`] composes two lower-level controls: [`TabList`] renders the
//! `pk-tabs__list` of `pk-tabs__tab` buttons (one carries `pk-tabs__tab--selected`), and
//! [`TabPanel`] renders the `pk-tabs__panel` for the active tab's content. Tab
//! buttons expose the [`Role::Tab`](prism_ui_a11y::Role::Tab) role. Colors and
//! spacing resolve from theme tokens via [`crate::preset`]; the controls attach
//! only kit class names.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// A single tab: its label and the content shown when selected.
#[derive(Clone, Debug, PartialEq)]
pub struct TabItem {
    /// The label rendered inside the tab button.
    pub label: Element,
    /// The content rendered in the panel when this tab is selected.
    pub content: Element,
}

impl TabItem {
    /// Creates a tab from a `label` and its `content`.
    #[must_use]
    pub fn new(label: Element, content: Element) -> Self {
        Self { label, content }
    }
}

/// Props for [`TabList`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TabListProps {
    /// The tab labels, in order.
    pub labels: Vec<Element>,
    /// The index of the selected tab.
    pub selected: usize,
}

impl TabListProps {
    /// Creates empty tab-list props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the labels with `labels`.
    #[must_use]
    pub fn labels<I: IntoIterator<Item = Element>>(mut self, labels: I) -> Self {
        self.labels = labels.into_iter().collect();
        self
    }

    /// Sets the selected tab index.
    #[must_use]
    pub fn selected(mut self, selected: usize) -> Self {
        self.selected = selected;
        self
    }
}

/// The tab-list control: a row of tab buttons. Config lives in [`TabListProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TabList;

impl TabList {
    /// The accessibility role each tab button exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Tab
    }
}

impl Component for TabList {
    type Props = TabListProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-tabs__list");
        for (index, label) in props.labels.iter().enumerate() {
            let mut tab = label.clone().class("pk-tabs__tab");
            if index == props.selected {
                tab = tab.class("pk-tabs__tab--selected");
            }
            el = el.child(tab);
        }
        el
    }
}

/// Props for [`TabPanel`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TabPanelProps {
    /// The content rendered inside the panel.
    pub content: Option<Element>,
}

impl TabPanelProps {
    /// Creates empty tab-panel props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the panel content.
    #[must_use]
    pub fn content(mut self, element: Element) -> Self {
        self.content = Some(element);
        self
    }
}

/// The tab-panel control: the body for the selected tab.
#[derive(Clone, Copy, Debug, Default)]
pub struct TabPanel;

impl Component for TabPanel {
    type Props = TabPanelProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-tabs__panel");
        if let Some(content) = props.content.clone() {
            el = el.child(content);
        }
        el
    }
}

/// Props for [`Tabs`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TabsProps {
    /// The tabs, in order.
    pub tabs: Vec<TabItem>,
    /// The index of the selected tab.
    pub selected: usize,
}

impl TabsProps {
    /// Creates empty tabs props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a tab.
    #[must_use]
    pub fn tab(mut self, tab: TabItem) -> Self {
        self.tabs.push(tab);
        self
    }

    /// Replaces the tabs with `tabs`.
    #[must_use]
    pub fn tabs<I: IntoIterator<Item = TabItem>>(mut self, tabs: I) -> Self {
        self.tabs = tabs.into_iter().collect();
        self
    }

    /// Sets the selected tab index.
    #[must_use]
    pub fn selected(mut self, selected: usize) -> Self {
        self.selected = selected;
        self
    }
}

/// The tabs control. Zero-sized; all configuration lives in [`TabsProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Tabs;

impl Component for Tabs {
    type Props = TabsProps;

    fn render(&self, props: &Self::Props) -> Element {
        let list = TabList.render(
            &TabListProps::new()
                .labels(props.tabs.iter().map(|t| t.label.clone()))
                .selected(props.selected),
        );
        let mut panel_props = TabPanelProps::new();
        if let Some(active) = props.tabs.get(props.selected) {
            panel_props = panel_props.content(active.content.clone());
        }
        let panel = TabPanel.render(&panel_props);
        Element::box_().class("pk-tabs").child(list).child(panel)
    }
}

/// Registers the `pk-tabs` class family: container, list, tab, panel.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok, transparent};

    // Container: a vertical stack of the tab list above its panel.
    sheet.insert(
        Class::new("pk-tabs")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    // Tab list: a row of tabs sharing a bottom separator.
    sheet.insert(
        Class::new("pk-tabs__list")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    // Tab: a transparent pill that fills on hover and selection.
    sheet.insert(
        Class::new("pk-tabs__tab")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::BackgroundColor, transparent())
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with(StyleProp::FontWeight, tok("font.weight.medium"))
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with_state(InteractionState::Hover, StyleProp::Color, tok("color.label")),
    );

    // Selected tab: accent label and a tinted fill.
    sheet.insert(
        Class::new("pk-tabs__tab--selected")
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );

    // Panel: the active tab's body.
    sheet.insert(
        Class::new("pk-tabs__panel")
            .with_padding_y(tok("space.sm"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tabs_render_list_then_panel() {
        let el = Tabs.render(
            &TabsProps::new()
                .tab(TabItem::new(Element::text("One"), Element::text("First")))
                .tab(TabItem::new(Element::text("Two"), Element::text("Second")))
                .selected(1),
        );
        assert_eq!(el.class_names(), ["pk-tabs"]);
        let children = el.child_elements();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].class_names(), ["pk-tabs__list"]);
        assert_eq!(children[1].class_names(), ["pk-tabs__panel"]);
    }

    #[test]
    fn selected_tab_carries_marker() {
        let el = TabList.render(
            &TabListProps::new()
                .labels([Element::text("A"), Element::text("B"), Element::text("C")])
                .selected(2),
        );
        let tabs = el.child_elements();
        assert_eq!(tabs.len(), 3);
        assert_eq!(tabs[0].class_names(), ["pk-tabs__tab"]);
        assert_eq!(tabs[2].class_names(), ["pk-tabs__tab", "pk-tabs__tab--selected"]);
    }

    #[test]
    fn panel_shows_selected_content() {
        let el = Tabs.render(
            &TabsProps::new()
                .tab(TabItem::new(Element::text("One"), Element::text("First")))
                .tab(TabItem::new(Element::text("Two"), Element::text("Second")))
                .selected(1),
        );
        let panel = &el.child_elements()[1];
        assert_eq!(panel.child_elements().len(), 1);
        assert_eq!(panel.child_elements()[0].text_content(), Some("Second"));
    }

    #[test]
    fn out_of_range_selection_yields_empty_panel() {
        let el = Tabs.render(&TabsProps::new().selected(5));
        let panel = &el.child_elements()[1];
        assert!(panel.child_elements().is_empty());
    }

    #[test]
    fn role_is_tab() {
        assert_eq!(TabList::role(), Role::Tab);
    }
}
