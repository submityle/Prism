//! [`AssetBrowser`] / [`AssetGrid`] — a thumbnail grid with a filter toolbar.
//!
//! An asset browser renders a `pk-asset-browser` column: a `__toolbar` strip
//! (search / filter controls) above a `__grid` of `__item` thumbnail cells.
//! Each item pairs an optional thumbnail element with a caption. Only kit class
//! names are attached; surface, grid metrics and type resolve from theme tokens
//! via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// A single asset entry: an optional thumbnail plus a caption label.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct AssetItem {
    /// The caption shown beneath the thumbnail.
    pub label: String,
    /// Optional thumbnail element (image, icon, preview).
    pub thumbnail: Option<Element>,
}

impl AssetItem {
    /// Creates an item with a caption and no thumbnail.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ..Self::default()
        }
    }

    /// Sets the thumbnail element.
    #[must_use]
    pub fn thumbnail(mut self, thumbnail: Element) -> Self {
        self.thumbnail = Some(thumbnail);
        self
    }
}

/// Props for [`AssetBrowser`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct AssetBrowserProps {
    /// Optional toolbar content (search field, filter controls).
    pub toolbar: Option<Element>,
    /// The assets shown in the grid.
    pub items: Vec<AssetItem>,
}

impl AssetBrowserProps {
    /// Creates empty asset-browser props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the toolbar content.
    #[must_use]
    pub fn toolbar(mut self, toolbar: Element) -> Self {
        self.toolbar = Some(toolbar);
        self
    }

    /// Appends an asset item.
    #[must_use]
    pub fn item(mut self, item: AssetItem) -> Self {
        self.items.push(item);
        self
    }

    /// Replaces the asset items.
    #[must_use]
    pub fn items<I: IntoIterator<Item = AssetItem>>(mut self, items: I) -> Self {
        self.items = items.into_iter().collect();
        self
    }
}

/// The asset-browser control. Zero-sized; config lives in [`AssetBrowserProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct AssetBrowser;

/// Alias matching the `AssetGrid` naming used by the design doc (section 12).
pub type AssetGrid = AssetBrowser;

impl AssetBrowser {
    /// The accessibility role an asset browser exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::List
    }
}

impl Component for AssetBrowser {
    type Props = AssetBrowserProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-asset-browser");

        let mut toolbar = Element::box_().class("pk-asset-browser__toolbar");
        if let Some(content) = props.toolbar.clone() {
            toolbar = toolbar.child(content);
        }
        el = el.child(toolbar);

        let mut grid = Element::box_().class("pk-asset-browser__grid");
        for item in &props.items {
            let mut cell = Element::box_().class("pk-asset-browser__item");
            if let Some(thumbnail) = item.thumbnail.clone() {
                cell = cell.child(thumbnail.class("pk-asset-browser__thumb"));
            }
            cell = cell.child(Element::text(item.label.clone()).class("pk-asset-browser__caption"));
            grid = grid.child(cell);
        }
        el.child(grid)
    }
}

/// Registers the `pk-asset-browser` family: container, toolbar, grid, item.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-asset-browser")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    sheet.insert(
        Class::new("pk-asset-browser__toolbar")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::BackgroundColor, tok("color.surface.secondary"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs")),
    );

    sheet.insert(
        Class::new("pk-asset-browser__grid")
            .with(StyleProp::Display, kw(Keyword::Grid))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    sheet.insert(
        Class::new("pk-asset-browser__item")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with_padding_x(tok("space.xs"))
            .with_padding_y(tok("space.xs")),
    );

    sheet.insert(
        Class::new("pk-asset-browser__thumb")
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::Height, StyleValue::px(72.0))
            .with(StyleProp::BackgroundColor, tok("color.fill.tertiary"))
            .with(StyleProp::BorderRadius, tok("radius.sm")),
    );

    sheet.insert(
        Class::new("pk-asset-browser__caption")
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.caption1")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: AssetBrowserProps) -> Element {
        AssetBrowser.render(&props)
    }

    #[test]
    fn renders_toolbar_then_grid() {
        let el = render(AssetBrowserProps::new());
        assert_eq!(el.class_names(), ["pk-asset-browser"]);
        let children = el.child_elements();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].class_names(), ["pk-asset-browser__toolbar"]);
        assert_eq!(children[1].class_names(), ["pk-asset-browser__grid"]);
    }

    #[test]
    fn grid_has_one_item_per_asset() {
        let el = render(
            AssetBrowserProps::new()
                .item(AssetItem::new("hero.png"))
                .item(AssetItem::new("ground.png")),
        );
        let grid = &el.child_elements()[1];
        assert_eq!(grid.child_elements().len(), 2);
        for item in grid.child_elements() {
            assert_eq!(item.class_names(), ["pk-asset-browser__item"]);
        }
    }

    #[test]
    fn item_with_thumbnail_has_thumb_then_caption() {
        let el = render(
            AssetBrowserProps::new()
                .item(AssetItem::new("hero.png").thumbnail(Element::box_().class("img"))),
        );
        let item = &el.child_elements()[1].child_elements()[0];
        let parts = item.child_elements();
        assert_eq!(parts.len(), 2);
        assert!(parts[0]
            .class_names()
            .iter()
            .any(|c| c == "pk-asset-browser__thumb"));
        assert_eq!(parts[1].class_names(), ["pk-asset-browser__caption"]);
        assert_eq!(parts[1].text_content(), Some("hero.png"));
    }

    #[test]
    fn role_is_list() {
        assert_eq!(AssetBrowser::role(), Role::List);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in [
            "pk-asset-browser",
            "pk-asset-browser__toolbar",
            "pk-asset-browser__grid",
            "pk-asset-browser__item",
            "pk-asset-browser__thumb",
            "pk-asset-browser__caption",
        ] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
