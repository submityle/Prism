//! [`Cascader`] — a multi-column cascading selector.
//!
//! A cascader renders a `pk-cascader` surface whose dropdown holds one
//! `__column` per level, each listing `__option`s with the active option on
//! that level marked `--active`. The dropdown is composed from the shared
//! [`Popover`] overlay base rather than a bespoke surface, per the kit's
//! single-overlay invariant. All color comes from theme tokens via
//! [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::feedback::{Popover, PopoverProps};
use crate::preset::StyleSheet;

/// Props for [`Cascader`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CascaderProps {
    /// One list of options per cascading level.
    pub columns: Vec<Vec<String>>,
    /// The active option index per level (`path[level]`).
    pub path: Vec<usize>,
    /// Whether the cascading dropdown is open.
    pub open: bool,
}

impl CascaderProps {
    /// Creates empty, closed cascader props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the columns with `columns`.
    #[must_use]
    pub fn columns<I, J, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.columns = columns
            .into_iter()
            .map(|col| col.into_iter().map(Into::into).collect())
            .collect();
        self
    }

    /// Replaces the active path with `path`.
    #[must_use]
    pub fn path<I: IntoIterator<Item = usize>>(mut self, path: I) -> Self {
        self.path = path.into_iter().collect();
        self
    }

    /// Sets whether the dropdown is open.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }
}

/// The cascader control. Zero-sized; config lives in [`CascaderProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Cascader;

impl Cascader {
    /// The accessibility role a cascader exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Textbox
    }
}

impl Component for Cascader {
    type Props = CascaderProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut root = Element::box_().class("pk-cascader");
        if props.open {
            root = root.class("pk-cascader--open");
        }

        let mut columns: Vec<Element> = Vec::with_capacity(props.columns.len());
        for (ci, col) in props.columns.iter().enumerate() {
            let active = props.path.get(ci).copied();
            let mut column = Element::box_().class("pk-cascader__column");
            for (oi, option) in col.iter().enumerate() {
                let mut opt = Element::text(option.clone()).class("pk-cascader__option");
                if active == Some(oi) {
                    opt = opt.class("pk-cascader__option--active");
                }
                column = column.child(opt);
            }
            columns.push(column);
        }

        let popover = Popover.render(&PopoverProps::new().open(props.open).children(columns));
        root.child(popover)
    }
}

/// Registers the `pk-cascader` class family: base surface, open modifier,
/// columns, options and the active-option marker.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-cascader")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column)),
    );
    sheet.insert(
        Class::new("pk-cascader--open").with(StyleProp::Opacity, StyleValue::number(1.0)),
    );

    sheet.insert(
        Class::new("pk-cascader__column")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::MinWidth, StyleValue::px(140.0))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_padding_x(tok("space.xs"))
            .with_padding_y(tok("space.xs")),
    );

    sheet.insert(
        Class::new("pk-cascader__option")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );
    sheet.insert(
        Class::new("pk-cascader__option--active")
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: CascaderProps) -> Element {
        Cascader.render(&props)
    }

    fn columns_of(root: &Element) -> Vec<Element> {
        // The columns live inside the popover surface (root's only child).
        let popover = root.child_elements().last().expect("popover");
        popover
            .child_elements()
            .iter()
            .flat_map(Element::child_elements)
            .filter(|c| c.class_names().iter().any(|n| n == "pk-cascader__column"))
            .cloned()
            .collect()
    }

    #[test]
    fn renders_one_column_per_level() {
        let el = render(
            CascaderProps::new()
                .columns([["a", "b"].to_vec(), ["c"].to_vec()])
                .open(true),
        );
        assert_eq!(columns_of(&el).len(), 2);
    }

    #[test]
    fn active_path_marks_options() {
        let el = render(
            CascaderProps::new()
                .columns([["a", "b"].to_vec(), ["c", "d"].to_vec()])
                .path([1, 0])
                .open(true),
        );
        let cols = columns_of(&el);
        assert!(cols[0].child_elements()[1]
            .class_names()
            .iter()
            .any(|n| n == "pk-cascader__option--active"));
        assert!(cols[1].child_elements()[0]
            .class_names()
            .iter()
            .any(|n| n == "pk-cascader__option--active"));
    }

    #[test]
    fn open_adds_block_modifier() {
        let el = render(CascaderProps::new().open(true));
        assert!(el.class_names().iter().any(|c| c == "pk-cascader--open"));
    }

    #[test]
    fn role_is_textbox() {
        assert_eq!(Cascader::role(), Role::Textbox);
    }
}
