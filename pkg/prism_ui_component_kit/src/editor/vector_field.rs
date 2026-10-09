//! [`VectorField`] — a row of N numeric scalar fields (e.g. x/y/z/w).
//!
//! A vector field groups several labelled scalar inputs on one row so a
//! multi-component value (a position, a color, a quaternion) edits as a unit.
//! Each axis renders a `pk-vector-field__axis` cell carrying a `__label` and an
//! optional `__value` text slot. The control attaches only kit class names;
//! spacing and typography resolve from theme tokens via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`VectorField`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct VectorFieldProps {
    /// The per-axis labels, in order (e.g. `["x", "y", "z"]`).
    pub axes: Vec<String>,
    /// Optional per-axis values as text, aligned by index with [`Self::axes`].
    pub values: Vec<String>,
}

impl VectorFieldProps {
    /// Creates vector-field props from a list of axis labels.
    #[must_use]
    pub fn new<I, S>(axes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            axes: axes.into_iter().map(Into::into).collect(),
            ..Self::default()
        }
    }

    /// Creates a conventional `x/y/z` three-axis field.
    #[must_use]
    pub fn xyz() -> Self {
        Self::new(["x", "y", "z"])
    }

    /// Creates a conventional `x/y/z/w` four-axis field.
    #[must_use]
    pub fn xyzw() -> Self {
        Self::new(["x", "y", "z", "w"])
    }

    /// Replaces the axis labels.
    #[must_use]
    pub fn axes<I, S>(mut self, axes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.axes = axes.into_iter().map(Into::into).collect();
        self
    }

    /// Replaces the per-axis values (text), aligned by index with the axes.
    #[must_use]
    pub fn values<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.values = values.into_iter().map(Into::into).collect();
        self
    }
}

/// The vector-field control. Zero-sized; config lives in [`VectorFieldProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct VectorField;

impl VectorField {
    /// The accessibility role a vector field exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for VectorField {
    type Props = VectorFieldProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-vector-field");
        for (i, axis) in props.axes.iter().enumerate() {
            let mut cell = Element::box_()
                .class("pk-vector-field__axis")
                .child(Element::text(axis.clone()).class("pk-vector-field__label"));
            if let Some(value) = props.values.get(i) {
                cell = cell.child(Element::text(value.clone()).class("pk-vector-field__value"));
            }
            el = el.child(cell);
        }
        el
    }
}

/// Registers the `pk-vector-field` family: row, axis cell, label, value.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-vector-field")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs")),
    );

    sheet.insert(
        Class::new("pk-vector-field__axis")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::FlexBasis, StyleValue::px(0.0))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with_padding_x(tok("space.xs"))
            .with_padding_y(tok("space.xxs")),
    );

    sheet.insert(
        Class::new("pk-vector-field__label")
            .with(StyleProp::Color, tok("color.label.tertiary"))
            .with(StyleProp::FontSize, tok("font.size.caption1"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );

    sheet.insert(
        Class::new("pk-vector-field__value")
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::FlexGrow, StyleValue::number(1.0)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: VectorFieldProps) -> Element {
        VectorField.render(&props)
    }

    #[test]
    fn empty_field_has_no_axes() {
        let el = render(VectorFieldProps::default());
        assert_eq!(el.class_names(), ["pk-vector-field"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn renders_one_axis_cell_per_label() {
        let el = render(VectorFieldProps::xyz());
        let axes = el.child_elements();
        assert_eq!(axes.len(), 3);
        for axis in axes {
            assert_eq!(axis.class_names(), ["pk-vector-field__axis"]);
            assert_eq!(
                axis.child_elements()[0].class_names(),
                ["pk-vector-field__label"]
            );
        }
    }

    #[test]
    fn values_align_by_index() {
        let el = render(VectorFieldProps::xyz().values(["1.0", "2.0"]));
        let axes = el.child_elements();
        // First two axes carry a value child; the third does not.
        assert_eq!(axes[0].child_elements().len(), 2);
        assert_eq!(axes[1].child_elements().len(), 2);
        assert_eq!(axes[2].child_elements().len(), 1);
        assert_eq!(axes[0].child_elements()[1].text_content(), Some("1.0"));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(VectorField::role(), Role::Group);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in [
            "pk-vector-field",
            "pk-vector-field__axis",
            "pk-vector-field__label",
            "pk-vector-field__value",
        ] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
