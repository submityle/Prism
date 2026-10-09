//! [`TransformField`] — translation / rotation / scale as three stacked
//! [`VectorField`]s.
//!
//! A transform field is the canonical "object transform" editor: three grouped
//! [`VectorField`]s (translation, rotation, scale), each introduced by a small
//! group label. It composes [`VectorField`] rather than re-implementing scalar
//! layout, and attaches only the `pk-transform-field` class family.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

use super::vector_field::{VectorField, VectorFieldProps};

/// Props for [`TransformField`].
#[derive(Clone, Debug, PartialEq)]
pub struct TransformFieldProps {
    /// The translation (position) component field.
    pub translation: VectorFieldProps,
    /// The rotation (euler/quaternion) component field.
    pub rotation: VectorFieldProps,
    /// The scale component field.
    pub scale: VectorFieldProps,
}

impl Default for TransformFieldProps {
    fn default() -> Self {
        Self {
            translation: VectorFieldProps::xyz(),
            rotation: VectorFieldProps::xyz(),
            scale: VectorFieldProps::xyz(),
        }
    }
}

impl TransformFieldProps {
    /// Creates default `x/y/z` translation/rotation/scale props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the translation field.
    #[must_use]
    pub fn translation(mut self, field: VectorFieldProps) -> Self {
        self.translation = field;
        self
    }

    /// Sets the rotation field.
    #[must_use]
    pub fn rotation(mut self, field: VectorFieldProps) -> Self {
        self.rotation = field;
        self
    }

    /// Sets the scale field.
    #[must_use]
    pub fn scale(mut self, field: VectorFieldProps) -> Self {
        self.scale = field;
        self
    }
}

/// The transform-field control. Zero-sized; config lives in
/// [`TransformFieldProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TransformField;

impl TransformField {
    /// The accessibility role a transform field exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

/// Builds one labelled group wrapping a rendered [`VectorField`].
fn group(label: &str, field: &VectorFieldProps) -> Element {
    let label_el = Element::text(String::from(label)).class("pk-transform-field__group-label");
    let field_el = VectorField.render(field);
    Element::box_()
        .class("pk-transform-field__group")
        .child(label_el)
        .child(field_el)
}

impl Component for TransformField {
    type Props = TransformFieldProps;

    fn render(&self, props: &Self::Props) -> Element {
        Element::box_()
            .class("pk-transform-field")
            .child(group("Translation", &props.translation))
            .child(group("Rotation", &props.rotation))
            .child(group("Scale", &props.scale))
    }
}

/// Registers the `pk-transform-field` family: container, group, group label.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-transform-field")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.md")),
    );

    sheet.insert(
        Class::new("pk-transform-field__group")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs")),
    );

    sheet.insert(
        Class::new("pk-transform-field__group-label")
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.subheadline"))
            .with(StyleProp::FontWeight, tok("font.weight.medium")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn renders_three_labelled_groups() {
        let el = TransformField.render(&TransformFieldProps::new());
        assert_eq!(el.class_names(), ["pk-transform-field"]);
        let groups = el.child_elements();
        assert_eq!(groups.len(), 3);
        for g in groups {
            assert_eq!(g.class_names(), ["pk-transform-field__group"]);
            assert_eq!(
                g.child_elements()[0].class_names(),
                ["pk-transform-field__group-label"]
            );
            assert!(g.child_elements()[1]
                .class_names()
                .iter()
                .any(|c| c == "pk-vector-field"));
        }
    }

    #[test]
    fn group_labels_are_in_order() {
        let el = TransformField.render(&TransformFieldProps::new());
        let labels: Vec<_> = el
            .child_elements()
            .iter()
            .map(|g| g.child_elements()[0].text_content().unwrap())
            .collect();
        assert_eq!(labels, ["Translation", "Rotation", "Scale"]);
    }

    #[test]
    fn custom_axes_propagate_to_rendered_field() {
        let props = TransformFieldProps::new().rotation(VectorFieldProps::xyzw());
        let el = TransformField.render(&props);
        let rotation_field = &el.child_elements()[1].child_elements()[1];
        assert_eq!(rotation_field.child_elements().len(), 4);
    }

    #[test]
    fn role_is_group() {
        assert_eq!(TransformField::role(), Role::Group);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in [
            "pk-transform-field",
            "pk-transform-field__group",
            "pk-transform-field__group-label",
        ] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
