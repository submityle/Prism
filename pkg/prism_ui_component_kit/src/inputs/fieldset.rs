//! [`Fieldset`] — a grouped set of fields under a legend.
//!
//! A fieldset renders a `pk-fieldset` column holding a `__legend` caption
//! followed by its child fields, marking itself `--disabled` when the whole
//! group is non-interactive. It exposes [`Role::Group`] so assistive tech
//! announces the fields as one unit. It is a pure layout aggregator: callers
//! pass fully-formed child [`Element`]s, and this control only positions them
//! and attaches the `pk-fieldset` class family.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`Fieldset`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct FieldsetProps {
    /// The group's caption, rendered as a legend.
    pub legend: String,
    /// The grouped field elements.
    pub children: Vec<Element>,
    /// Whether the whole group is non-interactive.
    pub disabled: bool,
}

impl FieldsetProps {
    /// Creates props for a captioned fieldset.
    #[must_use]
    pub fn new(legend: impl Into<String>) -> Self {
        Self {
            legend: legend.into(),
            ..Self::default()
        }
    }

    /// Replaces the grouped children.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children = children.into_iter().collect();
        self
    }

    /// Appends a single grouped child.
    #[must_use]
    pub fn child(mut self, child: Element) -> Self {
        self.children.push(child);
        self
    }

    /// Marks the group disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The fieldset control. Zero-sized; config lives in [`FieldsetProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Fieldset;

impl Fieldset {
    /// The accessibility role a fieldset exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for Fieldset {
    type Props = FieldsetProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-fieldset");
        if props.disabled {
            el = el.class("pk-fieldset--disabled");
        }
        el = el.child(Element::text(props.legend.clone()).class("pk-fieldset__legend"));
        el.children(props.children.iter().cloned())
    }
}

/// Registers the `pk-fieldset` class family: the vertical group, its disabled
/// modifier and the legend caption.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-fieldset")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.md"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    sheet.insert(
        Class::new("pk-fieldset--disabled").with(StyleProp::Opacity, StyleValue::number(0.4)),
    );

    sheet.insert(
        Class::new("pk-fieldset__legend")
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.headline"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legend_precedes_children() {
        let el = Fieldset.render(
            &FieldsetProps::new("Address")
                .child(Element::box_())
                .child(Element::box_()),
        );
        let kids = el.child_elements();
        assert_eq!(kids.len(), 3);
        assert_eq!(kids[0].class_names(), ["pk-fieldset__legend"]);
        assert_eq!(kids[0].text_content(), Some("Address"));
    }

    #[test]
    fn disabled_adds_modifier() {
        let el = Fieldset.render(&FieldsetProps::new("X").disabled(true));
        assert!(el.class_names().iter().any(|c| c == "pk-fieldset--disabled"));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(Fieldset::role(), Role::Group);
    }
}
