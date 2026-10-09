//! [`RadioGroup`] — a vertical set of mutually-exclusive [`Radio`]s.
//!
//! A radio group renders a `pk-radio-group` container whose children are
//! [`Radio`] rows, one per option, with exactly the `selected` index marked.
//! It carries the `Group` a11y role and attaches only kit class names; spacing
//! comes from theme tokens via [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::inputs::radio::{Radio, RadioProps};
use crate::preset::StyleSheet;

/// Props for [`RadioGroup`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RadioGroupProps {
    /// The option labels, rendered top-to-bottom.
    pub options: Vec<String>,
    /// The index of the selected option. Out-of-range selects nothing.
    pub selected: usize,
}

impl RadioGroupProps {
    /// Creates empty radio-group props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the options with `options`.
    #[must_use]
    pub fn options<I, S>(mut self, options: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.options = options.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the selected option index.
    #[must_use]
    pub fn selected(mut self, selected: usize) -> Self {
        self.selected = selected;
        self
    }
}

/// The radio-group control. Zero-sized; config lives in [`RadioGroupProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RadioGroup;

impl RadioGroup {
    /// The accessibility role a radio group exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for RadioGroup {
    type Props = RadioGroupProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-radio-group");
        for (index, option) in props.options.iter().enumerate() {
            let radio = Radio.render(
                &RadioProps::new()
                    .selected(index == props.selected)
                    .label(option.clone()),
            );
            el = el.child(radio);
        }
        el
    }
}

/// Registers the `pk-radio-group` container class.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-radio-group")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: RadioGroupProps) -> Element {
        RadioGroup.render(&props)
    }

    #[test]
    fn empty_group_has_no_children() {
        let el = render(RadioGroupProps::new());
        assert_eq!(el.class_names(), ["pk-radio-group"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn one_radio_per_option_with_selected_marked() {
        let el = render(
            RadioGroupProps::new()
                .options(["A", "B", "C"])
                .selected(1),
        );
        let kids = el.child_elements();
        assert_eq!(kids.len(), 3);
        // Selection is carried by the radio dot (`pk-radio__box--selected`),
        // not by a marker on the row.
        let dot_selected = |radio: &Element| {
            radio.child_elements()[0]
                .class_names()
                .iter()
                .any(|c| c == "pk-radio__box--selected")
        };
        assert!(!dot_selected(&kids[0]));
        assert!(dot_selected(&kids[1]));
        assert!(!dot_selected(&kids[2]));
    }

    #[test]
    fn out_of_range_selection_marks_nothing() {
        let el = render(RadioGroupProps::new().options(["A", "B"]).selected(9));
        for kid in el.child_elements() {
            assert!(kid.child_elements()[0]
                .class_names()
                .iter()
                .all(|c| c != "pk-radio__box--selected"));
        }
    }

    #[test]
    fn role_is_group() {
        assert_eq!(RadioGroup::role(), Role::Group);
    }
}
