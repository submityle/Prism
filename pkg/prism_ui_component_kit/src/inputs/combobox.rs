//! [`Combobox`] — a typeable field with a filtered dropdown of options.
//!
//! A combobox renders a `pk-combobox` root whose `__field` trigger pairs a
//! free-text `__input` with a `__caret` affordance. The candidate layer is
//! composed from the shared [`Popover`] overlay base (per the kit's
//! single-overlay invariant) and holds a `__list` of `__option`s; the option
//! matching the current value is marked `--active`. All color comes from theme
//! tokens via [`crate::preset`]. [`Autocomplete`] is a drop-in alias.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::feedback::{Popover, PopoverProps};
use crate::preset::StyleSheet;

/// A single combobox candidate: a human `label` and its backing `value`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ComboboxOption {
    /// The text shown in the dropdown.
    pub label: String,
    /// The value this option selects.
    pub value: String,
}

impl ComboboxOption {
    /// Creates an option from a label and value.
    #[must_use]
    pub fn new(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
        }
    }
}

/// Props for [`Combobox`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ComboboxProps {
    /// The current input/selected value. `None` renders an empty input.
    pub value: Option<String>,
    /// The available candidates.
    pub options: Vec<ComboboxOption>,
    /// Whether the candidate dropdown is open.
    pub open: bool,
    /// Whether the control is non-interactive.
    pub disabled: bool,
}

impl ComboboxProps {
    /// Creates empty, closed combobox props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the current value.
    #[must_use]
    pub fn value(mut self, value: impl Into<String>) -> Self {
        self.value = Some(value.into());
        self
    }

    /// Replaces the candidate options.
    #[must_use]
    pub fn options<I: IntoIterator<Item = ComboboxOption>>(mut self, options: I) -> Self {
        self.options = options.into_iter().collect();
        self
    }

    /// Sets whether the dropdown is open.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }

    /// Marks the control disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The combobox control. Zero-sized; config lives in [`ComboboxProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Combobox;

impl Combobox {
    /// The accessibility role a combobox exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Textbox
    }
}

impl Component for Combobox {
    type Props = ComboboxProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-combobox");
        if props.disabled {
            el = el.class("pk-combobox--disabled");
        }

        let input_text = props.value.clone().unwrap_or_default();
        let field = Element::box_()
            .class("pk-combobox__field")
            .child(Element::text(input_text).class("pk-combobox__input"))
            .child(Element::box_().class("pk-combobox__caret"));

        let mut list = Element::box_().class("pk-combobox__list");
        for option in &props.options {
            let mut opt = Element::text(option.label.clone()).class("pk-combobox__option");
            if props.value.as_deref() == Some(option.value.as_str()) {
                opt = opt.class("pk-combobox__option--active");
            }
            list = list.child(opt);
        }
        let popover = Popover.render(&PopoverProps::new().open(props.open).child(list));

        el.child(field).child(popover)
    }
}

/// A searchable autocomplete. An alias of [`Combobox`] with the same props.
pub type Autocomplete = Combobox;

/// Registers the `pk-combobox` class family: root, disabled modifier, field
/// trigger, input, caret, candidate list and its options.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-combobox")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column)),
    );
    sheet.insert(
        Class::new("pk-combobox--disabled").with(StyleProp::Opacity, StyleValue::number(0.4)),
    );

    sheet.insert(
        Class::new("pk-combobox__field")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with(StyleProp::Gap, tok("space.sm"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::Height, tok("size.control.height"))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_glass(0.0, tok("color.surface.secondary"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Focus, StyleProp::BorderColor, tok("color.tint")),
    );

    sheet.insert(
        Class::new("pk-combobox__input")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body")),
    );

    sheet.insert(
        Class::new("pk-combobox__caret")
            .with(StyleProp::Width, StyleValue::px(10.0))
            .with(StyleProp::Height, StyleValue::px(6.0))
            .with(StyleProp::MinWidth, StyleValue::px(10.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BackgroundColor, tok("color.label.tertiary"))
            .with(StyleProp::BorderRadius, tok("radius.xs")),
    );

    sheet.insert(
        Class::new("pk-combobox__list")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with(StyleProp::MinWidth, StyleValue::px(160.0)),
    );
    sheet.insert(
        Class::new("pk-combobox__option")
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );
    sheet.insert(
        Class::new("pk-combobox__option--active")
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: ComboboxProps) -> Element {
        Combobox.render(&props)
    }

    fn opts() -> Vec<ComboboxOption> {
        [("Apple", "a"), ("Banana", "b")]
            .into_iter()
            .map(|(l, v)| ComboboxOption::new(l, v))
            .collect()
    }

    #[test]
    fn field_holds_input_and_caret() {
        let el = render(ComboboxProps::new().value("Ap"));
        assert_eq!(el.class_names(), ["pk-combobox"]);
        let field = &el.child_elements()[0];
        assert_eq!(field.class_names(), ["pk-combobox__field"]);
        let kids = field.child_elements();
        assert_eq!(kids[0].class_names(), ["pk-combobox__input"]);
        assert_eq!(kids[0].text_content(), Some("Ap"));
        assert_eq!(kids[1].class_names(), ["pk-combobox__caret"]);
    }

    #[test]
    fn active_option_matches_value() {
        let el = render(ComboboxProps::new().value("b").options(opts()).open(true));
        let popover = el.child_elements().last().expect("popover");
        let list = popover
            .child_elements()
            .iter()
            .flat_map(Element::child_elements)
            .find(|c| c.class_names().iter().any(|n| n == "pk-combobox__list"))
            .expect("list");
        let options = list.child_elements();
        assert_eq!(options.len(), 2);
        assert!(!options[0].class_names().iter().any(|n| n == "pk-combobox__option--active"));
        assert!(options[1].class_names().iter().any(|n| n == "pk-combobox__option--active"));
    }

    #[test]
    fn none_value_renders_empty_input() {
        let el = render(ComboboxProps::new());
        let field = &el.child_elements()[0];
        assert_eq!(field.child_elements()[0].text_content(), Some(""));
    }

    #[test]
    fn disabled_adds_block_modifier() {
        let el = render(ComboboxProps::new().disabled(true));
        assert!(el.class_names().iter().any(|c| c == "pk-combobox--disabled"));
    }

    #[test]
    fn alias_renders_like_combobox() {
        let via_alias = Autocomplete::default().render(&ComboboxProps::new().value("x"));
        let direct = Combobox.render(&ComboboxProps::new().value("x"));
        assert_eq!(via_alias.class_names(), direct.class_names());
    }

    #[test]
    fn role_is_textbox() {
        assert_eq!(Combobox::role(), Role::Textbox);
    }
}
