//! [`ColorPicker`] — a discrete swatch grid.
//!
//! A color picker renders a `pk-color-picker` grid of
//! `pk-color-picker__swatch` boxes, one per preset color. The currently
//! selected swatch also carries `pk-color-picker__swatch--selected`. The value
//! model is a [`StyleValue`] (a concrete color or a token reference), so a
//! swatch's fill is a *data* value from props rather than a style literal — the
//! same pattern [`crate::inputs`]'s slider uses for its data-derived width.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;
use prism_ui_style::StyleValue;

use crate::preset::StyleSheet;

/// Props for [`ColorPicker`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ColorPickerProps {
    /// The currently selected color, if any. Matched against [`Self::swatches`]
    /// by value to decide which swatch is highlighted.
    pub value: Option<StyleValue>,
    /// The preset palette, each entry a concrete color or a token reference.
    pub swatches: Vec<StyleValue>,
    /// Whether to offer a trailing "custom color" affordance.
    pub allow_custom: bool,
}

impl ColorPickerProps {
    /// Creates empty picker props (no swatches, nothing selected).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the selected color value.
    #[must_use]
    pub fn value(mut self, value: StyleValue) -> Self {
        self.value = Some(value);
        self
    }

    /// Appends a swatch to the palette.
    #[must_use]
    pub fn swatch(mut self, swatch: StyleValue) -> Self {
        self.swatches.push(swatch);
        self
    }

    /// Replaces the palette with `swatches`.
    #[must_use]
    pub fn swatches<I: IntoIterator<Item = StyleValue>>(mut self, swatches: I) -> Self {
        self.swatches = swatches.into_iter().collect();
        self
    }

    /// Enables or disables the trailing custom-color affordance.
    #[must_use]
    pub fn allow_custom(mut self, allow_custom: bool) -> Self {
        self.allow_custom = allow_custom;
        self
    }
}

/// The color-picker control. Zero-sized; config lives in [`ColorPickerProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ColorPicker;

impl ColorPicker {
    /// The accessibility role a swatch grid exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for ColorPicker {
    type Props = ColorPickerProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::StyleProp;

        let mut el = Element::box_().class("pk-color-picker");
        for swatch in &props.swatches {
            let mut cell = Element::box_()
                .class("pk-color-picker__swatch")
                .style(StyleProp::BackgroundColor, swatch.clone());
            if props.value.as_ref() == Some(swatch) {
                cell = cell.class("pk-color-picker__swatch--selected");
            }
            el = el.child(cell);
        }
        if props.allow_custom {
            el = el.child(Element::box_().class("pk-color-picker__custom"));
        }
        el
    }
}

/// Registers the `pk-color-picker` class family: the grid, a swatch box, the
/// selected modifier and the custom-color affordance.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Grid: a dense flex row of fixed swatches.
    sheet.insert(
        Class::new("pk-color-picker")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xs")),
    );

    // Swatch: a fixed square chip with a hairline border (fill set inline).
    sheet.insert(
        Class::new("pk-color-picker__swatch")
            .with(StyleProp::Width, StyleValue::px(24.0))
            .with(StyleProp::Height, StyleValue::px(24.0))
            .with(StyleProp::MinWidth, StyleValue::px(24.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator")),
    );

    // Selected: a thicker accent ring to lift the active chip.
    sheet.insert(
        Class::new("pk-color-picker__swatch--selected")
            .with(StyleProp::BorderWidth, StyleValue::px(2.0))
            .with(StyleProp::BorderColor, tok("color.tint")),
    );

    // Custom: a neutral chip standing in for a free-form color entry point.
    sheet.insert(
        Class::new("pk-color-picker__custom")
            .with(StyleProp::Width, StyleValue::px(24.0))
            .with(StyleProp::Height, StyleValue::px(24.0))
            .with(StyleProp::MinWidth, StyleValue::px(24.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{StyleProp, StyleValue};

    fn render(props: ColorPickerProps) -> Element {
        ColorPicker.render(&props)
    }

    fn palette() -> Vec<StyleValue> {
        [StyleValue::token("color.red"), StyleValue::token("color.blue")].into()
    }

    #[test]
    fn renders_one_swatch_per_entry() {
        let el = render(ColorPickerProps::new().swatches(palette()));
        assert_eq!(el.class_names(), ["pk-color-picker"]);
        assert_eq!(el.child_elements().len(), 2);
        for cell in el.child_elements() {
            assert!(cell
                .class_names()
                .iter()
                .any(|c| c == "pk-color-picker__swatch"));
        }
    }

    #[test]
    fn swatch_fill_is_the_props_value_inline() {
        let el = render(ColorPickerProps::new().swatches(palette()));
        let first = &el.child_elements()[0];
        let bg = first
            .inline_pairs()
            .iter()
            .find(|(p, _)| *p == StyleProp::BackgroundColor)
            .map(|(_, v)| v.clone());
        assert_eq!(bg, Some(StyleValue::token("color.red")));
    }

    #[test]
    fn selected_swatch_gets_the_modifier() {
        let el = render(
            ColorPickerProps::new()
                .swatches(palette())
                .value(StyleValue::token("color.blue")),
        );
        let selected: Vec<_> = el
            .child_elements()
            .iter()
            .filter(|c| {
                c.class_names()
                    .iter()
                    .any(|n| n == "pk-color-picker__swatch--selected")
            })
            .collect();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].inline_pairs()[0].1, StyleValue::token("color.blue"));
    }

    #[test]
    fn custom_affordance_is_opt_in() {
        let plain = render(ColorPickerProps::new().swatches(palette()));
        assert!(plain
            .child_elements()
            .iter()
            .all(|c| !c.class_names().iter().any(|n| n == "pk-color-picker__custom")));
        let custom = render(ColorPickerProps::new().swatches(palette()).allow_custom(true));
        assert!(custom
            .child_elements()
            .iter()
            .any(|c| c.class_names().iter().any(|n| n == "pk-color-picker__custom")));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(ColorPicker::role(), Role::Group);
    }
}
