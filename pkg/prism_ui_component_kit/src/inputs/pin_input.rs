//! [`PinInput`] — a segmented one-time-code / PIN entry.
//!
//! A pin input renders a `pk-pin-input` row of `length` `__cell`s. Each cell
//! that already holds a character is marked `--filled`; the next empty cell is
//! marked `--active`. When `masked` is set, filled cells show a bullet instead
//! of the character. The logic is hand-rolled and `no_std`, and a `length` of
//! `0` renders an empty, panic-free row. All color comes from theme tokens via
//! [`crate::preset`]. [`OtpInput`] is a drop-in alias.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// The glyph shown in a filled cell while `masked` is set.
const MASK_GLYPH: char = '•';

/// Props for [`PinInput`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PinInputProps {
    /// How many cells to render.
    pub length: usize,
    /// The entered characters; extra characters past `length` are ignored.
    pub value: String,
    /// Whether filled cells mask their character.
    pub masked: bool,
    /// Whether the control is non-interactive.
    pub disabled: bool,
}

impl PinInputProps {
    /// Creates empty pin-input props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of cells.
    #[must_use]
    pub fn length(mut self, length: usize) -> Self {
        self.length = length;
        self
    }

    /// Sets the entered value.
    #[must_use]
    pub fn value(mut self, value: impl Into<String>) -> Self {
        self.value = value.into();
        self
    }

    /// Masks filled cells.
    #[must_use]
    pub fn masked(mut self, masked: bool) -> Self {
        self.masked = masked;
        self
    }

    /// Marks the control disabled.
    #[must_use]
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// The pin-input control. Zero-sized; config lives in [`PinInputProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct PinInput;

impl PinInput {
    /// The accessibility role the row exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }

    /// The accessibility role each individual cell exposes.
    #[must_use]
    pub const fn cell_role() -> Role {
        Role::Textbox
    }
}

impl Component for PinInput {
    type Props = PinInputProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-pin-input");
        if props.disabled {
            el = el.class("pk-pin-input--disabled");
        }

        let chars: Vec<char> = props.value.chars().collect();
        let filled = chars.len();

        #[expect(
            clippy::needless_range_loop,
            reason = "index drives cell state (filled/active) beyond chars.len()"
        )]
        for i in 0..props.length {
            let mut cell = Element::box_().class("pk-pin-input__cell");
            if i < filled {
                cell = cell.class("pk-pin-input__cell--filled");
                let shown = if props.masked { MASK_GLYPH } else { chars[i] };
                cell = cell.child(Element::text(shown.to_string()).class("pk-pin-input__char"));
            } else if i == filled {
                // The first empty cell is where the next character lands.
                cell = cell.class("pk-pin-input__cell--active");
            }
            el = el.child(cell);
        }

        el
    }
}

/// A one-time-passcode entry. An alias of [`PinInput`] with the same props.
pub type OtpInput = PinInput;

/// Registers the `pk-pin-input` class family: row, disabled modifier, cells,
/// their filled/active modifiers and the character glyph.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    sheet.insert(
        Class::new("pk-pin-input")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.sm")),
    );
    sheet.insert(
        Class::new("pk-pin-input--disabled").with(StyleProp::Opacity, StyleValue::number(0.4)),
    );

    sheet.insert(
        Class::new("pk-pin-input__cell")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Width, StyleValue::px(40.0))
            .with(StyleProp::Height, StyleValue::px(48.0))
            .with(StyleProp::MinWidth, StyleValue::px(40.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.md"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_glass(0.0, tok("color.surface.secondary"), Some(tok("glass.highlight")))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.title3")),
    );
    sheet.insert(
        Class::new("pk-pin-input__cell--filled")
            .with(StyleProp::BorderColor, tok("color.label.tertiary")),
    );
    sheet.insert(
        Class::new("pk-pin-input__cell--active")
            .with(StyleProp::BorderColor, tok("color.tint")),
    );

    sheet.insert(
        Class::new("pk-pin-input__char")
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.title3"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: PinInputProps) -> Element {
        PinInput.render(&props)
    }

    #[test]
    fn renders_one_cell_per_length() {
        let el = render(PinInputProps::new().length(4));
        assert_eq!(el.class_names(), ["pk-pin-input"]);
        assert_eq!(el.child_elements().len(), 4);
    }

    #[test]
    fn filled_active_and_empty_cells_are_marked() {
        let el = render(PinInputProps::new().length(4).value("12"));
        let cells = el.child_elements();
        assert!(cells[0].class_names().iter().any(|c| c == "pk-pin-input__cell--filled"));
        assert!(cells[1].class_names().iter().any(|c| c == "pk-pin-input__cell--filled"));
        assert!(cells[2].class_names().iter().any(|c| c == "pk-pin-input__cell--active"));
        assert!(!cells[3].class_names().iter().any(|c| c == "pk-pin-input__cell--active"));
        assert_eq!(cells[0].child_elements()[0].text_content(), Some("1"));
    }

    #[test]
    fn masked_cells_show_bullets() {
        let el = render(PinInputProps::new().length(3).value("9").masked(true));
        let cell = &el.child_elements()[0];
        assert_eq!(cell.child_elements()[0].text_content(), Some("•"));
    }

    #[test]
    fn full_value_leaves_no_active_cell() {
        let el = render(PinInputProps::new().length(2).value("ab"));
        let cells = el.child_elements();
        assert!(cells
            .iter()
            .all(|c| !c.class_names().iter().any(|n| n == "pk-pin-input__cell--active")));
    }

    #[test]
    fn zero_length_renders_no_cells_without_panicking() {
        let el = render(PinInputProps::new().length(0).value("overflow"));
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn disabled_adds_block_modifier() {
        let el = render(PinInputProps::new().length(1).disabled(true));
        assert!(el.class_names().iter().any(|c| c == "pk-pin-input--disabled"));
    }

    #[test]
    fn alias_renders_like_pin_input() {
        let via_alias = OtpInput::default().render(&PinInputProps::new().length(3).value("1"));
        let direct = PinInput.render(&PinInputProps::new().length(3).value("1"));
        assert_eq!(via_alias.child_elements().len(), direct.child_elements().len());
    }

    #[test]
    fn roles_are_group_and_textbox() {
        assert_eq!(PinInput::role(), Role::Group);
        assert_eq!(PinInput::cell_role(), Role::Textbox);
    }
}
