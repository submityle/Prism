//! [`TimePicker`] — three hour/minute/second wheel columns.
//!
//! A time picker renders a `pk-time-picker` row of `pk-time-picker__column`s,
//! each a short vertical *window* of `pk-time-picker__cell`s centred on the
//! chosen value and wrapping like a physical drum. Only [`WHEEL_RADIUS`] values
//! are shown on each side of the centre, so a column never overflows its cell
//! regardless of range (hours wrap `0..24`, minutes/seconds `0..60`). The
//! centre cell carries `pk-time-picker__cell--selected`; its neighbours fade
//! with `--near` / `--far`. The seconds column is optional.

use alloc::format;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`TimePicker`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TimePickerProps {
    /// The selected hour (`0..24`), if any.
    pub hour: Option<u8>,
    /// The selected minute (`0..60`), if any.
    pub minute: Option<u8>,
    /// The selected second (`0..60`), if any.
    pub second: Option<u8>,
    /// Whether to render the trailing seconds column.
    pub show_seconds: bool,
}

impl TimePickerProps {
    /// Creates default picker props (nothing selected, no seconds column).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the selected hour.
    #[must_use]
    pub fn hour(mut self, hour: u8) -> Self {
        self.hour = Some(hour);
        self
    }

    /// Sets the selected minute.
    #[must_use]
    pub fn minute(mut self, minute: u8) -> Self {
        self.minute = Some(minute);
        self
    }

    /// Sets the selected second (also enables the seconds column).
    #[must_use]
    pub fn second(mut self, second: u8) -> Self {
        self.second = Some(second);
        self.show_seconds = true;
        self
    }

    /// Enables or disables the seconds column.
    #[must_use]
    pub fn show_seconds(mut self, show_seconds: bool) -> Self {
        self.show_seconds = show_seconds;
        self
    }
}

/// How many values are shown on each side of the selection in a wheel column.
///
/// A column therefore renders `2 * WHEEL_RADIUS + 1` cells — a compact window
/// that fits the control without relying on overflow clipping.
pub const WHEEL_RADIUS: i32 = 2;

/// Builds one wheel column: a short window of `count` values centred on
/// `selected`, wrapping modulo `count` like a physical drum. The centre cell is
/// marked selected; cells one step away fade (`--near`) and further ones fade
/// more (`--far`), mirroring the curved iOS picker. Values are zero-padded to
/// two digits (e.g. `09`, `05`).
fn column(count: u8, selected: Option<u8>) -> Element {
    let count_i = i32::from(count);
    let center = i32::from(selected.unwrap_or(0));
    let mut col = Element::box_().class("pk-time-picker__column");
    for offset in -WHEEL_RADIUS..=WHEEL_RADIUS {
        let value = (center + offset).rem_euclid(count_i);
        let modifier = match offset.abs() {
            0 => "pk-time-picker__cell--selected",
            1 => "pk-time-picker__cell--near",
            _ => "pk-time-picker__cell--far",
        };
        let cell = Element::box_()
            .class("pk-time-picker__cell")
            .class(modifier)
            .child(Element::text(format!("{value:02}")));
        col = col.child(cell);
    }
    col
}

/// The time-picker control. Zero-sized; config lives in [`TimePickerProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct TimePicker;

impl TimePicker {
    /// The accessibility role a column group exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for TimePicker {
    type Props = TimePickerProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_()
            .class("pk-time-picker")
            .child(column(24, props.hour))
            .child(column(60, props.minute));
        if props.show_seconds {
            el = el.child(column(60, props.second));
        }
        el
    }
}

/// Registers the `pk-time-picker` class family: the row, a column, a cell and
/// the selected modifier.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Row: the columns side by side.
    sheet.insert(
        Class::new("pk-time-picker")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    // Column: a vertical stack of values.
    sheet.insert(
        Class::new("pk-time-picker__column")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.xxs")),
    );

    // Cell: a centered, tappable value row.
    sheet.insert(
        Class::new("pk-time-picker__cell")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Width, StyleValue::px(44.0))
            .with(StyleProp::MinWidth, StyleValue::px(44.0))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xxs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Selected (centre of the drum): a neutral selection band with a full
    // label-coloured, semibold numeral, matching the iOS wheel picker where the
    // centre row sits behind a tinted lozenge rather than an accent fill.
    sheet.insert(
        Class::new("pk-time-picker__cell--selected")
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontWeight, StyleValue::number(600.0)),
    );

    // Near neighbours (one step from centre): dimmed to the secondary label so
    // the drum reads as receding away from the selection.
    sheet.insert(
        Class::new("pk-time-picker__cell--near").with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Far neighbours (edges of the window): dimmed further to the tertiary
    // label for the strongest sense of curvature.
    sheet.insert(
        Class::new("pk-time-picker__cell--far").with(StyleProp::Color, tok("color.label.tertiary")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each wheel column renders a fixed window of `2 * WHEEL_RADIUS + 1` cells.
    const WINDOW: usize = (2 * WHEEL_RADIUS + 1) as usize;

    #[test]
    fn two_columns_by_default() {
        let el = TimePicker.render(&TimePickerProps::new());
        assert_eq!(el.class_names(), ["pk-time-picker"]);
        let cols = el.child_elements();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].child_elements().len(), WINDOW); // hours
        assert_eq!(cols[1].child_elements().len(), WINDOW); // minutes
    }

    #[test]
    fn seconds_column_is_opt_in() {
        let el = TimePicker.render(&TimePickerProps::new().show_seconds(true));
        let cols = el.child_elements();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[2].child_elements().len(), WINDOW);
    }

    #[test]
    fn selecting_a_second_enables_the_column() {
        let el = TimePicker.render(&TimePickerProps::new().second(30));
        assert_eq!(el.child_elements().len(), 3);
    }

    #[test]
    fn selected_cell_gets_the_modifier() {
        let el = TimePicker.render(&TimePickerProps::new().hour(9));
        let hours = &el.child_elements()[0];
        let marked: Vec<_> = hours
            .child_elements()
            .iter()
            .filter(|c| c.class_names().iter().any(|n| n == "pk-time-picker__cell--selected"))
            .collect();
        assert_eq!(marked.len(), 1);
        // The centre cell is the selection, zero-padded to two digits.
        assert_eq!(marked[0].child_elements()[0].text_content(), Some("09"));
    }

    #[test]
    fn window_wraps_around_the_bottom() {
        // Selecting hour 0 wraps the leading neighbours to 22, 23.
        let el = TimePicker.render(&TimePickerProps::new().hour(0));
        let hours = &el.child_elements()[0];
        let values: Vec<_> = hours
            .child_elements()
            .iter()
            .map(|c| c.child_elements()[0].text_content().unwrap())
            .collect();
        assert_eq!(values, ["22", "23", "00", "01", "02"]);
    }

    #[test]
    fn role_is_group() {
        assert_eq!(TimePicker::role(), Role::Group);
    }
}
