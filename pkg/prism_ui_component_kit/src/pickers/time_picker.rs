//! [`TimePicker`] — three scrollable hour/minute/second columns.
//!
//! A time picker renders a `pk-time-picker` row of `pk-time-picker__column`s,
//! each a vertical list of selectable `pk-time-picker__cell`s. The chosen value
//! in a column adds `pk-time-picker__cell--selected`. The seconds column is
//! optional. Hours run `0..24`, minutes and seconds `0..60`.

use alloc::string::ToString;

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

/// Builds one column of `count` selectable cells, highlighting `selected`.
fn column(count: u8, selected: Option<u8>) -> Element {
    let mut col = Element::box_().class("pk-time-picker__column");
    for value in 0..count {
        let mut cell = Element::box_()
            .class("pk-time-picker__cell")
            .child(Element::text(value.to_string()));
        if selected == Some(value) {
            cell = cell.class("pk-time-picker__cell--selected");
        }
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

    // Selected: solid accent fill with a white numeral.
    sheet.insert(
        Class::new("pk-time-picker__cell--selected")
            .with(StyleProp::BackgroundColor, tok("color.tint"))
            .with(StyleProp::Color, StyleValue::rgba8(255, 255, 255, 255)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_columns_by_default() {
        let el = TimePicker.render(&TimePickerProps::new());
        assert_eq!(el.class_names(), ["pk-time-picker"]);
        let cols = el.child_elements();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].child_elements().len(), 24); // hours
        assert_eq!(cols[1].child_elements().len(), 60); // minutes
    }

    #[test]
    fn seconds_column_is_opt_in() {
        let el = TimePicker.render(&TimePickerProps::new().show_seconds(true));
        let cols = el.child_elements();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[2].child_elements().len(), 60);
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
        assert_eq!(marked[0].child_elements()[0].text_content(), Some("9"));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(TimePicker::role(), Role::Group);
    }
}
