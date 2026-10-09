//! [`DateRangePicker`] — a start/end range across two calendar grids.
//!
//! The range picker shows two side-by-side `pk-calendar-grid`s (reusing the
//! family [`super::date_picker`] owns) inside a shared
//! [`crate::feedback::Popover`]. Endpoint days carry
//! `pk-calendar-grid__cell--range-start` / `--range-end` and the days strictly
//! between them carry `--in-range`. Only these three range modifiers are
//! registered here; the base grid classes come from [`super::date_picker`].

use alloc::string::ToString;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::feedback::{Popover, PopoverProps};
use crate::preset::StyleSheet;

use super::calendar::{self, Date};

/// Props for [`DateRangePicker`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DateRangePickerProps {
    /// Year of the left (start) month.
    pub start_year: i32,
    /// 1-based month of the left grid.
    pub start_month: u8,
    /// Year of the right (end) month.
    pub end_year: i32,
    /// 1-based month of the right grid.
    pub end_month: u8,
    /// The selected range start, if any.
    pub range_start: Option<Date>,
    /// The selected range end, if any.
    pub range_end: Option<Date>,
    /// Whether the popover surface is shown.
    pub open: bool,
}

impl DateRangePickerProps {
    /// Creates default range props (empty months, closed).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the left (start) month and year.
    #[must_use]
    pub fn start_month_of(mut self, year: i32, month: u8) -> Self {
        self.start_year = year;
        self.start_month = month;
        self
    }

    /// Sets the right (end) month and year.
    #[must_use]
    pub fn end_month_of(mut self, year: i32, month: u8) -> Self {
        self.end_year = year;
        self.end_month = month;
        self
    }

    /// Sets the selected range endpoints.
    #[must_use]
    pub fn range(mut self, start: Date, end: Date) -> Self {
        self.range_start = Some(start);
        self.range_end = Some(end);
        self
    }

    /// Sets whether the surface is shown.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }
}

/// Returns `true` when `date` falls strictly between the range endpoints.
fn in_range(start: Option<Date>, end: Option<Date>, date: Date) -> bool {
    match (start, end) {
        (Some(s), Some(e)) => date > s && date < e,
        _ => false,
    }
}

/// Builds one range-aware `pk-calendar-grid` for `(year, month)`.
fn range_grid(year: i32, month: u8, props: &DateRangePickerProps) -> Element {
    let grid = calendar::month_grid(year, month);
    let mut root = Element::box_().class("pk-calendar-grid");
    for week in grid {
        let mut row = Element::box_().class("pk-calendar-grid__row");
        for cell in week {
            let mut day_cell = Element::box_().class("pk-calendar-grid__cell");
            match cell {
                Some(day) => {
                    day_cell = day_cell.child(Element::text(day.to_string()));
                    let date = Date::new(year, month, day);
                    if props.range_start == Some(date) {
                        day_cell = day_cell.class("pk-calendar-grid__cell--range-start");
                    } else if props.range_end == Some(date) {
                        day_cell = day_cell.class("pk-calendar-grid__cell--range-end");
                    } else if in_range(props.range_start, props.range_end, date) {
                        day_cell = day_cell.class("pk-calendar-grid__cell--in-range");
                    }
                }
                None => {
                    day_cell = day_cell.class("pk-calendar-grid__cell--empty");
                }
            }
            row = row.child(day_cell);
        }
        root = root.child(row);
    }
    root
}

/// The range-picker control. Zero-sized; config lives in
/// [`DateRangePickerProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DateRangePicker;

impl DateRangePicker {
    /// The accessibility role a range popover exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Dialog
    }
}

impl Component for DateRangePicker {
    type Props = DateRangePickerProps;

    fn render(&self, props: &Self::Props) -> Element {
        let start = range_grid(props.start_year, props.start_month, props);
        let end = range_grid(props.end_year, props.end_month, props);
        let months = Element::box_()
            .class("pk-date-range-picker__months")
            .child(start)
            .child(end);
        let root = Element::box_().class("pk-date-range-picker").child(months);
        let popover = PopoverProps::new().open(props.open).child(root);
        Popover.render(&popover)
    }
}

/// Registers `pk-date-range-picker`, its two-month layout, and the three range
/// modifiers layered on top of the shared `pk-calendar-grid__cell`.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Container: a column holding the dual-month layout.
    sheet.insert(
        Class::new("pk-date-range-picker")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    // Months: the two grids laid out side by side.
    sheet.insert(
        Class::new("pk-date-range-picker__months")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::Gap, tok("space.lg")),
    );

    // Range endpoints: solid accent fill with white numerals.
    sheet.insert(
        Class::new("pk-calendar-grid__cell--range-start")
            .with(StyleProp::BackgroundColor, tok("color.tint"))
            .with(StyleProp::Color, StyleValue::rgba8(255, 255, 255, 255)),
    );
    sheet.insert(
        Class::new("pk-calendar-grid__cell--range-end")
            .with(StyleProp::BackgroundColor, tok("color.tint"))
            .with(StyleProp::Color, StyleValue::rgba8(255, 255, 255, 255)),
    );

    // In-range: a soft accent wash between the endpoints.
    sheet.insert(
        Class::new("pk-calendar-grid__cell--in-range")
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::Color, tok("color.label")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn month_grids(el: &Element) -> Vec<Element> {
        let surface = el
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-popover__surface"))
            .expect("surface");
        let root = &surface.child_elements()[0];
        let months = &root.child_elements()[0];
        months.child_elements().to_vec()
    }

    #[test]
    fn renders_two_grids_inside_a_popover() {
        let el = DateRangePicker.render(
            &DateRangePickerProps::new()
                .start_month_of(2024, 1)
                .end_month_of(2024, 2)
                .open(true),
        );
        assert!(el.class_names().iter().any(|c| c == "pk-popover"));
        let grids = month_grids(&el);
        assert_eq!(grids.len(), 2);
        for grid in &grids {
            assert!(grid.class_names().iter().any(|n| n == "pk-calendar-grid"));
        }
    }

    #[test]
    fn endpoints_and_interior_are_classified() {
        let el = DateRangePicker.render(
            &DateRangePickerProps::new()
                .start_month_of(2024, 2)
                .end_month_of(2024, 2)
                .range(Date::new(2024, 2, 10), Date::new(2024, 2, 14)),
        );
        let grids = month_grids(&el);
        let cells: Vec<_> = grids[0]
            .child_elements()
            .iter()
            .flat_map(|row| row.child_elements().iter().cloned())
            .collect();
        let count = |marker: &str| {
            cells
                .iter()
                .filter(|c| c.class_names().iter().any(|n| n == marker))
                .count()
        };
        assert_eq!(count("pk-calendar-grid__cell--range-start"), 1);
        assert_eq!(count("pk-calendar-grid__cell--range-end"), 1);
        // Days 11, 12, 13 are strictly inside the range.
        assert_eq!(count("pk-calendar-grid__cell--in-range"), 3);
    }

    #[test]
    fn role_is_dialog() {
        assert_eq!(DateRangePicker::role(), Role::Dialog);
    }
}
