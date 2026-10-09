//! [`DatePicker`] — a single-day calendar popover.
//!
//! A date picker builds a 7×6 `pk-calendar-grid` from [`super::calendar`] and
//! drops it into a shared [`crate::feedback::Popover`] surface (invariant K7:
//! overlays compose the one popover base rather than re-rolling a surface). Day
//! cells are `pk-calendar-grid__cell`; the selected day adds `--selected` and
//! today adds `--today`. This module owns the whole `pk-calendar-grid` family,
//! which [`super::date_range_picker`] reuses.

use alloc::string::ToString;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::feedback::{Popover, PopoverProps};
use crate::preset::StyleSheet;

use super::calendar::{self, Date};

/// Props for [`DatePicker`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DatePickerProps {
    /// The year of the displayed month.
    pub year: i32,
    /// The 1-based month on display (`1 = January`). Values outside `1..=12`
    /// render an empty grid.
    pub month: u8,
    /// The currently selected day, if any.
    pub selected: Option<Date>,
    /// The day to flag as "today", if any.
    pub today: Option<Date>,
    /// Whether the popover surface is shown.
    pub open: bool,
}

impl DatePickerProps {
    /// Creates default picker props (empty month, closed).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the displayed year and month.
    #[must_use]
    pub fn month_of(mut self, year: i32, month: u8) -> Self {
        self.year = year;
        self.month = month;
        self
    }

    /// Sets the selected day.
    #[must_use]
    pub fn selected(mut self, date: Date) -> Self {
        self.selected = Some(date);
        self
    }

    /// Sets the day flagged as today.
    #[must_use]
    pub fn today(mut self, date: Date) -> Self {
        self.today = Some(date);
        self
    }

    /// Sets whether the surface is shown.
    #[must_use]
    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }
}

/// Builds the `pk-calendar-grid` element for `(year, month)`, flagging the
/// `selected` and `today` days. Shared shape used by both date pickers.
pub(super) fn calendar_grid(
    year: i32,
    month: u8,
    selected: Option<Date>,
    today: Option<Date>,
) -> Element {
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
                    if selected == Some(date) {
                        day_cell = day_cell.class("pk-calendar-grid__cell--selected");
                    }
                    if today == Some(date) {
                        day_cell = day_cell.class("pk-calendar-grid__cell--today");
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

/// The date-picker control. Zero-sized; config lives in [`DatePickerProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DatePicker;

impl DatePicker {
    /// The accessibility role a calendar popover exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Dialog
    }
}

impl Component for DatePicker {
    type Props = DatePickerProps;

    fn render(&self, props: &Self::Props) -> Element {
        let grid = calendar_grid(props.year, props.month, props.selected, props.today);
        let popover = PopoverProps::new().open(props.open).child(grid);
        Popover.render(&popover)
    }
}

/// Registers the `pk-calendar-grid` class family: the grid, its week rows, day
/// cells, the empty filler, and the selected / today modifiers.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Grid: a vertical stack of week rows.
    sheet.insert(
        Class::new("pk-calendar-grid")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs")),
    );

    // Row: seven evenly spaced day cells.
    sheet.insert(
        Class::new("pk-calendar-grid__row")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::Gap, tok("space.xxs")),
    );

    // Cell: a fixed, centered square of footnote-sized text.
    sheet.insert(
        Class::new("pk-calendar-grid__cell")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Width, StyleValue::px(32.0))
            .with(StyleProp::Height, StyleValue::px(32.0))
            .with(StyleProp::MinWidth, StyleValue::px(32.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Empty filler: a blank, non-interactive placeholder.
    sheet.insert(
        Class::new("pk-calendar-grid__cell--empty").with(StyleProp::Opacity, StyleValue::number(0.0)),
    );

    // Selected: solid accent fill with a white numeral.
    sheet.insert(
        Class::new("pk-calendar-grid__cell--selected")
            .with(StyleProp::BackgroundColor, tok("color.tint"))
            .with(StyleProp::Color, StyleValue::rgba8(255, 255, 255, 255)),
    );

    // Today: an accent ring around the current day.
    sheet.insert(
        Class::new("pk-calendar-grid__cell--today")
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.tint")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface_grid(el: &Element) -> Element {
        // Popover root -> surface -> grid.
        let surface = el
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-popover__surface"))
            .expect("surface");
        surface.child_elements()[0].clone()
    }

    #[test]
    fn renders_grid_inside_a_popover() {
        let el = DatePicker.render(&DatePickerProps::new().month_of(2024, 2).open(true));
        assert!(el.class_names().iter().any(|c| c == "pk-popover"));
        assert!(el.class_names().iter().any(|c| c == "is-open"));
        let grid = surface_grid(&el);
        assert!(grid.class_names().iter().any(|c| c == "pk-calendar-grid"));
        assert_eq!(grid.child_elements().len(), 6); // six week rows
        assert_eq!(grid.child_elements()[0].child_elements().len(), 7); // seven cells
    }

    #[test]
    fn empty_cells_are_marked() {
        let el = DatePicker.render(&DatePickerProps::new().month_of(2024, 2));
        let grid = surface_grid(&el);
        // February 2024 starts on Thursday, so the first three cells are empty.
        let first_row = &grid.child_elements()[0];
        assert!(first_row.child_elements()[0]
            .class_names()
            .iter()
            .any(|n| n == "pk-calendar-grid__cell--empty"));
        assert!(first_row.child_elements()[4]
            .child_elements()
            .iter()
            .any(|t| t.text_content() == Some("1")));
    }

    #[test]
    fn selected_and_today_cells_get_modifiers() {
        let el = DatePicker.render(
            &DatePickerProps::new()
                .month_of(2024, 2)
                .selected(Date::new(2024, 2, 15))
                .today(Date::new(2024, 2, 1)),
        );
        let grid = surface_grid(&el);
        let cells: Vec<_> = grid
            .child_elements()
            .iter()
            .flat_map(|row| row.child_elements().iter().cloned())
            .collect();
        let selected = cells
            .iter()
            .filter(|c| c.class_names().iter().any(|n| n == "pk-calendar-grid__cell--selected"))
            .count();
        let today = cells
            .iter()
            .filter(|c| c.class_names().iter().any(|n| n == "pk-calendar-grid__cell--today"))
            .count();
        assert_eq!(selected, 1);
        assert_eq!(today, 1);
    }

    #[test]
    fn role_is_dialog() {
        assert_eq!(DatePicker::role(), Role::Dialog);
    }
}
