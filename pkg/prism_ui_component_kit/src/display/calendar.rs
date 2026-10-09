//! [`Calendar`] — a static month-grid view (read-only).
//!
//! A calendar renders a `pk-calendar` column: a `__weekdays` header row of
//! single-letter labels above a `__grid` of weeks. Each week is a `__week` row
//! of seven `__day` cells; empty leading/trailing cells carry `--empty`, the
//! selected day adds `--selected` and today adds `--today`. The month layout is
//! computed by the pure [`crate::pickers::calendar`] date algorithm, so this
//! control stays data-only and attaches only kit class names; color and spacing
//! resolve from theme tokens via [`crate::preset`].
//!
//! This differs from [`crate::pickers::DatePicker`]: the picker is an
//! interactive popover that owns the `pk-calendar-grid` family, whereas this is
//! a flat, always-visible month block for dashboards and summaries. The two
//! deliberately use distinct class families so their styles never collide.

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::pickers::calendar::{self, Date};
use crate::preset::StyleSheet;

/// Props for [`Calendar`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CalendarProps {
    /// The year of the displayed month.
    pub year: i32,
    /// The 1-based month on display (`1 = January`). Values outside `1..=12`
    /// render an empty grid.
    pub month: u8,
    /// The currently highlighted day, if any.
    pub selected: Option<Date>,
    /// The day to flag as "today", if any.
    pub today: Option<Date>,
}

impl CalendarProps {
    /// Creates props for a given year/month with no highlighted days.
    #[must_use]
    pub fn new(year: i32, month: u8) -> Self {
        Self {
            year,
            month,
            ..Self::default()
        }
    }

    /// Sets the highlighted (selected) day.
    #[must_use]
    pub fn selected(mut self, date: Date) -> Self {
        self.selected = Some(date);
        self
    }

    /// Sets the day flagged as "today".
    #[must_use]
    pub fn today(mut self, date: Date) -> Self {
        self.today = Some(date);
        self
    }
}

/// The static calendar control. Zero-sized; configuration lives in
/// [`CalendarProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Calendar;

impl Calendar {
    /// The accessibility role a calendar exposes (a grouping of day cells).
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }

    /// Whether a day in the displayed month matches `date`.
    fn day_matches(props_year: i32, props_month: u8, day: u8, date: Option<Date>) -> bool {
        match date {
            Some(d) => d.year == props_year && d.month == props_month && d.day == day,
            None => false,
        }
    }
}

/// The single-letter weekday headers, Sunday-first (matching
/// [`calendar::month_grid`]).
const WEEKDAY_LABELS: [&str; 7] = ["S", "M", "T", "W", "T", "F", "S"];

impl Component for Calendar {
    type Props = CalendarProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut root = Element::box_().class("pk-calendar");

        // Weekday header row.
        let mut header = Element::box_().class("pk-calendar__weekdays");
        for label in WEEKDAY_LABELS {
            header = header
                .child(Element::text(String::from(label)).class("pk-calendar__weekday"));
        }
        root = root.child(header);

        // Month grid: six weeks of seven day cells.
        let grid_data = calendar::month_grid(props.year, props.month);
        let mut grid = Element::box_().class("pk-calendar__grid");
        for week in grid_data {
            let mut week_row = Element::box_().class("pk-calendar__week");
            for cell in week {
                let day_el = match cell {
                    Some(day) => {
                        let mut d = Element::text(day_label(day)).class("pk-calendar__day");
                        if Calendar::day_matches(props.year, props.month, day, props.selected) {
                            d = d.class("pk-calendar__day--selected");
                        }
                        if Calendar::day_matches(props.year, props.month, day, props.today) {
                            d = d.class("pk-calendar__day--today");
                        }
                        d
                    }
                    None => Element::box_()
                        .class("pk-calendar__day")
                        .class("pk-calendar__day--empty"),
                };
                week_row = week_row.child(day_el);
            }
            grid = grid.child(week_row);
        }
        root = root.child(grid);
        root
    }
}

/// Renders a 1..=31 day number as a short decimal string without `std`.
fn day_label(day: u8) -> String {
    let mut s = String::new();
    if day >= 10 {
        s.push((b'0' + day / 10) as char);
    }
    s.push((b'0' + day % 10) as char);
    s
}

/// Registers the `pk-calendar` class family: container, weekday header, week
/// rows and day cells with selected/today/empty modifiers. All values resolve
/// from theme tokens so light/dark needs no change here.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok, transparent};

    // Container: a vertical column with small gaps between header and grid.
    sheet.insert(
        Class::new("pk-calendar")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xs"))
            .with(StyleProp::BackgroundColor, tok("color.surface"))
            .with(StyleProp::BorderRadius, tok("radius.lg"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.sm")),
    );

    // Weekday header: a seven-column row of centered labels.
    sheet.insert(
        Class::new("pk-calendar__weekdays")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::Gap, tok("space.xxs")),
    );
    sheet.insert(
        Class::new("pk-calendar__weekday")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::FlexBasis, StyleValue::px(0.0))
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::FontSize, tok("font.size.caption1"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Grid: a vertical stack of week rows.
    sheet.insert(
        Class::new("pk-calendar__grid")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs")),
    );
    sheet.insert(
        Class::new("pk-calendar__week")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::Gap, tok("space.xxs")),
    );

    // Day cell: an equal-width square-ish tappable target.
    sheet.insert(
        Class::new("pk-calendar__day")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::FlexBasis, StyleValue::px(0.0))
            .with(StyleProp::Height, StyleValue::px(32.0))
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with_state(InteractionState::Hover, StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    // Selected: solid accent fill with a white label.
    sheet.insert(
        Class::new("pk-calendar__day--selected")
            .with(StyleProp::BackgroundColor, tok("color.tint"))
            .with(StyleProp::Color, StyleValue::rgba8(255, 255, 255, 255))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );

    // Today: accent label with a subtle fill ring.
    sheet.insert(
        Class::new("pk-calendar__day--today")
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );

    // Empty padding cells: no surface, no interaction affordance.
    sheet.insert(
        Class::new("pk-calendar__day--empty")
            .with(StyleProp::BackgroundColor, transparent()),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn render(props: CalendarProps) -> Element {
        Calendar.render(&props)
    }

    fn weeks(el: &Element) -> Vec<Element> {
        let grid = el
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-calendar__grid"))
            .expect("grid")
            .clone();
        grid.child_elements().to_vec()
    }

    #[test]
    fn attaches_container_class() {
        let el = render(CalendarProps::new(2024, 2));
        assert_eq!(el.class_names(), ["pk-calendar"]);
    }

    #[test]
    fn header_has_seven_weekday_labels() {
        let el = render(CalendarProps::new(2024, 2));
        let header = el
            .child_elements()
            .iter()
            .find(|c| c.class_names().iter().any(|n| n == "pk-calendar__weekdays"))
            .expect("header")
            .clone();
        assert_eq!(header.child_elements().len(), 7);
    }

    #[test]
    fn grid_is_six_weeks_of_seven_days() {
        let el = render(CalendarProps::new(2024, 2));
        let rows = weeks(&el);
        assert_eq!(rows.len(), 6);
        for row in &rows {
            assert_eq!(row.child_elements().len(), 7);
        }
    }

    #[test]
    fn selected_day_gets_modifier() {
        let el = render(CalendarProps::new(2024, 2).selected(Date::new(2024, 2, 15)));
        let has_selected = weeks(&el).iter().any(|row| {
            row.child_elements().iter().any(|cell| {
                cell.text_content() == Some("15")
                    && cell
                        .class_names()
                        .iter()
                        .any(|n| n == "pk-calendar__day--selected")
            })
        });
        assert!(has_selected);
    }

    #[test]
    fn empty_cells_carry_empty_modifier() {
        // February 2024 starts on a Thursday, so the first row has leading
        // empty padding cells.
        let el = render(CalendarProps::new(2024, 2));
        let first_week = &weeks(&el)[0];
        let empties = first_week
            .child_elements()
            .iter()
            .filter(|c| c.class_names().iter().any(|n| n == "pk-calendar__day--empty"))
            .count();
        assert!(empties > 0);
    }

    #[test]
    fn invalid_month_renders_all_empty() {
        let el = render(CalendarProps::new(2024, 13));
        let all_empty = weeks(&el).iter().all(|row| {
            row.child_elements()
                .iter()
                .all(|c| c.class_names().iter().any(|n| n == "pk-calendar__day--empty"))
        });
        assert!(all_empty);
    }

    #[test]
    fn role_is_group() {
        assert_eq!(Calendar::role(), Role::Group);
    }
}
