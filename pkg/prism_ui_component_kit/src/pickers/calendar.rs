//! `calendar` — a `no_std`, dependency-free date algorithm module.
//!
//! This is **not** a [`Component`](prism_ui_component::Component): it owns no
//! [`Element`](prism_ui::Element) tree and no style class. It is the pure
//! arithmetic backbone shared by [`super::date_picker`] and
//! [`super::date_range_picker`] (and, per the design doc, the display-layer
//! `Calendar`). Everything here is integer-only — no `chrono`, no floats, no
//! `std` — so it compiles and unit-tests under `#![no_std]` with nothing but
//! `core`.
//!
//! Weekdays use **Sakamoto's algorithm** and are reported with `0 = Sunday`
//! through `6 = Saturday`. [`month_grid`] is therefore laid out **Sunday-first**:
//! column `0` is Sunday and column `6` is Saturday.
//!
//! The month argument is 1-based (`1 = January`). Inputs outside `1..=12` are
//! treated as empty rather than panicking, so callers can pass a
//! default-constructed state without a guard.

/// A calendar date as plain `(year, month, day)` fields.
///
/// Ordering is chronological because the fields are declared most-significant
/// first, which is exactly what the derived [`Ord`] compares. No validation is
/// performed on construction; the picker controls only ever build these from
/// values produced by [`month_grid`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date {
    /// The proleptic-Gregorian year (e.g. `2026`).
    pub year: i32,
    /// The 1-based month, where `1 = January` and `12 = December`.
    pub month: u8,
    /// The 1-based day of the month.
    pub day: u8,
}

impl Date {
    /// Creates a date from its parts. Performs no range validation.
    #[must_use]
    pub const fn new(year: i32, month: u8, day: u8) -> Self {
        Self { year, month, day }
    }
}

/// Returns `true` if `year` is a leap year in the proleptic Gregorian calendar.
///
/// A year is a leap year when it is divisible by 4, except centuries, which
/// must also be divisible by 400 (so 2000 is a leap year but 1900 is not).
#[must_use]
pub fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Returns the number of days in `month` of `year`, or `0` when `month` is
/// outside `1..=12`.
#[must_use]
pub fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Returns the weekday of `year-month-day` as `0 = Sunday` … `6 = Saturday`.
///
/// Uses Sakamoto's algorithm, which is exact for the proleptic Gregorian
/// calendar. Months outside `1..=12` return `0` so the function is total.
#[must_use]
pub fn weekday(year: i32, month: u8, day: u8) -> u8 {
    if !(1..=12).contains(&month) {
        return 0;
    }
    // Month offset table used by Sakamoto's algorithm (1-based month).
    const OFFSET: [i32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let mut y = year;
    if month < 3 {
        y -= 1;
    }
    let m = (month as usize) - 1;
    let dow = (y + y / 4 - y / 100 + y / 400 + OFFSET[m] + i32::from(day)).rem_euclid(7);
    dow as u8
}

/// Builds a 6-row × 7-column month grid, **Sunday-first**.
///
/// Each cell is `Some(day)` for a day belonging to `month`, or `None` for the
/// leading/trailing blanks around it. Six rows always suffice: the worst case
/// is a 31-day month whose first day lands on Saturday (offset 6), filling 37
/// of the 42 cells. A `month` outside `1..=12` yields an all-`None` grid.
#[must_use]
pub fn month_grid(year: i32, month: u8) -> [[Option<u8>; 7]; 6] {
    let mut grid = [[None; 7]; 6];
    let total = days_in_month(year, month);
    if total == 0 {
        return grid;
    }
    // Linear cell index of the 1st, where index 0 is the top-left (Sunday).
    let mut cell = weekday(year, month, 1) as usize;
    let mut day = 1u8;
    while day <= total {
        let row = cell / 7;
        let col = cell % 7;
        if row < 6 {
            grid[row][col] = Some(day);
        }
        cell += 1;
        day += 1;
    }
    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leap_years_follow_the_gregorian_rule() {
        assert!(is_leap_year(2000)); // divisible by 400
        assert!(!is_leap_year(1900)); // century, not div by 400
        assert!(is_leap_year(2024)); // ordinary leap year
        assert!(!is_leap_year(2023)); // common year
    }

    #[test]
    fn month_lengths_are_correct() {
        assert_eq!(days_in_month(2023, 1), 31);
        assert_eq!(days_in_month(2023, 4), 30);
        assert_eq!(days_in_month(2024, 2), 29); // leap February
        assert_eq!(days_in_month(2023, 2), 28); // common February
        assert_eq!(days_in_month(2023, 12), 31);
    }

    #[test]
    fn invalid_month_is_total_not_panicking() {
        assert_eq!(days_in_month(2024, 0), 0);
        assert_eq!(days_in_month(2024, 13), 0);
        assert_eq!(weekday(2024, 0, 1), 0);
        assert_eq!(month_grid(2024, 0), [[None; 7]; 6]);
    }

    #[test]
    fn known_weekdays_are_exact() {
        // 2000-01-01 was a Saturday (6); 2024-01-01 was a Monday (1).
        assert_eq!(weekday(2000, 1, 1), 6);
        assert_eq!(weekday(2024, 1, 1), 1);
    }

    #[test]
    fn grid_is_six_by_seven() {
        let grid = month_grid(2024, 2);
        assert_eq!(grid.len(), 6);
        for row in &grid {
            assert_eq!(row.len(), 7);
        }
    }

    #[test]
    fn grid_places_the_first_day_on_its_weekday_column() {
        // February 2024 began on a Thursday (weekday 4).
        let grid = month_grid(2024, 2);
        assert_eq!(grid[0][4], Some(1));
        assert_eq!(grid[0][3], None);
    }

    #[test]
    fn grid_contains_every_day_exactly_once() {
        let grid = month_grid(2024, 2);
        let present: usize = grid
            .iter()
            .flat_map(|row| row.iter())
            .filter(|cell| cell.is_some())
            .count();
        assert_eq!(present, 29);
        // Last day sits where it should: 2024-02-29 is a Thursday (col 4).
        assert_eq!(grid[4][4], Some(29));
    }
}
