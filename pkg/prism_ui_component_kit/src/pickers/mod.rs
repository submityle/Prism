//! `pickers/` controls. See the kit design doc, sections 5.3, 8 and 9.
//!
//! Each control emits a data-only [`Element`](prism_ui::Element) carrying only
//! kit class names; this module's [`register_styles`] owns the token-backed
//! [`Class`](prism_ui_style::Class) rules for those names. Controls stay plain
//! [`Component`](prism_ui_component::Component)s so they compose and unit-test
//! without a running runtime.
//!
//! [`calendar`] is the odd one out: a `no_std`, dependency-free date algorithm
//! module rather than a control. It carries no element tree and no style, so it
//! has **no** `register_styles`; it is the arithmetic backbone the date pickers
//! build their grids from.
//!
//! The date controls share the `pk-calendar-grid` base: [`date_picker`] owns
//! and registers that family, and [`date_range_picker`] reuses it, adding only
//! its range-highlight modifiers. [`register_styles`] therefore registers the
//! single-day picker before the range picker.

use crate::preset::StyleSheet;

pub mod calendar;
pub mod color_input;
pub mod color_picker;
pub mod color_wheel;
pub mod curve_editor;
pub mod date_picker;
pub mod date_range_picker;
pub mod gradient_editor;
pub mod time_picker;

pub use calendar::{days_in_month, is_leap_year, month_grid, weekday, Date};
pub use color_input::{ColorInput, ColorInputProps};
pub use color_picker::{ColorPicker, ColorPickerProps};
pub use color_wheel::{ColorWheel, ColorWheelProps};
pub use curve_editor::{CurveEditor, CurveEditorProps};
pub use date_picker::{DatePicker, DatePickerProps};
pub use date_range_picker::{DateRangePicker, DateRangePickerProps};
pub use gradient_editor::{GradientEditor, GradientEditorProps};
pub use time_picker::{TimePicker, TimePickerProps};

/// Registers every `pickers/` control's token-backed classes into `sheet`.
///
/// [`date_picker`] registers before [`date_range_picker`] because the range
/// picker reuses the `pk-calendar-grid` base the single-day picker owns.
/// [`calendar`] is a pure algorithm module and has no classes to register.
pub fn register_styles(sheet: &mut StyleSheet) {
    color_picker::register_styles(sheet);
    color_wheel::register_styles(sheet);
    color_input::register_styles(sheet);
    date_picker::register_styles(sheet);
    date_range_picker::register_styles(sheet);
    time_picker::register_styles(sheet);
    gradient_editor::register_styles(sheet);
    curve_editor::register_styles(sheet);
}
