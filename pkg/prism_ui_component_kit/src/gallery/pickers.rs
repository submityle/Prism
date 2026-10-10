//! Gallery instances for the `pickers` family. Dev-only; see `gallery/mod.rs`.

use alloc::vec::Vec;

use prism_ui_component::mount_component;
use prism_ui_style::StyleValue;

use super::Showcase;
use crate::pickers::{
    ColorInput, ColorInputProps, ColorPicker, ColorPickerProps, ColorWheel, ColorWheelProps,
    CurveEditor, CurveEditorProps, Date, DatePicker, DatePickerProps, DateRangePicker,
    DateRangePickerProps, GradientEditor, GradientEditorProps, TimePicker, TimePickerProps,
};

/// Real, named instances of every `pickers` control.
#[must_use]
pub fn instances() -> Vec<Showcase> {
    let red = StyleValue::rgba8(220, 60, 60, 255);
    let green = StyleValue::rgba8(60, 180, 90, 255);
    let blue = StyleValue::rgba8(60, 110, 220, 255);

    alloc::vec![
        Showcase::new(
            "ColorPicker",
            mount_component(
                &ColorPicker,
                ColorPickerProps::new()
                    .swatches([red.clone(), green.clone(), blue.clone()])
                    .value(green.clone())
                    .allow_custom(true),
            ),
        ),
        Showcase::new(
            "ColorWheel",
            mount_component(
                &ColorWheel,
                ColorWheelProps::new().value(blue.clone()).show_alpha(true),
            ),
        ),
        Showcase::new(
            "ColorInput",
            mount_component(
                &ColorInput,
                ColorInputProps::new().value(red.clone()).with_picker(true),
            ),
        ),
        Showcase::new(
            "GradientEditor",
            mount_component(
                &GradientEditor,
                GradientEditorProps::new().stops([
                    (0.0, red.clone()),
                    (0.5, green.clone()),
                    (1.0, blue.clone()),
                ]),
            ),
        ),
        Showcase::new(
            "CurveEditor",
            mount_component(
                &CurveEditor,
                CurveEditorProps::new().points([(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)]),
            ),
        ),
        Showcase::new(
            "TimePicker",
            mount_component(
                &TimePicker,
                TimePickerProps::new()
                    .hour(9)
                    .minute(30)
                    .second(15)
                    .show_seconds(true),
            ),
        ),
        Showcase::new(
            "DatePicker",
            mount_component(
                &DatePicker,
                DatePickerProps::new()
                    .month_of(2026, 10)
                    .selected(Date::new(2026, 10, 10))
                    .today(Date::new(2026, 10, 10))
                    .open(true),
            ),
        ),
        Showcase::new(
            "DateRangePicker",
            mount_component(
                &DateRangePicker,
                DateRangePickerProps::new()
                    .start_month_of(2026, 10)
                    .end_month_of(2026, 11)
                    .range(Date::new(2026, 10, 5), Date::new(2026, 11, 12))
                    .open(true),
            ),
        ),
    ]
}
