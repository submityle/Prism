//! [`NumberTicker`] — a formatted numeric readout.
//!
//! A number ticker renders a single `pk-number-ticker` text node showing
//! `prefix + value + suffix`, where `value` is formatted to a fixed number of
//! decimal places. Formatting is implemented locally (no `std`-only float
//! helpers) so the control works under `no_std`. The visual count-up animation
//! between values is driven by the runtime layer; this control only renders the
//! current formatted value. Typography resolves from theme tokens via
//! [`crate::preset`].

use alloc::format;
use alloc::string::{String, ToString};

use prism_ui::Element;
use prism_ui_component::Component;

use crate::preset::StyleSheet;

/// Props for [`NumberTicker`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NumberTickerProps {
    /// The value to display.
    pub value: f64,
    /// The number of fractional digits to render.
    pub precision: u8,
    /// A string rendered before the number (e.g. a currency symbol).
    pub prefix: String,
    /// A string rendered after the number (e.g. a unit).
    pub suffix: String,
}

impl NumberTickerProps {
    /// Creates props for `value` with zero fractional digits and no affixes.
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self {
            value,
            ..Self::default()
        }
    }

    /// Sets the fractional-digit count.
    #[must_use]
    pub fn precision(mut self, precision: u8) -> Self {
        self.precision = precision;
        self
    }

    /// Sets the leading affix.
    #[must_use]
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Sets the trailing affix.
    #[must_use]
    pub fn suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = suffix.into();
        self
    }
}

/// Formats `value` to exactly `precision` fractional digits, rounding
/// half away from zero. `no_std`-friendly: avoids `f64::round`/`clamp`.
fn format_fixed(value: f64, precision: u8) -> String {
    // Non-finite inputs degrade to a stable zero rather than panicking.
    if !value.is_finite() {
        return format_fixed(0.0, precision);
    }

    let negative = value < 0.0;
    let magnitude = if negative { -value } else { value };

    // Scale factor 10^precision as an integer.
    let mut scale: u128 = 1;
    for _ in 0..precision {
        scale = scale.saturating_mul(10);
    }

    // Round half up by adding 0.5 before truncating toward zero.
    let scaled = magnitude * scale as f64 + 0.5;
    let scaled_int: u128 = scaled as u128;

    let int_part = scaled_int / scale;
    let frac_part = scaled_int % scale;

    let mut out = String::new();
    if negative && scaled_int != 0 {
        out.push('-');
    }
    out.push_str(&int_part.to_string());
    if precision > 0 {
        out.push('.');
        // Left-pad the fractional digits to the requested width.
        out.push_str(&format!("{:0width$}", frac_part, width = precision as usize));
    }
    out
}

/// The number-ticker control. Zero-sized; config lives in [`NumberTickerProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct NumberTicker;

impl NumberTicker {
    /// The fully formatted display string, `prefix + value + suffix`.
    #[must_use]
    pub fn formatted(props: &NumberTickerProps) -> String {
        let mut out = String::with_capacity(props.prefix.len() + props.suffix.len() + 8);
        out.push_str(&props.prefix);
        out.push_str(&format_fixed(props.value, props.precision));
        out.push_str(&props.suffix);
        out
    }
}

impl Component for NumberTicker {
    type Props = NumberTickerProps;

    fn render(&self, props: &Self::Props) -> Element {
        Element::text(NumberTicker::formatted(props)).class("pk-number-ticker")
    }
}

/// Registers the `pk-number-ticker` class: a prominent numeric readout.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, StyleProp};

    use crate::preset::tok;

    sheet.insert(
        Class::new("pk-number-ticker")
            .with(StyleProp::FontSize, tok("font.size.title3"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: NumberTickerProps) -> Element {
        NumberTicker.render(&props)
    }

    #[test]
    fn renders_single_text_node() {
        let el = render(NumberTickerProps::new(42.0));
        assert_eq!(el.class_names(), ["pk-number-ticker"]);
        assert_eq!(el.text_content(), Some("42"));
    }

    #[test]
    fn formats_fixed_precision() {
        let props = NumberTickerProps::new(1.23456).precision(2);
        assert_eq!(NumberTicker::formatted(&props), "1.23");
    }

    #[test]
    fn rounds_half_away_from_zero() {
        // Use exactly-representable values so the test is not float-flaky.
        assert_eq!(format_fixed(2.5, 0), "3");
        assert_eq!(format_fixed(1.25, 1), "1.3");
        assert_eq!(format_fixed(2.3, 0), "2");
    }

    #[test]
    fn pads_fractional_digits() {
        assert_eq!(format_fixed(1.5, 3), "1.500");
        assert_eq!(format_fixed(0.0, 2), "0.00");
    }

    #[test]
    fn handles_negatives() {
        assert_eq!(format_fixed(-1.25, 1), "-1.3");
        // Negative values that round to zero drop the sign.
        assert_eq!(format_fixed(-0.001, 2), "0.00");
    }

    #[test]
    fn applies_prefix_and_suffix() {
        let props = NumberTickerProps::new(1234.0)
            .precision(0)
            .prefix("$")
            .suffix(" USD");
        assert_eq!(NumberTicker::formatted(&props), "$1234 USD");
    }

    #[test]
    fn non_finite_degrades_to_zero() {
        assert_eq!(format_fixed(f64::INFINITY, 2), "0.00");
        assert_eq!(format_fixed(f64::NAN, 0), "0");
    }
}
