//! Interpolation of [`StyleValue`]s for implicit style transitions.
//!
//! Only a subset of style values vary continuously: pixel lengths, percentage
//! lengths, unitless numbers and colors. These are interpolated numerically.
//! Everything else (keywords, `auto`, unresolved token references, or a pair
//! whose variants or length units differ) is **not** numerically meaningful to
//! blend, so it uses a discrete strategy: the value switches from the start to
//! the end once the eased progress crosses a threshold (the midpoint, `0.5`, by
//! default).
//!
//! [`AnimatableValue`] is a thin newtype implementing [`Lerp`], which lets a
//! [`StyleValue`] be driven by a [`prism_ui_anim::Tween`].

use prism_ui_anim::Lerp;
use prism_ui_style::{Color, Length, StyleValue};

/// The default discrete switch point: the midpoint of the eased progress.
pub const DISCRETE_MIDPOINT: f32 = 0.5;

/// Whether two values can be blended continuously (numerically).
///
/// Returns `true` only for matching, numerically meaningful variants: two
/// pixel lengths, two percentage lengths, two numbers or two colors.
#[must_use]
pub fn is_continuous(from: &StyleValue, to: &StyleValue) -> bool {
    matches!(
        (from, to),
        (
            StyleValue::Length(Length::Px(_)),
            StyleValue::Length(Length::Px(_)),
        ) | (
            StyleValue::Length(Length::Percent(_)),
            StyleValue::Length(Length::Percent(_)),
        ) | (StyleValue::Number(_), StyleValue::Number(_))
            | (StyleValue::Color(_), StyleValue::Color(_))
    )
}

/// Interpolates between two style values using the default
/// [`DISCRETE_MIDPOINT`] for non-continuous pairs.
///
/// See [`interpolate_with`] for the full strategy.
#[must_use]
pub fn interpolate(from: &StyleValue, to: &StyleValue, t: f32) -> StyleValue {
    interpolate_with(from, to, t, DISCRETE_MIDPOINT)
}

/// Interpolates between two style values.
///
/// Continuous pairs (see [`is_continuous`]) are blended numerically. All other
/// pairs are discrete: the result is `from` while `t < threshold` and `to`
/// afterwards, so the value changes exactly once at the threshold.
#[must_use]
pub fn interpolate_with(from: &StyleValue, to: &StyleValue, t: f32, threshold: f32) -> StyleValue {
    match (from, to) {
        (StyleValue::Length(Length::Px(a)), StyleValue::Length(Length::Px(b))) => {
            StyleValue::px(a.lerp(b, t))
        }
        (StyleValue::Length(Length::Percent(a)), StyleValue::Length(Length::Percent(b))) => {
            StyleValue::percent(a.lerp(b, t))
        }
        (StyleValue::Number(a), StyleValue::Number(b)) => StyleValue::number(a.lerp(b, t)),
        (StyleValue::Color(a), StyleValue::Color(b)) => StyleValue::Color(lerp_color(a, b, t)),
        _ => discrete(from, to, t, threshold),
    }
}

/// Discrete switch: `from` before `threshold`, `to` at or after it.
fn discrete(from: &StyleValue, to: &StyleValue, t: f32, threshold: f32) -> StyleValue {
    if t < threshold {
        from.clone()
    } else {
        to.clone()
    }
}

/// Interpolates each color channel independently, reusing the array [`Lerp`].
fn lerp_color(from: &Color, to: &Color, t: f32) -> Color {
    let a = [from.r, from.g, from.b, from.a];
    let b = [to.r, to.g, to.b, to.a];
    let out = a.lerp(&b, t);
    Color::rgba(out[0], out[1], out[2], out[3])
}

/// A [`StyleValue`] wrapper that implements [`Lerp`].
///
/// The orphan rule prevents implementing a foreign trait for the foreign
/// [`StyleValue`], so this newtype carries the [`Lerp`] implementation used by
/// [`prism_ui_anim::Tween`]. It blends with the default [`DISCRETE_MIDPOINT`].
#[derive(Clone, Debug, PartialEq)]
pub struct AnimatableValue(pub StyleValue);

impl AnimatableValue {
    /// Wraps a style value.
    #[must_use]
    pub const fn new(value: StyleValue) -> Self {
        Self(value)
    }

    /// Borrows the wrapped value.
    #[must_use]
    pub fn get(&self) -> &StyleValue {
        &self.0
    }

    /// Unwraps into the inner style value.
    #[must_use]
    pub fn into_inner(self) -> StyleValue {
        self.0
    }
}

impl From<StyleValue> for AnimatableValue {
    #[inline]
    fn from(value: StyleValue) -> Self {
        Self(value)
    }
}

impl Lerp for AnimatableValue {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        Self(interpolate(&self.0, &other.0, t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    #[test]
    fn pixels_blend_continuously() {
        let a = StyleValue::px(0.0);
        let b = StyleValue::px(100.0);
        assert!(is_continuous(&a, &b));
        match interpolate(&a, &b, 0.25) {
            StyleValue::Length(Length::Px(v)) => assert!((v - 25.0).abs() < EPS),
            other => panic!("expected px, got {other:?}"),
        }
    }

    #[test]
    fn percent_blends_continuously() {
        let a = StyleValue::percent(10.0);
        let b = StyleValue::percent(20.0);
        match interpolate(&a, &b, 0.5) {
            StyleValue::Length(Length::Percent(v)) => assert!((v - 15.0).abs() < EPS),
            other => panic!("expected percent, got {other:?}"),
        }
    }

    #[test]
    fn numbers_blend_continuously() {
        let a = StyleValue::number(1.0);
        let b = StyleValue::number(0.0);
        match interpolate(&a, &b, 0.75) {
            StyleValue::Number(v) => assert!((v - 0.25).abs() < EPS),
            other => panic!("expected number, got {other:?}"),
        }
    }

    #[test]
    fn colors_blend_per_channel() {
        let a = StyleValue::rgba(0.0, 0.0, 0.0, 1.0);
        let b = StyleValue::rgba(1.0, 0.5, 0.0, 1.0);
        match interpolate(&a, &b, 0.5) {
            StyleValue::Color(c) => {
                assert!((c.r - 0.5).abs() < EPS);
                assert!((c.g - 0.25).abs() < EPS);
                assert!((c.b - 0.0).abs() < EPS);
                assert!((c.a - 1.0).abs() < EPS);
            }
            other => panic!("expected color, got {other:?}"),
        }
    }

    #[test]
    fn mismatched_length_units_are_discrete() {
        let a = StyleValue::px(0.0);
        let b = StyleValue::percent(100.0);
        assert!(!is_continuous(&a, &b));
        assert_eq!(interpolate(&a, &b, 0.49), a);
        assert_eq!(interpolate(&a, &b, 0.51), b);
    }

    #[test]
    fn keywords_switch_at_midpoint() {
        use prism_ui_style::Keyword;
        let a = StyleValue::keyword(Keyword::Row);
        let b = StyleValue::keyword(Keyword::Column);
        assert!(!is_continuous(&a, &b));
        assert_eq!(interpolate(&a, &b, 0.0), a);
        assert_eq!(interpolate(&a, &b, 0.5), b);
        assert_eq!(interpolate(&a, &b, 1.0), b);
    }

    #[test]
    fn auto_length_is_discrete_with_custom_threshold() {
        let a = StyleValue::auto();
        let b = StyleValue::px(42.0);
        // Switch three quarters of the way through instead of the midpoint.
        assert_eq!(interpolate_with(&a, &b, 0.7, 0.75), a);
        assert_eq!(interpolate_with(&a, &b, 0.8, 0.75), b);
    }

    #[test]
    fn animatable_value_lerps_via_newtype() {
        let a = AnimatableValue::new(StyleValue::px(0.0));
        let b = AnimatableValue::from(StyleValue::px(10.0));
        let mid = a.lerp(&b, 0.5);
        assert_eq!(mid.into_inner(), StyleValue::px(5.0));
        assert_eq!(a.get(), &StyleValue::px(0.0));
    }
}
