//! CSS `<angle>` values: unit-tagged angles with exact-ratio conversions
//! and modular normalization.
//!
//! Style sheets express rotations, conic-gradient stops and `hue` components
//! as a number paired with one of the four CSS angle units: `deg`, `grad`,
//! `rad` or `turn`. This module models that `<angle>` value type from CSS
//! Values and Units, keeping the authored unit for round-tripping while
//! offering conversions and `[0, full-circle)` normalization.
//!
//! A full circle is 360 `deg`, 400 `grad`, 1 `turn` or 2π `rad`. Degree-based
//! ratios (360, 0.9, 360, 180/π per unit) are used so that the common
//! `deg`/`grad`/`turn` conversions are exact rationals and only the radian
//! path carries the floating-point π.
//!
//! # Example
//!
//! ```
//! use prism_ui_style::{Angle, AngleUnit};
//!
//! let quarter = Angle::grad(100.0);
//! assert!((quarter.to_degrees() - 90.0).abs() < 1e-3);
//!
//! // Keep the authored unit but convert the number.
//! let as_turns = quarter.to_unit(AngleUnit::Turn);
//! assert_eq!(as_turns.unit(), AngleUnit::Turn);
//! assert!((as_turns.value() - 0.25).abs() < 1e-3);
//!
//! // Normalization wraps into one revolution without changing the unit.
//! let wrapped = Angle::deg(450.0).normalized();
//! assert!((wrapped.to_degrees() - 90.0).abs() < 1e-3);
//! ```

/// Degrees in one full revolution.
const DEG_PER_CIRCLE: f32 = 360.0;
/// Degrees per gradian (`360 / 400`).
const DEG_PER_GRAD: f32 = 0.9;
/// Gradians in one full revolution.
const GRAD_PER_CIRCLE: f32 = 400.0;
/// Turns in one full revolution.
const TURN_PER_CIRCLE: f32 = 1.0;

/// The unit a CSS `<angle>` was authored in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AngleUnit {
    /// Degrees (`deg`); a full circle is 360.
    #[default]
    Deg,
    /// Gradians (`grad`); a full circle is 400.
    Grad,
    /// Radians (`rad`); a full circle is 2π.
    Rad,
    /// Turns (`turn`); a full circle is 1.
    Turn,
}

impl AngleUnit {
    /// Returns the value of one full revolution expressed in this unit.
    #[must_use]
    pub fn full_circle(self) -> f32 {
        match self {
            AngleUnit::Deg => DEG_PER_CIRCLE,
            AngleUnit::Grad => GRAD_PER_CIRCLE,
            AngleUnit::Rad => core::f32::consts::TAU,
            AngleUnit::Turn => TURN_PER_CIRCLE,
        }
    }

    /// Returns the CSS unit token (`"deg"`, `"grad"`, `"rad"`, `"turn"`).
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            AngleUnit::Deg => "deg",
            AngleUnit::Grad => "grad",
            AngleUnit::Rad => "rad",
            AngleUnit::Turn => "turn",
        }
    }
}

/// A CSS `<angle>`: a magnitude paired with its authored [`AngleUnit`].
///
/// Conversions go through degrees, so `deg`/`grad`/`turn` round-trips are
/// exact up to floating-point rounding and only the radian path introduces π.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Angle {
    value: f32,
    unit: AngleUnit,
}

impl Angle {
    /// Builds an angle in degrees.
    #[must_use]
    pub const fn deg(value: f32) -> Self {
        Self { value, unit: AngleUnit::Deg }
    }

    /// Builds an angle in gradians.
    #[must_use]
    pub const fn grad(value: f32) -> Self {
        Self { value, unit: AngleUnit::Grad }
    }

    /// Builds an angle in radians.
    #[must_use]
    pub const fn rad(value: f32) -> Self {
        Self { value, unit: AngleUnit::Rad }
    }

    /// Builds an angle in turns.
    #[must_use]
    pub const fn turn(value: f32) -> Self {
        Self { value, unit: AngleUnit::Turn }
    }

    /// Builds an angle with an explicit unit.
    #[must_use]
    pub const fn new(value: f32, unit: AngleUnit) -> Self {
        Self { value, unit }
    }

    /// Returns the raw magnitude in this angle's own unit.
    #[must_use]
    pub const fn value(self) -> f32 {
        self.value
    }

    /// Returns the unit this angle is expressed in.
    #[must_use]
    pub const fn unit(self) -> AngleUnit {
        self.unit
    }

    /// Returns the magnitude expressed in degrees.
    #[must_use]
    pub fn to_degrees(self) -> f32 {
        match self.unit {
            AngleUnit::Deg => self.value,
            AngleUnit::Grad => self.value * DEG_PER_GRAD,
            AngleUnit::Rad => self.value * (DEG_PER_CIRCLE / core::f32::consts::TAU),
            AngleUnit::Turn => self.value * DEG_PER_CIRCLE,
        }
    }

    /// Returns the magnitude expressed in gradians.
    #[must_use]
    pub fn to_gradians(self) -> f32 {
        self.to_degrees() / DEG_PER_GRAD
    }

    /// Returns the magnitude expressed in radians.
    #[must_use]
    pub fn to_radians(self) -> f32 {
        self.to_degrees() * (core::f32::consts::TAU / DEG_PER_CIRCLE)
    }

    /// Returns the magnitude expressed in turns.
    #[must_use]
    pub fn to_turns(self) -> f32 {
        self.to_degrees() / DEG_PER_CIRCLE
    }

    /// Converts this angle to `unit`, preserving the represented rotation.
    #[must_use]
    pub fn to_unit(self, unit: AngleUnit) -> Angle {
        let value = match unit {
            AngleUnit::Deg => self.to_degrees(),
            AngleUnit::Grad => self.to_gradians(),
            AngleUnit::Rad => self.to_radians(),
            AngleUnit::Turn => self.to_turns(),
        };
        Angle { value, unit }
    }

    /// Returns the magnitude in degrees wrapped into `[0, 360)`.
    #[must_use]
    pub fn normalized_degrees(self) -> f32 {
        wrap(self.to_degrees(), DEG_PER_CIRCLE)
    }

    /// Returns this angle wrapped into one revolution, keeping its unit.
    ///
    /// The result lies in `[0, full-circle)` for the angle's own unit.
    #[must_use]
    pub fn normalized(self) -> Angle {
        Angle { value: wrap(self.value, self.unit.full_circle()), unit: self.unit }
    }

    /// Returns `true` if this angle is already in `[0, full-circle)`.
    #[must_use]
    pub fn is_normalized(self) -> bool {
        (0.0..self.unit.full_circle()).contains(&self.value)
    }
}

/// Euclidean remainder for `f32` that stays in `core` (no_std-safe).
///
/// For a positive `modulus`, returns a value in `[0, modulus)` congruent to
/// `value`. `f32::rem_euclid` is unavailable without `std`, so this folds a
/// possibly-negative `%` result back into range.
fn wrap(value: f32, modulus: f32) -> f32 {
    let r = value % modulus;
    if r < 0.0 { r + modulus } else { r }
}

#[cfg(test)]
mod tests {
    use super::{Angle, AngleUnit};

    /// Small deterministic PRNG for property tests (`SplitMix64`).
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform `f32` in `[-range, range]`.
        fn signed(&mut self, range: f32) -> f32 {
            let frac = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
            (frac * 2.0 - 1.0) * range
        }
    }

    const UNITS: [AngleUnit; 4] =
        [AngleUnit::Deg, AngleUnit::Grad, AngleUnit::Rad, AngleUnit::Turn];

    #[test]
    fn quarter_turn_is_equivalent_across_units() {
        let variants = [
            Angle::deg(90.0),
            Angle::grad(100.0),
            Angle::rad(core::f32::consts::FRAC_PI_2),
            Angle::turn(0.25),
        ];
        for a in variants {
            assert!(
                (a.to_degrees() - 90.0).abs() < 1e-3,
                "{a:?} -> {} deg",
                a.to_degrees()
            );
        }
    }

    #[test]
    fn full_circle_values_match_units() {
        assert!((AngleUnit::Deg.full_circle() - 360.0).abs() < 1e-6);
        assert!((AngleUnit::Grad.full_circle() - 400.0).abs() < 1e-6);
        assert!((AngleUnit::Turn.full_circle() - 1.0).abs() < 1e-6);
        assert!((AngleUnit::Rad.full_circle() - core::f32::consts::TAU).abs() < 1e-6);
    }

    #[test]
    fn conversion_round_trips_through_every_unit() {
        let mut rng = SplitMix64(0x1234_5678_9ABC_DEF0);
        for _ in 0..2000 {
            let deg = rng.signed(720.0);
            let start = Angle::deg(deg);
            for &unit in &UNITS {
                let back = start.to_unit(unit).to_unit(AngleUnit::Deg).to_degrees();
                let tol = 1e-2 + deg.abs() * 1e-4;
                assert!(
                    (back - deg).abs() <= tol,
                    "deg {deg} via {unit:?} -> {back}"
                );
            }
        }
    }

    #[test]
    fn to_unit_preserves_rotation_in_degrees() {
        let mut rng = SplitMix64(0xDEAD_BEEF_CAFE_F00D);
        for _ in 0..2000 {
            let deg = rng.signed(720.0);
            let start = Angle::deg(deg);
            for &unit in &UNITS {
                let converted = start.to_unit(unit);
                assert_eq!(converted.unit(), unit);
                let tol = 1e-2 + deg.abs() * 1e-4;
                assert!(
                    (converted.to_degrees() - deg).abs() <= tol,
                    "deg {deg} as {unit:?} = {} deg",
                    converted.to_degrees()
                );
            }
        }
    }

    #[test]
    fn normalized_degrees_lies_in_half_open_circle() {
        let mut rng = SplitMix64(0x0BAD_C0DE_1337_7A5A);
        for _ in 0..4000 {
            let deg = rng.signed(5000.0);
            let n = Angle::deg(deg).normalized_degrees();
            assert!((0.0..360.0).contains(&n), "deg {deg} -> {n}");
            // The difference from the original must be a whole number of turns.
            let turns = (deg - n) / 360.0;
            assert!((turns - turns.round()).abs() < 1e-2, "deg {deg} -> {n}");
        }
    }

    #[test]
    fn normalized_preserves_unit_and_is_idempotent() {
        let mut rng = SplitMix64(0x5151_5151_2626_2626);
        for _ in 0..2000 {
            for &unit in &UNITS {
                let raw = rng.signed(10.0) * unit.full_circle();
                let a = Angle::new(raw, unit);
                let n = a.normalized();
                assert_eq!(n.unit(), unit);
                assert!(n.is_normalized(), "{n:?} not normalized");
                // Idempotence: normalizing again is a no-op.
                let n2 = n.normalized();
                assert!((n2.value() - n.value()).abs() < 1e-3);
                // Same rotation modulo a full circle.
                let diff = (a.to_degrees() - n.to_degrees()) / 360.0;
                assert!((diff - diff.round()).abs() < 1e-2, "{a:?} vs {n:?}");
            }
        }
    }

    #[test]
    fn unit_tokens_are_css_keywords() {
        assert_eq!(AngleUnit::Deg.token(), "deg");
        assert_eq!(AngleUnit::Grad.token(), "grad");
        assert_eq!(AngleUnit::Rad.token(), "rad");
        assert_eq!(AngleUnit::Turn.token(), "turn");
    }

    #[test]
    fn default_angle_is_zero_degrees() {
        let a = Angle::default();
        assert_eq!(a.unit(), AngleUnit::Deg);
        assert!(a.value().abs() < 1e-6);
    }
}
