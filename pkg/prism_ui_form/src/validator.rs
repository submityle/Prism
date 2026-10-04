//! The declarative validation model: the [`Validator`] trait and the ready-made
//! rules and combinators built on top of it.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;

use crate::error::ValidationError;
use crate::field::parse_i64;

/// A declarative validation rule over a value of type `T`.
///
/// Built-in validators return errors whose field is left anonymous; the owning
/// form rewrites it to the real field key when it runs the rule.
pub trait Validator<T: ?Sized> {
    /// Validate `value`, returning `Ok(())` when it satisfies the rule.
    fn validate(&self, value: &T) -> Result<(), ValidationError>;
}

/// A boxed string validator — the rule representation a form stores per field.
pub type BoxedValidator = Box<dyn Validator<String>>;

/// Rejects values that are empty or only whitespace.
struct Required;

impl Validator<String> for Required {
    fn validate(&self, value: &String) -> Result<(), ValidationError> {
        if value.trim().is_empty() {
            Err(ValidationError::message_only("This field is required."))
        } else {
            Ok(())
        }
    }
}

/// Rejects values with fewer than the configured number of characters.
struct MinLen(usize);

impl Validator<String> for MinLen {
    fn validate(&self, value: &String) -> Result<(), ValidationError> {
        if value.chars().count() < self.0 {
            Err(ValidationError::message_only(format!(
                "Must be at least {} characters.",
                self.0
            )))
        } else {
            Ok(())
        }
    }
}

/// Rejects values with more than the configured number of characters.
struct MaxLen(usize);

impl Validator<String> for MaxLen {
    fn validate(&self, value: &String) -> Result<(), ValidationError> {
        if value.chars().count() > self.0 {
            Err(ValidationError::message_only(format!(
                "Must be at most {} characters.",
                self.0
            )))
        } else {
            Ok(())
        }
    }
}

/// Parses the value as an integer and checks it lies within an inclusive range.
struct IntRange {
    lo: i64,
    hi: i64,
}

impl Validator<String> for IntRange {
    fn validate(&self, value: &String) -> Result<(), ValidationError> {
        match parse_i64(value) {
            Ok(number) if number >= self.lo && number <= self.hi => Ok(()),
            Ok(_) => Err(ValidationError::message_only(format!(
                "Must be between {} and {}.",
                self.lo, self.hi
            ))),
            Err(_) => Err(ValidationError::message_only("Must be a whole number.")),
        }
    }
}

/// Parses the value as an integer and checks it is aligned to a step: that is,
/// `(value - base)` is an exact multiple of the step. This mirrors the HTML5
/// `stepMismatch` constraint for integer number inputs.
struct IntStep {
    base: i64,
    /// Step magnitude; `0` disables the step check (every integer is aligned).
    step: u64,
}

impl Validator<String> for IntStep {
    fn validate(&self, value: &String) -> Result<(), ValidationError> {
        let Ok(number) = parse_i64(value) else {
            return Err(ValidationError::message_only("Must be a whole number."));
        };
        if self.step == 0 {
            return Ok(());
        }
        // Compute in `i128` so the difference cannot overflow for any `i64`
        // inputs, then test divisibility on the magnitude.
        let diff = i128::from(number) - i128::from(self.base);
        if diff.unsigned_abs().is_multiple_of(u128::from(self.step)) {
            Ok(())
        } else {
            Err(ValidationError::message_only(format!(
                "Must be {} plus a multiple of {}.",
                self.base, self.step
            )))
        }
    }
}

/// Accepts values for which a user predicate returns `true`.
struct Pattern<F> {
    predicate: F,
    message: String,
}

impl<F: Fn(&str) -> bool> Validator<String> for Pattern<F> {
    fn validate(&self, value: &String) -> Result<(), ValidationError> {
        if (self.predicate)(value) {
            Ok(())
        } else {
            Err(ValidationError::message_only(self.message.clone()))
        }
    }
}

/// Defers entirely to a user closure returning `Result<(), String>`.
struct Custom<F> {
    run: F,
}

impl<F: Fn(&str) -> Result<(), String>> Validator<String> for Custom<F> {
    fn validate(&self, value: &String) -> Result<(), ValidationError> {
        match (self.run)(value) {
            Ok(()) => Ok(()),
            Err(message) => Err(ValidationError::message_only(message)),
        }
    }
}

/// Require a non-empty (non-whitespace) value.
pub fn required() -> BoxedValidator {
    Box::new(Required)
}

/// Require at least `n` characters (counted by Unicode scalar value).
pub fn min_len(n: usize) -> BoxedValidator {
    Box::new(MinLen(n))
}

/// Require at most `n` characters (counted by Unicode scalar value).
pub fn max_len(n: usize) -> BoxedValidator {
    Box::new(MaxLen(n))
}

/// Require the value to parse as an integer within the inclusive range
/// `lo..=hi`.
pub fn int_range(lo: i64, hi: i64) -> BoxedValidator {
    Box::new(IntRange { lo, hi })
}

/// Require the integer value to be aligned to `step`, counting from `base`:
/// `(value - base)` must be an exact multiple of `step`. This mirrors the HTML5
/// step-mismatch constraint for integer number inputs.
///
/// The sign of `step` is ignored (its magnitude is used); a `step` of `0`
/// disables the alignment check so any whole number is accepted.
pub fn int_step(base: i64, step: i64) -> BoxedValidator {
    Box::new(IntStep {
        base,
        step: step.unsigned_abs(),
    })
}

/// Require `predicate` to return `true`, reporting `message` otherwise.
pub fn pattern(
    predicate: impl Fn(&str) -> bool + 'static,
    message: impl Into<String>,
) -> BoxedValidator {
    Box::new(Pattern {
        predicate,
        message: message.into(),
    })
}

/// Delegate to an arbitrary closure; its `Err(String)` becomes the message.
pub fn custom(run: impl Fn(&str) -> Result<(), String> + 'static) -> BoxedValidator {
    Box::new(Custom { run })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn ok(result: Result<(), ValidationError>) -> bool {
        result.is_ok()
    }

    #[test]
    fn required_pass_and_fail() {
        assert!(ok(required().validate(&"hi".to_string())));
        assert!(!ok(required().validate(&"   ".to_string())));
        assert!(!ok(required().validate(&String::new())));
    }

    #[test]
    fn min_len_pass_and_fail() {
        assert!(ok(min_len(3).validate(&"abc".to_string())));
        assert!(!ok(min_len(3).validate(&"ab".to_string())));
    }

    #[test]
    fn max_len_pass_and_fail() {
        assert!(ok(max_len(3).validate(&"abc".to_string())));
        assert!(!ok(max_len(3).validate(&"abcd".to_string())));
    }

    #[test]
    fn int_range_pass_and_fail() {
        assert!(ok(int_range(1, 10).validate(&"5".to_string())));
        assert!(ok(int_range(1, 10).validate(&" 10 ".to_string())));
        assert!(!ok(int_range(1, 10).validate(&"0".to_string())));
        assert!(!ok(int_range(1, 10).validate(&"nan".to_string())));
    }

    #[test]
    fn pattern_pass_and_fail() {
        let starts_with_a = pattern(|value| value.starts_with('a'), "must start with a");
        assert!(ok(starts_with_a.validate(&"apple".to_string())));
        let err = starts_with_a.validate(&"banana".to_string()).unwrap_err();
        assert_eq!(err.message, "must start with a");
    }

    #[test]
    fn custom_pass_and_fail() {
        let even_length = custom(|value| {
            if value.len() % 2 == 0 {
                Ok(())
            } else {
                Err("length must be even".to_string())
            }
        });
        assert!(ok(even_length.validate(&"ab".to_string())));
        let err = even_length.validate(&"abc".to_string()).unwrap_err();
        assert_eq!(err.message, "length must be even");
    }

    #[test]
    fn int_step_alignment_and_parse() {
        // base 0, step 5: multiples of 5 (either sign) pass.
        let by_five = int_step(0, 5);
        for value in ["0", "5", "-5", "10", "15", " -20 "] {
            assert!(ok(by_five.validate(&value.to_string())), "{value} should align");
        }
        for value in ["3", "7", "-3", "11"] {
            assert!(!ok(by_five.validate(&value.to_string())), "{value} should not align");
        }
        // Non-integers report the shared whole-number message.
        let err = by_five.validate(&"nan".to_string()).unwrap_err();
        assert_eq!(err.message, "Must be a whole number.");
    }

    #[test]
    fn int_step_offset_base_and_sign() {
        // base 1, step 2: odd numbers only. Negative step magnitude is used.
        let odd = int_step(1, -2);
        for value in ["1", "3", "-1", "7"] {
            assert!(ok(odd.validate(&value.to_string())));
        }
        for value in ["0", "2", "-2", "4"] {
            assert!(!ok(odd.validate(&value.to_string())));
        }
    }

    #[test]
    fn int_step_zero_disables_check() {
        let anything = int_step(0, 0);
        for value in ["0", "7", "-123", "999999"] {
            assert!(ok(anything.validate(&value.to_string())));
        }
        // Still requires an integer.
        assert!(!ok(anything.validate(&"abc".to_string())));
    }

    #[test]
    fn int_step_extremes_do_not_overflow() {
        // i128 arithmetic keeps this panic-free at the i64 boundaries.
        let v = int_step(i64::MIN, i64::MAX);
        // i64::MIN is the base, so it aligns (difference zero).
        assert!(ok(v.validate(&i64::MIN.to_string())));
        // i64::MAX differs from the base by 2^64-1, not a multiple of 2^63-1.
        assert!(!ok(v.validate(&i64::MAX.to_string())));
    }

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
            let span = (hi - lo + 1) as u64;
            lo + (self.next_u64() % span) as i64
        }
    }

    #[test]
    fn int_step_multiple_property() {
        // For any base/step, base + k*step always aligns, and adding a nonzero
        // remainder below the step never does.
        let mut rng = SplitMix64(0x5151_2626_3737_4848);
        for _ in 0..2_000 {
            let base = rng.range_i64(-1_000, 1_000);
            let step = rng.range_i64(1, 50);
            let k = rng.range_i64(-1_000, 1_000);
            let aligned = base + k * step;
            let rule = int_step(base, step);
            assert!(
                ok(rule.validate(&aligned.to_string())),
                "base={base} step={step} value={aligned} should align"
            );
            if step > 1 {
                let remainder = rng.range_i64(1, step - 1);
                let misaligned = aligned + remainder;
                assert!(
                    !ok(rule.validate(&misaligned.to_string())),
                    "base={base} step={step} value={misaligned} should not align"
                );
            }
        }
    }
}
