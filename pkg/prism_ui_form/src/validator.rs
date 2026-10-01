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
}
