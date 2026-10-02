//! Typed extraction of captured route parameters.
//!
//! A [`RouteMatch`](crate::RouteMatch) stores its captures as raw string
//! slices. Real handlers usually want them as integers, booleans or owned
//! strings, with a graceful fallback when a `:param` cannot be parsed (for
//! example `/users/:id` reached with a non-numeric `id`).
//!
//! This module provides a tiny [`FromParam`] conversion trait, blanket-free
//! implementations for the common primitive types, and the
//! [`parse_param`]/[`parse_param_or`] helpers that read a capture and parse it
//! in one call. Parsing never panics: a malformed value yields `None` (or the
//! supplied fallback) rather than aborting.

use alloc::string::{String, ToString};

use crate::route::RouteMatch;

/// A type that can be parsed from a raw captured route parameter.
///
/// Implementations return `None` when the raw text is not a valid value, so
/// callers can fall back gracefully instead of panicking.
pub trait FromParam: Sized {
    /// Parses `raw` into `Self`, or returns `None` when it is malformed.
    fn from_param(raw: &str) -> Option<Self>;
}

macro_rules! impl_from_param_via_parse {
    ($($ty:ty),* $(,)?) => {
        $(
            impl FromParam for $ty {
                fn from_param(raw: &str) -> Option<Self> {
                    raw.parse::<$ty>().ok()
                }
            }
        )*
    };
}

impl_from_param_via_parse!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, bool);

impl FromParam for String {
    fn from_param(raw: &str) -> Option<Self> {
        Some(raw.to_string())
    }
}

/// Reads parameter `name` from `matched` and parses it into `T`.
///
/// Returns `None` when the parameter is absent or cannot be parsed into `T`.
#[must_use]
pub fn parse_param<T: FromParam>(matched: &RouteMatch, name: &str) -> Option<T> {
    matched.param(name).and_then(T::from_param)
}

/// Reads parameter `name` and parses it into `T`, falling back to `fallback`
/// when the parameter is missing or malformed.
#[must_use]
pub fn parse_param_or<T: FromParam>(matched: &RouteMatch, name: &str, fallback: T) -> T {
    parse_param(matched, name).unwrap_or(fallback)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and reuse its String"
    )]

    use super::{parse_param, parse_param_or, FromParam};
    use crate::path::Location;
    use crate::route::RoutePattern;
    use alloc::string::String;

    fn match_users(path: &str) -> crate::route::RouteMatch {
        RoutePattern::new("/users/:id")
            .match_path(&Location::new(path))
            .expect("pattern should match")
    }

    #[test]
    fn parses_integer_param() {
        let matched = match_users("/users/42");
        assert_eq!(parse_param::<i64>(&matched, "id"), Some(42));
        assert_eq!(parse_param::<u32>(&matched, "id"), Some(42));
    }

    #[test]
    fn malformed_integer_param_is_none() {
        let matched = match_users("/users/abc");
        assert_eq!(parse_param::<i64>(&matched, "id"), None);
    }

    #[test]
    fn malformed_integer_falls_back() {
        let matched = match_users("/users/abc");
        assert_eq!(parse_param_or::<i64>(&matched, "id", -1), -1);
    }

    #[test]
    fn valid_param_ignores_fallback() {
        let matched = match_users("/users/7");
        assert_eq!(parse_param_or::<i64>(&matched, "id", -1), 7);
    }

    #[test]
    fn missing_param_is_none_and_uses_fallback() {
        let matched = match_users("/users/7");
        assert_eq!(parse_param::<i64>(&matched, "missing"), None);
        assert_eq!(parse_param_or::<i64>(&matched, "missing", 99), 99);
    }

    #[test]
    fn string_param_is_always_captured() {
        let matched = match_users("/users/ada");
        assert_eq!(
            parse_param::<String>(&matched, "id").as_deref(),
            Some("ada")
        );
    }

    #[test]
    fn bool_param_parses() {
        let matched = RoutePattern::new("/flag/:on")
            .match_path(&Location::new("/flag/true"))
            .expect("match");
        assert_eq!(parse_param::<bool>(&matched, "on"), Some(true));
        let matched = RoutePattern::new("/flag/:on")
            .match_path(&Location::new("/flag/nope"))
            .expect("match");
        assert_eq!(parse_param::<bool>(&matched, "on"), None);
    }

    #[test]
    fn negative_and_unsigned_edge_cases() {
        let matched = RoutePattern::new("/n/:v")
            .match_path(&Location::new("/n/-5"))
            .expect("match");
        assert_eq!(parse_param::<i32>(&matched, "v"), Some(-5));
        // A negative value is not a valid u32.
        assert_eq!(parse_param::<u32>(&matched, "v"), None);
    }

    #[test]
    fn from_param_trait_direct() {
        assert_eq!(<i64 as FromParam>::from_param("123"), Some(123));
        assert_eq!(<i64 as FromParam>::from_param(""), None);
    }
}
