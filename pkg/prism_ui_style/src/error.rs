//! Error type for token resolution and the cascade.
//!
//! Resolution never panics; every failure is surfaced as a [`StyleError`].

use alloc::string::String;
use core::fmt;

/// An error produced while resolving tokens or computing a cascade.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StyleError {
    /// A token name was referenced but is not present in the store.
    UnknownToken(String),
    /// A reference cycle was detected while resolving the named token.
    ///
    /// The contained name is the token at which the cycle closed.
    CycleDetected(String),
    /// A value had a type incompatible with the expected one.
    TypeMismatch {
        /// Human-readable description of the type that was expected.
        expected: &'static str,
        /// Human-readable description of the type that was found.
        found: &'static str,
    },
}

impl fmt::Display for StyleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StyleError::UnknownToken(name) => {
                write!(f, "unknown token: `{name}`")
            }
            StyleError::CycleDetected(name) => {
                write!(f, "token reference cycle detected at `{name}`")
            }
            StyleError::TypeMismatch { expected, found } => {
                write!(f, "type mismatch: expected {expected}, found {found}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StyleError {}
