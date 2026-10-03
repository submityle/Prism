//! Typed shader definitions (`shader-defs`) and the permutation they form.
//!
//! A *shader def* is a named compile-time value that selects a code path in a
//! shader: a boolean feature flag or a small integer parameter (quality tier,
//! light count, …). A set of defs is the *permutation* that specialises one
//! authored source into one concrete variant. Keeping defs typed (rather than
//! raw text macros) lets the `#if` evaluator reason about them numerically and
//! lets the permutation id be computed deterministically.

use alloc::collections::BTreeMap;
use alloc::string::String;

/// The value bound to a shader def.
///
/// Values are small and `Copy`. Booleans and integers unify into a single
/// signed numeric domain for the `#if` evaluator: a boolean is `1`/`0` and an
/// unsigned value is widened, so every def has a well-defined truthiness and
/// comparison behaviour.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ShaderDefValue {
    /// A boolean feature flag.
    Bool(bool),
    /// A signed integer parameter.
    Int(i32),
    /// An unsigned integer parameter.
    UInt(u32),
}

impl ShaderDefValue {
    /// The value as a signed 64-bit integer for use by the `#if` evaluator.
    ///
    /// `Bool(true)` is `1`, `Bool(false)` is `0`, and integer values widen
    /// losslessly.
    #[must_use]
    pub const fn as_i64(self) -> i64 {
        match self {
            Self::Bool(true) => 1,
            Self::Bool(false) => 0,
            Self::Int(value) => value as i64,
            Self::UInt(value) => value as i64,
        }
    }

    /// Whether the value is considered "true" by `#ifdef`-style truthiness.
    ///
    /// Any non-zero numeric value (and `Bool(true)`) is truthy.
    #[must_use]
    pub const fn is_truthy(self) -> bool {
        self.as_i64() != 0
    }
}

/// An ordered, deduplicated set of shader defs.
///
/// Backed by a [`BTreeMap`] so iteration is always in canonical (name-sorted)
/// order. That canonical order is what makes the derived permutation id stable:
/// the same logical set of defs always hashes identically regardless of the
/// order in which they were inserted.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct ShaderDefs {
    entries: BTreeMap<String, ShaderDefValue>,
}

impl ShaderDefs {
    /// Creates an empty def set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or overwrites a def, returning the previous value if any.
    pub fn insert(&mut self, name: impl Into<String>, value: ShaderDefValue) -> Option<ShaderDefValue> {
        self.entries.insert(name.into(), value)
    }

    /// Inserts a boolean flag set to `true` (the common `#define FEATURE` case).
    pub fn define(&mut self, name: impl Into<String>) -> Option<ShaderDefValue> {
        self.insert(name, ShaderDefValue::Bool(true))
    }

    /// Removes a def, returning its value if it was present.
    pub fn remove(&mut self, name: &str) -> Option<ShaderDefValue> {
        self.entries.remove(name)
    }

    /// Whether a def with this name exists (regardless of its value).
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// The value bound to `name`, if any.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<ShaderDefValue> {
        self.entries.get(name).copied()
    }

    /// The number of defs in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the set has no defs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterates defs in canonical name-sorted order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, ShaderDefValue)> {
        self.entries.iter().map(|(name, value)| (name.as_str(), *value))
    }
}
