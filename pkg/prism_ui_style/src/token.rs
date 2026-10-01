//! Design tokens and the [`TokenStore`] that resolves them.
//!
//! A design token is a named [`StyleValue`]. Tokens may reference other tokens
//! via [`StyleValue::TokenRef`], forming chains that are resolved on demand.
//! Resolution detects reference cycles and missing tokens and reports them as a
//! [`StyleError`] instead of panicking.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};

use crate::error::StyleError;
use crate::value::StyleValue;

/// A single named design token.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignToken {
    /// The token's unique name (for example `color.primary`).
    pub name: String,
    /// The token's value, which may itself be a [`StyleValue::TokenRef`].
    pub value: StyleValue,
}

impl DesignToken {
    /// Creates a new design token.
    #[must_use]
    pub fn new(name: impl Into<String>, value: StyleValue) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

/// A store of design tokens keyed by name.
///
/// Use [`TokenStore::resolve`] to fully resolve a token (following reference
/// chains) into a literal value, or [`TokenStore::resolve_value`] to resolve an
/// arbitrary [`StyleValue`] that may or may not be a reference.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TokenStore {
    tokens: BTreeMap<String, StyleValue>,
}

impl TokenStore {
    /// Creates an empty token store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tokens: BTreeMap::new(),
        }
    }

    /// Inserts or replaces a token by name, returning the store for chaining.
    #[must_use]
    pub fn with(mut self, name: impl Into<String>, value: StyleValue) -> Self {
        self.insert(name, value);
        self
    }

    /// Inserts or replaces a token by name.
    pub fn insert(&mut self, name: impl Into<String>, value: StyleValue) {
        self.tokens.insert(name.into(), value);
    }

    /// Inserts or replaces a [`DesignToken`].
    pub fn insert_token(&mut self, token: DesignToken) {
        self.tokens.insert(token.name, token.value);
    }

    /// Returns the raw (unresolved) value stored for `name`, if present.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&StyleValue> {
        self.tokens.get(name)
    }

    /// Returns the number of tokens in the store.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns `true` if the store contains no tokens.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Fully resolves the token named `name` into a literal value.
    ///
    /// Reference chains are followed until a non-reference value is reached.
    ///
    /// # Errors
    ///
    /// Returns [`StyleError::UnknownToken`] if a referenced name is absent, or
    /// [`StyleError::CycleDetected`] if the chain revisits a name.
    pub fn resolve(&self, name: &str) -> Result<StyleValue, StyleError> {
        let mut seen = BTreeSet::new();
        self.resolve_inner(name, &mut seen)
    }

    /// Resolves an arbitrary value, following it if it is a token reference.
    ///
    /// Non-reference values are returned unchanged (as a clone).
    ///
    /// # Errors
    ///
    /// Propagates any error from [`TokenStore::resolve`] when `value` is a
    /// [`StyleValue::TokenRef`].
    pub fn resolve_value(&self, value: &StyleValue) -> Result<StyleValue, StyleError> {
        match value {
            StyleValue::TokenRef(name) => self.resolve(name),
            other => Ok(other.clone()),
        }
    }

    fn resolve_inner(
        &self,
        name: &str,
        seen: &mut BTreeSet<String>,
    ) -> Result<StyleValue, StyleError> {
        if !seen.insert(name.to_string()) {
            return Err(StyleError::CycleDetected(name.to_string()));
        }
        let value = self
            .tokens
            .get(name)
            .ok_or_else(|| StyleError::UnknownToken(name.to_string()))?;
        match value {
            StyleValue::TokenRef(next) => self.resolve_inner(next, seen),
            other => Ok(other.clone()),
        }
    }
}
