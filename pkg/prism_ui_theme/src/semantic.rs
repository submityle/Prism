//! The semantic token layer.
//!
//! Components should reference *semantic* names — `color.surface`,
//! `color.text`, `color.border` — rather than raw palette primitives. A
//! [`SemanticToken`] is such a name; a [`SemanticMap`] records, per semantic
//! name, which value to use in each [`ThemeMode`] (with an optional
//! mode-independent default). The value is almost always a
//! [`StyleValue::token`](prism_ui_style::StyleValue::token) reference back into
//! the [`crate::Palette`], which is what decouples component code from concrete
//! colors.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use core::fmt;

use prism_ui_style::StyleValue;

use crate::theme::ThemeMode;

/// A semantic token name, such as `color.surface`.
///
/// This is a light newtype over [`String`] that documents intent at call sites
/// and keeps semantic names distinct from raw palette names in APIs.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemanticToken(String);

impl SemanticToken {
    /// Creates a semantic token from a name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Returns the semantic name as a string slice.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.0
    }

    /// Consumes the token, returning the owned name.
    #[must_use]
    pub fn into_name(self) -> String {
        self.0
    }
}

impl fmt::Display for SemanticToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for SemanticToken {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for SemanticToken {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// The per-mode values recorded for a single semantic token.
#[derive(Clone, Debug, Default, PartialEq)]
struct ModeValues {
    default: Option<StyleValue>,
    by_mode: BTreeMap<ThemeMode, StyleValue>,
}

impl ModeValues {
    fn resolve(&self, mode: &ThemeMode) -> Option<&StyleValue> {
        self.by_mode.get(mode).or(self.default.as_ref())
    }
}

/// A mapping from semantic token names to per-mode values.
///
/// Lookup for a mode falls back to the mode-independent default when that mode
/// has no explicit entry, so a token can be defined once and overridden only
/// where it actually differs.
///
/// # Example
///
/// ```
/// use prism_ui_theme::{SemanticMap, ThemeMode};
/// use prism_ui_style::StyleValue;
///
/// let mut map = SemanticMap::new();
/// // Surface flips between light and dark grays, with a light default.
/// map.set_default("color.surface", StyleValue::token("color.gray.50"));
/// map.set("color.surface", ThemeMode::Dark, StyleValue::token("color.gray.900"));
///
/// assert_eq!(
///     map.resolve_for("color.surface", &ThemeMode::Light),
///     Some(&StyleValue::token("color.gray.50")),
/// );
/// assert_eq!(
///     map.resolve_for("color.surface", &ThemeMode::Dark),
///     Some(&StyleValue::token("color.gray.900")),
/// );
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SemanticMap {
    entries: BTreeMap<String, ModeValues>,
}

impl SemanticMap {
    /// Creates an empty semantic map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Sets the mode-independent default value for a semantic token, returning
    /// `self` for chaining.
    #[must_use]
    pub fn with_default(mut self, name: impl Into<String>, value: StyleValue) -> Self {
        self.set_default(name, value);
        self
    }

    /// Sets the value for a semantic token in a specific mode, returning `self`
    /// for chaining.
    #[must_use]
    pub fn with_mode(
        mut self,
        name: impl Into<String>,
        mode: ThemeMode,
        value: StyleValue,
    ) -> Self {
        self.set(name, mode, value);
        self
    }

    /// Sets the mode-independent default value for a semantic token.
    pub fn set_default(&mut self, name: impl Into<String>, value: StyleValue) {
        self.entries.entry(name.into()).or_default().default = Some(value);
    }

    /// Sets the value for a semantic token in a specific mode.
    pub fn set(&mut self, name: impl Into<String>, mode: ThemeMode, value: StyleValue) {
        self.entries
            .entry(name.into())
            .or_default()
            .by_mode
            .insert(mode, value);
    }

    /// Resolves the value a semantic token takes in `mode`, falling back to its
    /// default. Returns [`None`] if the token is unknown and has no default.
    #[must_use]
    pub fn resolve_for(&self, name: &str, mode: &ThemeMode) -> Option<&StyleValue> {
        self.entries
            .get(name)
            .and_then(|values| values.resolve(mode))
    }

    /// Returns `true` if the map defines a semantic token named `name`.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Iterates over the semantic token names in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Returns the number of semantic tokens defined.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no semantic tokens are defined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_token_round_trips_name() {
        let token = SemanticToken::new("color.text");
        assert_eq!(token.name(), "color.text");
        assert_eq!(token.to_string(), "color.text");
        assert_eq!(token.into_name(), "color.text".to_string());
    }

    #[test]
    fn mode_entry_overrides_default() {
        let map = SemanticMap::new()
            .with_default("color.text", StyleValue::token("color.gray.900"))
            .with_mode(
                "color.text",
                ThemeMode::Dark,
                StyleValue::token("color.gray.50"),
            );

        assert_eq!(
            map.resolve_for("color.text", &ThemeMode::Light),
            Some(&StyleValue::token("color.gray.900")),
        );
        assert_eq!(
            map.resolve_for("color.text", &ThemeMode::Dark),
            Some(&StyleValue::token("color.gray.50")),
        );
    }

    #[test]
    fn unknown_semantic_without_default_is_none() {
        let map = SemanticMap::new();
        assert_eq!(map.resolve_for("color.ghost", &ThemeMode::Light), None);
        assert!(!map.contains("color.ghost"));
    }

    #[test]
    fn custom_mode_falls_back_to_default() {
        let map =
            SemanticMap::new().with_default("color.surface", StyleValue::token("color.gray.50"));
        assert_eq!(
            map.resolve_for("color.surface", &ThemeMode::custom("brand")),
            Some(&StyleValue::token("color.gray.50")),
        );
    }
}
