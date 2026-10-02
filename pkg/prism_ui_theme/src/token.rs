//! Primitive design tokens (the theme *palette*).
//!
//! A [`Palette`] is the mode-independent layer of the pipeline: raw color,
//! spacing and radius primitives such as `color.gray.100` or `space.md`. It is a
//! thin, name-tracking wrapper around [`prism_ui_style::TokenStore`], so token
//! reference chains and cycle detection are reused rather than reimplemented —
//! resolution is delegated straight to the style crate.
//!
//! Semantic indirection (`color.surface` -> `color.gray.100`) lives one layer up
//! in [`crate::semantic`]; compilation that folds both layers into concrete
//! values lives in [`crate::theme`].

use alloc::collections::BTreeSet;
use alloc::string::String;

use prism_ui_style::{StyleError, StyleValue, TokenStore};

/// The mode-independent palette of primitive design tokens.
///
/// Unlike the bare [`TokenStore`] it wraps, a `Palette` remembers the set of
/// token names it owns, which lets the compiler enumerate every primitive when
/// folding a theme (a [`TokenStore`] alone exposes lookup but not iteration).
///
/// # Example
///
/// ```
/// use prism_ui_theme::Palette;
/// use prism_ui_style::StyleValue;
///
/// let palette = Palette::new()
///     .with("color.gray.100", StyleValue::rgba8(243, 244, 246, 255))
///     .with("color.action", StyleValue::token("color.gray.100"));
///
/// // References resolve through the underlying store.
/// assert_eq!(
///     palette.resolve("color.action").unwrap(),
///     StyleValue::rgba8(243, 244, 246, 255),
/// );
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Palette {
    store: TokenStore,
    names: BTreeSet<String>,
}

impl Palette {
    /// Creates an empty palette.
    #[must_use]
    pub fn new() -> Self {
        Self {
            store: TokenStore::new(),
            names: BTreeSet::new(),
        }
    }

    /// Inserts or replaces a primitive token, returning `self` for chaining.
    #[must_use]
    pub fn with(mut self, name: impl Into<String>, value: StyleValue) -> Self {
        self.insert(name, value);
        self
    }

    /// Inserts or replaces a primitive token by name.
    pub fn insert(&mut self, name: impl Into<String>, value: StyleValue) {
        let name = name.into();
        self.names.insert(name.clone());
        self.store.insert(name, value);
    }

    /// Returns the underlying [`TokenStore`] used for resolution.
    #[must_use]
    pub fn store(&self) -> &TokenStore {
        &self.store
    }

    /// Returns the raw (unresolved) value stored for `name`, if present.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&StyleValue> {
        self.store.get(name)
    }

    /// Returns `true` if the palette owns a token named `name`.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    /// Iterates over the palette's token names in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(String::as_str)
    }

    /// Fully resolves `name` to a literal value, following reference chains.
    ///
    /// # Errors
    ///
    /// Propagates [`StyleError::UnknownToken`] for a missing name and
    /// [`StyleError::CycleDetected`] when a reference chain revisits a name.
    pub fn resolve(&self, name: &str) -> Result<StyleValue, StyleError> {
        self.store.resolve(name)
    }

    /// Returns the number of tokens in the palette.
    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Returns `true` if the palette holds no tokens.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec::Vec;

    #[test]
    fn insert_tracks_names_and_values() {
        let mut palette = Palette::new();
        palette.insert("space.md", StyleValue::px(16.0));
        assert!(palette.contains("space.md"));
        assert_eq!(palette.len(), 1);
        assert_eq!(palette.get("space.md"), Some(&StyleValue::px(16.0)));
    }

    #[test]
    fn names_are_sorted_and_complete() {
        let palette = Palette::new()
            .with("b", StyleValue::number(1.0))
            .with("a", StyleValue::number(2.0));
        let names: Vec<&str> = palette.names().collect();
        assert_eq!(names, alloc::vec!["a", "b"]);
    }

    #[test]
    fn resolve_follows_reference_chain() {
        let palette = Palette::new()
            .with("color.base", StyleValue::rgba8(10, 20, 30, 255))
            .with("color.alias", StyleValue::token("color.base"));
        assert_eq!(
            palette.resolve("color.alias").unwrap(),
            StyleValue::rgba8(10, 20, 30, 255),
        );
    }

    #[test]
    fn resolve_reports_cycles() {
        let palette = Palette::new()
            .with("a", StyleValue::token("b"))
            .with("b", StyleValue::token("a"));
        assert_eq!(
            palette.resolve("a"),
            Err(StyleError::CycleDetected("a".to_string())),
        );
    }

    #[test]
    fn resolve_reports_unknown_tokens() {
        let palette = Palette::new();
        assert_eq!(
            palette.resolve("missing"),
            Err(StyleError::UnknownToken("missing".to_string())),
        );
    }
}
