//! The output of scoping a style sheet.
//!
//! A [`ScopedSheet`] bundles the rewritten [`StyleSheet`] with the bidirectional
//! map between local and scoped class names, so callers can look a name up in
//! either direction and still drive the ordinary `prism_ui_style` cascade.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use prism_ui_style::StyleSheet;

use crate::scope::ScopeId;

/// A style sheet whose owned classes have been renamed into a component scope.
#[derive(Clone, Debug, PartialEq)]
pub struct ScopedSheet {
    id: ScopeId,
    sheet: StyleSheet,
    forward: BTreeMap<String, String>,
    reverse: BTreeMap<String, String>,
}

impl ScopedSheet {
    /// Builds a [`ScopedSheet`] from its already-computed parts.
    ///
    /// This is the constructor [`crate::scope::Scope::scope`] uses; the maps
    /// are expected to be mutually consistent (`forward` and `reverse` are
    /// inverses of each other).
    #[must_use]
    pub(crate) fn from_parts(
        id: ScopeId,
        sheet: StyleSheet,
        forward: BTreeMap<String, String>,
        reverse: BTreeMap<String, String>,
    ) -> Self {
        Self {
            id,
            sheet,
            forward,
            reverse,
        }
    }

    /// Returns the scope identity this sheet was rewritten for.
    #[must_use]
    pub fn id(&self) -> ScopeId {
        self.id
    }

    /// Returns the rewritten style sheet, ready for the cascade.
    #[must_use]
    pub fn sheet(&self) -> &StyleSheet {
        &self.sheet
    }

    /// Maps a local class name to its scoped form.
    #[must_use]
    pub fn scoped_name(&self, local: &str) -> Option<&str> {
        self.forward.get(local).map(String::as_str)
    }

    /// Maps a scoped class name back to its original local name.
    #[must_use]
    pub fn local_name(&self, scoped: &str) -> Option<&str> {
        self.reverse.get(scoped).map(String::as_str)
    }

    /// Returns `true` if `local` was registered in this scope.
    #[must_use]
    pub fn contains_local(&self, local: &str) -> bool {
        self.forward.contains_key(local)
    }

    /// Returns the number of local-to-scoped name mappings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.forward.len()
    }

    /// Returns `true` if the scope owns no local class names.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }

    /// Iterates over `(local, scoped)` name pairs in sorted local-name order.
    pub fn mappings(&self) -> impl Iterator<Item = (&str, &str)> {
        self.forward
            .iter()
            .map(|(local, scoped)| (local.as_str(), scoped.as_str()))
    }

    /// Returns every scoped class name, in sorted order.
    #[must_use]
    pub fn scoped_names(&self) -> Vec<&str> {
        self.reverse.keys().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use prism_ui_style::{Class, StyleProp, StyleSheet, StyleValue};

    use crate::scope::Scope;

    #[test]
    fn mappings_round_trip_in_both_directions() {
        let sheet = StyleSheet::new()
            .with_class(Class::new("a").with(StyleProp::Width, StyleValue::px(1.0)))
            .with_class(Class::new("b").with(StyleProp::Height, StyleValue::px(2.0)));
        let scoped = Scope::from_name("C")
            .with_local("a")
            .with_local("b")
            .scope(&sheet);

        assert!(!scoped.is_empty());
        assert_eq!(scoped.len(), 2);
        assert_eq!(scoped.scoped_names().len(), 2);

        for (local, scoped_name) in scoped.mappings() {
            assert_eq!(scoped.scoped_name(local), Some(scoped_name));
            assert_eq!(scoped.local_name(scoped_name), Some(local));
        }
    }

    #[test]
    fn id_is_exposed() {
        let scoped = Scope::from_name("C")
            .with_local("a")
            .scope(&StyleSheet::new());
        assert_eq!(scoped.id(), Scope::from_name("C").id());
    }
}
