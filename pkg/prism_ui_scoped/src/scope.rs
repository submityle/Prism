//! Component scope identity and the scoping engine.
//!
//! A [`ScopeId`] is a small, stable, collision-resistant identity for one
//! component instance (think Vue's `data-v-xxxxxxxx` attribute or the hashed
//! suffix a CSS-Modules compiler appends to a class name). A [`Scope`] couples
//! that identity with the set of *local* class names a component owns, so the
//! engine can rewrite exactly those names — and nothing else — both inside a
//! [`StyleSheet`] and across a [`prism_ui::Element`] subtree.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use prism_ui::{Element, ElementKind, Key};
use prism_ui_style::{Class, StyleSheet};

use crate::scoped_sheet::ScopedSheet;

/// The 64-bit [`Fowler-Noll-Vo`](https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function)
/// offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// The `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Hashes a byte string with the `FNV-1a` algorithm.
///
/// `FNV-1a` uses only wrapping integer multiplies and XORs, so it stays well
/// inside the crate's "no transcendental float math" budget and is identical
/// on every target.
#[must_use]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// A stable, unique identity for one component scope.
///
/// Two scopes built from the same component name share an id; a monotonically
/// increasing counter can be mixed in (via [`ScopeId::from_name_indexed`] or
/// [`ScopeAllocator`]) when several instances of the same component must not
/// collide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScopeId(u64);

impl ScopeId {
    /// Wraps a raw hash value as a [`ScopeId`].
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Derives a [`ScopeId`] from a component name by hashing it.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        Self(fnv1a(name.as_bytes()))
    }

    /// Derives a [`ScopeId`] from a component name mixed with an instance
    /// index, so repeated instances of one component get distinct ids.
    #[must_use]
    pub fn from_name_indexed(name: &str, index: u64) -> Self {
        let mut hash = fnv1a(name.as_bytes());
        // Fold the index in with the same wrapping-multiply mixing step.
        for byte in index.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        Self(hash)
    }

    /// Returns the raw 64-bit hash backing this id.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Returns the short, stable textual suffix for this scope.
    ///
    /// The suffix is the low 32 bits of the hash as zero-padded lowercase hex,
    /// mirroring the eight-hex-digit tag a CSS-Modules or Vue toolchain emits.
    #[must_use]
    pub fn suffix(self) -> String {
        format!("{:08x}", self.0 as u32)
    }

    /// Rewrites a local class name into its fully scoped form.
    ///
    /// The scheme is `"{local}__{suffix}"`, which keeps the human-readable
    /// stem for debugging while guaranteeing uniqueness across scopes.
    #[must_use]
    pub fn scoped_name(self, local: &str) -> String {
        format!("{local}__{}", self.suffix())
    }
}

/// Hands out successive, distinct [`ScopeId`]s for repeated component mounts.
///
/// The allocator seeds its counter from the component family name so ids are
/// both stable per family and unique per instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopeAllocator {
    family: String,
    next: u64,
}

impl ScopeAllocator {
    /// Creates an allocator for the given component family name.
    #[must_use]
    pub fn new(family: impl Into<String>) -> Self {
        Self {
            family: family.into(),
            next: 0,
        }
    }

    /// Returns the next distinct [`ScopeId`] and advances the counter.
    pub fn allocate(&mut self) -> ScopeId {
        let id = ScopeId::from_name_indexed(&self.family, self.next);
        self.next += 1;
        id
    }

    /// Returns how many ids have been allocated so far.
    #[must_use]
    pub fn allocated(&self) -> u64 {
        self.next
    }
}

/// A component scope: an identity plus the set of local class names it owns.
///
/// Only names registered on the scope are ever rewritten; every other class
/// (global utilities, names owned by a parent, unknown names) is passed through
/// untouched. This mirrors the Vue/CSS-Modules contract where `scoped` only
/// localises the component's own selectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    id: ScopeId,
    locals: BTreeSet<String>,
}

impl Scope {
    /// Creates an empty scope with an explicit id.
    #[must_use]
    pub fn new(id: ScopeId) -> Self {
        Self {
            id,
            locals: BTreeSet::new(),
        }
    }

    /// Creates an empty scope whose id is derived from a component name.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        Self::new(ScopeId::from_name(name))
    }

    /// Returns this scope's identity.
    #[must_use]
    pub fn id(&self) -> ScopeId {
        self.id
    }

    /// Registers a local class name in place.
    pub fn register(&mut self, name: impl Into<String>) {
        self.locals.insert(name.into());
    }

    /// Builder form of [`Scope::register`].
    #[must_use]
    pub fn with_local(mut self, name: impl Into<String>) -> Self {
        self.register(name);
        self
    }

    /// Registers every name yielded by an iterator.
    pub fn register_all<I, S>(&mut self, names: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for name in names {
            self.register(name);
        }
    }

    /// Returns `true` if `name` is a registered local class of this scope.
    #[must_use]
    pub fn owns(&self, name: &str) -> bool {
        self.locals.contains(name)
    }

    /// Returns the registered local class names in sorted order.
    #[must_use]
    pub fn locals(&self) -> Vec<&str> {
        self.locals.iter().map(String::as_str).collect()
    }

    /// Returns the scoped form of a local name, or `None` if the name is not
    /// owned by this scope.
    #[must_use]
    pub fn scoped_name(&self, local: &str) -> Option<String> {
        if self.owns(local) {
            Some(self.id.scoped_name(local))
        } else {
            None
        }
    }

    /// Rewrites a style sheet into a [`ScopedSheet`].
    ///
    /// Every registered local class that exists in `sheet` is cloned, renamed
    /// to its scoped form and inserted into the result. Registered names that
    /// are absent from `sheet` are still recorded in the name map (so element
    /// rewriting stays consistent) but contribute no class body.
    ///
    /// Because [`StyleSheet`] exposes no iterator, classes that are present in
    /// `sheet` but *not* registered on the scope are intentionally left out of
    /// the scoped sheet: the scope only localises what it owns.
    #[must_use]
    pub fn scope(&self, sheet: &StyleSheet) -> ScopedSheet {
        let mut scoped = StyleSheet::new();
        let mut forward: BTreeMap<String, String> = BTreeMap::new();
        let mut reverse: BTreeMap<String, String> = BTreeMap::new();

        for local in &self.locals {
            let scoped_name = self.id.scoped_name(local);
            forward.insert(local.clone(), scoped_name.clone());
            reverse.insert(scoped_name.clone(), local.clone());

            if let Some(class) = sheet.get(local) {
                let mut renamed = class.clone();
                renamed.name = scoped_name;
                scoped.insert(renamed);
            }
        }

        ScopedSheet::from_parts(self.id, scoped, forward, reverse)
    }

    /// Convenience: register the names of, and scope, a list of classes.
    ///
    /// The scope is *not* mutated; a fresh scope that owns exactly `classes`
    /// is used so the call is self-contained.
    #[must_use]
    pub fn scope_classes<I: IntoIterator<Item = Class>>(&self, classes: I) -> ScopedSheet {
        let mut sheet = StyleSheet::new();
        let mut scope = Scope::new(self.id);
        for class in classes {
            scope.register(class.name.clone());
            sheet.insert(class);
        }
        scope.scope(&sheet)
    }

    /// Rewrites an [`Element`] subtree, replacing owned local class names with
    /// their scoped forms and leaving every other class untouched.
    ///
    /// The element's kind, text, inline overrides, explicit key and child
    /// order are preserved exactly; only class names change.
    #[must_use]
    pub fn apply(&self, element: &Element) -> Element {
        self.rewrite(element)
    }

    /// Recursive worker for [`Scope::apply`].
    fn rewrite(&self, element: &Element) -> Element {
        let mut out = match element.kind() {
            ElementKind::Box => Element::box_(),
            ElementKind::Text => Element::text(element.text_content().unwrap_or_default()),
            ElementKind::Custom(name) => Element::custom(name.clone()),
        };

        if let Some(key) = element.explicit_key() {
            out = match key {
                Key::Int(value) => out.key_int(*value),
                Key::Str(value) => out.key_str(value.clone()),
                // Positional keys are assigned by the runtime, not re-emitted.
                Key::Index(_) => out,
            };
        }

        for name in element.class_names() {
            match self.scoped_name(name) {
                Some(scoped) => out = out.class(scoped),
                None => out = out.class(name.to_string()),
            }
        }

        for (prop, value) in element.inline_pairs() {
            out = out.style(*prop, value.clone());
        }

        for child in element.child_elements() {
            out = out.child(self.rewrite(child));
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{StyleProp, StyleValue};

    #[test]
    fn scope_id_is_deterministic_and_distinct() {
        assert_eq!(ScopeId::from_name("Button"), ScopeId::from_name("Button"));
        assert_ne!(ScopeId::from_name("Button"), ScopeId::from_name("Card"));
    }

    #[test]
    fn suffix_is_eight_hex_digits() {
        let suffix = ScopeId::from_name("Button").suffix();
        assert_eq!(suffix.len(), 8);
        assert!(suffix.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn indexed_ids_differ_per_instance() {
        let a = ScopeId::from_name_indexed("Row", 0);
        let b = ScopeId::from_name_indexed("Row", 1);
        assert_ne!(a, b);
    }

    #[test]
    fn allocator_hands_out_distinct_ids() {
        let mut alloc = ScopeAllocator::new("Item");
        let first = alloc.allocate();
        let second = alloc.allocate();
        assert_ne!(first, second);
        assert_eq!(alloc.allocated(), 2);
    }

    #[test]
    fn scoped_name_only_for_owned_locals() {
        let scope = Scope::from_name("Card").with_local("title");
        assert!(scope.owns("title"));
        assert!(scope.scoped_name("title").is_some());
        assert_eq!(scope.scoped_name("global"), None);
    }

    #[test]
    fn scope_renames_registered_classes_and_builds_maps() {
        let sheet = StyleSheet::new()
            .with_class(Class::new("title").with(StyleProp::FontSize, StyleValue::px(18.0)))
            .with_class(Class::new("body").with(StyleProp::Opacity, StyleValue::number(1.0)));

        let scope = Scope::from_name("Card")
            .with_local("title")
            .with_local("body");
        let scoped = scope.scope(&sheet);

        assert_eq!(scoped.len(), 2);
        let title_scoped = scoped.scoped_name("title").unwrap();
        assert!(title_scoped.starts_with("title__"));
        assert_eq!(scoped.local_name(title_scoped), Some("title"));
        // The renamed class carries its original body.
        let class = scoped.sheet().get(title_scoped).unwrap();
        assert_eq!(
            class.base.get(&StyleProp::FontSize),
            Some(&StyleValue::px(18.0))
        );
    }

    #[test]
    fn registered_name_absent_from_sheet_still_maps() {
        let sheet = StyleSheet::new();
        let scope = Scope::from_name("Card").with_local("ghost");
        let scoped = scope.scope(&sheet);
        // Name is mapped even though no class body exists for it.
        assert!(scoped.contains_local("ghost"));
        assert_eq!(scoped.sheet().len(), 0);
    }

    #[test]
    fn apply_rewrites_only_owned_classes() {
        let scope = Scope::from_name("Card").with_local("title");
        let view = Element::box_()
            .class("title")
            .class("global-util")
            .child(Element::text("hi").class("title"));
        let out = scope.apply(&view);

        let title_scoped = scope.scoped_name("title").unwrap();
        assert_eq!(
            out.class_names(),
            &[title_scoped.clone(), "global-util".to_string()]
        );
        // Nested references are rewritten identically.
        assert_eq!(out.child_elements()[0].class_names(), &[title_scoped]);
    }

    #[test]
    fn apply_preserves_structure() {
        let scope = Scope::from_name("Card").with_local("x");
        let view = Element::custom("widget")
            .key_str("k")
            .class("x")
            .style(StyleProp::Width, StyleValue::px(10.0))
            .child(Element::text("content").key_int(7));
        let out = scope.apply(&view);

        assert_eq!(out.kind(), &ElementKind::Custom("widget".into()));
        assert_eq!(out.explicit_key(), Some(&Key::Str("k".into())));
        assert_eq!(
            out.inline_pairs(),
            &[(StyleProp::Width, StyleValue::px(10.0))]
        );
        let child = &out.child_elements()[0];
        assert_eq!(child.text_content(), Some("content"));
        assert_eq!(child.explicit_key(), Some(&Key::Int(7)));
    }

    #[test]
    fn scope_classes_registers_and_renames() {
        let scope = Scope::new(ScopeId::from_name("List"));
        let scoped =
            scope.scope_classes([Class::new("row").with(StyleProp::Height, StyleValue::px(32.0))]);
        assert_eq!(scoped.len(), 1);
        assert!(scoped.scoped_name("row").is_some());
    }
}
