//! Scoped dependency injection via a [`ContextMap`].
//!
//! A [`ContextMap`] is a type-keyed bag of shared values: each stored value is
//! identified by its [`TypeId`], so at most one value of a given type lives in
//! a map at once. Values are held behind [`Rc`] so injection hands out cheap
//! shared handles rather than clones of the value itself.
//!
//! # Scope inheritance
//!
//! [`ContextMap::child`] produces a new map that starts as a clone of its
//! parent. Because the stored values are reference-counted, the child shares
//! the *same* underlying values until it overrides one. Calling
//! [`ContextMap::provide`] on the child replaces only the child's entry for
//! that type; the parent is untouched. This gives lexical, override-local
//! scoping: children inherit everything a parent provides but may shadow any
//! entry without leaking the change back up.

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use core::any::{Any, TypeId};

/// A type-keyed, reference-counted bag of injectable values.
///
/// See the [module docs](crate::context) for the inheritance semantics of
/// [`ContextMap::child`].
#[derive(Clone, Default)]
pub struct ContextMap {
    entries: BTreeMap<TypeId, Rc<dyn Any>>,
}

impl ContextMap {
    /// Creates an empty context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Provides `value`, replacing any existing entry of the same type `T`.
    pub fn provide<T: 'static>(&mut self, value: T) {
        self.entries.insert(TypeId::of::<T>(), Rc::new(value));
    }

    /// Injects the shared value of type `T`, or `None` if none was provided.
    ///
    /// The returned [`Rc`] shares ownership with every other handle to the same
    /// provided value.
    #[must_use]
    pub fn inject<T: 'static>(&self) -> Option<Rc<T>> {
        self.entries
            .get(&TypeId::of::<T>())
            .cloned()
            .and_then(|value| value.downcast::<T>().ok())
    }

    /// Returns `true` if a value of type `T` is currently provided.
    #[must_use]
    pub fn contains<T: 'static>(&self) -> bool {
        self.entries.contains_key(&TypeId::of::<T>())
    }

    /// Creates a child scope that inherits this map's entries.
    ///
    /// The child starts as a clone of `self`; subsequent calls to
    /// [`ContextMap::provide`] on the child shadow entries locally without
    /// mutating this parent map. See the [module docs](crate::context).
    #[must_use]
    pub fn child(&self) -> ContextMap {
        self.clone()
    }
}

impl core::fmt::Debug for ContextMap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ContextMap")
            .field("len", &self.entries.len())
            .finish()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::std_instead_of_alloc, reason = "tests run under std")]

    use super::*;

    #[derive(Debug, PartialEq)]
    struct Theme(&'static str);

    #[derive(Debug, PartialEq)]
    struct Locale(&'static str);

    #[test]
    fn provide_inject_roundtrips() {
        let mut ctx = ContextMap::new();
        ctx.provide(Theme("dark"));

        let theme = ctx.inject::<Theme>().expect("theme present");
        assert_eq!(*theme, Theme("dark"));
    }

    #[test]
    fn inject_absent_type_is_none() {
        let ctx = ContextMap::new();
        assert!(ctx.inject::<Theme>().is_none());
    }

    #[test]
    fn distinct_types_coexist() {
        let mut ctx = ContextMap::new();
        ctx.provide(Theme("dark"));
        ctx.provide(Locale("en"));

        assert_eq!(*ctx.inject::<Theme>().unwrap(), Theme("dark"));
        assert_eq!(*ctx.inject::<Locale>().unwrap(), Locale("en"));
    }

    #[test]
    fn child_inherits_parent_values() {
        let mut parent = ContextMap::new();
        parent.provide(Theme("dark"));

        let child = parent.child();
        assert_eq!(*child.inject::<Theme>().unwrap(), Theme("dark"));
    }

    #[test]
    fn child_override_does_not_mutate_parent() {
        let mut parent = ContextMap::new();
        parent.provide(Theme("dark"));

        let mut child = parent.child();
        child.provide(Theme("light"));

        assert_eq!(*child.inject::<Theme>().unwrap(), Theme("light"));
        assert_eq!(*parent.inject::<Theme>().unwrap(), Theme("dark"));
    }
}
