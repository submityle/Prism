//! The [`ArgList`]: a type-erased argument vector for reflected calls.
//!
//! An `ArgList` collects boxed [`Reflect`] values in positional order and hands
//! the call machinery a borrowed `&[&dyn Reflect]` view. It is a consuming
//! builder: each `push` returns the list so arguments can be chained.

use crate::reflect::Reflect;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A positional list of boxed reflected arguments for a [`DynamicFunction`](crate::DynamicFunction).
///
/// Build one with [`ArgList::new`] and chain [`push`](ArgList::push)/
/// [`push_boxed`](ArgList::push_boxed); the function machinery validates arity
/// and per-argument types before any value is read.
#[derive(Default)]
pub struct ArgList {
    args: Vec<Box<dyn Reflect>>,
}

impl ArgList {
    /// Create an empty argument list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an already-boxed reflected argument, returning the list.
    #[must_use]
    pub fn push_boxed(mut self, value: Box<dyn Reflect>) -> Self {
        self.args.push(value);
        self
    }

    /// Append a concrete reflected argument, returning the list.
    #[must_use]
    pub fn push<T: Reflect>(self, value: T) -> Self {
        self.push_boxed(Box::new(value))
    }

    /// The number of arguments collected so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.args.len()
    }

    /// Whether no arguments have been collected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }

    /// Borrow the argument at `index`, if present.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&dyn Reflect> {
        self.args.get(index).map(|value| &**value)
    }

    /// Iterate over the arguments as `&dyn Reflect`.
    pub fn iter(&self) -> impl Iterator<Item = &dyn Reflect> {
        self.args.iter().map(|value| &**value)
    }

    /// Borrow every argument as a contiguous `&[&dyn Reflect]` for dispatch.
    pub(crate) fn as_refs(&self) -> Vec<&dyn Reflect> {
        self.args.iter().map(|value| &**value).collect()
    }
}
