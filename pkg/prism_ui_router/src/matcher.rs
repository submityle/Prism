//! An ordered table of routes and first-match resolution.
//!
//! A [`RouteTable`] stores `(RoutePattern, RouteId)` entries in insertion
//! order. [`RouteTable::resolve`] walks them top to bottom and returns the
//! first pattern that matches, so earlier entries take precedence over later
//! ones. Specificity is therefore expressed purely by insertion order.

use alloc::vec::Vec;

use crate::path::Location;
use crate::route::{RouteMatch, RoutePattern};

/// A lightweight identifier associated with a route.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RouteId(u32);

impl RouteId {
    /// Creates a [`RouteId`] from a raw `u32`.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns the raw `u32` value of this [`RouteId`].
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for RouteId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

/// An ordered collection of routes resolved on a first-match-wins basis.
///
/// Build one fluently with [`RouteTable::new`] and [`RouteTable::route`], then
/// call [`RouteTable::resolve`] with a [`Location`] to find the matching route.
#[derive(Clone, Debug, Default)]
pub struct RouteTable {
    entries: Vec<(RoutePattern, RouteId)>,
}

impl RouteTable {
    /// Creates an empty [`RouteTable`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Appends a route compiled from `pattern` with identifier `id`.
    ///
    /// Returns `self` so calls can be chained. Routes added earlier take
    /// precedence during [`RouteTable::resolve`].
    #[must_use]
    pub fn route(mut self, pattern: &str, id: RouteId) -> Self {
        self.entries.push((RoutePattern::new(pattern), id));
        self
    }

    /// Returns the first route that matches `location`, with its match data.
    ///
    /// Entries are tried in insertion order; the first successful match wins.
    /// Returns `None` when no route matches.
    #[must_use]
    pub fn resolve(&self, location: &Location) -> Option<(RouteId, RouteMatch)> {
        for (pattern, id) in &self.entries {
            if let Some(matched) = pattern.match_path(location) {
                return Some((*id, matched));
            }
        }
        None
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::std_instead_of_alloc, reason = "tests run under std")]

    use super::{RouteId, RouteTable};
    use crate::path::Location;

    #[test]
    fn first_match_wins() {
        let table = RouteTable::new()
            .route("/users/:id", RouteId::new(1))
            .route("/users/me", RouteId::new(2));
        // Both patterns match "/users/me"; the earlier one is returned.
        let (id, matched) = table
            .resolve(&Location::new("/users/me"))
            .expect("should resolve");
        assert_eq!(id, RouteId::new(1));
        assert_eq!(matched.param("id"), Some("me"));
    }

    #[test]
    fn no_match_returns_none() {
        let table = RouteTable::new().route("/users/:id", RouteId::new(1));
        assert!(table.resolve(&Location::new("/posts/42")).is_none());
    }
}
