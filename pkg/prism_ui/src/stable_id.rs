//! Compile-time stable identity for element-tree nodes.
//!
//! A [`StableId`] is a structurally stable label assigned to a node based on
//! its *position in the source tree* (for example `""` for the root, `"0"` for
//! the first child, `"0/1"` for the second child of the first child). The
//! `loom!` macro computes these paths at compile time and attaches them to each
//! statically-known node.
//!
//! Unlike a positional reconciliation key, a stable id is independent of a
//! node's run-time sibling index: inserting or deleting a sibling in a *later*
//! revision of the view shifts positions, yet a node that keeps the same source
//! path keeps the same [`StableId`] and can therefore be recognised as "the
//! same node" by the hot-reload diff.
//!
//! A stable id is pure identity metadata: it is deliberately *not* part of an
//! [`Element`](crate::Element)'s value identity, so two visually identical
//! elements compare equal even if they came from different source positions.
//!
//! The identifier stores the raw path string in a [`Cow<'static, str>`], so the
//! common macro-generated case (a `'static` string literal) allocates nothing,
//! while run-time construction from an owned [`String`] is still supported. The
//! type is `no_std` friendly and relies only on `alloc`.

use alloc::borrow::Cow;
use alloc::string::String;
use core::fmt;

/// A compile-time, structurally stable identifier for a node in a view tree.
///
/// The wrapped string is a `/`-separated position path such as `"0/1"`. See the
/// [module documentation](self) for the full rationale.
///
/// # Examples
///
/// ```
/// use prism_ui::StableId;
///
/// let id = StableId::new("0/1");
/// assert_eq!(id.as_str(), "0/1");
/// assert_eq!(id, StableId::from("0/1"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StableId {
    /// The raw `/`-separated position path, borrowed when `'static`.
    path: Cow<'static, str>,
}

impl StableId {
    /// Creates a stable id from a position-path string.
    ///
    /// Accepts anything convertible into a [`Cow<'static, str>`], so both a
    /// `'static` string literal (zero-allocation, the macro-generated case) and
    /// an owned [`String`] work.
    #[must_use]
    pub fn new(path: impl Into<Cow<'static, str>>) -> Self {
        Self { path: path.into() }
    }

    /// Returns the raw position-path string this id wraps.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.path
    }

    /// Returns the number of path segments (the node's depth from the root).
    ///
    /// The root path (`""`) has depth `0`; `"0"` has depth `1`; `"0/1"` has
    /// depth `2`.
    #[must_use]
    pub fn depth(&self) -> usize {
        if self.path.is_empty() {
            0
        } else {
            self.path.as_ref().bytes().filter(|b| *b == b'/').count() + 1
        }
    }
}

impl fmt::Display for StableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.path)
    }
}

impl From<&'static str> for StableId {
    fn from(value: &'static str) -> Self {
        Self::new(value)
    }
}

impl From<String> for StableId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<Cow<'static, str>> for StableId {
    fn from(value: Cow<'static, str>) -> Self {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::StableId;
    use alloc::string::{String, ToString};

    #[test]
    fn as_str_round_trips_the_path() {
        let id = StableId::new("0/1/2");
        assert_eq!(id.as_str(), "0/1/2");
        assert_eq!(id.to_string(), "0/1/2");
    }

    #[test]
    fn from_static_str_and_string_agree() {
        let a = StableId::from("3");
        let b = StableId::from(String::from("3"));
        assert_eq!(a, b);
    }

    #[test]
    fn depth_counts_segments() {
        assert_eq!(StableId::new("").depth(), 0);
        assert_eq!(StableId::new("0").depth(), 1);
        assert_eq!(StableId::new("0/1").depth(), 2);
        assert_eq!(StableId::new("0/1/2").depth(), 3);
    }

    #[test]
    fn ordering_is_lexicographic_on_the_path() {
        assert!(StableId::new("0") < StableId::new("0/0"));
        assert!(StableId::new("0/0") < StableId::new("0/1"));
    }
}
