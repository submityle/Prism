//! Golden comparison of a serialized snapshot against expected text.
//!
//! A [`Snapshot`] owns the deterministic serialization of either an element
//! [`TreeSnapshot`] or a captured layout tree. Comparing it against a stored
//! "golden" string yields a structured [`Comparison`]: either [`Comparison::Match`]
//! or a [`Comparison::Mismatch`] carrying a human-readable [`crate::diff`].
//!
//! This mirrors the ergonomics of snapshot testing libraries: capture once,
//! store the golden text, and on later runs assert the fresh capture still
//! matches.

use alloc::string::String;

use prism_ui_devtools::TreeSnapshot;

use crate::diff::diff;
use crate::layout_snapshot::{serialize_layout, LayoutSnapshotNode};
use crate::serialize::serialize_tree;

/// An owned, serialized snapshot ready for golden comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    text: String,
}

impl Snapshot {
    /// Builds a snapshot from an element [`TreeSnapshot`].
    #[must_use]
    pub fn from_tree(tree: &TreeSnapshot) -> Self {
        Self {
            text: serialize_tree(tree),
        }
    }

    /// Builds a snapshot from a captured [`LayoutSnapshotNode`] tree.
    #[must_use]
    pub fn from_layout(root: &LayoutSnapshotNode) -> Self {
        Self {
            text: serialize_layout(root),
        }
    }

    /// Builds a snapshot directly from already-serialized text.
    #[must_use]
    pub fn from_text(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    /// Returns the serialized text backing this snapshot.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Consumes the snapshot, returning its serialized text.
    #[must_use]
    pub fn into_text(self) -> String {
        self.text
    }

    /// Returns `true` when this snapshot exactly equals `expected`.
    #[must_use]
    pub fn matches(&self, expected: &str) -> bool {
        self.text == expected
    }

    /// Compares this snapshot against `expected`, returning a [`Comparison`].
    ///
    /// On a mismatch the result carries a human-readable diff of `expected`
    /// against the snapshot's own text.
    #[must_use]
    pub fn compare(&self, expected: &str) -> Comparison {
        if self.text == expected {
            Comparison::Match
        } else {
            Comparison::Mismatch {
                diff: diff(expected, &self.text),
            }
        }
    }

    /// Snapshot-testing-style comparison against `expected`.
    ///
    /// This is an alias for [`Snapshot::compare`] that reads naturally at call
    /// sites: `snapshot.assert_matches(golden)`. It returns the structured
    /// [`Comparison`] rather than panicking, so callers choose how to react.
    #[must_use]
    pub fn assert_matches(&self, expected: &str) -> Comparison {
        self.compare(expected)
    }
}

/// The structured outcome of comparing a [`Snapshot`] with expected text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Comparison {
    /// The snapshot matched the expected text exactly.
    Match,
    /// The snapshot differed; the field holds a human-readable diff.
    Mismatch {
        /// A line-level diff of expected against actual, from [`crate::diff`].
        diff: String,
    },
}

impl Comparison {
    /// Returns `true` for [`Comparison::Match`].
    #[must_use]
    pub fn is_match(&self) -> bool {
        matches!(self, Comparison::Match)
    }

    /// Returns `true` for [`Comparison::Mismatch`].
    #[must_use]
    pub fn is_mismatch(&self) -> bool {
        matches!(self, Comparison::Mismatch { .. })
    }

    /// Returns the diff text on a mismatch, or `None` on a match.
    #[must_use]
    pub fn diff(&self) -> Option<&str> {
        match self {
            Comparison::Match => None,
            Comparison::Mismatch { diff } => Some(diff),
        }
    }

    /// Converts the comparison into a `Result`, with the diff as the error.
    ///
    /// # Errors
    ///
    /// Returns `Err` carrying the diff text when this is a
    /// [`Comparison::Mismatch`].
    pub fn into_result(self) -> Result<(), String> {
        match self {
            Comparison::Match => Ok(()),
            Comparison::Mismatch { diff } => Err(diff),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Comparison, Snapshot};
    use alloc::vec::Vec;
    use prism_ui_devtools::{SnapshotNode, TreeSnapshot};

    fn sample() -> TreeSnapshot {
        TreeSnapshot {
            root: SnapshotNode {
                kind: "Box".into(),
                text: None,
                classes: Vec::new(),
                children: alloc::vec![SnapshotNode {
                    kind: "Text".into(),
                    text: Some("hi".into()),
                    classes: Vec::new(),
                    children: Vec::new(),
                }],
            },
        }
    }

    #[test]
    fn match_reports_no_diff() {
        let snapshot = Snapshot::from_tree(&sample());
        let golden = snapshot.as_str().to_string();
        let comparison = snapshot.assert_matches(&golden);
        assert_eq!(comparison, Comparison::Match);
        assert!(comparison.is_match());
        assert_eq!(comparison.diff(), None);
        assert!(comparison.into_result().is_ok());
    }

    #[test]
    fn mismatch_reports_diff() {
        let snapshot = Snapshot::from_tree(&sample());
        let comparison = snapshot.compare("kind=Box text=- classes=\n");
        assert!(comparison.is_mismatch());
        let diff = comparison.diff().expect("diff present");
        assert!(diff.contains("kind=Text text=+hi classes="));
        assert!(snapshot
            .compare("kind=Box text=- classes=\n")
            .into_result()
            .is_err());
    }

    #[test]
    fn text_accessors_round_trip() {
        let snapshot = Snapshot::from_text("kind=Box text=- classes=\n");
        assert_eq!(snapshot.as_str(), "kind=Box text=- classes=\n");
        assert!(snapshot.matches("kind=Box text=- classes=\n"));
        assert_eq!(snapshot.into_text(), "kind=Box text=- classes=\n");
    }
}
