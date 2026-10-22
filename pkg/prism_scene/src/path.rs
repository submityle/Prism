//! [`FieldPath`] — the reflect-style field path an override targets
//! (design §7.1), e.g. `"transform.translation.x"`.
//!
//! A path is an ordered list of [`FieldSegment`]s: a [`Name`](FieldSegment::Name)
//! for a struct field and an [`Index`](FieldSegment::Index) for an array/list
//! element. The textual form joins segments with `.`, and a segment that is all
//! ASCII digits is decoded as an index — the same convention used by the editor
//! property tree the override set feeds.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// One step of a [`FieldPath`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FieldSegment {
    /// A named struct field.
    Name(Box<str>),
    /// A positional element of an array, list or tuple.
    Index(u32),
}

impl FieldSegment {
    /// Builds a name segment.
    #[must_use]
    pub fn name(name: impl Into<Box<str>>) -> Self {
        Self::Name(name.into())
    }

    /// Builds an index segment.
    #[must_use]
    pub const fn index(index: u32) -> Self {
        Self::Index(index)
    }
}

/// A reflect-style field path (design §7.1).
///
/// Paths are the unique key (alongside the target entity and component type) by
/// which a [`PropertyOverride`](crate::overrides::PropertyOverride) is stored and
/// deduplicated.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FieldPath {
    segments: Vec<FieldSegment>,
}

impl FieldPath {
    /// The empty path (targets the value as a whole).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a path from explicit segments.
    #[must_use]
    pub fn from_segments(segments: Vec<FieldSegment>) -> Self {
        Self { segments }
    }

    /// Parses a dotted textual path.
    ///
    /// An empty string is the empty path; a segment that is all ASCII digits
    /// becomes an index; an empty interior segment (`"a..b"`) or an index that
    /// overflows `u32` yields `None`.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        if text.is_empty() {
            return Some(Self::new());
        }
        let mut segments = Vec::new();
        for part in text.split('.') {
            if part.is_empty() {
                return None;
            }
            if part.bytes().all(|b| b.is_ascii_digit()) {
                let index = part.parse::<u32>().ok()?;
                segments.push(FieldSegment::Index(index));
            } else {
                segments.push(FieldSegment::Name(part.into()));
            }
        }
        Some(Self { segments })
    }

    /// The segments, in order.
    #[must_use]
    pub fn segments(&self) -> &[FieldSegment] {
        &self.segments
    }

    /// Number of segments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Whether the path is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Appends one segment.
    pub fn push(&mut self, segment: FieldSegment) {
        self.segments.push(segment);
    }

    /// Returns the canonical dotted form.
    #[must_use]
    pub fn to_canonical_string(&self) -> String {
        use core::fmt::Write as _;
        let mut out = String::new();
        for (i, segment) in self.segments.iter().enumerate() {
            if i > 0 {
                out.push('.');
            }
            match segment {
                FieldSegment::Name(name) => out.push_str(name),
                FieldSegment::Index(index) => {
                    let _ = write!(out, "{index}");
                }
            }
        }
        out
    }
}

impl fmt::Display for FieldPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, segment) in self.segments.iter().enumerate() {
            if i > 0 {
                f.write_str(".")?;
            }
            match segment {
                FieldSegment::Name(name) => f.write_str(name)?,
                FieldSegment::Index(index) => write!(f, "{index}")?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{FieldPath, FieldSegment};
    use alloc::string::ToString;
    use alloc::vec;

    #[test]
    fn parses_names_and_indices() {
        let path = FieldPath::parse("transform.translation.0.x").unwrap();
        assert_eq!(
            path.segments(),
            &[
                FieldSegment::name("transform"),
                FieldSegment::name("translation"),
                FieldSegment::Index(0),
                FieldSegment::name("x"),
            ]
        );
        assert_eq!(path.len(), 4);
        assert!(!path.is_empty());
    }

    #[test]
    fn empty_string_is_the_empty_path() {
        let path = FieldPath::parse("").unwrap();
        assert!(path.is_empty());
        assert_eq!(path, FieldPath::new());
    }

    #[test]
    fn rejects_degenerate_and_overflowing_segments() {
        assert_eq!(FieldPath::parse("a..b"), None);
        assert_eq!(FieldPath::parse(".a"), None);
        assert_eq!(FieldPath::parse("a."), None);
        // `4294967296` = 2^32, one past `u32::MAX`.
        assert_eq!(FieldPath::parse("a.4294967296"), None);
    }

    #[test]
    fn canonical_form_round_trips() {
        let text = "transform.translation.0.x";
        let path = FieldPath::parse(text).unwrap();
        assert_eq!(path.to_string(), text);
        assert_eq!(path.to_canonical_string(), text);
        assert_eq!(FieldPath::parse(&path.to_string()).unwrap(), path);
    }

    #[test]
    fn from_segments_and_push_agree() {
        let mut path = FieldPath::from_segments(vec![FieldSegment::name("a")]);
        path.push(FieldSegment::index(2));
        assert_eq!(path.to_string(), "a.2");
    }
}
