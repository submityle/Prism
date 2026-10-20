//! Canonical, semantic paths into reflected values.

use prism_reflect::{Access, ParsePathError, ParsedPath};
use std::fmt;
use std::str::FromStr;

/// An error returned while parsing or canonicalizing a field path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FieldPathError {
    /// The source text is not accepted by `prism_reflect`'s path grammar.
    Parse(ParsePathError),
    /// A map key cannot be represented by the dependency's quoted-key grammar.
    UnrepresentableMapKey(String),
}

impl fmt::Display for FieldPathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => error.fmt(formatter),
            Self::UnrepresentableMapKey(key) => write!(
                formatter,
                "map key `{key}` contains both quote characters and cannot be represented"
            ),
        }
    }
}

impl std::error::Error for FieldPathError {}

impl From<ParsePathError> for FieldPathError {
    fn from(error: ParsePathError) -> Self {
        Self::Parse(error)
    }
}

/// A canonical access path into a reflected root value.
///
/// Equality is semantic rather than textual: `name` and `.name` produce the
/// same path. Canonical formatting uses `.field`, `#tuple`, `[index]`, and a
/// quoted string key. Collection indices remain stable while the collection's
/// ordering is unchanged; named fields are stable across value changes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldPath {
    parsed: ParsedPath,
    canonical: String,
}

impl FieldPath {
    /// Return the empty path, which addresses the root value.
    #[must_use]
    pub fn root() -> Self {
        Self {
            parsed: ParsedPath::default(),
            canonical: String::new(),
        }
    }

    /// Parse and canonicalize a reflection path.
    pub fn parse(source: &str) -> Result<Self, FieldPathError> {
        let parsed = ParsedPath::parse(source)?;
        Self::from_parsed(parsed)
    }

    /// Build a path from reflection access segments.
    pub fn from_accesses(accesses: &[Access]) -> Result<Self, FieldPathError> {
        Self::from_parsed(ParsedPath::parse(&canonicalize(accesses)?)?)
    }

    /// Return the canonical path string accepted by the reflection kernel.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.canonical
    }

    /// Whether this path addresses the root value.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.parsed.is_empty()
    }

    /// Expose the semantic access segments.
    #[must_use]
    pub fn segments(&self) -> &[Access] {
        self.parsed.segments()
    }

    /// Expose the parsed path for zero-copy reflection access.
    #[must_use]
    pub(crate) fn parsed(&self) -> &ParsedPath {
        &self.parsed
    }

    /// Append one semantic access step and return a new canonical path.
    pub(crate) fn with_access(&self, access: Access) -> Result<Self, FieldPathError> {
        let mut accesses = self.parsed.segments().to_vec();
        accesses.push(access);
        Self::from_accesses(&accesses)
    }

    fn from_parsed(parsed: ParsedPath) -> Result<Self, FieldPathError> {
        let canonical = canonicalize(parsed.segments())?;
        Ok(Self { parsed, canonical })
    }
}

impl Default for FieldPath {
    fn default() -> Self {
        Self::root()
    }
}

impl FromStr for FieldPath {
    type Err = FieldPathError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        Self::parse(source)
    }
}

impl fmt::Display for FieldPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_root() {
            formatter.write_str("<root>")
        } else {
            formatter.write_str(self.as_str())
        }
    }
}

fn canonicalize(accesses: &[Access]) -> Result<String, FieldPathError> {
    let mut canonical = String::new();
    for access in accesses {
        match access {
            Access::Field(name) => {
                canonical.push('.');
                canonical.push_str(name);
            }
            Access::TupleIndex(index) => {
                canonical.push('#');
                canonical.push_str(&index.to_string());
            }
            Access::ListIndex(index) => {
                canonical.push('[');
                canonical.push_str(&index.to_string());
                canonical.push(']');
            }
            Access::Key(key) => {
                let quote = if !key.contains('"') {
                    '"'
                } else if !key.contains('\'') {
                    '\''
                } else {
                    return Err(FieldPathError::UnrepresentableMapKey(key.clone()));
                };
                canonical.push('[');
                canonical.push(quote);
                canonical.push_str(key);
                canonical.push(quote);
                canonical.push(']');
            }
        }
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_semantic_and_canonical() {
        let first = FieldPath::parse("settings.gain[2]").expect("path");
        let second = FieldPath::parse(".settings.gain[2]").expect("path");
        assert_eq!(first, second);
        assert_eq!(first.as_str(), ".settings.gain[2]");
    }

    #[test]
    fn quoted_keys_are_canonicalized() {
        let path = FieldPath::from_accesses(&[Access::Key(String::from("a'b"))]).expect("path");
        assert_eq!(path.as_str(), "[\"a'b\"]");
        assert_eq!(FieldPath::parse(path.as_str()).expect("round trip"), path);
    }
}
