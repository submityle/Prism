//! Asset source paths with an optional `source://` scheme and sub-asset label.

use alloc::string::String;
use core::fmt;

/// An owned asset source path with an optional `scheme://` prefix and an
/// optional sub-asset `#label`.
///
/// A full path has three parts, any of the outer two optional:
///
/// ```text
///   source://models/hero.gltf#Mesh0
///   └──┬──┘   └──────┬──────┘ └─┬─┘
///    scheme         path      label
/// ```
///
/// - **scheme** names a *mount / data source* (`source`, `game`, `engine`, a
///   DLC root, …) so the VFS can resolve the same relative path against
///   different roots with an explicit priority order (design §10.4). A path
///   with no scheme resolves against the default source.
/// - **path** is the source-relative locator; it never contains the scheme or
///   the label.
/// - **label** selects one sub-asset of a multi-asset source: a glTF scene
///   yields meshes, materials, and animations, and `scene.gltf#Mesh0` picks
///   one. A path without a `#` refers to the primary asset of the source.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssetPath {
    scheme: Option<String>,
    path: String,
    label: Option<String>,
}

impl AssetPath {
    /// Creates a path referring to a source's primary asset (no scheme, no
    /// label).
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            scheme: None,
            path: path.into(),
            label: None,
        }
    }

    /// Parses a path string, splitting an optional `scheme://` prefix and an
    /// optional `#label` suffix.
    ///
    /// The scheme is the text before the first `://`; a scheme must be
    /// non-empty to be recognised, otherwise the `://` is treated as part of
    /// the path. The label is the text after the first `#` in the remainder; a
    /// `#` with an empty remainder is treated as having no label.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let (scheme, rest) = match text.split_once("://") {
            Some((scheme, rest)) if !scheme.is_empty() => (Some(String::from(scheme)), rest),
            _ => (None, text),
        };
        let (path, label) = match rest.split_once('#') {
            Some((path, label)) if !label.is_empty() => (path, Some(String::from(label))),
            Some((path, _)) => (path, None),
            None => (rest, None),
        };
        Self {
            scheme,
            path: String::from(path),
            label,
        }
    }

    /// Returns this path with `scheme` set as its source selector.
    #[must_use]
    pub fn with_scheme(mut self, scheme: impl Into<String>) -> Self {
        self.scheme = Some(scheme.into());
        self
    }

    /// Returns this path with any `scheme://` prefix removed.
    #[must_use]
    pub fn without_scheme(mut self) -> Self {
        self.scheme = None;
        self
    }

    /// Returns this path with `label` set as its sub-asset selector.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns this path with any sub-asset label removed.
    #[must_use]
    pub fn without_label(mut self) -> Self {
        self.label = None;
        self
    }

    /// The source/mount scheme, if any (the text before `://`).
    #[must_use]
    pub fn scheme(&self) -> Option<&str> {
        self.scheme.as_deref()
    }

    /// Whether this path names an explicit source scheme.
    #[must_use]
    pub fn has_scheme(&self) -> bool {
        self.scheme.is_some()
    }

    /// The source-relative path portion (never includes the scheme or the
    /// `#label`).
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The sub-asset label, if any.
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Whether this path selects a labeled sub-asset.
    #[must_use]
    pub fn has_label(&self) -> bool {
        self.label.is_some()
    }
}

impl fmt::Display for AssetPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(scheme) = &self.scheme {
            write!(f, "{scheme}://")?;
        }
        f.write_str(&self.path)?;
        if let Some(label) = &self.label {
            write!(f, "#{label}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for AssetPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssetPath({self})")
    }
}

impl From<&str> for AssetPath {
    fn from(text: &str) -> Self {
        Self::parse(text)
    }
}

impl From<String> for AssetPath {
    fn from(text: String) -> Self {
        Self::parse(&text)
    }
}
