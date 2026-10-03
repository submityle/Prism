//! Asset source paths with an optional sub-asset label.

use alloc::string::String;
use core::fmt;

/// An owned asset source path with an optional sub-asset `#label`.
///
/// Many container formats expose several assets from one file (a glTF scene
/// yields meshes, materials, animations, and so on). A label selects one
/// sub-asset, written after a `#`, for example `"models/hero.gltf#Mesh0"`. A
/// path without a `#` refers to the primary asset of the source.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssetPath {
    path: String,
    label: Option<String>,
}

impl AssetPath {
    /// Creates a path referring to a source's primary asset (no label).
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            label: None,
        }
    }

    /// Parses a path string, splitting an optional `#label` suffix.
    ///
    /// The split uses the first `#`; the remainder becomes the label. A `#`
    /// with an empty remainder is treated as having no label.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        match text.split_once('#') {
            Some((path, label)) if !label.is_empty() => Self {
                path: String::from(path),
                label: Some(String::from(label)),
            },
            Some((path, _)) => Self {
                path: String::from(path),
                label: None,
            },
            None => Self {
                path: String::from(text),
                label: None,
            },
        }
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

    /// The source path portion (never includes the `#label`).
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
        match &self.label {
            Some(label) => write!(f, "{}#{label}", self.path),
            None => f.write_str(&self.path),
        }
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
