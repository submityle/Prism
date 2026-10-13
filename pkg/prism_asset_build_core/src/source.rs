//! Immutable source snapshots used by import and cook orchestration.

use crate::types::{ArtifactKey, ContentHash};
use std::fmt;

/// A canonical, immutable source file snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceSnapshot {
    logical_path: String,
    bytes: Vec<u8>,
    content_hash: ContentHash,
}

/// Errors raised while constructing a source snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceError {
    /// A source path is empty, absolute, or contains an unsafe parent segment.
    InvalidPath,
    /// A source path contains a NUL byte.
    NulInPath,
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPath => f.write_str("source path is not canonical and relative"),
            Self::NulInPath => f.write_str("source path contains a NUL byte"),
        }
    }
}

impl std::error::Error for SourceError {}

impl SourceSnapshot {
    /// Captures source bytes and computes their immutable content identity.
    pub fn new(path: &str, bytes: Vec<u8>) -> Result<Self, SourceError> {
        let logical_path = canonicalize_path(path)?;
        let content_hash = ContentHash::digest(&bytes);
        Ok(Self {
            logical_path,
            bytes,
            content_hash,
        })
    }

    /// Canonical logical path used for diagnostics and source identity.
    #[must_use]
    pub fn logical_path(&self) -> &str {
        &self.logical_path
    }

    /// Exact captured source bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// BLAKE3 identity of the exact captured bytes.
    #[must_use]
    pub const fn content_hash(&self) -> ContentHash {
        self.content_hash
    }

    /// Stable semantic key for this source snapshot.
    #[must_use]
    pub fn artifact_key(&self) -> ArtifactKey {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"Prism.AssetBuild.SourceKey\0");
        put_bytes(&mut hasher, self.logical_path.as_bytes());
        hasher.update(&self.content_hash.0);
        ArtifactKey(*hasher.finalize().as_bytes())
    }
}

fn canonicalize_path(path: &str) -> Result<String, SourceError> {
    if path.is_empty() || path.as_bytes().contains(&0) {
        return if path.as_bytes().contains(&0) {
            Err(SourceError::NulInPath)
        } else {
            Err(SourceError::InvalidPath)
        };
    }
    if path.starts_with('/') || path.starts_with('\\') || path.contains(':') {
        return Err(SourceError::InvalidPath);
    }
    let mut segments = Vec::new();
    for segment in path.split(['/', '\\']) {
        if segment.is_empty() || segment == "." {
            continue;
        }
        if segment == ".." {
            return Err(SourceError::InvalidPath);
        }
        segments.push(segment);
    }
    if segments.is_empty() {
        return Err(SourceError::InvalidPath);
    }
    Ok(segments.join("/"))
}

fn put_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_relative_source_paths_and_preserves_bytes() {
        let snapshot = SourceSnapshot::new("textures\\./hero.png", b"png".to_vec()).unwrap();
        assert_eq!(snapshot.logical_path(), "textures/hero.png");
        assert_eq!(snapshot.bytes(), b"png");
        assert_eq!(snapshot.content_hash(), ContentHash::digest(b"png"));
    }

    #[test]
    fn rejects_unsafe_source_paths() {
        for path in [
            "",
            "/hero.png",
            "../hero.png",
            "textures/../../hero.png",
            "C:hero.png",
        ] {
            assert_eq!(
                SourceSnapshot::new(path, Vec::new()),
                Err(SourceError::InvalidPath)
            );
        }
        assert_eq!(
            SourceSnapshot::new("hero\0.png", Vec::new()),
            Err(SourceError::NulInPath)
        );
    }

    #[test]
    fn source_key_changes_for_path_or_bytes() {
        let first = SourceSnapshot::new("hero.png", b"one".to_vec()).unwrap();
        let same = SourceSnapshot::new("./hero.png", b"one".to_vec()).unwrap();
        let other_bytes = SourceSnapshot::new("hero.png", b"two".to_vec()).unwrap();
        let other_path = SourceSnapshot::new("other.png", b"one".to_vec()).unwrap();
        assert_eq!(first.artifact_key(), same.artifact_key());
        assert_ne!(first.artifact_key(), other_bytes.artifact_key());
        assert_ne!(first.artifact_key(), other_path.artifact_key());
    }
}
