//! Immutable source snapshots used by import and cook orchestration.

use crate::executor::ArtifactSpec;
use crate::job::{BuildJob, JobId, ProcessorError};
use crate::types::{ArtifactKey, AssetId, ContentHash};
use std::fmt;

/// A canonical, immutable source file snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceSnapshot {
    logical_path: String,
    bytes: Vec<u8>,
    content_hash: ContentHash,
}

/// The persisted semantic identity of a source snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceIdentity {
    /// Canonical relative logical path.
    pub logical_path: String,
    /// Digest of the exact captured source bytes.
    pub content_hash: ContentHash,
}

/// Processor identity for source-root jobs in the deterministic build graph.
pub const SOURCE_PROCESSOR_ID: u128 = 0x5052_4953_4d5f_534f_5552_4345_0001;
/// Schema version for source-root outputs.
pub const SOURCE_PROCESSOR_VERSION: u32 = 1;
/// Artifact kind for immutable source-root outputs.
pub const SOURCE_ARTIFACT_KIND: u128 = 0x5052_4953_4d5f_534f_5552_4345;
/// Artifact schema for immutable source-root outputs.
pub const SOURCE_ARTIFACT_SCHEMA: u32 = 1;

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

    /// Returns the compact semantic identity used by invalidation reports.
    #[must_use]
    pub fn identity(&self) -> SourceIdentity {
        SourceIdentity {
            logical_path: self.logical_path.clone(),
            content_hash: self.content_hash,
        }
    }

    /// Stable semantic key for this source snapshot.
    #[must_use]
    pub fn artifact_key(&self) -> ArtifactKey {
        self.artifact_spec(AssetId(0))
            .artifact_key(1, &[])
            .expect("source key fields fit the artifact-key protocol")
    }

    /// Creates the root output spec used by the build executor.
    #[must_use]
    pub fn artifact_spec(&self, asset: AssetId) -> ArtifactSpec {
        ArtifactSpec {
            asset,
            processor_id: SOURCE_PROCESSOR_ID,
            processor_version: SOURCE_PROCESSOR_VERSION,
            canonical_settings: self.canonical_settings(),
            profile_hash: [0; 32],
            toolchain_fingerprint: Vec::new(),
            environment_fingerprint: Vec::new(),
            kind: SOURCE_ARTIFACT_KIND,
            schema: SOURCE_ARTIFACT_SCHEMA,
            alignment: 1,
        }
    }

    fn canonical_settings(&self) -> Vec<u8> {
        let path = self.logical_path.as_bytes();
        let mut settings = Vec::with_capacity(8 + path.len() + self.content_hash.0.len());
        settings.extend_from_slice(&(path.len() as u64).to_le_bytes());
        settings.extend_from_slice(path);
        settings.extend_from_slice(&self.content_hash.0);
        settings
    }

    /// Creates a graph root job that emits this exact captured source.
    #[must_use]
    pub fn build_job(&self, id: JobId) -> BuildJob {
        let bytes = self.bytes.clone();
        BuildJob::new(id, Vec::<JobId>::new(), move |inputs: &[&[u8]]| {
            if !inputs.is_empty() {
                return Err(ProcessorError::new(
                    "source root job cannot have dependencies",
                ));
            }
            Ok(bytes.clone())
        })
    }
}

pub(crate) fn is_canonical_path(path: &str) -> bool {
    canonicalize_path(path).is_ok_and(|canonical| canonical == path)
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
