//! Immutable, fully validated shader bytes; metadata alone is not executable.

use super::ShaderPackageManifest;
use alloc::{string::String, sync::Arc, vec::Vec};

/// BLAKE3 digest of canonical little-endian SPIR-V words, not of a filename.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ArtifactDigest(pub [u8; 32]);

/// A shader artifact was not valid for Prism's Naga-backed runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactValidationError {
    /// Package identity, revision or entry-point provenance is empty.
    EmptyProvenance,
    /// The binary header or instruction stream cannot be parsed.
    Parse(String),
    /// A parsed module violates type, control-flow or entry-point rules.
    Validation(String),
    /// The entry point declared by the manifest is not in the executable.
    EntryMissing(String),
    /// One entry name is exported with multiple shader stages.
    AmbiguousEntry(String),
}

/// Proof of Naga parsing and full validation, bound to immutable code bytes.
///
/// There is no unchecked constructor or deserialization implementation. Cooked
/// bytes must be validated again when loaded. This proves Naga IR validity, not
/// mathematical response semantics or support on every GPU; actual device
/// pipeline creation still checks device limits and features.
#[derive(Clone, Debug)]
pub struct VerifiedShaderArtifact {
    manifest: ShaderPackageManifest,
    words: Arc<[u32]>,
    digest: ArtifactDigest,
}

impl VerifiedShaderArtifact {
    /// Parses SPIR-V with strict capability handling and validates every entry.
    ///
    /// # Errors
    /// Rejects malformed, unsupported, ill-typed or missing-entry modules.
    pub fn from_spirv(
        manifest: ShaderPackageManifest,
        words: Vec<u32>,
    ) -> Result<Self, ArtifactValidationError> {
        if manifest.id.as_str().trim().is_empty()
            || manifest.source_revision.trim().is_empty()
            || manifest.entry_point.trim().is_empty()
        {
            return Err(ArtifactValidationError::EmptyProvenance);
        }
        let module = naga::front::spv::Frontend::new(
            words.iter().copied(),
            &naga::front::spv::Options::default(),
        )
        .parse()
        .map_err(|error| ArtifactValidationError::Parse(error.to_string()))?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|error| ArtifactValidationError::Validation(error.to_string()))?;
        match module
            .entry_points
            .iter()
            .filter(|entry| entry.name == manifest.entry_point)
            .count()
        {
            0 => return Err(ArtifactValidationError::EntryMissing(manifest.entry_point)),
            1 => {}
            _ => {
                return Err(ArtifactValidationError::AmbiguousEntry(
                    manifest.entry_point,
                ))
            }
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"prism.shader.spirv.v1\0");
        for word in &words {
            hasher.update(&word.to_le_bytes());
        }
        Ok(Self {
            manifest,
            digest: ArtifactDigest(*hasher.finalize().as_bytes()),
            words: words.into(),
        })
    }

    /// The metadata validated together with the executable entry.
    #[must_use]
    pub fn manifest(&self) -> &ShaderPackageManifest {
        &self.manifest
    }

    /// Immutable canonical words; clones share this allocation.
    #[must_use]
    pub fn spirv(&self) -> &[u32] {
        &self.words
    }

    /// Content identity suitable for comparing registry and response bindings.
    #[must_use]
    pub const fn digest(&self) -> ArtifactDigest {
        self.digest
    }
}
