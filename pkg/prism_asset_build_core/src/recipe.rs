//! Explicit importer/cook recipe identity.

use crate::hash::{derive_artifact_key, ArtifactKeyInputs, HashError};
use crate::source::SourceSnapshot;
use crate::types::ArtifactKey;

/// All semantic inputs needed to derive an imported artifact key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportRecipe {
    /// Stable importer/processor protocol identity.
    pub importer_id: u128,
    /// Importer implementation/schema version.
    pub importer_version: u32,
    /// Canonically encoded import settings.
    pub canonical_settings: Vec<u8>,
    /// Explicit platform/profile identity.
    pub profile_hash: [u8; 32],
    /// Explicit toolchain fingerprint.
    pub toolchain_fingerprint: Vec<u8>,
    /// Explicit declared environment fingerprint.
    pub environment_fingerprint: Vec<u8>,
}

impl ImportRecipe {
    /// Derives the artifact key for a source and ordered imported inputs.
    pub fn artifact_key(
        &self,
        protocol_version: u32,
        source: &SourceSnapshot,
        imported_inputs: &[ArtifactKey],
    ) -> Result<ArtifactKey, HashError> {
        let mut ordered_inputs = Vec::with_capacity(imported_inputs.len() + 1);
        ordered_inputs.push(source.artifact_key());
        ordered_inputs.extend_from_slice(imported_inputs);
        derive_artifact_key(&ArtifactKeyInputs {
            protocol_version,
            processor_id: self.importer_id,
            processor_version: self.importer_version,
            ordered_inputs: &ordered_inputs,
            canonical_settings: &self.canonical_settings,
            profile_hash: &self.profile_hash,
            toolchain_fingerprint: &self.toolchain_fingerprint,
            environment_fingerprint: &self.environment_fingerprint,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe() -> ImportRecipe {
        ImportRecipe {
            importer_id: 0x494d504f52544552,
            importer_version: 3,
            canonical_settings: b"{\"srgb\":true}".to_vec(),
            profile_hash: [7; 32],
            toolchain_fingerprint: b"toolchain-v1".to_vec(),
            environment_fingerprint: b"declared-platform".to_vec(),
        }
    }

    #[test]
    fn recipe_key_is_deterministic_and_all_declared_inputs_matter() {
        let source = SourceSnapshot::new("textures/hero.png", b"source".to_vec()).unwrap();
        let first = recipe().artifact_key(1, &source, &[]).unwrap();
        assert_eq!(first, recipe().artifact_key(1, &source, &[]).unwrap());

        let mut changed = recipe();
        changed.canonical_settings = b"{\"srgb\":false}".to_vec();
        assert_ne!(first, changed.artifact_key(1, &source, &[]).unwrap());
        assert_ne!(
            first,
            recipe()
                .artifact_key(1, &source, &[ArtifactKey([9; 32])])
                .unwrap()
        );
    }
}
