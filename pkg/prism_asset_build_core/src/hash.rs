use crate::types::ArtifactKey;

/// Fully explicit inputs to the artifact-key protocol. Settings and fingerprints
/// must already be canonical bytes; no host state is read implicitly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactKeyInputs<'a> {
    pub protocol_version: u32,
    pub processor_id: u128,
    pub processor_version: u32,
    pub ordered_inputs: &'a [ArtifactKey],
    pub canonical_settings: &'a [u8],
    pub profile_hash: &'a [u8; 32],
    pub toolchain_fingerprint: &'a [u8],
    pub environment_fingerprint: &'a [u8],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HashError {
    FieldTooLarge(&'static str),
}

impl std::fmt::Display for HashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FieldTooLarge(name) => write!(f, "artifact-key field too large: {name}"),
        }
    }
}

impl std::error::Error for HashError {}

/// Derives a deterministic key using a versioned, domain-separated, length-prefixed encoding.
pub fn derive_artifact_key(input: &ArtifactKeyInputs<'_>) -> Result<ArtifactKey, HashError> {
    let mut h = blake3::Hasher::new();
    h.update(b"Prism.AssetBuild.ArtifactKey\0");
    h.update(&input.protocol_version.to_le_bytes());
    h.update(&input.processor_id.to_le_bytes());
    h.update(&input.processor_version.to_le_bytes());
    put_len(&mut h, "ordered_inputs", input.ordered_inputs.len())?;
    for key in input.ordered_inputs {
        h.update(&key.0);
    }
    put_bytes(&mut h, "canonical_settings", input.canonical_settings)?;
    h.update(input.profile_hash);
    put_bytes(&mut h, "toolchain_fingerprint", input.toolchain_fingerprint)?;
    put_bytes(
        &mut h,
        "environment_fingerprint",
        input.environment_fingerprint,
    )?;
    Ok(ArtifactKey(*h.finalize().as_bytes()))
}

fn put_len(h: &mut blake3::Hasher, field: &'static str, len: usize) -> Result<(), HashError> {
    let len = u64::try_from(len).map_err(|_| HashError::FieldTooLarge(field))?;
    h.update(&len.to_le_bytes());
    Ok(())
}

fn put_bytes(h: &mut blake3::Hasher, field: &'static str, bytes: &[u8]) -> Result<(), HashError> {
    put_len(h, field, bytes.len())?;
    h.update(bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs<'a>(ordered_inputs: &'a [ArtifactKey], settings: &'a [u8]) -> ArtifactKeyInputs<'a> {
        ArtifactKeyInputs {
            protocol_version: 1,
            processor_id: 17,
            processor_version: 3,
            ordered_inputs,
            canonical_settings: settings,
            profile_hash: &[9; 32],
            toolchain_fingerprint: b"tool-v1",
            environment_fingerprint: b"declared-env",
        }
    }

    #[test]
    fn derivation_is_stable_and_all_explicit_inputs_matter() {
        let keys = [ArtifactKey([1; 32]), ArtifactKey([2; 32])];
        assert_eq!(
            derive_artifact_key(&inputs(&keys, b"canonical")).unwrap(),
            derive_artifact_key(&inputs(&keys, b"canonical")).unwrap()
        );
        assert_ne!(
            derive_artifact_key(&inputs(&keys, b"canonical")).unwrap(),
            derive_artifact_key(&inputs(&keys, b"other")).unwrap()
        );
        let reversed = [keys[1], keys[0]];
        assert_ne!(
            derive_artifact_key(&inputs(&keys, b"canonical")).unwrap(),
            derive_artifact_key(&inputs(&reversed, b"canonical")).unwrap()
        );
    }
}
