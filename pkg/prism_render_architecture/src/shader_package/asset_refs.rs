//! Portable shader-package references used by asset/build manifests.
//!
//! This module intentionally contains no [`ModuleId`], device fingerprint,
//! backend handle, or pipeline-cache state. Those values are allocated by the
//! renderer after a package is loaded and therefore belong to the driver-local
//! half of the pipeline, never to an asset artifact.

use alloc::{string::String, vec::Vec};

use super::ArtifactDigest;

/// One authored shader program referenced by a portable asset.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ShaderProgramAssetRef {
    /// Stable authored program identity, not a process-local module handle.
    pub program_id: String,
    /// Digest of the source/IR/code identity used by the build.
    pub code_digest: ArtifactDigest,
}

/// Portable shader references that may cross machines and backends.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderAssetRefs {
    /// Textual package identity retained at the asset boundary.
    pub package_name: String,
    /// Referenced programs, canonically ordered by `program_id`.
    pub programs: Vec<ShaderProgramAssetRef>,
    /// Named static features required by the authored material/package.
    /// Values are canonicalized and contain no backend/device state.
    pub static_features: Vec<String>,
}

/// Structural failure in the portable `asset_refs` contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AssetRefsError {
    EmptyPackageName,
    InvalidPackageName,
    NoPrograms,
    EmptyProgramId,
    InvalidProgramId,
    DuplicateProgramId(String),
    ProgramsNotCanonical,
    EmptyStaticFeature,
    InvalidStaticFeature,
    DuplicateStaticFeature(String),
    StaticFeaturesNotCanonical,
    ZeroCodeDigest(String),
}

impl core::fmt::Display for AssetRefsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyPackageName => f.write_str("shader asset refs have an empty package name"),
            Self::InvalidPackageName => {
                f.write_str("shader asset refs have an invalid package name")
            }
            Self::NoPrograms => f.write_str("shader asset refs must contain at least one program"),
            Self::EmptyProgramId => f.write_str("shader asset refs contain an empty program id"),
            Self::InvalidProgramId => {
                f.write_str("shader asset refs contain an invalid program id")
            }
            Self::DuplicateProgramId(id) => write!(f, "shader asset refs duplicate program {id:?}"),
            Self::ProgramsNotCanonical => {
                f.write_str("shader asset refs programs are not canonically ordered")
            }
            Self::EmptyStaticFeature => {
                f.write_str("shader asset refs contain an empty static feature")
            }
            Self::InvalidStaticFeature => {
                f.write_str("shader asset refs contain an invalid static feature")
            }
            Self::DuplicateStaticFeature(feature) => {
                write!(f, "shader asset refs duplicate static feature {feature:?}")
            }
            Self::StaticFeaturesNotCanonical => {
                f.write_str("shader asset refs static features are not canonically ordered")
            }
            Self::ZeroCodeDigest(id) => write!(f, "shader program {id:?} has a zero code digest"),
        }
    }
}

impl std::error::Error for AssetRefsError {}

impl ShaderAssetRefs {
    /// Validates the portable representation and its canonical ordering.
    pub fn validate(&self) -> Result<(), AssetRefsError> {
        validate_name(
            &self.package_name,
            AssetRefsError::EmptyPackageName,
            AssetRefsError::InvalidPackageName,
        )?;
        if self.programs.is_empty() {
            return Err(AssetRefsError::NoPrograms);
        }

        let mut previous = None;
        for program in &self.programs {
            validate_name(
                &program.program_id,
                AssetRefsError::EmptyProgramId,
                AssetRefsError::InvalidProgramId,
            )?;
            if program.code_digest.is_zero() {
                return Err(AssetRefsError::ZeroCodeDigest(program.program_id.clone()));
            }
            if previous.is_some_and(|id: &str| id >= program.program_id.as_str()) {
                if previous == Some(program.program_id.as_str()) {
                    return Err(AssetRefsError::DuplicateProgramId(
                        program.program_id.clone(),
                    ));
                }
                return Err(AssetRefsError::ProgramsNotCanonical);
            }
            previous = Some(program.program_id.as_str());
        }

        let mut previous = None;
        for feature in &self.static_features {
            validate_name(
                feature,
                AssetRefsError::EmptyStaticFeature,
                AssetRefsError::InvalidStaticFeature,
            )?;
            if previous.is_some_and(|value: &str| value >= feature.as_str()) {
                if previous == Some(feature.as_str()) {
                    return Err(AssetRefsError::DuplicateStaticFeature(feature.clone()));
                }
                return Err(AssetRefsError::StaticFeaturesNotCanonical);
            }
            previous = Some(feature.as_str());
        }
        Ok(())
    }
}

fn validate_name(
    value: &str,
    empty: AssetRefsError,
    invalid: AssetRefsError,
) -> Result<(), AssetRefsError> {
    if value.is_empty() {
        return Err(empty);
    }
    if value
        .chars()
        .any(|character| character.is_control() || character.is_whitespace())
    {
        return Err(invalid);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refs() -> ShaderAssetRefs {
        ShaderAssetRefs {
            package_name: "prism.response".into(),
            programs: vec![ShaderProgramAssetRef {
                program_id: "lit".into(),
                code_digest: ArtifactDigest([1; 32]),
            }],
            static_features: vec!["object_space_cache".into()],
        }
    }

    #[test]
    fn accepts_only_canonical_portable_fields() {
        assert!(refs().validate().is_ok());
    }

    #[test]
    fn rejects_duplicate_programs_and_features() {
        let mut value = refs();
        value.programs.push(ShaderProgramAssetRef {
            program_id: "lit".into(),
            code_digest: ArtifactDigest([2; 32]),
        });
        assert_eq!(
            value.validate(),
            Err(AssetRefsError::DuplicateProgramId("lit".into()))
        );

        let mut value = refs();
        value.static_features.push("object_space_cache".into());
        assert_eq!(
            value.validate(),
            Err(AssetRefsError::DuplicateStaticFeature(
                "object_space_cache".into()
            ))
        );
    }

    #[test]
    fn rejects_zero_digest_and_noncanonical_order() {
        let mut value = refs();
        value.programs[0].code_digest = ArtifactDigest([0; 32]);
        assert_eq!(
            value.validate(),
            Err(AssetRefsError::ZeroCodeDigest("lit".into()))
        );

        let mut value = refs();
        value.programs.push(ShaderProgramAssetRef {
            program_id: "aaa".into(),
            code_digest: ArtifactDigest([2; 32]),
        });
        assert_eq!(value.validate(), Err(AssetRefsError::ProgramsNotCanonical));
    }
}
