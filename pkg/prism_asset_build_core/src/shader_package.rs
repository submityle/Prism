//! Asset-build boundary for portable shader packages.
//!
//! `ShaderPackageAsset` is the only build-side bridge that combines portable
//! `asset_refs`, the complete variant manifest, and an artifact descriptor.
//! It validates the portable contract before exposing the descriptor to DDC
//! or Manifest construction. Driver-local module and pipeline-cache state has
//! no field in this type and is rejected by the descriptor storage gate.

use prism_render_architecture::abi::{AbiHash, AbiVersion};
use prism_render_architecture::shader_package::{
    AssetRefsError, ShaderAssetRefs, ShaderPackageVariantManifest, VariantManifestError,
};

use crate::types::{ArtifactDescriptor, ArtifactStorageClass, AssetId, ManifestEntry};

/// A validated, portable shader-package asset record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderPackageAsset {
    pub package_name: String,
    pub source_revision: String,
    pub entry_point: String,
    pub abi_version: AbiVersion,
    pub abi_hash: AbiHash,
    pub asset_refs: ShaderAssetRefs,
    pub variants: ShaderPackageVariantManifest,
    descriptor: ArtifactDescriptor,
}

/// Failure at the shader-package asset/DDC/Manifest boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShaderPackageAssetError {
    EmptyPackageName,
    EmptySourceRevision,
    EmptyEntryPoint,
    PackageNameMismatch,
    AssetRefs(AssetRefsError),
    Variants(VariantManifestError),
    InvalidDescriptor(&'static str),
    DriverLocalArtifact,
}

impl core::fmt::Display for ShaderPackageAssetError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyPackageName => f.write_str("shader package asset has an empty package name"),
            Self::EmptySourceRevision => {
                f.write_str("shader package asset has an empty source revision")
            }
            Self::EmptyEntryPoint => f.write_str("shader package asset has an empty entry point"),
            Self::PackageNameMismatch => {
                f.write_str("shader package name differs from asset_refs package name")
            }
            Self::AssetRefs(error) => write!(f, "shader package asset_refs are invalid: {error}"),
            Self::Variants(error) => {
                write!(f, "shader package variant manifest is invalid: {error}")
            }
            Self::InvalidDescriptor(reason) => {
                write!(f, "shader package artifact descriptor is invalid: {reason}")
            }
            Self::DriverLocalArtifact => {
                f.write_str("driver-local shader data cannot enter DDC or Manifest")
            }
        }
    }
}

impl std::error::Error for ShaderPackageAssetError {}

impl ShaderPackageAsset {
    /// Validates one package record against its complete permutation space.
    pub fn new(
        package_name: impl Into<String>,
        source_revision: impl Into<String>,
        entry_point: impl Into<String>,
        abi_version: AbiVersion,
        abi_hash: AbiHash,
        asset_refs: ShaderAssetRefs,
        variants: ShaderPackageVariantManifest,
        descriptor: ArtifactDescriptor,
        space: &prism_render_architecture::shader_package::PermutationSpace,
    ) -> Result<Self, ShaderPackageAssetError> {
        let package_name = package_name.into();
        let source_revision = source_revision.into();
        let entry_point = entry_point.into();
        if package_name.is_empty() {
            return Err(ShaderPackageAssetError::EmptyPackageName);
        }
        if source_revision.is_empty() {
            return Err(ShaderPackageAssetError::EmptySourceRevision);
        }
        if entry_point.is_empty() {
            return Err(ShaderPackageAssetError::EmptyEntryPoint);
        }
        if asset_refs.package_name != package_name {
            return Err(ShaderPackageAssetError::PackageNameMismatch);
        }
        asset_refs
            .validate()
            .map_err(ShaderPackageAssetError::AssetRefs)?;
        variants
            .validate_against(space)
            .map_err(ShaderPackageAssetError::Variants)?;
        if descriptor.storage_class != ArtifactStorageClass::Shared {
            return Err(ShaderPackageAssetError::DriverLocalArtifact);
        }
        descriptor
            .validate()
            .map_err(ShaderPackageAssetError::InvalidDescriptor)?;
        Ok(Self {
            package_name,
            source_revision,
            entry_point,
            abi_version,
            abi_hash,
            asset_refs,
            variants,
            descriptor,
        })
    }

    /// Returns the only descriptor that may be inserted into shared DDC.
    #[must_use]
    pub fn descriptor(&self) -> &ArtifactDescriptor {
        &self.descriptor
    }

    /// Produces a build-manifest entry after the same storage-class check.
    pub fn manifest_entry(&self, asset: AssetId) -> Result<ManifestEntry, ShaderPackageAssetError> {
        if self.descriptor.storage_class != ArtifactStorageClass::Shared {
            return Err(ShaderPackageAssetError::DriverLocalArtifact);
        }
        Ok(ManifestEntry {
            asset,
            descriptor: self.descriptor.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ddc::{DdcError, DerivedDataCache};
    use crate::types::{ArtifactKey, BuildRevision, Manifest, ManifestEntry};
    use prism_render_architecture::shader_package::{
        ArtifactDigest, PermutationSpace, ShaderPackageVariantManifest, ShaderProgramAssetRef,
    };
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEST_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "prism-shader-package-gate-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            TEST_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn valid_asset(descriptor: ArtifactDescriptor) -> ShaderPackageAsset {
        let space = PermutationSpace::new([2, 2]);
        let refs = ShaderAssetRefs {
            package_name: "prism.response".into(),
            programs: vec![ShaderProgramAssetRef {
                program_id: "lit".into(),
                code_digest: ArtifactDigest([1; 32]),
            }],
            static_features: vec!["object_space_cache".into()],
        };
        let variants = ShaderPackageVariantManifest::from_digests(
            &space,
            [
                ArtifactDigest([2; 32]),
                ArtifactDigest([3; 32]),
                ArtifactDigest([4; 32]),
                ArtifactDigest([5; 32]),
            ],
        )
        .unwrap();
        ShaderPackageAsset::new(
            "prism.response",
            "source-1",
            "main",
            AbiVersion::from_parts(1, 0),
            AbiHash::from_bytes(b"abi"),
            refs,
            variants,
            descriptor,
            &space,
        )
        .unwrap()
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_dir_all(path);
    }

    #[test]
    fn valid_asset_refs_and_all_variants_cross_only_as_shared_descriptor() {
        let descriptor = ArtifactDescriptor::for_bytes(
            ArtifactKey([1; 32]),
            7,
            1,
            16,
            Vec::new(),
            b"shader-package",
        );
        let asset = valid_asset(descriptor.clone());
        let entry = asset.manifest_entry(AssetId(9)).unwrap();
        let manifest = Manifest::new(BuildRevision(1), [entry]).unwrap();
        assert_eq!(manifest.entries[&AssetId(9)], descriptor);
    }

    #[test]
    fn asset_refs_not_in_ddc_or_manifest() {
        let descriptor = ArtifactDescriptor::for_bytes(
            ArtifactKey([9; 32]),
            7,
            1,
            16,
            Vec::new(),
            b"shader-package-payload",
        );
        let asset = valid_asset(descriptor.clone());
        let manifest = Manifest::new(
            BuildRevision(2),
            [asset.manifest_entry(AssetId(10)).unwrap()],
        )
        .unwrap();
        let manifest_bytes = manifest.canonical_bytes();

        let dir = temp_dir();
        let mut ddc = DerivedDataCache::open(&dir).unwrap();
        ddc.put_descriptor(descriptor).unwrap();
        let ddc_bytes = std::fs::read(ddc.index_path()).unwrap();
        cleanup(&dir);

        for portable_field in [
            asset.package_name.as_bytes(),
            asset.asset_refs.package_name.as_bytes(),
            asset.asset_refs.programs[0].program_id.as_bytes(),
            asset.asset_refs.static_features[0].as_bytes(),
        ] {
            assert!(!manifest_bytes
                .windows(portable_field.len())
                .any(|window| window == portable_field));
            assert!(!ddc_bytes
                .windows(portable_field.len())
                .any(|window| window == portable_field));
        }
    }

    #[test]
    fn driver_local_artifacts_are_rejected_by_asset_manifest_and_ddc() {
        let descriptor = ArtifactDescriptor::for_driver_local_bytes(
            ArtifactKey([2; 32]),
            8,
            1,
            16,
            Vec::new(),
            b"pso",
        );
        assert_eq!(
            ShaderPackageAsset::new(
                "prism.response",
                "source-1",
                "main",
                AbiVersion::from_parts(1, 0),
                AbiHash::from_bytes(b"abi"),
                valid_asset(ArtifactDescriptor::for_bytes(
                    ArtifactKey([3; 32]),
                    7,
                    1,
                    16,
                    Vec::new(),
                    b"shader-package",
                ))
                .asset_refs,
                valid_asset(ArtifactDescriptor::for_bytes(
                    ArtifactKey([4; 32]),
                    7,
                    1,
                    16,
                    Vec::new(),
                    b"shader-package",
                ))
                .variants,
                descriptor.clone(),
                &PermutationSpace::new([2, 2]),
            ),
            Err(ShaderPackageAssetError::DriverLocalArtifact)
        );

        assert!(matches!(
            Manifest::new(
                BuildRevision(1),
                [ManifestEntry {
                    asset: AssetId(1),
                    descriptor: descriptor.clone(),
                }]
            ),
            Err("manifest cannot contain driver-local artifacts")
        ));

        let dir = temp_dir();
        let mut ddc = DerivedDataCache::open(&dir).unwrap();
        assert!(matches!(
            ddc.put_descriptor(descriptor),
            Err(DdcError::DriverLocalArtifact { .. })
        ));
        cleanup(&dir);
    }

    #[test]
    fn malformed_variant_manifest_is_rejected_before_shared_storage() {
        let space = PermutationSpace::new([2, 2]);
        let variants = ShaderPackageVariantManifest::from_digests(
            &space,
            [
                ArtifactDigest([1; 32]),
                ArtifactDigest([2; 32]),
                ArtifactDigest([3; 32]),
            ],
        );
        assert!(matches!(
            variants,
            Err(VariantManifestError::CountMismatch { .. })
        ));
    }

    #[test]
    fn invalid_descriptor_is_not_reported_as_driver_local() {
        let descriptor = ArtifactDescriptor::for_bytes(
            ArtifactKey([8; 32]),
            7,
            1,
            3,
            Vec::new(),
            b"invalid-alignment",
        );
        let error = ShaderPackageAsset::new(
            "prism.response",
            "source-1",
            "main",
            AbiVersion::from_parts(1, 0),
            AbiHash::from_bytes(b"abi"),
            valid_asset(ArtifactDescriptor::for_bytes(
                ArtifactKey([3; 32]),
                7,
                1,
                16,
                Vec::new(),
                b"shader-package",
            ))
            .asset_refs,
            valid_asset(ArtifactDescriptor::for_bytes(
                ArtifactKey([4; 32]),
                7,
                1,
                16,
                Vec::new(),
                b"shader-package",
            ))
            .variants,
            descriptor,
            &PermutationSpace::new([2, 2]),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ShaderPackageAssetError::InvalidDescriptor("alignment must be a non-zero power of two")
        );
    }
}
