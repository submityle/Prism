//! PCM identity/capability bridge. The registry and manifests remain unchanged.
//! Graph declarations stay in the static sidecar and FeatureCatalog, not PCM.
use super::*;
use crate::backend::capability::Capability as DeviceCapability;
use alloc::collections::BTreeMap;
use core::fmt;
use prism_material_pcm::{
    Capability as PcmCapability, CapabilitySet as PcmCapabilities, ExtensionManifest,
    ExtensionRegistry, Kind as PcmKind, ManifestError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcmCapabilityError {
    UnknownBits(u32),
    Unrepresentable(PcmCapability),
}

/// Explicit, lossless conversion of representable device requirements.
/// ComputeScatter uses WebGPU compute's baseline integer storage atomics.
/// VT/cache/f16/heavy-sampling have no equivalent in the device vocabulary;
/// they must be negotiated by PCM, never erased or guessed from Compute.
pub fn pcm_device_requirements(
    requirements: PcmCapabilities,
) -> Result<CapabilitySet, PcmCapabilityError> {
    let unknown = requirements.bits() & !PcmCapabilities::full().bits();
    if unknown != 0 {
        return Err(PcmCapabilityError::UnknownBits(unknown));
    }
    let mut result = CapabilitySet::EMPTY;
    for capability in requirements.iter() {
        let device = match capability {
            PcmCapability::BindlessTextures => DeviceCapability::BindlessDescriptors,
            PcmCapability::ComputeScatter => DeviceCapability::Compute,
            other => return Err(PcmCapabilityError::Unrepresentable(other)),
        };
        result = result.union(CapabilitySet::from_slice(&[device]));
    }
    Ok(result)
}

/// Authored UID-to-feature relation, independent of mutable attribution,
/// registry slots, registration order and the manifest's derived display ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PcmFeatureBinding {
    pub uid: &'static str,
    pub feature: FeatureId,
    pub kind: PcmKind,
    pub version: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PcmBridgeError {
    Catalog(CatalogError),
    Manifest(ManifestError),
    DuplicateUid(&'static str),
    DuplicateFeature(FeatureId),
    UnknownUid(&'static str),
    UnknownFeature(FeatureId),
    KindMismatch {
        uid: &'static str,
        expected: PcmKind,
        actual: PcmKind,
    },
    VersionMismatch {
        uid: &'static str,
        expected: u32,
        actual: u32,
    },
    NotGraphExtension {
        uid: &'static str,
        kind: PcmKind,
    },
    FeatureKindMismatch {
        uid: &'static str,
        feature: FeatureId,
    },
    Capabilities {
        uid: &'static str,
        error: PcmCapabilityError,
    },
    MissingVariantRequirements {
        uid: &'static str,
        feature: FeatureId,
        variant: VariantId,
        missing: CapabilitySet,
    },
}
impl fmt::Display for PcmBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PCM feature bridge error: {self:?}")
    }
}
impl core::error::Error for PcmBridgeError {}

/// Immutable bridge product. It neither registers extensions nor allocates
/// semantic IDs. Unmapped PCM rows stay PCM concerns, not implicit graph roots.
#[derive(Debug)]
pub struct PcmFeatureCatalog {
    catalog: FeatureCatalog,
    manifests: BTreeMap<&'static str, ExtensionManifest>,
    features: BTreeMap<&'static str, FeatureId>,
}
impl PcmFeatureCatalog {
    pub fn new(
        registry: &ExtensionRegistry,
        bindings: &'static [PcmFeatureBinding],
        definitions: &'static [FeatureDefinition],
        keys: &'static [ResourceKey],
    ) -> Result<Self, PcmBridgeError> {
        let catalog = FeatureCatalog::new(definitions, keys).map_err(PcmBridgeError::Catalog)?;
        let mut result = Self {
            catalog,
            manifests: BTreeMap::new(),
            features: BTreeMap::new(),
        };
        let mut reverse = BTreeMap::new();
        for binding in bindings {
            if result
                .features
                .insert(binding.uid, binding.feature)
                .is_some()
            {
                return Err(PcmBridgeError::DuplicateUid(binding.uid));
            }
            if reverse.insert(binding.feature, binding.uid).is_some() {
                return Err(PcmBridgeError::DuplicateFeature(binding.feature));
            }
            let manifest = *registry
                .manifest_by_uid(binding.uid)
                .ok_or(PcmBridgeError::UnknownUid(binding.uid))?;
            manifest.validate().map_err(PcmBridgeError::Manifest)?;
            if manifest.kind() != binding.kind {
                return Err(PcmBridgeError::KindMismatch {
                    uid: binding.uid,
                    expected: binding.kind,
                    actual: manifest.kind(),
                });
            }
            if manifest.version() != binding.version {
                return Err(PcmBridgeError::VersionMismatch {
                    uid: binding.uid,
                    expected: binding.version,
                    actual: manifest.version(),
                });
            }
            let feature = result
                .catalog
                .feature(binding.feature)
                .ok_or(PcmBridgeError::UnknownFeature(binding.feature))?;
            let compatible = match manifest.kind() {
                PcmKind::Provider => matches!(feature.kind, FeatureKind::Service { .. }),
                PcmKind::ResponseMode | PcmKind::Composite => feature.kind == FeatureKind::Response,
                kind => {
                    return Err(PcmBridgeError::NotGraphExtension {
                        uid: binding.uid,
                        kind,
                    })
                }
            };
            if !compatible {
                return Err(PcmBridgeError::FeatureKindMismatch {
                    uid: binding.uid,
                    feature: binding.feature,
                });
            }
            let required = pcm_device_requirements(manifest.requires_caps()).map_err(|error| {
                PcmBridgeError::Capabilities {
                    uid: binding.uid,
                    error,
                }
            })?;
            // Do not mutate static variants or invent a second runtime gate.
            // Each authored candidate must already carry the manifest needs,
            // so the existing resolver rejects devices that cannot run it.
            for variant in feature.variants {
                let missing = variant.required_caps.missing_from(required);
                if !missing.is_empty() {
                    return Err(PcmBridgeError::MissingVariantRequirements {
                        uid: binding.uid,
                        feature: feature.id,
                        variant: variant.id,
                        missing,
                    });
                }
            }
            result.manifests.insert(binding.uid, manifest);
        }
        Ok(result)
    }
    pub const fn catalog(&self) -> &FeatureCatalog {
        &self.catalog
    }
    pub fn feature(&self, uid: &str) -> Option<&'static FeatureDefinition> {
        self.features
            .get(uid)
            .and_then(|&id| self.catalog.feature(id))
    }
    pub fn manifest(&self, uid: &str) -> Option<&ExtensionManifest> {
        self.manifests.get(uid)
    }
    /// Stable UID order, not registry iteration order.
    pub fn bindings(&self) -> impl Iterator<Item = (&str, FeatureId)> {
        self.features.iter().map(|(&uid, &feature)| (uid, feature))
    }
}
