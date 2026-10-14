use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use core::fmt;

use super::{topology, types::*};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequirementKind {
    Hard,
    Extra(VariantId),
}

/// Semantic diagnostics: no frame-local resource or pass handles escape here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CatalogError {
    DuplicateFeature(FeatureId),
    DuplicateVariant(VariantId),
    DuplicateRank {
        feature: FeatureId,
        rank: u16,
    },
    DuplicateResource(ResourceKey),
    UnknownResourceOwner(ResourceKey),
    UndeclaredResource(ResourceKey),
    KindDomainMismatch(FeatureId),
    NoVariants(FeatureId),
    DuplicateProvider {
        service: ServiceId,
        first: FeatureId,
        second: FeatureId,
    },
    DuplicateRequirement {
        feature: FeatureId,
        requirement: RequirementKind,
        service: ServiceId,
    },
    MissingProvider {
        feature: FeatureId,
        requirement: RequirementKind,
        service: ServiceId,
    },
    DuplicateOutputResource {
        feature: FeatureId,
        service: ServiceId,
        resource: ResourceKey,
    },
    DuplicatePass {
        variant: VariantId,
        pass: PassKey,
    },
    ConflictingPassKey(PassKey),
    AmbiguousWriter {
        feature: FeatureId,
        variant: VariantId,
        resource: ResourceKey,
    },
    MissingOutputWrite {
        feature: FeatureId,
        variant: VariantId,
        service: ServiceId,
        resource: ResourceKey,
    },
    HistoryWrite {
        variant: VariantId,
        pass: PassKey,
        resource: ResourceKey,
    },
    InvalidShaderPermutation(VariantId),
    CyclicHardDependencies,
    TopologyCapacityExceeded,
    TopologyCompilationFailed(String),
}

impl fmt::Display for CatalogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "render fabric catalog error: {self:?}")
    }
}
impl std::error::Error for CatalogError {}

/// Validated static declarations and their unique provider index.
///
/// Resource declarations are an explicit semantic-key table: repeated *uses*
/// are normal, but a repeated *declaration* is a conflict. A pass key may be
/// shared by alternate variants of one feature only when its full template is
/// identical. Distinct features must not share a pass key.
#[derive(Debug)]
pub struct FeatureCatalog {
    features: BTreeMap<FeatureId, &'static FeatureDefinition>,
    resources: BTreeSet<ResourceKey>,
    providers: BTreeMap<ServiceId, FeatureId>,
}

impl FeatureCatalog {
    pub fn new(
        definitions: &'static [FeatureDefinition],
        resource_declarations: &'static [ResourceKey],
    ) -> Result<Self, CatalogError> {
        let mut catalog = Self {
            features: BTreeMap::new(),
            resources: BTreeSet::new(),
            providers: BTreeMap::new(),
        };
        for feature in definitions {
            if catalog.features.insert(feature.id, feature).is_some() {
                return Err(CatalogError::DuplicateFeature(feature.id));
            }
        }
        for &key in resource_declarations {
            if !catalog.resources.insert(key) {
                return Err(CatalogError::DuplicateResource(key));
            }
            if let ResourceOwner::Feature(owner) = key.owner()
                && !catalog.features.contains_key(&owner)
            {
                return Err(CatalogError::UnknownResourceOwner(key));
            }
        }
        let mut variants = BTreeSet::new();
        let mut pass_keys: BTreeMap<PassKey, (FeatureId, &PassTemplate)> = BTreeMap::new();
        for &feature in catalog.features.values() {
            if matches!(feature.kind, FeatureKind::Subsystem) != feature.domain.is_some() {
                return Err(CatalogError::KindDomainMismatch(feature.id));
            }
            if feature.variants.is_empty() {
                return Err(CatalogError::NoVariants(feature.id));
            }
            for output in feature.kind.provides() {
                if let Some(first) = catalog.providers.insert(output.service, feature.id) {
                    return Err(CatalogError::DuplicateProvider {
                        service: output.service,
                        first,
                        second: feature.id,
                    });
                }
                let mut keys = BTreeSet::new();
                for &resource in output.resources {
                    catalog.validate_resource(resource)?;
                    if !keys.insert(resource) {
                        return Err(CatalogError::DuplicateOutputResource {
                            feature: feature.id,
                            service: output.service,
                            resource,
                        });
                    }
                }
            }
            let mut ranks = BTreeSet::new();
            for variant in feature.variants {
                if !variants.insert(variant.id) {
                    return Err(CatalogError::DuplicateVariant(variant.id));
                }
                if !ranks.insert(variant.rank) {
                    return Err(CatalogError::DuplicateRank {
                        feature: feature.id,
                        rank: variant.rank,
                    });
                }
                if let Some(binding) = variant.shader_permutation
                    && binding.space.index_of(binding.key).is_none()
                {
                    return Err(CatalogError::InvalidShaderPermutation(variant.id));
                }
                let mut local_passes = BTreeSet::new();
                let mut writes = BTreeSet::new();
                for pass in variant.passes {
                    if !local_passes.insert(pass.id) {
                        return Err(CatalogError::DuplicatePass {
                            variant: variant.id,
                            pass: pass.id,
                        });
                    }
                    if let Some(&(owner, previous)) = pass_keys.get(&pass.id) {
                        if owner != feature.id || previous != pass {
                            return Err(CatalogError::ConflictingPassKey(pass.id));
                        }
                    } else {
                        pass_keys.insert(pass.id, (feature.id, pass));
                    }
                    for usage in pass.accesses {
                        catalog.validate_resource(usage.key)?;
                        if usage.access.writes() {
                            if usage.frame_lag.is_some() {
                                return Err(CatalogError::HistoryWrite {
                                    variant: variant.id,
                                    pass: pass.id,
                                    resource: usage.key,
                                });
                            }
                            if !writes.insert(usage.key) {
                                return Err(CatalogError::AmbiguousWriter {
                                    feature: feature.id,
                                    variant: variant.id,
                                    resource: usage.key,
                                });
                            }
                        }
                    }
                }
                for output in feature.kind.provides() {
                    for &resource in output.resources {
                        if !writes.contains(&resource) {
                            return Err(CatalogError::MissingOutputWrite {
                                feature: feature.id,
                                variant: variant.id,
                                service: output.service,
                                resource,
                            });
                        }
                    }
                }
            }
        }
        for &feature in catalog.features.values() {
            catalog.validate_requirements(
                feature.id,
                RequirementKind::Hard,
                feature.requires_services,
            )?;
            for variant in feature.variants {
                catalog.validate_requirements(
                    feature.id,
                    RequirementKind::Extra(variant.id),
                    variant.requires_services_extra,
                )?;
            }
        }
        // Validation and selection both use the one authoritative scheduler.
        topology::order(&catalog, &catalog.features.keys().copied().collect())?;
        Ok(catalog)
    }

    fn validate_resource(&self, key: ResourceKey) -> Result<(), CatalogError> {
        if self.resources.contains(&key) {
            Ok(())
        } else {
            Err(CatalogError::UndeclaredResource(key))
        }
    }

    fn validate_requirements(
        &self,
        feature: FeatureId,
        requirement: RequirementKind,
        services: ServiceSet,
    ) -> Result<(), CatalogError> {
        let mut seen = BTreeSet::new();
        for &service in services.as_slice() {
            if !seen.insert(service) {
                return Err(CatalogError::DuplicateRequirement {
                    feature,
                    requirement,
                    service,
                });
            }
            // Only FeatureKind::Service populated this index. Thus every edge
            // target is a Service by construction, including soft requirements.
            if !self.providers.contains_key(&service) {
                return Err(CatalogError::MissingProvider {
                    feature,
                    requirement,
                    service,
                });
            }
        }
        Ok(())
    }

    pub fn feature(&self, id: FeatureId) -> Option<&'static FeatureDefinition> {
        self.features.get(&id).copied()
    }

    pub fn features(&self) -> impl Iterator<Item = &'static FeatureDefinition> + '_ {
        self.features.values().copied()
    }

    pub fn provider(&self, service: ServiceId) -> Option<FeatureId> {
        self.providers.get(&service).copied()
    }

    pub fn resources(&self) -> impl Iterator<Item = ResourceKey> + '_ {
        self.resources.iter().copied()
    }
}
