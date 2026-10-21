use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use core::fmt;

use super::{instances::*, topology, types::*};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequirementKind {
    Hard,
    Extra(VariantId),
}

/// The statically validated owner of one semantic service.
///
/// Keeping the provider feature and its published output together prevents
/// consumers from accidentally treating a service ID as if it were enough to
/// locate the resource interface. `FeatureCatalog` constructs exactly one of
/// these values for every declared service ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServiceProvider {
    service: ServiceId,
    feature: FeatureId,
    output: &'static ServiceOutput,
}

impl ServiceProvider {
    pub const fn service(self) -> ServiceId {
        self.service
    }

    pub const fn feature(self) -> FeatureId {
        self.feature
    }

    pub const fn output(self) -> &'static ServiceOutput {
        self.output
    }
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
    InvalidPassAlternativeGroup {
        feature: FeatureId,
        group: usize,
    },
    DuplicateAlternativePass {
        feature: FeatureId,
        pass: PassKey,
    },
    UnknownAlternativePass {
        feature: FeatureId,
        pass: PassKey,
    },
    IncompleteAlternativeGroup {
        feature: FeatureId,
        variant: VariantId,
        missing: PassKey,
    },
    AmbiguousWriter {
        feature: FeatureId,
        variant: VariantId,
        resource: ResourceKey,
    },
    InvalidObjectAccess {
        pass: PassKey,
        ordinal: usize,
    },
    InconsistentObjectResource(ResourceKey),
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
    InvalidResourceInput {
        pass: PassKey,
        resource: ResourceKey,
    },
    InvalidDependency {
        pass: PassKey,
        dependency: PassKey,
    },
    UnsupportedQueueAccess {
        pass: PassKey,
        queue: QueueClass,
        access: AccessKind,
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
    providers: BTreeMap<ServiceId, ServiceProvider>,
    object_accesses: BTreeMap<(PassKey, usize), ObjectResourceAccess>,
}

impl FeatureCatalog {
    pub fn new(
        definitions: &'static [FeatureDefinition],
        resource_declarations: &'static [ResourceKey],
    ) -> Result<Self, CatalogError> {
        Self::new_with_object_accesses(definitions, resource_declarations, &[])
    }

    /// Validate independent object-role writer chains before frame compilation.
    /// Every use of an object key must be explicitly scoped. The compiler must
    /// receive these same declarations (in any order); absent entries are shared.
    /// `new` retains the original key-level shared writer contract.
    pub fn new_with_object_accesses(
        definitions: &'static [FeatureDefinition],
        resource_declarations: &'static [ResourceKey],
        object_accesses: &[ObjectResourceAccess],
    ) -> Result<Self, CatalogError> {
        let mut catalog = Self {
            features: BTreeMap::new(),
            resources: BTreeSet::new(),
            providers: BTreeMap::new(),
            object_accesses: BTreeMap::new(),
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
        catalog.validate_object_accesses(object_accesses)?;
        let mut variants = BTreeSet::new();
        let mut pass_keys: BTreeMap<PassKey, (FeatureId, &PassTemplate)> = BTreeMap::new();
        for &feature in catalog.features.values() {
            if matches!(feature.kind, FeatureKind::Subsystem) != feature.domain.is_some() {
                return Err(CatalogError::KindDomainMismatch(feature.id));
            }
            if feature.variants.is_empty() {
                return Err(CatalogError::NoVariants(feature.id));
            }
            let mut alternative_members = BTreeSet::new();
            for (index, group) in feature.pass_alternatives.iter().enumerate() {
                if group.len() < 2 {
                    return Err(CatalogError::InvalidPassAlternativeGroup {
                        feature: feature.id,
                        group: index,
                    });
                }
                for &pass in *group {
                    if !alternative_members.insert(pass) {
                        return Err(CatalogError::DuplicateAlternativePass {
                            feature: feature.id,
                            pass,
                        });
                    }
                    if !feature
                        .variants
                        .iter()
                        .any(|variant| variant.passes.iter().any(|candidate| candidate.id == pass))
                    {
                        return Err(CatalogError::UnknownAlternativePass {
                            feature: feature.id,
                            pass,
                        });
                    }
                }
                for variant in feature.variants {
                    let present = |key| variant.passes.iter().any(|pass| pass.id == key);
                    if group.iter().copied().any(present)
                        && let Some(&missing) = group.iter().find(|&&key| !present(key))
                    {
                        return Err(CatalogError::IncompleteAlternativeGroup {
                            feature: feature.id,
                            variant: variant.id,
                            missing,
                        });
                    }
                }
            }
            for output in feature.kind.provides() {
                let provider = ServiceProvider {
                    service: output.service,
                    feature: feature.id,
                    output,
                };
                if let Some(first) = catalog.providers.insert(output.service, provider) {
                    return Err(CatalogError::DuplicateProvider {
                        service: output.service,
                        first: first.feature(),
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
                let mut writer_inputs = BTreeMap::new();
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
                    let mut dependencies = BTreeSet::new();
                    for &dependency in pass.depends_on {
                        if dependency == pass.id || !dependencies.insert(dependency) {
                            return Err(CatalogError::InvalidDependency {
                                pass: pass.id,
                                dependency,
                            });
                        }
                    }
                    let mut pass_writes = BTreeSet::new();
                    for (ordinal, usage) in pass.accesses.iter().enumerate() {
                        if (pass.queue == QueueClass::Transfer
                            && !matches!(
                                usage.access,
                                AccessKind::TransferRead | AccessKind::TransferWrite
                            ))
                            || ((usage.access.is_attachment()
                                || usage.access == AccessKind::IndexRead)
                                && pass.queue != QueueClass::Graphics)
                        {
                            return Err(CatalogError::UnsupportedQueueAccess {
                                pass: pass.id,
                                queue: pass.queue,
                                access: usage.access,
                            });
                        }
                        catalog.validate_resource(usage.key)?;
                        if let ResourceInput::ProducedBy(producer) = usage.input
                            && (producer == pass.id || usage.frame_lag.is_some())
                        {
                            return Err(CatalogError::InvalidResourceInput {
                                pass: pass.id,
                                resource: usage.key,
                            });
                        }
                        if usage.access.writes() {
                            if usage.frame_lag.is_some() {
                                return Err(CatalogError::HistoryWrite {
                                    variant: variant.id,
                                    pass: pass.id,
                                    resource: usage.key,
                                });
                            }
                            let writer_key = (
                                usage.key,
                                catalog
                                    .object_accesses
                                    .get(&(pass.id, ordinal))
                                    .map(|scope| scope.role),
                            );
                            if !pass_writes.insert(writer_key) {
                                return Err(CatalogError::AmbiguousWriter {
                                    feature: feature.id,
                                    variant: variant.id,
                                    resource: usage.key,
                                });
                            }
                            writes.insert(usage.key);
                            writer_inputs
                                .entry(writer_key)
                                .or_insert_with(Vec::new)
                                .push((pass.id, usage.input));
                        }
                    }
                }
                for ((resource, _role), inputs) in writer_inputs {
                    if inputs.len() <= 1 {
                        continue;
                    }
                    let mut predecessors: BTreeMap<_, Vec<PassKey>> = BTreeMap::new();
                    for (writer, input) in inputs {
                        let predecessor = match input {
                            ResourceInput::Automatic => {
                                return Err(CatalogError::AmbiguousWriter {
                                    feature: feature.id,
                                    variant: variant.id,
                                    resource,
                                });
                            }
                            ResourceInput::Initial => None,
                            ResourceInput::ProducedBy(pass) => Some(pass),
                        };
                        let siblings = predecessors.entry(predecessor).or_default();
                        if siblings
                            .iter()
                            .any(|&other| !feature.passes_are_alternatives(writer, other))
                        {
                            return Err(CatalogError::AmbiguousWriter {
                                feature: feature.id,
                                variant: variant.id,
                                resource,
                            });
                        }
                        siblings.push(writer);
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

    fn validate_object_accesses(
        &mut self,
        declarations: &[ObjectResourceAccess],
    ) -> Result<(), CatalogError> {
        if declarations.is_empty() {
            return Ok(());
        }
        let mut passes = BTreeMap::new();
        for feature in self.features() {
            for variant in feature.variants {
                for pass in variant.passes {
                    passes.entry(pass.id).or_insert((feature, pass));
                }
            }
        }
        let mut resources = BTreeMap::new();
        for &scope in declarations {
            let invalid = CatalogError::InvalidObjectAccess {
                pass: scope.pass,
                ordinal: scope.ordinal,
            };
            let &(feature, pass) = passes.get(&scope.pass).ok_or_else(|| invalid.clone())?;
            let domain = feature.domain.ok_or_else(|| invalid.clone())?;
            let usage = pass
                .accesses
                .get(scope.ordinal)
                .ok_or_else(|| invalid.clone())?;
            self.validate_resource(usage.key)?;
            let ResourceOwner::Feature(owner) = usage.key.owner() else {
                return Err(invalid);
            };
            if self.feature(owner).and_then(|owner| owner.domain) != Some(domain)
                || matches!(scope.layout, ObjectLayoutPolicy::Sized { max_bytes: 0 })
                || self
                    .object_accesses
                    .insert((scope.pass, scope.ordinal), scope)
                    .is_some()
            {
                return Err(invalid);
            }
            let contract = (domain, scope.layout);
            if resources
                .insert(usage.key, contract)
                .is_some_and(|previous| previous != contract)
            {
                return Err(CatalogError::InconsistentObjectResource(usage.key));
            }
        }
        // A role is an explicit address dimension, never permission to split a
        // shared service or reinterpret an unscoped access to the same key.
        for feature in self.features() {
            for output in feature.kind.provides() {
                for key in output.resources {
                    if resources.contains_key(key) {
                        return Err(CatalogError::InconsistentObjectResource(*key));
                    }
                }
            }
            for variant in feature.variants {
                for pass in variant.passes {
                    for (ordinal, usage) in pass.accesses.iter().enumerate() {
                        if resources.contains_key(&usage.key)
                            && !self.object_accesses.contains_key(&(pass.id, ordinal))
                        {
                            return Err(CatalogError::InconsistentObjectResource(usage.key));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn object_accesses(&self) -> &BTreeMap<(PassKey, usize), ObjectResourceAccess> {
        &self.object_accesses
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

    /// Return the complete statically validated provider contract.
    pub fn service_provider(&self, service: ServiceId) -> Option<ServiceProvider> {
        self.providers.get(&service).copied()
    }

    pub fn provider(&self, service: ServiceId) -> Option<FeatureId> {
        self.service_provider(service).map(ServiceProvider::feature)
    }

    /// Return the provider's feature-level output interface.
    pub fn service_output(&self, service: ServiceId) -> Option<&'static ServiceOutput> {
        self.service_provider(service).map(ServiceProvider::output)
    }

    pub fn resources(&self) -> impl Iterator<Item = ResourceKey> + '_ {
        self.resources.iter().copied()
    }
}
