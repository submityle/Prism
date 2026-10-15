use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use core::fmt;

use super::{topology, types::*, CatalogError, FeatureCatalog};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RootError {
    UnknownFeature(FeatureId),
    ExpectedSubsystem(FeatureId),
    ExpectedResponse(FeatureId),
    OutsideFrozenRoots(FeatureId),
}

impl fmt::Display for RootError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "render fabric root error: {self:?}")
    }
}
impl std::error::Error for RootError {}

/// Construct once from the build profile's enabled subsystems and statically
/// referenceable response programs, not from visible objects or frame metrics.
/// There is intentionally no API to insert/remove roots.
///
/// ```compile_fail
/// use prism_render_architecture::render_fabric::{FrozenRoots, FeatureId};
/// fn expand(roots: &mut FrozenRoots) { roots.ids.push(FeatureId(5)); }
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenRoots {
    ids: Vec<FeatureId>,
    subsystems: Vec<FeatureId>,
    responses: Vec<FeatureId>,
}

impl FrozenRoots {
    pub fn from_build_profile(
        catalog: &FeatureCatalog,
        enabled_subsystems: &'static [FeatureId],
        referenceable_responses: &'static [FeatureId],
    ) -> Result<Self, RootError> {
        let subsystems: BTreeSet<_> = enabled_subsystems.iter().copied().collect();
        let responses: BTreeSet<_> = referenceable_responses.iter().copied().collect();
        let roots = Self {
            ids: subsystems.union(&responses).copied().collect(),
            subsystems: subsystems.into_iter().collect(),
            responses: responses.into_iter().collect(),
        };
        roots.validate_catalog(catalog)?;
        Ok(roots)
    }

    fn validate_catalog(&self, catalog: &FeatureCatalog) -> Result<(), RootError> {
        for &id in &self.subsystems {
            let feature = catalog.feature(id).ok_or(RootError::UnknownFeature(id))?;
            if !matches!(feature.kind, FeatureKind::Subsystem) {
                return Err(RootError::ExpectedSubsystem(id));
            }
        }
        for &id in &self.responses {
            let feature = catalog.feature(id).ok_or(RootError::UnknownFeature(id))?;
            if !matches!(feature.kind, FeatureKind::Response) {
                return Err(RootError::ExpectedResponse(id));
            }
        }
        Ok(())
    }

    pub fn ids(&self) -> &[FeatureId] {
        &self.ids
    }

    /// Use at runtime when a material/program becomes referenceable. Membership
    /// in the dependency closure is NOT membership in the frozen root set.
    pub fn require_root(&self, id: FeatureId) -> Result<(), RootError> {
        if self.ids.binary_search(&id).is_ok() {
            Ok(())
        } else {
            Err(RootError::OutsideFrozenRoots(id))
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateRejection {
    pub variant: VariantId,
    pub missing_caps: CapabilitySet,
    pub required_quality: Option<RenderQualityTier>,
    pub unavailable_services: Vec<ServiceId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SelectionError {
    Roots(RootError),
    Catalog(CatalogError),
    NoAdmissibleVariant {
        feature: FeatureId,
        rejected: Vec<CandidateRejection>,
    },
}

impl fmt::Display for SelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "render fabric selection error: {self:?}")
    }
}
impl std::error::Error for SelectionError {}

#[derive(Clone, Debug)]
pub struct SelectedFeature {
    feature: FeatureId,
    variant: &'static FeatureVariant,
    definition: &'static FeatureDefinition,
}

// Equality describes the chosen implementation, not the registration order of
// its other candidates. Keep provenance validation separate from this semantic
// equality so shuffled declarations still produce equal selections.
impl PartialEq for SelectedFeature {
    fn eq(&self, other: &Self) -> bool {
        self.feature == other.feature && self.variant == other.variant
    }
}
impl Eq for SelectedFeature {}

impl SelectedFeature {
    pub(super) fn matches_definition(&self, definition: &FeatureDefinition) -> bool {
        // A shared variant alone is not proof of the same hard dependencies,
        // provider interface, root kind/domain, or candidate set. Never resolve
        // again here: consumers validate the frozen source declaration instead.
        (core::ptr::eq(self.definition, definition) || self.definition == definition)
            && definition
                .variants
                .iter()
                .any(|variant| core::ptr::eq(variant, self.variant))
    }

    pub const fn feature_id(&self) -> FeatureId {
        self.feature
    }
    pub const fn variant_id(&self) -> VariantId {
        self.variant.id
    }
    pub const fn variant(&self) -> &'static FeatureVariant {
        self.variant
    }
    pub const fn passes(&self) -> &'static [PassTemplate] {
        self.variant.passes
    }
}

/// An immutable, device-local result. Clone/read access cannot mutate the
/// selection. All consumers must share this result for the device lifetime.
///
/// ```compile_fail
/// use prism_render_architecture::render_fabric::FeatureSelection;
/// fn clear(selection: &mut FeatureSelection) { selection.selected.clear(); }
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeatureSelection {
    roots: FrozenRoots,
    device_caps: CapabilitySet,
    quality: RenderQualityTier,
    closure: Vec<FeatureId>,
    selected: Vec<SelectedFeature>,
    services: BTreeMap<ServiceId, (FeatureId, Vec<ResourceKey>)>,
}

impl FeatureSelection {
    /// Resolve once at device initialization. No frame inputs or backtracking.
    pub fn resolve(
        catalog: &FeatureCatalog,
        active_roots: &FrozenRoots,
        device_caps: CapabilitySet,
        quality: RenderQualityTier,
    ) -> Result<Self, SelectionError> {
        active_roots
            .validate_catalog(catalog)
            .map_err(SelectionError::Roots)?;
        let mut closure: BTreeSet<_> = active_roots.ids.iter().copied().collect();
        let mut pending = active_roots.ids.clone();
        while let Some(id) = pending.pop() {
            let feature = catalog.feature(id).expect("validated root or provider");
            for &service in feature.requires_services.as_slice() {
                let provider = catalog.provider(service).expect("validated provider");
                if closure.insert(provider) {
                    pending.push(provider);
                }
            }
        }
        let order = topology::order(catalog, &closure).map_err(SelectionError::Catalog)?;
        let mut result = Self {
            roots: active_roots.clone(),
            device_caps,
            quality,
            closure: closure.into_iter().collect(),
            selected: Vec::new(),
            services: BTreeMap::new(),
        };
        for id in order {
            let feature = catalog.feature(id).expect("validated closure");
            let mut candidates: Vec<_> = feature.variants.iter().collect();
            candidates.sort_by_key(|variant| (variant.rank, variant.id));
            let mut rejected = Vec::new();
            let mut chosen = None;
            for variant in candidates {
                let mut unavailable_services: Vec<_> = variant
                    .requires_services_extra
                    .as_slice()
                    .iter()
                    .copied()
                    .filter(|service| !result.services.contains_key(service))
                    .collect();
                unavailable_services.sort();
                let missing_caps = variant.required_caps.difference(device_caps);
                let required_quality =
                    (variant.min_quality > quality).then_some(variant.min_quality);
                if missing_caps.is_empty()
                    && required_quality.is_none()
                    && unavailable_services.is_empty()
                {
                    chosen = Some(variant);
                    break;
                }
                rejected.push(CandidateRejection {
                    variant: variant.id,
                    missing_caps,
                    required_quality,
                    unavailable_services,
                });
            }
            let variant = chosen.ok_or(SelectionError::NoAdmissibleVariant {
                feature: id,
                rejected,
            })?;
            result.selected.push(SelectedFeature {
                feature: id,
                variant,
                definition: feature,
            });
            for output in feature.kind.provides() {
                let mut resources = output.resources.to_vec();
                resources.sort();
                result.services.insert(output.service, (id, resources));
            }
        }
        Ok(result)
    }

    pub const fn roots(&self) -> &FrozenRoots {
        &self.roots
    }
    pub const fn device_caps(&self) -> CapabilitySet {
        self.device_caps
    }
    pub const fn quality(&self) -> RenderQualityTier {
        self.quality
    }
    /// Stable ID-sorted hard closure, independent of device/quality/extras.
    pub fn closure(&self) -> &[FeatureId] {
        &self.closure
    }
    /// Layered topological solve order, with `FeatureId` as the ready-node tie break.
    pub fn selected(&self) -> &[SelectedFeature] {
        &self.selected
    }
    pub fn feature(&self, id: FeatureId) -> Option<&SelectedFeature> {
        self.selected.iter().find(|selected| selected.feature == id)
    }
    pub fn provider(&self, service: ServiceId) -> Option<FeatureId> {
        self.services.get(&service).map(|(provider, _)| *provider)
    }
    pub fn service_resources(&self, service: ServiceId) -> Option<&[ResourceKey]> {
        self.services
            .get(&service)
            .map(|(_, resources)| resources.as_slice())
    }
    /// Selected pass declarations only; not an executable/resource-bound graph.
    pub fn passes(&self) -> impl Iterator<Item = (FeatureId, &'static PassTemplate)> + '_ {
        self.selected.iter().flat_map(|selected| {
            selected
                .passes()
                .iter()
                .map(move |pass| (selected.feature, pass))
        })
    }
}
