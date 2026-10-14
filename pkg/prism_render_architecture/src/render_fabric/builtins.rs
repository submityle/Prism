//! The production render-feature declarations used to bootstrap the device.
//!
//! This is intentionally a small semantic catalog: providers publish service
//! identities, while water/volume and the three response arms declare the
//! roots that consume them. Frame resources remain owned by the selected pass
//! templates and the authoritative frame graph.

use super::{
    FeatureCatalog, FeatureDefinition, FeatureId, FeatureVariant, RenderDomain,
    RenderQualityTier, ServiceId, ServiceOutput, ServiceSet, VariantId,
};
use crate::backend::capability::CapabilitySet;

pub const WATER_SERVICE_IDS: [ServiceId; 9] = [
    ServiceId(1),
    ServiceId(2),
    ServiceId(3),
    ServiceId(4),
    ServiceId(5),
    ServiceId(6),
    ServiceId(7),
    ServiceId(8),
    ServiceId(9),
];

pub const VOLUME_SERVICE_IDS: [ServiceId; 7] = [
    ServiceId(101),
    ServiceId(102),
    ServiceId(103),
    ServiceId(104),
    ServiceId(105),
    ServiceId(106),
    ServiceId(107),
];

pub const WATER_FEATURE_ID: FeatureId = FeatureId(0x1000);
pub const VOLUME_FEATURE_ID: FeatureId = FeatureId(0x1001);
pub const LIT_FEATURE_ID: FeatureId = FeatureId(0x1100);
pub const NPR_FEATURE_ID: FeatureId = FeatureId(0x1101);
pub const CUSTOM_FEATURE_ID: FeatureId = FeatureId(0x1102);

const WATER_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: VariantId(0x5000),
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: &[],
    shader_permutation: None,
}];
const VOLUME_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: VariantId(0x5001),
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: &[],
    shader_permutation: None,
}];
const LIT_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: VariantId(0x5002),
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: &[],
    shader_permutation: None,
}];
const NPR_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: VariantId(0x5003),
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: &[],
    shader_permutation: None,
}];
const CUSTOM_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: VariantId(0x5004),
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: &[],
    shader_permutation: None,
}];

macro_rules! service_definition {
    ($feature:expr, $variant:expr, $service:expr) => {{
        // These built-ins establish the stable semantic provider closure. The
        // concrete frame resources are declared by the feature pass templates
        // that consume the selected closure; leaving this bootstrap layer
        // resource-empty prevents it from inventing a second frame-graph ABI.
        const OUTPUTS: &[ServiceOutput] = &[ServiceOutput {
            service: $service,
            resources: &[],
        }];
        FeatureDefinition::service(
            $feature,
            OUTPUTS,
            ServiceSet::EMPTY,
            &[$crate::render_fabric::FeatureVariant {
                id: $variant,
                rank: 0,
                min_quality: $crate::render_fabric::RenderQualityTier::Low,
                required_caps: $crate::backend::capability::CapabilitySet::EMPTY,
                requires_services_extra: $crate::render_fabric::ServiceSet::EMPTY,
                passes: &[],
                shader_permutation: None,
            }],
        )
    }};
}

const SERVICE_FEATURES: [FeatureDefinition; 16] = [
    service_definition!(FeatureId(0x2000), VariantId(0x6000), WATER_SERVICE_IDS[0]),
    service_definition!(FeatureId(0x2001), VariantId(0x6001), WATER_SERVICE_IDS[1]),
    service_definition!(FeatureId(0x2002), VariantId(0x6002), WATER_SERVICE_IDS[2]),
    service_definition!(FeatureId(0x2003), VariantId(0x6003), WATER_SERVICE_IDS[3]),
    service_definition!(FeatureId(0x2004), VariantId(0x6004), WATER_SERVICE_IDS[4]),
    service_definition!(FeatureId(0x2005), VariantId(0x6005), WATER_SERVICE_IDS[5]),
    service_definition!(FeatureId(0x2006), VariantId(0x6006), WATER_SERVICE_IDS[6]),
    service_definition!(FeatureId(0x2007), VariantId(0x6007), WATER_SERVICE_IDS[7]),
    service_definition!(FeatureId(0x2008), VariantId(0x6008), WATER_SERVICE_IDS[8]),
    service_definition!(FeatureId(0x2010), VariantId(0x6010), VOLUME_SERVICE_IDS[0]),
    service_definition!(FeatureId(0x2011), VariantId(0x6011), VOLUME_SERVICE_IDS[1]),
    service_definition!(FeatureId(0x2012), VariantId(0x6012), VOLUME_SERVICE_IDS[2]),
    service_definition!(FeatureId(0x2013), VariantId(0x6013), VOLUME_SERVICE_IDS[3]),
    service_definition!(FeatureId(0x2014), VariantId(0x6014), VOLUME_SERVICE_IDS[4]),
    service_definition!(FeatureId(0x2015), VariantId(0x6015), VOLUME_SERVICE_IDS[5]),
    service_definition!(FeatureId(0x2016), VariantId(0x6016), VOLUME_SERVICE_IDS[6]),
];

const WATER_FEATURE: FeatureDefinition = FeatureDefinition::subsystem(
    WATER_FEATURE_ID,
    RenderDomain::Surface,
    ServiceSet::new(&WATER_SERVICE_IDS),
    WATER_VARIANTS,
);

const VOLUME_FEATURE: FeatureDefinition = FeatureDefinition::subsystem(
    VOLUME_FEATURE_ID,
    RenderDomain::Volume,
    ServiceSet::new(&VOLUME_SERVICE_IDS),
    VOLUME_VARIANTS,
);

const LIT_FEATURE: FeatureDefinition =
    FeatureDefinition::response(LIT_FEATURE_ID, ServiceSet::EMPTY, LIT_VARIANTS);
const NPR_FEATURE: FeatureDefinition =
    FeatureDefinition::response(NPR_FEATURE_ID, ServiceSet::EMPTY, NPR_VARIANTS);
const CUSTOM_FEATURE: FeatureDefinition =
    FeatureDefinition::response(CUSTOM_FEATURE_ID, ServiceSet::EMPTY, CUSTOM_VARIANTS);

static BUILTIN_FEATURES: [FeatureDefinition; 21] = [
    WATER_FEATURE,
    VOLUME_FEATURE,
    LIT_FEATURE,
    NPR_FEATURE,
    CUSTOM_FEATURE,
    SERVICE_FEATURES[0],
    SERVICE_FEATURES[1],
    SERVICE_FEATURES[2],
    SERVICE_FEATURES[3],
    SERVICE_FEATURES[4],
    SERVICE_FEATURES[5],
    SERVICE_FEATURES[6],
    SERVICE_FEATURES[7],
    SERVICE_FEATURES[8],
    SERVICE_FEATURES[9],
    SERVICE_FEATURES[10],
    SERVICE_FEATURES[11],
    SERVICE_FEATURES[12],
    SERVICE_FEATURES[13],
    SERVICE_FEATURES[14],
    SERVICE_FEATURES[15],
];

static BUILTIN_SUBSYSTEM_ROOTS: [FeatureId; 2] = [WATER_FEATURE_ID, VOLUME_FEATURE_ID];
static BUILTIN_RESPONSE_ROOTS: [FeatureId; 3] = [LIT_FEATURE_ID, NPR_FEATURE_ID, CUSTOM_FEATURE_ID];

/// Build the immutable semantic catalog used by renderer/device setup.
pub fn builtin_catalog() -> Result<FeatureCatalog, super::CatalogError> {
    FeatureCatalog::new(&BUILTIN_FEATURES, &[])
}

/// Freeze the subsystem and response roots for the device lifetime.
pub fn builtin_roots(
    catalog: &FeatureCatalog,
) -> Result<super::FrozenRoots, super::RootError> {
    super::FrozenRoots::from_build_profile(
        catalog,
        &BUILTIN_SUBSYSTEM_ROOTS,
        &BUILTIN_RESPONSE_ROOTS,
    )
}

pub const fn builtin_service_ids() -> &'static [ServiceId] {
    &WATER_SERVICE_IDS
}

pub const fn water_required_services() -> ServiceSet {
    WATER_FEATURE.requires_services
}

pub const fn volume_required_services() -> ServiceSet {
    VOLUME_FEATURE.requires_services
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_fabric::{FeatureSelection, RenderQualityTier};

    #[test]
    fn builtins_resolve_water_and_volume_service_closures() {
        let catalog = builtin_catalog().expect("built-in catalog is valid");
        let roots = builtin_roots(&catalog).expect("built-in roots are frozen");
        let selection = FeatureSelection::resolve(
            &catalog,
            &roots,
            CapabilitySet::EMPTY,
            RenderQualityTier::Low,
        )
        .expect("baseline built-ins resolve");
        assert!(selection.feature(WATER_FEATURE_ID).is_some());
        assert!(selection.feature(VOLUME_FEATURE_ID).is_some());
        assert_eq!(selection.provider(WATER_SERVICE_IDS[0]), Some(FeatureId(0x2000)));
        assert_eq!(selection.provider(VOLUME_SERVICE_IDS[0]), Some(FeatureId(0x2010)));
    }
}
