//! The production render-feature declarations used to bootstrap the device.
//!
//! This is intentionally a small semantic catalog: providers publish service
//! identities, while water/volume and the three response arms declare the
//! roots that consume them. Frame resources remain owned by the selected pass
//! templates and the authoritative frame graph.

use super::{
    AccessKind, FeatureCatalog, FeatureDefinition, FeatureId, FeatureVariant, FrameLag, PassKey,
    PassTemplate, QueueClass, RenderDomain, RenderQualityTier, ResourceKey, ResourceLayout,
    ResourceScope, ResourceTemplate, ResourceUse, ServiceId, ServiceOutput, ServiceSet, VariantId,
};
use crate::backend::capability::CapabilitySet;
use crate::frame_graph::ResourceLifetime;
use prism_render_driver::BufferUsages;

pub const SPRAY_FEEDBACK_SERVICE: ServiceId = ServiceId(10);
pub const SPRAY_FEEDBACK_FEATURE_ID: FeatureId = FeatureId(0x2020);

pub const WATER_SERVICE_IDS: [ServiceId; 10] = [
    ServiceId(1),
    ServiceId(2),
    ServiceId(3),
    ServiceId(4),
    ServiceId(5),
    ServiceId(6),
    ServiceId(7),
    ServiceId(8),
    ServiceId(9),
    SPRAY_FEEDBACK_SERVICE,
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

/// Stable semantic service identities used by particle shading capability
/// negotiation. These are global Render Fabric identities, not a private
/// particle provider list: virtual-shadow-map and global-illumination entries
/// intentionally alias the corresponding water identities. They are capability
/// inputs, not provider outputs; particle feature closure still uses
/// catalog-owned requirements.
pub const PARTICLE_LIGHTING_SERVICE_IDS: [ServiceId; 4] = [
    ServiceId(20), // clustered lights
    ServiceId(4),  // virtual shadow maps
    ServiceId(2),  // global illumination
    ServiceId(21), // ray-traced lighting
];
pub const PARTICLE_CLUSTERED_LIGHTS: ServiceId = PARTICLE_LIGHTING_SERVICE_IDS[0];
pub const PARTICLE_SHADOW_MAPS: ServiceId = PARTICLE_LIGHTING_SERVICE_IDS[1];
pub const PARTICLE_GLOBAL_ILLUMINATION: ServiceId = PARTICLE_LIGHTING_SERVICE_IDS[2];
pub const PARTICLE_RAY_TRACING: ServiceId = PARTICLE_LIGHTING_SERVICE_IDS[3];
pub const PARTICLE_LIGHTING_CAPABILITIES: ServiceSet = ServiceSet::new(&[
    PARTICLE_SHADOW_MAPS,
    PARTICLE_GLOBAL_ILLUMINATION,
    PARTICLE_RAY_TRACING,
]);

pub const WATER_FEATURE_ID: FeatureId = FeatureId(0x1000);
pub const VOLUME_FEATURE_ID: FeatureId = FeatureId(0x1001);
pub const PARTICLE_FEATURE_ID: FeatureId = FeatureId(0x1002);
pub const HAIR_FEATURE_ID: FeatureId = FeatureId(0x1003);
pub const DECAL_FEATURE_ID: FeatureId = FeatureId(0x1004);
pub const LIT_FEATURE_ID: FeatureId = FeatureId(0x1100);
pub const NPR_FEATURE_ID: FeatureId = FeatureId(0x1101);
pub const CUSTOM_FEATURE_ID: FeatureId = FeatureId(0x1102);

pub const WATER_SURFACE_RESOURCE: ResourceKey = ResourceKey::of(WATER_FEATURE_ID, 0);
/// Compatibility name for the particle service's persistent foam feedback.
pub const WATER_SPRAY_RESOURCE: ResourceKey = PARTICLE_SPRAY_RESOURCE;
pub const WATER_FOAM_RESOURCE: ResourceKey = ResourceKey::of(WATER_FEATURE_ID, 2);
pub const VOLUME_DENSITY_RESOURCE: ResourceKey = ResourceKey::of(VOLUME_FEATURE_ID, 0);
pub const VOLUME_FROXEL_RESOURCE: ResourceKey = ResourceKey::of(VOLUME_FEATURE_ID, 1);
pub const VOLUME_VELOCITY_RESOURCE: ResourceKey = ResourceKey::of(VOLUME_FEATURE_ID, 2);
pub const PARTICLE_STREAM_RESOURCE: ResourceKey = ResourceKey::of(PARTICLE_FEATURE_ID, 0);
pub const PARTICLE_SPRAY_RESOURCE: ResourceKey = ResourceKey::of(SPRAY_FEEDBACK_FEATURE_ID, 0);
pub const PARTICLE_ALIVE_RESOURCE: ResourceKey = ResourceKey::of(PARTICLE_FEATURE_ID, 2);
pub const HAIR_STRAND_RESOURCE: ResourceKey = ResourceKey::of(HAIR_FEATURE_ID, 0);
pub const HAIR_VISIBILITY_RESOURCE: ResourceKey = ResourceKey::of(HAIR_FEATURE_ID, 1);
pub const HAIR_DEEP_OPACITY_RESOURCE: ResourceKey = ResourceKey::of(HAIR_FEATURE_ID, 2);
pub const DECAL_INSTANCE_RESOURCE: ResourceKey = ResourceKey::of(DECAL_FEATURE_ID, 0);
pub const DECAL_INDIRECT_RESOURCE: ResourceKey = ResourceKey::of(DECAL_FEATURE_ID, 1);
pub const DECAL_RESOLVE_RESOURCE: ResourceKey = ResourceKey::of(DECAL_FEATURE_ID, 2);

const WATER_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        key: WATER_SURFACE_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: WATER_FOAM_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: WATER_SPRAY_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: FrameLag::new(1),
    },
];
const VOLUME_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        key: VOLUME_DENSITY_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: VOLUME_FROXEL_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: VOLUME_VELOCITY_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
];
const PARTICLE_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        key: PARTICLE_STREAM_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: PARTICLE_ALIVE_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
];
const HAIR_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        key: HAIR_STRAND_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: HAIR_VISIBILITY_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: HAIR_DEEP_OPACITY_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
];
const DECAL_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        key: DECAL_INSTANCE_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: DECAL_INDIRECT_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: DECAL_RESOLVE_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
];

const WATER_PASSES: &[PassTemplate] = &[PassTemplate {
    id: PassKey(0x5000),
    accesses: WATER_ACCESSES,
    queue: QueueClass::Compute,
}];
const VOLUME_PASSES: &[PassTemplate] = &[PassTemplate {
    id: PassKey(0x5001),
    accesses: VOLUME_ACCESSES,
    queue: QueueClass::Compute,
}];
const PARTICLE_PASSES: &[PassTemplate] = &[PassTemplate {
    id: PassKey(0x5005),
    accesses: PARTICLE_ACCESSES,
    queue: QueueClass::Compute,
}];
const HAIR_PASSES: &[PassTemplate] = &[PassTemplate {
    id: PassKey(0x5006),
    accesses: HAIR_ACCESSES,
    queue: QueueClass::Compute,
}];
const DECAL_PASSES: &[PassTemplate] = &[PassTemplate {
    id: PassKey(0x5007),
    accesses: DECAL_ACCESSES,
    queue: QueueClass::Compute,
}];

const WATER_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: VariantId(0x5000),
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: WATER_PASSES,
    shader_permutation: None,
}];
const VOLUME_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: VariantId(0x5001),
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: VOLUME_PASSES,
    shader_permutation: None,
}];
const PARTICLE_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: VariantId(0x5005),
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: PARTICLE_PASSES,
    shader_permutation: None,
}];
const HAIR_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: VariantId(0x5006),
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: HAIR_PASSES,
    shader_permutation: None,
}];
const DECAL_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: VariantId(0x5007),
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: DECAL_PASSES,
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
        const RESOURCE: ResourceKey = ResourceKey::of($feature, 0);
        const OUTPUT_RESOURCES: &[ResourceKey] = &[RESOURCE];
        const ACCESSES: &[ResourceUse] = &[ResourceUse {
            key: RESOURCE,
            access: AccessKind::StorageWrite,
            scope: ResourceScope::Whole,
            frame_lag: None,
        }];
        const PASSES: &[PassTemplate] = &[PassTemplate {
            id: PassKey($variant.0),
            accesses: ACCESSES,
            queue: QueueClass::Compute,
        }];
        const OUTPUTS: &[ServiceOutput] = &[ServiceOutput {
            service: $service,
            resources: OUTPUT_RESOURCES,
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
                passes: PASSES,
                shader_permutation: None,
            }],
        )
    }};
}

const SPRAY_FEEDBACK: FeatureDefinition = FeatureDefinition::service(
    SPRAY_FEEDBACK_FEATURE_ID,
    &[ServiceOutput {
        service: SPRAY_FEEDBACK_SERVICE,
        resources: &[PARTICLE_SPRAY_RESOURCE],
    }],
    ServiceSet::EMPTY,
    &[FeatureVariant {
        id: VariantId(0x6020),
        rank: 0,
        min_quality: RenderQualityTier::Low,
        required_caps: CapabilitySet::EMPTY,
        requires_services_extra: ServiceSet::EMPTY,
        passes: &[PassTemplate {
            id: PassKey(0x6020),
            queue: QueueClass::Compute,
            accesses: &[
                ResourceUse {
                    key: WATER_SURFACE_RESOURCE,
                    access: AccessKind::StorageRead,
                    scope: ResourceScope::Whole,
                    frame_lag: None,
                },
                ResourceUse {
                    key: PARTICLE_SPRAY_RESOURCE,
                    access: AccessKind::StorageWrite,
                    scope: ResourceScope::Whole,
                    frame_lag: None,
                },
            ],
        }],
        shader_permutation: None,
    }],
);

const SERVICE_FEATURES: [FeatureDefinition; 17] = [
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
    SPRAY_FEEDBACK,
];

const BUILTIN_RESOURCE_KEYS: [ResourceKey; 30] = [
    WATER_SURFACE_RESOURCE,
    WATER_FOAM_RESOURCE,
    VOLUME_DENSITY_RESOURCE,
    VOLUME_FROXEL_RESOURCE,
    VOLUME_VELOCITY_RESOURCE,
    PARTICLE_STREAM_RESOURCE,
    PARTICLE_SPRAY_RESOURCE,
    PARTICLE_ALIVE_RESOURCE,
    HAIR_STRAND_RESOURCE,
    HAIR_VISIBILITY_RESOURCE,
    HAIR_DEEP_OPACITY_RESOURCE,
    DECAL_INSTANCE_RESOURCE,
    DECAL_INDIRECT_RESOURCE,
    DECAL_RESOLVE_RESOURCE,
    ResourceKey::of(FeatureId(0x2000), 0),
    ResourceKey::of(FeatureId(0x2001), 0),
    ResourceKey::of(FeatureId(0x2002), 0),
    ResourceKey::of(FeatureId(0x2003), 0),
    ResourceKey::of(FeatureId(0x2004), 0),
    ResourceKey::of(FeatureId(0x2005), 0),
    ResourceKey::of(FeatureId(0x2006), 0),
    ResourceKey::of(FeatureId(0x2007), 0),
    ResourceKey::of(FeatureId(0x2008), 0),
    ResourceKey::of(FeatureId(0x2010), 0),
    ResourceKey::of(FeatureId(0x2011), 0),
    ResourceKey::of(FeatureId(0x2012), 0),
    ResourceKey::of(FeatureId(0x2013), 0),
    ResourceKey::of(FeatureId(0x2014), 0),
    ResourceKey::of(FeatureId(0x2015), 0),
    ResourceKey::of(FeatureId(0x2016), 0),
];

const fn resource_template(key: ResourceKey) -> ResourceTemplate {
    ResourceTemplate {
        key,
        lifetime: ResourceLifetime::Persistent,
        layout: ResourceLayout::Buffer {
            bytes: 256,
            usage: BufferUsages::STORAGE,
        },
        alignment: 16,
    }
}

pub const BUILTIN_RESOURCE_TEMPLATES: &[ResourceTemplate] = &[
    resource_template(WATER_SURFACE_RESOURCE),
    resource_template(WATER_FOAM_RESOURCE),
    resource_template(VOLUME_DENSITY_RESOURCE),
    resource_template(VOLUME_FROXEL_RESOURCE),
    resource_template(VOLUME_VELOCITY_RESOURCE),
    resource_template(PARTICLE_STREAM_RESOURCE),
    resource_template(PARTICLE_SPRAY_RESOURCE),
    resource_template(PARTICLE_ALIVE_RESOURCE),
    resource_template(HAIR_STRAND_RESOURCE),
    resource_template(HAIR_VISIBILITY_RESOURCE),
    resource_template(HAIR_DEEP_OPACITY_RESOURCE),
    resource_template(DECAL_INSTANCE_RESOURCE),
    resource_template(DECAL_INDIRECT_RESOURCE),
    resource_template(DECAL_RESOLVE_RESOURCE),
    resource_template(ResourceKey::of(FeatureId(0x2000), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2001), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2002), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2003), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2004), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2005), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2006), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2007), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2008), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2010), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2011), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2012), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2013), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2014), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2015), 0)),
    resource_template(ResourceKey::of(FeatureId(0x2016), 0)),
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
const PARTICLE_FEATURE: FeatureDefinition = FeatureDefinition::subsystem(
    PARTICLE_FEATURE_ID,
    RenderDomain::Particle,
    ServiceSet::new(&[SPRAY_FEEDBACK_SERVICE]),
    PARTICLE_VARIANTS,
);
const HAIR_FEATURE: FeatureDefinition = FeatureDefinition::subsystem(
    HAIR_FEATURE_ID,
    RenderDomain::Hair,
    ServiceSet::EMPTY,
    HAIR_VARIANTS,
);
const DECAL_FEATURE: FeatureDefinition = FeatureDefinition::subsystem(
    DECAL_FEATURE_ID,
    RenderDomain::Decal,
    ServiceSet::EMPTY,
    DECAL_VARIANTS,
);
const LIT_FEATURE: FeatureDefinition =
    FeatureDefinition::response(LIT_FEATURE_ID, ServiceSet::EMPTY, LIT_VARIANTS);
const NPR_FEATURE: FeatureDefinition =
    FeatureDefinition::response(NPR_FEATURE_ID, ServiceSet::EMPTY, NPR_VARIANTS);
const CUSTOM_FEATURE: FeatureDefinition =
    FeatureDefinition::response(CUSTOM_FEATURE_ID, ServiceSet::EMPTY, CUSTOM_VARIANTS);

static BUILTIN_FEATURES: [FeatureDefinition; 25] = [
    WATER_FEATURE,
    VOLUME_FEATURE,
    PARTICLE_FEATURE,
    HAIR_FEATURE,
    DECAL_FEATURE,
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
    SERVICE_FEATURES[16],
];

static BUILTIN_SUBSYSTEM_ROOTS: [FeatureId; 5] = [
    WATER_FEATURE_ID,
    VOLUME_FEATURE_ID,
    PARTICLE_FEATURE_ID,
    HAIR_FEATURE_ID,
    DECAL_FEATURE_ID,
];
static BUILTIN_RESPONSE_ROOTS: [FeatureId; 3] = [LIT_FEATURE_ID, NPR_FEATURE_ID, CUSTOM_FEATURE_ID];

/// Build the immutable semantic catalog used by renderer/device setup.
pub fn builtin_catalog() -> Result<FeatureCatalog, super::CatalogError> {
    FeatureCatalog::new(&BUILTIN_FEATURES, &BUILTIN_RESOURCE_KEYS)
}

pub const fn builtin_resource_templates() -> &'static [ResourceTemplate] {
    BUILTIN_RESOURCE_TEMPLATES
}

/// Freeze the subsystem and response roots for the device lifetime.
pub fn builtin_roots(catalog: &FeatureCatalog) -> Result<super::FrozenRoots, super::RootError> {
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
        assert_eq!(
            selection.provider(WATER_SERVICE_IDS[0]),
            Some(FeatureId(0x2000))
        );
        assert_eq!(
            selection.provider(VOLUME_SERVICE_IDS[0]),
            Some(FeatureId(0x2010))
        );
    }

    #[test]
    fn builtins_freeze_every_subsystem_and_response_root() {
        let catalog = builtin_catalog().expect("built-in catalog is valid");
        let roots = builtin_roots(&catalog).expect("all built-in roots are valid");
        let expected = [
            WATER_FEATURE_ID,
            VOLUME_FEATURE_ID,
            PARTICLE_FEATURE_ID,
            HAIR_FEATURE_ID,
            DECAL_FEATURE_ID,
            LIT_FEATURE_ID,
            NPR_FEATURE_ID,
            CUSTOM_FEATURE_ID,
        ];
        assert_eq!(roots.ids(), &expected);

        let selection = FeatureSelection::resolve(
            &catalog,
            &roots,
            CapabilitySet::EMPTY,
            RenderQualityTier::Low,
        )
        .expect("the complete built-in active-root fanout must resolve");
        for feature in expected {
            assert!(
                selection.feature(feature).is_some(),
                "active root {feature:?} must be selected"
            );
        }
        assert!(selection.feature(HAIR_FEATURE_ID).is_some());
        assert!(selection.feature(DECAL_FEATURE_ID).is_some());
    }

    #[test]
    fn built_in_service_ids_have_one_shared_global_identity_per_capability() {
        assert_eq!(PARTICLE_SHADOW_MAPS, WATER_SERVICE_IDS[3]);
        assert_eq!(PARTICLE_GLOBAL_ILLUMINATION, WATER_SERVICE_IDS[1]);
        assert_eq!(PARTICLE_LIGHTING_CAPABILITIES.len(), 3);
        assert!(!PARTICLE_LIGHTING_CAPABILITIES.contains(PARTICLE_CLUSTERED_LIGHTS));
    }
}
