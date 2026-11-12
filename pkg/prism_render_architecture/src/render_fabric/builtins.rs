//! The production render-feature declarations used to bootstrap the device.
//!
//! This is intentionally a small semantic catalog: providers publish service
//! identities, while water/volume and the three response arms declare the
//! roots that consume them. Frame resources remain owned by the selected pass
//! templates and the authoritative frame graph.

use super::{
    AccessKind, FeatureCatalog, FeatureDefinition, FeatureId, FeatureVariant, FrameLag,
    ObjectLayoutPolicy, ObjectResourceAccess, PassDescriptorIdentity, PassKey, PassTemplate,
    PayloadRole, QueueClass, RenderDomain, RenderQualityTier, ResourceInput, ResourceKey,
    ResourceLayout, ResourceScope, ResourceTemplate, ResourceUse, ServiceId, ServiceOutput,
    ServiceSet, ShaderBindingError, ShaderModuleBinding, ShaderPackageBinding, VariantId,
};
use crate::backend::capability::CapabilitySet;
use crate::shader_package::{ArtifactDigest, ModuleIdentity, PermutationSpace, ShaderPackageId};
use prism_render_driver::{BufferUsages, TextureDimension, TextureFormat, TextureUsages};
use prism_render_graph::frame_graph::ResourceLifetime;

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
/// Native Surface identity reconstruction is a first-class subsystem pass;
/// it is not a response/composition shortcut.
pub const SURFACE_FEATURE_ID: FeatureId = FeatureId(0x1005);
pub const LIT_FEATURE_ID: FeatureId = FeatureId(0x1100);
pub const NPR_FEATURE_ID: FeatureId = FeatureId(0x1101);
pub const CUSTOM_FEATURE_ID: FeatureId = FeatureId(0x1102);

/// Stable implementation identities shared by the catalog and host executors.
/// A host must explicitly register support for a variant; selecting a new
/// implementation must never silently reuse an executor for the old one.
pub const WATER_VARIANT_ID: VariantId = VariantId(0x5000);
pub const LIT_VARIANT_ID: VariantId = VariantId(0x5002);
pub const NPR_VARIANT_ID: VariantId = VariantId(0x5003);
pub const CUSTOM_VARIANT_ID: VariantId = VariantId(0x5004);
pub const SURFACE_VARIANT_ID: VariantId = VariantId(0x5009);

fn builtin_package_name(variant: VariantId) -> Option<&'static str> {
    Some(match variant {
        WATER_VARIANT_ID => "builtin.render_fabric.water",
        VariantId(0x5001) => "builtin.render_fabric.volume",
        LIT_VARIANT_ID => "builtin.render_fabric.lit",
        NPR_VARIANT_ID => "builtin.render_fabric.npr",
        CUSTOM_VARIANT_ID => "builtin.render_fabric.custom",
        SURFACE_VARIANT_ID => "builtin.render_fabric.surface_identity",
        VariantId(0x5005) => "builtin.render_fabric.particle",
        VariantId(0x5006) => "builtin.render_fabric.hair",
        VariantId(0x5007) => "builtin.render_fabric.decal",
        VariantId(0x6000) => "builtin.render_fabric.service.water.0",
        VariantId(0x6001) => "builtin.render_fabric.service.water.1",
        VariantId(0x6002) => "builtin.render_fabric.service.water.2",
        VariantId(0x6003) => "builtin.render_fabric.service.water.3",
        VariantId(0x6004) => "builtin.render_fabric.service.water.4",
        VariantId(0x6005) => "builtin.render_fabric.service.water.5",
        VariantId(0x6006) => "builtin.render_fabric.service.water.6",
        VariantId(0x6007) => "builtin.render_fabric.service.water.7",
        VariantId(0x6008) => "builtin.render_fabric.service.water.8",
        VariantId(0x6010) => "builtin.render_fabric.service.volume.0",
        VariantId(0x6011) => "builtin.render_fabric.service.volume.1",
        VariantId(0x6012) => "builtin.render_fabric.service.volume.2",
        VariantId(0x6013) => "builtin.render_fabric.service.volume.3",
        VariantId(0x6014) => "builtin.render_fabric.service.volume.4",
        VariantId(0x6015) => "builtin.render_fabric.service.volume.5",
        VariantId(0x6016) => "builtin.render_fabric.service.volume.6",
        VariantId(0x6020) => "builtin.render_fabric.service.spray_feedback",
        _ => return None,
    })
}

/// Builds the canonical package/module/pass identity for a built-in variant.
/// The returned module has no driver-local `ModuleId`; the backend allocates
/// that identity when it interns this declaration on a concrete device.
pub(super) fn builtin_shader_package_binding(
    variant: &FeatureVariant,
) -> Result<ShaderPackageBinding, ShaderBindingError> {
    let name =
        builtin_package_name(variant.id).ok_or(ShaderBindingError::BuiltinVariantUnavailable {
            variant: variant.id,
        })?;
    let package = ShaderPackageId::new(name);
    let binding = ShaderPackageBinding {
        package,
        module: ShaderModuleBinding {
            identity: ModuleIdentity {
                package,
                digest: ArtifactDigest::from_bytes(name.as_bytes()),
                entry_point: "main".into(),
                binding_layout_digest: ArtifactDigest::from_bytes(
                    b"prism.render_fabric.builtin.layout.v1",
                )
                .0,
            },
            permutations: variant
                .shader_permutation
                .map(|binding| binding.space.clone())
                .unwrap_or_else(|| PermutationSpace::new([])),
            full_lobe_fallback: None,
        },
        permutation: variant.shader_permutation,
        pass_descriptors: variant
            .passes
            .iter()
            .map(PassDescriptorIdentity::from_template)
            .collect(),
    };
    binding.validate_for(variant)?;
    Ok(binding)
}

pub const WATER_SURFACE_RESOURCE: ResourceKey = ResourceKey::of(WATER_FEATURE_ID, 0);
pub const WATER_FOAM_RESOURCE: ResourceKey = ResourceKey::of(WATER_FEATURE_ID, 2);
pub const VOLUME_SCATTERING_RESOURCE: ResourceKey = ResourceKey::of(VOLUME_FEATURE_ID, 0);
pub const VOLUME_EXTINCTION_RESOURCE: ResourceKey = ResourceKey::of(VOLUME_FEATURE_ID, 1);
pub const VOLUME_INTEGRATED_SCATTERING_RESOURCE: ResourceKey =
    ResourceKey::of(VOLUME_FEATURE_ID, 2);
pub const VOLUME_INTEGRATED_TRANSMITTANCE_RESOURCE: ResourceKey =
    ResourceKey::of(VOLUME_FEATURE_ID, 3);
pub const VOLUME_FOG_APPLIED_RESOURCE: ResourceKey = ResourceKey::of(VOLUME_FEATURE_ID, 4);
pub const VOLUME_SCENE_DEPTH_RESOURCE: ResourceKey = ResourceKey::of(VOLUME_FEATURE_ID, 5);
pub const VOLUME_SCENE_COLOR_RESOURCE: ResourceKey = ResourceKey::of(VOLUME_FEATURE_ID, 6);
/// Per-domain producer payload, distinct from froxel fog textures.
pub const VOLUME_DENSITY_RESOURCE: ResourceKey = ResourceKey::of(VOLUME_FEATURE_ID, 7);
pub const VOLUME_FROXEL_RESOURCE: ResourceKey = ResourceKey::of(VOLUME_FEATURE_ID, 8);
pub const VOLUME_VELOCITY_RESOURCE: ResourceKey = ResourceKey::of(VOLUME_FEATURE_ID, 9);
pub const PARTICLE_STREAM_RESOURCE: ResourceKey = ResourceKey::of(PARTICLE_FEATURE_ID, 0);
pub const PARTICLE_SPRAY_RESOURCE: ResourceKey = ResourceKey::of(SPRAY_FEEDBACK_FEATURE_ID, 0);
pub const PARTICLE_ALIVE_RESOURCE: ResourceKey = ResourceKey::of(PARTICLE_FEATURE_ID, 2);
pub const PARTICLE_SIMULATION_PARTICLES_RESOURCE: ResourceKey =
    ResourceKey::of(PARTICLE_FEATURE_ID, 3);
pub const PARTICLE_SIMULATION_ACTIVE_INDICES_RESOURCE: ResourceKey =
    ResourceKey::of(PARTICLE_FEATURE_ID, 4);
pub const PARTICLE_SIMULATION_COMPACT_EVENTS_RESOURCE: ResourceKey =
    ResourceKey::of(PARTICLE_FEATURE_ID, 5);
pub const PARTICLE_SIMULATION_CONTROL_RESOURCE: ResourceKey =
    ResourceKey::of(PARTICLE_FEATURE_ID, 6);
pub const PARTICLE_BRIDGE_EVENT_FLAGS_RESOURCE: ResourceKey =
    ResourceKey::of(PARTICLE_FEATURE_ID, 7);
pub const PARTICLE_RENDER_INPUT_COUNT_RESOURCE: ResourceKey =
    ResourceKey::of(PARTICLE_FEATURE_ID, 8);
pub const PARTICLE_BRIDGE_VALIDATION_RESOURCE: ResourceKey =
    ResourceKey::of(PARTICLE_FEATURE_ID, 9);
pub const PARTICLE_RENDER_ROUTE_TABLE_RESOURCE: ResourceKey =
    ResourceKey::of(PARTICLE_FEATURE_ID, 10);
pub const PARTICLE_BRIDGE_ROUTE_SCRATCH_RESOURCE: ResourceKey =
    ResourceKey::of(PARTICLE_FEATURE_ID, 11);
pub const PARTICLE_BRIDGE_ACTIVE_ROUTE_INDICES_RESOURCE: ResourceKey =
    ResourceKey::of(PARTICLE_FEATURE_ID, 12);
pub const HAIR_STRAND_RESOURCE: ResourceKey = ResourceKey::of(HAIR_FEATURE_ID, 0);
pub const HAIR_VISIBILITY_RESOURCE: ResourceKey = ResourceKey::of(HAIR_FEATURE_ID, 1);
pub const HAIR_DEEP_OPACITY_RESOURCE: ResourceKey = ResourceKey::of(HAIR_FEATURE_ID, 2);
pub const DECAL_INSTANCE_RESOURCE: ResourceKey = ResourceKey::of(DECAL_FEATURE_ID, 0);
pub const DECAL_MATERIAL_RESOURCE: ResourceKey = ResourceKey::of(DECAL_FEATURE_ID, 1);
pub const DECAL_RECEIVER_RESOURCE: ResourceKey = ResourceKey::of(DECAL_FEATURE_ID, 2);
pub const DECAL_CLUSTER_RANGE_RESOURCE: ResourceKey = ResourceKey::of(DECAL_FEATURE_ID, 3);
pub const DECAL_CLUSTER_INDEX_RESOURCE: ResourceKey = ResourceKey::of(DECAL_FEATURE_ID, 4);
pub const DECAL_RESOLVE_RESOURCE: ResourceKey = ResourceKey::of(DECAL_FEATURE_ID, 5);
pub const DECAL_RESOLVE_PASS: PassKey = PassKey(0x5007);
pub const SURFACE_IDENTITY_PASS: PassKey = PassKey(0x5009);

pub const SURFACE_NORMAL_RESOURCE: ResourceKey = ResourceKey::of(SURFACE_FEATURE_ID, 0);
pub const SURFACE_ID_RESOURCE: ResourceKey = ResourceKey::of(SURFACE_FEATURE_ID, 1);
pub const SURFACE_MATERIAL_HEADERS_RESOURCE: ResourceKey = ResourceKey::of(SURFACE_FEATURE_ID, 2);
pub const SURFACE_EXECUTION_PLANS_RESOURCE: ResourceKey = ResourceKey::of(SURFACE_FEATURE_ID, 3);
pub const SURFACE_EXECUTION_PROGRAMS_RESOURCE: ResourceKey = ResourceKey::of(SURFACE_FEATURE_ID, 4);
pub const SURFACE_LIGHTING_MAP_RESOURCE: ResourceKey = ResourceKey::of(SURFACE_FEATURE_ID, 5);
pub const SURFACE_PARAMS_RESOURCE: ResourceKey = ResourceKey::of(SURFACE_FEATURE_ID, 6);
pub const SURFACE_IDENTITY_RESOURCE: ResourceKey = ResourceKey::of(SURFACE_FEATURE_ID, 7);

/// Response-composition resources are owned by the water/frame fabric and
/// mirrored here so response variants use the same semantic identities.
pub const RESPONSE_LEFT_RESOURCE: ResourceKey = crate::water::fabric::RESPONSE_LEFT_RESOURCE;
pub const RESPONSE_RIGHT_RESOURCE: ResourceKey = crate::water::fabric::RESPONSE_RIGHT_RESOURCE;
pub const RESPONSE_CONTROLS_RESOURCE: ResourceKey =
    crate::water::fabric::RESPONSE_CONTROLS_RESOURCE;
pub const RESPONSE_AUXILIARY_RESOURCE: ResourceKey =
    crate::water::fabric::RESPONSE_AUXILIARY_RESOURCE;
pub const WATER_COVERAGE_RESOURCE: ResourceKey = crate::water::fabric::WATER_COVERAGE_RESOURCE;

/// Custom is a third response root and is selected alongside Lit/NPR.  It
/// therefore needs a root-owned output lane instead of becoming a second
/// writer of either shared composition plane.
pub const CUSTOM_RESPONSE_RESOURCE: ResourceKey = ResourceKey::of(CUSTOM_FEATURE_ID, 0);

/// Persistent per-view temporal resolve storage. The extent is supplied by the
/// active viewport, so the render-world producer creates the exact
/// `ResourceTemplate` for each allocation rather than putting a fake fixed
/// extent in the static catalog.
pub const TAA_HISTORY_RESOURCE: ResourceKey = ResourceKey::engine(0x7208);

const WATER_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        input: ResourceInput::Automatic,
        key: WATER_SURFACE_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: WATER_FOAM_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: PARTICLE_SPRAY_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: FrameLag::new(1),
    },
];
const fn volume_access(key: ResourceKey, access: AccessKind, input: ResourceInput) -> ResourceUse {
    ResourceUse {
        key,
        access,
        scope: ResourceScope::Whole,
        frame_lag: None,
        input,
    }
}
const VOLUME_SCATTER_ACCESSES: &[ResourceUse] = &[
    volume_access(
        VOLUME_SCATTERING_RESOURCE,
        AccessKind::StorageWrite,
        ResourceInput::Automatic,
    ),
    volume_access(
        VOLUME_EXTINCTION_RESOURCE,
        AccessKind::StorageWrite,
        ResourceInput::Automatic,
    ),
];
const VOLUME_INTEGRATE_ACCESSES: &[ResourceUse] = &[
    volume_access(
        VOLUME_SCATTERING_RESOURCE,
        AccessKind::SampledRead,
        ResourceInput::ProducedBy(VOLUME_SCATTER_PASS),
    ),
    volume_access(
        VOLUME_EXTINCTION_RESOURCE,
        AccessKind::SampledRead,
        ResourceInput::ProducedBy(VOLUME_SCATTER_PASS),
    ),
    volume_access(
        VOLUME_INTEGRATED_SCATTERING_RESOURCE,
        AccessKind::StorageWrite,
        ResourceInput::Automatic,
    ),
    volume_access(
        VOLUME_INTEGRATED_TRANSMITTANCE_RESOURCE,
        AccessKind::StorageWrite,
        ResourceInput::Automatic,
    ),
];
const VOLUME_APPLY_ACCESSES: &[ResourceUse] = &[
    volume_access(
        VOLUME_SCENE_DEPTH_RESOURCE,
        AccessKind::SampledRead,
        ResourceInput::Initial,
    ),
    volume_access(
        VOLUME_INTEGRATED_SCATTERING_RESOURCE,
        AccessKind::SampledRead,
        ResourceInput::ProducedBy(VOLUME_INTEGRATE_PASS),
    ),
    volume_access(
        VOLUME_INTEGRATED_TRANSMITTANCE_RESOURCE,
        AccessKind::SampledRead,
        ResourceInput::ProducedBy(VOLUME_INTEGRATE_PASS),
    ),
    volume_access(
        VOLUME_SCENE_COLOR_RESOURCE,
        AccessKind::SampledRead,
        ResourceInput::Initial,
    ),
    volume_access(
        VOLUME_FOG_APPLIED_RESOURCE,
        AccessKind::StorageWrite,
        ResourceInput::Automatic,
    ),
];
const VOLUME_COPY_ACCESSES: &[ResourceUse] = &[
    volume_access(
        VOLUME_FOG_APPLIED_RESOURCE,
        AccessKind::TransferRead,
        ResourceInput::ProducedBy(VOLUME_APPLY_PASS),
    ),
    volume_access(
        VOLUME_SCENE_COLOR_RESOURCE,
        AccessKind::TransferWrite,
        ResourceInput::Initial,
    ),
];
const PARTICLE_BRIDGE_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        input: ResourceInput::Initial,
        key: PARTICLE_SIMULATION_PARTICLES_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Initial,
        key: PARTICLE_SIMULATION_ACTIVE_INDICES_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Initial,
        key: PARTICLE_SIMULATION_COMPACT_EVENTS_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Initial,
        key: PARTICLE_SIMULATION_CONTROL_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: PARTICLE_BRIDGE_EVENT_FLAGS_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: PARTICLE_STREAM_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: PARTICLE_RENDER_INPUT_COUNT_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: PARTICLE_BRIDGE_VALIDATION_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: PARTICLE_RENDER_ROUTE_TABLE_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: PARTICLE_BRIDGE_ROUTE_SCRATCH_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: PARTICLE_BRIDGE_ACTIVE_ROUTE_INDICES_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
];
const PARTICLE_RENDER_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        input: ResourceInput::ProducedBy(PARTICLE_BRIDGE_PASS),
        key: PARTICLE_STREAM_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::ProducedBy(PARTICLE_BRIDGE_PASS),
        key: PARTICLE_RENDER_INPUT_COUNT_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::ProducedBy(PARTICLE_BRIDGE_PASS),
        key: PARTICLE_RENDER_ROUTE_TABLE_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: PARTICLE_ALIVE_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
];
const HAIR_VISIBILITY_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        input: ResourceInput::Automatic,
        key: HAIR_STRAND_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Initial,
        key: HAIR_VISIBILITY_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
];
const HAIR_DEEP_OPACITY_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        input: ResourceInput::Automatic,
        key: HAIR_STRAND_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: HAIR_VISIBILITY_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: HAIR_DEEP_OPACITY_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
];
const DECAL_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        input: ResourceInput::Automatic,
        key: DECAL_INSTANCE_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: DECAL_MATERIAL_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: DECAL_RECEIVER_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: DECAL_CLUSTER_RANGE_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: DECAL_CLUSTER_INDEX_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: DECAL_RESOLVE_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
];

const SURFACE_IDENTITY_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        input: ResourceInput::Initial,
        key: SURFACE_NORMAL_RESOURCE,
        access: AccessKind::SampledRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Initial,
        key: SURFACE_ID_RESOURCE,
        access: AccessKind::SampledRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Initial,
        key: SURFACE_MATERIAL_HEADERS_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Initial,
        key: SURFACE_EXECUTION_PLANS_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Initial,
        key: SURFACE_EXECUTION_PROGRAMS_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Initial,
        key: SURFACE_LIGHTING_MAP_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Initial,
        key: SURFACE_PARAMS_RESOURCE,
        access: AccessKind::UniformRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        input: ResourceInput::Automatic,
        key: SURFACE_IDENTITY_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
];

const WATER_PASSES: &[PassTemplate] = &[PassTemplate {
    depends_on: &[],
    id: PassKey(0x5000),
    accesses: WATER_ACCESSES,
    queue: QueueClass::Compute,
}];
pub const VOLUME_SCATTER_PASS: PassKey = PassKey(0x5201);
/// Semantic producer pass for the renderer-input particle stream.
pub const PARTICLE_BRIDGE_PASS: PassKey = PassKey(0x5205);
/// Semantic consumer pass that builds the renderer's live visible-index stream.
pub const PARTICLE_RENDER_PASS: PassKey = PassKey(0x5206);
pub const VOLUME_INTEGRATE_PASS: PassKey = PassKey(0x5202);
pub const VOLUME_APPLY_PASS: PassKey = PassKey(0x5203);
pub const VOLUME_COPY_BACK_PASS: PassKey = PassKey(0x5204);
const VOLUME_PASSES: &[PassTemplate] = &[
    PassTemplate {
        depends_on: &[],
        id: VOLUME_SCATTER_PASS,
        accesses: VOLUME_SCATTER_ACCESSES,
        queue: QueueClass::Compute,
    },
    PassTemplate {
        depends_on: &[VOLUME_SCATTER_PASS],
        id: VOLUME_INTEGRATE_PASS,
        accesses: VOLUME_INTEGRATE_ACCESSES,
        queue: QueueClass::Compute,
    },
    PassTemplate {
        depends_on: &[VOLUME_INTEGRATE_PASS],
        id: VOLUME_APPLY_PASS,
        accesses: VOLUME_APPLY_ACCESSES,
        queue: QueueClass::Compute,
    },
    PassTemplate {
        depends_on: &[VOLUME_APPLY_PASS],
        id: VOLUME_COPY_BACK_PASS,
        accesses: VOLUME_COPY_ACCESSES,
        queue: QueueClass::Transfer,
    },
];
const PARTICLE_PASSES: &[PassTemplate] = &[
    PassTemplate {
        depends_on: &[],
        id: PARTICLE_BRIDGE_PASS,
        accesses: PARTICLE_BRIDGE_ACCESSES,
        queue: QueueClass::Compute,
    },
    PassTemplate {
        depends_on: &[PARTICLE_BRIDGE_PASS],
        id: PARTICLE_RENDER_PASS,
        accesses: PARTICLE_RENDER_ACCESSES,
        queue: QueueClass::Compute,
    },
];
pub const HAIR_VISIBILITY_PASS: PassKey = PassKey(0x5006);
pub const HAIR_DEEP_OPACITY_PASS: PassKey = PassKey(0x5008);
const HAIR_PASSES: &[PassTemplate] = &[
    PassTemplate {
        depends_on: &[],
        id: HAIR_VISIBILITY_PASS,
        accesses: HAIR_VISIBILITY_ACCESSES,
        queue: QueueClass::Compute,
    },
    PassTemplate {
        depends_on: &[HAIR_VISIBILITY_PASS],
        id: HAIR_DEEP_OPACITY_PASS,
        accesses: HAIR_DEEP_OPACITY_ACCESSES,
        queue: QueueClass::Compute,
    },
];
const DECAL_PASSES: &[PassTemplate] = &[PassTemplate {
    depends_on: &[],
    id: PassKey(0x5007),
    accesses: DECAL_ACCESSES,
    queue: QueueClass::Compute,
}];
const SURFACE_IDENTITY_PASSES: &[PassTemplate] = &[PassTemplate {
    depends_on: &[],
    id: SURFACE_IDENTITY_PASS,
    accesses: SURFACE_IDENTITY_ACCESSES,
    queue: QueueClass::Compute,
}];

const LIT_RESPONSE_ACCESSES: &[ResourceUse] = &[ResourceUse {
    input: ResourceInput::Automatic,
    key: RESPONSE_LEFT_RESOURCE,
    access: AccessKind::StorageWrite,
    scope: ResourceScope::Whole,
    frame_lag: None,
}];

const NPR_RESPONSE_ACCESSES: &[ResourceUse] = &[ResourceUse {
    input: ResourceInput::Automatic,
    key: RESPONSE_RIGHT_RESOURCE,
    access: AccessKind::StorageWrite,
    scope: ResourceScope::Whole,
    frame_lag: None,
}];

const CUSTOM_RESPONSE_ACCESSES: &[ResourceUse] = &[ResourceUse {
    input: ResourceInput::Automatic,
    key: CUSTOM_RESPONSE_RESOURCE,
    access: AccessKind::StorageWrite,
    scope: ResourceScope::Whole,
    frame_lag: None,
}];

const LIT_RESPONSE_PASSES: &[PassTemplate] = &[PassTemplate {
    depends_on: &[],
    id: PassKey(0x7300),
    accesses: LIT_RESPONSE_ACCESSES,
    queue: QueueClass::Compute,
}];
const NPR_RESPONSE_PASSES: &[PassTemplate] = &[PassTemplate {
    depends_on: &[],
    id: PassKey(0x7301),
    accesses: NPR_RESPONSE_ACCESSES,
    queue: QueueClass::Compute,
}];
const CUSTOM_RESPONSE_PASSES: &[PassTemplate] = &[PassTemplate {
    depends_on: &[],
    id: PassKey(0x7302),
    accesses: CUSTOM_RESPONSE_ACCESSES,
    queue: QueueClass::Compute,
}];

const WATER_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: WATER_VARIANT_ID,
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
const SURFACE_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: SURFACE_VARIANT_ID,
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: SURFACE_IDENTITY_PASSES,
    shader_permutation: None,
}];
const LIT_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: LIT_VARIANT_ID,
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: LIT_RESPONSE_PASSES,
    shader_permutation: None,
}];
const NPR_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: NPR_VARIANT_ID,
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: NPR_RESPONSE_PASSES,
    shader_permutation: None,
}];
const CUSTOM_VARIANTS: &[FeatureVariant] = &[FeatureVariant {
    id: CUSTOM_VARIANT_ID,
    rank: 0,
    min_quality: RenderQualityTier::Low,
    required_caps: CapabilitySet::EMPTY,
    requires_services_extra: ServiceSet::EMPTY,
    passes: CUSTOM_RESPONSE_PASSES,
    shader_permutation: None,
}];

macro_rules! service_definition {
    ($feature:expr, $variant:expr, $service:expr) => {{
        const RESOURCE: ResourceKey = ResourceKey::of($feature, 0);
        const OUTPUT_RESOURCES: &[ResourceKey] = &[RESOURCE];
        const ACCESSES: &[ResourceUse] = &[ResourceUse {
            input: ResourceInput::Automatic,
            key: RESOURCE,
            access: AccessKind::StorageWrite,
            scope: ResourceScope::Whole,
            frame_lag: None,
        }];
        const PASSES: &[PassTemplate] = &[PassTemplate {
            depends_on: &[],
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
            depends_on: &[],
            id: PassKey(0x6020),
            queue: QueueClass::Compute,
            accesses: &[
                ResourceUse {
                    input: ResourceInput::Automatic,
                    key: WATER_SURFACE_RESOURCE,
                    access: AccessKind::StorageRead,
                    scope: ResourceScope::Whole,
                    frame_lag: None,
                },
                ResourceUse {
                    input: ResourceInput::Automatic,
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

const BUILTIN_RESOURCE_KEYS: [ResourceKey; 64] = [
    WATER_SURFACE_RESOURCE,
    WATER_FOAM_RESOURCE,
    VOLUME_SCATTERING_RESOURCE,
    VOLUME_EXTINCTION_RESOURCE,
    VOLUME_INTEGRATED_SCATTERING_RESOURCE,
    VOLUME_INTEGRATED_TRANSMITTANCE_RESOURCE,
    VOLUME_FOG_APPLIED_RESOURCE,
    VOLUME_SCENE_DEPTH_RESOURCE,
    VOLUME_SCENE_COLOR_RESOURCE,
    VOLUME_DENSITY_RESOURCE,
    VOLUME_FROXEL_RESOURCE,
    VOLUME_VELOCITY_RESOURCE,
    PARTICLE_STREAM_RESOURCE,
    PARTICLE_SPRAY_RESOURCE,
    PARTICLE_ALIVE_RESOURCE,
    PARTICLE_SIMULATION_PARTICLES_RESOURCE,
    PARTICLE_SIMULATION_ACTIVE_INDICES_RESOURCE,
    PARTICLE_SIMULATION_COMPACT_EVENTS_RESOURCE,
    PARTICLE_SIMULATION_CONTROL_RESOURCE,
    PARTICLE_BRIDGE_EVENT_FLAGS_RESOURCE,
    PARTICLE_RENDER_INPUT_COUNT_RESOURCE,
    PARTICLE_BRIDGE_VALIDATION_RESOURCE,
    PARTICLE_RENDER_ROUTE_TABLE_RESOURCE,
    PARTICLE_BRIDGE_ROUTE_SCRATCH_RESOURCE,
    PARTICLE_BRIDGE_ACTIVE_ROUTE_INDICES_RESOURCE,
    HAIR_STRAND_RESOURCE,
    HAIR_VISIBILITY_RESOURCE,
    HAIR_DEEP_OPACITY_RESOURCE,
    DECAL_INSTANCE_RESOURCE,
    DECAL_MATERIAL_RESOURCE,
    DECAL_RECEIVER_RESOURCE,
    DECAL_CLUSTER_RANGE_RESOURCE,
    DECAL_CLUSTER_INDEX_RESOURCE,
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
    RESPONSE_LEFT_RESOURCE,
    RESPONSE_RIGHT_RESOURCE,
    RESPONSE_CONTROLS_RESOURCE,
    RESPONSE_AUXILIARY_RESOURCE,
    WATER_COVERAGE_RESOURCE,
    CUSTOM_RESPONSE_RESOURCE,
    SURFACE_NORMAL_RESOURCE,
    SURFACE_ID_RESOURCE,
    SURFACE_MATERIAL_HEADERS_RESOURCE,
    SURFACE_EXECUTION_PLANS_RESOURCE,
    SURFACE_EXECUTION_PROGRAMS_RESOURCE,
    SURFACE_LIGHTING_MAP_RESOURCE,
    SURFACE_PARAMS_RESOURCE,
    SURFACE_IDENTITY_RESOURCE,
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

const fn decal_buffer_template(key: ResourceKey, bytes: u64) -> ResourceTemplate {
    ResourceTemplate {
        key,
        lifetime: ResourceLifetime::Persistent,
        layout: ResourceLayout::Buffer {
            bytes,
            usage: BufferUsages::STORAGE.union(BufferUsages::COPY_SRC),
        },
        alignment: 16,
    }
}

const fn response_resource_template(
    key: ResourceKey,
    lifetime: ResourceLifetime,
    bytes: u64,
    alignment: u64,
) -> ResourceTemplate {
    ResourceTemplate {
        key,
        lifetime,
        layout: ResourceLayout::Buffer {
            bytes,
            usage: BufferUsages::STORAGE,
        },
        alignment,
    }
}

const fn volume_texture_template(
    key: ResourceKey,
    dimension: TextureDimension,
    format: TextureFormat,
    usage: TextureUsages,
) -> ResourceTemplate {
    ResourceTemplate {
        key,
        lifetime: ResourceLifetime::Persistent,
        layout: ResourceLayout::Texture {
            extent: [1, 1, 1],
            dimension,
            format,
            mip_levels: 1,
            samples: 1,
            usage,
        },
        alignment: 1,
    }
}

pub const BUILTIN_RESOURCE_TEMPLATES: &[ResourceTemplate] = &[
    resource_template(WATER_SURFACE_RESOURCE),
    resource_template(WATER_FOAM_RESOURCE),
    volume_texture_template(
        VOLUME_SCATTERING_RESOURCE,
        TextureDimension::D3,
        TextureFormat::Rgba16Float,
        TextureUsages::STORAGE_BINDING.union(TextureUsages::TEXTURE_BINDING),
    ),
    volume_texture_template(
        VOLUME_EXTINCTION_RESOURCE,
        TextureDimension::D3,
        TextureFormat::Rgba16Float,
        TextureUsages::STORAGE_BINDING.union(TextureUsages::TEXTURE_BINDING),
    ),
    volume_texture_template(
        VOLUME_INTEGRATED_SCATTERING_RESOURCE,
        TextureDimension::D3,
        TextureFormat::Rgba16Float,
        TextureUsages::STORAGE_BINDING.union(TextureUsages::TEXTURE_BINDING),
    ),
    volume_texture_template(
        VOLUME_INTEGRATED_TRANSMITTANCE_RESOURCE,
        TextureDimension::D3,
        TextureFormat::Rgba16Float,
        TextureUsages::STORAGE_BINDING.union(TextureUsages::TEXTURE_BINDING),
    ),
    volume_texture_template(
        VOLUME_FOG_APPLIED_RESOURCE,
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        TextureUsages::STORAGE_BINDING.union(TextureUsages::COPY_SRC),
    ),
    volume_texture_template(
        VOLUME_SCENE_DEPTH_RESOURCE,
        TextureDimension::D2,
        TextureFormat::R32Float,
        TextureUsages::TEXTURE_BINDING
            .union(TextureUsages::STORAGE_BINDING)
            .union(TextureUsages::COPY_SRC),
    ),
    volume_texture_template(
        VOLUME_SCENE_COLOR_RESOURCE,
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        TextureUsages::TEXTURE_BINDING
            .union(TextureUsages::STORAGE_BINDING)
            .union(TextureUsages::COPY_DST),
    ),
    resource_template(VOLUME_DENSITY_RESOURCE),
    resource_template(VOLUME_FROXEL_RESOURCE),
    resource_template(VOLUME_VELOCITY_RESOURCE),
    resource_template(PARTICLE_STREAM_RESOURCE),
    resource_template(PARTICLE_SPRAY_RESOURCE),
    resource_template(PARTICLE_ALIVE_RESOURCE),
    resource_template(PARTICLE_SIMULATION_PARTICLES_RESOURCE),
    resource_template(PARTICLE_SIMULATION_ACTIVE_INDICES_RESOURCE),
    resource_template(PARTICLE_SIMULATION_COMPACT_EVENTS_RESOURCE),
    resource_template(PARTICLE_SIMULATION_CONTROL_RESOURCE),
    resource_template(PARTICLE_BRIDGE_EVENT_FLAGS_RESOURCE),
    resource_template(PARTICLE_RENDER_INPUT_COUNT_RESOURCE),
    resource_template(PARTICLE_BRIDGE_VALIDATION_RESOURCE),
    resource_template(PARTICLE_RENDER_ROUTE_TABLE_RESOURCE),
    resource_template(PARTICLE_BRIDGE_ROUTE_SCRATCH_RESOURCE),
    resource_template(PARTICLE_BRIDGE_ACTIVE_ROUTE_INDICES_RESOURCE),
    resource_template(HAIR_STRAND_RESOURCE),
    resource_template(HAIR_VISIBILITY_RESOURCE),
    resource_template(HAIR_DEEP_OPACITY_RESOURCE),
    decal_buffer_template(DECAL_INSTANCE_RESOURCE, 96),
    decal_buffer_template(DECAL_MATERIAL_RESOURCE, 48),
    decal_buffer_template(DECAL_RECEIVER_RESOURCE, 64),
    decal_buffer_template(DECAL_CLUSTER_RANGE_RESOURCE, 16),
    decal_buffer_template(DECAL_CLUSTER_INDEX_RESOURCE, 4),
    decal_buffer_template(DECAL_RESOLVE_RESOURCE, 48),
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
    crate::water::fabric::ResponseCompositionPlane::Left.template(64),
    crate::water::fabric::ResponseCompositionPlane::Right.template(64),
    crate::water::fabric::ResponseCompositionPlane::Controls.template(16),
    crate::water::fabric::ResponseCompositionPlane::Auxiliary.template(64),
    crate::water::fabric::ResponseCompositionPlane::WaterCoverage.template(4),
    response_resource_template(
        CUSTOM_RESPONSE_RESOURCE,
        ResourceLifetime::Transient,
        64,
        16,
    ),
    // These are catalog defaults only. Native frame owners override all eight
    // layouts with the exact target dimensions and published buffer ranges
    // through `compile_active_with_template_overrides`.
    ResourceTemplate {
        key: SURFACE_NORMAL_RESOURCE,
        lifetime: ResourceLifetime::Persistent,
        layout: ResourceLayout::Texture {
            extent: [1, 1, 1],
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsages::TEXTURE_BINDING,
        },
        alignment: 1,
    },
    ResourceTemplate {
        key: SURFACE_ID_RESOURCE,
        lifetime: ResourceLifetime::Persistent,
        layout: ResourceLayout::Texture {
            extent: [1, 1, 1],
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Uint,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsages::TEXTURE_BINDING,
        },
        alignment: 1,
    },
    resource_template(SURFACE_MATERIAL_HEADERS_RESOURCE),
    resource_template(SURFACE_EXECUTION_PLANS_RESOURCE),
    resource_template(SURFACE_EXECUTION_PROGRAMS_RESOURCE),
    resource_template(SURFACE_LIGHTING_MAP_RESOURCE),
    ResourceTemplate {
        key: SURFACE_PARAMS_RESOURCE,
        lifetime: ResourceLifetime::Persistent,
        layout: ResourceLayout::Buffer {
            bytes: 16,
            usage: BufferUsages::UNIFORM,
        },
        alignment: 16,
    },
    resource_template(SURFACE_IDENTITY_RESOURCE),
];

// Service-owned spray storage stays shared. Domain-owned resources that have
// independent physical storage per authored object/renderer are expanded into
// per-object payload addresses here.
const BUILTIN_OBJECT_ACCESSES: &[ObjectResourceAccess] = &[
    ObjectResourceAccess {
        pass: PassKey(0x5000),
        ordinal: 0,
        domain: Some(RenderDomain::Surface),
        role: PayloadRole(0x510),
        layout: ObjectLayoutPolicy::Sized {
            max_bytes: u64::MAX,
        },
    },
    ObjectResourceAccess {
        pass: PassKey(0x5000),
        ordinal: 1,
        domain: Some(RenderDomain::Surface),
        role: PayloadRole(0x511),
        layout: ObjectLayoutPolicy::Sized {
            max_bytes: u64::MAX,
        },
    },
    ObjectResourceAccess {
        pass: PassKey(0x6020),
        ordinal: 0,
        domain: Some(RenderDomain::Surface),
        role: PayloadRole(0x510),
        layout: ObjectLayoutPolicy::Sized {
            max_bytes: u64::MAX,
        },
    },
    // Particle renderer consumers own independent visible-index buffers. The
    // bridge stream and its input-count/route metadata remain shared, while
    // the renderer's alive-index output is addressed by stable object/history
    // identity.
    ObjectResourceAccess {
        pass: PARTICLE_RENDER_PASS,
        ordinal: 3,
        domain: Some(RenderDomain::Particle),
        role: PayloadRole(0x104),
        layout: ObjectLayoutPolicy::Sized {
            max_bytes: u64::MAX,
        },
    },
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
const SURFACE_FEATURE: FeatureDefinition = FeatureDefinition::subsystem(
    SURFACE_FEATURE_ID,
    RenderDomain::Surface,
    ServiceSet::EMPTY,
    SURFACE_VARIANTS,
);
const LIT_FEATURE: FeatureDefinition =
    FeatureDefinition::response(LIT_FEATURE_ID, ServiceSet::EMPTY, LIT_VARIANTS);
const NPR_FEATURE: FeatureDefinition =
    FeatureDefinition::response(NPR_FEATURE_ID, ServiceSet::EMPTY, NPR_VARIANTS);
const CUSTOM_FEATURE: FeatureDefinition =
    FeatureDefinition::response(CUSTOM_FEATURE_ID, ServiceSet::EMPTY, CUSTOM_VARIANTS);

static BUILTIN_FEATURES: [FeatureDefinition; 26] = [
    WATER_FEATURE,
    VOLUME_FEATURE,
    PARTICLE_FEATURE,
    HAIR_FEATURE,
    DECAL_FEATURE,
    SURFACE_FEATURE,
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

static BUILTIN_SUBSYSTEM_ROOTS: [FeatureId; 6] = [
    WATER_FEATURE_ID,
    VOLUME_FEATURE_ID,
    PARTICLE_FEATURE_ID,
    HAIR_FEATURE_ID,
    DECAL_FEATURE_ID,
    SURFACE_FEATURE_ID,
];
static BUILTIN_RESPONSE_ROOTS: [FeatureId; 3] = [LIT_FEATURE_ID, NPR_FEATURE_ID, CUSTOM_FEATURE_ID];

/// Build the immutable semantic catalog used by renderer/device setup.
pub fn builtin_catalog() -> Result<FeatureCatalog, super::CatalogError> {
    FeatureCatalog::new_with_object_accesses(
        &BUILTIN_FEATURES,
        &BUILTIN_RESOURCE_KEYS,
        BUILTIN_OBJECT_ACCESSES,
    )
}

pub const fn builtin_resource_templates() -> &'static [ResourceTemplate] {
    BUILTIN_RESOURCE_TEMPLATES
}

/// Object-scoped access declarations used by native lowering of built-ins.
/// Keeping this table beside the catalog prevents a caller from accidentally
/// compiling the built-in catalog with a different object-expansion policy.
pub const fn builtin_object_resource_accesses() -> &'static [ObjectResourceAccess] {
    BUILTIN_OBJECT_ACCESSES
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
    use crate::render_fabric::{
        FeatureSelection, PassDescriptorIdentity, RenderQualityTier, ShaderBindingCache,
        ShaderBindingCacheStatus, ShaderBindingError,
    };

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
    fn volumetrics_catalog_matches_the_production_texture_and_pass_contract() {
        let catalog = builtin_catalog().expect("built-in catalog is valid");
        let volume = catalog
            .feature(VOLUME_FEATURE_ID)
            .expect("volume subsystem is catalogued");
        let variant = volume.variants[0];
        assert_eq!(
            variant
                .passes
                .iter()
                .map(|pass| pass.id)
                .collect::<Vec<_>>(),
            [
                VOLUME_SCATTER_PASS,
                VOLUME_INTEGRATE_PASS,
                VOLUME_APPLY_PASS,
                VOLUME_COPY_BACK_PASS,
            ]
        );
        assert_eq!(variant.passes[1].depends_on, &[VOLUME_SCATTER_PASS]);
        assert_eq!(variant.passes[2].depends_on, &[VOLUME_INTEGRATE_PASS]);
        assert_eq!(variant.passes[3].depends_on, &[VOLUME_APPLY_PASS]);

        let accesses = |pass: PassKey| {
            variant
                .passes
                .iter()
                .find(|candidate| candidate.id == pass)
                .expect("volume pass exists")
                .accesses
        };
        assert!(accesses(VOLUME_INTEGRATE_PASS).iter().any(|access| {
            access.key == VOLUME_SCATTERING_RESOURCE
                && access.input == ResourceInput::ProducedBy(VOLUME_SCATTER_PASS)
                && access.access == AccessKind::SampledRead
        }));
        assert!(accesses(VOLUME_APPLY_PASS).iter().any(|access| {
            access.key == VOLUME_INTEGRATED_SCATTERING_RESOURCE
                && access.input == ResourceInput::ProducedBy(VOLUME_INTEGRATE_PASS)
        }));
        assert!(accesses(VOLUME_COPY_BACK_PASS).iter().any(|access| {
            access.key == VOLUME_FOG_APPLIED_RESOURCE
                && access.access == AccessKind::TransferRead
                && access.input == ResourceInput::ProducedBy(VOLUME_APPLY_PASS)
        }));
        assert!(accesses(VOLUME_COPY_BACK_PASS).iter().any(|access| {
            access.key == VOLUME_SCENE_COLOR_RESOURCE
                && access.access == AccessKind::TransferWrite
                && access.input == ResourceInput::Initial
        }));

        let layout = |key| {
            builtin_resource_templates()
                .iter()
                .find(|template| template.key == key)
                .expect("volume resource template exists")
                .layout
        };
        for key in [
            VOLUME_SCATTERING_RESOURCE,
            VOLUME_EXTINCTION_RESOURCE,
            VOLUME_INTEGRATED_SCATTERING_RESOURCE,
            VOLUME_INTEGRATED_TRANSMITTANCE_RESOURCE,
        ] {
            assert!(matches!(
                layout(key),
                ResourceLayout::Texture {
                    dimension: TextureDimension::D3,
                    format: TextureFormat::Rgba16Float,
                    mip_levels: 1,
                    samples: 1,
                    usage,
                    ..
                } if usage.contains(TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING)
            ));
        }
        assert!(matches!(
            layout(VOLUME_SCENE_DEPTH_RESOURCE),
            ResourceLayout::Texture {
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage,
                ..
            } if usage.contains(TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC)
        ));
        assert!(matches!(
            layout(VOLUME_SCENE_COLOR_RESOURCE),
            ResourceLayout::Texture {
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage,
                ..
            } if usage.contains(TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST)
        ));
    }

    #[test]
    fn hair_catalog_tracks_strand_input_visibility_and_deep_opacity_dependencies() {
        let catalog = builtin_catalog().expect("built-in catalog is valid");
        let feature = catalog
            .feature(HAIR_FEATURE_ID)
            .expect("hair is catalogued");
        let variant = feature.variants[0];
        assert_eq!(variant.passes.len(), 2);
        let visibility = variant.passes[0];
        let deep_opacity = variant.passes[1];
        assert_eq!(visibility.id, HAIR_VISIBILITY_PASS);
        assert_eq!(visibility.depends_on, &[]);
        assert_eq!(deep_opacity.id, HAIR_DEEP_OPACITY_PASS);
        assert_eq!(deep_opacity.depends_on, &[HAIR_VISIBILITY_PASS]);
        assert!(visibility.accesses.iter().any(|access| {
            access.key == HAIR_STRAND_RESOURCE && access.access == AccessKind::StorageRead
        }));
        assert!(visibility.accesses.iter().any(|access| {
            access.key == HAIR_VISIBILITY_RESOURCE && access.access == AccessKind::StorageWrite
        }));
        assert!(deep_opacity.accesses.iter().any(|access| {
            access.key == HAIR_VISIBILITY_RESOURCE && access.access == AccessKind::StorageRead
        }));
        assert!(deep_opacity.accesses.iter().any(|access| {
            access.key == HAIR_STRAND_RESOURCE && access.access == AccessKind::StorageRead
        }));
        assert!(deep_opacity.accesses.iter().any(|access| {
            access.key == HAIR_DEEP_OPACITY_RESOURCE && access.access == AccessKind::StorageWrite
        }));
        assert!(variant.passes.iter().all(|pass| {
            pass.accesses.iter().all(|access| {
                !(access.key == HAIR_STRAND_RESOURCE && access.access == AccessKind::StorageWrite)
            })
        }));
    }

    #[test]
    fn particle_catalog_models_bridge_then_renderer_resource_flow() {
        let catalog = builtin_catalog().expect("built-in catalog is valid");
        let feature = catalog
            .feature(PARTICLE_FEATURE_ID)
            .expect("particle subsystem is catalogued");
        let variant = feature.variants[0];
        assert_eq!(variant.passes.len(), 2);
        let bridge = variant.passes[0];
        let renderer = variant.passes[1];
        assert_eq!(bridge.id, PARTICLE_BRIDGE_PASS);
        assert_eq!(bridge.depends_on, &[]);
        assert_eq!(renderer.id, PARTICLE_RENDER_PASS);
        assert_eq!(renderer.depends_on, &[PARTICLE_BRIDGE_PASS]);
        assert!(bridge.accesses.iter().any(|access| {
            access.key == PARTICLE_STREAM_RESOURCE
                && access.input == ResourceInput::Automatic
                && access.access == AccessKind::StorageWrite
        }));
        for key in [
            PARTICLE_SIMULATION_PARTICLES_RESOURCE,
            PARTICLE_SIMULATION_ACTIVE_INDICES_RESOURCE,
            PARTICLE_SIMULATION_COMPACT_EVENTS_RESOURCE,
            PARTICLE_SIMULATION_CONTROL_RESOURCE,
        ] {
            assert!(bridge.accesses.iter().any(|access| {
                access.key == key
                    && access.input == ResourceInput::Initial
                    && access.access == AccessKind::StorageRead
            }));
        }
        for key in [
            PARTICLE_BRIDGE_EVENT_FLAGS_RESOURCE,
            PARTICLE_RENDER_INPUT_COUNT_RESOURCE,
            PARTICLE_BRIDGE_VALIDATION_RESOURCE,
            PARTICLE_RENDER_ROUTE_TABLE_RESOURCE,
            PARTICLE_BRIDGE_ROUTE_SCRATCH_RESOURCE,
            PARTICLE_BRIDGE_ACTIVE_ROUTE_INDICES_RESOURCE,
        ] {
            assert!(bridge
                .accesses
                .iter()
                .any(|access| { access.key == key && access.access == AccessKind::StorageWrite }));
        }
        assert!(renderer.accesses.iter().any(|access| {
            access.key == PARTICLE_STREAM_RESOURCE
                && access.input == ResourceInput::ProducedBy(PARTICLE_BRIDGE_PASS)
                && access.access == AccessKind::StorageRead
        }));
        for key in [
            PARTICLE_RENDER_INPUT_COUNT_RESOURCE,
            PARTICLE_RENDER_ROUTE_TABLE_RESOURCE,
        ] {
            assert!(renderer.accesses.iter().any(|access| {
                access.key == key
                    && access.input == ResourceInput::ProducedBy(PARTICLE_BRIDGE_PASS)
                    && access.access == AccessKind::StorageRead
            }));
        }
        assert!(renderer.accesses.iter().any(|access| {
            access.key == PARTICLE_ALIVE_RESOURCE
                && access.input == ResourceInput::Automatic
                && access.access == AccessKind::StorageWrite
        }));
        assert!(!bridge
            .accesses
            .iter()
            .any(|access| access.key == PARTICLE_ALIVE_RESOURCE));
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
            SURFACE_FEATURE_ID,
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

    #[test]
    fn response_templates_use_the_water_owned_plane_identity_table() {
        let expected = crate::water::fabric::response_composition_templates(64, 64, 16, 64, 4);
        let actual = [
            RESPONSE_LEFT_RESOURCE,
            RESPONSE_RIGHT_RESOURCE,
            RESPONSE_CONTROLS_RESOURCE,
            RESPONSE_AUXILIARY_RESOURCE,
            WATER_COVERAGE_RESOURCE,
        ]
        .map(|key| {
            BUILTIN_RESOURCE_TEMPLATES
                .iter()
                .copied()
                .find(|template| template.key == key)
                .expect("response resource has a builtin template")
        });
        assert_eq!(actual, expected);
        assert_eq!(
            crate::water::fabric::ResponseCompositionPlane::ALL
                .map(crate::water::fabric::ResponseCompositionPlane::key),
            actual.map(|template| template.key)
        );
    }

    #[test]
    fn selected_builtins_carry_the_complete_shader_package_identity() {
        let catalog = builtin_catalog().expect("built-in catalog is valid");
        let roots = builtin_roots(&catalog).expect("all built-in roots are valid");
        let selection = FeatureSelection::resolve(
            &catalog,
            &roots,
            CapabilitySet::EMPTY,
            RenderQualityTier::Low,
        )
        .expect("built-in selection resolves");

        for selected in selection.selected() {
            let binding = selected
                .bind_builtin_shader_package()
                .expect("every built-in selected variant has a package binding");
            assert_eq!(binding.feature_id(), selected.feature_id());
            assert_eq!(binding.variant_id(), selected.variant_id());
            assert_eq!(
                binding.package(),
                binding.module().identity.package,
                "package/module identity must be paired"
            );
            assert_eq!(
                binding.pass_descriptors().len(),
                selected.passes().len(),
                "package pass identity must cover every semantic pass"
            );
            assert_ne!(binding.identity_digest().0, [0; 32]);
        }
    }

    #[test]
    fn shader_binding_rejects_package_permutation_and_pass_mismatches() {
        let catalog = builtin_catalog().expect("built-in catalog is valid");
        let roots = builtin_roots(&catalog).expect("all built-in roots are valid");
        let selection = FeatureSelection::resolve(
            &catalog,
            &roots,
            CapabilitySet::EMPTY,
            RenderQualityTier::Low,
        )
        .expect("built-in selection resolves");
        let selected = selection
            .feature(WATER_FEATURE_ID)
            .expect("water is a root");
        let binding = selected
            .variant()
            .builtin_shader_package()
            .expect("water package exists");

        let mut package_mismatch = binding.clone();
        package_mismatch.package = ShaderPackageId::new("builtin.render_fabric.other");
        assert!(matches!(
            selected.bind_shader_package(package_mismatch),
            Err(ShaderBindingError::PackageModuleMismatch { .. })
        ));

        let mut permutation_mismatch = binding.clone();
        permutation_mismatch.module.permutations = PermutationSpace::new([2]);
        assert!(matches!(
            selected.bind_shader_package(permutation_mismatch),
            Err(ShaderBindingError::ModulePermutationMismatch { .. })
        ));

        let mut pass_mismatch = binding;
        pass_mismatch.pass_descriptors[0] = PassDescriptorIdentity {
            key: PassKey(0xdead),
            ..pass_mismatch.pass_descriptors[0].clone()
        };
        assert!(matches!(
            selected.bind_shader_package(pass_mismatch),
            Err(ShaderBindingError::PassDescriptorMismatch { .. })
        ));
    }

    #[test]
    fn shader_binding_cache_requires_explicit_invalidation_before_replacement() {
        let catalog = builtin_catalog().expect("built-in catalog is valid");
        let roots = builtin_roots(&catalog).expect("all built-in roots are valid");
        let selection = FeatureSelection::resolve(
            &catalog,
            &roots,
            CapabilitySet::EMPTY,
            RenderQualityTier::Low,
        )
        .expect("built-in selection resolves");
        let selected = selection
            .feature(WATER_FEATURE_ID)
            .expect("water is a root");
        let first = selected
            .variant()
            .builtin_shader_package()
            .expect("water package exists");
        let mut cache = ShaderBindingCache::default();
        assert_eq!(
            cache.register(selected.variant(), first.clone()),
            Ok(ShaderBindingCacheStatus::Inserted)
        );
        assert_eq!(
            cache.register(selected.variant(), first.clone()),
            Ok(ShaderBindingCacheStatus::Reused)
        );

        let replacement_package = ShaderPackageId::new("builtin.render_fabric.water.v2");
        let mut replacement = first.clone();
        replacement.package = replacement_package;
        replacement.module.identity.package = replacement_package;
        replacement.module.identity.digest = ArtifactDigest::from_bytes(b"water-v2");
        assert!(matches!(
            cache.register(selected.variant(), replacement.clone()),
            Err(ShaderBindingError::CacheIdentityChanged { .. })
        ));
        assert!(matches!(
            cache.invalidate(selected.variant_id(), &first),
            Ok(())
        ));
        assert_eq!(
            cache.register(selected.variant(), replacement),
            Ok(ShaderBindingCacheStatus::Inserted)
        );
        assert!(matches!(
            cache.invalidate(selected.variant_id(), &first),
            Err(ShaderBindingError::CacheIdentityMismatch { .. })
        ));
        assert_eq!(cache.len(), 1);
    }
}
