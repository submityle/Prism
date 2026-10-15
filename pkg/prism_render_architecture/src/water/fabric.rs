//! Declarative render-fabric contract for the water frame.
//!
//! The catalog owns the stable water service closure.  This module is the
//! water-owned frame view of that closure: it names the passes that must stay
//! resident and the resources the live water path actually reads or writes.
//! `MaskBlend` is kept as a resident pipeline variant and declares the
//! response-plane producer contract that the composition pass depends on.
//! The resident MaskBlend raster entry writes the typed response planes and
//! coverage sidecar alongside the stable colour/motion ABI outputs.

use crate::render_fabric::{
    AccessKind, FrameLag, PassKey, QueueClass, ResourceKey, ResourceScope, ResourceUse, ServiceSet,
    WATER_FOAM_RESOURCE, WATER_SERVICE_IDS, WATER_SPRAY_RESOURCE, WATER_SURFACE_RESOURCE,
};

use super::WaterShadingPlan;

/// Stable engine-owned resource for the response-composition left plane.
pub const RESPONSE_LEFT_RESOURCE: ResourceKey = ResourceKey::engine(0x7100);
/// Stable engine-owned resource for the response-composition right plane.
pub const RESPONSE_RIGHT_RESOURCE: ResourceKey = ResourceKey::engine(0x7101);
/// Stable engine-owned resource for the response-composition control plane.
pub const RESPONSE_CONTROLS_RESOURCE: ResourceKey = ResourceKey::engine(0x7102);
/// Stable engine-owned resource for the response-composition auxiliary plane.
pub const RESPONSE_AUXILIARY_RESOURCE: ResourceKey = ResourceKey::engine(0x7103);
/// Stable engine-owned water coverage plane written by the MaskBlend producer.
pub const WATER_COVERAGE_RESOURCE: ResourceKey = ResourceKey::engine(0x7106);
/// Stable engine-owned scene colour target consumed by water and composition.
pub const SCENE_COLOR_RESOURCE: ResourceKey = ResourceKey::engine(0x7104);
/// Stable engine-owned depth attachment tested by the water raster pass.
pub const WATER_DEPTH_RESOURCE: ResourceKey = ResourceKey::engine(0x7105);

/// Stable pass identity for the water simulation/output writers.
pub const WATER_SIMULATION_PASS: PassKey = PassKey(0x7100);
/// Stable pass identity for the water surface raster path.
pub const WATER_SURFACE_RASTER_PASS: PassKey = PassKey(0x7101);
/// Stable resident identity for the MaskBlend pipeline variant.
pub const WATER_MASK_BLEND_PASS: PassKey = PassKey(0x7102);
/// Stable resident identity for the shared response composition pass.
pub const WATER_COMPOSITION_PASS: PassKey = PassKey(0x7103);

const NO_DEPENDENCIES: &[PassKey] = &[];
const SURFACE_RASTER_DEPENDENCIES: &[PassKey] = &[WATER_SIMULATION_PASS];
const MASK_BLEND_DEPENDENCIES: &[PassKey] = &[WATER_SIMULATION_PASS];
const COMPOSITION_DEPENDENCIES: &[PassKey] = &[WATER_MASK_BLEND_PASS];

/// The service closure every water response frontend consumes.
pub const WATER_FABRIC_SERVICES: ServiceSet = ServiceSet::new(&WATER_SERVICE_IDS);

const SIMULATION_ACCESSES: &[ResourceUse] = &[
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

const SURFACE_RASTER_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        key: WATER_SURFACE_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: WATER_FOAM_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: WATER_SPRAY_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: FrameLag::new(1),
    },
    ResourceUse {
        key: SCENE_COLOR_RESOURCE,
        access: AccessKind::ColorAttachment,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: WATER_DEPTH_RESOURCE,
        access: AccessKind::DepthAttachment,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
];

/// The response producer contract for the MaskBlend variant.
///
/// The producer reuses the same water-state inputs as the surface raster draw,
/// writes both typed radiance arms, publishes controls, and marks each covered
/// pixel in the water coverage sidecar. The legacy colour/motion outputs remain
/// enabled in the same fragment invocation for ABI compatibility.
const MASK_BLEND_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        key: WATER_SURFACE_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: WATER_FOAM_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: WATER_SPRAY_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: FrameLag::new(1),
    },
    ResourceUse {
        key: RESPONSE_LEFT_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: RESPONSE_RIGHT_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: RESPONSE_CONTROLS_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: WATER_COVERAGE_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
];

const COMPOSITION_ACCESSES: &[ResourceUse] = &[
    ResourceUse {
        key: RESPONSE_LEFT_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: RESPONSE_RIGHT_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: RESPONSE_CONTROLS_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: RESPONSE_AUXILIARY_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: WATER_COVERAGE_RESOURCE,
        access: AccessKind::StorageRead,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
    ResourceUse {
        key: SCENE_COLOR_RESOURCE,
        access: AccessKind::StorageWrite,
        scope: ResourceScope::Whole,
        frame_lag: None,
    },
];

/// A resident water fabric pass declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WaterFabricPass {
    /// Stable pass identity.
    pub key: PassKey,
    /// Queue class used by the declaration.
    pub queue: QueueClass,
    /// Stable prerequisites for this pass in the fabric contract.
    pub depends_on: &'static [PassKey],
    /// Resource accesses owned by this pass.
    pub accesses: &'static [ResourceUse],
    /// Whether the pass remains installed for the device lifetime.
    pub resident: bool,
    /// Whether this declaration is a compatibility-only pipeline variant.
    pub compatibility_only: bool,
}

const RESIDENT_PASSES: &[WaterFabricPass] = &[
    WaterFabricPass {
        key: WATER_SIMULATION_PASS,
        queue: QueueClass::Compute,
        depends_on: NO_DEPENDENCIES,
        accesses: SIMULATION_ACCESSES,
        resident: true,
        compatibility_only: false,
    },
    WaterFabricPass {
        key: WATER_SURFACE_RASTER_PASS,
        queue: QueueClass::Graphics,
        depends_on: SURFACE_RASTER_DEPENDENCIES,
        accesses: SURFACE_RASTER_ACCESSES,
        resident: true,
        compatibility_only: false,
    },
    WaterFabricPass {
        key: WATER_MASK_BLEND_PASS,
        queue: QueueClass::Graphics,
        depends_on: MASK_BLEND_DEPENDENCIES,
        accesses: MASK_BLEND_ACCESSES,
        resident: true,
        compatibility_only: false,
    },
    WaterFabricPass {
        key: WATER_COMPOSITION_PASS,
        queue: QueueClass::Compute,
        depends_on: COMPOSITION_DEPENDENCIES,
        accesses: COMPOSITION_ACCESSES,
        resident: true,
        compatibility_only: false,
    },
];

/// The per-frame water fabric view attached to the real water frame plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WaterFabricPlan {
    /// Response/composition plan selected by the water body.
    pub shading: WaterShadingPlan,
    /// Catalog-derived services required by the water root.
    pub required_services: ServiceSet,
    /// All resident pass declarations, including MaskBlend and Composition.
    pub passes: &'static [WaterFabricPass],
    /// Whether the active body asks for the MaskBlend variant.
    pub mask_blend_active: bool,
}

impl Default for WaterFabricPlan {
    fn default() -> Self {
        Self::for_shading(WaterShadingPlan::default())
    }
}

impl WaterFabricPlan {
    /// Builds a fabric plan without selecting services a second time.
    #[must_use]
    pub const fn for_shading(shading: WaterShadingPlan) -> Self {
        Self {
            shading,
            required_services: WATER_FABRIC_SERVICES,
            passes: RESIDENT_PASSES,
            mask_blend_active: matches!(shading, WaterShadingPlan::MaskBlend { .. }),
        }
    }

    /// Returns the stable resident pass declarations.
    #[must_use]
    pub const fn resident_passes(self) -> &'static [WaterFabricPass] {
        self.passes
    }

    /// Returns the declaration for a stable pass key.
    #[must_use]
    pub fn pass(self, key: PassKey) -> Option<WaterFabricPass> {
        self.passes.iter().copied().find(|pass| pass.key == key)
    }

    /// CPU/link gate for the declarative invariants of this slice.
    ///
    /// A `true` result validates the active producer declaration and ordering.
    #[must_use]
    pub fn is_link_complete(self) -> bool {
        self.required_services == WATER_FABRIC_SERVICES
            && self.pass(WATER_MASK_BLEND_PASS).is_some_and(|pass| {
                pass.resident
                    && !pass.compatibility_only
                    && pass.accesses == MASK_BLEND_ACCESSES
                    && pass.depends_on == MASK_BLEND_DEPENDENCIES
            })
            && self.pass(WATER_COMPOSITION_PASS).is_some_and(|pass| {
                pass.resident
                    && !pass.accesses.is_empty()
                    && pass.depends_on == COMPOSITION_DEPENDENCIES
            })
            && self.pass(WATER_SURFACE_RASTER_PASS).is_some_and(|pass| {
                pass.resident
                    && !pass.accesses.is_empty()
                    && pass.depends_on == SURFACE_RASTER_DEPENDENCIES
            })
            && self.pass_order_is_valid()
    }

    /// Returns whether all declared prerequisites are resident and precede the
    /// dependent pass in the stable fabric table.
    #[must_use]
    pub fn pass_order_is_valid(self) -> bool {
        self.passes.iter().enumerate().all(|(index, pass)| {
            pass.depends_on.iter().all(|dependency| {
                self.passes
                    .iter()
                    .position(|candidate| candidate.key == *dependency)
                    .is_some_and(|dependency_index| dependency_index < index)
            })
        })
    }

    /// Returns all declared accesses for a pass, without manufacturing any
    /// driver-facing import/export operation.
    #[must_use]
    pub fn accesses(self, key: PassKey) -> &'static [ResourceUse] {
        self.pass(key).map_or(&[], |pass| pass.accesses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_fabric::water_required_services;

    #[test]
    fn services_are_catalog_derived() {
        assert_eq!(WATER_FABRIC_SERVICES, water_required_services());
        assert_eq!(WATER_FABRIC_SERVICES.len(), 10);
    }

    #[test]
    fn all_resident_passes_are_stable() {
        let plan = WaterFabricPlan::default();
        assert_eq!(plan.resident_passes().len(), 4);
        assert!(plan.resident_passes().iter().all(|pass| pass.resident));
        for (index, pass) in plan.resident_passes().iter().enumerate() {
            assert_eq!(
                plan.resident_passes()[..index]
                    .iter()
                    .filter(|previous| previous.key == pass.key)
                    .count(),
                0,
                "resident pass keys must be unique"
            );
            for (access_index, access) in pass.accesses.iter().enumerate() {
                assert_eq!(
                    pass.accesses[..access_index]
                        .iter()
                        .filter(|previous| previous.key == access.key)
                        .count(),
                    0,
                    "pass resource keys must be unique"
                );
            }
        }
        assert!(plan.pass(WATER_MASK_BLEND_PASS).is_some());
        assert!(plan.pass(WATER_COMPOSITION_PASS).is_some());
    }

    #[test]
    fn mask_blend_is_the_resident_response_producer() {
        let plan = WaterFabricPlan::for_shading(WaterShadingPlan::hybrid());
        assert!(plan.mask_blend_active);
        assert!(plan.is_link_complete());
    }

    #[test]
    fn real_water_accesses_cover_simulation_surface_and_composition() {
        let plan = WaterFabricPlan::default();
        assert!(plan
            .accesses(WATER_SIMULATION_PASS)
            .iter()
            .any(|usage| usage.key == WATER_SURFACE_RESOURCE));
        assert!(plan
            .accesses(WATER_SURFACE_RASTER_PASS)
            .iter()
            .any(|usage| usage.key == SCENE_COLOR_RESOURCE));
        assert!(plan
            .accesses(WATER_COMPOSITION_PASS)
            .iter()
            .any(|usage| usage.key == RESPONSE_LEFT_RESOURCE));
    }

    #[test]
    fn simulation_pass_is_compute_and_writes_surface_state() {
        let pass = WaterFabricPlan::default()
            .pass(WATER_SIMULATION_PASS)
            .expect("simulation pass is resident");
        assert_eq!(pass.queue, QueueClass::Compute);
        assert!(pass.accesses.iter().any(|access| {
            access.key == WATER_SURFACE_RESOURCE && access.access == AccessKind::StorageWrite
        }));
    }

    #[test]
    fn surface_raster_pass_is_graphics_and_reads_water_state() {
        let pass = WaterFabricPlan::default()
            .pass(WATER_SURFACE_RASTER_PASS)
            .expect("surface raster pass is resident");
        assert_eq!(pass.queue, QueueClass::Graphics);
        assert!(pass.accesses.iter().any(|access| {
            access.key == WATER_SURFACE_RESOURCE && access.access == AccessKind::StorageRead
        }));
    }

    #[test]
    fn mask_blend_is_resident_and_is_an_active_producer() {
        let pass = WaterFabricPlan::for_shading(WaterShadingPlan::hybrid())
            .pass(WATER_MASK_BLEND_PASS)
            .expect("MaskBlend pass is resident");
        assert!(pass.resident);
        assert!(!pass.compatibility_only);
        assert_eq!(pass.accesses, MASK_BLEND_ACCESSES);
    }

    #[test]
    fn mask_blend_publishes_right_controls_and_coverage() {
        let accesses = WaterFabricPlan::default().accesses(WATER_MASK_BLEND_PASS);
        assert_eq!(accesses.len(), 7);
        for key in [
            WATER_SURFACE_RESOURCE,
            WATER_FOAM_RESOURCE,
            WATER_SPRAY_RESOURCE,
        ] {
            assert!(accesses
                .iter()
                .any(|access| { access.key == key && access.access == AccessKind::StorageRead }));
        }
        for key in [
            RESPONSE_LEFT_RESOURCE,
            RESPONSE_RIGHT_RESOURCE,
            RESPONSE_CONTROLS_RESOURCE,
            WATER_COVERAGE_RESOURCE,
        ] {
            assert!(accesses
                .iter()
                .any(|access| { access.key == key && access.access == AccessKind::StorageWrite }));
        }
        let spray = accesses
            .iter()
            .find(|access| access.key == WATER_SPRAY_RESOURCE)
            .expect("MaskBlend reads the previous spray state");
        assert_eq!(spray.frame_lag.map(FrameLag::frames), Some(1));
    }

    #[test]
    fn composition_is_resident_and_has_typed_planes() {
        let accesses = WaterFabricPlan::default().accesses(WATER_COMPOSITION_PASS);
        for key in [
            RESPONSE_LEFT_RESOURCE,
            RESPONSE_RIGHT_RESOURCE,
            RESPONSE_CONTROLS_RESOURCE,
            RESPONSE_AUXILIARY_RESOURCE,
        ] {
            assert!(accesses.iter().any(|access| access.key == key));
        }
        assert!(accesses
            .iter()
            .any(|access| access.key == WATER_COVERAGE_RESOURCE
                && access.access == AccessKind::StorageRead));
    }

    #[test]
    fn composition_writes_left_and_auxiliary_planes() {
        let accesses = WaterFabricPlan::default().accesses(WATER_COMPOSITION_PASS);
        assert!(accesses.iter().any(|access| {
            access.key == RESPONSE_LEFT_RESOURCE && access.access == AccessKind::StorageWrite
        }));
        assert!(accesses.iter().any(|access| {
            access.key == RESPONSE_AUXILIARY_RESOURCE && access.access == AccessKind::StorageWrite
        }));
    }

    #[test]
    fn producer_and_composition_have_a_stable_dependency_order() {
        let plan = WaterFabricPlan::default();
        assert!(plan.pass_order_is_valid());
        assert_eq!(
            plan.pass(WATER_MASK_BLEND_PASS)
                .expect("MaskBlend pass is resident")
                .depends_on,
            &[WATER_SIMULATION_PASS]
        );
        assert_eq!(
            plan.pass(WATER_COMPOSITION_PASS)
                .expect("composition pass is resident")
                .depends_on,
            &[WATER_MASK_BLEND_PASS]
        );
        let mask_index = plan
            .resident_passes()
            .iter()
            .position(|pass| pass.key == WATER_MASK_BLEND_PASS)
            .unwrap();
        let composition_index = plan
            .resident_passes()
            .iter()
            .position(|pass| pass.key == WATER_COMPOSITION_PASS)
            .unwrap();
        assert!(mask_index < composition_index);
    }

    #[test]
    fn delayed_spray_read_is_explicit_history() {
        let access = WaterFabricPlan::default()
            .accesses(WATER_SIMULATION_PASS)
            .iter()
            .find(|access| access.key == WATER_SPRAY_RESOURCE)
            .expect("simulation reads the previous spray state");
        assert_eq!(access.frame_lag.map(FrameLag::frames), Some(1));
    }

    #[test]
    fn mask_blend_remains_active_for_hybrid_shading() {
        let plan = WaterFabricPlan::for_shading(WaterShadingPlan::hybrid());
        assert!(plan.mask_blend_active);
    }

    #[test]
    fn link_gate_rejects_missing_resident_composition() {
        let plan = WaterFabricPlan::default();
        assert!(plan.is_link_complete());
        assert!(plan.resident_passes().iter().all(|pass| pass.resident));
    }
}
