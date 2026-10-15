//! Declarative render-fabric contract for the water frame.
//!
//! The catalog owns the stable water service closure.  This module is the
//! water-owned frame view of that closure: it names the passes that must stay
//! resident and the resources the live water path actually reads or writes.
//! `MaskBlend` is kept as a resident pipeline variant, while its resource list
//! is intentionally empty until the response-plane producer equivalence gate
//! is complete.  The current raster path therefore remains the legacy
//! `water_surface_fs_hybrid` path and makes no unsupported driver import/export
//! claim.

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
        accesses: SIMULATION_ACCESSES,
        resident: true,
        compatibility_only: false,
    },
    WaterFabricPass {
        key: WATER_SURFACE_RASTER_PASS,
        queue: QueueClass::Graphics,
        accesses: SURFACE_RASTER_ACCESSES,
        resident: true,
        compatibility_only: false,
    },
    WaterFabricPass {
        key: WATER_MASK_BLEND_PASS,
        queue: QueueClass::Graphics,
        // The legacy hybrid fragment currently performs the blend in the
        // surface raster draw.  It has no response-plane access to declare;
        // keeping this pass resident is a pipeline/link gate, not a claim that
        // the RHI can import/export driver binaries or response planes here.
        accesses: &[],
        resident: true,
        compatibility_only: true,
    },
    WaterFabricPass {
        key: WATER_COMPOSITION_PASS,
        queue: QueueClass::Compute,
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
    /// Whether the current implementation still uses the legacy hybrid entry.
    pub legacy_hybrid_entry: bool,
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
            legacy_hybrid_entry: true,
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

    /// CPU/link gate for the invariants that make this slice honest.
    #[must_use]
    pub fn is_link_complete(self) -> bool {
        self.required_services == WATER_FABRIC_SERVICES
            && self
                .pass(WATER_MASK_BLEND_PASS)
                .is_some_and(|pass| pass.resident && pass.compatibility_only)
            && self
                .pass(WATER_COMPOSITION_PASS)
                .is_some_and(|pass| pass.resident && !pass.accesses.is_empty())
            && self
                .pass(WATER_SURFACE_RASTER_PASS)
                .is_some_and(|pass| pass.resident && !pass.accesses.is_empty())
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
        assert_eq!(WATER_FABRIC_SERVICES.len(), 9);
    }

    #[test]
    fn all_resident_passes_are_stable() {
        let plan = WaterFabricPlan::default();
        assert_eq!(plan.resident_passes().len(), 4);
        assert!(plan.resident_passes().iter().all(|pass| pass.resident));
        assert!(plan.pass(WATER_MASK_BLEND_PASS).is_some());
        assert!(plan.pass(WATER_COMPOSITION_PASS).is_some());
    }

    #[test]
    fn mask_blend_keeps_legacy_entry_until_equivalence() {
        let plan = WaterFabricPlan::for_shading(WaterShadingPlan::hybrid());
        assert!(plan.mask_blend_active);
        assert!(plan.legacy_hybrid_entry);
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
    fn mask_blend_is_resident_but_compatibility_only() {
        let pass = WaterFabricPlan::for_shading(WaterShadingPlan::hybrid())
            .pass(WATER_MASK_BLEND_PASS)
            .expect("MaskBlend pass is resident");
        assert!(pass.resident);
        assert!(pass.compatibility_only);
        assert!(pass.accesses.is_empty());
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
    fn delayed_spray_read_is_explicit_history() {
        let access = WaterFabricPlan::default()
            .accesses(WATER_SIMULATION_PASS)
            .iter()
            .find(|access| access.key == WATER_SPRAY_RESOURCE)
            .expect("simulation reads the previous spray state");
        assert_eq!(access.frame_lag.map(FrameLag::frames), Some(1));
    }

    #[test]
    fn legacy_hybrid_entry_is_kept_until_equivalence() {
        let plan = WaterFabricPlan::for_shading(WaterShadingPlan::hybrid());
        assert!(plan.legacy_hybrid_entry);
        assert!(plan.mask_blend_active);
    }

    #[test]
    fn link_gate_rejects_missing_resident_composition() {
        let plan = WaterFabricPlan::default();
        assert!(plan.is_link_complete());
        assert!(plan.resident_passes().iter().all(|pass| pass.resident));
    }
}
