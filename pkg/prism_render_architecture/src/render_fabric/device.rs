//! The device-lifetime owner. Frame consumers cannot replace its selection.
use super::*;
use crate::backend::capability::CapabilitySet;

#[derive(Debug)]
pub struct DeviceFabric {
    catalog: FeatureCatalog,
    selection: FeatureSelection,
}

impl DeviceFabric {
    pub fn new(
        catalog: FeatureCatalog,
        roots: FrozenRoots,
        caps: CapabilitySet,
        quality: RenderQualityTier,
    ) -> Result<Self, SelectionError> {
        let selection = FeatureSelection::resolve(&catalog, &roots, caps, quality)?;
        Ok(Self { catalog, selection })
    }

    pub fn selection(&self) -> &FeatureSelection {
        &self.selection
    }

    /// Admission is a root check, not a dependency-closure check.
    pub fn admit(&self, feature: FeatureId) -> Result<&SelectedFeature, RootError> {
        self.selection.roots().require_root(feature)?;
        Ok(self.selection.feature(feature).expect("resolved root"))
    }

    /// Graph consumers use exactly the selection used by pipeline/queue admission.
    pub fn compile(
        &self,
        templates: &'static [ResourceTemplate],
        frame: u64,
        bindings: &[FrameResourceBinding],
        items: &[FrameRenderItem],
        limits: GraphLimits,
    ) -> Result<CompiledFeatureGraph, FabricCompileError> {
        ResourceTemplates::new(&self.catalog, templates)?.compile(
            &self.selection,
            frame,
            bindings,
            items,
            limits,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_admission_does_not_expand_roots_or_reselect() {
        let catalog = builtin_catalog().unwrap();
        let roots =
            FrozenRoots::from_build_profile(&catalog, &[WATER_FEATURE_ID], &[LIT_FEATURE_ID])
                .unwrap();
        let device = DeviceFabric::new(
            catalog,
            roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::Low,
        )
        .unwrap();
        let snapshot = device.selection().clone();
        assert!(device.admit(WATER_FEATURE_ID).is_ok());
        assert_eq!(
            device.admit(PARTICLE_FEATURE_ID).unwrap_err(),
            RootError::OutsideFrozenRoots(PARTICLE_FEATURE_ID)
        );
        assert_eq!(
            device.admit(NPR_FEATURE_ID).unwrap_err(),
            RootError::OutsideFrozenRoots(NPR_FEATURE_ID)
        );
        // Selected service != root permission.
        assert!(device
            .selection()
            .feature(SPRAY_FEEDBACK_FEATURE_ID)
            .is_some());
        assert!(device.admit(SPRAY_FEEDBACK_FEATURE_ID).is_err());
        assert_eq!(device.selection(), &snapshot);
    }
}
