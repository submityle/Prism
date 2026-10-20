//! Read-only admission audit for the first Fabric -> response-composition slice.
//!
//! The current built-in Fabric selection is intentionally not lowered here.
//! This module records the exact reasons the lowering is unavailable while the
//! existing `response_composition_pass` remains the sole
//! real compositor recorder. It never selects a feature, allocates a resource,
//! creates a recorder, records commands, or submits work.

use bevy_ecs::prelude::*;
use bevy_log::warn;
use prism_render_architecture::render_fabric::{DeviceFabric, FeatureId, VariantId};

use crate::gpu::fabric::SceneFabric;

use crate::shading::composition::{
    ResidentCompositionPipelines, ResponseCompositionTarget, ViewResponseComposition,
};

/// The stable names used by the already-installed native composition path.
/// They are observations, not Fabric pass/resource declarations.
pub(crate) const RESPONSE_COMPOSITION_RECORDER: &str = "response_composition_pass";
pub(crate) const COMPOSITION_PLANES: [&str; 5] = [
    "response_left",
    "response_right",
    "response_controls",
    "response_auxiliary",
    "water_coverage",
];

/// Why the current frozen built-in selection cannot be prepared as a real
/// Fabric runtime graph for response composition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum FabricCompositionBlocker {
    /// A frozen response root selected no Fabric pass, so there is no stable
    /// pass key to which a real composition recorder could be bound.
    ResponseRootHasNoPass {
        feature: FeatureId,
        variant: VariantId,
    },
    /// The resident composition planes are owned by ViewResponseComposition,
    /// but the built-in Fabric declarations do not expose matching templates
    /// and frame bindings for them.
    ResidentPlanesHaveNoFabricBindings,
    /// The existing native recorder already owns this operation; adding a
    /// second RuntimePassBinding would record the response twice.
    ExistingRecorderIsTheOnlyProductionRecorder,
}

/// Read-only audit result kept in the render world for diagnostics/tests.
#[derive(Resource, Clone, Debug, Eq, PartialEq)]
pub(crate) struct FabricCompositionRuntimeDiagnostic {
    blockers: Vec<FabricCompositionBlocker>,
    resident_pipelines_present: bool,
    views_with_resident_composition: usize,
    views_with_prepared_targets: usize,
    warning_emitted: bool,
}

impl FabricCompositionRuntimeDiagnostic {
    /// Builds the structural result from the one device-lifetime selection.
    /// This reads `DeviceFabric::selection`; it never resolves another one.
    pub(crate) fn from_frozen_selection(fabric: &DeviceFabric) -> Self {
        let mut blockers = fabric
            .selection()
            .roots()
            .responses()
            .iter()
            .filter_map(|&feature| {
                let selected = fabric
                    .admit(feature)
                    .expect("frozen response root must be admitted by its own selection");
                selected.passes().is_empty().then_some(
                    FabricCompositionBlocker::ResponseRootHasNoPass {
                        feature,
                        variant: selected.variant_id(),
                    },
                )
            })
            .collect::<Vec<_>>();

        // The actual plane names are owned by ViewResponseComposition. The
        // built-in catalog currently declares only subsystem/service storage
        // templates and has no composition pass/resource binding contract.
        blockers.push(FabricCompositionBlocker::ResidentPlanesHaveNoFabricBindings);
        blockers.push(FabricCompositionBlocker::ExistingRecorderIsTheOnlyProductionRecorder);

        Self {
            blockers,
            resident_pipelines_present: false,
            views_with_resident_composition: 0,
            views_with_prepared_targets: 0,
            warning_emitted: false,
        }
    }

    pub(crate) fn blockers(&self) -> &[FabricCompositionBlocker] {
        &self.blockers
    }

    pub(crate) fn is_available(&self) -> bool {
        self.blockers.is_empty()
    }

    pub(crate) fn observe_resources(
        &mut self,
        resident_pipelines_present: bool,
        views_with_resident_composition: usize,
        views_with_prepared_targets: usize,
    ) {
        self.resident_pipelines_present = resident_pipelines_present;
        self.views_with_resident_composition = views_with_resident_composition;
        self.views_with_prepared_targets = views_with_prepared_targets;
    }

    pub(crate) fn resident_pipelines_present(&self) -> bool {
        self.resident_pipelines_present
    }

    pub(crate) fn views_with_resident_composition(&self) -> usize {
        self.views_with_resident_composition
    }

    pub(crate) fn views_with_prepared_targets(&self) -> usize {
        self.views_with_prepared_targets
    }
}

/// Installs the structural audit after the existing SceneFabric is frozen.
pub(crate) fn init_fabric_composition_runtime(mut commands: Commands, fabric: Res<SceneFabric>) {
    let mut diagnostic = FabricCompositionRuntimeDiagnostic::from_frozen_selection(&fabric.0);
    emit_warning_once(&mut diagnostic);
    commands.insert_resource(diagnostic);
}

/// Observes the real resident composition resources after their normal Bevy
/// prepare systems run. This is intentionally observation-only: it does not
/// import, allocate, publish, record, or replace `response_composition_pass`.
pub(crate) fn observe_fabric_composition_resources(
    mut diagnostic: ResMut<FabricCompositionRuntimeDiagnostic>,
    pipelines: Option<Res<ResidentCompositionPipelines>>,
    views: Query<(
        Option<&ViewResponseComposition>,
        Option<&ResponseCompositionTarget>,
    )>,
) {
    let mut resident = 0;
    let mut targets = 0;
    for (responses, target) in &views {
        if responses.is_some() {
            resident += 1;
        }
        if target.is_some() {
            targets += 1;
        }
    }
    diagnostic.observe_resources(pipelines.is_some(), resident, targets);
}

fn emit_warning_once(diagnostic: &mut FabricCompositionRuntimeDiagnostic) {
    if diagnostic.warning_emitted || diagnostic.is_available() {
        return;
    }
    warn!(
        "Fabric -> composition runtime unavailable: frozen response selection has no composition pass/resource lowering; preserve the existing {} recorder. planes={:?}, blockers={:?}",
        RESPONSE_COMPOSITION_RECORDER,
        COMPOSITION_PLANES,
        diagnostic.blockers
    );
    diagnostic.warning_emitted = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::{
        backend::capability::CapabilitySet,
        render_fabric::{builtin_catalog, builtin_roots, RenderQualityTier},
    };

    #[test]
    fn built_in_frozen_responses_report_the_real_unavailable_slice() {
        let catalog = builtin_catalog().expect("built-in catalog");
        let roots = builtin_roots(&catalog).expect("built-in roots");
        let fabric = DeviceFabric::new(
            catalog,
            roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::High,
        )
        .expect("built-in selection");
        let snapshot = fabric.selection().clone();
        let diagnostic = FabricCompositionRuntimeDiagnostic::from_frozen_selection(&fabric);

        assert!(!diagnostic.is_available());
        assert_eq!(
            diagnostic
                .blockers()
                .iter()
                .filter(|blocker| matches!(
                    blocker,
                    FabricCompositionBlocker::ResponseRootHasNoPass { .. }
                ))
                .count(),
            fabric.selection().roots().responses().len()
        );
        assert!(diagnostic
            .blockers()
            .contains(&FabricCompositionBlocker::ResidentPlanesHaveNoFabricBindings));
        assert!(diagnostic
            .blockers()
            .contains(&FabricCompositionBlocker::ExistingRecorderIsTheOnlyProductionRecorder));
        assert_eq!(fabric.selection(), &snapshot);
    }

    #[test]
    fn resource_observation_does_not_change_structural_admission() {
        let catalog = builtin_catalog().expect("built-in catalog");
        let roots = builtin_roots(&catalog).expect("built-in roots");
        let fabric = DeviceFabric::new(
            catalog,
            roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::High,
        )
        .expect("built-in selection");
        let mut diagnostic = FabricCompositionRuntimeDiagnostic::from_frozen_selection(&fabric);
        let blockers = diagnostic.blockers().to_vec();

        diagnostic.observe_resources(true, 3, 3);

        assert_eq!(diagnostic.blockers(), blockers.as_slice());
        assert!(!diagnostic.is_available());
        assert!(diagnostic.resident_pipelines_present());
        assert_eq!(diagnostic.views_with_resident_composition(), 3);
        assert_eq!(diagnostic.views_with_prepared_targets(), 3);
    }
}
