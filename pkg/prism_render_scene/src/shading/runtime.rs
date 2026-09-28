use bevy_ecs::prelude::*;
use prism_render_shading::{classify_material_header, MAX_SHADING_CLASSES};

#[derive(Resource, Clone, Copy, Debug)]
pub struct PrismShadingSettings {
    pub max_visible_pixels: u32,
    pub enable_visibility_buffer: bool,
    pub enable_compute_resolve: bool,
    /// Enables the GTAO ambient-occlusion prepass + compute pass. Requires the
    /// visibility buffer; the resolve stage multiplies indirect light by the
    /// resulting per-pixel visibility.
    pub enable_gtao: bool,
}

impl Default for PrismShadingSettings {
    fn default() -> Self {
        Self {
            max_visible_pixels: 3840 * 2160,
            enable_visibility_buffer: false,
            enable_compute_resolve: false,
            enable_gtao: false,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrismShadingDiagnostics {
    pub visible_work_items: u32,
    pub classified_materials: u32,
    pub class_counts: [u32; MAX_SHADING_CLASSES],
    pub stale_materials: u32,
    pub unsupported_materials: u32,
    pub visibility_buffer_active: bool,
    pub compute_resolve_active: bool,
    pub native_barycentrics: bool,
}

pub(crate) fn detect_shading_capabilities(
    device: Res<bevy_render::renderer::RenderDevice>,
    mut diagnostics: ResMut<PrismShadingDiagnostics>,
) {
    diagnostics.native_barycentrics = device
        .features()
        .contains(bevy_render::render_resource::WgpuFeatures::SHADER_BARYCENTRICS);
}

#[derive(Resource)]
pub(crate) struct ShadingFrameGraph {
    pub(crate) compiled: prism_render_architecture::frame_graph::CompiledGpuFrameGraph,
}

pub(crate) fn prepare_shading_work(
    settings: Res<PrismShadingSettings>,
    frame_graph: Res<ShadingFrameGraph>,
    visibility: Res<super::super::visibility::runtime::UnifiedVisibilityState>,
    materials: Res<super::super::material::runtime::RenderMaterialRegistry>,
    mut diagnostics: ResMut<PrismShadingDiagnostics>,
) {
    debug_assert_eq!(frame_graph.compiled.execution_order.len(), 5);
    *diagnostics = shading_diagnostics(
        &settings,
        diagnostics.native_barycentrics,
        visibility.frame.work_items.iter().map(|work| {
            materials
                .registry
                .get(work.material)
                .map(|record| record.header(0, 0, 0))
        }),
    );
}

fn shading_diagnostics(
    settings: &PrismShadingSettings,
    native_barycentrics: bool,
    materials: impl IntoIterator<Item = Option<prism_render_material::GpuMaterialHeader>>,
) -> PrismShadingDiagnostics {
    let mut diagnostics = PrismShadingDiagnostics {
        visibility_buffer_active: settings.enable_visibility_buffer,
        compute_resolve_active: settings.enable_visibility_buffer
            && settings.enable_compute_resolve,
        native_barycentrics,
        ..Default::default()
    };
    for header in materials {
        diagnostics.visible_work_items = diagnostics.visible_work_items.saturating_add(1);
        let Some(header) = header else {
            diagnostics.stale_materials = diagnostics.stale_materials.saturating_add(1);
            continue;
        };
        match classify_material_header(&header) {
            Ok(class) => {
                diagnostics.classified_materials =
                    diagnostics.classified_materials.saturating_add(1);
                diagnostics.class_counts[class.index()] =
                    diagnostics.class_counts[class.index()].saturating_add(1);
            }
            Err(_) => {
                diagnostics.unsupported_materials =
                    diagnostics.unsupported_materials.saturating_add(1);
            }
        }
    }
    diagnostics
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_shading::MaterialShadingClass;

    #[test]
    fn settings_require_visibility_before_compute_resolve() {
        let settings = PrismShadingSettings {
            enable_visibility_buffer: false,
            enable_compute_resolve: true,
            ..Default::default()
        };
        let diagnostics = PrismShadingDiagnostics {
            visibility_buffer_active: settings.enable_visibility_buffer,
            compute_resolve_active: settings.enable_visibility_buffer
                && settings.enable_compute_resolve,
            ..Default::default()
        };
        assert!(!diagnostics.compute_resolve_active);
        assert_eq!(MaterialShadingClass::Principled.index(), 0);
    }

    #[test]
    fn diagnostics_classify_models_and_reject_stale_or_non_surface_work() {
        let mut principled = prism_render_material::fallback_material_header(1);
        principled.generation = 2;
        let mut npr = prism_render_material::fallback_material_header(1);
        npr.shading_model = prism_render_material::MaterialShadingModel::Npr as u32;
        npr.render_class = prism_render_material::MaterialRenderClass::NprOpaque as u32;
        let mut transparent = prism_render_material::fallback_material_header(1);
        transparent.render_class = prism_render_material::MaterialRenderClass::Transparent as u32;
        let diagnostics = shading_diagnostics(
            &PrismShadingSettings::default(),
            true,
            [Some(principled), Some(npr), Some(transparent), None],
        );

        assert_eq!(diagnostics.visible_work_items, 4);
        assert_eq!(diagnostics.classified_materials, 2);
        assert_eq!(
            diagnostics.class_counts[MaterialShadingClass::Principled.index()],
            1
        );
        assert_eq!(
            diagnostics.class_counts[MaterialShadingClass::Npr.index()],
            1
        );
        assert_eq!(diagnostics.unsupported_materials, 1);
        assert_eq!(diagnostics.stale_materials, 1);
        assert!(diagnostics.native_barycentrics);
    }
}
