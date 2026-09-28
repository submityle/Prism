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
    /// GTAO sampling radius in world units; larger gathers more distant
    /// occluders. Clamped to a small positive minimum by the kernel.
    pub gtao_world_radius: f32,
    /// Fraction of the radius (`0..=1`) at which GTAO distance falloff begins.
    pub gtao_falloff: f32,
    /// GTAO occlusion contrast exponent; `> 1` darkens contact shadows.
    pub gtao_power: f32,
    /// GTAO slice directions swept per pixel (clamped `>= 1`). More slices
    /// reduce banding at proportional cost.
    pub gtao_slice_count: u32,
    /// GTAO marched samples per side per slice (clamped `>= 1`). More steps
    /// catch thinner horizons.
    pub gtao_steps_per_slice: u32,
    /// GTAO spatial-denoise bilateral kernel half-width in pixels (`0` disables
    /// the blur). The denoise covers `(2 * radius + 1)^2` taps.
    pub gtao_denoise_radius: u32,
    /// GTAO denoise Gaussian spatial falloff in pixels; larger smooths harder.
    pub gtao_denoise_spatial_sigma: f32,
    /// GTAO denoise depth edge-stop tolerance as a fraction of the centre
    /// pixel's view depth; smaller preserves silhouettes more aggressively.
    pub gtao_denoise_depth_sigma: f32,
    /// GTAO denoise normal edge-stop sharpness; higher rejects tilted
    /// neighbours faster, preserving occlusion contrast along curved edges.
    pub gtao_denoise_normal_power: f32,
    /// Enables the GTAO temporal accumulation pass. Requires `enable_gtao`;
    /// the pass reprojects last frame's converged AO through the previous
    /// frame's camera transform, variance-clips it to the current
    /// neighbourhood, and exponentially blends the fresh estimate to kill the
    /// under-motion boil. Off by default.
    pub enable_gtao_temporal: bool,
    /// Fraction of the reprojected GTAO history kept when it agrees with the
    /// current neighbourhood (`0..=1`); higher integrates more frames.
    pub gtao_temporal_history_weight: f32,
    /// Floor the GTAO temporal blend weight decays toward on a disocclusion
    /// (`0..=history_weight`); higher keeps a touch of smoothing through it.
    pub gtao_temporal_min_history_weight: f32,
    /// Stddev multiplier for the GTAO temporal variance clip band
    /// (`mean ± gamma·σ`); higher accepts more history before rejecting it.
    pub gtao_temporal_variance_gamma: f32,
    /// Enables image-based lighting: precomputes the split-sum environment
    /// BRDF ("DFG") table so the resolve stage can reconstruct specular
    /// reflectance from prefiltered radiance. Off by default.
    pub enable_ibl: bool,
    /// GGX importance samples integrated per DFG-table texel (clamped `>= 1`).
    /// More samples reduce the table's high-roughness noise at one-time cost.
    pub ibl_dfg_sample_count: u32,
    /// GGX importance samples convolved per prefiltered-radiance texel
    /// (clamped `>= 1`). More samples reduce speckle in glossy reflections
    /// at one-time cost; the convolution reruns only when the probe changes.
    pub ibl_prefilter_sample_count: u32,
    /// Enables the screen-space reflection (SSR) geometry prepass. Requires the
    /// visibility buffer; the prepass rebuilds the reverse-Z device depth and
    /// view-space normal the SSR trace consumes (there is no G-buffer). Off by
    /// default.
    pub enable_ssr: bool,
    /// Enables the temporal anti-aliasing (TAA) resolve compute pass. Requires
    /// the visibility buffer (for the composited `scene_color` and the
    /// motion-vector G-buffer); the pass motion-reprojects and YCoCg-variance
    /// blends the previous frame into the current one. Off by default. With no
    /// camera jitter yet injected it is a conservative temporal denoise on a
    /// static camera and a mild smoother under motion; the sub-pixel jitter
    /// that turns it into full supersampling lands in a follow-up.
    pub enable_taa: bool,
    /// Enables the screen-space global illumination (SSGI) gather. Requires the
    /// visibility buffer and SSR (it reuses SSR's rebuilt reverse-Z Hi-Z pyramid,
    /// packed `normal_roughness` and current-frame colour pyramid); the gather
    /// casts cosine-weighted hemisphere rays that pick up one indirect diffuse
    /// bounce of on-screen radiance, blended over the resolve's IBL/SH ambient
    /// under a confidence. Off by default.
    pub enable_ssgi: bool,
    /// Cosine-weighted hemisphere rays cast per pixel by the SSGI gather
    /// (clamped `>= 1`). More rays reduce the gather's noise at a linear march
    /// cost ahead of the dedicated denoise stage.
    pub ssgi_sample_count: u32,
    /// View-space march length (view units) for each SSGI hemisphere ray. A
    /// fixed budget keeps the gather bounded independent of scene scale; the
    /// confidence distance-fade tapers the tail so an over-long ray never
    /// hard-cuts.
    pub ssgi_max_distance: f32,
    /// Enables the histogram auto-exposure + eye-adaptation compute passes.
    /// Requires the visibility buffer (it meters the resolved HDR `scene_color`).
    /// The per-view exposure state buffer is always created and always bound by
    /// the composite, so when this is `false` the composite multiplies by a
    /// stationary `1.0` and the metering/adaptation passes are skipped. Off by
    /// default.
    pub enable_exposure: bool,
}

impl Default for PrismShadingSettings {
    fn default() -> Self {
        Self {
            max_visible_pixels: 3840 * 2160,
            enable_visibility_buffer: false,
            enable_compute_resolve: false,
            enable_gtao: false,
            gtao_world_radius: 1.0,
            gtao_falloff: 0.6,
            gtao_power: 1.0,
            gtao_slice_count: 4,
            gtao_steps_per_slice: 8,
            gtao_denoise_radius: 2,
            gtao_denoise_spatial_sigma: 2.0,
            gtao_denoise_depth_sigma: 0.05,
            gtao_denoise_normal_power: 8.0,
            enable_gtao_temporal: false,
            gtao_temporal_history_weight: 0.9,
            gtao_temporal_min_history_weight: 0.0,
            gtao_temporal_variance_gamma: 1.0,
            enable_ibl: false,
            ibl_dfg_sample_count: 1024,
            ibl_prefilter_sample_count: 256,
            enable_ssr: false,
            enable_taa: false,
            enable_ssgi: false,
            ssgi_sample_count: 8,
            ssgi_max_distance: 8.0,
            enable_exposure: false,
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
                .map(|record| record.header(0, 0, 0, 0))
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
        npr.illumination = prism_render_material::Illumination::Stylized as u32;
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
