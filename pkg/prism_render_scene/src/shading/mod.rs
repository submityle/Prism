mod ao;
mod area_light;
mod bloom;
mod cas;
mod chromatic_aberration;
mod classification_gpu;
mod classification_readback;
mod color_grade;
mod composite;
pub mod composition;
mod ddgi;
mod dof;
mod execution;
mod exposure;
mod film_grain;
mod gamut_map;
pub mod glint;
mod graph;
mod halftone;
mod hatching;
mod ibl;
mod kuwahara;
mod lens_flare;
mod light_routing;
mod lobe_budget;
pub mod lobe_sampling;
pub mod measured_brdf;
mod motion_blur;
mod ordered_dither;
mod outline;
mod plugin;
mod post_graph;
mod posterize;
mod raster;
mod resolve;
mod resources;
pub mod response_gpu;
mod runtime;
mod shadow;
mod sky;
mod spec_denoise;
mod spec_gi;
pub mod specular_aa;
mod ssgi;
mod ssr;
mod surface_cache;
mod taa;
pub mod texel_cache;
pub mod thin_film;
mod tonemap;
mod transparent;
mod upscale;
mod vignette;
mod virtual_shadow;
mod volumetric_clouds;
mod volumetrics;
mod world_restir;
mod world_space_gi;

pub use plugin::PrismShadingPlugin;
pub use runtime::{PrismShadingDiagnostics, PrismShadingSettings};

/// Re-exported for the water-surface raster draw: the water pipeline keys its
/// specialized render pipeline on the same per-view visibility-buffer path the
/// shading passes build, so the sibling `water` module needs the component.
pub(crate) use composite::composite_shading;
pub(crate) use composition::ViewResponseComposition;
pub(crate) use execution::{ExternalExecutionSite, ShadingExternalExecutors};
pub(crate) use resources::ViewVisibilityBuffer;
// Re-export the shared motion-vector G-buffer format so the transparent
// water surface pass declares a byte-compatible second render target when
// it writes its own screen-space motion over the pixels it covers (see
// `water::surface_motion`).
pub(crate) use resources::{MOTION_VECTOR_FORMAT, REACTIVE_MASK_FORMAT};
// Re-export the per-view reverse-Z Hi-Z "nearest depth" pyramid so the
// transparent water surface pass can march it directly for screen-space
// reflections (see `water::surface_ssr`), reusing the opaque prepass output.
pub(crate) use ssr::ViewSsrTextures;
// Re-exports the water surface pass consumes to shadow its directional term
// with the same demand-paged virtual shadow map the opaque resolve pass reads.
pub(crate) use resolve::GpuVsmResolveParams;
// Re-export the opt-in world-space (`Lumen`-style) GI settings and the per-view
// resident screen-probe field so the transparent water surface pass can gather
// the same far-field indirect diffuse the opaque resolve pass produces (see
// `water::surface_world_space_gi`).
pub(crate) use virtual_shadow::{
    PrismVirtualShadowSettings, ViewVsmPageTable, ViewVsmPhysicalAtlas, VsmPrimaryLight,
};
pub(crate) use world_space_gi::{PrismWorldSpaceGiSettings, ViewWorldSpaceGi};
