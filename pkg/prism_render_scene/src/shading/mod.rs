mod ao;
mod bloom;
mod cas;
mod chromatic_aberration;
mod classification_gpu;
mod classification_readback;
mod color_grade;
mod composite;
mod dof;
mod exposure;
mod film_grain;
mod gamut_map;
mod graph;
mod halftone;
mod hatching;
mod ibl;
mod kuwahara;
mod lens_flare;
mod light_routing;
mod motion_blur;
mod ordered_dither;
mod outline;
mod plugin;
mod posterize;
mod raster;
mod resolve;
mod resources;
mod runtime;
mod shadow;
mod ssgi;
mod ssr;
mod taa;
mod tonemap;
mod transparent;
mod upscale;
mod vignette;
mod virtual_shadow;
mod volumetric_clouds;
mod volumetrics;
mod world_space_gi;

pub use plugin::PrismShadingPlugin;
pub use runtime::{PrismShadingDiagnostics, PrismShadingSettings};

/// Re-exported for the water-surface raster draw: the water pipeline keys its
/// specialized render pipeline on the same per-view visibility-buffer path the
/// shading passes build, so the sibling `water` module needs the component.
pub(crate) use composite::composite_shading;
pub(crate) use resources::ViewVisibilityBuffer;
// Re-export the per-view reverse-Z Hi-Z "nearest depth" pyramid so the
// transparent water surface pass can march it directly for screen-space
// reflections (see `water::surface_ssr`), reusing the opaque prepass output.
pub(crate) use ssr::ViewSsrTextures;
// Re-exports the water surface pass consumes to shadow its directional term
// with the same demand-paged virtual shadow map the opaque resolve pass reads.
pub(crate) use resolve::GpuVsmResolveParams;
pub(crate) use virtual_shadow::{
    PrismVirtualShadowSettings, ViewVsmPageTable, ViewVsmPhysicalAtlas, VsmPrimaryLight,
};
