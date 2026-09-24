//! Backend-neutral visibility-buffer and shading-work contracts.

#![forbid(unsafe_code)]
#![expect(
    missing_docs,
    reason = "The experimental shading ABI is documented as it freezes."
)]

extern crate alloc;

mod classification;
mod cloth;
mod environment;
mod lighting;
mod punctual;
mod resolve;
mod subsurface;
mod surface;
mod vecmath;
mod visibility;

pub use classification::{
    classify_material_header, ClassificationError, MaterialShadingClass, ShadingWorkItem,
    ShadingWorkPlan, MAX_SHADING_CLASSES,
};
pub use cloth::evaluate_cloth_direct;
pub use environment::{
    env_brdf_approx, evaluate_image_based_light, ImageBasedLight, SphericalHarmonicsL2,
};
pub use lighting::{
    evaluate_principled_direct, evaluate_toon_direct, linear_furnace_response, DirectLightSample,
    ShadingFrame, SurfaceSample,
};
pub use punctual::PunctualLight;
pub use resolve::{
    resolve_pixel, surface_sample_from_parameters, DirectionalLight, LightingEnvironment,
    ResolveError, ResolveInput, ResolvedPixel,
};
pub use subsurface::evaluate_subsurface_direct;
pub use surface::{
    reconstruct_surface, GpuShadingPrimitive, GpuShadingVertex, SurfaceReconstructionError,
    SurfaceReconstructionFlags, SurfaceReconstructionInput, SurfaceSampleGeometry,
};
pub use visibility::{
    encode_barycentrics, BarycentricError, VisibilityPixel, VisibilityPixelTargets,
    INVALID_VISIBILITY_ID,
    VISIBILITY_BUFFER_ABI_VERSION,
};
