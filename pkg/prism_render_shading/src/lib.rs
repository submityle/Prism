//! Backend-neutral visibility-buffer and shading-work contracts.

#![forbid(unsafe_code)]
#![expect(
    missing_docs,
    reason = "The experimental shading ABI is documented as it freezes."
)]

extern crate alloc;

mod classification;
mod lighting;
mod visibility;

pub use classification::{
    classify_material_header, ClassificationError, MaterialShadingClass, ShadingWorkItem,
    ShadingWorkPlan, MAX_SHADING_CLASSES,
};
pub use lighting::{
    evaluate_principled_direct, evaluate_toon_direct, linear_furnace_response, DirectLightSample,
    ShadingFrame, SurfaceSample,
};
pub use visibility::{
    encode_barycentrics, BarycentricError, VisibilityPixel, VisibilityPixelTargets,
    INVALID_VISIBILITY_ID,
    VISIBILITY_BUFFER_ABI_VERSION,
};
