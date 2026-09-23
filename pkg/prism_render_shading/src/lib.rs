//! Backend-neutral visibility-buffer and shading-work contracts.

#![forbid(unsafe_code)]
#![expect(
    missing_docs,
    reason = "The experimental shading ABI is documented as it freezes."
)]

extern crate alloc;

mod classification;
mod visibility;

pub use classification::{
    classify_material_header, ClassificationError, MaterialShadingClass, ShadingWorkItem,
    ShadingWorkPlan, MAX_SHADING_CLASSES,
};
pub use visibility::{
    encode_barycentrics, BarycentricError, VisibilityPixel, INVALID_VISIBILITY_ID,
    VISIBILITY_BUFFER_ABI_VERSION,
};
