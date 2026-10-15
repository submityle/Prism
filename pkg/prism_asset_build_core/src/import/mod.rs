//! Source importers that emit versioned, deterministic runtime artifacts.

mod image;

pub use image::{CanonicalPngProcessor, ImageImportError, IMAGE_ARTIFACT_KIND};
pub use prism_image::{
    ImageArtifact, ImageArtifactError, ImageMipArtifact, IMAGE_ARTIFACT_SCHEMA,
    IMAGE_MIP_ARTIFACT_SCHEMA,
};
