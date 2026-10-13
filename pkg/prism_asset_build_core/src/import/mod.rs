//! Source importers that emit versioned, deterministic runtime artifacts.

mod image;

pub use image::{
    CanonicalPngProcessor, ImageArtifact, ImageImportError, IMAGE_ARTIFACT_KIND,
    IMAGE_ARTIFACT_SCHEMA,
};
