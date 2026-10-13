//! # `prism_image`
//!
//! The image asset: texel data and the description it must be uploaded with.
//!
//! An [`Image`] is an [`Asset`](prism_asset::Asset) holding an [`Extent3d`], a
//! [`TextureFormat`], the packed texel bytes, and the [`SamplerDescriptor`] the
//! image wants to be sampled through. It validates what a device would otherwise
//! discover: a data length that does not match the extents and format, an empty
//! axis, a depth format used as a colour image.
//!
//! It is deliberately the *data* half of the texture path: no ECS, no GPU
//! handles. The renderer's GPU form (texture, view, sampler, upload) is
//! `prism_render_app`'s `GpuImage`, prepared through the render-asset pipeline.

#![forbid(unsafe_code)]

extern crate alloc;

mod artifact;
mod error;
mod image;
mod runtime_loader;
mod upload;

#[cfg(test)]
mod tests;

pub use artifact::{ImageArtifact, ImageArtifactError, IMAGE_ARTIFACT_SCHEMA};
pub use error::ImageError;
pub use image::Image;
pub use runtime_loader::{RuntimeImageLoadError, RuntimeImageLoader};
pub use upload::{ImageUploadPlan, ImageUploadPlanError, IMAGE_UPLOAD_ROW_ALIGNMENT};
