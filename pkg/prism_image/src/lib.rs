//! # `prism_image`
//!
//! Engine-neutral image data for Prism's own renderer, and the RHI upload path
//! that turns it into a GPU texture. This is the `bevy_image` replacement on
//! Prism's render path: it holds decoded pixels on the CPU, describes their
//! format in terms the RHI understands, and hands them to a backend without
//! naming one.
//!
//! ## Layout
//! - [`Image`] — the CPU container: tightly packed pixel bytes plus an
//!   [`Extent3d`](prism_render_driver::Extent3d).
//! - [`ImageFormat`] — the pixel format, mapped one-to-one onto the RHI
//!   [`TextureFormat`](prism_render_driver::TextureFormat).
//! - Generators ([`checkerboard`], [`gradient_horizontal`],
//!   [`gradient_vertical`]) — deterministic builders for tests and demos.
//! - [`upload`] / [`Image::upload`] — create the texture and write the bytes
//!   through a [`RenderDevice`](prism_render_driver::RenderDevice) /
//!   [`RenderQueue`](prism_render_driver::RenderQueue).
//!
//! The crate is deliberately backend-agnostic: it depends only on the RHI
//! trait surface, so the same image uploads unchanged on every backend.
#![forbid(unsafe_code)]

mod format;
mod generators;
mod gpu;
mod image;

#[cfg(test)]
mod tests;

pub use format::ImageFormat;
pub use generators::{checkerboard, gradient_horizontal, gradient_vertical};
pub use gpu::{GpuImage, upload};
pub use image::{Image, ImageError};
