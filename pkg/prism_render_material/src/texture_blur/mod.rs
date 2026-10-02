//! Native-resolution image blurring for decoded `RGBA8` textures.
//!
//! Where [`texture_mipgen`](crate::texture_mipgen) reduces resolution and
//! [`texture_resize`](crate::texture_resize) changes it, this module blurs an
//! image at its own resolution -- the fixed-function primitive behind bloom,
//! SSAO / shadow denoise, screen-space diffusion, and coverage / prefilter
//! softening. See [`gaussian`] for the separable, gamma-correct, non-negative
//! Gaussian blur and [`box_blur`] for the running-sum box blur.
//!
//! Everything is deterministic analytic `f32` math -- no AI/ML -- so a CPU
//! golden matches a GPU compute blur to floating-point tolerance.

mod box_blur;
mod gaussian;

pub use box_blur::{box_blur, box_blur_plane};
pub use gaussian::{blur_plane, gaussian_blur, gaussian_weights_1d};
