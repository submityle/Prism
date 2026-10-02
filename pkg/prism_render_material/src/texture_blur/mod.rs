//! Native-resolution image blurring for decoded `RGBA8` textures.
//!
//! Where [`texture_mipgen`](crate::texture_mipgen) reduces resolution and
//! [`texture_resize`](crate::texture_resize) changes it, this module blurs an
//! image at its own resolution -- the fixed-function primitive behind bloom,
//! SSAO / shadow denoise, screen-space diffusion, and coverage / prefilter
//! softening. See [`gaussian`] for the separable, gamma-correct, non-negative
//! Gaussian blur, [`box_blur`] for the running-sum box blur,
//! [`bilateral`] for the edge-preserving bilateral blur,
//! [`joint_bilateral`] for the guide-steered joint / cross bilateral, and
//! [`unsharp`] for unsharp-mask sharpening.
//!
//! Everything is deterministic analytic `f32` math -- no AI/ML -- so a CPU
//! golden matches a GPU compute blur to floating-point tolerance.

mod bilateral;
mod box_blur;
mod gaussian;
mod joint_bilateral;
mod unsharp;

pub use bilateral::{bilateral_blur, bilateral_blur_plane};
pub use box_blur::{box_blur, box_blur_plane};
pub use gaussian::{blur_plane, gaussian_blur, gaussian_weights_1d};
pub use joint_bilateral::{joint_bilateral_blur, joint_bilateral_blur_plane};
pub use unsharp::{unsharp_mask, unsharp_mask_plane};
