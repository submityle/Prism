//! Elliptical Weighted Average (EWA) anisotropic texture filtering.
//!
//! See [`elliptical`] for the reference-quality anisotropic sampler that
//! gathers a Gaussian-weighted texel ellipse from a screen-space UV footprint,
//! the gold-standard alternative to the uniform-tap
//! [`filter_resolved`](crate::filter_resolved).
//!
//! Everything is deterministic analytic `f32` math -- no AI/ML -- so a CPU
//! golden matches a GPU compute EWA to floating-point tolerance.

mod elliptical;

pub use elliptical::{ewa_sample_plane, ewa_sample_rgba8};
