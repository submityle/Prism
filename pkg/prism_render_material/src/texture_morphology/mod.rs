//! Separable grayscale morphology (flat structuring element).
//!
//! See [`separable`] for the `(2*radius + 1)`-square min/max morphology
//! operators used to grow, shrink, open and close coverage masks and signed
//! distance fields.

mod separable;

pub use separable::{close_plane, dilate_plane, erode_plane, open_plane};
