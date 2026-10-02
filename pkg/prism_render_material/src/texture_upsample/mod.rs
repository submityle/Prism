//! Edge-aware **guided upsampling** of a low-resolution signal.
//!
//! See [`guided`] for joint bilateral upsampling (Kopf et al. 2007): a cheap
//! half/quarter-res solve (AO, GI, shadows, SSS) is upsampled to full
//! resolution under a full-res guide so edges stay crisp.

mod guided;
mod image;

pub use guided::joint_bilateral_upsample_plane;
pub use image::joint_bilateral_upsample_rgba8;
