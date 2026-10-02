//! Cylindrical colour-space conversions (`HSV` / `HSL`) for texture authoring.
//!
//! See [`hsv`] for the hue/saturation/value and hue/saturation/lightness
//! conversions used by tint, recolour and colour-grading workflows.
//!
//! See [`ycbcr`] for the ITU-R BT.601/BT.709 `YCbCr` codec transforms, and
//! [`ycocg`] for the luma/chroma `YCoCg` transforms (lossy orthogonal
//! float form plus the reversible `YCoCg-R` integer lifting) used by chroma
//! tooling and texture-compressor front-ends.

mod chroma_subsample;
mod hsv;
mod ycbcr;
mod ycocg;

pub use chroma_subsample::{downsample, upsample, ChromaPlane, ChromaSubsampling};
pub use hsv::{hsl_to_rgb, hsv_to_rgb, rgb_to_hsl, rgb_to_hsv};
pub use ycbcr::{rgb_to_ycbcr, ycbcr_to_rgb, YCbCrMatrix};
pub use ycocg::{rgb_to_ycocg, rgb_to_ycocg_r, ycocg_r_to_rgb, ycocg_to_rgb};
