//! Analytic texture level-of-detail (mip) selection for visibility-buffer and
//! ray-traced shading, where the fixed-function hardware screen-space
//! derivatives used by a classic forward raster pass are unavailable.
//!
//! Two complementary, fully classic-numerical strategies are provided:
//!
//! * [`RayCone`] — a single scalar cone (width + spread angle) propagated along
//!   a path. Cheap (a few FMAs per hit) and robust across reflection/refraction
//!   bounces. Based on Akenine-Moller et al., *Texture Level of Detail
//!   Strategies for Real-Time Ray Tracing* (Ray Tracing Gems, 2019, ch. 20).
//! * [`RayDifferential`] — Igehy-style screen-space UV partial derivatives,
//!   giving isotropic and anisotropic LOD identical to what hardware
//!   `textureGrad` would compute for a primary visibility hit. Based on Igehy,
//!   *Tracing Ray Differentials* (SIGGRAPH 1999).
//!
//! Both feed a shared [`TriangleLodConstant`] (the per-triangle texel/world
//! area ratio Delta) and a shared clamping policy in [`mip`].
//!
//! # Conventions
//! * Right-handed, world-space positions in meters; UVs in the unit square
//!   `[0, 1]^2` before wrapping; texture dimensions in texels.
//! * Mip 0 is the finest level; larger mip = coarser. Returned LODs are
//!   continuous (fractional) and clamped to `[0, max_mip]`.
//! * All inputs are defensively clamped so degenerate geometry (zero-area
//!   triangle, grazing `n . d -> 0`, zero-width cone) yields a finite,
//!   well-defined coarse LOD instead of `NaN`/`inf`.
//! * No AI/ML/neural path: these are closed-form analytic estimators that a CPU
//!   golden can reproduce bit-for-bit against a GPU twin.
//!
//! # References
//! * Akenine-Moller, Nilsson, Andersson, Barre-Brisebois, Deng, Wyman,
//!   "Texture Level of Detail Strategies for Real-Time Ray Tracing",
//!   Ray Tracing Gems, 2019.
//! * Igehy, "Tracing Ray Differentials", SIGGRAPH 1999.
//! * Ewins et al., "MIP-Map Level Selection for Texture Mapping", 1998.

mod math;
mod mip;
mod ray_cone;
mod ray_differential;
mod triangle;

pub use mip::{
    cone_mip_level, mip_from_isotropic_footprint, AnisotropicMip, MIN_COS_INCIDENCE,
};
pub use ray_cone::RayCone;
pub use ray_differential::RayDifferential;
pub use triangle::TriangleLodConstant;
