//! Input/output descriptors for the manual texture sampler.
//!
//! These types decouple *what* a shading path wants to sample ([`SampleRequest`])
//! from *how* the sampler resolves it ([`SampleResolved`]), so the resolve
//! functions in [`super`] stay free of ad-hoc tuples. All fields are plain data
//! with no interior mutability, keeping a CPU golden trivially reproducible by a
//! GPU twin.
//!
//! # Conventions
//! * `uv` is the raw, pre-wrap surface coordinate; the resolver folds it through
//!   the per-axis [`WrapMode`]s before any mip/page math.
//! * `max_anisotropy` mirrors the hardware `maxAnisotropy` sampler cap; it is
//!   clamped to `>= 1` downstream, so `0`/`NaN` degrade to isotropic.
//! * `pages` lists the fine-then-coarse virtual-texture pages a trilinear fetch
//!   must have resident; identical entries at the top of the pyramid are the
//!   caller's to dedup.
//! * `taps` are UV positions around the *addressed* centre. Per-tap re-wrapping
//!   for a tiling texture is the caller's responsibility, matching how hardware
//!   anisotropy re-folds each tap -- see [`super`] for the rationale.
//!
//! # References
//! * Vulkan `VkSamplerCreateInfo` (`addressModeU/V`, `maxAnisotropy`).
//! * Akenine-Moller et al., *Real-Time Rendering* 4th ed., Section 6.2.

use super::super::texture_addressing::WrapMode;
use super::super::texture_lod::{AnisoTaps, PageRequest};

/// A manual texture-sample request from a visibility-buffer or ray-traced
/// shading path, where the fixed-function sampler is unavailable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleRequest {
    /// Raw surface UV before wrap addressing.
    pub uv: [f32; 2],
    /// Wrap (address) mode for the U axis.
    pub wrap_u: WrapMode,
    /// Wrap (address) mode for the V axis.
    pub wrap_v: WrapMode,
    /// Maximum anisotropy ratio (hardware `maxAnisotropy`); `>= 1` after clamp.
    pub max_anisotropy: f32,
}

impl SampleRequest {
    /// Build a request with the same wrap mode on both axes and a given
    /// anisotropy cap.
    #[inline]
    #[must_use]
    pub fn new(uv: [f32; 2], wrap: WrapMode, max_anisotropy: f32) -> Self {
        Self {
            uv,
            wrap_u: wrap,
            wrap_v: wrap,
            max_anisotropy,
        }
    }

    /// Build an isotropic (single-tap, `maxAnisotropy = 1`) request with the
    /// same wrap mode on both axes.
    #[inline]
    #[must_use]
    pub fn isotropic(uv: [f32; 2], wrap: WrapMode) -> Self {
        Self::new(uv, wrap, 1.0)
    }
}

/// The resolved sampling plan: everything the fetch loop needs after LOD,
/// addressing, anisotropy and residency have been computed.
#[derive(Clone, Copy, Debug)]
pub struct SampleResolved {
    /// `true` when a [`WrapMode::ClampToBorder`] axis was out of range and the
    /// caller must substitute the sampler border colour instead of fetching.
    pub border: bool,
    /// Continuous (fractional) mip level the fetch should blend around.
    pub lod: f32,
    /// Fine-then-coarse virtual-texture pages that must be resident.
    pub pages: [PageRequest; 2],
    /// Anisotropic sample taps (centre tap only in the isotropic case).
    pub taps: AnisoTaps,
}
