//! ABI shared between the chromatic-aberration compute pass and
//! `shaders/chromatic_aberration.wesl`.
//!
//! The subsystem is a single full-screen pass, so it carries exactly one
//! immediate (push-constant) block, [`GpuChromaticAberrationParams`], mirroring
//! the shader's single `var<immediate>` global. Every field mirrors its shader
//! struct byte-for-byte so machines with and without a GPU agree with the CPU
//! golden in [`prism_render_shading::chromatic_aberration`].
//!
//! The block leads with the `vec2<f32>` optical `center` (WGSL align 8), then
//! the `intensity` and spectral `samples` scalars, then the `vec2<u32>`
//! framebuffer `screen_size`. WGSL places `screen_size` at offset 16 (the next
//! multiple of its 8-byte alignment), which happens to fall exactly after the
//! two scalars with no padding, so the block is a tight 24 bytes on both sides.

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismChromaticAberrationSettings;

/// Workgroup size (per axis) of the chromatic-aberration compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `chromatic_aberration.wesl`; the
/// dispatch rounds the framebuffer extent up to a multiple of this on both axes
/// and the shader bounds-checks every invocation against `screen_size`.
pub(crate) const CHROMATIC_ABERRATION_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `chromatic_aberration_main`
/// entry point.
///
/// Mirrors the shader's `ChromaticAberrationParams`: the optical `center` in
/// `uv` space, the split `intensity`, the spectral `samples` count and the
/// framebuffer `screen_size` in texels. Layout is byte-identical to the WGSL
/// struct (see the module docs): 24 tight bytes, no padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuChromaticAberrationParams {
    /// Optical centre in `uv` space; the radial split fans out from here
    /// (golden `center`).
    pub center: [f32; 2],
    /// Split strength; `0` collapses every channel offset (golden `intensity`).
    pub intensity: f32,
    /// Spectral tap count carried for parity with the golden; the shipped
    /// three-tap pass reads the red/green/blue endpoints (golden `samples`).
    pub samples: u32,
    /// Full-resolution framebuffer extent in texels, for the per-pixel `uv`
    /// reconstruction and the invocation bounds check.
    pub screen_size: [u32; 2],
}

impl GpuChromaticAberrationParams {
    /// Builds the immediate block from the framebuffer extent and the live
    /// [`PrismChromaticAberrationSettings`].
    ///
    /// The optics (`center`, `intensity`, `samples`) come straight from the
    /// settings — which fold the golden [`prism_render_shading::ChromaticAberrationParams`]
    /// defaults — so the GPU twin sees exactly the controls the CPU reference
    /// was validated against.
    #[must_use]
    pub(crate) fn from_settings(
        screen_size: UVec2,
        settings: &PrismChromaticAberrationSettings,
    ) -> Self {
        Self {
            center: settings.center,
            intensity: settings.intensity,
            samples: settings.samples,
            screen_size: [screen_size.x, screen_size.y],
        }
    }
}
