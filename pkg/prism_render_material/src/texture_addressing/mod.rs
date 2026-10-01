//! Texture coordinate addressing: apply a sampler wrap mode to a UV before any
//! mip / page math runs.
//!
//! Hardware samplers fold out-of-range UVs through a per-axis *wrap mode*
//! (repeat, clamp, mirror, border) before a fetch. In a visibility-buffer or
//! ray-traced shading path the fetch is issued manually, so the wrap transform
//! must be reproduced explicitly -- otherwise tiling materials seam and
//! `texture_lod::residency` addresses the wrong page. This module performs that
//! transform in closed form, matching the Vulkan/OpenGL/D3D addressing modes,
//! with no learning path so a CPU golden mirrors a GPU twin exactly.
//!
//! # Conventions
//! * Inputs are normalized UVs; the normalized unit cell is `[0, 1)`.
//! * Every mode returns a coordinate inside `[0, 1]` plus a `border` flag; only
//!   [`WrapMode::ClampToBorder`] ever sets `border`, signalling the caller to
//!   substitute the sampler border colour instead of a texel fetch.
//! * All transforms are total and finite: a non-finite input collapses to `0.0`
//!   (non-border) rather than propagating `NaN` into page addressing.
//!
//! # References
//! Vulkan `VkSamplerAddressMode`; OpenGL `GL_TEXTURE_WRAP_*`; Direct3D
//! `D3D12_TEXTURE_ADDRESS_MODE`.

mod wrap;

pub use wrap::{address_uv, wrap_coord, AddressResult, WrapMode};
