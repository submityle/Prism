//! Bind-group layout for the virtual-shadow-map caster depth pass's `@group(0)`
//! per-page uniform.
//!
//! The single entry -- a dynamic-offset uniform holding one
//! [`GpuVsmCasterDepthView`](super::abi::GpuVsmCasterDepthView) per resident page
//! -- is declared once here and shared by both the specialized pipeline's
//! `@group(0)` ([`super::pipeline::VsmCasterDepthPipeline`]) and the uniform
//! buffer's bind group ([`super::pipeline::VsmCasterDepthViewUniform`]), so the
//! two layouts can never drift apart. `@group(1)` (the shared GPU-scene storage
//! table) is owned by [`crate::buffers::GpuSceneBindGroup`] and reused verbatim,
//! exactly as the classic shadow depth pass does.

use bevy_material::bind_group_layout_entries::{binding_types::uniform_buffer, BindGroupLayoutEntries};
use bevy_render::render_resource::{BindGroupLayoutEntry, ShaderStages};

use super::abi::GpuVsmCasterDepthView;

/// The single `@group(0)` layout entry for the caster-depth per-page uniform.
///
/// Bound with a dynamic offset (`true`) so one buffer holds every page's
/// projection and each draw selects its slice; visible to the vertex stage
/// (transforms casters) and the fragment stage (kept identical to the classic
/// shadow view layout so the two passes' `@group(0)` are interchangeable at the
/// binding level).
pub(crate) fn caster_depth_view_layout_entries() -> [BindGroupLayoutEntry; 1] {
    BindGroupLayoutEntries::single(
        ShaderStages::VERTEX | ShaderStages::FRAGMENT,
        uniform_buffer::<GpuVsmCasterDepthView>(true),
    )
}
