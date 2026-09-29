//! `ExtractSchedule` extraction of the primary directional light's world-space
//! direction for the virtual-shadow-map receiver-generation pass.
//!
//! The receiver-generation shader projects each reconstructed world position
//! onto the light's clipmap plane, whose orthonormal basis
//! ([`prism_render_shading::ReceiverProjection`]) is derived from the light
//! *direction*. The shadow ABI [`super::abi::GpuVsmReceiverGenParams`] already
//! carries that basis, but the render-world shadow records
//! ([`super::super::shadow`]) only retain cascade view-projections, not the raw
//! direction. This system therefore extracts the primary directional light's
//! forward vector directly from the main-world ECS, mirroring the light query
//! [`super::super::shadow::extract_shadows`] already runs.
//!
//! "Primary" is the first visible directional light, matching the convention
//! the cascade extraction uses for its dominant shadow caster. When no
//! directional light is present the resource holds `None` and the
//! receiver-generation prepare step clears its per-view state so the pass is a
//! no-op that frame.

use bevy_ecs::prelude::*;
use bevy_light::DirectionalLight;
use bevy_math::Vec3;
use bevy_render::Extract;
use bevy_transform::components::GlobalTransform;

/// Render-world resource holding the primary directional light's world-space
/// forward direction, or `None` when the scene has no directional light this
/// frame.
///
/// Refreshed every frame by [`extract_vsm_primary_light`]; consumed by
/// [`super::resources::prepare_vsm_receiver_resources`] to build the light's
/// clipmap-plane basis.
#[derive(Resource, Default, Clone, Copy, Debug, PartialEq)]
pub(crate) struct VsmPrimaryLight {
    /// Unit-ish forward direction of the primary directional light (the
    /// direction its rays travel), or `None` when there is no directional light.
    pub direction: Option<Vec3>,
}

/// `ExtractSchedule` system copying the first directional light's world-space
/// forward vector into [`VsmPrimaryLight`].
///
/// Runs alongside [`super::super::shadow::extract_shadows`] and picks the same
/// "first directional light" the cascade fitting treats as dominant, so the
/// virtual shadow map is driven by the scene's primary sun.
pub(crate) fn extract_vsm_primary_light(
    mut primary: ResMut<VsmPrimaryLight>,
    directionals: Extract<Query<&GlobalTransform, With<DirectionalLight>>>,
) {
    primary.direction = directionals
        .iter()
        .next()
        .map(|transform| Vec3::from(transform.forward()));
}
