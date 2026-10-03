//! Render-world extraction of main-world [`AreaLight`] components into the flat
//! std430 [`GpuAreaLight`] records the clustered-lighting resolve integrates.
//!
//! This system runs in [`ExtractSchedule`](bevy_render::ExtractSchedule) and
//! mirrors every visible [`AreaLight`] entity into one rectangle record. The
//! world placement comes entirely from the entity's
//! [`GlobalTransform`]: the emitter centre is the translation, the in-plane U
//! axis is the rotated local `+X`, the in-plane V axis is the rotated local
//! `+Y`, and the non-uniform scale stretches the half-extents along each axis
//! (so an entity scaled `(2, 3, 1)` doubles the width and triples the height).
//! This matches the authoring contract documented on [`AreaLight`] and the
//! golden rectangle parameterization in
//! [`prism_render_shading::gi::area_light`]; the conversion is pinned by the
//! unit tests below.
//!
//! Entities whose [`ViewVisibility`] has been computed to hidden are skipped so
//! the GPU buffer only carries contributing emitters; entities without the
//! component (lights added without the full visibility bundle) are visible.

use bevy_camera::visibility::ViewVisibility;
use bevy_color::{Color, ColorToComponents};
use bevy_ecs::prelude::*;
use bevy_math::Vec3;
use bevy_render::Extract;
use bevy_transform::components::GlobalTransform;

use super::abi::GpuAreaLight;
use super::component::AreaLight;

/// The area lights extracted from the main world for a single frame.
///
/// Mirrors the single GPU storage buffer one-to-one; the resolve pass loops
/// over the whole array (shape and degenerate-extent guards in the shader skip
/// non-contributing records).
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub(crate) struct ExtractedAreaLights {
    /// Every visible rectangle area light, already in world space.
    pub lights: Vec<GpuAreaLight>,
}

impl ExtractedAreaLights {
    /// Clears the per-frame collection while keeping the allocation.
    fn clear(&mut self) {
        self.lights.clear();
    }
}

/// Returns the linear RGB triple of a Bevy [`Color`], dropping alpha.
fn linear_rgb(color: Color) -> [f32; 3] {
    let [red, green, blue, _] = color.to_linear().to_f32_array();
    [red, green, blue]
}

/// Returns `true` when a view-visibility component is present and computed off.
fn is_hidden(visibility: Option<&ViewVisibility>) -> bool {
    visibility.is_some_and(|visibility| !visibility.get())
}

/// Converts one authoring [`AreaLight`] plus its world transform into the flat
/// std430 [`GpuAreaLight`] rectangle record.
///
/// Split out from the system so the unit tests can pin the transform math
/// without constructing a render world.
fn to_gpu(light: &AreaLight, transform: &GlobalTransform) -> GpuAreaLight {
    let (scale, rotation, translation) = transform.to_scale_rotation_translation();
    let axis_u = rotation * Vec3::X;
    let axis_v = rotation * Vec3::Y;
    let mut gpu = GpuAreaLight::rect(
        translation.to_array(),
        axis_u.to_array(),
        axis_v.to_array(),
        light.half_width * scale.x,
        light.half_height * scale.y,
        linear_rgb(light.color),
        light.intensity,
    );
    gpu.two_sided = u32::from(light.two_sided);
    gpu.range = light.range.max(0.0);
    gpu
}

/// Extracts every visible [`AreaLight`] into [`ExtractedAreaLights`].
pub(crate) fn extract_area_lights(
    mut extracted: ResMut<ExtractedAreaLights>,
    lights: Extract<Query<(&AreaLight, &GlobalTransform, Option<&ViewVisibility>)>>,
) {
    extracted.clear();
    for (light, transform, visibility) in &lights {
        if is_hidden(visibility) {
            continue;
        }
        extracted.lights.push(to_gpu(light, transform));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::{Quat, Vec3};
    use core::f32::consts::FRAC_PI_2;

    #[test]
    fn identity_transform_keeps_axes_and_extents() {
        let light = AreaLight::rect(2.0, 0.5, Color::WHITE, 3.0);
        let transform = GlobalTransform::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let gpu = to_gpu(&light, &transform);
        assert_eq!([gpu.pos_x, gpu.pos_y, gpu.pos_z], [1.0, 2.0, 3.0]);
        // Local +X / +Y under identity rotation.
        assert!((gpu.axis_u_x - 1.0).abs() < 1e-6);
        assert!(gpu.axis_u_y.abs() < 1e-6);
        assert!((gpu.axis_v_y - 1.0).abs() < 1e-6);
        assert_eq!(gpu.half_width, 2.0);
        assert_eq!(gpu.half_height, 0.5);
        assert_eq!(gpu.intensity, 3.0);
        assert_eq!(gpu.two_sided, 0);
        assert_eq!(gpu.range, 0.0);
    }

    #[test]
    fn scale_stretches_half_extents_per_axis() {
        let light = AreaLight::rect(1.0, 1.0, Color::WHITE, 1.0);
        let transform = GlobalTransform::from_scale(Vec3::new(2.0, 3.0, 1.0));
        let gpu = to_gpu(&light, &transform);
        assert!((gpu.half_width - 2.0).abs() < 1e-6);
        assert!((gpu.half_height - 3.0).abs() < 1e-6);
    }

    #[test]
    fn rotation_rotates_the_in_plane_axes() {
        let light = AreaLight::rect(1.0, 1.0, Color::WHITE, 1.0);
        // 90 deg about +Z sends local +X -> +Y and local +Y -> -X.
        let transform = GlobalTransform::from_rotation(Quat::from_rotation_z(FRAC_PI_2));
        let gpu = to_gpu(&light, &transform);
        assert!(gpu.axis_u_x.abs() < 1e-6);
        assert!((gpu.axis_u_y - 1.0).abs() < 1e-6);
        assert!((gpu.axis_v_x + 1.0).abs() < 1e-6);
        assert!(gpu.axis_v_y.abs() < 1e-6);
    }

    #[test]
    fn double_sided_and_range_round_trip() {
        let light = AreaLight::rect(1.0, 1.0, Color::WHITE, 1.0)
            .double_sided()
            .with_range(5.0);
        let gpu = to_gpu(&light, &GlobalTransform::IDENTITY);
        assert_eq!(gpu.two_sided, 1);
        assert_eq!(gpu.range, 5.0);
    }

    #[test]
    fn negative_range_is_clamped_to_zero() {
        let light = AreaLight::rect(1.0, 1.0, Color::WHITE, 1.0).with_range(-3.0);
        let gpu = to_gpu(&light, &GlobalTransform::IDENTITY);
        assert_eq!(gpu.range, 0.0);
    }
}
