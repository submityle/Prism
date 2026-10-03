//! Main-world component tagging a polygonal (rectangle) area light.
//!
//! An [`AreaLight`] is the authoring-side emitter the render world extracts into
//! the std430 [`GpuAreaLight`](super::abi::GpuAreaLight) record every frame
//! (see [`super::extract`]). Its world placement comes entirely from the
//! entity's [`GlobalTransform`](bevy_transform::components::GlobalTransform):
//!
//! * the emitter centre is the transform translation,
//! * the in-plane U axis (half-extent [`AreaLight::half_width`]) is the
//!   transform's local `+X` (`right`),
//! * the in-plane V axis (half-extent [`AreaLight::half_height`]) is the local
//!   `+Y` (`up`), and
//! * the radiating face normal is `cross(U, V)` = the local `+Z`
//!   (`GlobalTransform::back`). Orient the entity so that `+Z` faces the scene,
//!   or set [`AreaLight::two_sided`] to radiate from both faces.
//!
//! The non-uniform transform scale stretches the half-extents along each axis,
//! so an entity scaled `(2, 3, 1)` doubles the width and triples the height.
//! This mirrors the golden rectangle parameterization in
//! [`prism_render_shading::gi::area_light`] exactly; the extraction unit tests
//! pin the conversion.

use bevy_color::Color;
use bevy_ecs::prelude::Component;

/// A rectangle (quad) area light authored in the main world.
///
/// Attach it alongside a `Transform` / `GlobalTransform`; the render world turns
/// it into one [`GpuAreaLight`](super::abi::GpuAreaLight) rectangle record the
/// clustered-lighting resolve integrates with Linearly Transformed Cosines.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub(crate) struct AreaLight {
    /// Linear emitter colour (alpha ignored).
    pub color: Color,
    /// Scalar radiance multiplier applied to [`AreaLight::color`].
    pub intensity: f32,
    /// Half-extent along the local `+X` axis in world units (before scale).
    pub half_width: f32,
    /// Half-extent along the local `+Y` axis in world units (before scale).
    pub half_height: f32,
    /// When `true` the emitter radiates from both faces; when `false` only the
    /// local `+Z` face lights the scene.
    pub two_sided: bool,
    /// Influence radius in world units; `0` (or negative) disables the smooth
    /// range falloff so the light reaches the whole scene.
    pub range: f32,
}

impl Default for AreaLight {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            intensity: 1.0,
            half_width: 0.5,
            half_height: 0.5,
            two_sided: false,
            range: 0.0,
        }
    }
}

impl AreaLight {
    /// Builds a one-sided rectangle emitter from its half-extents, colour and
    /// radiance multiplier. The placement is supplied by the entity transform.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "authoring builders exercised by the unit tests; the extract path reads fields directly until the public authoring API lands"
        )
    )]
    pub(crate) fn rect(half_width: f32, half_height: f32, color: Color, intensity: f32) -> Self {
        Self {
            color,
            intensity,
            half_width,
            half_height,
            two_sided: false,
            range: 0.0,
        }
    }

    /// Returns a copy radiating from both faces.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "authoring builders exercised by the unit tests; the extract path reads fields directly until the public authoring API lands"
        )
    )]
    pub(crate) fn double_sided(mut self) -> Self {
        self.two_sided = true;
        self
    }

    /// Returns a copy with a finite influence radius (smooth range falloff).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "authoring builders exercised by the unit tests; the extract path reads fields directly until the public authoring API lands"
        )
    )]
    pub(crate) fn with_range(mut self, range: f32) -> Self {
        self.range = range;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_a_unit_one_sided_white_quad() {
        let light = AreaLight::default();
        assert_eq!(light.color, Color::WHITE);
        assert_eq!(light.intensity, 1.0);
        assert_eq!(light.half_width, 0.5);
        assert_eq!(light.half_height, 0.5);
        assert!(!light.two_sided);
        assert_eq!(light.range, 0.0);
    }

    #[test]
    fn rect_builder_sets_extents_and_keeps_one_sided() {
        let light = AreaLight::rect(2.0, 0.75, Color::srgb(0.1, 0.2, 0.3), 4.0);
        assert_eq!(light.half_width, 2.0);
        assert_eq!(light.half_height, 0.75);
        assert_eq!(light.intensity, 4.0);
        assert!(!light.two_sided);
    }

    #[test]
    fn modifiers_flip_sidedness_and_range() {
        let light = AreaLight::rect(1.0, 1.0, Color::WHITE, 1.0)
            .double_sided()
            .with_range(12.5);
        assert!(light.two_sided);
        assert_eq!(light.range, 12.5);
    }
}
