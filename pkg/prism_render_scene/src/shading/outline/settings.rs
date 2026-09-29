//! Render-world resource gating the outline pass and feeding the golden
//! tunables into the [`GpuOutlineParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::OutlineParams`] defaults so the
//! render-world resource and the CPU reference stay in lockstep, with three
//! device-side additions the golden does not model: a master `enabled` gate, an
//! authored `line_color` and a coverage->opacity `line_strength`.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `OutlineParams`; a game can overwrite the resource to
//! retune globally without touching any pass code, mirroring
//! [`super::super::color_grade`]'s `PrismColorGradeSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::{Mat4, UVec2};
use prism_render_shading::OutlineParams;

use super::abi::GpuOutlineParams;

/// Global outline settings consumed by the outline pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass
/// allocates nothing and dispatches nothing). Every edge control ships the
/// golden default, so the on-device edge reduction is the exact golden twin; a
/// host raises `enabled` and dials in the line colour and thresholds.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismOutlineSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Relative depth-jump threshold where the depth edge begins (golden
    /// `depth_threshold`).
    pub depth_threshold: f32,
    /// Half-width of the depth-edge transition (golden `depth_softness`).
    pub depth_softness: f32,
    /// Normal-turn threshold where the crease edge begins (golden
    /// `normal_threshold`).
    pub normal_threshold: f32,
    /// Half-width of the crease-edge transition (golden `normal_softness`).
    pub normal_softness: f32,
    /// When `true`, a differing neighbour id draws a hard edge (golden
    /// `id_edges`). Inert until an outline-id G-buffer exists in the binding
    /// layout; see [`GpuOutlineParams::id_edges`].
    pub id_edges: bool,
    /// Authored ink line colour (device-only), composited over the shaded pixel
    /// by the outline coverage. Default is black.
    pub line_color: [f32; 3],
    /// Coverage->opacity strength in `[0, 1]` (device-only) scaling the golden
    /// coverage before the composite lerp. `1` draws the coverage at full
    /// opacity.
    pub line_strength: f32,
}

impl Default for PrismOutlineSettings {
    fn default() -> Self {
        // Fold the golden `OutlineParams` defaults so the render-world resource
        // and the CPU reference stay in lockstep. `enabled` is off (opt-in) and
        // the line is full-strength black for when a host dials it in.
        let golden = OutlineParams::default();
        Self {
            // Opt-in: the subsystem allocates and dispatches nothing until the
            // host flips this on.
            enabled: false,
            depth_threshold: golden.depth_threshold,
            depth_softness: golden.depth_softness,
            normal_threshold: golden.normal_threshold,
            normal_softness: golden.normal_softness,
            id_edges: golden.id_edges,
            // Full-strength black ink by default.
            line_color: [0.0, 0.0, 0.0],
            line_strength: 1.0,
        }
    }
}

impl PrismOutlineSettings {
    /// Builds the [`GpuOutlineParams`] immediate block for a framebuffer of the
    /// given extent from the inverse projection and these settings.
    pub(crate) fn params(&self, view_from_clip: Mat4, size: UVec2) -> GpuOutlineParams {
        GpuOutlineParams::from_settings(view_from_clip, size, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_outline_params() {
        let settings = PrismOutlineSettings::default();
        let golden = OutlineParams::default();
        assert_eq!(settings.depth_threshold, golden.depth_threshold);
        assert_eq!(settings.depth_softness, golden.depth_softness);
        assert_eq!(settings.normal_threshold, golden.normal_threshold);
        assert_eq!(settings.normal_softness, golden.normal_softness);
        assert_eq!(settings.id_edges, golden.id_edges);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
        // Full-strength black ink.
        assert_eq!(settings.line_color, [0.0, 0.0, 0.0]);
        assert_eq!(settings.line_strength, 1.0);
    }

    #[test]
    fn params_folds_the_extent_matrix_and_thresholds() {
        let settings = PrismOutlineSettings::default();
        let params = settings.params(Mat4::IDENTITY, UVec2::new(1280, 720));
        assert_eq!(params.view_from_clip, Mat4::IDENTITY.to_cols_array());
        assert_eq!(params.screen_size, [1280, 720]);
        assert_eq!(params.depth_threshold, settings.depth_threshold);
        assert_eq!(params.normal_threshold, settings.normal_threshold);
        assert_eq!(params.id_edges, 1);
    }
}
