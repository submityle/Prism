//! View-space reconstruction helpers shared by the GTAO horizon search.
//!
//! The screen-space AO pass only has the depth prepass (linear view depth) and
//! the view-space normal buffer to work with, so every sample must be lifted
//! back into view space before horizon angles can be measured.  A pinhole
//! camera is fully described by the tangents of its half field-of-view angles,
//! which is exactly what the GPU passes down from the projection matrix
//! (`tan_half_fov_x = 1 / P[0][0]`, `tan_half_fov_y = 1 / P[1][1]`).

/// Pinhole projection parameters needed to lift a screen sample into view
/// space.  View space is right-handed with the camera at the origin looking
/// down `-Z`, so a visible surface has positive `linear_depth`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GtaoCamera {
    /// `tan(fov_x / 2)`; horizontal half-extent of the frustum at unit depth.
    pub tan_half_fov_x: f32,
    /// `tan(fov_y / 2)`; vertical half-extent of the frustum at unit depth.
    pub tan_half_fov_y: f32,
}

impl GtaoCamera {
    /// Builds a camera from the two non-trivial diagonal entries of a standard
    /// perspective projection matrix (`proj[0][0]`, `proj[1][1]`).
    pub fn from_projection(proj_m00: f32, proj_m11: f32) -> Self {
        Self {
            tan_half_fov_x: proj_m00.abs().recip(),
            tan_half_fov_y: proj_m11.abs().recip(),
        }
    }

    /// Reconstructs the view-space position of a sample at texture coordinate
    /// `uv` (origin top-left, `y` down) whose linear view depth is
    /// `linear_depth` (positive distance in front of the camera).
    pub fn reconstruct(&self, uv: [f32; 2], linear_depth: f32) -> [f32; 3] {
        let ndc_x = 2.0 * uv[0] - 1.0;
        let ndc_y = 1.0 - 2.0 * uv[1];
        [
            ndc_x * self.tan_half_fov_x * linear_depth,
            ndc_y * self.tan_half_fov_y * linear_depth,
            -linear_depth,
        ]
    }

    /// Converts a world-space search radius into the per-axis texture-space
    /// radius at the given depth.  A world offset `r` perpendicular to the view
    /// direction at depth `z` maps to an NDC offset of `r / (z * tan_half_fov)`,
    /// and texture coordinates span twice the NDC range, hence the `2 * z`.
    pub fn uv_radius(&self, world_radius: f32, linear_depth: f32) -> [f32; 2] {
        let inv = (2.0 * linear_depth.max(1.0e-4)).recip();
        [
            world_radius * inv / self.tan_half_fov_x,
            world_radius * inv / self.tan_half_fov_y,
        ]
    }
}
