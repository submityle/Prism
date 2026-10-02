//! Ground-Truth Ambient Occlusion (GTAO) golden reference.
//!
//! GTAO (Jimenez et al., SIGGRAPH 2016) is a screen-space estimate of how much
//! of the hemisphere above a surface is blocked by nearby geometry.  For each
//! shaded pixel it sweeps a set of *slices* through the view vector, searches
//! each slice for the highest occluding *horizon* on both sides, and integrates
//! the cosine-weighted visible arc analytically (see [`integral`]).  Averaging
//! the slices yields ambient **visibility** in `[0, 1]` (`1` unoccluded); the
//! renderer multiplies indirect/ambient light by this factor.
//!
//! This module is backend neutral and deterministic: it consumes the linear
//! view-depth prepass plus the view-space normal buffer and reproduces exactly
//! the arithmetic of the `gtao.wesl` compute twin, so it is the CPU golden the
//! shader is validated against on machines without a GPU.

mod denoise;
mod integral;
mod reconstruct;
mod temporal;

use bevy_math::ops;

use crate::vecmath::{dot, mul_scalar, normalize_or, sub};
use integral::{combine_horizon, distance_weight, slice_visibility};

pub use denoise::{denoise_gtao, denoise_gtao_pixel, GtaoDenoiseConfig};
pub use reconstruct::GtaoCamera;
pub use temporal::{
    accumulate_ao, accumulate_moment, clip_history, gtao_adaptive_history_weight,
    reproject_prev_uv_gtao, variance_clip_band, GtaoClipResult, GtaoTemporalParams,
};

/// Read-only view over the depth/normal prepass GTAO samples.
///
/// `linear_depth` holds positive view-space distance per pixel (`<= 0` or
/// non-finite marks background/sky), and `view_normals` holds unit view-space
/// normals.  Both are row-major, `width * height` long, top-left origin.
#[derive(Clone, Copy, Debug)]
pub struct GtaoBuffers<'a> {
    pub width: usize,
    pub height: usize,
    pub linear_depth: &'a [f32],
    pub view_normals: &'a [[f32; 3]],
}

impl GtaoBuffers<'_> {
    fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    fn depth_at(&self, x: usize, y: usize) -> Option<f32> {
        let d = self.linear_depth[self.index(x, y)];
        (d.is_finite() && d > 0.0).then_some(d)
    }

    /// Nearest-neighbour depth fetch for a `uv` inside the image, `None` when
    /// the coordinate leaves the frame or lands on background.
    fn sample_depth(&self, uv: [f32; 2]) -> Option<f32> {
        if !(0.0..=1.0).contains(&uv[0]) || !(0.0..=1.0).contains(&uv[1]) {
            return None;
        }
        let x = ((uv[0] * self.width as f32 - 0.5).round() as i64).clamp(0, self.width as i64 - 1)
            as usize;
        let y = ((uv[1] * self.height as f32 - 0.5).round() as i64).clamp(0, self.height as i64 - 1)
            as usize;
        self.depth_at(x, y)
    }

    fn pixel_uv(&self, x: usize, y: usize) -> [f32; 2] {
        [
            (x as f32 + 0.5) / self.width as f32,
            (y as f32 + 0.5) / self.height as f32,
        ]
    }
}

/// Tunables for the horizon search and final remap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GtaoConfig {
    /// Directions swept through the view vector; higher is smoother/slower.
    pub slice_count: u32,
    /// Marched samples per side per slice; higher catches thinner horizons.
    pub steps_per_slice: u32,
    /// World-space search radius; occluders past it are ignored.
    pub world_radius: f32,
    /// Fraction of the radius (`[0, 1]`) after which distance attenuation
    /// begins; `0.8` keeps near occluders solid while easing the radius edge.
    pub falloff: f32,
    /// Occlusion contrast: final visibility is raised to this power (`> 1`
    /// darkens creases, `< 1` softens).
    pub power: f32,
}

impl Default for GtaoConfig {
    fn default() -> Self {
        Self {
            slice_count: 4,
            steps_per_slice: 8,
            world_radius: 1.0,
            falloff: 0.6,
            power: 1.0,
        }
    }
}

/// Small offset (texture space) used to derive the in-view slice axis.
const AXIS_EPSILON: f32 = 1.0e-3;
/// Guards against dividing by a near-zero horizon distance.
const DISTANCE_EPSILON: f32 = 1.0e-4;

/// Computes ambient visibility (`1` = unoccluded) for a single pixel.
///
/// Background pixels and degenerate normals return `1`.  Prefer
/// [`compute_gtao`] for a whole frame.
pub fn gtao_pixel(
    buffers: GtaoBuffers<'_>,
    camera: GtaoCamera,
    config: GtaoConfig,
    x: usize,
    y: usize,
) -> f32 {
    let Some(depth) = buffers.depth_at(x, y) else {
        return 1.0;
    };
    let slice_count = config.slice_count.max(1);
    let steps = config.steps_per_slice.max(1);
    let radius_world = config.world_radius.max(DISTANCE_EPSILON);
    let falloff_start = radius_world * config.falloff.clamp(0.0, 1.0);

    let center_uv = buffers.pixel_uv(x, y);
    let position = camera.reconstruct(center_uv, depth);
    // View vector points from the surface back toward the camera at the origin.
    let view = normalize_or(mul_scalar(position, -1.0), [0.0, 0.0, 1.0]);
    let normal = buffers.view_normals[buffers.index(x, y)];

    let radius_uv = camera.uv_radius(radius_world, depth);

    let search = HorizonSearch {
        buffers: &buffers,
        camera,
        position,
        view,
        center_uv,
        radius_uv,
        radius_world,
        falloff_start,
        steps,
    };

    let mut visibility = 0.0;
    for slice in 0..slice_count {
        let phi = core::f32::consts::PI * slice as f32 / slice_count as f32;
        let omega = [ops::cos(phi), ops::sin(phi)];

        // The slice plane is spanned by the view vector and the screen tangent
        // along `omega`; recover that tangent by reconstructing a nudged point
        // at the same depth and stripping its view-parallel component.
        let nudged_uv = [
            center_uv[0] + omega[0] * AXIS_EPSILON,
            center_uv[1] + omega[1] * AXIS_EPSILON,
        ];
        let nudged = camera.reconstruct(nudged_uv, depth);
        let tangent = sub(nudged, position);
        let axis = normalize_or(
            sub(tangent, mul_scalar(view, dot(tangent, view))),
            [1.0, 0.0, 0.0],
        );

        // Project the normal into the slice plane: its view/axis components give
        // the in-plane angle `gamma` and the slice weight `proj_len`.
        let n_view = dot(normal, view);
        let n_axis = dot(normal, axis);
        let proj_len = (n_view * n_view + n_axis * n_axis).sqrt();
        if proj_len < DISTANCE_EPSILON {
            continue;
        }
        let gamma = ops::atan2(n_axis, n_view);

        // Horizons start flush with the tangent plane (cos = 0, no occluder).
        let cos_h2 = search.horizon(omega, 1.0);
        let cos_h1 = search.horizon(omega, -1.0);

        visibility += slice_visibility(cos_h1, cos_h2, gamma, proj_len);
    }

    let visibility = (visibility / slice_count as f32).clamp(0.0, 1.0);
    ops::powf(visibility, config.power.max(0.0))
}

/// Per-pixel invariants shared by both horizon sweeps of every slice.
struct HorizonSearch<'a, 'b> {
    buffers: &'a GtaoBuffers<'b>,
    camera: GtaoCamera,
    position: [f32; 3],
    view: [f32; 3],
    center_uv: [f32; 2],
    radius_uv: [f32; 2],
    radius_world: f32,
    falloff_start: f32,
    steps: u32,
}

impl HorizonSearch<'_, '_> {
    /// Marches one side of a slice, returning the highest horizon cosine found.
    ///
    /// `side` is `+1` for the positive axis (paired with `cos_h2`) or `-1` for
    /// the negative axis (`cos_h1`).
    fn horizon(&self, omega: [f32; 2], side: f32) -> f32 {
        let mut cos_horizon = 0.0;
        for step in 0..self.steps {
            let s = (step as f32 + 1.0) / self.steps as f32;
            let sample_uv = [
                self.center_uv[0] + side * omega[0] * self.radius_uv[0] * s,
                self.center_uv[1] + side * omega[1] * self.radius_uv[1] * s,
            ];
            let Some(sample_depth) = self.buffers.sample_depth(sample_uv) else {
                continue;
            };
            let sample_pos = self.camera.reconstruct(sample_uv, sample_depth);
            let horizon = sub(sample_pos, self.position);
            let dist_sq = dot(horizon, horizon);
            if dist_sq < DISTANCE_EPSILON * DISTANCE_EPSILON {
                continue;
            }
            let dist = dist_sq.sqrt();
            // Only occluders above the tangent plane (positive cosine) block light.
            let cos_angle = dot(horizon, self.view) / dist;
            if cos_angle <= 0.0 {
                continue;
            }
            let weighted = cos_angle * distance_weight(dist, self.falloff_start, self.radius_world);
            cos_horizon = combine_horizon(cos_horizon, weighted);
        }
        cos_horizon
    }
}

/// Computes ambient visibility for the whole frame, row-major and
/// `width * height` long (`1` = unoccluded).
pub fn compute_gtao(buffers: GtaoBuffers<'_>, camera: GtaoCamera, config: GtaoConfig) -> Vec<f32> {
    let mut out = Vec::with_capacity(buffers.width * buffers.height);
    for y in 0..buffers.height {
        for x in 0..buffers.width {
            out.push(gtao_pixel(buffers, camera, config, x, y));
        }
    }
    out
}

#[cfg(test)]
mod tests;
