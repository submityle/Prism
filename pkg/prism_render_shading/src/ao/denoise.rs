//! Edge-aware spatial denoiser for the raw GTAO buffer.
//!
//! A single GTAO pass sweeps only a handful of slices per pixel, so its raw
//! ambient-visibility buffer is peppered with slice-aliasing noise — grainy in
//! the open, ropey along creases.  Shipping that directly looks nothing like
//! AAA AO, which always runs the raw estimate through a denoiser.  This module
//! is the first (spatial) half: a joint **bilateral** blur that averages each
//! pixel with its neighbours while a depth *and* normal edge-stopping term
//! keeps the blur from bleeding occlusion across geometry seams (`XeGTAO`'s
//! denoise pass, generalised to an `N x N` kernel).
//!
//! It is the CPU golden the `gtao_denoise.wesl` compute twin reproduces
//! bit-for-bit within tolerance, so both agree on machines with and without a
//! GPU.  Every transcendental routes through [`bevy_math::ops`] for
//! determinism, matching the rest of [`crate::ao`].
//!
//! The temporal half (motion-reprojected history accumulation) layers on top of
//! this in a following slice; keeping the spatial filter standalone means it
//! can run on its own on the very first frame (no history yet) and stays the
//! sole denoiser when temporal reprojection is disabled.

use bevy_math::ops;

use super::GtaoBuffers;
use crate::vecmath::dot;

/// Tunables for the joint bilateral GTAO blur.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GtaoDenoiseConfig {
    /// Kernel half-width in pixels; the blur covers `(2 * radius + 1)^2` taps.
    /// `2` (a 5x5 kernel) is the sweet spot between smoothing and cost.
    pub radius: u32,
    /// Gaussian spatial falloff in pixels: larger smooths harder, blurring more
    /// of the noise but also more real detail.
    pub spatial_sigma: f32,
    /// Depth edge-stopping tolerance as a *fraction* of the centre pixel's view
    /// depth, so the same value holds near and far.  A neighbour whose depth
    /// differs by more than a few multiples of `depth_sigma * centre_depth` is
    /// effectively rejected, stopping the blur at silhouettes.
    pub depth_sigma: f32,
    /// Normal edge-stopping sharpness: the neighbour weight is
    /// `max(dot(n, n_c), 0)^normal_power`, so a higher power rejects tilted
    /// neighbours faster and preserves occlusion contrast along curved edges.
    pub normal_power: f32,
}

impl Default for GtaoDenoiseConfig {
    fn default() -> Self {
        Self {
            radius: 2,
            spatial_sigma: 2.0,
            depth_sigma: 0.05,
            normal_power: 8.0,
        }
    }
}

/// Guards the weight normalisation and the sigma denominators against zero.
const WEIGHT_EPSILON: f32 = 1.0e-6;

/// Denoises a single pixel of the raw AO buffer.
///
/// `raw_ao` is the row-major, `width * height` ambient-visibility buffer the
/// GTAO kernel produced; `buffers` supplies the matching linear depth and
/// view-space normals used for edge stopping.  Background pixels (no valid
/// centre depth) and fully rejected neighbourhoods pass the raw value through
/// unchanged.  Prefer [`denoise_gtao`] for a whole frame.
pub fn denoise_gtao_pixel(
    raw_ao: &[f32],
    buffers: GtaoBuffers<'_>,
    config: GtaoDenoiseConfig,
    x: usize,
    y: usize,
) -> f32 {
    let center = raw_ao[buffers.index(x, y)];
    // Background/sky has no meaningful surface to blur along: keep it as-is.
    let Some(center_depth) = buffers.depth_at(x, y) else {
        return center;
    };
    let center_normal = buffers.view_normals[buffers.index(x, y)];

    let radius = config.radius as i64;
    let spatial_sigma = config.spatial_sigma.max(WEIGHT_EPSILON);
    let two_spatial_sq = 2.0 * spatial_sigma * spatial_sigma;
    // Relative depth tolerance: scale the world tolerance by the centre depth
    // so the edge stop is perspective-correct.
    let depth_scale = (config.depth_sigma.max(WEIGHT_EPSILON)) * center_depth;
    let two_depth_sq = 2.0 * depth_scale * depth_scale;
    let normal_power = config.normal_power.max(0.0);

    let mut weighted_sum = 0.0;
    let mut weight_total = 0.0;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let sx = x as i64 + dx;
            let sy = y as i64 + dy;
            if sx < 0 || sy < 0 || sx >= buffers.width as i64 || sy >= buffers.height as i64 {
                continue;
            }
            let (sx, sy) = (sx as usize, sy as usize);
            // Neighbours on background carry no valid occlusion; skip them so a
            // silhouette against the sky does not darken the foreground.
            let Some(sample_depth) = buffers.depth_at(sx, sy) else {
                continue;
            };

            let dist_sq = (dx * dx + dy * dy) as f32;
            let w_spatial = ops::exp(-dist_sq / two_spatial_sq);

            let depth_delta = sample_depth - center_depth;
            let w_depth = ops::exp(-(depth_delta * depth_delta) / two_depth_sq);

            let sample_normal = buffers.view_normals[buffers.index(sx, sy)];
            let n_dot = dot(center_normal, sample_normal).max(0.0);
            let w_normal = if normal_power > 0.0 {
                ops::powf(n_dot, normal_power)
            } else {
                1.0
            };

            let weight = w_spatial * w_depth * w_normal;
            weighted_sum += weight * raw_ao[buffers.index(sx, sy)];
            weight_total += weight;
        }
    }

    if weight_total > WEIGHT_EPSILON {
        (weighted_sum / weight_total).clamp(0.0, 1.0)
    } else {
        center
    }
}

/// Denoises the whole frame, returning a row-major, `width * height` AO buffer
/// (`1` = unoccluded) — the direct input to the shading resolve's ambient term.
pub fn denoise_gtao(
    raw_ao: &[f32],
    buffers: GtaoBuffers<'_>,
    config: GtaoDenoiseConfig,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(buffers.width * buffers.height);
    for y in 0..buffers.height {
        for x in 0..buffers.width {
            out.push(denoise_gtao_pixel(raw_ao, buffers, config, x, y));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normals(width: usize, height: usize) -> Vec<[f32; 3]> {
        vec![[0.0, 0.0, 1.0]; width * height]
    }

    #[test]
    fn flat_surface_smooths_checkerboard_noise() {
        // A fronto-parallel plane at constant depth with alternating raw AO.
        // The bilateral blur has no edges to stop at, so it should pull every
        // pixel toward the 0.5 mean.
        let (width, height) = (8, 8);
        let depth = vec![10.0_f32; width * height];
        let view_normals = normals(width, height);
        let mut raw = vec![0.0_f32; width * height];
        for y in 0..height {
            for x in 0..width {
                raw[y * width + x] = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let buffers = GtaoBuffers {
            width,
            height,
            linear_depth: &depth,
            view_normals: &view_normals,
        };
        let out = denoise_gtao(&raw, buffers, GtaoDenoiseConfig::default());
        let center = out[4 * width + 4];
        assert!(
            (center - 0.5).abs() < 0.15,
            "checkerboard should average toward 0.5, got {center}"
        );
    }

    #[test]
    fn depth_edge_stops_the_blur() {
        // Left half far (dark AO), right half much nearer (bright AO). The seam
        // pixel on the far side must not inherit the near side's brightness:
        // the relative depth stop rejects the 5-unit jump.
        let (width, height) = (16, 4);
        let mut depth = vec![10.0_f32; width * height];
        let mut raw = vec![0.2_f32; width * height];
        for y in 0..height {
            for x in (width / 2)..width {
                depth[y * width + x] = 5.0;
                raw[y * width + x] = 1.0;
            }
        }
        let view_normals = normals(width, height);
        let buffers = GtaoBuffers {
            width,
            height,
            linear_depth: &depth,
            view_normals: &view_normals,
        };
        let out = denoise_gtao(&raw, buffers, GtaoDenoiseConfig::default());
        let far_seam = out[2 * width + (width / 2 - 1)];
        assert!(
            far_seam < 0.4,
            "depth edge must keep the far seam dark, got {far_seam}"
        );
    }

    #[test]
    fn normal_edge_stops_the_blur() {
        // Same AO split, identical depth, but the two halves face away from each
        // other. The normal stop alone must keep the dark side dark.
        let (width, height) = (16, 4);
        let depth = vec![10.0_f32; width * height];
        let mut raw = vec![0.2_f32; width * height];
        let mut view_normals = vec![[1.0, 0.0, 0.0]; width * height];
        for y in 0..height {
            for x in (width / 2)..width {
                raw[y * width + x] = 1.0;
                view_normals[y * width + x] = [-1.0, 0.0, 0.0];
            }
        }
        let buffers = GtaoBuffers {
            width,
            height,
            linear_depth: &depth,
            view_normals: &view_normals,
        };
        let out = denoise_gtao(&raw, buffers, GtaoDenoiseConfig::default());
        let far_seam = out[2 * width + (width / 2 - 1)];
        assert!(
            far_seam < 0.4,
            "normal edge must keep the far seam dark, got {far_seam}"
        );
    }

    #[test]
    fn background_pixels_pass_through() {
        let (width, height) = (4, 4);
        let mut depth = vec![10.0_f32; width * height];
        depth[2 * width + 2] = 0.0; // sky
        let view_normals = normals(width, height);
        let mut raw = vec![0.5_f32; width * height];
        raw[2 * width + 2] = 0.123;
        let buffers = GtaoBuffers {
            width,
            height,
            linear_depth: &depth,
            view_normals: &view_normals,
        };
        let out = denoise_gtao(&raw, buffers, GtaoDenoiseConfig::default());
        assert_eq!(out[2 * width + 2], 0.123, "background AO must be untouched");
    }

    #[test]
    fn output_stays_in_unit_range() {
        let (width, height) = (12, 12);
        let mut depth = vec![0.0_f32; width * height];
        let mut raw = vec![0.0_f32; width * height];
        for y in 0..height {
            for x in 0..width {
                depth[y * width + x] = 4.0 + ((x * 5 + y * 3) % 7) as f32 * 0.5;
                raw[y * width + x] = ((x * 3 + y) % 5) as f32 * 0.25;
            }
        }
        let view_normals = normals(width, height);
        let buffers = GtaoBuffers {
            width,
            height,
            linear_depth: &depth,
            view_normals: &view_normals,
        };
        for value in denoise_gtao(&raw, buffers, GtaoDenoiseConfig::default()) {
            assert!((0.0..=1.0).contains(&value), "denoised AO out of range: {value}");
        }
    }

    #[test]
    fn zero_radius_is_identity_on_surfaces() {
        let (width, height) = (4, 4);
        let depth = vec![10.0_f32; width * height];
        let view_normals = normals(width, height);
        let raw: Vec<f32> = (0..width * height).map(|i| i as f32 * 0.05).collect();
        let buffers = GtaoBuffers {
            width,
            height,
            linear_depth: &depth,
            view_normals: &view_normals,
        };
        let config = GtaoDenoiseConfig {
            radius: 0,
            ..GtaoDenoiseConfig::default()
        };
        let out = denoise_gtao(&raw, buffers, config);
        for (o, r) in out.iter().zip(raw.iter()) {
            assert!((o - r).abs() < 1.0e-6, "radius 0 must be identity: {o} vs {r}");
        }
    }
}
