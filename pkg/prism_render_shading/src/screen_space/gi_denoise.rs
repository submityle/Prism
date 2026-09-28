//! Edge-aware spatial denoiser for the raw SSGI buffer.
//!
//! A single SSGI gather traces only a handful of cosine-weighted hemisphere
//! rays per pixel, so its raw indirect-diffuse buffer is heavy with Monte-Carlo
//! noise — grainy on open surfaces and splotchy where only a few rays find a
//! valid hit. Shipping that directly looks nothing like AAA GI, which always
//! runs the raw estimate through a denoiser. This module is the spatial half:
//! a joint **bilateral** blur that averages each pixel with its
//! `(2 * radius + 1)^2` neighbours while depth *and* normal edge-stopping terms
//! keep the indirect radiance from bleeding across geometry seams — the same
//! `XeGTAO`-style edge-aware kernel [`crate::ao::denoise_gtao`] applies to AO,
//! generalised here to the four-channel SSGI output (`rgb` = pre-albedo mean
//! indirect radiance, `a` = blend confidence).
//!
//! It is the CPU golden the `ssgi_denoise.wesl` compute twin reproduces
//! bit-for-bit within tolerance, so both agree on machines with and without a
//! GPU. Every transcendental routes through [`bevy_math::ops`] for determinism,
//! matching the rest of [`crate::screen_space`].
//!
//! Keeping the spatial filter standalone means it can run on its own on the
//! very first frame and stays the sole denoiser when no temporal history is
//! available — exactly the reflection subsystem's motion-vector-free stance,
//! where blur (not reprojection) is the noise-suppression path.

use bevy_math::ops;

use crate::vecmath::dot;

/// Tunables for the joint bilateral SSGI blur. Mirrors
/// [`crate::ao::GtaoDenoiseConfig`] so the two denoisers share one mental model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsgiDenoiseConfig {
    /// Kernel half-width in pixels; the blur covers `(2 * radius + 1)^2` taps.
    /// `2` (a 5x5 kernel) balances smoothing against cost; noisy low-ray-count
    /// gathers benefit from `3`.
    pub radius: u32,
    /// Gaussian spatial falloff in pixels: larger smooths harder, blurring more
    /// of the noise but also more real indirect detail.
    pub spatial_sigma: f32,
    /// Depth edge-stopping tolerance as a *fraction* of the centre pixel's view
    /// depth, so the same value holds near and far. A neighbour whose depth
    /// differs by more than a few multiples of `depth_sigma * centre_depth` is
    /// effectively rejected, stopping the blur at silhouettes.
    pub depth_sigma: f32,
    /// Normal edge-stopping sharpness: the neighbour weight is
    /// `max(dot(n, n_c), 0)^normal_power`, so a higher power rejects tilted
    /// neighbours faster and preserves indirect contrast along curved edges.
    pub normal_power: f32,
}

impl Default for SsgiDenoiseConfig {
    fn default() -> Self {
        Self {
            radius: 2,
            spatial_sigma: 2.0,
            depth_sigma: 0.05,
            normal_power: 8.0,
        }
    }
}

/// Read-only view over the depth/normal prepass targets the SSGI denoiser reads
/// for edge stopping.
///
/// `linear_depth` holds positive view-space distance per pixel (`<= 0` or
/// non-finite marks background/sky), and `view_normals` holds unit view-space
/// normals. Both are row-major, `width * height` long, top-left origin — the
/// same layout [`crate::ao::GtaoBuffers`] uses.
#[derive(Clone, Copy, Debug)]
pub struct SsgiDenoiseBuffers<'a> {
    pub width: usize,
    pub height: usize,
    pub linear_depth: &'a [f32],
    pub view_normals: &'a [[f32; 3]],
}

impl SsgiDenoiseBuffers<'_> {
    fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    fn depth_at(&self, x: usize, y: usize) -> Option<f32> {
        let d = self.linear_depth[self.index(x, y)];
        (d.is_finite() && d > 0.0).then_some(d)
    }
}

/// Guards the weight normalisation and the sigma denominators against zero;
/// matches the golden `WEIGHT_EPSILON` in [`crate::ao::denoise`].
const WEIGHT_EPSILON: f32 = 1.0e-6;

/// Denoises a single pixel of the raw SSGI buffer.
///
/// `raw` is the row-major, `width * height` indirect-diffuse buffer the SSGI
/// gather produced (`rgb` = pre-albedo mean indirect radiance, `a` = blend
/// confidence); `buffers` supplies the matching linear depth and view-space
/// normals used for edge stopping. Background pixels (no valid centre depth)
/// and fully rejected neighbourhoods pass the raw value through unchanged.
/// Prefer [`denoise_ssgi`] for a whole frame.
///
/// All four channels share one bilateral weight, so the confidence channel is
/// smoothed exactly like the radiance and the composite sees a spatially
/// coherent blend factor. `rgb` is clamped to non-negative (HDR, no upper
/// bound); `a` is clamped to `[0, 1]`.
pub fn denoise_ssgi_pixel(
    raw: &[[f32; 4]],
    buffers: SsgiDenoiseBuffers<'_>,
    config: SsgiDenoiseConfig,
    x: usize,
    y: usize,
) -> [f32; 4] {
    let center = raw[buffers.index(x, y)];
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
    let depth_scale = config.depth_sigma.max(WEIGHT_EPSILON) * center_depth;
    let two_depth_sq = 2.0 * depth_scale * depth_scale;
    let normal_power = config.normal_power.max(0.0);

    let mut weighted_sum = [0.0_f32; 4];
    let mut weight_total = 0.0_f32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let sx = x as i64 + dx;
            let sy = y as i64 + dy;
            if sx < 0 || sy < 0 || sx >= buffers.width as i64 || sy >= buffers.height as i64 {
                continue;
            }
            let (sx, sy) = (sx as usize, sy as usize);
            // Neighbours on background carry no valid indirect radiance; skip
            // them so a silhouette against the sky does not darken the surface.
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
            let sample = raw[buffers.index(sx, sy)];
            weighted_sum[0] += weight * sample[0];
            weighted_sum[1] += weight * sample[1];
            weighted_sum[2] += weight * sample[2];
            weighted_sum[3] += weight * sample[3];
            weight_total += weight;
        }
    }

    if weight_total > WEIGHT_EPSILON {
        let inv = 1.0 / weight_total;
        [
            (weighted_sum[0] * inv).max(0.0),
            (weighted_sum[1] * inv).max(0.0),
            (weighted_sum[2] * inv).max(0.0),
            (weighted_sum[3] * inv).clamp(0.0, 1.0),
        ]
    } else {
        center
    }
}

/// Denoises the whole frame, returning a row-major, `width * height` SSGI buffer
/// (`rgb` = pre-albedo mean indirect radiance, `a` = blend confidence) — the
/// direct input to the SSGI composite's fold over the IBL/SH ambient.
pub fn denoise_ssgi(
    raw: &[[f32; 4]],
    buffers: SsgiDenoiseBuffers<'_>,
    config: SsgiDenoiseConfig,
) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(buffers.width * buffers.height);
    for y in 0..buffers.height {
        for x in 0..buffers.width {
            out.push(denoise_ssgi_pixel(raw, buffers, config, x, y));
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
        // A fronto-parallel plane at constant depth with alternating raw
        // radiance. The bilateral blur has no edges to stop at, so it should
        // pull every pixel toward the 0.5 mean on each channel.
        let (width, height) = (8, 8);
        let depth = vec![10.0_f32; width * height];
        let view_normals = normals(width, height);
        let mut raw = vec![[0.0_f32; 4]; width * height];
        for y in 0..height {
            for x in 0..width {
                let v = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                raw[y * width + x] = [v, v, v, v];
            }
        }
        let buffers = SsgiDenoiseBuffers {
            width,
            height,
            linear_depth: &depth,
            view_normals: &view_normals,
        };
        let out = denoise_ssgi(&raw, buffers, SsgiDenoiseConfig::default());
        let center = out[4 * width + 4];
        for (c, &value) in center.iter().enumerate().take(4) {
            assert!(
                (value - 0.5).abs() < 0.15,
                "checkerboard channel {c} should average toward 0.5, got {value}"
            );
        }
    }

    #[test]
    fn depth_edge_stops_the_blur() {
        // Left half far (dark GI), right half much nearer (bright GI). The seam
        // pixel on the far side must not inherit the near side's brightness.
        let (width, height) = (16, 4);
        let mut depth = vec![10.0_f32; width * height];
        let mut raw = vec![[0.2_f32, 0.2, 0.2, 1.0]; width * height];
        for y in 0..height {
            for x in (width / 2)..width {
                depth[y * width + x] = 5.0;
                raw[y * width + x] = [1.0, 1.0, 1.0, 1.0];
            }
        }
        let view_normals = normals(width, height);
        let buffers = SsgiDenoiseBuffers {
            width,
            height,
            linear_depth: &depth,
            view_normals: &view_normals,
        };
        let out = denoise_ssgi(&raw, buffers, SsgiDenoiseConfig::default());
        let far_seam = out[2 * width + (width / 2 - 1)][0];
        assert!(
            far_seam < 0.4,
            "depth edge must keep the far seam dark, got {far_seam}"
        );
    }

    #[test]
    fn normal_edge_stops_the_blur() {
        // Same split, identical depth, but the two halves face away from each
        // other. The normal stop alone must keep the dark side dark.
        let (width, height) = (16, 4);
        let depth = vec![10.0_f32; width * height];
        let mut raw = vec![[0.2_f32, 0.2, 0.2, 1.0]; width * height];
        let mut view_normals = vec![[1.0, 0.0, 0.0]; width * height];
        for y in 0..height {
            for x in (width / 2)..width {
                raw[y * width + x] = [1.0, 1.0, 1.0, 1.0];
                view_normals[y * width + x] = [-1.0, 0.0, 0.0];
            }
        }
        let buffers = SsgiDenoiseBuffers {
            width,
            height,
            linear_depth: &depth,
            view_normals: &view_normals,
        };
        let out = denoise_ssgi(&raw, buffers, SsgiDenoiseConfig::default());
        let far_seam = out[2 * width + (width / 2 - 1)][0];
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
        let mut raw = vec![[0.5_f32, 0.5, 0.5, 1.0]; width * height];
        raw[2 * width + 2] = [0.123, 0.456, 0.789, 0.321];
        let buffers = SsgiDenoiseBuffers {
            width,
            height,
            linear_depth: &depth,
            view_normals: &view_normals,
        };
        let out = denoise_ssgi(&raw, buffers, SsgiDenoiseConfig::default());
        assert_eq!(
            out[2 * width + 2],
            [0.123, 0.456, 0.789, 0.321],
            "background GI must be untouched"
        );
    }

    #[test]
    fn confidence_stays_in_unit_range_radiance_non_negative() {
        let (width, height) = (12, 12);
        let mut depth = vec![0.0_f32; width * height];
        let mut raw = vec![[0.0_f32; 4]; width * height];
        for y in 0..height {
            for x in 0..width {
                depth[y * width + x] = 4.0 + ((x * 5 + y * 3) % 7) as f32 * 0.5;
                let v = ((x * 3 + y) % 5) as f32 * 0.25;
                raw[y * width + x] = [v, v * 2.0, v * 3.0, (v).min(1.0)];
            }
        }
        let view_normals = normals(width, height);
        let buffers = SsgiDenoiseBuffers {
            width,
            height,
            linear_depth: &depth,
            view_normals: &view_normals,
        };
        for px in denoise_ssgi(&raw, buffers, SsgiDenoiseConfig::default()) {
            assert!(px[0] >= 0.0 && px[1] >= 0.0 && px[2] >= 0.0, "radiance negative: {px:?}");
            assert!((0.0..=1.0).contains(&px[3]), "confidence out of range: {px:?}");
        }
    }

    #[test]
    fn zero_radius_is_identity_on_surfaces() {
        let (width, height) = (4, 4);
        let depth = vec![10.0_f32; width * height];
        let view_normals = normals(width, height);
        let raw: Vec<[f32; 4]> = (0..width * height)
            .map(|i| {
                let v = i as f32 * 0.05;
                [v, v, v, (v).min(1.0)]
            })
            .collect();
        let buffers = SsgiDenoiseBuffers {
            width,
            height,
            linear_depth: &depth,
            view_normals: &view_normals,
        };
        let config = SsgiDenoiseConfig {
            radius: 0,
            ..SsgiDenoiseConfig::default()
        };
        let out = denoise_ssgi(&raw, buffers, config);
        for (o, r) in out.iter().zip(raw.iter()) {
            for c in 0..4 {
                assert!((o[c] - r[c]).abs() < 1.0e-6, "radius 0 must be identity: {o:?} vs {r:?}");
            }
        }
    }
}
