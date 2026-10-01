//! Screen-space light-shaft ("god ray") CPU golden references.
//!
//! Deterministic, GPU-free crepuscular-ray models that complement the froxel
//! volumetrics in [`crate::gi::volumetric_gi`] and the analytic sun inscatter
//! in [`crate::gi::fog`].  Where those integrate media in world space, this
//! module reconstructs shafts cheaply in screen space:
//!
//! * [`radial`] — Mitchell 2007 radial-blur occlusion scattering toward the
//!   screen-space light position (decay/density/weight/exposure march).
//! * [`occlusion`] — screen-space occlusion mask build + depth/sky gating that
//!   feeds the radial march.
//! * [`mask`] — half-resolution shaft mask upsample + bilateral depth-aware
//!   recombine against the full-resolution scene.

use alloc::vec::Vec;

use bevy_math::Vec3;

pub mod mask;
pub mod occlusion;
pub mod radial;

pub use mask::{
    bilateral_weights, composite_additive, composite_screen, upsample_bilateral, upsample_buffer,
    HalfResShaft,
};
pub use occlusion::{
    build_emission, build_emission_buffer, luminance, occlusion_mask_binary, sun_disk, DepthRange,
    OcclusionConfig,
};
pub use radial::{
    radial_scatter, radial_scatter_color, sample_mask_bilinear, RadialScatterParams, MAX_SAMPLES,
};

/// Final full-resolution compositing operator for the shaft contribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// Additive screen blend `scene + shaft - scene*shaft` (default).
    Screen,
    /// Plain additive blend `scene + shaft`.
    Additive,
}

impl Default for BlendMode {
    #[inline]
    fn default() -> Self {
        Self::Screen
    }
}

/// Bundles every stage parameter for [`render_light_shaft`].
///
/// The screen-space light position used by the radial march is taken from
/// `occlusion.sun_uv`, so the march, the emission build, and the sun-disk
/// injection all agree on where the light sits on screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightShaftConfig {
    /// Emission-mask build parameters (sky gating + sun-disk injection).
    pub occlusion: OcclusionConfig,
    /// Radial-scatter march parameters.
    pub scatter: RadialScatterParams,
    /// Number of march steps per texel.
    pub sample_count: u32,
    /// Linear-RGB tint applied to the scalar shaft intensity.
    pub shaft_color: Vec3,
    /// Depth sigma for the bilateral upsample.
    pub sigma_z: f32,
    /// Scalar gain applied to the shaft at composite time.
    pub intensity: f32,
    /// How the shaft is combined with the scene.
    pub blend: BlendMode,
}

impl Default for LightShaftConfig {
    #[inline]
    fn default() -> Self {
        Self {
            occlusion: OcclusionConfig::default(),
            scatter: RadialScatterParams::default(),
            sample_count: 64,
            shaft_color: Vec3::new(1.0, 0.9, 0.7),
            sigma_z: 0.05,
            intensity: 1.0,
            blend: BlendMode::Screen,
        }
    }
}

/// Runs the full screen-space god-ray pipeline and returns the composited
/// full-resolution image (row-major `full_width * full_height`).
///
/// The three stages are chained end to end:
///
/// 1. [`occlusion::build_emission_buffer`] turns the half-resolution depth and
///    colour into an emission mask (sky emits, geometry occludes, sun disk
///    injected through sky).
/// 2. [`radial::radial_scatter`] marches that mask toward `occlusion.sun_uv` at
///    half resolution, producing a half-resolution shaft-intensity buffer.
/// 3. [`mask::upsample_buffer`] bilaterally upsamples the shaft to full
///    resolution and it is composited over `scene_color` with the configured
///    [`BlendMode`].
///
/// Mismatched or degenerate buffer sizes yield an empty result.  Every output
/// channel is finite and non-negative.
pub fn render_light_shaft(
    scene_color: &[Vec3],
    full_depth: &[f32],
    full_width: usize,
    full_height: usize,
    half_depth: &[f32],
    half_color: &[Vec3],
    half_width: usize,
    half_height: usize,
    config: &LightShaftConfig,
) -> Vec<Vec3> {
    let full_len = full_width.saturating_mul(full_height);
    let half_len = half_width.saturating_mul(half_height);
    if full_width == 0
        || full_height == 0
        || half_width == 0
        || half_height == 0
        || scene_color.len() < full_len
        || full_depth.len() < full_len
        || half_depth.len() < half_len
        || half_color.len() < half_len
    {
        return Vec::new();
    }

    // Stage 1: emission mask at half resolution.
    let emission =
        build_emission_buffer(half_depth, half_color, half_width, half_height, &config.occlusion);
    if emission.len() < half_len {
        return Vec::new();
    }

    // Stage 2: radial scatter at half resolution.
    let light_uv = config.occlusion.sun_uv;
    let mut shaft = alloc::vec![0.0_f32; half_len];
    let inv_hw = 1.0 / half_width as f32;
    let inv_hh = 1.0 / half_height as f32;
    for y in 0..half_height {
        for x in 0..half_width {
            let i = y * half_width + x;
            let uv = bevy_math::Vec2::new((x as f32 + 0.5) * inv_hw, (y as f32 + 0.5) * inv_hh);
            shaft[i] = radial_scatter(uv, light_uv, config.sample_count, config.scatter, |p| {
                sample_mask_bilinear(&emission, half_width, half_height, p)
            });
        }
    }

    // Stage 3: bilateral upsample + composite.
    let Some(half_res_shaft) = HalfResShaft::new(half_width, half_height, shaft, half_depth[..half_len].to_vec())
    else {
        return Vec::new();
    };
    let full_shaft =
        upsample_buffer(&half_res_shaft, full_depth, full_width, full_height, config.sigma_z);
    if full_shaft.len() < full_len {
        return Vec::new();
    }

    let mut out = alloc::vec![Vec3::ZERO; full_len];
    for i in 0..full_len {
        let shaft_v = full_shaft[i];
        out[i] = match config.blend {
            BlendMode::Screen => {
                composite_screen(scene_color[i], shaft_v, config.shaft_color, config.intensity)
            }
            BlendMode::Additive => {
                composite_additive(scene_color[i], shaft_v, config.shaft_color, config.intensity)
            }
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec2;

    /// Builds a synthetic half-resolution scene: a bright sky with the sun at
    /// the top centre and a horizontal opaque occluder bar across the middle.
    fn synthetic_half(
        hw: usize,
        hh: usize,
        sky_depth: f32,
        geo_depth: f32,
        bar_row: usize,
    ) -> (Vec<f32>, Vec<Vec3>) {
        let mut depth = alloc::vec![sky_depth; hw * hh];
        let mut color = alloc::vec![Vec3::splat(0.3); hw * hh];
        for x in 0..hw {
            let i = bar_row * hw + x;
            depth[i] = geo_depth; // occluder geometry
            color[i] = Vec3::ZERO; // black bar
        }
        (depth, color)
    }

    #[test]
    fn pipeline_rejects_degenerate_sizes() {
        let cfg = LightShaftConfig::default();
        let empty: Vec<Vec3> = Vec::new();
        let out = render_light_shaft(&empty, &[], 0, 0, &[], &empty, 0, 0, &cfg);
        assert!(out.is_empty());
    }

    #[test]
    fn pipeline_produces_finite_full_res_image() {
        let (hw, hh) = (16, 16);
        let (fw, fh) = (32, 32);
        let (half_depth, half_color) = synthetic_half(hw, hh, 1.0, 0.5, hh / 2);
        let scene = alloc::vec![Vec3::splat(0.2); fw * fh];
        let full_depth = alloc::vec![1.0_f32; fw * fh];
        let cfg = LightShaftConfig::default();
        let out = render_light_shaft(
            &scene, &full_depth, fw, fh, &half_depth, &half_color, hw, hh, &cfg,
        );
        assert_eq!(out.len(), fw * fh);
        for c in &out {
            assert!(c.is_finite());
            assert!(c.x >= 0.0 && c.y >= 0.0 && c.z >= 0.0);
        }
    }

    #[test]
    fn screen_blend_never_darkens_the_scene() {
        let (hw, hh) = (16, 16);
        let (fw, fh) = (16, 16);
        let (half_depth, half_color) = synthetic_half(hw, hh, 1.0, 0.5, hh / 2);
        let scene = alloc::vec![Vec3::splat(0.25); fw * fh];
        let full_depth = alloc::vec![1.0_f32; fw * fh];
        let cfg = LightShaftConfig::default();
        let out = render_light_shaft(
            &scene, &full_depth, fw, fh, &half_depth, &half_color, hw, hh, &cfg,
        );
        for c in &out {
            // Additive screen blend can only brighten a positive scene.
            assert!(c.x >= 0.25 - 1e-5, "darkened: {}", c.x);
        }
    }

    #[test]
    fn shaft_is_brighter_near_the_sun_than_in_shadow() {
        // Full-res sampling: compare a pixel high in the sky near the sun column
        // against a pixel directly below the occluder bar.  The near-sun pixel
        // should receive more scattered energy.
        let (hw, hh) = (32, 32);
        let (fw, fh) = (32, 32);
        let bar_row = hh / 2;
        let (half_depth, half_color) = synthetic_half(hw, hh, 1.0, 0.5, bar_row);
        let scene = alloc::vec![Vec3::splat(0.1); fw * fh];
        let full_depth = alloc::vec![1.0_f32; fw * fh];
        let mut cfg = LightShaftConfig::default();
        cfg.occlusion.sun_uv = Vec2::new(0.5, 0.06);
        cfg.blend = BlendMode::Additive;
        let out = render_light_shaft(
            &scene, &full_depth, fw, fh, &half_depth, &half_color, hw, hh, &cfg,
        );
        // Near-sun pixel (top centre).
        let near = out[2 * fw + fw / 2].x;
        // Pixel well below the occluder, same column.
        let shadow = out[(bar_row + 6) * fw + fw / 2].x;
        assert!(near > shadow, "near={near} shadow={shadow}");
    }

    #[test]
    fn mismatched_emission_or_half_buffers_are_safe() {
        // half buffers too small for the stated dimensions.
        let cfg = LightShaftConfig::default();
        let scene = alloc::vec![Vec3::splat(0.2); 16];
        let full_depth = alloc::vec![1.0_f32; 16];
        let half_depth = alloc::vec![1.0_f32; 2];
        let half_color = alloc::vec![Vec3::splat(0.3); 2];
        let out =
            render_light_shaft(&scene, &full_depth, 4, 4, &half_depth, &half_color, 4, 4, &cfg);
        assert!(out.is_empty());
    }
}
