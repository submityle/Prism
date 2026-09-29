//! Screen-adaptive shadow texel-size policy.
//!
//! A clipmap only helps if each receiver is shadowed at a resolution that
//! matches how large it appears on screen: a near receiver filling many pixels
//! wants fine shadow texels, while a distant one covering a few pixels wastes
//! memory and bandwidth on the same density. This module turns a receiver's
//! camera distance into the world-space shadow texel size it warrants, which
//! [`super::clipmap::ClipmapConfig::select_level`] then maps to a clip level.
//!
//! The model is a pinhole camera: at distance `d` a single screen pixel spans
//! `d / focal_length_pixels` world units, so matching one shadow texel per
//! screen pixel gives that same world size. `texels_per_pixel` oversamples
//! (values above one shrink the texel for crisper shadow edges at a memory
//! cost). The policy is GPU-independent and deterministic; it uses only a divide
//! and, in the caster helper, a single square root for distance.

use super::frame::ShadowCaster;
use crate::gpu_scene::SceneBounds;

/// Maps receiver distance to the world-space shadow texel size it needs.
///
/// `focal_length_pixels` is the camera's pinhole focal length in pixels (larger
/// is more zoomed-in, finer texels); `texels_per_pixel` oversamples the shadow
/// relative to screen density (>= 1 for crisper edges). Both are validated at
/// use so a misconfigured policy degrades gracefully instead of dividing by
/// zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowQuality {
    /// Camera pinhole focal length in pixels.
    pub focal_length_pixels: f32,
    /// Shadow texels desired per screen pixel; higher is crisper.
    pub texels_per_pixel: f32,
}

impl ShadowQuality {
    /// World-space shadow texel size a receiver at `distance` warrants.
    ///
    /// Returns the pinhole per-pixel world size at `distance` divided by the
    /// oversample factor. A non-positive distance yields the finest texel this
    /// policy can express (`0.0`), and non-positive tunables fall back to sane
    /// unit values so the result stays finite and non-negative.
    #[must_use]
    pub fn required_texel_size(&self, distance: f32) -> f32 {
        if distance <= 0.0 {
            return 0.0;
        }
        let focal = if self.focal_length_pixels > 0.0 {
            self.focal_length_pixels
        } else {
            1.0
        };
        let oversample = if self.texels_per_pixel > 0.0 {
            self.texels_per_pixel
        } else {
            1.0
        };
        (distance / focal) / oversample
    }
}

/// Builds a [`ShadowCaster`] for a receiver, sizing its texel from the policy.
///
/// The caster distance is the world-space distance from `camera_world` to the
/// bounds centre; the returned caster carries the policy-derived
/// `required_texel_size` so the frame planner selects a distance-appropriate
/// clip level automatically.
#[must_use]
pub fn caster_for_receiver(
    bounds: SceneBounds,
    priority: f32,
    camera_world: [f32; 3],
    quality: &ShadowQuality,
) -> ShadowCaster {
    let dx = bounds.center[0] - camera_world[0];
    let dy = bounds.center[1] - camera_world[1];
    let dz = bounds.center[2] - camera_world[2];
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
    ShadowCaster {
        bounds,
        priority,
        required_texel_size: quality.required_texel_size(distance),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quality() -> ShadowQuality {
        ShadowQuality {
            focal_length_pixels: 1000.0,
            texels_per_pixel: 1.0,
        }
    }

    fn approx(a: f32, b: f32) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= 1.0e-4
    }

    #[test]
    fn texel_grows_linearly_with_distance() {
        let q = quality();
        // At focal 1000, distance 1000 => 1 world unit per pixel.
        assert!(approx(q.required_texel_size(1000.0), 1.0));
        // Twice as far => twice the texel size.
        assert!(approx(q.required_texel_size(2000.0), 2.0));
    }

    #[test]
    fn oversampling_shrinks_the_texel() {
        let q = ShadowQuality {
            focal_length_pixels: 1000.0,
            texels_per_pixel: 4.0,
        };
        // Four shadow texels per screen pixel => quarter-size texel.
        assert!(approx(q.required_texel_size(1000.0), 0.25));
    }

    #[test]
    fn nonpositive_inputs_stay_finite() {
        let q = quality();
        assert!(approx(q.required_texel_size(0.0), 0.0));
        assert!(approx(q.required_texel_size(-5.0), 0.0));
        let broken = ShadowQuality {
            focal_length_pixels: 0.0,
            texels_per_pixel: 0.0,
        };
        // Falls back to focal=1, oversample=1 => texel == distance.
        assert!(approx(broken.required_texel_size(3.0), 3.0));
    }

    #[test]
    fn caster_helper_sizes_texel_from_distance() {
        let q = quality();
        let bounds = SceneBounds {
            center: [0.0, 0.0, 2000.0],
            radius: 1.0,
            half_extents: [1.0, 1.0, 1.0],
            _padding: 0.0,
        };
        let caster = caster_for_receiver(bounds, 3.0, [0.0, 0.0, 0.0], &q);
        // Distance 2000 at focal 1000 => texel 2.0.
        assert!(approx(caster.required_texel_size, 2.0));
        assert!(approx(caster.priority, 3.0));
    }
}
