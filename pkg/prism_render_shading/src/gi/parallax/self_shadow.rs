//! Height-field self-shadowing — soft parallax shadows (CPU golden reference).
//!
//! Once parallax mapping has displaced a texel to its true position on the
//! relief, that position can still be shadowed by *other* parts of the same
//! height field standing between it and the light.  This module marches a
//! second ray — this time from the resolved surface point toward the light —
//! and reports how much of the light survives, following the soft
//! self-shadowing of Tatarchuk's *Dynamic Parallax Occlusion Mapping*
//! (ATI, 2006): instead of a hard in/out test, each would-be occluder along the
//! light ray contributes a *partial* shadow scaled by how far it rises above
//! the light ray and by how close it sits to the shaded point, and the strongest
//! such contribution drives a soft penumbra.
//!
//! The height-field and tangent-space conventions match
//! [`crate::gi::parallax`]: `height_at(uv) -> f32` returns the surface height in
//! `[0, 1]` (`1` on the polygon plane / ridge, `0` at the deepest valley), and
//! the tangent-space light direction `light_ts` points from the surface toward
//! the light with `+z` along the geometric normal.  The march climbs from the
//! shaded point's height toward the ceiling (`height = 1`); at a height rise
//! `dh` it steps the coordinate by `(light_ts.xy / light_ts.z) * height_scale`
//! per unit height, so the ray tracks the light through texture space.
//!
//! [`crate::gi::parallax`]: crate::gi::parallax
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; `sqrt` uses the inherent
//!   method.  No `f32::exp()`-style free functions.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * Height is `[0, 1]` with `1` at the polygon plane / ridge.
//! * The returned shadow factor is in `[0, 1]`: `1` fully lit, `0` fully
//!   shadowed.  Closer occluders weigh more than distant ones, giving a soft
//!   penumbra that strengthens at grazing light angles.
//! * Defensive clamping everywhere: non-finite inputs, a light at or below the
//!   surface horizon, and a point already on the ceiling all resolve to a
//!   finite factor so no `NaN`/`inf` ever escapes.

use bevy_math::{Vec2, Vec3};

/// Smallest `light_ts.z` treated as a light above the surface horizon.
///
/// At or below this the light grazes the tangent plane (or sits behind it), the
/// `light_ts.xy / light_ts.z` climb rate explodes, and the point cannot be
/// directly lit, so the reference reports full shadow.
const MIN_LIGHT_Z: f32 = 1.0e-4;

/// Lower bound on the march step count.
const MIN_STEPS: u32 = 1;

/// Upper bound on the march step count, bounding work.
const MAX_STEPS: u32 = 1024;

/// Height below which the shaded point is already effectively on the ceiling,
/// leaving no room for an occluder to rise above the light ray.
const CEILING_EPS: f32 = 1.0e-6;

/// Tuning for a [`self_shadow`] march.
///
/// Field layout mirrors the packed `u32`/`f32` constants the GPU twin consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfShadowConfig {
    /// Number of samples taken climbing from the shaded point to the ceiling.
    pub steps: u32,
    /// Scales height-field units into tangent-plane texture units (must match
    /// the scale used for the parallax displacement that produced the point).
    pub height_scale: f32,
    /// Penumbra hardness: larger values darken partial occluders faster, so the
    /// shadow approaches a hard edge; smaller values soften it.
    pub strength: f32,
}

impl Default for SelfShadowConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl SelfShadowConfig {
    /// A sensible default: 16 steps, unit height, moderate hardness.
    pub const DEFAULT: Self = Self {
        steps: 16,
        height_scale: 1.0,
        strength: 4.0,
    };

    /// Builds a configuration, clamping every field to a finite, sane range.
    pub fn new(steps: u32, height_scale: f32, strength: f32) -> Self {
        let steps = steps.clamp(MIN_STEPS, MAX_STEPS);
        let height_scale = if height_scale.is_finite() {
            height_scale.max(0.0)
        } else {
            0.0
        };
        let strength = if strength.is_finite() {
            strength.max(0.0)
        } else {
            0.0
        };
        Self {
            steps,
            height_scale,
            strength,
        }
    }
}

/// Marches from a resolved surface point toward the light and returns the soft
/// self-shadow factor in `[0, 1]` (`1` fully lit, `0` fully shadowed).
///
/// `hit_uv` is the coordinate of the point being shaded (typically the output
/// of a parallax march), `light_ts` the tangent-space surface-to-light
/// direction (`+z` along the normal, need not be normalised), and `height_at`
/// the height-field closure (values outside `[0, 1]` are clamped).
pub fn self_shadow<F>(hit_uv: Vec2, light_ts: Vec3, cfg: SelfShadowConfig, height_at: F) -> f32
where
    F: Fn(Vec2) -> f32,
{
    let cfg = SelfShadowConfig::new(cfg.steps, cfg.height_scale, cfg.strength);
    let hit_uv = sanitize_vec2(hit_uv);

    let light = sanitize_vec3(light_ts);
    let len = light.length();
    if len <= MIN_LIGHT_Z {
        return 0.0;
    }
    let light = light / len;
    // A light at or below the surface horizon cannot reach the point.
    if light.z <= MIN_LIGHT_Z {
        return 0.0;
    }

    let h0 = height_at_clamped(&height_at, hit_uv);
    let climb = 1.0 - h0;
    if climb <= CEILING_EPS {
        // Already on the ceiling: nothing can rise above the light ray.
        return 1.0;
    }

    let n = cfg.steps;
    let inv_n = 1.0 / n as f32;
    let step_height = climb * inv_n;

    // Texture-space move per unit of height climbed along the light ray.
    let dir_uv = Vec2::new(light.x, light.y) / light.z * cfg.height_scale;
    let step_uv = dir_uv * step_height;

    // Soft accumulation: track the strongest weighted occluder along the ray.
    let mut max_blocker = 0.0f32;
    for i in 1..=n {
        let fi = i as f32;
        let sample_uv = hit_uv + step_uv * fi;
        let ray_height = h0 + step_height * fi;
        let surface = height_at_clamped(&height_at, sample_uv);

        let rise = surface - ray_height;
        if rise > 0.0 {
            // Closer samples (small `i`) weigh more; weight fades to 0 at the
            // ceiling so a far-off ridge barely darkens the point.
            let weight = (1.0 - (fi - 0.5) * inv_n).clamp(0.0, 1.0);
            let blocker = rise * weight;
            if blocker > max_blocker {
                max_blocker = blocker;
            }
        }
    }

    let shadow = (max_blocker * cfg.strength).clamp(0.0, 1.0);
    (1.0 - shadow).clamp(0.0, 1.0)
}

/// Height at `uv` clamped to `[0, 1]`, with non-finite samples treated as the
/// ceiling (`height = 1`) so a broken closure never injects a phantom occluder.
fn height_at_clamped<F>(height_at: &F, uv: Vec2) -> f32
where
    F: Fn(Vec2) -> f32,
{
    let h = height_at(uv);
    if h.is_finite() {
        h.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Replaces any non-finite component of a `Vec2` with `0`.
fn sanitize_vec2(v: Vec2) -> Vec2 {
    Vec2::new(finite_or_zero(v.x), finite_or_zero(v.y))
}

/// Replaces any non-finite component of a `Vec3` with `0`.
fn sanitize_vec3(v: Vec3) -> Vec3 {
    Vec3::new(finite_or_zero(v.x), finite_or_zero(v.y), finite_or_zero(v.z))
}

/// Returns `x` when finite, otherwise `0`.
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;

    fn flat(h: f32) -> impl Fn(Vec2) -> f32 {
        move |_uv: Vec2| h
    }

    /// A tall wall: everything with `uv.x >= edge` is at the ceiling (height 1),
    /// everything before it sits at `floor`.
    fn wall(edge: f32, floor: f32) -> impl Fn(Vec2) -> f32 {
        move |uv: Vec2| if uv.x >= edge { 1.0 } else { floor }
    }

    fn light_from_angle(deg: f32) -> Vec3 {
        let r = deg * core::f32::consts::PI / 180.0;
        // Light leans toward +x (toward the wall) above the surface.
        Vec3::new(ops::sin(r), 0.0, ops::cos(r)).normalize()
    }

    #[test]
    fn flat_surface_is_fully_lit() {
        // Nothing rises above the climbing ray on a flat field.
        let s = self_shadow(Vec2::new(0.3, 0.3), light_from_angle(40.0), SelfShadowConfig::DEFAULT, flat(0.5));
        assert!((s - 1.0).abs() < 1e-6, "expected full light, got {}", s);
    }

    #[test]
    fn point_on_ceiling_is_fully_lit() {
        let s = self_shadow(Vec2::new(0.5, 0.5), light_from_angle(30.0), SelfShadowConfig::DEFAULT, flat(1.0));
        assert!((s - 1.0).abs() < 1e-6, "got {}", s);
    }

    #[test]
    fn tall_wall_casts_a_strong_shadow() {
        // A deep point just before a ceiling-high wall, light grazing toward it.
        let s = self_shadow(
            Vec2::new(0.49, 0.5),
            light_from_angle(60.0),
            SelfShadowConfig::new(32, 1.0, 8.0),
            wall(0.5, 0.0),
        );
        assert!(s < 0.1, "expected near-black shadow, got {}", s);
    }

    #[test]
    fn grazing_light_shadows_more_than_steep_light() {
        // Same wall and point; shallower light (bigger angle from normal)
        // should produce a smaller (darker) factor, monotonically.
        // Low hardness keeps both ends off the fully-shadowed floor so the
        // grazing-vs-steep ordering is observable rather than saturated to 0.
        let cfg = SelfShadowConfig::new(64, 1.0, 1.0);
        let point = Vec2::new(0.4, 0.5);
        let f = |deg: f32| self_shadow(point, light_from_angle(deg), cfg, wall(0.5, 0.0));
        let steep = f(20.0);
        let mid = f(45.0);
        let grazing = f(70.0);
        assert!(grazing <= mid, "grazing {} should be <= mid {}", grazing, mid);
        assert!(mid <= steep, "mid {} should be <= steep {}", mid, steep);
        assert!(grazing < steep, "expected a real difference: {} vs {}", grazing, steep);
    }

    #[test]
    fn light_below_horizon_is_fully_shadowed() {
        let below = Vec3::new(1.0, 0.0, -0.2);
        let s = self_shadow(Vec2::new(0.4, 0.5), below, SelfShadowConfig::DEFAULT, wall(0.5, 0.0));
        assert_eq!(s, 0.0);
        let flat_light = Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(self_shadow(Vec2::new(0.4, 0.5), flat_light, SelfShadowConfig::DEFAULT, wall(0.5, 0.0)), 0.0);
    }

    #[test]
    fn factor_is_always_in_unit_range() {
        for deg in [5.0f32, 25.0, 45.0, 65.0, 85.0] {
            for floor in [0.0f32, 0.3, 0.6] {
                let s = self_shadow(
                    Vec2::new(0.45, 0.5),
                    light_from_angle(deg),
                    SelfShadowConfig::DEFAULT,
                    wall(0.5, floor),
                );
                assert!(s.is_finite() && (0.0..=1.0).contains(&s), "deg {} floor {} -> {}", deg, floor, s);
            }
        }
    }

    #[test]
    fn non_finite_inputs_never_produce_nan() {
        let s = self_shadow(
            Vec2::new(f32::NAN, 0.5),
            Vec3::new(0.0, f32::INFINITY, 1.0),
            SelfShadowConfig::new(0, f32::NAN, f32::INFINITY),
            wall(0.5, 0.0),
        );
        assert!(s.is_finite() && (0.0..=1.0).contains(&s));
    }

    #[test]
    fn results_are_deterministic() {
        let l = light_from_angle(50.0);
        let a = self_shadow(Vec2::new(0.42, 0.5), l, SelfShadowConfig::DEFAULT, wall(0.5, 0.1));
        let b = self_shadow(Vec2::new(0.42, 0.5), l, SelfShadowConfig::DEFAULT, wall(0.5, 0.1));
        assert_eq!(a, b);
    }
}
