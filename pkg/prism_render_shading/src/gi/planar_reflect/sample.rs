//! Reflection-buffer sampling: UV projection, blur LOD, and edge fade — CPU golden.
//!
//! Once the mirror pass has rendered the reflected scene into an off-screen
//! colour buffer (from the mirrored camera of [`super::mirror`], optionally with
//! the oblique near-clip of [`super::clip`]), the reflective surface shader must
//! *read back* from that buffer.  This module is the backend-neutral reference
//! for that read:
//!
//! * [`project_to_uv`] maps a world-space point through the main
//!   view-projection to clip → NDC → screen UV, clamped to `[0, 1]`.
//! * [`roughness_to_lod`] converts a surface roughness and reflection travel
//!   distance into a mip (blur) level — rough surfaces and distant hits read
//!   from blurrier mips, matching a pre-filtered reflection pyramid.
//! * [`edge_fade`] computes a `[0, 1]` weight that is `1` in the screen interior
//!   and smoothly falls to `0` toward the frame border, hiding the hard seam
//!   where a planar reflection runs out of on-screen information.
//! * [`sample_reflection`] combines the three into a single
//!   [`ReflectionSample`] (UV, LOD, weight) for a reflected point.
//!
//! # Conventions
//! * `no_std`: math via `bevy_math`; transcendental-free (uses `sqrt`, `clamp`,
//!   and polynomial `smoothstep` only).  No `alloc` is required.
//! * UV lives in `[0, 1]^2` with the origin at the **top-left** of the frame:
//!   `u = ndc.x * 0.5 + 0.5` and `v = 0.5 - ndc.y * 0.5` (NDC `+Y` is up, UV `+V`
//!   is down).  The frame centre `ndc = 0` maps to `uv = (0.5, 0.5)`.
//! * Clip space uses the `[0, 1]` depth convention of
//!   a right-handed `wgpu`-style perspective; the perspective divide is guarded so a
//!   point at or behind the eye (`w ≤ eps`) is reported as off-screen rather
//!   than producing a wrapped UV.
//! * `roughness` is the perceptual roughness in `[0, 1]`; `0` is a perfect
//!   mirror (LOD 0) and `1` is fully diffuse (max LOD).
//! * Every function is deterministic and pure, clamps its inputs, and never
//!   returns `NaN`/`inf`.

use bevy_math::{Mat4, Vec2, Vec3};

/// Smallest clip-space `w` whose reciprocal is trusted; a point with `w` at or
/// below this is treated as behind the eye (off-screen).
const MIN_W: f32 = 1.0e-6;

/// Parameters controlling how a reflection fades and blurs across the frame.
///
/// Collected into one struct so [`sample_reflection`] has a stable signature and
/// callers can tweak the look without juggling positional arguments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectionParams {
    /// Half-width (in UV units) of the smooth border fade band; `0.1` fades the
    /// outer 10% of the frame on each side.  Clamped to `[0, 0.5]`.
    pub edge_fade_width: f32,
    /// Maximum mip level selectable by [`roughness_to_lod`] (the blurriest mip).
    pub max_lod: f32,
    /// Reflection travel distance (world units) at which the distance term adds
    /// one full extra LOD; larger values make distance blur more gradual.
    pub distance_lod_scale: f32,
}

impl ReflectionParams {
    /// A reasonable default: 8%-per-side fade, up to 6 mips, 1 LOD per 20 units.
    pub const DEFAULT: ReflectionParams = ReflectionParams {
        edge_fade_width: 0.08,
        max_lod: 6.0,
        distance_lod_scale: 20.0,
    };
}

impl Default for ReflectionParams {
    #[inline]
    fn default() -> Self {
        ReflectionParams::DEFAULT
    }
}

/// A sampled reflection tap: where to read, which mip, and how much to trust it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectionSample {
    /// Screen-space UV in `[0, 1]^2` to read the reflection buffer at.
    pub uv: Vec2,
    /// Mip / blur level to read, in `[0, max_lod]`.
    pub lod: f32,
    /// Confidence weight in `[0, 1]`: `0` means the tap is invalid/off-screen
    /// (fall back to a probe/environment reflection), `1` means fully trusted.
    pub weight: f32,
    /// `true` when the point projected in front of the camera and on-screen
    /// (before the soft edge fade is applied).
    pub on_screen: bool,
}

/// Hermite `smoothstep` over `[edge0, edge1]`, clamped and robust to `edge0 ==
/// edge1` (which degenerates to a hard step at that value).
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = edge1 - edge0;
    if !denom.is_finite() || denom.abs() <= f32::EPSILON {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / denom).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Projects a world point through `clip_from_world` to screen UV in `[0, 1]^2`.
///
/// Returns the clamped UV together with a flag that is `true` only when the
/// point was in front of the camera (`w > eps`) and the raw UV fell inside
/// `[0, 1]^2` before clamping.  A point behind the eye returns the frame centre
/// with `on_screen = false` so callers can reject it.
pub fn project_to_uv(clip_from_world: &Mat4, p_world: Vec3) -> (Vec2, bool) {
    let p = sanitize3(p_world);
    let clip = clip_from_world.mul_vec4(p.extend(1.0));
    if !clip.is_finite() || !clip.w.is_finite() || clip.w <= MIN_W {
        return (Vec2::splat(0.5), false);
    }
    let inv_w = 1.0 / clip.w;
    let ndc_x = clip.x * inv_w;
    let ndc_y = clip.y * inv_w;
    if !ndc_x.is_finite() || !ndc_y.is_finite() {
        return (Vec2::splat(0.5), false);
    }
    let u = ndc_x * 0.5 + 0.5;
    let v = 0.5 - ndc_y * 0.5;
    let on_screen = (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v);
    (Vec2::new(u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)), on_screen)
}

/// Maps roughness and reflection distance to a blur mip level in `[0, max_lod]`.
///
/// The roughness term uses `sqrt(roughness)` so the LOD climbs quickly out of
/// the mirror regime (matching the way GGX lobe width grows with roughness),
/// scaled to span `[0, max_lod]`.  The distance term adds `distance /
/// distance_lod_scale` extra levels for far-away reflections.  The sum is
/// clamped to `[0, max_lod]` and is monotonically non-decreasing in both
/// roughness and distance.
pub fn roughness_to_lod(roughness: f32, distance: f32, params: &ReflectionParams) -> f32 {
    let max_lod = if params.max_lod.is_finite() {
        params.max_lod.max(0.0)
    } else {
        0.0
    };
    // Sanitise inputs: a non-finite roughness collapses to the mirror case and a
    // non-finite distance saturates to the blurriest mip.
    let r = if roughness.is_finite() {
        roughness.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let rough_lod = r.sqrt() * max_lod;
    let scale = params.distance_lod_scale;
    let dist = if distance.is_finite() {
        distance.max(0.0)
    } else {
        f32::MAX
    };
    let dist_lod = if scale.is_finite() && scale > f32::EPSILON {
        dist / scale
    } else {
        0.0
    };
    let lod = rough_lod + dist_lod;
    if lod.is_finite() {
        lod.clamp(0.0, max_lod)
    } else {
        max_lod
    }
}

/// Smooth border weight for a UV: `1` in the interior, `0` at the frame edge.
///
/// Uses a [`smoothstep`] ramp of half-width `fade_width` on each of the four
/// borders and multiplies the horizontal and vertical ramps, so corners fade
/// faster than edges.  `fade_width` is clamped to `[0, 0.5]`; a width of `0`
/// disables the fade (returns `1` everywhere inside, `0` only exactly on the
/// border).
pub fn edge_fade(uv: Vec2, fade_width: f32) -> f32 {
    let w = if fade_width.is_finite() {
        fade_width.clamp(0.0, 0.5)
    } else {
        0.0
    };
    // A non-finite coordinate is treated as sitting on the border (fully faded).
    let u = if uv.x.is_finite() { uv.x.clamp(0.0, 1.0) } else { 0.0 };
    let v = if uv.y.is_finite() { uv.y.clamp(0.0, 1.0) } else { 0.0 };
    if w <= f32::EPSILON {
        // Degenerate band: full weight strictly inside, zero on the border.
        let inside = u > 0.0 && u < 1.0 && v > 0.0 && v < 1.0;
        return if inside { 1.0 } else { 0.0 };
    }
    // Distance to the nearest horizontal / vertical border.
    let du = u.min(1.0 - u);
    let dv = v.min(1.0 - v);
    let fu = smoothstep(0.0, w, du);
    let fv = smoothstep(0.0, w, dv);
    (fu * fv).clamp(0.0, 1.0)
}

/// Projects a reflected world point and returns a full [`ReflectionSample`].
///
/// Combines [`project_to_uv`], [`edge_fade`], and [`roughness_to_lod`].  The
/// weight is the edge fade when the point is on-screen and `0` when it projected
/// behind the camera or off-frame, so an invalid tap is cleanly rejected.
///
/// * `clip_from_world` — the main (non-mirrored) camera view-projection.
/// * `reflected_point` — the world-space reflected hit position to look up.
/// * `distance` — world-space reflection travel distance (reflector → hit).
/// * `roughness` — perceptual roughness of the reflective surface in `[0, 1]`.
pub fn sample_reflection(
    clip_from_world: &Mat4,
    reflected_point: Vec3,
    distance: f32,
    roughness: f32,
    params: &ReflectionParams,
) -> ReflectionSample {
    let (uv, on_screen) = project_to_uv(clip_from_world, reflected_point);
    let lod = roughness_to_lod(roughness, distance, params);
    let weight = if on_screen {
        edge_fade(uv, params.edge_fade_width)
    } else {
        0.0
    };
    ReflectionSample {
        uv,
        lod,
        weight,
        on_screen,
    }
}

/// Clamps a position to all-finite components, mapping non-finite axes to 0.
#[inline]
fn sanitize3(v: Vec3) -> Vec3 {
    Vec3::new(
        if v.x.is_finite() { v.x } else { 0.0 },
        if v.y.is_finite() { v.y } else { 0.0 },
        if v.z.is_finite() { v.z } else { 0.0 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A right-handed `[0, 1]`-depth perspective projection (wgpu/D3D style),
    /// built explicitly so the test avoids deprecated helpers.  Looks down `-Z`.
    fn perspective() -> Mat4 {
        let fov_y = core::f32::consts::FRAC_PI_3;
        let (aspect, near, far) = (1.0, 0.1, 100.0);
        let f = 1.0 / (fov_y * 0.5).tan();
        let r = far / (near - far);
        Mat4::from_cols(
            bevy_math::Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            bevy_math::Vec4::new(0.0, f, 0.0, 0.0),
            bevy_math::Vec4::new(0.0, 0.0, r, -1.0),
            bevy_math::Vec4::new(0.0, 0.0, near * far / (near - far), 0.0),
        )
    }

    #[test]
    fn frame_centre_projects_to_uv_centre() {
        let proj = perspective();
        // A point straight ahead on the view axis (view looks down -Z).
        let (uv, on_screen) = project_to_uv(&proj, Vec3::new(0.0, 0.0, -5.0));
        assert!(on_screen);
        assert!((uv - Vec2::splat(0.5)).length() < 1e-5, "centre uv = {uv:?}");
    }

    #[test]
    fn projected_uv_is_always_clamped_to_unit_square() {
        let proj = perspective();
        for p in [
            Vec3::new(100.0, 0.0, -5.0),
            Vec3::new(-50.0, 80.0, -2.0),
            Vec3::new(0.0, -200.0, -10.0),
        ] {
            let (uv, _) = project_to_uv(&proj, p);
            assert!((0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y));
        }
    }

    #[test]
    fn point_behind_eye_is_off_screen() {
        let proj = perspective();
        // Positive view-space z is behind the camera.
        let (uv, on_screen) = project_to_uv(&proj, Vec3::new(0.0, 0.0, 5.0));
        assert!(!on_screen);
        assert_eq!(uv, Vec2::splat(0.5));
    }

    #[test]
    fn edge_fade_is_full_at_centre_and_zero_at_border() {
        let p = ReflectionParams::DEFAULT;
        assert!((edge_fade(Vec2::splat(0.5), p.edge_fade_width) - 1.0).abs() < 1e-6);
        // Exactly on the border: zero.
        assert!(edge_fade(Vec2::new(0.0, 0.5), p.edge_fade_width) < 1e-6);
        assert!(edge_fade(Vec2::new(0.5, 1.0), p.edge_fade_width) < 1e-6);
        // Just inside the fade band: strictly between 0 and 1.
        let near = edge_fade(Vec2::new(0.04, 0.5), p.edge_fade_width);
        assert!(near > 0.0 && near < 1.0, "edge band weight = {near}");
    }

    #[test]
    fn corners_fade_faster_than_edges() {
        let w = 0.1;
        let edge = edge_fade(Vec2::new(0.05, 0.5), w);
        let corner = edge_fade(Vec2::new(0.05, 0.05), w);
        assert!(corner < edge, "corner {corner} should fade below edge {edge}");
    }

    #[test]
    fn lod_is_monotonic_in_roughness() {
        let p = ReflectionParams::DEFAULT;
        let mut prev = -1.0;
        let mut r = 0.0;
        while r <= 1.0 + 1e-6 {
            let lod = roughness_to_lod(r, 0.0, &p);
            assert!(lod >= prev - 1e-6, "lod decreased at roughness {r}");
            assert!(lod >= 0.0 && lod <= p.max_lod + 1e-6);
            prev = lod;
            r += 0.05;
        }
        // Endpoints: mirror -> 0, fully rough -> max.
        assert!(roughness_to_lod(0.0, 0.0, &p).abs() < 1e-6);
        assert!((roughness_to_lod(1.0, 0.0, &p) - p.max_lod).abs() < 1e-6);
    }

    #[test]
    fn lod_is_monotonic_in_distance_and_clamped() {
        let p = ReflectionParams::DEFAULT;
        let mut prev = -1.0;
        for d in [0.0, 5.0, 20.0, 100.0, 1.0e6] {
            let lod = roughness_to_lod(0.3, d, &p);
            assert!(lod >= prev - 1e-6, "lod decreased at distance {d}");
            assert!(lod <= p.max_lod + 1e-6, "lod exceeded max at distance {d}");
            prev = lod;
        }
    }

    #[test]
    fn sample_reflection_rejects_behind_camera() {
        let proj = perspective();
        let s = sample_reflection(&proj, Vec3::new(0.0, 0.0, 10.0), 3.0, 0.2, &ReflectionParams::DEFAULT);
        assert!(!s.on_screen);
        assert_eq!(s.weight, 0.0);
    }

    #[test]
    fn sample_reflection_centre_is_fully_weighted() {
        let proj = perspective();
        let s = sample_reflection(&proj, Vec3::new(0.0, 0.0, -5.0), 0.0, 0.0, &ReflectionParams::DEFAULT);
        assert!(s.on_screen);
        assert!((s.weight - 1.0).abs() < 1e-5);
        assert!(s.lod.abs() < 1e-6);
        assert!((s.uv - Vec2::splat(0.5)).length() < 1e-5);
    }

    #[test]
    fn non_finite_inputs_stay_finite() {
        let proj = perspective();
        let (uv, on_screen) = project_to_uv(&proj, Vec3::new(f32::NAN, 0.0, -5.0));
        assert!(uv.is_finite());
        let _ = on_screen;
        let lod = roughness_to_lod(f32::NAN, f32::INFINITY, &ReflectionParams::DEFAULT);
        assert!(lod.is_finite());
        let fade = edge_fade(Vec2::new(f32::NAN, 0.5), 0.1);
        assert!(fade.is_finite());
    }

    #[test]
    fn smoothstep_handles_degenerate_edges() {
        assert_eq!(smoothstep(1.0, 1.0, 0.5), 0.0);
        assert_eq!(smoothstep(1.0, 1.0, 2.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
    }
}
