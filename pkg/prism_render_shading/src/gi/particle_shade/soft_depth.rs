//! Soft-particle depth fade: depth linearisation and intersection-aware
//! opacity falloff — CPU golden reference.
//!
//! Billboarded particles (smoke, fire, dust, magic) are flat camera-facing
//! quads that visibly *clip* against opaque scene geometry: the quad plane
//! slices through the floor or a wall and leaves a razor-sharp seam where the
//! particle depth equals the scene depth.  "Soft particles" (Lorach 2007) hide
//! that seam by fading a fragment's opacity to zero as its camera-space depth
//! approaches the depth already stored in the scene depth buffer, so the
//! particle appears to dissolve into the surface rather than intersect it.
//!
//! A second, independent falloff fades particles that drift too close to the
//! near plane ("camera fade"): a particle quad straddling the camera would
//! otherwise fill the screen with a single giant texel.
//!
//! This module provides the backend-neutral numerical reference for both
//! falloffs plus the depth-buffer linearisation they operate on:
//!
//! * [`linearize_depth01`] / [`linearize_depth_ndc`] — convert a non-linear
//!   hardware depth (clip `[0, 1]` or OpenGL NDC `[-1, 1]`) into a positive,
//!   linear camera-space eye distance using the projection `near` / `far`.
//! * [`soft_fade`] — the smooth intersection falloff from the particle and
//!   scene eye-depths and a `fade_distance`.
//! * [`soft_fade_linear`] — the cheaper linear ramp variant.
//! * [`camera_fade`] — the near-plane dissolve as the particle approaches the
//!   camera.
//!
//! # Conventions
//! * All depths passed to the fades are **linear eye distances** (positive, in
//!   view units); use the `linearize_*` helpers first on raw buffer samples.
//!   The eye looks down `-Z`, so eye distance is `-view_z`; the fades take the
//!   positive magnitude.
//! * A fade returns an **opacity multiplier** in `[0, 1]`: `1` keeps the
//!   particle fully opaque, `0` fully dissolves it.  Scene geometry *behind*
//!   the particle (`scene_depth > particle_depth`) is the lit case.
//! * Every function is a deterministic pure function (no RNG / I/O / GPU /
//!   globals / `unsafe`).  Inputs are defensively clamped — `near` / `far`
//!   degenerate ranges, non-finite samples, and zero fade widths all fall back
//!   to a safe branch and never yield `NaN` or infinity.
//!
//! # References
//! * T. Lorach, "Soft Particles", NVIDIA SDK white paper, 2007.
//! * NVIDIA GPU Gems / DirectX SDK "SoftParticles" sample (depth-buffer fade).
//! * E. Lengyel, *Foundations of Game Engine Development, Vol. 2: Rendering*,
//!   §6 (perspective depth and reverse-Z linearisation).

/// Smallest separation between `near` and `far` treated as a valid frustum.
const MIN_RANGE: f32 = 1.0e-6;

/// Smallest fade width that still produces a smooth ramp; below this the fade
/// collapses to a hard step to avoid dividing by (near-)zero.
const MIN_WIDTH: f32 = 1.0e-6;

/// Clamp `x` into `[lo, hi]`, mapping any non-finite input to `lo`.
///
/// `lo` is assumed `<= hi`; callers pass ordered bounds.
#[inline]
fn clamp_finite(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_finite() {
        x.clamp(lo, hi)
    } else {
        lo
    }
}

/// Saturate `x` to `[0, 1]`, mapping non-finite input to `0`.
#[inline]
fn saturate(x: f32) -> f32 {
    clamp_finite(x, 0.0, 1.0)
}

/// Hermite `smoothstep` on an already-saturated parameter `t`.
///
/// Returns `t * t * (3 - 2 t)`; `t` is saturated to `[0, 1]` first so the
/// result is monotone in `[0, 1]` with zero first derivative at both ends.
#[inline]
fn smoothstep01(t: f32) -> f32 {
    let t = saturate(t);
    t * t * (3.0 - 2.0 * t)
}

/// Normalise a positive camera-space eye distance from a raw sample.
///
/// Depth buffers and view-space `z` can arrive signed (eye looks down `-Z`);
/// this returns the finite, non-negative magnitude used by the fades.
#[inline]
fn eye_distance(z: f32) -> f32 {
    if z.is_finite() {
        z.abs()
    } else {
        0.0
    }
}

/// Linearise a clip-space depth in `[0, 1]` (D3D / Metal / Vulkan convention)
/// into a positive linear eye distance.
///
/// For a standard perspective projection mapping eye distance `[near, far]` to
/// clip depth `[0, 1]`, the inverse is
/// `z_eye = (near * far) / (far - d * (far - near))`, giving `near` at `d = 0`
/// and `far` at `d = 1`.  `d` is saturated to `[0, 1]`; `near` is clamped
/// strictly positive and `far` is forced at least `near + MIN_RANGE`, so a
/// degenerate or inverted frustum still returns a finite value in
/// `[near, far]`.
#[inline]
pub fn linearize_depth01(d: f32, near: f32, far: f32) -> f32 {
    let d = saturate(d);
    let near = clamp_finite(near, MIN_RANGE, f32::MAX);
    let far = clamp_finite(far, near + MIN_RANGE, f32::MAX);
    let denom = far - d * (far - near);
    // `denom` ranges over `[near, far]` and is strictly positive here.
    let denom = denom.max(MIN_RANGE);
    let z = (near * far) / denom;
    clamp_finite(z, near, far)
}

/// Linearise an OpenGL-style NDC depth in `[-1, 1]` into a positive linear eye
/// distance.
///
/// The NDC depth is first remapped to the clip `[0, 1]` convention via
/// `d01 = 0.5 * ndc + 0.5` and then forwarded to [`linearize_depth01`], so the
/// same clamping guarantees apply.
#[inline]
pub fn linearize_depth_ndc(ndc: f32, near: f32, far: f32) -> f32 {
    let ndc = clamp_finite(ndc, -1.0, 1.0);
    linearize_depth01(0.5 * ndc + 0.5, near, far)
}

/// Smooth soft-particle intersection fade (Hermite `smoothstep`).
///
/// Both depths are linear eye distances; `scene_depth` is the opaque surface
/// behind the particle and `particle_depth` is the fragment being shaded.  The
/// signed separation `scene_depth - particle_depth` is normalised by
/// `fade_distance` and passed through [`smoothstep01`], returning an opacity
/// multiplier in `[0, 1]`:
///
/// * `0` when the particle sits on or behind the surface (`separation <= 0`),
/// * ramping smoothly up as the surface recedes,
/// * `1` once the surface is at least `fade_distance` behind the particle.
///
/// A non-positive or non-finite `fade_distance` degrades to a hard visibility
/// test (opaque where the particle is strictly in front).
#[inline]
pub fn soft_fade(particle_depth: f32, scene_depth: f32, fade_distance: f32) -> f32 {
    let particle = eye_distance(particle_depth);
    let scene = eye_distance(scene_depth);
    let separation = scene - particle;
    let width = clamp_finite(fade_distance, 0.0, f32::MAX);
    if width < MIN_WIDTH {
        // Hard test: opaque only where the particle is strictly in front.
        return if separation > 0.0 { 1.0 } else { 0.0 };
    }
    smoothstep01(separation / width)
}

/// Linear-ramp soft-particle intersection fade.
///
/// Identical to [`soft_fade`] but uses a plain clamped ramp
/// `saturate(separation / fade_distance)` instead of the Hermite curve; the
/// cheaper variant some engines expose.  Shares the same degenerate-width
/// hard-test fallback.
#[inline]
pub fn soft_fade_linear(particle_depth: f32, scene_depth: f32, fade_distance: f32) -> f32 {
    let particle = eye_distance(particle_depth);
    let scene = eye_distance(scene_depth);
    let separation = scene - particle;
    let width = clamp_finite(fade_distance, 0.0, f32::MAX);
    if width < MIN_WIDTH {
        return if separation > 0.0 { 1.0 } else { 0.0 };
    }
    saturate(separation / width)
}

/// Near-plane camera fade: dissolve particles that drift too close to the eye.
///
/// `particle_depth` is the linear eye distance of the fragment.  Particles at
/// or nearer than `fade_end` are fully transparent (`0`); particles at or
/// farther than `fade_start` are fully opaque (`1`); in between the opacity
/// follows a Hermite `smoothstep`.  The two radii are reordered defensively so
/// `fade_start >= fade_end`, and a degenerate (zero-width) band becomes a hard
/// step at the shared radius.
#[inline]
pub fn camera_fade(particle_depth: f32, fade_start: f32, fade_end: f32) -> f32 {
    let depth = eye_distance(particle_depth);
    let a = clamp_finite(fade_start, 0.0, f32::MAX);
    let b = clamp_finite(fade_end, 0.0, f32::MAX);
    // Enforce `start >= end` (fully opaque is the farther radius).
    let (start, end) = if a >= b { (a, b) } else { (b, a) };
    let width = start - end;
    if width < MIN_WIDTH {
        return if depth >= start { 1.0 } else { 0.0 };
    }
    smoothstep01((depth - end) / width)
}

/// Combined depth + camera opacity multiplier.
///
/// Convenience product of [`soft_fade`] (intersection) and [`camera_fade`]
/// (near plane); the result is the opacity scale a soft particle should apply
/// before any texture / lighting alpha.  Any disabled falloff (zero width)
/// contributes its hard-test or `1` branch, so this is always in `[0, 1]`.
#[inline]
pub fn depth_opacity(
    particle_depth: f32,
    scene_depth: f32,
    fade_distance: f32,
    camera_fade_start: f32,
    camera_fade_end: f32,
) -> f32 {
    let intersect = soft_fade(particle_depth, scene_depth, fade_distance);
    let near = camera_fade(particle_depth, camera_fade_start, camera_fade_end);
    saturate(intersect * near)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0e-5;

    #[test]
    fn linearize01_hits_near_and_far_endpoints() {
        let (near, far) = (0.1, 100.0);
        assert!((linearize_depth01(0.0, near, far) - near).abs() < TOL);
        assert!((linearize_depth01(1.0, near, far) - far).abs() < TOL);
    }

    #[test]
    fn linearize01_is_monotone_increasing() {
        let (near, far) = (0.5, 50.0);
        let mut prev = linearize_depth01(0.0, near, far);
        for i in 1..=20 {
            let d = i as f32 / 20.0;
            let z = linearize_depth01(d, near, far);
            assert!(z >= prev - TOL, "non-monotone at d={d}: {z} < {prev}");
            assert!(z.is_finite());
            prev = z;
        }
    }

    #[test]
    fn linearize01_matches_closed_form_midpoint() {
        let (near, far) = (1.0, 11.0);
        // Eye distance z=2 maps to clip d = far*(z-near)/(z*(far-near)).
        let z = 2.0_f32;
        let d = far * (z - near) / (z * (far - near));
        assert!((linearize_depth01(d, near, far) - z).abs() < 1.0e-4);
    }

    #[test]
    fn linearize_ndc_matches_clip_convention() {
        let (near, far) = (0.2, 20.0);
        // NDC -1 == clip 0 == near; NDC +1 == clip 1 == far (float-eps slack).
        assert!((linearize_depth_ndc(-1.0, near, far) - near).abs() < 1.0e-3);
        assert!((linearize_depth_ndc(1.0, near, far) - far).abs() < 1.0e-3);
    }

    #[test]
    fn linearize_degenerate_range_is_finite() {
        // far <= near is repaired to a minimal valid frustum.
        let z = linearize_depth01(0.5, 10.0, 10.0);
        assert!(z.is_finite() && z >= 10.0);
        // Non-finite inputs fall back without producing NaN.
        assert!(linearize_depth01(f32::NAN, f32::NAN, f32::NAN).is_finite());
    }

    #[test]
    fn soft_fade_zero_at_intersection_one_when_clear() {
        // Particle and surface coincident -> fully dissolved.
        assert!(soft_fade(5.0, 5.0, 2.0).abs() < TOL);
        // Surface far behind -> fully opaque.
        assert!((soft_fade(5.0, 50.0, 2.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn soft_fade_particle_in_front_of_surface_is_transparent() {
        // Particle behind the opaque surface: separation negative -> 0.
        assert!(soft_fade(20.0, 5.0, 3.0).abs() < TOL);
    }

    #[test]
    fn soft_fade_is_monotone_and_smooth() {
        let particle = 10.0;
        let fade = 4.0;
        let mut prev = soft_fade(particle, particle, fade);
        for i in 0..=16 {
            let scene = particle + i as f32 * (fade / 8.0);
            let v = soft_fade(particle, scene, fade);
            assert!((0.0..=1.0).contains(&v));
            assert!(v >= prev - TOL, "non-monotone: {v} < {prev}");
            prev = v;
        }
    }

    #[test]
    fn soft_fade_smooth_vs_linear_endpoints_agree() {
        let (p, fade) = (3.0, 2.0);
        // Both agree at the clamped endpoints.
        assert!((soft_fade(p, p, fade)).abs() < TOL);
        assert!((soft_fade_linear(p, p, fade)).abs() < TOL);
        assert!((soft_fade(p, p + 5.0, fade) - 1.0).abs() < TOL);
        assert!((soft_fade_linear(p, p + 5.0, fade) - 1.0).abs() < TOL);
        // Linear ramp sits at exactly 0.5 at the half-separation; smoothstep too.
        assert!((soft_fade_linear(p, p + fade * 0.5, fade) - 0.5).abs() < TOL);
        assert!((soft_fade(p, p + fade * 0.5, fade) - 0.5).abs() < TOL);
    }

    #[test]
    fn soft_fade_zero_width_is_hard_test() {
        assert!((soft_fade(5.0, 6.0, 0.0) - 1.0).abs() < TOL);
        assert!(soft_fade(6.0, 5.0, 0.0).abs() < TOL);
        assert!(soft_fade(5.0, 5.0, 0.0).abs() < TOL);
    }

    #[test]
    fn camera_fade_near_and_far_branches() {
        // Within fade_end -> transparent; beyond fade_start -> opaque.
        assert!(camera_fade(0.1, 2.0, 0.5).abs() < TOL);
        assert!((camera_fade(10.0, 2.0, 0.5) - 1.0).abs() < TOL);
    }

    #[test]
    fn camera_fade_handles_swapped_radii() {
        // Swapped args are reordered to the same band.
        let a = camera_fade(1.25, 2.0, 0.5);
        let b = camera_fade(1.25, 0.5, 2.0);
        assert!((a - b).abs() < TOL);
        assert!((0.0..=1.0).contains(&a));
    }

    #[test]
    fn camera_fade_zero_width_is_hard_step() {
        assert!((camera_fade(3.0, 2.0, 2.0) - 1.0).abs() < TOL);
        assert!(camera_fade(1.0, 2.0, 2.0).abs() < TOL);
    }

    #[test]
    fn depth_opacity_is_product_in_unit_range() {
        let v = depth_opacity(5.0, 7.0, 4.0, 2.0, 0.5);
        let expect = soft_fade(5.0, 7.0, 4.0) * camera_fade(5.0, 2.0, 0.5);
        assert!((v - expect).abs() < TOL);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn no_nan_on_pathological_inputs() {
        for &x in &[f32::NAN, f32::INFINITY, -f32::INFINITY, 0.0, -3.0] {
            assert!(soft_fade(x, x, x).is_finite());
            assert!(soft_fade_linear(x, x, x).is_finite());
            assert!(camera_fade(x, x, x).is_finite());
            assert!(depth_opacity(x, x, x, x, x).is_finite());
            assert!(linearize_depth01(x, x, x).is_finite());
            assert!(linearize_depth_ndc(x, x, x).is_finite());
        }
    }
}
