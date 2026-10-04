//! Resolve-level analytic rectangle area-light accumulation (CPU golden).
//!
//! This is the backend-neutral authoritative reference for the
//! `accumulate_area_lights` block of `prism_render_scene`'s
//! `shaders/shading_resolve.wesl`: given a reconstructed [`SurfaceSample`], its
//! [`ShadingFrame`] and world position, it integrates every resident polygonal
//! rectangle area light with Heitz et al. 2016 Linearly Transformed Cosines.
//! The GPU resolve kernel is a byte-exact twin of this logic (it inlines the
//! same clipped-polygon maths from [`crate::gi::area_light::polygon`] and
//! samples the same baked LUT from [`crate::gi::area_light::ltc_lut`]).
//!
//! The specular lobe uses the LTC-fit `M⁻¹` transform (so it tracks the GGX
//! roughness), the diffuse lobe the identity clamped-cosine form factor; both
//! are weighted by the principled material response exactly like the
//! clustered-lighting direct loops in [`crate::resolve`]. No environment-BRDF
//! factor is applied, so this never double-counts the image-based specular
//! already folded into the indirect term.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG / IO / GPU / `unsafe`.
//! * Defensive clamping everywhere; the public entry points never emit `NaN`.
//! * Winding, culling and attenuation mirror the GPU kernel bit-for-bit:
//!   - single-sided emitters cull receivers strictly behind the emitter plane
//!     (`dot(p - center, normalize(cross(axis_u, axis_v))) < 0`),
//!   - the optional range window is `atten = x²` with
//!     `x = saturate(1 - dist / range)` and a hard cut at `dist >= range`,
//!   - degenerate rectangles (`half_width <= 0 || half_height <= 0`) contribute
//!     nothing.

use bevy_math::Vec3;

use crate::gi::area_light::ltc_lut::LtcLut;
use crate::gi::area_light::polygon::{quad_form_factor, quad_ltc_evaluate, rectangle_points};
use crate::{ShadingFrame, SurfaceSample};

/// A planar rectangle (quad) polygonal area light expressed in world space.
///
/// Mirrors the resident `GpuAreaLight` record consumed by the GPU resolve
/// (`prism_render_scene::shading::area_light::abi`) restricted to the rectangle
/// shape. `axis_u` / `axis_v` are the orthonormal in-plane axes; `half_width` /
/// `half_height` are the half-extents along them. `color * intensity` is the
/// emitted radiance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AreaLightRect {
    /// World-space center of the rectangle.
    pub center: [f32; 3],
    /// Unit in-plane axis spanning the width.
    pub axis_u: [f32; 3],
    /// Unit in-plane axis spanning the height.
    pub axis_v: [f32; 3],
    /// Half-extent along `axis_u`. Non-positive marks a disabled/degenerate
    /// light that contributes nothing.
    pub half_width: f32,
    /// Half-extent along `axis_v`. Non-positive marks a disabled/degenerate
    /// light that contributes nothing.
    pub half_height: f32,
    /// Linear emitter color.
    pub color: [f32; 3],
    /// Scalar radiance multiplier applied to [`Self::color`].
    pub intensity: f32,
    /// Optional smooth range window. `range <= 0` disables attenuation; a
    /// receiver farther than `range` receives nothing.
    pub range: f32,
    /// When `false`, only the front face (the `+cross(axis_u, axis_v)`
    /// half-space) emits; receivers behind the plane are culled.
    pub two_sided: bool,
}

impl Default for AreaLightRect {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            axis_u: [1.0, 0.0, 0.0],
            axis_v: [0.0, 1.0, 0.0],
            half_width: 0.0,
            half_height: 0.0,
            color: [1.0, 1.0, 1.0],
            intensity: 0.0,
            range: 0.0,
            two_sided: false,
        }
    }
}

#[inline]
fn to_vec3(v: [f32; 3]) -> Vec3 {
    Vec3::new(v[0], v[1], v[2])
}

/// Normalizes `value`, returning `fallback` for degenerate/non-finite inputs.
/// Mirrors `brdf_normalize_or` in the GPU kernel and `normalize_or` in the CPU
/// reference.
#[inline]
fn normalize_or(value: Vec3, fallback: Vec3) -> Vec3 {
    let length_squared = value.dot(value);
    if length_squared > 1.0e-12 && length_squared.is_finite() {
        value * length_squared.sqrt().recip()
    } else {
        fallback
    }
}

/// Grazing-angle `F0` for the surface, shared by every indirect path.
/// Mirrors `indirect_f0` in `shading_resolve.wesl`.
#[inline]
fn indirect_f0(surface: &SurfaceSample) -> Vec3 {
    let metallic = surface.metallic.clamp(0.0, 1.0);
    let reflectance = surface.reflectance.clamp(0.0, 1.0);
    let f0_dielectric = 0.16 * reflectance * reflectance;
    Vec3::splat(f0_dielectric).lerp(to_vec3(surface.base_color), metallic)
}

#[inline]
fn sanitize(v: Vec3) -> [f32; 3] {
    if v.is_finite() {
        [v.x, v.y, v.z]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Evaluates the principled analytic response of a single rectangle area light
/// at the shaded point.
///
/// Returns the linear radiance contribution `[r, g, b]`. The specular lobe is
/// only evaluated when a baked LUT is supplied; with `lut == None` the result
/// is the diffuse-only clamped-cosine term (production always binds the baked
/// LUT, matching the GPU path). The result is always finite.
pub fn principled_area_contribution(
    surface: &SurfaceSample,
    frame: ShadingFrame,
    position: [f32; 3],
    light: &AreaLightRect,
    lut: Option<&LtcLut>,
) -> [f32; 3] {
    // Degenerate / disabled record (mirrors the GPU `half <= 0` skip).
    if light.half_width <= 0.0 || light.half_height <= 0.0 {
        return [0.0; 3];
    }

    let position = to_vec3(position);
    let center = to_vec3(light.center);
    let axis_u = to_vec3(light.axis_u);
    let axis_v = to_vec3(light.axis_v);

    // One-sided emitters cull receivers strictly behind the emitter plane.
    if !light.two_sided {
        let light_normal = axis_u.cross(axis_v).normalize_or_zero();
        if (position - center).dot(light_normal) < 0.0 {
            return [0.0; 3];
        }
    }

    // Optional smooth range window (`range <= 0` disables it).
    let atten = if light.range > 0.0 {
        let dist = (center - position).length();
        if dist >= light.range {
            return [0.0; 3];
        }
        let x = (1.0 - dist / light.range).clamp(0.0, 1.0);
        x * x
    } else {
        1.0
    };

    let normal = normalize_or(to_vec3(frame.normal), Vec3::Y);
    let view = normalize_or(to_vec3(frame.view), normal);
    let n_dot_v = normal.dot(view).max(1.0e-4);
    let roughness = surface.perceptual_roughness.clamp(0.0, 1.0);

    // Orthonormal shading frame with the normal as `+Z` (the LTC convention).
    // The usual tangent `normalize(view - normal * dot(normal, view))`
    // degenerates to zero when the view is parallel to the normal
    // (perpendicular viewing), which would collapse every quad corner onto the
    // `+Z` axis and yield a spurious zero form factor. Fall back to any stable
    // axis perpendicular to the normal: at normal incidence the lobe is
    // rotationally symmetric about `+Z`, so the in-plane choice is immaterial.
    let projected = view - normal * normal.dot(view);
    let tangent = if projected.dot(projected) > 1.0e-12 {
        projected.normalize_or_zero()
    } else {
        let helper = if normal.y.abs() < 0.999 { Vec3::Y } else { Vec3::X };
        helper.cross(normal).normalize_or_zero()
    };
    let bitangent = normal.cross(tangent);
    let to_frame = |world: Vec3| {
        let rel = world - position;
        Vec3::new(tangent.dot(rel), bitangent.dot(rel), normal.dot(rel))
    };

    // Counter-clockwise corners: c-u-v, c+u-v, c+u+v, c-u+v.
    let corners = rectangle_points(center, axis_u, axis_v, light.half_width, light.half_height);
    let p0 = to_frame(corners[0]);
    let p1 = to_frame(corners[1]);
    let p2 = to_frame(corners[2]);
    let p3 = to_frame(corners[3]);

    let diffuse_response = quad_form_factor(p0, p1, p2, p3);
    let specular_response = match lut {
        Some(lut) => {
            let coeffs = lut.sample(n_dot_v, roughness);
            quad_ltc_evaluate(p0, p1, p2, p3, &coeffs)
        }
        None => 0.0,
    };

    let f0 = indirect_f0(surface);
    let metallic = surface.metallic.clamp(0.0, 1.0);
    let diffuse_color = to_vec3(surface.base_color) * (1.0 - metallic);
    let radiance = to_vec3(light.color) * light.intensity;

    let contribution = radiance * (f0 * specular_response + diffuse_color * diffuse_response) * atten;
    sanitize(contribution)
}

/// Accumulates the principled analytic response of every rectangle area light
/// in `lights` at the shaded point. Mirrors `accumulate_area_lights` in
/// `shading_resolve.wesl`. The result is always finite.
pub fn accumulate_area_lights(
    surface: &SurfaceSample,
    frame: ShadingFrame,
    position: [f32; 3],
    lights: &[AreaLightRect],
    lut: Option<&LtcLut>,
) -> [f32; 3] {
    let mut total = Vec3::ZERO;
    for light in lights {
        total += to_vec3(principled_area_contribution(surface, frame, position, light, lut));
    }
    sanitize(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi::area_light::ltc_lut::bake_ltc_lut_default;

    fn test_surface() -> SurfaceSample {
        SurfaceSample {
            base_color: [0.8, 0.6, 0.4],
            metallic: 0.0,
            perceptual_roughness: 0.5,
            reflectance: 0.5,
            ..SurfaceSample::default()
        }
    }

    /// A unit-ish rectangle hovering above the origin, front face pointing down
    /// (`-Z` emitter) toward a receiver whose normal points up (`+Z`).
    fn overhead_light() -> AreaLightRect {
        AreaLightRect {
            center: [0.0, 0.0, 2.0],
            axis_u: [1.0, 0.0, 0.0],
            axis_v: [0.0, 1.0, 0.0],
            half_width: 1.0,
            half_height: 1.0,
            color: [1.0, 1.0, 1.0],
            intensity: 4.0,
            range: 0.0,
            // Emitter hovers above the receiver; its front face (+Z) points
            // up/away, so make it two-sided to light the surface below.
            two_sided: true,
        }
    }

    fn up_frame() -> ShadingFrame {
        ShadingFrame {
            normal: [0.0, 0.0, 1.0],
            view: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
        }
    }

    fn is_finite3(v: [f32; 3]) -> bool {
        v.iter().all(|c| c.is_finite())
    }

    #[test]
    fn facing_quad_illuminates() {
        let lut = bake_ltc_lut_default();
        let out = principled_area_contribution(
            &test_surface(),
            up_frame(),
            [0.0, 0.0, 0.0],
            &overhead_light(),
            Some(&lut),
        );
        assert!(is_finite3(out));
        assert!(out.iter().all(|c| *c > 0.0), "expected positive radiance, got {out:?}");
    }

    #[test]
    fn back_face_is_culled_unless_two_sided() {
        let lut = bake_ltc_lut_default();
        // `overhead_light` hovers above the receiver with its front face (+Z)
        // pointing up/away, so the receiver below sits strictly behind the
        // single-sided emission plane: `dot(p - center, +Z) = -2 < 0`.
        let mut light = overhead_light();
        light.two_sided = false;
        let culled = principled_area_contribution(
            &test_surface(),
            up_frame(),
            [0.0, 0.0, 0.0],
            &light,
            Some(&lut),
        );
        assert_eq!(culled, [0.0; 3], "single-sided emitter facing away must not illuminate");

        // Flipping to two-sided bypasses the plane cull while keeping the
        // correct (+Y) winding, so the same quad now illuminates from behind.
        light.two_sided = true;
        let lit = principled_area_contribution(
            &test_surface(),
            up_frame(),
            [0.0, 0.0, 0.0],
            &light,
            Some(&lut),
        );
        assert!(lit.iter().any(|c| *c > 0.0), "two-sided emitter must illuminate from behind");
    }

    #[test]
    fn out_of_range_contributes_nothing() {
        let lut = bake_ltc_lut_default();
        let mut light = overhead_light();
        light.range = 1.0; // receiver is 2 units away -> culled.
        let out = principled_area_contribution(
            &test_surface(),
            up_frame(),
            [0.0, 0.0, 0.0],
            &light,
            Some(&lut),
        );
        assert_eq!(out, [0.0; 3]);
    }

    #[test]
    fn range_window_attenuates_with_distance() {
        let lut = bake_ltc_lut_default();
        let mut near = overhead_light();
        near.center = [0.0, 0.0, 1.0];
        near.range = 10.0;
        let mut far = near;
        far.center = [0.0, 0.0, 5.0];
        let near_out = principled_area_contribution(
            &test_surface(),
            up_frame(),
            [0.0, 0.0, 0.0],
            &near,
            Some(&lut),
        );
        let far_out = principled_area_contribution(
            &test_surface(),
            up_frame(),
            [0.0, 0.0, 0.0],
            &far,
            Some(&lut),
        );
        assert!(near_out[0] > far_out[0], "nearer light must dominate: {near_out:?} vs {far_out:?}");
    }

    #[test]
    fn degenerate_rectangle_is_skipped() {
        let lut = bake_ltc_lut_default();
        let mut light = overhead_light();
        light.half_width = 0.0;
        let out = principled_area_contribution(
            &test_surface(),
            up_frame(),
            [0.0, 0.0, 0.0],
            &light,
            Some(&lut),
        );
        assert_eq!(out, [0.0; 3]);
    }

    #[test]
    fn none_lut_is_diffuse_only_but_positive() {
        let out = principled_area_contribution(
            &test_surface(),
            up_frame(),
            [0.0, 0.0, 0.0],
            &overhead_light(),
            None,
        );
        assert!(is_finite3(out));
        assert!(out.iter().all(|c| *c > 0.0), "diffuse-only term must still illuminate");
    }

    #[test]
    fn rougher_surface_broadens_specular() {
        let lut = bake_ltc_lut_default();
        let mut smooth = test_surface();
        smooth.base_color = [0.0, 0.0, 0.0]; // isolate the specular (f0) lobe.
        smooth.reflectance = 1.0;
        smooth.perceptual_roughness = 0.1;
        let mut rough = smooth;
        rough.perceptual_roughness = 0.9;
        let smooth_out = principled_area_contribution(
            &smooth,
            up_frame(),
            [0.0, 0.0, 0.0],
            &overhead_light(),
            Some(&lut),
        );
        let rough_out = principled_area_contribution(
            &rough,
            up_frame(),
            [0.0, 0.0, 0.0],
            &overhead_light(),
            Some(&lut),
        );
        assert!(is_finite3(smooth_out) && is_finite3(rough_out));
        assert!(
            (smooth_out[0] - rough_out[0]).abs() > 1.0e-5,
            "roughness must change the specular response: {smooth_out:?} vs {rough_out:?}"
        );
    }

    #[test]
    fn accumulate_sums_lights_and_stays_finite() {
        let lut = bake_ltc_lut_default();
        let light = overhead_light();
        let one = accumulate_area_lights(
            &test_surface(),
            up_frame(),
            [0.0, 0.0, 0.0],
            &[light],
            Some(&lut),
        );
        let two = accumulate_area_lights(
            &test_surface(),
            up_frame(),
            [0.0, 0.0, 0.0],
            &[light, light],
            Some(&lut),
        );
        assert!(is_finite3(one) && is_finite3(two));
        for c in 0..3 {
            assert!((two[c] - 2.0 * one[c]).abs() <= 1.0e-5 * one[c].max(1.0));
        }
    }

    #[test]
    fn empty_slice_is_zero() {
        let lut = bake_ltc_lut_default();
        let out = accumulate_area_lights(
            &test_surface(),
            up_frame(),
            [0.0, 0.0, 0.0],
            &[],
            Some(&lut),
        );
        assert_eq!(out, [0.0; 3]);
    }

    #[test]
    fn non_finite_frame_does_not_emit_nan() {
        let lut = bake_ltc_lut_default();
        let frame = ShadingFrame {
            normal: [0.0, 0.0, 0.0],
            view: [0.0, 0.0, 0.0],
            tangent: [0.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, 0.0],
        };
        let out = principled_area_contribution(
            &test_surface(),
            frame,
            [0.0, 0.0, 0.0],
            &overhead_light(),
            Some(&lut),
        );
        assert!(is_finite3(out));
    }
}
