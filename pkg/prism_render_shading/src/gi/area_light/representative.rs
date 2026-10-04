//! Most-Representative-Point (MRP) area-light sampling for the specialized
//! shading lobes (toon / hair / water) — CPU golden reference.
//!
//! The GGX-family closures (principled / cloth / subsurface / clearcoat) take
//! the full Linearly-Transformed-Cosine `area_light_term` from
//! [`crate::gi::area_light::resolve`].  The stylized (NPR), hair and water lobes
//! use specialized BRDFs where the raw LTC specular transform is physically
//! meaningless, so they instead convert each rectangle area light into an
//! equivalent [`DirectLightSample`] with the Most-Representative-Point technique
//! (Karis 2013, *Real Shading in Unreal Engine 4*) and run each model's own
//! analytic direct BRDF.
//!
//! # Energy formulation
//! For a rectangle light at shaded point `p` with normal `n`:
//! * `L = normalize(closest_point_on_rect(p) - p)` is the representative
//!   direction; it places the specialized specular highlight at the point on the
//!   emitter nearest the receiver.
//! * `F_diff` is the clamped-cosine polygon form factor evaluated in an LTC
//!   frame about `n`.  The clamped cosine depends only on the `+Z` (normal)
//!   component, so it is view-independent and the in-plane tangent choice is
//!   immaterial; this is byte-identical to the diffuse lobe of
//!   [`crate::gi::area_light::resolve::principled_area_contribution`].
//! * The returned illuminance is `radiance * F_diff / max(dot(n, L), 1e-3) *
//!   atten`, so a Lambertian lobe's own trailing `* NdotL` reproduces the
//!   analytic area result `albedo * radiance * F_diff`.  This keeps the diffuse
//!   response energy-consistent with the GGX area term while the specialized
//!   specular highlight is placed by `L`.
//!
//! The degenerate / one-sided-cull / range-window gates are byte-identical to
//! `principled_area_contribution`, so an emitter either lights every lobe or
//! none of them.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG / IO / GPU / `unsafe`.
//! * Defensive clamping everywhere; returns [`None`] for a non-contributing
//!   light and never produces `NaN`.

use bevy_math::Vec3;

use crate::gi::area_light::polygon::{quad_form_factor, rectangle_points};
use crate::gi::area_light::resolve::AreaLightRect;
use crate::DirectLightSample;

#[inline]
fn to_vec3(v: [f32; 3]) -> Vec3 {
    Vec3::new(v[0], v[1], v[2])
}

/// Normalizes `value`, returning `fallback` for degenerate/non-finite inputs.
/// Mirrors `brdf_normalize_or` in the GPU kernel.
#[inline]
fn normalize_or(value: Vec3, fallback: Vec3) -> Vec3 {
    let length_squared = value.dot(value);
    if length_squared > 1.0e-12 && length_squared.is_finite() {
        value * length_squared.sqrt().recip()
    } else {
        fallback
    }
}

/// Closest point on the rectangle to `p`: project `p - center` onto the in-plane
/// axes and clamp to the half-extents.  `axis_u` / `axis_v` are unit by
/// contract; they are renormalized defensively so a mildly non-unit axis still
/// clamps against true world distances.
#[inline]
fn closest_point_on_rect(
    p: Vec3,
    center: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    half_u: f32,
    half_v: f32,
) -> Vec3 {
    let au = normalize_or(axis_u, Vec3::X);
    let av = normalize_or(axis_v, Vec3::Y);
    let d = p - center;
    let u = d.dot(au).clamp(-half_u, half_u);
    let v = d.dot(av).clamp(-half_v, half_v);
    center + au * u + av * v
}

/// Converts a rectangle area light into an equivalent [`DirectLightSample`] for
/// the specialized (toon / hair / water) lobes using the Most-Representative-
/// Point technique.
///
/// Returns [`None`] when the light does not illuminate the shaded point
/// (degenerate record, culled back face, outside the range window, or a zero
/// form factor) so the caller can skip it.  Mirrors `rect_representative_sample`
/// in `shading_resolve.wesl`.  The returned sample is always finite.
pub fn rect_representative_sample(
    position: [f32; 3],
    normal: [f32; 3],
    light: &AreaLightRect,
) -> Option<DirectLightSample> {
    // Degenerate / disabled record (mirrors the GPU `half <= 0` skip).
    if light.half_width <= 0.0 || light.half_height <= 0.0 {
        return None;
    }

    let position = to_vec3(position);
    let center = to_vec3(light.center);
    let axis_u = to_vec3(light.axis_u);
    let axis_v = to_vec3(light.axis_v);

    // One-sided emitters cull receivers strictly behind the emitter plane.
    if !light.two_sided {
        let light_normal = axis_u.cross(axis_v).normalize_or_zero();
        if (position - center).dot(light_normal) < 0.0 {
            return None;
        }
    }

    // Optional smooth range window (`range <= 0` disables it).
    let atten = if light.range > 0.0 {
        let dist = (center - position).length();
        if dist >= light.range {
            return None;
        }
        let x = (1.0 - dist / light.range).clamp(0.0, 1.0);
        x * x
    } else {
        1.0
    };

    let normal = normalize_or(to_vec3(normal), Vec3::Y);

    // Clamped-cosine form factor in an orthonormal LTC frame about the normal.
    // The clamped cosine only reads the `+Z` (normal) component, so pick any
    // stable in-plane axis.
    let helper = if normal.y.abs() < 0.999 { Vec3::Y } else { Vec3::X };
    let tangent = helper.cross(normal).normalize_or_zero();
    let bitangent = normal.cross(tangent);
    let to_frame = |world: Vec3| {
        let rel = world - position;
        Vec3::new(tangent.dot(rel), bitangent.dot(rel), normal.dot(rel))
    };
    let corners = rectangle_points(center, axis_u, axis_v, light.half_width, light.half_height);
    let form_factor = quad_form_factor(
        to_frame(corners[0]),
        to_frame(corners[1]),
        to_frame(corners[2]),
        to_frame(corners[3]),
    );
    if form_factor <= 0.0 || !form_factor.is_finite() {
        return None;
    }

    // Representative direction: toward the point on the rectangle nearest the
    // receiver (places the specialized specular highlight).
    let closest = closest_point_on_rect(
        position,
        center,
        axis_u,
        axis_v,
        light.half_width,
        light.half_height,
    );
    let direction = normalize_or(closest - position, normal);
    let n_dot_l = normal.dot(direction).max(1.0e-3);

    // `illuminance = radiance * F_diff / NdotL * atten` so a lobe's own trailing
    // `* NdotL` reproduces the analytic Lambert area result
    // `albedo * radiance * F_diff`.
    let radiance = to_vec3(light.color) * light.intensity;
    let illuminance = radiance * (form_factor / n_dot_l * atten);
    if !illuminance.is_finite() {
        return None;
    }

    Some(DirectLightSample {
        direction: [direction.x, direction.y, direction.z],
        illuminance: [illuminance.x, illuminance.y, illuminance.z],
        visibility: 1.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi::area_light::resolve::principled_area_contribution;
    use crate::{ShadingFrame, SurfaceSample};

    /// A unit rectangle hovering above the origin, made two-sided so its back
    /// face still lights the receiver below.
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
            two_sided: true,
        }
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn overhead_light_illuminates_from_above() {
        let sample = rect_representative_sample([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &overhead_light())
            .expect("overhead light must illuminate");
        assert!(sample.illuminance.iter().all(|c| c.is_finite() && *c > 0.0));
        assert!((sample.visibility - 1.0).abs() < 1.0e-6);
        // Representative direction points toward the emitter (straight up).
        assert!(sample.direction[2] > 0.9, "direction = {:?}", sample.direction);
    }

    #[test]
    fn degenerate_rectangle_is_skipped() {
        let mut light = overhead_light();
        light.half_width = 0.0;
        assert!(rect_representative_sample([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &light).is_none());
    }

    #[test]
    fn one_sided_back_face_is_culled() {
        let mut light = overhead_light();
        light.two_sided = false; // front face (+Z) points up/away from the receiver below.
        assert!(rect_representative_sample([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &light).is_none());
    }

    #[test]
    fn out_of_range_is_skipped() {
        let mut light = overhead_light();
        light.range = 1.0; // receiver is 2 units away.
        assert!(rect_representative_sample([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &light).is_none());
    }

    #[test]
    fn off_axis_direction_points_at_closest_point() {
        // Receiver offset along +X; the closest point on the rect is clamped to
        // its +X edge (x = +1), so the representative direction leans toward +X.
        let sample = rect_representative_sample([5.0, 0.0, 0.0], [0.0, 0.0, 1.0], &overhead_light())
            .expect("off-axis overhead light must illuminate");
        assert!(sample.direction[0] < 0.0, "should lean back toward the rect: {:?}", sample.direction);
        assert!(sample.direction[2] > 0.0);
    }

    /// The MRP illuminance is calibrated so a Lambertian reconstruction
    /// (`albedo * illuminance * NdotL`) reproduces the diffuse-only analytic
    /// area term from `principled_area_contribution` (lut = `None`, metallic 0
    /// so `diffuse_color == base_color`).
    #[test]
    fn mrp_diffuse_matches_principled_area_diffuse() {
        let base = [0.8, 0.6, 0.4];
        let normal = [0.0, 0.0, 1.0];
        let position = [0.0, 0.0, 0.0];
        let light = overhead_light();

        let sample = rect_representative_sample(position, normal, &light)
            .expect("overhead light must illuminate");
        let n_dot_l = dot(normal, sample.direction).max(0.0);
        let lambert = [
            base[0] * sample.illuminance[0] * n_dot_l,
            base[1] * sample.illuminance[1] * n_dot_l,
            base[2] * sample.illuminance[2] * n_dot_l,
        ];

        let surface = SurfaceSample {
            base_color: base,
            metallic: 0.0,
            perceptual_roughness: 0.5,
            reflectance: 0.5,
            ..SurfaceSample::default()
        };
        let frame = ShadingFrame {
            normal,
            view: normal,
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 1.0, 0.0],
        };
        let principled = principled_area_contribution(&surface, frame, position, &light, None);

        for c in 0..3 {
            assert!(
                (lambert[c] - principled[c]).abs() <= 1.0e-5,
                "channel {c}: lambert={lambert:?} principled={principled:?}"
            );
        }
    }

    #[test]
    fn non_finite_normal_does_not_emit_nan() {
        let sample =
            rect_representative_sample([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], &overhead_light());
        if let Some(sample) = sample {
            assert!(sample.direction.iter().all(|c| c.is_finite()));
            assert!(sample.illuminance.iter().all(|c| c.is_finite()));
        }
    }
}
