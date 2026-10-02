//! Triangle-mesh area lights for next-event estimation.
//!
//! Emissive geometry (a [`Material`](super::integrator::Material) with a
//! non-zero `emission`) is a finite-extent light that a `BSDF`-sampled bounce
//! ray can strike directly. Relying on that alone is high variance for small or
//! bright emitters, so the integrator also connects to the emitters explicitly
//! by next-event estimation and combines both strategies with the power
//! heuristic (see [`super::mis`]).
//!
//! All emissive triangles are treated as a single compound light. A triangle is
//! chosen in proportion to its emitted power (its luminance times its area) and
//! then a point is drawn uniformly inside it. This power-weighted importance
//! sampling spends samples where the light actually radiates, so a scene with a
//! bright emitter beside a dim one resolves with far less variance than uniform
//! area selection would give. The surface density at a point on a triangle of
//! luminance `L` is therefore `L / total_power`, and its solid-angle form at a
//! shading point is `L * distance^2 / (|cos_light| * total_power)`. The same
//! closed form gives the light-sampling density of a `BSDF`-sampled ray that
//! happens to hit an emitter, which is what the multiple-importance (`MIS`)
//! weight needs; both strategies must weight identically or the estimator is
//! biased.
//!
//! Emission is two-sided (the facing cosine is taken in magnitude), matching the
//! integrator's convention that surface emission is collected on whichever side
//! a path arrives from.

use alloc::vec::Vec;

use super::bsdf::Bsdf;
use super::mis::power_heuristic;
#[cfg(test)]
use super::sampler::Rng;
use super::sampler::SampleSource;
use super::{Vec3, EPS_LEN_SQ, RAY_EPS};

/// Luma weight for the red channel, used to collapse an emitter's `RGB`
/// radiance to the scalar importance it is sampled by.
const LUMA_R: f32 = 0.2126;
/// Luma weight for the green channel (see [`LUMA_R`]).
const LUMA_G: f32 = 0.7152;
/// Luma weight for the blue channel (see [`LUMA_R`]).
const LUMA_B: f32 = 0.0722;

/// The photometric luminance of an `RGB` radiance, the scalar an emitter is
/// importance-sampled by. The weights are the standard luma coefficients and
/// sum to one, so a white emitter keeps unit importance per unit area.
#[must_use]
fn luminance(emission: Vec3) -> f32 {
    LUMA_R * emission.x + LUMA_G * emission.y + LUMA_B * emission.z
}

/// A single emissive triangle registered as part of the compound area light.
#[derive(Clone, Copy, Debug)]
struct TriangleEmitter {
    /// First vertex, used as the barycentric origin for point sampling.
    anchor: Vec3,
    /// Edge vector from [`Self::anchor`] to the second vertex.
    edge1: Vec3,
    /// Edge vector from [`Self::anchor`] to the third vertex.
    edge2: Vec3,
    /// Unit geometric normal; emission is two-sided so only its direction (not
    /// its sign) matters.
    normal: Vec3,
    /// Emitted radiance per channel.
    emission: Vec3,
    /// Sampling importance (`luminance(emission) * area`): the unnormalised
    /// probability of selecting this triangle, driving both the power-weighted
    /// emitter choice and the matching `MIS` density.
    importance: f32,
}

/// Every emissive triangle in a scene, sampled as one compound area light.
///
/// Empty when the scene has no emissive geometry, in which case all queries are
/// inert (they return zero), leaving `BSDF`-sampled hits as the only path to any
/// stray emission.
#[derive(Clone, Debug, Default)]
pub struct AreaLights {
    /// The emissive triangles, in registration order.
    emitters: Vec<TriangleEmitter>,
    /// Sum of every emitter's importance (`luminance * area`), the normaliser of
    /// the power-weighted surface density; zero when there are no emitters.
    total_importance: f32,
}

impl AreaLights {
    /// Builds the compound light from each emissive triangle's three world-space
    /// vertices and its emitted radiance.
    ///
    /// Degenerate (zero-area) triangles are skipped so they can never be chosen
    /// and can never divide by zero. The caller is expected to pass only
    /// triangles whose material actually emits.
    pub fn new(triangles: impl IntoIterator<Item = ([[f32; 3]; 3], Vec3)>) -> Self {
        let mut emitters = Vec::new();
        let mut total_importance = 0.0f32;
        for (positions, emission) in triangles {
            let anchor = Vec3::from_array(positions[0]);
            let edge1 = Vec3::from_array(positions[1]).sub(anchor);
            let edge2 = Vec3::from_array(positions[2]).sub(anchor);
            let cross = edge1.cross(edge2);
            let twice_area = cross.length();
            if twice_area <= 0.0 {
                continue;
            }
            let area = 0.5 * twice_area;
            let importance = luminance(emission) * area;
            // A triangle that is degenerate or carries no luminance can never be
            // chosen and would only risk a zero divisor, so it is dropped.
            if importance <= 0.0 {
                continue;
            }
            emitters.push(TriangleEmitter {
                anchor,
                edge1,
                edge2,
                normal: cross.scale(1.0 / twice_area),
                emission,
                importance,
            });
            total_importance += importance;
        }
        Self {
            emitters,
            total_importance,
        }
    }

    /// `true` when the scene has no sampleable emissive geometry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.emitters.is_empty() || self.total_importance <= 0.0
    }

    /// Solid-angle density this light would assign to a shading point at `from`
    /// looking toward the surface point `hit_point` with geometric normal
    /// `hit_normal` on a triangle of radiance `emission`.
    ///
    /// This is the light-sampling density that competes with `BSDF` sampling in
    /// the multiple-importance weight when a bounce ray strikes an emitter. It
    /// must mirror the power-weighted draw of [`Self::sample_direct`] exactly:
    /// the probability of landing on this point is its luminance over the total
    /// emitted power, converted from area to solid angle. Returns zero for an
    /// empty light or a grazing/degenerate configuration, in which case the
    /// caller must treat the `BSDF` strategy as the only one.
    #[must_use]
    pub fn pdf(&self, from: Vec3, hit_point: Vec3, hit_normal: Vec3, emission: Vec3) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let emitter_luminance = luminance(emission);
        if emitter_luminance <= 0.0 {
            return 0.0;
        }
        let to_light = hit_point.sub(from);
        let dist_sq = to_light.length_squared();
        if dist_sq <= EPS_LEN_SQ {
            return 0.0;
        }
        let dist = dist_sq.sqrt();
        let wi = to_light.scale(1.0 / dist);
        let cos_light = hit_normal.dot(wi).abs();
        if cos_light <= 1.0e-8 {
            return 0.0;
        }
        emitter_luminance * dist_sq / (cos_light * self.total_importance)
    }

    /// Estimates the direct lighting at a shading point from the compound area
    /// light by next-event estimation, weighted against `BSDF` sampling.
    ///
    /// Draws one surface point by power-weighted importance sampling of the
    /// emitters (luminance times area), tests visibility with `occluded`, and
    /// returns the power-heuristic-weighted
    /// contribution `f_r * L_e * cos_surface / light_pdf`. Returns [`Vec3::ZERO`]
    /// for an empty light, a sample below the horizon, an occluded connection, or
    /// a degenerate geometry term.
    pub fn sample_direct<F>(
        &self,
        point: Vec3,
        normal: Vec3,
        wo: Vec3,
        bsdf: &Bsdf,
        rng: &mut impl SampleSource,
        occluded: &F,
    ) -> Vec3
    where
        F: Fn(Vec3, Vec3, f32) -> bool,
    {
        let Some(emitter) = self.select(rng) else {
            return Vec3::ZERO;
        };
        let on_light = Self::sample_point(emitter, rng);
        let to_light = on_light.sub(point);
        let dist_sq = to_light.length_squared();
        if dist_sq <= EPS_LEN_SQ {
            return Vec3::ZERO;
        }
        let dist = dist_sq.sqrt();
        let wi = to_light.scale(1.0 / dist);
        let cos_surface = normal.dot(wi);
        let cos_light = emitter.normal.dot(wi).abs();
        if cos_surface <= 0.0 || cos_light <= 0.0 {
            return Vec3::ZERO;
        }
        let fr = bsdf.evaluate(wo, wi, normal);
        if fr.max_component() <= 0.0 || occluded(point, wi, dist * (1.0 - RAY_EPS)) {
            return Vec3::ZERO;
        }
        // The point was drawn with probability `luminance / total_power` over
        // the combined emitter area; folding the area-to-solid-angle Jacobian in
        // gives its solid-angle density. The reciprocal weights the estimator,
        // and the same density feeds the `MIS` power heuristic so the paired
        // `BSDF` strategy stays consistent.
        let emitter_luminance = luminance(emitter.emission);
        let light_pdf = emitter_luminance * dist_sq / (cos_light * self.total_importance);
        let bsdf_pdf = bsdf.pdf(wo, wi, normal);
        let weight = power_heuristic(light_pdf, bsdf_pdf);
        let inv_pdf = (cos_light * self.total_importance) / (emitter_luminance * dist_sq);
        fr.mul(emitter.emission)
            .scale(cos_surface * inv_pdf * weight)
    }

    /// Chooses an emitter in proportion to its emitted power (importance), or
    /// [`None`] when the light is empty.
    fn select(&self, rng: &mut impl SampleSource) -> Option<TriangleEmitter> {
        if self.is_empty() {
            return None;
        }
        let target = rng.next_f32() * self.total_importance;
        let mut cumulative = 0.0f32;
        for emitter in &self.emitters {
            cumulative += emitter.importance;
            if target <= cumulative {
                return Some(*emitter);
            }
        }
        // Floating-point drift can let `target` edge past the final cumulative
        // sum; fall back to the last emitter so a sample is never wasted.
        self.emitters.last().copied()
    }

    /// Draws a point uniformly inside `emitter` using the square-root barycentric
    /// warp (only `sqrt` is needed, honouring the determinism policy).
    fn sample_point(emitter: TriangleEmitter, rng: &mut impl SampleSource) -> Vec3 {
        let u0 = rng.next_f32();
        let u1 = rng.next_f32();
        let su0 = u0.sqrt();
        let b1 = u1 * su0;
        let b2 = su0 * (1.0 - u1);
        emitter
            .anchor
            .add(emitter.edge1.scale(b1))
            .add(emitter.edge2.scale(b2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit emitter: a right triangle of area `0.5` in the `y = 2` plane,
    /// facing down, emitting white.
    fn single_emitter() -> AreaLights {
        AreaLights::new([(
            [[0.0_f32, 2.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 1.0]],
            Vec3::splat(1.0),
        )])
    }

    #[test]
    fn empty_light_is_inert() {
        let lights = AreaLights::new([]);
        assert!(lights.is_empty());
        assert_eq!(
            lights.pdf(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), Vec3::ONE, Vec3::ONE),
            0.0
        );
        let occluded = |_: Vec3, _: Vec3, _: f32| false;
        let v = lights.sample_direct(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            &Bsdf::Lambert { albedo: Vec3::ONE },
            &mut Rng::seed(1),
            &occluded,
        );
        assert_eq!(v, Vec3::ZERO);
    }

    #[test]
    fn degenerate_triangle_is_skipped() {
        let lights = AreaLights::new([(
            [[0.0_f32, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            Vec3::splat(5.0),
        )]);
        assert!(lights.is_empty());
    }

    #[test]
    fn sampled_points_lie_in_the_triangle_plane() {
        let lights = single_emitter();
        let emitter = lights.select(&mut Rng::seed(7)).expect("one emitter");
        let mut rng = Rng::seed(7);
        for _ in 0..1_000 {
            let p = AreaLights::sample_point(emitter, &mut rng);
            // Plane `y = 2`.
            assert!((p.y - 2.0).abs() < 1e-5, "point off plane: {}", p.y);
            // Barycentric membership of the right triangle `x >= 0, z >= 0,
            // x + z <= 1`.
            assert!(p.x >= -1e-5 && p.z >= -1e-5 && p.x + p.z <= 1.0 + 1e-5);
        }
    }

    /// Two disjoint downward emitters in the `y = 2` plane with a 6:1 power
    /// ratio: a bright one over `x >= 0` and a dim one over `x <= 0`, each of
    /// area `0.5`. Luminance of a white emitter equals its scalar level, so the
    /// importances are `3.0 * 0.5 = 1.5` and `0.5 * 0.5 = 0.25`.
    fn mixed_emitters() -> AreaLights {
        AreaLights::new([
            (
                [[0.0_f32, 2.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 1.0]],
                Vec3::splat(3.0),
            ),
            (
                [[-1.0_f32, 2.0, 0.0], [0.0, 2.0, 0.0], [-1.0, 2.0, 1.0]],
                Vec3::splat(0.5),
            ),
        ])
    }

    #[test]
    fn emitters_are_chosen_in_proportion_to_power() {
        // The bright emitter carries importance 1.5 and the dim one 0.25, so the
        // bright half of the plane must be chosen about 1.5 / 1.75 of the time.
        // Uniform-area selection would instead split the two evenly, so this
        // frequency is the signature of power-weighted importance sampling.
        let lights = mixed_emitters();
        let mut rng = Rng::with_stream(11, 3);
        let count = 400_000u32;
        let mut bright = 0u32;
        for _ in 0..count {
            let emitter = lights.select(&mut rng).expect("two emitters");
            let point = AreaLights::sample_point(emitter, &mut rng);
            if point.x > 1.0e-6 {
                bright += 1;
            }
        }
        let fraction = f64::from(bright) / f64::from(count);
        let expected = 1.5_f64 / 1.75_f64;
        assert!(
            (fraction - expected).abs() < 1.0e-2,
            "bright fraction {fraction} should match the power ratio {expected}"
        );
    }

    #[test]
    fn power_weighted_sampling_is_unbiased_for_mixed_emitters() {
        // Power-weighted importance sampling changes which emitter is picked but
        // must leave the direct-lighting integral unbiased. Compare the
        // `MIS`-weighted estimate against an independent uniform-area reference
        // over the same two emitters; the small, distant emitters give a light
        // density far above the diffuse `BSDF` density, so the `MIS` weight is
        // effectively one and the two means must agree.
        let lights = mixed_emitters();
        let point = Vec3::ZERO;
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.0, 1.0, 0.0);
        let bsdf = Bsdf::Lambert {
            albedo: Vec3::splat(0.8),
        };
        let never = |_: Vec3, _: Vec3, _: f32| false;

        // Reference emitters: (anchor, edge1, edge2, emission), each area 0.5.
        let refs = [
            (
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::splat(3.0),
            ),
            (
                Vec3::new(-1.0, 2.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::splat(0.5),
            ),
        ];
        let light_normal = Vec3::new(0.0, 1.0, 0.0);
        let total_area = 1.0_f32;

        let count = 400_000u32;
        let mut mis_sum = 0.0f64;
        let mut ref_sum = 0.0f64;
        let mut rng_mis = Rng::with_stream(13, 1);
        let mut rng_ref = Rng::with_stream(13, 2);
        for _ in 0..count {
            mis_sum += f64::from(
                lights
                    .sample_direct(point, normal, wo, &bsdf, &mut rng_mis, &never)
                    .x,
            );

            // Reference: pick one of the two emitters uniformly (each area 0.5,
            // so this is a uniform draw over the combined area) and estimate the
            // same direct term with no `MIS` weight.
            let pick = rng_ref.next_f32();
            let (anchor, edge1, edge2, emission) = if pick < 0.5 { refs[0] } else { refs[1] };
            let u0 = rng_ref.next_f32();
            let u1 = rng_ref.next_f32();
            let su0 = u0.sqrt();
            let on_light = anchor
                .add(edge1.scale(u1 * su0))
                .add(edge2.scale(su0 * (1.0 - u1)));
            let to_light = on_light.sub(point);
            let dist_sq = to_light.length_squared();
            if dist_sq > EPS_LEN_SQ {
                let dist = dist_sq.sqrt();
                let wi = to_light.scale(1.0 / dist);
                let cos_surface = normal.dot(wi);
                let cos_light = light_normal.dot(wi).abs();
                if cos_surface > 0.0 && cos_light > 0.0 {
                    let fr = bsdf.evaluate(wo, wi, normal);
                    // `1 / p_A = total_area` because the uniform-area density is
                    // `1 / total_area` across both triangles.
                    let geom = cos_surface * cos_light / dist_sq;
                    ref_sum += f64::from(fr.mul(emission).scale(geom * total_area).x);
                }
            }
        }
        let mis_mean = mis_sum / f64::from(count);
        let ref_mean = ref_sum / f64::from(count);
        assert!(mis_mean > 0.0, "surface should receive light");
        assert!(
            (mis_mean / ref_mean - 1.0).abs() < 2.0e-2,
            "power-weighted MIS mean {mis_mean} vs uniform-area reference {ref_mean}"
        );
    }

    #[test]
    fn pdf_matches_the_analytic_solid_angle_density() {
        let lights = single_emitter();
        let from = Vec3::ZERO;
        let hit = Vec3::new(0.2, 2.0, 0.2);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let dist_sq = hit.distance_squared(from);
        let dist = dist_sq.sqrt();
        let wi = hit.sub(from).scale(1.0 / dist);
        let cos_light = normal.dot(wi).abs();
        let expected = dist_sq / (cos_light * 0.5);
        let got = lights.pdf(from, hit, normal, Vec3::splat(1.0));
        assert!(
            (got - expected).abs() <= 1e-4 * expected,
            "pdf {got} vs expected {expected}"
        );
    }

    #[test]
    fn direct_lighting_mean_matches_a_light_sampling_reference() {
        // A Lambertian point under a small downward emitter. The `MIS`-weighted
        // estimator and an unweighted light-sampling reference must share a mean,
        // confirming the weight and the area-to-solid-angle Jacobian are correct.
        let lights = single_emitter();
        let point = Vec3::ZERO;
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.0, 1.0, 0.0);
        let bsdf = Bsdf::Lambert {
            albedo: Vec3::splat(0.8),
        };
        let never = |_: Vec3, _: Vec3, _: f32| false;

        let count = 200_000u32;
        let mut mis_sum = 0.0f64;
        let mut ref_sum = 0.0f64;
        let mut rng_mis = Rng::with_stream(5, 1);
        let mut rng_ref = Rng::with_stream(5, 2);
        let anchor = Vec3::new(0.0, 2.0, 0.0);
        let edge1 = Vec3::new(1.0, 0.0, 0.0);
        let edge2 = Vec3::new(0.0, 0.0, 1.0);
        let area = 0.5;
        let light_normal = Vec3::new(0.0, 1.0, 0.0);
        for _ in 0..count {
            mis_sum += f64::from(
                lights
                    .sample_direct(point, normal, wo, &bsdf, &mut rng_mis, &never)
                    .x,
            );

            // Reference: uniform point on the triangle, converted to solid angle,
            // with no `MIS` weight. It integrates the same direct term.
            let u0 = rng_ref.next_f32();
            let u1 = rng_ref.next_f32();
            let su0 = u0.sqrt();
            let on_light = anchor
                .add(edge1.scale(u1 * su0))
                .add(edge2.scale(su0 * (1.0 - u1)));
            let to_light = on_light.sub(point);
            let dist_sq = to_light.length_squared();
            if dist_sq > EPS_LEN_SQ {
                let dist = dist_sq.sqrt();
                let wi = to_light.scale(1.0 / dist);
                let cos_surface = normal.dot(wi);
                let cos_light = light_normal.dot(wi).abs();
                if cos_surface > 0.0 && cos_light > 0.0 {
                    let fr = bsdf.evaluate(wo, wi, normal);
                    let geom = cos_surface * cos_light * area / dist_sq;
                    ref_sum += f64::from(fr.mul(Vec3::splat(1.0)).scale(geom).x);
                }
            }
        }
        let mis_mean = mis_sum / f64::from(count);
        let ref_mean = ref_sum / f64::from(count);
        assert!(mis_mean > 0.0, "surface should receive light");
        assert!(
            (mis_mean / ref_mean - 1.0).abs() < 2.0e-2,
            "MIS mean {mis_mean} vs light-sampling reference {ref_mean}"
        );
    }
}
