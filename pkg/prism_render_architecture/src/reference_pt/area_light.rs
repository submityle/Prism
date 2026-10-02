//! Triangle-mesh area lights for next-event estimation.
//!
//! Emissive geometry (a [`Material`](super::integrator::Material) with a
//! non-zero `emission`) is a finite-extent light that a `BSDF`-sampled bounce
//! ray can strike directly. Relying on that alone is high variance for small or
//! bright emitters, so the integrator also connects to the emitters explicitly
//! by next-event estimation and combines both strategies with the power
//! heuristic (see [`super::mis`]).
//!
//! All emissive triangles are treated as a single compound light: a surface
//! point is drawn by choosing a triangle in proportion to its area and then a
//! point uniformly inside it, which is exactly a uniform draw over the combined
//! area. The surface density is therefore the constant `1 / total_area`, and its
//! solid-angle form at a shading point is `distance^2 / (|cos_light| * area)`,
//! independent of which triangle was chosen. The same closed form gives the
//! light-sampling density of a `BSDF`-sampled ray that happens to hit an
//! emitter, which is what the multiple-importance weight needs.
//!
//! Emission is two-sided (the facing cosine is taken in magnitude), matching the
//! integrator's convention that surface emission is collected on whichever side
//! a path arrives from.

use alloc::vec::Vec;

use super::bsdf::Bsdf;
use super::mis::power_heuristic;
use super::sampler::Rng;
use super::{Vec3, EPS_LEN_SQ, RAY_EPS};

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
    /// Triangle surface area in world units.
    area: f32,
    /// Emitted radiance per channel.
    emission: Vec3,
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
    /// Sum of all emitter areas, i.e. the normalisation of the uniform surface
    /// density; zero when there are no emitters.
    total_area: f32,
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
        let mut total_area = 0.0f32;
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
            emitters.push(TriangleEmitter {
                anchor,
                edge1,
                edge2,
                normal: cross.scale(1.0 / twice_area),
                area,
                emission,
            });
            total_area += area;
        }
        Self {
            emitters,
            total_area,
        }
    }

    /// `true` when the scene has no sampleable emissive geometry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.emitters.is_empty() || self.total_area <= 0.0
    }

    /// Solid-angle density this light would assign to a shading point at `from`
    /// looking toward the surface point `hit_point` with geometric normal
    /// `hit_normal`.
    ///
    /// This is the light-sampling density that competes with `BSDF` sampling in
    /// the multiple-importance weight when a bounce ray strikes an emitter.
    /// Returns zero for an empty light or a grazing/degenerate configuration, in
    /// which case the caller must treat the `BSDF` strategy as the only one.
    #[must_use]
    pub fn pdf(&self, from: Vec3, hit_point: Vec3, hit_normal: Vec3) -> f32 {
        if self.is_empty() {
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
        dist_sq / (cos_light * self.total_area)
    }

    /// Estimates the direct lighting at a shading point from the compound area
    /// light by next-event estimation, weighted against `BSDF` sampling.
    ///
    /// Draws one surface point uniformly over the combined emitter area, tests
    /// visibility with `occluded`, and returns the power-heuristic-weighted
    /// contribution `f_r * L_e * cos_surface / light_pdf`. Returns [`Vec3::ZERO`]
    /// for an empty light, a sample below the horizon, an occluded connection, or
    /// a degenerate geometry term.
    pub fn sample_direct<F>(
        &self,
        point: Vec3,
        normal: Vec3,
        wo: Vec3,
        bsdf: &Bsdf,
        rng: &mut Rng,
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
        // Combined surface density is `1 / total_area`, so the solid-angle
        // density folds the area-to-solid-angle Jacobian over the whole light.
        let light_pdf = dist_sq / (cos_light * self.total_area);
        let bsdf_pdf = bsdf.pdf(wo, wi, normal);
        let weight = power_heuristic(light_pdf, bsdf_pdf);
        let inv_pdf = (cos_light * self.total_area) / dist_sq;
        fr.mul(emitter.emission)
            .scale(cos_surface * inv_pdf * weight)
    }

    /// Chooses an emitter in proportion to its area, or [`None`] when the light
    /// is empty.
    fn select(&self, rng: &mut Rng) -> Option<TriangleEmitter> {
        if self.is_empty() {
            return None;
        }
        let target = rng.next_f32() * self.total_area;
        let mut cumulative = 0.0f32;
        for emitter in &self.emitters {
            cumulative += emitter.area;
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
    fn sample_point(emitter: TriangleEmitter, rng: &mut Rng) -> Vec3 {
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
            lights.pdf(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), Vec3::ONE),
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
        let got = lights.pdf(from, hit, normal);
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
