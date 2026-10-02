//! Direct-lighting estimation (next-event estimation) for the reference tracer.
//!
//! Next-event estimation (`NEE`) adds, at every non-specular path vertex, an
//! explicit connection to a light source. This drastically lowers variance
//! compared to hoping a `BSDF`-sampled bounce ray happens to strike a small or
//! distant emitter. Three classical light types are supported:
//!
//! - [`Light::Point`] — an isotropic point emitter with inverse-square falloff
//!   (a Dirac delta in direction, so there is no `BSDF`-sampling counterpart).
//! - [`Light::Directional`] — an infinitely distant emitter (a "sun"): a single
//!   incident direction with no distance falloff.
//! - [`Light::Quad`] — a flat rectangular area emitter sampled uniformly by
//!   area, with the area-to-solid-angle Jacobian folded into the estimator.
//!
//! Visibility is delegated to a caller-supplied occlusion predicate so this
//! module stays decoupled from the scene container: the integrator passes a
//! closure backed by the `BVH` any-hit query.

use super::bsdf::Bsdf;
use super::mis::power_heuristic;
use super::sampler::Rng;
use super::{Vec3, EPS_LEN_SQ, RAY_EPS};

/// A light source usable by next-event estimation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Light {
    /// Isotropic point light with radiant `intensity` (per steradian) and
    /// inverse-square falloff from `position`.
    Point {
        /// World-space position of the emitter.
        position: Vec3,
        /// Radiant intensity per channel (power per unit solid angle).
        intensity: Vec3,
    },
    /// Infinitely distant directional light ("sun").
    Directional {
        /// Unit direction *from the surface toward the light*.
        to_light: Vec3,
        /// Incident radiance per channel arriving along `to_light`.
        radiance: Vec3,
    },
    /// Flat rectangular area light spanned by two edge vectors from `origin`,
    /// emitting `emission` radiance from both faces.
    Quad {
        /// A corner of the rectangle.
        origin: Vec3,
        /// First edge vector from `origin`.
        edge_u: Vec3,
        /// Second edge vector from `origin`.
        edge_v: Vec3,
        /// Emitted radiance per channel (two-sided).
        emission: Vec3,
    },
}

/// A ray-light intersection used by the integrator to gather area-light
/// emission along a `BSDF`-sampled continuation ray.
///
/// Only emitters with finite extent (currently [`Light::Quad`]) can be struck;
/// [`Light::Point`] and [`Light::Directional`] are deltas with zero measure and
/// never return a hit. The hit carries the emitted radiance and the solid-angle
/// density that next-event estimation would have assigned to this direction, so
/// the integrator can multiple-importance weight the gathered emission against
/// the light-sampling strategy and avoid double counting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightHit {
    /// Distance along the (unit) ray direction to the emitter surface.
    pub distance: f32,
    /// Emitted radiance per channel at the hit.
    pub emission: Vec3,
    /// Solid-angle density of light sampling for this direction, used as the
    /// competing strategy density in the integrator's `MIS` weight.
    pub light_pdf: f32,
}

impl Light {
    /// Estimates the direct-lighting contribution reflected toward `wo` at a
    /// surface point `point` with viewer-facing shading `normal` and surface
    /// `bsdf`.
    ///
    /// `occluded(origin, direction, max_distance)` must return `true` when any
    /// geometry blocks the segment from `origin` along unit `direction` up to
    /// `max_distance`. The returned value is already divided by the sampling
    /// density, so the integrator simply adds `throughput * contribution`.
    #[must_use]
    pub fn direct<F>(
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
        // A perfectly specular surface cannot be connected to a light by light
        // sampling (its lobe is a delta), so `NEE` contributes nothing.
        if bsdf.is_specular() {
            return Vec3::ZERO;
        }
        match *self {
            Self::Point {
                position,
                intensity,
            } => Self::direct_point(point, normal, wo, bsdf, occluded, position, intensity),
            Self::Directional { to_light, radiance } => {
                Self::direct_directional(point, normal, wo, bsdf, occluded, to_light, radiance)
            }
            Self::Quad {
                origin,
                edge_u,
                edge_v,
                emission,
            } => Self::direct_quad(
                point, normal, wo, bsdf, rng, occluded, origin, edge_u, edge_v, emission,
            ),
        }
    }

    fn direct_point<F>(
        point: Vec3,
        normal: Vec3,
        wo: Vec3,
        bsdf: &Bsdf,
        occluded: &F,
        position: Vec3,
        intensity: Vec3,
    ) -> Vec3
    where
        F: Fn(Vec3, Vec3, f32) -> bool,
    {
        let to_light = position.sub(point);
        let dist_sq = to_light.length_squared();
        if dist_sq <= EPS_LEN_SQ {
            return Vec3::ZERO;
        }
        let dist = dist_sq.sqrt();
        let wi = to_light.scale(1.0 / dist);
        let cos_surface = normal.dot(wi);
        if cos_surface <= 0.0 {
            return Vec3::ZERO;
        }
        let fr = bsdf.evaluate(wo, wi, normal);
        if fr.max_component() <= 0.0 {
            return Vec3::ZERO;
        }
        if occluded(point, wi, dist * (1.0 - RAY_EPS)) {
            return Vec3::ZERO;
        }
        fr.mul(intensity).scale(cos_surface / dist_sq)
    }

    fn direct_directional<F>(
        point: Vec3,
        normal: Vec3,
        wo: Vec3,
        bsdf: &Bsdf,
        occluded: &F,
        to_light: Vec3,
        radiance: Vec3,
    ) -> Vec3
    where
        F: Fn(Vec3, Vec3, f32) -> bool,
    {
        let wi = to_light.normalize_or_zero();
        if wi.length_squared() <= 0.0 {
            return Vec3::ZERO;
        }
        let cos_surface = normal.dot(wi);
        if cos_surface <= 0.0 {
            return Vec3::ZERO;
        }
        let fr = bsdf.evaluate(wo, wi, normal);
        if fr.max_component() <= 0.0 {
            return Vec3::ZERO;
        }
        if occluded(point, wi, f32::INFINITY) {
            return Vec3::ZERO;
        }
        fr.mul(radiance).scale(cos_surface)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Area-light sampling needs the shading point, frame, surface, RNG, occlusion test and the quad's four parameters; grouping them into a struct would obscure the estimator math."
    )]
    fn direct_quad<F>(
        point: Vec3,
        normal: Vec3,
        wo: Vec3,
        bsdf: &Bsdf,
        rng: &mut Rng,
        occluded: &F,
        origin: Vec3,
        edge_u: Vec3,
        edge_v: Vec3,
        emission: Vec3,
    ) -> Vec3
    where
        F: Fn(Vec3, Vec3, f32) -> bool,
    {
        let cross = edge_u.cross(edge_v);
        let area = cross.length();
        if area <= 0.0 {
            return Vec3::ZERO;
        }
        let light_normal = cross.scale(1.0 / area);

        // Strategy A: sample a point on the emitter (low variance on broad,
        // near-diffuse lobes), weighted against the `BSDF` density by the power
        // heuristic.
        let su = rng.next_f32();
        let sv = rng.next_f32();
        let on_light = origin.add(edge_u.scale(su)).add(edge_v.scale(sv));
        let to_light = on_light.sub(point);
        let dist_sq = to_light.length_squared();
        let mut result = Vec3::ZERO;
        if dist_sq > EPS_LEN_SQ {
            let dist = dist_sq.sqrt();
            let wi = to_light.scale(1.0 / dist);
            let cos_surface = normal.dot(wi);
            // Two-sided emitter: use the magnitude of the facing cosine.
            let cos_light = light_normal.dot(wi).abs();
            if cos_surface > 0.0 && cos_light > 0.0 {
                let fr = bsdf.evaluate(wo, wi, normal);
                if fr.max_component() > 0.0 && !occluded(point, wi, dist * (1.0 - RAY_EPS)) {
                    // Solid-angle density of this strategy and the competing
                    // `BSDF` density at the same direction.
                    let light_pdf = dist_sq / (cos_light * area);
                    let bsdf_pdf = bsdf.pdf(wo, wi, normal);
                    let weight = power_heuristic(light_pdf, bsdf_pdf);
                    // fr * L_e * cos_surface / light_pdf, with the `MIS` weight.
                    let inv_pdf = (cos_light * area) / dist_sq;
                    result = result.add(fr.mul(emission).scale(cos_surface * inv_pdf * weight));
                }
            }
        }

        // The competing strategy — sampling the `BSDF` lobe and detecting
        // emitter hits along the continuation ray — lives in the integrator's
        // path loop (see `Scene::nearest_light_hit`), so this estimator only
        // performs next-event estimation (Strategy A). Specular vertices skip
        // next-event estimation entirely and reach the emitter through that
        // continuation path, which keeps mirror reflections of area lights
        // unbiased.
        result
    }

    /// Intersects the ray `point + t * wi` with the rectangular emitter spanned
    /// by `edge_u` and `edge_v` from `origin`.
    ///
    /// On a hit inside the rectangle in front of the shading point, returns the
    /// hit distance `t` and the emitter's solid-angle density
    /// `t^2 / (|cos_light| * area)` at that direction, used as the light-sampling
    /// density of the competing strategy for the `MIS` weight. `light_normal` is
    /// the unit face normal and `area` the rectangle area, both precomputed by
    /// the caller.
    fn quad_hit(
        point: Vec3,
        wi: Vec3,
        origin: Vec3,
        edge_u: Vec3,
        edge_v: Vec3,
        light_normal: Vec3,
        area: f32,
    ) -> Option<(f32, f32)> {
        let denom = wi.dot(light_normal);
        let cos_light = denom.abs();
        if cos_light <= 1.0e-8 {
            return None;
        }
        let t = origin.sub(point).dot(light_normal) / denom;
        if t <= 0.0 {
            return None;
        }
        let hit = point.add(wi.scale(t));
        let d = hit.sub(origin);
        // Solve `d = a * edge_u + b * edge_v` for possibly non-orthogonal edges.
        let uu = edge_u.dot(edge_u);
        let vv = edge_v.dot(edge_v);
        let uv = edge_u.dot(edge_v);
        let du = d.dot(edge_u);
        let dv = d.dot(edge_v);
        let det = uu * vv - uv * uv;
        if det <= 0.0 {
            return None;
        }
        let a = (du * vv - dv * uv) / det;
        let b = (dv * uu - du * uv) / det;
        if !(0.0..=1.0).contains(&a) || !(0.0..=1.0).contains(&b) {
            return None;
        }
        Some((t, (t * t) / (cos_light * area)))
    }

    /// Intersects a world-space ray `origin + t * direction` (with `direction`
    /// unit length) against this light's emitting surface.
    ///
    /// Returns the nearest forward hit as a [`LightHit`], or [`None`] for delta
    /// emitters and for rays that miss the surface. The integrator uses this to
    /// let `BSDF`-sampled continuation rays — including perfectly specular ones
    /// that next-event estimation cannot connect — pick up area-light emission.
    #[must_use]
    pub fn intersect(&self, origin: Vec3, direction: Vec3) -> Option<LightHit> {
        match *self {
            Self::Point { .. } | Self::Directional { .. } => None,
            Self::Quad {
                origin: quad_origin,
                edge_u,
                edge_v,
                emission,
            } => {
                let cross = edge_u.cross(edge_v);
                let area = cross.length();
                if area <= 0.0 {
                    return None;
                }
                let light_normal = cross.scale(1.0 / area);
                let (distance, light_pdf) = Self::quad_hit(
                    origin,
                    direction,
                    quad_origin,
                    edge_u,
                    edge_v,
                    light_normal,
                    area,
                )?;
                Some(LightHit {
                    distance,
                    emission,
                    light_pdf,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    fn never_occluded(_: Vec3, _: Vec3, _: f32) -> bool {
        false
    }

    fn always_occluded(_: Vec3, _: Vec3, _: f32) -> bool {
        true
    }

    #[test]
    fn point_light_inverse_square() {
        // Lambert albedo 1, light straight up at height h; expected irradiance
        // reflected = (albedo/pi) * intensity * cos / dist^2 with cos = 1.
        let bsdf = Bsdf::Lambert { albedo: Vec3::ONE };
        let mut rng = Rng::seed(1);
        let h = 2.0f32;
        let light = Light::Point {
            position: Vec3::new(0.0, h, 0.0),
            intensity: Vec3::splat(4.0),
        };
        let v = light.direct(Vec3::ZERO, N, N, &bsdf, &mut rng, &never_occluded);
        let expected = super::super::INV_PI * 4.0 * 1.0 / (h * h);
        assert!(
            (v.x - expected).abs() < 1e-6,
            "got {} expected {expected}",
            v.x
        );
    }

    #[test]
    fn point_light_occlusion_blocks() {
        let bsdf = Bsdf::Lambert { albedo: Vec3::ONE };
        let mut rng = Rng::seed(1);
        let light = Light::Point {
            position: Vec3::new(0.0, 2.0, 0.0),
            intensity: Vec3::splat(4.0),
        };
        let v = light.direct(Vec3::ZERO, N, N, &bsdf, &mut rng, &always_occluded);
        assert_eq!(v, Vec3::ZERO);
    }

    #[test]
    fn point_light_below_surface_is_dark() {
        let bsdf = Bsdf::Lambert { albedo: Vec3::ONE };
        let mut rng = Rng::seed(1);
        let light = Light::Point {
            position: Vec3::new(0.0, -2.0, 0.0),
            intensity: Vec3::splat(4.0),
        };
        let v = light.direct(Vec3::ZERO, N, N, &bsdf, &mut rng, &never_occluded);
        assert_eq!(v, Vec3::ZERO);
    }

    #[test]
    fn directional_light_no_falloff() {
        let bsdf = Bsdf::Lambert { albedo: Vec3::ONE };
        let mut rng = Rng::seed(1);
        let light = Light::Directional {
            to_light: Vec3::new(0.0, 1.0, 0.0),
            radiance: Vec3::splat(3.0),
        };
        let v = light.direct(Vec3::ZERO, N, N, &bsdf, &mut rng, &never_occluded);
        let expected = super::super::INV_PI * 3.0;
        assert!((v.x - expected).abs() < 1e-6);
    }

    #[test]
    fn specular_surface_has_no_nee() {
        let bsdf = Bsdf::Mirror {
            reflectance: Vec3::ONE,
        };
        let mut rng = Rng::seed(1);
        let light = Light::Point {
            position: Vec3::new(0.0, 2.0, 0.0),
            intensity: Vec3::splat(4.0),
        };
        let v = light.direct(Vec3::ZERO, N, N, &bsdf, &mut rng, &never_occluded);
        assert_eq!(v, Vec3::ZERO);
    }

    #[test]
    fn quad_light_positive_and_scales_linearly() {
        let bsdf = Bsdf::Lambert { albedo: Vec3::ONE };
        let quad1 = Light::Quad {
            origin: Vec3::new(-0.5, 2.0, -0.5),
            edge_u: Vec3::new(1.0, 0.0, 0.0),
            edge_v: Vec3::new(0.0, 0.0, 1.0),
            emission: Vec3::splat(1.0),
        };
        let quad2 = Light::Quad {
            origin: Vec3::new(-0.5, 2.0, -0.5),
            edge_u: Vec3::new(1.0, 0.0, 0.0),
            edge_v: Vec3::new(0.0, 0.0, 1.0),
            emission: Vec3::splat(2.0),
        };
        let mut sum1 = 0.0f64;
        let mut sum2 = 0.0f64;
        let count = 20_000u32;
        let mut rng = Rng::seed(55);
        for _ in 0..count {
            sum1 += quad1
                .direct(Vec3::ZERO, N, N, &bsdf, &mut rng, &never_occluded)
                .x as f64;
            sum2 += quad2
                .direct(Vec3::ZERO, N, N, &bsdf, &mut rng, &never_occluded)
                .x as f64;
        }
        let mean1 = sum1 / f64::from(count);
        let mean2 = sum2 / f64::from(count);
        assert!(mean1 > 0.0, "quad light should illuminate the point");
        // Doubling emission must double the estimate (within Monte Carlo noise).
        assert!(
            (mean2 / mean1 - 2.0).abs() < 1e-2,
            "ratio {}",
            mean2 / mean1
        );
    }

    #[test]
    fn quad_light_occlusion_blocks() {
        let bsdf = Bsdf::Lambert { albedo: Vec3::ONE };
        let mut rng = Rng::seed(55);
        let quad = Light::Quad {
            origin: Vec3::new(-0.5, 2.0, -0.5),
            edge_u: Vec3::new(1.0, 0.0, 0.0),
            edge_v: Vec3::new(0.0, 0.0, 1.0),
            emission: Vec3::splat(1.0),
        };
        let v = quad.direct(Vec3::ZERO, N, N, &bsdf, &mut rng, &always_occluded);
        assert_eq!(v, Vec3::ZERO);
    }

    /// Reference direct-lighting estimator that only samples the emitter, i.e.
    /// the single-strategy estimator the `MIS` path must agree with in the mean.
    ///
    /// Returns `fr * L_e * cos_surface * cos_light * area / dist^2` for one point
    /// drawn uniformly on the quad, with no `BSDF`-sampling strategy and no `MIS`
    /// weight. Averaged over many samples it converges to the same integral as
    /// `Light::direct` for a quad, so the two means must match to within Monte
    /// Carlo noise if the `MIS` combination is unbiased.
    fn light_sample_only(
        point: Vec3,
        normal: Vec3,
        wo: Vec3,
        bsdf: &Bsdf,
        origin: Vec3,
        edge_u: Vec3,
        edge_v: Vec3,
        emission: Vec3,
        rng: &mut Rng,
    ) -> Vec3 {
        let cross = edge_u.cross(edge_v);
        let area = cross.length();
        if area <= 0.0 {
            return Vec3::ZERO;
        }
        let light_normal = cross.scale(1.0 / area);
        let on_light = origin
            .add(edge_u.scale(rng.next_f32()))
            .add(edge_v.scale(rng.next_f32()));
        let to_light = on_light.sub(point);
        let dist_sq = to_light.length_squared();
        if dist_sq <= EPS_LEN_SQ {
            return Vec3::ZERO;
        }
        let dist = dist_sq.sqrt();
        let wi = to_light.scale(1.0 / dist);
        let cos_surface = normal.dot(wi);
        let cos_light = light_normal.dot(wi).abs();
        if cos_surface <= 0.0 || cos_light <= 0.0 {
            return Vec3::ZERO;
        }
        let fr = bsdf.evaluate(wo, wi, normal);
        let geom = cos_surface * cos_light * area / dist_sq;
        fr.mul(emission).scale(geom)
    }

    /// `BSDF`-sampling half of the area-light `MIS` pair, mirroring what the
    /// path integrator does on a continuation ray: draw a lobe sample, intersect
    /// the emitter with [`Light::intersect`], and weight the hit against the
    /// light-sampling density by the power heuristic. Combined with
    /// [`Light::direct`] (the emitter-sampling half) the two means must match the
    /// single-strategy reference, proving the split is unbiased.
    fn bsdf_sample_only(
        point: Vec3,
        normal: Vec3,
        wo: Vec3,
        bsdf: &Bsdf,
        light: &Light,
        rng: &mut Rng,
    ) -> Vec3 {
        let Some(sample) = bsdf.sample(wo, normal, rng) else {
            return Vec3::ZERO;
        };
        if sample.pdf <= 0.0 {
            return Vec3::ZERO;
        }
        let wi = sample.direction;
        let cos_surface = normal.dot(wi);
        if cos_surface <= 0.0 {
            return Vec3::ZERO;
        }
        let Some(hit) = light.intersect(point, wi) else {
            return Vec3::ZERO;
        };
        if hit.light_pdf <= 0.0 {
            return Vec3::ZERO;
        }
        let weight = power_heuristic(sample.pdf, hit.light_pdf);
        sample
            .value
            .mul(hit.emission)
            .scale(cos_surface / sample.pdf * weight)
    }

    #[test]
    fn mis_quad_matches_light_sampling_in_the_mean() {
        // A glossy conductor exercises both `MIS` strategies: the emitter sample
        // and the `BSDF` lobe sample. The combined estimator must converge to the
        // same radiance as the pure light-sampling reference above, proving the
        // power-heuristic combination adds no bias.
        let bsdf = Bsdf::GgxConductor {
            reflectance: Vec3::splat(0.95),
            roughness: 0.25,
        };
        let wo = Vec3::new(0.0, 1.0, 0.0);
        let origin = Vec3::new(-1.0, 2.0, -1.0);
        let edge_u = Vec3::new(2.0, 0.0, 0.0);
        let edge_v = Vec3::new(0.0, 0.0, 2.0);
        let emission = Vec3::splat(1.0);
        let quad = Light::Quad {
            origin,
            edge_u,
            edge_v,
            emission,
        };

        let count = 200_000u32;
        let mut mis_sum = 0.0f64;
        let mut ref_sum = 0.0f64;
        let mut rng_mis = Rng::seed(900);
        let mut rng_ref = Rng::seed(901);
        for _ in 0..count {
            let strategy_a = quad.direct(Vec3::ZERO, N, wo, &bsdf, &mut rng_mis, &never_occluded);
            let strategy_b = bsdf_sample_only(Vec3::ZERO, N, wo, &bsdf, &quad, &mut rng_mis);
            mis_sum += f64::from(strategy_a.add(strategy_b).x);
            ref_sum += f64::from(
                light_sample_only(
                    Vec3::ZERO,
                    N,
                    wo,
                    &bsdf,
                    origin,
                    edge_u,
                    edge_v,
                    emission,
                    &mut rng_ref,
                )
                .x,
            );
        }
        let mis_mean = mis_sum / f64::from(count);
        let ref_mean = ref_sum / f64::from(count);
        assert!(mis_mean > 0.0, "glossy surface should receive light");
        // Relative agreement within Monte Carlo noise; glossy light sampling is
        // noisy, so the tolerance is loose but still catches any systematic bias.
        assert!(
            (mis_mean / ref_mean - 1.0).abs() < 4.0e-2,
            "MIS mean {mis_mean} vs light-sampling reference {ref_mean}"
        );
    }

    #[test]
    fn degenerate_quad_is_dark() {
        let bsdf = Bsdf::Lambert { albedo: Vec3::ONE };
        let mut rng = Rng::seed(55);
        let quad = Light::Quad {
            origin: Vec3::new(-0.5, 2.0, -0.5),
            edge_u: Vec3::ZERO,
            edge_v: Vec3::ZERO,
            emission: Vec3::splat(1.0),
        };
        let v = quad.direct(Vec3::ZERO, N, N, &bsdf, &mut rng, &never_occluded);
        assert_eq!(v, Vec3::ZERO);
    }
}
