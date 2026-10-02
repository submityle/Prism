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
        let su = rng.next_f32();
        let sv = rng.next_f32();
        let on_light = origin.add(edge_u.scale(su)).add(edge_v.scale(sv));
        let to_light = on_light.sub(point);
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
        let cross = edge_u.cross(edge_v);
        let area = cross.length();
        if area <= 0.0 {
            return Vec3::ZERO;
        }
        let light_normal = cross.scale(1.0 / area);
        // Two-sided emitter: use the magnitude of the facing cosine.
        let cos_light = light_normal.dot(wi).abs();
        if cos_light <= 0.0 {
            return Vec3::ZERO;
        }
        let fr = bsdf.evaluate(wo, wi, normal);
        if fr.max_component() <= 0.0 {
            return Vec3::ZERO;
        }
        if occluded(point, wi, dist * (1.0 - RAY_EPS)) {
            return Vec3::ZERO;
        }
        // pdf (solid angle) = dist^2 / (cos_light * area); contribution is
        // fr * L_e * cos_surface / pdf = fr * L_e * cos_surface * cos_light * area / dist^2.
        let geom = cos_surface * cos_light * area / dist_sq;
        fr.mul(emission).scale(geom)
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
