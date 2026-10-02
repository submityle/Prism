//! The path-tracing loop and the scene it traces.
//!
//! This is the heart of the offline reference renderer: an unbiased
//! `Monte Carlo` path integrator that estimates the radiance returning along a
//! camera ray by random-walking light transport through the scene. Each vertex
//! of the walk combines two classical variance-reduction techniques:
//!
//! - **next-event estimation** (`NEE`): an explicit, shadow-tested connection
//!   to every analytic light (see [`crate::reference_pt::estimator`]), which
//!   captures direct illumination cheaply at non-specular vertices.
//! - **`BSDF` importance sampling**: the surface scatters the path into a new
//!   direction drawn from its own lobe (see [`crate::reference_pt::bsdf`]),
//!   which carries indirect light and specular transport.
//!
//! The walk is terminated without bias by Russian roulette once the path
//! throughput has decayed, and hard-capped at [`PathIntegrator::max_depth`]
//! bounces so a perfectly reflective (albedo-one) environment cannot loop
//! forever. Geometry queries are delegated to the software `BVH` in
//! [`crate::ray_scene`]; this module owns no acceleration structure of its own.
//!
//! Emission accounting keeps the two light systems disjoint to avoid double
//! counting: analytic lights in [`Scene`] are gathered *only* through `NEE`,
//! while surface emission ([`Material::emission`]) is gathered *only* when a
//! `BSDF`-sampled path (or the primary camera ray) lands on the emitter.

use alloc::vec::Vec;

use super::bsdf::Bsdf;
use super::estimator::Light;
use super::sampler::Rng;
use super::{Vec3, RAY_EPS};
use crate::ray_scene::traversal::Ray;
use crate::ray_scene::triangle_mesh::TriangleMeshBvh;

/// The maximum Russian-roulette survival probability.
///
/// Clamping below one keeps a bright path from surviving with certainty (which
/// would never terminate in a fully reflective environment) while leaving the
/// estimator unbiased: survivors are reweighted by the reciprocal probability.
const RR_MAX_SURVIVAL: f32 = 0.95;

/// A surface material: how it scatters light plus any light it emits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// The scattering lobe (diffuse or specular) at the surface.
    pub bsdf: Bsdf,
    /// Emitted radiance per channel; [`Vec3::ZERO`] for a non-emitter.
    pub emission: Vec3,
}

impl Material {
    /// A non-emitting material with the given scattering lobe.
    #[must_use]
    pub const fn new(bsdf: Bsdf) -> Self {
        Self {
            bsdf,
            emission: Vec3::ZERO,
        }
    }

    /// A material that both scatters (via `bsdf`) and emits `emission`.
    #[must_use]
    pub const fn emissive(bsdf: Bsdf, emission: Vec3) -> Self {
        Self { bsdf, emission }
    }
}

/// The error returned when a [`Scene`] is built with inconsistent inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneError {
    /// The per-triangle material list length does not match the mesh triangle
    /// count, so some triangle would have no (or an ambiguous) material.
    MaterialCountMismatch {
        /// Number of materials supplied.
        materials: usize,
        /// Number of triangles in the geometry.
        triangles: usize,
    },
}

impl core::fmt::Display for SceneError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::MaterialCountMismatch {
                materials,
                triangles,
            } => write!(
                f,
                "material count {materials} does not match triangle count {triangles}"
            ),
        }
    }
}

/// A traced scene: geometry, a per-triangle material table, analytic lights, and
/// a constant environment radiance for rays that escape to infinity.
#[derive(Clone, Debug)]
pub struct Scene {
    geometry: TriangleMeshBvh,
    materials: Vec<Material>,
    lights: Vec<Light>,
    environment: Vec3,
}

impl Scene {
    /// Builds a scene, binding one [`Material`] to each triangle (indexed by the
    /// mesh's original triangle id) and a constant `environment` radiance.
    ///
    /// # Errors
    ///
    /// Returns [`SceneError::MaterialCountMismatch`] when `materials.len()` is
    /// not exactly the geometry's triangle count.
    pub fn new(
        geometry: TriangleMeshBvh,
        materials: Vec<Material>,
        lights: Vec<Light>,
        environment: Vec3,
    ) -> Result<Self, SceneError> {
        let triangles = geometry.mesh().triangle_count();
        if materials.len() != triangles {
            return Err(SceneError::MaterialCountMismatch {
                materials: materials.len(),
                triangles,
            });
        }
        Ok(Self {
            geometry,
            materials,
            lights,
            environment,
        })
    }

    /// The constant radiance returned by rays that leave the scene.
    #[must_use]
    pub fn environment(&self) -> Vec3 {
        self.environment
    }

    /// Finds the closest surface hit by `ray`, resolved into a shading record.
    ///
    /// Returns [`None`] when the ray escapes the geometry (an environment miss).
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<SurfaceInteraction> {
        let hit = self.geometry.closest_hit(ray)?;
        Some(SurfaceInteraction {
            position: Vec3::from_array(hit.position),
            normal: Vec3::from_array(hit.normal).normalize_or_zero(),
            front_face: hit.front_face,
            material: hit.triangle as usize,
        })
    }

    /// `true` when any geometry blocks the segment from `origin` along unit
    /// `direction` up to `max_distance` (the `NEE` visibility predicate).
    #[must_use]
    fn occluded(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> bool {
        let ray = Ray::new(origin.to_array(), direction.to_array(), 0.0, max_distance);
        self.geometry.any_hit(&ray)
    }
}

/// A resolved surface interaction: everything a path vertex needs to shade.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceInteraction {
    /// World-space hit position.
    pub position: Vec3,
    /// Unit shading normal as reported by the mesh (not yet viewer-oriented).
    pub normal: Vec3,
    /// `true` when the primary side (front face) of the triangle was struck.
    pub front_face: bool,
    /// Index of the hit triangle's material in [`Scene`]'s material table.
    pub material: usize,
}

/// The unbiased `Monte Carlo` path integrator.
///
/// [`PathIntegrator::radiance`] estimates the radiance arriving along one camera
/// ray with a single random walk; averaging many walks (with distinct `RNG`
/// streams) over the same ray converges to the true value. The integrator is
/// stateless apart from its termination parameters, so it is cheap to copy.
#[derive(Clone, Copy, Debug)]
pub struct PathIntegrator {
    /// Hard cap on the number of surface bounces along a single path.
    pub max_depth: u32,
    /// Bounce depth at which Russian-roulette termination begins; before this
    /// depth every path survives so near-field transport stays low-variance.
    pub rr_start_depth: u32,
}

impl Default for PathIntegrator {
    fn default() -> Self {
        Self {
            max_depth: 16,
            rr_start_depth: 4,
        }
    }
}

impl PathIntegrator {
    /// Builds an integrator with an explicit bounce cap and roulette start.
    #[must_use]
    pub const fn new(max_depth: u32, rr_start_depth: u32) -> Self {
        Self {
            max_depth,
            rr_start_depth,
        }
    }

    /// Estimates the radiance returning along `primary` through `scene` using a
    /// single random walk driven by `rng`.
    ///
    /// The returned value is always finite: degenerate rays, grazing samples,
    /// and empty scenes terminate the walk cleanly rather than panicking or
    /// producing `NaN`.
    #[must_use]
    pub fn radiance(&self, scene: &Scene, primary: Ray, rng: &mut Rng) -> Vec3 {
        let mut radiance = Vec3::ZERO;
        let mut throughput = Vec3::ONE;
        let mut ray = primary;
        let mut depth: u32 = 0;

        loop {
            let Some(isect) = scene.intersect(&ray) else {
                radiance = radiance.add(throughput.mul(scene.environment));
                break;
            };
            let material = scene.materials[isect.material];

            // Surface emission is gathered on the arriving path (never via `NEE`).
            radiance = radiance.add(throughput.mul(material.emission));

            // View direction, pointing away from the surface toward the sensor.
            let wo = Vec3::from_array(ray.direction())
                .normalize_or_zero()
                .negate();
            // Shading normal oriented into the viewer's hemisphere, used for
            // the hemispherical operations (`NEE` and ray-origin offsets).
            let shading_normal = isect.normal.faced_toward(wo);

            // Direct lighting by next-event estimation (skipped on specular lobes).
            if !material.bsdf.is_specular() {
                let shadow_origin = isect.position.add(shading_normal.scale(RAY_EPS));
                let occluded = |origin: Vec3, direction: Vec3, max_distance: f32| {
                    scene.occluded(origin, direction, max_distance)
                };
                for light in &scene.lights {
                    let contribution = light.direct(
                        shadow_origin,
                        shading_normal,
                        wo,
                        &material.bsdf,
                        rng,
                        &occluded,
                    );
                    radiance = radiance.add(throughput.mul(contribution));
                }
            }

            // Stop before exceeding the configured bounce budget.
            if depth + 1 >= self.max_depth {
                break;
            }

            // Extend the path by importance-sampling the surface lobe. The raw
            // geometric normal is handed to `sample`: reflective lobes orient it
            // internally, while the dielectric needs the unoriented side to tell
            // whether the ray is entering or leaving the medium.
            let Some(sample) = material.bsdf.sample(wo, isect.normal, rng) else {
                break;
            };
            if sample.pdf <= 0.0 {
                break;
            }
            // Transmission puts `wi` on the far side of the geometric normal, so
            // the cosine is taken in absolute value (reflection keeps the sign).
            let cos_i = isect.normal.dot(sample.direction);
            let cos_abs = cos_i.abs();
            if cos_abs <= 0.0 {
                break;
            }
            throughput = throughput.mul(sample.value).scale(cos_abs / sample.pdf);
            if !throughput.is_finite() {
                break;
            }

            depth += 1;

            // Russian roulette: terminate dim paths without bias.
            if depth >= self.rr_start_depth {
                let survive = throughput.max_component().clamp(0.0, RR_MAX_SURVIVAL);
                if survive <= 0.0 || rng.next_f32() >= survive {
                    break;
                }
                throughput = throughput.scale(1.0 / survive);
            }

            // Offset the bounce origin along the geometric normal, on whichever
            // side the sampled direction leaves (reflection stays above, a
            // transmitted ray drops below), to avoid self-intersection.
            let offset_sign = if isect.normal.dot(sample.direction) >= 0.0 {
                1.0
            } else {
                -1.0
            };
            let next_origin = isect
                .position
                .add(isect.normal.scale(RAY_EPS * offset_sign));
            ray = Ray::new(
                next_origin.to_array(),
                sample.direction.to_array(),
                0.0,
                f32::INFINITY,
            );
        }

        radiance
    }
}

#[cfg(test)]
mod tests {
    use super::super::sampler::Rng;
    use super::*;

    /// A single large upward-facing triangle in the `y = 0` plane.
    fn ground_plane() -> TriangleMeshBvh {
        let positions = alloc::vec![
            [-10.0_f32, 0.0, -10.0],
            [10.0, 0.0, -10.0],
            [0.0, 0.0, 10.0],
        ];
        let indices = alloc::vec![[0u32, 1, 2]];
        let mesh = crate::ray_scene::triangle_mesh::TriangleMesh::new(
            positions,
            alloc::vec![],
            alloc::vec![],
            indices,
        )
        .expect("valid ground triangle");
        TriangleMeshBvh::build(mesh)
    }

    /// A right-angle corner: a floor plus a wall, so bounces can recurse.
    fn corner() -> TriangleMeshBvh {
        let positions = alloc::vec![
            // Floor (y = 0).
            [-5.0_f32, 0.0, -5.0],
            [5.0, 0.0, -5.0],
            [0.0, 0.0, 5.0],
            // Wall (x = -5).
            [-5.0, 0.0, -5.0],
            [-5.0, 0.0, 5.0],
            [-5.0, 10.0, 0.0],
        ];
        let indices = alloc::vec![[0u32, 1, 2], [3, 4, 5]];
        let mesh = crate::ray_scene::triangle_mesh::TriangleMesh::new(
            positions,
            alloc::vec![],
            alloc::vec![],
            indices,
        )
        .expect("valid corner");
        TriangleMeshBvh::build(mesh)
    }

    fn empty_geometry() -> TriangleMeshBvh {
        let mesh = crate::ray_scene::triangle_mesh::TriangleMesh::new(
            alloc::vec![],
            alloc::vec![],
            alloc::vec![],
            alloc::vec![],
        )
        .expect("empty mesh is valid");
        TriangleMeshBvh::build(mesh)
    }

    /// Camera ray straight down onto the origin.
    fn down_ray() -> Ray {
        Ray::new([0.0, 2.0, 0.0], [0.0, -1.0, 0.0], 0.0, f32::INFINITY)
    }

    #[test]
    fn material_count_mismatch_is_rejected() {
        let geometry = ground_plane();
        let err = Scene::new(geometry, alloc::vec![], alloc::vec![], Vec3::ONE)
            .expect_err("empty material table must be rejected for a 1-triangle mesh");
        assert_eq!(
            err,
            SceneError::MaterialCountMismatch {
                materials: 0,
                triangles: 1,
            }
        );
    }

    #[test]
    fn white_furnace_single_bounce_returns_unit() {
        // Albedo-one surface under unit environment radiance must reflect unit
        // radiance back (the classic white-furnace energy-conservation test).
        let geometry = ground_plane();
        let materials = alloc::vec![Material::new(Bsdf::Lambert { albedo: Vec3::ONE })];
        let scene =
            Scene::new(geometry, materials, alloc::vec![], Vec3::ONE).expect("white furnace scene");
        let integrator = PathIntegrator::new(8, 5);
        let mut sum = 0.0f64;
        let count = 4_000u32;
        for s in 0..count {
            let mut rng = Rng::with_stream(1, u64::from(s) + 1);
            let value = integrator.radiance(&scene, down_ray(), &mut rng);
            assert!(value.is_finite());
            sum += f64::from(value.x);
        }
        let mean = sum / f64::from(count);
        assert!(
            (mean - 1.0).abs() < 1e-3,
            "white furnace mean {mean} should converge to 1"
        );
    }

    #[test]
    fn multi_bounce_furnace_conserves_energy() {
        // A reflective corner under unit environment still returns unit radiance
        // in the mean, exercising recursive bounces and Russian roulette.
        let geometry = corner();
        let materials = alloc::vec![
            Material::new(Bsdf::Lambert { albedo: Vec3::ONE }),
            Material::new(Bsdf::Lambert { albedo: Vec3::ONE }),
        ];
        let scene =
            Scene::new(geometry, materials, alloc::vec![], Vec3::ONE).expect("corner furnace");
        let integrator = PathIntegrator::new(32, 4);
        let mut sum = 0.0f64;
        let count = 40_000u32;
        for s in 0..count {
            let mut rng = Rng::with_stream(7, u64::from(s) + 1);
            let value = integrator.radiance(&scene, down_ray(), &mut rng);
            assert!(value.is_finite());
            sum += f64::from(value.x);
        }
        let mean = sum / f64::from(count);
        assert!(
            (mean - 1.0).abs() < 3e-2,
            "multi-bounce furnace mean {mean} should stay near 1"
        );
    }

    #[test]
    fn darker_albedo_reflects_less_than_environment() {
        // With albedo < 1 the single-bounce reflected radiance must be below the
        // unit environment (strict energy loss, averaged over samples).
        let geometry = ground_plane();
        let materials = alloc::vec![Material::new(Bsdf::Lambert {
            albedo: Vec3::splat(0.5),
        })];
        let scene =
            Scene::new(geometry, materials, alloc::vec![], Vec3::ONE).expect("gray furnace");
        let integrator = PathIntegrator::new(8, 5);
        let mut sum = 0.0f64;
        let count = 4_000u32;
        for s in 0..count {
            let mut rng = Rng::with_stream(3, u64::from(s) + 1);
            sum += f64::from(integrator.radiance(&scene, down_ray(), &mut rng).x);
        }
        let mean = sum / f64::from(count);
        assert!(
            (mean - 0.5).abs() < 1e-3,
            "gray furnace mean {mean} should be ~0.5"
        );
        assert!(
            mean < 1.0,
            "reflected radiance must be below the environment"
        );
    }

    #[test]
    fn emissive_surface_is_directly_visible() {
        // A primary ray onto an emitter returns its emission plus the reflected
        // environment term; it must at least exceed the emission floor.
        let geometry = ground_plane();
        let materials = alloc::vec![Material::emissive(
            Bsdf::Lambert { albedo: Vec3::ZERO },
            Vec3::splat(3.0),
        )];
        let scene =
            Scene::new(geometry, materials, alloc::vec![], Vec3::ZERO).expect("emitter scene");
        let integrator = PathIntegrator::new(8, 5);
        let mut rng = Rng::seed(123);
        let value = integrator.radiance(&scene, down_ray(), &mut rng);
        assert!(
            (value.x - 3.0).abs() < 1e-5,
            "emission should pass through, got {}",
            value.x
        );
    }

    #[test]
    fn direct_lighting_from_point_source() {
        // A black (non-reflecting) floor lit by a point light isolates the `NEE`
        // direct term; it must equal the analytic reflected irradiance.
        let geometry = ground_plane();
        let materials = alloc::vec![Material::new(Bsdf::Lambert { albedo: Vec3::ONE })];
        let light = Light::Point {
            position: Vec3::new(0.0, 2.0, 0.0),
            intensity: Vec3::splat(4.0),
        };
        let scene =
            Scene::new(geometry, materials, alloc::vec![light], Vec3::ZERO).expect("lit scene");
        // One bounce only, so no indirect term pollutes the direct estimate.
        let integrator = PathIntegrator::new(1, 1);
        let mut rng = Rng::seed(55);
        let value = integrator.radiance(&scene, down_ray(), &mut rng);
        let expected = super::super::INV_PI * 4.0 / (2.0 * 2.0);
        assert!(
            (value.x - expected).abs() < 1e-4,
            "direct point-light term {} should equal {expected}",
            value.x
        );
    }

    #[test]
    fn deterministic_same_seed_same_result() {
        let geometry = corner();
        let materials = alloc::vec![
            Material::new(Bsdf::Lambert {
                albedo: Vec3::splat(0.7),
            }),
            Material::new(Bsdf::Lambert {
                albedo: Vec3::splat(0.7),
            }),
        ];
        let scene =
            Scene::new(geometry, materials, alloc::vec![], Vec3::splat(0.6)).expect("scene");
        let integrator = PathIntegrator::default();
        let mut rng_a = Rng::with_stream(99, 7);
        let mut rng_b = Rng::with_stream(99, 7);
        for _ in 0..256 {
            let a = integrator.radiance(&scene, down_ray(), &mut rng_a);
            let b = integrator.radiance(&scene, down_ray(), &mut rng_b);
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
            assert_eq!(a.z.to_bits(), b.z.to_bits());
        }
    }

    #[test]
    fn empty_scene_returns_environment() {
        let scene = Scene::new(
            empty_geometry(),
            alloc::vec![],
            alloc::vec![],
            Vec3::splat(0.3),
        )
        .expect("empty scene");
        let integrator = PathIntegrator::default();
        let mut rng = Rng::seed(1);
        let value = integrator.radiance(&scene, down_ray(), &mut rng);
        assert!(value.is_finite());
        assert!((value.x - 0.3).abs() < 1e-6);
    }

    #[test]
    fn degenerate_ray_does_not_panic() {
        let geometry = ground_plane();
        let materials = alloc::vec![Material::new(Bsdf::Lambert { albedo: Vec3::ONE })];
        let scene = Scene::new(geometry, materials, alloc::vec![], Vec3::ONE).expect("scene");
        let integrator = PathIntegrator::default();
        let mut rng = Rng::seed(1);
        // Zero-length direction and NaN/extreme origins must not panic.
        let zero_dir = Ray::new([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], 0.0, f32::INFINITY);
        let huge = Ray::new(
            [f32::MAX, f32::MAX, f32::MAX],
            [1.0, 0.0, 0.0],
            0.0,
            f32::INFINITY,
        );
        assert!(integrator.radiance(&scene, zero_dir, &mut rng).is_finite());
        let _ = integrator.radiance(&scene, huge, &mut rng);
    }
}
