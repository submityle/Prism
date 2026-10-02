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

use super::area_light::AreaLights;
use super::bsdf::Bsdf;
use super::environment::EnvironmentMap;
use super::estimator::{Light, LightHit};
use super::mis::power_heuristic;
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

/// The distant lighting returned by rays that escape the geometry.
///
/// A `Constant` dome is uniform radiance with no light-sampling strategy, so
/// escaped rays claim it in full (its [`Environment::pdf`] is zero and
/// multiple-importance weighting falls back to the `BSDF` arrival). A `Map`
/// dome is an image-based [`EnvironmentMap`] that is importance-sampled by
/// next-event estimation and weighted against `BSDF`-sampled escapes.
#[derive(Clone, Debug)]
pub enum Environment {
    /// A spatially uniform radiance dome.
    Constant(Vec3),
    /// An image-based radiance dome in octahedral layout.
    Map(EnvironmentMap),
}

impl Environment {
    /// The radiance arriving from the dome along the unit direction `dir`.
    #[must_use]
    pub fn radiance(&self, dir: Vec3) -> Vec3 {
        match self {
            Self::Constant(radiance) => *radiance,
            Self::Map(map) => map.radiance(dir),
        }
    }

    /// The solid-angle density of drawing `dir` from this dome's next-event
    /// estimator, or zero when the dome offers no light-sampling strategy.
    #[must_use]
    pub fn pdf(&self, dir: Vec3) -> f32 {
        match self {
            Self::Constant(_) => 0.0,
            Self::Map(map) => map.pdf(dir),
        }
    }

    /// Estimates direct lighting from the dome by next-event estimation,
    /// returning black for a `Constant` dome (which has no light-sampling
    /// strategy and is gathered only through `BSDF`-sampled escapes).
    #[must_use]
    fn sample_direct<F>(
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
        match self {
            Self::Constant(_) => Vec3::ZERO,
            Self::Map(map) => map.sample_direct(point, normal, wo, bsdf, rng, occluded),
        }
    }
}

impl From<Vec3> for Environment {
    fn from(radiance: Vec3) -> Self {
        Self::Constant(radiance)
    }
}

/// A traced scene: geometry, a per-triangle material table, analytic lights, and
/// a distant environment dome for rays that escape to infinity.
#[derive(Clone, Debug)]
pub struct Scene {
    geometry: TriangleMeshBvh,
    materials: Vec<Material>,
    lights: Vec<Light>,
    /// Emissive mesh triangles gathered into one sampleable compound light for
    /// next-event estimation; empty when no material emits.
    area_lights: AreaLights,
    environment: Environment,
}

impl Scene {
    /// Builds a scene, binding one [`Material`] to each triangle (indexed by the
    /// mesh's original triangle id) and a distant `environment` dome.
    ///
    /// `environment` accepts anything convertible into an [`Environment`]: a
    /// [`Vec3`] becomes a uniform [`Environment::Constant`] dome, while an
    /// [`EnvironmentMap`] can be passed as [`Environment::Map`] for an
    /// importance-sampled image-based dome.
    ///
    /// # Errors
    ///
    /// Returns [`SceneError::MaterialCountMismatch`] when `materials.len()` is
    /// not exactly the geometry's triangle count.
    pub fn new(
        geometry: TriangleMeshBvh,
        materials: Vec<Material>,
        lights: Vec<Light>,
        environment: impl Into<Environment>,
    ) -> Result<Self, SceneError> {
        let triangles = geometry.mesh().triangle_count();
        if materials.len() != triangles {
            return Err(SceneError::MaterialCountMismatch {
                materials: materials.len(),
                triangles,
            });
        }
        // Gather every emissive triangle into one compound area light so the
        // integrator can connect to glowing mesh geometry by next-event
        // estimation, not only by chance `BSDF`-sampled hits.
        let mesh = geometry.mesh();
        let area_lights = AreaLights::new((0..triangles).filter_map(|tri| {
            let emission = materials[tri].emission;
            if emission.max_component() > 0.0 {
                Some((mesh.triangle_positions(tri), emission))
            } else {
                None
            }
        }));
        Ok(Self {
            geometry,
            materials,
            lights,
            area_lights,
            environment: environment.into(),
        })
    }

    /// The distant dome returned by rays that leave the scene.
    #[must_use]
    pub fn environment(&self) -> &Environment {
        &self.environment
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

    /// Finds the nearest analytic light struck by the ray `origin + t *
    /// direction` (with `direction` unit length), or [`None`] when no finite
    /// emitter lies in front of the ray.
    ///
    /// Delta lights ([`Light::Point`], [`Light::Directional`]) are never hit;
    /// only area emitters contribute. The returned hit is the closest one, so a
    /// continuation ray cannot pick up an emitter hidden behind another.
    #[must_use]
    fn nearest_light_hit(&self, origin: Vec3, direction: Vec3) -> Option<LightHit> {
        let mut best: Option<LightHit> = None;
        for light in &self.lights {
            let Some(hit) = light.intersect(origin, direction) else {
                continue;
            };
            if hit.distance <= RAY_EPS {
                continue;
            }
            if best.is_none_or(|current| hit.distance < current.distance) {
                best = Some(hit);
            }
        }
        best
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
        // Multiple-importance-sampling bookkeeping for the previous scattering
        // event. The camera ray is treated as a delta event (weight 1), so a
        // primary ray that stares straight into an emitter collects its full
        // radiance.
        let mut prev_delta = true;
        let mut prev_bsdf_pdf = 0.0f32;

        loop {
            // World-space ray origin and unit direction, shared by the geometry
            // query and the analytic area-light intersection.
            let origin_v = Vec3::from_array(ray.origin());
            let dir_v = Vec3::from_array(ray.direction()).normalize_or_zero();
            let geo = scene.intersect(&ray);
            let light_hit = scene.nearest_light_hit(origin_v, dir_v);

            // An area light struck before any surface terminates the path with
            // its emission. This is the `BSDF`-sampling half of the area-light
            // `MIS` pair; next-event estimation (Strategy A) handles the other
            // half inside `Light::direct`.
            let geo_distance = geo.map(|isect| isect.position.sub(origin_v).length());
            if let Some(lh) = light_hit {
                let closer_than_geo = geo_distance.is_none_or(|gd| lh.distance < gd);
                if closer_than_geo {
                    // Specular bounces and the camera ray cannot be reached by
                    // next-event estimation, so they claim the emitter in full;
                    // diffuse/glossy bounces share it with `NEE` by the power
                    // heuristic.
                    let weight = if prev_delta {
                        1.0
                    } else {
                        power_heuristic(prev_bsdf_pdf, lh.light_pdf)
                    };
                    radiance = radiance.add(throughput.mul(lh.emission).scale(weight));
                    break;
                }
            }

            let Some(isect) = geo else {
                // An escaped ray gathers the environment dome. This is the
                // `BSDF`-sampling half of the dome's `MIS` pair: a specular or
                // camera bounce claims it in full, while a diffuse/glossy bounce
                // shares it with the dome's next-event estimation (which only
                // exists for an importance-sampled map) under the power
                // heuristic.
                let env_radiance = scene.environment.radiance(dir_v);
                let weight = if prev_delta {
                    1.0
                } else {
                    let env_pdf = scene.environment.pdf(dir_v);
                    if env_pdf > 0.0 {
                        power_heuristic(prev_bsdf_pdf, env_pdf)
                    } else {
                        1.0
                    }
                };
                radiance = radiance.add(throughput.mul(env_radiance).scale(weight));
                break;
            };
            let material = scene.materials[isect.material];

            // Surface emission on an emissive mesh triangle. Because those
            // triangles are also sampled explicitly by next-event estimation
            // (via `scene.area_lights`), the `BSDF`-sampled arrival is weighted
            // against light sampling with the power heuristic. A specular or
            // camera bounce cannot be reached by `NEE`, so it claims the full
            // emission (weight 1).
            if material.emission.max_component() > 0.0 {
                let weight = if prev_delta {
                    1.0
                } else {
                    let light_pdf = scene.area_lights.pdf(
                        origin_v,
                        isect.position,
                        isect.normal,
                        material.emission,
                    );
                    if light_pdf > 0.0 {
                        power_heuristic(prev_bsdf_pdf, light_pdf)
                    } else {
                        1.0
                    }
                };
                radiance = radiance.add(throughput.mul(material.emission).scale(weight));
            }

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
                // Next-event estimation against emissive mesh triangles, the
                // Strategy-A half of their `MIS` pair (the arrival above is
                // Strategy B).
                let mesh_direct = scene.area_lights.sample_direct(
                    shadow_origin,
                    shading_normal,
                    wo,
                    &material.bsdf,
                    rng,
                    &occluded,
                );
                radiance = radiance.add(throughput.mul(mesh_direct));
                // Next-event estimation against the environment dome, the
                // Strategy-A half of its `MIS` pair (an importance-sampled map
                // only; a constant dome contributes nothing here and is gathered
                // through `BSDF`-sampled escapes instead).
                let env_direct = scene.environment.sample_direct(
                    shadow_origin,
                    shading_normal,
                    wo,
                    &material.bsdf,
                    rng,
                    &occluded,
                );
                radiance = radiance.add(throughput.mul(env_direct));
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

            // Carry this bounce's sampling density forward so the next vertex
            // can weight an area-light hit against next-event estimation.
            prev_delta = sample.specular;
            prev_bsdf_pdf = sample.pdf;
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

    /// Rectangular area light hovering face-down over the origin, spanning
    /// `x, z` in `[-1, 1]` at `y = 4`.
    fn overhead_quad(emission: Vec3) -> Light {
        Light::Quad {
            origin: Vec3::new(-1.0, 4.0, -1.0),
            edge_u: Vec3::new(2.0, 0.0, 0.0),
            edge_v: Vec3::new(0.0, 0.0, 2.0),
            emission,
        }
    }

    /// A floor plus a small downward-facing emissive triangle overhead, as a
    /// single mesh so the glowing triangle lives in the `BVH` alongside the
    /// floor. The floor is triangle 0, the emitter triangle 1.
    fn floor_with_overhead_emitter(
        floor_albedo: Vec3,
        emission: Vec3,
    ) -> (TriangleMeshBvh, Vec<Material>) {
        let positions = alloc::vec![
            // Floor (y = 0).
            [-10.0_f32, 0.0, -10.0],
            [10.0, 0.0, -10.0],
            [0.0, 0.0, 10.0],
            // Downward-facing emitter at y = 4, area 2.
            [-1.0, 4.0, -1.0],
            [1.0, 4.0, -1.0],
            [-1.0, 4.0, 1.0],
        ];
        let indices = alloc::vec![[0u32, 1, 2], [3, 4, 5]];
        let mesh = crate::ray_scene::triangle_mesh::TriangleMesh::new(
            positions,
            alloc::vec![],
            alloc::vec![],
            indices,
        )
        .expect("valid floor + emitter mesh");
        let materials = alloc::vec![
            Material::new(Bsdf::Lambert {
                albedo: floor_albedo,
            }),
            Material::emissive(Bsdf::Lambert { albedo: Vec3::ZERO }, emission),
        ];
        (TriangleMeshBvh::build(mesh), materials)
    }

    #[test]
    fn emissive_mesh_mean_matches_light_sampling() {
        // The full integrator connects to the emissive triangle both by
        // next-event estimation at the floor vertex (Strategy A) and by the
        // `BSDF`-sampled continuation ray that strikes it (Strategy B), combined
        // with the power heuristic. Their mean must equal a single-strategy
        // light-sampling reference, proving the mesh-emitter wiring carries no
        // bias and no double counting.
        let floor_albedo = Vec3::splat(0.8);
        let emission = Vec3::splat(5.0);
        let (geometry, materials) = floor_with_overhead_emitter(floor_albedo, emission);
        let scene = Scene::new(geometry, materials, alloc::vec![], Vec3::ZERO)
            .expect("emissive mesh scene");
        // Two bounces: the floor vertex plus the continuation ray that can reach
        // the emitter. There is no other geometry, so the mean is the direct
        // term at the floor point under the camera ray.
        let integrator = PathIntegrator::new(2, 2);

        // Shading point, normal and view direction for the camera ray onto the
        // origin, shared by the brute-force reference below.
        let point = Vec3::ZERO;
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.0, 1.0, 0.0);
        let bsdf = Bsdf::Lambert {
            albedo: floor_albedo,
        };
        // The emitter triangle, with its uniform-area sampling parameters.
        let em_anchor = Vec3::new(-1.0, 4.0, -1.0);
        let em_edge1 = Vec3::new(2.0, 0.0, 0.0);
        let em_edge2 = Vec3::new(0.0, 0.0, 2.0);
        let em_area = 2.0f32;
        let em_normal = Vec3::new(0.0, -1.0, 0.0);

        let count = 200_000u32;
        let mut path_sum = 0.0f64;
        let mut ref_sum = 0.0f64;
        let mut rng_ref = Rng::with_stream(31, 2);
        for s in 0..count {
            let mut rng = Rng::with_stream(31, u64::from(s) + 3);
            path_sum += f64::from(integrator.radiance(&scene, down_ray(), &mut rng).x);

            // Reference: a point drawn uniformly over the emitter triangle,
            // converted to a solid-angle direct term, with no `MIS` weight.
            let u0 = rng_ref.next_f32();
            let u1 = rng_ref.next_f32();
            let su0 = u0.sqrt();
            let on_light = em_anchor
                .add(em_edge1.scale(u1 * su0))
                .add(em_edge2.scale(su0 * (1.0 - u1)));
            let to_light = on_light.sub(point);
            let dist_sq = to_light.length_squared();
            if dist_sq > super::super::EPS_LEN_SQ {
                let dist = dist_sq.sqrt();
                let wi = to_light.scale(1.0 / dist);
                let cos_surface = normal.dot(wi);
                let cos_light = em_normal.dot(wi).abs();
                if cos_surface > 0.0 && cos_light > 0.0 {
                    let fr = bsdf.evaluate(wo, wi, normal);
                    let geom = cos_surface * cos_light * em_area / dist_sq;
                    ref_sum += f64::from(fr.mul(emission).scale(geom).x);
                }
            }
        }
        let path_mean = path_sum / f64::from(count);
        let ref_mean = ref_sum / f64::from(count);
        assert!(path_mean > 0.0, "floor should receive emitted light");
        assert!(
            (path_mean / ref_mean - 1.0).abs() < 3.0e-2,
            "integrator mean {path_mean} vs light-sampling reference {ref_mean}"
        );
    }

    #[test]
    fn mirror_reflects_area_light() {
        // A perfect mirror cannot be connected to an emitter by next-event
        // estimation, so before hittable area lights its reflection of the quad
        // was pure black. The `BSDF`-sampled continuation ray must now strike the
        // light and carry its emission back, scaled only by the mirror's
        // reflectance (the specular vertex claims the full, unweighted radiance).
        let geometry = ground_plane();
        let materials = alloc::vec![Material::new(Bsdf::Mirror {
            reflectance: Vec3::splat(0.9),
        })];
        let light = overhead_quad(Vec3::splat(5.0));
        let scene = Scene::new(geometry, materials, alloc::vec![light], Vec3::ZERO)
            .expect("mirror + area light scene");
        let integrator = PathIntegrator::new(4, 4);
        let mut rng = Rng::seed(2024);
        let value = integrator.radiance(&scene, down_ray(), &mut rng);
        // Mirror reflectance (0.9) times emission (5.0): the delta vertex takes
        // the emitter in full, so the reflection is exactly 4.5.
        assert!(
            (value.x - 4.5).abs() < 1e-4,
            "mirror should reflect the area light, got {}",
            value.x
        );
    }

    #[test]
    fn glossy_area_light_matches_light_sampling_in_the_mean() {
        // A glossy floor lit by the quad exercises both halves of the area-light
        // `MIS` pair: next-event estimation at the floor vertex (Strategy A) and
        // the continuation ray striking the emitter (Strategy B). Their combined
        // mean must equal the single-strategy light-sampling reference, proving
        // the split carries no bias and no double counting.
        let reflectance = Vec3::splat(0.9);
        let roughness = 0.3;
        let bsdf = Bsdf::GgxConductor {
            reflectance,
            roughness,
        };
        let emission = Vec3::splat(5.0);
        let geometry = ground_plane();
        let materials = alloc::vec![Material::new(bsdf)];
        let light = overhead_quad(emission);
        let scene = Scene::new(geometry, materials, alloc::vec![light], Vec3::ZERO)
            .expect("glossy + area light scene");
        // Two bounces let the continuation ray reach the emitter; there is no
        // other geometry, so the mean is exactly the single-bounce direct term.
        let integrator = PathIntegrator::new(2, 2);

        // Shading point, normal and view direction for the camera ray onto the
        // origin, shared by the brute-force reference below.
        let point = Vec3::ZERO;
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let wo = Vec3::new(0.0, 1.0, 0.0);
        let origin = Vec3::new(-1.0, 4.0, -1.0);
        let edge_u = Vec3::new(2.0, 0.0, 0.0);
        let edge_v = Vec3::new(0.0, 0.0, 2.0);
        let cross = edge_u.cross(edge_v);
        let area = cross.length();
        let light_normal = cross.scale(1.0 / area);

        let count = 200_000u32;
        let mut path_sum = 0.0f64;
        let mut ref_sum = 0.0f64;
        let mut rng_path = Rng::with_stream(1337, 1);
        let mut rng_ref = Rng::with_stream(1337, 2);
        for _ in 0..count {
            path_sum += f64::from(integrator.radiance(&scene, down_ray(), &mut rng_path).x);

            // Pure light-sampling reference: a point drawn uniformly on the quad,
            // converted to a solid-angle estimate with no `MIS` weight.
            let on_light = origin
                .add(edge_u.scale(rng_ref.next_f32()))
                .add(edge_v.scale(rng_ref.next_f32()));
            let to_light = on_light.sub(point);
            let dist_sq = to_light.length_squared();
            if dist_sq > 0.0 {
                let dist = dist_sq.sqrt();
                let wi = to_light.scale(1.0 / dist);
                let cos_surface = normal.dot(wi);
                let cos_light = light_normal.dot(wi).abs();
                if cos_surface > 0.0 && cos_light > 0.0 {
                    let fr = bsdf.evaluate(wo, wi, normal);
                    let geom = cos_surface * cos_light * area / dist_sq;
                    ref_sum += f64::from(fr.mul(emission).scale(geom).x);
                }
            }
        }
        let path_mean = path_sum / f64::from(count);
        let ref_mean = ref_sum / f64::from(count);
        assert!(path_mean > 0.0, "glossy floor should see the area light");
        assert!(
            (path_mean / ref_mean - 1.0).abs() < 4.0e-2,
            "path-traced mean {path_mean} vs light-sampling reference {ref_mean}"
        );
    }

    /// Builds a smoothly varying octahedral radiance dome for the environment
    /// integration tests.
    fn gradient_env() -> EnvironmentMap {
        let width = 16;
        let height = 16;
        let mut texels = Vec::with_capacity(width * height);
        for row in 0..height {
            for col in 0..width {
                let u = (col as f32 + 0.5) / width as f32;
                let v = (row as f32 + 0.5) / height as f32;
                let brightness = 0.2 + 3.0 * u * u + 1.5 * v;
                texels.push(Vec3::new(brightness, 0.6 * brightness, 0.3 * brightness));
            }
        }
        EnvironmentMap::new(width, height, texels)
    }

    /// Rejection-samples a direction uniformly over the upper hemisphere about
    /// `+y`, returning the direction and its cosine with the normal.
    fn uniform_upper_hemisphere(rng: &mut Rng) -> Vec3 {
        loop {
            let x = 2.0 * rng.next_f32() - 1.0;
            let y = 2.0 * rng.next_f32() - 1.0;
            let z = 2.0 * rng.next_f32() - 1.0;
            let len_sq = x * x + y * y + z * z;
            if len_sq > 1e-6 && len_sq <= 1.0 {
                let inv = 1.0 / len_sq.sqrt();
                let dir = Vec3::new(x * inv, y * inv, z * inv);
                return if dir.y >= 0.0 {
                    dir
                } else {
                    Vec3::new(dir.x, -dir.y, dir.z)
                };
            }
        }
    }

    #[test]
    fn environment_map_visible_on_escape() {
        // A camera ray into empty geometry is a delta vertex, so it claims the
        // dome radiance along its direction in full.
        let map = gradient_env();
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let expected = map.radiance(dir);
        let scene = Scene::new(
            empty_geometry(),
            alloc::vec![],
            alloc::vec![],
            Environment::Map(map),
        )
        .expect("environment scene");
        let integrator = PathIntegrator::new(8, 5);
        let mut rng = Rng::seed(5);
        let value = integrator.radiance(&scene, down_ray(), &mut rng);
        assert!(value.sub(expected).length() < 1e-5, "got {value:?}");
    }

    #[test]
    fn environment_map_lights_lambert_floor() {
        // A Lambertian floor under an image-based dome must return the same
        // single-bounce reflected radiance the integrator's two sampling
        // strategies estimate and an independent hemisphere integral predicts.
        let albedo = 0.5f32;
        let map = gradient_env();

        // Reference reflected radiance L_o = (albedo / pi) * integral over the
        // upper hemisphere of L(w) cos(theta) dw, estimated with uniform
        // hemisphere sampling (density 1 / (2 pi)); the (albedo/pi)*(2 pi)
        // prefactor collapses to 2 * albedo.
        let mut ref_rng = Rng::with_stream(2, 9);
        let ref_samples = 400_000;
        let mut sum = [0.0f64; 3];
        for _ in 0..ref_samples {
            let dir = uniform_upper_hemisphere(&mut ref_rng);
            let radiance = map.radiance(dir);
            let cos = f64::from(dir.y);
            sum[0] += f64::from(radiance.x) * cos;
            sum[1] += f64::from(radiance.y) * cos;
            sum[2] += f64::from(radiance.z) * cos;
        }
        let scale = 2.0 * f64::from(albedo) / f64::from(ref_samples);
        let reference = [sum[0] * scale, sum[1] * scale, sum[2] * scale];

        // Integrator estimate: a single scattering event (camera -> floor ->
        // escape) with next-event estimation against the dome plus the
        // `BSDF`-sampled escape, combined under multiple importance sampling.
        let materials = alloc::vec![Material::new(Bsdf::Lambert {
            albedo: Vec3::splat(albedo),
        })];
        let scene = Scene::new(
            ground_plane(),
            materials,
            alloc::vec![],
            Environment::Map(map.clone()),
        )
        .expect("floor under environment");
        let integrator = PathIntegrator::new(2, 8);
        let path_samples = 200_000u32;
        let mut path_sum = [0.0f64; 3];
        for s in 0..path_samples {
            let mut rng = Rng::with_stream(41, u64::from(s) + 1);
            let value = integrator.radiance(&scene, down_ray(), &mut rng);
            path_sum[0] += f64::from(value.x);
            path_sum[1] += f64::from(value.y);
            path_sum[2] += f64::from(value.z);
        }
        let path = [
            path_sum[0] / f64::from(path_samples),
            path_sum[1] / f64::from(path_samples),
            path_sum[2] / f64::from(path_samples),
        ];

        for channel in 0..3 {
            assert!(reference[channel] > 0.0);
            let rel = (path[channel] - reference[channel]).abs() / reference[channel];
            assert!(
                rel < 3e-2,
                "channel {channel}: path {} vs reference {}",
                path[channel],
                reference[channel]
            );
        }
    }
}
