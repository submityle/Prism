//! Ray-traced particle collision and lighting contracts (design §22).
//!
//! This is the deterministic `CPU` reference for Ember's optional ray-tracing
//! integration with `bevy_solari`: per-particle short rays cast along the motion
//! direction to detect scene collisions (bounce + friction response), a quality
//! ladder that degrades to an `SDF` or depth-buffer query when ray-tracing
//! hardware is absent, and a lit-sampling contract that turns a ray-traced
//! visibility query into soft shadows and a `GI` bounce colour for the shading
//! closures.
//!
//! Ray tracing is a *differentiating extra*, never a hard dependency: the
//! collision-method chooser always resolves to a runnable fallback, so the core
//! pipeline is never blocked when `bevy_solari` is unavailable. Both `PBR` and
//! `NPR` particles consume the ray-traced lighting result (design §16, §22),
//! routed by [`enables_ray_traced_lighting`]; `Unlit` opts out because it does
//! not respond to scene light.
//!
//! All math is spelled out on the shared hand-rolled [`Vec3`]; only `sqrt` (via
//! [`Vec3::length`]/[`Vec3::normalize_or_zero`]) is used and every normalization
//! and division is zero-guarded, so no transcendental call or `NaN` can leak in
//! and the `CPU` reference stays bit-reproducible against a future `GPU`/`RT`
//! kernel.

use super::{EmberShadingModel, ShadingBasis, Vec3, EPS_LEN_SQ};

// ---------------------------------------------------------------------------
// Collision queries and hits (design §22).
// ---------------------------------------------------------------------------

/// A short collision ray cast from a particle along its motion (design §22).
///
/// `direction` need not be unit length; it is normalized robustly at use. The
/// `radius` models a thick ray / sphere cast so small particles still register
/// contact. A query with a (near) zero direction or non-positive `max_distance`
/// is inactive (see [`RayCollisionQuery::is_active`]) and yields no collision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayCollisionQuery {
    /// Ray origin (the particle position).
    pub origin: Vec3,
    /// Ray direction (motion direction); normalized robustly when used.
    pub direction: Vec3,
    /// Maximum query distance along the ray (typically `speed * dt`).
    pub max_distance: f32,
    /// Particle radius for the thick-ray / sphere cast.
    pub radius: f32,
}

impl RayCollisionQuery {
    /// Builds a query that casts along a particle's motion over a timestep.
    ///
    /// The direction is the velocity direction and the reach is `speed * dt`
    /// (plus the particle `radius`), so a particle only tests the space it is
    /// about to sweep through. A (near) zero velocity produces an inactive
    /// query.
    #[must_use]
    pub fn along_motion(origin: Vec3, velocity: Vec3, dt: f32, radius: f32) -> Self {
        let speed = velocity.length();
        let step = speed * dt.max(0.0);
        Self {
            origin,
            direction: velocity.normalize_or_zero(),
            max_distance: step + radius.max(0.0),
            radius: radius.max(0.0),
        }
    }

    /// Whether the query can actually hit anything: a non-degenerate direction
    /// and a positive reach.
    #[must_use]
    pub fn is_active(self) -> bool {
        self.direction.length_squared() > EPS_LEN_SQ && self.max_distance > 0.0
    }
}

/// A resolved ray-scene intersection (design §22).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// Distance from the ray origin to the hit point.
    pub distance: f32,
    /// Geometric surface normal at the hit point (unit-ish; normalized at use).
    pub normal: Vec3,
    /// Material slot of the hit surface, for restitution/friction lookup.
    pub material_slot: u32,
    /// Whether the ray struck a back face (it started inside the geometry).
    pub back_face: bool,
}

impl RayHit {
    /// The collision normal oriented against the incoming ray.
    ///
    /// A back-face hit means the stored geometric normal points *away* from the
    /// approaching particle, so it is flipped to face the particle before the
    /// bounce response uses it.
    #[must_use]
    pub fn oriented_normal(self) -> Vec3 {
        if self.back_face {
            self.normal.scale(-1.0)
        } else {
            self.normal
        }
    }
}

/// Per-material collision response coefficients (design §22).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CollisionResponse {
    /// Coefficient of restitution `e` in `0..=1`: `0` is a dead stop along the
    /// normal, `1` is a perfectly elastic bounce.
    pub restitution: f32,
    /// Tangential friction in `0..=1`: `0` keeps all sliding speed, `1` stops
    /// tangential motion.
    pub friction: f32,
}

impl CollisionResponse {
    /// A fully inelastic, high-friction response (particle sticks).
    pub const STICKY: Self = Self {
        restitution: 0.0,
        friction: 1.0,
    };

    /// A perfectly elastic, frictionless response (mirror bounce).
    pub const ELASTIC: Self = Self {
        restitution: 1.0,
        friction: 0.0,
    };
}

/// Mirror-reflects a velocity about a surface normal: `v' = v - 2 (v·n) n`.
///
/// The normal is normalized robustly; a (near) zero normal is a no-op and
/// returns `velocity` unchanged, so the result is never `NaN`.
#[must_use]
pub fn reflect(velocity: Vec3, normal: Vec3) -> Vec3 {
    let n = normal.normalize_or_zero();
    if n.length_squared() <= EPS_LEN_SQ {
        return velocity;
    }
    let vn = velocity.dot(n);
    velocity.sub(n.scale(2.0 * vn))
}

/// Applies a bounce + friction collision response to a velocity (design §22).
///
/// Decomposes `velocity` into a normal and a tangential component about `normal`
/// and returns `v' = v_t (1 - friction) - e v_n`, i.e. the tangential part is
/// damped by friction and the normal part is reversed and scaled by restitution
/// `e`. Equivalent to the reflection formula `v' = v - (1 + e)(v·n) n` before
/// friction. Coefficients are clamped to `0..=1`.
///
/// The response is a no-op (returns `velocity`) when the normal is (near) zero
/// or when the particle is already moving away from the surface (`v·n >= 0`),
/// so a grazing or separating particle is never spuriously pushed and no `NaN`
/// is produced.
#[must_use]
pub fn resolve_bounce(velocity: Vec3, normal: Vec3, response: CollisionResponse) -> Vec3 {
    let n = normal.normalize_or_zero();
    if n.length_squared() <= EPS_LEN_SQ {
        return velocity;
    }
    let vn_scalar = velocity.dot(n);
    if vn_scalar >= 0.0 {
        // Separating or purely tangential: no collision impulse.
        return velocity;
    }
    let v_n = n.scale(vn_scalar);
    let v_t = velocity.sub(v_n);
    let e = response.restitution.clamp(0.0, 1.0);
    let f = response.friction.clamp(0.0, 1.0);
    v_t.scale(1.0 - f).sub(v_n.scale(e))
}

/// Applies a collision response using a hit's ray-oriented normal (design §22).
///
/// Convenience wrapper that flips the normal for back-face hits (see
/// [`RayHit::oriented_normal`]) before delegating to [`resolve_bounce`].
#[must_use]
pub fn apply_collision(velocity: Vec3, hit: RayHit, response: CollisionResponse) -> Vec3 {
    resolve_bounce(velocity, hit.oriented_normal(), response)
}

// ---------------------------------------------------------------------------
// Collision method quality ladder and degradation (design §22, §28).
// ---------------------------------------------------------------------------

/// How particle-scene collisions are queried (design §22).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CollisionMethod {
    /// Hardware ray tracing through `bevy_solari`: the high-quality path.
    Raytrace,
    /// Signed-distance-field (`SDF`) scene query: a cheaper approximation.
    Sdf,
    /// Depth-buffer collision against the camera depth: the always-available
    /// fallback (screen-space, misses off-screen geometry).
    DepthBuffer,
}

impl CollisionMethod {
    /// Whether this method needs ray-tracing hardware (`bevy_solari`).
    #[must_use]
    pub const fn uses_ray_tracing(self) -> bool {
        matches!(self, CollisionMethod::Raytrace)
    }
}

/// Ray-tracing-relevant hardware capabilities of the running platform.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HardwareCaps {
    /// Hardware/`API` ray tracing is available (`bevy_solari` usable).
    pub ray_tracing: bool,
    /// A scene `SDF` volume is available for distance queries.
    pub sdf_volume: bool,
}

/// The requested collision quality tier (design §28).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CollisionQuality {
    /// Best available accuracy (prefers ray tracing).
    High,
    /// Balanced accuracy (prefers the `SDF` query).
    Medium,
    /// Cheapest path (depth-buffer fallback).
    Low,
}

/// Chooses the collision method for the given hardware and quality (design §22).
///
/// Ray tracing is optional and never blocks the pipeline, so the chooser always
/// resolves to a runnable method by degrading gracefully:
///
/// - `High` uses [`CollisionMethod::Raytrace`] when ray tracing is available,
///   else the `SDF` query, else the depth-buffer fallback.
/// - `Medium` uses the `SDF` query when available, else the depth-buffer.
/// - `Low` always uses the depth-buffer fallback.
#[must_use]
pub fn choose_collision_method(caps: HardwareCaps, quality: CollisionQuality) -> CollisionMethod {
    match quality {
        CollisionQuality::High => {
            if caps.ray_tracing {
                CollisionMethod::Raytrace
            } else if caps.sdf_volume {
                CollisionMethod::Sdf
            } else {
                CollisionMethod::DepthBuffer
            }
        }
        CollisionQuality::Medium => {
            if caps.sdf_volume {
                CollisionMethod::Sdf
            } else {
                CollisionMethod::DepthBuffer
            }
        }
        CollisionQuality::Low => CollisionMethod::DepthBuffer,
    }
}

// ---------------------------------------------------------------------------
// Ray-traced lighting / shadows / GI (design §22).
// ---------------------------------------------------------------------------

/// A request to sample ray-traced lighting for one particle (design §22).
///
/// The visibility and bounce colour themselves come from the ray-scene query;
/// this contract carries the surface data needed to turn a raw visibility and
/// bounce sample into an incident-irradiance estimate. A (near) zero `normal`
/// marks a volumetric sample, which is treated as isotropic (`N·L = 1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LitSampleRequest {
    /// World-space position of the sample.
    pub position: Vec3,
    /// Surface normal; zero means an isotropic (volumetric) sample.
    pub normal: Vec3,
    /// Direction toward the light (normalized robustly at use).
    pub to_light: Vec3,
    /// Incoming light radiance (colour times intensity).
    pub light_radiance: Vec3,
    /// Whether to gather a ray-traced `GI` bounce colour as well as shadows.
    pub sample_gi: bool,
}

/// The result of a ray-traced lighting sample (design §22).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LitSampleResult {
    /// Light visibility in `0..=1` (soft shadow term; `0` fully occluded).
    pub visibility: f32,
    /// Approximate incident irradiance (`visibility * N·L * radiance`).
    pub irradiance: Vec3,
    /// Ray-traced `GI` bounce colour (zero when `sample_gi` is false).
    pub bounce: Vec3,
}

/// Evaluates a ray-traced lighting sample from raw query outputs (design §22).
///
/// `raw_visibility` is the shadow-ray visibility (clamped to `0..=1`) and
/// `raw_bounce` is the gathered `GI` colour. The incident irradiance is
/// `visibility * max(N·L, 0) * radiance`; a volumetric sample (zero normal) uses
/// `N·L = 1`. The bounce colour is passed through only when the request opted
/// into `GI`, otherwise it is zero. Direction and normal are normalized
/// robustly, so a zero light direction simply yields no directional response.
#[must_use]
pub fn evaluate_lit_sample(
    request: LitSampleRequest,
    raw_visibility: f32,
    raw_bounce: Vec3,
) -> LitSampleResult {
    let visibility = raw_visibility.clamp(0.0, 1.0);
    let n = request.normal.normalize_or_zero();
    let l = request.to_light.normalize_or_zero();
    let n_dot_l = if n.length_squared() <= EPS_LEN_SQ {
        // Isotropic / volumetric sample: no oriented surface.
        1.0
    } else {
        n.dot(l).max(0.0)
    };
    let irradiance = request.light_radiance.scale(visibility * n_dot_l);
    let bounce = if request.sample_gi {
        raw_bounce
    } else {
        Vec3::ZERO
    };
    LitSampleResult {
        visibility,
        irradiance,
        bounce,
    }
}

/// Whether a shading basis lobe consumes ray-traced lighting (design §22).
///
/// Every lit basis (`PBR`/`NPR`/custom) receives ray-traced shadows and `GI`;
/// only `Unlit` opts out, since it does not respond to scene light.
#[must_use]
pub const fn basis_enables_ray_traced_lighting(basis: ShadingBasis) -> bool {
    match basis {
        ShadingBasis::Unlit => false,
        ShadingBasis::Pbr | ShadingBasis::Npr | ShadingBasis::Custom(_) => true,
    }
}

/// Whether a shading model consumes ray-traced lighting (design §16, §22).
///
/// `PBR` and `NPR` are equal citizens here: both receive ray-traced shadows and
/// `GI` bounce, as do custom closures. `Unlit` opts out. A `Hybrid` model opts
/// in when either of its lobes is lit.
#[must_use]
pub const fn enables_ray_traced_lighting(model: EmberShadingModel) -> bool {
    match model {
        EmberShadingModel::Unlit => false,
        EmberShadingModel::Pbr | EmberShadingModel::Npr | EmberShadingModel::Custom(_) => true,
        EmberShadingModel::Hybrid { base, overlay, .. } => {
            basis_enables_ray_traced_lighting(base) || basis_enables_ray_traced_lighting(overlay)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    fn vec_approx(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    // --- query construction --------------------------------------------

    #[test]
    fn along_motion_reaches_swept_distance() {
        let q = RayCollisionQuery::along_motion(Vec3::ZERO, Vec3::new(0.0, -10.0, 0.0), 0.1, 0.5);
        assert!(q.is_active());
        // speed 10 * dt 0.1 = 1.0, plus radius 0.5.
        assert!(approx(q.max_distance, 1.5));
        assert!(vec_approx(q.direction, Vec3::new(0.0, -1.0, 0.0)));
    }

    #[test]
    fn zero_velocity_query_is_inactive() {
        let q = RayCollisionQuery::along_motion(Vec3::ZERO, Vec3::ZERO, 0.1, 0.0);
        assert!(!q.is_active());
        assert!(vec_approx(q.direction, Vec3::ZERO));
    }

    // --- reflection geometry -------------------------------------------

    #[test]
    fn reflect_head_on_reverses_normal_component() {
        let v = Vec3::new(0.0, -1.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        assert!(vec_approx(reflect(v, n), Vec3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn reflect_preserves_tangential_component() {
        // 45-degree incidence onto a floor: x kept, y flipped.
        let v = Vec3::new(1.0, -1.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        assert!(vec_approx(reflect(v, n), Vec3::new(1.0, 1.0, 0.0)));
    }

    #[test]
    fn reflect_zero_normal_is_noop() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert!(vec_approx(reflect(v, Vec3::ZERO), v));
    }

    #[test]
    fn reflect_normalizes_non_unit_normal() {
        let v = Vec3::new(0.0, -1.0, 0.0);
        let n = Vec3::new(0.0, 5.0, 0.0);
        assert!(vec_approx(reflect(v, n), Vec3::new(0.0, 1.0, 0.0)));
    }

    // --- bounce restitution / friction ---------------------------------

    #[test]
    fn elastic_bounce_equals_reflection() {
        let v = Vec3::new(1.0, -1.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let bounced = resolve_bounce(v, n, CollisionResponse::ELASTIC);
        assert!(vec_approx(bounced, reflect(v, n)));
    }

    #[test]
    fn zero_restitution_kills_normal_component() {
        let v = Vec3::new(1.0, -1.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let r = CollisionResponse {
            restitution: 0.0,
            friction: 0.0,
        };
        // Normal part removed, tangential x preserved.
        assert!(vec_approx(
            resolve_bounce(v, n, r),
            Vec3::new(1.0, 0.0, 0.0)
        ));
    }

    #[test]
    fn full_friction_kills_tangential_component() {
        let v = Vec3::new(1.0, -1.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let r = CollisionResponse {
            restitution: 1.0,
            friction: 1.0,
        };
        // Tangential x removed, normal fully bounced (e = 1).
        assert!(vec_approx(
            resolve_bounce(v, n, r),
            Vec3::new(0.0, 1.0, 0.0)
        ));
    }

    #[test]
    fn partial_restitution_and_friction() {
        let v = Vec3::new(2.0, -4.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let r = CollisionResponse {
            restitution: 0.5,
            friction: 0.25,
        };
        // v_n = (0,-4,0); v_t = (2,0,0).
        // result = v_t*(1-0.25) - 0.5*v_n = (1.5, 0, 0) - (0,-2,0) = (1.5, 2, 0).
        assert!(vec_approx(
            resolve_bounce(v, n, r),
            Vec3::new(1.5, 2.0, 0.0)
        ));
    }

    #[test]
    fn separating_velocity_is_noop() {
        // Already moving away from the surface (v·n > 0): no impulse.
        let v = Vec3::new(0.0, 1.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        assert!(vec_approx(
            resolve_bounce(v, n, CollisionResponse::ELASTIC),
            v
        ));
    }

    #[test]
    fn grazing_velocity_barely_changes() {
        // Nearly tangential approach: tangential preserved, tiny normal bounce.
        let v = Vec3::new(1.0, -0.001, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let bounced = resolve_bounce(v, n, CollisionResponse::ELASTIC);
        assert!(approx(bounced.x, 1.0));
        assert!(approx(bounced.y, 0.001));
    }

    #[test]
    fn bounce_zero_normal_is_noop() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        assert!(vec_approx(
            resolve_bounce(v, Vec3::ZERO, CollisionResponse::ELASTIC),
            v
        ));
    }

    #[test]
    fn coefficients_are_clamped() {
        let v = Vec3::new(0.0, -1.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let over = CollisionResponse {
            restitution: 5.0,
            friction: 5.0,
        };
        // Clamped to e=1, f=1 -> full elastic normal bounce.
        assert!(vec_approx(
            resolve_bounce(v, n, over),
            Vec3::new(0.0, 1.0, 0.0)
        ));
    }

    // --- back-face handling --------------------------------------------

    #[test]
    fn back_face_hit_flips_normal() {
        let hit = RayHit {
            distance: 1.0,
            normal: Vec3::new(0.0, 1.0, 0.0),
            material_slot: 0,
            back_face: true,
        };
        assert!(vec_approx(hit.oriented_normal(), Vec3::new(0.0, -1.0, 0.0)));
    }

    #[test]
    fn front_face_hit_keeps_normal_and_bounces() {
        // Particle falling onto a floor from outside.
        let hit = RayHit {
            distance: 0.5,
            normal: Vec3::new(0.0, 1.0, 0.0),
            material_slot: 3,
            back_face: false,
        };
        let v = Vec3::new(0.0, -2.0, 0.0);
        let bounced = apply_collision(v, hit, CollisionResponse::ELASTIC);
        assert!(vec_approx(bounced, Vec3::new(0.0, 2.0, 0.0)));
    }

    // --- collision-method degradation matrix ---------------------------

    #[test]
    fn high_quality_prefers_ray_tracing() {
        let caps = HardwareCaps {
            ray_tracing: true,
            sdf_volume: true,
        };
        assert_eq!(
            choose_collision_method(caps, CollisionQuality::High),
            CollisionMethod::Raytrace
        );
    }

    #[test]
    fn high_quality_degrades_to_sdf_without_ray_tracing() {
        let caps = HardwareCaps {
            ray_tracing: false,
            sdf_volume: true,
        };
        assert_eq!(
            choose_collision_method(caps, CollisionQuality::High),
            CollisionMethod::Sdf
        );
    }

    #[test]
    fn high_quality_degrades_to_depth_without_ray_tracing_or_sdf() {
        let caps = HardwareCaps {
            ray_tracing: false,
            sdf_volume: false,
        };
        assert_eq!(
            choose_collision_method(caps, CollisionQuality::High),
            CollisionMethod::DepthBuffer
        );
    }

    #[test]
    fn medium_quality_never_uses_ray_tracing() {
        let caps = HardwareCaps {
            ray_tracing: true,
            sdf_volume: true,
        };
        assert_eq!(
            choose_collision_method(caps, CollisionQuality::Medium),
            CollisionMethod::Sdf
        );
        let no_sdf = HardwareCaps {
            ray_tracing: true,
            sdf_volume: false,
        };
        assert_eq!(
            choose_collision_method(no_sdf, CollisionQuality::Medium),
            CollisionMethod::DepthBuffer
        );
    }

    #[test]
    fn low_quality_always_depth_buffer() {
        let caps = HardwareCaps {
            ray_tracing: true,
            sdf_volume: true,
        };
        assert_eq!(
            choose_collision_method(caps, CollisionQuality::Low),
            CollisionMethod::DepthBuffer
        );
    }

    #[test]
    fn ray_tracing_never_blocks_pipeline() {
        // Whatever the caps/quality, a runnable method is always returned.
        for rt in [false, true] {
            for sdf in [false, true] {
                for q in [
                    CollisionQuality::High,
                    CollisionQuality::Medium,
                    CollisionQuality::Low,
                ] {
                    let caps = HardwareCaps {
                        ray_tracing: rt,
                        sdf_volume: sdf,
                    };
                    let method = choose_collision_method(caps, q);
                    if method.uses_ray_tracing() {
                        assert!(rt, "chose ray tracing without hardware");
                    }
                }
            }
        }
    }

    // --- lit sampling ---------------------------------------------------

    fn lit_request(normal: Vec3, to_light: Vec3, sample_gi: bool) -> LitSampleRequest {
        LitSampleRequest {
            position: Vec3::ZERO,
            normal,
            to_light,
            light_radiance: Vec3::new(4.0, 4.0, 4.0),
            sample_gi,
        }
    }

    #[test]
    fn head_on_light_full_irradiance() {
        let req = lit_request(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), false);
        let r = evaluate_lit_sample(req, 1.0, Vec3::ZERO);
        assert!(approx(r.visibility, 1.0));
        assert!(vec_approx(r.irradiance, Vec3::new(4.0, 4.0, 4.0)));
        assert!(vec_approx(r.bounce, Vec3::ZERO));
    }

    #[test]
    fn grazing_light_no_irradiance() {
        // Light perpendicular to the normal -> N·L = 0.
        let req = lit_request(Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0), false);
        let r = evaluate_lit_sample(req, 1.0, Vec3::ZERO);
        assert!(vec_approx(r.irradiance, Vec3::ZERO));
    }

    #[test]
    fn back_light_clamps_ndotl_to_zero() {
        let req = lit_request(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -1.0, 0.0), false);
        let r = evaluate_lit_sample(req, 1.0, Vec3::ZERO);
        assert!(vec_approx(r.irradiance, Vec3::ZERO));
    }

    #[test]
    fn visibility_scales_and_clamps() {
        let req = lit_request(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), false);
        let half = evaluate_lit_sample(req, 0.5, Vec3::ZERO);
        assert!(vec_approx(half.irradiance, Vec3::new(2.0, 2.0, 2.0)));

        let over = evaluate_lit_sample(req, 9.0, Vec3::ZERO);
        assert!(approx(over.visibility, 1.0));
        let under = evaluate_lit_sample(req, -3.0, Vec3::ZERO);
        assert!(approx(under.visibility, 0.0));
        assert!(vec_approx(under.irradiance, Vec3::ZERO));
    }

    #[test]
    fn volumetric_sample_is_isotropic() {
        // Zero normal -> N·L = 1 regardless of light direction.
        let req = lit_request(Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0), false);
        let r = evaluate_lit_sample(req, 1.0, Vec3::ZERO);
        assert!(vec_approx(r.irradiance, Vec3::new(4.0, 4.0, 4.0)));
    }

    #[test]
    fn gi_bounce_gated_by_request() {
        let bounce = Vec3::new(0.1, 0.2, 0.3);
        let on = evaluate_lit_sample(
            lit_request(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), true),
            1.0,
            bounce,
        );
        assert!(vec_approx(on.bounce, bounce));
        let off = evaluate_lit_sample(
            lit_request(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), false),
            1.0,
            bounce,
        );
        assert!(vec_approx(off.bounce, Vec3::ZERO));
    }

    #[test]
    fn lit_sample_is_deterministic() {
        let req = lit_request(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.3, 1.0, 0.0), true);
        let bounce = Vec3::new(0.2, 0.2, 0.2);
        let a = evaluate_lit_sample(req, 0.7, bounce);
        let b = evaluate_lit_sample(req, 0.7, bounce);
        assert_eq!(a, b);
    }

    // --- shading-model routing -----------------------------------------

    #[test]
    fn lighting_routing_by_shading_model() {
        assert!(!enables_ray_traced_lighting(EmberShadingModel::Unlit));
        assert!(enables_ray_traced_lighting(EmberShadingModel::Pbr));
        assert!(enables_ray_traced_lighting(EmberShadingModel::Npr));
        assert!(enables_ray_traced_lighting(EmberShadingModel::Custom(9)));
    }

    #[test]
    fn hybrid_lighting_routing() {
        let lit = EmberShadingModel::Hybrid {
            base: ShadingBasis::Unlit,
            overlay: ShadingBasis::Npr,
            weight: 0.5,
        };
        assert!(enables_ray_traced_lighting(lit));
        let unlit = EmberShadingModel::Hybrid {
            base: ShadingBasis::Unlit,
            overlay: ShadingBasis::Unlit,
            weight: 0.5,
        };
        assert!(!enables_ray_traced_lighting(unlit));
    }

    #[test]
    fn basis_lighting_routing() {
        assert!(!basis_enables_ray_traced_lighting(ShadingBasis::Unlit));
        assert!(basis_enables_ray_traced_lighting(ShadingBasis::Pbr));
        assert!(basis_enables_ray_traced_lighting(ShadingBasis::Npr));
        assert!(basis_enables_ray_traced_lighting(ShadingBasis::Custom(1)));
    }

    #[test]
    fn collision_method_ray_tracing_flag() {
        assert!(CollisionMethod::Raytrace.uses_ray_tracing());
        assert!(!CollisionMethod::Sdf.uses_ray_tracing());
        assert!(!CollisionMethod::DepthBuffer.uses_ray_tracing());
    }
}
