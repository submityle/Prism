//! Particle-versus-environment collision solving: analytic primitives, signed-
//! distance-field response, screen-space depth-buffer contacts, and continuous
//! (swept) collision (design §10, §13).
//!
//! This is the deterministic `CPU` reference for the collision layer that a
//! future `GPU` compute kernel fills in. It aligns at the algorithm level with
//! Unreal `Niagara`'s *Collision* module, Unity `VFX Graph`'s *Collide with
//! Sphere / Plane / Depth Buffer* blocks, and the `GPU` screen-space depth
//! collision every production particle stack ships — without reusing any of
//! their code.
//!
//! # Relationship to [`super::constraints`]
//!
//! This module and [`super::constraints`] are orthogonal:
//!
//! - [`super::constraints`] owns the `XPBD`/`VBD` *internal* constraint solve
//!   (distance / bend / volume) that holds a body together, plus
//!   [`super::constraints::FractureCollisionScheme`], which only *selects* which
//!   collision proxy a fracture chunk uses (analytic plane, `SDF`, convex hull,
//!   or particle proxy).
//! - This module owns the *actual* per-particle collision solve against the
//!   environment: it detects the contact between a particle (a sphere of a
//!   given radius) and an analytic primitive, a signed-distance field, or the
//!   scene depth buffer, and it produces the position-level and velocity-level
//!   response (push-out, restitution bounce, Coulomb friction).
//!
//! # Determinism
//!
//! Everything here is pure and deterministic: the only floating-point primitive
//! beyond ordinary arithmetic is `sqrt` (through the hand-rolled [`Vec3`] math
//! and the quadratic `TOI` solves). There are no transcendental calls, so a
//! `GPU` kernel evaluating the same contacts in the same order produces bit-
//! identical results. Analytic primitives reuse [`Aabb`] and [`Plane`] from
//! [`super::sort_cull`] verbatim rather than redefining them.

use super::sort_cull::{Aabb, Plane};
use super::{Vec3, EPS_LEN_SQ};

/// Absolute tolerance for scalar floating-point comparisons in this module.
///
/// Direct `==` / `!=` on `f32` is forbidden by the workspace lints; equality is
/// always expressed as `(a - b).abs() < EPS`. This guards near-degenerate
/// denominators (a parallel sweep, a stationary sphere) rather than squared
/// lengths, which use [`EPS_LEN_SQ`].
pub const EPS: f32 = 1e-6;

/// The kinematic state of one particle entering the collision solve.
///
/// A particle is modeled as a sphere of `radius` centered at `pos` moving with
/// `vel`. A zero radius collapses to a point particle, which is valid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleState {
    /// Current center position.
    pub pos: Vec3,
    /// Current linear velocity.
    pub vel: Vec3,
    /// Collision radius (sphere thickness); `0` for a point particle.
    pub radius: f32,
}

impl ParticleState {
    /// Builds a particle state from its center, velocity, and radius.
    #[must_use]
    pub fn new(pos: Vec3, vel: Vec3, radius: f32) -> Self {
        Self { pos, vel, radius }
    }
}

/// The elastic and frictional response coefficients of a collision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResponseParams {
    /// Normal restitution (bounciness) in `0..=1`: `0` is a dead stop along the
    /// normal, `1` conserves the normal speed (a perfect bounce).
    pub restitution: f32,
    /// Coulomb friction coefficient in `0..=1`: the tangential speed loses at
    /// most `friction * |normal impulse|`. `0` preserves the slide, `1` is the
    /// maximum stick this contact can apply.
    pub friction: f32,
}

impl ResponseParams {
    /// Builds response parameters, clamping both coefficients into `0..=1`.
    #[must_use]
    pub fn new(restitution: f32, friction: f32) -> Self {
        Self {
            restitution: restitution.clamp(0.0, 1.0),
            friction: friction.clamp(0.0, 1.0),
        }
    }
}

/// A resolved contact between a particle sphere and a collider surface.
///
/// `normal` points *outward* from the solid toward the particle (the direction
/// the particle is pushed), `penetration` is the non-negative overlap depth
/// along that normal, and `point` is a representative surface contact point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// `true` when the particle overlaps the collider.
    pub hit: bool,
    /// Unit outward normal pointing from the surface toward the particle.
    pub normal: Vec3,
    /// Overlap depth along `normal`; `0` when there is no hit.
    pub penetration: f32,
    /// Representative surface contact point.
    pub point: Vec3,
}

impl Contact {
    /// The "no contact" result.
    #[must_use]
    pub fn miss() -> Self {
        Self {
            hit: false,
            normal: Vec3::ZERO,
            penetration: 0.0,
            point: Vec3::ZERO,
        }
    }

    /// Builds a hit contact.
    #[must_use]
    pub fn hit(normal: Vec3, penetration: f32, point: Vec3) -> Self {
        Self {
            hit: true,
            normal,
            penetration,
            point,
        }
    }
}

/// The result of resolving a collision: the corrected state and whether a
/// contact was applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CollisionResponse {
    /// Position after the penetration push-out.
    pub new_pos: Vec3,
    /// Velocity after the restitution and friction response.
    pub new_vel: Vec3,
    /// `true` when a contact was resolved (a push-out / bounce happened).
    pub hit: bool,
}

impl CollisionResponse {
    /// The pass-through response used when nothing was hit.
    #[must_use]
    pub fn unchanged(state: ParticleState) -> Self {
        Self {
            new_pos: state.pos,
            new_vel: state.vel,
            hit: false,
        }
    }
}

/// A solid sphere collider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    /// Center of the sphere.
    pub center: Vec3,
    /// Radius of the sphere.
    pub radius: f32,
}

impl Sphere {
    /// Builds a sphere from its center and radius.
    #[must_use]
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }
}

/// A solid capsule collider: all points within `radius` of the segment `a..b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule {
    /// First segment endpoint.
    pub a: Vec3,
    /// Second segment endpoint.
    pub b: Vec3,
    /// Capsule radius around the segment.
    pub radius: f32,
}

impl Capsule {
    /// Builds a capsule from its segment endpoints and radius.
    #[must_use]
    pub fn new(a: Vec3, b: Vec3, radius: f32) -> Self {
        Self { a, b, radius }
    }
}

/// A single sample of a signed-distance field at a query point (design §10).
///
/// `distance` is the signed distance from the query point to the solid surface,
/// positive outside the solid and negative inside; `gradient` is the field's
/// spatial gradient, which points *away* from the surface (outward) and whose
/// normalization is the collision normal. This is the contract a `GPU` `SDF`
/// texture lookup fills in per particle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSample {
    /// Signed distance to the surface (positive outside, negative inside).
    pub distance: f32,
    /// Field gradient; its normalization is the outward collision normal.
    pub gradient: Vec3,
}

impl SdfSample {
    /// Builds a signed-distance-field sample.
    #[must_use]
    pub fn new(distance: f32, gradient: Vec3) -> Self {
        Self { distance, gradient }
    }
}

/// A collider the batch solver understands.
///
/// The [`Collider::Sdf`] variant carries a signed-distance sample already
/// evaluated at the particle center (as a `GPU` `SDF` texture fetch would), so
/// the batch entry can treat every collider uniformly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Collider {
    /// A half-space whose [`Plane`] normal points into free space; the solid is
    /// the negative-distance side.
    HalfSpace(Plane),
    /// A solid sphere.
    Sphere(Sphere),
    /// A solid axis-aligned box (reusing [`Aabb`]).
    Box(Aabb),
    /// A solid capsule.
    Capsule(Capsule),
    /// A signed-distance-field sample evaluated at the particle center.
    Sdf(SdfSample),
}

/// Returns a unit normal along `v`, falling back to `fallback` when `v` is
/// (numerically) zero so a degenerate contact still has a defined direction.
#[must_use]
fn safe_normal(v: Vec3, fallback: Vec3) -> Vec3 {
    if v.length_squared() > EPS_LEN_SQ {
        v.normalize_or_zero()
    } else {
        fallback
    }
}

/// The closest point to `p` on the segment `a..b`.
#[must_use]
pub fn closest_point_on_segment(a: Vec3, b: Vec3, p: Vec3) -> Vec3 {
    let ab = b.sub(a);
    let denom = ab.length_squared();
    if denom <= EPS_LEN_SQ {
        return a;
    }
    let t = (p.sub(a).dot(ab) / denom).clamp(0.0, 1.0);
    a.add(ab.scale(t))
}

/// Signed distance from `p` to a solid sphere surface (positive outside).
#[must_use]
pub fn sd_sphere(sphere: Sphere, p: Vec3) -> f32 {
    p.distance(sphere.center) - sphere.radius
}

/// The closest point on a sphere's surface to `p` (the sphere center when `p`
/// coincides with the center).
#[must_use]
pub fn closest_point_sphere(sphere: Sphere, p: Vec3) -> Vec3 {
    let dir = p.sub(sphere.center);
    if dir.length_squared() <= EPS_LEN_SQ {
        return sphere.center;
    }
    sphere
        .center
        .add(dir.normalize_or_zero().scale(sphere.radius))
}

/// Signed distance from `p` to a solid capsule surface (positive outside).
#[must_use]
pub fn sd_capsule(capsule: Capsule, p: Vec3) -> f32 {
    let closest = closest_point_on_segment(capsule.a, capsule.b, p);
    p.distance(closest) - capsule.radius
}

/// The closest point on a capsule's surface to `p`.
#[must_use]
pub fn closest_point_capsule(capsule: Capsule, p: Vec3) -> Vec3 {
    let axis = closest_point_on_segment(capsule.a, capsule.b, p);
    let dir = p.sub(axis);
    if dir.length_squared() <= EPS_LEN_SQ {
        return axis;
    }
    axis.add(dir.normalize_or_zero().scale(capsule.radius))
}

/// The closest point to `p` inside or on a solid box (a component-wise clamp).
#[must_use]
pub fn closest_point_aabb(box_: Aabb, p: Vec3) -> Vec3 {
    Vec3::new(
        p.x.clamp(box_.min.x, box_.max.x),
        p.y.clamp(box_.min.y, box_.max.y),
        p.z.clamp(box_.min.z, box_.max.z),
    )
}

/// Signed distance from `p` to a solid axis-aligned box (positive outside,
/// negative inside).
#[must_use]
pub fn sd_aabb(box_: Aabb, p: Vec3) -> f32 {
    let c = box_.center();
    let e = box_.half_extents();
    let q = Vec3::new(
        (p.x - c.x).abs() - e.x,
        (p.y - c.y).abs() - e.y,
        (p.z - c.z).abs() - e.z,
    );
    let outside = q.max(Vec3::ZERO).length();
    let inside = q.x.max(q.y.max(q.z)).min(0.0);
    outside + inside
}

/// The outward face normal for a point inside a box: the axis whose face is
/// nearest, so an interior particle is pushed straight out the closest wall.
#[must_use]
fn aabb_interior_normal(box_: Aabb, p: Vec3) -> Vec3 {
    let c = box_.center();
    let e = box_.half_extents();
    // Distance from each face (negative inside); the largest (closest to zero)
    // is the exit face.
    let dx = (p.x - c.x).abs() - e.x;
    let dy = (p.y - c.y).abs() - e.y;
    let dz = (p.z - c.z).abs() - e.z;
    if dx >= dy && dx >= dz {
        Vec3::new((p.x - c.x).signum(), 0.0, 0.0)
    } else if dy >= dz {
        Vec3::new(0.0, (p.y - c.y).signum(), 0.0)
    } else {
        Vec3::new(0.0, 0.0, (p.z - c.z).signum())
    }
}

/// Detects the contact between a particle sphere and a half-space.
///
/// The [`Plane`] normal points into free space; the particle collides once its
/// signed distance drops below its radius.
#[must_use]
pub fn collide_half_space(plane: Plane, state: ParticleState) -> Contact {
    let sd = plane.signed_distance(state.pos);
    let penetration = state.radius - sd;
    if penetration <= 0.0 {
        return Contact::miss();
    }
    let normal = safe_normal(plane.normal, Vec3::new(0.0, 1.0, 0.0));
    let surface = state.pos.sub(normal.scale(sd));
    Contact::hit(normal, penetration, surface)
}

/// Detects the contact between a particle sphere and a solid sphere.
#[must_use]
pub fn collide_sphere(sphere: Sphere, state: ParticleState) -> Contact {
    let sd = sd_sphere(sphere, state.pos);
    let penetration = state.radius - sd;
    if penetration <= 0.0 {
        return Contact::miss();
    }
    let normal = safe_normal(state.pos.sub(sphere.center), Vec3::new(0.0, 1.0, 0.0));
    let surface = sphere.center.add(normal.scale(sphere.radius));
    Contact::hit(normal, penetration, surface)
}

/// Detects the contact between a particle sphere and a solid capsule.
#[must_use]
pub fn collide_capsule(capsule: Capsule, state: ParticleState) -> Contact {
    let axis = closest_point_on_segment(capsule.a, capsule.b, state.pos);
    let sd = state.pos.distance(axis) - capsule.radius;
    let penetration = state.radius - sd;
    if penetration <= 0.0 {
        return Contact::miss();
    }
    let normal = safe_normal(state.pos.sub(axis), Vec3::new(0.0, 1.0, 0.0));
    let surface = axis.add(normal.scale(capsule.radius));
    Contact::hit(normal, penetration, surface)
}

/// Detects the contact between a particle sphere and a solid axis-aligned box.
#[must_use]
pub fn collide_aabb(box_: Aabb, state: ParticleState) -> Contact {
    let closest = closest_point_aabb(box_, state.pos);
    let outward = state.pos.sub(closest);
    if outward.length_squared() > EPS_LEN_SQ {
        // Particle center is outside the box.
        let dist = outward.length();
        let penetration = state.radius - dist;
        if penetration <= 0.0 {
            return Contact::miss();
        }
        let normal = outward.normalize_or_zero();
        return Contact::hit(normal, penetration, closest);
    }
    // Particle center is inside the box: always a contact, pushed out the
    // nearest face.
    let sd = sd_aabb(box_, state.pos);
    let penetration = state.radius - sd;
    let normal = safe_normal(
        aabb_interior_normal(box_, state.pos),
        Vec3::new(0.0, 1.0, 0.0),
    );
    let surface = closest_point_aabb(box_, state.pos.add(normal.scale(state.radius)));
    Contact::hit(normal, penetration, surface)
}

/// Detects the contact between a particle sphere and a signed-distance field.
///
/// The collision normal is the (normalized) field gradient, which points
/// outward from the solid; the particle collides once the sampled distance
/// drops below its radius (design §10).
#[must_use]
pub fn collide_sdf(sample: SdfSample, state: ParticleState) -> Contact {
    let penetration = state.radius - sample.distance;
    if penetration <= 0.0 {
        return Contact::miss();
    }
    let normal = safe_normal(sample.gradient, Vec3::new(0.0, 1.0, 0.0));
    let surface = state.pos.sub(normal.scale(sample.distance));
    Contact::hit(normal, penetration, surface)
}

/// Detects the contact for any [`Collider`] variant.
#[must_use]
pub fn collide(collider: Collider, state: ParticleState) -> Contact {
    match collider {
        Collider::HalfSpace(plane) => collide_half_space(plane, state),
        Collider::Sphere(sphere) => collide_sphere(sphere, state),
        Collider::Box(box_) => collide_aabb(box_, state),
        Collider::Capsule(capsule) => collide_capsule(capsule, state),
        Collider::Sdf(sample) => collide_sdf(sample, state),
    }
}

/// Applies a resolved [`Contact`] to a particle: positional push-out, normal
/// restitution, and Coulomb friction.
///
/// The particle center is pushed out along the outward normal by the
/// penetration depth. The normal velocity is reflected and scaled by
/// `restitution` *only when the particle is approaching* the surface (so
/// separating particles are never sucked back). The tangential velocity loses
/// at most `friction * |normal impulse|` (Coulomb's cone), never reversing
/// direction.
///
/// - `restitution == 0` gives no normal bounce; `restitution == 1` conserves
///   the normal speed.
/// - `friction == 0` preserves the tangential slide; `friction == 1` applies
///   the maximum stick the normal impulse allows.
#[must_use]
pub fn resolve_contact(
    state: ParticleState,
    contact: Contact,
    params: ResponseParams,
) -> CollisionResponse {
    if !contact.hit {
        return CollisionResponse::unchanged(state);
    }
    let normal = contact.normal;
    let new_pos = state.pos.add(normal.scale(contact.penetration.max(0.0)));

    let vn = state.vel.dot(normal);
    let vt_vec = state.vel.sub(normal.scale(vn));
    let vt_speed = vt_vec.length();

    // Reflect the normal component only when moving into the surface.
    let vn_after = if vn < 0.0 {
        -params.restitution * vn
    } else {
        vn
    };

    // Coulomb friction: the change in normal momentum bounds the tangential
    // speed that can be removed this contact.
    let normal_impulse = (vn_after - vn).abs();
    let friction_delta = params.friction * normal_impulse;
    let new_vt_speed = (vt_speed - friction_delta).max(0.0);
    let vt_dir = vt_vec.normalize_or_zero();

    let new_vel = normal.scale(vn_after).add(vt_dir.scale(new_vt_speed));
    CollisionResponse {
        new_pos,
        new_vel,
        hit: true,
    }
}

/// Resolves a particle against a batch of colliders, applying the response for
/// the *deepest* penetrating contact (design §13).
///
/// Colliders are scanned in slice order and the one with the greatest
/// penetration wins (ties keep the earliest), so the result is independent of
/// how the caller ordered equal contacts and is deterministic. Returns an
/// unchanged response when nothing is hit.
#[must_use]
pub fn resolve_particle_collision(
    state: ParticleState,
    colliders: &[Collider],
    params: ResponseParams,
) -> CollisionResponse {
    let mut deepest = Contact::miss();
    for collider in colliders {
        let contact = collide(*collider, state);
        if contact.hit && contact.penetration > deepest.penetration {
            deepest = contact;
        }
    }
    resolve_contact(state, deepest, params)
}

/// The time-of-impact of a sphere of `radius` swept from `p0` to `p1` against a
/// half-space, or `None` when the sweep never enters the surface (design §13).
///
/// This is the continuous (`TOI`) test that stops a fast particle from
/// tunneling through a thin wall in a single step: it returns the fraction `t`
/// in `0..=1` at which the swept sphere first touches the plane. The particle
/// must start on the free side (its signed distance at least `radius`); a sweep
/// that is already penetrating or moving away returns `None`.
#[must_use]
pub fn swept_sphere_vs_plane(plane: Plane, p0: Vec3, p1: Vec3, radius: f32) -> Option<f32> {
    let d0 = plane.signed_distance(p0);
    let d1 = plane.signed_distance(p1);
    let denom = d1 - d0;
    if denom.abs() < EPS {
        // Parallel to the plane: no crossing this step.
        return None;
    }
    if d0 < radius - EPS {
        // Already penetrating at the start; handled by the discrete solve.
        return None;
    }
    let t = (radius - d0) / denom;
    if (0.0..=1.0).contains(&t) {
        Some(t)
    } else {
        None
    }
}

/// The time-of-impact of a moving particle sphere against a static solid
/// sphere, or `None` when the sweep never touches it (design §13).
///
/// The particle center moves from `p0` to `p1` over the unit step. Returns the
/// earliest fraction `t` in `0..=1` at which the surfaces first touch (a
/// quadratic solved with a single `sqrt`), or `Some(0.0)` when they already
/// overlap at the start. This catches tunneling between two small fast bodies.
#[must_use]
pub fn swept_sphere_vs_sphere(p0: Vec3, p1: Vec3, radius: f32, sphere: Sphere) -> Option<f32> {
    let combined = radius + sphere.radius;
    let rel = p0.sub(sphere.center);
    let disp = p1.sub(p0);

    let c = rel.length_squared() - combined * combined;
    if c <= 0.0 {
        // Overlapping at the start of the step.
        return Some(0.0);
    }
    let a = disp.length_squared();
    if a < EPS {
        // Not moving and not already overlapping.
        return None;
    }
    let b = 2.0 * rel.dot(disp);
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()) / (2.0 * a);
    if (0.0..=1.0).contains(&t) {
        Some(t)
    } else {
        None
    }
}

/// Parameters for a screen-space depth-buffer collision (design §13).
///
/// `thickness` is how far *behind* the visible surface a particle is still
/// considered to be colliding with it — the collidable slab the scene depth
/// buffer implies, since the buffer only stores the front-most surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthCollisionParams {
    /// Depth of the collidable slab behind the visible surface (view units).
    pub thickness: f32,
}

impl DepthCollisionParams {
    /// Builds depth-collision parameters from the slab thickness.
    #[must_use]
    pub fn new(thickness: f32) -> Self {
        Self { thickness }
    }
}

/// Returns `true` when a particle at `particle_depth` collides with the scene
/// depth buffer sampled as `scene_depth` (design §13).
///
/// This mirrors the `GPU` depth-collision test: a particle collides when it is
/// *behind* the visible surface but within the collidable slab, i.e.
/// `scene_depth < particle_depth < scene_depth + thickness`. A particle in
/// front of the surface (nearer the camera) or beyond the slab passes through.
#[must_use]
pub fn depth_buffer_hit(
    particle_depth: f32,
    scene_depth: f32,
    params: DepthCollisionParams,
) -> bool {
    particle_depth > scene_depth && particle_depth < scene_depth + params.thickness
}

/// Resolves a particle against the scene depth buffer (design §13).
///
/// When [`depth_buffer_hit`] passes, the reconstructed `scene_normal` (the
/// surface normal recovered from neighboring depth samples, pointing toward the
/// camera / out of the surface) becomes the collision normal, and the amount
/// the particle sits behind the surface (`particle_depth - scene_depth`) is the
/// penetration. The contact is then resolved through [`resolve_contact`].
/// Returns an unchanged response when there is no hit.
#[must_use]
pub fn resolve_depth_collision(
    state: ParticleState,
    scene_normal: Vec3,
    particle_depth: f32,
    scene_depth: f32,
    response: ResponseParams,
    depth: DepthCollisionParams,
) -> CollisionResponse {
    if !depth_buffer_hit(particle_depth, scene_depth, depth) {
        return CollisionResponse::unchanged(state);
    }
    let penetration = particle_depth - scene_depth;
    let normal = safe_normal(scene_normal, Vec3::new(0.0, 0.0, 1.0));
    let contact = Contact::hit(normal, penetration, state.pos);
    resolve_contact(state, contact, response)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    fn approx_vec(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    // ----- analytic primitives -----------------------------------------

    #[test]
    fn closest_point_on_segment_clamps_to_endpoints() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        assert!(approx_vec(
            closest_point_on_segment(a, b, Vec3::new(1.0, 1.0, 0.0)),
            Vec3::new(1.0, 0.0, 0.0)
        ));
        // Before `a`.
        assert!(approx_vec(
            closest_point_on_segment(a, b, Vec3::new(-5.0, 3.0, 0.0)),
            a
        ));
        // After `b`.
        assert!(approx_vec(
            closest_point_on_segment(a, b, Vec3::new(9.0, -3.0, 0.0)),
            b
        ));
        // Degenerate segment returns `a`.
        assert!(approx_vec(
            closest_point_on_segment(a, a, Vec3::splat(4.0)),
            a
        ));
    }

    #[test]
    fn sphere_signed_distance_and_closest_point() {
        let s = Sphere::new(Vec3::ZERO, 2.0);
        assert!(approx(sd_sphere(s, Vec3::new(5.0, 0.0, 0.0)), 3.0));
        assert!(approx(sd_sphere(s, Vec3::new(1.0, 0.0, 0.0)), -1.0));
        assert!(approx_vec(
            closest_point_sphere(s, Vec3::new(10.0, 0.0, 0.0)),
            Vec3::new(2.0, 0.0, 0.0)
        ));
        // Center is degenerate: returns the center itself.
        assert!(approx_vec(closest_point_sphere(s, Vec3::ZERO), Vec3::ZERO));
    }

    #[test]
    fn capsule_signed_distance_matches_segment() {
        let c = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0);
        // Above the middle of the segment.
        assert!(approx(sd_capsule(c, Vec3::new(0.0, 3.0, 0.0)), 2.0));
        // Off the rounded cap.
        assert!(approx(sd_capsule(c, Vec3::new(4.0, 0.0, 0.0)), 2.0));
        assert!(approx_vec(
            closest_point_capsule(c, Vec3::new(0.0, 3.0, 0.0)),
            Vec3::new(0.0, 1.0, 0.0)
        ));
    }

    #[test]
    fn aabb_signed_distance_outside_and_inside() {
        let b = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        // Outside on one axis.
        assert!(approx(sd_aabb(b, Vec3::new(3.0, 0.0, 0.0)), 2.0));
        // Diagonally outside a corner: sqrt(3) beyond the corner.
        let corner = sd_aabb(b, Vec3::new(2.0, 2.0, 2.0));
        assert!(approx(corner, (3.0f32).sqrt()));
        // Inside: negative distance to the nearest face.
        assert!(approx(sd_aabb(b, Vec3::new(0.5, 0.0, 0.0)), -0.5));
        assert!(approx_vec(
            closest_point_aabb(b, Vec3::new(3.0, 0.5, -9.0)),
            Vec3::new(1.0, 0.5, -1.0)
        ));
    }

    // ----- collision detection ------------------------------------------

    #[test]
    fn half_space_contact_pushes_along_normal() {
        // Floor at y = 0, free space above.
        let plane = Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0);
        let state = ParticleState::new(Vec3::new(0.0, 0.3, 0.0), Vec3::ZERO, 0.5);
        let c = collide_half_space(plane, state);
        assert!(c.hit);
        assert!(approx(c.penetration, 0.2));
        assert!(approx_vec(c.normal, Vec3::new(0.0, 1.0, 0.0)));
        // Well above: no contact.
        let high = ParticleState::new(Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO, 0.5);
        assert!(!collide_half_space(plane, high).hit);
    }

    #[test]
    fn sphere_contact_and_miss() {
        let s = Sphere::new(Vec3::ZERO, 1.0);
        let hit = collide_sphere(
            s,
            ParticleState::new(Vec3::new(1.2, 0.0, 0.0), Vec3::ZERO, 0.5),
        );
        assert!(hit.hit);
        assert!(approx(hit.penetration, 0.3));
        assert!(approx_vec(hit.normal, Vec3::new(1.0, 0.0, 0.0)));
        let miss = collide_sphere(
            s,
            ParticleState::new(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO, 0.5),
        );
        assert!(!miss.hit);
    }

    #[test]
    fn capsule_contact_normal_is_radial() {
        let c = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1.0);
        let hit = collide_capsule(
            c,
            ParticleState::new(Vec3::new(0.0, 1.2, 0.0), Vec3::ZERO, 0.5),
        );
        assert!(hit.hit);
        assert!(approx(hit.penetration, 0.3));
        assert!(approx_vec(hit.normal, Vec3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn aabb_contact_outside_and_inside() {
        let b = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        // Just outside the +x face.
        let outside = collide_aabb(
            b,
            ParticleState::new(Vec3::new(1.2, 0.0, 0.0), Vec3::ZERO, 0.5),
        );
        assert!(outside.hit);
        assert!(approx(outside.penetration, 0.3));
        assert!(approx_vec(outside.normal, Vec3::new(1.0, 0.0, 0.0)));
        // Deep inside, nearest to the +x face.
        let inside = collide_aabb(
            b,
            ParticleState::new(Vec3::new(0.8, 0.0, 0.0), Vec3::ZERO, 0.1),
        );
        assert!(inside.hit);
        assert!(approx_vec(inside.normal, Vec3::new(1.0, 0.0, 0.0)));
        // sd = -0.2, penetration = radius - sd = 0.1 + 0.2 = 0.3.
        assert!(approx(inside.penetration, 0.3));
    }

    #[test]
    fn sdf_contact_uses_gradient_as_normal() {
        let sample = SdfSample::new(0.2, Vec3::new(0.0, 2.0, 0.0));
        let hit = collide_sdf(sample, ParticleState::new(Vec3::ZERO, Vec3::ZERO, 0.5));
        assert!(hit.hit);
        assert!(approx(hit.penetration, 0.3));
        assert!(approx_vec(hit.normal, Vec3::new(0.0, 1.0, 0.0)));
        let miss = collide_sdf(
            SdfSample::new(2.0, Vec3::new(0.0, 1.0, 0.0)),
            ParticleState::new(Vec3::ZERO, Vec3::ZERO, 0.5),
        );
        assert!(!miss.hit);
    }

    // ----- response -----------------------------------------------------

    #[test]
    fn push_out_moves_particle_to_the_surface() {
        let plane = Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0);
        let state = ParticleState::new(Vec3::new(0.0, 0.3, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.5);
        let c = collide_half_space(plane, state);
        let r = resolve_contact(state, c, ResponseParams::new(0.0, 0.0));
        assert!(r.hit);
        // Pushed to y = 0.5 (radius above the floor).
        assert!(approx(r.new_pos.y, 0.5));
    }

    #[test]
    fn restitution_zero_kills_normal_velocity() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let contact = Contact::hit(normal, 0.1, Vec3::ZERO);
        let state = ParticleState::new(Vec3::ZERO, Vec3::new(0.0, -4.0, 0.0), 0.0);
        let r = resolve_contact(state, contact, ResponseParams::new(0.0, 0.0));
        assert!(approx(r.new_vel.y, 0.0));
    }

    #[test]
    fn restitution_one_conserves_normal_speed() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let contact = Contact::hit(normal, 0.1, Vec3::ZERO);
        let state = ParticleState::new(Vec3::ZERO, Vec3::new(0.0, -4.0, 0.0), 0.0);
        let r = resolve_contact(state, contact, ResponseParams::new(1.0, 0.0));
        // Perfect bounce: reflected with the same magnitude.
        assert!(approx(r.new_vel.y, 4.0));
    }

    #[test]
    fn separating_particle_keeps_its_normal_velocity() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let contact = Contact::hit(normal, 0.1, Vec3::ZERO);
        // Already moving away from the surface.
        let state = ParticleState::new(Vec3::ZERO, Vec3::new(0.0, 3.0, 0.0), 0.0);
        let r = resolve_contact(state, contact, ResponseParams::new(1.0, 0.0));
        assert!(approx(r.new_vel.y, 3.0));
    }

    #[test]
    fn friction_zero_preserves_tangential_velocity() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let contact = Contact::hit(normal, 0.1, Vec3::ZERO);
        let state = ParticleState::new(Vec3::ZERO, Vec3::new(1.0, -2.0, 0.0), 0.0);
        let r = resolve_contact(state, contact, ResponseParams::new(0.0, 0.0));
        assert!(approx(r.new_vel.x, 1.0));
    }

    #[test]
    fn friction_one_zeroes_tangential_velocity() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let contact = Contact::hit(normal, 0.1, Vec3::ZERO);
        // Tangential speed (1) <= normal impulse (|−2| = 2), so it is fully
        // removed by maximal Coulomb friction.
        let state = ParticleState::new(Vec3::ZERO, Vec3::new(1.0, -2.0, 0.0), 0.0);
        let r = resolve_contact(state, contact, ResponseParams::new(0.0, 1.0));
        assert!(approx(r.new_vel.x, 0.0));
        assert!(approx(r.new_vel.y, 0.0));
    }

    #[test]
    fn no_hit_contact_passes_state_through() {
        let state = ParticleState::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0), 0.5);
        let r = resolve_contact(state, Contact::miss(), ResponseParams::new(0.5, 0.5));
        assert!(!r.hit);
        assert!(approx_vec(r.new_pos, state.pos));
        assert!(approx_vec(r.new_vel, state.vel));
    }

    // ----- batch entry --------------------------------------------------

    #[test]
    fn batch_resolves_the_deepest_contact() {
        let shallow = Collider::HalfSpace(Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0));
        let deep = Collider::Sphere(Sphere::new(Vec3::new(0.0, -0.5, 0.0), 1.0));
        let state = ParticleState::new(Vec3::new(0.0, 0.2, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.5);
        // The sphere penetrates far more deeply than the floor here.
        let sphere_pen = collide(deep, state).penetration;
        let plane_pen = collide(shallow, state).penetration;
        assert!(sphere_pen > plane_pen);
        let r = resolve_particle_collision(state, &[shallow, deep], ResponseParams::new(0.0, 0.0));
        assert!(r.hit);
        // Deepest is the sphere: its normal points up (+y) from the center below.
        assert!(approx_vec(
            r.new_pos,
            state.pos.add(Vec3::new(0.0, 1.0, 0.0).scale(sphere_pen))
        ));
    }

    #[test]
    fn batch_with_no_colliders_is_unchanged() {
        let state = ParticleState::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.5);
        let r = resolve_particle_collision(state, &[], ResponseParams::new(0.5, 0.5));
        assert!(!r.hit);
        assert!(approx_vec(r.new_pos, state.pos));
    }

    #[test]
    fn batch_reports_miss_when_all_clear() {
        let colliders = [
            Collider::HalfSpace(Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0)),
            Collider::Sphere(Sphere::new(Vec3::new(100.0, 0.0, 0.0), 1.0)),
        ];
        let state = ParticleState::new(Vec3::new(0.0, 50.0, 0.0), Vec3::ZERO, 0.5);
        assert!(!resolve_particle_collision(state, &colliders, ResponseParams::new(0.0, 0.0)).hit);
    }

    // ----- continuous (TOI) collision -----------------------------------

    #[test]
    fn swept_plane_catches_tunneling() {
        // Floor at y = 0; particle starts well above and ends well below in one
        // step — a discrete test at the endpoints could miss the wall.
        let plane = Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0);
        let p0 = Vec3::new(0.0, 5.0, 0.0);
        let p1 = Vec3::new(0.0, -5.0, 0.0);
        let toi = swept_sphere_vs_plane(plane, p0, p1, 0.5).expect("should catch the crossing");
        // Contact when the center reaches y = radius = 0.5: t = (5 - 0.5) / 10.
        assert!(approx(toi, 0.45));
    }

    #[test]
    fn swept_plane_ignores_parallel_and_separating() {
        let plane = Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0);
        // Parallel sweep well above the plane.
        assert!(swept_sphere_vs_plane(
            plane,
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(3.0, 5.0, 0.0),
            0.5
        )
        .is_none());
        // Moving away from the plane.
        assert!(swept_sphere_vs_plane(
            plane,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 9.0, 0.0),
            0.5
        )
        .is_none());
    }

    #[test]
    fn swept_sphere_catches_fast_pass_through() {
        let target = Sphere::new(Vec3::ZERO, 1.0);
        let p0 = Vec3::new(-5.0, 0.0, 0.0);
        let p1 = Vec3::new(5.0, 0.0, 0.0);
        let toi = swept_sphere_vs_sphere(p0, p1, 0.5, target).expect("should catch the crossing");
        // Touch when center distance = 1.5: at x = -1.5, t = (−1.5 − (−5)) / 10.
        assert!(approx(toi, 0.35));
    }

    #[test]
    fn swept_sphere_overlap_and_miss() {
        let target = Sphere::new(Vec3::ZERO, 1.0);
        // Already overlapping at the start.
        match swept_sphere_vs_sphere(
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.6, 0.0, 0.0),
            0.2,
            target,
        ) {
            Some(t) => assert!(approx(t, 0.0)),
            None => panic!("overlap at t=0 should be reported"),
        }
        // Passes by without touching.
        assert!(swept_sphere_vs_sphere(
            Vec3::new(-5.0, 5.0, 0.0),
            Vec3::new(5.0, 5.0, 0.0),
            0.2,
            target
        )
        .is_none());
    }

    // ----- depth-buffer collision ---------------------------------------

    #[test]
    fn depth_buffer_hit_slab() {
        let params = DepthCollisionParams::new(0.5);
        // Behind the surface, inside the slab.
        assert!(depth_buffer_hit(10.2, 10.0, params));
        // In front of the surface (nearer the camera): passes through.
        assert!(!depth_buffer_hit(9.8, 10.0, params));
        // Beyond the slab: passes through.
        assert!(!depth_buffer_hit(10.9, 10.0, params));
    }

    #[test]
    fn depth_collision_resolves_with_scene_normal() {
        let state = ParticleState::new(Vec3::new(0.0, 0.0, 10.2), Vec3::new(0.0, 0.0, -3.0), 0.0);
        let scene_normal = Vec3::new(0.0, 0.0, 1.0);
        let r = resolve_depth_collision(
            state,
            scene_normal,
            10.2,
            10.0,
            ResponseParams::new(1.0, 0.0),
            DepthCollisionParams::new(0.5),
        );
        assert!(r.hit);
        // Approaching along -z with restitution 1: reflected to +3.
        assert!(approx(r.new_vel.z, 3.0));
        // Pushed out by the penetration (0.2) along +z.
        assert!(approx(r.new_pos.z, 10.4));
    }

    #[test]
    fn depth_collision_passes_through_when_not_behind_surface() {
        let state = ParticleState::new(Vec3::new(0.0, 0.0, 9.0), Vec3::new(0.0, 0.0, -3.0), 0.0);
        let r = resolve_depth_collision(
            state,
            Vec3::new(0.0, 0.0, 1.0),
            9.0,
            10.0,
            ResponseParams::new(1.0, 0.0),
            DepthCollisionParams::new(0.5),
        );
        assert!(!r.hit);
        assert!(approx_vec(r.new_vel, state.vel));
    }

    #[test]
    fn response_params_clamp_coefficients() {
        let p = ResponseParams::new(2.0, -1.0);
        assert!(approx(p.restitution, 1.0));
        assert!(approx(p.friction, 0.0));
    }
}
