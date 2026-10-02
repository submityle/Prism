//! Real-device parity for the particle-versus-environment collision twin:
//! [`GpuCollision`](prism_volumetric_gpu::collision::GpuCollision) must
//! reproduce the `CPU` golden
//! [`collision`](prism_render_architecture::particle::collision) routine for
//! routine across the three signed-distance primitives
//! ([`sd_sphere`](prism_render_architecture::particle::collision::sd_sphere),
//! [`sd_capsule`](prism_render_architecture::particle::collision::sd_capsule),
//! [`sd_aabb`](prism_render_architecture::particle::collision::sd_aabb)), the
//! four closest-point projections
//! ([`closest_point_on_segment`](prism_render_architecture::particle::collision::closest_point_on_segment),
//! [`closest_point_sphere`](prism_render_architecture::particle::collision::closest_point_sphere),
//! [`closest_point_capsule`](prism_render_architecture::particle::collision::closest_point_capsule),
//! [`closest_point_aabb`](prism_render_architecture::particle::collision::closest_point_aabb)),
//! the five contact predicates
//! ([`collide_half_space`](prism_render_architecture::particle::collision::collide_half_space),
//! [`collide_sphere`](prism_render_architecture::particle::collision::collide_sphere),
//! [`collide_capsule`](prism_render_architecture::particle::collision::collide_capsule),
//! [`collide_aabb`](prism_render_architecture::particle::collision::collide_aabb),
//! [`collide_sdf`](prism_render_architecture::particle::collision::collide_sdf)),
//! the single and batch impulse solves
//! ([`resolve_contact`](prism_render_architecture::particle::collision::resolve_contact),
//! [`resolve_particle_collision`](prism_render_architecture::particle::collision::resolve_particle_collision))
//! and the two continuous sweeps
//! ([`swept_sphere_vs_plane`](prism_render_architecture::particle::collision::swept_sphere_vs_plane),
//! [`swept_sphere_vs_sphere`](prism_render_architecture::particle::collision::swept_sphere_vs_sphere)).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each routine threads through multiplies, adds, guarded divisions and at most
//! one `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every
//! continuous quantity and asserts an *exact* match on the discrete hit flags.
//! Every fixture is placed clear of every contact and sweep boundary (a
//! constructive margin of at least `BAND`), and every normal is axis-aligned or
//! an explicit direction so the outward direction is unique; the fixtures use no
//! transcendental method (only algebraic `sqrt` through the reused vector math),
//! and the random batch draws from a host-side integer `LCG` so it needs no
//! external math library.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::collision`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::collision::{
    Capsule, Collider, Contact, ParticleState, ResponseParams, SdfSample, Sphere,
};
use prism_render_architecture::particle::sort_cull::{Aabb, Plane};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::collision::cpu_reference;
use prism_volumetric_gpu::{CollisionQuery, CollisionResult, GpuCollision, GpuContext};

/// Absolute parity bound on every continuous quantity. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Constructive margin that keeps every fixture clear of a contact or sweep
/// boundary, so the discrete hit flags are unambiguous.
const BAND: f32 = 0.1;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn approx(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Returns whether two vectors agree component-wise within the parity bound.
fn approx_vec(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

/// Returns whether two contacts agree: the hit flag exactly, and when hit the
/// normal, penetration and surface point within the parity bound.
fn contacts_match(a: &Contact, b: &Contact) -> bool {
    if a.hit != b.hit {
        return false;
    }
    if !a.hit {
        return true;
    }
    approx_vec(a.normal, b.normal)
        && approx(a.penetration, b.penetration)
        && approx_vec(a.point, b.point)
}

/// Returns whether two sweep results agree: the hit flag exactly, and when hit
/// the time of impact within the parity bound.
fn sweeps_match(h0: bool, t0: f32, h1: bool, t1: f32) -> bool {
    if h0 != h1 {
        return false;
    }
    !h0 || approx(t0, t1)
}

/// Returns whether a `GPU` result matches the `CPU` reference, matching variant
/// against variant and applying the discrete / continuous split per field.
fn results_match(got: &CollisionResult, want: &CollisionResult) -> bool {
    match (got, want) {
        (CollisionResult::Scalar(a), CollisionResult::Scalar(b)) => approx(*a, *b),
        (CollisionResult::Vector(a), CollisionResult::Vector(b)) => approx_vec(*a, *b),
        (CollisionResult::Contact(a), CollisionResult::Contact(b)) => contacts_match(a, b),
        (CollisionResult::Response(a), CollisionResult::Response(b)) => {
            a.hit == b.hit && approx_vec(a.new_pos, b.new_pos) && approx_vec(a.new_vel, b.new_vel)
        }
        (
            CollisionResult::Swept { hit: h0, toi: t0 },
            CollisionResult::Swept { hit: h1, toi: t1 },
        ) => sweeps_match(*h0, *t0, *h1, *t1),
        _ => false,
    }
}

/// Evaluates a single query on device and asserts it matches the `CPU` golden.
fn check(gpu: &GpuCollision, ctx: &GpuContext, query: CollisionQuery) {
    let want = cpu_reference(&query);
    let got = gpu.evaluate(ctx, std::slice::from_ref(&query));
    assert_eq!(got.len(), 1, "one query yields one result");
    assert!(
        results_match(&got[0], &want),
        "GPU result {:?} must match CPU {:?} for {:?}",
        got[0],
        want,
        query
    );
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Draws a uniform `f32` in `[lo, hi)` from the generator.
fn rand_range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// Draws a point whose components lie in `[-2, 2)`.
fn rand_point(state: &mut u64) -> Vec3 {
    Vec3::new(
        rand_range(state, -2.0, 2.0),
        rand_range(state, -2.0, 2.0),
        rand_range(state, -2.0, 2.0),
    )
}

/// Draws a unit direction by rejection sampling, keeping only draws whose
/// squared length clears `0.1` so the normalization is well conditioned.
fn rand_dir(state: &mut u64) -> Vec3 {
    loop {
        let v = Vec3::new(
            rand_range(state, -1.0, 1.0),
            rand_range(state, -1.0, 1.0),
            rand_range(state, -1.0, 1.0),
        );
        if v.length_squared() > 0.1 {
            return v.normalize_or_zero();
        }
    }
}

/// The cross product of two vectors (the reused vector math exposes no cross).
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

/// A unit vector perpendicular to `dir`, crossing against a reference axis far
/// from parallel so the result is well conditioned.
fn perpendicular(dir: Vec3) -> Vec3 {
    let axis = if dir.x.abs() >= 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    cross(dir, axis).normalize_or_zero()
}

/// Builds an axis-aligned box from a center and (positive) half-extents.
fn box_from(center: Vec3, half: Vec3) -> Aabb {
    Aabb {
        min: center.sub(half),
        max: center.add(half),
    }
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCollision::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn signed_distances_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCollision::new(&ctx);
    check(
        &gpu,
        &ctx,
        CollisionQuery::SdSphere {
            sphere: Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            point: Vec3::new(0.0, 3.0, 0.0),
        },
    );
    check(
        &gpu,
        &ctx,
        CollisionQuery::SdCapsule {
            capsule: Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5),
            point: Vec3::new(0.0, 2.0, 0.0),
        },
    );
    check(
        &gpu,
        &ctx,
        CollisionQuery::SdAabb {
            box_: box_from(Vec3::ZERO, Vec3::splat(1.0)),
            point: Vec3::new(0.0, 3.0, 0.0),
        },
    );
    // Interior point: a strictly negative signed distance.
    check(
        &gpu,
        &ctx,
        CollisionQuery::SdAabb {
            box_: box_from(Vec3::ZERO, Vec3::splat(1.0)),
            point: Vec3::new(0.3, 0.1, 0.1),
        },
    );
}

#[test]
fn closest_points_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCollision::new(&ctx);
    // Interior projection on the segment.
    check(
        &gpu,
        &ctx,
        CollisionQuery::ClosestPointOnSegment {
            a: Vec3::new(-2.0, 0.0, 0.0),
            b: Vec3::new(2.0, 0.0, 0.0),
            point: Vec3::new(0.5, 1.0, 0.0),
        },
    );
    // Beyond an endpoint: the clamp pins to `b`.
    check(
        &gpu,
        &ctx,
        CollisionQuery::ClosestPointOnSegment {
            a: Vec3::new(-2.0, 0.0, 0.0),
            b: Vec3::new(2.0, 0.0, 0.0),
            point: Vec3::new(5.0, 1.0, 0.0),
        },
    );
    check(
        &gpu,
        &ctx,
        CollisionQuery::ClosestPointSphere {
            sphere: Sphere::new(Vec3::ZERO, 1.0),
            point: Vec3::new(0.0, 4.0, 0.0),
        },
    );
    check(
        &gpu,
        &ctx,
        CollisionQuery::ClosestPointCapsule {
            capsule: Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5),
            point: Vec3::new(0.0, 3.0, 0.0),
        },
    );
    check(
        &gpu,
        &ctx,
        CollisionQuery::ClosestPointAabb {
            box_: box_from(Vec3::ZERO, Vec3::splat(1.0)),
            point: Vec3::new(3.0, 0.2, -0.2),
        },
    );
}

#[test]
fn half_space_hit_and_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCollision::new(&ctx);
    let plane = Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0);
    // Signed distance 0.2 < radius 0.5: a clear hit, outward normal +y.
    let hit = CollisionQuery::CollideHalfSpace {
        plane,
        state: ParticleState::new(Vec3::new(0.0, 0.2, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.5),
    };
    let CollisionResult::Contact(c) = cpu_reference(&hit) else {
        panic!("half-space hit yields a contact");
    };
    assert!(c.hit, "fixture must be a clear hit");
    check(&gpu, &ctx, hit);
    // Signed distance 2.0 > radius 0.5: a clear miss.
    let miss = CollisionQuery::CollideHalfSpace {
        plane,
        state: ParticleState::new(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.5),
    };
    let CollisionResult::Contact(c) = cpu_reference(&miss) else {
        panic!("half-space miss yields a contact");
    };
    assert!(!c.hit, "fixture must be a clear miss");
    check(&gpu, &ctx, miss);
}

#[test]
fn sphere_hit_and_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCollision::new(&ctx);
    let sphere = Sphere::new(Vec3::ZERO, 1.0);
    let hit = CollisionQuery::CollideSphere {
        sphere,
        state: ParticleState::new(Vec3::new(0.0, 1.2, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.5),
    };
    let CollisionResult::Contact(c) = cpu_reference(&hit) else {
        panic!("sphere hit yields a contact");
    };
    assert!(c.hit, "fixture must be a clear hit");
    check(&gpu, &ctx, hit);
    let miss = CollisionQuery::CollideSphere {
        sphere,
        state: ParticleState::new(Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.5),
    };
    let CollisionResult::Contact(c) = cpu_reference(&miss) else {
        panic!("sphere miss yields a contact");
    };
    assert!(!c.hit, "fixture must be a clear miss");
    check(&gpu, &ctx, miss);
}

#[test]
fn capsule_hit_and_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCollision::new(&ctx);
    let capsule = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0);
    let hit = CollisionQuery::CollideCapsule {
        capsule,
        state: ParticleState::new(Vec3::new(0.0, 1.2, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.5),
    };
    let CollisionResult::Contact(c) = cpu_reference(&hit) else {
        panic!("capsule hit yields a contact");
    };
    assert!(c.hit, "fixture must be a clear hit");
    check(&gpu, &ctx, hit);
    let miss = CollisionQuery::CollideCapsule {
        capsule,
        state: ParticleState::new(Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.5),
    };
    let CollisionResult::Contact(c) = cpu_reference(&miss) else {
        panic!("capsule miss yields a contact");
    };
    assert!(!c.hit, "fixture must be a clear miss");
    check(&gpu, &ctx, miss);
}

#[test]
fn aabb_hit_miss_and_interior() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCollision::new(&ctx);
    let box_ = box_from(Vec3::ZERO, Vec3::splat(1.0));
    // Outside the +x face by 0.2 with radius 0.5: a clear hit, normal +x.
    let hit = CollisionQuery::CollideAabb {
        box_,
        state: ParticleState::new(Vec3::new(1.2, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 0.5),
    };
    let CollisionResult::Contact(c) = cpu_reference(&hit) else {
        panic!("box hit yields a contact");
    };
    assert!(c.hit, "fixture must be a clear hit");
    check(&gpu, &ctx, hit);
    // Clearly outside: a miss.
    let miss = CollisionQuery::CollideAabb {
        box_,
        state: ParticleState::new(Vec3::new(3.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 0.5),
    };
    let CollisionResult::Contact(c) = cpu_reference(&miss) else {
        panic!("box miss yields a contact");
    };
    assert!(!c.hit, "fixture must be a clear miss");
    check(&gpu, &ctx, miss);
    // Interior, off-center so the +x face is strictly nearest (distance -0.5 vs
    // -0.9 on y and z): avoids any signed-zero ambiguity in the exit normal.
    let inside = CollisionQuery::CollideAabb {
        box_,
        state: ParticleState::new(Vec3::new(0.5, 0.1, 0.1), Vec3::new(0.0, 0.0, 0.0), 0.5),
    };
    let CollisionResult::Contact(c) = cpu_reference(&inside) else {
        panic!("box interior yields a contact");
    };
    assert!(c.hit, "an interior particle always contacts");
    check(&gpu, &ctx, inside);
}

#[test]
fn sdf_hit_and_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCollision::new(&ctx);
    let hit = CollisionQuery::CollideSdf {
        sample: SdfSample::new(0.2, Vec3::new(0.0, 1.0, 0.0)),
        state: ParticleState::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.5),
    };
    let CollisionResult::Contact(c) = cpu_reference(&hit) else {
        panic!("sdf hit yields a contact");
    };
    assert!(c.hit, "fixture must be a clear hit");
    check(&gpu, &ctx, hit);
    let miss = CollisionQuery::CollideSdf {
        sample: SdfSample::new(2.0, Vec3::new(0.0, 1.0, 0.0)),
        state: ParticleState::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.5),
    };
    let CollisionResult::Contact(c) = cpu_reference(&miss) else {
        panic!("sdf miss yields a contact");
    };
    assert!(!c.hit, "fixture must be a clear miss");
    check(&gpu, &ctx, miss);
}

#[test]
fn resolve_contact_bounce_and_friction() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCollision::new(&ctx);
    // Approaching along -y with a tangential +x slide: restitution reflects the
    // normal component, friction trims the slide within Coulomb's cone.
    let hit = CollisionQuery::ResolveContact {
        state: ParticleState::new(Vec3::new(0.0, 0.1, 0.0), Vec3::new(1.0, -2.0, 0.0), 0.5),
        contact: Contact::hit(Vec3::new(0.0, 1.0, 0.0), 0.3, Vec3::ZERO),
        params: ResponseParams::new(0.5, 0.25),
    };
    let CollisionResult::Response(r) = cpu_reference(&hit) else {
        panic!("resolve yields a response");
    };
    assert!(r.hit, "a hit contact resolves to a hit response");
    check(&gpu, &ctx, hit);
    // A miss contact leaves the particle unchanged.
    let miss = CollisionQuery::ResolveContact {
        state: ParticleState::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, -2.0, 0.0), 0.5),
        contact: Contact::miss(),
        params: ResponseParams::new(0.5, 0.25),
    };
    let CollisionResult::Response(r) = cpu_reference(&miss) else {
        panic!("resolve yields a response");
    };
    assert!(!r.hit, "a miss contact leaves the particle unchanged");
    check(&gpu, &ctx, miss);
}

#[test]
fn resolve_particle_batch_picks_deepest() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCollision::new(&ctx);
    let state = ParticleState::new(Vec3::new(0.0, -0.4, 0.0), Vec3::new(0.0, -3.0, 0.0), 0.5);
    // Half-space penetration 0.9 (deepest), sphere grazes at 0.1, box misses.
    let half_space = Collider::HalfSpace(Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0));
    let sphere = Collider::Sphere(Sphere::new(Vec3::new(3.0, -0.4, 0.0), 2.6));
    let far_box = Collider::Box(box_from(Vec3::new(10.0, 10.0, 10.0), Vec3::splat(0.5)));
    let hit = CollisionQuery::ResolveParticleCollision {
        state,
        colliders: vec![sphere, half_space, far_box],
        params: ResponseParams::new(0.5, 0.25),
    };
    let CollisionResult::Response(r) = cpu_reference(&hit) else {
        panic!("batch yields a response");
    };
    assert!(r.hit, "the deepest contact drives a hit response");
    check(&gpu, &ctx, hit);
    // Nothing within reach: an unchanged response.
    let all_miss = CollisionQuery::ResolveParticleCollision {
        state,
        colliders: vec![
            Collider::Sphere(Sphere::new(Vec3::new(20.0, 0.0, 0.0), 1.0)),
            Collider::Box(box_from(Vec3::new(-20.0, 0.0, 0.0), Vec3::splat(0.5))),
        ],
        params: ResponseParams::new(0.5, 0.25),
    };
    let CollisionResult::Response(r) = cpu_reference(&all_miss) else {
        panic!("batch yields a response");
    };
    assert!(!r.hit, "an all-miss batch leaves the particle unchanged");
    check(&gpu, &ctx, all_miss);
}

#[test]
fn swept_plane_tunnel_and_separating() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCollision::new(&ctx);
    let plane = Plane::new(Vec3::new(0.0, 1.0, 0.0), 0.0);
    // Starts at y=3 (free side, signed distance >= radius) and sweeps to y=-1,
    // crossing the radius 0.5 shell: toi 0.625, clear of 0 and 1.
    let tunnel = CollisionQuery::SweptSphereVsPlane {
        plane,
        p0: Vec3::new(0.0, 3.0, 0.0),
        p1: Vec3::new(0.0, -1.0, 0.0),
        radius: 0.5,
    };
    let CollisionResult::Swept { hit, .. } = cpu_reference(&tunnel) else {
        panic!("swept plane yields a sweep");
    };
    assert!(hit, "the sweep must tunnel through the plane");
    check(&gpu, &ctx, tunnel);
    // Starts free and moves further away: no crossing.
    let separating = CollisionQuery::SweptSphereVsPlane {
        plane,
        p0: Vec3::new(0.0, 3.0, 0.0),
        p1: Vec3::new(0.0, 5.0, 0.0),
        radius: 0.5,
    };
    let CollisionResult::Swept { hit, .. } = cpu_reference(&separating) else {
        panic!("swept plane yields a sweep");
    };
    assert!(!hit, "a separating sweep never touches the plane");
    check(&gpu, &ctx, separating);
}

#[test]
fn swept_sphere_pass_overlap_and_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCollision::new(&ctx);
    let target = Sphere::new(Vec3::ZERO, 1.0);
    // Passes straight through: toi 0.35, clear of 0 and 1.
    let pass = CollisionQuery::SweptSphereVsSphere {
        p0: Vec3::new(-5.0, 0.0, 0.0),
        p1: Vec3::new(5.0, 0.0, 0.0),
        radius: 0.5,
        target,
    };
    let CollisionResult::Swept { hit, .. } = cpu_reference(&pass) else {
        panic!("swept sphere yields a sweep");
    };
    assert!(hit, "the sweep must pass through the target");
    check(&gpu, &ctx, pass);
    // Already overlapping at the start: toi 0.
    let overlap = CollisionQuery::SweptSphereVsSphere {
        p0: Vec3::new(0.0, 0.0, 0.0),
        p1: Vec3::new(5.0, 0.0, 0.0),
        radius: 0.5,
        target,
    };
    let CollisionResult::Swept { hit, toi } = cpu_reference(&overlap) else {
        panic!("swept sphere yields a sweep");
    };
    assert!(hit && approx(toi, 0.0), "an initial overlap reports toi 0");
    check(&gpu, &ctx, overlap);
    // Parallel miss offset in y: never within the combined radius.
    let miss = CollisionQuery::SweptSphereVsSphere {
        p0: Vec3::new(-5.0, 3.0, 0.0),
        p1: Vec3::new(5.0, 3.0, 0.0),
        radius: 0.5,
        target,
    };
    let CollisionResult::Swept { hit, .. } = cpu_reference(&miss) else {
        panic!("swept sphere yields a sweep");
    };
    assert!(!hit, "an offset parallel sweep misses the target");
    check(&gpu, &ctx, miss);
}

/// Appends one clear query per routine, with the hit-flagged routines flipped
/// between a constructive hit and a constructive miss by `hit`. Every contact
/// and sweep is kept at least `BAND` from its boundary.
#[expect(
    clippy::too_many_lines,
    reason = "one builder appends a fixture for each of the sixteen routines"
)]
fn push_suite(state: &mut u64, hit: bool, queries: &mut Vec<CollisionQuery>) {
    // 1-3: signed distances (continuous, no hit flag).
    let sc = rand_point(state);
    let sr = rand_range(state, 0.5, 1.5);
    let sdir = rand_dir(state);
    queries.push(CollisionQuery::SdSphere {
        sphere: Sphere::new(sc, sr),
        point: sc.add(sdir.scale(sr + rand_range(state, 0.5, 2.0))),
    });
    let ca = rand_point(state);
    let cb = ca.add(rand_dir(state).scale(rand_range(state, 1.0, 3.0)));
    let cr = rand_range(state, 0.3, 0.9);
    queries.push(CollisionQuery::SdCapsule {
        capsule: Capsule::new(ca, cb, cr),
        point: ca.add(perpendicular(cb.sub(ca)).scale(cr + rand_range(state, 0.5, 2.0))),
    });
    let bc = rand_point(state);
    let be = Vec3::new(
        rand_range(state, 0.5, 1.5),
        rand_range(state, 0.5, 1.5),
        rand_range(state, 0.5, 1.5),
    );
    queries.push(CollisionQuery::SdAabb {
        box_: box_from(bc, be),
        point: bc.add(Vec3::new(be.x + rand_range(state, 0.5, 2.0), 0.0, 0.0)),
    });

    // 4-7: closest points (continuous).
    let a = rand_point(state);
    let b = a.add(rand_dir(state).scale(rand_range(state, 1.0, 3.0)));
    queries.push(CollisionQuery::ClosestPointOnSegment {
        a,
        b,
        point: a
            .add(b.sub(a).scale(0.5))
            .add(perpendicular(b.sub(a)).scale(rand_range(state, 0.5, 2.0))),
    });
    let qc = rand_point(state);
    let qr = rand_range(state, 0.5, 1.5);
    queries.push(CollisionQuery::ClosestPointSphere {
        sphere: Sphere::new(qc, qr),
        point: qc.add(rand_dir(state).scale(qr + rand_range(state, 0.5, 2.0))),
    });
    let pa = rand_point(state);
    let pb = pa.add(rand_dir(state).scale(rand_range(state, 1.0, 3.0)));
    let pr = rand_range(state, 0.3, 0.9);
    queries.push(CollisionQuery::ClosestPointCapsule {
        capsule: Capsule::new(pa, pb, pr),
        point: pa.add(perpendicular(pb.sub(pa)).scale(pr + rand_range(state, 0.5, 2.0))),
    });
    let xc = rand_point(state);
    let xe = Vec3::new(
        rand_range(state, 0.5, 1.5),
        rand_range(state, 0.5, 1.5),
        rand_range(state, 0.5, 1.5),
    );
    queries.push(CollisionQuery::ClosestPointAabb {
        box_: box_from(xc, xe),
        point: xc.add(Vec3::new(xe.x + rand_range(state, 0.5, 2.0), 0.0, 0.0)),
    });

    // Shared particle radius for the contact predicates.
    let radius = rand_range(state, 0.4, 0.9);

    // 8: half-space contact. `sd` chosen directly, then `d` back-solved.
    let hn = rand_dir(state);
    let hpos = rand_point(state);
    let sd = if hit {
        radius - rand_range(state, BAND, 2.0)
    } else {
        radius + rand_range(state, BAND, 2.0)
    };
    let hd = sd - hn.dot(hpos);
    queries.push(CollisionQuery::CollideHalfSpace {
        plane: Plane::new(hn, hd),
        state: ParticleState::new(hpos, rand_point(state), radius),
    });

    // 9: sphere contact. Place the particle at a clear hit or miss distance.
    let scen = rand_point(state);
    let srad = rand_range(state, 0.5, 1.5);
    let sdirc = rand_dir(state);
    let sdist = if hit {
        srad + radius * 0.3
    } else {
        srad + radius + rand_range(state, BAND, 2.0)
    };
    queries.push(CollisionQuery::CollideSphere {
        sphere: Sphere::new(scen, srad),
        state: ParticleState::new(scen.add(sdirc.scale(sdist)), rand_point(state), radius),
    });

    // 10: capsule contact, offset perpendicular from an interior axis point.
    let kca = rand_point(state);
    let kcb = kca.add(rand_dir(state).scale(rand_range(state, 1.5, 3.0)));
    let krad = rand_range(state, 0.5, 1.0);
    let s = rand_range(state, 0.3, 0.7);
    let axis = kca.add(kcb.sub(kca).scale(s));
    let perp = perpendicular(kcb.sub(kca));
    let kdist = if hit {
        krad + radius * 0.3
    } else {
        krad + radius + rand_range(state, BAND, 2.0)
    };
    queries.push(CollisionQuery::CollideCapsule {
        capsule: Capsule::new(kca, kcb, krad),
        state: ParticleState::new(axis.add(perp.scale(kdist)), rand_point(state), radius),
    });

    // 11: box contact, approaching the +x face with y and z inside the slab.
    let abc = rand_point(state);
    let abe = Vec3::new(
        rand_range(state, 0.6, 1.5),
        rand_range(state, 0.6, 1.5),
        rand_range(state, 0.6, 1.5),
    );
    let gap = if hit {
        radius * 0.3
    } else {
        radius + rand_range(state, BAND, 2.0)
    };
    let bpos = Vec3::new(
        abc.x + abe.x + gap,
        abc.y + rand_range(state, -0.4, 0.4) * abe.y,
        abc.z + rand_range(state, -0.4, 0.4) * abe.z,
    );
    queries.push(CollisionQuery::CollideAabb {
        box_: box_from(abc, abe),
        state: ParticleState::new(bpos, rand_point(state), radius),
    });

    // 12: SDF contact; the sampled distance decides the verdict.
    let fdist = if hit {
        radius * 0.3
    } else {
        radius + rand_range(state, BAND, 2.0)
    };
    queries.push(CollisionQuery::CollideSdf {
        sample: SdfSample::new(fdist, rand_dir(state)),
        state: ParticleState::new(rand_point(state), rand_point(state), radius),
    });

    // 13: single-contact resolve, hit or miss contact.
    let rn = rand_dir(state);
    let rtan = perpendicular(rn);
    let rvel = rn
        .scale(-rand_range(state, 0.5, 3.0))
        .add(rtan.scale(rand_range(state, 0.3, 2.0)));
    let rstate = ParticleState::new(rand_point(state), rvel, radius);
    let rparams = ResponseParams::new(rand_range(state, 0.0, 1.0), rand_range(state, 0.0, 1.0));
    let rcontact = if hit {
        Contact::hit(rn, rand_range(state, BAND, 1.0), rand_point(state))
    } else {
        Contact::miss()
    };
    queries.push(CollisionQuery::ResolveContact {
        state: rstate,
        contact: rcontact,
        params: rparams,
    });

    // 14: batch resolve. A half-space is the only reachable collider; it hits or
    // misses by construction while two decoy colliders stay far away.
    let bn = rand_dir(state);
    let bpos2 = rand_point(state);
    let bsd = if hit {
        radius - rand_range(state, BAND, 1.0)
    } else {
        radius + rand_range(state, BAND, 2.0)
    };
    let bd = bsd - bn.dot(bpos2);
    let bvel = bn.scale(-rand_range(state, 0.5, 3.0));
    queries.push(CollisionQuery::ResolveParticleCollision {
        state: ParticleState::new(bpos2, bvel, radius),
        colliders: vec![
            Collider::HalfSpace(Plane::new(bn, bd)),
            Collider::Sphere(Sphere::new(bpos2.add(Vec3::new(50.0, 0.0, 0.0)), 1.0)),
            Collider::Box(box_from(
                bpos2.add(Vec3::new(-50.0, 0.0, 0.0)),
                Vec3::splat(0.5),
            )),
        ],
        params: ResponseParams::new(rand_range(state, 0.0, 1.0), rand_range(state, 0.0, 1.0)),
    });

    // 15: swept sphere vs plane, moving along the plane normal.
    let wn = rand_dir(state);
    let wrad = rand_range(state, 0.3, 0.8);
    let wp0 = rand_point(state);
    let d0 = wrad + rand_range(state, BAND, 2.0);
    let d1 = if hit {
        wrad - rand_range(state, BAND, 2.0)
    } else {
        d0 + rand_range(state, BAND, 2.0)
    };
    // Place the plane so `signed_distance(wp0) == d0`, then move along `wn` by
    // `d1 - d0` so `signed_distance(wp1) == d1`.
    let wd = d0 - wn.dot(wp0);
    let wp1 = wp0.add(wn.scale(d1 - d0));
    queries.push(CollisionQuery::SweptSphereVsPlane {
        plane: Plane::new(wn, wd),
        p0: wp0,
        p1: wp1,
        radius: wrad,
    });

    // 16: swept sphere vs sphere, passing through or receding.
    let tc = rand_point(state);
    let trad = rand_range(state, 0.5, 1.5);
    let prad = rand_range(state, 0.3, 0.8);
    let combined = trad + prad;
    let dir = rand_dir(state);
    let d_start = combined + rand_range(state, 0.5, 2.0);
    let (ep0, ep1) = if hit {
        let d_end = combined + rand_range(state, 0.5, 2.0);
        (tc.add(dir.scale(d_start)), tc.sub(dir.scale(d_end)))
    } else {
        let extra = rand_range(state, 0.5, 2.0);
        (
            tc.add(dir.scale(d_start)),
            tc.add(dir.scale(d_start + extra)),
        )
    };
    queries.push(CollisionQuery::SweptSphereVsSphere {
        p0: ep0,
        p1: ep1,
        radius: prad,
        target: Sphere::new(tc, trad),
    });
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCollision::new(&ctx);
    let mut state = 0x_c011_1510_0f0f_0001_u64;
    let mut queries = Vec::new();
    for round in 0..24 {
        push_suite(&mut state, round % 2 == 0, &mut queries);
    }
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");

    let mut saw_contact_hit = false;
    let mut saw_contact_miss = false;
    let mut saw_sweep_hit = false;
    let mut saw_sweep_miss = false;
    for (q, g) in queries.iter().zip(got.iter()) {
        let want = cpu_reference(q);
        match want {
            CollisionResult::Contact(c) => {
                if c.hit {
                    saw_contact_hit = true;
                } else {
                    saw_contact_miss = true;
                }
            }
            CollisionResult::Swept { hit, .. } => {
                if hit {
                    saw_sweep_hit = true;
                } else {
                    saw_sweep_miss = true;
                }
            }
            _ => {}
        }
        assert!(
            results_match(g, &want),
            "GPU result {g:?} must match CPU {want:?} for {q:?}"
        );
    }
    assert!(
        saw_contact_hit && saw_contact_miss,
        "the batch must exercise both contact verdicts"
    );
    assert!(
        saw_sweep_hit && saw_sweep_miss,
        "the batch must exercise both sweep verdicts"
    );
}
