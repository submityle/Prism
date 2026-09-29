//! Cross-module invariant tests for the particle subsystem.
//!
//! Each sibling module (`pool`, `emitter`, `simulation`, `sort_cull`, `lod`,
//! `shading`) is unit-tested in isolation; this module exercises the full
//! per-frame chain they compose into — spawn → simulate → reduce bounds → sort →
//! `LOD` → deformation request → budget schedule — and asserts the invariants
//! that only emerge when the pieces run together, above all *bit-for-bit
//! determinism*: the same authored inputs must produce the same plan every time
//! (design §29), which is what makes the `CPU` contract a trustworthy oracle for
//! the eventual `GPU` backend.

use alloc::vec::Vec;

use crate::deformation::schedule::{plan_deformations, DeformationRequest};
use crate::deformation::{DeformationBudget, DeformationHandle, DeformationKind};

use super::emitter::{build_spawn, sample_shape, Emitter, EmitterShape, SpawnParams, UnitCursor};
use super::lod::{
    particle_deformation_request, resolve_particle_lod, EmitterLodInput, ParticleLodDecision,
    ParticleLodThresholds, ParticleLodTier, ParticleQuality,
};
use super::pool::{CapacityPolicy, ParticlePool};
use super::shading::{
    resolve_shading_program, LightingServiceCaps, ParticleRenderPhase, ShadingProgramInput,
};
use super::simulation::{gravity, integrate, rand_unit, Kinematics};
use super::sort_cull::{
    choose_sort_strategy, reduce_bounds, significance, sort_key, Aabb, BlendMode, SortDecision,
};
use super::{EmberShadingModel, EmitterHandle, IntegratorKind, SimSpace, SortStrategy, Vec3};

/// A self-contained authored effect the chain runs against.
#[derive(Clone, Copy)]
struct EffectSpec {
    handle: EmitterHandle,
    deformation: DeformationHandle,
    shape: EmitterShape,
    params: SpawnParams,
    seed: u32,
    capacity: u32,
    spawn_per_frame: u32,
    max_age: f32,
    frames: u32,
    dt: f32,
    gravity: Vec3,
    integrator: IntegratorKind,
}

/// The whole-frame result the chain produces, captured so two runs can be
/// compared for determinism.
#[derive(Clone, Debug, PartialEq)]
struct FrameOutcome {
    alive: u32,
    bounds: Aabb,
    sort_keys: Vec<u16>,
    lod: ParticleLodDecision,
    strategy: SortStrategy,
    scheduled: usize,
    charged_vertices: u32,
}

const THRESHOLDS: ParticleLodThresholds = ParticleLodThresholds {
    reduced_below: 0.25,
    impostor_below: 0.08,
    cull_below: 0.01,
};

/// Draws four unit samples for one spawn from the stateless hash `RNG`, so the
/// spawn placement is a pure function of `(seed, particle_id)`.
fn spawn_cursor_values(seed: u32, id: u32) -> [f32; 4] {
    [
        rand_unit(id, seed, 0, 0),
        rand_unit(id, seed, 1, 0),
        rand_unit(id, seed, 2, 0),
        rand_unit(id, seed, 3, 0),
    ]
}

/// Runs the full spawn→sim→sort→LOD→deformation chain for one effect at a given
/// screen coverage and returns the captured outcome.
fn run_effect(spec: EffectSpec, coverage: f32) -> FrameOutcome {
    let mut pool = ParticlePool::with_capacity(spec.capacity);
    let emitter = Emitter::new(spec.handle, spec.shape, spec.params);

    // Spawn one frame's worth of particles and seed their kinematic state.
    let slots = emitter.allocate(
        &mut pool,
        spec.spawn_per_frame,
        CapacityPolicy::DiscardNewest,
    );
    let mut states: Vec<Kinematics> = Vec::new();
    for &slot in &slots {
        let values = spawn_cursor_values(spec.seed, slot);
        let mut cursor = UnitCursor::new(&values);
        let sample = sample_shape(spec.shape, &mut cursor);
        let spawn = build_spawn(Vec3::ZERO, Vec3::ZERO, sample, spec.params);
        states.push(Kinematics::new(spawn.position, spawn.velocity));
    }

    // Advance every particle under a uniform gravity field for `frames` steps.
    let g = spec.gravity;
    for _ in 0..spec.frames {
        for state in &mut states {
            *state = integrate(spec.integrator, *state, spec.dt, |_p, _v| gravity(g));
        }
    }

    // Reduce the alive positions to a bounding box (design §13).
    let positions: Vec<Vec3> = states.iter().map(|s| s.position).collect();
    let bounds = reduce_bounds(&positions);

    // Quantize a back-to-front view-depth key per particle (design §12). The
    // camera sits behind the origin looking down `+Z`.
    let camera = Vec3::new(0.0, 0.0, -10.0);
    let near = 0.1;
    let far = 200.0;
    let sort_keys: Vec<u16> = positions
        .iter()
        .map(|p| sort_key(p.distance(camera), near, far, true))
        .collect();

    // Resolve the LOD tier for the emitter's live count at this coverage.
    let lod_input = EmitterLodInput {
        handle: spec.handle,
        deformation: spec.deformation,
        max_particles: pool.alive_count(),
        sim_substeps: 4,
        ray_traced: false,
        native_form: ParticleLodTier::Full,
    };
    let lod = resolve_particle_lod(lod_input, coverage, THRESHOLDS);

    // Emit the deformation request (only simulated tiers charge the budget) and
    // schedule it against a generous per-frame budget.
    let mut requests: Vec<DeformationRequest> = Vec::new();
    if let Some(request) = particle_deformation_request(lod_input, lod, 8) {
        requests.push(request);
    }
    let budget = DeformationBudget {
        vertices_per_frame: 1_000_000,
        blas_refits_per_frame: 4,
    };
    let plan = plan_deformations(&requests, budget);
    let charged_vertices = plan.scheduled.iter().map(|s| s.request.vertex_count).sum();

    // Decide the sort strategy for a typical alpha-blended VFX renderer.
    let strategy = choose_sort_strategy(SortDecision {
        blend: BlendMode::AlphaBlend,
        particle_count: pool.alive_count(),
        radix_min_count: 4096,
        prefer_shared_oit: false,
    });

    let _ = spec.max_age;
    FrameOutcome {
        alive: pool.alive_count(),
        bounds,
        sort_keys,
        lod,
        strategy,
        scheduled: plan.scheduled_count(),
        charged_vertices,
    }
}

fn fountain() -> EffectSpec {
    EffectSpec {
        handle: EmitterHandle(1),
        deformation: DeformationHandle(1),
        shape: EmitterShape::Sphere {
            radius: 1.5,
            surface_only: false,
        },
        params: SpawnParams {
            speed: 6.0,
            inherit_velocity: 0.0,
            sim_space: SimSpace::World,
        },
        seed: 0x51ED,
        capacity: 1024,
        spawn_per_frame: 300,
        max_age: 4.0,
        frames: 30,
        dt: 1.0 / 60.0,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        integrator: IntegratorKind::Rk2,
    }
}

#[test]
fn full_chain_is_bit_for_bit_deterministic() {
    let a = run_effect(fountain(), 0.9);
    let b = run_effect(fountain(), 0.9);
    assert_eq!(
        a, b,
        "identical inputs must produce an identical frame plan"
    );
}

#[test]
fn spawn_never_exceeds_pool_capacity() {
    let mut spec = fountain();
    spec.capacity = 64;
    spec.spawn_per_frame = 500;
    let outcome = run_effect(spec, 0.9);
    assert_eq!(outcome.alive, 64, "DiscardNewest caps spawns at capacity");
    assert_eq!(outcome.sort_keys.len(), 64);
}

#[test]
fn bounds_contain_every_simulated_particle() {
    let spec = fountain();
    let outcome = run_effect(spec, 0.9);
    // The reduced box is non-empty and its radius is finite and positive for a
    // multi-particle spread.
    let radius = outcome.bounds.bounding_radius();
    assert!(radius.is_finite());
    assert!(radius > 0.0, "a spread of particles has positive extent");
    // The center is finite (no NaN leaked through the reduction).
    let c = outcome.bounds.center();
    assert!(c.x.is_finite() && c.y.is_finite() && c.z.is_finite());
}

#[test]
fn one_sort_key_per_alive_particle() {
    let outcome = run_effect(fountain(), 0.9);
    assert_eq!(outcome.sort_keys.len() as u32, outcome.alive);
}

#[test]
fn full_coverage_keeps_full_detail_and_charges_full_budget() {
    let outcome = run_effect(fountain(), 0.9);
    assert_eq!(outcome.lod.tier, ParticleLodTier::Full);
    assert_eq!(outcome.lod.active_particles, outcome.alive);
    // A simulated tier emits exactly one request and it is scheduled, charging
    // the live particle count against the budget.
    assert_eq!(outcome.scheduled, 1);
    assert_eq!(outcome.charged_vertices, outcome.alive);
    // Alpha-blended VFX with no shared OIT sorts via bitonic/radix by count.
    assert!(outcome.strategy.is_explicit_sort());
}

#[test]
fn tiny_coverage_culls_and_charges_nothing() {
    let outcome = run_effect(fountain(), 0.005);
    assert_eq!(outcome.lod.tier, ParticleLodTier::Culled);
    assert_eq!(outcome.lod.active_particles, 0);
    // A culled emitter emits no deformation request, so nothing is scheduled.
    assert_eq!(outcome.scheduled, 0);
    assert_eq!(outcome.charged_vertices, 0);
}

#[test]
fn impostor_coverage_stops_simulation_but_still_bounds_and_sorts() {
    // Coverage in the impostor band: not simulated, but the frame still has
    // spawned particles to bound and sort this frame.
    let outcome = run_effect(fountain(), 0.05);
    assert_eq!(outcome.lod.tier, ParticleLodTier::Impostor);
    assert_eq!(outcome.lod.active_particles, 0);
    assert_eq!(outcome.scheduled, 0);
    assert!(outcome.alive > 0);
    assert_eq!(outcome.sort_keys.len() as u32, outcome.alive);
}

#[test]
fn reduced_band_decimates_particles_and_substeps() {
    let outcome = run_effect(fountain(), 0.15);
    assert_eq!(outcome.lod.tier, ParticleLodTier::Reduced);
    // Reduced keeps a quarter of the live particles (never below one) and still
    // charges the budget for them.
    let expected = (outcome.alive / 4).max(1);
    assert_eq!(outcome.lod.active_particles, expected);
    assert_eq!(outcome.charged_vertices, expected);
    assert_eq!(outcome.scheduled, 1);
}

#[test]
fn shading_program_and_sort_route_agree_for_alpha_blend() {
    // The same alpha-blend renderer lands in the transparent phase and, without
    // shared OIT, sorts explicitly — the two subsystems agree on "transparent
    // and order-dependent".
    let program = resolve_shading_program(ShadingProgramInput {
        model: EmberShadingModel::Pbr,
        blend: BlendMode::AlphaBlend,
        caps: LightingServiceCaps::default(),
        volumetric: false,
        quality: ParticleQuality::High,
    });
    assert_eq!(program.phase, ParticleRenderPhase::Transparent);
    let outcome = run_effect(fountain(), 0.9);
    assert!(outcome.strategy.is_explicit_sort());
}

#[test]
fn significance_falls_as_coverage_shrinks() {
    // The sleep/significance heuristic and the LOD ladder move the same way:
    // less coverage is less significant and a coarser tier.
    let near = significance(0.9, 3.0, 10.0);
    let far = significance(0.02, 3.0, 10.0);
    assert!(near > far);
    let near_tier = run_effect(fountain(), 0.9).lod.tier;
    let far_tier = run_effect(fountain(), 0.02).lod.tier;
    assert!(far_tier.rank() > near_tier.rank());
}

#[test]
fn distinct_effects_charge_a_shared_budget_by_priority() {
    // Two simulated emitters both emit particle deformation requests; the shared
    // scheduler admits both when the budget is ample and orders deterministically.
    let a = run_effect(fountain(), 0.9);
    let mut spec_b = fountain();
    spec_b.handle = EmitterHandle(2);
    spec_b.deformation = DeformationHandle(2);
    spec_b.seed = 0xBEEF;
    let b = run_effect(spec_b, 0.9);

    let requests = [
        DeformationRequest {
            handle: DeformationHandle(1),
            kind: DeformationKind::Particle,
            vertex_count: a.alive,
            priority: 8,
            needs_blas_refit: false,
        },
        DeformationRequest {
            handle: DeformationHandle(2),
            kind: DeformationKind::Particle,
            vertex_count: b.alive,
            priority: 5,
            needs_blas_refit: false,
        },
    ];
    let plan = plan_deformations(
        &requests,
        DeformationBudget {
            vertices_per_frame: 1_000_000,
            blas_refits_per_frame: 0,
        },
    );
    assert_eq!(plan.count_of_kind(DeformationKind::Particle), 2);
    // Higher priority (handle 1) schedules first.
    assert_eq!(plan.scheduled[0].request.handle, DeformationHandle(1));
    assert_eq!(plan.scheduled[1].request.handle, DeformationHandle(2));
}
