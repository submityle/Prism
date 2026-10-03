//! Real-device parity for the cloth isometric-bending twin:
//! [`GpuClothBendingApply`](prism_volumetric_gpu::cloth_bending_apply::GpuClothBendingApply)
//! must reproduce the `CPU` golden
//! [`apply_bending`](prism_render_architecture::cloth::bending::apply_bending)
//! across single- and multi-hinge systems, the clamped sweep count, the
//! `dt <= 0` and all-pinned no-ops, an out-of-bounds stencil slot, and a
//! randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected positions and first-sweep energy are produced by calling the
//! golden `apply_bending` directly on a cloned particle set built from the
//! hand-rolled render [`ClothParticle`](prism_render_architecture::cloth::ClothParticle)
//! and [`BendingConstraint`](prism_render_architecture::cloth::bending::BendingConstraint),
//! so each test pins `GPU == golden`, not merely that the shader compiles. Real
//! hinge weights and scales come from
//! [`build_dihedral_bending`](prism_render_architecture::cloth::bending::build_dihedral_bending)
//! on a flat rest mesh; the particle positions are then folded so the bend
//! vector `S` is non-trivial.
//!
//! # Parity criterion
//!
//! Positions and energy are continuous `f32` quantities compared with the
//! tolerance `abs <= 2e-4` or `rel <= 2e-3` (relative floor `1e-6`), relaxed
//! modestly from the base `1e-4`/`1e-3` because multiple in-place Gauss-Seidel
//! sweeps accumulate rounding. The randomized sweep rejects fixtures whose per
//! hinge `|S|²` sits near the `1e-12` degeneracy threshold and keeps `dt > 0`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::bending`；无第三方引擎源码或衍生代码。

use prism_render_architecture::cloth::bending::{
    apply_bending, build_dihedral_bending, BendingConstraint,
};
use prism_render_architecture::cloth::{ClothParticle, Compliance, Vec3};
use prism_volumetric_gpu::cloth_bending_apply::{
    ClothBendingApplyQuery, ClothBendingApplyResult, GpuClothBendingApply,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for a continuous `f32` parity comparison (relaxed for the
/// multi-sweep accumulation).
const EPS: f32 = 2e-4;
/// Relative tolerance for a continuous `f32` parity comparison.
const REL: f32 = 2e-3;
/// Floor on the relative-tolerance denominator so tiny magnitudes stay stable.
const REL_FLOOR: f32 = 1e-6;

/// Returns `true` when `a` and `b` agree within the continuous tolerance
/// (`abs <= 2e-4` or `rel <= 2e-3`, floor `1e-6`).
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// One cloth system under test: its particles, bending hinges, sweep count and
/// substep `dt`.
struct System {
    /// Pre-solve particles.
    particles: Vec<ClothParticle>,
    /// Bending hinges projected in slice order.
    constraints: Vec<BendingConstraint>,
    /// Gauss-Seidel sweep count.
    iterations: u32,
    /// Substep `dt`.
    dt: f32,
}

/// Builds the matching [`ClothBendingApplyQuery`] from a [`System`], flattening
/// the particles and hinges into the twin's parallel vectors.
fn to_query(system: &System) -> ClothBendingApplyQuery {
    ClothBendingApplyQuery {
        positions: system
            .particles
            .iter()
            .map(|p| [p.position.x, p.position.y, p.position.z])
            .collect(),
        inverse_masses: system.particles.iter().map(|p| p.inverse_mass).collect(),
        vertices: system.constraints.iter().map(|c| c.vertices).collect(),
        weights: system.constraints.iter().map(|c| c.weights).collect(),
        scales: system.constraints.iter().map(|c| c.scale).collect(),
        compliances: system
            .constraints
            .iter()
            .map(|c| c.compliance.value())
            .collect(),
        iterations: system.iterations,
        dt: system.dt,
    }
}

/// Runs the golden `apply_bending` on a clone of the system's particles,
/// returning the solved positions and the first-sweep energy.
fn oracle(system: &System) -> (Vec<[f32; 3]>, f32) {
    let mut particles = system.particles.clone();
    let first = apply_bending(
        &mut particles,
        &system.constraints,
        system.iterations,
        system.dt,
    );
    let positions = particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect();
    (positions, first)
}

/// Pins one `GPU` result against the in-host oracle, position-by-position and
/// on the first-sweep energy.
fn check(idx: usize, system: &System, got: &ClothBendingApplyResult) {
    let (want_pos, want_energy) = oracle(system);
    assert_eq!(
        got.positions.len(),
        want_pos.len(),
        "system {idx}: position count mismatch",
    );
    for (p, (g, w)) in got.positions.iter().zip(want_pos.iter()).enumerate() {
        for axis in 0..3 {
            assert!(
                close(g[axis], w[axis]),
                "system {idx} particle {p} axis {axis}: gpu {g:?} golden {w:?}",
            );
        }
    }
    assert!(
        close(got.first_energy, want_energy),
        "system {idx} first_energy: gpu {} golden {want_energy}",
        got.first_energy,
    );
}

/// Builds the one-hinge flat-rest quad and returns its bending constraints.
///
/// The rest mesh is a planar unit square split along its diagonal; the single
/// interior edge yields one hinge whose cotangent weights and area scale are
/// used verbatim with folded particle positions.
fn quad_hinge(compliance: f32) -> Vec<BendingConstraint> {
    let rest = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
    ];
    let triangles = [[0u32, 1, 2], [1, 3, 2]];
    build_dihedral_bending(&rest, &triangles, Compliance(compliance))
}

/// Builds the folded four-particle quad with a uniform inverse mass; the apex
/// is lifted out of plane so the bend vector `S` is non-trivial.
fn quad_particles(inv_mass: f32, fold: f32) -> Vec<ClothParticle> {
    vec![
        ClothParticle::new(Vec3::new(0.0, 0.0, 0.0), inv_mass),
        ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), inv_mass),
        ClothParticle::new(Vec3::new(0.0, 1.0, 0.0), inv_mass),
        ClothParticle::new(Vec3::new(1.0, 1.0, fold), inv_mass),
    ]
}

/// Builds a three-hinge flat-rest strip (two stacked quads) and returns its
/// bending constraints, exercising a multi-constraint Gauss-Seidel sweep.
fn strip_hinges(compliance: f32) -> Vec<BendingConstraint> {
    let rest = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(2.0, 1.0, 0.0),
    ];
    let triangles = [[0u32, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]];
    build_dihedral_bending(&rest, &triangles, Compliance(compliance))
}

/// Builds the folded six-particle strip with the given inverse mass; the top
/// row is lifted into a shallow ridge so every hinge sees a non-trivial fold.
fn strip_particles(inv_mass: f32) -> Vec<ClothParticle> {
    vec![
        ClothParticle::new(Vec3::new(0.0, 0.0, 0.0), inv_mass),
        ClothParticle::new(Vec3::new(1.0, 0.0, 0.1), inv_mass),
        ClothParticle::new(Vec3::new(2.0, 0.0, 0.0), inv_mass),
        ClothParticle::new(Vec3::new(0.0, 1.0, 0.3), inv_mass),
        ClothParticle::new(Vec3::new(1.0, 1.0, 0.5), inv_mass),
        ClothParticle::new(Vec3::new(2.0, 1.0, 0.2), inv_mass),
    ]
}

/// Squared length of the bend vector `S = Σ wᵢ · xᵢ` for one hinge, computed
/// with pure arithmetic so the randomized sweep can reject near-degenerate
/// folds without a transcendental call.
fn bend_len_sq(positions: &[[f32; 3]], constraint: &BendingConstraint) -> f32 {
    let mut s = [0.0f32; 3];
    for (idx, weight) in constraint.vertices.iter().zip(constraint.weights.iter()) {
        if let Some(p) = positions.get(*idx as usize) {
            s[0] += p[0] * *weight;
            s[1] += p[1] * *weight;
            s[2] += p[2] * *weight;
        }
    }
    s[0] * s[0] + s[1] * s[1] + s[2] * s[2]
}

/// A tiny host-side `LCG` producing a stream of `u32` words; used only to drive
/// the randomized fixture sweep (no `GPU` state depends on it).
struct Lcg {
    /// Current generator state.
    state: u64,
}

impl Lcg {
    /// Seeds the generator.
    fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    /// Advances the state and returns the next `u32` word.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as u32
    }

    /// Returns the next `f32` uniformly in `[lo, hi)`.
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0);
        lo + (hi - lo) * unit
    }
}

/// Dispatches every system in one batch and pins each against the oracle.
fn run(ctx: &GpuContext, systems: &[System]) {
    let twin = GpuClothBendingApply::new(ctx);
    let queries: Vec<ClothBendingApplyQuery> = systems.iter().map(to_query).collect();
    let out = twin.evaluate(ctx, &queries);
    assert_eq!(out.len(), systems.len());
    for (idx, (system, got)) in systems.iter().zip(out.iter()).enumerate() {
        check(idx, system, got);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuClothBendingApply::new(&ctx);
    let out = twin.evaluate(&ctx, &[]);
    assert!(out.is_empty(), "empty batch must return an empty vector");
}

#[test]
fn single_hinge_sweeps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // One hinge, one sweep and three sweeps, plus a stiffer compliance.
    let systems = [
        System {
            particles: quad_particles(1.0, 0.6),
            constraints: quad_hinge(0.0),
            iterations: 1,
            dt: 0.016,
        },
        System {
            particles: quad_particles(1.0, 0.6),
            constraints: quad_hinge(0.0),
            iterations: 3,
            dt: 0.016,
        },
        System {
            particles: quad_particles(0.75, 0.45),
            constraints: quad_hinge(0.002),
            iterations: 2,
            dt: 0.01,
        },
    ];
    run(&ctx, &systems);
}

#[test]
fn multi_hinge_strip() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let systems = [
        System {
            particles: strip_particles(1.0),
            constraints: strip_hinges(0.0),
            iterations: 1,
            dt: 0.016,
        },
        System {
            particles: strip_particles(0.5),
            constraints: strip_hinges(0.001),
            iterations: 3,
            dt: 0.012,
        },
    ];
    run(&ctx, &systems);
}

#[test]
fn degenerate_cases_are_no_ops() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // dt = 0: every projection is a no-op (positions unchanged, energy 0).
    let dt_zero = System {
        particles: quad_particles(1.0, 0.6),
        constraints: quad_hinge(0.0),
        iterations: 3,
        dt: 0.0,
    };
    // All pinned: S and energy still accumulate but no particle moves.
    let all_pinned = System {
        particles: quad_particles(0.0, 0.6),
        constraints: quad_hinge(0.0),
        iterations: 2,
        dt: 0.016,
    };
    // Out-of-bounds stencil slot: a vertex index at the particle count is
    // skipped everywhere, matching the golden `positions.get` guard.
    let mut oob_constraints = quad_hinge(0.0);
    oob_constraints[0].vertices[3] = 4;
    let out_of_bounds = System {
        particles: quad_particles(1.0, 0.6),
        constraints: oob_constraints,
        iterations: 2,
        dt: 0.016,
    };
    run(&ctx, &[dt_zero, all_pinned, out_of_bounds]);
}

#[test]
fn randomized_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };

    let mut rng = Lcg::new(0x1234_5678_9ABC_DEF0);
    let mut systems: Vec<System> = Vec::with_capacity(256);
    while systems.len() < 256 {
        let compliance = rng.next_f32(0.0, 0.01);
        let constraints = quad_hinge(compliance);
        // Slot 0 stays free (inverse mass > 0) and the apex is folded; reject
        // folds whose |S|² sits near the 1e-12 degeneracy tie.
        let positions = [
            [
                rng.next_f32(-1.5, 1.5),
                rng.next_f32(-1.5, 1.5),
                rng.next_f32(-1.5, 1.5),
            ],
            [
                rng.next_f32(-1.5, 1.5),
                rng.next_f32(-1.5, 1.5),
                rng.next_f32(-1.5, 1.5),
            ],
            [
                rng.next_f32(-1.5, 1.5),
                rng.next_f32(-1.5, 1.5),
                rng.next_f32(-1.5, 1.5),
            ],
            [
                rng.next_f32(-1.5, 1.5),
                rng.next_f32(-1.5, 1.5),
                rng.next_f32(-1.5, 1.5),
            ],
        ];
        if bend_len_sq(&positions, &constraints[0]) <= 0.25 {
            continue;
        }
        let inv_masses = [
            rng.next_f32(0.5, 2.0),
            rng.next_f32(0.5, 2.0),
            rng.next_f32(0.5, 2.0),
            rng.next_f32(0.5, 2.0),
        ];
        let particles = positions
            .iter()
            .zip(inv_masses.iter())
            .map(|(p, m)| ClothParticle::new(Vec3::new(p[0], p[1], p[2]), *m))
            .collect();
        let iterations = (rng.next_u32() % 3) + 1;
        let dt = rng.next_f32(0.006, 0.02);
        systems.push(System {
            particles,
            constraints,
            iterations,
            dt,
        });
    }

    run(&ctx, &systems);
}
