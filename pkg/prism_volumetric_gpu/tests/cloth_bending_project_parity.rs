//! Real-device parity for the isometric-bending projection twin:
//! [`GpuClothBendingProject`](prism_volumetric_gpu::cloth_bending_project::GpuClothBendingProject)
//! must reproduce the `CPU` golden
//! [`project_bending`](prism_render_architecture::cloth::bending::project_bending)
//! — the four updated stencil positions and the pre-step bending energy of one
//! four-vertex hinge — across the degenerate branches, an out-of-range slot, and
//! a randomized sweep compared slot-for-slot.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The oracle is the public golden
//! [`project_bending`](prism_render_architecture::cloth::bending::project_bending)
//! itself: a clone of the particle slice is projected on the host, its returned
//! energy and its mutated positions are read back, and the `GPU` is pinned
//! against them. The `GPU` query is built from the *original* particles by
//! resolving the hinge `vertices` into four stencil slots, mirroring the
//! reference's `positions.get` / `unwrap_or(0.0)` out-of-range handling.
//!
//! # Parity criterion
//!
//! The energy and the updated positions thread through only `+ - * /` with no
//! `sqrt`, so a `GPU` result may land a few units in the last place from the
//! scalar reference; both are asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`. No `f32` `==` is used anywhere.
//!
//! # Conditioning
//!
//! The randomized sweep rejection-samples hinges whose squared bend length
//! `|S|²` is near the `1e-12` flat threshold (requiring `|S|² >= 1e-3`) so the
//! `CPU` and `GPU` stay on the same side of the flat-stencil short-circuit, and
//! keeps at least one free slot so the denominator is strictly positive.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::bending`；无第三方引擎源码或衍生代码。

use prism_render_architecture::cloth::{
    bending::{project_bending, BendingConstraint},
    ClothParticle, Compliance, Vec3,
};
use prism_volumetric_gpu::cloth_bending_project::{
    ClothBendingProjectQuery, ClothBendingProjectResult, GpuClothBendingProject,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous quantity. A `GPU` arithmetic pipeline
/// may land a few units in the last place from the scalar reference; `1e-4`
/// admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws an `f32` in `[lo, hi]` at ten-thousandth resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (lcg(state) % 10_001) as f32 / 10_000.0 * (hi - lo)
}

/// Builds a `GPU` query from the original particles by resolving the hinge
/// `vertices` into four stencil slots.
fn build_query(
    particles: &[ClothParticle],
    constraint: &BendingConstraint,
    dt: f32,
) -> ClothBendingProjectQuery {
    let mut positions = [[0.0f32; 3]; 4];
    let mut inverse_masses = [0.0f32; 4];
    let mut valid = [false; 4];
    for (((pos_slot, inv_slot), valid_slot), &vi) in positions
        .iter_mut()
        .zip(inverse_masses.iter_mut())
        .zip(valid.iter_mut())
        .zip(constraint.vertices.iter())
    {
        if let Some(p) = particles.get(vi as usize) {
            *pos_slot = [p.position.x, p.position.y, p.position.z];
            *inv_slot = if p.is_pinned() { 0.0 } else { p.inverse_mass };
            *valid_slot = true;
        }
    }
    ClothBendingProjectQuery {
        positions,
        inverse_masses,
        valid,
        weights: constraint.weights,
        scale: constraint.scale,
        compliance: constraint.compliance.0,
        dt,
    }
}

/// Projects one hinge on the `GPU` and pins the energy and every in-range slot's
/// updated position against the host golden `project_bending`.
fn check_case(
    ctx: &GpuContext,
    gpu: &GpuClothBendingProject,
    particles: &[ClothParticle],
    constraint: BendingConstraint,
    dt: f32,
) {
    let query = build_query(particles, &constraint, dt);

    let mut clone: Vec<ClothParticle> = particles.to_vec();
    let energy_golden = project_bending(&mut clone, constraint, dt);

    let got = gpu.evaluate(ctx, &[query]);
    assert_eq!(got.len(), 1, "one hinge produces one result");
    let result: &ClothBendingProjectResult = &got[0];

    assert!(
        close(result.energy, energy_golden),
        "energy: gpu {} vs cpu {}",
        result.energy,
        energy_golden
    );

    for (&vi, got_pos) in constraint.vertices.iter().zip(result.positions.iter()) {
        if let Some(p) = clone.get(vi as usize) {
            let want = [p.position.x, p.position.y, p.position.z];
            for (g, w) in got_pos.iter().zip(want.iter()) {
                assert!(
                    close(*g, *w),
                    "slot vertex {vi} position component: gpu {g} vs cpu {w}"
                );
            }
        }
    }
}

/// A hinge of four distinct free particles with the given per-slot weights.
fn free_particles(points: [[f32; 3]; 4], inverse_mass: f32) -> Vec<ClothParticle> {
    points
        .iter()
        .map(|p| ClothParticle::new(Vec3::new(p[0], p[1], p[2]), inverse_mass))
        .collect()
}

/// A constraint over the four stencil slots in order with distinct vertices.
fn constraint(weights: [f32; 4], scale: f32, compliance: f32) -> BendingConstraint {
    BendingConstraint {
        vertices: [0, 1, 2, 3],
        weights,
        scale,
        compliance: Compliance(compliance),
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_bending_project parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuClothBendingProject::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn flat_stencil_is_a_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBendingProject::new(&ctx);
    // Four coincident particles with weights summing (weighted) to zero give
    // S = 0, so |S|^2 <= EPS_LEN_SQ: the reference returns 0 and moves nothing.
    let particles = free_particles(
        [
            [0.5, -0.25, 1.0],
            [0.5, -0.25, 1.0],
            [0.5, -0.25, 1.0],
            [0.5, -0.25, 1.0],
        ],
        1.5,
    );
    check_case(
        &ctx,
        &gpu,
        &particles,
        constraint([1.0, -1.0, 1.0, -1.0], 2.0, 0.01),
        1.0 / 60.0,
    );
}

#[test]
fn all_pinned_returns_energy_without_moving() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBendingProject::new(&ctx);
    // All pinned (inverse_mass 0) with zero compliance: the denominator is 0, so
    // the reference returns the energy and leaves every position fixed.
    let points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.2],
        [0.5, -1.0, -0.3],
    ];
    let particles: Vec<ClothParticle> = points
        .iter()
        .map(|p| ClothParticle::new(Vec3::new(p[0], p[1], p[2]), 0.0))
        .collect();
    check_case(
        &ctx,
        &gpu,
        &particles,
        constraint([1.0, 1.0, -1.0, -1.0], 1.5, 0.0),
        1.0 / 60.0,
    );
}

#[test]
fn non_positive_dt_is_a_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBendingProject::new(&ctx);
    let particles = free_particles(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.2],
            [0.5, -1.0, -0.3],
        ],
        1.0,
    );
    check_case(
        &ctx,
        &gpu,
        &particles,
        constraint([1.0, 1.0, -1.0, -1.0], 1.5, 0.01),
        0.0,
    );
}

#[test]
fn out_of_range_slot_is_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBendingProject::new(&ctx);
    // Four particles but a hinge whose fourth vertex (9) is out of range: that
    // slot contributes nothing to S and is never moved, mirroring the reference.
    let particles = free_particles(
        [
            [0.1, 0.2, 0.0],
            [1.0, 0.1, 0.3],
            [0.4, 1.0, -0.2],
            [0.0, 0.0, 0.0],
        ],
        1.2,
    );
    let constraint = BendingConstraint {
        vertices: [0, 1, 2, 9],
        weights: [1.0, -0.5, -0.5, 0.75],
        scale: 1.8,
        compliance: Compliance(0.02),
    };
    check_case(&ctx, &gpu, &particles, constraint, 1.0 / 60.0);
}

#[test]
fn mixed_pinned_and_free_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBendingProject::new(&ctx);
    // Slots 0 and 2 free, slots 1 and 3 pinned: pinned slots still contribute to
    // S but never move, free slots take the whole correction.
    let particles = vec![
        ClothParticle::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
        ClothParticle::new(Vec3::new(1.0, 0.1, 0.2), 0.0),
        ClothParticle::new(Vec3::new(0.4, 1.0, -0.1), 2.0),
        ClothParticle::new(Vec3::new(0.5, -1.0, 0.3), 0.0),
    ];
    check_case(
        &ctx,
        &gpu,
        &particles,
        constraint([1.0, -0.5, 1.0, -0.5], 1.3, 0.015),
        1.0 / 60.0,
    );
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothBendingProject::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;

    let mut built = 0u32;
    let mut attempts = 0u32;
    while built < 256 && attempts < 20_000 {
        attempts += 1;

        // Four distinct particles with random positions; slot 0 is always free so
        // the denominator is strictly positive. Other slots are randomly pinned.
        let mut particles: Vec<ClothParticle> = Vec::with_capacity(4);
        for slot in 0..4u32 {
            let position = Vec3::new(
                draw(&mut state, -2.0, 2.0),
                draw(&mut state, -2.0, 2.0),
                draw(&mut state, -2.0, 2.0),
            );
            let pinned = slot != 0 && lcg(&mut state).is_multiple_of(3);
            let inverse_mass = if pinned {
                0.0
            } else {
                draw(&mut state, 0.5, 3.0)
            };
            particles.push(ClothParticle::new(position, inverse_mass));
        }

        let weights = [
            draw(&mut state, 0.3, 2.0)
                * if lcg(&mut state).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                },
            draw(&mut state, 0.3, 2.0)
                * if lcg(&mut state).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                },
            draw(&mut state, 0.3, 2.0)
                * if lcg(&mut state).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                },
            draw(&mut state, 0.3, 2.0)
                * if lcg(&mut state).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                },
        ];
        let scale = draw(&mut state, 0.5, 3.0);
        let compliance = draw(&mut state, 0.0, 0.05);

        // Reject hinges near the flat threshold so CPU and GPU stay on the same
        // side of the |S|^2 <= 1e-12 short-circuit.
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        let mut sz = 0.0f32;
        for (p, w) in particles.iter().zip(weights.iter()) {
            sx += w * p.position.x;
            sy += w * p.position.y;
            sz += w * p.position.z;
        }
        let s_len_sq = sx * sx + sy * sy + sz * sz;
        if s_len_sq < 1.0e-3 {
            continue;
        }

        check_case(
            &ctx,
            &gpu,
            &particles,
            constraint(weights, scale, compliance),
            1.0 / 60.0,
        );
        built += 1;
    }
    assert!(built >= 256, "expected 256 random hinges, built {built}");
}
