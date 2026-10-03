//! Real-device parity for the pressure (closed-mesh volume) projection twin:
//! [`GpuClothPressureProject`](prism_volumetric_gpu::cloth_pressure_project::GpuClothPressureProject)
//! must reproduce the `CPU` golden
//! [`project_pressure`](prism_render_architecture::cloth::pressure::project_pressure)
//! — the updated world-space position of every vertex of one closed,
//! outward-wound triangle mesh after a single compliant `XPBD` pressure step —
//! across the degenerate branches (empty batch, empty mesh, non-positive `dt`,
//! an all-pinned rigid mesh), an out-of-range triangle, and a randomized sweep
//! compared slot-for-slot.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The oracle is the public golden
//! [`project_pressure`](prism_render_architecture::cloth::pressure::project_pressure)
//! itself: a clone of the particle slice is projected on the host, its mutated
//! positions are read back, and the `GPU` is pinned against them. The `GPU`
//! query is built from the *original* particles' positions and inverse masses
//! and the same triangle list, mirroring the reference's structure-of-arrays
//! conversion and its `positions.get` out-of-range triangle skip. The raw
//! pressure parameters are passed through unchanged to both paths, so the
//! device's in-shader sanitize is pinned against the reference `sanitized`.
//!
//! # Parity criterion
//!
//! The updated positions thread through only `+ - * /`, `cross` and `dot` with
//! no `sqrt`, so a `GPU` result may land a few units in the last place from the
//! scalar reference; both are asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`. No `f32` `==` is used anywhere.
//!
//! # Conditioning
//!
//! The randomized sweep keeps every vertex free and rejection-samples meshes
//! whose solve denominator is near the `1e-12` degenerate threshold (requiring
//! `denom >= 1e-3`) so the `CPU` and `GPU` stay on the same side of the
//! near-zero-denominator short-circuit.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::pressure`；无第三方引擎源码或衍生代码。

use prism_render_architecture::cloth::{
    pressure::{project_pressure, PressureParams},
    ClothParticle, Compliance, Vec3,
};
use prism_volumetric_gpu::cloth_pressure_project::{
    ClothPressureProjectQuery, ClothPressureProjectResult, GpuClothPressureProject,
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

/// One sixth, the divergence-theorem volume prefactor; used only for host-side
/// rejection-sampling conditioning, never as the parity oracle.
const INV_SIX: f32 = 1.0 / 6.0;

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

/// The eight corners of the axis-aligned unit cube, matching the reference
/// fixture; the shell encloses a volume of one.
fn unit_cube_positions() -> Vec<[f32; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
}

/// The twelve outward-wound triangles of the unit cube, matching the reference.
fn unit_cube_triangles() -> Vec<[u32; 3]> {
    vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 6, 2],
        [3, 7, 6],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ]
}

/// The cross product of two three-vectors, for host-side conditioning only.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Recomputes the solve denominator `Σ wᵢ |∇ᵢ|² + compliance / dt²` for the
/// rejection-sampling guard so the sweep stays well clear of the degenerate
/// near-zero-denominator branch. This is conditioning only; the parity oracle
/// is always the golden `project_pressure`.
fn solve_denominator(
    positions: &[[f32; 3]],
    inverse_masses: &[f32],
    triangles: &[[u32; 3]],
    compliance: f32,
    dt: f32,
) -> f32 {
    let count = positions.len();
    let mut grad = vec![[0.0f32; 3]; count];
    for tri in triangles {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= count || i1 >= count || i2 >= count {
            continue;
        }
        let (p0, p1, p2) = (positions[i0], positions[i1], positions[i2]);
        let c12 = cross(p1, p2);
        let c20 = cross(p2, p0);
        let c01 = cross(p0, p1);
        for (slot, src) in [(i0, c12), (i1, c20), (i2, c01)] {
            grad[slot][0] += src[0] * INV_SIX;
            grad[slot][1] += src[1] * INV_SIX;
            grad[slot][2] += src[2] * INV_SIX;
        }
    }
    let mut denom = 0.0f32;
    for (g, &w) in grad.iter().zip(inverse_masses.iter()) {
        if w <= 0.0 {
            continue;
        }
        denom += w * (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    }
    denom + compliance / (dt * dt)
}

/// Builds a `GPU` query from the original particles' positions and inverse
/// masses plus the raw pressure parameters.
fn build_query(
    particles: &[ClothParticle],
    triangles: &[[u32; 3]],
    rest_volume: f32,
    overpressure: f32,
    compliance: f32,
    dt: f32,
) -> ClothPressureProjectQuery {
    let positions = particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect();
    let inverse_masses = particles.iter().map(|p| p.inverse_mass).collect();
    ClothPressureProjectQuery {
        positions,
        inverse_masses,
        triangles: triangles.to_vec(),
        rest_volume,
        overpressure,
        compliance,
        dt,
    }
}

/// Projects one mesh on the `GPU` and pins every vertex position against the
/// host golden `project_pressure`.
fn check_case(
    ctx: &GpuContext,
    gpu: &GpuClothPressureProject,
    particles: &[ClothParticle],
    triangles: &[[u32; 3]],
    rest_volume: f32,
    overpressure: f32,
    compliance: f32,
    dt: f32,
) {
    let query = build_query(
        particles,
        triangles,
        rest_volume,
        overpressure,
        compliance,
        dt,
    );

    let mut clone: Vec<ClothParticle> = particles.to_vec();
    project_pressure(
        &mut clone,
        triangles,
        PressureParams::new(rest_volume, overpressure, Compliance(compliance)),
        dt,
    );

    let got = gpu.evaluate(ctx, &[query]);
    assert_eq!(got.len(), 1, "one mesh produces one result");
    let result: &ClothPressureProjectResult = &got[0];
    assert_eq!(
        result.positions.len(),
        clone.len(),
        "the result echoes one position per vertex"
    );

    for (got_pos, p) in result.positions.iter().zip(clone.iter()) {
        let want = [p.position.x, p.position.y, p.position.z];
        for (g, w) in got_pos.iter().zip(want.iter()) {
            assert!(close(*g, *w), "position component: gpu {g} vs cpu {w}");
        }
    }
}

/// Builds a movable-particle mesh from positions with unit inverse mass.
fn free_particles(positions: &[[f32; 3]]) -> Vec<ClothParticle> {
    positions
        .iter()
        .map(|p| ClothParticle::new(Vec3::new(p[0], p[1], p[2]), 1.0))
        .collect()
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_pressure_project parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuClothPressureProject::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn inflated_unit_cube_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPressureProject::new(&ctx);
    // Rest volume 1, overpressure 2 (so C = 1 - 2 = -1): the rigid shell gets
    // pushed outward along its vertex gradients.
    let particles = free_particles(&unit_cube_positions());
    check_case(
        &ctx,
        &gpu,
        &particles,
        &unit_cube_triangles(),
        1.0,
        2.0,
        0.0,
        1.0 / 60.0,
    );
}

#[test]
fn compliant_deflation_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPressureProject::new(&ctx);
    // Overpressure below one deflates; a non-zero compliance softens the step.
    let particles = free_particles(&unit_cube_positions());
    check_case(
        &ctx,
        &gpu,
        &particles,
        &unit_cube_triangles(),
        1.0,
        0.5,
        0.01,
        1.0 / 60.0,
    );
}

#[test]
fn empty_mesh_is_a_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPressureProject::new(&ctx);
    // No triangles: the reference returns before touching any position.
    let particles = free_particles(&unit_cube_positions());
    check_case(&ctx, &gpu, &particles, &[], 1.0, 2.0, 0.0, 1.0 / 60.0);
}

#[test]
fn non_positive_dt_is_a_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPressureProject::new(&ctx);
    let particles = free_particles(&unit_cube_positions());
    check_case(
        &ctx,
        &gpu,
        &particles,
        &unit_cube_triangles(),
        1.0,
        2.0,
        0.0,
        0.0,
    );
}

#[test]
fn all_pinned_is_a_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPressureProject::new(&ctx);
    // Every vertex pinned (inverse_mass 0) with zero compliance: the denominator
    // is zero, so the reference leaves every position fixed.
    let particles: Vec<ClothParticle> = unit_cube_positions()
        .iter()
        .map(|p| ClothParticle::new(Vec3::new(p[0], p[1], p[2]), 0.0))
        .collect();
    check_case(
        &ctx,
        &gpu,
        &particles,
        &unit_cube_triangles(),
        1.0,
        2.0,
        0.0,
        1.0 / 60.0,
    );
}

#[test]
fn out_of_range_triangle_is_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPressureProject::new(&ctx);
    // A junk triangle indexing past the vertex array contributes nothing to the
    // volume or gradient, exactly like the reference's `positions.get` skip.
    let particles = free_particles(&unit_cube_positions());
    let mut triangles = unit_cube_triangles();
    triangles.push([9, 10, 11]);
    check_case(
        &ctx,
        &gpu,
        &particles,
        &triangles,
        1.0,
        2.0,
        0.0,
        1.0 / 60.0,
    );
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPressureProject::new(&ctx);
    let triangles = unit_cube_triangles();
    let mut state = 0x51a2_c3d4_e5f6_0718_u64;

    let mut built = 0u32;
    let mut attempts = 0u32;
    while built < 256 && attempts < 20_000 {
        attempts += 1;

        // Perturb the cube corners so the shell is non-degenerate but still a
        // valid closed mesh; keep every vertex free so the denominator comes
        // only from the volume gradient.
        let positions: Vec<[f32; 3]> = unit_cube_positions()
            .iter()
            .map(|p| {
                [
                    p[0] + draw(&mut state, -0.2, 0.2),
                    p[1] + draw(&mut state, -0.2, 0.2),
                    p[2] + draw(&mut state, -0.2, 0.2),
                ]
            })
            .collect();
        let inverse_masses: Vec<f32> = (0..positions.len())
            .map(|_| draw(&mut state, 0.5, 3.0))
            .collect();
        let rest_volume = draw(&mut state, 0.3, 1.8);
        let overpressure = draw(&mut state, 0.5, 2.0);
        let compliance = draw(&mut state, 0.0, 0.05);
        let dt = 1.0 / 60.0;

        // Reject meshes whose denominator is near the 1e-12 degenerate threshold
        // so CPU and GPU stay on the same side of the no-op short-circuit.
        let denom = solve_denominator(&positions, &inverse_masses, &triangles, compliance, dt);
        if denom < 1.0e-3 {
            continue;
        }

        let particles: Vec<ClothParticle> = positions
            .iter()
            .zip(inverse_masses.iter())
            .map(|(p, &w)| ClothParticle::new(Vec3::new(p[0], p[1], p[2]), w))
            .collect();
        check_case(
            &ctx,
            &gpu,
            &particles,
            &triangles,
            rest_volume,
            overpressure,
            compliance,
            dt,
        );
        built += 1;
    }
    assert!(built >= 256, "expected 256 random meshes, built {built}");
}
