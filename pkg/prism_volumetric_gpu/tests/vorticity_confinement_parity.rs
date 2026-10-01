//! Real-device parity for the vorticity-confinement twin:
//! [`GpuVorticityConfinement`](prism_volumetric_gpu::vorticity_confinement::GpuVorticityConfinement)
//! must reproduce the `CPU` golden
//! [`particle::vorticity_confinement`](prism_render_architecture::particle::vorticity_confinement)
//! across several velocity fields, grid resolutions and confinement strengths.
//!
//! The tests skip (with a printed notice on the first) when the host has no
//! `wgpu` adapter, so the suite stays green everywhere while still exercising
//! the full dispatch-and-readback on any real device such as an Apple
//! `M`-series `GPU`. The kernel is portable core-`WGSL`, so it needs no optional
//! device feature.
//!
//! # Parity criterion
//!
//! Every cell is finite-difference algebra with no transcendental call and no
//! reorderable reduction, so `CPU` and `GPU` evaluate the same closed form in
//! the same order. The comparison allows `abs_diff <= 1e-5` or
//! `rel_diff <= 1e-5` — loose enough to admit a legal fused multiply-add
//! contraction across the six neighbor curls feeding each gradient, yet tight
//! enough to fail a wrong port (a swapped curl component, a dropped boundary
//! clamp, a missing normalization guard, a wrong force scale). Each scenario
//! checks the intermediate vorticity field, the vorticity-magnitude field and
//! the final confinement force, and several scenes additionally assert a
//! non-trivial field (a non-zero curl and force somewhere) so a degenerate
//! all-zero kernel could not pass.
//!
//! Provenance: standard Fedkiw-Stam-Jensen vorticity confinement; no Unreal
//! Engine source or derived code.

use prism_render_architecture::particle::vorticity_confinement::{
    confinement_force, vorticity_magnitude_field, Vec3, VelocityGrid,
};
use prism_volumetric_gpu::vorticity_confinement::{GpuVorticityConfinement, VorticityResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute/relative parity bound. A `GPU` may fuse a multiply-add the scalar
/// reference leaves separate, perturbing the low mantissa bits by a few units
/// in the last place; `1e-5` admits that legal slack while still failing a
/// genuinely wrong port.
const EPS: f32 = 1.0e-5;

/// The reference `MIN_LENGTH` floor used to keep the relative-error denominator
/// away from zero.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= EPS
}

/// Asserts two vectors agree component-wise within [`close`].
fn close_vec(a: Vec3, b: Vec3, what: &str, idx: usize) {
    assert!(
        close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z),
        "{what} mismatch at cell {idx}: gpu ({}, {}, {}), cpu ({}, {}, {})",
        a.x,
        a.y,
        a.z,
        b.x,
        b.y,
        b.z
    );
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[-1, 1]`.
fn lcg(state: &mut u64) -> f32 {
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    // 24 usable mantissa bits mapped onto [0, 1) then onto [-1, 1).
    let unit = (bits & 0x00ff_ffff) as f32 / 16_777_216.0;
    unit * 2.0 - 1.0
}

/// Builds a random velocity grid of the given dimensions and spacing.
fn random_grid(dims: [usize; 3], cell_size: f32, seed: u64) -> VelocityGrid {
    let count = dims[0] * dims[1] * dims[2];
    let mut state = seed;
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        let x = lcg(&mut state);
        let y = lcg(&mut state);
        let z = lcg(&mut state);
        data.push(Vec3::new(x, y, z));
    }
    VelocityGrid::new(dims, cell_size, data)
}

/// Builds a rigid-rotation field `v = (-y, x, 0)`; analytic curl is `(0, 0, 2)`.
fn rotation_grid(dims: [usize; 3], cell_size: f32) -> VelocityGrid {
    let count = dims[0] * dims[1] * dims[2];
    let mut data = Vec::with_capacity(count);
    for k in 0..dims[2] {
        let _ = k;
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let x = f32::from(u8::try_from(i).unwrap_or(0)) * cell_size;
                let y = f32::from(u8::try_from(j).unwrap_or(0)) * cell_size;
                data.push(Vec3::new(-y, x, 0.0));
            }
        }
    }
    VelocityGrid::new(dims, cell_size, data)
}

/// Builds a sheared field `v = (-y*x, x, 0)` whose vorticity magnitude varies in
/// space, giving a genuine (non-zero) confinement direction.
fn shear_grid(dims: [usize; 3], cell_size: f32) -> VelocityGrid {
    let count = dims[0] * dims[1] * dims[2];
    let mut data = Vec::with_capacity(count);
    for k in 0..dims[2] {
        let _ = k;
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let x = f32::from(u8::try_from(i).unwrap_or(0)) * cell_size;
                let y = f32::from(u8::try_from(j).unwrap_or(0)) * cell_size;
                data.push(Vec3::new(-y * x, x, 0.0));
            }
        }
    }
    VelocityGrid::new(dims, cell_size, data)
}

/// Builds a uniform field (constant velocity everywhere); curl is zero.
fn uniform_grid(dims: [usize; 3], cell_size: f32, v: Vec3) -> VelocityGrid {
    let count = dims[0] * dims[1] * dims[2];
    VelocityGrid::new(dims, cell_size, vec![v; count])
}

/// Runs the `GPU` twin over `grid` at `epsilon` and asserts cell-by-cell parity
/// of the vorticity, the vorticity-magnitude field and the confinement force
/// against the `CPU` reference. Returns the `GPU` results for extra assertions.
fn check_grid(
    ctx: &GpuContext,
    gpu: &GpuVorticityConfinement,
    grid: &VelocityGrid,
    epsilon: f32,
) -> Vec<VorticityResult> {
    let results = gpu.eval(ctx, grid, epsilon);
    assert_eq!(
        results.len(),
        grid.cell_count(),
        "one result per grid cell"
    );
    let cpu_mag = vorticity_magnitude_field(grid);
    assert_eq!(cpu_mag.len(), grid.cell_count(), "one magnitude per cell");

    for k in 0..grid.dims[2] {
        for j in 0..grid.dims[1] {
            for i in 0..grid.dims[0] {
                let idx = grid.index(i, j, k);
                let got = results[idx];

                // Intermediate vorticity parity.
                let cpu_curl = grid.curl_at(i, j, k);
                close_vec(got.curl, cpu_curl, "curl", idx);

                // Vorticity-magnitude field parity (derived from the curl).
                assert!(
                    close(got.curl.length(), cpu_mag[idx]),
                    "magnitude mismatch at cell {idx}: gpu {}, cpu {}",
                    got.curl.length(),
                    cpu_mag[idx]
                );

                // Final confinement-force parity.
                let cpu_force = confinement_force(grid, i, j, k, epsilon);
                close_vec(got.force, cpu_force, "force", idx);
            }
        }
    }
    results
}

/// Asserts that at least one cell carries a non-trivial curl and force, so a
/// degenerate all-zero kernel could not pass this scene.
fn assert_non_trivial(results: &[VorticityResult]) {
    let any_curl = results.iter().any(|r| r.curl.length() > 1.0e-3);
    let any_force = results.iter().any(|r| r.force.length() > 1.0e-3);
    assert!(any_curl, "scene should produce a non-zero vorticity somewhere");
    assert!(
        any_force,
        "scene should produce a non-zero confinement force somewhere"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_rotation_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping vorticity-confinement parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuVorticityConfinement::new(&ctx);
    let grid = rotation_grid([5, 5, 3], 0.25);
    let results = check_grid(&ctx, &gpu, &grid, 2.5);

    // The interior curl of a rigid rotation is the constant (0, 0, 2).
    let interior = results[grid.index(2, 2, 1)];
    assert!(
        close(interior.curl.z, 2.0),
        "interior rigid-rotation curl.z should be 2.0, got {}",
        interior.curl.z
    );
    // A rigid rotation has a spatially constant vorticity, so its magnitude
    // gradient is zero and the confinement force vanishes everywhere.
    for r in &results {
        assert!(
            r.force.length() <= EPS,
            "rigid rotation should have zero confinement force, got {}",
            r.force.length()
        );
    }
}

#[test]
fn gpu_matches_cpu_on_uniform_field_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVorticityConfinement::new(&ctx);
    let grid = uniform_grid([4, 4, 4], 0.5, Vec3::new(3.0, -1.0, 2.0));
    let results = check_grid(&ctx, &gpu, &grid, 5.0);
    for r in &results {
        assert!(r.curl.length() <= EPS, "uniform field has zero curl");
        assert!(r.force.length() <= EPS, "uniform field has zero force");
    }
}

#[test]
fn gpu_matches_cpu_on_shear_field_nontrivial() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVorticityConfinement::new(&ctx);
    let grid = shear_grid([6, 6, 3], 0.5);
    let results = check_grid(&ctx, &gpu, &grid, 2.0);
    assert_non_trivial(&results);
}

#[test]
fn gpu_matches_cpu_on_random_fields_multiple_resolutions() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVorticityConfinement::new(&ctx);
    // A spread of resolutions (including non-cubic and a thin slab) and spacings
    // and confinement strengths, each with its own random field.
    let cases: [([usize; 3], f32, f32, u64); 4] = [
        ([4, 4, 4], 0.5, 1.0, 0x1234_5678_9abc_def0),
        ([7, 3, 4], 0.3, 3.5, 0x0f0f_0f0f_1234_5678),
        ([8, 6, 2], 1.0, 0.75, 0xdead_beef_cafe_babe),
        ([3, 9, 5], 0.2, 2.25, 0x5555_aaaa_3333_cccc),
    ];
    for (dims, cell, eps, seed) in cases {
        let grid = random_grid(dims, cell, seed);
        let results = check_grid(&ctx, &gpu, &grid, eps);
        assert_non_trivial(&results);
    }
}

#[test]
fn gpu_matches_cpu_with_negative_epsilon() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVorticityConfinement::new(&ctx);
    // A negative confinement strength just flips the force sign; parity must
    // still hold against the reference.
    let grid = random_grid([5, 5, 5], 0.4, 0x9e37_79b9_7f4a_7c15);
    let results = check_grid(&ctx, &gpu, &grid, -1.75);
    assert_non_trivial(&results);
}

#[test]
fn gpu_matches_cpu_on_zero_field_is_safe() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVorticityConfinement::new(&ctx);
    // A degenerate all-zero velocity field: curl, gradient and force all vanish
    // and the normalization guard must keep every output finite (no NaN).
    let grid = uniform_grid([4, 3, 2], 0.5, Vec3::ZERO);
    let results = check_grid(&ctx, &gpu, &grid, 4.0);
    for r in &results {
        assert!(
            r.curl.length() <= EPS && r.force.length() <= EPS,
            "zero field must stay zero and finite"
        );
        assert!(r.force.x.is_finite() && r.force.y.is_finite() && r.force.z.is_finite());
    }
}

#[test]
fn gpu_matches_cpu_on_single_cell_grid_is_degenerate_guarded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVorticityConfinement::new(&ctx);
    // A 1x1x1 grid has a zero-span stencil on every axis: curl and force must be
    // the guarded zero, matching the reference.
    let grid = uniform_grid([1, 1, 1], 1.0, Vec3::new(1.0, 2.0, 3.0));
    let results = check_grid(&ctx, &gpu, &grid, 10.0);
    assert_eq!(results.len(), 1);
    assert!(results[0].curl.length() <= EPS && results[0].force.length() <= EPS);
}

#[test]
fn gpu_matches_cpu_on_boundary_cells() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVorticityConfinement::new(&ctx);
    // The rotation field exercises one-sided boundary stencils; check_grid
    // already compares every cell including the corners, and the corner curl.z
    // must still recover the rigid-rotation value of 2.
    let grid = rotation_grid([4, 4, 2], 0.5);
    let results = check_grid(&ctx, &gpu, &grid, 1.5);
    let corner = results[grid.index(0, 0, 0)];
    assert!(
        close(corner.curl.z, 2.0),
        "one-sided boundary curl.z should still be 2.0, got {}",
        corner.curl.z
    );
}

#[test]
fn empty_grid_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVorticityConfinement::new(&ctx);
    let grid = VelocityGrid::new([0, 0, 0], 1.0, Vec::new());
    let out = gpu.eval(&ctx, &grid, 1.0);
    assert!(out.is_empty(), "an empty grid yields no results");
}
