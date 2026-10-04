//! Real-device parity for the full foam-update twin:
//! [`GpuWaterFoamStep`](prism_volumetric_gpu::water_foam_step::GpuWaterFoamStep)
//! must reproduce the dependency-free `CPU` golden
//! [`step_foam`](prism_render_architecture::water::foam::step_foam) — advect,
//! flow-aware decay, add sources, clamp — across still, uniform, structured and
//! divergent flows, several decay configs, non-square grids, a zero step, and
//! the degenerate request.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`step_foam`] is public and pure, so the expected field is built
//! in-host and compared cell by cell against the `GPU` readback. A
//! `GPU == oracle` pass is therefore directly a `GPU == golden` pass.
//!
//! # Parity criterion
//!
//! Each cell performs the identical backtrace-and-bilinear advection, `sqrt`
//! flow speed, flow-aware rate, bit-replicated `exp_approx` decay, source add
//! and clamp as the golden, in the same order, so the only residual is the
//! last-place slack of a `GPU` fused multiply-add. Each cell is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::foam`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::foam::{step_foam, FoamConfig};
use prism_volumetric_gpu::water_foam_step::{GpuWaterFoamStep, WaterFoamStep};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on one updated sample.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes.
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

/// A foam config with the given grid and decay parameters.
fn cfg(nx: u32, nz: u32, dx: f32, base_decay: f32, floor: f32, reference: f32) -> FoamConfig {
    FoamConfig {
        nx,
        nz,
        dx,
        base_decay,
        persistence_floor: floor,
        reference_speed: reference,
    }
}

/// A deterministic blob-plus-ripple foam field over `nx * nz` cells.
fn foam(nx: u32, nz: u32) -> Vec<f32> {
    let mut d = Vec::with_capacity((nx * nz) as usize);
    for z in 0..nz {
        for x in 0..nx {
            let fx = x as f32;
            let fz = z as f32;
            let cx = (nx as f32 - 1.0) * 0.5;
            let cz = (nz as f32 - 1.0) * 0.5;
            let r2 = (fx - cx) * (fx - cx) + (fz - cz) * (fz - cz);
            let bump = (1.0 - 0.05 * r2).max(0.0);
            let checker = if (x + z) % 2 == 0 { 0.1 } else { 0.0 };
            d.push(bump + checker);
        }
    }
    d
}

/// A deterministic per-cell source field (fresh foam from crests/contacts),
/// including a few negatives so the `max(source, 0)` clamp is exercised.
fn source_field(nx: u32, nz: u32) -> Vec<f32> {
    let mut s = Vec::with_capacity((nx * nz) as usize);
    for z in 0..nz {
        for x in 0..nx {
            let v = match (x + 2 * z) % 5 {
                0 => 0.2,
                1 => 0.0,
                2 => -0.3,
                3 => 0.05,
                _ => 0.4,
            };
            s.push(v);
        }
    }
    s
}

/// Pins one `GPU` foam step against the `CPU` golden, cell by cell.
#[expect(clippy::too_many_arguments, reason = "mirrors the golden signature")]
fn check(
    ctx: &GpuContext,
    gpu: &GpuWaterFoamStep,
    density: &[f32],
    u: &[f32],
    v: &[f32],
    sources: &[f32],
    nx: u32,
    nz: u32,
    dx: f32,
    dt: f32,
    base_decay: f32,
    floor: f32,
    reference: f32,
) {
    let config = cfg(nx, nz, dx, base_decay, floor, reference);
    let want = step_foam(density, u, v, sources, config, dt);
    let got = gpu.evaluate(
        ctx, density, u, v, sources, nx, nz, dx, dt, base_decay, floor, reference,
    );
    let label =
        format!("nx={nx} nz={nz} dx={dx} dt={dt} decay={base_decay} floor={floor} ref={reference}");
    assert_eq!(got.density.len(), want.len(), "{label}: len");
    for (i, (&g, &w)) in got.density.iter().zip(want.iter()).enumerate() {
        assert!(close(g, w), "{label}: cell[{i}] gpu {g} vs cpu {w}");
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn degenerate_request_returns_input() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_foam_step parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterFoamStep::new(&ctx);
    // An empty grid returns empty, exactly like the golden.
    assert_eq!(
        gpu.evaluate(&ctx, &[], &[], &[], &[], 0, 0, 1.0, 0.1, 0.5, 0.1, 1.0),
        WaterFoamStep {
            density: Vec::new()
        },
        "empty grid"
    );
    // A short input is the twin's device-buffer no-op contract: it returns the
    // density unchanged rather than dispatching with undersized device buffers.
    let short = vec![0.3_f32, 0.4];
    assert_eq!(
        gpu.evaluate(&ctx, &short, &short, &short, &short, 4, 4, 1.0, 0.1, 0.5, 0.1, 1.0),
        WaterFoamStep {
            density: short.clone()
        },
        "short input"
    );
}

#[test]
fn still_water_is_pure_decay_plus_source() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamStep::new(&ctx);
    let (nx, nz) = (12u32, 9u32);
    let d = foam(nx, nz);
    let s = source_field(nx, nz);
    let n = (nx * nz) as usize;
    // Zero flow: no advection, so the step is persistence-floor decay plus the
    // clamped source — the calm-water branch of the decay law.
    let u = vec![0.0_f32; n];
    let v = vec![0.0_f32; n];
    check(&ctx, &gpu, &d, &u, &v, &s, nx, nz, 1.0, 0.5, 0.5, 0.15, 1.0);
}

#[test]
fn zero_step_adds_only_clamped_sources() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamStep::new(&ctx);
    let (nx, nz) = (14u32, 10u32);
    let d = foam(nx, nz);
    let s = source_field(nx, nz);
    let n = (nx * nz) as usize;
    let u = vec![1.5_f32; n];
    let v = vec![-0.7_f32; n];
    // A zero step backtraces nowhere and `exp_approx(0) == 1`, so the result is
    // `clamp(density + max(source, 0), 0, 1)`.
    check(&ctx, &gpu, &d, &u, &v, &s, nx, nz, 1.0, 0.0, 0.4, 0.1, 1.0);
}

#[test]
fn matches_golden_under_uniform_flow() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamStep::new(&ctx);
    let (nx, nz) = (16u32, 16u32);
    let d = foam(nx, nz);
    let s = source_field(nx, nz);
    let n = (nx * nz) as usize;
    // Uniform drift in several directions crossed with decay configs, including
    // churning (fast) and nearly calm flow relative to the reference speed.
    for &(ux, vz, dt, decay, floor, reference) in &[
        (1.0, 0.0, 0.3, 0.5, 0.1, 1.0),
        (0.0, 2.0, 0.5, 0.8, 0.0, 2.5),
        (-1.5, -2.5, 0.4, 1.2, 0.3, 1.5),
    ] {
        let u = vec![ux; n];
        let v = vec![vz; n];
        check(
            &ctx, &gpu, &d, &u, &v, &s, nx, nz, 0.5, dt, decay, floor, reference,
        );
    }
}

#[test]
fn matches_golden_under_divergent_flow() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamStep::new(&ctx);
    let (nx, nz) = (13u32, 11u32);
    let d = foam(nx, nz);
    let s = source_field(nx, nz);
    // A swirling/divergent flow so each cell both backtraces elsewhere and sees
    // a distinct local flow speed (hence a distinct decay rate).
    let mut u = Vec::with_capacity((nx * nz) as usize);
    let mut v = Vec::with_capacity((nx * nz) as usize);
    let cx = (nx as f32 - 1.0) * 0.5;
    let cz = (nz as f32 - 1.0) * 0.5;
    for z in 0..nz {
        for x in 0..nx {
            let dxp = x as f32 - cx;
            let dzp = z as f32 - cz;
            u.push(-dzp * 0.6);
            v.push(dxp * 0.6);
        }
    }
    check(&ctx, &gpu, &d, &u, &v, &s, nx, nz, 1.0, 0.35, 0.9, 0.2, 1.3);
}

#[test]
fn matches_golden_on_non_square_grid_and_partial_workgroup() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamStep::new(&ctx);
    // Non-square grids whose cell counts are not multiples of the 256-wide
    // workgroup exercise the tail guard.
    for &(nx, nz) in &[(1u32, 1u32), (3, 7), (17, 15), (31, 9)] {
        let d = foam(nx, nz);
        let s = source_field(nx, nz);
        let n = (nx * nz) as usize;
        let u = vec![0.8_f32; n];
        let v = vec![-1.1_f32; n];
        check(
            &ctx, &gpu, &d, &u, &v, &s, nx, nz, 0.75, 0.45, 0.6, 0.15, 1.0,
        );
    }
}

#[test]
fn is_deterministic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamStep::new(&ctx);
    let (nx, nz) = (20u32, 14u32);
    let d = foam(nx, nz);
    let s = source_field(nx, nz);
    let n = (nx * nz) as usize;
    let u = vec![1.2_f32; n];
    let v = vec![0.9_f32; n];
    let a = gpu.evaluate(&ctx, &d, &u, &v, &s, nx, nz, 0.6, 0.4, 0.7, 0.2, 1.1);
    let b = gpu.evaluate(&ctx, &d, &u, &v, &s, nx, nz, 0.6, 0.4, 0.7, 0.2, 1.1);
    assert_eq!(a, b, "the same request steps identically");
}
