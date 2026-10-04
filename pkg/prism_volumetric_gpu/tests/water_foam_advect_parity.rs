//! Real-device parity for the foam semi-Lagrangian advection twin:
//! [`GpuWaterFoamAdvect`](prism_volumetric_gpu::water_foam_advect::GpuWaterFoamAdvect)
//! must reproduce the dependency-free `CPU` golden
//! [`advect_foam_field`](prism_render_architecture::water::foam::advect_foam_field)
//! — the unconditionally stable backtrace-and-bilinear-gather `Niagara` and
//! `Crest` foam passes run — across uniform, structured and divergent flows,
//! non-square grids, a zero step, and the degenerate request.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`advect_foam_field`] is public and pure, so the expected field
//! is built in-host and compared cell by cell against the `GPU` readback. A
//! `GPU == oracle` pass is therefore directly a `GPU == golden` pass.
//!
//! # Parity criterion
//!
//! Each cell performs the identical backtrace and two-tap-then-one-tap bilinear
//! blend as the golden, in the same order, so the only residual is the
//! last-place slack of a `GPU` fused multiply-add. Each cell is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::foam`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::foam::{advect_foam_field, FoamConfig};
use prism_volumetric_gpu::water_foam_advect::{GpuWaterFoamAdvect, WaterFoamAdvect};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on one advected sample.
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

/// A foam config with the given grid; the decay fields are unused by advection
/// but carried so the config matches the golden's type.
fn cfg(nx: u32, nz: u32, dx: f32) -> FoamConfig {
    FoamConfig {
        nx,
        nz,
        dx,
        base_decay: 0.5,
        persistence_floor: 0.1,
        reference_speed: 1.0,
    }
}

/// A deterministic blob-plus-ripple foam field over `nx * nz` cells.
fn foam(nx: u32, nz: u32) -> Vec<f32> {
    let mut d = Vec::with_capacity((nx * nz) as usize);
    for z in 0..nz {
        for x in 0..nx {
            let fx = x as f32;
            let fz = z as f32;
            // A smooth bump centred on the grid plus a mild checker so the
            // bilinear taps are non-trivial; clamped non-negative.
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

/// Pins one `GPU` advection against the `CPU` golden, cell by cell.
#[expect(clippy::too_many_arguments, reason = "mirrors the golden signature")]
fn check(
    ctx: &GpuContext,
    gpu: &GpuWaterFoamAdvect,
    density: &[f32],
    u: &[f32],
    v: &[f32],
    nx: u32,
    nz: u32,
    dx: f32,
    dt: f32,
) {
    let config = cfg(nx, nz, dx);
    let want = advect_foam_field(density, u, v, config, dt);
    let got = gpu.evaluate(ctx, density, u, v, nx, nz, dx, dt);
    let label = format!("nx={nx} nz={nz} dx={dx} dt={dt}");
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
        eprintln!("skipping water_foam_advect parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterFoamAdvect::new(&ctx);
    // An empty grid and a short input are both honest no-ops that return the
    // density unchanged, exactly like the golden.
    assert_eq!(
        gpu.evaluate(&ctx, &[], &[], &[], 0, 0, 1.0, 0.1),
        WaterFoamAdvect {
            density: Vec::new()
        },
        "empty grid"
    );
    let short = vec![0.3_f32, 0.4];
    assert_eq!(
        gpu.evaluate(&ctx, &short, &short, &short, 4, 4, 1.0, 0.1),
        WaterFoamAdvect {
            density: short.clone()
        },
        "short input"
    );
}

#[test]
fn zero_step_preserves_the_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamAdvect::new(&ctx);
    let (nx, nz) = (12u32, 9u32);
    let d = foam(nx, nz);
    let u = vec![1.5_f32; (nx * nz) as usize];
    let v = vec![-0.7_f32; (nx * nz) as usize];
    // A zero step backtraces nowhere; every cell samples itself.
    check(&ctx, &gpu, &d, &u, &v, nx, nz, 1.0, 0.0);
}

#[test]
fn matches_golden_under_uniform_flow() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamAdvect::new(&ctx);
    let (nx, nz) = (16u32, 16u32);
    let d = foam(nx, nz);
    let n = (nx * nz) as usize;
    // Uniform drift in several directions, including sub-cell and multi-cell
    // backtraces and off-grid clamps.
    for &(ux, vz, dt) in &[(1.0, 0.0, 0.3), (0.0, 2.0, 0.5), (-1.5, -2.5, 0.4)] {
        let u = vec![ux; n];
        let v = vec![vz; n];
        check(&ctx, &gpu, &d, &u, &v, nx, nz, 0.5, dt);
    }
}

#[test]
fn matches_golden_under_divergent_flow() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamAdvect::new(&ctx);
    let (nx, nz) = (13u32, 11u32);
    let d = foam(nx, nz);
    // A swirling/divergent flow so each cell backtraces to a different place.
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
    check(&ctx, &gpu, &d, &u, &v, nx, nz, 1.0, 0.35);
}

#[test]
fn matches_golden_on_non_square_grid_and_partial_workgroup() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamAdvect::new(&ctx);
    // Non-square grids whose cell counts are not multiples of the 256-wide
    // workgroup exercise the tail guard.
    for &(nx, nz) in &[(1u32, 1u32), (3, 7), (17, 15), (31, 9)] {
        let d = foam(nx, nz);
        let n = (nx * nz) as usize;
        let u = vec![0.8_f32; n];
        let v = vec![-1.1_f32; n];
        check(&ctx, &gpu, &d, &u, &v, nx, nz, 0.75, 0.45);
    }
}

#[test]
fn is_deterministic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFoamAdvect::new(&ctx);
    let (nx, nz) = (20u32, 14u32);
    let d = foam(nx, nz);
    let n = (nx * nz) as usize;
    let u = vec![1.2_f32; n];
    let v = vec![0.9_f32; n];
    let a = gpu.evaluate(&ctx, &d, &u, &v, nx, nz, 0.6, 0.4);
    let b = gpu.evaluate(&ctx, &d, &u, &v, nx, nz, 0.6, 0.4);
    assert_eq!(a, b, "the same request advects identically");
}
