//! Real-device parity test for the wind-field weather-map advection twin.
//!
//! Builds real [`WeatherField`](prism_render_architecture::volumetric::weather::WeatherField)
//! grids, advects them one step under several
//! [`WindField`](prism_render_architecture::volumetric::weather::WindField)s
//! (base flow plus divergence-free curl gust) and step sizes on the CPU via
//! [`advect_with_wind`](prism_render_architecture::volumetric::weather::advect_with_wind),
//! then confirms the `wgpu` [`GpuAdvectWithWind`] kernel reproduces every output
//! cell. Because each cell samples its own velocity (including the hand-rolled
//! `sin_approx`/`cos_approx` curl term modulated by the cell's wind-disturbance
//! channel), this also exercises the shader's polynomial trig against the
//! golden. Winds are chosen so some backtraces land inside the grid and others
//! fall off the edge (clamp-to-edge). It also checks the empty-field fast path.

#![expect(
    clippy::print_stderr,
    reason = "test prints a skip notice when no GPU adapter is available"
)]

use prism_render_architecture::volumetric::math::Vec2;
use prism_render_architecture::volumetric::weather::{advect_with_wind, WeatherField, WindField};
use prism_render_architecture::volumetric::WeatherMapHandle;
use prism_render_architecture::volumetric::WeatherSample;
use prism_volumetric_gpu::GpuContext;
use prism_volumetric_gpu::{GpuAdvectWithWind, WeatherAdvectSample};

/// Builds a deterministic, spatially varying weather field.
fn build_field(width: u32, height: u32) -> WeatherField {
    let mut cells = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / (width as f32);
            let fy = y as f32 / (height as f32);
            // Deterministic ramps kept inside [0, 1] by construction. The alpha
            // channel (wind_disturbance) varies so the per-cell gust differs.
            let r = 0.5 * fx + 0.5 * fy;
            let g = fx;
            let b = fy;
            let a = 0.25 + 0.5 * fx * fy;
            cells.push(WeatherSample::from_rgba(r, g, b, a));
        }
    }
    WeatherField::from_cells(WeatherMapHandle(1), width, height, cells)
        .expect("cell count matches dimensions")
}

/// Snapshots a field's cells into the GPU sample type in row-major order.
fn snapshot(field: &WeatherField) -> Vec<WeatherAdvectSample> {
    field
        .cells()
        .iter()
        .map(|s| WeatherAdvectSample {
            coverage: s.coverage,
            cloud_type: s.cloud_type,
            precipitation: s.precipitation,
            wind_disturbance: s.wind_disturbance,
        })
        .collect()
}

fn check(
    ctx: &GpuContext,
    kernel: &GpuAdvectWithWind,
    field: &WeatherField,
    wind: &WindField,
    dt: f32,
    label: &str,
) {
    let cpu = advect_with_wind(field, wind, dt);
    let input = snapshot(field);
    let dir = wind.direction();
    let gpu = kernel.eval(
        ctx,
        &input,
        field.width(),
        field.height(),
        (dir.x, dir.y),
        wind.speed(),
        wind.curl_strength(),
        dt,
    );
    let cpu_cells = cpu.cells();
    assert_eq!(gpu.len(), cpu_cells.len(), "{label}: length mismatch");
    for (i, (g, c)) in gpu.iter().zip(cpu_cells.iter()).enumerate() {
        assert!(
            (g.coverage - c.coverage).abs() < 1e-5
                && (g.cloud_type - c.cloud_type).abs() < 1e-5
                && (g.precipitation - c.precipitation).abs() < 1e-5
                && (g.wind_disturbance - c.wind_disturbance).abs() < 1e-5,
            "{label}: cell {i} mismatch: gpu {g:?} vs cpu {c:?}"
        );
    }
}

#[test]
fn advect_with_wind_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping advect_with_wind_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuAdvectWithWind::new(&ctx);

    let field = build_field(17, 11);

    // Interior fractional backtrace with an active curl gust.
    let interior = WindField::new(Vec2::new(1.0, -0.5), 1.3, 0.8);
    check(&ctx, &kernel, &field, &interior, 1.0, "interior");

    // Pure uniform wind (no curl): identity of the curl term, still traces.
    let uniform = WindField::new(Vec2::new(0.7, 0.2), 1.1, 0.0);
    check(&ctx, &kernel, &field, &uniform, 1.0, "uniform");

    // Large wind that pushes most backtraces off the grid (clamp-to-edge).
    let off_grid = WindField::new(Vec2::new(-1.0, 0.9), 45.0, 1.5);
    check(&ctx, &kernel, &field, &off_grid, 1.0, "off_grid");

    // Polar construction, fractional step not a whole cell.
    let polar = WindField::from_polar(0.9, 2.4, 1.2);
    check(&ctx, &kernel, &field, &polar, 0.37, "polar_fractional_dt");

    // Zero speed: pure fixed point (velocity is exactly zero everywhere).
    let calm = WindField::new(Vec2::new(1.0, 1.0), 0.0, 0.0);
    check(&ctx, &kernel, &field, &calm, 1.0, "calm");

    // A degenerate single-row field: every vertical backtrace clamps.
    let row = build_field(9, 1);
    check(&ctx, &kernel, &row, &interior, 1.0, "single_row");

    // A single-cell field is a fixed point regardless of wind.
    let one = build_field(1, 1);
    check(&ctx, &kernel, &one, &off_grid, 1.0, "single_cell");
}

#[test]
fn advect_with_wind_empty_field_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping advect_with_wind_parity empty: no GPU adapter available");
        return;
    };
    let kernel = GpuAdvectWithWind::new(&ctx);
    let gpu = kernel.eval(&ctx, &[], 0, 0, (1.0, 0.0), 1.0, 1.0, 1.0);
    assert!(gpu.is_empty());
}
