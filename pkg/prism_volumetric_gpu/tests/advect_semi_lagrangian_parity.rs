//! Real-device parity test for the semi-Lagrangian weather-map advection twin.
//!
//! Builds real [`WeatherField`](prism_render_architecture::volumetric::weather::WeatherField)
//! grids, advects them one step under several uniform winds and step sizes on
//! the CPU via
//! [`advect_semi_lagrangian`](prism_render_architecture::volumetric::weather::advect_semi_lagrangian),
//! then confirms the `wgpu` [`GpuAdvectSemiLagrangian`] kernel reproduces every
//! output cell. Winds are chosen so some backtraces land inside the grid and
//! others fall off the edge (exercising the clamp-to-edge path). It also checks
//! the empty-field fast path.

#![expect(
    clippy::print_stderr,
    reason = "test prints a skip notice when no GPU adapter is available"
)]

use prism_render_architecture::volumetric::math::Vec2;
use prism_render_architecture::volumetric::weather::{advect_semi_lagrangian, WeatherField};
use prism_render_architecture::volumetric::WeatherMapHandle;
use prism_render_architecture::volumetric::WeatherSample;
use prism_volumetric_gpu::GpuContext;
use prism_volumetric_gpu::{GpuAdvectSemiLagrangian, WeatherAdvectSample};

/// Builds a deterministic, spatially varying weather field.
fn build_field(width: u32, height: u32) -> WeatherField {
    let mut cells = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / (width as f32);
            let fy = y as f32 / (height as f32);
            // Deterministic ramps kept inside [0, 1] by construction.
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
    kernel: &GpuAdvectSemiLagrangian,
    field: &WeatherField,
    wind: Vec2,
    dt: f32,
    label: &str,
) {
    let cpu = advect_semi_lagrangian(field, wind, dt);
    let input = snapshot(field);
    let gpu = kernel.eval(
        ctx,
        &input,
        field.width(),
        field.height(),
        (wind.x, wind.y),
        dt,
    );
    let cpu_cells = cpu.cells();
    assert_eq!(gpu.len(), cpu_cells.len(), "{label}: length mismatch");
    for (i, (g, c)) in gpu.iter().zip(cpu_cells.iter()).enumerate() {
        assert!(
            (g.coverage - c.coverage).abs() < 1e-6
                && (g.cloud_type - c.cloud_type).abs() < 1e-6
                && (g.precipitation - c.precipitation).abs() < 1e-6
                && (g.wind_disturbance - c.wind_disturbance).abs() < 1e-6,
            "{label}: cell {i} mismatch: gpu {g:?} vs cpu {c:?}"
        );
    }
}

#[test]
fn advect_semi_lagrangian_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping advect_semi_lagrangian_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuAdvectSemiLagrangian::new(&ctx);

    let field = build_field(17, 11);

    // Interior fractional backtrace.
    check(&ctx, &kernel, &field, Vec2::new(1.3, -0.7), 1.0, "interior");
    // Zero wind is an identity map.
    check(&ctx, &kernel, &field, Vec2::ZERO, 1.0, "identity");
    // Large winds that push most backtraces off the grid (clamp-to-edge).
    check(
        &ctx,
        &kernel,
        &field,
        Vec2::new(-50.0, 40.0),
        1.0,
        "off_grid",
    );
    // Fractional step that is not a whole cell.
    check(
        &ctx,
        &kernel,
        &field,
        Vec2::new(2.5, 3.25),
        0.37,
        "fractional_dt",
    );

    // A degenerate single-row field: every vertical backtrace clamps.
    let row = build_field(9, 1);
    check(&ctx, &kernel, &row, Vec2::new(1.5, 2.0), 1.0, "single_row");

    // A single-cell field is a fixed point regardless of wind.
    let one = build_field(1, 1);
    check(
        &ctx,
        &kernel,
        &one,
        Vec2::new(3.0, -2.0),
        1.0,
        "single_cell",
    );
}

#[test]
fn advect_semi_lagrangian_empty_field_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping advect_semi_lagrangian_parity empty: no GPU adapter available");
        return;
    };
    let kernel = GpuAdvectSemiLagrangian::new(&ctx);
    let gpu = kernel.eval(&ctx, &[], 0, 0, (1.0, 1.0), 1.0);
    assert!(gpu.is_empty());
}
