//! Real-device parity test for the weather-map coverage-mass reduction twin.
//!
//! Builds real [`WeatherField`](prism_render_architecture::volumetric::weather::WeatherField)
//! grids (including a pre/post-advection pair, the mass-conservation case the
//! reduction exists to check) and confirms the `wgpu` [`GpuTotalCoverage`]
//! kernel reproduces each field's
//! [`WeatherField::total_coverage`](prism_render_architecture::volumetric::weather::WeatherField::total_coverage)
//! sum. Fields of differing sizes are reduced in one batched dispatch, so this
//! exercises the flattened-array-plus-ranges layout. It also checks the
//! empty-batch and all-empty-fields fast paths.

#![expect(
    clippy::print_stderr,
    reason = "test prints a skip notice when no GPU adapter is available"
)]

use prism_render_architecture::volumetric::math::Vec2;
use prism_render_architecture::volumetric::weather::{advect_semi_lagrangian, WeatherField};
use prism_render_architecture::volumetric::WeatherMapHandle;
use prism_render_architecture::volumetric::WeatherSample;
use prism_volumetric_gpu::GpuContext;
use prism_volumetric_gpu::{GpuTotalCoverage, WeatherAdvectSample};

/// Builds a deterministic, spatially varying weather field.
fn build_field(width: u32, height: u32) -> WeatherField {
    let mut cells = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / (width as f32 + 1.0);
            let fy = y as f32 / (height as f32 + 1.0);
            // Deterministic ramps kept inside [0, 1] by construction; the
            // coverage (red) channel varies so the sum is non-trivial.
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

#[test]
fn total_coverage_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping total_coverage_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuTotalCoverage::new(&ctx);

    let a = build_field(17, 11);
    let b = build_field(9, 1);
    let c = build_field(1, 1);
    // A pre/post-advection pair: the reduction bounds mass drift across a step.
    let advected = advect_semi_lagrangian(&a, Vec2::new(1.3, -0.7), 1.0);

    let fields = [
        snapshot(&a),
        snapshot(&b),
        snapshot(&c),
        snapshot(&advected),
    ];
    let expected = [
        a.total_coverage(),
        b.total_coverage(),
        c.total_coverage(),
        advected.total_coverage(),
    ];

    let gpu = kernel.eval(&ctx, &fields);
    assert_eq!(gpu.len(), expected.len(), "length mismatch");
    for (i, (g, e)) in gpu.iter().zip(expected.iter()).enumerate() {
        // Tolerance scales with the summed cell count (interior fields have a
        // few hundred cells), so allow a small relative slack too.
        assert!(
            (g - e).abs() < 1e-4 || (g - e).abs() < 1e-4 * e.abs(),
            "field {i} mismatch: gpu {g} vs cpu {e}"
        );
    }
}

#[test]
fn total_coverage_empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping total_coverage_parity empty: no GPU adapter available");
        return;
    };
    let kernel = GpuTotalCoverage::new(&ctx);
    let gpu = kernel.eval(&ctx, &[]);
    assert!(gpu.is_empty());
}

#[test]
fn total_coverage_all_empty_fields_are_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping total_coverage_parity all-empty: no GPU adapter available");
        return;
    };
    let kernel = GpuTotalCoverage::new(&ctx);
    let fields: [Vec<WeatherAdvectSample>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let gpu = kernel.eval(&ctx, &fields);
    assert_eq!(gpu, vec![0.0, 0.0, 0.0]);
}
