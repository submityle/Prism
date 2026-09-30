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

/// Builds an `n x n` field whose `coverage` is a compact interior bump that
/// vanishes well before the border, mirroring the CPU golden's `interior_bump`.
/// Because the bump never touches the edge, a fractional interior advection
/// backtrace stays inside the domain, so clamp-to-edge sampling neither gains
/// nor loses mass — the setup under which coverage is a conserved quantity.
fn interior_bump_field(n: u32) -> WeatherField {
    let center = (n as f32 - 1.0) * 0.5;
    let mut cells = Vec::with_capacity((n * n) as usize);
    for y in 0..n {
        for x in 0..n {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let r2 = dx * dx + dy * dy;
            // A compact quadratic bump that reaches zero before the border.
            let cov = (1.0 - r2 / 9.0).clamp(0.0, 1.0);
            cells.push(WeatherSample::from_rgba(cov, 0.5, 0.0, 0.0));
        }
    }
    WeatherField::from_cells(WeatherMapHandle(2), n, n, cells)
        .expect("cell count matches dimensions")
}

/// Section 16 invariant, certified on the on-device reduction (not merely
/// implied by CPU parity): under a divergence-free (uniform) wind the weather
/// map's total `coverage` is conserved up to bilinear rounding. Advecting a
/// compact interior bump by a fractional interior shift keeps every backtrace
/// tap inside the domain, so no mass leaves through the clamped border; the
/// GPU-reduced pre/post totals must therefore agree to within the documented
/// mass-drift bound. A lossy or mis-strided GPU reduction would report false
/// drift and fail this guard even while a same-field parity check still passed.
#[test]
fn total_coverage_certifies_mass_conservation_on_gpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping total_coverage mass conservation: no GPU adapter available");
        return;
    };
    let kernel = GpuTotalCoverage::new(&ctx);

    let before_field = interior_bump_field(24);
    // A fractional interior shift: every backtrace tap stays interior, matching
    // the CPU golden's `uniform_wind_advection_conserves_mass`.
    let after_field = advect_semi_lagrangian(&before_field, Vec2::new(0.73, -0.41), 1.0);

    let gpu = kernel.eval(&ctx, &[snapshot(&before_field), snapshot(&after_field)]);
    assert_eq!(gpu.len(), 2, "one total per field");

    // Anchor both on-device totals to the CPU golden so the drift below is
    // measured against trustworthy sums, not two coincidentally-equal errors.
    let before_cpu = before_field.total_coverage();
    let after_cpu = after_field.total_coverage();
    assert!(
        (gpu[0] - before_cpu).abs() < 1e-4 * before_cpu.max(1.0),
        "pre-advection total: gpu {} vs cpu {before_cpu}",
        gpu[0]
    );
    assert!(
        (gpu[1] - after_cpu).abs() < 1e-4 * after_cpu.max(1.0),
        "post-advection total: gpu {} vs cpu {after_cpu}",
        gpu[1]
    );

    // The mass-conservation bound, asserted on the GPU-reduced totals.
    let before_gpu = gpu[0];
    let after_gpu = gpu[1];
    assert!(
        before_gpu > 0.0,
        "the interior bump must carry positive mass"
    );
    let drift = (after_gpu - before_gpu).abs() / before_gpu;
    assert!(
        drift < 1e-3,
        "gpu-reduced mass drift {drift} exceeds the conservation bound \
         (before {before_gpu}, after {after_gpu})"
    );
}
