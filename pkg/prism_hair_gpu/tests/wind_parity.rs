//! Real-device parity for the guide-strand wind-field twin:
//! [`GpuWindField`] must reproduce the `CPU` golden
//! [`wind_acceleration`](prism_render_architecture::hair::wind::wind_acceleration)
//! for every query across steady, gusting and turbulent fields, including the
//! calm and clamped edge cases the reference guards.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The field is a closed-form polynomial (the reference deliberately avoids
//! `f32::sin`), so the `CPU` and `GPU` evaluate the same expression and diverge
//! only through legal fused-multiply-add contraction. Each component is
//! asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to
//! fail a genuinely wrong port (a dropped term, a swapped axis, a missing range
//! reduction), loose enough to admit fma contraction. The sweeps also assert
//! the physical result (the steady term is recovered, and a turbulent field
//! produces position-varying output) so a degenerate all-constant kernel could
//! not pass.
//!
//! Provenance: standard steady+gust+turbulence wind acceleration with a
//! hand-written Taylor sine; no Unreal Engine source or derived code.

use prism_hair_gpu::wind::{query_for_wind, GpuWindField, WindQuery};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::dynamics::Vec3;
use prism_render_architecture::hair::wind::{wind_acceleration, WindField};

/// Asserts `gpu` matches the `CPU` golden [`wind_acceleration`] for every query
/// to within the documented fma tolerance. `fields`, `points` and `times` are
/// the reference inputs, aligned with `gpu` by index.
fn assert_parity(fields: &[WindField], points: &[Vec3], times: &[f32], gpu: &[[f32; 3]]) {
    assert_eq!(gpu.len(), fields.len(), "one acceleration per query");
    assert_eq!(points.len(), fields.len(), "one point per field");
    assert_eq!(times.len(), fields.len(), "one time per field");
    for i in 0..fields.len() {
        let expected = wind_acceleration(fields[i], points[i], times[i]);
        let got = gpu[i];
        for (axis, (g, e)) in [
            (got[0], expected.x),
            (got[1], expected.y),
            (got[2], expected.z),
        ]
        .into_iter()
        .enumerate()
        {
            let abs_diff = (g - e).abs();
            let rel_diff = abs_diff / e.abs().max(1e-6);
            assert!(
                abs_diff < 1e-4 || rel_diff < 1e-3,
                "wind mismatch for query {i} axis {axis}: gpu {g}, cpu {e} (abs {abs_diff}, rel {rel_diff})"
            );
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_steady_and_gust_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping wind parity: no wgpu adapter on this host");
        return;
    };
    let field = WindField {
        direction: Vec3::new(1.0, 0.0, 0.0),
        speed: 3.5,
        gust_amplitude: 1.25,
        gust_frequency: 0.7,
        turbulence: 0.0,
    };

    let mut fields = Vec::new();
    let mut points = Vec::new();
    let mut times = Vec::new();
    let mut queries = Vec::new();
    for k in 0..48u32 {
        let t = k as f32 * 0.13;
        let p = Vec3::new(0.2 * k as f32, -0.1 * k as f32, 0.05 * k as f32);
        fields.push(field);
        points.push(p);
        times.push(t);
        queries.push(query_for_wind(field, p, t));
    }

    let out = GpuWindField::new(&ctx).eval(&ctx, &queries);
    assert_parity(&fields, &points, &times, &out);

    // Physical shape: with no turbulence the along-wind (x) accel stays within
    // speed +/- amplitude, and the cross-wind (y, z) accels are zero.
    for got in &out {
        assert!(
            got[0] >= 3.5 - 1.25 - 1e-3 && got[0] <= 3.5 + 1.25 + 1e-3,
            "along-wind accel {} outside steady +/- gust band",
            got[0]
        );
        assert!(
            got[1].abs() < 1e-4,
            "cross-wind y should be zero: {}",
            got[1]
        );
        assert!(
            got[2].abs() < 1e-4,
            "cross-wind z should be zero: {}",
            got[2]
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_full_turbulent_field_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping wind parity: no wgpu adapter on this host");
        return;
    };
    let field = WindField {
        direction: Vec3::new(0.3, 1.0, -0.4),
        speed: 2.0,
        gust_amplitude: 0.9,
        gust_frequency: 1.4,
        turbulence: 0.6,
    };

    let mut fields = Vec::new();
    let mut points = Vec::new();
    let mut times = Vec::new();
    let mut queries = Vec::new();
    for k in 0..64u32 {
        let t = 0.37 + k as f32 * 0.091;
        let p = Vec3::new(
            0.11 * k as f32 - 1.0,
            0.23 * k as f32 - 2.0,
            -0.17 * k as f32 + 0.5,
        );
        fields.push(field);
        points.push(p);
        times.push(t);
        queries.push(query_for_wind(field, p, t));
    }

    let out = GpuWindField::new(&ctx).eval(&ctx, &queries);
    assert_parity(&fields, &points, &times, &out);

    // Non-degenerate: a turbulent field must produce output that varies across
    // sample points, so a constant-returning kernel could not pass.
    let first = out[0];
    let varies = out
        .iter()
        .any(|g| (g[0] - first[0]).abs() > 1e-4 || (g[1] - first[1]).abs() > 1e-4);
    assert!(varies, "turbulent field output should vary across samples");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_calm_and_clamped_edge_fields_match_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping wind parity: no wgpu adapter on this host");
        return;
    };

    // A calm field exerts no force; a zero-direction field cannot push; a
    // negative gust frequency is clamped to zero by the kernel and the golden.
    let calm = WindField::CALM;
    let zero_dir = WindField {
        direction: Vec3::ZERO,
        speed: 5.0,
        gust_amplitude: 2.0,
        gust_frequency: 1.0,
        turbulence: 0.0,
    };
    let neg_freq = WindField {
        direction: Vec3::new(0.0, 0.0, 1.0),
        speed: 1.0,
        gust_amplitude: 3.0,
        gust_frequency: -5.0,
        turbulence: 0.4,
    };

    let fields = [calm, zero_dir, neg_freq, neg_freq];
    let points = [
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(-0.5, 0.5, 0.25),
        Vec3::new(0.7, -0.2, 1.1),
        Vec3::new(4.0, 4.0, 4.0),
    ];
    let times = [5.0, 1.5, 0.0, 2.75];
    let queries: Vec<WindQuery> = (0..fields.len())
        .map(|i| query_for_wind(fields[i], points[i], times[i]))
        .collect();

    let out = GpuWindField::new(&ctx).eval(&ctx, &queries);
    assert_parity(&fields, &points, &times, &out);

    // Calm field yields exactly zero acceleration.
    assert!(out[0][0].abs() < 1e-6 && out[0][1].abs() < 1e-6 && out[0][2].abs() < 1e-6);
    // Zero direction cannot push, even with speed and gust set.
    assert!(out[1][0].abs() < 1e-4 && out[1][1].abs() < 1e-4 && out[1][2].abs() < 1e-4);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_empty_queries_yields_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping wind parity: no wgpu adapter on this host");
        return;
    };
    let out = GpuWindField::new(&ctx).eval(&ctx, &[]);
    assert!(out.is_empty(), "empty input yields empty output");
}
