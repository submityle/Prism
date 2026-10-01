//! Real-device parity for the divergence-free curl-noise wind-field twin:
//! [`GpuHairCurlWind`] must reproduce the `CPU` golden
//! [`reference_curl_wind_map`](prism_hair_gpu::wind_field::reference_curl_wind_map)
//! (built on
//! [`curl_wind_map`](prism_render_architecture::hair::wind_field::curl_wind_map))
//! for a batch of world positions sharing one [`WindFieldParams`] and time,
//! mapping each position to its wind velocity independently. The suite drives a
//! deterministic repeat, independent fields from different seeds, time
//! evolution, amplitude scaling, negative / non-finite `frequency` repaired to
//! `1.0`, negative / non-finite `amplitude` repaired to `0.0` (a dead field),
//! the empty no-op, and a large batch that crosses the 64-wide dispatch
//! boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each velocity component is a long multiply-add chain (trilinear value noise
//! plus a central-difference curl scaled by `inv_two_eps = 500`) a `GPU` may
//! fuse, so every component is asserted within `abs_diff < 5e-3` or
//! `rel_diff < 5e-3` (a band widened from the usual tolerance because the
//! central-difference curl scales the potential difference by
//! `inv_two_eps = 500`, amplifying any fma reassociation 500-fold). Beyond matching the golden component by component, every
//! result is asserted finite. All positions are finite explicit literals or
//! integer-derived fractions; no `sin`/`cos` appears anywhere. The golden does
//! not sanitize positions, so the batches use only finite coordinates (the
//! sanitizer contract covers `frequency` / `amplitude`, which the dedicated
//! cases exercise).
//!
//! Provenance: `Bridson` 2007 curl-noise and `wgpu` compute dispatch; no
//! third-party engine source or derived code.

use prism_hair_gpu::wind_field::{reference_curl_wind_map, GpuHairCurlWind};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::wind_field::WindFieldParams;

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// Asserts two scalars agree within the documented fma tolerance.
fn assert_close(got: f32, want: f32, what: &str) {
    let abs = (got - want).abs();
    let rel = abs / want.abs().max(1.0);
    assert!(
        abs < 5e-3 || rel < 5e-3,
        "{what}: got {got}, want {want} (abs {abs}, rel {rel})"
    );
}

/// Asserts a whole batch matches the `CPU` golden component by component and
/// that every component is finite.
fn assert_batch(got: &[[f32; 3]], params: WindFieldParams, positions: &[[f32; 3]], time: f32) {
    let want = reference_curl_wind_map(params, positions, time);
    assert_eq!(got.len(), positions.len(), "one velocity per position");
    assert_eq!(got.len(), want.len(), "golden length matches query length");
    for (i, (&out, &reference)) in got.iter().zip(want.iter()).enumerate() {
        for axis in 0..3 {
            assert_close(
                out[axis],
                reference[axis],
                &format!("position {i} axis {axis}"),
            );
            assert!(
                out[axis].is_finite(),
                "position {i} axis {axis} must be finite, got {}",
                out[axis]
            );
        }
    }
}

/// Dispatches one batch through the device twin.
fn run(
    ctx: &GpuContext,
    params: WindFieldParams,
    positions: &[[f32; 3]],
    time: f32,
) -> Vec<[f32; 3]> {
    GpuHairCurlWind::new(ctx).eval(ctx, params, positions, time)
}

/// A deterministic finite sample cloud spanning several lattice cells and signs.
fn sample_positions() -> Vec<[f32; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [0.3, -1.2, 4.7],
        [-2.5, 3.1, -0.8],
        [1.25, 1.25, 1.25],
        [-5.0, -5.0, 5.0],
        [7.3, -0.1, 2.2],
        [0.75, -3.4, -6.1],
        [12.0, 0.5, -2.5],
    ]
}

#[test]
fn deterministic_batch_matches_golden() {
    let Some(ctx) = context_or_skip("deterministic_batch_matches_golden") else {
        return;
    };
    let params = WindFieldParams::new(1.5, 2.0, 7);
    let positions = sample_positions();
    assert_batch(
        &run(&ctx, params, &positions, 0.25),
        params,
        &positions,
        0.25,
    );
}

#[test]
fn repeat_run_is_stable() {
    let Some(ctx) = context_or_skip("repeat_run_is_stable") else {
        return;
    };
    // Same input twice must read back identical bits — the field is fully
    // deterministic on-device.
    let params = WindFieldParams::new(0.8, 1.3, 42);
    let positions = sample_positions();
    let a = run(&ctx, params, &positions, 0.5);
    let b = run(&ctx, params, &positions, 0.5);
    assert_eq!(a, b, "repeated dispatch must be bit-identical");
}

#[test]
fn different_seeds_give_independent_fields() {
    let Some(ctx) = context_or_skip("different_seeds_give_independent_fields") else {
        return;
    };
    let positions = sample_positions();
    let p0 = WindFieldParams::new(1.0, 1.0, 1);
    let p1 = WindFieldParams::new(1.0, 1.0, 9999);
    let a = run(&ctx, p0, &positions, 0.0);
    let b = run(&ctx, p1, &positions, 0.0);
    assert_batch(&a, p0, &positions, 0.0);
    assert_batch(&b, p1, &positions, 0.0);
    // Different seeds reseed every lattice corner, so the two fields must
    // disagree somewhere beyond fma noise.
    let differs = a
        .iter()
        .zip(b.iter())
        .any(|(va, vb)| (0..3).any(|k| (va[k] - vb[k]).abs() > 1e-2));
    assert!(differs, "distinct seeds must produce distinct fields");
}

#[test]
fn time_evolves_the_field() {
    let Some(ctx) = context_or_skip("time_evolves_the_field") else {
        return;
    };
    let params = WindFieldParams::new(1.2, 1.0, 5);
    let positions = sample_positions();
    let a = run(&ctx, params, &positions, 0.0);
    let b = run(&ctx, params, &positions, 3.0);
    assert_batch(&a, params, &positions, 0.0);
    assert_batch(&b, params, &positions, 3.0);
    let differs = a
        .iter()
        .zip(b.iter())
        .any(|(va, vb)| (0..3).any(|k| (va[k] - vb[k]).abs() > 1e-2));
    assert!(differs, "advancing time must evolve the field");
}

#[test]
fn amplitude_scales_the_field() {
    let Some(ctx) = context_or_skip("amplitude_scales_the_field") else {
        return;
    };
    // Amplitude is a pure output scale; both scales must match their own golden.
    let positions = sample_positions();
    for amp in [0.5, 1.0, 4.0] {
        let params = WindFieldParams::new(1.1, amp, 3);
        assert_batch(
            &run(&ctx, params, &positions, 0.75),
            params,
            &positions,
            0.75,
        );
    }
}

#[test]
fn non_finite_or_negative_frequency_repairs_to_one() {
    let Some(ctx) = context_or_skip("non_finite_or_negative_frequency_repairs_to_one") else {
        return;
    };
    // A non-finite / non-positive frequency falls back to 1.0 on both sides, so
    // each batch still matches its golden (evaluated with the same raw params).
    let positions = sample_positions();
    for freq in [-2.0, 0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let params = WindFieldParams::new(freq, 1.5, 11);
        assert_batch(&run(&ctx, params, &positions, 0.2), params, &positions, 0.2);
    }
}

#[test]
fn non_finite_or_negative_amplitude_is_dead_field() {
    let Some(ctx) = context_or_skip("non_finite_or_negative_amplitude_is_dead_field") else {
        return;
    };
    // A non-finite / negative amplitude falls back to 0.0 (no wind), so the
    // whole batch reads back exactly zero everywhere.
    let positions = sample_positions();
    for amp in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let params = WindFieldParams::new(1.0, amp, 2);
        let got = run(&ctx, params, &positions, 0.4);
        assert_batch(&got, params, &positions, 0.4);
        for (i, v) in got.iter().enumerate() {
            for (axis, &component) in v.iter().enumerate() {
                assert_close(
                    component,
                    0.0,
                    &format!("dead field position {i} axis {axis}"),
                );
            }
        }
    }
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let got = run(&ctx, WindFieldParams::new(1.0, 1.0, 0), &[], 0.0);
    assert!(got.is_empty(), "empty batch yields no velocities");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 130 positions span three 64-wide workgroups over a deterministic finite
    // sweep across several lattice cells and signs. The golden does not sanitize
    // positions, so every coordinate stays finite.
    let mut positions = Vec::new();
    for k in 0u32..130 {
        let f = k as f32;
        positions.push([(f * 0.37) - 24.0, (f * -0.21) + 7.0, (f * 0.13) - 3.5]);
    }
    let params = WindFieldParams::new(0.9, 1.7, 123);
    assert_batch(
        &run(&ctx, params, &positions, 1.25),
        params,
        &positions,
        1.25,
    );
}
