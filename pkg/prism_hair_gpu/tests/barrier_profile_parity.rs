//! Real-device parity for the per-sample soft-barrier profile twin:
//! [`GpuBarrierProfile`] must reproduce the `CPU` goldens
//! [`barrier_energy`](prism_render_architecture::hair::barrier_contact::barrier_energy)
//! and
//! [`barrier_force_magnitude`](prism_render_architecture::hair::barrier_contact::barrier_force_magnitude)
//! for a batch of contact distances under shared `BarrierParams`, emitting one
//! `(energy, force)` pair per distance in input order.
//!
//! # Parity criterion
//!
//! The barrier profile is a single closed-form evaluation (a handful of
//! add/sub/mul/divide) with no chained recurrence, so the only `CPU` vs `GPU`
//! divergence is legal fused-multiply-add contraction and correctly rounded
//! division. Each component (energy and force) is asserted to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3`. Every distance here is chosen so the
//! branch taken (free region `d >= dhat`, interior active window, or clamped at
//! `d_floor`) is far from its decision boundary, so a stray `fma` cannot flip
//! the branch and both sides evaluate the identical formula.
//!
//! The suite drives interior active-window samples, a below-floor sample (the
//! `d_floor` clamp), free-region samples beyond `dhat` (bit-exact zero on both
//! outputs), a second param set, an empty batch (handled with no dispatch), and
//! a 100-sample batch that crosses the 64-wide dispatch boundary. The free
//! region yields a bit-exact zero (asserted on raw `f32` bit patterns) and the
//! active window carries strictly positive energy/force, so neither a
//! zero-writing nor a constant-writing no-op kernel could pass.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature. All distances are
//! explicit decimal literals — never `f32::sin`/`cos` — so test data stays
//! deterministic without introducing transcendental divergence.
//!
//! Provenance: Prism's own rational soft-barrier (IPC-style, no Unreal Engine
//! source or derived code) plus a `wgpu` compute dispatch.

use prism_hair_gpu::barrier_profile::{reference_barrier_profile, GpuBarrierProfile};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::barrier_contact::BarrierParams;

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

/// True when `got` matches `want` within the fused-multiply-add tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// Already-sanitized barrier tuning: `dhat = 1e-2`, `d_floor = 1e-3` (strictly
/// below `dhat * 0.5`), `stiffness = 2`, so `BarrierParams::sanitized` is the
/// identity and the device sees exactly these values.
fn params_a() -> BarrierParams {
    BarrierParams {
        dhat: 1.0e-2,
        stiffness: 2.0,
        d_floor: 1.0e-3,
        friction_mu: 0.3,
    }
}

/// A second already-sanitized tuning with a wider window and softer stiffness.
fn params_b() -> BarrierParams {
    BarrierParams {
        dhat: 5.0e-2,
        stiffness: 0.5,
        d_floor: 5.0e-3,
        friction_mu: 0.1,
    }
}

/// Asserts every sample's `(energy, force)` pair matches the `CPU` goldens.
fn assert_batch_matches(gpu: &[(f32, f32)], params: BarrierParams, dists: &[f32]) {
    assert_eq!(gpu.len(), dists.len(), "one output pair per distance");
    for (i, &d) in dists.iter().enumerate() {
        let (we, wf) = reference_barrier_profile(d, params);
        let (ge, gf) = gpu[i];
        assert!(
            close(ge, we),
            "sample {i} (d = {d}) energy: gpu {ge} vs cpu {we}",
        );
        assert!(
            close(gf, wf),
            "sample {i} (d = {d}) force: gpu {gf} vs cpu {wf}",
        );
    }
}

#[test]
fn gpu_interior_window_matches_cpu() {
    let Some(ctx) = context_or_skip("gpu_interior_window_matches_cpu") else {
        return;
    };
    let twin = GpuBarrierProfile::new(&ctx);
    let params = params_a();
    // All strictly inside (d_floor = 1e-3, dhat = 1e-2), clear of both bounds.
    let dists = [0.002_f32, 0.004, 0.006, 0.008, 0.009];
    let gpu = twin.eval(&ctx, params, &dists);
    assert_batch_matches(&gpu, params, &dists);
    // The active window must carry strictly positive energy and force so a
    // zero-writing no-op kernel could not pass.
    assert!(
        gpu[0].0 > 0.0 && gpu[0].1 > 0.0,
        "interior sample must have positive energy/force, got {:?}",
        gpu[0],
    );
    // Energy is monotonically non-increasing across the window; the first
    // (closest) sample must dominate the last.
    assert!(
        gpu[0].0 > gpu[4].0,
        "energy must fall as distance grows: {} vs {}",
        gpu[0].0,
        gpu[4].0,
    );
}

#[test]
fn gpu_below_floor_clamps_like_cpu() {
    let Some(ctx) = context_or_skip("gpu_below_floor_clamps_like_cpu") else {
        return;
    };
    let twin = GpuBarrierProfile::new(&ctx);
    let params = params_a();
    // 5e-4 is clearly below the 1e-3 floor, so both sides clamp to d_floor.
    let dists = [0.0005_f32];
    let gpu = twin.eval(&ctx, params, &dists);
    assert_batch_matches(&gpu, params, &dists);
    // Clamping at the floor yields the same finite value as evaluating at the
    // floor itself.
    let (fe, ff) = reference_barrier_profile(0.001_f32, params);
    assert!(
        close(gpu[0].0, fe) && close(gpu[0].1, ff),
        "below-floor sample must clamp to the floor value {fe}/{ff}, got {:?}",
        gpu[0],
    );
}

#[test]
fn gpu_free_region_is_bit_exact_zero() {
    let Some(ctx) = context_or_skip("gpu_free_region_is_bit_exact_zero") else {
        return;
    };
    let twin = GpuBarrierProfile::new(&ctx);
    let params = params_a();
    // Both beyond dhat = 1e-2, well clear of the boundary.
    let dists = [0.011_f32, 0.05, 1.0];
    let gpu = twin.eval(&ctx, params, &dists);
    assert_batch_matches(&gpu, params, &dists);
    for (i, pair) in gpu.iter().enumerate() {
        assert_eq!(
            pair.0.to_bits(),
            0.0_f32.to_bits(),
            "free-region sample {i} energy must be bit-exact zero, got {}",
            pair.0,
        );
        assert_eq!(
            pair.1.to_bits(),
            0.0_f32.to_bits(),
            "free-region sample {i} force must be bit-exact zero, got {}",
            pair.1,
        );
    }
}

#[test]
fn gpu_second_param_set_matches_cpu() {
    let Some(ctx) = context_or_skip("gpu_second_param_set_matches_cpu") else {
        return;
    };
    let twin = GpuBarrierProfile::new(&ctx);
    let params = params_b();
    // Interior of (5e-3, 5e-2), plus a below-floor and a free-region sample.
    let dists = [0.001_f32, 0.01, 0.02, 0.03, 0.06];
    let gpu = twin.eval(&ctx, params, &dists);
    assert_batch_matches(&gpu, params, &dists);
}

#[test]
fn gpu_empty_batch_is_empty() {
    let Some(ctx) = context_or_skip("gpu_empty_batch_is_empty") else {
        return;
    };
    let twin = GpuBarrierProfile::new(&ctx);
    let gpu = twin.eval(&ctx, params_a(), &[]);
    assert!(gpu.is_empty(), "an empty batch must return no pairs");
}

#[test]
fn gpu_many_samples_cross_workgroup_boundary() {
    let Some(ctx) = context_or_skip("gpu_many_samples_cross_workgroup_boundary") else {
        return;
    };
    let twin = GpuBarrierProfile::new(&ctx);
    let params = params_a();
    // 100 distances spanning the active window cross the 64-wide dispatch
    // boundary, so threads in distinct workgroups must each recover their own
    // profile value. Step 9e-5 keeps every sample strictly inside (1e-3, 1e-2).
    let dists: Vec<f32> = (0..100).map(|i| 0.001_f32 + (i as f32) * 9.0e-5).collect();
    let gpu = twin.eval(&ctx, params, &dists);
    assert_batch_matches(&gpu, params, &dists);
}
