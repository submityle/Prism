//! Real-device parity for the isolated hair temporal reactive-mask twin:
//! [`GpuHairReactiveMask`] must reproduce the `CPU` golden
//! [`reference_reactivity`](prism_hair_gpu::reactive_mask::reference_reactivity)
//! (built on
//! [`pixel_reactivity`](prism_render_architecture::hair::reactive_mask::pixel_reactivity))
//! for a batch of per-pixel inputs, mapping each `(coverage, screen_velocity,
//! depth_delta)` triple to its `reactivity` in `[0, max_reactivity]`
//! independently. The suite drives the balanced default policy, a low-coverage
//! pixel pushing `reactivity` up, a fast pixel whose velocity ramp saturates, a
//! `depth_delta`-driven pixel, negative weights collapsing to zero, non-finite
//! weights / inputs sanitizing to the same bounded result, a tightened
//! `max_reactivity` cap, the empty no-op, and a large multi-workgroup batch that
//! crosses the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The reactivity is a weighted sum of ramps plus a reciprocal a `GPU` may fuse,
//! so every value is asserted within `abs_diff < 1e-4` or `rel_diff < 1e-3`.
//! Beyond matching the golden element by element, every value is asserted finite
//! and inside `[0, max_reactivity]` (the sanitized upper clamp). No `sin`/`cos`
//! appears anywhere; all inputs are explicit literals or integer-derived
//! fractions.
//!
//! Provenance: Prism's own deterministic temporal reactive-mask fold plus
//! `wgpu` compute dispatch; no third-party engine source or derived code.

use prism_hair_gpu::reactive_mask::{reference_reactivity, GpuHairReactiveMask};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::reactive_mask::{PixelReactiveInput, ReactiveParams};

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
        abs < 1e-4 || rel < 1e-3,
        "{what}: got {got}, want {want} (abs {abs}, rel {rel})"
    );
}

/// Asserts a whole batch of reactivity values matches the `CPU` golden element
/// by element, that every value is finite, and that each lands in
/// `[0, max_reactivity]` under the sanitized policy (the range invariant).
fn assert_batch(got: &[f32], params: ReactiveParams, inputs: &[PixelReactiveInput]) {
    assert_eq!(got.len(), inputs.len(), "one reactivity value per pixel");
    let max_reactivity = params.sanitized().max_reactivity;
    for (i, &input) in inputs.iter().enumerate() {
        let want = reference_reactivity(params, input);
        assert_close(got[i], want, &format!("element {i}"));
        assert!(
            got[i].is_finite() && got[i] >= 0.0 && got[i] <= max_reactivity,
            "element {i} must be finite and in [0, {max_reactivity}], got {}",
            got[i]
        );
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, params: ReactiveParams, inputs: &[PixelReactiveInput]) -> Vec<f32> {
    GpuHairReactiveMask::new(ctx).eval(ctx, params, inputs)
}

#[test]
fn default_params_balanced() {
    let Some(ctx) = context_or_skip("default_params_balanced") else {
        return;
    };
    // The balanced default policy over a spread of coverage / velocity / depth.
    let params = ReactiveParams::default();
    let inputs = [
        PixelReactiveInput::new(1.0, 0.0, 0.0),
        PixelReactiveInput::new(0.5, 2.0, 0.1),
        PixelReactiveInput::new(0.0, 10.0, 1.0),
    ];
    assert_batch(&run(&ctx, params, &inputs), params, &inputs);
}

#[test]
fn low_coverage_raises_reactivity() {
    let Some(ctx) = context_or_skip("low_coverage_raises_reactivity") else {
        return;
    };
    // With everything else held still, a fully covered pixel is less reactive
    // than a barely covered one (the 1 - coverage term dominates).
    let params = ReactiveParams::default();
    let covered = PixelReactiveInput::new(1.0, 0.0, 0.0);
    let bare = PixelReactiveInput::new(0.0, 0.0, 0.0);
    let got = run(&ctx, params, &[covered, bare]);
    assert_batch(&got, params, &[covered, bare]);
    assert!(
        got[1] > got[0],
        "low coverage should be more reactive, got {got:?}"
    );
}

#[test]
fn high_velocity_saturates() {
    let Some(ctx) = context_or_skip("high_velocity_saturates") else {
        return;
    };
    // The velocity ramp v/(v+k) saturates toward 1, so a very fast pixel is more
    // reactive than a slow one but both match the golden exactly.
    let params = ReactiveParams::default();
    let slow = PixelReactiveInput::new(0.5, 1.0, 0.0);
    let fast = PixelReactiveInput::new(0.5, 1000.0, 0.0);
    let got = run(&ctx, params, &[slow, fast]);
    assert_batch(&got, params, &[slow, fast]);
    assert!(
        got[1] > got[0],
        "higher velocity should be more reactive, got {got:?}"
    );
}

#[test]
fn depth_delta_contributes() {
    let Some(ctx) = context_or_skip("depth_delta_contributes") else {
        return;
    };
    // A large depth delta (disocclusion edge) raises reactivity through the
    // depth ramp; the magnitude is used, so a signed delta behaves like its abs.
    let params = ReactiveParams::default();
    let flat = PixelReactiveInput::new(0.5, 0.0, 0.0);
    let edge = PixelReactiveInput::new(0.5, 0.0, -5.0);
    let got = run(&ctx, params, &[flat, edge]);
    assert_batch(&got, params, &[flat, edge]);
    assert!(
        got[1] > got[0],
        "a depth edge should be more reactive, got {got:?}"
    );
}

#[test]
fn negative_weights_clamp_to_zero() {
    let Some(ctx) = context_or_skip("negative_weights_clamp_to_zero") else {
        return;
    };
    // Negative weights sanitize to zero, so an all-negative-weight policy yields
    // exactly zero reactivity regardless of input (matching the golden).
    let params = ReactiveParams::new(-1.0, -2.0, -3.0, 1.0);
    let inputs = [
        PixelReactiveInput::new(0.0, 100.0, 1.0),
        PixelReactiveInput::new(0.5, 2.0, 0.1),
    ];
    let got = run(&ctx, params, &inputs);
    assert_batch(&got, params, &inputs);
    for (i, &v) in got.iter().enumerate() {
        assert_close(v, 0.0, &format!("zeroed-weight element {i}"));
    }
}

#[test]
fn non_finite_inputs_sanitize() {
    let Some(ctx) = context_or_skip("non_finite_inputs_sanitize") else {
        return;
    };
    // Non-finite inputs sanitize like the golden: NaN coverage -> fully covered
    // (1), +/-inf velocity / depth -> 0 magnitude; a NaN weight -> 0 and a
    // non-finite max_reactivity -> 1. Everything stays bounded and finite.
    let params = ReactiveParams::new(0.3, f32::NAN, 0.25, f32::INFINITY);
    let inputs = [
        PixelReactiveInput::new(f32::NAN, 1.0, 0.1),
        PixelReactiveInput::new(0.5, f32::INFINITY, 0.1),
        PixelReactiveInput::new(0.5, 1.0, f32::NEG_INFINITY),
        PixelReactiveInput::new(f32::NEG_INFINITY, f32::NAN, f32::INFINITY),
    ];
    let got = run(&ctx, params, &inputs);
    assert_batch(&got, params, &inputs);
    assert!(
        got.iter().all(|v| v.is_finite()),
        "reactivity must stay finite for non-finite inputs, got {got:?}"
    );
}

#[test]
fn max_reactivity_caps() {
    let Some(ctx) = context_or_skip("max_reactivity_caps") else {
        return;
    };
    // A tightened max_reactivity of 0.5 clamps the weighted sum: a maximally
    // reactive pixel (zero coverage, huge velocity, huge depth) saturates at the
    // cap on both the CPU and the GPU.
    let params = ReactiveParams::new(0.5, 0.5, 0.5, 0.5);
    let inputs = [PixelReactiveInput::new(0.0, 1.0e6, 1.0e6)];
    let got = run(&ctx, params, &inputs);
    assert_batch(&got, params, &inputs);
    assert_close(got[0], 0.5, "capped reactivity");
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let got = run(&ctx, ReactiveParams::default(), &[]);
    assert!(got.is_empty(), "empty batch yields no reactivity values");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 130 pixels span three 64-wide workgroups over a deterministic sweep of
    // coverage / velocity / depth derived from integer slots (with every 13th
    // slot forced non-finite to exercise the sanitizer across the boundary).
    let params = ReactiveParams::default();
    let mut inputs = Vec::new();
    for k in 0u32..130 {
        if k % 13 == 0 {
            inputs.push(PixelReactiveInput::new(
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ));
        } else {
            let coverage = (k % 101) as f32 / 100.0;
            let velocity = (k % 17) as f32;
            let depth = (k % 5) as f32 / 10.0;
            inputs.push(PixelReactiveInput::new(coverage, velocity, depth));
        }
    }
    assert_batch(&run(&ctx, params, &inputs), params, &inputs);
}
