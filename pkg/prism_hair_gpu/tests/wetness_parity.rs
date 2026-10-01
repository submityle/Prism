//! Real-device parity for the isolated wet-hair coupling twin:
//! [`GpuHairWetness`] must reproduce the `CPU` golden
//! [`reference_response`](prism_hair_gpu::wetness::reference_response)
//! (built on
//! [`wet_hair_response`](prism_render_architecture::hair::wetness::wet_hair_response))
//! for a batch of saturation fractions, mapping each to its five parameter
//! modifiers independently. The suite drives the dry endpoint, full saturation,
//! partial saturations, clamping above one, negative and non-finite inputs
//! collapsing to dry, the empty no-op, a mixed batch, and a large multi-workgroup
//! batch that crosses the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each modifier is one clamped multiply-add the scalar reference may leave
//! separate while a `GPU` fuses it, so every value is asserted within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3`. No `sin`/`cos` appears anywhere; all
//! inputs are explicit literals or integer-derived fractions.
//!
//! Provenance: Prism's own deterministic wet-hair coupling plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use prism_hair_gpu::wetness::{reference_response, GpuHairWetness, MODIFIERS_PER_ELEMENT};
use prism_hair_gpu::GpuContext;

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

/// Asserts a whole batch of flattened responses matches the `CPU` golden element
/// by element, field by field, and that every modifier stays finite.
fn assert_batch(got: &[f32], wetness: &[f32]) {
    assert_eq!(
        got.len(),
        wetness.len() * MODIFIERS_PER_ELEMENT,
        "five modifiers per element"
    );
    for (i, &w) in wetness.iter().enumerate() {
        let want = reference_response(w);
        let base = i * MODIFIERS_PER_ELEMENT;
        let fields = [
            (got[base], want.clump_scale, "clump_scale"),
            (got[base + 1], want.mass_mul, "mass_mul"),
            (got[base + 2], want.damping_mul, "damping_mul"),
            (got[base + 3], want.sigma_a_mul, "sigma_a_mul"),
            (got[base + 4], want.roughness_delta, "roughness_delta"),
        ];
        for (out, reference, name) in fields {
            assert_close(out, reference, &format!("element {i} {name}"));
            assert!(
                out.is_finite(),
                "element {i} {name} must stay finite, got {out}"
            );
        }
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, wetness: &[f32]) -> Vec<f32> {
    GpuHairWetness::new(ctx).eval(ctx, wetness)
}

#[test]
fn dry_input_is_identity() {
    let Some(ctx) = context_or_skip("dry_input_is_identity") else {
        return;
    };
    // w = 0 must leave every multiplier at 1.0 and the roughness delta at 0.0.
    let wetness = [0.0];
    let got = run(&ctx, &wetness);
    assert_batch(&got, &wetness);
    assert_close(got[0], 1.0, "dry clump_scale");
    assert_close(got[1], 1.0, "dry mass_mul");
    assert_close(got[2], 1.0, "dry damping_mul");
    assert_close(got[3], 1.0, "dry sigma_a_mul");
    assert_close(got[4], 0.0, "dry roughness_delta");
}

#[test]
fn full_saturation_matches_golden() {
    let Some(ctx) = context_or_skip("full_saturation_matches_golden") else {
        return;
    };
    let wetness = [1.0];
    assert_batch(&run(&ctx, &wetness), &wetness);
}

#[test]
fn partial_saturation_matches_golden() {
    let Some(ctx) = context_or_skip("partial_saturation_matches_golden") else {
        return;
    };
    // Several interior fractions exercise the full interpolation ramp.
    let wetness = [0.1, 0.25, 0.5, 0.75, 0.9];
    assert_batch(&run(&ctx, &wetness), &wetness);
}

#[test]
fn above_one_clamps_to_full_saturation() {
    let Some(ctx) = context_or_skip("above_one_clamps_to_full_saturation") else {
        return;
    };
    // Inputs greater than one are unphysical and must clamp to the w = 1 result.
    let wetness = [1.5, 2.0, 1000.0];
    assert_batch(&run(&ctx, &wetness), &wetness);
}

#[test]
fn negative_input_clamps_to_dry() {
    let Some(ctx) = context_or_skip("negative_input_clamps_to_dry") else {
        return;
    };
    // Finite negative inputs clamp up to the fully dry (w = 0) response.
    let wetness = [-0.5, -5.0];
    assert_batch(&run(&ctx, &wetness), &wetness);
}

#[test]
fn non_finite_input_clamps_to_dry() {
    let Some(ctx) = context_or_skip("non_finite_input_clamps_to_dry") else {
        return;
    };
    // NaN and +/-inf must each collapse to the fully dry response like the golden.
    let wetness = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let got = run(&ctx, &wetness);
    assert_batch(&got, &wetness);
    assert!(
        got.iter().all(|v| v.is_finite()),
        "modifiers must stay finite for non-finite inputs, got {got:?}"
    );
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = context_or_skip("mixed_batch_matches_golden") else {
        return;
    };
    // A single batch mixing dry, partial, full, over-range, negative and
    // non-finite inputs preserves order and matches the golden element by element.
    let wetness = [0.0, 0.33, 1.0, 2.5, -1.0, f32::NAN, 0.6];
    assert_batch(&run(&ctx, &wetness), &wetness);
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty(), "empty batch yields no modifiers");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 130 saturations span three 64-wide workgroups over a deterministic sweep
    // from 0 to 1 (with every 11th slot forced non-finite to exercise the guard
    // across the dispatch boundary).
    let mut wetness = Vec::new();
    for k in 0u32..130 {
        if k % 11 == 0 {
            wetness.push(f32::NAN);
        } else {
            wetness.push((k % 101) as f32 / 100.0);
        }
    }
    assert_batch(&run(&ctx, &wetness), &wetness);
}
