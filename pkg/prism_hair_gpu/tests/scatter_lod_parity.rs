//! Real-device parity for the isolated hair BSDF lobe-importance pdf twin:
//! [`GpuHairScatterLod`] must reproduce the `CPU` golden
//! [`reference_pdf`](prism_hair_gpu::scatter_lod::reference_pdf)
//! (built on
//! [`lobe_pdf`](prism_render_architecture::hair::scatter_lod::lobe_pdf)
//! composed with
//! [`lobe_weights`](prism_render_architecture::hair::scatter_lod::lobe_weights))
//! for a batch of per-fibre optical inputs, mapping each `(fresnel, absorption)`
//! pair to its normalized `R`/`TT`/`TRT` sampling pdf independently. The suite
//! drives a reflection-dominant fibre, absorption suppressing the transmission
//! lobes, a degenerate dead stack collapsing to the uniform `1/3` fallback,
//! negative and non-finite inputs sanitizing to the same bounded pdf, a fresnel
//! above one clamping, a mixed batch, the empty no-op, and a large
//! multi-workgroup batch that crosses the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each probability is a chain of products and a normalising reciprocal a `GPU`
//! may fuse, so every value is asserted within `abs_diff < 1e-4` or
//! `rel_diff < 1e-3`. Beyond matching the golden field by field, every element's
//! three probabilities are asserted non-negative and summing to one (the pdf
//! invariant). No `sin`/`cos` appears anywhere; all inputs are explicit literals
//! or integer-derived fractions.
//!
//! Provenance: Prism's own deterministic Marschner lobe-importance fold plus
//! `wgpu` compute dispatch; no third-party engine source or derived code.

use prism_hair_gpu::scatter_lod::{reference_pdf, GpuHairScatterLod, PDF_PER_ELEMENT};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::scatter_lod::LobeOpticalInput;

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

/// Asserts a whole batch of flattened pdfs matches the `CPU` golden element by
/// element, lobe by lobe, that every probability is non-negative, and that each
/// element's three probabilities sum to one (the pdf invariant).
fn assert_batch(got: &[f32], inputs: &[LobeOpticalInput]) {
    assert_eq!(
        got.len(),
        inputs.len() * PDF_PER_ELEMENT,
        "three lobe probabilities per element"
    );
    for (i, &input) in inputs.iter().enumerate() {
        let want = reference_pdf(input);
        let base = i * PDF_PER_ELEMENT;
        let fields = [
            (got[base], want.r, "r"),
            (got[base + 1], want.tt, "tt"),
            (got[base + 2], want.trt, "trt"),
        ];
        for (out, reference, name) in fields {
            assert_close(out, reference, &format!("element {i} lobe {name}"));
            assert!(
                out.is_finite() && out >= 0.0,
                "element {i} lobe {name} must be finite and non-negative, got {out}"
            );
        }
        let sum = got[base] + got[base + 1] + got[base + 2];
        assert_close(sum, 1.0, &format!("element {i} pdf sum"));
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, inputs: &[LobeOpticalInput]) -> Vec<f32> {
    GpuHairScatterLod::new(ctx).eval(ctx, inputs)
}

#[test]
fn reflection_dominant_fibre() {
    let Some(ctx) = context_or_skip("reflection_dominant_fibre") else {
        return;
    };
    // High fresnel with zero absorption (T = 1): the R lobe should dominate while
    // the two transmission lobes stay tiny but present.
    let inputs = [LobeOpticalInput::new(0.9, 0.0)];
    let got = run(&ctx, &inputs);
    assert_batch(&got, &inputs);
    // R = 0.9, TT = 0.01, TRT = 0.009 pre-normalisation -> R clearly largest.
    assert!(
        got[0] > got[1] && got[0] > got[2],
        "R should dominate, got {got:?}"
    );
}

#[test]
fn transmission_dominant_fibre() {
    let Some(ctx) = context_or_skip("transmission_dominant_fibre") else {
        return;
    };
    // Low fresnel with low absorption keeps T near one, so the single-transmission
    // TT lobe carries most of the energy (R = 0.05, TT = 0.9025, TRT = 0.045).
    let inputs = [LobeOpticalInput::new(0.05, 0.0)];
    let got = run(&ctx, &inputs);
    assert_batch(&got, &inputs);
    assert!(
        got[1] > got[0] && got[1] > got[2],
        "TT should dominate, got {got:?}"
    );
}

#[test]
fn high_absorption_suppresses_transmission() {
    let Some(ctx) = context_or_skip("high_absorption_suppresses_transmission") else {
        return;
    };
    // Large absorption drives T = 1/(1+a) toward zero, collapsing TT and TRT and
    // pushing almost all of the pdf mass onto R.
    let inputs = [LobeOpticalInput::new(0.5, 50.0)];
    let got = run(&ctx, &inputs);
    assert_batch(&got, &inputs);
    assert!(
        got[0] > got[1] && got[0] > got[2],
        "absorption should concentrate mass on R, got {got:?}"
    );
}

#[test]
fn dead_stack_uniform_fallback() {
    let Some(ctx) = context_or_skip("dead_stack_uniform_fallback") else {
        return;
    };
    // Zero fresnel with an enormous absorption drives every energy below the EPS
    // dead-stack threshold (R = 0, TT = T ~ 1e-30, TRT = 0), so the pdf falls back
    // to the uniform 1/3 per lobe on both the CPU and the GPU.
    let inputs = [LobeOpticalInput::new(0.0, 1.0e30)];
    let got = run(&ctx, &inputs);
    assert_batch(&got, &inputs);
    let third = 1.0 / 3.0;
    assert_close(got[0], third, "fallback R");
    assert_close(got[1], third, "fallback TT");
    assert_close(got[2], third, "fallback TRT");
}

#[test]
fn negative_and_non_finite_clamp() {
    let Some(ctx) = context_or_skip("negative_and_non_finite_clamp") else {
        return;
    };
    // Negative fresnel/absorption and NaN/+/-inf must each sanitize exactly like
    // the golden (fresnel -> clamp01, absorption -> sanitize_nonneg) and still
    // yield a valid, non-negative pdf summing to one.
    let inputs = [
        LobeOpticalInput::new(-1.0, -5.0),
        LobeOpticalInput::new(f32::NAN, 0.3),
        LobeOpticalInput::new(0.4, f32::NAN),
        LobeOpticalInput::new(f32::INFINITY, f32::NEG_INFINITY),
    ];
    let got = run(&ctx, &inputs);
    assert_batch(&got, &inputs);
    assert!(
        got.iter().all(|v| v.is_finite()),
        "probabilities must stay finite for non-finite inputs, got {got:?}"
    );
}

#[test]
fn fresnel_above_one_clamps() {
    let Some(ctx) = context_or_skip("fresnel_above_one_clamps") else {
        return;
    };
    // Fresnel above one is unphysical and clamps to 1: R = 1 drives both
    // transmission lobes to zero (TT = TRT = 0) -> pdf (1, 0, 0).
    let inputs = [LobeOpticalInput::new(5.0, 0.0)];
    let got = run(&ctx, &inputs);
    assert_batch(&got, &inputs);
    assert_close(got[0], 1.0, "clamped R");
    assert_close(got[1], 0.0, "clamped TT");
    assert_close(got[2], 0.0, "clamped TRT");
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = context_or_skip("mixed_batch_matches_golden") else {
        return;
    };
    // A single batch mixing reflection-dominant, balanced, transmission-heavy,
    // clamped, negative and non-finite inputs preserves order and matches the
    // golden element by element.
    let inputs = [
        LobeOpticalInput::new(0.9, 0.0),
        LobeOpticalInput::new(0.3, 0.5),
        LobeOpticalInput::new(0.05, 0.1),
        LobeOpticalInput::new(5.0, 0.0),
        LobeOpticalInput::new(-1.0, -2.0),
        LobeOpticalInput::new(f32::NAN, f32::INFINITY),
        LobeOpticalInput::new(0.6, 3.0),
    ];
    assert_batch(&run(&ctx, &inputs), &inputs);
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty(), "empty batch yields no probabilities");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 130 fibres span three 64-wide workgroups over a deterministic sweep of
    // fresnel and absorption derived from integer slots (with every 13th slot
    // forced non-finite to exercise the sanitizer across the dispatch boundary).
    let mut inputs = Vec::new();
    for k in 0u32..130 {
        if k % 13 == 0 {
            inputs.push(LobeOpticalInput::new(f32::NAN, f32::INFINITY));
        } else {
            let fresnel = (k % 101) as f32 / 100.0;
            let absorption = (k % 7) as f32 / 2.0;
            inputs.push(LobeOpticalInput::new(fresnel, absorption));
        }
    }
    assert_batch(&run(&ctx, &inputs), &inputs);
}
