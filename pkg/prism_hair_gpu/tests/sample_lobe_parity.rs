//! Real-device parity for the isolated Marschner lobe importance-sampling twin:
//! [`GpuHairSampleLobe`] must reproduce the `CPU` golden
//! [`reference_sample_lobe`](prism_hair_gpu::sample_lobe::reference_sample_lobe)
//! (which forwards to
//! [`sample_lobe`](prism_render_architecture::hair::scatter_lod::sample_lobe))
//! for a batch of `(pdf, u)` tuples.
//!
//! # Parity criterion
//!
//! The chosen lobe index and its probability are exact (a cdf walk and a pass of
//! a sanitised pdf component), so the index is asserted exactly; only the
//! `(u - lo) / p` remap divides, which the `GPU` may round a few `ULP`
//! differently, so `remapped_u` and the returned probability are compared
//! against a tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`).
//!
//! The suite drives samples that land in each of the three cdf slices, the exact
//! slice boundaries, `u == 1` (absorbed by `TRT`), a degenerate all-zero pdf
//! (uniform fallback is handled by the pdf stage, so here it samples `TRT` with
//! probability `0` and `remapped_u = 0`), sanitised non-finite / out-of-range
//! `u` and pdf components, the empty no-op batch, and a large multi-workgroup
//! batch crossing the 64-wide dispatch boundary. It also asserts the structural
//! invariants: the lobe index is always in `0..3` and `remapped_u` is always in
//! `[0, 1]`.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: `Marschner` 2003 three-lobe hair BSDF importance sampling plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::sample_lobe::{reference_sample_lobe, GpuHairSampleLobe};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::scatter_lod::{LobePdf, LobeSample};

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

/// Builds a `LobePdf` from its three components.
fn pdf(r: f32, tt: f32, trt: f32) -> LobePdf {
    LobePdf { r, tt, trt }
}

/// Dispatches one batch of `(pdf, u)` tuples through the device twin.
fn run(ctx: &GpuContext, tuples: &[(LobePdf, f32)]) -> Vec<LobeSample> {
    GpuHairSampleLobe::new(ctx).eval(ctx, tuples)
}

/// True when `got` matches `want` within the fma tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// Asserts a whole batch matches the `CPU` golden: exact lobe index, tolerant
/// floats.
fn assert_batch_close(got: &[LobeSample], tuples: &[(LobePdf, f32)]) {
    assert_eq!(
        got.len(),
        tuples.len(),
        "one sample per tuple (got {}, want {})",
        got.len(),
        tuples.len()
    );
    for (i, (g, &(p, u))) in got.iter().zip(tuples.iter()).enumerate() {
        let want = reference_sample_lobe(p, u);
        assert_eq!(
            g.lobe, want.lobe,
            "tuple {i} (u {u}): device lobe {} must match golden {}",
            g.lobe, want.lobe
        );
        assert!(
            close(g.remapped_u, want.remapped_u),
            "tuple {i} (u {u}): device remapped_u {} must match golden {}",
            g.remapped_u,
            want.remapped_u
        );
        assert!(
            close(g.pdf, want.pdf),
            "tuple {i} (u {u}): device pdf {} must match golden {}",
            g.pdf,
            want.pdf
        );
    }
}

#[test]
fn samples_land_in_each_cdf_slice() {
    let Some(ctx) = context_or_skip("samples_land_in_each_cdf_slice") else {
        return;
    };
    // pdf (0.5, 0.3, 0.2): cdf boundaries at 0.5 and 0.8.
    let p = pdf(0.5, 0.3, 0.2);
    let tuples = [
        (p, 0.0),  // R
        (p, 0.25), // R
        (p, 0.5),  // TT (boundary absorbed upward)
        (p, 0.65), // TT
        (p, 0.8),  // TRT (boundary absorbed upward)
        (p, 0.95), // TRT
        (p, 1.0),  // TRT (upper end)
    ];
    let got = run(&ctx, &tuples);
    assert_batch_close(&got, &tuples);
    assert_eq!(got[0].lobe, 0, "u=0 samples R");
    assert_eq!(got[2].lobe, 1, "u=0.5 samples TT");
    assert_eq!(got[4].lobe, 2, "u=0.8 samples TRT");
    assert_eq!(got[6].lobe, 2, "u=1 is absorbed by TRT");
    // Midpoint of the R slice remaps to its slice midpoint 0.5.
    assert!(
        close(got[1].remapped_u, 0.5),
        "u=0.25 of a [0,0.5) slice -> 0.5"
    );
}

#[test]
fn invariants_hold_across_a_sweep() {
    let Some(ctx) = context_or_skip("invariants_hold_across_a_sweep") else {
        return;
    };
    let p = pdf(0.2, 0.5, 0.3);
    let mut tuples = Vec::with_capacity(50);
    for i in 0..50u32 {
        tuples.push((p, i as f32 / 49.0));
    }
    let got = run(&ctx, &tuples);
    assert_batch_close(&got, &tuples);
    for s in &got {
        assert!(s.lobe < 3, "lobe index in 0..3: {}", s.lobe);
        assert!(
            (0.0..=1.0).contains(&s.remapped_u),
            "remapped_u in [0,1]: {}",
            s.remapped_u
        );
    }
}

#[test]
fn degenerate_all_zero_pdf_absorbs_into_trt() {
    let Some(ctx) = context_or_skip("degenerate_all_zero_pdf_absorbs_into_trt") else {
        return;
    };
    // An all-zero pdf: every cdf slice is empty, so TRT absorbs any u and its
    // probability is 0, forcing remapped_u = 0 (the <= EPS guard).
    let p = pdf(0.0, 0.0, 0.0);
    let tuples = [(p, 0.0), (p, 0.4), (p, 1.0)];
    let got = run(&ctx, &tuples);
    assert_batch_close(&got, &tuples);
    for s in &got {
        assert_eq!(s.lobe, 2, "empty cdf -> TRT absorbs");
        assert!(close(s.pdf, 0.0), "dead lobe probability is 0");
        assert!(close(s.remapped_u, 0.0), "dead lobe remap collapses to 0");
    }
}

#[test]
fn nonfinite_and_out_of_range_inputs_sanitise() {
    let Some(ctx) = context_or_skip("nonfinite_and_out_of_range_inputs_sanitise") else {
        return;
    };
    let p = pdf(0.5, 0.3, 0.2);
    let bad = pdf(-1.0, f32::NAN, f32::INFINITY); // sanitises to (0, 0, 0)
    let tuples = [
        (p, f32::NAN),          // u -> 0 -> R
        (p, f32::INFINITY),     // u -> 0 -> R
        (p, f32::NEG_INFINITY), // u -> 0 -> R
        (p, -0.5),              // u clamps to 0 -> R
        (p, 2.0),               // u clamps to 1 -> TRT
        (bad, 0.3),             // dead pdf -> TRT, prob 0
    ];
    let got = run(&ctx, &tuples);
    assert_batch_close(&got, &tuples);
    assert_eq!(got[0].lobe, 0, "NaN u -> 0 -> R");
    assert_eq!(got[4].lobe, 2, "u>1 clamps to 1 -> TRT");
    assert_eq!(got[5].lobe, 2, "sanitised dead pdf -> TRT");
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_batch_is_a_no_op") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields no samples");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 tuples > 3 full 64-wide workgroups: every tuple index must map to its
    // own sample independent of the dispatch tiling.
    let p = pdf(0.4, 0.35, 0.25);
    let mut tuples = Vec::with_capacity(200);
    for i in 0..200u32 {
        tuples.push((p, i as f32 / 199.0));
    }
    let got = run(&ctx, &tuples);
    assert_batch_close(&got, &tuples);
}
