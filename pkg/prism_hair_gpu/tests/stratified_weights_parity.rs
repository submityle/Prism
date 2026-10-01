//! Real-device parity for the isolated stratified per-lobe weight twin:
//! [`GpuHairStratifiedWeights`] must reproduce the `CPU` golden
//! [`reference_stratified_weights`](prism_hair_gpu::stratified_weights::reference_stratified_weights)
//! (which forwards to
//! [`stratified_weights`](prism_render_architecture::hair::scatter_lod::stratified_weights))
//! for a batch of `(pdf, total)` tuples.
//!
//! # Parity criterion
//!
//! The allocation underneath is integer-exact, so the only inexact step is the
//! final `pdf_i * total / count_i` divide, which the `GPU` may round a few `ULP`
//! differently; weights are therefore compared against a tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`). A dead lobe (`count_i == 0`) or a
//! zero budget yields weight exactly `0`.
//!
//! The suite drives a normalised pdf at several budgets, a zero budget
//! (all-zero weights), a uniform pdf, a sanitised negative component that stays
//! normalised, the empty no-op batch, and a large multi-workgroup batch crossing
//! the 64-wide dispatch boundary. All test pdfs are normalised so the
//! three-step leftover bound stays exact (see the allocation twin).
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: largest-remainder (Hamilton) apportionment of a `Marschner`
//! three-lobe pdf and its unbiased stratified weights plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::stratified_weights::{reference_stratified_weights, GpuHairStratifiedWeights};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::scatter_lod::{LobePdf, LOBE_COUNT};

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

/// Dispatches one batch of `(pdf, total)` tuples through the device twin.
fn run(ctx: &GpuContext, tuples: &[(LobePdf, usize)]) -> Vec<[f32; LOBE_COUNT]> {
    GpuHairStratifiedWeights::new(ctx).eval(ctx, tuples)
}

/// True when `got` matches `want` within the fma tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// Asserts a whole batch matches the `CPU` golden within tolerance.
fn assert_batch_close(got: &[[f32; LOBE_COUNT]], tuples: &[(LobePdf, usize)]) {
    assert_eq!(
        got.len(),
        tuples.len(),
        "one weight vector per tuple (got {}, want {})",
        got.len(),
        tuples.len()
    );
    for (i, (g, &(p, total))) in got.iter().zip(tuples.iter()).enumerate() {
        let want = reference_stratified_weights(p, total);
        for (k, (&gw, &ww)) in g.iter().zip(want.iter()).enumerate() {
            assert!(
                close(gw, ww),
                "tuple {i} lobe {k} (total {total}): device weight {gw} must match golden {ww}"
            );
        }
    }
}

#[test]
fn normalised_pdf_weights_several_budgets() {
    let Some(ctx) = context_or_skip("normalised_pdf_weights_several_budgets") else {
        return;
    };
    let p = pdf(0.5, 0.3, 0.2);
    let tuples = [(p, 1usize), (p, 2), (p, 5), (p, 10), (p, 17), (p, 64)];
    let got = run(&ctx, &tuples);
    assert_batch_close(&got, &tuples);
}

#[test]
fn zero_budget_weights_are_zero() {
    let Some(ctx) = context_or_skip("zero_budget_weights_are_zero") else {
        return;
    };
    let tuples = [
        (pdf(0.5, 0.3, 0.2), 0usize),
        (pdf(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 0),
    ];
    let got = run(&ctx, &tuples);
    assert_batch_close(&got, &tuples);
    for g in &got {
        assert_eq!(*g, [0.0, 0.0, 0.0], "a zero budget yields zero weights");
    }
}

#[test]
fn uniform_pdf_weights() {
    let Some(ctx) = context_or_skip("uniform_pdf_weights") else {
        return;
    };
    let p = pdf(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    let tuples = [(p, 3usize), (p, 7), (p, 30), (p, 100)];
    let got = run(&ctx, &tuples);
    assert_batch_close(&got, &tuples);
    // An even split of 3 across 3 lobes gives each weight pdf*total/count =
    // (1/3)*3/1 = 1.
    for &w in &got[0] {
        assert!(close(w, 1.0), "uniform 3-sample weight is 1");
    }
}

#[test]
fn negative_component_sanitises_but_stays_normalised() {
    let Some(ctx) = context_or_skip("negative_component_sanitises_but_stays_normalised") else {
        return;
    };
    // A negative R sanitises to 0 while TT + TRT already sum to 1.
    let p = pdf(-1.0, 0.6, 0.4);
    let tuples = [(p, 1usize), (p, 5), (p, 10), (p, 50)];
    let got = run(&ctx, &tuples);
    assert_batch_close(&got, &tuples);
    for g in &got {
        assert!(
            close(g[0], 0.0),
            "the sanitised dead R lobe has zero weight"
        );
    }
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_batch_is_a_no_op") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields no weight vectors");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 tuples > 3 full 64-wide workgroups: every tuple index must map to its
    // own weight vector independent of the dispatch tiling.
    let p = pdf(0.2, 0.5, 0.3);
    let mut tuples = Vec::with_capacity(200);
    for i in 0..200u32 {
        tuples.push((p, (i as usize) % 37 + 1));
    }
    let got = run(&ctx, &tuples);
    assert_batch_close(&got, &tuples);
}
