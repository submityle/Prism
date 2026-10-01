//! Real-device parity for the isolated stratified lobe-allocation twin:
//! [`GpuHairStratifiedAllocation`] must reproduce the `CPU` golden
//! [`reference_stratified_allocation`](prism_hair_gpu::stratified_allocation::reference_stratified_allocation)
//! (which forwards to
//! [`stratified_allocation`](prism_render_architecture::hair::scatter_lod::stratified_allocation))
//! for a batch of `(pdf, total)` tuples.
//!
//! # Parity criterion
//!
//! The allocation is pure integer arithmetic (`floor` of an exact `prob * total`
//! multiply plus largest-remainder leftover handling), so the per-lobe counts
//! are asserted with exact integer equality — no tolerance.
//!
//! The suite drives a normalised pdf at several budgets, a zero budget
//! (all-zero counts), a uniform pdf, a sanitised negative component that stays
//! normalised, the empty no-op batch, and a large multi-workgroup batch crossing
//! the 64-wide dispatch boundary. It also asserts the structural invariant that
//! for a normalised pdf the counts sum to the budget.
//!
//! All test pdfs are normalised (their three sanitised components sum to `1`).
//! That keeps the floored quotas at or below the budget so only the bounded
//! leftover top-up runs, which both the golden and the device twin resolve in at
//! most two picks for three lobes; a non-normalised pdf is intentionally out of
//! scope because its leftover can exceed the three-step device bound.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: largest-remainder (Hamilton) apportionment of a `Marschner`
//! three-lobe pdf plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use prism_hair_gpu::stratified_allocation::{
    reference_stratified_allocation, GpuHairStratifiedAllocation,
};
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
fn run(ctx: &GpuContext, tuples: &[(LobePdf, usize)]) -> Vec<[usize; LOBE_COUNT]> {
    GpuHairStratifiedAllocation::new(ctx).eval(ctx, tuples)
}

/// Asserts a whole batch matches the `CPU` golden exactly, lobe count by lobe
/// count.
fn assert_batch_exact(got: &[[usize; LOBE_COUNT]], tuples: &[(LobePdf, usize)]) {
    assert_eq!(
        got.len(),
        tuples.len(),
        "one allocation per tuple (got {}, want {})",
        got.len(),
        tuples.len()
    );
    for (i, (g, &(p, total))) in got.iter().zip(tuples.iter()).enumerate() {
        let want = reference_stratified_allocation(p, total);
        assert_eq!(
            *g, want,
            "tuple {i} (total {total}): device counts {g:?} must match golden {want:?}"
        );
    }
}

#[test]
fn normalised_pdf_apportions_several_budgets() {
    let Some(ctx) = context_or_skip("normalised_pdf_apportions_several_budgets") else {
        return;
    };
    // A normalised pdf (0.5, 0.3, 0.2): components sum to 1.
    let p = pdf(0.5, 0.3, 0.2);
    let tuples = [(p, 1usize), (p, 2), (p, 5), (p, 10), (p, 17), (p, 64)];
    let got = run(&ctx, &tuples);
    assert_batch_exact(&got, &tuples);
    // Counts must sum to the budget for a normalised pdf.
    for (g, &(_, total)) in got.iter().zip(tuples.iter()) {
        assert_eq!(
            g.iter().sum::<usize>(),
            total,
            "normalised allocation sums to the budget"
        );
    }
}

#[test]
fn zero_budget_allocates_nothing() {
    let Some(ctx) = context_or_skip("zero_budget_allocates_nothing") else {
        return;
    };
    let tuples = [
        (pdf(0.5, 0.3, 0.2), 0usize),
        (pdf(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 0),
    ];
    let got = run(&ctx, &tuples);
    assert_batch_exact(&got, &tuples);
    for g in &got {
        assert_eq!(*g, [0, 0, 0], "a zero budget allocates nothing");
    }
}

#[test]
fn uniform_pdf_splits_evenly_when_divisible() {
    let Some(ctx) = context_or_skip("uniform_pdf_splits_evenly_when_divisible") else {
        return;
    };
    let p = pdf(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    let tuples = [(p, 3usize), (p, 7), (p, 30), (p, 100)];
    let got = run(&ctx, &tuples);
    assert_batch_exact(&got, &tuples);
    // A budget divisible by three splits evenly.
    assert_eq!(got[0], [1, 1, 1], "3 samples split evenly across 3 lobes");
    for (g, &(_, total)) in got.iter().zip(tuples.iter()) {
        assert_eq!(
            g.iter().sum::<usize>(),
            total,
            "uniform split sums to budget"
        );
    }
}

#[test]
fn negative_component_sanitises_but_stays_normalised() {
    let Some(ctx) = context_or_skip("negative_component_sanitises_but_stays_normalised") else {
        return;
    };
    // A negative R sanitises to 0 while TT + TRT already sum to 1, so the pdf
    // stays normalised after sanitisation.
    let p = pdf(-1.0, 0.6, 0.4);
    let tuples = [(p, 1usize), (p, 5), (p, 10), (p, 50)];
    let got = run(&ctx, &tuples);
    assert_batch_exact(&got, &tuples);
    for (g, &(_, total)) in got.iter().zip(tuples.iter()) {
        assert_eq!(g[0], 0, "the sanitised dead R lobe gets no samples");
        assert_eq!(
            g.iter().sum::<usize>(),
            total,
            "sanitised-but-normalised allocation sums to the budget"
        );
    }
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_batch_is_a_no_op") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields no allocations");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 tuples > 3 full 64-wide workgroups: every tuple index must map to its
    // own allocation independent of the dispatch tiling.
    let p = pdf(0.2, 0.5, 0.3);
    let mut tuples = Vec::with_capacity(200);
    for i in 0..200u32 {
        tuples.push((p, (i as usize) % 37 + 1));
    }
    let got = run(&ctx, &tuples);
    assert_batch_exact(&got, &tuples);
    for (g, &(_, total)) in got.iter().zip(tuples.iter()) {
        assert_eq!(
            g.iter().sum::<usize>(),
            total,
            "every tuple sums to its budget"
        );
    }
}
