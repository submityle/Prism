//! Real-device parity for the multigrid coarsest-level predicate twin:
//! [`GpuMgIsCoarsest`](prism_volumetric_gpu::mg_is_coarsest::GpuMgIsCoarsest)
//! must reproduce the `CPU` golden
//! [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
//! `is_coarsest` rule exactly — the `0`/`1` stopping verdict for a
//! `(resolution, coarsest_axis)` pair.
//!
//! The three golden rules (`coarse_axis`, `coarsen`, `is_coarsest`) are private
//! to the golden module, so this test mirrors them verbatim (see the
//! `MIRROR of ...` helpers) and compares the `GPU` batch against that mirror.
//!
//! The fixtures cover the shapes the hierarchy builder exercises: a `1x1x1`
//! grid (within budget and stuck), grids already at or below the budget on every
//! axis, grids that still exceed the budget, odd and even axes (the round-up
//! halving), a single coarsenable axis still above budget (not within, not
//! stuck), mixed single-cell and larger extents, large grids kept well away from
//! `u32` overflow, plus a deterministic integer sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every input and intermediate is an unsigned integer, so the `CPU` and `GPU`
//! verdicts are bit-identical and the comparison is exact equality (`==`), not a
//! tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::multigrid_pressure`
//! 的最粗层停止判定整数规则；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::fluid::GridResolution;
use prism_volumetric_gpu::mg_is_coarsest::{
    GpuMgIsCoarsest, GpuMgIsCoarsestQuery, GpuMgIsCoarsestResult,
};
use prism_volumetric_gpu::GpuContext;

// MIRROR of multigrid_pressure::coarse_axis (private, exact transcription).
fn golden_coarse_axis(n: u32) -> u32 {
    if n <= 1 {
        n
    } else {
        // Round up so an odd axis still produces a strictly smaller grid.
        n.div_ceil(2)
    }
}

// MIRROR of multigrid_pressure::coarsen (private, exact transcription).
fn golden_coarsen(res: GridResolution) -> GridResolution {
    GridResolution::new(
        golden_coarse_axis(res.nx),
        golden_coarse_axis(res.ny),
        golden_coarse_axis(res.nz),
    )
}

// MIRROR of multigrid_pressure::is_coarsest (private, exact transcription).
fn golden_is_coarsest(res: GridResolution, coarsest_axis: u32) -> bool {
    let within = res.nx <= coarsest_axis && res.ny <= coarsest_axis && res.nz <= coarsest_axis;
    let stuck = golden_coarsen(res) == res;
    within || stuck
}

/// The golden verdict folded to the `0`/`1` flag the kernel emits.
fn golden_flag(res: GridResolution, coarsest_axis: u32) -> u32 {
    u32::from(golden_is_coarsest(res, coarsest_axis))
}

/// A small integer `LCG` used to synthesize deterministic queries without any
/// external math library or transcendental call.
struct Lcg {
    /// The current `64`-bit state.
    state: u64,
}

impl Lcg {
    /// Seeds the generator, forcing an odd state so the stream never degenerates.
    fn new(seed: u64) -> Self {
        Lcg { state: seed | 1 }
    }

    /// Advances the state and returns the high `32` bits.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// A deterministic axis extent in `1..=max`, kept well away from `u32`
    /// overflow, pure integer math with no transcendental call.
    fn axis(&mut self, max: u32) -> u32 {
        (self.next_u32() % max) + 1
    }
}

/// One fixture entry: a grid plus the `coarsest_axis` budget it is tested at.
#[derive(Clone, Copy)]
struct Case {
    /// The grid resolution under test.
    res: GridResolution,
    /// The coarsest-axis budget the predicate is evaluated against.
    coarsest_axis: u32,
}

impl Case {
    /// Builds a case from explicit extents and a budget.
    fn new(nx: u32, ny: u32, nz: u32, coarsest_axis: u32) -> Self {
        Case {
            res: GridResolution::new(nx, ny, nz),
            coarsest_axis,
        }
    }
}

/// Maps fixture cases to kernel queries in the same order.
fn to_queries(cases: &[Case]) -> Vec<GpuMgIsCoarsestQuery> {
    cases
        .iter()
        .map(|c| GpuMgIsCoarsestQuery {
            nx: c.res.nx,
            ny: c.res.ny,
            nz: c.res.nz,
            coarsest_axis: c.coarsest_axis,
        })
        .collect()
}

/// Runs the whole batch through the kernel and asserts exact parity, case by
/// case, against the mirrored golden rule.
fn assert_parity(gpu: &[GpuMgIsCoarsestResult], cases: &[Case], label: &str) {
    assert_eq!(gpu.len(), cases.len(), "{label}: result count");
    for (i, g) in gpu.iter().enumerate() {
        let c = cases[i];
        let expected = golden_flag(c.res, c.coarsest_axis);
        assert_eq!(
            g.flag, expected,
            "{label}: case[{i}] res=({}, {}, {}) coarsest_axis={}",
            c.res.nx, c.res.ny, c.res.nz, c.coarsest_axis
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mg_is_coarsest parity: no wgpu adapter on this host");
        return;
    };
    let predicate = GpuMgIsCoarsest::new(&ctx);
    let out = predicate.is_coarsest(&ctx, &[]);
    assert!(
        out.is_empty(),
        "empty query batch must return an empty vector"
    );
}

#[test]
fn explicit_shape_parity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let predicate = GpuMgIsCoarsest::new(&ctx);
    let cases = [
        // 1x1x1: within any non-zero budget and stuck -> flag 1.
        Case::new(1, 1, 1, 2),
        // 1x1x1 against a zero budget: not within, but stuck -> flag 1.
        Case::new(1, 1, 1, 0),
        // Every axis exactly at the budget: within -> flag 1.
        Case::new(2, 2, 2, 2),
        // Every axis below the budget: within -> flag 1.
        Case::new(2, 3, 4, 8),
        // One axis just above the budget: not within; coarsens -> not stuck -> 0.
        Case::new(2, 2, 3, 2),
        // All even and above budget: coarsens, not within -> flag 0.
        Case::new(8, 16, 32, 2),
        // All odd (> 1) and above budget: round-up halving, not within -> 0.
        Case::new(7, 9, 15, 2),
        // Single coarsenable axis still above budget: not within, not stuck -> 0.
        Case::new(1, 1, 4, 2),
        // Single coarsenable axis within budget: within -> flag 1.
        Case::new(1, 1, 2, 2),
        // Two-cell axis against a one-cell budget: not within; halves to one so
        // not stuck -> flag 0.
        Case::new(2, 1, 1, 1),
        // Mixed single-cell and even/odd, all within a generous budget -> 1.
        Case::new(1, 10, 5, 16),
        // Mixed, one axis above budget -> coarsens, not within -> 0.
        Case::new(1, 10, 5, 8),
        // Large but far from overflow, above budget -> flag 0.
        Case::new(1024, 1023, 512, 2),
        // Cinematic cube above budget -> flag 0.
        Case::new(128, 128, 128, 2),
    ];
    let out = predicate.is_coarsest(&ctx, &to_queries(&cases));
    assert_parity(&out, &cases, "explicit");
}

#[test]
fn single_cell_axes_boundary_parity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let predicate = GpuMgIsCoarsest::new(&ctx);
    // Every combination of single-cell versus coarsenable axis, tested against a
    // one-cell budget so the stuck branch is isolated from the within branch:
    // only an all-single-cell grid is stuck here, every coarsenable axis flips
    // the verdict to 0.
    let cases = [
        Case::new(1, 1, 1, 1),
        Case::new(4, 1, 1, 1),
        Case::new(1, 4, 1, 1),
        Case::new(1, 1, 4, 1),
        Case::new(4, 4, 1, 1),
        Case::new(4, 1, 4, 1),
        Case::new(1, 4, 4, 1),
        Case::new(4, 4, 4, 1),
    ];
    let out = predicate.is_coarsest(&ctx, &to_queries(&cases));
    assert_parity(&out, &cases, "single-cell");
}

#[test]
fn large_deterministic_sweep_parity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let predicate = GpuMgIsCoarsest::new(&ctx);
    let mut rng = Lcg::new(0x5E27_C0A8);
    let mut cases = Vec::new();
    // A spread of extents from single cells up to a few thousand, well clear of
    // overflow, paired with a range of budgets so both the within and stuck
    // branches fire across the sweep.
    for _ in 0..257 {
        let budget = rng.axis(2048);
        cases.push(Case::new(
            rng.axis(4096),
            rng.axis(4096),
            rng.axis(4096),
            budget,
        ));
    }
    // Force single-cell axes (stuck candidates) into the stream against both a
    // one-cell budget and larger budgets.
    for _ in 0..32 {
        cases.push(Case::new(1, 1, 1, 1));
        cases.push(Case::new(1, rng.axis(256), 1, 1));
        cases.push(Case::new(1, 1, rng.axis(256), rng.axis(512)));
        cases.push(Case::new(rng.axis(256), 1, 1, rng.axis(512)));
    }
    let out = predicate.is_coarsest(&ctx, &to_queries(&cases));
    assert_parity(&out, &cases, "sweep");
}
