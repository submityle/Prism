//! Real-device parity for the multigrid coarsening twin:
//! [`GpuMgCoarsen`](prism_volumetric_gpu::mg_coarsen::GpuMgCoarsen) must
//! reproduce the `CPU` golden
//! [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
//! coarsening rules exactly — the per-axis coarsened extents and the
//! coarsened-axis count.
//!
//! The three golden rules (`coarse_axis`, `coarsen`, `coarsened_axis_count`) are
//! private to the golden module, so this test mirrors them verbatim (see the
//! `MIRROR of ...` helpers) and compares the `GPU` batch against that mirror.
//!
//! The fixtures cover the shapes the hierarchy builder exercises: even axes,
//! odd axes (the round-up halving), single-cell axes left untouched, a `1x1x1`
//! grid that coarsens to itself, mixed per-axis extents and large grids kept
//! well away from `u32` overflow, plus a deterministic integer sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every output is pure unsigned integer arithmetic, so the `CPU` and `GPU`
//! results are bit-identical and the comparison is exact equality (`==`), not a
//! tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::multigrid_pressure`
//! 的分辨率粗化整数规则；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::fluid::GridResolution;
use prism_volumetric_gpu::mg_coarsen::{GpuMgCoarsen, GpuMgCoarsenQuery, GpuMgCoarsenResult};
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

// MIRROR of multigrid_pressure::coarsened_axis_count (private, exact
// transcription).
fn golden_coarsened_axis_count(fine: GridResolution, coarse: GridResolution) -> u32 {
    let mut k = 0u32;
    if coarse.nx < fine.nx {
        k += 1;
    }
    if coarse.ny < fine.ny {
        k += 1;
    }
    if coarse.nz < fine.nz {
        k += 1;
    }
    k
}

/// A small integer `LCG` used to synthesize deterministic grids without any
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

/// Runs the whole batch through the kernel and asserts exact parity, grid by
/// grid, against the mirrored golden rules.
fn assert_parity(gpu: &[GpuMgCoarsenResult], fines: &[GridResolution], label: &str) {
    assert_eq!(gpu.len(), fines.len(), "{label}: result count");
    for (i, g) in gpu.iter().enumerate() {
        let fine = fines[i];
        let coarse = golden_coarsen(fine);
        let k = golden_coarsened_axis_count(fine, coarse);
        assert_eq!(
            (g.cx, g.cy, g.cz, g.k),
            (coarse.nx, coarse.ny, coarse.nz, k),
            "{label}: grid[{i}] fine=({}, {}, {})",
            fine.nx,
            fine.ny,
            fine.nz
        );
    }
}

/// Maps fine grids to kernel queries in the same order.
fn to_queries(fines: &[GridResolution]) -> Vec<GpuMgCoarsenQuery> {
    fines
        .iter()
        .map(|f| GpuMgCoarsenQuery {
            nx: f.nx,
            ny: f.ny,
            nz: f.nz,
        })
        .collect()
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let coarsener = GpuMgCoarsen::new(&ctx);
    let out = coarsener.coarsen(&ctx, &[]);
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
    let coarsener = GpuMgCoarsen::new(&ctx);
    let fines = [
        // Single cell on every axis: coarsens to itself, k == 0.
        GridResolution::new(1, 1, 1),
        // All even: each axis halves exactly.
        GridResolution::new(8, 16, 32),
        // All odd (> 1): round-up halving on every axis.
        GridResolution::new(7, 9, 15),
        // Mixed: one single-cell axis (untouched), one even, one odd.
        GridResolution::new(1, 10, 5),
        // Two-cell axes: the smallest coarsenable extent, halves to one.
        GridResolution::new(2, 2, 2),
        // Mixed single-cell and larger extents.
        GridResolution::new(3, 1, 1),
        // Large but far from overflow: even and odd mixed.
        GridResolution::new(1024, 1023, 512),
        // Cinematic cube.
        GridResolution::uniform(128),
    ];
    let out = coarsener.coarsen(&ctx, &to_queries(&fines));
    assert_parity(&out, &fines, "explicit");
}

#[test]
fn single_cell_axes_untouched() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let coarsener = GpuMgCoarsen::new(&ctx);
    // Every combination of single-cell versus coarsenable axis, so each axis's
    // n <= 1 branch is exercised independently.
    let fines = [
        GridResolution::new(1, 1, 1),
        GridResolution::new(4, 1, 1),
        GridResolution::new(1, 4, 1),
        GridResolution::new(1, 1, 4),
        GridResolution::new(4, 4, 1),
        GridResolution::new(4, 1, 4),
        GridResolution::new(1, 4, 4),
    ];
    let out = coarsener.coarsen(&ctx, &to_queries(&fines));
    assert_parity(&out, &fines, "single-cell");
}

#[test]
fn large_deterministic_sweep_parity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let coarsener = GpuMgCoarsen::new(&ctx);
    let mut rng = Lcg::new(0xC0A8_5E27);
    let mut fines = Vec::new();
    // A spread of extents from single cells up to a few thousand, well clear of
    // overflow; integer-only generation keeps the fixture reproducible.
    for _ in 0..257 {
        fines.push(GridResolution::new(
            rng.axis(4096),
            rng.axis(4096),
            rng.axis(4096),
        ));
    }
    // Force some single-cell axes into the stream so the n <= 1 branch is hit.
    for _ in 0..32 {
        fines.push(GridResolution::new(1, rng.axis(256), rng.axis(256)));
        fines.push(GridResolution::new(rng.axis(256), 1, rng.axis(256)));
        fines.push(GridResolution::new(rng.axis(256), rng.axis(256), 1));
    }
    let out = coarsener.coarsen(&ctx, &to_queries(&fines));
    assert_parity(&out, &fines, "sweep");
}
