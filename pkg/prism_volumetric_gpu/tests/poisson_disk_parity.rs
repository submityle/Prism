//! Real-device parity for the deterministic, closed-form core of the
//! blue-noise Poisson-disk sampler:
//! [`GpuPoissonDisk`](prism_volumetric_gpu::GpuPoissonDisk) must reproduce the
//! `CPU` golden
//! [`poisson_disk`](prism_render_architecture::particle::poisson_disk) numeric
//! spine routine for routine across the pseudo-random stream
//! ([`Rng::next_u32`](prism_render_architecture::particle::poisson_disk::Rng::next_u32),
//! [`Rng::next_unit`](prism_render_architecture::particle::poisson_disk::Rng::next_unit)),
//! the background-grid cell size
//! ([`cell_size`](prism_render_architecture::particle::poisson_disk::cell_size)),
//! the single-pair distance core of
//! [`min_pairwise_distance`](prism_render_architecture::particle::poisson_disk::min_pairwise_distance)
//! and its squared companion, the coordinate-to-cell linear index map (the
//! golden private `cell_index_1d`), and the per-candidate min-distance
//! accept/reject predicate (the golden private `candidate_fits`).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The integer routines (`next_u32`, the cell linear index and the
//! accept/reject verdict) are bit-exact and are compared with `==`. The
//! continuous entries (`next_unit`, `cell_size` and the pair distances) thread
//! through a `u32`-to-`f32` widen, a divide or a `sqrt`, so `CPU` and `GPU` are
//! not guaranteed bit-exact: a `GPU` may round a widen or a divide a hair
//! differently. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every continuous quantity. Every
//! fixture is placed clear of every branch boundary: the cell size stays
//! strictly positive, the cell-index coordinates sit at cell midpoints well
//! away from a `floor` seam, the candidate-fits neighbors sit either well
//! inside or well outside the `r` radius, and the pair-distance points are
//! chosen so the distance is well conditioned. The fixtures use no
//! transcendental method (only algebraic multiplies and `sqrt` through the
//! reused math), and the random batch draws from a host-side integer `LCG` so
//! it needs no external math library.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::poisson_disk`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::poisson_disk::cpu_reference;
use prism_volumetric_gpu::{GpuContext, GpuPoissonDisk, PoissonDiskQuery, PoissonDiskResult};

/// Absolute parity bound on every continuous quantity. A `GPU` may round a
/// widen or a divide a hair differently from the scalar reference, perturbing
/// the low mantissa bits by a few units in the last place; `1e-4` admits that
/// legal slack while still failing a genuinely wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes (hot `next_u32`-derived
/// draws, large pair distances) where a few units in the last place exceed the
/// absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn approx(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Returns whether a `GPU` result matches the `CPU` reference, matching variant
/// against variant: the integer and boolean lanes compare exactly, the scalar
/// lane applies the continuous-field bound.
#[expect(
    clippy::match_same_arms,
    reason = "the integer Word and Index lanes read identically but guard distinct result variants, so the arms stay separate for clarity"
)]
fn results_match(got: &PoissonDiskResult, want: &PoissonDiskResult) -> bool {
    match (got, want) {
        (PoissonDiskResult::Scalar(a), PoissonDiskResult::Scalar(b)) => approx(*a, *b),
        (PoissonDiskResult::Word(a), PoissonDiskResult::Word(b)) => a == b,
        (PoissonDiskResult::Index(a), PoissonDiskResult::Index(b)) => a == b,
        (PoissonDiskResult::Fits(a), PoissonDiskResult::Fits(b)) => a == b,
        _ => false,
    }
}

/// Evaluates a single query on device and asserts it matches the `CPU` golden.
fn check(gpu: &GpuPoissonDisk, ctx: &GpuContext, query: PoissonDiskQuery) {
    let want = cpu_reference(&query);
    let got = gpu.evaluate(ctx, std::slice::from_ref(&query));
    assert_eq!(got.len(), 1, "one query yields one result");
    assert!(
        results_match(&got[0], &want),
        "GPU result {:?} must match CPU {:?} for {:?}",
        got[0],
        want,
        query
    );
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Advances the generator and returns a raw 32-bit word, feeding the integer
/// fixtures (generator seeds, column counts, cell coordinates).
fn lcg_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// Draws a uniform `f32` in `[lo, hi)` from the generator.
fn rand_range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// Draws a point whose components lie in `[-2, 2)`.
fn rand_point2(state: &mut u64) -> [f32; 2] {
    [rand_range(state, -2.0, 2.0), rand_range(state, -2.0, 2.0)]
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPoissonDisk::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn next_u32_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPoissonDisk::new(&ctx);
    for state in [0u32, 1, 2, 0x12345678, 0xDEADBEEF, 0xFFFFFFFF, 0x9E3779B9] {
        check(&gpu, &ctx, PoissonDiskQuery::NextU32 { state });
    }
}

#[test]
fn next_unit_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPoissonDisk::new(&ctx);
    for state in [0u32, 7, 42, 0x0BADF00D, 0xC0FFEE11, 0x7FFFFFFF] {
        check(&gpu, &ctx, PoissonDiskQuery::NextUnit { state });
    }
}

#[test]
fn cell_size_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPoissonDisk::new(&ctx);
    // Strictly positive radii, well away from the degenerate zero point.
    for radius in [0.5f32, 1.0, 2.0, 3.5, 10.0] {
        check(&gpu, &ctx, PoissonDiskQuery::CellSize { radius });
    }
}

#[test]
fn pair_distance_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPoissonDisk::new(&ctx);
    // Clean 3-4-5 triangle: distance is exactly 5.
    check(
        &gpu,
        &ctx,
        PoissonDiskQuery::PairDistance {
            a: [0.0, 0.0],
            b: [3.0, 4.0],
        },
    );
    check(
        &gpu,
        &ctx,
        PoissonDiskQuery::PairDistance {
            a: [-1.5, 2.0],
            b: [4.5, -2.0],
        },
    );
}

#[test]
fn pair_distance_squared_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPoissonDisk::new(&ctx);
    // 3^2 + 4^2 = 25.
    check(
        &gpu,
        &ctx,
        PoissonDiskQuery::PairDistanceSquared {
            a: [1.0, 2.0],
            b: [4.0, 6.0],
        },
    );
    check(
        &gpu,
        &ctx,
        PoissonDiskQuery::PairDistanceSquared {
            a: [-2.0, -1.0],
            b: [1.0, 3.0],
        },
    );
}

#[test]
fn cell_index_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPoissonDisk::new(&ctx);
    let cell = 0.5f32;
    let cols = 8u32;
    // Coordinates sit at cell midpoints, well clear of any `floor` seam:
    // col = floor(2.5) = 2, row = floor(1.5) = 1, index = 1*8 + 2 = 10.
    check(
        &gpu,
        &ctx,
        PoissonDiskQuery::CellIndex {
            coord: [cell * 2.5, cell * 1.5],
            cell,
            cols,
        },
    );
    // col = floor(0.5) = 0, row = floor(5.5) = 5, index = 5*8 + 0 = 40.
    check(
        &gpu,
        &ctx,
        PoissonDiskQuery::CellIndex {
            coord: [cell * 0.5, cell * 5.5],
            cell,
            cols,
        },
    );
}

#[test]
fn candidate_fits_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPoissonDisk::new(&ctx);
    let candidate = [1.0f32, 1.0];
    let r = 2.0f32;
    let r_squared = r * r;

    // Accept: every neighbor sits well outside r (axis-aligned offsets of
    // 3.0 and 3.5 > r = 2.0), so the squared distances all clear r^2.
    let mut neighbors = [[0.0f32; 2]; 8];
    neighbors[0] = [candidate[0] + 3.0, candidate[1]];
    neighbors[1] = [candidate[0], candidate[1] + 3.5];
    neighbors[2] = [candidate[0] - 3.0, candidate[1] - 3.0];
    check(
        &gpu,
        &ctx,
        PoissonDiskQuery::CandidateFits {
            candidate,
            neighbors,
            neighbor_count: 3,
            r_squared,
        },
    );

    // Reject: neighbor 1 sits well inside r (offset 0.8 < r = 2.0), so its
    // squared distance falls below r^2 and the verdict flips to false.
    let mut neighbors = [[0.0f32; 2]; 8];
    neighbors[0] = [candidate[0] + 3.0, candidate[1]];
    neighbors[1] = [candidate[0] + 0.8, candidate[1]];
    neighbors[2] = [candidate[0] - 3.0, candidate[1] - 3.0];
    check(
        &gpu,
        &ctx,
        PoissonDiskQuery::CandidateFits {
            candidate,
            neighbors,
            neighbor_count: 3,
            r_squared,
        },
    );
}

/// Appends one query of every routine, drawing fixtures clear of each branch
/// boundary. `reject` forces the candidate-fits case to its rejecting branch so
/// the batch exercises both verdicts.
fn push_suite(state: &mut u64, reject: bool, queries: &mut Vec<PoissonDiskQuery>) {
    queries.push(PoissonDiskQuery::NextU32 {
        state: lcg_u32(state),
    });
    queries.push(PoissonDiskQuery::NextUnit {
        state: lcg_u32(state),
    });
    queries.push(PoissonDiskQuery::CellSize {
        radius: rand_range(state, 0.5, 4.0),
    });
    queries.push(PoissonDiskQuery::PairDistance {
        a: rand_point2(state),
        b: rand_point2(state),
    });
    queries.push(PoissonDiskQuery::PairDistanceSquared {
        a: rand_point2(state),
        b: rand_point2(state),
    });

    // Cell index: coordinates land at cell midpoints (`k + 0.5`), well clear of
    // every `floor` seam.
    let cell = rand_range(state, 0.5, 3.0);
    let cols = 1 + (lcg_u32(state) % 32);
    let kx = lcg_u32(state) % cols;
    let ky = lcg_u32(state) % 20;
    queries.push(PoissonDiskQuery::CellIndex {
        coord: [cell * (kx as f32 + 0.5), cell * (ky as f32 + 0.5)],
        cell,
        cols,
    });

    // Candidate fits: four neighbors placed either well outside r (accept) or,
    // when `reject` is set, with the first neighbor well inside r (reject).
    let r = rand_range(state, 3.0, 6.0);
    let r_squared = r * r;
    let candidate = rand_point2(state);
    let mut neighbors = [[0.0f32; 2]; 8];
    for (i, slot) in neighbors.iter_mut().take(4).enumerate() {
        let d = rand_range(state, 1.3 * r, 2.5 * r);
        *slot = [candidate[0] + d, candidate[1] + i as f32];
    }
    if reject {
        let d = rand_range(state, 0.3 * r, 0.7 * r);
        neighbors[0] = [candidate[0] + d, candidate[1]];
    }
    queries.push(PoissonDiskQuery::CandidateFits {
        candidate,
        neighbors,
        neighbor_count: 4,
        r_squared,
    });
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPoissonDisk::new(&ctx);
    let mut state = 0x_f1d1_0f0f_51c3_0001_u64;
    let mut queries = Vec::new();
    for round in 0..24 {
        push_suite(&mut state, round % 2 == 0, &mut queries);
    }
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");

    let mut saw_fit = false;
    let mut saw_reject = false;
    for (q, g) in queries.iter().zip(got.iter()) {
        let want = cpu_reference(q);
        if let PoissonDiskResult::Fits(fits) = want {
            if fits {
                saw_fit = true;
            } else {
                saw_reject = true;
            }
        }
        assert!(
            results_match(g, &want),
            "GPU result {g:?} must match CPU {want:?} for {q:?}"
        );
    }
    assert!(
        saw_fit && saw_reject,
        "the batch must exercise both candidate-fits verdicts"
    );
}
