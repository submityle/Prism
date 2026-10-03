//! Real-device parity for the stateless Worley / cellular (`Voronoi`) noise
//! twin:
//! [`GpuWorleyNoise`](prism_volumetric_gpu::worley_noise::GpuWorleyNoise) must
//! reproduce the closed form of the golden
//! `prism_render_architecture::particle::worley` — the nearest (`F1`) and
//! next-nearest (`F2`) feature-point distances over the `3x3x3` candidate
//! block, plus the derived `edges = F2 - F1` and `inverted = 1 - F1` — across a
//! battery of named fixtures and a conditioned randomized sweep compared
//! query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle below is
//! an *independent* reimplementation of the same closed form: the identical
//! `32`-bit avalanche (`mix` / `finalize` with the same salts and constants),
//! the same `[0, 1)` jitter, the same `3x3x3` neighbourhood search and the same
//! `Euclidean` / `Manhattan` / `Chebyshev` metrics. Because the reference and
//! this oracle are both scalar `f32`, a `GPU == oracle` pass is direct evidence
//! the ported kernel computes the same field the reference does, not merely
//! that the shader compiles.
//!
//! # Parity criterion
//!
//! The hash and jitter are pure integer / exact-multiply arithmetic, so those
//! agree bit for bit; only the metric distance threads through `sqrt` or
//! `abs` / `max`, so a `GPU` result may land a few units in the last place from
//! the scalar oracle. Each field is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, with a `rel_diff` floor of `1e-6` so a near-zero
//! expected value (such as `F1` at a feature point) does not inflate the
//! relative error.
//!
//! # Conditioning
//!
//! The `F1` / `F2` selection is a `min` over `27` candidate distances. When two
//! candidates tie to within a few units in the last place, the host scalar
//! order and the device order (which may fuse multiply-adds differently) can
//! disagree about which distance is `F1` versus `F2`, flipping the reported
//! pair even though every individual distance agrees. The randomized sweep
//! therefore computes all `27` candidate distances on the host, sorts them, and
//! rejects (resamples) any draw whose two smallest gaps are below a `2e-3`
//! margin, keeping the sweep clear of that knife-edge. The named fixtures stay
//! well inside single cells for the same reason.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::worley`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::worley_noise::{
    GpuWorleyNoise, WorleyNoiseQuery, WorleyNoiseResult, METRIC_CHEBYSHEV, METRIC_EUCLIDEAN,
    METRIC_MANHATTAN,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each cellular field. A `GPU` `sqrt` / `abs` / `max` may
/// land a few units in the last place from the scalar oracle; `1e-4` admits
/// that legal slack while still failing a wrong port.
const FIELD_ABS: f32 = 1.0e-4;

/// Relative bound on each cellular field, applied for larger magnitudes where a
/// few units in the last place exceed the absolute floor.
const FIELD_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Small ordering slack for the invariant assertions, matching the golden
/// module's `EPS`, so `F1 <= F2` and `edges >= 0` are checked with slack rather
/// than a forbidden bare comparison.
const EPS: f32 = 1.0e-6;

/// Odd-integer salt for the X jitter axis (mirrors the golden `SALT_X`).
const SALT_X: u32 = 0x68E3_1DA4;
/// Odd-integer salt for the Y jitter axis (mirrors the golden `SALT_Y`).
const SALT_Y: u32 = 0xB529_7A4D;
/// Odd-integer salt for the Z jitter axis (mirrors the golden `SALT_Z`).
const SALT_Z: u32 = 0x1B56_C4E9;
/// Scale turning a `16`-bit hash segment into `[0, 1)` (`1 / 65536`).
const INV_2POW16: f32 = 1.0 / 65_536.0;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// One folding step of the lattice hash: xor-in a multiplied input word, then a
/// `15`-bit left rotate and a multiply spread the bits. Integer multiply wraps
/// mod `2^32`, matching the device `u32` arithmetic.
fn mix_hash(mut h: u32, v: u32) -> u32 {
    h ^= v.wrapping_mul(0x9E37_79B1);
    h = h.rotate_left(15).wrapping_mul(0x85EB_CA6B);
    h
}

/// Final avalanche applied once after all inputs are folded.
fn finalize_hash(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h
}

/// Stateless integer hash of a lattice cell and seed; the signed cell indices
/// are reinterpreted through their two's-complement bits (`as u32`).
fn hash_lattice(i: i32, j: i32, k: i32, seed: u32) -> u32 {
    let mut h = seed ^ 0x811C_9DC5;
    h = mix_hash(h, i as u32);
    h = mix_hash(h, j as u32);
    h = mix_hash(h, k as u32);
    finalize_hash(h)
}

/// Maps a `32`-bit hash to `[0, 1)` using its low `16`-bit segment.
fn unit01(h: u32) -> f32 {
    ((h & 0xFFFF) as f32) * INV_2POW16
}

/// The jittered feature point of integer cell `(i, j, k)`: the cell's corner
/// offset by an independent per-axis jitter in `[0, 1)`.
fn feature_point(i: i32, j: i32, k: i32, seed: u32) -> [f32; 3] {
    [
        i as f32 + unit01(hash_lattice(i, j, k, seed ^ SALT_X)),
        j as f32 + unit01(hash_lattice(i, j, k, seed ^ SALT_Y)),
        k as f32 + unit01(hash_lattice(i, j, k, seed ^ SALT_Z)),
    ]
}

/// Distance from `a` to `b` under the selected metric, summing the squared
/// components in the same `x, y, z` order as the device before the single
/// `sqrt`.
fn metric_distance(metric: u32, a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    match metric {
        METRIC_MANHATTAN => dx.abs() + dy.abs() + dz.abs(),
        METRIC_CHEBYSHEV => dx.abs().max(dy.abs()).max(dz.abs()),
        _ => (dx * dx + dy * dy + dz * dz).sqrt(),
    }
}

/// Collects all `27` candidate distances for the sample's `3x3x3` block, in the
/// device's `di, dj, dk` iteration order.
fn candidate_distances(pos: [f32; 3], seed: u32, metric: u32) -> Vec<f32> {
    let bi = pos[0].floor() as i32;
    let bj = pos[1].floor() as i32;
    let bk = pos[2].floor() as i32;
    let mut ds = Vec::with_capacity(27);
    for di in -1..=1 {
        for dj in -1..=1 {
            for dk in -1..=1 {
                let fp = feature_point(bi + di, bj + dj, bk + dk, seed);
                ds.push(metric_distance(metric, pos, fp));
            }
        }
    }
    ds
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against. Mirrors the golden `search` keeping
/// the two smallest distances with strict `<` comparisons.
fn oracle(q: &WorleyNoiseQuery) -> WorleyNoiseResult {
    let mut f1 = f32::INFINITY;
    let mut f2 = f32::INFINITY;
    for d in candidate_distances([q.pos_x, q.pos_y, q.pos_z], q.seed, q.metric) {
        if d < f1 {
            f2 = f1;
            f1 = d;
        } else if d < f2 {
            f2 = d;
        }
    }
    WorleyNoiseResult {
        f1,
        f2,
        edges: f2 - f1,
        inverted: 1.0 - f1,
    }
}

/// Pins one `GPU` result against the host oracle, each field under the field
/// bound.
fn check_one(idx: usize, got: &WorleyNoiseResult, want: &WorleyNoiseResult) {
    assert!(
        close(got.f1, want.f1, FIELD_ABS, FIELD_REL),
        "query {idx} f1: gpu {} vs cpu {}",
        got.f1,
        want.f1
    );
    assert!(
        close(got.f2, want.f2, FIELD_ABS, FIELD_REL),
        "query {idx} f2: gpu {} vs cpu {}",
        got.f2,
        want.f2
    );
    assert!(
        close(got.edges, want.edges, FIELD_ABS, FIELD_REL),
        "query {idx} edges: gpu {} vs cpu {}",
        got.edges,
        want.edges
    );
    assert!(
        close(got.inverted, want.inverted, FIELD_ABS, FIELD_REL),
        "query {idx} inverted: gpu {} vs cpu {}",
        got.inverted,
        want.inverted
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWorleyNoise, queries: &[WorleyNoiseQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, result, &want);
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a `f32` in `[lo, hi)` from the generator, using only integer-to-float
/// division (no transcendental).
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = lcg(state) as f32 * (1.0 / 4_294_967_296.0);
    lo + (hi - lo) * u
}

/// Smallest and second-smallest gap among a sample's `27` candidate distances;
/// a draw is only accepted when both exceed the conditioning margin so the
/// `F1` / `F2` selection cannot flip between host and device.
fn two_smallest_gaps(pos: [f32; 3], seed: u32, metric: u32) -> (f32, f32) {
    let mut ds = candidate_distances(pos, seed, metric);
    ds.sort_by(|a, b| a.partial_cmp(b).expect("candidate distances are finite"));
    (ds[1] - ds[0], ds[2] - ds[1])
}

/// Builds one well-conditioned random query: a position in `[-4, 4]^3`, a
/// random seed and a cycling metric. Positions whose two smallest candidate
/// gaps fall under the margin are resampled so the sweep never straddles an
/// `F1` / `F2` tie.
fn random_query(state: &mut u64, metric: u32) -> WorleyNoiseQuery {
    /// Minimum separation between the two closest candidate distances.
    const MARGIN: f32 = 2.0e-3;
    loop {
        let pos = [
            uniform(state, -4.0, 4.0),
            uniform(state, -4.0, 4.0),
            uniform(state, -4.0, 4.0),
        ];
        let seed = lcg(state);
        let (gap1, gap2) = two_smallest_gaps(pos, seed, metric);
        if gap1 >= MARGIN && gap2 >= MARGIN {
            return WorleyNoiseQuery::new(pos[0], pos[1], pos[2], seed, metric);
        }
    }
}

/// A fixed battery of named cases spanning the three metrics and several seeds,
/// each kept well inside a single cell so no two candidates tie.
fn fixture_queries() -> Vec<WorleyNoiseQuery> {
    vec![
        WorleyNoiseQuery::new(0.37, 1.11, -0.62, 21, METRIC_EUCLIDEAN),
        WorleyNoiseQuery::new(0.37, 1.11, -0.62, 21, METRIC_MANHATTAN),
        WorleyNoiseQuery::new(0.37, 1.11, -0.62, 21, METRIC_CHEBYSHEV),
        WorleyNoiseQuery::new(2.5, -3.25, 4.75, 9, METRIC_EUCLIDEAN),
        WorleyNoiseQuery::new(2.5, -3.25, 4.75, 9, METRIC_MANHATTAN),
        WorleyNoiseQuery::new(2.5, -3.25, 4.75, 9, METRIC_CHEBYSHEV),
        WorleyNoiseQuery::new(-1.2, 0.8, 3.3, 44, METRIC_EUCLIDEAN),
        WorleyNoiseQuery::new(-1.2, 0.8, 3.3, 44, METRIC_MANHATTAN),
        WorleyNoiseQuery::new(10.1, -7.4, 0.15, 13, METRIC_EUCLIDEAN),
        WorleyNoiseQuery::new(0.5, 0.5, 0.5, 6, METRIC_EUCLIDEAN),
        WorleyNoiseQuery::new(-5.9, 5.9, -5.9, 99, METRIC_CHEBYSHEV),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping worley_noise parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWorleyNoise::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer
    // cannot be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn f1_is_zero_at_a_feature_point() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWorleyNoise::new(&ctx);
    // A point placed exactly on a cell's feature point has F1 == 0, because
    // that point lives inside the searched 3x3x3 block.
    let seed = 55;
    let fp = feature_point(0, 0, 0, seed);
    let queries = [
        WorleyNoiseQuery::new(fp[0], fp[1], fp[2], seed, METRIC_EUCLIDEAN),
        WorleyNoiseQuery::new(fp[0], fp[1], fp[2], seed, METRIC_MANHATTAN),
        WorleyNoiseQuery::new(fp[0], fp[1], fp[2], seed, METRIC_CHEBYSHEV),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        check_one(idx, result, &oracle(q));
        assert!(
            result.f1.abs() <= FIELD_ABS,
            "F1 at a feature point should be ~0, got {}",
            result.f1
        );
    }
}

#[test]
fn metric_ordering_manhattan_ge_euclidean_ge_chebyshev() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWorleyNoise::new(&ctx);
    // F1 is a minimum over the same candidate set for every metric, so the
    // per-pair ordering L1 >= L2 >= Linf is inherited by the F1 values.
    let samples = [
        [0.37_f32, 1.11, -0.62],
        [2.5, -3.25, 4.75],
        [-1.2, 0.8, 3.3],
        [10.1, -7.4, 0.15],
    ];
    let mut queries = Vec::new();
    for p in samples {
        queries.push(WorleyNoiseQuery::new(p[0], p[1], p[2], 9, METRIC_MANHATTAN));
        queries.push(WorleyNoiseQuery::new(p[0], p[1], p[2], 9, METRIC_EUCLIDEAN));
        queries.push(WorleyNoiseQuery::new(p[0], p[1], p[2], 9, METRIC_CHEBYSHEV));
    }
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        check_one(idx, result, &oracle(q));
    }
    for triple in got.chunks_exact(3) {
        let man = triple[0].f1;
        let euc = triple[1].f1;
        let che = triple[2].f1;
        assert!(man >= euc - EPS, "Manhattan {man} < Euclidean {euc}");
        assert!(euc >= che - EPS, "Euclidean {euc} < Chebyshev {che}");
    }
}

#[test]
fn fields_satisfy_cellular_invariants() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWorleyNoise::new(&ctx);
    // The device must honour the structural invariants of the fields:
    // F1 <= F2, edges = F2 - F1 >= 0, and inverted = 1 - F1.
    let queries = fixture_queries();
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, result) in got.iter().enumerate() {
        assert!(
            result.f1 <= result.f2 + EPS,
            "query {idx}: f1 {} must not exceed f2 {}",
            result.f1,
            result.f2
        );
        assert!(
            result.edges >= -EPS,
            "query {idx}: edges {} must be non-negative",
            result.edges
        );
        assert!(
            close(result.edges, result.f2 - result.f1, FIELD_ABS, FIELD_REL),
            "query {idx}: edges {} must equal f2 - f1 {}",
            result.edges,
            result.f2 - result.f1
        );
        assert!(
            close(result.inverted, 1.0 - result.f1, FIELD_ABS, FIELD_REL),
            "query {idx}: inverted {} must equal 1 - f1 {}",
            result.inverted,
            1.0 - result.f1
        );
    }
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWorleyNoise::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWorleyNoise::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let metrics = [METRIC_EUCLIDEAN, METRIC_MANHATTAN, METRIC_CHEBYSHEV];
    let mut queries = fixture_queries();
    // Several workgroups' worth of conditioned random queries pin every field
    // across a wide span of positions, seeds and all three metrics.
    for i in 0..384 {
        let metric = metrics[i % metrics.len()];
        queries.push(random_query(&mut state, metric));
    }
    check(&ctx, &gpu, &queries);
}
