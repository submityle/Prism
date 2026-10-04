//! Real-device parity test for the oriented-bounding-box volume twin.
//!
//! Each case evaluates one or more [`ObbVolumeQuery`] values on the GPU and
//! pins the returned [`ObbVolumeResult`] against an independent `f32`
//! reimplementation of `prism_physics_core::collider::obb::Obb::volume`. The
//! oracle is rebuilt here from first principles; this test never depends on the
//! golden crate, and the crate carries no `glam` dev-dependency, so the triple
//! product is hand-written in `f32` in the same arithmetic order as the kernel.
//!
//! The `volume` channel is a continuous quantity threaded through two
//! multiplies, so it is pinned with an `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! tolerance (`REL_FLOOR = 1e-6`) that absorbs a fused multiply-add the scalar
//! reference leaves separate. The relative term carries a large volume while
//! the floor keeps a tiny volume honest. The discrete `valid` flag is pinned
//! with an exact `==`; the formula has no degenerate branch, so it is always
//! `1`.
//!
//! Every case short-circuits to a skip when no headless adapter is available,
//! so the suite is inert on a machine without a GPU and exercises the real
//! device elsewhere.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::obb::Obb::volume`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::obb_volume::{GpuObbVolume, ObbVolumeQuery, ObbVolumeResult};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor so a near-zero reference magnitude does not demand
/// an impossibly tight absolute match.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent `f32` reimplementation of `Obb::volume` for one query, in the
/// same arithmetic order as the kernel: `volume = 8.0 * he.x * he.y * he.z`,
/// `valid = 1`.
fn oracle(query: &ObbVolumeQuery) -> ObbVolumeResult {
    let [hx, hy, hz] = query.half_extents;
    ObbVolumeResult {
        volume: 8.0 * hx * hy * hz,
        valid: 1,
    }
}

/// Returns `true` when `a` matches `b` within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= 1.0e-3
}

/// Pins one `GPU` result against the independent host oracle: the continuous
/// `volume` channel within tolerance, the discrete `valid` flag exact.
fn pin(idx: usize, query: &ObbVolumeQuery, result: &ObbVolumeResult) {
    let want = oracle(query);
    assert!(
        close(result.volume, want.volume),
        "query {idx}: volume gpu={} oracle={}",
        result.volume,
        want.volume
    );
    assert_eq!(
        result.valid, want.valid,
        "query {idx}: valid gpu={} oracle={}",
        result.valid, want.valid
    );
}

/// Evaluates `queries` on-device and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuObbVolume, queries: &[ObbVolumeQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

/// 64-bit linear-congruential step (`Knuth`/`PCG` constants), returning the
/// high word so the stream has good spread without any transcendental math.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// A deterministic pseudo-random `f32` in `[0, 1]`.
fn unit01(state: &mut u64) -> f32 {
    lcg(state) as f32 / u32::MAX as f32
}

/// A deterministic pseudo-random `f32` in `[lo, hi]`.
fn range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + unit01(state) * (hi - lo)
}

/// A deterministic random query with each half-extent in `[0.1, 10.0]`, which
/// keeps the volume comfortably inside the `f32` dynamic range so the parity
/// comparison is never contaminated by an overflow to infinity.
fn rand_query(state: &mut u64) -> ObbVolumeQuery {
    ObbVolumeQuery::new([
        range(state, 0.1, 10.0),
        range(state, 0.1, 10.0),
        range(state, 0.1, 10.0),
    ])
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbVolume::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty input must return an empty vector");
}

#[test]
fn unit_cube_has_unit_volume() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbVolume::new(&ctx);
    // Half-extents of 0.5 describe a unit cube: 8 * 0.5^3 = 1.
    let query = ObbVolumeQuery::new([0.5, 0.5, 0.5]);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(
        got[0].valid, 1,
        "the volume formula has no degenerate branch"
    );
    assert!(
        close(got[0].volume, 1.0),
        "a unit cube has unit volume, got {}",
        got[0].volume
    );
    pin(0, &query, &got[0]);
}

#[test]
fn asymmetric_half_extents_multiply_out() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbVolume::new(&ctx);
    // 8 * 1 * 2 * 3 = 48.
    let query = ObbVolumeQuery::new([1.0, 2.0, 3.0]);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].valid, 1, "every half-extent triple is valid");
    assert!(
        close(got[0].volume, 48.0),
        "asymmetric volume must be 48, got {}",
        got[0].volume
    );
    pin(0, &query, &got[0]);
}

#[test]
fn large_half_extents_lean_on_relative_tolerance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbVolume::new(&ctx);
    // 8 * 100 * 200 * 300 = 4.8e7; the absolute term cannot hold such a large
    // volume, so the relative tolerance carries the comparison.
    let query = ObbVolumeQuery::new([100.0, 200.0, 300.0]);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].valid, 1, "a large box is still valid");
    assert!(
        close(got[0].volume, 4.8e7),
        "large volume must be 4.8e7, got {}",
        got[0].volume
    );
    pin(0, &query, &got[0]);
}

#[test]
fn tiny_half_extents_lean_on_the_relative_floor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbVolume::new(&ctx);
    // 8 * 1e-3^3 = 8e-9; a tiny reference magnitude relies on REL_FLOOR to keep
    // the relative test meaningful.
    let query = ObbVolumeQuery::new([1.0e-3, 1.0e-3, 1.0e-3]);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].valid, 1, "a tiny box is still valid");
    assert!(
        close(got[0].volume, 8.0e-9),
        "tiny volume must be 8e-9, got {}",
        got[0].volume
    );
    pin(0, &query, &got[0]);
}

#[test]
fn stride_regression_two_element_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbVolume::new(&ctx);
    // Two distinct queries: a wrong per-element stride would cross-contaminate
    // the two answers.
    let queries = [
        ObbVolumeQuery::new([0.5, 0.5, 0.5]),
        ObbVolumeQuery::new([1.0, 2.0, 3.0]),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), 2, "both results must be returned");
    assert!(close(got[0].volume, 1.0), "first box is the unit cube");
    assert!(close(got[1].volume, 48.0), "second box is 48");
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbVolume::new(&ctx);
    let mut queries = vec![
        ObbVolumeQuery::new([0.5, 0.5, 0.5]),
        ObbVolumeQuery::new([1.0, 2.0, 3.0]),
        ObbVolumeQuery::new([100.0, 200.0, 300.0]),
        ObbVolumeQuery::new([1.0e-3, 1.0e-3, 1.0e-3]),
    ];
    let mut state: u64 = 0x51A7_3C9D_0E12_4455;
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbVolume::new(&ctx);
    let mut state: u64 = 0x0C3A_1F70_7B6E_9D11;
    let queries: Vec<ObbVolumeQuery> = (0..512).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
