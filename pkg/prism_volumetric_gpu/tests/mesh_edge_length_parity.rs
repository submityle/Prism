//! Real-device parity for the mesh edge-length per-edge twin:
//! [`GpuMeshEdgeLength`](prism_volumetric_gpu::mesh_edge_length::GpuMeshEdgeLength)
//! must reproduce the numeric core of the `CPU` golden
//! [`mesh_edge_length_stats`](prism_render_architecture::ray_scene::mesh_edge_length_stats)
//! — the Euclidean length of one undirected edge and its strict classification
//! against a threshold — across named fixtures and a randomized sweep compared
//! edge-for-edge.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden's `distance` helper and its `count_shorter_than` /
//! `count_longer_than` filters are the exact closed form twinned here. Because
//! the public `edge_length_stats` is a variable-length, de-duplicated, sorted
//! aggregate, the expected per-edge values are reconstructed in-host from that
//! closed form (`length = sqrt(dx^2 + dy^2 + dz^2)`, `shorter = length <
//! threshold`, `longer = length > threshold`). The oracle is an independent
//! re-implementation and does not import `prism_render_architecture`.
//!
//! # Parity criterion
//!
//! The `length` threads through a `sqrt`, so a `GPU` `sqrt` may land a few units
//! in the last place from the scalar reference; it is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`). The `shorter`
//! and `longer` flags are discrete decisions and are asserted with `==`.
//!
//! # Conditioning
//!
//! The flags are strict ordered comparisons against the threshold, so a fixture
//! whose length sits a hair from the threshold could flip once a `GPU` `sqrt`
//! and a `CPU` `sqrt` disagree by a unit in the last place. The named fixtures
//! and the random sweep keep `|length - threshold|` a comfortable margin clear
//! of zero before asserting the flags; a separate named fixture pins the exact
//! at-threshold tie, where the strict inequalities leave both flags `0`, and a
//! zero-length fixture pins the degenerate `a == b` edge.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_edge_length_stats`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_edge_length::{
    GpuMeshEdgeLength, MeshEdgeLengthQuery, MeshEdgeLengthResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a length. A `GPU` `sqrt` may land a few units in the
/// last place from the scalar reference; `1e-4` admits that legal slack while
/// still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Margin kept between a length and the threshold before asserting the discrete
/// flags, so a last-place `sqrt` difference can never flip a strict comparison.
const CLASSIFY_MARGIN: f32 = 1.0e-2;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Reconstructs the golden's exact per-edge closed form in-host: the faithful
/// oracle the `GPU` is pinned against. Uses only `sqrt` and strict ordered
/// comparisons, never an `f32` `==`, matching the house rules.
fn oracle(a: [f32; 3], b: [f32; 3], threshold: f32) -> MeshEdgeLengthResult {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    let length = (dx * dx + dy * dy + dz * dz).sqrt();
    MeshEdgeLengthResult {
        length,
        shorter: length < threshold,
        longer: length > threshold,
    }
}

/// Pins one `GPU` edge result against the in-host oracle: the length within
/// tolerance, both flags exactly.
fn check_edge(idx: usize, got: &MeshEdgeLengthResult, want: &MeshEdgeLengthResult) {
    assert!(
        close(got.length, want.length),
        "edge {idx} length: gpu {} vs cpu {}",
        got.length,
        want.length
    );
    assert_eq!(
        got.shorter, want.shorter,
        "edge {idx} shorter: gpu {} vs cpu {}",
        got.shorter, want.shorter
    );
    assert_eq!(
        got.longer, want.longer,
        "edge {idx} longer: gpu {} vs cpu {}",
        got.longer, want.longer
    );
}

/// Dispatches every edge and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuMeshEdgeLength, queries: &[MeshEdgeLengthQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q.a, q.b, q.threshold);
        check_edge(idx, result, &want);
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

/// Draws a coordinate in `[-50.0, 50.0]` at milli resolution from `state`.
fn coord(state: &mut u64) -> f32 {
    -50.0 + (lcg(state) % 100_000) as f32 / 1000.0
}

/// Draws a positive threshold in `[0.5, 120.0]` at milli resolution from
/// `state`.
fn threshold(state: &mut u64) -> f32 {
    0.5 + (lcg(state) % 119_500) as f32 / 1000.0
}

/// Whether an edge is clear of the at-threshold tie: its length stays a full
/// [`CLASSIFY_MARGIN`] clear of the threshold, so both the `CPU` and the `GPU`
/// land on the same side of each strict comparison. Uses only `sqrt` and an
/// ordered comparison, never an `f32` `==`.
fn well_conditioned(q: &MeshEdgeLengthQuery) -> bool {
    let want = oracle(q.a, q.b, q.threshold);
    (want.length - q.threshold).abs() >= CLASSIFY_MARGIN
}

/// The deterministic named edge fixtures spanning the shorter/longer branches
/// and the degenerate zero-length edge. Each keeps a comfortable margin from the
/// threshold except the explicit at-threshold tie, handled separately.
fn fixture_edges() -> Vec<MeshEdgeLengthQuery> {
    vec![
        // Unit axis edge, length 1, well under the threshold -> shorter.
        MeshEdgeLengthQuery::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 5.0),
        // 3-4-5 right triangle hypotenuse, length 5, over the threshold ->
        // longer.
        MeshEdgeLengthQuery::new([0.0, 0.0, 0.0], [3.0, 4.0, 0.0], 2.0),
        // 1-2-2 edge, length 3, under the threshold -> shorter.
        MeshEdgeLengthQuery::new([1.0, 1.0, 1.0], [2.0, 3.0, 3.0], 10.0),
        // Full 3-D diagonal, length sqrt(27) ~= 5.196, over the threshold ->
        // longer.
        MeshEdgeLengthQuery::new([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0], 4.0),
        // Negative-coordinate edge, length 2, over the threshold -> longer.
        MeshEdgeLengthQuery::new([-3.0, -4.0, 0.0], [-3.0, -4.0, 2.0], 1.0),
        // Degenerate zero-length edge: a == b, length 0, under any positive
        // threshold -> shorter.
        MeshEdgeLengthQuery::new([7.5, -2.25, 3.0], [7.5, -2.25, 3.0], 0.5),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_edge_length parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuMeshEdgeLength::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn named_fixtures_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshEdgeLength::new(&ctx);
    let edges = fixture_edges();
    // The named fixtures are chosen clear of the at-threshold tie.
    for q in &edges {
        assert!(
            well_conditioned(q),
            "named fixture must stay clear of the at-threshold tie"
        );
    }
    check(&ctx, &gpu, &edges);
}

#[test]
fn zero_length_edge_is_classified_shorter() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshEdgeLength::new(&ctx);
    // A degenerate edge collapses to length 0 and is strictly shorter than any
    // positive threshold.
    let q = MeshEdgeLengthQuery::new([2.0, -1.0, 4.0], [2.0, -1.0, 4.0], 3.0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1, "one edge in, one result out");
    assert!(
        close(got[0].length, 0.0),
        "zero-length edge: {}",
        got[0].length
    );
    assert!(
        got[0].shorter,
        "zero length is shorter than a positive threshold"
    );
    assert!(
        !got[0].longer,
        "zero length is not longer than a positive threshold"
    );
}

#[test]
fn at_threshold_edge_sets_neither_flag() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshEdgeLength::new(&ctx);
    // An edge of length exactly 4 (a clean power-of-two length computed without
    // rounding) classified against threshold 4: the strict `<` and `>` both
    // reject it, so neither flag is set on either side.
    let q = MeshEdgeLengthQuery::new([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 4.0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1, "one edge in, one result out");
    assert!(
        close(got[0].length, 4.0),
        "unit-scaled length: {}",
        got[0].length
    );
    assert!(
        !got[0].shorter && !got[0].longer,
        "an exactly-at-threshold edge sets neither flag: shorter {} longer {}",
        got[0].shorter,
        got[0].longer
    );
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshEdgeLength::new(&ctx);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut queries: Vec<MeshEdgeLengthQuery> = Vec::new();
    while queries.len() < 512 {
        let a = [coord(&mut state), coord(&mut state), coord(&mut state)];
        let b = [coord(&mut state), coord(&mut state), coord(&mut state)];
        let t = threshold(&mut state);
        let q = MeshEdgeLengthQuery::new(a, b, t);
        // Reject the near-tie band so the discrete flags never straddle a
        // last-place `sqrt` disagreement.
        if well_conditioned(&q) {
            queries.push(q);
        }
    }
    check(&ctx, &gpu, &queries);
}
