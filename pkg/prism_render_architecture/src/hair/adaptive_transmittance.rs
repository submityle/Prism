//! Adaptive variable-node transmittance compression for hair self-shadowing
//! (design doc §8.5 item7): a deep shadow map / adaptive transmittance-function
//! back end.
//!
//! The fixed-layer deep opacity map in [`super::deep_transmittance`] slices the
//! light-space depth range into a *constant* number of equal-width layers. That
//! is the baseline the §5 transmittance service runs on, and it is cheap and
//! upload-friendly, but equal-width layers spend the same storage on flat
//! regions (long runs where transmittance barely changes) as on the sharp drops
//! where a dense clump of strands occludes the light. Hair self-shadowing is
//! exactly that kind of signal: mostly flat with a few steep steps.
//!
//! The deep shadow map of Lokovic & Veach (2000), with Salvi's adaptive
//! transmittance-storage variant, instead stores the transmittance as a
//! *piecewise-linear curve with a variable number of control nodes* and
//! compresses it by dropping interior nodes that lie within an error tolerance
//! of the line between their neighbours. The budget then follows the signal:
//! flat runs collapse to two nodes, steep steps keep theirs. This module is the
//! deterministic `CPU` golden contract for that compression — the optional
//! higher-fidelity back end the §5 service can select instead of the fixed-layer
//! slab, feeding the same shadow / `OIT` route (`crate::transparency`'s
//! `HairVisibility` path) that the baseline does.
//!
//! Everything here is array-in / array-out, stable under input reordering,
//! golden-comparable, and panic-free on empty or out-of-range input. Like the
//! sibling transmittance paths it is **exponential-free**: the optical
//! transmittance of a stack of independent `alpha`-blended occluders is the
//! multiplicative composite `T = product(1 - alpha_i)`, computed directly, so no
//! Beer-Lambert `exp` is ever evaluated and the goldens are bit-stable. The
//! device-side build that would produce such a curve on the `GPU` is separate
//! scheduling and is not owned here.

use alloc::vec::Vec;

/// Depths closer together than this (in light space) are treated as the same
/// node when accumulating, so near-coincident samples compose into one control
/// point instead of a degenerate zero-width segment.
const DEPTH_EPS: f32 = 1e-6;

/// One strand sample projected onto a single light ray/texel: its light-space
/// `depth` and the `alpha` coverage it occludes there.
///
/// `alpha` is in `0..=1` (`0` = fully transparent, `1` = fully opaque) and
/// `depth` is a non-negative distance along the light ray. [`TransmittanceSample::new`]
/// clamps both invariants and maps non-finite inputs to safe values; the
/// accumulation path re-sanitises through `new` as well, so a hand-built sample
/// with public fields set out of range can never drive transmittance outside
/// `0..=1` or panic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransmittanceSample {
    /// Light-space depth of the sample; larger is farther from the light.
    pub depth: f32,
    /// Opacity contribution in `0..=1`.
    pub alpha: f32,
}

impl TransmittanceSample {
    /// Builds a sample, clamping `depth` to `>= 0` and `alpha` to `0..=1`, and
    /// mapping any non-finite component to `0`.
    #[must_use]
    pub fn new(depth: f32, alpha: f32) -> Self {
        Self {
            depth: if depth.is_finite() {
                depth.max(0.0)
            } else {
                0.0
            },
            alpha: if alpha.is_finite() {
                alpha.clamp(0.0, 1.0)
            } else {
                0.0
            },
        }
    }
}

/// One control node of a transmittance curve: the surviving light fraction
/// [`Self::transmittance`] at light-space [`Self::depth`].
///
/// Along any curve the nodes are ordered by non-decreasing `depth` and
/// non-increasing `transmittance` (each occluder can only remove light), and
/// `transmittance` always lies in `0..=1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransmittanceNode {
    /// Light-space depth of this control point.
    pub depth: f32,
    /// Surviving transmittance `T` at this depth, in `0..=1`.
    pub transmittance: f32,
}

/// The full (uncompressed) transmittance curve: one node per distinct sample
/// depth, with the cumulative composite `T = product(1 - alpha)` of every sample
/// at or in front of that depth.
///
/// This is the reference signal the adaptive compressor approximates. The node
/// list is monotone non-increasing in transmittance and sorted by depth; an
/// empty curve means "no occluders" and reads fully transmissive (`1.0`)
/// everywhere.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TransmittanceCurve {
    /// Control nodes, sorted by depth, transmittance non-increasing.
    pub nodes: Vec<TransmittanceNode>,
}

impl TransmittanceCurve {
    /// Number of control nodes in the curve.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when the curve has no nodes (fully transmissive).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Samples the transmittance a receiver at `depth` sees through this curve;
    /// see [`sample_nodes_shadow`] for the exact lookup rule.
    #[must_use]
    pub fn sample(&self, depth: f32) -> f32 {
        sample_nodes_shadow(&self.nodes, depth)
    }
}

/// An adaptively compressed transmittance curve: a variable number of
/// control nodes whose piecewise-linear reconstruction stays within the
/// requested tolerance of the full [`TransmittanceCurve`].
///
/// The two endpoints (shallowest and deepest nodes) are always preserved, so
/// the compressed curve covers the same depth span as the source. Nodes remain
/// sorted by depth with non-increasing transmittance in `0..=1`. An empty curve
/// reads fully transmissive (`1.0`) everywhere.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompressedCurve {
    /// Surviving control nodes, sorted by depth, transmittance non-increasing.
    pub nodes: Vec<TransmittanceNode>,
}

impl CompressedCurve {
    /// Number of control nodes retained after compression.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when the curve has no nodes (fully transmissive).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Accumulates strand samples on one light ray/texel into a full transmittance
/// curve.
///
/// `samples` may arrive in any order; they are re-sanitised through
/// [`TransmittanceSample::new`] and stably sorted by depth, so the result is
/// independent of arrival order. Samples within [`DEPTH_EPS`] of one another in
/// depth compose into a single node. Each node stores the running composite
/// `T = product(1 - alpha)` of every sample at or in front of it, which is
/// monotone non-increasing and always in `0..=1`. Empty input yields an empty
/// (fully transmissive) curve. Never panics.
#[must_use]
pub fn accumulate(samples: &[TransmittanceSample]) -> TransmittanceCurve {
    if samples.is_empty() {
        return TransmittanceCurve { nodes: Vec::new() };
    }

    // Re-sanitise so struct-literal samples with out-of-range public fields are
    // clamped before they can touch the composite, then stable-sort by depth.
    let mut sorted: Vec<TransmittanceSample> = samples
        .iter()
        .map(|s| TransmittanceSample::new(s.depth, s.alpha))
        .collect();
    sorted.sort_by(|a, b| a.depth.total_cmp(&b.depth));

    let mut nodes: Vec<TransmittanceNode> = Vec::new();
    let mut running = 1.0_f32;
    let mut i = 0usize;
    while i < sorted.len() {
        let depth = sorted[i].depth;
        // Fold every sample sharing this depth (within DEPTH_EPS) into one node.
        while i < sorted.len() && (sorted[i].depth - depth).abs() < DEPTH_EPS {
            running *= 1.0 - sorted[i].alpha;
            i += 1;
        }
        running = running.clamp(0.0, 1.0);
        nodes.push(TransmittanceNode {
            depth,
            transmittance: running,
        });
    }

    TransmittanceCurve { nodes }
}

/// Linearly interpolates the transmittance of a node polyline at `depth`,
/// clamping to the end node values outside the node span.
///
/// This is the plain piecewise-linear reconstruction used as the compression
/// error metric: between two nodes it lerps, and before the first / after the
/// last node it holds the respective endpoint value. It deliberately does *not*
/// apply the "fully lit in front of the first node" shadow rule — that belongs
/// to receiver queries ([`sample_nodes_shadow`]), whereas the compressor only
/// needs to measure how well the retained polyline reproduces the source nodes.
fn interp_nodes(nodes: &[TransmittanceNode], depth: f32) -> f32 {
    if nodes.is_empty() {
        return 1.0;
    }
    let first = nodes[0];
    if depth <= first.depth {
        return first.transmittance;
    }
    let last = nodes[nodes.len() - 1];
    if depth >= last.depth {
        return last.transmittance;
    }
    for pair in nodes.windows(2) {
        if depth <= pair[1].depth {
            let span = pair[1].depth - pair[0].depth;
            let t = if span > 0.0 {
                (depth - pair[0].depth) / span
            } else {
                0.0
            };
            return pair[0].transmittance + (pair[1].transmittance - pair[0].transmittance) * t;
        }
    }
    last.transmittance
}

/// Samples the transmittance a receiver at `depth` sees through a node curve.
///
/// Receivers in front of the frontmost node are fully lit (`1.0`); receivers at
/// or beyond the deepest node take the last (most occluded) value; in between,
/// the two bracketing nodes are linearly interpolated. The result is monotone
/// non-increasing in `depth` and always in `0..=1`. An empty curve returns
/// `1.0`. This matches the receiver-query convention of
/// [`super::deep_transmittance::sample_transmittance`] so the adaptive back end
/// decodes under the same rule as the fixed-layer baseline. Never panics.
fn sample_nodes_shadow(nodes: &[TransmittanceNode], depth: f32) -> f32 {
    if nodes.is_empty() {
        return 1.0;
    }
    if depth <= nodes[0].depth {
        return 1.0;
    }
    let last = nodes.len() - 1;
    if depth >= nodes[last].depth {
        return nodes[last].transmittance;
    }
    for pair in nodes.windows(2) {
        if depth <= pair[1].depth {
            let span = pair[1].depth - pair[0].depth;
            let t = if span > 0.0 {
                (depth - pair[0].depth) / span
            } else {
                0.0
            };
            return pair[0].transmittance + (pair[1].transmittance - pair[0].transmittance) * t;
        }
    }
    nodes[last].transmittance
}

/// Maximum reconstruction error over every original node when the node at
/// position `pos` in `kept` is dropped.
///
/// Builds the candidate node polyline from `kept` minus `kept[pos]`, then
/// returns the largest absolute difference between each original node's
/// transmittance and the candidate polyline sampled at that node's depth. The
/// error is always measured against the *original* curve (not the already-thinned
/// one) so the greedy loop bounds total drift, not just the drift of a single
/// step.
fn max_error_without(original: &[TransmittanceNode], kept: &[usize], pos: usize) -> f32 {
    let mut candidate: Vec<TransmittanceNode> = Vec::with_capacity(kept.len() - 1);
    for (j, &idx) in kept.iter().enumerate() {
        if j == pos {
            continue;
        }
        candidate.push(original[idx]);
    }

    let mut max_err = 0.0_f32;
    for node in original {
        let approx = interp_nodes(&candidate, node.depth);
        let err = (node.transmittance - approx).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

/// Compresses strand samples into an adaptive variable-node transmittance curve.
///
/// The samples are first accumulated into the full [`TransmittanceCurve`], then
/// greedily thinned: each pass drops the single interior node whose removal adds
/// the least reconstruction error (measured against the full curve). A node is
/// dropped when either that error is within `tol`, or the curve still exceeds
/// `max_nodes` and must shed a node regardless. The two endpoints are never
/// dropped, so the compressed curve keeps the full depth span, stays monotone
/// non-increasing, and reconstructs within `tol` of the source whenever the
/// `max_nodes` budget is not the binding limit.
///
/// `tol` is sanitised to a finite value `>= 0` (non-finite becomes `0`, i.e.
/// lossless). `max_nodes` is clamped up to `2` (the endpoints are mandatory);
/// a source curve of two or fewer nodes is returned unchanged. Empty input
/// yields an empty (fully transmissive) curve. Never panics.
#[must_use]
pub fn compress_adaptive(
    samples: &[TransmittanceSample],
    max_nodes: usize,
    tol: f32,
) -> CompressedCurve {
    let original = accumulate(samples).nodes;
    if original.len() <= 2 {
        // Nothing to drop: endpoints are mandatory and there are no interior
        // nodes (this also covers the empty and single-node cases).
        return CompressedCurve { nodes: original };
    }

    let tol = if tol.is_finite() { tol.max(0.0) } else { 0.0 };
    let max_nodes = max_nodes.max(2);

    let mut kept: Vec<usize> = (0..original.len()).collect();
    while kept.len() > 2 {
        // Pick the interior node whose removal perturbs the curve the least.
        let mut best_pos: Option<usize> = None;
        let mut best_err = f32::INFINITY;
        for pos in 1..kept.len() - 1 {
            let err = max_error_without(&original, &kept, pos);
            if err < best_err {
                best_err = err;
                best_pos = Some(pos);
            }
        }

        let Some(pos) = best_pos else { break };
        let over_budget = kept.len() > max_nodes;
        if best_err <= tol || over_budget {
            kept.remove(pos);
        } else {
            break;
        }
    }

    let nodes = kept.iter().map(|&i| original[i]).collect();
    CompressedCurve { nodes }
}

/// Samples the transmittance a receiver at `depth` sees through a compressed
/// curve.
///
/// Applies the receiver-query rule of [`sample_nodes_shadow`]: fully lit
/// (`1.0`) in front of the frontmost node, the last value at or beyond the
/// deepest node, and a linear interpolation of the two bracketing nodes in
/// between. Monotone non-increasing in `depth`, always in `0..=1`, and
/// panic-free on an empty curve or out-of-range depth.
#[must_use]
pub fn sample_transmittance(curve: &CompressedCurve, depth: f32) -> f32 {
    sample_nodes_shadow(&curve.nodes, depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn accumulate_is_monotone_in_range_and_golden() {
        // Two alpha-0.5 occluders at depth 0 and 10: nodes (0, 0.5), (10, 0.25).
        let curve = accumulate(&[
            TransmittanceSample::new(0.0, 0.5),
            TransmittanceSample::new(10.0, 0.5),
        ]);
        assert_eq!(curve.node_count(), 2);
        assert!(close(curve.nodes[0].depth, 0.0));
        assert!(close(curve.nodes[0].transmittance, 0.5));
        assert!(close(curve.nodes[1].depth, 10.0));
        assert!(close(curve.nodes[1].transmittance, 0.25));

        // Monotone non-increasing and bounded over a longer run.
        let curve = accumulate(&[
            TransmittanceSample::new(0.0, 0.2),
            TransmittanceSample::new(1.0, 0.3),
            TransmittanceSample::new(2.0, 0.4),
            TransmittanceSample::new(3.0, 0.5),
        ]);
        let mut prev = 1.0_f32;
        for node in &curve.nodes {
            assert!(node.transmittance <= prev + EPS);
            assert!((0.0..=1.0).contains(&node.transmittance));
            prev = node.transmittance;
        }
    }

    #[test]
    fn coincident_depths_compose_into_one_node() {
        // Three samples at (near-)identical depth collapse to a single node that
        // composites all three: (1-0.5)(1-0.5)(1-0.5) = 0.125.
        let curve = accumulate(&[
            TransmittanceSample::new(2.0, 0.5),
            TransmittanceSample::new(2.0, 0.5),
            TransmittanceSample::new(2.0 + 1e-9, 0.5),
        ]);
        assert_eq!(curve.node_count(), 1);
        assert!(close(curve.nodes[0].transmittance, 0.125));
    }

    #[test]
    fn empty_samples_are_fully_transmissive() {
        let curve = accumulate(&[]);
        assert!(curve.is_empty());
        assert!(close(curve.sample(0.0), 1.0));
        assert!(close(curve.sample(123.0), 1.0));

        let compressed = compress_adaptive(&[], 8, 1e-4);
        assert!(compressed.is_empty());
        assert!(close(sample_transmittance(&compressed, 0.0), 1.0));
        assert!(close(sample_transmittance(&compressed, 9.0), 1.0));
    }

    #[test]
    fn sample_boundaries_and_midpoint_golden() {
        let compressed = compress_adaptive(
            &[
                TransmittanceSample::new(0.0, 0.5),
                TransmittanceSample::new(10.0, 0.5),
            ],
            8,
            0.0,
        );
        // In front of the frontmost node -> fully lit.
        assert!(close(sample_transmittance(&compressed, -1.0), 1.0));
        assert!(close(sample_transmittance(&compressed, 0.0), 1.0));
        // At / behind the last node -> last value.
        assert!(close(sample_transmittance(&compressed, 10.0), 0.25));
        assert!(close(sample_transmittance(&compressed, 50.0), 0.25));
        // Midpoint depth 5 between nodes (0, 0.5) and (10, 0.25): lerp = 0.375.
        assert!(close(sample_transmittance(&compressed, 5.0), 0.375));
    }

    #[test]
    fn compress_collapses_flat_curve_to_endpoints() {
        // All-transparent samples give a constant T=1 curve: every interior node
        // is exactly collinear and must collapse to the two endpoints.
        let samples: Vec<TransmittanceSample> = (0..6)
            .map(|i| TransmittanceSample::new(i as f32, 0.0))
            .collect();
        let full = accumulate(&samples);
        assert_eq!(full.node_count(), 6);
        let compressed = compress_adaptive(&samples, 16, 1e-6);
        assert_eq!(compressed.node_count(), 2);
        assert!(close(compressed.nodes[0].depth, 0.0));
        assert!(close(compressed.nodes[1].depth, 5.0));
        for &n in &compressed.nodes {
            assert!(close(n.transmittance, 1.0));
        }
    }

    #[test]
    fn compress_keeps_sharp_step_and_drops_collinear_node() {
        // Flat, a sharp drop at depth 2, then flat again. The pre-step node and
        // the step node are not collinear and survive; the trailing flat node is
        // collinear and is dropped.
        let samples = [
            TransmittanceSample::new(0.0, 0.0),
            TransmittanceSample::new(1.0, 0.0),
            TransmittanceSample::new(2.0, 0.9),
            TransmittanceSample::new(3.0, 0.0),
            TransmittanceSample::new(4.0, 0.0),
        ];
        let full = accumulate(&samples);
        assert_eq!(full.node_count(), 5);

        let compressed = compress_adaptive(&samples, 16, 1e-4);
        assert_eq!(compressed.node_count(), 4);
        let depths: Vec<f32> = compressed.nodes.iter().map(|n| n.depth).collect();
        assert!(close(depths[0], 0.0));
        assert!(close(depths[1], 1.0));
        assert!(close(depths[2], 2.0));
        assert!(close(depths[3], 4.0));
        // Endpoints preserve the source transmittance exactly.
        assert!(close(compressed.nodes[0].transmittance, 1.0));
        assert!(close(compressed.nodes[3].transmittance, 0.1));
    }

    #[test]
    fn compress_respects_max_nodes_cap() {
        let samples = [
            TransmittanceSample::new(0.0, 0.0),
            TransmittanceSample::new(1.0, 0.0),
            TransmittanceSample::new(2.0, 0.9),
            TransmittanceSample::new(3.0, 0.0),
            TransmittanceSample::new(4.0, 0.0),
        ];
        // A tight tolerance would keep 4 nodes, but max_nodes forces 3.
        let compressed = compress_adaptive(&samples, 3, 1e-4);
        assert_eq!(compressed.node_count(), 3);
        // Endpoints are always retained.
        assert!(close(compressed.nodes[0].depth, 0.0));
        assert!(close(
            compressed.nodes[compressed.node_count() - 1].depth,
            4.0
        ));

        // max_nodes below 2 is clamped to the mandatory endpoints.
        let two = compress_adaptive(&samples, 0, 0.0);
        assert!(two.node_count() >= 2);
    }

    #[test]
    fn compressed_curve_reconstructs_within_tolerance() {
        let samples = [
            TransmittanceSample::new(0.0, 0.1),
            TransmittanceSample::new(1.0, 0.2),
            TransmittanceSample::new(2.0, 0.6),
            TransmittanceSample::new(3.0, 0.2),
            TransmittanceSample::new(4.0, 0.3),
            TransmittanceSample::new(5.0, 0.1),
        ];
        let tol = 0.1_f32;
        let full = accumulate(&samples);
        let compressed = compress_adaptive(&samples, 64, tol);

        // Every original node is reproduced within tol by the compressed curve.
        for node in &full.nodes {
            let approx = interp_nodes(&compressed.nodes, node.depth);
            assert!(
                (node.transmittance - approx).abs() <= tol + EPS,
                "node at depth {} drifted: {} vs {}",
                node.depth,
                node.transmittance,
                approx,
            );
        }
        // Compression actually removed nodes without exceeding the source span.
        assert!(compressed.node_count() <= full.node_count());
        assert!(close(compressed.nodes[0].depth, 0.0));
        assert!(close(
            compressed.nodes[compressed.node_count() - 1].depth,
            5.0
        ));
    }

    #[test]
    fn compressed_curve_is_monotone_non_increasing() {
        let samples = [
            TransmittanceSample::new(0.0, 0.3),
            TransmittanceSample::new(1.5, 0.5),
            TransmittanceSample::new(3.0, 0.2),
            TransmittanceSample::new(4.5, 0.7),
            TransmittanceSample::new(6.0, 0.4),
        ];
        let compressed = compress_adaptive(&samples, 8, 0.05);
        for pair in compressed.nodes.windows(2) {
            assert!(pair[1].depth >= pair[0].depth - EPS);
            assert!(pair[1].transmittance <= pair[0].transmittance + EPS);
        }
        // Sampling is monotone non-increasing and bounded across the span.
        let mut prev = 1.0_f32;
        for step in 0..30 {
            let d = step as f32 * 0.25;
            let t = sample_transmittance(&compressed, d);
            assert!(t <= prev + EPS, "rose at depth {d}: {prev} -> {t}");
            assert!((0.0..=1.0).contains(&t));
            prev = t;
        }
    }

    #[test]
    fn accumulate_is_order_independent() {
        let ordered = [
            TransmittanceSample::new(0.0, 0.2),
            TransmittanceSample::new(2.0, 0.4),
            TransmittanceSample::new(5.0, 0.3),
            TransmittanceSample::new(8.0, 0.6),
        ];
        let shuffled = [ordered[3], ordered[0], ordered[2], ordered[1]];
        let a = accumulate(&ordered);
        let b = accumulate(&shuffled);
        assert_eq!(a.node_count(), b.node_count());
        for (x, y) in a.nodes.iter().zip(&b.nodes) {
            assert!(close(x.depth, y.depth));
            assert!(close(x.transmittance, y.transmittance));
        }
    }

    #[test]
    fn non_finite_and_negative_inputs_are_sanitised() {
        let samples = [
            TransmittanceSample::new(f32::NAN, 0.5),
            TransmittanceSample::new(-4.0, 2.0),
            TransmittanceSample::new(f32::INFINITY, f32::NEG_INFINITY),
            TransmittanceSample::new(3.0, -1.0),
        ];
        let curve = accumulate(&samples);
        for node in &curve.nodes {
            assert!(node.depth.is_finite() && node.depth >= 0.0);
            assert!((0.0..=1.0).contains(&node.transmittance));
        }
        // Compression tolerates non-finite tol (treated as lossless) and never
        // panics on out-of-range queries.
        let compressed = compress_adaptive(&samples, 8, f32::NAN);
        for node in &compressed.nodes {
            assert!((0.0..=1.0).contains(&node.transmittance));
        }
        assert!(close(sample_transmittance(&compressed, -100.0), 1.0));
        let far = sample_transmittance(&compressed, 1e9);
        assert!((0.0..=1.0).contains(&far));
    }

    #[test]
    fn single_node_curve_queries_do_not_panic() {
        let compressed = compress_adaptive(&[TransmittanceSample::new(2.0, 0.5)], 8, 0.0);
        assert_eq!(compressed.node_count(), 1);
        // At/in front of the sole node -> lit (the frontmost-node convention
        // from `sample_boundaries_and_midpoint_golden`); strictly behind it ->
        // the node value.
        assert!(close(sample_transmittance(&compressed, 0.0), 1.0));
        assert!(close(sample_transmittance(&compressed, 2.0), 1.0));
        assert!(close(sample_transmittance(&compressed, 100.0), 0.5));
    }
}
