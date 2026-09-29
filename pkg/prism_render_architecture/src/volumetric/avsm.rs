//! Adaptive Volumetric Shadow Maps (`AVSM`) control-point compression and
//! `curve` queries (design section 6b).
//!
//! Cone shadow sampling (section 6) is cheap but under-samples thin clouds and
//! inter-layer self-shadowing. `AVSM` instead records, from the light's point
//! of view along one ray, the piecewise-linear `transmittance`-versus-`depth`
//! `curve` as a small set of `(depth, transmittance)` control points. When the
//! ray deposits more points than the fixed per-pixel budget allows, the
//! `curve` is *adaptively compressed*: the interior control point whose removal
//! perturbs the area under the `curve` the least (a Salvi-style area
//! minimisation) is dropped, preserving the sharp inflections that carry the
//! visual self-shadow while staying inside a fixed memory budget. Main-view
//! shading then queries this pre-integrated `curve` by `depth` to obtain light
//! visibility, so one integration serves many view samples and the expensive
//! secondary `raymarch` is avoided.
//!
//! `transmittance` is the fraction of light that survives from the light to a
//! given `depth`; it starts at `1` (nothing occluded yet) and decays
//! monotonically as optical thickness accumulates, so the recorded `curve` is
//! monotone non-increasing by construction. The `opacity` seen by a shaded
//! sample is `1 - transmittance`. On the `GPU` the `WESL` kernel stores the
//! same control points in a light-space texture (a deep-shadow / `AVSM`
//! `LUT`); this `CPU` reference is the deterministic, unit-testable path (the
//! sandbox has no `GPU`). The stylised `NPR` path quantises the same `curve`
//! into hard steps, so it consumes identical data.
//!
//! Everything here is pure and deterministic. The only transcendental used is
//! the shared [`super::math::exp_approx`] for Beer-Lambert decay; no `f32`
//! transcendental intrinsic is reached, so results are bit-reproducible.

use alloc::vec::Vec;

use super::math::{exp_approx, lerp, saturate};

/// One `AVSM` control point: the surviving light `transmittance` recorded at a
/// given light-space `depth`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AvsmNode {
    /// Light-space distance from the light at which this sample was taken.
    pub depth: f32,
    /// Fraction of light surviving to [`AvsmNode::depth`], always in `[0, 1]`
    /// and monotone non-increasing with `depth` across a valid `curve`.
    pub transmittance: f32,
}

impl AvsmNode {
    /// Builds a node, saturating `transmittance` into `[0, 1]` defensively.
    #[must_use]
    fn new(depth: f32, transmittance: f32) -> Self {
        Self {
            depth,
            transmittance: saturate(transmittance),
        }
    }
}

/// A single ray's adaptive `transmittance` `curve`: an ordered set of
/// `(depth, transmittance)` control points capped at a fixed budget.
///
/// Nodes are kept sorted by ascending `depth`. The running optical thickness is
/// carried so each [`AvsmCurve::insert`] extends the Beer-Lambert decay, which
/// keeps the stored `transmittance` sequence monotone non-increasing without a
/// separate sort of the values. When the node count exceeds
/// [`AvsmCurve::max_nodes`] the `curve` self-compresses.
#[derive(Clone, Debug, PartialEq)]
pub struct AvsmCurve {
    /// Control points in ascending `depth` order.
    nodes: Vec<AvsmNode>,
    /// Maximum control points retained after compression (at least two, so both
    /// `curve` endpoints always survive).
    max_nodes: usize,
    /// Accumulated optical thickness deposited so far, feeding the next node's
    /// Beer-Lambert `transmittance`.
    accumulated_optical_thickness: f32,
}

impl AvsmCurve {
    /// Creates an empty `curve` with the given control-point budget.
    ///
    /// A budget below two is raised to two: an `AVSM` `curve` needs both a near
    /// and a far endpoint, and compression never removes endpoints.
    #[must_use]
    pub fn new(max_nodes: usize) -> Self {
        Self {
            nodes: Vec::new(),
            max_nodes: if max_nodes < 2 { 2 } else { max_nodes },
            accumulated_optical_thickness: 0.0,
        }
    }

    /// The retained control-point budget (always at least two).
    #[must_use]
    pub fn max_nodes(&self) -> usize {
        self.max_nodes
    }

    /// The current control points in ascending `depth` order.
    #[must_use]
    pub fn nodes(&self) -> &[AvsmNode] {
        &self.nodes
    }

    /// Number of control points currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// `true` when the `curve` holds no control points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Records a segment ending at `depth` with optical thickness
    /// `extinction_segment`, then compresses if the budget is exceeded.
    ///
    /// `extinction_segment` is the optical thickness `integral of sigma_t ds`
    /// deposited between the previous node and this one; negative inputs are
    /// clamped to zero so the accumulated thickness never decreases and the
    /// `transmittance` never rises. The node is inserted at its sorted `depth`
    /// position (duplicate depths append after existing equals) so callers may
    /// feed samples in any order without panicking, though the physically
    /// meaningful ordering is front-to-back (non-decreasing `depth`). After the
    /// insert the stored values are forced monotone non-increasing and, if the
    /// node count exceeds [`AvsmCurve::max_nodes`], [`AvsmCurve::compress`]
    /// runs.
    pub fn insert(&mut self, depth: f32, extinction_segment: f32) {
        let segment = if extinction_segment > 0.0 {
            extinction_segment
        } else {
            0.0
        };
        self.accumulated_optical_thickness += segment;
        let transmittance = saturate(exp_approx(-self.accumulated_optical_thickness));
        let node = AvsmNode::new(depth, transmittance);

        let index = self.sorted_insert_index(depth);
        self.nodes.insert(index, node);
        self.enforce_monotone();

        if self.nodes.len() > self.max_nodes {
            self.compress();
        }
    }

    /// Compresses the `curve` down to the budget by repeatedly removing the
    /// interior control point whose removal perturbs the area under the `curve`
    /// the least.
    ///
    /// The perturbation of removing interior point `i` is exactly the area of
    /// the triangle formed by its neighbours `i - 1`, `i`, `i + 1`: replacing
    /// the two segments through `i` with the single chord `i-1 -> i+1` adds or
    /// removes precisely that triangle. Removing the minimum-area triangle each
    /// step is the Salvi-style adaptive merge that keeps the sharp inflections.
    /// Endpoints are never removed, so the near and far `depth` bounds are
    /// preserved, and the surviving `transmittance` sequence is re-forced
    /// monotone non-increasing.
    pub fn compress(&mut self) {
        while self.nodes.len() > self.max_nodes && self.nodes.len() > 2 {
            let mut best_index = 1_usize;
            let mut best_area = f32::INFINITY;
            let mut i = 1;
            while i + 1 < self.nodes.len() {
                let area = triangle_area(self.nodes[i - 1], self.nodes[i], self.nodes[i + 1]);
                if area < best_area {
                    best_area = area;
                    best_index = i;
                }
                i += 1;
            }
            self.nodes.remove(best_index);
        }
        self.enforce_monotone();
    }

    /// Samples the piecewise-linear `transmittance` `curve` at `depth`.
    ///
    /// An empty `curve` returns `1` (nothing occludes yet). A `depth` before
    /// the first node returns `1`; a `depth` at or after the last node returns
    /// the last node's `transmittance`. In between, the two bracketing nodes are
    /// linearly interpolated. The result is saturated into `[0, 1]`, and any
    /// `depth` (including non-finite-adjacent extremes) is handled without a
    /// panic.
    #[must_use]
    pub fn transmittance_at(&self, depth: f32) -> f32 {
        let count = self.nodes.len();
        if count == 0 {
            return 1.0;
        }
        let first = self.nodes[0];
        if depth < first.depth {
            return 1.0;
        }
        let last = self.nodes[count - 1];
        if depth >= last.depth {
            return saturate(last.transmittance);
        }
        // Locate the bracketing segment `[i, i + 1]`; `depth` is within the
        // interior span so the loop always finds one.
        let mut i = 0;
        while i + 1 < count {
            let lo = self.nodes[i];
            let hi = self.nodes[i + 1];
            if depth <= hi.depth {
                let span = hi.depth - lo.depth;
                let t = if span > super::math::EPS {
                    (depth - lo.depth) / span
                } else {
                    0.0
                };
                return saturate(lerp(lo.transmittance, hi.transmittance, t));
            }
            i += 1;
        }
        saturate(last.transmittance)
    }

    /// Area under the `transmittance` `curve` via the trapezoidal rule.
    ///
    /// Used as the compression error metric: comparing the area before and
    /// after [`AvsmCurve::compress`] bounds how much the adaptive merge shifted
    /// the `curve`. Fewer than two nodes enclose no area, so the result is `0`.
    #[must_use]
    pub fn area(&self) -> f32 {
        let count = self.nodes.len();
        if count < 2 {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut i = 0;
        while i + 1 < count {
            let lo = self.nodes[i];
            let hi = self.nodes[i + 1];
            let width = hi.depth - lo.depth;
            sum += 0.5 * (lo.transmittance + hi.transmittance) * width;
            i += 1;
        }
        sum
    }

    /// Returns the index at which a node of the given `depth` should be
    /// inserted to keep [`AvsmCurve::nodes`] ascending, placing equal depths
    /// after existing equal entries for a stable, deterministic order.
    #[must_use]
    fn sorted_insert_index(&self, depth: f32) -> usize {
        let mut i = 0;
        while i < self.nodes.len() {
            if depth < self.nodes[i].depth {
                break;
            }
            i += 1;
        }
        i
    }

    /// Forces the stored `transmittance` values monotone non-increasing in
    /// ascending `depth` order.
    ///
    /// In the physical front-to-back insertion order the values are already
    /// monotone (accumulated optical thickness only grows), so this is a no-op
    /// there; it makes the invariant hold unconditionally even if a caller
    /// feeds samples out of `depth` order, clamping each value to no more than
    /// its predecessor.
    fn enforce_monotone(&mut self) {
        let mut i = 1;
        while i < self.nodes.len() {
            let prev = self.nodes[i - 1].transmittance;
            if self.nodes[i].transmittance > prev {
                self.nodes[i].transmittance = prev;
            }
            i += 1;
        }
    }
}

/// Area of the triangle spanned by three control points, using `depth` as the
/// `x` axis and `transmittance` as the `y` axis.
///
/// This is the exact area perturbation of dropping the middle point `b` from a
/// piecewise-linear `curve` through `a`, `b`, `c`; the shoelace formula keeps
/// it free of any transcendental intrinsic.
#[must_use]
fn triangle_area(a: AvsmNode, b: AvsmNode, c: AvsmNode) -> f32 {
    let cross = a.depth * (b.transmittance - c.transmittance)
        + b.depth * (c.transmittance - a.transmittance)
        + c.depth * (a.transmittance - b.transmittance);
    0.5 * cross.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance used to compare computed areas in the tests.
    const AREA_TOL: f32 = 1e-4;

    /// Builds a `curve` from a `(depth, extinction)` script for the given
    /// budget, so tests share one deterministic construction path.
    fn build(max_nodes: usize, script: &[(f32, f32)]) -> AvsmCurve {
        let mut curve = AvsmCurve::new(max_nodes);
        for &(depth, extinction) in script {
            curve.insert(depth, extinction);
        }
        curve
    }

    #[test]
    fn empty_curve_queries_full_transmittance() {
        let curve = AvsmCurve::new(8);
        assert!(curve.is_empty());
        assert_eq!(curve.transmittance_at(-10.0), 1.0);
        assert_eq!(curve.transmittance_at(0.0), 1.0);
        assert_eq!(curve.transmittance_at(1234.5), 1.0);
        assert_eq!(curve.area(), 0.0);
    }

    #[test]
    fn transmittance_is_monotone_non_increasing() {
        let curve = build(
            16,
            &[
                (0.0, 0.0),
                (1.0, 0.3),
                (2.0, 0.4),
                (3.0, 0.2),
                (4.0, 0.5),
                (5.0, 0.6),
            ],
        );
        let mut prev = curve.transmittance_at(-1.0);
        let mut depth = -1.0;
        while depth <= 6.0 {
            let cur = curve.transmittance_at(depth);
            assert!(
                cur <= prev + super::super::math::EPS,
                "transmittance must be non-increasing at depth={depth}: {cur} > {prev}"
            );
            prev = cur;
            depth += 0.1;
        }
    }

    #[test]
    fn queries_stay_in_unit_range_and_do_not_panic_out_of_range() {
        let curve = build(8, &[(0.5, 0.0), (1.0, 0.7), (2.0, 0.9), (3.0, 1.1)]);
        // Far below the first node returns full transmittance.
        assert_eq!(curve.transmittance_at(-1_000.0), 1.0);
        // Far beyond the last node returns the last stored value.
        let last = curve.nodes()[curve.len() - 1].transmittance;
        assert_eq!(curve.transmittance_at(1_000.0), saturate(last));
        // Every sample across and beyond the range is a valid unit fraction.
        let mut depth = -5.0;
        while depth <= 8.0 {
            let v = curve.transmittance_at(depth);
            assert!(
                (0.0..=1.0).contains(&v),
                "transmittance {v} out of range at depth={depth}"
            );
            depth += 0.05;
        }
    }

    #[test]
    fn compression_bounds_node_count_and_area_error() {
        // A dense, smooth extinction script sampled front-to-back.
        let mut script = Vec::new();
        let mut depth = 0.0;
        while depth <= 24.0 {
            // Small per-segment optical thickness -> smooth decaying curve.
            script.push((depth, 0.12));
            depth += 1.0;
        }

        // Reference curve keeps every node; budgeted curve self-compresses.
        let full = build(256, &script);
        let budgeted = build(12, &script);

        assert!(full.len() > budgeted.max_nodes());
        assert!(
            budgeted.len() <= budgeted.max_nodes(),
            "compressed node count {} exceeds budget {}",
            budgeted.len(),
            budgeted.max_nodes()
        );

        // Endpoints are preserved by compression.
        assert_eq!(budgeted.nodes()[0].depth, full.nodes()[0].depth);
        assert_eq!(
            budgeted.nodes()[budgeted.len() - 1].depth,
            full.nodes()[full.len() - 1].depth
        );

        // The area error stays small for a smooth curve.
        let full_area = full.area();
        let budgeted_area = budgeted.area();
        assert!(full_area > 0.0);
        let rel_err = (full_area - budgeted_area).abs() / full_area;
        assert!(
            rel_err < 0.1,
            "compression area relative error {rel_err} too large ({budgeted_area} vs {full_area})"
        );

        // Compression preserves the monotone non-increasing invariant.
        let mut prev = 1.0;
        for node in budgeted.nodes() {
            assert!(
                node.transmittance <= prev + AREA_TOL,
                "compressed transmittance rose: {} > {prev}",
                node.transmittance
            );
            prev = node.transmittance;
        }
    }

    #[test]
    fn explicit_compress_is_idempotent_within_budget() {
        let mut curve = build(6, &[(0.0, 0.0), (1.0, 0.2), (2.0, 0.2), (3.0, 0.2)]);
        let before = curve.clone();
        // Already within budget: compression must not change anything.
        curve.compress();
        assert_eq!(curve, before);
        assert!((curve.area() - before.area()).abs() <= AREA_TOL);
    }

    #[test]
    fn insertion_is_deterministic() {
        let script = [
            (0.0, 0.0),
            (0.7, 0.25),
            (1.4, 0.4),
            (2.1, 0.15),
            (2.8, 0.5),
            (3.5, 0.35),
            (4.2, 0.6),
            (4.9, 0.2),
        ];
        let a = build(4, &script);
        let b = build(4, &script);
        assert_eq!(a, b);
        assert_eq!(a.nodes(), b.nodes());
        // Bit-identical node payloads.
        for (na, nb) in a.nodes().iter().zip(b.nodes()) {
            assert_eq!(na.depth.to_bits(), nb.depth.to_bits());
            assert_eq!(na.transmittance.to_bits(), nb.transmittance.to_bits());
        }
    }

    #[test]
    fn out_of_order_inserts_stay_monotone() {
        // Feed depths out of order; the monotone invariant must still hold.
        let curve = build(
            16,
            &[(3.0, 0.4), (1.0, 0.3), (5.0, 0.5), (2.0, 0.2), (4.0, 0.6)],
        );
        let mut prev = 1.0;
        for node in curve.nodes() {
            assert!(
                node.transmittance <= prev + super::super::math::EPS,
                "out-of-order insert broke monotonicity: {} > {prev}",
                node.transmittance
            );
            prev = node.transmittance;
        }
        // Queries remain valid unit fractions.
        let mut depth = 0.0;
        while depth <= 6.0 {
            let v = curve.transmittance_at(depth);
            assert!((0.0..=1.0).contains(&v));
            depth += 0.2;
        }
    }

    #[test]
    fn tiny_budget_is_raised_to_two() {
        let curve = AvsmCurve::new(0);
        assert_eq!(curve.max_nodes(), 2);
        let curve = build(1, &[(0.0, 0.0), (1.0, 0.5), (2.0, 0.5), (3.0, 0.5)]);
        assert!(curve.len() <= 2);
        // Endpoints survive the aggressive compression.
        assert_eq!(curve.nodes()[0].depth, 0.0);
        assert_eq!(curve.nodes()[curve.len() - 1].depth, 3.0);
    }
}
