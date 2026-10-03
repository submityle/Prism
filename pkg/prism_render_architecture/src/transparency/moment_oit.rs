//! Moment-Based Order-Independent Transparency (`MBOIT`), power-moment variant.
//!
//! This is the device-free numeric core behind the
//! [`TransparencyPath::MomentOit`](super::TransparencyPath::MomentOit) bucket.
//! It implements the power-moment form of Munstermann, Krause, Weyrich &
//! Thiedemann, *"Moment-Based Order-Independent Transparency"* (`PACMCGIT`
//! 2018): transparent fragments are summarized by the first four power moments
//! of their optical depth, and the transmittance in front of any depth is
//! reconstructed from those moments in a single resolve pass — no per-pixel
//! sorting and no fixed layer count.
//!
//! `MBOIT` is more accurate than
//! [`weighted_oit`](super::weighted_oit): where weighted-blended `OIT` collapses
//! all depth ordering into a single heuristic weight, `MBOIT` reconstructs an
//! actual (approximate) transmittance curve `T(z)`, so overlapping layers
//! occlude each other in roughly the right order. The two-node reconstruction
//! used here is **exact** for one or two transparent layers (it reproduces
//! ordinary back-to-front `src-over` compositing) and degrades gracefully for
//! three or more.
//!
//! # How it works
//!
//! Each fragment with coverage `alpha` contributes optical depth
//! `a = -ln(1 - alpha)` at a warped depth `z` (see [`warp_depth`]). The
//! generation pass accumulates
//!
//! ```text
//! b_0 = sum a_i                 (total optical depth)
//! b_k = sum a_i * z_i^k   k=1..4 (power moments)
//! ```
//!
//! Those moments define a non-negative measure on the depth axis. Its
//! two-point Gauss quadrature — nodes `z1, z2` and weights `w1, w2` with
//! `w1 + w2 = 1` — is recovered by solving the Hankel system for the monic
//! orthogonal polynomial `z^2 + p z + q` and reading off its roots. The optical
//! depth strictly in front of a query depth `z` is then
//!
//! ```text
//! front(z) = b_0 * ( w1 * [z1 < z] + w2 * [z2 < z] )
//! T(z)     = exp(-front(z))
//! ```
//!
//! and the pixel resolves to
//!
//! ```text
//! out = sum_i color_i * alpha_i * T(z_i) + background * exp(-b_0).
//! ```
//!
//! Every accumulated quantity is a sum, so the result is order independent, and
//! the reconstruction reproduces the input moments `m_0..m_3` exactly, which the
//! tests assert directly.

use super::weighted_oit::{OitFragment, Rgb};

/// Below this total optical depth a pixel is treated as having no transparent
/// coverage at all, so the background shows through unchanged.
const MIN_OPTICAL_DEPTH: f64 = 1e-9;
/// Below this Hankel determinant the measure is treated as a single Dirac
/// (one node), avoiding a divide-by-zero in the quadrature solve.
const MIN_VARIANCE: f64 = 1e-12;
/// Depth half-window within which a reconstruction node is considered to lie
/// *at* the query depth rather than strictly in front of or behind it.
const DEPTH_EPSILON: f64 = 1e-6;
/// Upper clamp on per-fragment optical depth so a fully opaque fragment
/// (`alpha == 1`, optical depth `+inf`) stays finite.
const MAX_FRAGMENT_OPTICAL_DEPTH: f64 = 42.0;

/// Map a positive view-space distance into the `[-1, 1]` range the moment
/// reconstruction is conditioned for, nearer surfaces mapping to smaller
/// values.
///
/// The mapping is logarithmic so that depth precision is spread across the
/// view frustum the way a reversed-Z buffer would. `view_z`, `near`, and `far`
/// are positive eye-space distances with `near < far`; values outside
/// `[near, far]` are clamped to the range endpoints.
pub fn warp_depth(view_z: f32, near: f32, far: f32) -> f32 {
    let near = (near as f64).max(1e-6);
    let far = (far as f64).max(near + 1e-6);
    let z = (view_z as f64).clamp(near, far);
    let ln_near = near.ln();
    let ln_far = far.ln();
    let t = (z.ln() - ln_near) / (ln_far - ln_near);
    (t * 2.0 - 1.0) as f32
}

/// Accumulator for the four power moments of a pixel's transparent fragments.
///
/// Fragments are added with an already-[`warp_depth`]ed depth. Accumulation is
/// commutative and associative, so fragments may be added in any order and
/// partial accumulators combined with [`merge`](MomentOitGenerator::merge).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MomentOitGenerator {
    /// `b_0..b_4`: total optical depth followed by its four power moments.
    b: [f64; 5],
}

impl Default for MomentOitGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl MomentOitGenerator {
    /// Create an empty generator (all moments zero).
    #[inline]
    pub const fn new() -> Self {
        Self { b: [0.0; 5] }
    }

    /// Accumulate one fragment given its warped depth in `[-1, 1]`.
    ///
    /// The alpha is clamped to `[0, 1]`; its optical depth
    /// `-ln(1 - alpha)` is clamped to a finite maximum so an opaque fragment
    /// does not poison the moments with an infinity.
    pub fn add_warped(&mut self, warped_depth: f32, alpha: f32) {
        let alpha = (alpha as f64).clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return;
        }
        let optical = (-(1.0 - alpha).ln()).min(MAX_FRAGMENT_OPTICAL_DEPTH);
        let z = warped_depth as f64;
        let z2 = z * z;
        self.b[0] += optical;
        self.b[1] += optical * z;
        self.b[2] += optical * z2;
        self.b[3] += optical * z2 * z;
        self.b[4] += optical * z2 * z2;
    }

    /// Merge another generator's moments into this one.
    #[inline]
    pub fn merge(&mut self, other: &MomentOitGenerator) {
        for i in 0..5 {
            self.b[i] += other.b[i];
        }
    }

    /// Total optical depth `b_0` accumulated so far.
    #[inline]
    pub fn total_optical_depth(&self) -> f32 {
        self.b[0] as f32
    }

    /// Build the transmittance reconstruction from the accumulated moments.
    ///
    /// Returns [`None`] when no meaningful coverage was accumulated, in which
    /// case the background is fully visible.
    pub fn reconstruct(&self) -> Option<MomentReconstruction> {
        MomentReconstruction::from_moments(&self.b)
    }
}

/// A two-node Gauss-quadrature reconstruction of the transmittance curve.
///
/// Produced by [`MomentOitGenerator::reconstruct`]. It stores the total optical
/// depth and up to two `(node, weight)` pairs that reproduce the pixel's
/// optical-depth measure; [`transmittance_in_front`](MomentReconstruction::transmittance_in_front)
/// evaluates `T(z)` from them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MomentReconstruction {
    /// Total optical depth `b_0`.
    total: f64,
    /// Reconstruction nodes (warped depths), ascending.
    nodes: [f64; 2],
    /// Weights for each node, summing to one.
    weights: [f64; 2],
    /// Number of distinct nodes actually used (1 or 2).
    node_count: u8,
}

impl MomentReconstruction {
    /// Solve the Hankel system for the two-node quadrature of `b_0..b_4`.
    fn from_moments(b: &[f64; 5]) -> Option<Self> {
        let total = b[0];
        if total < MIN_OPTICAL_DEPTH {
            return None;
        }
        // Normalized power moments m_1..m_3 (m_0 == 1 by construction).
        let m1 = b[1] / total;
        let m2 = b[2] / total;
        let m3 = b[3] / total;

        // Hankel determinant == variance of the depth measure.
        let det = m2 - m1 * m1;
        if det <= MIN_VARIANCE {
            // Degenerate measure: a single Dirac at the mean depth.
            return Some(Self {
                total,
                nodes: [m1, m1],
                weights: [1.0, 0.0],
                node_count: 1,
            });
        }

        // Monic orthogonal polynomial z^2 + p z + q (Cramer's rule on the
        // 2x2 Hankel system [[1, m1], [m1, m2]] [q; p]^T = [-m2; -m3]^T).
        let q = (-m2 * m2 + m1 * m3) / det;
        let p = (-m3 + m1 * m2) / det;

        let disc = (p * p - 4.0 * q).max(0.0);
        let sq = disc.sqrt();
        let z1 = 0.5 * (-p - sq);
        let z2 = 0.5 * (-p + sq);

        if (z2 - z1).abs() <= MIN_VARIANCE {
            return Some(Self {
                total,
                nodes: [m1, m1],
                weights: [1.0, 0.0],
                node_count: 1,
            });
        }

        // Weights from the first moment: w1 + w2 = 1, w1 z1 + w2 z2 = m1.
        let w1 = (m1 - z2) / (z1 - z2);
        let w2 = 1.0 - w1;

        Some(Self {
            total,
            nodes: [z1, z2],
            weights: [w1, w2],
            node_count: 2,
        })
    }

    /// Total optical depth `b_0` of the pixel.
    #[inline]
    pub fn total_optical_depth(&self) -> f32 {
        self.total as f32
    }

    /// Transmittance of everything the camera sees *before* `warped_depth`.
    ///
    /// `overestimation` in `[0, 1]` controls how a node sitting exactly at the
    /// query depth is counted: `0.0` excludes it (the usual choice when
    /// querying a fragment's own depth), `1.0` includes it fully.
    pub fn transmittance_in_front(&self, warped_depth: f32, overestimation: f32) -> f32 {
        let z = warped_depth as f64;
        let over = (overestimation as f64).clamp(0.0, 1.0);
        let mut fraction = 0.0;
        for i in 0..self.node_count as usize {
            let node = self.nodes[i];
            let coverage = if node < z - DEPTH_EPSILON {
                1.0
            } else if node > z + DEPTH_EPSILON {
                0.0
            } else {
                over
            };
            fraction += self.weights[i] * coverage;
        }
        (-self.total * fraction).exp() as f32
    }

    /// Net transmittance behind every transparent layer, i.e. how much of the
    /// background survives: `exp(-b_0)`.
    #[inline]
    pub fn background_transmittance(&self) -> f32 {
        (-self.total).exp() as f32
    }
}

/// Composite transparent fragments over a background using power-moment `MBOIT`.
///
/// `near`/`far` are the positive eye-space clip distances used to
/// [`warp_depth`] each fragment's view-space depth. Because every accumulated
/// quantity is a sum, the result is invariant to the order of `fragments`.
pub fn composite(fragments: &[OitFragment], near: f32, far: f32, background: Rgb) -> Rgb {
    let mut generator = MomentOitGenerator::new();
    for frag in fragments {
        let z = warp_depth(frag.depth, near, far);
        generator.add_warped(z, frag.alpha);
    }
    let Some(recon) = generator.reconstruct() else {
        return background;
    };

    let bg_t = recon.background_transmittance();
    let mut out = Rgb {
        r: background.r * bg_t,
        g: background.g * bg_t,
        b: background.b * bg_t,
    };
    for frag in fragments {
        let alpha = frag.alpha.clamp(0.0, 1.0);
        if alpha <= 0.0 {
            continue;
        }
        let z = warp_depth(frag.depth, near, far);
        // Exclude the fragment's own layer from what occludes it.
        let t = recon.transmittance_in_front(z, 0.0);
        let w = alpha * t;
        out.r += frag.color.r * w;
        out.g += frag.color.g * w;
        out.b += frag.color.b * w;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn rgb_approx(a: Rgb, b: Rgb, eps: f32) -> bool {
        approx(a.r, b.r, eps) && approx(a.g, b.g, eps) && approx(a.b, b.b, eps)
    }

    /// Reference back-to-front `src-over` compositing of depth-sorted layers.
    fn sorted_reference(fragments: &[OitFragment], background: Rgb) -> Rgb {
        let mut order: Vec<usize> = (0..fragments.len()).collect();
        // Farthest first so nearer layers composite last (on top).
        order.sort_by(|&a, &b| fragments[b].depth.partial_cmp(&fragments[a].depth).unwrap());
        let mut out = background;
        for &i in &order {
            let f = fragments[i];
            let a = f.alpha.clamp(0.0, 1.0);
            out = Rgb {
                r: f.color.r * a + out.r * (1.0 - a),
                g: f.color.g * a + out.g * (1.0 - a),
                b: f.color.b * a + out.b * (1.0 - a),
            };
        }
        out
    }

    #[test]
    fn empty_stack_shows_background() {
        let bg = Rgb::new(0.2, 0.4, 0.6);
        assert!(rgb_approx(composite(&[], 0.1, 100.0, bg), bg, 1e-6));
    }

    #[test]
    fn single_layer_matches_exact_alpha_compositing() {
        let bg = Rgb::new(0.1, 0.1, 0.1);
        for &alpha in &[0.1_f32, 0.25, 0.5, 0.75, 0.95] {
            for &depth in &[0.5_f32, 7.0, 50.0, 90.0] {
                let frag = OitFragment::new(Rgb::new(0.9, 0.3, 0.2), alpha, depth);
                let out = composite(&[frag], 0.1, 100.0, bg);
                let want = sorted_reference(&[frag], bg);
                assert!(
                    rgb_approx(out, want, 2e-3),
                    "alpha={alpha} depth={depth} out={out:?} want={want:?}",
                );
            }
        }
    }

    #[test]
    fn two_separated_layers_match_sorted_reference() {
        let bg = Rgb::new(0.05, 0.05, 0.1);
        let frags = [
            OitFragment::new(Rgb::new(0.9, 0.1, 0.1), 0.6, 5.0),
            OitFragment::new(Rgb::new(0.1, 0.2, 0.9), 0.4, 60.0),
        ];
        let out = composite(&frags, 0.1, 100.0, bg);
        let want = sorted_reference(&frags, bg);
        assert!(rgb_approx(out, want, 3e-3), "out={out:?} want={want:?}");
    }

    #[test]
    fn result_is_order_independent() {
        let bg = Rgb::new(0.05, 0.05, 0.2);
        let frags = [
            OitFragment::new(Rgb::new(0.9, 0.1, 0.1), 0.6, 3.0),
            OitFragment::new(Rgb::new(0.1, 0.9, 0.1), 0.4, 12.0),
            OitFragment::new(Rgb::new(0.1, 0.1, 0.9), 0.8, 1.5),
            OitFragment::new(Rgb::new(0.8, 0.8, 0.2), 0.3, 42.0),
        ];
        let base = composite(&frags, 0.1, 100.0, bg);
        let mut rev = frags;
        rev.reverse();
        assert!(rgb_approx(base, composite(&rev, 0.1, 100.0, bg), 1e-5));
        let sh = [frags[2], frags[0], frags[3], frags[1]];
        assert!(rgb_approx(base, composite(&sh, 0.1, 100.0, bg), 1e-5));
    }

    #[test]
    fn transmittance_is_monotone_non_increasing() {
        let mut g = MomentOitGenerator::new();
        g.add_warped(warp_depth(5.0, 0.1, 100.0), 0.5);
        g.add_warped(warp_depth(40.0, 0.1, 100.0), 0.5);
        let r = g.reconstruct().unwrap();
        let mut prev = 1.0_f32;
        let mut z = -1.0_f32;
        while z <= 1.0 {
            let t = r.transmittance_in_front(z, 1.0);
            assert!(t <= prev + 1e-5, "z={z} t={t} prev={prev}");
            assert!((0.0..=1.0).contains(&t), "t out of range: {t}");
            prev = t;
            z += 0.05;
        }
    }

    #[test]
    fn quadrature_reproduces_input_moments() {
        // The two-node reconstruction must match m_0..m_3 of the measure.
        let mut g = MomentOitGenerator::new();
        let samples = [
            (warp_depth(3.0, 0.1, 100.0), 0.5_f32),
            (warp_depth(20.0, 0.1, 100.0), 0.3),
            (warp_depth(70.0, 0.1, 100.0), 0.7),
        ];
        for &(z, a) in &samples {
            g.add_warped(z, a);
        }
        let r = g.reconstruct().unwrap();
        let total = r.total;
        // Rebuild moments from the recovered nodes/weights.
        let mut m = [0.0_f64; 4];
        for i in 0..r.node_count as usize {
            let z = r.nodes[i];
            let w = r.weights[i];
            m[0] += w;
            m[1] += w * z;
            m[2] += w * z * z;
            m[3] += w * z * z * z;
        }
        // Expected normalized moments from the raw accumulator.
        let em1 = g.b[1] / total;
        let em2 = g.b[2] / total;
        let em3 = g.b[3] / total;
        assert!((m[0] - 1.0).abs() < 1e-9, "m0={}", m[0]);
        assert!((m[1] - em1).abs() < 1e-9, "m1 {} vs {em1}", m[1]);
        assert!((m[2] - em2).abs() < 1e-9, "m2 {} vs {em2}", m[2]);
        assert!((m[3] - em3).abs() < 1e-9, "m3 {} vs {em3}", m[3]);
    }

    #[test]
    fn merge_matches_single_generator() {
        let a = [
            (warp_depth(4.0, 0.1, 100.0), 0.5_f32),
            (warp_depth(20.0, 0.1, 100.0), 0.35),
        ];
        let b = [
            (warp_depth(2.0, 0.1, 100.0), 0.6_f32),
            (warp_depth(60.0, 0.1, 100.0), 0.2),
        ];
        let mut whole = MomentOitGenerator::new();
        for &(z, al) in a.iter().chain(b.iter()) {
            whole.add_warped(z, al);
        }
        let mut pa = MomentOitGenerator::new();
        for &(z, al) in &a {
            pa.add_warped(z, al);
        }
        let mut pb = MomentOitGenerator::new();
        for &(z, al) in &b {
            pb.add_warped(z, al);
        }
        pa.merge(&pb);
        // Merging accumulates the moments in a different summation grouping
        // than the single-pass generator, so the two results agree only up to
        // floating-point round-off (last-ULP differences in the higher
        // moments). Compare moment-by-moment within a tight tolerance rather
        // than requiring a bit-exact struct equality.
        for (w, m) in whole.b.iter().zip(pa.b.iter()) {
            assert!(
                (w - m).abs() < 1.0e-9,
                "merged moments diverge: whole={w} merged={m}"
            );
        }
    }

    #[test]
    fn mboit_beats_weighted_for_three_layers() {
        // Against the sorted ground truth, MBOIT should be at least as close
        // as weighted-blended OIT for a modest overlapping stack.
        use super::super::weighted_oit::{self, WeightFunction};
        let bg = Rgb::new(0.1, 0.1, 0.1);
        let frags = [
            OitFragment::new(Rgb::new(0.9, 0.1, 0.1), 0.5, 5.0),
            OitFragment::new(Rgb::new(0.1, 0.9, 0.1), 0.5, 25.0),
            OitFragment::new(Rgb::new(0.1, 0.1, 0.9), 0.5, 55.0),
        ];
        let truth = sorted_reference(&frags, bg);
        let m = composite(&frags, 0.1, 100.0, bg);
        let w = weighted_oit::composite(&frags, WeightFunction::Eq9, bg);
        let err = |o: Rgb| (o.r - truth.r).abs() + (o.g - truth.g).abs() + (o.b - truth.b).abs();
        assert!(
            err(m) <= err(w) + 1e-4,
            "mboit_err={} weighted_err={}",
            err(m),
            err(w),
        );
    }
}
