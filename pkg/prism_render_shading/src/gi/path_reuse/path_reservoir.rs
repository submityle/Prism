//! Path reservoirs and reconnection-shift GRIS resampling — CPU golden.
//!
//! This is the resampling backend of path-space `ReSTIR` (Lin et al. 2022,
//! *ReSTIR PT* / *Generalized Resampled Importance Sampling*).  It reuses the
//! screen-probe [`Reservoir`] verbatim — the payload is the shared [`GiSample`]
//! and the scalar target is [`target_function`] — but combines neighbours
//! through the *reconnection shift map*: a neighbour's reusable path tail is
//! re-anchored to the destination pixel's primary vertex and its contribution
//! is corrected by the shift Jacobian from [`super::shift_map`].
//!
//! The pipeline mirrors a frame of path-space `ReSTIR`:
//!
//! * **Fill** — [`stream_path`] folds one freshly traced path suffix into a
//!   reservoir with the standard `RIS` weight `target / source_pdf`.
//! * **Finalize** — [`finalize_path`] converts the accumulated state into the
//!   unbiased contribution weight `W` using the selected sample's own target.
//! * **Reuse** — [`merge_path`] shifts a neighbour's selected path to the
//!   destination primary vertex via [`shift_mapped_target`] and feeds the
//!   `GRIS` induced weight `m * p_hat * W * |J|` into [`Reservoir::merge`].
//!   [`pairwise_path_mis_weight`] exposes the balance-heuristic MIS weight for
//!   schemes that weight neighbours explicitly.
//!
//! # Conventions
//! * The induced reuse weight folds the shift Jacobian `|J|` into the target
//!   density handed to [`Reservoir::merge`]: passing `p_hat * |J|` makes the
//!   merge's internal `other.m * pdf * other.W` equal the `GRIS` weight
//!   `m * p_hat * W * |J|` exactly.
//! * Every weight is clamped finite and non-negative; a degenerate shift falls
//!   back to the identity Jacobian (see [`super::shift_map`]) so reuse never
//!   injects `NaN` energy and a same-pixel reuse reduces to plain `GRIS`.
//! * Randomness enters only as caller-supplied uniforms `u in [0, 1)`.  Every
//!   function is a deterministic pure function: no RNG, no I/O, no GPU, no
//!   global state, and no `unsafe`.  State is `f32`-packed to match the GPU
//!   reservoir-buffer twin.

use bevy_math::Vec3;

use crate::gi::screen_probe::restir::{balance_heuristic, target_function, GiSample, Reservoir};

use super::shift_map::reconnection_shift;
use super::vertex::{PathSuffix, PathVertex};

/// Resampled-importance weight of a freshly traced path: `target / source_pdf`.
///
/// Returns `0` for a non-positive / non-finite `source_pdf` or a non-finite
/// quotient, so a degenerate candidate carries no weight and is discarded by
/// the reservoir's own [`Reservoir::update`] guard.
#[inline]
pub fn path_ris_weight(suffix: &PathSuffix, source_pdf: f32) -> f32 {
    if !source_pdf.is_finite() || source_pdf <= 0.0 {
        return 0.0;
    }
    let w = target_function(&suffix.to_gi_sample()) / source_pdf;
    if w.is_finite() {
        w.max(0.0)
    } else {
        0.0
    }
}

/// Streams one freshly traced path suffix into `reservoir` using the uniform `u`.
///
/// Computes the `RIS` weight via [`path_ris_weight`] and folds the sample in
/// with [`Reservoir::update`].  Returns `true` when the candidate became the
/// surviving sample.
#[inline]
pub fn stream_path(
    reservoir: &mut Reservoir<GiSample>,
    suffix: PathSuffix,
    source_pdf: f32,
    u: f32,
) -> bool {
    let weight = path_ris_weight(&suffix, source_pdf);
    reservoir.update(suffix.to_gi_sample(), weight, u)
}

/// Finalizes a reservoir with the selected sample's own target density.
///
/// Equivalent to `finalize_weight(target_function(selected))`; an empty
/// reservoir is finalized to `W = 0`.
#[inline]
pub fn finalize_path(reservoir: &mut Reservoir<GiSample>) {
    match reservoir.sample() {
        Some(sample) => reservoir.finalize_weight(target_function(&sample)),
        None => reservoir.finalize_weight(0.0),
    }
}

/// Shift-mapped induced target density of a neighbour's path at `dst_primary`.
///
/// Reconnects the neighbour's stored path tail to the destination primary
/// vertex and returns `p_hat_shifted * |J|`, the measure-corrected target the
/// `GRIS` merge needs: `p_hat_shifted` is [`target_function`] of the shifted
/// path and `|J|` is the reconnection-shift Jacobian from
/// [`reconnection_shift`].  Always finite and non-negative; a dark or
/// degenerate shift yields `0`.
#[inline]
pub fn shift_mapped_target(neighbor_sample: &GiSample, dst_primary: PathVertex) -> f32 {
    let base = PathSuffix::from_gi_sample(neighbor_sample);
    let (shifted, jacobian) = reconnection_shift(&base, dst_primary);
    let p_hat = target_function(&shifted.to_gi_sample());
    let induced = p_hat * jacobian;
    if induced.is_finite() {
        induced.max(0.0)
    } else {
        0.0
    }
}

/// Merges a neighbour reservoir into `dst` under reconnection-shift `GRIS`.
///
/// The neighbour's selected path is shifted to `dst`'s primary vertex (`x_v`,
/// `n_v`) by [`shift_mapped_target`], and the resulting Jacobian-weighted target
/// density is passed to [`Reservoir::merge`] (whose induced weight becomes
/// `other.m * p_hat * other.W * |J|`).  `dst` and the neighbour must already be
/// finalized.  Returns `true` when the neighbour's sample was selected; an empty
/// neighbour is a no-op.
#[inline]
pub fn merge_path(
    dst: &mut Reservoir<GiSample>,
    visible_point: Vec3,
    visible_normal: Vec3,
    neighbor: &Reservoir<GiSample>,
    u: f32,
) -> bool {
    match neighbor.sample() {
        Some(sample) => {
            let dst_primary = PathVertex::new(visible_point, visible_normal);
            let induced = shift_mapped_target(&sample, dst_primary);
            dst.merge(neighbor, induced, u)
        }
        None => false,
    }
}

/// Balance-heuristic MIS weight for the neighbour term of a two-way path reuse.
///
/// A thin wrapper over [`balance_heuristic`] with the canonical reservoir as
/// technique `0` and the (shift-mapped) neighbour as technique `1`; it returns
/// the neighbour's pairing-count weight `(c_n p_n) / (c_c p_c + c_n p_n)`.  Pass
/// the Jacobian-weighted shifted density (as produced by [`shift_mapped_target`])
/// as `shifted_pdf` so the measure change is reflected in the MIS split.
#[inline]
pub fn pairwise_path_mis_weight(
    canonical_pdf: f32,
    canonical_count: f32,
    shifted_pdf: f32,
    neighbor_count: f32,
) -> f32 {
    let pdfs = [canonical_pdf, shifted_pdf];
    let counts = [canonical_count, neighbor_count];
    balance_heuristic(&pdfs, &counts, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A head-on facing suffix at unit distance: `geometric_term == 1`, so its
    /// target density equals the radiance luminance.  With a grey radiance of
    /// `splat(x)` the Rec. 709 luminance (whose coefficients sum to one) is
    /// exactly `x`, giving a target of `x`.
    fn facing_suffix(x: f32) -> PathSuffix {
        PathSuffix::new(
            PathVertex::new(Vec3::ZERO, Vec3::Z),
            PathVertex::new(Vec3::new(0.0, 0.0, 1.0), Vec3::NEG_Z),
            Vec3::splat(x),
        )
    }

    /// Deterministic integer hash (an invertible bit-mix) for the Monte-Carlo
    /// test's pseudo-uniforms; it keeps the test a pure function with no RNG.
    fn hash_u32(mut x: u32) -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^= x >> 16;
        x
    }

    /// A deterministic pseudo-uniform in `[0, 1)` derived from `seed`.
    fn uniform(seed: u32) -> f32 {
        (hash_u32(seed) >> 8) as f32 / ((1u32 << 24) as f32)
    }

    #[test]
    fn ris_weight_matches_target_over_pdf() {
        let s = facing_suffix(2.0); // target == 2.0
        assert!((path_ris_weight(&s, 0.5) - 4.0).abs() < 1e-5);
        // Degenerate source pdf -> zero weight.
        assert_eq!(path_ris_weight(&s, 0.0), 0.0);
        assert_eq!(path_ris_weight(&s, -1.0), 0.0);
        assert_eq!(path_ris_weight(&s, f32::NAN), 0.0);
    }

    #[test]
    fn stream_and_finalize_single_candidate_gives_inverse_pdf() {
        // One candidate: W = (w_sum / m) / p_hat = (target/pdf) / target = 1/pdf.
        let mut r = Reservoir::<GiSample>::new();
        assert!(stream_path(&mut r, facing_suffix(1.0), 0.25, 0.0));
        finalize_path(&mut r);
        assert!((r.contribution_weight() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn empty_finalize_is_zero() {
        let mut r = Reservoir::<GiSample>::new();
        finalize_path(&mut r);
        assert_eq!(r.contribution_weight(), 0.0);
    }

    #[test]
    fn shift_mapped_target_is_identity_at_same_primary() {
        let suffix = facing_suffix(1.5);
        let sample = suffix.to_gi_sample();
        // Shifting to the sample's own primary vertex must not change its target
        // (Jacobian 1), so the induced density equals the raw target.
        let same = shift_mapped_target(&sample, suffix.primary);
        let raw = target_function(&sample);
        assert!((same - raw).abs() < 1e-6, "same={same} raw={raw}");
    }

    #[test]
    fn merge_path_same_domain_selects_strong_neighbor() {
        // Canonical: a weak path. Neighbour: a strong, high-confidence path.
        let mut canonical = Reservoir::<GiSample>::new();
        stream_path(&mut canonical, facing_suffix(0.01), 1.0, 0.0);
        finalize_path(&mut canonical);

        let strong = facing_suffix(10.0);
        let mut neighbor = Reservoir::<GiSample>::new();
        for _ in 0..16 {
            stream_path(&mut neighbor, strong, 1.0, 0.0);
        }
        finalize_path(&mut neighbor);

        // Reuse at the same primary (same-domain -> Jacobian 1); u = 0 forces
        // selection of the strong induced weight.
        let selected = merge_path(&mut canonical, Vec3::ZERO, Vec3::Z, &neighbor, 0.0);
        assert!(selected);
        assert!(canonical.confidence() >= 17.0);
        finalize_path(&mut canonical);
        assert!(canonical.contribution_weight() > 0.0);
    }

    #[test]
    fn merge_path_empty_neighbor_is_noop() {
        let mut canonical = Reservoir::<GiSample>::new();
        stream_path(&mut canonical, facing_suffix(1.0), 1.0, 0.0);
        let empty = Reservoir::<GiSample>::new();
        assert!(!merge_path(&mut canonical, Vec3::ZERO, Vec3::Z, &empty, 0.0));
        assert_eq!(canonical.confidence(), 1.0);
    }

    #[test]
    fn pairwise_mis_weight_partitions_with_canonical() {
        // Equal pdfs & counts -> neighbour weighted 0.5.
        assert!((pairwise_path_mis_weight(2.0, 1.0, 2.0, 1.0) - 0.5).abs() < 1e-6);
        // A stronger, more-confident neighbour earns more weight.
        let w = pairwise_path_mis_weight(1.0, 1.0, 3.0, 2.0);
        assert!((w - (6.0 / 7.0)).abs() < 1e-6, "w={w}");
    }

    #[test]
    fn gris_estimator_is_unbiased_monte_carlo() {
        // Three candidate paths with known targets 0.5, 1.0, 2.0 drawn with
        // equal probability (source pdf = 1/3).  The RIS estimator
        // `target(y) * W` is unbiased for the sum of the per-sample targets,
        // which is 0.5 + 1.0 + 2.0 = 3.5 (see the module derivation).
        let targets = [0.5f32, 1.0, 2.0];
        let samples: [GiSample; 3] = [
            facing_suffix(targets[0]).to_gi_sample(),
            facing_suffix(targets[1]).to_gi_sample(),
            facing_suffix(targets[2]).to_gi_sample(),
        ];
        let k = samples.len();
        let source_pdf = 1.0 / k as f32;
        let true_integral: f32 = targets.iter().sum(); // 3.5

        const TRIALS: u32 = 30_000;
        const CANDIDATES: u32 = 8;
        let mut acc = 0.0f64;
        let mut seed = 1u32;
        for _ in 0..TRIALS {
            let mut r = Reservoir::<GiSample>::new();
            for _ in 0..CANDIDATES {
                let u_pick = uniform(seed.wrapping_mul(2));
                let u_accept = uniform(seed.wrapping_mul(2).wrapping_add(1));
                seed = seed.wrapping_add(1);
                let idx = ((u_pick * k as f32) as usize).min(k - 1);
                let suffix = PathSuffix::from_gi_sample(&samples[idx]);
                stream_path(&mut r, suffix, source_pdf, u_accept);
            }
            finalize_path(&mut r);
            if let Some(sel) = r.sample() {
                acc += (target_function(&sel) * r.contribution_weight()) as f64;
            }
        }
        let mean = (acc / TRIALS as f64) as f32;
        assert!(
            (mean - true_integral).abs() < 0.1,
            "mean={mean} expected={true_integral}"
        );
    }

    #[test]
    fn results_are_deterministic() {
        let build = || {
            let mut canonical = Reservoir::<GiSample>::new();
            stream_path(&mut canonical, facing_suffix(1.0), 0.5, 0.3);
            finalize_path(&mut canonical);

            let mut neighbor = Reservoir::<GiSample>::new();
            stream_path(&mut neighbor, facing_suffix(3.0), 0.5, 0.1);
            neighbor.cap_confidence(8.0);
            finalize_path(&mut neighbor);

            merge_path(&mut canonical, Vec3::new(0.1, 0.0, 0.0), Vec3::Z, &neighbor, 0.7);
            finalize_path(&mut canonical);
            canonical.contribution_weight()
        };
        assert_eq!(build(), build());
    }
}
