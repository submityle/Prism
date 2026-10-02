//! Spatiotemporal reservoir resampling for direct illumination (`ReSTIR` DI).
//!
//! The stochastic per-cluster sampler ([`super::stochastic`]) bounds how many
//! lights a tile evaluates, but with a small budget its single-frame estimate
//! is noisy: each pixel only ever folds a handful of lights into its shading.
//! `ReSTIR` DI (Bitterli et al. 2020, *Spatiotemporal reservoir resampling for
//! real-time ray tracing with dynamic direct lighting*) drives that variance
//! down for free by **resampling** — every pixel keeps a one-entry reservoir
//! (a running weighted sample via [`Reservoir`]) and reuses the reservoirs of
//! its previous frame (temporal reuse) and screen neighbors (spatial reuse), so
//! each pixel effectively shades against hundreds of candidate lights while
//! only ever storing and shading one.
//!
//! This module borrows the *form* of UE5's reservoir lighting (bounded reuse,
//! `M`-capped history, biased fast path plus an unbiased path) without reusing
//! any of its code. It is pure classical Monte Carlo: no neural, learned, or
//! data-driven components. Selection is deterministic given a seed (it reuses
//! the stateless [`Rng`] from [`crate::particle::reservoir_sample`]), so results
//! are reproducible in golden tests and a future GPU twin.
//!
//! # Pipeline (per pixel, per frame)
//! 1. **Initial candidates / RIS.** Draw `budget.initial_candidates` tentative
//!    lights (e.g. from [`super::stochastic::select_tile_lights`]) and resample
//!    them into one reservoir with resampled importance sampling: candidate `i`
//!    folds in with weight `p̂(i) / p(i)`, the ratio of its *target* function
//!    `p̂` (an estimate of the light's unshadowed contribution at this pixel) to
//!    the *source* pdf `p` it was drawn from. See [`stream_initial`].
//! 2. **Temporal reuse.** Combine with the previous frame's reservoir for this
//!    surface, after capping its [`Reservoir::m`] to bound how long stale
//!    history lingers. See [`cap_history`] and [`combine_biased`] /
//!    [`combine_unbiased`].
//! 3. **Spatial reuse.** Combine with `budget.spatial_neighbors` nearby pixels'
//!    reservoirs, re-evaluating each candidate's target function at *this*
//!    pixel. See [`combine_unbiased`].
//! 4. **Finalize.** [`DiReservoir::finalize`] computes the unbiased contribution
//!    weight `W`; the shaded radiance is `contribution(y) * W` for the single
//!    held light `y`.
//!
//! # Unbiasedness
//! A finalized reservoir yields an unbiased estimate of the full many-light sum
//! `Σ_i contribution_i`: with a target function proportional to the true
//! contribution, `E[contribution(y) · W] = Σ_i contribution_i`. Spatial/temporal
//! reuse across pixels with *different* target functions stays unbiased only
//! with the [`combine_unbiased`] normalization (divide by the count `Z` of
//! source reservoirs whose domain actually contains the chosen sample, not by
//! the raw sample count `M`); [`combine_biased`] drops that correction for speed
//! and is slightly bias-darkened at depth/normal discontinuities — the standard
//! real-time trade.

use alloc::vec::Vec;

use super::ReservoirBudget;
use crate::particle::reservoir_sample::{Reservoir, Rng};

/// Below this target density a reservoir is treated as having no valid sample.
const TARGET_PDF_EPS: f32 = 1.0e-6;

/// One tentative light for the initial resampling pass.
///
/// `target_pdf` is `p̂(i)`, an (unshadowed) estimate of this light's shading
/// contribution at the pixel — the function the reservoir is importance
/// sampling. `source_pdf` is `p(i)`, the density the candidate was actually
/// drawn from (e.g. the `MegaLights` importance pdf). The RIS weight is their
/// ratio, so a cheap, mismatched source pdf stays unbiased as long as it is
/// nonzero wherever `target_pdf` is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiCandidate {
    /// Light identifier carried into shading (e.g. a global light index).
    pub light_index: u32,
    /// Target function value `p̂(i)` at this pixel (unshadowed contribution).
    pub target_pdf: f32,
    /// Source density `p(i)` the candidate was drawn from (must be `> 0`).
    pub source_pdf: f32,
}

/// A direct-illumination reservoir: a [`Reservoir`] plus the target density of
/// the sample it currently holds.
///
/// [`Reservoir`] stores the held light index (`sample`), the running weight sum
/// and count, and the finalized weight `W`. `ReSTIR` additionally needs the
/// *target* density `p̂(y)` of the held sample to finalize `W` and to re-weight
/// the sample when it migrates to another pixel during reuse; that is kept here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiReservoir {
    /// The underlying weighted reservoir (held light index, `w_sum`, `M`, `W`).
    pub reservoir: Reservoir,
    /// Target density `p̂(y)` of the held sample at the pixel that owns this
    /// reservoir. `0` when the reservoir is empty.
    pub target_pdf: f32,
}

impl Default for DiReservoir {
    fn default() -> Self {
        Self::empty()
    }
}

impl DiReservoir {
    /// An empty reservoir: no sample, zero weights, zero target density.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            reservoir: Reservoir::empty(),
            target_pdf: 0.0,
        }
    }

    /// The light index currently held, meaningful only when [`Self::is_empty`]
    /// is `false`.
    #[must_use]
    pub const fn light_index(&self) -> u32 {
        self.reservoir.sample
    }

    /// Whether no sample has been folded in yet.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.reservoir.m == 0
    }

    /// Folds one tentative light into this reservoir with its RIS weight
    /// `target_pdf / source_pdf`, returning `true` when it became the held
    /// sample. A non-positive source pdf contributes zero weight (never chosen).
    pub fn stream(&mut self, candidate: DiCandidate, rand_u01: f32) -> bool {
        let ris_weight = if candidate.source_pdf > 0.0 {
            candidate.target_pdf / candidate.source_pdf
        } else {
            0.0
        };
        let replaced = self
            .reservoir
            .update(candidate.light_index, ris_weight, rand_u01);
        if replaced {
            self.target_pdf = candidate.target_pdf;
        }
        replaced
    }

    /// Caps the sample count `M` so stale temporal history cannot dominate.
    ///
    /// `ReSTIR` temporal reuse would otherwise let a pixel's reservoir age
    /// without bound, pinning it to a light that is no longer the best choice
    /// after the camera or lights move. Clamping `M` to a small multiple of the
    /// per-frame candidate budget (typically ~20x) keeps history influential but
    /// responsive. The weight sum is left intact; only the count is clamped.
    pub fn cap_history(&mut self, max_m: u32) {
        if self.reservoir.m > max_m {
            self.reservoir.m = max_m;
        }
    }

    /// Computes the unbiased contribution weight `W = w_sum / (M · p̂(y))`.
    ///
    /// Use this on a single-pixel reservoir (initial RIS, or reuse where all
    /// sources share this pixel's target function). After this call the shaded
    /// radiance is `contribution(y) · W`. An empty reservoir or a vanishing
    /// target density yields `W = 0`.
    pub fn finalize(&mut self) {
        self.reservoir.finalize_w(self.target_pdf);
    }
}

/// Streams a slice of tentative lights into a fresh reservoir via RIS.
///
/// Only the first `budget.initial_candidates` candidates are considered, matching
/// the configured per-pixel candidate budget; pass fewer to use all of them.
/// The returned reservoir is **not** finalized — reuse it first (temporal /
/// spatial), then call [`DiReservoir::finalize`].
#[must_use]
pub fn stream_initial(
    candidates: &[DiCandidate],
    budget: ReservoirBudget,
    rng: &mut Rng,
) -> DiReservoir {
    let limit = (budget.initial_candidates as usize).min(candidates.len());
    let mut reservoir = DiReservoir::empty();
    for candidate in &candidates[..limit] {
        reservoir.stream(*candidate, rng.next_u01());
    }
    reservoir
}

/// Re-weights a source reservoir's held sample for the destination pixel: the
/// combined weight contribution `p̂_dst(y) · W · M`.
fn reuse_weight(source: &DiReservoir, target_pdf_at_dst: f32) -> f32 {
    target_pdf_at_dst * source.reservoir.w * (source.reservoir.m as f32)
}

/// Combines reservoirs with the **biased** normalization (divide by total `M`).
///
/// `target_at_center(y)` returns the target density of light `y` at the pixel
/// that owns the result. Every source must already be finalized
/// ([`DiReservoir::finalize`]).
/// This is the fast real-time path: it ignores whether each source's domain
/// actually contains the chosen sample, which slightly darkens depth/normal
/// edges. For a reference-quality result use [`combine_unbiased`].
///
/// Returns a finalized reservoir (its `W` is already set).
#[must_use]
pub fn combine_biased<F>(
    sources: &[DiReservoir],
    mut target_at_center: F,
    rng: &mut Rng,
) -> DiReservoir
where
    F: FnMut(u32) -> f32,
{
    let (mut out, total_m) = fold_sources(sources, &mut target_at_center, rng);
    out.reservoir.m = total_m;
    let chosen = out.reservoir.sample;
    out.target_pdf = if out.reservoir.m == 0 {
        0.0
    } else {
        target_at_center(chosen)
    };
    out.finalize();
    out
}

/// Combines reservoirs with the **unbiased** normalization (divide by `Z`, the
/// count of source samples whose domain contains the chosen light).
///
/// `target_at(source_index, light)` returns the target density of `light` at the
/// pixel that produced `sources[source_index]`; `center` selects which source's
/// pixel owns the result. Every source must already be finalized. This is the
/// reference path — it stays unbiased when neighbors have different visibility
/// (occlusion, grazing normals) at the cost of re-testing each source's target.
///
/// Returns a finalized reservoir (its `W` is already set).
#[must_use]
pub fn combine_unbiased<F>(
    sources: &[DiReservoir],
    center: usize,
    mut target_at: F,
    rng: &mut Rng,
) -> DiReservoir
where
    F: FnMut(usize, u32) -> f32,
{
    let mut target_at_center = |light: u32| target_at(center, light);
    let (mut out, total_m) = fold_sources(sources, &mut target_at_center, rng);
    out.reservoir.m = total_m;

    if out.reservoir.m == 0 {
        out.target_pdf = 0.0;
        out.reservoir.w = 0.0;
        return out;
    }

    let chosen = out.reservoir.sample;
    out.target_pdf = target_at(center, chosen);

    // Z = number of source samples whose domain contains the chosen light.
    let mut z: u32 = 0;
    for (i, source) in sources.iter().enumerate() {
        if source.reservoir.m == 0 {
            continue;
        }
        if target_at(i, chosen) > TARGET_PDF_EPS {
            z += source.reservoir.m;
        }
    }

    out.reservoir.w = if z == 0 || out.target_pdf <= TARGET_PDF_EPS {
        0.0
    } else {
        out.reservoir.w_sum / (z as f32 * out.target_pdf)
    };
    out
}

/// Combines reservoirs with **balance-heuristic multiple importance sampling**
/// (the generalized balance heuristic, a.k.a. Talbot `MIS`) — the
/// reference-quality normalization used by production spatiotemporal
/// resamplers.
///
/// Like [`combine_unbiased`] this stays unbiased when neighbors have different
/// target functions, but instead of the `1/Z` count heuristic it weights each
/// source `i`'s held sample `y` by
/// `m_i(y) = M_i * p̂_i(y) / Σ_j M_j * p̂_j(y)` — the balance heuristic over the
/// per-source candidate counts `M_j`. The balance heuristic is provably within
/// a bounded term of the minimum-variance `MIS` combination (Veach), so in the
/// heterogeneous-neighbor case (occlusion, grazing normals, disjoint light
/// domains) it drives reuse variance below the `1/Z` path. The cost is
/// evaluating every held sample's target at every source domain — `O(N^2)`
/// target calls for `N` sources, cheap for the handful of neighbors reuse uses.
///
/// This is generalized `RIS` (`GRIS`, Lin et al. 2022): each source contributes
/// its finalized sample `(y, W_i)` and the resampling weight at the result
/// pixel is `w_i = m_i(y) * p̂(y) * W_i`, where `p̂` is the target at `center`.
/// One sample is selected proportional to `w_i` and the finalized contribution
/// weight is `W = (Σ_i w_i) / p̂(y_selected)` — no extra `1/M` or `1/Z` factor,
/// because the `MIS` weights already sum to one and normalize the estimate.
///
/// `target_at(source_index, light)` returns the target density of `light` at the
/// pixel that produced `sources[source_index]`; `center` selects which source's
/// pixel owns the result. Every source must already be finalized
/// ([`DiReservoir::finalize`]). Returns a finalized reservoir (its `W` is set).
#[must_use]
pub fn combine_mis<F>(
    sources: &[DiReservoir],
    center: usize,
    mut target_at: F,
    rng: &mut Rng,
) -> DiReservoir
where
    F: FnMut(usize, u32) -> f32,
{
    let mut out = DiReservoir::empty();
    let mut total_m: u32 = 0;
    let mut w_sum = 0.0_f32;

    for (i, source) in sources.iter().enumerate() {
        if source.reservoir.m == 0 {
            continue;
        }
        total_m += source.reservoir.m;
        let light = source.reservoir.sample;

        // Balance-heuristic denominator Σ_j M_j * p̂_j(light): re-evaluate this
        // held light's target at every populated source's pixel.
        let mut denom = 0.0_f32;
        for (j, other) in sources.iter().enumerate() {
            if other.reservoir.m == 0 {
                continue;
            }
            denom += (other.reservoir.m as f32) * target_at(j, light);
        }
        if denom <= TARGET_PDF_EPS {
            continue;
        }

        let p_source = target_at(i, light);
        let mis = (source.reservoir.m as f32) * p_source / denom;
        let p_center = target_at(center, light);
        let weight = mis * p_center * source.reservoir.w;
        w_sum += weight;

        if out.reservoir.update(light, weight, rng.next_u01()) {
            out.target_pdf = p_center;
        }
    }

    out.reservoir.m = total_m;
    out.reservoir.w_sum = w_sum;

    if total_m == 0 || out.target_pdf <= TARGET_PDF_EPS || w_sum <= 0.0 {
        out.reservoir.w = 0.0;
        out.target_pdf = 0.0;
    } else {
        out.reservoir.w = w_sum / out.target_pdf;
    }
    out
}

/// Shared stream-and-select over `sources`, re-weighting each held sample for
/// the destination pixel. Returns the accumulating reservoir (its `M` is a raw
/// call count that callers overwrite with the true summed `M`) and the summed
/// source `M`.
fn fold_sources<F>(
    sources: &[DiReservoir],
    target_at_center: &mut F,
    rng: &mut Rng,
) -> (DiReservoir, u32)
where
    F: FnMut(u32) -> f32,
{
    let mut out = DiReservoir::empty();
    let mut total_m: u32 = 0;
    for source in sources {
        if source.reservoir.m == 0 {
            continue;
        }
        total_m += source.reservoir.m;
        let dst_target = target_at_center(source.reservoir.sample);
        let weight = reuse_weight(source, dst_target);
        let replaced = out
            .reservoir
            .update(source.reservoir.sample, weight, rng.next_u01());
        if replaced {
            out.target_pdf = dst_target;
        }
    }
    (out, total_m)
}

/// Collects the finalized sources for an unbiased spatial combine from a center
/// reservoir plus up to `budget.spatial_neighbors` neighbor reservoirs.
///
/// A small convenience mirroring the configured budget: the center is always
/// included first (index `0`), followed by at most `spatial_neighbors` of the
/// supplied neighbors. Pair with [`combine_unbiased`] passing `center = 0`.
#[must_use]
pub fn gather_spatial_sources(
    center: DiReservoir,
    neighbors: &[DiReservoir],
    budget: ReservoirBudget,
) -> Vec<DiReservoir> {
    let take = (budget.spatial_neighbors as usize).min(neighbors.len());
    let mut sources = Vec::with_capacity(1 + take);
    sources.push(center);
    sources.extend_from_slice(&neighbors[..take]);
    sources
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(initial: u16, spatial: u16, temporal: bool) -> ReservoirBudget {
        ReservoirBudget {
            initial_candidates: initial,
            spatial_neighbors: spatial,
            temporal_reuse: temporal,
        }
    }

    /// Draws one index from an unnormalized positive pmf using `rng`.
    fn sample_index(pmf: &[f32], rng: &mut Rng) -> usize {
        let total: f32 = pmf.iter().sum();
        let u = rng.next_u01() * total;
        let mut acc = 0.0;
        for (i, &p) in pmf.iter().enumerate() {
            acc += p;
            if u < acc {
                return i;
            }
        }
        pmf.len() - 1
    }

    /// Builds `m` RIS candidates by sampling lights from `source_pmf`; each
    /// candidate's `target_pdf` is `contribution[i]` and its `source_pdf` is the
    /// normalized draw probability `source_pmf[i] / Σ source_pmf`.
    fn sampled_candidates(
        contribution: &[f32],
        source_pmf: &[f32],
        m: u32,
        rng: &mut Rng,
    ) -> Vec<DiCandidate> {
        let total: f32 = source_pmf.iter().sum();
        (0..m)
            .map(|_| {
                let i = sample_index(source_pmf, rng);
                DiCandidate {
                    light_index: i as u32,
                    target_pdf: contribution[i],
                    source_pdf: source_pmf[i] / total,
                }
            })
            .collect()
    }

    #[test]
    fn empty_reservoir_reports_empty() {
        let r = DiReservoir::empty();
        assert!(r.is_empty());
        assert_eq!(r.reservoir.m, 0);
    }

    #[test]
    fn single_candidate_is_always_held() {
        let mut r = DiReservoir::empty();
        let replaced = r.stream(
            DiCandidate {
                light_index: 9,
                target_pdf: 2.0,
                source_pdf: 1.0,
            },
            0.5,
        );
        assert!(replaced);
        assert_eq!(r.light_index(), 9);
        assert_eq!(r.target_pdf, 2.0);
    }

    #[test]
    fn zero_source_pdf_contributes_nothing() {
        // A zero-source-pdf candidate still counts toward M (it was drawn) but
        // can never be selected and yields zero finalized weight.
        let mut r = DiReservoir::empty();
        let replaced = r.stream(
            DiCandidate {
                light_index: 3,
                target_pdf: 5.0,
                source_pdf: 0.0,
            },
            0.0,
        );
        assert!(!replaced);
        assert_eq!(r.reservoir.m, 1);
        assert_eq!(r.reservoir.w_sum, 0.0);
        r.finalize();
        assert_eq!(r.reservoir.w, 0.0);
    }

    #[test]
    fn cap_history_clamps_count_only() {
        let mut r = DiReservoir::empty();
        for _ in 0..100 {
            r.stream(
                DiCandidate {
                    light_index: 1,
                    target_pdf: 1.0,
                    source_pdf: 1.0,
                },
                0.5,
            );
        }
        assert_eq!(r.reservoir.m, 100);
        let w_sum = r.reservoir.w_sum;
        r.cap_history(20);
        assert_eq!(r.reservoir.m, 20);
        assert_eq!(r.reservoir.w_sum, w_sum);
    }

    #[test]
    fn deterministic_given_seed() {
        let contribution = [1.0, 2.0, 3.0, 0.5, 4.0];
        let source = [1.0, 2.0, 1.0, 3.0, 1.0];
        let mut a = Rng::new(123);
        let mut b = Rng::new(123);
        let ca = sampled_candidates(&contribution, &source, 5, &mut a);
        let cb = sampled_candidates(&contribution, &source, 5, &mut b);
        let ra = stream_initial(&ca, budget(5, 0, false), &mut a);
        let rb = stream_initial(&cb, budget(5, 0, false), &mut b);
        assert_eq!(ra, rb);
    }

    #[test]
    fn initial_candidate_budget_is_respected() {
        let cands = [DiCandidate {
            light_index: 0,
            target_pdf: 1.0,
            source_pdf: 1.0,
        }; 6];
        let mut rng = Rng::new(7);
        let r = stream_initial(&cands, budget(3, 0, false), &mut rng);
        assert!(r.reservoir.m <= 3);
        assert!(r.reservoir.m >= 1);
    }

    #[test]
    fn initial_ris_is_unbiased() {
        // Averaging contribution(y) * W over many seeds converges to the sum of
        // per-light contributions, even with a source pmf that does not match
        // the target (that mismatch is exactly what RIS corrects for).
        let contribution = [0.4, 1.3, 2.1, 0.7, 3.0, 1.1, 0.9, 2.4];
        let source_pmf = [3.0, 1.0, 2.0, 5.0, 1.0, 4.0, 2.0, 1.0];
        let exact: f32 = contribution.iter().sum();
        let m = 16u32;

        let seeds = 300_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(2_654_435_761).wrapping_add(1));
            let cands = sampled_candidates(&contribution, &source_pmf, m, &mut rng);
            let mut r = stream_initial(&cands, budget(m as u16, 0, false), &mut rng);
            r.finalize();
            acc += f64::from(r.target_pdf * r.reservoir.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel_err = (mean - exact).abs() / exact;
        assert!(rel_err < 0.02, "mean {mean} vs exact {exact} ({rel_err})");
    }

    #[test]
    fn biased_combine_single_source_is_identity_in_expectation() {
        let contribution = [1.0, 2.0, 3.0, 4.0];
        let source_pmf = [2.0, 1.0, 1.0, 3.0];
        let exact: f32 = contribution.iter().sum();
        let target = |light: u32| contribution[light as usize];
        let m = 8u32;

        let seeds = 300_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(40_503).wrapping_add(1));
            let cands = sampled_candidates(&contribution, &source_pmf, m, &mut rng);
            let mut initial = stream_initial(&cands, budget(m as u16, 0, false), &mut rng);
            initial.finalize();
            let combined = combine_biased(&[initial], target, &mut rng);
            acc += f64::from(combined.target_pdf * combined.reservoir.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel_err = (mean - exact).abs() / exact;
        assert!(rel_err < 0.02, "mean {mean} vs exact {exact} ({rel_err})");
    }

    /// Per-source light visibility for the reuse tests: source 0 (center) sees
    /// every light; source 1 sees lights `0..3`; source 2 sees lights `3..6`.
    fn visible(src: usize, light: u32) -> bool {
        match src {
            0 => true,
            1 => (light as usize) < 3,
            _ => (light as usize) >= 3,
        }
    }

    fn build_sources(contribution: &[f32], m: u32, rng: &mut Rng) -> Vec<DiReservoir> {
        let n = contribution.len();
        (0..3usize)
            .map(|src| {
                // Source pmf is uniform over this source's visible lights.
                let pmf: Vec<f32> = (0..n)
                    .map(|l| if visible(src, l as u32) { 1.0 } else { 0.0 })
                    .collect();
                let cands = sampled_candidates(contribution, &pmf, m, rng);
                let mut r = stream_initial(&cands, budget(m as u16, 0, false), rng);
                r.finalize();
                r
            })
            .collect()
    }

    #[test]
    fn unbiased_spatial_reuse_with_partial_domains_converges() {
        // Center sees all lights; neighbors see disjoint-ish subsets. The
        // unbiased combine must still converge to the center's full sum.
        let contribution = [0.5, 1.5, 2.5, 1.0, 3.0, 0.8];
        let target_at = |src: usize, light: u32| -> f32 {
            if visible(src, light) {
                contribution[light as usize]
            } else {
                0.0
            }
        };
        let exact: f32 = contribution.iter().sum();
        let m = 12u32;

        let seeds = 600_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(2_246_822_519).wrapping_add(7));
            let sources = build_sources(&contribution, m, &mut rng);
            let combined = combine_unbiased(&sources, 0, target_at, &mut rng);
            acc += f64::from(combined.target_pdf * combined.reservoir.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel_err = (mean - exact).abs() / exact;
        assert!(rel_err < 0.03, "mean {mean} vs exact {exact} ({rel_err})");
    }

    #[test]
    fn biased_combine_darkens_with_partial_domains() {
        // Same partial-visibility setup: the biased normalization (divide by
        // total M, not Z) underestimates, which is why the unbiased path exists.
        let contribution = [0.5, 1.5, 2.5, 1.0, 3.0, 0.8];
        let target_center = |light: u32| contribution[light as usize];
        let exact: f32 = contribution.iter().sum();
        let m = 12u32;

        let seeds = 300_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(915_488_749).wrapping_add(3));
            let sources = build_sources(&contribution, m, &mut rng);
            let combined = combine_biased(&sources, target_center, &mut rng);
            acc += f64::from(combined.target_pdf * combined.reservoir.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        assert!(mean < exact * 0.97, "mean {mean} not darker than {exact}");
    }

    #[test]
    fn gather_spatial_sources_honors_neighbor_budget() {
        let center = DiReservoir::empty();
        let neighbors = [DiReservoir::empty(); 5];
        let sources = gather_spatial_sources(center, &neighbors, budget(0, 3, true));
        assert_eq!(sources.len(), 4); // center + 3 neighbors
    }

    #[test]
    fn mis_spatial_reuse_with_partial_domains_converges() {
        // Balance-heuristic MIS must stay unbiased under the same partial-
        // visibility setup as the Z-normalized combine: center sees all lights,
        // neighbors see disjoint-ish subsets, yet the estimate converges to the
        // center's full many-light sum.
        let contribution = [0.5, 1.5, 2.5, 1.0, 3.0, 0.8];
        let target_at = |src: usize, light: u32| -> f32 {
            if visible(src, light) {
                contribution[light as usize]
            } else {
                0.0
            }
        };
        let exact: f32 = contribution.iter().sum();
        let m = 12u32;

        let seeds = 600_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(2_654_435_761).wrapping_add(11));
            let sources = build_sources(&contribution, m, &mut rng);
            let combined = combine_mis(&sources, 0, target_at, &mut rng);
            acc += f64::from(combined.target_pdf * combined.reservoir.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel_err = (mean - exact).abs() / exact;
        assert!(rel_err < 0.03, "mean {mean} vs exact {exact} ({rel_err})");
    }

    #[test]
    fn mis_combine_has_no_higher_variance_than_z_norm() {
        // Both combines are unbiased, so their estimator means agree; the
        // balance heuristic is the reference minimum-variance MIS weighting, so
        // under heterogeneous (partial-domain) neighbors its per-seed estimator
        // spread must not exceed the 1/Z path. Common random numbers (shared
        // sources per seed) keep the comparison low-noise.
        let contribution = [0.5, 1.5, 2.5, 1.0, 3.0, 0.8];
        let target_at = |src: usize, light: u32| -> f32 {
            if visible(src, light) {
                contribution[light as usize]
            } else {
                0.0
            }
        };

        let seeds = 400_000u32;
        let mut sum_z = 0.0f64;
        let mut sumsq_z = 0.0f64;
        let mut sum_m = 0.0f64;
        let mut sumsq_m = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(2_246_822_519).wrapping_add(5));
            let sources = build_sources(&contribution, 12, &mut rng);
            let mut rng_z = Rng::new(s.wrapping_mul(747_796_405).wrapping_add(1));
            let mut rng_m = Rng::new(s.wrapping_mul(747_796_405).wrapping_add(1));
            let cz = combine_unbiased(&sources, 0, target_at, &mut rng_z);
            let cm = combine_mis(&sources, 0, target_at, &mut rng_m);
            let ez = f64::from(cz.target_pdf * cz.reservoir.w);
            let em = f64::from(cm.target_pdf * cm.reservoir.w);
            sum_z += ez;
            sumsq_z += ez * ez;
            sum_m += em;
            sumsq_m += em * em;
        }
        let n = f64::from(seeds);
        let mean_z = sum_z / n;
        let mean_m = sum_m / n;
        let var_z = sumsq_z / n - mean_z * mean_z;
        let var_m = sumsq_m / n - mean_m * mean_m;
        assert!(
            (mean_z - mean_m).abs() < 0.02,
            "unbiased means should agree: z {mean_z} m {mean_m}"
        );
        assert!(
            var_m <= var_z * 1.02,
            "MIS variance {var_m} should not exceed Z-norm variance {var_z}"
        );
    }
}
