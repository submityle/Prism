//! Path resampling for indirect illumination (`ReSTIR` GI).
//!
//! Where [`super::restir_di`] resamples *lights* for direct illumination, this
//! module resamples *indirect paths* (Ouyang et al. 2021, *`ReSTIR` GI: Path
//! Resampling for Real-Time Path Tracing*). Each pixel traces one short path and
//! keeps a reservoir over **sample points**: the second path vertex `x_s` (a
//! surface hit one bounce away), its normal, and the outgoing radiance `L_o`
//! that vertex sends back toward the shading point. Temporal and spatial reuse
//! then lets a pixel import its neighbors' sample points, so each pixel
//! effectively integrates many indirect paths while tracing only one.
//!
//! Unlike direct-light reuse, importing a GI sample means **reconnecting**: the
//! sample point is a fixed location in space, so reusing it at a different
//! shading point changes the connecting direction and therefore the solid-angle
//! measure. That change of measure is the *reconnection Jacobian*
//! ([`reconnection_jacobian`]); every reuse weight carries it so the estimator
//! stays unbiased across pixels with different geometry.
//!
//! This borrows the *form* of UE5-era reservoir path reuse (bounded reuse,
//! `M`-capped history, biased fast path plus an unbiased path, reconnection
//! Jacobian) without reusing any of its code. It is pure classical Monte Carlo:
//! no neural, learned, or data-driven components. Selection is deterministic
//! given a seed (it reuses the stateless [`Rng`] from
//! [`crate::particle::reservoir_sample`]), so results reproduce in golden tests
//! and a future GPU twin.
//!
//! Because a GI reservoir holds a *struct* sample (point, normal, radiance)
//! rather than a `u32` light index, it cannot reuse the integer
//! [`crate::particle::reservoir_sample::Reservoir`]; the reservoir arithmetic
//! (weighted update, `M`-cap, unbiased `W`) is reimplemented here over the
//! [`GiSample`] payload, but follows the exact same algebra.
//!
//! # Pipeline (per pixel, per frame)
//! 1. **Initial resampling.** Trace one path, draw
//!    `budget.initial_candidates` tentative sample points, and resample them
//!    into one reservoir with RIS (see [`stream_initial`]).
//! 2. **Temporal reuse.** Combine with the reprojected previous-frame reservoir
//!    (after [`GiReservoir::cap_history`]), applying the reconnection Jacobian.
//! 3. **Spatial reuse.** Combine with nearby pixels' reservoirs, re-evaluating
//!    the target function and Jacobian at *this* shading point.
//! 4. **Finalize.** [`GiReservoir::finalize`] computes the unbiased weight `W`;
//!    the resolved radiance is `sample.radiance · W` scaled by the shading
//!    integrand the caller folds into the target function.

use alloc::vec::Vec;

use super::ReservoirBudget;
use crate::particle::reservoir_sample::Rng;

/// Below this target density a reservoir is treated as holding no valid sample.
const TARGET_PDF_EPS: f32 = 1.0e-6;

/// Below this (squared) distance or cosine the reconnection is treated as
/// degenerate and its Jacobian is zero (the reuse is rejected).
const GEOM_EPS: f32 = 1.0e-8;

fn abs_f32(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The second path vertex carried by a GI reservoir: a surface point one bounce
/// from the shading point, plus the radiance it reflects back toward it.
///
/// `sample_point` and `sample_normal` are in a world space shared by all
/// reservoirs being combined; `radiance` is the outgoing radiance `L_o(x_s →
/// x)` in linear RGB. The sample point and its normal are what the reconnection
/// Jacobian is computed from; the radiance is what shading ultimately reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiSample {
    /// World-space position of the sample point `x_s`.
    pub sample_point: [f32; 3],
    /// Unit surface normal at the sample point.
    pub sample_normal: [f32; 3],
    /// Outgoing radiance from the sample point toward the shading point (RGB).
    pub radiance: [f32; 3],
}

impl GiSample {
    /// A zeroed sample (used as the placeholder in an empty reservoir).
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            sample_point: [0.0; 3],
            sample_normal: [0.0; 3],
            radiance: [0.0; 3],
        }
    }
}

/// The surface a reservoir is anchored to: the shading point `x` and its
/// normal. Needed to compute the reconnection Jacobian and to re-evaluate the
/// target function when a sample migrates between pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadingPoint {
    /// World-space position of the shading point `x`.
    pub position: [f32; 3],
    /// Unit surface normal at the shading point.
    pub normal: [f32; 3],
}

impl ShadingPoint {
    /// Builds a shading point from a position and (assumed unit) normal.
    #[must_use]
    pub const fn new(position: [f32; 3], normal: [f32; 3]) -> Self {
        Self { position, normal }
    }
}

/// One tentative sample point for the initial resampling pass.
///
/// `target_pdf` is `p̂`, the (luminance of the) shading contribution this sample
/// produces at the pixel — the function the reservoir importance samples.
/// `source_pdf` is the solid-angle density the sample direction was drawn from
/// at the shading point (must be `> 0`). The RIS weight is their ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiCandidate {
    /// The sample point / radiance payload.
    pub sample: GiSample,
    /// Target function value `p̂` at this pixel.
    pub target_pdf: f32,
    /// Source solid-angle density the sample was drawn from (must be `> 0`).
    pub source_pdf: f32,
}

/// Computes the reconnection Jacobian for importing a sample built at `src` into
/// the shading point `dst`.
///
/// A GI sample point is fixed in space, so the solid angle it subtends differs
/// between shading points. The Jacobian converts the source pixel's solid-angle
/// density into the destination pixel's:
///
/// ```text
///       cosθ_dst / ‖x_dst − x_s‖²
/// J  =  ─────────────────────────
///       cosθ_src / ‖x_src − x_s‖²
/// ```
///
/// where `θ` is the angle at the *sample point's* normal with the direction to
/// the shading point. Multiplying a reused reservoir's resampling weight by `J`
/// makes `E[reused estimate] = I(dst)` rather than `I(src)` (verified against a
/// brute-force reference in the tests). A degenerate reconnection (sample point
/// coincident with a shading point, or a grazing sample normal) returns `0`,
/// rejecting the reuse.
#[must_use]
pub fn reconnection_jacobian(src: ShadingPoint, dst: ShadingPoint, sample: &GiSample) -> f32 {
    let to_src = sub3(src.position, sample.sample_point);
    let to_dst = sub3(dst.position, sample.sample_point);
    let d_src2 = dot3(to_src, to_src);
    let d_dst2 = dot3(to_dst, to_dst);
    if d_src2 < GEOM_EPS || d_dst2 < GEOM_EPS {
        return 0.0;
    }
    let d_src = d_src2.sqrt();
    let d_dst = d_dst2.sqrt();
    // Cosine at the sample-point normal toward each shading point.
    let cos_src = abs_f32(dot3(sample.sample_normal, to_src)) / d_src;
    let cos_dst = abs_f32(dot3(sample.sample_normal, to_dst)) / d_dst;
    let den = cos_src / d_src2;
    if den < GEOM_EPS {
        return 0.0;
    }
    let num = cos_dst / d_dst2;
    num / den
}

/// A GI reservoir: a running weighted sample ([`GiSample`]) plus the bookkeeping
/// needed for the unbiased contribution weight `W`.
///
/// Mirrors [`crate::particle::reservoir_sample::Reservoir`] but over a struct
/// payload: `w_sum` is the running weight sum, `m` the sample count, `w` the
/// finalized contribution weight, and `target_pdf` the target density `p̂(y)` of
/// the held sample at the pixel that owns this reservoir.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiReservoir {
    /// The currently held sample (meaningful only when `m > 0`).
    pub sample: GiSample,
    /// Target density `p̂(y)` of the held sample at the owning pixel.
    pub target_pdf: f32,
    /// Running sum of all resampling weights observed.
    pub w_sum: f32,
    /// Number of samples folded into this reservoir.
    pub m: u32,
    /// Unbiased contribution weight, set by [`Self::finalize`].
    pub w: f32,
}

impl Default for GiReservoir {
    fn default() -> Self {
        Self::empty()
    }
}

impl GiReservoir {
    /// An empty reservoir: no sample, zero weights.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            sample: GiSample::zero(),
            target_pdf: 0.0,
            w_sum: 0.0,
            m: 0,
            w: 0.0,
        }
    }

    /// Whether no sample has been folded in yet.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.m == 0
    }

    /// Folds one tentative sample into this reservoir with its RIS weight
    /// `target_pdf / source_pdf`, returning `true` when it became the held
    /// sample. A non-positive source pdf contributes zero weight.
    pub fn stream(&mut self, candidate: GiCandidate, rand_u01: f32) -> bool {
        let weight = if candidate.source_pdf > 0.0 {
            candidate.target_pdf / candidate.source_pdf
        } else {
            0.0
        };
        self.push_weighted(candidate.sample, candidate.target_pdf, weight, rand_u01)
    }

    /// Low-level weighted update: adds `weight` to `w_sum`, bumps `M`, and
    /// replaces the held sample with probability `weight / w_sum`. Returns
    /// whether the held sample was replaced. `dst_target_pdf` is the target
    /// density of `sample` at this reservoir's owning pixel, adopted alongside
    /// the sample when it wins.
    fn push_weighted(
        &mut self,
        sample: GiSample,
        dst_target_pdf: f32,
        weight: f32,
        rand_u01: f32,
    ) -> bool {
        self.w_sum += weight;
        self.m += 1;
        let replace = rand_u01 * self.w_sum < weight;
        if replace {
            self.sample = sample;
            self.target_pdf = dst_target_pdf;
        }
        replace
    }

    /// Caps the sample count `M` so stale temporal history cannot dominate.
    /// Only the count is clamped; the weight sum is left intact.
    pub fn cap_history(&mut self, max_m: u32) {
        if self.m > max_m {
            self.m = max_m;
        }
    }

    /// Computes the unbiased contribution weight `W = w_sum / (M · p̂(y))`.
    ///
    /// Use on a single-pixel reservoir (initial RIS, or reuse where every
    /// source shares this pixel). An empty reservoir or vanishing target
    /// density yields `W = 0`.
    pub fn finalize(&mut self) {
        if self.m == 0 || self.target_pdf <= TARGET_PDF_EPS {
            self.w = 0.0;
            return;
        }
        self.w = self.w_sum / (self.m as f32 * self.target_pdf);
    }
}

/// Streams a slice of tentative samples into a fresh reservoir via RIS.
///
/// Only the first `budget.initial_candidates` candidates are considered. The
/// returned reservoir is **not** finalized — reuse it first (temporal /
/// spatial), then call [`GiReservoir::finalize`].
#[must_use]
pub fn stream_initial(
    candidates: &[GiCandidate],
    budget: ReservoirBudget,
    rng: &mut Rng,
) -> GiReservoir {
    let limit = (budget.initial_candidates as usize).min(candidates.len());
    let mut reservoir = GiReservoir::empty();
    for candidate in &candidates[..limit] {
        reservoir.stream(*candidate, rng.next_u01());
    }
    reservoir
}

/// A finalized GI reservoir paired with the shading point it was produced on,
/// supplied as a temporal-history or spatial-neighbor reuse source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiReuseSource {
    /// The finalized source reservoir (its `W` is already set).
    pub reservoir: GiReservoir,
    /// The shading point this reservoir was computed on.
    pub shading_point: ShadingPoint,
}

impl GiReuseSource {
    /// Pairs a reservoir with the shading point it belongs to.
    #[must_use]
    pub const fn new(reservoir: GiReservoir, shading_point: ShadingPoint) -> Self {
        Self {
            reservoir,
            shading_point,
        }
    }
}

/// The combined resampling weight contributed by a reuse source at the
/// destination shading point: `p̂_dst(y) · W · M · J`.
fn reuse_weight(source: &GiReuseSource, center: ShadingPoint, dst_target_pdf: f32) -> f32 {
    let jacobian = reconnection_jacobian(source.shading_point, center, &source.reservoir.sample);
    dst_target_pdf * source.reservoir.w * (source.reservoir.m as f32) * jacobian
}

/// Shared stream-and-select over reuse sources, re-weighting each held sample
/// for the destination pixel with the reconnection Jacobian. Returns the
/// accumulating reservoir (its `M` is a raw call count that callers overwrite)
/// and the summed source `M`.
fn fold_sources<F>(
    sources: &[GiReuseSource],
    center: ShadingPoint,
    target_at_center: &mut F,
    rng: &mut Rng,
) -> (GiReservoir, u32)
where
    F: FnMut(&GiSample) -> f32,
{
    let mut out = GiReservoir::empty();
    let mut total_m: u32 = 0;
    for source in sources {
        if source.reservoir.m == 0 {
            continue;
        }
        total_m += source.reservoir.m;
        let dst_target = target_at_center(&source.reservoir.sample);
        let weight = reuse_weight(source, center, dst_target);
        out.push_weighted(source.reservoir.sample, dst_target, weight, rng.next_u01());
    }
    (out, total_m)
}

/// Combines reuse sources with the **biased** normalization (divide by total
/// `M`). `target_at_center(sample)` returns the sample's target density at the
/// destination shading point `center`. Every source must already be finalized.
///
/// This is the fast real-time path; it slightly darkens depth / normal edges.
/// Returns a finalized reservoir.
#[must_use]
pub fn combine_biased<F>(
    sources: &[GiReuseSource],
    center: ShadingPoint,
    mut target_at_center: F,
    rng: &mut Rng,
) -> GiReservoir
where
    F: FnMut(&GiSample) -> f32,
{
    let (mut out, total_m) = fold_sources(sources, center, &mut target_at_center, rng);
    out.m = total_m;
    out.target_pdf = if out.m == 0 {
        0.0
    } else {
        target_at_center(&out.sample)
    };
    out.finalize();
    out
}

/// Combines reuse sources with the **unbiased** normalization (divide by `Z`,
/// the count of source samples whose domain contains the chosen sample).
///
/// `target_at(shading_point, sample)` returns the target density of `sample` at
/// `shading_point`; `center` is the destination shading point that owns the
/// result. Every source must already be finalized. Stays unbiased when sources
/// have different visibility / geometry, at the cost of re-testing each source.
///
/// Returns a finalized reservoir.
#[must_use]
pub fn combine_unbiased<F>(
    sources: &[GiReuseSource],
    center: ShadingPoint,
    mut target_at: F,
    rng: &mut Rng,
) -> GiReservoir
where
    F: FnMut(&ShadingPoint, &GiSample) -> f32,
{
    let (mut out, total_m) = {
        let mut target_at_center = |sample: &GiSample| target_at(&center, sample);
        fold_sources(sources, center, &mut target_at_center, rng)
    };
    out.m = total_m;
    if out.m == 0 {
        out.target_pdf = 0.0;
        out.w = 0.0;
        return out;
    }

    let chosen = out.sample;
    out.target_pdf = target_at(&center, &chosen);

    // Z = number of source samples whose domain contains the chosen sample.
    let mut z: u32 = 0;
    for source in sources {
        if source.reservoir.m == 0 {
            continue;
        }
        if target_at(&source.shading_point, &chosen) > TARGET_PDF_EPS {
            z += source.reservoir.m;
        }
    }

    out.w = if z == 0 || out.target_pdf <= TARGET_PDF_EPS {
        0.0
    } else {
        out.w_sum / (z as f32 * out.target_pdf)
    };
    out
}

/// Combines reuse sources with **balance-heuristic multiple importance
/// sampling** (generalized `RIS`, Lin et al. 2022), the minimum-variance
/// reference weighting that supersedes the `1/Z` normalization of
/// [`combine_unbiased`].
///
/// `target_at(shading_point, sample)` returns the target density of `sample` at
/// `shading_point`; `center` is the canonical destination domain that owns the
/// result and **must** be the shading point of `sources[0]` (the pixel's own
/// reservoir), so the balance-heuristic denominator contains the canonical
/// term. Every source must already be finalized.
///
/// For a sample `y` held by source `i`, the balance-heuristic `MIS` weight in
/// the canonical domain is
///
/// ```text
///          M_i · p̂_i(y) · J(center→i)
/// m_i(y) = ───────────────────────────────
///          Σ_j M_j · p̂_j(y) · J(center→j)
/// ```
///
/// where `p̂_i(y) = target_at(sp_i, y)` and `J(center→i)` is the reconnection
/// Jacobian from the canonical domain into source `i` (computed directly to
/// avoid dividing by a possibly degenerate Jacobian). The generalized-`RIS`
/// resampling weight carries the forward shift Jacobian `J(i→center)`, which is
/// the reciprocal of `J(center→i)`, so the two cancel exactly and the pushed
/// weight reduces to
///
/// ```text
/// w_i = M_i · p̂_i(y) · p̂_center(y) · W_i / D   with   D = Σ_j M_j · p̂_j(y) · J(center→j).
/// ```
///
/// A single source whose shading point is `center` reduces to the identity
/// `W_out = W_in` (`J(center→center) = 1`). Returns a finalized reservoir;
/// sources with a vanishing denominator or target are dropped unbiasedly.
#[must_use]
pub fn combine_mis<F>(
    sources: &[GiReuseSource],
    center: ShadingPoint,
    mut target_at: F,
    rng: &mut Rng,
) -> GiReservoir
where
    F: FnMut(&ShadingPoint, &GiSample) -> f32,
{
    let mut out = GiReservoir::empty();
    let mut total_m: u32 = 0;
    for source in sources {
        if source.reservoir.m == 0 {
            continue;
        }
        total_m += source.reservoir.m;
        let sample = source.reservoir.sample;

        // Balance-heuristic denominator in the canonical domain: every source's
        // sample count scaled by its target density for this sample and the
        // reconnection Jacobian from the center into that source's domain.
        let mut denom = 0.0_f32;
        for other in sources {
            if other.reservoir.m == 0 {
                continue;
            }
            let p_other = target_at(&other.shading_point, &sample);
            if p_other <= TARGET_PDF_EPS {
                continue;
            }
            let jac = reconnection_jacobian(center, other.shading_point, &sample);
            denom += (other.reservoir.m as f32) * p_other * jac;
        }
        if denom <= TARGET_PDF_EPS {
            continue;
        }

        let p_source = target_at(&source.shading_point, &sample);
        let p_center = target_at(&center, &sample);
        let weight = (source.reservoir.m as f32) * p_source * p_center * source.reservoir.w / denom;
        out.push_weighted(sample, p_center, weight, rng.next_u01());
    }

    out.m = total_m;
    if total_m == 0 {
        out.target_pdf = 0.0;
        out.w = 0.0;
        return out;
    }
    out.target_pdf = target_at(&center, &out.sample);
    out.w = if out.target_pdf <= TARGET_PDF_EPS || out.w_sum <= 0.0 {
        0.0
    } else {
        out.w_sum / out.target_pdf
    };
    out
}

/// Gathers reuse sources for a spatial combine: the center reservoir first,
/// then up to `budget.spatial_neighbors` neighbors. Pair with
/// [`combine_unbiased`].
#[must_use]
pub fn gather_spatial_sources(
    center: GiReuseSource,
    neighbors: &[GiReuseSource],
    budget: ReservoirBudget,
) -> Vec<GiReuseSource> {
    let take = (budget.spatial_neighbors as usize).min(neighbors.len());
    let mut sources = Vec::with_capacity(1 + take);
    sources.push(center);
    sources.extend_from_slice(&neighbors[..take]);
    sources
}

/// Default boiling-filter strength for GI reservoirs (`RTXDI` convention): a
/// reservoir is cleared only when its finalized weight exceeds `11x` the tile
/// mean.
pub const DEFAULT_BOILING_FILTER_STRENGTH: f32 = 0.5;

/// Tile-local **boiling filter** for GI reservoirs: the GI twin of the direct
/// illumination [`super::restir_di::boiling_filter_di`].
///
/// Indirect bounces are even more prone to fireflies than direct lighting (a
/// single bright, rarely sampled bounce dominates `w_sum`), and temporal reuse
/// keeps that outlier alive for several frames ("boiling"). This clears any
/// reservoir whose finalized contribution weight `w` exceeds `multiplier *
/// mean_w`, where `mean_w` averages `w` over the non-empty reservoirs of one
/// screen tile and
///
/// ```text
/// multiplier = 10 / clamp(filter_strength, eps, 1) - 9
/// ```
///
/// so `filter_strength` lives in `(0, 1]` (`1` thresholds at the mean, the
/// default `0.5` at `11x` the mean, `<= 0` or non-finite disables it).
///
/// A biased but standard real-time firefly suppressor: when nothing is clipped
/// the per-pixel estimate is left exactly as the unbiased resample produced it.
/// A tile with fewer than two non-empty reservoirs is left untouched. `tile`
/// holds the **finalized** reservoirs (post-[`GiReservoir::finalize`]) of one
/// screen tile.
pub fn boiling_filter_gi(tile: &mut [GiReservoir], filter_strength: f32) {
    if filter_strength <= 0.0 || !filter_strength.is_finite() {
        return;
    }
    let strength = if filter_strength < 1.0 {
        filter_strength
    } else {
        1.0
    };
    let multiplier = 10.0 / strength - 9.0;

    let mut sum = 0.0_f32;
    let mut count = 0_u32;
    for r in tile.iter() {
        if !r.is_empty() && r.w > 0.0 {
            sum += r.w;
            count += 1;
        }
    }
    if count < 2 {
        return;
    }
    let threshold = (sum / count as f32) * multiplier;
    for r in tile.iter_mut() {
        if !r.is_empty() && r.w > threshold {
            *r = GiReservoir::empty();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A finalized GI reservoir holding a zero sample with contribution weight `w`.
    fn gi_with_w(w: f32) -> GiReservoir {
        GiReservoir {
            sample: GiSample::zero(),
            target_pdf: 1.0,
            w_sum: w,
            m: 1,
            w,
        }
    }

    #[test]
    fn boiling_filter_gi_clears_single_outlier() {
        // A representative 8x8 tile of unit-weight reservoirs plus one 200x
        // firefly: mean ~= 4.1, threshold ~= 45, so only the firefly goes.
        let mut tile: Vec<GiReservoir> = (0..63).map(|_| gi_with_w(1.0)).collect();
        tile.push(gi_with_w(200.0));
        boiling_filter_gi(&mut tile, DEFAULT_BOILING_FILTER_STRENGTH);
        assert!(tile[63].is_empty());
        for (i, r) in tile.iter().enumerate().take(63) {
            assert!(!r.is_empty(), "reservoir {i} wrongly cleared");
        }
    }

    #[test]
    fn boiling_filter_gi_preserves_uniform_tile() {
        let before = [gi_with_w(1.0), gi_with_w(2.0), gi_with_w(3.0)];
        let mut tile = before;
        boiling_filter_gi(&mut tile, DEFAULT_BOILING_FILTER_STRENGTH);
        assert_eq!(tile, before);
    }

    #[test]
    fn boiling_filter_gi_disabled_is_identity() {
        let before = [gi_with_w(1.0), gi_with_w(5000.0)];
        for strength in [0.0_f32, -2.0, f32::NAN, f32::INFINITY] {
            let mut tile = before;
            boiling_filter_gi(&mut tile, strength);
            assert_eq!(tile, before, "strength {strength} should be a no-op");
        }
    }

    #[test]
    fn boiling_filter_gi_needs_two_samples() {
        let mut tile = [gi_with_w(5000.0), GiReservoir::empty()];
        boiling_filter_gi(&mut tile, DEFAULT_BOILING_FILTER_STRENGTH);
        assert!(!tile[0].is_empty());
    }

    fn budget(initial: u16, spatial: u16, temporal: bool) -> ReservoirBudget {
        ReservoirBudget {
            initial_candidates: initial,
            spatial_neighbors: spatial,
            temporal_reuse: temporal,
        }
    }

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        abs_f32(a - b) <= eps
    }

    #[test]
    fn jacobian_is_identity_for_same_shading_point() {
        let sp = ShadingPoint::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let sample = GiSample {
            sample_point: [0.3, -0.2, 2.0],
            sample_normal: [0.0, 0.0, -1.0],
            radiance: [1.0, 1.0, 1.0],
        };
        assert!(approx(reconnection_jacobian(sp, sp, &sample), 1.0, 1e-5));
    }

    #[test]
    fn jacobian_is_reciprocal() {
        let a = ShadingPoint::new([-0.5, 0.1, 0.0], [0.0, 0.0, 1.0]);
        let b = ShadingPoint::new([0.6, -0.3, 0.0], [0.0, 0.0, 1.0]);
        let sample = GiSample {
            sample_point: [0.1, 0.2, 2.0],
            sample_normal: [0.0, 0.0, -1.0],
            radiance: [1.0; 3],
        };
        let fwd = reconnection_jacobian(a, b, &sample);
        let bwd = reconnection_jacobian(b, a, &sample);
        assert!(approx(fwd * bwd, 1.0, 1e-4), "{fwd} * {bwd}");
    }

    #[test]
    fn jacobian_matches_hand_computed_value() {
        // Sample point at origin, normal +z. src straight above at z=2
        // (cosθ=1, d²=4); dst offset so direction is (3,0,4)/5, cosθ=4/5,
        // d²=25. J = (0.8/25) / (1/4) = 0.032 * 4 = 0.128.
        let sample = GiSample {
            sample_point: [0.0, 0.0, 0.0],
            sample_normal: [0.0, 0.0, 1.0],
            radiance: [1.0; 3],
        };
        let src = ShadingPoint::new([0.0, 0.0, 2.0], [0.0, 0.0, -1.0]);
        let dst = ShadingPoint::new([3.0, 0.0, 4.0], [0.0, 0.0, -1.0]);
        let j = reconnection_jacobian(src, dst, &sample);
        assert!(approx(j, 0.128, 1e-4), "jacobian {j}");
    }

    #[test]
    fn jacobian_rejects_degenerate_reconnection() {
        let sample = GiSample {
            sample_point: [0.0, 0.0, 0.0],
            sample_normal: [0.0, 0.0, 1.0],
            radiance: [1.0; 3],
        };
        // Grazing sample normal at src (direction perpendicular to normal).
        let src = ShadingPoint::new([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let dst = ShadingPoint::new([0.0, 0.0, 2.0], [0.0, 0.0, -1.0]);
        assert_eq!(reconnection_jacobian(src, dst, &sample), 0.0);
    }

    // ---- A small discretized area emitter used by the estimator tests. ----

    const H: f32 = 2.0; // emitter plane height
    const GRID: usize = 24; // emitter is GRID x GRID points over [-1,1]^2
    const AREA_TOTAL: f32 = 4.0; // (2 x 2)

    fn emitter_point(ix: usize, iy: usize) -> [f32; 3] {
        let step = 2.0 / GRID as f32;
        let x = -1.0 + (ix as f32 + 0.5) * step;
        let y = -1.0 + (iy as f32 + 0.5) * step;
        [x, y, H]
    }

    const EMITTER_NORMAL: [f32; 3] = [0.0, 0.0, -1.0];

    /// Cosine at the shading point normal toward a sample point.
    fn cos_shade(sp: ShadingPoint, point: [f32; 3]) -> f32 {
        let to = sub3(point, sp.position);
        let d = dot3(to, to).sqrt();
        abs_f32(dot3(sp.normal, to)) / d
    }

    /// Cosine at the emitter normal toward a shading point.
    fn cos_emitter(point: [f32; 3], sp: ShadingPoint) -> f32 {
        let to = sub3(sp.position, point);
        let d = dot3(to, to).sqrt();
        abs_f32(dot3(EMITTER_NORMAL, to)) / d
    }

    fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
        let d = sub3(a, b);
        dot3(d, d)
    }

    /// Target function `p̂` at a shading point for a sample point: the
    /// solid-angle integrand `L · cosθ_shade` (radiance luminance = 1 here).
    fn target_pdf_at(sp: ShadingPoint, point: [f32; 3]) -> f32 {
        cos_shade(sp, point)
    }

    /// Brute-force reference irradiance (solid-angle integral) at `sp`.
    fn reference_irradiance(sp: ShadingPoint) -> f32 {
        let da = AREA_TOTAL / (GRID * GRID) as f32;
        let mut sum = 0.0;
        for ix in 0..GRID {
            for iy in 0..GRID {
                let p = emitter_point(ix, iy);
                let d2 = dist2(p, sp.position);
                let dw = da * cos_emitter(p, sp) / d2; // solid angle subtended
                sum += target_pdf_at(sp, p) * dw;
            }
        }
        sum
    }

    /// Builds a finalized reservoir at `sp` by RIS over emitter points sampled
    /// uniformly by index (area-uniform), with the source pdf expressed in the
    /// shading point's solid-angle measure.
    fn build_reservoir(sp: ShadingPoint, m: u32, rng: &mut Rng) -> GiReservoir {
        // Uniform area density over the emitter: p_A = 1 / AREA_TOTAL.
        let p_area = 1.0 / AREA_TOTAL;
        let mut r = GiReservoir::empty();
        for _ in 0..m {
            let ix = (rng.next_u32() as usize) % GRID;
            let iy = (rng.next_u32() as usize) % GRID;
            let point = emitter_point(ix, iy);
            let d2 = dist2(point, sp.position);
            // Convert area density to solid-angle density at the shading point.
            let source_pdf = p_area * d2 / cos_emitter(point, sp);
            let sample = GiSample {
                sample_point: point,
                sample_normal: EMITTER_NORMAL,
                radiance: [1.0; 3],
            };
            r.stream(
                GiCandidate {
                    sample,
                    target_pdf: target_pdf_at(sp, point),
                    source_pdf,
                },
                rng.next_u01(),
            );
        }
        r.finalize();
        r
    }

    #[test]
    fn initial_ris_is_unbiased() {
        let sp = ShadingPoint::new([0.1, -0.2, 0.0], [0.0, 0.0, 1.0]);
        let exact = reference_irradiance(sp);
        let m = 16u32;
        let seeds = 200_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(2_654_435_761).wrapping_add(1));
            let r = build_reservoir(sp, m, &mut rng);
            acc += f64::from(r.target_pdf * r.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel = abs_f32(mean - exact) / exact;
        assert!(rel < 0.02, "mean {mean} vs exact {exact} ({rel})");
    }

    #[test]
    fn reused_reservoir_retargets_to_destination() {
        // The reconnection Jacobian must make a reservoir built at `src` an
        // unbiased estimator of the irradiance at `dst`, not at `src`.
        let src = ShadingPoint::new([-0.6, -0.3, 0.0], [0.0, 0.0, 1.0]);
        let dst = ShadingPoint::new([0.8, 0.6, 0.0], [0.0, 0.0, 1.0]);
        let exact_dst = reference_irradiance(dst);
        let exact_src = reference_irradiance(src);
        // Sanity: the two differ, so retargeting is actually being tested.
        assert!(abs_f32(exact_dst - exact_src) / exact_src > 0.05);

        let m = 16u32;
        let seeds = 400_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(40_503).wrapping_add(7));
            let src_res = build_reservoir(src, m, &mut rng);
            let source = GiReuseSource::new(src_res, src);
            let combined = combine_unbiased(
                &[source],
                dst,
                |sp: &ShadingPoint, sample: &GiSample| target_pdf_at(*sp, sample.sample_point),
                &mut rng,
            );
            acc += f64::from(combined.target_pdf * combined.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel = abs_f32(mean - exact_dst) / exact_dst;
        assert!(
            rel < 0.03,
            "retargeted mean {mean} vs dst {exact_dst} (src {exact_src}, rel {rel})"
        );
    }

    #[test]
    fn unbiased_combine_of_two_pixels_converges() {
        let center = ShadingPoint::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let neighbor = ShadingPoint::new([0.5, -0.4, 0.0], [0.0, 0.0, 1.0]);
        let exact = reference_irradiance(center);
        let m = 12u32;
        let seeds = 400_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(2_246_822_519).wrapping_add(3));
            let c = GiReuseSource::new(build_reservoir(center, m, &mut rng), center);
            let n = GiReuseSource::new(build_reservoir(neighbor, m, &mut rng), neighbor);
            let combined = combine_unbiased(
                &[c, n],
                center,
                |sp: &ShadingPoint, sample: &GiSample| target_pdf_at(*sp, sample.sample_point),
                &mut rng,
            );
            acc += f64::from(combined.target_pdf * combined.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel = abs_f32(mean - exact) / exact;
        assert!(rel < 0.03, "mean {mean} vs exact {exact} ({rel})");
    }

    #[test]
    fn cap_history_clamps_count_only() {
        let sp = ShadingPoint::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let mut rng = Rng::new(11);
        let mut r = build_reservoir(sp, 200, &mut rng);
        let w_sum = r.w_sum;
        r.cap_history(32);
        assert_eq!(r.m, 32);
        assert!(approx(r.w_sum, w_sum, 1e-6));
    }

    #[test]
    fn empty_reservoir_is_empty() {
        let r = GiReservoir::empty();
        assert!(r.is_empty());
        let mut r2 = r;
        r2.finalize();
        assert_eq!(r2.w, 0.0);
    }

    #[test]
    fn stream_respects_initial_budget() {
        let sp = ShadingPoint::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let sample = GiSample {
            sample_point: [0.0, 0.0, 2.0],
            sample_normal: EMITTER_NORMAL,
            radiance: [1.0; 3],
        };
        let cands = [GiCandidate {
            sample,
            target_pdf: target_pdf_at(sp, sample.sample_point),
            source_pdf: 1.0,
        }; 6];
        let mut rng = Rng::new(5);
        let r = stream_initial(&cands, budget(3, 0, false), &mut rng);
        assert!(r.m <= 3 && r.m >= 1);
    }

    #[test]
    fn mis_combine_with_nontrivial_jacobian_converges() {
        // Balance-heuristic MIS over a center and a neighbor at *different*
        // shading points must stay unbiased: the per-source reconnection
        // Jacobian retargets each held sample to the canonical center domain,
        // so the estimate converges to the center's irradiance even though the
        // neighbor subtends the emitter at a different solid angle.
        let center = ShadingPoint::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let neighbor = ShadingPoint::new([0.6, -0.5, 0.0], [0.0, 0.0, 1.0]);
        let exact = reference_irradiance(center);
        // Sanity: the neighbor's own irradiance differs, so the Jacobian is
        // doing real retargeting work.
        assert!(abs_f32(reference_irradiance(neighbor) - exact) / exact > 0.05);

        let m = 12u32;
        let seeds = 600_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(2_654_435_761).wrapping_add(13));
            let c = GiReuseSource::new(build_reservoir(center, m, &mut rng), center);
            let n = GiReuseSource::new(build_reservoir(neighbor, m, &mut rng), neighbor);
            let combined = combine_mis(
                &[c, n],
                center,
                |sp: &ShadingPoint, sample: &GiSample| target_pdf_at(*sp, sample.sample_point),
                &mut rng,
            );
            acc += f64::from(combined.target_pdf * combined.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel = abs_f32(mean - exact) / exact;
        assert!(rel < 0.03, "mis mean {mean} vs exact {exact} ({rel})");
    }

    #[test]
    fn mis_combine_has_no_higher_variance_than_z_norm() {
        // Both combines are unbiased, so their estimator means agree; the
        // balance heuristic is the reference minimum-variance MIS weighting, so
        // under neighbors with a non-trivial reconnection Jacobian its per-seed
        // estimator spread must not exceed the 1/Z path. Common random numbers
        // (shared sources per seed) keep the comparison low-noise.
        let center = ShadingPoint::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let neighbor = ShadingPoint::new([0.6, -0.5, 0.0], [0.0, 0.0, 1.0]);
        let m = 12u32;
        let seeds = 400_000u32;
        let mut sum_z = 0.0f64;
        let mut sumsq_z = 0.0f64;
        let mut sum_m = 0.0f64;
        let mut sumsq_m = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(2_246_822_519).wrapping_add(17));
            let c = GiReuseSource::new(build_reservoir(center, m, &mut rng), center);
            let n = GiReuseSource::new(build_reservoir(neighbor, m, &mut rng), neighbor);
            let sources = [c, n];
            let mut rng_z = Rng::new(s.wrapping_mul(747_796_405).wrapping_add(1));
            let mut rng_m = Rng::new(s.wrapping_mul(747_796_405).wrapping_add(1));
            let target =
                |sp: &ShadingPoint, sample: &GiSample| target_pdf_at(*sp, sample.sample_point);
            let cz = combine_unbiased(&sources, center, target, &mut rng_z);
            let cm = combine_mis(&sources, center, target, &mut rng_m);
            let ez = f64::from(cz.target_pdf * cz.w);
            let em = f64::from(cm.target_pdf * cm.w);
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
