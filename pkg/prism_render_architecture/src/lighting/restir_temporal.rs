//! High-level `ReSTIR` DI resolve: motion-reprojected temporal reuse chained
//! with spatial reuse into a single per-pixel entry point.
//!
//! [`super::restir_di`] supplies the reservoir primitives (initial RIS,
//! `M`-capping, biased / unbiased combines) but leaves the frame-to-frame
//! orchestration to the caller. This module is that orchestration: given a
//! pixel's current surface, its freshly drawn light candidates, the reservoir
//! reprojected from last frame, and a handful of this frame's spatial neighbors,
//! [`resolve_di`] runs the canonical `ReSTIR` DI pipeline and returns one
//! finalized reservoir ready for shading.
//!
//! It stays a **CPU contract**: the actual motion-vector texture fetch (which
//! previous-frame texel maps to this pixel) is a GPU operation, so the caller
//! performs the addressing and hands us the reprojected history reservoir plus
//! the surface it came from. Our job is the part that must match the GPU twin
//! bit-for-bit: deciding whether that history is geometrically admissible
//! (depth / normal consistency, i.e. disocclusion rejection) and folding the
//! admissible reservoirs together with unbiased normalization.
//!
//! Pure classical Monte Carlo — no neural, learned, or data-driven components.
//! Determinism comes from the stateless [`Rng`]; a given seed reproduces the
//! same selection, which the golden tests and a future GPU kernel rely on.
//!
//! # Pipeline (per pixel, per frame)
//! 1. **Initial RIS.** [`super::restir_di::stream_initial`] over the candidate
//!    budget, then finalize — the single-frame estimate for this pixel.
//! 2. **Temporal reuse.** If `budget.temporal_reuse` and the reprojected
//!    history passes [`reproject_history`] (same surface, not a disocclusion),
//!    `M`-cap it and unbiased-combine it with the initial reservoir, re-testing
//!    each sample's target function at the relevant surface.
//! 3. **Spatial reuse.** Unbiased-combine the temporal result with up to
//!    `budget.spatial_neighbors` neighbor reservoirs, again re-testing targets
//!    at each neighbor's surface so partial visibility stays unbiased.
//!
//! The result is finalized: shaded radiance is `contribution(y) · W` for the
//! single held light `y`.

use alloc::vec::Vec;

use super::restir_di::{combine_mis, stream_initial, DiCandidate, DiReservoir};
use super::ReservoirBudget;
use crate::particle::reservoir_sample::Rng;

/// Default `M` cap for reprojected history (≈20× a typical candidate budget):
/// large enough that history dominates the per-frame noise, small enough that
/// the reservoir re-adapts within a few frames after motion.
pub const DEFAULT_MAX_HISTORY_M: u32 = 500;

/// Default relative view-depth tolerance for accepting reprojected history.
/// History is rejected when `|z_prev − z_curr| > tol · z_curr`.
pub const DEFAULT_DEPTH_REL_TOLERANCE: f32 = 0.1;

/// Default minimum normal agreement (cosine) for accepting history, ≈25°.
pub const DEFAULT_NORMAL_COS_TOLERANCE: f32 = 0.906;

/// Branchless `f32` magnitude (this crate is `no_std`; avoid relying on the
/// `std` `f32::abs` surface so the reference stays portable to the GPU twin).
fn abs_f32(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Dot product of two 3-vectors (plain multiply / add — no transcendentals).
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The surface a reservoir belongs to, used to re-evaluate target functions
/// during reuse and to validate temporal reprojection.
///
/// `view_depth` is linear (positive) view-space depth; `normal` is a unit
/// surface normal in a consistent space (view or world) shared across the
/// reservoirs being combined. These are exactly the quantities a deferred
/// renderer already has in its G-buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceGeometry {
    /// Linear view-space depth of the surface (must be `> 0` to be in front of
    /// the camera).
    pub view_depth: f32,
    /// Unit surface normal in the space shared by the combined reservoirs.
    pub normal: [f32; 3],
}

impl SurfaceGeometry {
    /// Builds a surface descriptor from a view depth and (assumed unit) normal.
    #[must_use]
    pub const fn new(view_depth: f32, normal: [f32; 3]) -> Self {
        Self { view_depth, normal }
    }

    /// Whether this surface is a real, in-front-of-camera hit (finite positive
    /// depth). Background / sky texels report a non-positive depth and never
    /// reproject.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.view_depth > 0.0 && self.view_depth.is_finite()
    }
}

/// A finalized reservoir paired with the surface it was produced on.
///
/// This is what temporal history and spatial neighbors are supplied as: the
/// reservoir plus enough geometry to decide admissibility and re-weight the
/// held sample at the destination pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeomReservoir {
    /// The finalized reservoir (its `W` is already set).
    pub reservoir: DiReservoir,
    /// The surface this reservoir was computed on.
    pub geometry: SurfaceGeometry,
}

impl GeomReservoir {
    /// Pairs a reservoir with its surface.
    #[must_use]
    pub const fn new(reservoir: DiReservoir, geometry: SurfaceGeometry) -> Self {
        Self {
            reservoir,
            geometry,
        }
    }
}

/// Tunables for the temporal / reprojection stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalParams {
    /// Upper bound on the reprojected history's sample count `M`.
    pub max_history_m: u32,
    /// Relative view-depth tolerance for accepting reprojected history.
    pub depth_rel_tolerance: f32,
    /// Minimum normal agreement (cosine) for accepting reprojected history.
    pub normal_cos_tolerance: f32,
}

impl Default for TemporalParams {
    fn default() -> Self {
        Self {
            max_history_m: DEFAULT_MAX_HISTORY_M,
            depth_rel_tolerance: DEFAULT_DEPTH_REL_TOLERANCE,
            normal_cos_tolerance: DEFAULT_NORMAL_COS_TOLERANCE,
        }
    }
}

/// Validates and `M`-caps a reprojected history reservoir for the current
/// pixel, returning `None` on a disocclusion (geometry mismatch) or empty
/// history.
///
/// `history` is the reservoir the caller fetched from last frame via the motion
/// vector, together with the surface it was produced on; `current` is this
/// pixel's surface. The sample is admissible when both surfaces are valid, their
/// view depths agree to within `params.depth_rel_tolerance` (relative to the
/// current depth), and their normals agree to within `params.normal_cos_tolerance`.
/// On success the returned reservoir has its `M` clamped to
/// `params.max_history_m` (weights untouched) so stale history cannot pin the
/// pixel to an outdated light.
#[must_use]
pub fn reproject_history(
    history: GeomReservoir,
    current: SurfaceGeometry,
    params: TemporalParams,
) -> Option<DiReservoir> {
    if history.reservoir.is_empty() {
        return None;
    }
    if !current.is_valid() || !history.geometry.is_valid() {
        return None;
    }

    let depth_diff = abs_f32(history.geometry.view_depth - current.view_depth);
    if depth_diff > params.depth_rel_tolerance * current.view_depth {
        return None;
    }

    if dot3(history.geometry.normal, current.normal) < params.normal_cos_tolerance {
        return None;
    }

    let mut reservoir = history.reservoir;
    reservoir.cap_history(params.max_history_m);
    Some(reservoir)
}

/// Runs the full `ReSTIR` DI resolve for one pixel and returns a finalized
/// reservoir.
///
/// * `current` — the pixel's surface (depth / normal).
/// * `candidates` — tentative lights for the initial RIS pass (see
///   [`super::restir_di::DiCandidate`]); only the first
///   `budget.initial_candidates` are used.
/// * `history` — the reservoir reprojected from last frame via the motion
///   vector, or `None` on a first frame / off-screen reprojection. Reuse is
///   skipped entirely unless `budget.temporal_reuse` is set.
/// * `spatial` — this frame's neighbor reservoirs; up to
///   `budget.spatial_neighbors` non-empty ones are folded in.
/// * `target` — the target function `p̂`: `target(surface, light)` returns the
///   (unshadowed) contribution estimate of `light` at `surface`. It is
///   re-evaluated at the relevant surface for every reuse source so partial
///   visibility stays unbiased.
///
/// All combines use balance-heuristic multiple importance sampling (generalized
/// `RIS`), so the result is an unbiased, minimum-variance estimator of the full
/// many-light sum at `current` even when history and neighbors see different
/// lights.
#[must_use]
pub fn resolve_di<T>(
    current: SurfaceGeometry,
    candidates: &[DiCandidate],
    history: Option<GeomReservoir>,
    spatial: &[GeomReservoir],
    budget: ReservoirBudget,
    params: TemporalParams,
    mut target: T,
    rng: &mut Rng,
) -> DiReservoir
where
    T: FnMut(&SurfaceGeometry, u32) -> f32,
{
    // 1. Initial RIS at the current pixel.
    let mut initial = stream_initial(candidates, budget, rng);
    initial.finalize();

    // 2. Temporal reuse: fold in reprojected, admissible history.
    let temporal = match (budget.temporal_reuse, history) {
        (true, Some(hist)) => match reproject_history(hist, current, params) {
            Some(prev) => {
                let sources = [initial, prev];
                let geoms = [current, hist.geometry];
                combine_mis(&sources, 0, |i, light| target(&geoms[i], light), rng)
            }
            None => initial,
        },
        _ => initial,
    };

    // 3. Spatial reuse: fold in this frame's neighbor reservoirs.
    let take = (budget.spatial_neighbors as usize).min(spatial.len());
    if take == 0 {
        return temporal;
    }

    let mut sources: Vec<DiReservoir> = Vec::with_capacity(1 + take);
    let mut geoms: Vec<SurfaceGeometry> = Vec::with_capacity(1 + take);
    sources.push(temporal);
    geoms.push(current);
    for neighbor in &spatial[..take] {
        if neighbor.reservoir.is_empty() {
            continue;
        }
        sources.push(neighbor.reservoir);
        geoms.push(neighbor.geometry);
    }

    if sources.len() == 1 {
        return temporal;
    }

    combine_mis(&sources, 0, |i, light| target(&geoms[i], light), rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRONT: [f32; 3] = [0.0, 0.0, 1.0];

    fn budget(initial: u16, spatial: u16, temporal: bool) -> ReservoirBudget {
        ReservoirBudget {
            initial_candidates: initial,
            spatial_neighbors: spatial,
            temporal_reuse: temporal,
        }
    }

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
    fn reproject_accepts_matching_surface() {
        let hist = GeomReservoir::new(
            {
                let mut r = DiReservoir::empty();
                r.stream(
                    DiCandidate {
                        light_index: 2,
                        target_pdf: 1.0,
                        source_pdf: 1.0,
                    },
                    0.5,
                );
                r.finalize();
                r
            },
            SurfaceGeometry::new(10.0, FRONT),
        );
        let current = SurfaceGeometry::new(10.3, FRONT);
        assert!(reproject_history(hist, current, TemporalParams::default()).is_some());
    }

    #[test]
    fn reproject_rejects_depth_discontinuity() {
        let mut r = DiReservoir::empty();
        r.stream(
            DiCandidate {
                light_index: 2,
                target_pdf: 1.0,
                source_pdf: 1.0,
            },
            0.5,
        );
        r.finalize();
        let hist = GeomReservoir::new(r, SurfaceGeometry::new(10.0, FRONT));
        // A far-away foreground surface: large relative depth jump = disocclusion.
        let current = SurfaceGeometry::new(50.0, FRONT);
        assert!(reproject_history(hist, current, TemporalParams::default()).is_none());
    }

    #[test]
    fn reproject_rejects_normal_flip() {
        let mut r = DiReservoir::empty();
        r.stream(
            DiCandidate {
                light_index: 2,
                target_pdf: 1.0,
                source_pdf: 1.0,
            },
            0.5,
        );
        r.finalize();
        let hist = GeomReservoir::new(r, SurfaceGeometry::new(10.0, FRONT));
        let current = SurfaceGeometry::new(10.0, [1.0, 0.0, 0.0]); // 90° apart
        assert!(reproject_history(hist, current, TemporalParams::default()).is_none());
    }

    #[test]
    fn reproject_rejects_empty_history() {
        let hist = GeomReservoir::new(DiReservoir::empty(), SurfaceGeometry::new(10.0, FRONT));
        let current = SurfaceGeometry::new(10.0, FRONT);
        assert!(reproject_history(hist, current, TemporalParams::default()).is_none());
    }

    #[test]
    fn reproject_caps_history_count() {
        let mut r = DiReservoir::empty();
        for _ in 0..1000 {
            r.stream(
                DiCandidate {
                    light_index: 1,
                    target_pdf: 1.0,
                    source_pdf: 1.0,
                },
                0.5,
            );
        }
        r.finalize();
        let hist = GeomReservoir::new(r, SurfaceGeometry::new(10.0, FRONT));
        let params = TemporalParams {
            max_history_m: 32,
            ..TemporalParams::default()
        };
        let capped = reproject_history(hist, SurfaceGeometry::new(10.0, FRONT), params).unwrap();
        assert_eq!(capped.reservoir.m, 32);
    }

    #[test]
    fn resolve_without_reuse_matches_initial() {
        let contribution = [1.0, 2.0, 3.0, 0.5, 4.0];
        let source = [1.0, 2.0, 1.0, 3.0, 1.0];
        let geom = SurfaceGeometry::new(5.0, FRONT);

        let mut a = Rng::new(99);
        let cands = sampled_candidates(&contribution, &source, 5, &mut a);
        let mut expected = stream_initial(&cands, budget(5, 0, false), &mut a);
        expected.finalize();

        let mut b = Rng::new(99);
        let cands_b = sampled_candidates(&contribution, &source, 5, &mut b);
        let target = |_g: &SurfaceGeometry, light: u32| contribution[light as usize];
        let got = resolve_di(
            geom,
            &cands_b,
            None,
            &[],
            budget(5, 0, false),
            TemporalParams::default(),
            target,
            &mut b,
        );
        assert_eq!(got, expected);
    }

    #[test]
    fn resolve_temporal_reuse_is_unbiased() {
        // Averaging contribution(y) * W over many frames must converge to the
        // full many-light sum even though the estimate folds in reprojected
        // history on an identical surface each frame.
        let contribution = [0.5, 1.4, 2.2, 0.9, 3.1, 1.0];
        let source_pmf = [2.0, 1.0, 3.0, 1.0, 2.0, 1.0];
        let exact: f32 = contribution.iter().sum();
        let geom = SurfaceGeometry::new(8.0, FRONT);
        let target = |_g: &SurfaceGeometry, light: u32| contribution[light as usize];
        let m = 8u32;
        let b = budget(m as u16, 0, true);

        let seeds = 400_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(2_654_435_761).wrapping_add(1));

            // Previous frame: an initial-only resolve on the same surface.
            let prev_cands = sampled_candidates(&contribution, &source_pmf, m, &mut rng);
            let prev = resolve_di(
                geom,
                &prev_cands,
                None,
                &[],
                b,
                TemporalParams::default(),
                target,
                &mut rng,
            );
            let history = GeomReservoir::new(prev, geom);

            // Current frame: fresh candidates + reprojected history.
            let cands = sampled_candidates(&contribution, &source_pmf, m, &mut rng);
            let out = resolve_di(
                geom,
                &cands,
                Some(history),
                &[],
                b,
                TemporalParams::default(),
                target,
                &mut rng,
            );
            acc += f64::from(out.target_pdf * out.reservoir.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel_err = (mean - exact).abs() / exact;
        assert!(rel_err < 0.03, "mean {mean} vs exact {exact} ({rel_err})");
    }

    #[test]
    fn resolve_full_pipeline_is_unbiased() {
        // Temporal + spatial reuse together, neighbors on matching surfaces.
        let contribution = [0.6, 1.2, 2.4, 0.8, 2.9, 1.1, 0.7];
        let source_pmf = [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0];
        let exact: f32 = contribution.iter().sum();
        let geom = SurfaceGeometry::new(12.0, FRONT);
        let target = |_g: &SurfaceGeometry, light: u32| contribution[light as usize];
        let m = 8u32;
        let b = budget(m as u16, 2, true);

        let seeds = 500_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(40_503).wrapping_add(7));

            let mk_initial = |rng: &mut Rng| {
                let cands = sampled_candidates(&contribution, &source_pmf, m, rng);
                let mut r = stream_initial(&cands, budget(m as u16, 0, false), rng);
                r.finalize();
                GeomReservoir::new(r, geom)
            };

            let history = mk_initial(&mut rng);
            let n0 = mk_initial(&mut rng);
            let n1 = mk_initial(&mut rng);

            let cands = sampled_candidates(&contribution, &source_pmf, m, &mut rng);
            let out = resolve_di(
                geom,
                &cands,
                Some(history),
                &[n0, n1],
                b,
                TemporalParams::default(),
                target,
                &mut rng,
            );
            acc += f64::from(out.target_pdf * out.reservoir.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel_err = (mean - exact).abs() / exact;
        assert!(rel_err < 0.03, "mean {mean} vs exact {exact} ({rel_err})");
    }

    #[test]
    fn resolve_skips_history_on_disocclusion() {
        // A mismatched history surface must be dropped, leaving the initial-only
        // result (same rng path as temporal_reuse=false after reprojection fails
        // before consuming rng).
        let contribution = [1.0, 2.0, 3.0];
        let source = [1.0, 1.0, 1.0];
        let near = SurfaceGeometry::new(5.0, FRONT);
        let far = SurfaceGeometry::new(100.0, FRONT);
        let target = |_g: &SurfaceGeometry, light: u32| contribution[light as usize];

        let mut r = DiReservoir::empty();
        r.stream(
            DiCandidate {
                light_index: 0,
                target_pdf: 1.0,
                source_pdf: 1.0,
            },
            0.5,
        );
        r.finalize();
        let stale = GeomReservoir::new(r, far);

        let mut a = Rng::new(55);
        let cands_a = sampled_candidates(&contribution, &source, 4, &mut a);
        let with_stale = resolve_di(
            near,
            &cands_a,
            Some(stale),
            &[],
            budget(4, 0, true),
            TemporalParams::default(),
            target,
            &mut a,
        );

        let mut b = Rng::new(55);
        let cands_b = sampled_candidates(&contribution, &source, 4, &mut b);
        let no_history = resolve_di(
            near,
            &cands_b,
            None,
            &[],
            budget(4, 0, true),
            TemporalParams::default(),
            target,
            &mut b,
        );
        assert_eq!(with_stale, no_history);
    }
}
