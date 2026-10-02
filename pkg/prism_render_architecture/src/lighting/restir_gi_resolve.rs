//! High-level `ReSTIR` GI resolve: motion-reprojected temporal reuse chained
//! with spatial reuse into a single per-pixel entry point.
//!
//! [`super::restir_gi`] supplies the GI reservoir primitives (initial RIS over
//! sample points, `M`-capping, biased / unbiased combines carrying the
//! reconnection Jacobian) but leaves the frame-to-frame orchestration to the
//! caller. This module is that orchestration, the GI analogue of
//! [`super::restir_temporal`]: given a pixel's current surface, its freshly
//! traced indirect-path candidates, the reservoir reprojected from last frame,
//! and a handful of this frame's spatial neighbors, [`resolve_gi`] runs the
//! canonical `ReSTIR` GI pipeline and returns one finalized reservoir ready for
//! shading.
//!
//! Like the DI resolve, it stays a **CPU contract**: the motion-vector texture
//! fetch (which previous-frame texel feeds this pixel) is a GPU operation, so
//! the caller performs the addressing and hands us the reprojected history
//! reservoir plus the surface it came from. Our job is the part that must match
//! the GPU twin bit-for-bit: deciding whether that history is geometrically
//! admissible (depth / normal consistency, i.e. disocclusion rejection) and
//! folding the admissible reservoirs together with unbiased normalization. The
//! change-of-measure between shading points — the piece unique to GI — lives in
//! [`super::restir_gi::reconnection_jacobian`] and is applied inside every
//! combine, so no `+1`-frame latency is introduced here: history is consumed
//! the same frame it is reprojected.
//!
//! Pure classical Monte Carlo — no neural, learned, or data-driven components.
//! Determinism comes from the stateless [`Rng`]; a given seed reproduces the
//! same selection, which the golden tests and a future GPU kernel rely on.
//!
//! # Pipeline (per pixel, per frame)
//! 1. **Initial RIS.** [`super::restir_gi::stream_initial`] over the candidate
//!    budget, then finalize — the single-frame indirect estimate for this
//!    pixel.
//! 2. **Temporal reuse.** If `budget.temporal_reuse` and the reprojected
//!    history passes [`reproject_gi_history`] (same surface, not a
//!    disocclusion), `M`-cap it and unbiased-combine it with the initial
//!    reservoir. The combine re-evaluates each sample's target function at the
//!    destination shading point and multiplies the history weight by the
//!    reconnection Jacobian, so the reused sample points retarget to *this*
//!    pixel's geometry.
//! 3. **Spatial reuse.** Unbiased-combine the temporal result with up to
//!    `budget.spatial_neighbors` neighbor reservoirs, again retargeting each
//!    neighbor's sample point to this pixel via the Jacobian so partial
//!    visibility and differing geometry stay unbiased.
//!
//! The result is finalized: shaded indirect radiance is `L_o(y) · W` for the
//! single held sample point `y`.

use alloc::vec::Vec;

use super::restir_gi::{
    combine_unbiased, gather_spatial_sources, stream_initial, GiCandidate, GiReservoir,
    GiReuseSource, GiSample, ShadingPoint,
};
use super::ReservoirBudget;
use crate::particle::reservoir_sample::Rng;

/// Default `M` cap for reprojected GI history.
///
/// Lower than the DI cap: indirect lighting responds to moving occluders and
/// emitters, so an over-long history lags visibly (ghosted bounce light). This
/// keeps roughly a third of a second of accumulation at 60 Hz while still
/// re-adapting within a few frames after a disocclusion.
pub const DEFAULT_GI_MAX_HISTORY_M: u32 = 30;

/// Default relative view-depth tolerance for accepting reprojected history.
/// History is rejected when `|z_prev − z_curr| > tol · z_curr`.
pub const DEFAULT_GI_DEPTH_REL_TOLERANCE: f32 = 0.1;

/// Default minimum normal agreement (cosine) for accepting history, ≈25°.
pub const DEFAULT_GI_NORMAL_COS_TOLERANCE: f32 = 0.906;

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

/// The surface a GI reservoir belongs to: the shading point used for the
/// reconnection Jacobian and target re-evaluation, plus a linear view-space
/// depth used only for temporal disocclusion rejection.
///
/// `view_depth` is linear (positive) view-space depth; `shading_point` carries
/// the world-space position and unit normal. These are exactly the quantities a
/// deferred renderer already has in its G-buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiSurface {
    /// Linear view-space depth of the surface (must be `> 0` to be in front of
    /// the camera).
    pub view_depth: f32,
    /// The world-space shading point (position + unit normal).
    pub shading_point: ShadingPoint,
}

impl GiSurface {
    /// Builds a surface descriptor from a view depth and shading point.
    #[must_use]
    pub const fn new(view_depth: f32, shading_point: ShadingPoint) -> Self {
        Self {
            view_depth,
            shading_point,
        }
    }

    /// Whether this surface is a real, in-front-of-camera hit (finite positive
    /// depth). Background / sky texels report a non-positive depth and never
    /// reproject.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.view_depth > 0.0 && self.view_depth.is_finite()
    }
}

/// A finalized GI reservoir paired with the surface it was produced on.
///
/// This is what temporal history and spatial neighbors are supplied as: the
/// reservoir plus enough geometry to decide admissibility and to re-anchor the
/// held sample point at the destination pixel via the reconnection Jacobian.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiGeomReservoir {
    /// The finalized reservoir (its `W` is already set).
    pub reservoir: GiReservoir,
    /// The surface this reservoir was computed on.
    pub surface: GiSurface,
}

impl GiGeomReservoir {
    /// Pairs a reservoir with its surface.
    #[must_use]
    pub const fn new(reservoir: GiReservoir, surface: GiSurface) -> Self {
        Self { reservoir, surface }
    }

    /// Repackages this as a [`GiReuseSource`] for the combine primitives,
    /// dropping the depth (needed only for the disocclusion gate).
    #[must_use]
    fn as_source(&self) -> GiReuseSource {
        GiReuseSource::new(self.reservoir, self.surface.shading_point)
    }
}

/// Tunables for the temporal / reprojection stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiTemporalParams {
    /// Upper bound on the reprojected history's sample count `M`.
    pub max_history_m: u32,
    /// Relative view-depth tolerance for accepting reprojected history.
    pub depth_rel_tolerance: f32,
    /// Minimum normal agreement (cosine) for accepting reprojected history.
    pub normal_cos_tolerance: f32,
}

impl Default for GiTemporalParams {
    fn default() -> Self {
        Self {
            max_history_m: DEFAULT_GI_MAX_HISTORY_M,
            depth_rel_tolerance: DEFAULT_GI_DEPTH_REL_TOLERANCE,
            normal_cos_tolerance: DEFAULT_GI_NORMAL_COS_TOLERANCE,
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
/// current depth), and their normals agree to within
/// `params.normal_cos_tolerance`. On success the returned reservoir has its `M`
/// clamped to `params.max_history_m` (weights untouched) so stale history
/// cannot pin the pixel to an outdated bounce.
///
/// Note the Jacobian is **not** applied here — the shading points legitimately
/// differ under camera motion, and the retargeting happens inside the combine.
/// This gate only rejects genuine disocclusions (a different surface now
/// occupies the pixel).
#[must_use]
pub fn reproject_gi_history(
    history: GiGeomReservoir,
    current: GiSurface,
    params: GiTemporalParams,
) -> Option<GiReservoir> {
    if history.reservoir.is_empty() {
        return None;
    }
    if !current.is_valid() || !history.surface.is_valid() {
        return None;
    }

    let depth_diff = abs_f32(history.surface.view_depth - current.view_depth);
    if depth_diff > params.depth_rel_tolerance * current.view_depth {
        return None;
    }

    if dot3(
        history.surface.shading_point.normal,
        current.shading_point.normal,
    ) < params.normal_cos_tolerance
    {
        return None;
    }

    let mut reservoir = history.reservoir;
    reservoir.cap_history(params.max_history_m);
    Some(reservoir)
}

/// Default relative view-depth tolerance for accepting a GI spatial neighbor.
/// A neighbor is rejected when `|z_n − z_c| > tol · z_c`.
pub const DEFAULT_GI_SPATIAL_DEPTH_REL_TOLERANCE: f32 = 0.1;

/// Default minimum normal agreement (cosine) for accepting a GI spatial
/// neighbor, ≈25°.
pub const DEFAULT_GI_SPATIAL_NORMAL_COS_TOLERANCE: f32 = 0.906;

/// Tunables for the GI spatial-neighbor admissibility gate.
///
/// These mirror the temporal gate's tolerances but apply between two same-frame
/// pixels rather than across a reprojection. Spatial reuse stays unbiased for
/// *any* neighbor set (the combine retargets each sample via the reconnection
/// Jacobian), so this gate is purely variance reduction / quality control: it
/// drops neighbors straddling a depth silhouette or normal crease whose samples
/// would almost always be rejected at this pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiSpatialParams {
    /// Relative view-depth tolerance: a neighbor is rejected when its depth
    /// differs from the center's by more than `depth_rel_tolerance · z_center`.
    pub depth_rel_tolerance: f32,
    /// Minimum normal agreement (cosine) between center and neighbor normals.
    pub normal_cos_tolerance: f32,
}

impl Default for GiSpatialParams {
    fn default() -> Self {
        Self {
            depth_rel_tolerance: DEFAULT_GI_SPATIAL_DEPTH_REL_TOLERANCE,
            normal_cos_tolerance: DEFAULT_GI_SPATIAL_NORMAL_COS_TOLERANCE,
        }
    }
}

/// Whether a neighbor GI surface is geometrically compatible with the center
/// pixel and may be folded into its reservoir.
///
/// A neighbor is admissible when both surfaces are valid (finite, in front of
/// the camera), their view depths agree to within `params.depth_rel_tolerance`
/// (relative to the center depth), and their normals agree to within
/// `params.normal_cos_tolerance`. This is the same disocclusion-style test
/// [`reproject_gi_history`] applies to temporal history, evaluated here between
/// two same-frame pixels. It inspects surface geometry only — never the held
/// sample — so it cannot bias the estimator.
#[must_use]
pub fn gi_spatial_admissible(
    center: GiSurface,
    neighbor: GiSurface,
    params: GiSpatialParams,
) -> bool {
    if !center.is_valid() || !neighbor.is_valid() {
        return false;
    }
    let depth_diff = abs_f32(neighbor.view_depth - center.view_depth);
    if depth_diff > params.depth_rel_tolerance * center.view_depth {
        return false;
    }
    dot3(center.shading_point.normal, neighbor.shading_point.normal) >= params.normal_cos_tolerance
}

/// Selects up to `max` geometrically admissible, non-empty GI neighbors for the
/// center pixel, preserving the caller's neighbor order.
///
/// `neighbors` is this frame's candidate neighbor set (each a finalized
/// reservoir paired with its surface); the caller chose *which* screen texels
/// those are. We keep only the ones that pass [`gi_spatial_admissible`] and
/// still hold a sample, stopping once `max` have been collected (typically
/// `budget.spatial_neighbors`). The returned reservoirs are ready to fold into
/// [`resolve_gi`]'s spatial stage.
#[must_use]
pub fn gather_admissible_gi_neighbors(
    center: GiSurface,
    neighbors: &[GiGeomReservoir],
    max: usize,
    params: GiSpatialParams,
) -> Vec<GiGeomReservoir> {
    let mut out: Vec<GiGeomReservoir> = Vec::with_capacity(max.min(neighbors.len()));
    if max == 0 {
        return out;
    }
    for neighbor in neighbors {
        if out.len() >= max {
            break;
        }
        if neighbor.reservoir.is_empty() {
            continue;
        }
        if gi_spatial_admissible(center, neighbor.surface, params) {
            out.push(*neighbor);
        }
    }
    out
}

/// Runs the full `ReSTIR` GI resolve for one pixel and returns a finalized
/// reservoir.
///
/// * `current` — the pixel's surface (depth + shading point).
/// * `candidates` — tentative sample points for the initial RIS pass (see
///   [`super::restir_gi::GiCandidate`]); only the first
///   `budget.initial_candidates` are used.
/// * `history` — the reservoir reprojected from last frame via the motion
///   vector, or `None` on a first frame / off-screen reprojection. Reuse is
///   skipped entirely unless `budget.temporal_reuse` is set.
/// * `spatial` — this frame's neighbor reservoirs; up to
///   `budget.spatial_neighbors` geometrically admissible, non-empty ones are
///   folded in (screened by `spatial_params`).
/// * `spatial_params` — geometric admissibility tolerances applied to the
///   spatial neighbors before they are folded in (see [`GiSpatialParams`]).
/// * `target` — the target function `p̂`: `target(shading_point, sample)`
///   returns the indirect contribution estimate of `sample` at `shading_point`.
///   It is re-evaluated at the relevant shading point for every reuse source so
///   differing geometry / visibility stays unbiased.
///
/// All combines use the unbiased normalization and carry the reconnection
/// Jacobian, so the result is an unbiased estimator of the indirect
/// illumination at `current` even when history and neighbors anchored their
/// sample points to different shading geometry.
#[must_use]
pub fn resolve_gi<T>(
    current: GiSurface,
    candidates: &[GiCandidate],
    history: Option<GiGeomReservoir>,
    spatial: &[GiGeomReservoir],
    budget: ReservoirBudget,
    params: GiTemporalParams,
    spatial_params: GiSpatialParams,
    mut target: T,
    rng: &mut Rng,
) -> GiReservoir
where
    T: FnMut(&ShadingPoint, &GiSample) -> f32,
{
    let center = current.shading_point;

    // 1. Initial RIS at the current pixel, then finalize so it can act as a
    //    reuse source.
    let mut initial = stream_initial(candidates, budget, rng);
    initial.finalize();

    // 2. Temporal reuse: fold in reprojected, admissible history. The combine
    //    retargets the history's sample point to `center` via the Jacobian.
    let temporal = match (budget.temporal_reuse, history) {
        (true, Some(hist)) => match reproject_gi_history(hist, current, params) {
            Some(prev) => {
                let sources = [
                    GiReuseSource::new(initial, center),
                    GiReuseSource::new(prev, hist.surface.shading_point),
                ];
                combine_unbiased(&sources, center, |sp, sample| target(sp, sample), rng)
            }
            None => initial,
        },
        _ => initial,
    };

    // 3. Spatial reuse: fold in this frame's geometrically admissible
    //    neighbors. The gate screens by surface geometry only (never the held
    //    sample), so dropping disoccluded neighbors removes variance without
    //    biasing the estimator, and it runs before any spatial RNG draw.
    let admissible = gather_admissible_gi_neighbors(
        current,
        spatial,
        budget.spatial_neighbors as usize,
        spatial_params,
    );
    if admissible.is_empty() {
        return temporal;
    }

    let mut neighbors: Vec<GiReuseSource> = Vec::with_capacity(admissible.len());
    for neighbor in &admissible {
        neighbors.push(neighbor.as_source());
    }

    let sources = gather_spatial_sources(GiReuseSource::new(temporal, center), &neighbors, budget);
    combine_unbiased(&sources, center, |sp, sample| target(sp, sample), rng)
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

    fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    // ---- A small discretized area emitter shared by the estimator tests. ----
    // (Mirrors the one in `restir_gi`'s own tests so convergence targets match.)

    const H: f32 = 2.0;
    const GRID: usize = 24;
    const AREA_TOTAL: f32 = 4.0;
    const EMITTER_NORMAL: [f32; 3] = [0.0, 0.0, -1.0];

    fn emitter_point(ix: usize, iy: usize) -> [f32; 3] {
        let step = 2.0 / GRID as f32;
        let x = -1.0 + (ix as f32 + 0.5) * step;
        let y = -1.0 + (iy as f32 + 0.5) * step;
        [x, y, H]
    }

    fn cos_shade(sp: ShadingPoint, point: [f32; 3]) -> f32 {
        let to = sub3(point, sp.position);
        let d = dot3(to, to).sqrt();
        abs_f32(dot3(sp.normal, to)) / d
    }

    fn cos_emitter(point: [f32; 3], sp: ShadingPoint) -> f32 {
        let to = sub3(sp.position, point);
        let d = dot3(to, to).sqrt();
        abs_f32(dot3(EMITTER_NORMAL, to)) / d
    }

    fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
        let d = sub3(a, b);
        dot3(d, d)
    }

    fn target_pdf_at(sp: &ShadingPoint, sample: &GiSample) -> f32 {
        cos_shade(*sp, sample.sample_point)
    }

    fn reference_irradiance(sp: ShadingPoint) -> f32 {
        let da = AREA_TOTAL / (GRID * GRID) as f32;
        let mut sum = 0.0;
        for ix in 0..GRID {
            for iy in 0..GRID {
                let p = emitter_point(ix, iy);
                let d2 = dist2(p, sp.position);
                let dw = da * cos_emitter(p, sp) / d2;
                sum += cos_shade(sp, p) * dw;
            }
        }
        sum
    }

    /// Builds a finalized reservoir at `sp` by RIS over emitter points sampled
    /// area-uniformly, with the source pdf expressed in solid-angle measure.
    fn build_reservoir(sp: ShadingPoint, m: u32, rng: &mut Rng) -> GiReservoir {
        let p_area = 1.0 / AREA_TOTAL;
        let mut r = GiReservoir::empty();
        for _ in 0..m {
            let ix = (rng.next_u32() as usize) % GRID;
            let iy = (rng.next_u32() as usize) % GRID;
            let point = emitter_point(ix, iy);
            let d2 = dist2(point, sp.position);
            let source_pdf = p_area * d2 / cos_emitter(point, sp);
            let sample = GiSample {
                sample_point: point,
                sample_normal: EMITTER_NORMAL,
                radiance: [1.0; 3],
            };
            r.stream(
                GiCandidate {
                    sample,
                    target_pdf: cos_shade(sp, point),
                    source_pdf,
                },
                rng.next_u01(),
            );
        }
        r.finalize();
        r
    }

    fn candidates_at(sp: ShadingPoint, m: u32, rng: &mut Rng) -> Vec<GiCandidate> {
        let p_area = 1.0 / AREA_TOTAL;
        (0..m)
            .map(|_| {
                let ix = (rng.next_u32() as usize) % GRID;
                let iy = (rng.next_u32() as usize) % GRID;
                let point = emitter_point(ix, iy);
                let d2 = dist2(point, sp.position);
                let source_pdf = p_area * d2 / cos_emitter(point, sp);
                GiCandidate {
                    sample: GiSample {
                        sample_point: point,
                        sample_normal: EMITTER_NORMAL,
                        radiance: [1.0; 3],
                    },
                    target_pdf: cos_shade(sp, point),
                    source_pdf,
                }
            })
            .collect()
    }

    fn surface(pos: [f32; 3]) -> GiSurface {
        GiSurface::new(H - pos[2], ShadingPoint::new(pos, FRONT))
    }

    fn abs(x: f32) -> f32 {
        abs_f32(x)
    }

    // ---------- reprojection gate ----------

    #[test]
    fn reproject_accepts_matching_surface() {
        let mut rng = Rng::new(1);
        let sp = ShadingPoint::new([0.1, 0.0, 0.0], FRONT);
        let hist = GiGeomReservoir::new(build_reservoir(sp, 8, &mut rng), surface(sp.position));
        let current = GiSurface::new(hist.surface.view_depth * 1.02, sp);
        let out = reproject_gi_history(hist, current, GiTemporalParams::default());
        assert!(out.is_some());
    }

    #[test]
    fn reproject_rejects_empty_history() {
        let current = surface([0.0, 0.0, 0.0]);
        let hist = GiGeomReservoir::new(GiReservoir::empty(), current);
        assert!(reproject_gi_history(hist, current, GiTemporalParams::default()).is_none());
    }

    #[test]
    fn reproject_rejects_depth_mismatch() {
        let mut rng = Rng::new(2);
        let sp = ShadingPoint::new([0.0, 0.0, 0.0], FRONT);
        let hist = GiGeomReservoir::new(build_reservoir(sp, 8, &mut rng), surface(sp.position));
        // 50% deeper than the default 10% tolerance allows.
        let current = GiSurface::new(hist.surface.view_depth * 1.5, sp);
        assert!(reproject_gi_history(hist, current, GiTemporalParams::default()).is_none());
    }

    #[test]
    fn reproject_rejects_normal_mismatch() {
        let mut rng = Rng::new(3);
        let sp = ShadingPoint::new([0.0, 0.0, 0.0], FRONT);
        let hist = GiGeomReservoir::new(build_reservoir(sp, 8, &mut rng), surface(sp.position));
        // Current pixel faces sideways: well past the ~25° cone.
        let current = GiSurface::new(
            hist.surface.view_depth,
            ShadingPoint::new(sp.position, [1.0, 0.0, 0.0]),
        );
        assert!(reproject_gi_history(hist, current, GiTemporalParams::default()).is_none());
    }

    #[test]
    fn reproject_rejects_background_surface() {
        let mut rng = Rng::new(4);
        let sp = ShadingPoint::new([0.0, 0.0, 0.0], FRONT);
        let hist = GiGeomReservoir::new(build_reservoir(sp, 8, &mut rng), surface(sp.position));
        let background = GiSurface::new(0.0, sp); // non-positive depth = sky
        assert!(reproject_gi_history(hist, background, GiTemporalParams::default()).is_none());
    }

    #[test]
    fn reproject_caps_history_m() {
        let mut rng = Rng::new(5);
        let sp = ShadingPoint::new([0.0, 0.0, 0.0], FRONT);
        let hist = GiGeomReservoir::new(build_reservoir(sp, 400, &mut rng), surface(sp.position));
        let current = surface(sp.position);
        let params = GiTemporalParams {
            max_history_m: 30,
            ..Default::default()
        };
        let out = reproject_gi_history(hist, current, params).unwrap();
        assert_eq!(out.m, 30);
    }

    // ---------- resolve estimator ----------

    #[test]
    fn resolve_initial_only_is_unbiased() {
        let sp = ShadingPoint::new([0.1, -0.1, 0.0], FRONT);
        let current = surface(sp.position);
        let exact = reference_irradiance(sp);
        let seeds = 150_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(2_654_435_761).wrapping_add(1));
            let cands = candidates_at(sp, 16, &mut rng);
            let r = resolve_gi(
                current,
                &cands,
                None,
                &[],
                budget(16, 0, false),
                GiTemporalParams::default(),
                GiSpatialParams::default(),
                target_pdf_at,
                &mut rng,
            );
            acc += f64::from(r.target_pdf * r.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel = abs(mean - exact) / exact;
        assert!(rel < 0.02, "mean {mean} vs exact {exact} ({rel})");
    }

    #[test]
    fn resolve_with_temporal_history_stays_unbiased() {
        // History anchored to a *different* shading point must retarget to the
        // current pixel's irradiance via the Jacobian, not drag it toward the
        // history's own irradiance.
        let cur_sp = ShadingPoint::new([0.5, 0.2, 0.0], FRONT);
        let hist_sp = ShadingPoint::new([-0.8, -0.6, 0.0], FRONT);
        let current = surface(cur_sp.position);
        let exact_cur = reference_irradiance(cur_sp);
        let exact_hist = reference_irradiance(hist_sp);
        // Sanity: history differs, so retargeting is actually exercised.
        assert!(abs(exact_cur - exact_hist) / exact_cur > 0.05);

        let seeds = 400_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(40_503).wrapping_add(7));
            let hist = GiGeomReservoir::new(
                build_reservoir(hist_sp, 16, &mut rng),
                surface(hist_sp.position),
            );
            let cands = candidates_at(cur_sp, 16, &mut rng);
            let r = resolve_gi(
                current,
                &cands,
                Some(hist),
                &[],
                budget(16, 0, true),
                GiTemporalParams::default(),
                GiSpatialParams::default(),
                target_pdf_at,
                &mut rng,
            );
            acc += f64::from(r.target_pdf * r.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel = abs(mean - exact_cur) / exact_cur;
        assert!(
            rel < 0.03,
            "temporal mean {mean} vs current {exact_cur} (hist {exact_hist}, rel {rel})"
        );
    }

    #[test]
    fn resolve_full_pipeline_converges() {
        // Initial + temporal history + two spatial neighbors, all anchored to
        // distinct shading points, must still estimate the current pixel's
        // irradiance.
        let cur_sp = ShadingPoint::new([0.0, 0.0, 0.0], FRONT);
        let hist_sp = ShadingPoint::new([-0.4, 0.3, 0.0], FRONT);
        let n0 = ShadingPoint::new([0.3, -0.3, 0.0], FRONT);
        let n1 = ShadingPoint::new([-0.2, -0.4, 0.0], FRONT);
        let current = surface(cur_sp.position);
        let exact = reference_irradiance(cur_sp);

        let seeds = 400_000u32;
        let mut acc = 0.0f64;
        for s in 0..seeds {
            let mut rng = Rng::new(s.wrapping_mul(2_246_822_519).wrapping_add(3));
            let hist = GiGeomReservoir::new(
                build_reservoir(hist_sp, 16, &mut rng),
                surface(hist_sp.position),
            );
            let spatial = [
                GiGeomReservoir::new(build_reservoir(n0, 16, &mut rng), surface(n0.position)),
                GiGeomReservoir::new(build_reservoir(n1, 16, &mut rng), surface(n1.position)),
            ];
            let cands = candidates_at(cur_sp, 16, &mut rng);
            let r = resolve_gi(
                current,
                &cands,
                Some(hist),
                &spatial,
                budget(16, 2, true),
                GiTemporalParams::default(),
                GiSpatialParams::default(),
                target_pdf_at,
                &mut rng,
            );
            acc += f64::from(r.target_pdf * r.w);
        }
        let mean = (acc / f64::from(seeds)) as f32;
        let rel = abs(mean - exact) / exact;
        assert!(rel < 0.03, "pipeline mean {mean} vs exact {exact} ({rel})");
    }

    #[test]
    fn disoccluded_history_is_dropped() {
        // A history whose depth is wildly off is rejected; the resolve then
        // equals the initial-only estimate for the same seed.
        let sp = ShadingPoint::new([0.2, 0.1, 0.0], FRONT);
        let current = surface(sp.position);
        let mut rng_a = Rng::new(99);
        let bad_hist = GiGeomReservoir::new(
            build_reservoir(sp, 16, &mut rng_a),
            GiSurface::new(current.view_depth * 3.0, sp),
        );
        let mut rng1 = Rng::new(123);
        let cands1 = candidates_at(sp, 16, &mut rng1);
        let with_bad = resolve_gi(
            current,
            &cands1,
            Some(bad_hist),
            &[],
            budget(16, 0, true),
            GiTemporalParams::default(),
            GiSpatialParams::default(),
            target_pdf_at,
            &mut rng1,
        );
        let mut rng2 = Rng::new(123);
        let cands2 = candidates_at(sp, 16, &mut rng2);
        let without = resolve_gi(
            current,
            &cands2,
            None,
            &[],
            budget(16, 0, true),
            GiTemporalParams::default(),
            GiSpatialParams::default(),
            target_pdf_at,
            &mut rng2,
        );
        assert!(abs(with_bad.w - without.w) < 1e-6);
        assert_eq!(with_bad.m, without.m);
    }

    #[test]
    fn empty_candidates_yield_empty_reservoir() {
        let current = surface([0.0, 0.0, 0.0]);
        let mut rng = Rng::new(7);
        let r = resolve_gi(
            current,
            &[],
            None,
            &[],
            budget(16, 2, true),
            GiTemporalParams::default(),
            GiSpatialParams::default(),
            target_pdf_at,
            &mut rng,
        );
        assert!(r.is_empty());
        assert_eq!(r.w, 0.0);
    }

    #[test]
    fn spatial_budget_zero_skips_neighbors() {
        let sp = ShadingPoint::new([0.0, 0.0, 0.0], FRONT);
        let current = surface(sp.position);
        let mut rng_n = Rng::new(11);
        let neighbor = GiGeomReservoir::new(
            build_reservoir(ShadingPoint::new([0.4, 0.0, 0.0], FRONT), 16, &mut rng_n),
            surface([0.4, 0.0, 0.0]),
        );
        let mut rng1 = Rng::new(55);
        let cands1 = candidates_at(sp, 16, &mut rng1);
        let no_spatial = resolve_gi(
            current,
            &cands1,
            None,
            &[neighbor],
            budget(16, 0, false),
            GiTemporalParams::default(),
            GiSpatialParams::default(),
            target_pdf_at,
            &mut rng1,
        );
        let mut rng2 = Rng::new(55);
        let cands2 = candidates_at(sp, 16, &mut rng2);
        let empty_spatial = resolve_gi(
            current,
            &cands2,
            None,
            &[],
            budget(16, 0, false),
            GiTemporalParams::default(),
            GiSpatialParams::default(),
            target_pdf_at,
            &mut rng2,
        );
        assert!(abs(no_spatial.w - empty_spatial.w) < 1e-6);
    }

    #[test]
    fn default_params_are_sane() {
        let p = GiTemporalParams::default();
        assert_eq!(p.max_history_m, DEFAULT_GI_MAX_HISTORY_M);
        assert!(p.depth_rel_tolerance > 0.0 && p.depth_rel_tolerance < 1.0);
        assert!(p.normal_cos_tolerance > 0.0 && p.normal_cos_tolerance < 1.0);
    }

    // ---------- spatial admissibility gate ----------

    #[test]
    fn gi_spatial_admits_matching_surface() {
        let c = surface([0.0, 0.0, 0.0]);
        // Within 10% depth and well inside the normal cone.
        let n = GiSurface::new(
            c.view_depth * 1.05,
            ShadingPoint::new([0.3, 0.0, 0.0], FRONT),
        );
        assert!(gi_spatial_admissible(c, n, GiSpatialParams::default()));
    }

    #[test]
    fn gi_spatial_rejects_depth_silhouette() {
        let c = surface([0.0, 0.0, 0.0]);
        // 50% deeper than the default 10% tolerance allows.
        let n = GiSurface::new(
            c.view_depth * 1.5,
            ShadingPoint::new([0.3, 0.0, 0.0], FRONT),
        );
        assert!(!gi_spatial_admissible(c, n, GiSpatialParams::default()));
    }

    #[test]
    fn gi_spatial_rejects_normal_crease() {
        let c = surface([0.0, 0.0, 0.0]);
        // Orthogonal normal (90°) is far outside the ≈25° cone.
        let n = GiSurface::new(
            c.view_depth,
            ShadingPoint::new([0.3, 0.0, 0.0], [1.0, 0.0, 0.0]),
        );
        assert!(!gi_spatial_admissible(c, n, GiSpatialParams::default()));
    }

    #[test]
    fn gi_spatial_rejects_background() {
        let c = surface([0.0, 0.0, 0.0]);
        // Sky / background neighbor (non-positive depth) never reuses.
        let bg = GiSurface::new(0.0, ShadingPoint::new([0.3, 0.0, 0.0], FRONT));
        assert!(!gi_spatial_admissible(c, bg, GiSpatialParams::default()));
        // A center that is itself background rejects everything.
        assert!(!gi_spatial_admissible(
            GiSurface::new(-1.0, c.shading_point),
            surface([0.3, 0.0, 0.0]),
            GiSpatialParams::default()
        ));
    }

    #[test]
    fn gather_gi_filters_empty_and_inadmissible_and_caps() {
        let center = surface([0.0, 0.0, 0.0]);
        let mut rng = Rng::new(71);
        let good0 = GiGeomReservoir::new(
            build_reservoir(ShadingPoint::new([0.2, 0.0, 0.0], FRONT), 8, &mut rng),
            GiSurface::new(
                center.view_depth * 1.02,
                ShadingPoint::new([0.2, 0.0, 0.0], FRONT),
            ),
        );
        let empty = GiGeomReservoir::new(GiReservoir::empty(), center);
        let deep = GiGeomReservoir::new(
            build_reservoir(ShadingPoint::new([0.4, 0.0, 0.0], FRONT), 8, &mut rng),
            GiSurface::new(
                center.view_depth * 3.0,
                ShadingPoint::new([0.4, 0.0, 0.0], FRONT),
            ),
        );
        let good1 = GiGeomReservoir::new(
            build_reservoir(ShadingPoint::new([-0.2, 0.0, 0.0], FRONT), 8, &mut rng),
            GiSurface::new(
                center.view_depth * 0.98,
                ShadingPoint::new([-0.2, 0.0, 0.0], FRONT),
            ),
        );
        let neighbors = [good0, empty, deep, good1];
        let kept =
            gather_admissible_gi_neighbors(center, &neighbors, 8, GiSpatialParams::default());
        assert_eq!(kept.len(), 2);
        // Order preserved: good0 then good1 (empty and deep screened out).
        assert_eq!(kept[0].surface, good0.surface);
        assert_eq!(kept[1].surface, good1.surface);
        // Cap honored, and max = 0 selects nothing.
        let capped =
            gather_admissible_gi_neighbors(center, &neighbors, 1, GiSpatialParams::default());
        assert_eq!(capped.len(), 1);
        assert_eq!(capped[0].surface, good0.surface);
        assert!(
            gather_admissible_gi_neighbors(center, &neighbors, 0, GiSpatialParams::default())
                .is_empty()
        );
    }

    #[test]
    fn disoccluded_spatial_neighbor_is_dropped() {
        // A neighbor across a depth silhouette is screened out before any
        // spatial RNG draw, so the resolve matches the no-neighbor estimate for
        // the same seed.
        let sp = ShadingPoint::new([0.0, 0.0, 0.0], FRONT);
        let current = surface(sp.position);
        let mut rng_n = Rng::new(21);
        let bad = GiGeomReservoir::new(
            build_reservoir(ShadingPoint::new([0.3, 0.0, 0.0], FRONT), 16, &mut rng_n),
            GiSurface::new(
                current.view_depth * 3.0,
                ShadingPoint::new([0.3, 0.0, 0.0], FRONT),
            ),
        );
        let mut rng1 = Rng::new(321);
        let cands1 = candidates_at(sp, 16, &mut rng1);
        let with_bad = resolve_gi(
            current,
            &cands1,
            None,
            &[bad],
            budget(16, 4, false),
            GiTemporalParams::default(),
            GiSpatialParams::default(),
            target_pdf_at,
            &mut rng1,
        );
        let mut rng2 = Rng::new(321);
        let cands2 = candidates_at(sp, 16, &mut rng2);
        let without = resolve_gi(
            current,
            &cands2,
            None,
            &[],
            budget(16, 4, false),
            GiTemporalParams::default(),
            GiSpatialParams::default(),
            target_pdf_at,
            &mut rng2,
        );
        assert!(abs(with_bad.w - without.w) < 1e-6);
        assert_eq!(with_bad.m, without.m);
    }

    #[test]
    fn admissible_spatial_neighbor_changes_estimate() {
        // Sanity companion to the drop test: a geometrically compatible
        // neighbor is actually folded in, so the resolve differs from the
        // no-neighbor estimate for the same seed.
        let sp = ShadingPoint::new([0.0, 0.0, 0.0], FRONT);
        let current = surface(sp.position);
        let mut rng_n = Rng::new(22);
        let good = GiGeomReservoir::new(
            build_reservoir(ShadingPoint::new([0.3, 0.0, 0.0], FRONT), 16, &mut rng_n),
            surface([0.3, 0.0, 0.0]),
        );
        let mut rng1 = Rng::new(321);
        let cands1 = candidates_at(sp, 16, &mut rng1);
        let with_good = resolve_gi(
            current,
            &cands1,
            None,
            &[good],
            budget(16, 4, false),
            GiTemporalParams::default(),
            GiSpatialParams::default(),
            target_pdf_at,
            &mut rng1,
        );
        let mut rng2 = Rng::new(321);
        let cands2 = candidates_at(sp, 16, &mut rng2);
        let without = resolve_gi(
            current,
            &cands2,
            None,
            &[],
            budget(16, 4, false),
            GiTemporalParams::default(),
            GiSpatialParams::default(),
            target_pdf_at,
            &mut rng2,
        );
        assert!(with_good.m > without.m);
    }

    #[test]
    fn gi_spatial_default_params_are_sane() {
        let p = GiSpatialParams::default();
        assert!(p.depth_rel_tolerance > 0.0 && p.depth_rel_tolerance < 1.0);
        assert!(p.normal_cos_tolerance > 0.0 && p.normal_cos_tolerance < 1.0);
    }
}
