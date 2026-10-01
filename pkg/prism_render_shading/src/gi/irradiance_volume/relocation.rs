//! DDGI probe relocation, classification, grid mapping, and interpolation.
//!
//! This is the CPU golden reference for the probe-management layer of Majercik
//! et al. 2019 together with its RTXGI follow-ups ("Scrolling / Infinite
//! Scrolling Volumes" and "Probe Classification").  It covers four cooperating
//! concerns:
//!
//! * [`ProbeGrid`] — the regular 3-D lattice: world↔grid coordinate mapping,
//!   trilinear cell fractions, and toroidal *scrolling* of the volume origin as
//!   the camera moves.
//! * [`relocate_offset`] — per-probe *relocation*: a bounded offset nudging a
//!   probe out of the geometry it is embedded in (too many back-face hits) or
//!   away from a surface it sits too close to, while relaxing back toward the
//!   grid centre when the probe sits in open space.
//! * [`classify_probe`] / [`transition_state`] — probe *classification* into
//!   [`ProbeState::Active`], [`ProbeState::Inactive`], and the transient
//!   [`ProbeState::NewlyVacated`] marker that tells the integrator to reset a
//!   probe's temporal history after it re-activates.
//! * [`interpolation_weights`] — the eight-probe blend weights that fuse
//!   trilinear position, a smooth normal/back-face term, and the Chebyshev
//!   depth-visibility weight (reusing [`crate::gi::world_space::visibility`]).
//!
//! # Conventions
//! * World space is right-handed `Vec3`; grid coordinates are integer
//!   [`IVec3`] probe indices.  Probe `coord` sits at `origin + coord * spacing`
//!   plus its relocation offset.
//! * Relocation offsets are clamped per axis to `relocation_limit * spacing`
//!   (default fraction below) so a probe never leaves its own cell neighbourhood.
//! * All weights are non-negative and renormalised to sum to one, with a
//!   uniform fallback when every probe is rejected, so the blend is always a
//!   valid partition of unity.
//! * Every function is deterministic: no RNG, no I/O, no GPU, no `unsafe`, no
//!   heap allocation; transcendental use goes through [`bevy_math::ops`].

use bevy_math::{IVec3, Vec3};

use crate::gi::world_space::visibility::chebyshev_weight;

/// Default fraction of a cell a probe may be relocated from its grid position.
///
/// RTXGI keeps relocation inside the cell (`< 0.5`) so the eight-corner
/// interpolation footprint stays valid; `0.45` leaves a small safety margin.
pub const DEFAULT_RELOCATION_LIMIT: f32 = 0.45;

/// A regular 3-D DDGI probe lattice with world↔grid mapping and scrolling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeGrid {
    /// World-space position of probe coordinate `(0, 0, 0)`.
    pub origin: Vec3,
    /// World-space spacing between adjacent probes along each axis; each
    /// component is clamped to a tiny positive value on read to avoid division
    /// by zero.
    pub spacing: Vec3,
    /// Number of probes along each axis; each component is treated as at
    /// least `1`.
    pub counts: IVec3,
}

impl ProbeGrid {
    /// Builds a grid, sanitising `spacing` (positive) and `counts` (`>= 1`).
    #[inline]
    pub fn new(origin: Vec3, spacing: Vec3, counts: IVec3) -> Self {
        Self {
            origin,
            spacing: sanitize_spacing(spacing),
            counts: IVec3::new(counts.x.max(1), counts.y.max(1), counts.z.max(1)),
        }
    }

    /// Clamps a probe coordinate into the valid `0..counts` range per axis.
    #[inline]
    pub fn clamp_coord(&self, coord: IVec3) -> IVec3 {
        IVec3::new(
            coord.x.clamp(0, self.counts.x - 1),
            coord.y.clamp(0, self.counts.y - 1),
            coord.z.clamp(0, self.counts.z - 1),
        )
    }

    /// World-space base position of a probe coordinate (no relocation offset).
    #[inline]
    pub fn probe_base_position(&self, coord: IVec3) -> Vec3 {
        self.origin + coord.as_vec3() * self.spacing
    }

    /// World-space position of a probe including its relocation `offset`.
    #[inline]
    pub fn probe_position(&self, coord: IVec3, offset: Vec3) -> Vec3 {
        self.probe_base_position(coord) + offset
    }

    /// Maps a world point to its base cell coordinate and trilinear fractions.
    ///
    /// Returns `(base, frac)` where `base` is the low corner of the enclosing
    /// cell (clamped so `base` and `base + 1` are both in range) and `frac` is
    /// the per-axis position within the cell in `[0, 1]`.  Points outside the
    /// volume clamp to the boundary cell with a saturated fraction.
    #[inline]
    pub fn world_to_cell(&self, point: Vec3) -> (IVec3, Vec3) {
        let inv = Vec3::ONE / self.spacing;
        let g = (point - self.origin) * inv;
        let mut base = IVec3::ZERO;
        let mut frac = Vec3::ZERO;
        for a in 0..3 {
            let max_base = (self.counts[a] - 1).max(0);
            let gi = g[a];
            // Lowest cell whose [base, base+1] brackets the point.
            let fi = libm_floor(gi);
            let bi = (fi as i32).clamp(0, (max_base - 1).max(0));
            base[a] = bi;
            frac[a] = (gi - bi as f32).clamp(0.0, 1.0);
        }
        (base, frac)
    }

    /// Returns the eight corner coordinates of a cell with low corner `base`.
    ///
    /// Corner `c` is indexed by its `(x, y, z)` bits (bit 0 = x, bit 1 = y,
    /// bit 2 = z), matching [`trilinear_weights`].  Coordinates are clamped
    /// into range so degenerate single-probe axes repeat the same probe.
    #[inline]
    pub fn cell_corners(&self, base: IVec3) -> [IVec3; 8] {
        let mut out = [IVec3::ZERO; 8];
        for (c, slot) in out.iter_mut().enumerate() {
            let dx = (c & 1) as i32;
            let dy = ((c >> 1) & 1) as i32;
            let dz = ((c >> 2) & 1) as i32;
            *slot = self.clamp_coord(base + IVec3::new(dx, dy, dz));
        }
        out
    }

    /// Scrolls the volume by an integer number of probe cells.
    ///
    /// Infinite-scrolling DDGI volumes track the camera by shifting the origin
    /// in whole-probe increments (so already-converged probes keep their world
    /// position and only a thin shell of probes is recomputed).  The returned
    /// grid has `origin += amount * spacing`; `counts`/`spacing` are unchanged.
    #[inline]
    pub fn scrolled(&self, amount: IVec3) -> Self {
        Self {
            origin: self.origin + amount.as_vec3() * self.spacing,
            spacing: self.spacing,
            counts: self.counts,
        }
    }
}

/// Computes the eight trilinear corner weights for cell fractions `frac`.
///
/// `frac` is clamped to `[0, 1]^3`; corner `c` uses the `(x, y, z)` bit layout
/// described on [`ProbeGrid::cell_corners`].  The weights sum to exactly one.
#[inline]
pub fn trilinear_weights(frac: Vec3) -> [f32; 8] {
    let fx = frac.x.clamp(0.0, 1.0);
    let fy = frac.y.clamp(0.0, 1.0);
    let fz = frac.z.clamp(0.0, 1.0);
    let wx = [1.0 - fx, fx];
    let wy = [1.0 - fy, fy];
    let wz = [1.0 - fz, fz];
    let mut out = [0.0f32; 8];
    for (c, slot) in out.iter_mut().enumerate() {
        *slot = wx[c & 1] * wy[(c >> 1) & 1] * wz[(c >> 2) & 1];
    }
    out
}

/// Smooth DDGI "wrap" back-face weight for a probe seen from a shading point.
///
/// Returns `(0.5 + 0.5 * dot(dir_to_probe, normal))^2`: `1` when the probe lies
/// along the surface normal, fading to `0` as it rotates behind the surface.  A
/// degenerate normal or coincident probe/point returns `1` (no-op term).
#[inline]
pub fn normal_backface_weight(point: Vec3, normal: Vec3, probe_position: Vec3) -> f32 {
    let to_probe = probe_position - point;
    let len_sq = to_probe.length_squared();
    let n_len_sq = normal.length_squared();
    if len_sq <= f32::MIN_POSITIVE || n_len_sq <= f32::MIN_POSITIVE {
        return 1.0;
    }
    let dir = to_probe * len_sq.sqrt().recip();
    let n = normal * n_len_sq.sqrt().recip();
    let wrap = (0.5 + 0.5 * dir.dot(n)).max(0.0);
    wrap * wrap
}

/// One corner probe participating in an eight-probe interpolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeSample {
    /// World-space probe position (base + relocation offset).
    pub position: Vec3,
    /// Mean occluder depth `E[d]` along the probe→point direction.
    pub depth_mean: f32,
    /// Mean-squared occluder depth `E[d^2]` along the probe→point direction.
    pub depth_mean_sq: f32,
    /// Whether the probe is active; inactive probes receive zero weight.
    pub active: bool,
}

impl ProbeSample {
    /// Convenience constructor for a fully open, active probe.
    #[inline]
    pub fn open(position: Vec3) -> Self {
        let big = 1.0e9;
        Self {
            position,
            depth_mean: big,
            depth_mean_sq: big * big,
            active: true,
        }
    }
}

/// Computes the fully combined, renormalised eight-probe interpolation weights.
///
/// For each corner the trilinear weight is multiplied by:
/// * [`normal_backface_weight`] (cosine / back-face cull), and
/// * the Chebyshev depth-visibility weight for the probe→point distance biased
///   by `bias` (reusing [`chebyshev_weight`]), and
/// * `0` for inactive probes.
///
/// The eight products are renormalised to sum to one.  If every probe is
/// rejected (sum non-positive) a uniform `1/8` fallback is returned so the
/// blend never collapses to zero or `NaN`.
#[inline]
pub fn interpolation_weights(
    corners: &[ProbeSample; 8],
    frac: Vec3,
    point: Vec3,
    normal: Vec3,
    bias: f32,
) -> [f32; 8] {
    let tri = trilinear_weights(frac);
    let bias = bias.max(0.0);
    let mut combined = [0.0f32; 8];
    let mut sum = 0.0f32;
    for c in 0..8 {
        let probe = &corners[c];
        if !probe.active {
            continue;
        }
        let n_w = normal_backface_weight(point, normal, probe.position);
        let distance = ((probe.position - point).length() - bias).max(0.0);
        let d_w = chebyshev_weight(probe.depth_mean, probe.depth_mean_sq, distance);
        let w = (tri[c] * n_w * d_w).max(0.0);
        combined[c] = w;
        sum += w;
    }
    if sum > f32::MIN_POSITIVE {
        let inv = sum.recip();
        for w in combined.iter_mut() {
            *w *= inv;
        }
    } else {
        combined = [0.125; 8];
    }
    combined
}

/// Aggregated per-probe ray statistics driving relocation and classification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeRayStats {
    /// Fraction of rays that hit a *back* face in `[0, 1]` (inside geometry).
    pub backface_fraction: f32,
    /// Distance to the closest *front*-face hit (positive; `f32::INFINITY`
    /// when none were hit).
    pub closest_frontface_distance: f32,
    /// Direction (from probe) toward the closest front-face hit; need not be
    /// normalised.
    pub closest_frontface_dir: Vec3,
    /// Distance to the farthest front-face hit used as the "most open"
    /// direction target.
    pub farthest_frontface_distance: f32,
    /// Direction toward the farthest front-face hit (most open direction).
    pub farthest_frontface_dir: Vec3,
    /// Direction toward the closest back-face hit (used to escape geometry).
    pub closest_backface_dir: Vec3,
    /// Distance to the closest back-face hit.
    pub closest_backface_distance: f32,
}

impl ProbeRayStats {
    /// Stats for a probe sitting in fully open space (no hits).
    #[inline]
    pub fn open() -> Self {
        Self {
            backface_fraction: 0.0,
            closest_frontface_distance: f32::INFINITY,
            closest_frontface_dir: Vec3::ZERO,
            farthest_frontface_distance: 0.0,
            farthest_frontface_dir: Vec3::ZERO,
            closest_backface_dir: Vec3::ZERO,
            closest_backface_distance: f32::INFINITY,
        }
    }
}

/// Computes the next relocation offset for a probe (bounded, deterministic).
///
/// Starting from `prev_offset` (world-space) the rule mirrors RTXGI relocation:
///
/// * If the back-face fraction exceeds `backface_threshold` the probe is inside
///   geometry; it is pushed along the closest back-face direction just past
///   that occluder (`closest_backface_distance + min_frontface_distance`).
/// * Otherwise, if the closest front face is nearer than `min_frontface_distance`
///   the probe sits too close to a surface; it is pushed along the farthest
///   (most open) front-face direction by `min_frontface_distance`.
/// * Otherwise the probe is in open space and the offset relaxes toward zero
///   (back to the exact grid position), which keeps interpolation footprints
///   clean once geometry moves away.
///
/// The resulting offset is clamped per axis to
/// `relocation_limit * spacing` (with `relocation_limit` clamped to
/// `[0, 0.5)`), so a probe always stays inside its cell neighbourhood.  All
/// directions are normalised defensively; degenerate inputs leave the offset
/// unchanged (minus the open-space relaxation).
#[inline]
pub fn relocate_offset(
    prev_offset: Vec3,
    stats: &ProbeRayStats,
    spacing: Vec3,
    min_frontface_distance: f32,
    backface_threshold: f32,
    relocation_limit: f32,
) -> Vec3 {
    let spacing = sanitize_spacing(spacing);
    let min_front = min_frontface_distance.max(0.0);
    let backface_threshold = backface_threshold.clamp(0.0, 1.0);
    let limit = relocation_limit.clamp(0.0, 0.4999);

    let mut offset = prev_offset;
    if stats.backface_fraction > backface_threshold {
        // Probe is embedded in geometry: escape along the closest back face.
        if let Some(dir) = normalize_or_none(stats.closest_backface_dir) {
            let push = stats.closest_backface_distance.max(0.0) + min_front;
            offset = prev_offset + dir * push;
        }
    } else if stats.closest_frontface_distance < min_front {
        // Too close to a surface: back off along the most open direction.
        if let Some(dir) = normalize_or_none(stats.farthest_frontface_dir) {
            offset = prev_offset + dir * min_front;
        }
    } else {
        // Open space: relax toward the grid centre so the offset decays.
        offset = prev_offset * 0.5;
    }

    // Clamp per axis to the allowed fraction of the cell.
    let max_axis = spacing * limit;
    Vec3::new(
        offset.x.clamp(-max_axis.x, max_axis.x),
        offset.y.clamp(-max_axis.y, max_axis.y),
        offset.z.clamp(-max_axis.z, max_axis.z),
    )
}

/// Classification state of a DDGI probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeState {
    /// The probe is near relevant geometry and contributes to interpolation.
    Active,
    /// The probe is inside geometry or isolated in empty space; it is skipped.
    Inactive,
    /// The probe just transitioned from inactive to active: its temporal
    /// history is stale and must be reset before it is trusted.
    NewlyVacated,
}

/// Classifies a probe from its ray statistics (ignoring prior state).
///
/// A probe is [`ProbeState::Inactive`] when it is embedded in geometry (the
/// back-face fraction exceeds `backface_threshold`) or when it has no geometry
/// within `activity_distance` (isolated in empty space, so it cannot usefully
/// shade nearby surfaces).  Otherwise it is [`ProbeState::Active`].
///
/// `activity_distance` is typically a small multiple of the cell diagonal.
/// This function never returns [`ProbeState::NewlyVacated`]; that transient is
/// produced only by [`transition_state`].
#[inline]
pub fn classify_probe(
    stats: &ProbeRayStats,
    activity_distance: f32,
    backface_threshold: f32,
) -> ProbeState {
    let backface_threshold = backface_threshold.clamp(0.0, 1.0);
    let activity_distance = activity_distance.max(0.0);
    if stats.backface_fraction > backface_threshold {
        return ProbeState::Inactive;
    }
    if stats.closest_frontface_distance <= activity_distance {
        ProbeState::Active
    } else {
        ProbeState::Inactive
    }
}

/// Folds the previous state into the freshly classified state.
///
/// If the probe was [`ProbeState::Inactive`] (or had never settled) and is now
/// [`ProbeState::Active`], it is reported as [`ProbeState::NewlyVacated`] so the
/// integrator knows to reset the probe's temporal history before trusting it.
/// Otherwise the fresh classification is returned unchanged.  `raw` is treated
/// as the output of [`classify_probe`] (never itself `NewlyVacated`).
#[inline]
pub fn transition_state(prev: ProbeState, raw: ProbeState) -> ProbeState {
    match (prev, raw) {
        (ProbeState::Inactive, ProbeState::Active)
        | (ProbeState::NewlyVacated, ProbeState::Active) => {
            if prev == ProbeState::Inactive {
                ProbeState::NewlyVacated
            } else {
                ProbeState::Active
            }
        }
        _ => raw,
    }
}

/// Clamps spacing components to a tiny positive value to avoid division by zero.
#[inline]
fn sanitize_spacing(spacing: Vec3) -> Vec3 {
    const MIN: f32 = 1.0e-6;
    Vec3::new(
        if spacing.x.is_finite() { spacing.x.abs().max(MIN) } else { MIN },
        if spacing.y.is_finite() { spacing.y.abs().max(MIN) } else { MIN },
        if spacing.z.is_finite() { spacing.z.abs().max(MIN) } else { MIN },
    )
}

/// Normalises a vector, returning `None` for degenerate (near-zero) inputs.
#[inline]
fn normalize_or_none(v: Vec3) -> Option<Vec3> {
    let len_sq = v.length_squared();
    if len_sq <= f32::MIN_POSITIVE || !len_sq.is_finite() {
        None
    } else {
        Some(v * len_sq.sqrt().recip())
    }
}

/// `floor` for `f32` without pulling in `std`; exact for the finite range used
/// by grid addressing.
#[inline]
fn libm_floor(x: f32) -> f32 {
    let t = x as i64 as f32;
    if t > x {
        t - 1.0
    } else {
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> ProbeGrid {
        ProbeGrid::new(Vec3::ZERO, Vec3::splat(2.0), IVec3::new(4, 4, 4))
    }

    #[test]
    fn probe_positions_follow_origin_and_spacing() {
        let g = grid();
        assert_eq!(g.probe_base_position(IVec3::new(1, 2, 3)), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(
            g.probe_position(IVec3::new(1, 0, 0), Vec3::new(0.1, 0.0, 0.0)),
            Vec3::new(2.1, 0.0, 0.0)
        );
    }

    #[test]
    fn world_to_cell_is_inverse_in_interior() {
        let g = grid();
        // Point at 3.5 along x (between probes 1 and 2) -> base 1, frac 0.75.
        let (base, frac) = g.world_to_cell(Vec3::new(3.5, 0.0, 0.0));
        assert_eq!(base.x, 1);
        assert!((frac.x - 0.75).abs() < 1e-5, "frac {}", frac.x);
        // The low corner of the cell is the base coordinate itself.
        let corners = g.cell_corners(base);
        assert_eq!(corners[0], base);
        assert_eq!(corners[1], base + IVec3::new(1, 0, 0));
    }

    #[test]
    fn world_to_cell_clamps_outside_volume() {
        let g = grid();
        let (base, frac) = g.world_to_cell(Vec3::new(-100.0, 1000.0, 3.0));
        assert!(base.x >= 0 && base.x < g.counts.x - 1 || g.counts.x == 1);
        assert!(frac.x >= 0.0 && frac.x <= 1.0);
        assert!(frac.y >= 0.0 && frac.y <= 1.0);
    }

    #[test]
    fn scrolling_shifts_origin_by_whole_cells() {
        let g = grid();
        let s = g.scrolled(IVec3::new(1, 0, -2));
        assert_eq!(s.origin, Vec3::new(2.0, 0.0, -4.0));
        assert_eq!(s.spacing, g.spacing);
        assert_eq!(s.counts, g.counts);
    }

    #[test]
    fn trilinear_weights_are_partition_of_unity() {
        for frac in [Vec3::ZERO, Vec3::ONE, Vec3::splat(0.5), Vec3::new(0.2, 0.7, 0.9)] {
            let w = trilinear_weights(frac);
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "sum {sum} for {frac:?}");
            assert!(w.iter().all(|&x| x >= 0.0));
        }
        // Corner (0,0,0) with zero frac gets all the weight.
        assert!((trilinear_weights(Vec3::ZERO)[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn interpolation_weights_normalise_and_cull() {
        let base = Vec3::new(0.0, 0.0, 0.0);
        let mut corners = [ProbeSample::open(base); 8];
        // Lay corners on a unit cube.
        for c in 0..8 {
            let dx = (c & 1) as f32;
            let dy = ((c >> 1) & 1) as f32;
            let dz = ((c >> 2) & 1) as f32;
            corners[c] = ProbeSample::open(Vec3::new(dx, dy, dz) * 2.0);
        }
        let point = Vec3::new(1.0, 1.0, 1.0);
        let normal = Vec3::Y;
        let w = interpolation_weights(&corners, Vec3::splat(0.5), point, normal, 0.0);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum {sum}");
        assert!(w.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn inactive_probes_get_zero_weight() {
        let mut corners = [ProbeSample::open(Vec3::ZERO); 8];
        for c in 0..8 {
            let dx = (c & 1) as f32;
            let dy = ((c >> 1) & 1) as f32;
            let dz = ((c >> 2) & 1) as f32;
            corners[c] = ProbeSample::open(Vec3::new(dx, dy, dz));
            corners[c].active = c % 2 == 0; // deactivate odd corners
        }
        let w = interpolation_weights(&corners, Vec3::splat(0.5), Vec3::splat(0.5), Vec3::Y, 0.0);
        for c in 0..8 {
            if c % 2 == 1 {
                assert_eq!(w[c], 0.0, "inactive corner {c} must have zero weight");
            }
        }
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum {sum}");
    }

    #[test]
    fn all_rejected_falls_back_to_uniform() {
        let corners = [ProbeSample {
            position: Vec3::ZERO,
            depth_mean: 0.0,
            depth_mean_sq: 0.0,
            active: false,
        }; 8];
        let w = interpolation_weights(&corners, Vec3::splat(0.5), Vec3::splat(5.0), Vec3::Y, 0.0);
        for x in w {
            assert!((x - 0.125).abs() < 1e-6, "expected uniform fallback, got {x}");
        }
    }

    #[test]
    fn relocation_escapes_backface_and_relaxes_open() {
        let spacing = Vec3::splat(2.0);
        // Inside geometry: pushed along +X back-face direction, clamped to cell.
        let mut stats = ProbeRayStats::open();
        stats.backface_fraction = 0.6;
        stats.closest_backface_dir = Vec3::X;
        stats.closest_backface_distance = 0.1;
        let off = relocate_offset(Vec3::ZERO, &stats, spacing, 0.5, 0.25, 0.45);
        assert!(off.x > 0.0, "should move along +X, got {off:?}");
        assert!(off.x <= 2.0 * 0.45 + 1e-6, "offset exceeds cell limit: {off:?}");

        // Open space: offset relaxes toward zero.
        let open = ProbeRayStats::open();
        let relaxed = relocate_offset(Vec3::new(0.4, 0.0, 0.0), &open, spacing, 0.5, 0.25, 0.45);
        assert!(relaxed.x < 0.4 && relaxed.x >= 0.0, "should relax, got {relaxed:?}");
    }

    #[test]
    fn relocation_backs_off_near_surface() {
        let spacing = Vec3::splat(2.0);
        let mut stats = ProbeRayStats::open();
        stats.closest_frontface_distance = 0.05; // closer than min_front
        stats.farthest_frontface_dir = Vec3::NEG_X;
        stats.farthest_frontface_distance = 3.0;
        let off = relocate_offset(Vec3::ZERO, &stats, spacing, 0.5, 0.25, 0.45);
        assert!(off.x < 0.0, "should back off along -X, got {off:?}");
    }

    #[test]
    fn relocation_offset_is_always_bounded() {
        let spacing = Vec3::new(2.0, 4.0, 1.0);
        let mut stats = ProbeRayStats::open();
        stats.backface_fraction = 1.0;
        stats.closest_backface_dir = Vec3::new(1.0, 1.0, 1.0);
        stats.closest_backface_distance = 1000.0;
        let off = relocate_offset(Vec3::ZERO, &stats, spacing, 10.0, 0.25, 0.45);
        assert!(off.x.abs() <= 2.0 * 0.45 + 1e-6);
        assert!(off.y.abs() <= 4.0 * 0.45 + 1e-6);
        assert!(off.z.abs() <= 1.0 * 0.45 + 1e-6);
    }

    #[test]
    fn classification_detects_inside_and_isolated() {
        // Inside geometry -> inactive.
        let mut stats = ProbeRayStats::open();
        stats.backface_fraction = 0.5;
        assert_eq!(classify_probe(&stats, 10.0, 0.25), ProbeState::Inactive);

        // Near geometry -> active.
        let mut near = ProbeRayStats::open();
        near.closest_frontface_distance = 2.0;
        assert_eq!(classify_probe(&near, 10.0, 0.25), ProbeState::Active);

        // Isolated (no geometry within activity distance) -> inactive.
        let mut far = ProbeRayStats::open();
        far.closest_frontface_distance = 100.0;
        assert_eq!(classify_probe(&far, 10.0, 0.25), ProbeState::Inactive);
    }

    #[test]
    fn transition_reports_newly_vacated_then_active() {
        assert_eq!(
            transition_state(ProbeState::Inactive, ProbeState::Active),
            ProbeState::NewlyVacated
        );
        assert_eq!(
            transition_state(ProbeState::NewlyVacated, ProbeState::Active),
            ProbeState::Active
        );
        assert_eq!(
            transition_state(ProbeState::Active, ProbeState::Active),
            ProbeState::Active
        );
        assert_eq!(
            transition_state(ProbeState::Active, ProbeState::Inactive),
            ProbeState::Inactive
        );
    }

    #[test]
    fn sanitize_handles_degenerate_spacing() {
        let g = ProbeGrid::new(Vec3::ZERO, Vec3::new(0.0, -2.0, f32::NAN), IVec3::new(0, 0, 0));
        assert!(g.spacing.x > 0.0 && g.spacing.y > 0.0 && g.spacing.z > 0.0);
        assert_eq!(g.counts, IVec3::new(1, 1, 1));
        // world_to_cell never panics on a single-probe axis.
        let (_base, frac) = g.world_to_cell(Vec3::splat(5.0));
        assert!(frac.is_finite());
    }

    #[test]
    fn floor_matches_reference() {
        for v in [-2.5f32, -1.0, -0.1, 0.0, 0.1, 1.9, 3.0, 7.75] {
            assert_eq!(libm_floor(v), v.floor(), "floor mismatch at {v}");
        }
    }

    #[test]
    fn results_are_deterministic() {
        let g = grid();
        let a = g.world_to_cell(Vec3::new(1.3, 2.7, 5.1));
        let b = g.world_to_cell(Vec3::new(1.3, 2.7, 5.1));
        assert_eq!(a, b);
    }
}
