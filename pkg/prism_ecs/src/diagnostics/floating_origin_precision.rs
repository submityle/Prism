//! Floating-origin rebase precision budget audit (design §13.3 / §16.6).
//!
//! The whole point of the 64-bit [`FloatingOrigin`] kernel (design §13.3) is to
//! keep the single-precision coordinate that render and physics actually
//! consume *jitter-free*: an entity is stored as a coarse `i64`
//! [`GridCell`] plus a fine `f32` [`LocalPos`], and every frame
//! [`FloatingOrigin::rebase`] folds it into the active origin cell's local
//! space. Near the origin the rebased magnitude is tiny, so the `f32` keeps its
//! full ~24-bit significand; far from the origin the magnitude grows and the
//! gap between representable `f32` values (one *ULP*) widens until geometry
//! visibly steps. Culling / LOD (design §13.1 / §13.2) are supposed to retire
//! an entity before its rebased precision decays below what the content needs.
//!
//! Nothing in the kernel checks that the cull/LOD reach actually stays inside
//! the usable-precision radius, nor that stored local offsets have not drifted
//! out of their cell. This report makes both legible against a caller-supplied
//! **target resolution** (the finest spacing the content cares about, e.g.
//! `0.001` for one millimetre):
//!
//! * a **precision ladder** — for a geometric sweep of ring radii (powers of
//!   two cells out from the active origin) it reports the magnitude in metres
//!   and the exact `f32` ULP there, flagging the first ring whose resolution no
//!   longer meets the target. This answers "how far can I draw before `f32`
//!   stops resolving the target", read from the grid's cell size alone;
//! * an analytic **safe radius** (`target × 2^23`, the 24-bit significand's
//!   relative-precision bound) in both metres and whole cells;
//! * an optional **sample census** — for a set of `(cell, local)` entity
//!   positions it rebases each against the active origin and reports the
//!   resulting magnitude, its ULP, whether it meets the target, and whether the
//!   local offset has drifted outside `[0, cell_size)` and needs a
//!   [`normalize`](FloatingOrigin::normalize) — plus world-wide roll-ups.
//!
//! All ULP figures are computed by pure bit inspection of the `f32`
//! representation (no `std` float intrinsics), so the audit is `no_std`-clean
//! and deterministic. It owns no [`World`](crate::world::World) state and reads
//! the grid in `O(rings + samples)`.

use alloc::vec::Vec;

use crate::partition::floating_origin::{FloatingOrigin, GridCell, LocalPos};

/// `2^23`, the implicit scale of an `f32`'s 24-bit significand: at magnitude
/// `m` the relative precision is about `m / 2^23`, so a target resolution `t`
/// is held up to roughly `t × 2^23` metres.
const F32_MANTISSA_SCALE: f32 = 8_388_608.0;

/// Highest ring exponent swept by the ladder (`2^20` ≈ one million cells).
const MAX_RING_EXP: u32 = 20;

/// `|x|` for an `f32` via sign-bit masking — the `no_std`-clean absolute value
/// (the inherent `f32::abs` lives in `std`).
#[inline]
fn f32_abs(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}

/// The ULP (unit in the last place) at `|x|`: the gap in metres to the next
/// representable `f32` above `|x|`, i.e. the spatial resolution available there.
/// Returns the value itself for a non-finite input.
#[inline]
fn ulp_f32(x: f32) -> f32 {
    let a = f32_abs(x);
    if !a.is_finite() {
        return a;
    }
    // `a` is finite and non-negative, so `a.to_bits() + 1` is the next
    // representable magnitude (at most the `+inf` bit pattern for `f32::MAX`).
    let next = f32::from_bits(a.to_bits() + 1);
    next - a
}

/// Largest absolute axis of a rebased triple, in metres.
#[inline]
fn max3_abs(x: f32, y: f32, z: f32) -> f32 {
    let ax = f32_abs(x);
    let ay = f32_abs(y);
    let az = f32_abs(z);
    let mut m = ax;
    if ay > m {
        m = ay;
    }
    if az > m {
        m = az;
    }
    m
}

/// Whether a single local axis lies outside `[0, cell_size)` (has drifted and
/// needs re-normalising before its `f32` precision decays).
#[inline]
fn axis_drifted(v: f32, cell_size: f64) -> bool {
    let d = v as f64;
    d < 0.0 || d >= cell_size
}

/// One rung of the precision ladder: the `f32` resolution available at a given
/// Chebyshev ring radius out from the active origin (design §13.3).
#[derive(Clone, Copy, Debug)]
pub struct PrecisionRingEntry {
    /// Chebyshev radius of this ring, in whole cells (a power of two).
    pub radius_cells: u32,
    /// That radius in metres (`radius_cells × cell_size`), as the `f32` the
    /// rebased pipeline would see.
    pub radius_m: f32,
    /// Spatial resolution available at `radius_m`: one `f32` ULP there, in
    /// metres.
    pub resolution_m: f32,
    /// Whether `resolution_m` is at least as fine as the audit's target
    /// (`resolution_m <= target`).
    pub meets_target: bool,
}

/// Per-sample rebase audit: one stored `(cell, local)` entity position folded
/// into the active origin's local `f32` space (design §13.3).
#[derive(Clone, Copy, Debug)]
pub struct RebaseSampleAudit {
    /// Chebyshev distance, in cells, from the sample's cell to the active
    /// origin.
    pub cell_delta_max: u64,
    /// Largest-magnitude axis of the sample after rebasing into origin-local
    /// space, in metres.
    pub rebased_magnitude_m: f32,
    /// Spatial resolution (one ULP) at `rebased_magnitude_m`, in metres.
    pub resolution_m: f32,
    /// Whether `resolution_m` meets the audit target.
    pub meets_target: bool,
    /// Whether any local axis lies outside `[0, cell_size)` — the sample has
    /// drifted and should be re-[`normalize`](FloatingOrigin::normalize)d.
    pub is_drifted: bool,
}

/// Read-only precision budget audit of a [`FloatingOrigin`] grid: a resolution
/// ladder over distance, an analytic safe radius, and an optional per-entity
/// rebase census (design §13.3 / §16.6).
#[derive(Clone, Debug)]
pub struct FloatingOriginPrecisionAudit {
    cell_size: f64,
    origin: GridCell,
    target_resolution_m: f32,
    rings: Vec<PrecisionRingEntry>,
    samples: Vec<RebaseSampleAudit>,
}

/// Builds the precision ladder for a cell size: one entry per power-of-two ring
/// radius, stopping once the radius overflows the finite `f32` range.
fn build_rings(cell_size: f64, target: f32) -> Vec<PrecisionRingEntry> {
    let mut rings = Vec::new();
    for exp in 0..=MAX_RING_EXP {
        let radius_cells = 1u32 << exp;
        let radius_m = (radius_cells as f64 * cell_size) as f32;
        if !radius_m.is_finite() {
            break;
        }
        let resolution_m = ulp_f32(radius_m);
        rings.push(PrecisionRingEntry {
            radius_cells,
            radius_m,
            resolution_m,
            meets_target: resolution_m <= target,
        });
    }
    rings
}

impl FloatingOriginPrecisionAudit {
    /// Audit a grid's precision budget from its configuration alone (no sample
    /// census). `target_resolution_m` is the finest spacing the content needs.
    pub fn from_origin(grid: &FloatingOrigin, target_resolution_m: f32) -> Self {
        let cell_size = grid.cell_size();
        Self {
            cell_size,
            origin: grid.origin(),
            target_resolution_m,
            rings: build_rings(cell_size, target_resolution_m),
            samples: Vec::new(),
        }
    }

    /// Audit a grid's precision budget and, in addition, rebase each
    /// `(cell, local)` sample against the active origin to census its actual
    /// resolution and drift (design §13.3). Read-only; `O(rings + samples)`.
    pub fn from_origin_with_samples(
        grid: &FloatingOrigin,
        target_resolution_m: f32,
        samples: &[(GridCell, LocalPos)],
    ) -> Self {
        let cell_size = grid.cell_size();
        let origin = grid.origin();
        let mut audited = Vec::with_capacity(samples.len());
        for &(cell, local) in samples {
            let (dx, dy, dz) = cell.offset_from(origin);
            let cell_delta_max = dx
                .unsigned_abs()
                .max(dy.unsigned_abs())
                .max(dz.unsigned_abs());
            let rebased = grid.rebase(cell, local);
            let rebased_magnitude_m = max3_abs(rebased.x, rebased.y, rebased.z);
            let resolution_m = ulp_f32(rebased_magnitude_m);
            let is_drifted = axis_drifted(local.x, cell_size)
                || axis_drifted(local.y, cell_size)
                || axis_drifted(local.z, cell_size);
            audited.push(RebaseSampleAudit {
                cell_delta_max,
                rebased_magnitude_m,
                resolution_m,
                meets_target: resolution_m <= target_resolution_m,
                is_drifted,
            });
        }
        Self {
            cell_size,
            origin,
            target_resolution_m,
            rings: build_rings(cell_size, target_resolution_m),
            samples: audited,
        }
    }

    /// The grid's cell edge length in metres.
    #[inline]
    pub fn cell_size(&self) -> f64 {
        self.cell_size
    }

    /// The active origin cell the audit rebased against.
    #[inline]
    pub fn origin(&self) -> GridCell {
        self.origin
    }

    /// The target spatial resolution the audit grades against, in metres.
    #[inline]
    pub fn target_resolution_m(&self) -> f32 {
        self.target_resolution_m
    }

    /// Whether the grid's cell size is finite and strictly positive.
    #[inline]
    pub fn config_is_sane(&self) -> bool {
        self.cell_size.is_finite() && self.cell_size > 0.0
    }

    /// Chebyshev distance of the active origin from world cell `(0, 0, 0)`, in
    /// cells — how far the frame has already rebased.
    pub fn origin_offset_cells(&self) -> u64 {
        self.origin
            .x
            .unsigned_abs()
            .max(self.origin.y.unsigned_abs())
            .max(self.origin.z.unsigned_abs())
    }

    /// World distance of the active origin from `(0, 0, 0)` along its farthest
    /// axis, in metres (`origin_offset_cells × cell_size`).
    pub fn origin_world_offset_m(&self) -> f64 {
        self.origin_offset_cells() as f64 * self.cell_size
    }

    /// Resolution floor inside the origin cell itself: one ULP at a magnitude
    /// of `cell_size`, the coarsest `f32` gap an un-drifted local offset in the
    /// origin cell can see.
    pub fn cell_boundary_resolution_m(&self) -> f32 {
        ulp_f32(self.cell_size as f32)
    }

    /// The precision ladder: one entry per power-of-two ring radius out from
    /// the active origin, nearest first.
    #[inline]
    pub fn rings(&self) -> &[PrecisionRingEntry] {
        &self.rings
    }

    /// Number of ladder rings computed.
    #[inline]
    pub fn ring_count(&self) -> usize {
        self.rings.len()
    }

    /// Number of ladder rings whose resolution meets the target.
    pub fn safe_ring_count(&self) -> usize {
        self.rings.iter().filter(|r| r.meets_target).count()
    }

    /// Whether every computed ring meets the target (the whole addressable
    /// sweep resolves the target).
    pub fn all_rings_meet_target(&self) -> bool {
        self.rings.iter().all(|r| r.meets_target)
    }

    /// The farthest ring that still meets the target resolution, or `None` if
    /// even the nearest ring is already too coarse.
    pub fn deepest_safe_ring(&self) -> Option<&PrecisionRingEntry> {
        self.rings.iter().rev().find(|r| r.meets_target)
    }

    /// The nearest ring that fails the target resolution, or `None` if every
    /// computed ring is fine enough — the radius at which culling / LOD must
    /// already have retired the entity.
    pub fn first_unsafe_ring(&self) -> Option<&PrecisionRingEntry> {
        self.rings.iter().find(|r| !r.meets_target)
    }

    /// Analytic bound on how far from the origin an `f32` still resolves the
    /// target: `target × 2^23` metres, from the 24-bit significand's relative
    /// precision. Approximate; the exact per-magnitude values are in
    /// [`rings`](Self::rings).
    pub fn max_safe_radius_m(&self) -> f64 {
        self.target_resolution_m as f64 * F32_MANTISSA_SCALE as f64
    }

    /// [`max_safe_radius_m`](Self::max_safe_radius_m) in whole cells (floored).
    /// `0` when the cell size is not usable.
    pub fn max_safe_radius_cells(&self) -> i64 {
        if !self.config_is_sane() {
            return 0;
        }
        (self.max_safe_radius_m() / self.cell_size) as i64
    }

    /// The per-sample rebase audits (empty unless built with samples).
    #[inline]
    pub fn samples(&self) -> &[RebaseSampleAudit] {
        &self.samples
    }

    /// Number of audited samples.
    #[inline]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Samples whose local offset lies outside `[0, cell_size)` and need a
    /// [`normalize`](FloatingOrigin::normalize).
    pub fn drifted_sample_count(&self) -> usize {
        self.samples.iter().filter(|s| s.is_drifted).count()
    }

    /// Whether any audited sample has drifted out of its cell.
    pub fn has_drift(&self) -> bool {
        self.samples.iter().any(|s| s.is_drifted)
    }

    /// Samples whose rebased resolution is coarser than the target.
    pub fn below_target_sample_count(&self) -> usize {
        self.samples.iter().filter(|s| !s.meets_target).count()
    }

    /// Whether every audited sample meets the target resolution (vacuously
    /// `true` with no samples).
    pub fn all_samples_meet_target(&self) -> bool {
        self.samples.iter().all(|s| s.meets_target)
    }

    /// Largest Chebyshev cell-delta among the samples — the farthest rebased
    /// entity. `0` with no samples.
    pub fn max_sample_cell_delta(&self) -> u64 {
        self.samples
            .iter()
            .map(|s| s.cell_delta_max)
            .max()
            .unwrap_or(0)
    }

    /// Coarsest (largest) rebased resolution among the samples, in metres.
    /// `0.0` with no samples.
    pub fn worst_sample_resolution_m(&self) -> f32 {
        let mut worst = 0.0f32;
        for s in &self.samples {
            if s.resolution_m > worst {
                worst = s.resolution_m;
            }
        }
        worst
    }

    /// Largest rebased magnitude among the samples, in metres. `0.0` with no
    /// samples.
    pub fn max_rebased_magnitude_m(&self) -> f32 {
        let mut m = 0.0f32;
        for s in &self.samples {
            if s.rebased_magnitude_m > m {
                m = s.rebased_magnitude_m;
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_is_monotone_coarsening() {
        let grid = FloatingOrigin::new(1024.0);
        let audit = FloatingOriginPrecisionAudit::from_origin(&grid, 0.001);
        assert!(audit.config_is_sane());
        assert!(audit.ring_count() > 0);
        // Resolution never improves as the radius grows, and radii are the
        // powers of two we swept.
        let mut prev_res = 0.0f32;
        let mut prev_cells = 0u32;
        for r in audit.rings() {
            assert!(r.resolution_m >= prev_res);
            assert!(r.radius_cells > prev_cells);
            prev_res = r.resolution_m;
            prev_cells = r.radius_cells;
        }
        // The origin cell resolves far better than a millimetre.
        assert!(audit.cell_boundary_resolution_m() < 0.001);
    }

    #[test]
    fn near_rings_meet_target_far_rings_fail() {
        let grid = FloatingOrigin::new(1024.0);
        let audit = FloatingOriginPrecisionAudit::from_origin(&grid, 0.001);
        // Some rings pass and some fail at a 1 mm target with 1 km cells.
        assert!(audit.safe_ring_count() > 0);
        assert!(audit.safe_ring_count() < audit.ring_count());
        assert!(!audit.all_rings_meet_target());
        let deepest = audit.deepest_safe_ring().expect("a safe ring");
        let first_bad = audit.first_unsafe_ring().expect("an unsafe ring");
        // The last safe ring is strictly nearer than the first failing one.
        assert!(deepest.radius_cells < first_bad.radius_cells);
        assert!(deepest.meets_target);
        assert!(!first_bad.meets_target);
    }

    #[test]
    fn safe_radius_matches_analytic_bound() {
        let grid = FloatingOrigin::new(1024.0);
        let audit = FloatingOriginPrecisionAudit::from_origin(&grid, 0.001);
        // 0.001 * 2^23 = 8388.608 m -> 8388.608 / 1024 = 8.19 -> 8 cells.
        assert!((audit.max_safe_radius_m() - 8388.608).abs() < 1.0);
        assert_eq!(audit.max_safe_radius_cells(), 8);
        // The deepest safe ladder ring agrees with the analytic cell bound.
        let deepest = audit.deepest_safe_ring().expect("a safe ring");
        assert!(deepest.radius_cells as i64 <= audit.max_safe_radius_cells());
    }

    #[test]
    fn config_only_has_no_samples() {
        let grid = FloatingOrigin::new(256.0);
        let audit = FloatingOriginPrecisionAudit::from_origin(&grid, 0.01);
        assert_eq!(audit.sample_count(), 0);
        assert!(!audit.has_drift());
        assert!(audit.all_samples_meet_target());
        assert_eq!(audit.max_sample_cell_delta(), 0);
        assert!(audit.worst_sample_resolution_m() <= f32::EPSILON);
        assert!(audit.max_rebased_magnitude_m() <= f32::EPSILON);
    }

    #[test]
    fn near_sample_is_precise_and_undrifted() {
        let grid = FloatingOrigin::new(1024.0).with_origin(GridCell::new(10_000, 0, 0));
        let samples = [(GridCell::new(10_000, 0, 0), LocalPos::new(12.5, 3.0, 7.0))];
        let audit = FloatingOriginPrecisionAudit::from_origin_with_samples(&grid, 0.001, &samples);
        assert_eq!(audit.sample_count(), 1);
        let s = audit.samples()[0];
        assert_eq!(s.cell_delta_max, 0);
        assert!((s.rebased_magnitude_m - 12.5).abs() < 1.0e-3);
        assert!(s.meets_target);
        assert!(!s.is_drifted);
        assert!(audit.all_samples_meet_target());
        assert!(!audit.has_drift());
    }

    #[test]
    fn far_sample_fails_target_and_is_counted() {
        // Origin at cell 0; an entity 4000 cells (≈ 4096 km) away rebases to a
        // huge f32 whose ULP is far coarser than a millimetre.
        let grid = FloatingOrigin::new(1024.0);
        let samples = [(GridCell::new(4000, 0, 0), LocalPos::new(1.0, 0.0, 0.0))];
        let audit = FloatingOriginPrecisionAudit::from_origin_with_samples(&grid, 0.001, &samples);
        let s = audit.samples()[0];
        assert_eq!(s.cell_delta_max, 4000);
        assert!(s.rebased_magnitude_m > 4_000_000.0);
        assert!(!s.meets_target);
        assert_eq!(audit.below_target_sample_count(), 1);
        assert_eq!(audit.max_sample_cell_delta(), 4000);
        assert!(audit.worst_sample_resolution_m() > 0.001);
        assert!(!audit.all_samples_meet_target());
    }

    #[test]
    fn drifted_local_is_flagged() {
        let grid = FloatingOrigin::new(100.0);
        let samples = [
            (GridCell::new(0, 0, 0), LocalPos::new(150.0, 0.0, 0.0)), // past +X edge
            (GridCell::new(0, 0, 0), LocalPos::new(10.0, -5.0, 0.0)), // below 0 on Y
            (GridCell::new(0, 0, 0), LocalPos::new(10.0, 20.0, 30.0)), // in cell
        ];
        let audit = FloatingOriginPrecisionAudit::from_origin_with_samples(&grid, 0.001, &samples);
        assert_eq!(audit.sample_count(), 3);
        assert_eq!(audit.drifted_sample_count(), 2);
        assert!(audit.has_drift());
        assert!(audit.samples()[0].is_drifted);
        assert!(audit.samples()[1].is_drifted);
        assert!(!audit.samples()[2].is_drifted);
    }

    #[test]
    fn origin_offset_reported_in_cells_and_metres() {
        let grid = FloatingOrigin::new(1024.0).with_origin(GridCell::new(10_000, -3, 7));
        let audit = FloatingOriginPrecisionAudit::from_origin(&grid, 0.001);
        assert_eq!(audit.origin(), GridCell::new(10_000, -3, 7));
        assert_eq!(audit.origin_offset_cells(), 10_000);
        assert!((audit.origin_world_offset_m() - 10_000.0 * 1024.0).abs() < 1.0);
    }
}
