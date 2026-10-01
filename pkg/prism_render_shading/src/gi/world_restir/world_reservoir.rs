//! World-space reservoirs: hashed RIS fill + cross-cell GRIS reuse — CPU golden.
//!
//! This module turns the backend-neutral [`Reservoir`] into the storage and
//! resampling backend of a SHARC-style world-space ReSTIR cache.  Candidate GI
//! paths are streamed into per-cell reservoirs addressed by the
//! [`super::spatial_hash`] key, and neighbouring cells are then recombined with
//! generalized resampled importance sampling (`GRIS`) so a shading point draws
//! on the pooled history of its whole neighbourhood rather than one noisy cell.
//!
//! The pipeline mirrors a frame of world-space ReSTIR:
//!
//! * **Fill** — [`stream_candidate`] folds one path sample into a cell's
//!   reservoir with the resampling weight `w = target / source_pdf` (standard
//!   RIS).  [`WorldHashGrid::insert_candidate`] routes the sample to the right
//!   slot, allocating it on first use and rejecting hash collisions by
//!   [`checksum`].
//! * **Finalize** — [`finalize`] (and [`WorldHashGrid::finalize_all`]) convert
//!   each cell's accumulated state into the unbiased contribution weight `W`
//!   using the selected sample's own target density.
//! * **Reuse** — [`merge_spatial`] re-evaluates a neighbour's sample from the
//!   destination's shading point (a *reconnection shift*, [`reconnection_target`])
//!   and feeds the induced weight `other.m * p_hat * other.W` into
//!   [`Reservoir::merge`]; [`WorldHashGrid::spatial_reuse`] drives this across a
//!   ring of neighbour cells and re-finalizes the pooled result.
//!
//! # Conventions
//! * Reservoirs reuse [`crate::gi::screen_probe::restir`] verbatim:
//!   [`GiSample`] is the payload, [`target_function`] is the scalar target
//!   `p_hat`, [`Reservoir::merge`] applies the `GRIS` pairing-count MIS, and
//!   [`balance_heuristic`] backs the explicit [`pairwise_mis_weight`] helper.
//! * World-space samples are view independent: `sample_point`, `sample_normal`,
//!   and `radiance` are reused unchanged, while a spatial reuse re-evaluates the
//!   geometry against the destination's `visible_point` / `visible_normal`.  The
//!   reservoir therefore stores the neighbour's record as-is and the consumer
//!   reconnects it at query time with its own visible point.
//! * The table is open-addressed with linear probing capped at
//!   [`PROBE_LIMIT`]; a full probe window drops the sample rather than evicting
//!   a live reservoir, keeping the fill deterministic and bounded.
//! * Every helper is a deterministic pure function: randomness enters only as
//!   caller-supplied uniforms `u in [0, 1)`, and every weight is clamped finite
//!   and non-negative so the cache can never inject `NaN` energy.  State is
//!   `f32`-packed to match the GPU reservoir-buffer twin.

use alloc::vec::Vec;

use bevy_math::Vec3;

use crate::gi::screen_probe::restir::{balance_heuristic, target_function, GiSample, Reservoir};

use super::spatial_hash::{bucket_index, checksum, HashGridKey};

/// Maximum linear-probing steps when locating a key's slot.
pub const PROBE_LIMIT: u32 = 32;

/// Resampled-importance weight of a candidate: `target_function / source_pdf`.
///
/// Returns `0` for a non-positive / non-finite `source_pdf` or a non-finite
/// quotient, so a degenerate candidate carries no weight and is discarded by
/// the reservoir's own [`Reservoir::update`] guard.
#[inline]
pub fn ris_weight(sample: &GiSample, source_pdf: f32) -> f32 {
    if !source_pdf.is_finite() || source_pdf <= 0.0 {
        return 0.0;
    }
    let w = target_function(sample) / source_pdf;
    if w.is_finite() {
        w.max(0.0)
    } else {
        0.0
    }
}

/// Streams one candidate path into a reservoir using the external uniform `u`.
///
/// Computes the RIS weight via [`ris_weight`] and folds the sample in with
/// [`Reservoir::update`].  Returns `true` when the candidate became the
/// surviving sample.
#[inline]
pub fn stream_candidate(
    reservoir: &mut Reservoir<GiSample>,
    sample: GiSample,
    source_pdf: f32,
    u: f32,
) -> bool {
    let weight = ris_weight(&sample, source_pdf);
    reservoir.update(sample, weight, u)
}

/// Finalizes a reservoir with the selected sample's own target density.
///
/// Equivalent to `finalize_weight(target_function(selected))`; an empty
/// reservoir is finalized to `W = 0`.
#[inline]
pub fn finalize(reservoir: &mut Reservoir<GiSample>) {
    match reservoir.sample() {
        Some(sample) => {
            let p_hat = target_function(&sample);
            reservoir.finalize_weight(p_hat);
        }
        None => reservoir.finalize_weight(0.0),
    }
}

/// Re-evaluates a neighbour's sample as seen from a destination shading point.
///
/// This is the *reconnection shift map*: the sample's secondary point, normal,
/// and radiance (all view independent) are kept, while the visible point and
/// normal are replaced by the destination's.  Returns the shift-mapped target
/// density `p_hat` to drive the `GRIS` induced weight; it is finite and
/// non-negative by [`target_function`]'s contract.
#[inline]
pub fn reconnection_target(
    neighbor_sample: &GiSample,
    visible_point: Vec3,
    visible_normal: Vec3,
) -> f32 {
    let shifted = GiSample {
        visible_point,
        visible_normal,
        ..*neighbor_sample
    };
    target_function(&shifted)
}

/// Merges a neighbour reservoir into `dst` under `GRIS` spatial reuse.
///
/// The neighbour's selected sample is reconnected to `dst`'s shading point via
/// [`reconnection_target`], and the resulting density is passed to
/// [`Reservoir::merge`] (whose induced weight is `other.m * p_hat * other.W`).
/// `dst` and the neighbour must already be finalized.  Returns `true` when the
/// neighbour's sample was selected; an empty neighbour is a no-op.
#[inline]
pub fn merge_spatial(
    dst: &mut Reservoir<GiSample>,
    visible_point: Vec3,
    visible_normal: Vec3,
    neighbor: &Reservoir<GiSample>,
    u: f32,
) -> bool {
    match neighbor.sample() {
        Some(sample) => {
            let p_hat = reconnection_target(&sample, visible_point, visible_normal);
            dst.merge(neighbor, p_hat, u)
        }
        None => false,
    }
}

/// Balance-heuristic MIS weight for the neighbour term of a two-way reuse.
///
/// A thin wrapper over [`balance_heuristic`] with the canonical reservoir as
/// technique `0` and the neighbour as technique `1`; it returns the neighbour's
/// pairing-count weight `(c_n p_n) / (c_c p_c + c_n p_n)`.  Useful for pairwise
/// reuse schemes that weight neighbours explicitly instead of relying on the
/// streaming confidence accumulation inside [`Reservoir::merge`].
#[inline]
pub fn pairwise_mis_weight(
    canonical_pdf: f32,
    canonical_count: f32,
    neighbor_pdf: f32,
    neighbor_count: f32,
) -> f32 {
    let pdfs = [canonical_pdf, neighbor_pdf];
    let counts = [canonical_count, neighbor_count];
    balance_heuristic(&pdfs, &counts, 1)
}

/// One slot of the open-addressed reservoir table.
///
/// Stores the cell's reservoir plus the owning key's [`checksum`] and an
/// occupancy flag, so a probe can tell an empty slot from a colliding one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldCell {
    /// The cell's reservoir of GI samples.
    pub reservoir: Reservoir<GiSample>,
    /// Collision checksum of the key that owns this slot (valid iff `occupied`).
    pub checksum: u32,
    /// Whether this slot has been claimed by a key.
    pub occupied: bool,
}

impl Default for WorldCell {
    #[inline]
    fn default() -> Self {
        Self::EMPTY
    }
}

impl WorldCell {
    /// A free, empty slot.
    pub const EMPTY: Self = Self {
        reservoir: Reservoir::EMPTY,
        checksum: 0,
        occupied: false,
    };
}

/// A fixed-capacity, open-addressed world-space reservoir table.
///
/// Keys hash to a bucket ([`bucket_index`]) and probe linearly up to
/// [`PROBE_LIMIT`] slots; each slot records a [`checksum`] so colliding keys are
/// kept apart.  The table never grows or evicts, matching the bounded GPU
/// buffer twin.
#[derive(Clone, Debug)]
pub struct WorldHashGrid {
    cells: Vec<WorldCell>,
    capacity: u32,
}

impl WorldHashGrid {
    /// Creates an empty table with `capacity` slots (clamped to at least `1`).
    #[inline]
    pub fn new(capacity: u32) -> Self {
        let cap = capacity.max(1);
        let mut cells = Vec::with_capacity(cap as usize);
        for _ in 0..cap {
            cells.push(WorldCell::EMPTY);
        }
        Self { cells, capacity: cap }
    }

    /// Number of slots in the table.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Number of currently occupied slots.
    #[inline]
    pub fn occupied_len(&self) -> usize {
        self.cells.iter().filter(|c| c.occupied).count()
    }

    /// Finds the slot owning `key`, or the first free slot in its probe window,
    /// claiming that free slot for `key`.  Returns `None` when the probe window
    /// is exhausted (the table is locally full).
    #[inline]
    fn find_or_alloc(&mut self, key: &HashGridKey) -> Option<usize> {
        let cap = self.capacity;
        let cs = checksum(key);
        let base = bucket_index(key, cap);
        let steps = PROBE_LIMIT.min(cap);
        for i in 0..steps {
            let idx = ((base + i) % cap) as usize;
            let cell = &mut self.cells[idx];
            if !cell.occupied {
                cell.occupied = true;
                cell.checksum = cs;
                cell.reservoir = Reservoir::new();
                return Some(idx);
            }
            if cell.checksum == cs {
                return Some(idx);
            }
        }
        None
    }

    /// Finds the slot owning `key`, if it is present.  Stops at the first free
    /// slot in the probe window because an inserted key would have claimed it.
    #[inline]
    fn find_slot(&self, key: &HashGridKey) -> Option<usize> {
        let cap = self.capacity;
        let cs = checksum(key);
        let base = bucket_index(key, cap);
        let steps = PROBE_LIMIT.min(cap);
        for i in 0..steps {
            let idx = ((base + i) % cap) as usize;
            let cell = &self.cells[idx];
            if !cell.occupied {
                return None;
            }
            if cell.checksum == cs {
                return Some(idx);
            }
        }
        None
    }

    /// Streams one candidate into the reservoir for `key`.
    ///
    /// Allocates the slot on first use.  Returns `true` when the candidate
    /// became the surviving sample, `false` when it was discarded or the probe
    /// window was full.
    #[inline]
    pub fn insert_candidate(
        &mut self,
        key: &HashGridKey,
        sample: GiSample,
        source_pdf: f32,
        u: f32,
    ) -> bool {
        match self.find_or_alloc(key) {
            Some(idx) => stream_candidate(&mut self.cells[idx].reservoir, sample, source_pdf, u),
            None => false,
        }
    }

    /// Borrows the reservoir for `key`, if the cell exists.
    #[inline]
    pub fn reservoir(&self, key: &HashGridKey) -> Option<&Reservoir<GiSample>> {
        self.find_slot(key).map(|idx| &self.cells[idx].reservoir)
    }

    /// Mutably borrows the reservoir for `key`, if the cell exists.
    #[inline]
    pub fn reservoir_mut(&mut self, key: &HashGridKey) -> Option<&mut Reservoir<GiSample>> {
        match self.find_slot(key) {
            Some(idx) => Some(&mut self.cells[idx].reservoir),
            None => None,
        }
    }

    /// Finalizes every occupied cell in place via [`finalize`].
    #[inline]
    pub fn finalize_all(&mut self) {
        for cell in self.cells.iter_mut() {
            if cell.occupied {
                finalize(&mut cell.reservoir);
            }
        }
    }

    /// Pools the `center` cell with its `neighbors` under `GRIS` spatial reuse.
    ///
    /// Starts from a copy of the center reservoir, merges each existing
    /// neighbour (reconnected to `visible_point` / `visible_normal`) with the
    /// matching uniform from `us` (defaulting to `0.5` when `us` is shorter),
    /// then re-finalizes the pooled reservoir.  All cells should already be
    /// finalized via [`finalize_all`].  Returns the pooled reservoir, which is
    /// empty when the center cell does not exist and no neighbour is selected.
    #[inline]
    pub fn spatial_reuse(
        &self,
        center: &HashGridKey,
        neighbors: &[HashGridKey],
        visible_point: Vec3,
        visible_normal: Vec3,
        us: &[f32],
    ) -> Reservoir<GiSample> {
        let mut out = self
            .reservoir(center)
            .copied()
            .unwrap_or_else(Reservoir::new);
        for (i, neighbor_key) in neighbors.iter().enumerate() {
            if let Some(neighbor) = self.reservoir(neighbor_key) {
                let u = us.get(i).copied().unwrap_or(0.5);
                merge_spatial(&mut out, visible_point, visible_normal, neighbor, u);
            }
        }
        finalize(&mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a facing visible/sample pair with the given radiance so the
    /// target function is a predictable non-zero value.
    fn facing_sample(radiance: Vec3) -> GiSample {
        GiSample {
            visible_point: Vec3::ZERO,
            visible_normal: Vec3::Z,
            sample_point: Vec3::new(0.0, 0.0, 1.0),
            sample_normal: Vec3::NEG_Z,
            radiance,
        }
    }

    fn key(cell: (i32, i32, i32), normal_bin: u32) -> HashGridKey {
        HashGridKey {
            cell_coord: bevy_math::IVec3::new(cell.0, cell.1, cell.2),
            level: 0,
            normal_bin,
        }
    }

    #[test]
    fn ris_weight_guards_degenerate_pdf() {
        let s = facing_sample(Vec3::ONE);
        assert!(ris_weight(&s, 1.0) > 0.0);
        assert_eq!(ris_weight(&s, 0.0), 0.0);
        assert_eq!(ris_weight(&s, -1.0), 0.0);
        assert_eq!(ris_weight(&s, f32::NAN), 0.0);
        // A dark sample has zero target, hence zero weight.
        assert_eq!(ris_weight(&facing_sample(Vec3::ZERO), 1.0), 0.0);
    }

    #[test]
    fn stream_candidate_fills_and_finalizes() {
        let mut r = Reservoir::<GiSample>::new();
        let s = facing_sample(Vec3::ONE);
        // Source pdf below the target -> weight > 1, candidate accepted.
        assert!(stream_candidate(&mut r, s, 0.5, 0.0));
        assert!(r.weight_sum() > 0.0);
        assert_eq!(r.confidence(), 1.0);
        finalize(&mut r);
        // W = (w_sum / m) / p_hat = (target/0.5) / target = 2.
        assert!((r.contribution_weight() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn empty_reservoir_finalizes_to_zero() {
        let mut r = Reservoir::<GiSample>::new();
        finalize(&mut r);
        assert_eq!(r.contribution_weight(), 0.0);
    }

    #[test]
    fn reconnection_target_tracks_visible_point() {
        let neighbor = facing_sample(Vec3::ONE);
        // Reconnecting to the original visible point reproduces the target.
        let p0 = reconnection_target(&neighbor, Vec3::ZERO, Vec3::Z);
        assert!((p0 - target_function(&neighbor)).abs() < 1e-6);
        // A back-facing visible normal kills the geometric term.
        let p1 = reconnection_target(&neighbor, Vec3::ZERO, Vec3::NEG_Z);
        assert_eq!(p1, 0.0);
        // Moving the visible point far away reduces the target (inverse square).
        let p2 = reconnection_target(&neighbor, Vec3::new(0.0, 0.0, -3.0), Vec3::Z);
        assert!(p2 < p0 && p2 > 0.0, "p2={p2} p0={p0}");
    }

    #[test]
    fn merge_spatial_selects_strong_neighbor() {
        // Canonical cell: a weak sample.
        let mut canonical = Reservoir::<GiSample>::new();
        stream_candidate(&mut canonical, facing_sample(Vec3::splat(0.01)), 1.0, 0.0);
        finalize(&mut canonical);

        // Neighbour cell: a strong sample with lots of history.
        let mut neighbor = Reservoir::<GiSample>::new();
        for _ in 0..16 {
            stream_candidate(&mut neighbor, facing_sample(Vec3::splat(10.0)), 1.0, 0.0);
        }
        finalize(&mut neighbor);

        // u = 0 forces selection of the induced neighbour weight.
        let selected = merge_spatial(&mut canonical, Vec3::ZERO, Vec3::Z, &neighbor, 0.0);
        assert!(selected);
        assert!(canonical.confidence() >= 17.0);
        finalize(&mut canonical);
        assert!(canonical.contribution_weight() > 0.0);
    }

    #[test]
    fn merge_spatial_empty_neighbor_is_noop() {
        let mut canonical = Reservoir::<GiSample>::new();
        stream_candidate(&mut canonical, facing_sample(Vec3::ONE), 1.0, 0.0);
        let empty = Reservoir::<GiSample>::new();
        assert!(!merge_spatial(&mut canonical, Vec3::ZERO, Vec3::Z, &empty, 0.0));
        assert_eq!(canonical.confidence(), 1.0);
    }

    #[test]
    fn pairwise_mis_weight_partitions_with_canonical() {
        // Equal pdfs & counts -> each technique weighted 0.5.
        let w = pairwise_mis_weight(2.0, 1.0, 2.0, 1.0);
        assert!((w - 0.5).abs() < 1e-6);
        // A stronger, more-confident neighbour gets more weight.
        let w = pairwise_mis_weight(1.0, 1.0, 3.0, 2.0);
        assert!((w - (6.0 / 7.0)).abs() < 1e-6, "w={w}");
    }

    #[test]
    fn grid_insert_and_query_roundtrip() {
        let mut grid = WorldHashGrid::new(256);
        let k = key((1, 2, 3), 5);
        assert!(grid.reservoir(&k).is_none());
        assert!(grid.insert_candidate(&k, facing_sample(Vec3::ONE), 0.5, 0.0));
        assert_eq!(grid.occupied_len(), 1);
        let r = grid.reservoir(&k).expect("cell exists");
        assert_eq!(r.confidence(), 1.0);
        // A second sample into the same key reuses the slot (no new allocation).
        assert!(grid.insert_candidate(&k, facing_sample(Vec3::splat(2.0)), 0.5, 0.0));
        assert_eq!(grid.occupied_len(), 1);
        assert_eq!(grid.reservoir(&k).unwrap().confidence(), 2.0);
    }

    #[test]
    fn grid_separates_distinct_keys() {
        let mut grid = WorldHashGrid::new(256);
        let a = key((0, 0, 0), 0);
        let b = key((0, 0, 0), 1); // same cell, different normal bin.
        grid.insert_candidate(&a, facing_sample(Vec3::ONE), 1.0, 0.0);
        grid.insert_candidate(&b, facing_sample(Vec3::splat(5.0)), 1.0, 0.0);
        assert_eq!(grid.occupied_len(), 2);
        assert!(grid.reservoir(&a).is_some());
        assert!(grid.reservoir(&b).is_some());
    }

    #[test]
    fn grid_finalize_all_sets_weights() {
        let mut grid = WorldHashGrid::new(64);
        let k = key((4, 4, 4), 2);
        grid.insert_candidate(&k, facing_sample(Vec3::ONE), 0.5, 0.0);
        grid.finalize_all();
        let w = grid.reservoir(&k).unwrap().contribution_weight();
        assert!((w - 2.0).abs() < 1e-5, "W={w}");
    }

    #[test]
    fn spatial_reuse_pools_neighbors() {
        let mut grid = WorldHashGrid::new(512);
        let center = key((0, 0, 0), 0);
        let neighbor = key((1, 0, 0), 0);
        // Weak center, strong neighbour.
        grid.insert_candidate(&center, facing_sample(Vec3::splat(0.01)), 1.0, 0.0);
        for _ in 0..8 {
            grid.insert_candidate(&neighbor, facing_sample(Vec3::splat(5.0)), 1.0, 0.0);
        }
        grid.finalize_all();
        let pooled = grid.spatial_reuse(&center, &[neighbor], Vec3::ZERO, Vec3::Z, &[0.0]);
        // Pooling accumulates both histories and stays finite / non-negative.
        assert!(pooled.confidence() >= 9.0);
        assert!(pooled.contribution_weight().is_finite());
        assert!(pooled.contribution_weight() >= 0.0);
    }

    #[test]
    fn spatial_reuse_missing_center_is_empty() {
        let grid = WorldHashGrid::new(16);
        let center = key((9, 9, 9), 0);
        let pooled = grid.spatial_reuse(&center, &[], Vec3::ZERO, Vec3::Z, &[]);
        assert!(pooled.is_empty());
        assert_eq!(pooled.contribution_weight(), 0.0);
    }

    #[test]
    fn results_are_deterministic() {
        let build = || {
            let mut grid = WorldHashGrid::new(128);
            let k = key((2, -1, 3), 4);
            grid.insert_candidate(&k, facing_sample(Vec3::ONE), 0.7, 0.25);
            grid.insert_candidate(&k, facing_sample(Vec3::splat(2.0)), 0.4, 0.8);
            grid.finalize_all();
            grid.reservoir(&k).unwrap().contribution_weight()
        };
        assert_eq!(build(), build());
    }
}
