//! Ocean-surface tile streaming and residency scheduling.
//!
//! [`ocean_lod`](super::ocean_lod) decides, for a surface patch at a known
//! camera distance, which `clipmap` ring and which spectral cascades drive it.
//! That answers *how finely to tessellate* what is already in memory. It does
//! not answer *what to keep in memory at all* once the playable ocean is far
//! larger than any affordable displacement/foam cache — the regime `UE5` large
//! worlds, `Crest`, and `WaveWorks` handle with tiled streaming: the sea plane
//! is cut into fixed world-space tiles, only the tiles near the camera hold
//! their displacement/depth/foam caches resident, and distant tiles are paged
//! out and re-paged as the camera moves.
//!
//! This module owns that residency decision as a pure, deterministic classifier
//! over a camera position and a view radius. It names the tiles that *should* be
//! resident this frame, each at the coarsest level of detail its distance
//! allows, with a screen-importance priority that falls off with distance. The
//! output plugs straight into the generic residency machine in
//! [`crate::paging`]: a tile key is the page key `K`, the per-tile priority is
//! the streaming priority, and the budgeted admit/evict policy is reused rather
//! than reinvented here.
//!
//! Everything is classical and allocation-light: tiles are enumerated by
//! integer index ranges, distances compare against level-of-detail bands, and
//! every returned list is sorted into tile-key order so a frame's request,
//! admit, and evict sequences are reproducible run to run. There is no GPU
//! handle, no transcendental math beyond the shared `sqrt`-based distance, and
//! no AI/ML anywhere.
//!
//! Provenance: standard distance-banded tile residency; algorithm-level only,
//! no `Unreal Engine`/`Crest` source or derived code.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::paging::RequestBatch;

use super::{Vec2, EPS};

/// Address of one ocean surface tile: its level of detail and integer grid
/// position at that level.
///
/// A tile at level `lod` tiles the sea plane into squares of
/// [`OceanTileConfig::lod_edge`] meters; `ix`/`iy` are the signed grid indices
/// of the tile, so the tile spans world `x` in `[ix*edge, (ix+1)*edge)` and `y`
/// likewise. The derived ordering (`lod`, then `iy`, then `ix`) gives a total,
/// deterministic key order so this type can serve directly as the page key `K`
/// in [`crate::paging`], whose containers all iterate in key order.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct OceanTileKey {
    /// Level of detail; `0` is the finest, each level doubles the tile edge.
    pub lod: u8,
    /// Signed tile index along world `y` at this level.
    pub iy: i32,
    /// Signed tile index along world `x` at this level.
    pub ix: i32,
}

impl OceanTileKey {
    /// Builds a tile key from its level and grid indices.
    #[must_use]
    pub const fn new(lod: u8, ix: i32, iy: i32) -> Self {
        Self { lod, iy, ix }
    }
}

/// Layout of the ocean tile pyramid: tile size, level count, and the radial
/// width of each level-of-detail band.
///
/// Level `l` uses tiles of `base_tile_size * 2^l` meters and covers the
/// distance band `[l*lod_band, (l+1)*lod_band)` from the camera; the last level
/// extends out to the planning view radius. Because the edge grows with the
/// level and the bands grow linearly, the resident tile count stays bounded as
/// the view radius grows: near water is fine and dense, far water is coarse and
/// sparse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OceanTileConfig {
    /// World-space edge of a level-`0` tile, in meters (`> 0`).
    pub base_tile_size: f32,
    /// Number of levels of detail (`>= 1`).
    pub lod_count: u8,
    /// Radial width of each level-of-detail distance band, in meters (`> 0`).
    pub lod_band: f32,
}

impl OceanTileConfig {
    /// Highest valid level index (`lod_count - 1`), or `0` when no levels exist.
    #[must_use]
    pub fn last_lod(self) -> u8 {
        self.lod_count.saturating_sub(1)
    }

    /// Whether the configuration is usable: positive tile size and band and at
    /// least one level. Planning against an invalid config yields no tiles.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.base_tile_size > EPS && self.lod_band > EPS && self.lod_count >= 1
    }

    /// World-space edge of a tile at `lod`, in meters.
    ///
    /// Computed as `base_tile_size * 2^lod` with a shift (no `powf`), clamping
    /// the level to the last valid one and capping the shift so it cannot
    /// overflow for pathological level counts.
    #[must_use]
    pub fn lod_edge(self, lod: u8) -> f32 {
        let shift = lod.min(self.last_lod()).min(31);
        self.base_tile_size * (1u32 << shift) as f32
    }

    /// Coarsest level whose distance band contains `distance`.
    ///
    /// Monotonic non-decreasing in `distance`: nearer water selects a finer
    /// (lower) level, farther water a coarser (higher) one, saturating at the
    /// last level. Non-positive distances map to level `0`.
    #[must_use]
    pub fn lod_for_distance(self, distance: f32) -> u8 {
        if distance <= 0.0 {
            return 0;
        }
        let idx = (distance / self.lod_band).floor();
        let last = f32::from(self.last_lod());
        if idx <= 0.0 {
            0
        } else if idx >= last {
            self.last_lod()
        } else {
            idx as u8
        }
    }

    /// World-space center of a tile.
    #[must_use]
    pub fn tile_center(self, key: OceanTileKey) -> Vec2 {
        let edge = self.lod_edge(key.lod);
        Vec2::new((key.ix as f32 + 0.5) * edge, (key.iy as f32 + 0.5) * edge)
    }
}

/// One tile the camera wants resident this frame, with its ranking inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OceanTileRequest {
    /// The tile to keep (or stream) resident.
    pub key: OceanTileKey,
    /// Screen importance in `0..=1`; higher survives eviction and streams in
    /// first. Falls off linearly with distance to the view-radius edge.
    pub priority: f32,
    /// Distance from the camera to the tile center, in meters.
    pub distance: f32,
}

/// Names the tiles that should be resident around `camera` out to `view_radius`.
///
/// Each covered tile is assigned the level its center distance selects (so a
/// tile belongs to exactly one level and the level rises monotonically with
/// distance), and a priority that fades from `1` at the camera to `0` at the
/// radius edge. The result is deduplicated by construction and returned in
/// tile-key order, so it is a stable, deterministic description of the frame's
/// desired residency set. An invalid config or a non-positive radius yields an
/// empty plan.
///
/// This does not enforce a residency budget; feed the result through
/// [`admit_within_budget`] or the [`crate::paging`] stream manager to bound it.
#[must_use]
pub fn plan_resident_tiles(
    camera: Vec2,
    view_radius: f32,
    cfg: OceanTileConfig,
) -> Vec<OceanTileRequest> {
    let mut out = Vec::new();
    if !cfg.is_valid() || view_radius <= EPS {
        return out;
    }

    for lod in 0..cfg.lod_count {
        let band_outer = if lod == cfg.last_lod() {
            view_radius
        } else {
            (f32::from(lod) + 1.0) * cfg.lod_band
        };
        let reach = band_outer.min(view_radius);
        if reach <= 0.0 {
            continue;
        }
        // A tile whose center lies within `reach` of the camera is a candidate;
        // its center must additionally fall in this level's distance band.
        let edge = cfg.lod_edge(lod);
        let min_ix = ((camera.x - reach) / edge).floor() as i32;
        let max_ix = ((camera.x + reach) / edge).floor() as i32;
        let min_iy = ((camera.y - reach) / edge).floor() as i32;
        let max_iy = ((camera.y + reach) / edge).floor() as i32;

        let mut iy = min_iy;
        while iy <= max_iy {
            let mut ix = min_ix;
            while ix <= max_ix {
                let key = OceanTileKey::new(lod, ix, iy);
                let distance = camera.distance(cfg.tile_center(key));
                if distance <= view_radius + EPS && cfg.lod_for_distance(distance) == lod {
                    let priority = (1.0 - distance / view_radius).clamp(0.0, 1.0);
                    out.push(OceanTileRequest {
                        key,
                        priority,
                        distance,
                    });
                }
                ix += 1;
            }
            iy += 1;
        }
    }

    out.sort_by_key(|req| req.key);
    out
}

/// Splits a desired tile set into the tiles that fit a residency budget and the
/// tiles deferred past it.
///
/// Admission is greedy by priority (highest first), breaking ties by tile key
/// so the choice is deterministic. At most `budget` tiles are admitted; the
/// rest are deferred. Both returned lists are in tile-key order. A `budget` at
/// or above the request count admits everything with no deferrals.
#[must_use]
pub fn admit_within_budget(
    requests: &[OceanTileRequest],
    budget: usize,
) -> (Vec<OceanTileKey>, Vec<OceanTileKey>) {
    let mut ranked: Vec<&OceanTileRequest> = requests.iter().collect();
    ranked.sort_by(|a, b| {
        b.priority
            .total_cmp(&a.priority)
            .then_with(|| a.key.cmp(&b.key))
    });

    let mut admitted = Vec::new();
    let mut deferred = Vec::new();
    for (rank, req) in ranked.into_iter().enumerate() {
        if rank < budget {
            admitted.push(req.key);
        } else {
            deferred.push(req.key);
        }
    }
    admitted.sort_unstable();
    deferred.sort_unstable();
    (admitted, deferred)
}

/// The per-frame residency transition: which desired tiles must be loaded,
/// which stay, and which currently-resident tiles can be evicted.
///
/// The three lists are disjoint and each is in tile-key order, so a transition
/// is fully described by this diff.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OceanTileDiff {
    /// Desired tiles not yet resident (stream in).
    pub load: Vec<OceanTileKey>,
    /// Desired tiles already resident (keep).
    pub keep: Vec<OceanTileKey>,
    /// Resident tiles no longer desired (evict).
    pub evict: Vec<OceanTileKey>,
}

/// Diffs a desired tile set against the currently resident set.
///
/// `desired` may contain duplicates or arbitrary order; it is treated as a set.
/// Returns the load/keep/evict partition in tile-key order.
#[must_use]
pub fn diff_tiles(desired: &[OceanTileKey], resident: &BTreeSet<OceanTileKey>) -> OceanTileDiff {
    let desired_set: BTreeSet<OceanTileKey> = desired.iter().copied().collect();
    let mut diff = OceanTileDiff::default();
    for &key in &desired_set {
        if resident.contains(&key) {
            diff.keep.push(key);
        } else {
            diff.load.push(key);
        }
    }
    for &key in resident {
        if !desired_set.contains(&key) {
            diff.evict.push(key);
        }
    }
    diff
}

/// Records a desired tile set into a [`RequestBatch`] for the generic streaming
/// controller, keeping each tile's streaming priority.
///
/// This is the seam that hands an ocean frame's residency plan to
/// [`crate::paging`]: flush the batch through
/// [`PageStreamManager::reconcile`](crate::paging::PageStreamManager::reconcile)
/// and the shared budgeted admit/evict policy takes over.
pub fn record_into_batch(requests: &[OceanTileRequest], batch: &mut RequestBatch<OceanTileKey>) {
    for req in requests {
        batch.record(req.key, req.priority);
    }
}

/// Desired tiles grouped by level of detail, mirroring
/// [`OceanClipmapPlan`](super::ocean_lod::OceanClipmapPlan).
///
/// A geometry/streaming stage walks each level's bucket to size per-level tile
/// caches without re-filtering the flat request list.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OceanTilePlan {
    buckets: Vec<Vec<OceanTileRequest>>,
}

impl OceanTilePlan {
    /// Builds an empty plan with `lod_count` buckets.
    #[must_use]
    pub fn with_lod_count(lod_count: u8) -> Self {
        let mut buckets = Vec::new();
        buckets.resize_with(lod_count as usize, Vec::new);
        Self { buckets }
    }

    /// Number of level buckets.
    #[must_use]
    pub fn lod_count(&self) -> usize {
        self.buckets.len()
    }

    /// Total tiles across all levels.
    #[must_use]
    pub fn total(&self) -> usize {
        self.buckets.iter().map(Vec::len).sum()
    }

    /// Whether the plan holds no tiles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    /// Tiles at a level, or an empty slice when the level is out of range.
    #[must_use]
    pub fn bucket(&self, lod: u8) -> &[OceanTileRequest] {
        self.buckets.get(lod as usize).map_or(&[], Vec::as_slice)
    }

    /// Tile count at a level.
    #[must_use]
    pub fn count_of_lod(&self, lod: u8) -> usize {
        self.bucket(lod).len()
    }

    /// Adds a tile to its level bucket, ignoring out-of-range levels so a stray
    /// level never panics the geometry stage.
    pub fn push(&mut self, request: OceanTileRequest) {
        if let Some(bucket) = self.buckets.get_mut(request.key.lod as usize) {
            bucket.push(request);
        }
    }
}

/// Bins a flat request list into per-level buckets, preserving tile-key order
/// within each level and dropping requests whose level exceeds `lod_count`.
#[must_use]
pub fn bin_tiles(requests: &[OceanTileRequest], lod_count: u8) -> OceanTilePlan {
    let mut plan = OceanTilePlan::with_lod_count(lod_count);
    for &req in requests {
        plan.push(req);
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paging::{PageSource, PageStreamManager, ResidencyTable};
    use alloc::vec;

    const CFG: OceanTileConfig = OceanTileConfig {
        base_tile_size: 10.0,
        lod_count: 3,
        lod_band: 30.0,
    };

    #[test]
    fn lod_edge_doubles_per_level_and_clamps() {
        assert!((CFG.lod_edge(0) - 10.0).abs() < EPS);
        assert!((CFG.lod_edge(1) - 20.0).abs() < EPS);
        assert!((CFG.lod_edge(2) - 40.0).abs() < EPS);
        // Out-of-range level clamps to the last valid edge, never panics.
        assert!((CFG.lod_edge(9) - CFG.lod_edge(2)).abs() < EPS);
    }

    #[test]
    fn lod_for_distance_is_monotonic() {
        let mut prev = 0u8;
        let mut d = 0.0;
        while d < 200.0 {
            let lod = CFG.lod_for_distance(d);
            assert!(lod >= prev, "lod must not decrease with distance");
            assert!(lod <= CFG.last_lod());
            prev = lod;
            d += 1.0;
        }
        assert_eq!(CFG.lod_for_distance(-5.0), 0);
        assert_eq!(CFG.lod_for_distance(0.0), 0);
        assert_eq!(CFG.lod_for_distance(29.0), 0);
        assert_eq!(CFG.lod_for_distance(31.0), 1);
        assert_eq!(CFG.lod_for_distance(1_000.0), CFG.last_lod());
    }

    #[test]
    fn empty_plan_on_invalid_or_zero_radius() {
        assert!(plan_resident_tiles(Vec2::ZERO, 0.0, CFG).is_empty());
        assert!(plan_resident_tiles(Vec2::ZERO, -1.0, CFG).is_empty());
        let bad = OceanTileConfig {
            base_tile_size: 0.0,
            lod_count: 0,
            lod_band: 0.0,
        };
        assert!(plan_resident_tiles(Vec2::ZERO, 100.0, bad).is_empty());
    }

    #[test]
    fn plan_is_deterministic_and_key_ordered() {
        let a = plan_resident_tiles(Vec2::new(3.0, -7.0), 80.0, CFG);
        let b = plan_resident_tiles(Vec2::new(3.0, -7.0), 80.0, CFG);
        assert_eq!(a, b, "same input must give identical plan");
        assert!(!a.is_empty());
        for pair in a.windows(2) {
            assert!(pair[0].key < pair[1].key, "requests must be key-ordered");
        }
    }

    #[test]
    fn plan_tiles_lie_within_radius_at_their_distance_level() {
        let camera = Vec2::new(12.5, -4.0);
        let radius = 90.0;
        let plan = plan_resident_tiles(camera, radius, CFG);
        assert!(!plan.is_empty());
        for req in &plan {
            // Each tile is at the level its own center distance selects, so the
            // level rises monotonically with distance across the whole plan.
            assert_eq!(CFG.lod_for_distance(req.distance), req.key.lod);
            assert!(req.distance <= radius + EPS);
            assert!((0.0..=1.0).contains(&req.priority));
            // Nearer tiles outrank farther tiles.
            let expect = (1.0 - req.distance / radius).clamp(0.0, 1.0);
            assert!((req.priority - expect).abs() < EPS);
        }
    }

    #[test]
    fn nearer_tiles_have_higher_priority() {
        let plan = plan_resident_tiles(Vec2::ZERO, 90.0, CFG);
        let mut near = plan[0];
        let mut far = plan[0];
        for req in &plan {
            if req.distance < near.distance {
                near = *req;
            }
            if req.distance > far.distance {
                far = *req;
            }
        }
        assert!(near.priority >= far.priority);
    }

    #[test]
    fn admit_within_budget_conserves_and_prioritizes() {
        let plan = plan_resident_tiles(Vec2::ZERO, 90.0, CFG);
        let budget = plan.len() / 2;
        let (admitted, deferred) = admit_within_budget(&plan, budget);
        assert_eq!(admitted.len(), budget);
        assert_eq!(admitted.len() + deferred.len(), plan.len());
        // No key appears in both partitions.
        let adm: BTreeSet<OceanTileKey> = admitted.iter().copied().collect();
        for key in &deferred {
            assert!(!adm.contains(key));
        }
        // The lowest admitted priority is at least the highest deferred one.
        let min_admit = plan
            .iter()
            .filter(|r| adm.contains(&r.key))
            .map(|r| r.priority)
            .fold(f32::INFINITY, f32::min);
        let def: BTreeSet<OceanTileKey> = deferred.iter().copied().collect();
        let max_defer = plan
            .iter()
            .filter(|r| def.contains(&r.key))
            .map(|r| r.priority)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(min_admit + EPS >= max_defer);
    }

    #[test]
    fn admit_everything_when_budget_exceeds_demand() {
        let plan = plan_resident_tiles(Vec2::ZERO, 60.0, CFG);
        let (admitted, deferred) = admit_within_budget(&plan, plan.len() + 10);
        assert_eq!(admitted.len(), plan.len());
        assert!(deferred.is_empty());
    }

    #[test]
    fn diff_partitions_load_keep_evict() {
        let a = OceanTileKey::new(0, 0, 0);
        let b = OceanTileKey::new(0, 1, 0);
        let c = OceanTileKey::new(1, 0, 0);
        let desired = [b, a, b]; // unordered + duplicate
        let resident: BTreeSet<OceanTileKey> = [a, c].into_iter().collect();
        let diff = diff_tiles(&desired, &resident);
        assert_eq!(diff.load, vec![b]);
        assert_eq!(diff.keep, vec![a]);
        assert_eq!(diff.evict, vec![c]);
    }

    #[test]
    fn bin_tiles_groups_by_level_and_drops_overflow() {
        let plan = plan_resident_tiles(Vec2::ZERO, 90.0, CFG);
        let binned = bin_tiles(&plan, CFG.lod_count);
        assert_eq!(binned.total(), plan.len());
        let mut sum = 0;
        for lod in 0..CFG.lod_count {
            sum += binned.count_of_lod(lod);
            for req in binned.bucket(lod) {
                assert_eq!(req.key.lod, lod);
            }
        }
        assert_eq!(sum, plan.len());
        // A level beyond the bucket count is dropped, never panics.
        let narrow = bin_tiles(&plan, 1);
        assert!(narrow.total() <= plan.len());
        assert_eq!(narrow.bucket(9).len(), 0);
    }

    /// A page source handing out a fixed-size zero payload: tile streaming only
    /// needs residency, not page contents, so the backing words are irrelevant.
    struct ZeroTileSource {
        page_words: usize,
    }

    impl PageSource<OceanTileKey> for ZeroTileSource {
        fn load(&mut self, _key: OceanTileKey) -> Vec<u32> {
            vec![0u32; self.page_words]
        }
    }

    #[test]
    fn record_into_batch_drives_residency_table() {
        let plan = plan_resident_tiles(Vec2::ZERO, 60.0, CFG);
        assert!(!plan.is_empty());
        let mut batch = RequestBatch::new();
        record_into_batch(&plan, &mut batch);
        assert_eq!(batch.len(), plan.len());

        let mut table: ResidencyTable<OceanTileKey> = ResidencyTable::new();
        batch.flush(&mut table, 1);
        assert_eq!(table.pending_requests().len(), plan.len());
        for req in &plan {
            table.mark_resident(req.key);
        }
        assert_eq!(table.resident_count(), plan.len());
    }

    #[test]
    fn paging_manager_streams_tiles_within_budget() {
        let plan = plan_resident_tiles(Vec2::ZERO, 90.0, CFG);
        let budget = plan.len() / 2;
        assert!(budget > 0);
        let capacity = plan.len() as u32;
        let mut manager: PageStreamManager<OceanTileKey> =
            PageStreamManager::new(capacity, 4, budget);
        let mut source = ZeroTileSource { page_words: 4 };

        let mut batch = RequestBatch::new();
        record_into_batch(&plan, &mut batch);
        let report = manager.reconcile(&batch, 1, &mut source);

        // The budget caps residency; the rest are deferred, nothing lost.
        assert_eq!(manager.resident_count(), budget);
        assert_eq!(report.streamed_in.len(), budget);
        assert_eq!(report.streamed_in.len() + report.deferred.len(), plan.len());
    }
}
