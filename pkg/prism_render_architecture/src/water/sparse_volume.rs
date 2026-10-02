//! Sparse `FLIP`/`APIC` volume brick residency.
//!
//! A film- or game-scale liquid volume cannot afford a dense `MAC` grid over
//! its whole bounding box: the fluid occupies a thin, moving sheet of that box
//! and the rest is empty air. Production fluid engines (`Houdini`, `OpenVDB`,
//! and `VDB`-style sparse grids generally) therefore allocate the grid in
//! fixed-size *bricks* — small `B×B×B` blocks of cells — and keep resident only
//! the bricks that actually contain fluid, plus a thin apron of neighbor bricks
//! so advection and the pressure stencil can reach across brick seams without
//! sampling unallocated memory.
//!
//! [`flip`](super::flip) owns the per-cell transfer and projection numerics and
//! assumes the cells it touches exist. This module owns the complementary
//! decision those numerics depend on: given where the fluid particles are this
//! frame, which bricks must be resident, at what priority, and which previously
//! resident bricks can be released. It is the volumetric sibling of
//! [`tile_stream`](super::tile_stream): a brick coordinate is the page key `K`
//! for the generic residency machine in [`crate::paging`], occupied bricks
//! outrank apron bricks, and the budgeted admit/evict policy is reused rather
//! than reinvented.
//!
//! Everything is classical, deterministic, and allocation-light: points map to
//! bricks by integer floor division, the active set dilates by a bounded apron,
//! and every returned list is in brick-key order so a frame's request, admit,
//! and evict sequences are reproducible run to run. Only the shared `sqrt`-based
//! distance is used; there is no `f32` equality test and no AI/ML.
//!
//! Provenance: standard sparse-grid brick residency; algorithm-level only, no
//! `Houdini`/`OpenVDB` source or derived code.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::paging::RequestBatch;

use super::{Vec3, EPS};

/// Integer coordinate of one sparse volume brick.
///
/// A brick at `(bx, by, bz)` spans world `x` in `[bx*edge, (bx+1)*edge)` and
/// `y`/`z` likewise, where `edge` is [`SparseVolumeConfig::brick_edge`]. The
/// derived ordering (`bz`, then `by`, then `bx`) is a total, deterministic key
/// order so this type serves directly as the page key `K` in
/// [`crate::paging`], whose containers iterate in key order.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct BrickCoord {
    /// Signed brick index along world `z`.
    pub bz: i32,
    /// Signed brick index along world `y`.
    pub by: i32,
    /// Signed brick index along world `x`.
    pub bx: i32,
}

impl BrickCoord {
    /// Builds a brick coordinate from its signed indices.
    #[must_use]
    pub const fn new(bx: i32, by: i32, bz: i32) -> Self {
        Self { bz, by, bx }
    }
}

/// Layout of the sparse volume: cell size, brick size, and apron width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseVolumeConfig {
    /// World-space edge of one `MAC` cell, in meters (`> 0`).
    pub cell_size: f32,
    /// Cells along one brick edge (`>= 1`); a brick holds `brick_cells^3` cells.
    pub brick_cells: u32,
    /// Apron width in bricks dilated around the occupied set so stencils and
    /// advection reach across brick seams. `0` keeps only occupied bricks.
    pub apron: u32,
}

impl SparseVolumeConfig {
    /// Whether the configuration is usable: positive cell size and at least one
    /// cell per brick edge. Planning against an invalid config yields no bricks.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.cell_size > EPS && self.brick_cells >= 1
    }

    /// World-space edge of a brick, in meters.
    #[must_use]
    pub fn brick_edge(self) -> f32 {
        self.cell_size * self.brick_cells as f32
    }

    /// Brick containing a world-space point (floor division by the brick edge).
    #[must_use]
    pub fn brick_of_point(self, p: Vec3) -> BrickCoord {
        let edge = self.brick_edge();
        BrickCoord::new(
            (p.x / edge).floor() as i32,
            (p.y / edge).floor() as i32,
            (p.z / edge).floor() as i32,
        )
    }

    /// World-space minimum corner of a brick.
    #[must_use]
    pub fn brick_min_corner(self, coord: BrickCoord) -> Vec3 {
        let edge = self.brick_edge();
        Vec3::new(
            coord.bx as f32 * edge,
            coord.by as f32 * edge,
            coord.bz as f32 * edge,
        )
    }

    /// World-space center of a brick.
    #[must_use]
    pub fn brick_center(self, coord: BrickCoord) -> Vec3 {
        let edge = self.brick_edge();
        let half = edge * 0.5;
        let min = self.brick_min_corner(coord);
        Vec3::new(min.x + half, min.y + half, min.z + half)
    }
}

/// One brick that must be resident this frame, with its ranking inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrickRequest {
    /// The brick to keep (or stream) resident.
    pub coord: BrickCoord,
    /// Residency priority in `0..=1`; occupied bricks rank above the apron, and
    /// denser occupied bricks above sparser ones.
    pub priority: f32,
    /// Number of input points that fell inside the brick. `0` marks an
    /// apron-only brick kept solely to support a neighbor's stencil.
    pub occupancy: u32,
}

/// Screen-independent residency priority for an occupied brick, rising
/// monotonically with occupancy and saturating below `1`. Lands in `[0.75, 1)`
/// so it always outranks the apron tier below.
fn occupied_priority(occupancy: u32) -> f32 {
    0.5 + 0.5 * (1.0 - 1.0 / (1.0 + occupancy as f32))
}

/// Names the bricks that should be resident for a `FLIP`/`APIC` volume given the
/// fluid point cloud this frame.
///
/// Points are binned into occupied bricks (occupancy counted per brick), then
/// the occupied set is dilated by the configured apron so stencil neighbors are
/// resident too. Occupied bricks carry a density-ranked priority; apron-only
/// bricks carry half the priority of the occupied brick that pulled them in, so
/// under budget pressure the fluid core survives and the apron sheds first. The
/// result is deduplicated by construction and returned in brick-key order, a
/// stable deterministic description of the frame's desired residency set.
///
/// This does not enforce a residency budget; feed the result through
/// [`admit_within_budget`] or the [`crate::paging`] stream manager to bound it.
#[must_use]
pub fn plan_resident_bricks(points: &[Vec3], cfg: SparseVolumeConfig) -> Vec<BrickRequest> {
    if !cfg.is_valid() {
        return Vec::new();
    }

    // 1. Count occupancy per occupied brick.
    let mut occupancy: BTreeMap<BrickCoord, u32> = BTreeMap::new();
    for &p in points {
        *occupancy.entry(cfg.brick_of_point(p)).or_insert(0) += 1;
    }

    // 2. Seed the resident map with occupied bricks at their density priority.
    let mut resident: BTreeMap<BrickCoord, BrickRequest> = BTreeMap::new();
    for (&coord, &occ) in &occupancy {
        resident.insert(
            coord,
            BrickRequest {
                coord,
                priority: occupied_priority(occ),
                occupancy: occ,
            },
        );
    }

    // 3. Dilate by the apron: each occupied brick stamps its neighbors as
    //    support bricks, never downgrading an already-occupied neighbor.
    let apron = cfg.apron as i32;
    if apron > 0 {
        for (&coord, &occ) in &occupancy {
            let support = 0.5 * occupied_priority(occ);
            let mut dz = -apron;
            while dz <= apron {
                let mut dy = -apron;
                while dy <= apron {
                    let mut dx = -apron;
                    while dx <= apron {
                        if dx != 0 || dy != 0 || dz != 0 {
                            let ncoord =
                                BrickCoord::new(coord.bx + dx, coord.by + dy, coord.bz + dz);
                            if !occupancy.contains_key(&ncoord) {
                                let entry = resident.entry(ncoord).or_insert(BrickRequest {
                                    coord: ncoord,
                                    priority: support,
                                    occupancy: 0,
                                });
                                entry.priority = entry.priority.max(support);
                            }
                        }
                        dx += 1;
                    }
                    dy += 1;
                }
                dz += 1;
            }
        }
    }

    resident.into_values().collect()
}

/// Splits a desired brick set into the bricks that fit a residency budget and
/// the bricks deferred past it.
///
/// Admission is greedy by priority (highest first), breaking ties by brick key
/// so the choice is deterministic. At most `budget` bricks are admitted; the
/// rest are deferred. Both returned lists are in brick-key order. A `budget` at
/// or above the request count admits everything with no deferrals.
#[must_use]
pub fn admit_within_budget(
    requests: &[BrickRequest],
    budget: usize,
) -> (Vec<BrickCoord>, Vec<BrickCoord>) {
    let mut ranked: Vec<&BrickRequest> = requests.iter().collect();
    ranked.sort_by(|a, b| {
        b.priority
            .total_cmp(&a.priority)
            .then_with(|| a.coord.cmp(&b.coord))
    });

    let mut admitted = Vec::new();
    let mut deferred = Vec::new();
    for (rank, req) in ranked.into_iter().enumerate() {
        if rank < budget {
            admitted.push(req.coord);
        } else {
            deferred.push(req.coord);
        }
    }
    admitted.sort_unstable();
    deferred.sort_unstable();
    (admitted, deferred)
}

/// The per-frame residency transition: which desired bricks must be loaded,
/// which stay, and which currently-resident bricks can be evicted.
///
/// The three lists are disjoint and each is in brick-key order, so a transition
/// is fully described by this diff.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BrickDiff {
    /// Desired bricks not yet resident (stream in).
    pub load: Vec<BrickCoord>,
    /// Desired bricks already resident (keep).
    pub keep: Vec<BrickCoord>,
    /// Resident bricks no longer desired (evict).
    pub evict: Vec<BrickCoord>,
}

/// Diffs a desired brick set against the currently resident set.
///
/// `desired` may contain duplicates or arbitrary order; it is treated as a set.
/// Returns the load/keep/evict partition in brick-key order.
#[must_use]
pub fn diff_bricks(desired: &[BrickCoord], resident: &BTreeSet<BrickCoord>) -> BrickDiff {
    let desired_set: BTreeSet<BrickCoord> = desired.iter().copied().collect();
    let mut diff = BrickDiff::default();
    for &coord in &desired_set {
        if resident.contains(&coord) {
            diff.keep.push(coord);
        } else {
            diff.load.push(coord);
        }
    }
    for &coord in resident {
        if !desired_set.contains(&coord) {
            diff.evict.push(coord);
        }
    }
    diff
}

/// Records a desired brick set into a [`RequestBatch`] for the generic streaming
/// controller, keeping each brick's residency priority.
///
/// This is the seam that hands a volume frame's residency plan to
/// [`crate::paging`]: flush the batch through
/// [`PageStreamManager::reconcile`](crate::paging::PageStreamManager::reconcile)
/// and the shared budgeted admit/evict policy takes over.
pub fn record_into_batch(requests: &[BrickRequest], batch: &mut RequestBatch<BrickCoord>) {
    for req in requests {
        batch.record(req.coord, req.priority);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paging::{PageSource, PageStreamManager, ResidencyTable};
    use alloc::vec;

    const CFG: SparseVolumeConfig = SparseVolumeConfig {
        cell_size: 0.5,
        brick_cells: 4,
        apron: 1,
    };

    #[test]
    fn brick_edge_and_point_mapping_handle_negatives() {
        // edge = 0.5 * 4 = 2.0
        assert!((CFG.brick_edge() - 2.0).abs() < EPS);
        assert_eq!(
            CFG.brick_of_point(Vec3::new(0.0, 0.0, 0.0)),
            BrickCoord::new(0, 0, 0)
        );
        assert_eq!(
            CFG.brick_of_point(Vec3::new(1.9, 0.1, 0.1)),
            BrickCoord::new(0, 0, 0)
        );
        assert_eq!(
            CFG.brick_of_point(Vec3::new(2.1, 0.0, 0.0)),
            BrickCoord::new(1, 0, 0)
        );
        // Negative coordinates floor toward minus infinity, not toward zero.
        assert_eq!(
            CFG.brick_of_point(Vec3::new(-0.1, 0.0, 0.0)),
            BrickCoord::new(-1, 0, 0)
        );
    }

    #[test]
    fn invalid_config_yields_no_bricks() {
        let bad = SparseVolumeConfig {
            cell_size: 0.0,
            brick_cells: 0,
            apron: 1,
        };
        assert!(plan_resident_bricks(&[Vec3::ZERO], bad).is_empty());
        assert!(plan_resident_bricks(&[], CFG).is_empty());
    }

    #[test]
    fn occupied_set_covers_all_points_and_sums_occupancy() {
        let points = [
            Vec3::new(0.1, 0.1, 0.1),
            Vec3::new(0.2, 0.2, 0.2),
            Vec3::new(2.5, 0.0, 0.0),
        ];
        let plan = plan_resident_bricks(&points, CFG);
        // Every point's brick must be present and occupied.
        for &p in &points {
            let b = CFG.brick_of_point(p);
            let req = plan
                .iter()
                .find(|r| r.coord == b)
                .expect("occupied brick present");
            assert!(req.occupancy >= 1);
        }
        let total_occ: u32 = plan.iter().map(|r| r.occupancy).sum();
        assert_eq!(total_occ, points.len() as u32);
    }

    #[test]
    fn single_brick_dilates_to_full_apron_ring() {
        let plan = plan_resident_bricks(&[Vec3::new(0.5, 0.5, 0.5)], CFG);
        // One occupied brick + a 1-brick apron on every side => 3^3 = 27.
        assert_eq!(plan.len(), 27);
        let occupied: usize = plan.iter().filter(|r| r.occupancy > 0).count();
        assert_eq!(occupied, 1);
        // apron=0 keeps only the occupied brick.
        let no_apron = SparseVolumeConfig { apron: 0, ..CFG };
        let plan0 = plan_resident_bricks(&[Vec3::new(0.5, 0.5, 0.5)], no_apron);
        assert_eq!(plan0.len(), 1);
        assert_eq!(plan0[0].occupancy, 1);
    }

    #[test]
    fn occupied_outranks_apron_and_denser_outranks_sparser() {
        // Brick (0,0,0) gets three points, a far brick gets one.
        let mut points = vec![
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.6, 0.6, 0.6),
            Vec3::new(0.7, 0.7, 0.7),
        ];
        points.push(Vec3::new(20.5, 20.5, 20.5));
        let plan = plan_resident_bricks(&points, CFG);
        let dense = plan.iter().find(|r| r.occupancy == 3).unwrap();
        let sparse = plan.iter().find(|r| r.occupancy == 1).unwrap();
        let apron = plan.iter().find(|r| r.occupancy == 0).unwrap();
        assert!(dense.priority > sparse.priority);
        assert!(sparse.priority > apron.priority);
        for req in &plan {
            assert!((0.0..=1.0).contains(&req.priority));
        }
    }

    #[test]
    fn plan_is_deterministic_and_key_ordered() {
        let points = [
            Vec3::new(0.3, 0.3, 0.3),
            Vec3::new(3.1, -2.0, 1.0),
            Vec3::new(-4.0, 5.0, -1.0),
        ];
        let a = plan_resident_bricks(&points, CFG);
        let b = plan_resident_bricks(&points, CFG);
        assert_eq!(a, b);
        for pair in a.windows(2) {
            assert!(pair[0].coord < pair[1].coord, "bricks must be key-ordered");
        }
    }

    #[test]
    fn admit_within_budget_conserves_and_prioritizes() {
        let points = [
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.6, 0.6, 0.6),
            Vec3::new(10.5, 10.5, 10.5),
        ];
        let plan = plan_resident_bricks(&points, CFG);
        let budget = plan.len() / 2;
        let (admitted, deferred) = admit_within_budget(&plan, budget);
        assert_eq!(admitted.len(), budget);
        assert_eq!(admitted.len() + deferred.len(), plan.len());
        let adm: BTreeSet<BrickCoord> = admitted.iter().copied().collect();
        // The occupied (highest-priority) bricks must survive into the budget.
        for req in plan.iter().filter(|r| r.occupancy > 0) {
            if admitted.len() >= plan.iter().filter(|r| r.occupancy > 0).count() {
                assert!(adm.contains(&req.coord));
            }
        }
        let min_admit = plan
            .iter()
            .filter(|r| adm.contains(&r.coord))
            .map(|r| r.priority)
            .fold(f32::INFINITY, f32::min);
        let def: BTreeSet<BrickCoord> = deferred.iter().copied().collect();
        let max_defer = plan
            .iter()
            .filter(|r| def.contains(&r.coord))
            .map(|r| r.priority)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(min_admit + EPS >= max_defer);
    }

    #[test]
    fn diff_partitions_load_keep_evict() {
        let a = BrickCoord::new(0, 0, 0);
        let b = BrickCoord::new(1, 0, 0);
        let c = BrickCoord::new(0, 1, 0);
        let desired = [b, a, b];
        let resident: BTreeSet<BrickCoord> = [a, c].into_iter().collect();
        let diff = diff_bricks(&desired, &resident);
        assert_eq!(diff.load, vec![b]);
        assert_eq!(diff.keep, vec![a]);
        assert_eq!(diff.evict, vec![c]);
    }

    /// A page source handing out a fixed-size zero payload: brick residency only
    /// needs the key lifecycle, not page contents.
    struct ZeroBrickSource {
        page_words: usize,
    }

    impl PageSource<BrickCoord> for ZeroBrickSource {
        fn load(&mut self, _key: BrickCoord) -> Vec<u32> {
            vec![0u32; self.page_words]
        }
    }

    #[test]
    fn record_into_batch_drives_residency_table() {
        let plan = plan_resident_bricks(&[Vec3::new(0.5, 0.5, 0.5)], CFG);
        let mut batch = RequestBatch::new();
        record_into_batch(&plan, &mut batch);
        assert_eq!(batch.len(), plan.len());
        let mut table: ResidencyTable<BrickCoord> = ResidencyTable::new();
        batch.flush(&mut table, 1);
        assert_eq!(table.pending_requests().len(), plan.len());
        for req in &plan {
            table.mark_resident(req.coord);
        }
        assert_eq!(table.resident_count(), plan.len());
    }

    #[test]
    fn paging_manager_streams_bricks_within_budget() {
        let plan = plan_resident_bricks(&[Vec3::new(0.5, 0.5, 0.5)], CFG);
        let budget = plan.len() / 2;
        assert!(budget > 0);
        let capacity = plan.len() as u32;
        let mut manager: PageStreamManager<BrickCoord> =
            PageStreamManager::new(capacity, 4, budget);
        let mut source = ZeroBrickSource { page_words: 4 };
        let mut batch = RequestBatch::new();
        record_into_batch(&plan, &mut batch);
        let report = manager.reconcile(&batch, 1, &mut source);
        assert_eq!(manager.resident_count(), budget);
        assert_eq!(report.streamed_in.len(), budget);
        assert_eq!(report.streamed_in.len() + report.deferred.len(), plan.len());
        // The occupied brick is highest priority, so it must be among the
        // streamed-in set rather than deferred.
        let occupied = CFG.brick_of_point(Vec3::new(0.5, 0.5, 0.5));
        assert!(report.streamed_in.contains(&occupied));
    }
}
