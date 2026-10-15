//! Deterministic dirty-region merging and budgeted incremental rebake planning.

use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::{RepresentationChoice, RepresentationError, RepresentationSpec, TileKey};

/// A closed tile-local dirty rectangle in Q16 normalized coordinates.
///
/// The endpoints `0` and `u16::MAX` map to the first and last sample of
/// each XY axis. Planning encloses the rectangle with floor/ceil grid
/// indices, including interpolation endpoints for a non-aligned edit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirtyRegion {
    tile: TileKey,
    min_u_q16: u16,
    min_v_q16: u16,
    max_u_q16: u16,
    max_v_q16: u16,
}

impl DirtyRegion {
    /// Creates a non-empty closed dirty rectangle.
    pub const fn new(
        tile: TileKey,
        min_u_q16: u16,
        min_v_q16: u16,
        max_u_q16: u16,
        max_v_q16: u16,
    ) -> Result<Self, RebakeError> {
        if min_u_q16 > max_u_q16 || min_v_q16 > max_v_q16 {
            return Err(RebakeError::InvalidDirtyRegion);
        }
        Ok(Self {
            tile,
            min_u_q16,
            min_v_q16,
            max_u_q16,
            max_v_q16,
        })
    }

    /// Returns the owning tile.
    pub const fn tile(self) -> TileKey {
        self.tile
    }

    /// Returns the inclusive minimum U coordinate.
    pub const fn min_u_q16(self) -> u16 {
        self.min_u_q16
    }

    /// Returns the inclusive minimum V coordinate.
    pub const fn min_v_q16(self) -> u16 {
        self.min_v_q16
    }

    /// Returns the inclusive maximum U coordinate.
    pub const fn max_u_q16(self) -> u16 {
        self.max_u_q16
    }

    /// Returns the inclusive maximum V coordinate.
    pub const fn max_v_q16(self) -> u16 {
        self.max_v_q16
    }

    /// Returns the number of Q16 cells covered by the rectangle.
    pub fn area_cells(self) -> Result<u64, RebakeError> {
        let width = u64::from(self.max_u_q16) - u64::from(self.min_u_q16) + 1;
        let height = u64::from(self.max_v_q16) - u64::from(self.min_v_q16) + 1;
        width.checked_mul(height).ok_or(RebakeError::CostOverflow)
    }

    fn can_merge(self, other: Self) -> bool {
        if self.tile != other.tile {
            return false;
        }
        let u_overlap = self.min_u_q16 <= other.max_u_q16 && other.min_u_q16 <= self.max_u_q16;
        let v_overlap = self.min_v_q16 <= other.max_v_q16 && other.min_v_q16 <= self.max_v_q16;
        let u_touch = self.max_u_q16.checked_add(1) == Some(other.min_u_q16)
            || other.max_u_q16.checked_add(1) == Some(self.min_u_q16);
        let v_touch = self.max_v_q16.checked_add(1) == Some(other.min_v_q16)
            || other.max_v_q16.checked_add(1) == Some(self.min_v_q16);
        (u_overlap && (v_overlap || v_touch)) || (v_overlap && u_touch)
    }

    const fn union(self, other: Self) -> Self {
        Self {
            tile: self.tile,
            min_u_q16: if self.min_u_q16 < other.min_u_q16 {
                self.min_u_q16
            } else {
                other.min_u_q16
            },
            min_v_q16: if self.min_v_q16 < other.min_v_q16 {
                self.min_v_q16
            } else {
                other.min_v_q16
            },
            max_u_q16: if self.max_u_q16 > other.max_u_q16 {
                self.max_u_q16
            } else {
                other.max_u_q16
            },
            max_v_q16: if self.max_v_q16 > other.max_v_q16 {
                self.max_v_q16
            } else {
                other.max_v_q16
            },
        }
    }
}

/// A hard admission budget for one incremental rebake plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RebakeBudget {
    /// Maximum number of admitted tasks.
    pub max_tasks: u32,
    /// Maximum abstract work units.
    pub max_work_units: u64,
    /// Maximum estimated output bytes.
    pub max_bytes: u64,
}

impl RebakeBudget {
    /// Creates a budget. Zero values are valid and intentionally admit no work.
    pub const fn new(max_tasks: u32, max_work_units: u64, max_bytes: u64) -> Self {
        Self {
            max_tasks,
            max_work_units,
            max_bytes,
        }
    }
}

/// Source and target representations for an incremental rebake.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RebakeRequest {
    choice: RepresentationChoice,
}

impl RebakeRequest {
    /// Creates a request from validated representation descriptions.
    pub const fn new(source: RepresentationSpec, target: RepresentationSpec) -> Self {
        Self {
            choice: RepresentationChoice::new(source, target),
        }
    }

    /// Returns the source and target choice.
    pub const fn choice(self) -> RepresentationChoice {
        self.choice
    }
}

/// One admitted incremental rebake task.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RebakeTask {
    /// Dirty area to process.
    pub region: DirtyRegion,
    /// Source and target representation contract.
    pub choice: RepresentationChoice,
    /// Estimated work units.
    pub work_units: u64,
    /// Estimated output bytes.
    pub bytes: u64,
}

/// Result of deterministic region merging and budget admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RebakePlan {
    /// Admitted tasks in deterministic order.
    pub tasks: Vec<RebakeTask>,
    /// Merged regions not admitted by the budget.
    pub deferred: Vec<DirtyRegion>,
    /// Sum of admitted work units.
    pub total_work_units: u64,
    /// Sum of admitted output bytes.
    pub total_bytes: u64,
}

/// Planner for incremental, engine-neutral terrain rebakes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RebakePlanner;

impl RebakePlanner {
    /// Merges overlapping/edge-adjacent regions and admits tasks in stable order.
    ///
    /// A merged bounding rectangle may include unchanged samples. Its entire
    /// enclosing XY grid footprint is charged, including complete Z columns
    /// for voxel and SDF targets. Costs describe packed sample payloads, not
    /// padded GPU buffers or whole-brick transfers. Per-task cost overflow
    /// returns an error; tasks whose cumulative costs exceed the remaining
    /// budget are deferred without wrapping the admitted totals.
    pub fn plan(
        &self,
        request: RebakeRequest,
        regions: &[DirtyRegion],
        budget: RebakeBudget,
    ) -> Result<RebakePlan, RebakeError> {
        let merged = merge_regions(regions);
        let mut tasks = Vec::new();
        let mut deferred = Vec::new();
        let mut total_work_units: u64 = 0;
        let mut total_bytes: u64 = 0;

        for region in merged {
            let sample_count = sample_count(request.choice().target, region)?;
            let work = request
                .choice()
                .target
                .estimate_work(region.tile(), sample_count)
                .map_err(RebakeError::Representation)?;
            let fits = u32::try_from(tasks.len()).is_ok_and(|count| count < budget.max_tasks)
                && total_work_units
                    .checked_add(work.work_units)
                    .is_some_and(|value| value <= budget.max_work_units)
                && total_bytes
                    .checked_add(work.bytes)
                    .is_some_and(|value| value <= budget.max_bytes);
            if fits {
                total_work_units = total_work_units
                    .checked_add(work.work_units)
                    .ok_or(RebakeError::CostOverflow)?;
                total_bytes = total_bytes
                    .checked_add(work.bytes)
                    .ok_or(RebakeError::CostOverflow)?;
                tasks.push(RebakeTask {
                    region,
                    choice: request.choice(),
                    work_units: work.work_units,
                    bytes: work.bytes,
                });
            } else {
                deferred.push(region);
            }
        }

        Ok(RebakePlan {
            tasks,
            deferred,
            total_work_units,
            total_bytes,
        })
    }
}

fn sample_count(spec: RepresentationSpec, region: DirtyRegion) -> Result<u64, RebakeError> {
    let (resolution_u, resolution_v) = match spec {
        RepresentationSpec::Heightfield(spec) => (spec.samples_per_edge(), spec.samples_per_edge()),
        RepresentationSpec::Voxel(spec) => {
            let dimensions = spec.layout().dimensions();
            (dimensions[0], dimensions[1])
        }
        RepresentationSpec::Sdf(spec) => {
            let dimensions = spec.dimensions();
            (dimensions[0], dimensions[1])
        }
    };
    let samples_u = covered_axis_samples(region.min_u_q16(), region.max_u_q16(), resolution_u);
    let samples_v = covered_axis_samples(region.min_v_q16(), region.max_v_q16(), resolution_v);
    samples_u
        .checked_mul(samples_v)
        .ok_or(RebakeError::CostOverflow)
}

fn covered_axis_samples(min: u16, max: u16, resolution: u32) -> u64 {
    // Validated specs always have resolution >= 1. Products of a u16 coordinate
    // and (u32 resolution - 1) fit u64, including MAX_TILE_LEVEL tiles. Counting
    // by endpoint indices instead of span length preserves the grid phase:
    // a short interval straddling a sample can need three enclosing points.
    let intervals = u64::from(resolution - 1);
    let denominator = u64::from(u16::MAX);
    let first = u64::from(min) * intervals / denominator;
    let last = (u64::from(max) * intervals).div_ceil(denominator);
    last - first + 1
}

fn merge_regions(regions: &[DirtyRegion]) -> Vec<DirtyRegion> {
    let mut sorted = regions.to_vec();
    sorted.sort_unstable_by(region_order);
    let mut merged = Vec::with_capacity(sorted.len());
    for region in sorted {
        let mut current = region;
        let mut index = 0;
        while index < merged.len() {
            if current.can_merge(merged[index]) {
                current = current.union(merged.remove(index));
                index = 0;
            } else {
                index += 1;
            }
        }
        merged.push(current);
        merged.sort_unstable_by(region_order);
    }
    merged
}

fn region_order(left: &DirtyRegion, right: &DirtyRegion) -> Ordering {
    (
        left.tile().face().index(),
        left.tile().level(),
        left.tile().x(),
        left.tile().y(),
        left.min_v_q16(),
        left.min_u_q16(),
        left.max_v_q16(),
        left.max_u_q16(),
    )
        .cmp(&(
            right.tile().face().index(),
            right.tile().level(),
            right.tile().x(),
            right.tile().y(),
            right.min_v_q16(),
            right.min_u_q16(),
            right.max_v_q16(),
            right.max_u_q16(),
        ))
}

/// Errors returned while creating or planning dirty regions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RebakeError {
    /// Minimum coordinates exceed maximum coordinates.
    InvalidDirtyRegion,
    /// A sample or cost calculation overflowed.
    CostOverflow,
    /// The target representation rejected the cost request.
    Representation(RepresentationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CubeFace, HeightfieldSpec, SdfSpec, VoxelComponentFormat, VoxelSpec};

    fn request() -> RebakeRequest {
        RebakeRequest::new(
            RepresentationSpec::Heightfield(HeightfieldSpec::new(129, 10_000).unwrap()),
            RepresentationSpec::Sdf(SdfSpec::new([32, 32, 32], 8, 1).unwrap()),
        )
    }

    fn region(tile: TileKey, u0: u16, v0: u16, u1: u16, v1: u16) -> DirtyRegion {
        DirtyRegion::new(tile, u0, v0, u1, v1).unwrap()
    }

    #[test]
    fn invalid_and_boundary_regions_are_rejected_or_preserved() {
        let tile = TileKey::new(CubeFace::NegativeX, 30, 0, (1_u32 << 30) - 1).unwrap();
        assert_eq!(
            DirtyRegion::new(tile, 5, 9, 4, 9),
            Err(RebakeError::InvalidDirtyRegion)
        );
        let full = region(tile, 0, 0, u16::MAX, u16::MAX);
        assert_eq!(full.area_cells().unwrap(), 4_294_967_296);
    }

    #[test]
    fn adjacent_regions_merge_transitively_and_deterministically() {
        let tile = TileKey::new(CubeFace::PositiveZ, 3, 2, 4).unwrap();
        let inputs = [
            region(tile, 10, 10, 19, 19),
            region(tile, 0, 10, 9, 19),
            region(tile, 20, 10, 29, 19),
        ];
        let planner = RebakePlanner;
        let budget = RebakeBudget::new(8, u64::MAX, u64::MAX);
        let first = planner.plan(request(), &inputs, budget).unwrap();
        let second = planner
            .plan(request(), &[inputs[2], inputs[0], inputs[1]], budget)
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.tasks.len(), 1);
        assert_eq!(first.tasks[0].region.min_u_q16(), 0);
        assert_eq!(first.tasks[0].region.max_u_q16(), 29);
    }

    #[test]
    fn budget_defers_tasks_without_nondeterministic_reordering() {
        let tile = TileKey::new(CubeFace::PositiveX, 2, 1, 1).unwrap();
        let regions = [
            region(tile, 0, 0, 1023, 1023),
            region(tile, 40_000, 40_000, 40_010, 40_010),
        ];
        let planner = RebakePlanner;
        let plan = planner
            .plan(
                request(),
                &regions,
                RebakeBudget::new(1, u64::MAX, u64::MAX),
            )
            .unwrap();
        assert_eq!(plan.tasks.len(), 1);
        assert_eq!(plan.deferred.len(), 1);
        assert_eq!(plan.tasks[0].region.min_u_q16(), 0);

        let zero = planner
            .plan(request(), &regions, RebakeBudget::new(0, 0, 0))
            .unwrap();
        assert!(zero.tasks.is_empty());
        assert_eq!(zero.deferred.len(), 2);
    }

    #[test]
    fn all_three_target_contracts_have_finite_plans() {
        let tile = TileKey::root(CubeFace::PositiveY);
        let source = RepresentationSpec::Voxel(
            VoxelSpec::new([8, 8, 8], 2, VoxelComponentFormat::DensityU16).unwrap(),
        );
        let targets = [
            RepresentationSpec::Heightfield(HeightfieldSpec::new(33, 1).unwrap()),
            source,
            RepresentationSpec::Sdf(SdfSpec::new([8, 8, 8], 2, 1).unwrap()),
        ];
        for target in targets {
            let plan = RebakePlanner
                .plan(
                    RebakeRequest::new(source, target),
                    &[region(tile, 0, 0, 255, 255)],
                    RebakeBudget::new(4, u64::MAX, u64::MAX),
                )
                .unwrap();
            assert_eq!(plan.tasks.len(), 1);
            assert!(plan.total_work_units > 0);
            assert!(plan.total_bytes > 0);
        }
    }

    #[test]
    fn full_regions_charge_exact_packed_volume_bytes() {
        let tile = TileKey::new(CubeFace::NegativeX, crate::MAX_TILE_LEVEL, 0, 0).unwrap();
        let full = region(tile, 0, 0, u16::MAX, u16::MAX);
        let voxel = VoxelSpec::new([7, 5, 3], 2, VoxelComponentFormat::DensityU16).unwrap();
        let targets = [
            (
                RepresentationSpec::Heightfield(HeightfieldSpec::new(5, 1).unwrap()),
                50,
                25,
            ),
            (
                RepresentationSpec::Voxel(voxel),
                voxel.layout().expected_byte_len().unwrap() as u64,
                105,
            ),
            (
                RepresentationSpec::Sdf(SdfSpec::new([7, 5, 3], 2, 1).unwrap()),
                420,
                105,
            ),
        ];
        for (target, bytes, output_samples) in targets {
            let work_units = output_samples * (u64::from(tile.level()) + 1);
            let plan = RebakePlanner
                .plan(
                    RebakeRequest::new(target, target),
                    &[full],
                    RebakeBudget::new(1, work_units, bytes),
                )
                .unwrap();
            assert_eq!(plan.tasks.len(), 1);
            assert!(plan.deferred.is_empty());
            assert_eq!(plan.total_bytes, bytes);
            assert_eq!(plan.total_work_units, work_units);
            assert_eq!(plan.tasks[0].bytes, bytes);
            assert_eq!(plan.tasks[0].work_units, work_units);

            for budget in [
                RebakeBudget::new(1, work_units, bytes - 1),
                RebakeBudget::new(1, work_units - 1, bytes),
            ] {
                let deferred = RebakePlanner
                    .plan(RebakeRequest::new(target, target), &[full], budget)
                    .unwrap();
                assert!(deferred.tasks.is_empty());
                assert_eq!(deferred.deferred, [full]);
                assert_eq!(deferred.total_bytes, 0);
                assert_eq!(deferred.total_work_units, 0);
            }
        }
    }

    #[test]
    fn non_aligned_footprints_account_for_grid_position_and_endpoints() {
        let tile = TileKey::root(CubeFace::PositiveX);
        let target = RepresentationSpec::Heightfield(HeightfieldSpec::new(4, 1).unwrap());
        // Grid points are 0, 21845, 43690, and 65535. Equal-length intervals
        // have different conservative footprints depending on the grid phase.
        assert_eq!(sample_count(target, region(tile, 0, 0, 2, 0)).unwrap(), 2);
        assert_eq!(
            sample_count(target, region(tile, 21844, 0, 21846, 0)).unwrap(),
            3
        );
        assert_eq!(
            sample_count(target, region(tile, 21844, 21844, 21846, 21846)).unwrap(),
            9
        );
        assert_eq!(
            sample_count(target, region(tile, 21845, 21845, 21845, 21845)).unwrap(),
            1
        );
        assert_eq!(
            sample_count(target, region(tile, 21846, 21846, 21846, 21846)).unwrap(),
            4
        );
        assert_eq!(
            sample_count(target, region(tile, 65535, 65535, 65535, 65535)).unwrap(),
            1
        );
        let sdf = RepresentationSpec::Sdf(SdfSpec::new([4, 4, 3], 1, 1).unwrap());
        let dirty = region(tile, 21844, 21844, 21846, 21846);
        let plan = RebakePlanner
            .plan(
                RebakeRequest::new(sdf, sdf),
                &[dirty],
                RebakeBudget::new(1, 27, 108),
            )
            .unwrap();
        assert_eq!(plan.total_bytes, 3 * 3 * 3 * 4);
        assert_eq!(plan.total_work_units, 27);
        let singleton = RepresentationSpec::Sdf(SdfSpec::new([1, 1, 3], 1, 1).unwrap());
        assert_eq!(
            sample_count(singleton, region(tile, 0, 0, 65535, 65535)).unwrap(),
            1
        );
    }

    #[test]
    fn axis_coverage_is_conservative_for_representative_grids() {
        for resolution in [1, 2, 3, 4, 17, 129, 65_536, u32::MAX] {
            for min in [0_u16, 1, 21844, 21845, 21846, 32767, 65534, 65535] {
                for max in [
                    min,
                    min.saturating_add(1),
                    min.saturating_add(100),
                    u16::MAX,
                ] {
                    let count = covered_axis_samples(min, max, resolution);
                    let intervals = u128::from(resolution - 1);
                    let denominator = u128::from(u16::MAX);
                    // Wide-integer cross multiplication verifies the floor/ceil
                    // enclosure without reusing the u64 division-based result.
                    let first = u128::from(min) * intervals / denominator;
                    let last = first + u128::from(count) - 1;
                    assert!(first * denominator <= u128::from(min) * intervals);
                    assert!((first + 1) * denominator > u128::from(min) * intervals);
                    assert!(last * denominator >= u128::from(max) * intervals);
                    assert!(last == 0 || (last - 1) * denominator < u128::from(max) * intervals);
                    assert!(count <= u64::from(resolution));
                }
            }
        }
    }

    #[test]
    fn per_task_overflow_errors_and_cumulative_overflow_defers() {
        let tile = TileKey::root(CubeFace::PositiveZ);
        let full = region(tile, 0, 0, u16::MAX, u16::MAX);
        let huge = RepresentationSpec::Sdf(SdfSpec::new([u32::MAX; 3], 1, 1).unwrap());
        assert_eq!(
            RebakePlanner.plan(
                RebakeRequest::new(huge, huge),
                &[full],
                RebakeBudget::new(1, u64::MAX, u64::MAX)
            ),
            Err(RebakeError::Representation(
                RepresentationError::CostOverflow
            ))
        );
        let byte_overflow =
            RepresentationSpec::Sdf(SdfSpec::new([u32::MAX, u32::MAX, 1], 1, 1).unwrap());
        assert_eq!(
            RebakePlanner.plan(
                RebakeRequest::new(byte_overflow, byte_overflow),
                &[full],
                RebakeBudget::new(1, u64::MAX, u64::MAX)
            ),
            Err(RebakeError::Representation(
                RepresentationError::CostOverflow
            ))
        );
        // Each independent tile fits, but two 2^63-byte outputs cannot fit one
        // u64 budget. No buffers are allocated by this pure data planner.
        let large = RepresentationSpec::Heightfield(HeightfieldSpec::new(1_u32 << 31, 1).unwrap());
        let second = region(TileKey::root(CubeFace::NegativeZ), 0, 0, u16::MAX, u16::MAX);
        let plan = RebakePlanner
            .plan(
                RebakeRequest::new(large, large),
                &[second, full],
                RebakeBudget::new(2, u64::MAX, u64::MAX),
            )
            .unwrap();
        assert_eq!(plan.tasks.len(), 1);
        assert_eq!(plan.deferred.len(), 1);
        assert_eq!(plan.total_bytes, 1_u64 << 63);
        assert_eq!(plan.total_work_units, 1_u64 << 62);
    }

    #[test]
    fn merges_and_budget_results_are_invariant_under_all_input_permutations() {
        let tile = TileKey::new(
            CubeFace::NegativeY,
            crate::MAX_TILE_LEVEL,
            (1_u32 << 30) - 1,
            0,
        )
        .unwrap();
        let inputs = [
            region(tile, 21844, 0, 21844, 50),
            region(tile, 21845, 25, 21846, 100),
            region(tile, 50000, 50000, 51000, 51000),
            region(tile, 21844, 0, 21844, 50),
        ];
        let target = RepresentationSpec::Sdf(SdfSpec::new([4, 4, 3], 1, 1).unwrap());
        let request = RebakeRequest::new(target, target);
        let budget = RebakeBudget::new(1, u64::MAX, u64::MAX);
        let reference = RebakePlanner.plan(request, &inputs, budget).unwrap();
        assert_eq!(reference.tasks.len(), 1);
        assert_eq!(reference.deferred.len(), 1);
        // Bounding union spans three U points and two V points; depth is 3.
        assert_eq!(reference.total_bytes, 3 * 2 * 3 * 4);
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        if a == b || a == c || a == d || b == c || b == d || c == d {
                            continue;
                        }
                        let permutation = [inputs[a], inputs[b], inputs[c], inputs[d]];
                        assert_eq!(
                            RebakePlanner.plan(request, &permutation, budget).unwrap(),
                            reference
                        );
                    }
                }
            }
        }
    }
}
