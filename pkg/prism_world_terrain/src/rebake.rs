//! Deterministic dirty-region merging and budgeted incremental rebake planning.

use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::{RepresentationChoice, RepresentationError, RepresentationSpec, TileKey};

/// A closed tile-local dirty rectangle in Q16 normalized coordinates.
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
            let fits = (tasks.len() as u32) < budget.max_tasks
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
    let span_u = u64::from(region.max_u_q16()) - u64::from(region.min_u_q16()) + 1;
    let span_v = u64::from(region.max_v_q16()) - u64::from(region.min_v_q16()) + 1;
    let (resolution_u, resolution_v) = match spec {
        RepresentationSpec::Heightfield(spec) => (
            u64::from(spec.samples_per_edge()),
            u64::from(spec.samples_per_edge()),
        ),
        RepresentationSpec::Voxel(spec) => {
            let dimensions = spec.layout().dimensions();
            (u64::from(dimensions[0]), u64::from(dimensions[1]))
        }
        RepresentationSpec::Sdf(spec) => {
            let dimensions = spec.dimensions();
            (u64::from(dimensions[0]), u64::from(dimensions[1]))
        }
    };
    let samples_u = span_u
        .checked_mul(resolution_u)
        .and_then(|value| value.checked_add(65_535))
        .map(|value| value / 65_536)
        .ok_or(RebakeError::CostOverflow)?
        .max(1);
    let samples_v = span_v
        .checked_mul(resolution_v)
        .and_then(|value| value.checked_add(65_535))
        .map(|value| value / 65_536)
        .ok_or(RebakeError::CostOverflow)?
        .max(1);
    samples_u
        .checked_mul(samples_v)
        .ok_or(RebakeError::CostOverflow)
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
        let tile = TileKey::new(CubeFace::NegativeX, 30, 0, 1_u32 << 30 - 1).unwrap();
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
}
