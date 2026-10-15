use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::sim::BudgetLedger;
use crate::types::{CellCoord, DirtyRegions, Region};

const D8_OFFSETS: [(HydrologyDirection, i32, i32); 8] = [
    (HydrologyDirection::North, 0, -1),
    (HydrologyDirection::NorthEast, 1, -1),
    (HydrologyDirection::East, 1, 0),
    (HydrologyDirection::SouthEast, 1, 1),
    (HydrologyDirection::South, 0, 1),
    (HydrologyDirection::SouthWest, -1, 1),
    (HydrologyDirection::West, -1, 0),
    (HydrologyDirection::NorthWest, -1, -1),
];

/// Stable D8 direction from one cell to its selected downstream cell.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum HydrologyDirection {
    /// Negative Z direction.
    North,
    /// Positive X, negative Z direction.
    NorthEast,
    /// Positive X direction.
    East,
    /// Positive X, positive Z direction.
    SouthEast,
    /// Positive Z direction.
    South,
    /// Negative X, positive Z direction.
    SouthWest,
    /// Negative X direction.
    West,
    /// Negative X, negative Z direction.
    NorthWest,
    /// No strictly lower in-region neighbor exists.
    Sink,
}

/// Fixed limits for one D8 hydrology analysis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HydrologyAnalysisBudget {
    /// Maximum number of cells included in the analysis.
    pub max_cells: u64,
    /// Maximum number of neighbor probes. Every analyzed cell consumes eight.
    pub max_neighbor_probes: u64,
}

impl HydrologyAnalysisBudget {
    /// Creates an analysis budget.
    pub const fn new(max_cells: u64, max_neighbor_probes: u64) -> Self {
        Self {
            max_cells,
            max_neighbor_probes,
        }
    }
}

/// Failure returned when a D8 analysis cannot be completed within its bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HydrologyAnalysisError {
    /// The selected region contains more cells than the caller allows.
    CellBudgetExceeded {
        /// Number of cells requested.
        required: u64,
        /// Configured cell limit.
        limit: u64,
    },
    /// Eight neighbor probes per cell exceed the caller's limit.
    NeighborProbeBudgetExceeded {
        /// Number of probes required.
        required: u64,
        /// Configured probe limit.
        limit: u64,
    },
    /// A checked count conversion or multiplication overflowed.
    CountOverflow,
    /// A flow accumulation could not be represented by `u64`.
    AccumulationOverflow {
        /// Cell whose accumulation overflowed.
        coord: CellCoord,
    },
}

/// Deterministic D8 analysis result for one sparse hydrology field region.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HydrologyAnalysis {
    cells: Vec<HydrologyAnalysisCell>,
    river_threshold: u64,
}

impl HydrologyAnalysis {
    /// Returns cells in canonical [`CellCoord`] order.
    pub fn cells(&self) -> &[HydrologyAnalysisCell] {
        &self.cells
    }

    /// Returns one analyzed cell by coordinate.
    pub fn get(&self, coord: CellCoord) -> Option<HydrologyAnalysisCell> {
        self.cells
            .binary_search_by_key(&coord, |cell| cell.coord)
            .ok()
            .map(|index| self.cells[index])
    }

    /// Returns coordinates whose accumulation reaches the configured threshold.
    pub fn river_cells(&self) -> Vec<CellCoord> {
        self.cells
            .iter()
            .filter(|cell| cell.is_river)
            .map(|cell| cell.coord)
            .collect()
    }

    /// Returns the accumulation threshold used to classify river cells.
    pub const fn river_threshold(&self) -> u64 {
        self.river_threshold
    }
}

/// One analyzed cell with D8 routing and basin metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HydrologyAnalysisCell {
    /// Coordinate of this cell.
    pub coord: CellCoord,
    /// Terrain elevation copied from the source field.
    pub elevation: i32,
    /// Selected D8 route, or [`HydrologyDirection::Sink`].
    pub direction: HydrologyDirection,
    /// Selected downstream coordinate, if any.
    pub downstream: Option<CellCoord>,
    /// Number of analyzed cells draining through this cell, including itself.
    pub accumulation: u64,
    /// Terminal sink coordinate for this cell's basin.
    pub basin: CellCoord,
    /// Whether `accumulation >= river_threshold`.
    pub is_river: bool,
}

/// Discrete hydrology state at one cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HydrologyCell {
    /// Integer terrain elevation used for downhill routing.
    pub elevation: i32,
    /// Water volume in fixed integer units.
    pub water: u32,
    /// Maximum water volume retained by this cell.
    pub capacity: u32,
    /// Permeability in the range 0..=65535.
    pub permeability: u16,
}

impl HydrologyCell {
    /// Creates a hydrology cell.
    pub const fn new(elevation: i32, water: u32, capacity: u32, permeability: u16) -> Self {
        Self {
            elevation,
            water,
            capacity,
            permeability,
        }
    }
}

/// A stable downhill water transfer emitted by a hydrology step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlowDelta {
    /// Source coordinate.
    pub from: CellCoord,
    /// Destination coordinate.
    pub to: CellCoord,
    /// Transferred water volume.
    pub amount: u32,
}

/// Sparse discrete hydrology state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HydrologyField {
    cells: BTreeMap<CellCoord, HydrologyCell>,
}

impl HydrologyField {
    /// Creates an empty hydrology field.
    pub const fn new() -> Self {
        Self {
            cells: BTreeMap::new(),
        }
    }

    /// Inserts or replaces a hydrology cell.
    pub fn set(&mut self, coord: CellCoord, cell: HydrologyCell) {
        self.cells.insert(coord, cell);
    }

    /// Returns a hydrology cell by coordinate.
    pub fn get(&self, coord: CellCoord) -> Option<HydrologyCell> {
        self.cells.get(&coord).copied()
    }

    /// Returns water at a coordinate, or zero when the cell is absent.
    pub fn water_at(&self, coord: CellCoord) -> u32 {
        self.cells.get(&coord).map_or(0, |cell| cell.water)
    }

    /// Returns all coordinates in stable map order, optionally clipped to a region.
    pub fn coordinates(&self, region: Option<Region>) -> Vec<CellCoord> {
        self.cells
            .keys()
            .copied()
            .filter(|coord| region.is_none_or(|bounds| bounds.contains(*coord)))
            .collect()
    }

    /// Computes deterministic D8 routing, flow accumulation, basins, and rivers.
    ///
    /// Routing is evaluated only in the X/Z plane and never crosses a Y layer.
    /// A candidate must be strictly lower than its source; equal-height
    /// candidates are therefore never part of a route. When multiple lower
    /// candidates have the same elevation, the canonical [`CellCoord`] order
    /// selects the destination. The optional region constrains both analyzed
    /// sources and downstream candidates.
    ///
    /// The analysis is read-only. It consumes a fixed eight neighbor probes per
    /// selected cell, sorts only bounded coordinate lists, and reports checked
    /// arithmetic failures instead of returning partial output.
    pub fn analyze_d8(
        &self,
        region: Option<Region>,
        river_threshold: u64,
        budget: HydrologyAnalysisBudget,
    ) -> Result<HydrologyAnalysis, HydrologyAnalysisError> {
        let coordinates = self.coordinates(region);
        let cell_count =
            u64::try_from(coordinates.len()).map_err(|_| HydrologyAnalysisError::CountOverflow)?;
        if cell_count > budget.max_cells {
            return Err(HydrologyAnalysisError::CellBudgetExceeded {
                required: cell_count,
                limit: budget.max_cells,
            });
        }
        let probe_count = cell_count
            .checked_mul(D8_OFFSETS.len() as u64)
            .ok_or(HydrologyAnalysisError::CountOverflow)?;
        if probe_count > budget.max_neighbor_probes {
            return Err(HydrologyAnalysisError::NeighborProbeBudgetExceeded {
                required: probe_count,
                limit: budget.max_neighbor_probes,
            });
        }

        let mut routes = BTreeMap::<CellCoord, Option<(HydrologyDirection, CellCoord)>>::new();
        for source in coordinates.iter().copied() {
            let source_elevation = self
                .cells
                .get(&source)
                .map(|cell| cell.elevation)
                .expect("coordinates are sourced from the hydrology map");
            let mut selected = None;
            for (direction, dx, dz) in D8_OFFSETS {
                let Some(x) = source.x.checked_add(dx) else {
                    continue;
                };
                let Some(z) = source.z.checked_add(dz) else {
                    continue;
                };
                let candidate = CellCoord::new(x, source.y, z);
                if !region.is_none_or(|bounds| bounds.contains(candidate)) {
                    continue;
                }
                let Some(candidate_cell) = self.cells.get(&candidate) else {
                    continue;
                };
                if candidate_cell.elevation >= source_elevation {
                    continue;
                }
                let candidate_key = (candidate_cell.elevation, candidate);
                let replace = selected.is_none_or(|(_, current)| {
                    let current_cell = self
                        .cells
                        .get(&current)
                        .expect("selected route is sourced from the hydrology map");
                    candidate_key < (current_cell.elevation, current)
                });
                if replace {
                    selected = Some((direction, candidate));
                }
            }
            routes.insert(source, selected);
        }

        let mut descending = coordinates.clone();
        descending.sort_by(|left, right| {
            let left_elevation = self
                .cells
                .get(left)
                .expect("coordinates are sourced from the hydrology map")
                .elevation;
            let right_elevation = self
                .cells
                .get(right)
                .expect("coordinates are sourced from the hydrology map")
                .elevation;
            right_elevation
                .cmp(&left_elevation)
                .then_with(|| left.cmp(right))
        });

        let mut accumulations = BTreeMap::<CellCoord, u64>::new();
        for source in descending {
            let accumulation = accumulations
                .get(&source)
                .copied()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or(HydrologyAnalysisError::AccumulationOverflow { coord: source })?;
            accumulations.insert(source, accumulation);
            if let Some((_, destination)) = routes.get(&source).copied().flatten() {
                let destination_accumulation = accumulations
                    .get(&destination)
                    .copied()
                    .unwrap_or(0)
                    .checked_add(accumulation)
                    .ok_or(HydrologyAnalysisError::AccumulationOverflow { coord: destination })?;
                accumulations.insert(destination, destination_accumulation);
            }
        }

        let mut ascending = coordinates.clone();
        ascending.sort_by(|left, right| {
            let left_elevation = self
                .cells
                .get(left)
                .expect("coordinates are sourced from the hydrology map")
                .elevation;
            let right_elevation = self
                .cells
                .get(right)
                .expect("coordinates are sourced from the hydrology map")
                .elevation;
            left_elevation
                .cmp(&right_elevation)
                .then_with(|| left.cmp(right))
        });

        let mut basins = BTreeMap::<CellCoord, CellCoord>::new();
        for source in ascending {
            let basin = routes
                .get(&source)
                .copied()
                .flatten()
                .map(|(_, destination)| {
                    basins
                        .get(&destination)
                        .copied()
                        .expect("strictly lower destinations sort before their sources")
                })
                .unwrap_or(source);
            basins.insert(source, basin);
        }

        let mut cells = Vec::with_capacity(coordinates.len());
        for coord in coordinates {
            let (direction, downstream) = routes.get(&coord).copied().flatten().map_or(
                (HydrologyDirection::Sink, None),
                |(direction, destination)| (direction, Some(destination)),
            );
            let accumulation = accumulations
                .get(&coord)
                .copied()
                .expect("every analyzed coordinate has an accumulation");
            cells.push(HydrologyAnalysisCell {
                coord,
                elevation: self
                    .cells
                    .get(&coord)
                    .expect("coordinates are sourced from the hydrology map")
                    .elevation,
                direction,
                downstream,
                accumulation,
                basin: basins
                    .get(&coord)
                    .copied()
                    .expect("every analyzed coordinate has a basin"),
                is_river: accumulation >= river_threshold,
            });
        }

        Ok(HydrologyAnalysis {
            cells,
            river_threshold,
        })
    }

    pub(crate) fn route(
        &mut self,
        region: Option<Region>,
        budget: &mut BudgetLedger,
        dirty: &mut DirtyRegions,
        output: &mut Vec<FlowDelta>,
    ) {
        let coordinates = self.coordinates(region);
        let original_water: BTreeMap<CellCoord, u32> = self
            .cells
            .iter()
            .map(|(coord, cell)| (*coord, cell.water))
            .collect();
        let mut outgoing = BTreeMap::<CellCoord, u32>::new();
        let mut incoming = BTreeMap::<CellCoord, u32>::new();

        for source in coordinates {
            let Some(source_cell) = self.cells.get(&source).copied() else {
                continue;
            };
            let Some(source_water) = original_water.get(&source).copied() else {
                continue;
            };
            let Some((destination, destination_cell)) = source
                .horizontal_neighbors()
                .into_iter()
                .filter_map(|candidate| {
                    self.cells
                        .get(&candidate)
                        .copied()
                        .map(|cell| (candidate, cell))
                })
                .filter(|(_, cell)| cell.elevation < source_cell.elevation)
                .min_by_key(|(candidate, cell)| (cell.elevation, *candidate))
            else {
                if budget.can_take(1, 0) {
                    budget.take(1, 0);
                }
                continue;
            };

            let available =
                source_water.saturating_sub(outgoing.get(&source).copied().unwrap_or(0));
            let downhill = u32::try_from(
                i64::from(source_cell.elevation) - i64::from(destination_cell.elevation),
            )
            .unwrap_or(u32::MAX);
            let quarter = available.saturating_add(3) / 4;
            let permeability = u32::from(source_cell.permeability);
            let slope_limit = downhill.saturating_add(1).saturating_add(permeability / 16);
            let free_space = destination_cell
                .capacity
                .saturating_sub(destination_cell.water)
                .saturating_sub(incoming.get(&destination).copied().unwrap_or(0));
            let amount = available.min(quarter).min(slope_limit).min(free_space);
            if amount == 0 {
                if budget.can_take(1, 0) {
                    budget.take(1, 0);
                }
                continue;
            }
            let dirty_region = Region::covering(source, destination);
            if !budget.can_take(1, 1) || !dirty.try_insert(dirty_region) {
                continue;
            }
            budget.take(1, 1);
            outgoing.insert(source, outgoing.get(&source).copied().unwrap_or(0) + amount);
            incoming.insert(
                destination,
                incoming.get(&destination).copied().unwrap_or(0) + amount,
            );
            if let Some(cell) = self.cells.get_mut(&source) {
                cell.water = cell.water.saturating_sub(amount);
            }
            if let Some(cell) = self.cells.get_mut(&destination) {
                cell.water = cell.water.saturating_add(amount).min(cell.capacity);
            }
            output.push(FlowDelta {
                from: source,
                to: destination,
                amount,
            });
        }
    }

    pub(crate) fn cells_mut(&mut self) -> &mut BTreeMap<CellCoord, HydrologyCell> {
        &mut self.cells
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(cells: u64) -> HydrologyAnalysisBudget {
        HydrologyAnalysisBudget::new(cells, cells * 8)
    }

    #[test]
    fn d8_chain_accumulates_and_classifies_rivers() {
        let mut field = HydrologyField::new();
        let first = CellCoord::new(0, 0, 0);
        let second = CellCoord::new(1, 0, 0);
        let third = CellCoord::new(2, 0, 0);
        field.set(first, HydrologyCell::new(3, 0, 1, 0));
        field.set(second, HydrologyCell::new(2, 0, 1, 0));
        field.set(third, HydrologyCell::new(1, 0, 1, 0));

        let analysis = field.analyze_d8(None, 2, budget(3)).unwrap();

        assert_eq!(analysis.cells().len(), 3);
        assert_eq!(analysis.get(first).unwrap().accumulation, 1);
        assert_eq!(analysis.get(second).unwrap().accumulation, 2);
        assert_eq!(analysis.get(third).unwrap().accumulation, 3);
        assert_eq!(analysis.get(first).unwrap().basin, third);
        assert_eq!(analysis.get(second).unwrap().basin, third);
        assert_eq!(
            analysis.get(third).unwrap().direction,
            HydrologyDirection::Sink
        );
        assert_eq!(analysis.river_cells(), alloc::vec![second, third]);
    }

    #[test]
    fn d8_selects_diagonal_and_stable_equal_height_tie_break() {
        let mut diagonal = HydrologyField::new();
        let source = CellCoord::new(0, 0, 0);
        let destination = CellCoord::new(1, 0, 1);
        diagonal.set(source, HydrologyCell::new(2, 0, 1, 0));
        diagonal.set(destination, HydrologyCell::new(1, 0, 1, 0));
        let diagonal_analysis = diagonal.analyze_d8(None, 1, budget(2)).unwrap();
        let diagonal_cell = diagonal_analysis.get(source).unwrap();
        assert_eq!(diagonal_cell.direction, HydrologyDirection::SouthEast);
        assert_eq!(diagonal_cell.downstream, Some(destination));

        let mut tied = HydrologyField::new();
        let west = CellCoord::new(-1, 0, 0);
        let east = CellCoord::new(1, 0, 0);
        tied.set(source, HydrologyCell::new(2, 0, 1, 0));
        tied.set(west, HydrologyCell::new(1, 0, 1, 0));
        tied.set(east, HydrologyCell::new(1, 0, 1, 0));
        let tied_analysis = tied.analyze_d8(None, 1, budget(3)).unwrap();
        let tied_cell = tied_analysis.get(source).unwrap();
        assert_eq!(tied_cell.direction, HydrologyDirection::West);
        assert_eq!(tied_cell.downstream, Some(west));
    }

    #[test]
    fn region_clips_sources_and_downstream_candidates() {
        let mut field = HydrologyField::new();
        let source = CellCoord::new(0, 0, 0);
        let outside = CellCoord::new(1, 0, 0);
        field.set(source, HydrologyCell::new(2, 0, 1, 0));
        field.set(outside, HydrologyCell::new(1, 0, 1, 0));

        let region = Region::cell(source);
        let analysis = field.analyze_d8(Some(region), 1, budget(1)).unwrap();

        assert_eq!(analysis.cells().len(), 1);
        assert_eq!(
            analysis.get(source).unwrap().direction,
            HydrologyDirection::Sink
        );
        assert_eq!(analysis.get(source).unwrap().basin, source);
    }

    #[test]
    fn budget_failure_is_explicit_and_analysis_does_not_mutate_field() {
        let mut field = HydrologyField::new();
        field.set(
            CellCoord::new(0, 0, 0),
            HydrologyCell::new(1, 7, 9, u16::MAX),
        );
        field.set(
            CellCoord::new(1, 0, 0),
            HydrologyCell::new(0, 3, 9, u16::MAX),
        );
        let before = field.clone();

        assert_eq!(
            field.analyze_d8(None, 1, HydrologyAnalysisBudget::new(2, 15)),
            Err(HydrologyAnalysisError::NeighborProbeBudgetExceeded {
                required: 16,
                limit: 15,
            })
        );
        assert_eq!(field, before);
    }

    #[test]
    fn d8_neighbor_arithmetic_is_safe_at_i32_boundaries() {
        let mut field = HydrologyField::new();
        let minimum = CellCoord::new(i32::MIN, 0, 0);
        let maximum = CellCoord::new(i32::MAX, 0, 0);
        field.set(minimum, HydrologyCell::new(0, 0, 1, 0));
        field.set(maximum, HydrologyCell::new(0, 0, 1, 0));

        let analysis = field.analyze_d8(None, 1, budget(2)).unwrap();

        assert_eq!(
            analysis.get(minimum).unwrap().direction,
            HydrologyDirection::Sink
        );
        assert_eq!(
            analysis.get(maximum).unwrap().direction,
            HydrologyDirection::Sink
        );
    }
}
