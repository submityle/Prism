use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::sim::BudgetLedger;
use crate::types::{CellCoord, DirtyRegions, Region};

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
