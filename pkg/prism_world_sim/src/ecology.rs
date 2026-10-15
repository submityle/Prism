use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::hydrology::HydrologyField;
use crate::sim::BudgetLedger;
use crate::types::{CellCoord, DirtyRegions, Region};

/// Integer ecological state stored at one simulation cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EcologyCell {
    /// Current biomass in arbitrary integer units.
    pub biomass: u32,
    /// Local fertility in the range 0..=65535.
    pub fertility: u16,
    /// Maximum biomass capacity.
    pub capacity: u32,
}

impl EcologyCell {
    /// Creates an ecological cell.
    pub const fn new(biomass: u32, fertility: u16, capacity: u32) -> Option<Self> {
        if capacity == 0 || biomass > capacity {
            None
        } else {
            Some(Self {
                biomass,
                fertility,
                capacity,
            })
        }
    }
}

/// Fixed-point parameters for deterministic ecological updates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EcologyConfig {
    /// Moisture at which growth is preferred, in 0..=65535.
    pub preferred_moisture: u16,
    /// Divisor applied to potential growth.
    pub growth_divisor: u16,
    /// Divisor applied to moisture stress.
    pub stress_divisor: u16,
}

impl EcologyConfig {
    /// Creates parameters, rejecting zero divisors.
    pub const fn new(
        preferred_moisture: u16,
        growth_divisor: u16,
        stress_divisor: u16,
    ) -> Option<Self> {
        if growth_divisor == 0 || stress_divisor == 0 {
            None
        } else {
            Some(Self {
                preferred_moisture,
                growth_divisor,
                stress_divisor,
            })
        }
    }
}

impl Default for EcologyConfig {
    fn default() -> Self {
        Self {
            preferred_moisture: 32_768,
            growth_divisor: 16,
            stress_divisor: 1_024,
        }
    }
}

/// A persistent biomass transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EcologyDelta {
    /// Changed coordinate.
    pub coord: CellCoord,
    /// Biomass before the update.
    pub before_biomass: u32,
    /// Biomass after the update.
    pub after_biomass: u32,
    /// Moisture used by the update.
    pub moisture: u16,
    /// Moisture stress used by the update.
    pub stress: u16,
}

/// Sparse ecological state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EcologyField {
    cells: BTreeMap<CellCoord, EcologyCell>,
}

impl EcologyField {
    /// Creates an empty ecological field.
    pub const fn new() -> Self {
        Self {
            cells: BTreeMap::new(),
        }
    }

    /// Inserts or replaces an ecological cell.
    pub fn set(&mut self, coord: CellCoord, cell: EcologyCell) {
        self.cells.insert(coord, cell);
    }

    /// Returns an ecological cell by coordinate.
    pub fn get(&self, coord: CellCoord) -> Option<EcologyCell> {
        self.cells.get(&coord).copied()
    }

    /// Returns coordinates in stable map order, optionally clipped to a region.
    pub fn coordinates(&self, region: Option<Region>) -> Vec<CellCoord> {
        self.cells
            .keys()
            .copied()
            .filter(|coord| region.is_none_or(|bounds| bounds.contains(*coord)))
            .collect()
    }

    pub(crate) fn update(
        &mut self,
        config: EcologyConfig,
        hydrology: &HydrologyField,
        region: Option<Region>,
        budget: &mut BudgetLedger,
        dirty: &mut DirtyRegions,
        output: &mut Vec<EcologyDelta>,
    ) {
        let coordinates = self.coordinates(region);
        let snapshot = self.cells.clone();
        for coord in coordinates {
            let Some(cell) = snapshot.get(&coord).copied() else {
                continue;
            };
            let moisture = hydrology
                .get(coord)
                .map(|water| {
                    if water.capacity == 0 {
                        0
                    } else {
                        ((u64::from(water.water) * 65_535) / u64::from(water.capacity)).min(65_535)
                            as u16
                    }
                })
                .unwrap_or(0);
            let stress = moisture.abs_diff(config.preferred_moisture);
            let neighbor_sum: u64 = coord
                .horizontal_neighbors()
                .into_iter()
                .filter_map(|neighbor| {
                    snapshot
                        .get(&neighbor)
                        .map(|value| u64::from(value.biomass))
                })
                .sum();
            let neighbor_average = (neighbor_sum / 4).min(u64::from(cell.capacity)) as u32;
            let potential = (u64::from(cell.fertility) * u64::from(moisture) / 65_535) as u32
                / u32::from(config.growth_divisor);
            let crowding_loss = neighbor_average.saturating_sub(cell.biomass) / 32;
            let stress_loss = u32::from(stress) / u32::from(config.stress_divisor);
            let growth = potential.min(cell.capacity.saturating_sub(cell.biomass));
            let after = cell
                .biomass
                .saturating_add(growth)
                .saturating_sub(crowding_loss.saturating_add(stress_loss));
            if after == cell.biomass {
                if budget.can_take(1, 0) {
                    budget.take(1, 0);
                }
                continue;
            }
            if !budget.can_take(1, 1) || !dirty.try_insert(Region::cell(coord)) {
                continue;
            }
            budget.take(1, 1);
            if let Some(current) = self.cells.get_mut(&coord) {
                current.biomass = after;
            }
            output.push(EcologyDelta {
                coord,
                before_biomass: cell.biomass,
                after_biomass: after,
                moisture,
                stress,
            });
        }
    }
}
