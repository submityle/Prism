use alloc::vec::Vec;

use crate::ecology::{EcologyConfig, EcologyDelta, EcologyField};
use crate::hydrology::{FlowDelta, HydrologyField};
use crate::types::{DirtyRegion, DirtyRegions, Region};
use crate::voxel::{stable_fragment_key, FragmentDelta, VoxelCell, VoxelDelta, VoxelField};

/// Per-tick admission limits shared by all simulation phases.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TickBudget {
    /// Maximum number of candidate cells/nodes processed.
    pub work_units: u32,
    /// Maximum number of emitted delta records.
    pub output_records: u32,
    /// Maximum number of disjoint dirty regions returned.
    pub dirty_regions: u32,
}

impl TickBudget {
    /// Creates a tick budget.
    pub const fn new(work_units: u32, output_records: u32, dirty_regions: u32) -> Self {
        Self {
            work_units,
            output_records,
            dirty_regions,
        }
    }
}

/// Work and output actually consumed by one tick.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BudgetUsage {
    /// Candidate cells/nodes processed.
    pub work_units: u32,
    /// Delta records emitted.
    pub output_records: u32,
}

pub(crate) struct BudgetLedger {
    limit: TickBudget,
    usage: BudgetUsage,
}

impl BudgetLedger {
    pub(crate) const fn new(limit: TickBudget) -> Self {
        Self {
            limit,
            usage: BudgetUsage {
                work_units: 0,
                output_records: 0,
            },
        }
    }

    pub(crate) const fn can_take(&self, work_units: u32, output_records: u32) -> bool {
        self.usage.work_units.saturating_add(work_units) <= self.limit.work_units
            && self.usage.output_records.saturating_add(output_records) <= self.limit.output_records
    }

    pub(crate) fn take(&mut self, work_units: u32, output_records: u32) {
        self.usage.work_units += work_units;
        self.usage.output_records += output_records;
    }

    pub(crate) const fn usage(&self) -> BudgetUsage {
        self.usage
    }
}

/// Persistent simulation changes emitted by a tick.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SimulationDelta {
    /// Tick that produced the delta.
    pub tick: u64,
    /// Destructive voxel changes.
    pub voxel_deltas: Vec<VoxelDelta>,
    /// Spawned fragments.
    pub fragment_deltas: Vec<FragmentDelta>,
    /// Water transfers.
    pub flow_deltas: Vec<FlowDelta>,
    /// Biomass changes.
    pub ecology_deltas: Vec<EcologyDelta>,
}

/// Result of one deterministic simulation tick.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TickReport {
    /// Persistent changes produced by this tick.
    pub delta: SimulationDelta,
    /// Dirty regions needing host-side refresh.
    pub dirty_regions: Vec<DirtyRegion>,
    /// Actual shared budget usage.
    pub budget: BudgetUsage,
}

/// Configuration for the simulation owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSimConfig {
    /// Optional local simulation bounds. `None` means all stored cells.
    pub active_region: Option<Region>,
    /// Water required per unit of removed voxel mass.
    pub water_cost_per_mass: u32,
    /// Maximum mass removed from one voxel in one visit.
    pub max_erosion_mass: u32,
    /// Ecological fixed-point parameters.
    pub ecology: EcologyConfig,
}

impl WorldSimConfig {
    /// Creates a valid simulation configuration.
    pub const fn new(
        active_region: Option<Region>,
        water_cost_per_mass: u32,
        max_erosion_mass: u32,
        ecology: EcologyConfig,
    ) -> Option<Self> {
        if water_cost_per_mass == 0 || max_erosion_mass == 0 {
            None
        } else {
            Some(Self {
                active_region,
                water_cost_per_mass,
                max_erosion_mass,
                ecology,
            })
        }
    }
}

impl Default for WorldSimConfig {
    fn default() -> Self {
        Self {
            active_region: None,
            water_cost_per_mass: 8,
            max_erosion_mass: 8,
            ecology: EcologyConfig::default(),
        }
    }
}

/// Failure returned when a simulation configuration cannot be constructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimulationError {
    /// The erosion water cost was zero.
    ZeroWaterCost,
    /// The per-cell erosion limit was zero.
    ZeroErosionLimit,
    /// The ecology configuration contained a zero divisor.
    InvalidEcologyConfig,
}

/// Deterministic engine-neutral world simulation owner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldSim {
    /// Sparse destructive voxel state.
    pub voxels: VoxelField,
    /// Sparse hydrology state.
    pub hydrology: HydrologyField,
    /// Sparse ecological state.
    pub ecology: EcologyField,
    config: WorldSimConfig,
    tick: u64,
}

impl WorldSim {
    /// Creates a simulation owner with validated configuration.
    pub fn new(config: WorldSimConfig) -> Result<Self, SimulationError> {
        if config.water_cost_per_mass == 0 {
            return Err(SimulationError::ZeroWaterCost);
        }
        if config.max_erosion_mass == 0 {
            return Err(SimulationError::ZeroErosionLimit);
        }
        if config.ecology.growth_divisor == 0 || config.ecology.stress_divisor == 0 {
            return Err(SimulationError::InvalidEcologyConfig);
        }
        Ok(Self {
            voxels: VoxelField::new(),
            hydrology: HydrologyField::new(),
            ecology: EcologyField::new(),
            config,
            tick: 0,
        })
    }

    /// Returns the next tick number that will be emitted.
    pub const fn next_tick(&self) -> u64 {
        self.tick.wrapping_add(1)
    }

    /// Runs hydrology, local erosion, and ecology in stable phase order.
    pub fn step(&mut self, budget: TickBudget) -> TickReport {
        self.tick = self.tick.wrapping_add(1);
        let mut ledger = BudgetLedger::new(budget);
        let mut dirty = DirtyRegions::new(budget.dirty_regions);
        let mut delta = SimulationDelta {
            tick: self.tick,
            ..SimulationDelta::default()
        };
        self.hydrology.route(
            self.config.active_region,
            &mut ledger,
            &mut dirty,
            &mut delta.flow_deltas,
        );
        self.erode(
            self.config.active_region,
            &mut ledger,
            &mut dirty,
            &mut delta.voxel_deltas,
            &mut delta.fragment_deltas,
        );
        self.ecology.update(
            self.config.ecology,
            &self.hydrology,
            self.config.active_region,
            &mut ledger,
            &mut dirty,
            &mut delta.ecology_deltas,
        );
        TickReport {
            delta,
            dirty_regions: dirty.into_vec(),
            budget: ledger.usage(),
        }
    }

    fn erode(
        &mut self,
        region: Option<Region>,
        budget: &mut BudgetLedger,
        dirty: &mut DirtyRegions,
        voxel_output: &mut Vec<VoxelDelta>,
        fragment_output: &mut Vec<FragmentDelta>,
    ) {
        let coordinates = self.voxels.coordinates(region);
        for coord in coordinates {
            let Some(before) = self.voxels.get(coord) else {
                continue;
            };
            let Some(water) = self.hydrology.get(coord) else {
                continue;
            };
            let removable = water.water / self.config.water_cost_per_mass;
            let removed = before
                .mass
                .min(removable)
                .min(self.config.max_erosion_mass)
                .min(u32::from(before.hardness).saturating_add(1));
            if removed == 0 {
                if budget.can_take(1, 0) {
                    budget.take(1, 0);
                }
                continue;
            }
            if !budget.can_take(1, 2) || !dirty.try_insert(Region::cell(coord).expand(1)) {
                continue;
            }
            budget.take(1, 2);
            let after_mass = before.mass - removed;
            if after_mass == 0 {
                self.voxels.remove(coord);
            } else if let Some(cell) = self.voxels.get(coord) {
                self.voxels.set(
                    coord,
                    VoxelCell {
                        mass: after_mass,
                        ..cell
                    },
                );
            }
            let water_cost = removed.saturating_mul(self.config.water_cost_per_mass);
            if let Some(cell) = self.hydrology.cells_mut().get_mut(&coord) {
                cell.water = cell.water.saturating_sub(water_cost);
            }
            voxel_output.push(VoxelDelta {
                coord,
                material: before.material,
                before_mass: before.mass,
                after_mass,
                removed_mass: removed,
            });
            fragment_output.push(FragmentDelta {
                stable_key: stable_fragment_key(self.tick, coord, 0),
                origin: coord,
                mass: removed,
                material: before.material,
                velocity: [0, -1, 0],
            });
        }
    }
}
