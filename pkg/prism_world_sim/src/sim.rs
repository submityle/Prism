use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::ecology::{EcologyConfig, EcologyDelta, EcologyField};
use crate::hydrology::{FlowDelta, HydrologyField};
use crate::types::{CellCoord, DirtyRegion, DirtyRegions, Region};
use crate::voxel::{
    stable_fragment_key, FragmentDelta, FragmentField, VoxelCell, VoxelDelta, VoxelField,
};

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
    /// The simulation tick would wrap past the largest representable value.
    TickOverflow,
}

/// Failure while applying one persisted simulation delta.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimulationApplyError {
    /// The current tick cannot advance to the delta's tick.
    TickOverflow,
    /// The delta is not the next tick and cannot be applied exactly once.
    UnexpectedTick {
        /// Tick required by the receiver.
        expected: u64,
        /// Tick carried by the delta.
        actual: u64,
    },
    /// A flow record has no positive amount or routes to itself.
    InvalidFlow,
    /// A flow references a cell absent from the receiver's hydrology field.
    MissingHydrologyCell(CellCoord),
    /// A flow would remove more water than is currently available.
    WaterUnderflow {
        /// Cell whose water would become negative.
        coord: CellCoord,
        /// Water available after prior records in this delta.
        available: u32,
        /// Water requested by this record.
        required: u32,
    },
    /// A flow would exceed the destination capacity or integer range.
    WaterCapacityExceeded {
        /// Destination cell.
        coord: CellCoord,
        /// Water present before this incoming amount.
        current: u32,
        /// Incoming amount.
        incoming: u32,
        /// Destination capacity.
        capacity: u32,
    },
    /// More than one flow record originates at the same cell.
    DuplicateFlowSource(CellCoord),
    /// Flow records are not in the stable source-coordinate order emitted by the simulator.
    NonCanonicalFlowOrder {
        /// Previous source coordinate.
        previous: CellCoord,
        /// Current source coordinate.
        current: CellCoord,
    },
    /// A flow record does not match the deterministic routing equation.
    FlowTransitionMismatch {
        /// Source coordinate.
        from: CellCoord,
        /// Destination coordinate.
        to: CellCoord,
        /// Amount carried by the delta.
        actual: u32,
        /// Amount required by replay.
        expected: u32,
    },
    /// More than one voxel record targets the same coordinate.
    DuplicateVoxelCoordinate(CellCoord),
    /// A voxel record references a cell absent from the receiver.
    MissingVoxelCell(CellCoord),
    /// A voxel record names a different material than the receiver.
    VoxelMaterialMismatch {
        /// Voxel coordinate.
        coord: CellCoord,
        /// Material in the receiver.
        actual: u16,
        /// Material carried by the delta.
        expected: u16,
    },
    /// A voxel record's mass differs from the receiver.
    VoxelMassMismatch {
        /// Voxel coordinate.
        coord: CellCoord,
        /// Mass in the receiver.
        actual: u32,
        /// Before mass carried by the delta.
        expected: u32,
    },
    /// A voxel record's hardness differs from the receiver.
    VoxelHardnessMismatch {
        /// Voxel coordinate.
        coord: CellCoord,
        /// Hardness in the receiver.
        actual: u16,
        /// Hardness carried by the delta.
        expected: u16,
    },
    /// A voxel record does not satisfy before-after-removed arithmetic.
    InvalidVoxelTransition,
    /// A voxel record does not match the deterministic erosion equation.
    VoxelRemovalMismatch {
        /// Voxel coordinate.
        coord: CellCoord,
        /// Removed mass carried by the delta.
        actual: u32,
        /// Removed mass required by replay.
        expected: u32,
    },
    /// Erosion claims more water than is available after routing.
    ErosionWaterUnderflow {
        /// Voxel coordinate.
        coord: CellCoord,
        /// Water available after prior adjustments.
        available: u32,
        /// Water required for the claimed removal.
        required: u32,
    },
    /// More than one ecology record targets the same coordinate.
    DuplicateEcologyCoordinate(CellCoord),
    /// An ecology record references a cell absent from the receiver.
    MissingEcologyCell(CellCoord),
    /// An ecology record's before biomass differs from the receiver.
    EcologyBiomassMismatch {
        /// Ecology coordinate.
        coord: CellCoord,
        /// Biomass in the receiver.
        actual: u32,
        /// Before biomass carried by the delta.
        expected: u32,
    },
    /// An ecology record would exceed its cell capacity.
    EcologyCapacityExceeded {
        /// Ecology coordinate.
        coord: CellCoord,
        /// Biomass carried by the delta.
        biomass: u32,
        /// Cell capacity.
        capacity: u32,
    },
    /// An ecology record reports moisture different from replay hydrology.
    EcologyMoistureMismatch {
        /// Ecology coordinate.
        coord: CellCoord,
        /// Moisture carried by the delta.
        actual: u16,
        /// Moisture computed during replay.
        expected: u16,
    },
    /// An ecology record reports stress different from the replay equation.
    EcologyStressMismatch {
        /// Ecology coordinate.
        coord: CellCoord,
        /// Stress carried by the delta.
        actual: u16,
        /// Stress computed during replay.
        expected: u16,
    },
    /// An ecology record does not match the deterministic transition equation.
    EcologyTransitionMismatch {
        /// Ecology coordinate.
        coord: CellCoord,
        /// Biomass carried by the delta.
        actual: u32,
        /// Biomass required by replay.
        expected: u32,
    },
    /// A fragment record has no positive mass.
    InvalidFragment,
    /// More than one fragment record uses the same stable key.
    DuplicateFragmentKey(u64),
    /// A reverse operation expected a fragment that is not present.
    MissingFragmentKey(u64),
    /// A reverse operation found a different fragment for the same key.
    FragmentMismatch(u64),
    /// A reverse operation was requested for a delta other than the current tick.
    ReverseTickMismatch {
        /// Current simulation tick.
        expected: u64,
        /// Tick carried by the delta.
        actual: u64,
    },
    /// Internal checked arithmetic could not represent a replay adjustment.
    ArithmeticOverflow,
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
    /// Sparse fragments emitted by destructive simulation.
    pub fragments: FragmentField,
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
            fragments: FragmentField::new(),
            config,
            tick: 0,
        })
    }

    /// Returns the next tick number that will be emitted.
    pub const fn next_tick(&self) -> Result<u64, SimulationError> {
        match self.tick.checked_add(1) {
            Some(tick) => Ok(tick),
            None => Err(SimulationError::TickOverflow),
        }
    }

    /// Returns the last successfully advanced simulation tick.
    pub const fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Runs hydrology, local erosion, and ecology in stable phase order.
    pub fn step(&mut self, budget: TickBudget) -> Result<TickReport, SimulationError> {
        self.tick = self.next_tick()?;
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
        Ok(TickReport {
            delta,
            dirty_regions: dirty.into_vec(),
            budget: ledger.usage(),
        })
    }

    /// Applies one previously emitted delta without re-running simulation.
    ///
    /// Validation is completed against the current state before any field is
    /// mutated. Flow adjustments are accumulated only for coordinates touched
    /// by the delta; voxel and ecology coordinates are required to be unique.
    /// This keeps replay linear in the delta size instead of cloning the full
    /// sparse world, while rejecting divergence before publication.
    pub fn apply_delta(&mut self, delta: &SimulationDelta) -> Result<(), SimulationApplyError> {
        let expected = self
            .next_tick()
            .map_err(|_| SimulationApplyError::TickOverflow)?;
        if delta.tick != expected {
            return Err(SimulationApplyError::UnexpectedTick {
                expected,
                actual: delta.tick,
            });
        }

        let mut flow_adjustments = BTreeMap::<CellCoord, i64>::new();
        let mut flow_sources = BTreeSet::new();
        let mut outgoing = BTreeMap::<CellCoord, u32>::new();
        let mut incoming = BTreeMap::<CellCoord, u32>::new();
        let mut previous_source = None;
        for flow in &delta.flow_deltas {
            if flow.amount == 0 || flow.from == flow.to {
                return Err(SimulationApplyError::InvalidFlow);
            }
            if let Some(previous) = previous_source
                && previous >= flow.from
            {
                return Err(SimulationApplyError::NonCanonicalFlowOrder {
                    previous,
                    current: flow.from,
                });
            }
            previous_source = Some(flow.from);
            if !flow_sources.insert(flow.from) {
                return Err(SimulationApplyError::DuplicateFlowSource(flow.from));
            }
            let source = self
                .hydrology
                .get(flow.from)
                .ok_or(SimulationApplyError::MissingHydrologyCell(flow.from))?;
            self.hydrology
                .get(flow.to)
                .ok_or(SimulationApplyError::MissingHydrologyCell(flow.to))?;
            let source_water = i64::from(source.water)
                .checked_sub(i64::from(outgoing.get(&flow.from).copied().unwrap_or(0)))
                .ok_or(SimulationApplyError::ArithmeticOverflow)?;
            if source_water < i64::from(flow.amount) {
                return Err(SimulationApplyError::WaterUnderflow {
                    coord: flow.from,
                    available: source_water as u32,
                    required: flow.amount,
                });
            }
            let expected_destination = flow
                .from
                .horizontal_neighbors()
                .into_iter()
                .filter_map(|candidate| self.hydrology.get(candidate).map(|cell| (candidate, cell)))
                .filter(|(_, cell)| cell.elevation < source.elevation)
                .min_by_key(|(candidate, cell)| (cell.elevation, *candidate));
            let Some((expected_to, expected_destination_cell)) = expected_destination else {
                return Err(SimulationApplyError::FlowTransitionMismatch {
                    from: flow.from,
                    to: flow.to,
                    actual: flow.amount,
                    expected: 0,
                });
            };
            if expected_to != flow.to {
                return Err(SimulationApplyError::FlowTransitionMismatch {
                    from: flow.from,
                    to: flow.to,
                    actual: flow.amount,
                    expected: 0,
                });
            }
            let available = u32::try_from(source_water)
                .map_err(|_| SimulationApplyError::ArithmeticOverflow)?;
            let downhill = u32::try_from(
                i64::from(source.elevation) - i64::from(expected_destination_cell.elevation),
            )
            .unwrap_or(u32::MAX);
            let quarter = available.saturating_add(3) / 4;
            let slope_limit = downhill
                .saturating_add(1)
                .saturating_add(u32::from(source.permeability) / 16);
            let free_space = expected_destination_cell
                .capacity
                .saturating_sub(expected_destination_cell.water)
                .saturating_sub(incoming.get(&flow.to).copied().unwrap_or(0));
            let expected_amount = available.min(quarter).min(slope_limit).min(free_space);
            if flow.amount != expected_amount {
                return Err(SimulationApplyError::FlowTransitionMismatch {
                    from: flow.from,
                    to: flow.to,
                    actual: flow.amount,
                    expected: expected_amount,
                });
            }
            let incoming_amount = flow.amount;
            outgoing
                .entry(flow.from)
                .and_modify(|value| *value = value.saturating_add(flow.amount))
                .or_insert(flow.amount);
            incoming
                .entry(flow.to)
                .and_modify(|value| *value = value.saturating_add(incoming_amount))
                .or_insert(incoming_amount);
            let incoming = i64::from(flow.amount);
            adjust_water(&mut flow_adjustments, flow.from, -incoming)?;
            adjust_water(&mut flow_adjustments, flow.to, incoming)?;
        }

        let mut voxel_coordinates = BTreeSet::new();
        for voxel in &delta.voxel_deltas {
            if voxel.before_mass < voxel.after_mass
                || voxel.before_mass - voxel.after_mass != voxel.removed_mass
            {
                return Err(SimulationApplyError::InvalidVoxelTransition);
            }
            if !voxel_coordinates.insert(voxel.coord) {
                return Err(SimulationApplyError::DuplicateVoxelCoordinate(voxel.coord));
            }
            let current = self
                .voxels
                .get(voxel.coord)
                .ok_or(SimulationApplyError::MissingVoxelCell(voxel.coord))?;
            if current.material != voxel.material {
                return Err(SimulationApplyError::VoxelMaterialMismatch {
                    coord: voxel.coord,
                    actual: current.material,
                    expected: voxel.material,
                });
            }
            if current.hardness != voxel.hardness {
                return Err(SimulationApplyError::VoxelHardnessMismatch {
                    coord: voxel.coord,
                    actual: current.hardness,
                    expected: voxel.hardness,
                });
            }
            if current.mass != voxel.before_mass {
                return Err(SimulationApplyError::VoxelMassMismatch {
                    coord: voxel.coord,
                    actual: current.mass,
                    expected: voxel.before_mass,
                });
            }
            if voxel.removed_mass != 0 {
                let water = self
                    .hydrology
                    .get(voxel.coord)
                    .ok_or(SimulationApplyError::MissingHydrologyCell(voxel.coord))?;
                let available = adjusted_water(
                    water.water,
                    flow_adjustments.get(&voxel.coord).copied().unwrap_or(0),
                )?;
                let water_cost = voxel
                    .removed_mass
                    .checked_mul(self.config.water_cost_per_mass)
                    .ok_or(SimulationApplyError::ArithmeticOverflow)?;
                let expected_removed = current
                    .mass
                    .min(
                        u32::try_from(available)
                            .map_err(|_| SimulationApplyError::ArithmeticOverflow)?
                            / self.config.water_cost_per_mass,
                    )
                    .min(self.config.max_erosion_mass)
                    .min(u32::from(current.hardness).saturating_add(1));
                if voxel.removed_mass != expected_removed {
                    return Err(SimulationApplyError::VoxelRemovalMismatch {
                        coord: voxel.coord,
                        actual: voxel.removed_mass,
                        expected: expected_removed,
                    });
                }
                if i64::from(water_cost) > available {
                    return Err(SimulationApplyError::ErosionWaterUnderflow {
                        coord: voxel.coord,
                        available: u32::try_from(available)
                            .map_err(|_| SimulationApplyError::ArithmeticOverflow)?,
                        required: water_cost,
                    });
                }
                adjust_water(&mut flow_adjustments, voxel.coord, -i64::from(water_cost))?;
            }
        }

        let mut fragment_keys = BTreeSet::new();
        for fragment in &delta.fragment_deltas {
            if fragment.mass == 0 {
                return Err(SimulationApplyError::InvalidFragment);
            }
            if !fragment_keys.insert(fragment.stable_key)
                || self.fragments.get(fragment.stable_key).is_some()
            {
                return Err(SimulationApplyError::DuplicateFragmentKey(
                    fragment.stable_key,
                ));
            }
        }

        let mut ecology_coordinates = BTreeSet::new();
        for ecology in &delta.ecology_deltas {
            if !ecology_coordinates.insert(ecology.coord) {
                return Err(SimulationApplyError::DuplicateEcologyCoordinate(
                    ecology.coord,
                ));
            }
            let current = self
                .ecology
                .get(ecology.coord)
                .ok_or(SimulationApplyError::MissingEcologyCell(ecology.coord))?;
            if current.biomass != ecology.before_biomass {
                return Err(SimulationApplyError::EcologyBiomassMismatch {
                    coord: ecology.coord,
                    actual: current.biomass,
                    expected: ecology.before_biomass,
                });
            }
            if ecology.after_biomass > current.capacity {
                return Err(SimulationApplyError::EcologyCapacityExceeded {
                    coord: ecology.coord,
                    biomass: ecology.after_biomass,
                    capacity: current.capacity,
                });
            }
            let moisture = match self.hydrology.get(ecology.coord) {
                Some(hydrology) => {
                    let water = adjusted_water(
                        hydrology.water,
                        flow_adjustments.get(&ecology.coord).copied().unwrap_or(0),
                    )?;
                    if hydrology.capacity == 0 {
                        0
                    } else {
                        ((u64::try_from(water)
                            .map_err(|_| SimulationApplyError::ArithmeticOverflow)?
                            * 65_535)
                            / u64::from(hydrology.capacity))
                        .min(65_535) as u16
                    }
                }
                None => 0,
            };
            if ecology.moisture != moisture {
                return Err(SimulationApplyError::EcologyMoistureMismatch {
                    coord: ecology.coord,
                    actual: ecology.moisture,
                    expected: moisture,
                });
            }
            let stress = moisture.abs_diff(self.config.ecology.preferred_moisture);
            if ecology.stress != stress {
                return Err(SimulationApplyError::EcologyStressMismatch {
                    coord: ecology.coord,
                    actual: ecology.stress,
                    expected: stress,
                });
            }
            let neighbor_sum: u64 = ecology
                .coord
                .horizontal_neighbors()
                .into_iter()
                .filter_map(|neighbor| {
                    self.ecology
                        .get(neighbor)
                        .map(|value| u64::from(value.biomass))
                })
                .sum();
            let neighbor_average = (neighbor_sum / 4).min(u64::from(current.capacity)) as u32;
            let potential = (u64::from(current.fertility) * u64::from(moisture) / 65_535) as u32
                / u32::from(self.config.ecology.growth_divisor);
            let crowding_loss = neighbor_average.saturating_sub(current.biomass) / 32;
            let stress_loss = u32::from(stress) / u32::from(self.config.ecology.stress_divisor);
            let growth = potential.min(current.capacity.saturating_sub(current.biomass));
            let expected_after = current
                .biomass
                .saturating_add(growth)
                .saturating_sub(crowding_loss.saturating_add(stress_loss));
            if ecology.after_biomass != expected_after {
                return Err(SimulationApplyError::EcologyTransitionMismatch {
                    coord: ecology.coord,
                    actual: ecology.after_biomass,
                    expected: expected_after,
                });
            }
        }

        for (coord, adjustment) in flow_adjustments {
            let cell = self
                .hydrology
                .cells_mut()
                .get_mut(&coord)
                .ok_or(SimulationApplyError::MissingHydrologyCell(coord))?;
            let water = adjusted_water(cell.water, adjustment)?;
            cell.water =
                u32::try_from(water).map_err(|_| SimulationApplyError::ArithmeticOverflow)?;
        }
        for voxel in &delta.voxel_deltas {
            let current = self
                .voxels
                .get(voxel.coord)
                .ok_or(SimulationApplyError::MissingVoxelCell(voxel.coord))?;
            if voxel.after_mass == 0 {
                self.voxels.remove(voxel.coord);
            } else {
                self.voxels.set(
                    voxel.coord,
                    VoxelCell {
                        mass: voxel.after_mass,
                        ..current
                    },
                );
            }
        }
        for ecology in &delta.ecology_deltas {
            let current = self
                .ecology
                .get(ecology.coord)
                .ok_or(SimulationApplyError::MissingEcologyCell(ecology.coord))?;
            self.ecology.set(
                ecology.coord,
                crate::ecology::EcologyCell {
                    biomass: ecology.after_biomass,
                    ..current
                },
            );
        }
        for fragment in &delta.fragment_deltas {
            self.fragments.set(*fragment);
        }
        self.tick = delta.tick;
        Ok(())
    }

    /// Reverses one previously applied simulation delta without rerunning the
    /// simulation.
    ///
    /// The receiver must currently be at `delta.tick`. Validation completes
    /// against the post-delta state before any field is mutated, so a failed
    /// rollback leaves the simulation unchanged. Voxel hardness and fragment
    /// payloads are part of the delta because restoring only mass would lose
    /// the state needed by a later deterministic tick.
    pub fn reverse_delta(&mut self, delta: &SimulationDelta) -> Result<(), SimulationApplyError> {
        if self.tick != delta.tick {
            return Err(SimulationApplyError::ReverseTickMismatch {
                expected: self.tick,
                actual: delta.tick,
            });
        }
        let previous_tick = delta
            .tick
            .checked_sub(1)
            .ok_or(SimulationApplyError::ArithmeticOverflow)?;

        let mut water_adjustments = BTreeMap::<CellCoord, i64>::new();
        let mut flow_sources = BTreeSet::new();
        let mut previous_source = None;
        for flow in &delta.flow_deltas {
            if flow.amount == 0 || flow.from == flow.to {
                return Err(SimulationApplyError::InvalidFlow);
            }
            if let Some(previous) = previous_source
                && previous >= flow.from
            {
                return Err(SimulationApplyError::NonCanonicalFlowOrder {
                    previous,
                    current: flow.from,
                });
            }
            previous_source = Some(flow.from);
            if !flow_sources.insert(flow.from) {
                return Err(SimulationApplyError::DuplicateFlowSource(flow.from));
            }
            self.hydrology
                .get(flow.from)
                .ok_or(SimulationApplyError::MissingHydrologyCell(flow.from))?;
            let destination = self
                .hydrology
                .get(flow.to)
                .ok_or(SimulationApplyError::MissingHydrologyCell(flow.to))?;
            if destination.water < flow.amount {
                return Err(SimulationApplyError::WaterUnderflow {
                    coord: flow.to,
                    available: destination.water,
                    required: flow.amount,
                });
            }
            adjust_water(&mut water_adjustments, flow.from, i64::from(flow.amount))?;
            adjust_water(&mut water_adjustments, flow.to, -i64::from(flow.amount))?;
        }

        let mut voxel_coordinates = BTreeSet::new();
        for voxel in &delta.voxel_deltas {
            if voxel.before_mass < voxel.after_mass
                || voxel.before_mass - voxel.after_mass != voxel.removed_mass
                || voxel.before_mass == 0
            {
                return Err(SimulationApplyError::InvalidVoxelTransition);
            }
            if !voxel_coordinates.insert(voxel.coord) {
                return Err(SimulationApplyError::DuplicateVoxelCoordinate(voxel.coord));
            }
            match self.voxels.get(voxel.coord) {
                Some(current) => {
                    if current.material != voxel.material {
                        return Err(SimulationApplyError::VoxelMaterialMismatch {
                            coord: voxel.coord,
                            actual: current.material,
                            expected: voxel.material,
                        });
                    }
                    if current.hardness != voxel.hardness {
                        return Err(SimulationApplyError::VoxelHardnessMismatch {
                            coord: voxel.coord,
                            actual: current.hardness,
                            expected: voxel.hardness,
                        });
                    }
                    if current.mass != voxel.after_mass {
                        return Err(SimulationApplyError::VoxelMassMismatch {
                            coord: voxel.coord,
                            actual: current.mass,
                            expected: voxel.after_mass,
                        });
                    }
                }
                None if voxel.after_mass == 0 => {}
                None => {
                    return Err(SimulationApplyError::MissingVoxelCell(voxel.coord));
                }
            }
            let water_cost = voxel
                .removed_mass
                .checked_mul(self.config.water_cost_per_mass)
                .ok_or(SimulationApplyError::ArithmeticOverflow)?;
            adjust_water(&mut water_adjustments, voxel.coord, i64::from(water_cost))?;
        }

        let mut fragment_keys = BTreeSet::new();
        for fragment in &delta.fragment_deltas {
            if fragment.mass == 0 {
                return Err(SimulationApplyError::InvalidFragment);
            }
            if !fragment_keys.insert(fragment.stable_key) {
                return Err(SimulationApplyError::DuplicateFragmentKey(
                    fragment.stable_key,
                ));
            }
            let current = self.fragments.get(fragment.stable_key).ok_or(
                SimulationApplyError::MissingFragmentKey(fragment.stable_key),
            )?;
            if current != *fragment {
                return Err(SimulationApplyError::FragmentMismatch(fragment.stable_key));
            }
        }

        let mut ecology_coordinates = BTreeSet::new();
        for ecology in &delta.ecology_deltas {
            if !ecology_coordinates.insert(ecology.coord) {
                return Err(SimulationApplyError::DuplicateEcologyCoordinate(
                    ecology.coord,
                ));
            }
            let current = self
                .ecology
                .get(ecology.coord)
                .ok_or(SimulationApplyError::MissingEcologyCell(ecology.coord))?;
            if current.biomass != ecology.after_biomass {
                return Err(SimulationApplyError::EcologyBiomassMismatch {
                    coord: ecology.coord,
                    actual: current.biomass,
                    expected: ecology.after_biomass,
                });
            }
            if ecology.before_biomass > current.capacity {
                return Err(SimulationApplyError::EcologyCapacityExceeded {
                    coord: ecology.coord,
                    biomass: ecology.before_biomass,
                    capacity: current.capacity,
                });
            }
        }

        for (coord, adjustment) in &water_adjustments {
            let cell = self
                .hydrology
                .get(*coord)
                .ok_or(SimulationApplyError::MissingHydrologyCell(*coord))?;
            let water = adjusted_water(cell.water, *adjustment)?;
            if water > i64::from(cell.capacity) {
                return Err(SimulationApplyError::WaterCapacityExceeded {
                    coord: *coord,
                    current: cell.water,
                    incoming: u32::try_from(*adjustment).unwrap_or(u32::MAX),
                    capacity: cell.capacity,
                });
            }
        }

        for (coord, adjustment) in water_adjustments {
            let cell = self
                .hydrology
                .cells_mut()
                .get_mut(&coord)
                .ok_or(SimulationApplyError::MissingHydrologyCell(coord))?;
            cell.water = u32::try_from(adjusted_water(cell.water, adjustment)?)
                .map_err(|_| SimulationApplyError::ArithmeticOverflow)?;
        }
        for voxel in &delta.voxel_deltas {
            self.voxels.set(
                voxel.coord,
                VoxelCell {
                    material: voxel.material,
                    mass: voxel.before_mass,
                    hardness: voxel.hardness,
                },
            );
        }
        for ecology in &delta.ecology_deltas {
            let current = self
                .ecology
                .get(ecology.coord)
                .ok_or(SimulationApplyError::MissingEcologyCell(ecology.coord))?;
            self.ecology.set(
                ecology.coord,
                crate::ecology::EcologyCell {
                    biomass: ecology.before_biomass,
                    ..current
                },
            );
        }
        for fragment in &delta.fragment_deltas {
            self.fragments.remove(fragment.stable_key);
        }
        self.tick = previous_tick;
        Ok(())
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
                hardness: before.hardness,
                before_mass: before.mass,
                after_mass,
                removed_mass: removed,
            });
            let fragment = FragmentDelta {
                stable_key: stable_fragment_key(self.tick, coord, 0),
                origin: coord,
                mass: removed,
                material: before.material,
                velocity: [0, -1, 0],
            };
            fragment_output.push(fragment);
            self.fragments.set(fragment);
        }
    }
}

fn adjusted_water(base: u32, adjustment: i64) -> Result<i64, SimulationApplyError> {
    i64::from(base)
        .checked_add(adjustment)
        .filter(|water| *water >= 0 && *water <= i64::from(u32::MAX))
        .ok_or(SimulationApplyError::ArithmeticOverflow)
}

fn adjust_water(
    adjustments: &mut BTreeMap<CellCoord, i64>,
    coord: CellCoord,
    amount: i64,
) -> Result<(), SimulationApplyError> {
    let entry = adjustments.entry(coord).or_insert(0);
    *entry = entry
        .checked_add(amount)
        .ok_or(SimulationApplyError::ArithmeticOverflow)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_overflow_is_reported_without_mutating_state() {
        let mut sim = WorldSim::new(WorldSimConfig::default()).unwrap();
        sim.tick = u64::MAX;
        assert_eq!(sim.next_tick(), Err(SimulationError::TickOverflow));
        assert_eq!(
            sim.step(TickBudget::new(1, 1, 1)),
            Err(SimulationError::TickOverflow)
        );
        assert_eq!(sim.tick, u64::MAX);
    }
}
