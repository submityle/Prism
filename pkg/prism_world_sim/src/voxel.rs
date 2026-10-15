use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::types::{CellCoord, Region};

/// One sparse voxel's material and remaining solid mass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VoxelCell {
    /// Stable material identifier.
    pub material: u16,
    /// Remaining mass in arbitrary fixed integer units.
    pub mass: u32,
    /// Resistance used by the local erosion primitive.
    pub hardness: u16,
}

impl VoxelCell {
    /// Creates a non-empty voxel cell.
    pub const fn new(material: u16, mass: u32, hardness: u16) -> Option<Self> {
        if mass == 0 {
            None
        } else {
            Some(Self {
                material,
                mass,
                hardness,
            })
        }
    }
}

/// Sparse authoritative voxel storage owned by the simulation host.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VoxelField {
    cells: BTreeMap<CellCoord, VoxelCell>,
}

impl VoxelField {
    /// Creates an empty sparse field.
    pub const fn new() -> Self {
        Self {
            cells: BTreeMap::new(),
        }
    }

    /// Inserts or replaces a non-empty cell.
    pub fn set(&mut self, coord: CellCoord, cell: VoxelCell) {
        self.cells.insert(coord, cell);
    }

    /// Returns a cell by coordinate.
    pub fn get(&self, coord: CellCoord) -> Option<VoxelCell> {
        self.cells.get(&coord).copied()
    }

    /// Removes a cell and returns its previous value.
    pub fn remove(&mut self, coord: CellCoord) -> Option<VoxelCell> {
        self.cells.remove(&coord)
    }

    /// Returns all coordinates in stable map order, optionally clipped to a region.
    pub fn coordinates(&self, region: Option<Region>) -> Vec<CellCoord> {
        self.cells
            .keys()
            .copied()
            .filter(|coord| region.is_none_or(|bounds| bounds.contains(*coord)))
            .collect()
    }

    /// Returns the number of stored cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Returns whether no voxels are stored.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

/// A persistent destructive change to one voxel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VoxelDelta {
    /// Changed coordinate.
    pub coord: CellCoord,
    /// Material identifier retained by the delta.
    pub material: u16,
    /// Mass before the change.
    pub before_mass: u32,
    /// Mass after the change.
    pub after_mass: u32,
    /// Mass removed from the voxel.
    pub removed_mass: u32,
}

/// A persistent fragment spawned by destructive voxel simulation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FragmentDelta {
    /// Stable key for replay, persistence, and deduplication.
    pub stable_key: u64,
    /// Source coordinate.
    pub origin: CellCoord,
    /// Fragment mass.
    pub mass: u32,
    /// Material identifier.
    pub material: u16,
    /// Integer initial velocity hint, in engine-neutral units.
    pub velocity: [i16; 3],
}

/// Derives a stable fragment key without a random hash seed.
pub const fn stable_fragment_key(tick: u64, origin: CellCoord, ordinal: u32) -> u64 {
    let mut value = tick ^ 0x9e37_79b9_7f4a_7c15;
    value ^= (origin.x as i64 as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
    value = value.rotate_left(17) ^ (origin.y as i64 as u64).wrapping_mul(0x6eed_0e9d_a4d9_4a4f);
    value = value.rotate_left(23) ^ (origin.z as i64 as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= (ordinal as u64).wrapping_mul(0x2545_f491_4f6c_dd1d);
    value ^ (value >> 29)
}
