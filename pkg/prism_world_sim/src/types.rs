use alloc::vec::Vec;

/// A signed integer cell coordinate in engine-neutral world space.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CellCoord {
    /// X coordinate.
    pub x: i32,
    /// Y coordinate.
    pub y: i32,
    /// Z coordinate.
    pub z: i32,
}

impl CellCoord {
    /// Creates a cell coordinate.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Returns the four horizontal cardinal neighbors in stable order.
    pub const fn horizontal_neighbors(self) -> [Self; 4] {
        [
            Self::new(self.x - 1, self.y, self.z),
            Self::new(self.x + 1, self.y, self.z),
            Self::new(self.x, self.y, self.z - 1),
            Self::new(self.x, self.y, self.z + 1),
        ]
    }
}

/// Inclusive integer bounds for a simulation region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Region {
    /// Inclusive minimum corner.
    pub min: CellCoord,
    /// Inclusive maximum corner.
    pub max: CellCoord,
}

impl Region {
    /// Creates bounds when every minimum is no greater than its maximum.
    pub const fn new(min: CellCoord, max: CellCoord) -> Option<Self> {
        if min.x <= max.x && min.y <= max.y && min.z <= max.z {
            Some(Self { min, max })
        } else {
            None
        }
    }

    /// Returns the one-cell region containing a coordinate.
    pub const fn cell(cell: CellCoord) -> Self {
        Self {
            min: cell,
            max: cell,
        }
    }

    /// Returns whether the coordinate is inside the inclusive bounds.
    pub const fn contains(self, cell: CellCoord) -> bool {
        cell.x >= self.min.x
            && cell.x <= self.max.x
            && cell.y >= self.min.y
            && cell.y <= self.max.y
            && cell.z >= self.min.z
            && cell.z <= self.max.z
    }

    /// Returns a region covering two coordinates.
    pub fn covering(a: CellCoord, b: CellCoord) -> Self {
        Self {
            min: CellCoord::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z)),
            max: CellCoord::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z)),
        }
    }

    /// Expands the region by a non-negative number of cells with saturation.
    pub const fn expand(self, amount: i32) -> Self {
        let amount = if amount < 0 { 0 } else { amount };
        Self {
            min: CellCoord::new(
                self.min.x.saturating_sub(amount),
                self.min.y.saturating_sub(amount),
                self.min.z.saturating_sub(amount),
            ),
            max: CellCoord::new(
                self.max.x.saturating_add(amount),
                self.max.y.saturating_add(amount),
                self.max.z.saturating_add(amount),
            ),
        }
    }

    /// Returns whether two regions overlap or touch on a face/edge/corner.
    pub const fn intersects_or_touches(self, other: Self) -> bool {
        self.min.x <= other.max.x.saturating_add(1)
            && other.min.x <= self.max.x.saturating_add(1)
            && self.min.y <= other.max.y.saturating_add(1)
            && other.min.y <= self.max.y.saturating_add(1)
            && self.min.z <= other.max.z.saturating_add(1)
            && other.min.z <= self.max.z.saturating_add(1)
    }

    /// Returns the smallest region containing both regions.
    pub fn merge(self, other: Self) -> Self {
        Self::covering(
            CellCoord::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            CellCoord::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        )
    }
}

/// A dirty output region produced by one simulation tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirtyRegion {
    /// The affected inclusive bounds.
    pub region: Region,
}

pub(crate) struct DirtyRegions {
    items: Vec<DirtyRegion>,
    max_items: u32,
}

impl DirtyRegions {
    pub(crate) fn new(max_items: u32) -> Self {
        Self {
            items: Vec::new(),
            max_items,
        }
    }

    pub(crate) fn try_insert(&mut self, region: Region) -> bool {
        if self.max_items == 0 {
            return false;
        }
        let mut merged = region;
        let mut merged_existing = false;
        let mut index = 0;
        while index < self.items.len() {
            if self.items[index].region.intersects_or_touches(merged) {
                merged = merged.merge(self.items[index].region);
                merged_existing = true;
                self.items.remove(index);
            } else {
                index += 1;
            }
        }
        if !merged_existing && self.items.len() as u32 >= self.max_items {
            return false;
        }
        self.items.push(DirtyRegion { region: merged });
        self.items.sort_by_key(|item| item.region.min);
        true
    }

    pub(crate) fn into_vec(self) -> Vec<DirtyRegion> {
        self.items
    }
}
