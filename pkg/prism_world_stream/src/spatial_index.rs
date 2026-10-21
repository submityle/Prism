//! Deterministic hierarchical indexing for world cell manifests.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::{
    validate_manifests, AabbI64, CellManifest, ManifestValidationError, SectorGrid, SectorId,
};

const MAX_LEAF_ENTRIES: usize = 16;
const MAX_DEPTH: u8 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SectorBounds {
    min: [i64; 3],
    max: [i64; 3],
}

impl SectorBounds {
    fn from_sector(sector: SectorId) -> Self {
        Self {
            min: [sector.x, sector.y, sector.z],
            max: [sector.x, sector.y, sector.z],
        }
    }

    fn include(&mut self, sector: SectorId) {
        self.min[0] = self.min[0].min(sector.x);
        self.min[1] = self.min[1].min(sector.y);
        self.min[2] = self.min[2].min(sector.z);
        self.max[0] = self.max[0].max(sector.x);
        self.max[1] = self.max[1].max(sector.y);
        self.max[2] = self.max[2].max(sector.z);
    }

    fn intersects(self, other: Self) -> bool {
        (0..3).all(|axis| self.min[axis] <= other.max[axis] && other.min[axis] <= self.max[axis])
    }

    fn is_point(self) -> bool {
        self.min == self.max
    }
}

#[derive(Debug)]
struct Node {
    bounds: SectorBounds,
    children: [Option<usize>; 8],
    entries: Vec<usize>,
}

/// A deterministic hierarchy over the lossless sector identities in a manifest set.
///
/// The index is immutable after construction. Each level owns an independent octree;
/// queries never compare sectors from different levels. The hierarchy is an acceleration
/// structure only: returned manifests are sorted by their complete [`SectorId`] identity.
#[derive(Debug)]
pub struct ManifestSpatialIndex {
    manifests: Vec<CellManifest>,
    roots: BTreeMap<u8, usize>,
    nodes: Vec<Node>,
}

impl ManifestSpatialIndex {
    /// Builds an index and rejects duplicate sector identities before allocating nodes.
    pub fn build(manifests: &[CellManifest]) -> Result<Self, ManifestValidationError> {
        validate_manifests(manifests)?;
        let mut ordered = manifests.to_vec();
        ordered.sort_unstable_by_key(|manifest| manifest.sector);

        let mut by_level: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
        for (index, manifest) in ordered.iter().enumerate() {
            by_level
                .entry(manifest.sector.level)
                .or_default()
                .push(index);
        }

        let mut index = Self {
            manifests: ordered,
            roots: BTreeMap::new(),
            nodes: Vec::new(),
        };
        for (level, entries) in by_level {
            let mut bounds = SectorBounds::from_sector(index.manifests[entries[0]].sector);
            for &entry in &entries[1..] {
                bounds.include(index.manifests[entry].sector);
            }
            let root = index.build_node(level, bounds, entries, 0);
            index.roots.insert(level, root);
        }
        Ok(index)
    }

    /// Returns the number of indexed manifests.
    #[must_use]
    pub fn len(&self) -> usize {
        self.manifests.len()
    }

    /// Returns whether the index contains no manifests.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.manifests.is_empty()
    }

    /// Looks up one complete sector identity without scanning the hierarchy.
    #[must_use]
    pub fn get(&self, sector: SectorId) -> Option<CellManifest> {
        self.manifests
            .binary_search_by_key(&sector, |manifest| manifest.sector)
            .ok()
            .map(|index| self.manifests[index])
    }

    /// Queries an inclusive sector-coordinate box at one hierarchy level.
    #[must_use]
    pub fn query_sector_range(&self, level: u8, min: SectorId, max: SectorId) -> Vec<CellManifest> {
        if min.level != level
            || max.level != level
            || min.x > max.x
            || min.y > max.y
            || min.z > max.z
        {
            return Vec::new();
        }
        let Some(&root) = self.roots.get(&level) else {
            return Vec::new();
        };
        let query = SectorBounds {
            min: [min.x, min.y, min.z],
            max: [max.x, max.y, max.z],
        };
        let mut indices = Vec::new();
        self.query_node(root, query, &mut indices);
        indices.sort_unstable_by_key(|&index| self.manifests[index].sector);
        indices
            .into_iter()
            .map(|index| self.manifests[index])
            .collect()
    }

    /// Queries an inclusive Chebyshev-radius cube around a sector.
    #[must_use]
    pub fn query_sector_cube(&self, center: SectorId, radius: u64) -> Vec<CellManifest> {
        let min = SectorId::new(
            center.level,
            saturating_add_signed(center.x, -(radius as i128)),
            saturating_add_signed(center.y, -(radius as i128)),
            saturating_add_signed(center.z, -(radius as i128)),
        );
        let max = SectorId::new(
            center.level,
            saturating_add_signed(center.x, radius as i128),
            saturating_add_signed(center.y, radius as i128),
            saturating_add_signed(center.z, radius as i128),
        );
        self.query_sector_range(center.level, min, max)
    }

    /// Queries manifests whose authoritative world bounds intersect an AABB.
    ///
    /// `grid` must describe the sector size for `level`. Sector indexing first
    /// prunes candidates hierarchically, then performs an exact inclusive AABB test.
    #[must_use]
    pub fn query_bounds(&self, grid: SectorGrid, level: u8, bounds: AabbI64) -> Vec<CellManifest> {
        let (min_sector, _) = grid.locate(level, bounds.min);
        let (max_sector, _) = grid.locate(level, bounds.max);
        self.query_sector_range(level, min_sector, max_sector)
            .into_iter()
            .filter(|manifest| aabb_intersects(manifest.bounds, bounds))
            .collect()
    }

    fn build_node(
        &mut self,
        level: u8,
        bounds: SectorBounds,
        entries: Vec<usize>,
        depth: u8,
    ) -> usize {
        let node_index = self.nodes.len();
        self.nodes.push(Node {
            bounds,
            children: [None; 8],
            entries: Vec::new(),
        });
        if entries.len() <= MAX_LEAF_ENTRIES || depth >= MAX_DEPTH || bounds.is_point() {
            self.nodes[node_index].entries = entries;
            return node_index;
        }

        let midpoint = [
            midpoint(bounds.min[0], bounds.max[0]),
            midpoint(bounds.min[1], bounds.max[1]),
            midpoint(bounds.min[2], bounds.max[2]),
        ];
        let mut buckets: [Vec<usize>; 8] = core::array::from_fn(|_| Vec::new());
        for entry in entries {
            let sector = self.manifests[entry].sector;
            let child = usize::from(sector.x > midpoint[0])
                | (usize::from(sector.y > midpoint[1]) << 1)
                | (usize::from(sector.z > midpoint[2]) << 2);
            buckets[child].push(entry);
        }
        let mut children = [None; 8];
        for (child, entries) in buckets.into_iter().enumerate() {
            if entries.is_empty() {
                continue;
            }
            let child_bounds = child_bounds(bounds, midpoint, child);
            children[child] = Some(self.build_node(level, child_bounds, entries, depth + 1));
        }
        self.nodes[node_index].children = children;
        let _ = level;
        node_index
    }

    fn query_node(&self, node_index: usize, query: SectorBounds, output: &mut Vec<usize>) {
        let node = &self.nodes[node_index];
        if !node.bounds.intersects(query) {
            return;
        }
        if node.children.iter().all(Option::is_none) {
            output.extend(
                node.entries
                    .iter()
                    .copied()
                    .filter(|&index| query.contains(self.manifests[index].sector)),
            );
            return;
        }
        for child in node.children.iter().flatten().copied() {
            self.query_node(child, query, output);
        }
    }
}

impl SectorBounds {
    fn contains(self, sector: SectorId) -> bool {
        sector.x >= self.min[0]
            && sector.x <= self.max[0]
            && sector.y >= self.min[1]
            && sector.y <= self.max[1]
            && sector.z >= self.min[2]
            && sector.z <= self.max[2]
    }
}

fn midpoint(min: i64, max: i64) -> i64 {
    ((min as i128 + max as i128) / 2) as i64
}

fn child_bounds(parent: SectorBounds, midpoint: [i64; 3], child: usize) -> SectorBounds {
    let mut min = parent.min;
    let mut max = parent.max;
    for axis in 0..3 {
        if child & (1 << axis) == 0 {
            max[axis] = midpoint[axis];
        } else {
            min[axis] = midpoint[axis].saturating_add(1);
        }
    }
    SectorBounds { min, max }
}

fn saturating_add_signed(value: i64, delta: i128) -> i64 {
    let result = value as i128 + delta;
    result.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

fn aabb_intersects(left: AabbI64, right: AabbI64) -> bool {
    left.min.x <= right.max.x
        && right.min.x <= left.max.x
        && left.min.y <= right.max.y
        && right.min.y <= left.max.y
        && left.min.z <= right.max.z
        && right.min.z <= left.max.z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataLayerMask;

    fn manifest(level: u8, x: i64, y: i64, z: i64) -> CellManifest {
        CellManifest {
            sector: SectorId::new(level, x, y, z),
            bounds: AabbI64::new(
                GlobalPos::new(
                    x.saturating_mul(10),
                    y.saturating_mul(10),
                    z.saturating_mul(10),
                ),
                GlobalPos::new(
                    x.saturating_mul(10).saturating_add(9),
                    y.saturating_mul(10).saturating_add(9),
                    z.saturating_mul(10).saturating_add(9),
                ),
            )
            .unwrap(),
            scene_key: (x as i128 ^ y as i128 ^ z as i128) as u128,
            content_version: 1,
            derived_hash: 2,
            io_bytes: 3,
            data_layers: DataLayerMask::EMPTY,
        }
    }

    #[test]
    fn hierarchy_queries_negative_extreme_and_level_isolation_deterministically() {
        let manifests = [
            manifest(1, i64::MAX, 0, 0),
            manifest(0, -100, 2, 3),
            manifest(0, i64::MIN, 0, 0),
            manifest(0, -99, 2, 3),
            manifest(1, 0, 0, 0),
        ];
        let index = ManifestSpatialIndex::build(&manifests).unwrap();
        let result = index.query_sector_cube(SectorId::new(0, -100, 2, 3), 1);
        let sectors: Vec<_> = result.into_iter().map(|manifest| manifest.sector).collect();
        assert_eq!(
            sectors,
            vec![SectorId::new(0, -100, 2, 3), SectorId::new(0, -99, 2, 3)]
        );
        assert!(index
            .query_sector_cube(SectorId::new(0, i64::MIN, 0, 0), 0)
            .iter()
            .any(|manifest| manifest.sector.x == i64::MIN));
        assert!(index
            .query_sector_cube(SectorId::new(1, 0, 0, 0), 0)
            .iter()
            .all(|manifest| manifest.sector.level == 1));
    }

    #[test]
    fn exact_bounds_query_prunes_then_checks_authoritative_aabbs() {
        let manifests = [manifest(0, 0, 0, 0), manifest(0, 1, 0, 0)];
        let index = ManifestSpatialIndex::build(&manifests).unwrap();
        let grid = SectorGrid::new(10).unwrap();
        let bounds = AabbI64::new(GlobalPos::new(9, 0, 0), GlobalPos::new(10, 0, 0)).unwrap();
        let sectors: Vec<_> = index
            .query_bounds(grid, 0, bounds)
            .into_iter()
            .map(|manifest| manifest.sector)
            .collect();
        assert_eq!(
            sectors,
            vec![SectorId::new(0, 0, 0, 0), SectorId::new(0, 1, 0, 0)]
        );
    }

    #[test]
    fn duplicate_sector_is_rejected_before_index_construction() {
        let duplicate = [manifest(0, 0, 0, 0), manifest(0, 0, 0, 0)];
        assert!(matches!(
            ManifestSpatialIndex::build(&duplicate),
            Err(ManifestValidationError::DuplicateSector { .. })
        ));
    }
}
