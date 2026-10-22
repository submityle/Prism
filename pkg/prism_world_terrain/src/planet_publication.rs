//! Atomic publication of a bounded set of planetary heightfield meshes.
//!
//! Mesh construction is intentionally separate from publication. Callers stage
//! any number of replacements and removals, then commit the complete batch only
//! after capacity, source identity, and every affected shared edge have been
//! validated. A failed transaction leaves the previous snapshot untouched.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

use crate::{
    validate_planet_heightfield_edge, BakeError, BakeLimits, GridChange, HeightfieldTile,
    PlanetHeightfieldBoundaryError, PlanetHeightfieldMesh, PlanetHeightfieldNormalNeighborhood,
    PlanetMeshConfig, TileEdge, TileKey,
};

/// Revision of the atomic planetary mesh publication contract.
pub const PLANET_MESH_PUBLICATION_VERSION: u32 = 1;

/// Hard limits for one published mesh snapshot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PlanetMeshPublicationLimits {
    /// Maximum number of simultaneously published tiles.
    pub max_tiles: usize,
    /// Maximum logical mesh payload bytes across the published snapshot.
    pub max_mesh_bytes: usize,
    /// Maximum decoded samples compared while validating affected shared edges.
    pub max_neighbor_comparisons: u64,
}

/// One immutable source/mesh pair in the published snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct PublishedPlanetHeightfieldMesh {
    source: Arc<HeightfieldTile>,
    mesh: Arc<PlanetHeightfieldMesh>,
}

impl PublishedPlanetHeightfieldMesh {
    /// Returns the source tile snapshot retained by the publication.
    pub fn source(&self) -> &HeightfieldTile {
        &self.source
    }

    /// Returns the origin-relative projected mesh.
    pub fn mesh(&self) -> &PlanetHeightfieldMesh {
        self.mesh.as_ref()
    }

    /// Returns the stable tile identity.
    pub fn tile(&self) -> TileKey {
        self.mesh.tile()
    }
}

/// Read-only result of a successful atomic commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanetMeshPublicationReceipt {
    /// New monotonically increasing publication generation.
    pub generation: u64,
    /// Number of tiles in the committed snapshot.
    pub published_tiles: usize,
    /// Number of tile replacements in the transaction.
    pub replaced_tiles: usize,
    /// Number of tile removals in the transaction.
    pub removed_tiles: usize,
    /// Total logical mesh bytes in the committed snapshot.
    pub mesh_bytes: usize,
}

/// Transaction-local staging area for one atomic publication.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlanetMeshPublicationTransaction {
    base_generation: u64,
    replacements: Vec<PublishedPlanetHeightfieldMesh>,
    removals: Vec<TileKey>,
}

impl PlanetMeshPublicationTransaction {
    /// Generation observed when this transaction began.
    pub const fn base_generation(&self) -> u64 {
        self.base_generation
    }

    /// Stages a fully built mesh replacement or a new tile.
    pub fn stage_build(
        &mut self,
        source: Arc<HeightfieldTile>,
        config: PlanetMeshConfig,
        limits: BakeLimits,
        neighborhood: PlanetHeightfieldNormalNeighborhood<'_>,
    ) -> Result<(), PlanetMeshPublicationError> {
        let mesh =
            PlanetHeightfieldMesh::build_with_neighborhood(&source, config, limits, neighborhood)
                .map_err(PlanetMeshPublicationError::Build)?;
        self.stage_prebuilt(source, mesh)
    }

    /// Stages a planetary mesh with a validated resident-derived LOD seam plan.
    ///
    /// Commit additionally verifies that every coarse neighbor referenced by
    /// the plan remains resident in the final atomic snapshot.
    pub fn stage_build_with_lod_seam_plan(
        &mut self,
        source: Arc<HeightfieldTile>,
        config: PlanetMeshConfig,
        limits: BakeLimits,
        neighborhood: PlanetHeightfieldNormalNeighborhood<'_>,
        plan: &crate::HeightfieldLodSeamPlan,
        max_mesh_bytes: usize,
    ) -> Result<(), PlanetMeshPublicationError> {
        let mut mesh =
            PlanetHeightfieldMesh::build_with_neighborhood(&source, config, limits, neighborhood)
                .map_err(PlanetMeshPublicationError::Build)?;
        mesh.apply_lod_seam_plan(plan, max_mesh_bytes)
            .map_err(PlanetMeshPublicationError::Build)?;
        self.stage_prebuilt(source, mesh)
    }

    fn stage_prebuilt(
        &mut self,
        source: Arc<HeightfieldTile>,
        mesh: PlanetHeightfieldMesh,
    ) -> Result<(), PlanetMeshPublicationError> {
        let tile = source.tile();
        if mesh.tile() != tile || mesh.revision() != source.revision() {
            return Err(PlanetMeshPublicationError::SourceMismatch { tile });
        }
        if self.removals.binary_search(&tile).is_ok() {
            return Err(PlanetMeshPublicationError::ConflictingOperation { tile });
        }
        if self
            .replacements
            .binary_search_by_key(&tile, |entry| entry.tile())
            .is_ok()
        {
            return Err(PlanetMeshPublicationError::DuplicateReplacement { tile });
        }
        let index = self
            .replacements
            .binary_search_by_key(&tile, |entry| entry.tile())
            .unwrap_or_else(|index| index);
        self.replacements
            .try_reserve(1)
            .map_err(|_| PlanetMeshPublicationError::AllocationFailed)?;
        self.replacements.insert(
            index,
            PublishedPlanetHeightfieldMesh {
                source,
                mesh: Arc::new(mesh),
            },
        );
        Ok(())
    }

    /// Stages a tile removal.
    pub fn stage_remove(&mut self, tile: TileKey) -> Result<(), PlanetMeshPublicationError> {
        if self
            .replacements
            .binary_search_by_key(&tile, |entry| entry.tile())
            .is_ok()
        {
            return Err(PlanetMeshPublicationError::ConflictingOperation { tile });
        }
        if self.removals.binary_search(&tile).is_ok() {
            return Err(PlanetMeshPublicationError::DuplicateRemoval { tile });
        }
        let index = self
            .removals
            .binary_search(&tile)
            .unwrap_or_else(|index| index);
        self.removals
            .try_reserve(1)
            .map_err(|_| PlanetMeshPublicationError::AllocationFailed)?;
        self.removals.insert(index, tile);
        Ok(())
    }

    /// Number of staged replacements.
    pub const fn replaced_tiles(&self) -> usize {
        self.replacements.len()
    }

    /// Number of staged removals.
    pub const fn removed_tiles(&self) -> usize {
        self.removals.len()
    }
}

/// Atomically published planetary mesh snapshot.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlanetMeshPublication {
    limits: PlanetMeshPublicationLimits,
    generation: u64,
    meshes: Vec<PublishedPlanetHeightfieldMesh>,
}

impl PlanetMeshPublication {
    /// Creates an empty publication with fixed capacity and validation limits.
    pub const fn new(limits: PlanetMeshPublicationLimits) -> Self {
        Self {
            limits,
            generation: 0,
            meshes: Vec::new(),
        }
    }

    /// Returns the immutable publication limits.
    pub const fn limits(&self) -> PlanetMeshPublicationLimits {
        self.limits
    }

    /// Returns the current publication generation.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns all published tiles in stable `TileKey` order.
    pub fn meshes(&self) -> &[PublishedPlanetHeightfieldMesh] {
        &self.meshes
    }

    /// Looks up one published tile without copying its mesh.
    pub fn mesh(&self, tile: TileKey) -> Option<&PublishedPlanetHeightfieldMesh> {
        self.meshes
            .binary_search_by_key(&tile, |entry| entry.tile())
            .ok()
            .map(|index| &self.meshes[index])
    }

    /// Starts a transaction against the current generation.
    pub const fn begin(&self) -> PlanetMeshPublicationTransaction {
        PlanetMeshPublicationTransaction {
            base_generation: self.generation,
            replacements: Vec::new(),
            removals: Vec::new(),
        }
    }

    /// Stages an incremental update against the current published mesh.
    ///
    /// The update is built in transaction-owned storage. The published mesh is
    /// not changed until [`Self::commit`] succeeds.
    pub fn stage_update(
        &self,
        transaction: &mut PlanetMeshPublicationTransaction,
        source: Arc<HeightfieldTile>,
        change: GridChange,
        neighborhood: PlanetHeightfieldNormalNeighborhood<'_>,
    ) -> Result<(), PlanetMeshPublicationError> {
        let tile = source.tile();
        let current = self
            .mesh(tile)
            .ok_or(PlanetMeshPublicationError::TileNotPublished { tile })?;
        let mut mesh = current.mesh().clone();
        mesh.update(&source, change, neighborhood)
            .map_err(PlanetMeshPublicationError::Build)?;
        transaction.stage_prebuilt(source, mesh)
    }

    /// Validates and atomically commits a transaction.
    pub fn commit(
        &mut self,
        transaction: PlanetMeshPublicationTransaction,
    ) -> Result<PlanetMeshPublicationReceipt, PlanetMeshPublicationError> {
        if transaction.base_generation != self.generation {
            return Err(PlanetMeshPublicationError::StaleTransaction {
                expected: self.generation,
                actual: transaction.base_generation,
            });
        }
        let next_generation = self
            .generation
            .checked_add(1)
            .ok_or(PlanetMeshPublicationError::GenerationExhausted)?;
        let final_count = final_tile_count(&self.meshes, &transaction)?;
        if final_count > self.limits.max_tiles {
            return Err(PlanetMeshPublicationError::TileCapacityExceeded {
                requested: final_count,
                limit: self.limits.max_tiles,
            });
        }
        let final_bytes = final_mesh_bytes(&self.meshes, &transaction)?;
        if final_bytes > self.limits.max_mesh_bytes {
            return Err(PlanetMeshPublicationError::MeshCapacityExceeded {
                requested: final_bytes,
                limit: self.limits.max_mesh_bytes,
            });
        }
        validate_replacements(&transaction)?;
        validate_resident_seam_neighbors(&self.meshes, &transaction)?;
        validate_affected_boundaries(
            &self.meshes,
            &transaction,
            self.limits.max_neighbor_comparisons,
        )?;

        let mut next = Vec::new();
        next.try_reserve_exact(final_count)
            .map_err(|_| PlanetMeshPublicationError::AllocationFailed)?;
        merge_snapshot(&self.meshes, &transaction, &mut next);
        self.meshes = next;
        self.generation = next_generation;
        Ok(PlanetMeshPublicationReceipt {
            generation: next_generation,
            published_tiles: final_count,
            replaced_tiles: transaction.replacements.len(),
            removed_tiles: transaction.removals.len(),
            mesh_bytes: final_bytes,
        })
    }
}

/// Publication validation, identity, capacity, or allocation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanetMeshPublicationError {
    /// Mesh construction or incremental update failed.
    Build(BakeError<core::convert::Infallible>),
    /// A staged source and mesh disagree about tile or revision identity.
    SourceMismatch {
        /// Tile whose source/mesh identity disagrees.
        tile: TileKey,
    },
    /// A tile was staged more than once for replacement.
    DuplicateReplacement {
        /// Tile already staged for replacement.
        tile: TileKey,
    },
    /// A tile was staged more than once for removal.
    DuplicateRemoval {
        /// Tile already staged for removal.
        tile: TileKey,
    },
    /// A tile was both removed and replaced.
    ConflictingOperation {
        /// Tile with conflicting staged operations.
        tile: TileKey,
    },
    /// An incremental update targeted a tile not in the publication.
    TileNotPublished {
        /// Tile absent from the current publication.
        tile: TileKey,
    },
    /// The transaction was created from an older snapshot.
    StaleTransaction {
        /// Current publication generation.
        expected: u64,
        /// Generation observed by the transaction.
        actual: u64,
    },
    /// The committed tile count exceeds the fixed publication capacity.
    TileCapacityExceeded {
        /// Number of tiles required by the next snapshot.
        requested: usize,
        /// Maximum number of published tiles.
        limit: usize,
    },
    /// The committed logical mesh bytes exceed the fixed publication capacity.
    MeshCapacityExceeded {
        /// Logical bytes required by the next snapshot.
        requested: usize,
        /// Maximum logical mesh bytes.
        limit: usize,
    },
    /// A published replacement has an invalid source/mesh identity.
    InvalidReplacement {
        /// Tile whose replacement failed identity validation.
        tile: TileKey,
    },
    /// Affected neighboring source edges are not identical.
    Boundary(PlanetHeightfieldBoundaryError),
    /// A published seam references a coarse tile absent from the final snapshot.
    MissingSeamNeighbor {
        /// Tile that owns the transition strip.
        tile: TileKey,
        /// Coarse neighbor required by the transition strip.
        neighbor: TileKey,
    },
    /// The staged seam topology does not match the final resident LOD set.
    SeamTopologyMismatch {
        /// Tile whose transition plan is stale or missing.
        tile: TileKey,
        /// Coarse neighbor required by final residency, if any.
        expected: Option<TileKey>,
        /// Coarse neighbor encoded in the staged mesh, if any.
        actual: Option<TileKey>,
    },
    /// The publication generation cannot advance further.
    GenerationExhausted,
    /// A bounded staging or commit allocation failed.
    AllocationFailed,
}

impl fmt::Display for PlanetMeshPublicationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Build(error) => write!(formatter, "planet mesh build failed: {error:?}"),
            Self::SourceMismatch { tile } => write!(formatter, "source/mesh mismatch for {tile:?}"),
            Self::DuplicateReplacement { tile } => {
                write!(formatter, "duplicate replacement for {tile:?}")
            }
            Self::DuplicateRemoval { tile } => write!(formatter, "duplicate removal for {tile:?}"),
            Self::ConflictingOperation { tile } => {
                write!(formatter, "conflicting operation for {tile:?}")
            }
            Self::TileNotPublished { tile } => write!(formatter, "tile {tile:?} is not published"),
            Self::StaleTransaction { expected, actual } => {
                write!(
                    formatter,
                    "stale publication transaction: expected {expected}, got {actual}"
                )
            }
            Self::TileCapacityExceeded { requested, limit } => {
                write!(
                    formatter,
                    "publication needs {requested} tiles, limit {limit}"
                )
            }
            Self::MeshCapacityExceeded { requested, limit } => {
                write!(
                    formatter,
                    "publication needs {requested} mesh bytes, limit {limit}"
                )
            }
            Self::InvalidReplacement { tile } => {
                write!(formatter, "invalid replacement for {tile:?}")
            }
            Self::Boundary(error) => {
                write!(formatter, "planet mesh boundary validation failed: {error}")
            }
            Self::MissingSeamNeighbor { tile, neighbor } => write!(
                formatter,
                "planet mesh {tile:?} has a seam requiring non-resident neighbor {neighbor:?}"
            ),
            Self::SeamTopologyMismatch {
                tile,
                expected,
                actual,
            } => write!(
                formatter,
                "planet mesh {tile:?} seam topology does not match residency: expected {expected:?}, got {actual:?}"
            ),
            Self::GenerationExhausted => {
                formatter.write_str("planet mesh publication generation exhausted")
            }
            Self::AllocationFailed => {
                formatter.write_str("planet mesh publication allocation failed")
            }
        }
    }
}

impl core::error::Error for PlanetMeshPublicationError {}

fn mesh_bytes(mesh: &PlanetHeightfieldMesh) -> Result<usize, PlanetMeshPublicationError> {
    mesh.positions()
        .len()
        .checked_mul(12)
        .and_then(|bytes| {
            mesh.normals()
                .len()
                .checked_mul(12)
                .and_then(|n| bytes.checked_add(n))
        })
        .and_then(|bytes| {
            mesh.uvs()
                .len()
                .checked_mul(8)
                .and_then(|n| bytes.checked_add(n))
        })
        .and_then(|bytes| {
            mesh.indices()
                .len()
                .checked_mul(4)
                .and_then(|n| bytes.checked_add(n))
        })
        .ok_or(PlanetMeshPublicationError::AllocationFailed)
}

fn final_tile_count(
    current: &[PublishedPlanetHeightfieldMesh],
    transaction: &PlanetMeshPublicationTransaction,
) -> Result<usize, PlanetMeshPublicationError> {
    let removed_existing = transaction
        .removals
        .iter()
        .filter(|tile| {
            current
                .binary_search_by(|entry| entry.tile().cmp(tile))
                .is_ok()
        })
        .count();
    let new_replacements = transaction
        .replacements
        .iter()
        .filter(|entry| {
            current
                .binary_search_by_key(&entry.tile(), |item| item.tile())
                .is_err()
        })
        .count();
    current
        .len()
        .checked_sub(removed_existing)
        .and_then(|count| count.checked_add(new_replacements))
        .ok_or(PlanetMeshPublicationError::AllocationFailed)
}

fn final_mesh_bytes(
    current: &[PublishedPlanetHeightfieldMesh],
    transaction: &PlanetMeshPublicationTransaction,
) -> Result<usize, PlanetMeshPublicationError> {
    let mut bytes = 0usize;
    for entry in current {
        if transaction.removals.binary_search(&entry.tile()).is_ok() {
            continue;
        }
        let mesh = transaction
            .replacements
            .binary_search_by_key(&entry.tile(), |item| item.tile())
            .ok()
            .map(|index| transaction.replacements[index].mesh.as_ref())
            .unwrap_or(entry.mesh.as_ref());
        bytes = bytes
            .checked_add(mesh_bytes(mesh)?)
            .ok_or(PlanetMeshPublicationError::AllocationFailed)?;
    }
    for entry in &transaction.replacements {
        if current
            .binary_search_by_key(&entry.tile(), |item| item.tile())
            .is_err()
        {
            bytes = bytes
                .checked_add(mesh_bytes(entry.mesh.as_ref())?)
                .ok_or(PlanetMeshPublicationError::AllocationFailed)?;
        }
    }
    Ok(bytes)
}

fn validate_replacements(
    transaction: &PlanetMeshPublicationTransaction,
) -> Result<(), PlanetMeshPublicationError> {
    for entry in &transaction.replacements {
        if entry.tile() != entry.source.tile() || entry.mesh.revision() != entry.source.revision() {
            return Err(PlanetMeshPublicationError::InvalidReplacement { tile: entry.tile() });
        }
    }
    Ok(())
}

fn validate_resident_seam_neighbors(
    current: &[PublishedPlanetHeightfieldMesh],
    transaction: &PlanetMeshPublicationTransaction,
) -> Result<(), PlanetMeshPublicationError> {
    for replacement in &transaction.replacements {
        validate_changed_lod_neighborhood(replacement.tile(), current, transaction)?;
    }
    for &removed in &transaction.removals {
        validate_changed_lod_neighborhood(removed, current, transaction)?;
    }
    Ok(())
}

fn validate_changed_lod_neighborhood(
    changed: TileKey,
    current: &[PublishedPlanetHeightfieldMesh],
    transaction: &PlanetMeshPublicationTransaction,
) -> Result<(), PlanetMeshPublicationError> {
    validate_final_seam_topology(changed, current, transaction)?;
    for edge in TileEdge::ALL {
        let (same_level, facing_edge) = changed.neighbor_with_edge(edge);
        validate_final_seam_topology(same_level, current, transaction)?;
        if let Ok(children) = same_level.children() {
            for child in children {
                if tile_child_touches_edge(child, facing_edge) {
                    validate_final_seam_topology(child, current, transaction)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_final_seam_topology(
    tile: TileKey,
    current: &[PublishedPlanetHeightfieldMesh],
    transaction: &PlanetMeshPublicationTransaction,
) -> Result<(), PlanetMeshPublicationError> {
    if !final_contains_tile(current, transaction, tile) {
        return Ok(());
    }
    let entry = match transaction
        .replacements
        .binary_search_by_key(&tile, |replacement| replacement.tile())
    {
        Ok(index) => &transaction.replacements[index],
        Err(_) => match current.binary_search_by_key(&tile, |resident| resident.tile()) {
            Ok(index) => &current[index],
            Err(_) => return Err(PlanetMeshPublicationError::InvalidReplacement { tile }),
        },
    };
    let seam_neighbors = entry.mesh.lod_seam_neighbors();
    for edge in TileEdge::ALL {
        let slot = match edge {
            TileEdge::West => 0,
            TileEdge::East => 1,
            TileEdge::South => 2,
            TileEdge::North => 3,
        };
        let (same_level, _) = tile.neighbor_with_edge(edge);
        let same_level_resident = final_contains_tile(current, transaction, same_level);
        let coarse = if same_level_resident {
            None
        } else {
            same_level
                .parent()
                .filter(|parent| final_contains_tile(current, transaction, *parent))
        };
        if seam_neighbors[slot] != coarse {
            if let Some(neighbor) = coarse.or(seam_neighbors[slot]) {
                if !final_contains_tile(current, transaction, neighbor) {
                    return Err(PlanetMeshPublicationError::MissingSeamNeighbor { tile, neighbor });
                }
            }
            return Err(PlanetMeshPublicationError::SeamTopologyMismatch {
                tile,
                expected: coarse,
                actual: seam_neighbors[slot],
            });
        }
    }
    Ok(())
}

fn tile_child_touches_edge(child: TileKey, edge: TileEdge) -> bool {
    let Some(parent) = child.parent() else {
        return false;
    };
    match edge {
        TileEdge::West => child.x() == parent.x() * 2,
        TileEdge::East => child.x() == parent.x() * 2 + 1,
        TileEdge::South => child.y() == parent.y() * 2,
        TileEdge::North => child.y() == parent.y() * 2 + 1,
    }
}

fn final_contains_tile(
    current: &[PublishedPlanetHeightfieldMesh],
    transaction: &PlanetMeshPublicationTransaction,
    tile: TileKey,
) -> bool {
    transaction.removals.binary_search(&tile).is_err()
        && (transaction
            .replacements
            .binary_search_by_key(&tile, |replacement| replacement.tile())
            .is_ok()
            || current
                .binary_search_by_key(&tile, |resident| resident.tile())
                .is_ok())
}

fn source_for<'a>(
    current: &'a [PublishedPlanetHeightfieldMesh],
    transaction: &'a PlanetMeshPublicationTransaction,
    tile: TileKey,
) -> Option<&'a HeightfieldTile> {
    if transaction.removals.binary_search(&tile).is_ok() {
        return None;
    }
    transaction
        .replacements
        .binary_search_by_key(&tile, |entry| entry.tile())
        .ok()
        .map(|index| transaction.replacements[index].source.as_ref())
        .or_else(|| {
            current
                .binary_search_by_key(&tile, |entry| entry.tile())
                .ok()
                .map(|index| current[index].source.as_ref())
        })
}

fn validate_affected_boundaries(
    current: &[PublishedPlanetHeightfieldMesh],
    transaction: &PlanetMeshPublicationTransaction,
    max_comparisons: u64,
) -> Result<(), PlanetMeshPublicationError> {
    let mut used = 0u64;
    for entry in &transaction.replacements {
        for edge in TileEdge::ALL {
            let (neighbor, _) = entry.tile().neighbor_with_edge(edge);
            let Some(right) = source_for(current, transaction, neighbor) else {
                continue;
            };
            if entry.tile() > neighbor
                && transaction
                    .replacements
                    .binary_search_by_key(&neighbor, |replacement| replacement.tile())
                    .is_ok()
            {
                continue;
            }
            let side = u64::from(entry.source.spec().samples_per_edge());
            let next = used
                .checked_add(side)
                .ok_or(PlanetMeshPublicationError::AllocationFailed)?;
            if next > max_comparisons {
                return Err(PlanetMeshPublicationError::Boundary(
                    PlanetHeightfieldBoundaryError::ComparisonBudgetExceeded {
                        required: next,
                        limit: max_comparisons,
                    },
                ));
            }
            validate_planet_heightfield_edge(&entry.source, edge, right, side)
                .map_err(PlanetMeshPublicationError::Boundary)?;
            used = next;
        }
    }
    Ok(())
}

fn merge_snapshot(
    current: &[PublishedPlanetHeightfieldMesh],
    transaction: &PlanetMeshPublicationTransaction,
    output: &mut Vec<PublishedPlanetHeightfieldMesh>,
) {
    let mut current_index = 0;
    let mut replacement_index = 0;
    while current_index < current.len() || replacement_index < transaction.replacements.len() {
        let current_tile = current
            .get(current_index)
            .map(PublishedPlanetHeightfieldMesh::tile);
        let replacement_tile = transaction
            .replacements
            .get(replacement_index)
            .map(PublishedPlanetHeightfieldMesh::tile);
        let next_tile = match (current_tile, replacement_tile) {
            (Some(left), Some(right)) => left.min(right),
            (Some(tile), None) | (None, Some(tile)) => tile,
            (None, None) => break,
        };
        if transaction.removals.binary_search(&next_tile).is_ok() {
            if current_tile == Some(next_tile) {
                current_index += 1;
            }
            if replacement_tile == Some(next_tile) {
                replacement_index += 1;
            }
            continue;
        }
        if replacement_tile == Some(next_tile) {
            output.push(transaction.replacements[replacement_index].clone());
            replacement_index += 1;
            if current_tile == Some(next_tile) {
                current_index += 1;
            }
        } else {
            output.push(current[current_index].clone());
            current_index += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::Infallible;

    use crate::{
        CubeFace, DirtyRegion, HeightSample, HeightSampleValue, HeightSampler,
        HeightfieldLodSeamPlan, HeightfieldSpec, LodSeamCoverage, LodSeamLimits, LodSeamNeighbor,
    };

    struct Constant(i64);
    impl HeightSampler for Constant {
        type Error = Infallible;

        fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            Ok(HeightSampleValue::from_millimetres(self.0))
        }
    }

    fn source(tile: TileKey, height: i64) -> Arc<HeightfieldTile> {
        Arc::new(
            HeightfieldTile::bake(
                tile,
                HeightfieldSpec::new(5, 1).unwrap(),
                &Constant(height),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 50,
                },
            )
            .unwrap(),
        )
    }

    fn limits() -> PlanetMeshPublicationLimits {
        PlanetMeshPublicationLimits {
            max_tiles: 4,
            max_mesh_bytes: 4096,
            max_neighbor_comparisons: 32,
        }
    }

    fn config() -> PlanetMeshConfig {
        PlanetMeshConfig::new(6_371_000_000, [0; 3], 10_000.0).unwrap()
    }

    fn seam_plan(tile: TileKey) -> (TileKey, HeightfieldLodSeamPlan) {
        let neighbor = tile.neighbor(TileEdge::East).parent().unwrap();
        let plan = HeightfieldLodSeamPlan::build_strict(
            tile,
            4,
            &[LodSeamNeighbor::with_boundary(
                TileEdge::East,
                neighbor,
                4,
                LodSeamCoverage::full(4),
            )],
            LodSeamLimits {
                max_segments: 4,
                max_vertices: 32,
                max_indices: 96,
                max_bytes: 4096,
            },
        )
        .unwrap();
        (neighbor, plan)
    }

    fn stage_seamed_tile(
        transaction: &mut PlanetMeshPublicationTransaction,
        tile: TileKey,
        plan: &HeightfieldLodSeamPlan,
    ) {
        transaction
            .stage_build_with_lod_seam_plan(
                source(tile, 7),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
                plan,
                4096,
            )
            .unwrap();
    }

    #[test]
    fn seam_and_coarse_neighbor_publish_in_one_transaction() {
        let tile = TileKey::new(CubeFace::PositiveZ, 3, 3, 3).unwrap();
        let (neighbor, plan) = seam_plan(tile);
        let mut publication = PlanetMeshPublication::new(limits());
        let mut transaction = publication.begin();
        stage_seamed_tile(&mut transaction, tile, &plan);
        transaction
            .stage_build(
                source(neighbor, 7),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();

        let receipt = publication.commit(transaction).unwrap();
        assert_eq!(receipt.published_tiles, 2);
        assert_eq!(publication.generation(), 1);
    }

    #[test]
    fn missing_seam_neighbor_rejects_commit_without_changing_snapshot() {
        let tile = TileKey::new(CubeFace::PositiveZ, 3, 3, 3).unwrap();
        let (neighbor, plan) = seam_plan(tile);
        let mut publication = PlanetMeshPublication::new(limits());
        let mut transaction = publication.begin();
        stage_seamed_tile(&mut transaction, tile, &plan);

        assert_eq!(
            publication.commit(transaction),
            Err(PlanetMeshPublicationError::MissingSeamNeighbor { tile, neighbor })
        );
        assert_eq!(publication.generation(), 0);
        assert!(publication.meshes().is_empty());
    }

    #[test]
    fn resident_coarse_neighbor_requires_matching_seam_topology() {
        let tile = TileKey::new(CubeFace::PositiveZ, 3, 3, 3).unwrap();
        let (neighbor, _) = seam_plan(tile);
        let mut publication = PlanetMeshPublication::new(limits());
        let mut transaction = publication.begin();
        transaction
            .stage_build(
                source(tile, 7),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();
        transaction
            .stage_build(
                source(neighbor, 7),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();

        assert_eq!(
            publication.commit(transaction),
            Err(PlanetMeshPublicationError::SeamTopologyMismatch {
                tile,
                expected: Some(neighbor),
                actual: None,
            })
        );
        assert_eq!(publication.generation(), 0);
        assert!(publication.meshes().is_empty());
    }

    #[test]
    fn removing_seam_neighbor_requires_removing_or_replacing_owner_seam() {
        let tile = TileKey::new(CubeFace::PositiveZ, 3, 3, 3).unwrap();
        let (neighbor, plan) = seam_plan(tile);
        let mut publication = PlanetMeshPublication::new(limits());
        let mut initial = publication.begin();
        stage_seamed_tile(&mut initial, tile, &plan);
        initial
            .stage_build(
                source(neighbor, 7),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();
        publication.commit(initial).unwrap();

        let mut remove_neighbor = publication.begin();
        remove_neighbor.stage_remove(neighbor).unwrap();
        let snapshot = publication.clone();
        assert_eq!(
            publication.commit(remove_neighbor),
            Err(PlanetMeshPublicationError::MissingSeamNeighbor { tile, neighbor })
        );
        assert_eq!(publication, snapshot);

        let mut clear_seam_and_remove = publication.begin();
        clear_seam_and_remove.stage_remove(neighbor).unwrap();
        clear_seam_and_remove
            .stage_build(
                source(tile, 7),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();
        let receipt = publication.commit(clear_seam_and_remove).unwrap();
        assert_eq!(receipt.published_tiles, 1);
        assert_eq!(publication.generation(), 2);

        let mut remove_owner = publication.begin();
        remove_owner.stage_remove(tile).unwrap();
        let receipt = publication.commit(remove_owner).unwrap();
        assert_eq!(receipt.published_tiles, 0);
    }

    #[test]
    fn replacing_resident_seam_neighbor_keeps_seam_publication_valid() {
        let tile = TileKey::new(CubeFace::PositiveZ, 3, 3, 3).unwrap();
        let (neighbor, plan) = seam_plan(tile);
        let mut publication = PlanetMeshPublication::new(limits());
        let mut initial = publication.begin();
        stage_seamed_tile(&mut initial, tile, &plan);
        initial
            .stage_build(
                source(neighbor, 7),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();
        publication.commit(initial).unwrap();

        let mut replacement = publication.begin();
        replacement
            .stage_build(
                source(neighbor, 7),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();
        let receipt = publication.commit(replacement).unwrap();
        assert_eq!(receipt.replaced_tiles, 1);
        assert_eq!(publication.generation(), 2);
    }

    #[test]
    fn matching_adjacent_batch_publishes_once() {
        let left_key = TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap();
        let right_key = left_key.neighbor(TileEdge::East);
        let left = source(left_key, 7);
        let right = source(right_key, 7);
        let mut publication = PlanetMeshPublication::new(limits());
        let mut transaction = publication.begin();
        transaction
            .stage_build(
                left,
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();
        transaction
            .stage_build(
                right,
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();
        let receipt = publication.commit(transaction).unwrap();
        assert_eq!(receipt.generation, 1);
        assert_eq!(receipt.published_tiles, 2);
        assert_eq!(publication.meshes().len(), 2);
    }

    #[test]
    fn mismatched_shared_edge_does_not_publish_any_tile() {
        let left_key = TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap();
        let right_key = left_key.neighbor(TileEdge::East);
        let mut publication = PlanetMeshPublication::new(limits());
        let mut transaction = publication.begin();
        transaction
            .stage_build(
                source(left_key, 7),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();
        transaction
            .stage_build(
                source(right_key, 8),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();
        assert!(matches!(
            publication.commit(transaction),
            Err(PlanetMeshPublicationError::Boundary(_))
        ));
        assert_eq!(publication.generation(), 0);
        assert!(publication.meshes().is_empty());
    }

    #[test]
    fn replacement_of_higher_sorted_tile_checks_existing_lower_neighbor() {
        let lower = TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap();
        let higher = lower.neighbor(TileEdge::East);
        assert!(lower < higher);
        let mut publication = PlanetMeshPublication::new(limits());
        let mut initial = publication.begin();
        for tile in [lower, higher] {
            initial
                .stage_build(
                    source(tile, 7),
                    config(),
                    BakeLimits {
                        max_samples: 25,
                        max_bytes: 4096,
                    },
                    PlanetHeightfieldNormalNeighborhood::empty(),
                )
                .unwrap();
        }
        publication.commit(initial).unwrap();
        let previous = publication.clone();

        let mut replacement = publication.begin();
        replacement
            .stage_build(
                source(higher, 8),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();
        assert!(matches!(
            publication.commit(replacement),
            Err(PlanetMeshPublicationError::Boundary(_))
        ));
        assert_eq!(publication, previous);
    }

    #[test]
    fn stale_transaction_cannot_overwrite_new_snapshot() {
        let tile = TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap();
        let mut publication = PlanetMeshPublication::new(limits());
        let mut first = publication.begin();
        first
            .stage_build(
                source(tile, 1),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();
        let second = publication.begin();
        publication.commit(first).unwrap();
        assert!(matches!(
            publication.commit(second),
            Err(PlanetMeshPublicationError::StaleTransaction { .. })
        ));
        assert_eq!(publication.meshes()[0].source().height_mm([0, 0]), Some(1));
    }

    #[test]
    fn capacity_failure_preserves_previous_snapshot() {
        let tile = TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap();
        let mut publication = PlanetMeshPublication::new(PlanetMeshPublicationLimits {
            max_tiles: 1,
            max_mesh_bytes: 1,
            max_neighbor_comparisons: 32,
        });
        let mut transaction = publication.begin();
        transaction
            .stage_build(
                source(tile, 1),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();
        assert!(matches!(
            publication.commit(transaction),
            Err(PlanetMeshPublicationError::MeshCapacityExceeded { .. })
        ));
        assert_eq!(publication.generation(), 0);
        assert!(publication.meshes().is_empty());
    }

    #[test]
    fn staged_incremental_update_keeps_published_mesh_immutable_until_commit() {
        let tile = TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap();
        let mut publication = PlanetMeshPublication::new(limits());
        let mut initial = publication.begin();
        initial
            .stage_build(
                source(tile, 1),
                config(),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 4096,
                },
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();
        publication.commit(initial).unwrap();
        let original_mesh = publication.mesh(tile).unwrap().mesh().clone();

        let mut updated_source = publication.mesh(tile).unwrap().source().clone();
        let change = updated_source
            .rebake(
                DirtyRegion::new(tile, 0, 0, u16::MAX, u16::MAX).unwrap(),
                &Constant(2),
                BakeLimits {
                    max_samples: 25,
                    max_bytes: 50,
                },
            )
            .unwrap();
        let mut transaction = publication.begin();
        publication
            .stage_update(
                &mut transaction,
                Arc::new(updated_source),
                change,
                PlanetHeightfieldNormalNeighborhood::empty(),
            )
            .unwrap();

        assert_eq!(publication.mesh(tile).unwrap().mesh(), &original_mesh);
        assert_eq!(publication.generation(), 1);
        let receipt = publication.commit(transaction).unwrap();
        assert_eq!(receipt.generation, 2);
        assert_eq!(
            publication.mesh(tile).unwrap().source().height_mm([0, 0]),
            Some(2)
        );
        assert_eq!(
            publication.mesh(tile).unwrap().mesh().revision(),
            change.revision()
        );
    }
}
