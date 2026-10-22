//! Strictly bounded GPU consumer for published planetary heightfield meshes.
//!
//! This is a tile-buffer cache and indexed-draw encoding boundary, not scene
//! extraction or a product renderer. Callers must preserve one publication
//! lineage between delta applications. A full snapshot can be applied at any
//! time to resynchronize (including after a missed delta).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use prism_world_terrain::{PlanetMeshPublication, PlanetMeshPublicationDelta, TileKey};
use wgpu::util::DeviceExt;

use crate::GpuContext;

/// Explicit vertex ABI: eight little-endian `f32` words (position, normal, UV).
pub const PLANET_TERRAIN_VERTEX_STRIDE: u64 = 32;

/// Hard limits for the resident GPU mesh cache and atomic update overlap.
/// Buffers are per tile and their exact ABI payload bytes count toward both
/// applicable byte budgets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanetTerrainRhiLimits {
    /// Maximum resident tiles.
    pub max_tiles: usize,
    /// Maximum number of native GPU buffers (two per resident tile).
    pub max_buffers: usize,
    /// Maximum cache-owned GPU buffers while an atomic update is staged.
    pub max_transaction_buffers: usize,
    /// Maximum total vertex and index buffer bytes.
    pub max_buffer_bytes: u64,
    /// Maximum cache-owned GPU bytes while an atomic snapshot or delta is
    /// staged. This includes the old resident buffers plus all replacement or
    /// added buffers that must coexist before the cache can commit.
    pub max_transaction_buffer_bytes: u64,
}

/// Logical cache accounting, useful for admission decisions and tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PlanetTerrainRhiStats {
    /// Current publication generation represented by the cache.
    pub generation: u64,
    /// Number of resident tile meshes.
    pub tiles: usize,
    /// Total allocated vertex and index buffer payload bytes.
    pub buffer_bytes: u64,
}

/// Validation, sequencing, budget, or resource ownership failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanetTerrainRhiError {
    /// This consumer has not been initialized from a full publication snapshot.
    SnapshotRequired,
    /// Delta is stale, skipped a generation, or disagrees with its publication.
    Generation { expected: u64, actual: u64 },
    /// Delta was produced by a different publication lineage than the cache.
    ForeignLineage,
    /// Delta does not consume the exact snapshot currently held by this cache.
    StaleBase,
    /// Delta target is not the exact publication snapshot supplied with it.
    TargetStateMismatch,
    /// Caller supplied a different logical GPU device.
    ForeignDevice,
    /// Invalid or duplicate mesh/tile references in supplied data.
    InvalidMesh,
    /// A removal names a tile that is not in the cached lineage.
    UnknownRemoval,
    /// Replacements and removals contain the same tile or duplicate a key.
    ConflictingDelta,
    /// Tile or byte budget would be exceeded.
    BudgetExceeded { tiles: usize, bytes: u64 },
    /// Atomic replacement would exceed the configured temporary GPU peak.
    TransactionBudgetExceeded {
        resident_bytes: u64,
        staged_bytes: u64,
        byte_limit: u64,
        resident_buffers: usize,
        staged_buffers: usize,
        buffer_limit: usize,
    },
    /// Buffer size exceeds the selected device's maximum buffer size.
    DeviceBufferLimit { bytes: u64, limit: u64 },
    /// A checked byte-size or indexed-draw conversion overflowed.
    SizeOverflow,
    /// CPU staging allocation failed without mutating the old cache.
    AllocationFailed,
}

impl std::fmt::Display for PlanetTerrainRhiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SnapshotRequired => f.write_str("planet terrain requires a full snapshot first"),
            Self::Generation { expected, actual } => write!(
                f,
                "planet terrain generation mismatch: expected {expected}, got {actual}"
            ),
            Self::ForeignLineage => {
                f.write_str("planet terrain delta belongs to another publication lineage")
            }
            Self::StaleBase => {
                f.write_str("planet terrain delta does not consume the cached snapshot state")
            }
            Self::TargetStateMismatch => {
                f.write_str("planet terrain delta target differs from the supplied snapshot")
            }
            Self::ForeignDevice => {
                f.write_str("planet terrain cache belongs to another GPU device")
            }
            Self::InvalidMesh => {
                f.write_str("planet terrain mesh has invalid identity, data, or indices")
            }
            Self::UnknownRemoval => f.write_str("planet terrain delta removes an unknown tile"),
            Self::ConflictingDelta => f.write_str(
                "planet terrain delta contains duplicate or conflicting tile operations",
            ),
            Self::BudgetExceeded { tiles, bytes } => write!(
                f,
                "planet terrain cache needs {tiles} tiles and {bytes} bytes"
            ),
            Self::TransactionBudgetExceeded {
                resident_bytes,
                staged_bytes,
                byte_limit,
                resident_buffers,
                staged_buffers,
                buffer_limit,
            } => write!(
                f,
                "planet terrain update peak is {resident_bytes} resident + {staged_bytes} staged bytes (limit {byte_limit}) and {resident_buffers} resident + {staged_buffers} staged buffers (limit {buffer_limit})"
            ),
            Self::DeviceBufferLimit { bytes, limit } => write!(
                f,
                "planet terrain buffer size {bytes} exceeds device limit {limit}"
            ),
            Self::SizeOverflow => f.write_str("planet terrain buffer or draw size overflow"),
            Self::AllocationFailed => f.write_str("planet terrain CPU staging allocation failed"),
        }
    }
}

impl std::error::Error for PlanetTerrainRhiError {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuPlanetTerrainVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

struct GpuTile {
    key: TileKey,
    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    index_count: u32,
    bytes: u64,
}

struct StagedTile {
    key: TileKey,
    vertices: Vec<GpuPlanetTerrainVertex>,
    indices: Vec<u32>,
    index_count: u32,
    bytes: u64,
}

/// Same-device planetary heightfield GPU cache with atomic snapshot/delta
/// application and a minimal indexed-draw entry point.
pub struct PlanetTerrainRhiCache {
    device: wgpu::Device,
    identity: Arc<()>,
    limits: PlanetTerrainRhiLimits,
    generation: Option<u64>,
    lineage: Option<Arc<()>>,
    state: Option<Arc<()>>,
    tiles: Vec<GpuTile>,
    buffer_bytes: u64,
}

impl PlanetTerrainRhiCache {
    /// Creates an empty cache. A full snapshot is mandatory before any delta.
    #[must_use]
    pub fn new(context: &GpuContext, limits: PlanetTerrainRhiLimits) -> Self {
        Self {
            device: context.device().clone(),
            identity: context.device_identity().clone(),
            limits,
            generation: None,
            lineage: None,
            state: None,
            tiles: Vec::new(),
            buffer_bytes: 0,
        }
    }

    /// Replaces the complete cache from a publication snapshot. This also
    /// provides recovery after a missed delta and may move generation forward
    /// or backward because it is an authoritative resynchronization operation.
    pub fn apply_snapshot(
        &mut self,
        context: &GpuContext,
        publication: &PlanetMeshPublication,
    ) -> Result<PlanetTerrainRhiStats, PlanetTerrainRhiError> {
        self.check_device(context)?;
        let mut staged = Vec::new();
        staged
            .try_reserve_exact(publication.meshes().len())
            .map_err(|_| PlanetTerrainRhiError::AllocationFailed)?;
        for published in publication.meshes() {
            staged.push(stage_mesh(published.tile(), published.mesh())?);
        }
        let bytes = total_bytes(&staged)?;
        self.check_budget(staged.len(), bytes, &staged)?;
        self.check_transaction_budget(staged.len(), bytes)?;
        let next = self.upload_all(staged)?;
        self.tiles = next;
        self.buffer_bytes = bytes;
        self.generation = Some(publication.generation());
        self.lineage = Some(Arc::clone(publication.lineage_token()));
        self.state = Some(Arc::clone(publication.state_token()));
        Ok(self.stats())
    }

    /// Applies one delta against the exact next publication generation.
    /// The publication must be the same lineage used to create `delta`; the
    /// source API has no publisher token, so callers are responsible for that
    /// provenance. All CPU validation/staging and final-cache budget checks
    /// complete before GPU allocation or cache mutation.
    pub fn apply_delta(
        &mut self,
        context: &GpuContext,
        publication: &PlanetMeshPublication,
        delta: &PlanetMeshPublicationDelta,
    ) -> Result<PlanetTerrainRhiStats, PlanetTerrainRhiError> {
        self.check_device(context)?;
        let current = self
            .generation
            .ok_or(PlanetTerrainRhiError::SnapshotRequired)?;
        let expected = current
            .checked_add(1)
            .ok_or(PlanetTerrainRhiError::SizeOverflow)?;
        if delta.generation() != expected || publication.generation() != delta.generation() {
            return Err(PlanetTerrainRhiError::Generation {
                expected,
                actual: delta.generation(),
            });
        }
        let Some(lineage) = self.lineage.as_ref() else {
            return Err(PlanetTerrainRhiError::SnapshotRequired);
        };
        if !Arc::ptr_eq(lineage, publication.lineage_token())
            || !Arc::ptr_eq(lineage, delta.lineage_token())
        {
            return Err(PlanetTerrainRhiError::ForeignLineage);
        }
        let Some(state) = self.state.as_ref() else {
            return Err(PlanetTerrainRhiError::SnapshotRequired);
        };
        if !Arc::ptr_eq(state, delta.base_state_token()) {
            return Err(PlanetTerrainRhiError::StaleBase);
        }
        if !Arc::ptr_eq(publication.state_token(), delta.target_state_token()) {
            return Err(PlanetTerrainRhiError::TargetStateMismatch);
        }
        let mut cached_keys = Vec::new();
        cached_keys
            .try_reserve_exact(self.tiles.len())
            .map_err(|_| PlanetTerrainRhiError::AllocationFailed)?;
        cached_keys.extend(self.tiles.iter().map(|tile| tile.key));
        validate_delta_keys(delta.changed_tiles(), delta.removed_tiles(), &cached_keys)?;

        let mut operations = Vec::new();
        operations
            .try_reserve_exact(
                delta
                    .changed_tiles()
                    .len()
                    .checked_add(delta.removed_tiles().len())
                    .ok_or(PlanetTerrainRhiError::SizeOverflow)?,
            )
            .map_err(|_| PlanetTerrainRhiError::AllocationFailed)?;
        for &key in delta.changed_tiles() {
            let published = publication
                .mesh(key)
                .ok_or(PlanetTerrainRhiError::InvalidMesh)?;
            if published.tile() != key {
                return Err(PlanetTerrainRhiError::InvalidMesh);
            }
            operations.push((key, Some(stage_mesh(key, published.mesh())?)));
        }
        for &key in delta.removed_tiles() {
            operations.push((key, None));
        }
        operations.sort_by_key(|(key, _)| *key);

        let additions = operations
            .iter()
            .filter(|(key, staged)| {
                staged.is_some()
                    && self
                        .tiles
                        .binary_search_by_key(key, |tile| tile.key)
                        .is_err()
            })
            .count();
        let removals = operations
            .iter()
            .filter(|(_, staged)| staged.is_none())
            .count();
        let final_count = self
            .tiles
            .len()
            .checked_add(additions)
            .and_then(|count| count.checked_sub(removals))
            .ok_or(PlanetTerrainRhiError::SizeOverflow)?;
        let mut final_bytes = 0_u64;
        let mut old_index = 0;
        let mut operation_index = 0;
        while old_index < self.tiles.len() || operation_index < operations.len() {
            let old_key = self.tiles.get(old_index).map(|tile| tile.key);
            let op_key = operations.get(operation_index).map(|(key, _)| *key);
            match (old_key, op_key) {
                (Some(old), Some(op)) if old < op => {
                    final_bytes = final_bytes
                        .checked_add(self.tiles[old_index].bytes)
                        .ok_or(PlanetTerrainRhiError::SizeOverflow)?;
                    old_index += 1;
                }
                (_, Some(op)) if old_key.is_none_or(|old| op <= old) => {
                    if let Some(staged) = operations[operation_index].1.as_ref() {
                        final_bytes = final_bytes
                            .checked_add(staged.bytes)
                            .ok_or(PlanetTerrainRhiError::SizeOverflow)?;
                    }
                    if old_key == Some(op) {
                        old_index += 1;
                    }
                    operation_index += 1;
                }
                (Some(_), None) => {
                    final_bytes = final_bytes
                        .checked_add(self.tiles[old_index].bytes)
                        .ok_or(PlanetTerrainRhiError::SizeOverflow)?;
                    old_index += 1;
                }
                (None, Some(_)) | (Some(_), Some(_)) => {
                    // The guarded arms above handle the ordered cases. This
                    // fallback is unreachable for valid sorted keys, but it
                    // keeps validation total if a future key ordering changes.
                    return Err(PlanetTerrainRhiError::ConflictingDelta);
                }
                (None, None) => break,
            }
        }
        if final_count > self.limits.max_tiles
            || final_count
                .checked_mul(2)
                .is_none_or(|count| count > self.limits.max_buffers)
            || final_bytes > self.limits.max_buffer_bytes
        {
            return Err(PlanetTerrainRhiError::BudgetExceeded {
                tiles: final_count,
                bytes: final_bytes,
            });
        }
        let (staged_tiles, staged_bytes) =
            operations
                .iter()
                .try_fold((0_usize, 0_u64), |(tiles, bytes), (_, tile)| {
                    let Some(tile) = tile.as_ref() else {
                        return Ok((tiles, bytes));
                    };
                    Ok((
                        tiles
                            .checked_add(1)
                            .ok_or(PlanetTerrainRhiError::SizeOverflow)?,
                        bytes
                            .checked_add(tile.bytes)
                            .ok_or(PlanetTerrainRhiError::SizeOverflow)?,
                    ))
                })?;
        self.check_transaction_budget(staged_tiles, staged_bytes)?;
        let device_limit = self.device.limits().max_buffer_size;
        if let Some(tile) = operations
            .iter()
            .filter_map(|(_, tile)| tile.as_ref())
            .find(|tile| tile.bytes > device_limit)
        {
            return Err(PlanetTerrainRhiError::DeviceBufferLimit {
                bytes: tile.bytes,
                limit: device_limit,
            });
        }

        let mut next = Vec::new();
        next.try_reserve_exact(final_count)
            .map_err(|_| PlanetTerrainRhiError::AllocationFailed)?;
        old_index = 0;
        operation_index = 0;
        let mut final_staged = Vec::new();
        final_staged
            .try_reserve_exact(operations.len())
            .map_err(|_| PlanetTerrainRhiError::AllocationFailed)?;
        while old_index < self.tiles.len() || operation_index < operations.len() {
            let old_key = self.tiles.get(old_index).map(|tile| tile.key);
            let op_key = operations.get(operation_index).map(|(key, _)| *key);
            match (old_key, op_key) {
                (Some(old), Some(op)) if old < op => {
                    next.push(self.tiles[old_index].clone_handle());
                    old_index += 1;
                }
                (_, Some(op)) if old_key.is_none_or(|old| op <= old) => {
                    if let Some(staged) = operations[operation_index].1.take() {
                        final_staged.push(staged);
                    }
                    if old_key == Some(op) {
                        old_index += 1;
                    }
                    operation_index += 1;
                }
                (Some(_), None) => {
                    next.push(self.tiles[old_index].clone_handle());
                    old_index += 1;
                }
                (None, Some(_)) | (Some(_), Some(_)) => {
                    return Err(PlanetTerrainRhiError::ConflictingDelta);
                }
                (None, None) => break,
            }
        }
        let replacements = self.upload_all(final_staged)?;
        // Operations are sorted and old cache entries are sorted; merge while
        // preserving ordering and reuse unchanged native buffer handles.
        let mut merged = Vec::new();
        merged
            .try_reserve_exact(final_count)
            .map_err(|_| PlanetTerrainRhiError::AllocationFailed)?;
        let mut old_cursor = 0;
        let mut replacement_cursor = 0;
        while old_cursor < next.len() || replacement_cursor < replacements.len() {
            match (
                next.get(old_cursor).map(|tile| tile.key),
                replacements.get(replacement_cursor).map(|tile| tile.key),
            ) {
                (Some(a), Some(b)) if a < b => {
                    merged.push(next[old_cursor].clone_handle());
                    old_cursor += 1;
                }
                (Some(_), Some(_)) => {
                    merged.push(replacements[replacement_cursor].clone_handle());
                    replacement_cursor += 1;
                }
                (Some(_), None) => {
                    merged.push(next[old_cursor].clone_handle());
                    old_cursor += 1;
                }
                (None, Some(_)) => {
                    merged.push(replacements[replacement_cursor].clone_handle());
                    replacement_cursor += 1;
                }
                (None, None) => break,
            }
        }
        next = merged;
        debug_assert_eq!(next.len(), final_count);
        self.tiles = next;
        self.buffer_bytes = final_bytes;
        self.generation = Some(delta.generation());
        self.state = Some(Arc::clone(delta.target_state_token()));
        Ok(self.stats())
    }

    /// Encodes one tile's indexed draw into a caller-owned render pass.
    pub fn encode_draw<'pass>(
        &'pass self,
        tile: TileKey,
        pass: &mut wgpu::RenderPass<'pass>,
    ) -> Result<(), PlanetTerrainRhiError> {
        let gpu_tile = self
            .tiles
            .binary_search_by_key(&tile, |entry| entry.key)
            .ok()
            .map(|index| &self.tiles[index])
            .ok_or(PlanetTerrainRhiError::InvalidMesh)?;
        pass.set_vertex_buffer(0, gpu_tile.vertex.slice(..));
        pass.set_index_buffer(gpu_tile.index.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..gpu_tile.index_count, 0, 0..1);
        Ok(())
    }

    /// Returns the current committed cache accounting.
    #[must_use]
    pub fn stats(&self) -> PlanetTerrainRhiStats {
        PlanetTerrainRhiStats {
            generation: self.generation.unwrap_or(0),
            tiles: self.tiles.len(),
            buffer_bytes: self.buffer_bytes,
        }
    }

    fn check_device(&self, context: &GpuContext) -> Result<(), PlanetTerrainRhiError> {
        if Arc::ptr_eq(&self.identity, context.device_identity()) {
            Ok(())
        } else {
            Err(PlanetTerrainRhiError::ForeignDevice)
        }
    }

    fn check_budget(
        &self,
        tiles: usize,
        bytes: u64,
        staged: &[StagedTile],
    ) -> Result<(), PlanetTerrainRhiError> {
        if tiles > self.limits.max_tiles
            || tiles
                .checked_mul(2)
                .is_none_or(|count| count > self.limits.max_buffers)
            || bytes > self.limits.max_buffer_bytes
        {
            return Err(PlanetTerrainRhiError::BudgetExceeded { tiles, bytes });
        }
        let limit = self.device.limits().max_buffer_size;
        if let Some(tile) = staged.iter().find(|tile| tile.bytes > limit) {
            return Err(PlanetTerrainRhiError::DeviceBufferLimit {
                bytes: tile.bytes,
                limit,
            });
        }
        Ok(())
    }

    fn check_transaction_budget(
        &self,
        staged_tiles: usize,
        staged_bytes: u64,
    ) -> Result<(), PlanetTerrainRhiError> {
        let peak_bytes = self
            .buffer_bytes
            .checked_add(staged_bytes)
            .ok_or(PlanetTerrainRhiError::SizeOverflow)?;
        let resident_buffers = self
            .tiles
            .len()
            .checked_mul(2)
            .ok_or(PlanetTerrainRhiError::SizeOverflow)?;
        let staged_buffers = staged_tiles
            .checked_mul(2)
            .ok_or(PlanetTerrainRhiError::SizeOverflow)?;
        let peak_buffers = resident_buffers
            .checked_add(staged_buffers)
            .ok_or(PlanetTerrainRhiError::SizeOverflow)?;
        if peak_bytes > self.limits.max_transaction_buffer_bytes
            || peak_buffers > self.limits.max_transaction_buffers
        {
            return Err(PlanetTerrainRhiError::TransactionBudgetExceeded {
                resident_bytes: self.buffer_bytes,
                staged_bytes,
                byte_limit: self.limits.max_transaction_buffer_bytes,
                resident_buffers,
                staged_buffers,
                buffer_limit: self.limits.max_transaction_buffers,
            });
        }
        Ok(())
    }

    fn upload_all(&self, staged: Vec<StagedTile>) -> Result<Vec<GpuTile>, PlanetTerrainRhiError> {
        let mut next = Vec::new();
        next.try_reserve_exact(staged.len())
            .map_err(|_| PlanetTerrainRhiError::AllocationFailed)?;
        for tile in staged {
            let vertex = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("planet terrain packed vertex ABI"),
                    contents: bytemuck::cast_slice(&tile.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let index = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("planet terrain u32 indices"),
                    contents: bytemuck::cast_slice(&tile.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
            next.push(GpuTile {
                key: tile.key,
                vertex,
                index,
                index_count: tile.index_count,
                bytes: tile.bytes,
            });
        }
        Ok(next)
    }
}

impl GpuTile {
    fn clone_handle(&self) -> Self {
        Self {
            key: self.key,
            vertex: self.vertex.clone(),
            index: self.index.clone(),
            index_count: self.index_count,
            bytes: self.bytes,
        }
    }
}

fn stage_mesh(
    key: TileKey,
    mesh: &prism_world_terrain::PlanetHeightfieldMesh,
) -> Result<StagedTile, PlanetTerrainRhiError> {
    if mesh.tile() != key
        || mesh.positions().is_empty()
        || mesh.positions().len() != mesh.normals().len()
        || mesh.positions().len() != mesh.uvs().len()
        || mesh.indices().is_empty()
        || !mesh.indices().len().is_multiple_of(3)
        || mesh.positions().len() > u32::MAX as usize
        || mesh.indices().len() > u32::MAX as usize
    {
        return Err(PlanetTerrainRhiError::InvalidMesh);
    }
    if mesh
        .positions()
        .iter()
        .flatten()
        .chain(mesh.normals().iter().flatten())
        .chain(mesh.uvs().iter().flatten())
        .any(|value| !value.is_finite())
        || mesh
            .indices()
            .iter()
            .any(|&index| index as usize >= mesh.positions().len())
    {
        return Err(PlanetTerrainRhiError::InvalidMesh);
    }
    let index_count =
        u32::try_from(mesh.indices().len()).map_err(|_| PlanetTerrainRhiError::SizeOverflow)?;
    let vertex_bytes = (mesh.positions().len() as u64)
        .checked_mul(PLANET_TERRAIN_VERTEX_STRIDE)
        .ok_or(PlanetTerrainRhiError::SizeOverflow)?;
    let index_bytes = (mesh.indices().len() as u64)
        .checked_mul(4)
        .ok_or(PlanetTerrainRhiError::SizeOverflow)?;
    let bytes = vertex_bytes
        .checked_add(index_bytes)
        .ok_or(PlanetTerrainRhiError::SizeOverflow)?;
    let mut vertices = Vec::new();
    vertices
        .try_reserve_exact(mesh.positions().len())
        .map_err(|_| PlanetTerrainRhiError::AllocationFailed)?;
    vertices.extend(
        mesh.positions()
            .iter()
            .zip(mesh.normals())
            .zip(mesh.uvs())
            .map(|((&position, &normal), &uv)| GpuPlanetTerrainVertex {
                position,
                normal,
                uv,
            }),
    );
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(mesh.indices().len())
        .map_err(|_| PlanetTerrainRhiError::AllocationFailed)?;
    indices.extend_from_slice(mesh.indices());
    Ok(StagedTile {
        key,
        vertices,
        indices,
        index_count,
        bytes,
    })
}

fn total_bytes(staged: &[StagedTile]) -> Result<u64, PlanetTerrainRhiError> {
    staged.iter().try_fold(0_u64, |total, tile| {
        total
            .checked_add(tile.bytes)
            .ok_or(PlanetTerrainRhiError::SizeOverflow)
    })
}

fn validate_delta_keys(
    changed: &[TileKey],
    removed: &[TileKey],
    cached: &[TileKey],
) -> Result<(), PlanetTerrainRhiError> {
    for keys in [changed, removed] {
        if keys.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PlanetTerrainRhiError::ConflictingDelta);
        }
    }
    if changed.iter().any(|key| removed.binary_search(key).is_ok()) {
        return Err(PlanetTerrainRhiError::ConflictingDelta);
    }
    if removed.iter().any(|key| cached.binary_search(key).is_err()) {
        return Err(PlanetTerrainRhiError::UnknownRemoval);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_world_terrain::CubeFace;

    #[test]
    fn delta_key_validation_rejects_duplicate_conflicts_and_unknown_removal() {
        let a = TileKey::new(CubeFace::PositiveZ, 2, 0, 0).unwrap();
        let b = a.neighbor(prism_world_terrain::TileEdge::East);
        assert_eq!(
            validate_delta_keys(&[a, a], &[], &[a]),
            Err(PlanetTerrainRhiError::ConflictingDelta)
        );
        assert_eq!(
            validate_delta_keys(&[a], &[a], &[a]),
            Err(PlanetTerrainRhiError::ConflictingDelta)
        );
        assert_eq!(
            validate_delta_keys(&[], &[b], &[a]),
            Err(PlanetTerrainRhiError::UnknownRemoval)
        );
        assert_eq!(validate_delta_keys(&[], &[a], &[a]), Ok(()));
    }
}
