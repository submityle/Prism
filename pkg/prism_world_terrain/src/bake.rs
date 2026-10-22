//! Executable heightfield baking, atomic local rebakes, and indexed render geometry.
//!
//! Encoded heights are signed i16 values; `height_scale_mm` is millimetres per
//! encoded unit. Quantization rounds ties away from zero and rejects overflow.
//! Geometry is a local XZ patch in metres, not a cube-sphere projection. Consumers
//! place it using the world's high-precision transform boundary.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::{DirtyRegion, HeightSample, HeightSampler, HeightfieldSpec, TileEdge, TileKey};

/// Hard limits checked before sampling or allocating derived payloads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BakeLimits {
    /// Maximum sampled height count, or mesh vertex count.
    pub max_samples: usize,
    /// Maximum output payload bytes, excluding container headers.
    ///
    /// A local rebake stages two bytes per affected sample before committing;
    /// its scratch payload is bounded by this same limit. Existing tile/mesh
    /// payloads are retained during the operation.
    pub max_bytes: usize,
}

/// Baking or update failure. Failed operations do not publish partial results.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BakeError<E> {
    /// A checked count, index, or generation calculation overflowed.
    SizeOverflow,
    /// The supplied admission limit would be exceeded.
    BudgetExceeded,
    /// The allocator refused a bounded allocation.
    AllocationFailed,
    /// The source sampler failed.
    Sample(E),
    /// A height does not fit the representation's signed encoding.
    HeightOutOfRange,
    /// A dirty rectangle belongs to a different tile.
    TileMismatch,
    /// A mesh missed an update, or a change belongs to another representation.
    StaleChange,
    /// The process-wide disposable cache identity space is exhausted.
    CacheIdExhausted,
    /// Patch size is non-positive/non-finite, or geometry cannot be represented.
    InvalidGeometry,
}

/// Failure while restoring an immutable heightfield from validated encoded samples.
///
/// Import allocates a fresh exact logical payload instead of retaining caller-owned
/// spare capacity. This keeps the retained sample storage subject to [`BakeLimits`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeightfieldImportError {
    /// The square sample count or byte count cannot be represented by `usize`.
    SizeOverflow,
    /// The supplied sample count does not match `samples_per_edge squared`.
    SampleCountMismatch {
        /// Required number of samples.
        expected: usize,
        /// Supplied number of samples.
        actual: usize,
    },
    /// The sample count or retained logical payload exceeds the configured limits.
    BudgetExceeded,
    /// The allocator refused the bounded sample allocation.
    AllocationFailed,
    /// The process-wide disposable cache identity space is exhausted.
    CacheIdExhausted,
}

/// Inclusive grid coordinates touched by a bake, with X as the fast axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampleBounds {
    min: [u32; 2],
    max: [u32; 2],
}

impl SampleBounds {
    /// Inclusive lower grid coordinate.
    pub const fn min(self) -> [u32; 2] {
        self.min
    }

    /// Inclusive upper grid coordinate.
    pub const fn max(self) -> [u32; 2] {
        self.max
    }

    fn count<E>(self) -> Result<usize, BakeError<E>> {
        usize::try_from(
            u64::from(self.max[0] - self.min[0] + 1) * u64::from(self.max[1] - self.min[1] + 1),
        )
        .map_err(|_| BakeError::SizeOverflow)
    }

    pub(crate) fn expanded(self, side: u32) -> Self {
        Self {
            min: [self.min[0].saturating_sub(1), self.min[1].saturating_sub(1)],
            max: [
                (self.max[0] + 1).min(side - 1),
                (self.max[1] + 1).min(side - 1),
            ],
        }
    }
}

/// An ordered change token produced only after a successful tile rebake.
///
/// Revisions are local to one stable cache owner for a tile/specification. Do not
/// mix independently baked caches or their tokens: rebuilding the owning cache
/// also requires rebuilding its mesh. Tokens are not global content identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GridChange {
    cache_id: u64,
    tile: TileKey,
    spec: HeightfieldSpec,
    previous_revision: u64,
    revision: u64,
    bounds: SampleBounds,
}

impl GridChange {
    /// Grid samples resampled in the transaction.
    pub const fn bounds(self) -> SampleBounds {
        self.bounds
    }

    /// Revision after the change.
    pub const fn revision(self) -> u64 {
        self.revision
    }

    pub(crate) fn matches(
        self,
        field: &HeightfieldTile,
        cache_id: u64,
        current_revision: u64,
    ) -> bool {
        self.cache_id == field.cache_id
            && self.cache_id == cache_id
            && self.tile == field.tile
            && self.spec == field.spec
            && self.previous_revision == current_revision
            && self.revision == field.revision
    }
}

/// Disposable baked height data; authored source remains in the scene/provider.
#[derive(Clone, Debug)]
pub struct HeightfieldTile {
    cache_id: u64,
    tile: TileKey,
    spec: HeightfieldSpec,
    revision: u64,
    samples: Vec<i16>,
}

impl PartialEq for HeightfieldTile {
    fn eq(&self, other: &Self) -> bool {
        self.tile == other.tile
            && self.spec == other.spec
            && self.revision == other.revision
            && self.samples == other.samples
    }
}

impl Eq for HeightfieldTile {}

impl HeightfieldTile {
    /// Bakes the complete tile after validating both count and payload limits.
    pub fn bake<S: HeightSampler>(
        tile: TileKey,
        spec: HeightfieldSpec,
        sampler: &S,
        limits: BakeLimits,
    ) -> Result<Self, BakeError<S::Error>> {
        let side = spec.samples_per_edge();
        let bounds = SampleBounds {
            min: [0, 0],
            max: [side - 1, side - 1],
        };
        let samples = sample_region(tile, spec, bounds, sampler, limits)?;
        Ok(Self {
            cache_id: next_cache_id().ok_or(BakeError::CacheIdExhausted)?,
            tile,
            spec,
            revision: 0,
            samples,
        })
    }

    /// Restores a complete tile from already encoded signed samples.
    ///
    /// The iterator must contain exactly `samples_per_edge squared` values. Count
    /// and byte budgets are checked before allocation, allocation is fallible, and
    /// a fresh process-local cache identity is published only after the complete
    /// sample payload has been staged. The persisted revision is retained so a
    /// reloaded tile can participate in monotonic downstream publication.
    pub fn from_encoded_samples<I>(
        tile: TileKey,
        spec: HeightfieldSpec,
        revision: u64,
        samples: I,
        limits: BakeLimits,
    ) -> Result<Self, HeightfieldImportError>
    where
        I: IntoIterator<Item = i16>,
        I::IntoIter: ExactSizeIterator,
    {
        let side = usize::try_from(spec.samples_per_edge())
            .map_err(|_| HeightfieldImportError::SizeOverflow)?;
        let expected = side
            .checked_mul(side)
            .ok_or(HeightfieldImportError::SizeOverflow)?;
        let bytes = expected
            .checked_mul(core::mem::size_of::<i16>())
            .ok_or(HeightfieldImportError::SizeOverflow)?;
        let samples = samples.into_iter();
        let actual = samples.len();
        if actual != expected {
            return Err(HeightfieldImportError::SampleCountMismatch { expected, actual });
        }
        if expected > limits.max_samples || bytes > limits.max_bytes {
            return Err(HeightfieldImportError::BudgetExceeded);
        }
        let mut restored = Vec::new();
        restored
            .try_reserve_exact(expected)
            .map_err(|_| HeightfieldImportError::AllocationFailed)?;
        let mut samples = samples;
        for _ in 0..expected {
            let Some(sample) = samples.next() else {
                return Err(HeightfieldImportError::SampleCountMismatch {
                    expected,
                    actual: restored.len(),
                });
            };
            restored.push(sample);
        }
        if samples.next().is_some() {
            return Err(HeightfieldImportError::SampleCountMismatch {
                expected,
                actual: expected.saturating_add(1),
            });
        }
        let cache_id = next_cache_id().ok_or(HeightfieldImportError::CacheIdExhausted)?;
        Ok(Self {
            cache_id,
            tile,
            spec,
            revision,
            samples: restored,
        })
    }

    /// Owning spatial tile.
    pub const fn tile(&self) -> TileKey {
        self.tile
    }

    pub(crate) const fn cache_id(&self) -> u64 {
        self.cache_id
    }

    /// Validated encoded-height contract.
    pub const fn spec(&self) -> HeightfieldSpec {
        self.spec
    }

    /// Monotonic local cache revision.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Row-major signed sample payload.
    pub fn samples(&self) -> &[i16] {
        &self.samples
    }

    /// Exact decoded height in millimetres without integer multiplication overflow.
    pub fn height_mm(&self, coordinate: [u32; 2]) -> Option<i128> {
        let side = self.spec.samples_per_edge();
        if coordinate[0] >= side || coordinate[1] >= side {
            return None;
        }
        Some(
            i128::from(self.samples[index(side, coordinate[0], coordinate[1])])
                * i128::from(self.spec.height_scale_mm()),
        )
    }

    /// Rebuilds only samples in a conservative grid cover of `region`.
    ///
    /// Floor/ceil cover includes boundary samples even for sub-grid dirty areas.
    /// Sampling and encoding finish in scratch storage before mutating the cache.
    pub fn rebake<S: HeightSampler>(
        &mut self,
        region: DirtyRegion,
        sampler: &S,
        limits: BakeLimits,
    ) -> Result<GridChange, BakeError<S::Error>> {
        if region.tile() != self.tile {
            return Err(BakeError::TileMismatch);
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(BakeError::SizeOverflow)?;
        let side = self.spec.samples_per_edge();
        let map_min = |value| (u64::from(value) * u64::from(side - 1) / 65_535) as u32;
        let map_max = |value| (u64::from(value) * u64::from(side - 1)).div_ceil(65_535) as u32;
        let bounds = SampleBounds {
            min: [map_min(region.min_u_q16()), map_min(region.min_v_q16())],
            max: [map_max(region.max_u_q16()), map_max(region.max_v_q16())],
        };
        let staged = sample_region(self.tile, self.spec, bounds, sampler, limits)?;
        let width = (bounds.max[0] - bounds.min[0] + 1) as usize;
        for (row, z) in (bounds.min[1]..=bounds.max[1]).enumerate() {
            let start = index(side, bounds.min[0], z);
            self.samples[start..start + width]
                .copy_from_slice(&staged[row * width..(row + 1) * width]);
        }
        let change = GridChange {
            cache_id: self.cache_id,
            tile: self.tile,
            spec: self.spec,
            previous_revision: self.revision,
            revision,
            bounds,
        };
        self.revision = revision;
        Ok(change)
    }
}

fn allocate<T, E>(count: usize) -> Result<Vec<T>, BakeError<E>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| BakeError::AllocationFailed)?;
    Ok(result)
}

fn sample_region<S: HeightSampler>(
    tile: TileKey,
    spec: HeightfieldSpec,
    bounds: SampleBounds,
    sampler: &S,
    limits: BakeLimits,
) -> Result<Vec<i16>, BakeError<S::Error>> {
    let count = bounds.count()?;
    let bytes = count.checked_mul(2).ok_or(BakeError::SizeOverflow)?;
    if count > limits.max_samples || bytes > limits.max_bytes {
        return Err(BakeError::BudgetExceeded);
    }
    let mut samples = allocate(count)?;
    let side = spec.samples_per_edge();
    let quantize_coordinate = |value| {
        ((u64::from(value) * 65_535 + u64::from(side - 1) / 2) / u64::from(side - 1)) as u16
    };
    let scale = i128::from(spec.height_scale_mm());
    for z in bounds.min[1]..=bounds.max[1] {
        for x in bounds.min[0]..=bounds.max[0] {
            let height = i128::from(
                sampler
                    .sample_height(HeightSample::new(
                        tile,
                        quantize_coordinate(x),
                        quantize_coordinate(z),
                    ))
                    .map_err(BakeError::Sample)?
                    .height_mm(),
            );
            let rounded = if height < 0 {
                (height - scale / 2) / scale
            } else {
                (height + scale / 2) / scale
            };
            samples.push(i16::try_from(rounded).map_err(|_| BakeError::HeightOutOfRange)?);
        }
    }
    Ok(samples)
}

fn index(side: u32, x: u32, z: u32) -> usize {
    (u64::from(z) * u64::from(side) + u64::from(x)) as usize
}

fn mesh_edges_are_opposite(a: TileEdge, b: TileEdge) -> bool {
    matches!(
        (a, b),
        (TileEdge::West, TileEdge::East)
            | (TileEdge::East, TileEdge::West)
            | (TileEdge::South, TileEdge::North)
            | (TileEdge::North, TileEdge::South)
    )
}

/// Indexed local heightfield geometry with in-place one-ring normal updates.
#[derive(Clone, Debug)]
pub struct HeightfieldMesh {
    cache_id: u64,
    tile: TileKey,
    spec: HeightfieldSpec,
    revision: u64,
    size_metres: [f32; 2],
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
    lod_seams_applied: bool,
}

impl PartialEq for HeightfieldMesh {
    fn eq(&self, other: &Self) -> bool {
        self.tile == other.tile
            && self.spec == other.spec
            && self.revision == other.revision
            && self.size_metres == other.size_metres
            && self.positions == other.positions
            && self.normals == other.normals
            && self.uvs == other.uvs
            && self.indices == other.indices
    }
}

impl HeightfieldMesh {
    /// Builds CCW +Y-facing triangles, UVs, and normalized finite-difference normals.
    ///
    /// Mesh payload is exactly 32 bytes per vertex and 24 per grid quad. Limits
    /// are checked before allocation. Normal math uses f64 intermediates to avoid
    /// squaring large valid slopes in f32.
    pub fn build(
        field: &HeightfieldTile,
        size_metres: [f32; 2],
        limits: BakeLimits,
    ) -> Result<Self, BakeError<core::convert::Infallible>> {
        if size_metres
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(BakeError::InvalidGeometry);
        }
        let side = field.spec.samples_per_edge();
        let vertices = field.samples.len();
        // Indices are u32, even on 64-bit hosts.
        if vertices > u32::MAX as usize {
            return Err(BakeError::SizeOverflow);
        }
        let segments = (side - 1) as usize;
        let indices = segments
            .checked_mul(segments)
            .and_then(|n| n.checked_mul(6))
            .ok_or(BakeError::SizeOverflow)?;
        let bytes = vertices
            .checked_mul(32)
            .and_then(|n| indices.checked_mul(4).and_then(|i| n.checked_add(i)))
            .ok_or(BakeError::SizeOverflow)?;
        if vertices > limits.max_samples || bytes > limits.max_bytes {
            return Err(BakeError::BudgetExceeded);
        }
        let mut result = Self {
            cache_id: field.cache_id,
            tile: field.tile,
            spec: field.spec,
            revision: field.revision,
            size_metres,
            positions: allocate(vertices)?,
            normals: allocate(vertices)?,
            uvs: allocate(vertices)?,
            indices: allocate(indices)?,
            lod_seams_applied: false,
        };
        for z in 0..side {
            for x in 0..side {
                let u = x as f32 / (side - 1) as f32;
                let v = z as f32 / (side - 1) as f32;
                result.positions.push([
                    (u - 0.5) * size_metres[0],
                    height_metres(field, x, z),
                    (v - 0.5) * size_metres[1],
                ]);
                result.normals.push(normal(field, size_metres, x, z));
                result.uvs.push([u, v]);
            }
        }
        for z in 0..side - 1 {
            for x in 0..side - 1 {
                let a = z * side + x;
                let b = a + 1;
                let c = a + side;
                result.indices.extend_from_slice(&[a, c, b, b, c, c + 1]);
            }
        }
        Ok(result)
    }

    /// Applies the next committed tile change without reallocating mesh storage.
    ///
    /// Returns the conservative rectangle whose vertex data changed: heights
    /// only depend on the patch; normals additionally depend on one grid ring.
    /// Change tokens must be consumed in order and before the next tile rebake.
    pub fn update(
        &mut self,
        field: &HeightfieldTile,
        change: GridChange,
    ) -> Result<SampleBounds, BakeError<core::convert::Infallible>> {
        if self.cache_id != field.cache_id
            || self.tile != field.tile
            || change.tile != field.tile
            || change.cache_id != field.cache_id
            || self.spec != field.spec
            || change.spec != field.spec
            || self.revision != change.previous_revision
            || field.revision != change.revision
        {
            return Err(BakeError::StaleChange);
        }
        let side = self.spec.samples_per_edge();
        for z in change.bounds.min[1]..=change.bounds.max[1] {
            for x in change.bounds.min[0]..=change.bounds.max[0] {
                self.positions[index(side, x, z)][1] = height_metres(field, x, z);
            }
        }
        let updated = change.bounds.expanded(side);
        for z in updated.min[1]..=updated.max[1] {
            for x in updated.min[0]..=updated.max[0] {
                self.normals[index(side, x, z)] = normal(field, self.size_metres, x, z);
            }
        }
        self.revision = change.revision;
        Ok(updated)
    }

    /// Packed-position source for renderer/collision adapters, in metres.
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }
    /// Unit normals in the same local space.
    pub fn normals(&self) -> &[[f32; 3]] {
        &self.normals
    }
    /// Tile-normalized texture coordinates.
    pub fn uvs(&self) -> &[[f32; 2]] {
        &self.uvs
    }
    /// CCW triangle indices.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Removes active transition strips and restores the regular grid topology.
    ///
    /// The replacement index stream is fully allocated and budget-checked before
    /// the current topology is changed. Vertex attributes and their allocations
    /// are retained, so callers can apply a new seam plan without rebaking the
    /// heightfield.
    pub fn clear_lod_seam_plan(
        &mut self,
        max_bytes: usize,
    ) -> Result<(), BakeError<core::convert::Infallible>> {
        if !self.lod_seams_applied {
            return Ok(());
        }
        let segments = self.spec.samples_per_edge() - 1;
        let base_indices = usize::try_from(segments)
            .ok()
            .and_then(|n| n.checked_mul(n))
            .and_then(|n| n.checked_mul(6))
            .ok_or(BakeError::SizeOverflow)?;
        let bytes = self
            .positions
            .len()
            .checked_mul(32)
            .and_then(|n| {
                base_indices
                    .checked_mul(core::mem::size_of::<u32>())
                    .and_then(|indices| n.checked_add(indices))
            })
            .ok_or(BakeError::SizeOverflow)?;
        if bytes > max_bytes {
            return Err(BakeError::BudgetExceeded);
        }
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(base_indices)
            .map_err(|_| BakeError::AllocationFailed)?;
        let side = self.spec.samples_per_edge();
        for y in 0..segments {
            for x in 0..segments {
                let a = y * side + x;
                let b = a + 1;
                let c = a + side;
                indices.extend_from_slice(&[a, c, b, b, c, c + 1]);
            }
        }
        self.indices = indices;
        self.lod_seams_applied = false;
        Ok(())
    }

    /// Replaces edge quads with the validated 2:1 coarse-neighbor strips in a
    /// seam plan. Transition vertices reuse this mesh's grid vertices; no
    /// duplicate vertex buffer is created. The operation stages the complete
    /// index stream and swaps it only after validation and budget admission.
    ///
    /// The plan must target this tile and use `samples_per_edge - 1` segments.
    /// At most two coarse edges may be applied, and when there are two they
    /// must be opposite; adjacent coarse edges share a corner quad and require
    /// a corner fan contract that this mesh ABI does not yet define. A mesh
    /// accepts one active seam plan at a time; call [`Self::clear_lod_seam_plan`] to
    /// restore the base topology before applying a new LOD configuration.
    pub fn apply_lod_seam_plan(
        &mut self,
        plan: &crate::HeightfieldLodSeamPlan,
        max_bytes: usize,
    ) -> Result<(), BakeError<core::convert::Infallible>> {
        if self.lod_seams_applied || plan.tile() != self.tile {
            return Err(BakeError::InvalidGeometry);
        }
        let side = self.spec.samples_per_edge();
        let segments = side - 1;
        if plan.segments() != segments {
            return Err(BakeError::InvalidGeometry);
        }
        let transitions = plan.transitions();
        if transitions.is_empty() {
            return Ok(());
        }
        if transitions.len() > 2
            || (transitions.len() == 2
                && !mesh_edges_are_opposite(transitions[0].edge(), transitions[1].edge()))
        {
            return Err(BakeError::InvalidGeometry);
        }

        let mut coarse_edges = [false; 4];
        for transition in transitions {
            let index = match transition.edge() {
                crate::TileEdge::West => 0,
                crate::TileEdge::East => 1,
                crate::TileEdge::South => 2,
                crate::TileEdge::North => 3,
            };
            if coarse_edges[index] {
                return Err(BakeError::InvalidGeometry);
            }
            coarse_edges[index] = true;
        }

        let vertex_count = self.positions.len();
        let base_indices = usize::try_from(segments)
            .ok()
            .and_then(|n| n.checked_mul(n))
            .and_then(|n| n.checked_mul(6))
            .ok_or(BakeError::SizeOverflow)?;
        if self.indices.len() != base_indices {
            return Err(BakeError::InvalidGeometry);
        }
        let replaced_quads = transitions
            .len()
            .checked_mul(segments as usize)
            .ok_or(BakeError::SizeOverflow)?;
        let removed_indices = replaced_quads
            .checked_mul(6)
            .ok_or(BakeError::SizeOverflow)?;
        let added_indices = transitions
            .iter()
            .try_fold(0usize, |total, transition| {
                total.checked_add(transition.indices().len())
            })
            .ok_or(BakeError::SizeOverflow)?;
        let next_len = base_indices
            .checked_sub(removed_indices)
            .and_then(|n| n.checked_add(added_indices))
            .ok_or(BakeError::SizeOverflow)?;
        let bytes = vertex_count
            .checked_mul(32)
            .and_then(|n| {
                next_len
                    .checked_mul(core::mem::size_of::<u32>())
                    .and_then(|i| n.checked_add(i))
            })
            .ok_or(BakeError::SizeOverflow)?;
        if bytes > max_bytes {
            return Err(BakeError::BudgetExceeded);
        }

        let mut next_indices = Vec::new();
        next_indices
            .try_reserve_exact(next_len)
            .map_err(|_| BakeError::AllocationFailed)?;
        for y in 0..segments {
            for x in 0..segments {
                let removed = (coarse_edges[0] && x == 0)
                    || (coarse_edges[1] && x + 1 == segments)
                    || (coarse_edges[2] && y == 0)
                    || (coarse_edges[3] && y + 1 == segments);
                if !removed {
                    let first = ((y * segments + x) * 6) as usize;
                    next_indices.extend_from_slice(&self.indices[first..first + 6]);
                }
            }
        }
        for transition in transitions {
            for &index in transition.indices() {
                let grid = transition.vertices()[index as usize].grid;
                let vertex = grid[1]
                    .checked_mul(side)
                    .and_then(|row| row.checked_add(grid[0]))
                    .ok_or(BakeError::SizeOverflow)?;
                if vertex as usize >= vertex_count {
                    return Err(BakeError::InvalidGeometry);
                }
                next_indices.push(vertex);
            }
        }
        if next_indices.len() != next_len {
            return Err(BakeError::InvalidGeometry);
        }
        self.indices = next_indices;
        self.lod_seams_applied = true;
        Ok(())
    }
    /// Last consumed tile-cache revision.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Consumes the geometry as a real Prism mesh asset, not an opaque proxy.
    #[cfg(feature = "mesh")]
    pub fn into_mesh(self) -> Result<prism_mesh::Mesh, prism_mesh::MeshError> {
        use prism_mesh::{Indices, Mesh, MeshVertexAttribute, VertexAttributeValues};
        use prism_render_driver::VertexFormat;
        let position =
            MeshVertexAttribute::new(prism_mesh::POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0);
        let normal = MeshVertexAttribute::new("Vertex_Normal", VertexFormat::Float32x3, 1);
        let uv = MeshVertexAttribute::new("Vertex_Uv0", VertexFormat::Float32x2, 2);
        let mesh = Mesh::new(prism_render_driver::PrimitiveTopology::TriangleList)
            .with_attribute(position, VertexAttributeValues::Float32x3(self.positions))?
            .with_attribute(normal, VertexAttributeValues::Float32x3(self.normals))?
            .with_attribute(uv, VertexAttributeValues::Float32x2(self.uvs))?
            .indexed(Indices::Uint32(self.indices));
        mesh.validate()?;
        Ok(mesh)
    }
}

fn height_metres(field: &HeightfieldTile, x: u32, z: u32) -> f32 {
    (f64::from(field.samples[index(field.spec.samples_per_edge(), x, z)])
        * field.spec.height_scale_mm() as f64
        * 0.001) as f32
}

static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(1);

fn next_cache_id() -> Option<u64> {
    let mut current = NEXT_CACHE_ID.load(Ordering::Relaxed);
    loop {
        if current == 0 || current == u64::MAX {
            return None;
        }
        match NEXT_CACHE_ID.compare_exchange_weak(
            current,
            current + 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return Some(current),
            Err(observed) => current = observed,
        }
    }
}

#[cfg(test)]
fn next_cache_id_candidate(current: u64) -> Option<u64> {
    (current != 0 && current != u64::MAX).then_some(current)
}

fn normal(field: &HeightfieldTile, size: [f32; 2], x: u32, z: u32) -> [f32; 3] {
    let side = field.spec.samples_per_edge();
    let left = x.saturating_sub(1);
    let right = (x + 1).min(side - 1);
    let back = z.saturating_sub(1);
    let front = (z + 1).min(side - 1);
    let scale_metres = field.spec.height_scale_mm() as f64 * 0.001;
    let dx = (f64::from(field.samples[index(side, right, z)])
        - f64::from(field.samples[index(side, left, z)]))
        * scale_metres
        / (f64::from(size[0]) * f64::from(right - left) / f64::from(side - 1));
    let dz = (f64::from(field.samples[index(side, x, front)])
        - f64::from(field.samples[index(side, x, back)]))
        * scale_metres
        / (f64::from(size[1]) * f64::from(front - back) / f64::from(side - 1));
    let length = libm::sqrt(dx * dx + 1.0 + dz * dz);
    [
        (-dx / length) as f32,
        (1.0 / length) as f32,
        (-dz / length) as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    struct MisreportedSamples {
        samples: alloc::vec::IntoIter<i16>,
        reported_len: usize,
    }

    impl Iterator for MisreportedSamples {
        type Item = i16;

        fn next(&mut self) -> Option<Self::Item> {
            self.samples.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.reported_len, Some(self.reported_len))
        }
    }

    impl ExactSizeIterator for MisreportedSamples {
        fn len(&self) -> usize {
            self.reported_len
        }
    }
    use crate::{CubeFace, HeightSampleValue};
    use core::cell::Cell;

    const LIMITS: BakeLimits = BakeLimits {
        max_samples: 1024,
        max_bytes: 100_000,
    };
    struct Plane {
        calls: Cell<usize>,
        offset: i64,
        fail_at: Option<usize>,
    }
    impl HeightSampler for Plane {
        type Error = &'static str;
        fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            if self.fail_at == Some(call) {
                return Err("sample failure");
            }
            Ok(HeightSampleValue::from_millimetres(
                self.offset + i64::from(sample.u_q16()) * 2,
            ))
        }
    }
    fn plane(offset: i64) -> Plane {
        Plane {
            calls: Cell::new(0),
            offset,
            fail_at: None,
        }
    }
    fn field() -> HeightfieldTile {
        HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveY),
            HeightfieldSpec::new(9, 10).unwrap(),
            &plane(0),
            LIMITS,
        )
        .unwrap()
    }

    fn field_at(tile: TileKey) -> HeightfieldTile {
        HeightfieldTile::bake(
            tile,
            HeightfieldSpec::new(9, 10).unwrap(),
            &plane(0),
            LIMITS,
        )
        .unwrap()
    }

    #[test]
    fn lod_seam_plan_replaces_boundary_quads_transactionally() {
        use crate::{CubeFace, HeightfieldLodSeamPlan, LodSeamLimits, TileEdge};

        let tile = TileKey::new(CubeFace::PositiveZ, 3, 3, 2).unwrap();
        let coarse = tile.neighbor(TileEdge::East).parent().unwrap();
        let plan = HeightfieldLodSeamPlan::build_strict_for_residency(
            tile,
            8,
            &[coarse],
            LodSeamLimits {
                max_segments: 8,
                max_vertices: 128,
                max_indices: 1024,
                max_bytes: 16_384,
            },
        )
        .unwrap();
        assert_eq!(plan.transitions().len(), 1);
        let mut mesh = HeightfieldMesh::build(&field_at(tile), [8.0, 8.0], LIMITS).unwrap();
        let original_indices = mesh.indices().len();
        mesh.apply_lod_seam_plan(&plan, 16_384).unwrap();
        assert_eq!(mesh.indices().len(), original_indices - 8 * 6 + 4 * 9);
        assert!(mesh.indices().iter().all(|&index| (index as usize) < 81));
        assert_eq!(
            mesh.apply_lod_seam_plan(&plan, 16_384),
            Err(BakeError::InvalidGeometry)
        );
        let south_coarse = tile.neighbor(TileEdge::South).parent().unwrap();
        let next_plan = HeightfieldLodSeamPlan::build_strict_for_residency(
            tile,
            8,
            &[south_coarse],
            LodSeamLimits {
                max_segments: 8,
                max_vertices: 128,
                max_indices: 1024,
                max_bytes: 16_384,
            },
        )
        .unwrap();
        assert_eq!(mesh.clear_lod_seam_plan(1), Err(BakeError::BudgetExceeded));
        assert_eq!(mesh.indices().len(), original_indices - 8 * 6 + 4 * 9);
        mesh.clear_lod_seam_plan(16_384).unwrap();
        assert_eq!(mesh.indices().len(), original_indices);
        mesh.apply_lod_seam_plan(&next_plan, 16_384).unwrap();
        assert_eq!(mesh.indices().len(), original_indices - 8 * 6 + 4 * 9);
        assert!(mesh.indices().iter().all(|&index| (index as usize) < 81));
    }

    #[test]
    fn lod_seam_budget_failure_preserves_the_original_mesh() {
        use crate::{CubeFace, HeightfieldLodSeamPlan, LodSeamLimits, TileEdge};

        let tile = TileKey::new(CubeFace::PositiveZ, 3, 3, 2).unwrap();
        let coarse = tile.neighbor(TileEdge::East).parent().unwrap();
        let plan = HeightfieldLodSeamPlan::build_strict_for_residency(
            tile,
            8,
            &[coarse],
            LodSeamLimits {
                max_segments: 8,
                max_vertices: 128,
                max_indices: 1024,
                max_bytes: 16_384,
            },
        )
        .unwrap();
        let mut mesh = HeightfieldMesh::build(&field_at(tile), [8.0, 8.0], LIMITS).unwrap();
        let original = mesh.indices().to_vec();
        assert_eq!(
            mesh.apply_lod_seam_plan(&plan, 1),
            Err(BakeError::BudgetExceeded)
        );
        assert_eq!(mesh.indices(), original);
    }

    #[test]
    fn bake_is_quantized_bounded_and_deterministic() {
        let first = field();
        assert_eq!(first, field());
        assert_eq!(first.samples.len(), 81);
        assert_eq!(first.height_mm([8, 8]), Some(131_070));
        assert_eq!(first.height_mm([9, 0]), None);
        let sampler = plane(0);
        assert_eq!(
            HeightfieldTile::bake(
                first.tile,
                first.spec,
                &sampler,
                BakeLimits {
                    max_samples: 80,
                    ..LIMITS
                }
            ),
            Err(BakeError::BudgetExceeded)
        );
        assert_eq!(sampler.calls.get(), 0);
        assert_eq!(
            HeightfieldTile::bake(first.tile, first.spec, &plane(i64::MAX), LIMITS),
            Err(BakeError::HeightOutOfRange)
        );
    }

    #[test]
    fn failed_rebake_is_atomic_and_local_update_matches_full_rebuild() {
        let mut field = field();
        let old = field.clone();
        let mut mesh = HeightfieldMesh::build(&field, [8.0, 8.0], LIMITS).unwrap();
        let region = DirtyRegion::new(field.tile, 30_000, 30_000, 34_000, 34_000).unwrap();
        let mut bad = plane(1_000);
        bad.fail_at = Some(2);
        assert_eq!(
            field.rebake(region, &bad, LIMITS),
            Err(BakeError::Sample("sample failure"))
        );
        assert_eq!(field, old);
        let sampler = plane(1_000);
        let change = field.rebake(region, &sampler, LIMITS).unwrap();
        assert_eq!(change.bounds.min, [3, 3]);
        assert_eq!(change.bounds.max, [5, 5]);
        assert_eq!(sampler.calls.get(), 9);
        let positions_address = mesh.positions.as_ptr();
        let updated = mesh.update(&field, change).unwrap();
        assert_eq!(updated.min, [2, 2]);
        assert_eq!(updated.max, [6, 6]);
        assert_eq!(positions_address, mesh.positions.as_ptr());
        assert_eq!(
            mesh,
            HeightfieldMesh::build(&field, [8.0, 8.0], LIMITS).unwrap()
        );
        assert_eq!(mesh.update(&field, change), Err(BakeError::StaleChange));
        assert_eq!(field.samples[0], old.samples[0]);
    }

    #[test]
    fn planar_normals_and_winding_are_correct() {
        let field = field();
        let mesh = HeightfieldMesh::build(&field, [131.07, 8.0], LIMITS).unwrap();
        for n in mesh.normals() {
            assert!((n[0] + core::f32::consts::FRAC_1_SQRT_2).abs() < 0.0005);
            assert!((n[1] - core::f32::consts::FRAC_1_SQRT_2).abs() < 0.0005);
            assert_eq!(n[2], 0.0);
        }
        for triangle in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [
                mesh.positions[triangle[0] as usize],
                mesh.positions[triangle[1] as usize],
                mesh.positions[triangle[2] as usize],
            ];
            assert!((b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]) > 0.0);
        }
        assert_eq!(mesh.indices.len(), 8 * 8 * 6);
    }

    #[test]
    fn mesh_limits_and_change_lineage_are_enforced() {
        let mut field = field();
        let mut mesh = HeightfieldMesh::build(&field, [1.0, 1.0], LIMITS).unwrap();
        assert_eq!(
            HeightfieldMesh::build(&field, [f32::NAN, 1.0], LIMITS),
            Err(BakeError::InvalidGeometry)
        );
        let bytes = 81 * 32 + 64 * 24;
        assert!(HeightfieldMesh::build(
            &field,
            [1.0, 1.0],
            BakeLimits {
                max_bytes: bytes,
                ..LIMITS
            }
        )
        .is_ok());
        assert_eq!(
            HeightfieldMesh::build(
                &field,
                [1.0, 1.0],
                BakeLimits {
                    max_bytes: bytes - 1,
                    ..LIMITS
                }
            ),
            Err(BakeError::BudgetExceeded)
        );
        let region = DirtyRegion::new(field.tile, u16::MAX, u16::MAX, u16::MAX, u16::MAX).unwrap();
        let first = field.rebake(region, &plane(0), LIMITS).unwrap();
        let second = field.rebake(region, &plane(0), LIMITS).unwrap();
        assert_eq!(second.bounds.count::<()>().unwrap(), 1);
        assert_eq!(mesh.update(&field, first), Err(BakeError::StaleChange));
        assert_eq!(mesh.update(&field, second), Err(BakeError::StaleChange));
        let mut independent =
            HeightfieldTile::bake(field.tile, field.spec, &plane(10_000), LIMITS).unwrap();
        let independent_change = independent.rebake(region, &plane(10_000), LIMITS).unwrap();
        let mut independent_mesh =
            HeightfieldMesh::build(&independent, [1.0, 1.0], LIMITS).unwrap();
        assert_eq!(
            independent_mesh.update(&field, independent_change),
            Err(BakeError::StaleChange)
        );
        let other = DirtyRegion::new(TileKey::root(CubeFace::NegativeY), 0, 0, 0, 0).unwrap();
        assert_eq!(
            field.rebake(other, &plane(0), LIMITS),
            Err(BakeError::TileMismatch)
        );

        let same_content =
            HeightfieldTile::bake(field.tile, field.spec, &plane(0), LIMITS).unwrap();
        let same_mesh = HeightfieldMesh::build(&same_content, [1.0, 1.0], LIMITS).unwrap();
        assert_eq!(mesh, same_mesh);
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn native_mesh_has_valid_packed_vertices_indices_and_bounds() {
        let mesh = HeightfieldMesh::build(&field(), [8.0, 8.0], LIMITS)
            .unwrap()
            .into_mesh()
            .unwrap();
        mesh.validate().unwrap();
        assert_eq!(mesh.vertex_count(), 81);
        assert_eq!(mesh.vertex_buffer_data().len(), 81 * 32);
        assert_eq!(mesh.indices().unwrap().len(), 384);
        assert!(mesh.bounds().is_some());
    }

    #[test]
    fn cache_id_allocator_fails_closed_at_extremes() {
        assert_eq!(next_cache_id_candidate(1), Some(1));
        assert_eq!(next_cache_id_candidate(u64::MAX - 1), Some(u64::MAX - 1));
        assert_eq!(next_cache_id_candidate(u64::MAX), None);
        assert_eq!(next_cache_id_candidate(0), None);
    }

    #[test]
    fn encoded_sample_import_is_exact_bounded_and_cache_local() {
        let tile = TileKey::root(CubeFace::PositiveZ);
        let spec = HeightfieldSpec::new(3, 25).unwrap();
        let samples = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
        let limits = BakeLimits {
            max_samples: samples.len(),
            max_bytes: samples.len() * core::mem::size_of::<i16>(),
        };
        let first =
            HeightfieldTile::from_encoded_samples(tile, spec, 17, samples.into_iter(), limits)
                .unwrap();
        let second =
            HeightfieldTile::from_encoded_samples(tile, spec, 17, samples.into_iter(), limits)
                .unwrap();
        assert_eq!(first, second);
        assert_ne!(first.cache_id(), second.cache_id());
        assert_eq!(first.revision(), 17);
        assert_eq!(first.samples(), &samples);

        assert_eq!(
            HeightfieldTile::from_encoded_samples(
                tile,
                spec,
                17,
                samples[..8].iter().copied(),
                limits,
            ),
            Err(HeightfieldImportError::SampleCountMismatch {
                expected: 9,
                actual: 8,
            })
        );
        assert_eq!(
            HeightfieldTile::from_encoded_samples(
                tile,
                spec,
                17,
                samples.into_iter(),
                BakeLimits {
                    max_samples: 8,
                    max_bytes: limits.max_bytes,
                },
            ),
            Err(HeightfieldImportError::BudgetExceeded)
        );
        assert_eq!(
            HeightfieldTile::from_encoded_samples(
                tile,
                spec,
                17,
                samples.into_iter(),
                BakeLimits {
                    max_samples: limits.max_samples,
                    max_bytes: limits.max_bytes - 1,
                },
            ),
            Err(HeightfieldImportError::BudgetExceeded)
        );
    }

    #[test]
    fn encoded_sample_import_does_not_trust_misreported_exact_length() {
        let tile = TileKey::root(CubeFace::PositiveZ);
        let spec = HeightfieldSpec::new(2, 25).unwrap();
        let limits = BakeLimits {
            max_samples: 4,
            max_bytes: 8,
        };
        let short = MisreportedSamples {
            samples: vec![1, 2, 3].into_iter(),
            reported_len: 4,
        };
        assert_eq!(
            HeightfieldTile::from_encoded_samples(tile, spec, 3, short, limits),
            Err(HeightfieldImportError::SampleCountMismatch {
                expected: 4,
                actual: 3,
            })
        );

        let long = MisreportedSamples {
            samples: vec![1, 2, 3, 4, 5].into_iter(),
            reported_len: 4,
        };
        assert_eq!(
            HeightfieldTile::from_encoded_samples(tile, spec, 3, long, limits),
            Err(HeightfieldImportError::SampleCountMismatch {
                expected: 4,
                actual: 5,
            })
        );
    }
}
