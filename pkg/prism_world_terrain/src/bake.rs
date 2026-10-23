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

fn heightfield_transition_skipped_segments(
    edge: TileEdge,
    coarse_edges: [bool; 4],
) -> (bool, bool) {
    match edge {
        TileEdge::West | TileEdge::East => (coarse_edges[2], coarse_edges[3]),
        TileEdge::South | TileEdge::North => (coarse_edges[0], coarse_edges[1]),
    }
}

fn heightfield_offset_grid(
    value: u32,
    offset: i64,
    limit: u32,
) -> Result<u32, BakeError<core::convert::Infallible>> {
    i64::from(value)
        .checked_add(offset)
        .and_then(|coordinate| u32::try_from(coordinate).ok())
        .filter(|&coordinate| coordinate <= limit)
        .ok_or(BakeError::InvalidGeometry)
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
    lod_seam_edges: [bool; 4],
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
            && self.lod_seam_edges == other.lod_seam_edges
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
            lod_seam_edges: [false; 4],
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

    #[cfg(feature = "bevy_mesh_adapter")]
    pub(crate) fn into_bevy_parts(self) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>) {
        (self.positions, self.normals, self.uvs, self.indices)
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
        self.lod_seam_edges = [false; 4];
        Ok(())
    }

    /// Replaces edge quads with the validated 2:1 coarse-neighbor strips in a
    /// seam plan. Transition vertices reuse this mesh's grid vertices; no
    /// duplicate vertex buffer is created. The operation stages the complete
    /// index stream and swaps it only after validation and budget admission.
    ///
    /// The plan must target this tile and use `samples_per_edge - 1` segments.
    /// Any subset of the four coarse edges may be applied. Adjacent edges are
    /// joined by deterministic corner fans so their transition strips neither
    /// overlap nor leave a hole. A mesh accepts one active seam plan at a time;
    /// call [`Self::clear_lod_seam_plan`] to restore the base topology before
    /// applying a new LOD configuration.
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
        let corner_fans = [
            coarse_edges[0] && coarse_edges[2],
            coarse_edges[1] && coarse_edges[2],
            coarse_edges[0] && coarse_edges[3],
            coarse_edges[1] && coarse_edges[3],
        ];
        let corner_count = corner_fans.iter().filter(|&&fan| fan).count();
        let replaced_quads = transitions
            .len()
            .checked_mul(segments as usize)
            .and_then(|count| count.checked_sub(corner_count))
            .ok_or(BakeError::SizeOverflow)?;
        let removed_indices = replaced_quads
            .checked_mul(6)
            .ok_or(BakeError::SizeOverflow)?;
        let added_indices = transitions
            .iter()
            .try_fold(0usize, |total, transition| {
                let (skip_start, skip_end) =
                    heightfield_transition_skipped_segments(transition.edge(), coarse_edges);
                if !transition.indices().len().is_multiple_of(9) {
                    return None;
                }
                let skipped_indices =
                    (usize::from(skip_start) + usize::from(skip_end)).checked_mul(9)?;
                total.checked_add(transition.indices().len().checked_sub(skipped_indices)?)
            })
            .and_then(|count| count.checked_add(corner_count.checked_mul(12)?))
            .ok_or(BakeError::InvalidGeometry)?;
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
            let (skip_start, skip_end) =
                heightfield_transition_skipped_segments(transition.edge(), coarse_edges);
            for (segment, segment_indices) in transition.indices().chunks_exact(9).enumerate() {
                if (skip_start && segment == 0)
                    || (skip_end && segment + 1 == transition.indices().len() / 9)
                {
                    continue;
                }
                for triangle in segment_indices.chunks_exact(3) {
                    let mapped = [triangle[0], triangle[1], triangle[2]].map(|index| {
                        let grid = transition.vertices().get(index as usize)?.grid;
                        let vertex = grid[1].checked_mul(side)?.checked_add(grid[0])?;
                        ((vertex as usize) < vertex_count).then_some(vertex)
                    });
                    let [Some(a), Some(b), Some(c)] = mapped else {
                        return Err(BakeError::InvalidGeometry);
                    };
                    let point = |vertex: u32| [i64::from(vertex % side), i64::from(vertex / side)];
                    let [pa, pb, pc] = [a, b, c].map(point);
                    let ab = [pb[0] - pa[0], pb[1] - pa[1]];
                    let ac = [pc[0] - pa[0], pc[1] - pa[1]];
                    let area = ab[0] * ac[1] - ab[1] * ac[0];
                    if area == 0 {
                        return Err(BakeError::InvalidGeometry);
                    }
                    if area < 0 {
                        next_indices.extend_from_slice(&[a, b, c]);
                    } else {
                        next_indices.extend_from_slice(&[a, c, b]);
                    }
                }
            }
        }
        let corner_coordinates = [
            (0, 0, 1_i64, 1_i64),
            (segments, 0, -1, 1),
            (0, segments, 1, -1),
            (segments, segments, -1, -1),
        ];
        for (corner_index, &(x, z, dx, dz)) in corner_coordinates.iter().enumerate() {
            if !corner_fans[corner_index] {
                continue;
            }
            let corner = [x, z];
            let top_outer = [heightfield_offset_grid(x, dx * 2, segments)?, z];
            let top_inner_far = [
                heightfield_offset_grid(x, dx * 2, segments)?,
                heightfield_offset_grid(z, dz, segments)?,
            ];
            let top_inner_near = [
                heightfield_offset_grid(x, dx, segments)?,
                heightfield_offset_grid(z, dz, segments)?,
            ];
            let side_inner_far = [
                heightfield_offset_grid(x, dx, segments)?,
                heightfield_offset_grid(z, dz * 2, segments)?,
            ];
            let side_outer = [x, heightfield_offset_grid(z, dz * 2, segments)?];
            let patch = [
                [corner, top_outer, top_inner_far],
                [corner, top_inner_far, top_inner_near],
                [corner, top_inner_near, side_inner_far],
                [corner, side_inner_far, side_outer],
            ];
            for triangle in patch {
                let vertices = triangle.map(|[grid_x, grid_z]| {
                    grid_z
                        .checked_mul(side)
                        .and_then(|row| row.checked_add(grid_x))
                });
                let [Some(a), Some(b), Some(c)] = vertices else {
                    return Err(BakeError::SizeOverflow);
                };
                if [a, b, c]
                    .iter()
                    .any(|&vertex| vertex as usize >= vertex_count)
                {
                    return Err(BakeError::InvalidGeometry);
                }
                let point = |vertex: u32| [i64::from(vertex % side), i64::from(vertex / side)];
                let [pa, pb, pc] = [a, b, c].map(point);
                let ab = [pb[0] - pa[0], pb[1] - pa[1]];
                let ac = [pc[0] - pa[0], pc[1] - pa[1]];
                let area = ab[0] * ac[1] - ab[1] * ac[0];
                if area == 0 {
                    return Err(BakeError::InvalidGeometry);
                }
                if area < 0 {
                    next_indices.extend_from_slice(&[a, b, c]);
                } else {
                    next_indices.extend_from_slice(&[a, c, b]);
                }
            }
        }
        if next_indices.len() != next_len {
            return Err(BakeError::InvalidGeometry);
        }
        self.indices = next_indices;
        self.lod_seams_applied = true;
        self.lod_seam_edges = coarse_edges;
        Ok(())
    }

    /// Applies a continuous 2:1 LOD morph to the fine-only boundary samples.
    ///
    /// `alpha` is evaluated from the current screen-space LOD decision: zero
    /// restores the baked fine surface and one places every omitted odd edge
    /// sample on its containing coarse edge segment. The operation reuses mesh
    /// storage, validates the field/plan/topology before mutation, and updates
    /// only boundary positions plus the one-sample normal ring. Call again
    /// after a field rebake to restore the current morph from the new field.
    pub fn apply_lod_morph(
        &mut self,
        field: &HeightfieldTile,
        plan: &crate::HeightfieldLodSeamPlan,
        alpha: f32,
    ) -> Result<(), BakeError<core::convert::Infallible>> {
        if !alpha.is_finite()
            || !(0.0..=1.0).contains(&alpha)
            || !self.lod_seams_applied
            || self.tile != plan.tile()
            || self.tile != field.tile
            || self.cache_id != field.cache_id
            || self.revision != field.revision
            || self.spec != field.spec
            || plan.segments() + 1 != self.spec.samples_per_edge()
        {
            return Err(BakeError::InvalidGeometry);
        }
        let mut plan_edges = [false; 4];
        for transition in plan.transitions() {
            let edge_index = match transition.edge() {
                TileEdge::West => 0,
                TileEdge::East => 1,
                TileEdge::South => 2,
                TileEdge::North => 3,
            };
            if plan_edges[edge_index] {
                return Err(BakeError::InvalidGeometry);
            }
            plan_edges[edge_index] = true;
        }
        if plan_edges != self.lod_seam_edges {
            return Err(BakeError::InvalidGeometry);
        }

        // Preflight all values first so the following writes cannot fail.
        for transition in plan.transitions() {
            for morph in transition.morph_vertices() {
                let source = morph.source;
                let a = morph.target_a;
                let b = morph.target_b;
                if [source, a, b].into_iter().any(|point| {
                    point[0] >= self.spec.samples_per_edge()
                        || point[1] >= self.spec.samples_per_edge()
                }) {
                    return Err(BakeError::InvalidGeometry);
                }
                let source_height = f64::from(height_metres(field, source[0], source[1]));
                let target_a = f64::from(height_metres(field, a[0], a[1]));
                let target_b = f64::from(height_metres(field, b[0], b[1]));
                let height = (source_height
                    + (0.5 * (target_a + target_b) - source_height) * f64::from(alpha))
                    as f32;
                if !height.is_finite() {
                    return Err(BakeError::InvalidGeometry);
                }
            }
        }

        let side = self.spec.samples_per_edge();
        for transition in plan.transitions() {
            for morph in transition.morph_vertices() {
                let [x, z] = morph.source;
                let source_height = f64::from(height_metres(field, x, z));
                let target_a =
                    f64::from(height_metres(field, morph.target_a[0], morph.target_a[1]));
                let target_b =
                    f64::from(height_metres(field, morph.target_b[0], morph.target_b[1]));
                let height = source_height
                    + (0.5 * (target_a + target_b) - source_height) * f64::from(alpha);
                self.positions[index(side, x, z)][1] = height as f32;
            }
        }
        for transition in plan.transitions() {
            for morph in transition.morph_vertices() {
                let [x, z] = morph.source;
                let (min_x, max_x, min_z, max_z) = match transition.edge() {
                    TileEdge::West => (0, 1, z.saturating_sub(1), (z + 1).min(side - 1)),
                    TileEdge::East => (
                        side - 2,
                        side - 1,
                        z.saturating_sub(1),
                        (z + 1).min(side - 1),
                    ),
                    TileEdge::South => (x.saturating_sub(1), (x + 1).min(side - 1), 0, 1),
                    TileEdge::North => (
                        x.saturating_sub(1),
                        (x + 1).min(side - 1),
                        side - 2,
                        side - 1,
                    ),
                };
                for normal_z in min_z..=max_z {
                    for normal_x in min_x..=max_x {
                        let vertex = index(side, normal_x, normal_z);
                        self.normals[vertex] = if alpha == 0.0 {
                            normal(field, self.size_metres, normal_x, normal_z)
                        } else {
                            normal_from_positions(
                                &self.positions,
                                side,
                                self.size_metres,
                                normal_x,
                                normal_z,
                            )
                        };
                    }
                }
            }
        }
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

fn normal_from_positions(
    positions: &[[f32; 3]],
    side: u32,
    size: [f32; 2],
    x: u32,
    z: u32,
) -> [f32; 3] {
    let left = x.saturating_sub(1);
    let right = (x + 1).min(side - 1);
    let back = z.saturating_sub(1);
    let front = (z + 1).min(side - 1);
    let dx = (f64::from(positions[index(side, right, z)][1])
        - f64::from(positions[index(side, left, z)][1]))
        / (f64::from(size[0]) * f64::from(right - left) / f64::from(side - 1));
    let dz = (f64::from(positions[index(side, x, front)][1])
        - f64::from(positions[index(side, x, back)][1]))
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
    use core::convert::Infallible;

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
    use crate::{HeightfieldLodSeamPlan, LodSeamLimits};
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
        let coarse = tile.coarser_neighbor(TileEdge::East).unwrap();
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
        let mut changed_field = field_at(tile);
        let mut mesh = HeightfieldMesh::build(&changed_field, [8.0, 8.0], LIMITS).unwrap();
        let original_indices = mesh.indices().len();
        mesh.apply_lod_seam_plan(&plan, 16_384).unwrap();
        assert_eq!(mesh.indices().len(), original_indices - 8 * 6 + 4 * 9);
        assert!(mesh.indices().iter().all(|&index| (index as usize) < 81));
        assert_eq!(
            mesh.apply_lod_seam_plan(&plan, 16_384),
            Err(BakeError::InvalidGeometry)
        );
        let south_coarse = tile.coarser_neighbor(TileEdge::South).unwrap();
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
        let changed = changed_field
            .rebake(
                DirtyRegion::new(tile, 30_000, 30_000, 34_000, 34_000).unwrap(),
                &plane(2_000),
                LIMITS,
            )
            .unwrap();
        mesh.update(&changed_field, changed).unwrap();
        let mut rebuilt = HeightfieldMesh::build(&changed_field, [8.0, 8.0], LIMITS).unwrap();
        rebuilt.apply_lod_seam_plan(&next_plan, 16_384).unwrap();
        assert_eq!(mesh.positions(), rebuilt.positions());
        assert_eq!(mesh.normals(), rebuilt.normals());
        assert_eq!(mesh.indices(), rebuilt.indices());
    }

    #[test]
    fn lod_seam_budget_failure_preserves_the_original_mesh() {
        use crate::{CubeFace, HeightfieldLodSeamPlan, LodSeamLimits, TileEdge};

        let tile = TileKey::new(CubeFace::PositiveZ, 3, 3, 2).unwrap();
        let coarse = tile.coarser_neighbor(TileEdge::East).unwrap();
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
    fn lod_seam_corner_fans_cover_adjacent_coarse_edges_without_cracks() {
        use crate::{
            CubeFace, HeightfieldLodSeamPlan, LodSeamCoverage, LodSeamLimits, LodSeamNeighbor,
            TileEdge,
        };

        let tile = TileKey::new(CubeFace::PositiveZ, 3, 3, 2).unwrap();
        let seam_limits = LodSeamLimits {
            max_segments: 8,
            max_vertices: 128,
            max_indices: 1024,
            max_bytes: 16_384,
        };
        let corner_pairs = [
            (TileEdge::West, TileEdge::South),
            (TileEdge::East, TileEdge::South),
            (TileEdge::West, TileEdge::North),
            (TileEdge::East, TileEdge::North),
        ];
        let mut configurations = Vec::new();
        for (first, second) in corner_pairs {
            configurations.push(vec![first, second]);
        }
        configurations.push(TileEdge::ALL.to_vec());

        for edges in configurations {
            let neighbors: Vec<_> = edges
                .iter()
                .map(|&edge| {
                    LodSeamNeighbor::with_boundary(
                        edge,
                        tile.coarser_neighbor(edge).unwrap(),
                        8,
                        LodSeamCoverage::full(8),
                    )
                })
                .collect();
            let plan =
                HeightfieldLodSeamPlan::build_strict(tile, 8, &neighbors, seam_limits).unwrap();
            let mut mesh = HeightfieldMesh::build(&field_at(tile), [8.0, 8.0], LIMITS).unwrap();
            let original = mesh.clone();
            assert_eq!(
                mesh.apply_lod_seam_plan(&plan, 1),
                Err(BakeError::BudgetExceeded)
            );
            assert_eq!(
                mesh.indices(),
                original.indices(),
                "budget mutation for {edges:?}"
            );
            mesh.apply_lod_seam_plan(&plan, 16_384).unwrap();

            let mut signed_area = 0_i64;
            let mut edge_counts = Vec::new();
            for triangle in mesh.indices().chunks_exact(3) {
                let points = [triangle[0], triangle[1], triangle[2]].map(|index| {
                    let index = i64::from(index);
                    [index % 9, index / 9]
                });
                let ab = [points[1][0] - points[0][0], points[1][1] - points[0][1]];
                let ac = [points[2][0] - points[0][0], points[2][1] - points[0][1]];
                let area = ab[0] * ac[1] - ab[1] * ac[0];
                assert_ne!(area, 0, "degenerate triangle for {edges:?}");
                assert!(
                    area < 0,
                    "heightfield triangle winding flipped for {edges:?}: {triangle:?} => {points:?} area={area}"
                );
                signed_area += area;
                for (start, end) in [
                    (triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[2], triangle[0]),
                ] {
                    edge_counts.push((start.min(end), start.max(end)));
                }
            }
            assert_eq!(signed_area, -2 * 8 * 8, "hole/overlap for {edges:?}");
            edge_counts.sort_unstable();
            let mut boundary_edges = 0;
            for group in edge_counts.chunk_by(|left, right| left == right) {
                assert!(group.len() <= 2, "non-manifold edge for {edges:?}");
                boundary_edges += usize::from(group.len() == 1);
            }
            assert_eq!(
                boundary_edges,
                4 * 8 - edges.len() * 4,
                "unexpected boundary segmentation for {edges:?}"
            );
        }
    }

    #[test]
    fn lod_seam_morph_moves_only_odd_boundary_samples_and_restores_fine_surface() {
        struct Curved;
        impl HeightSampler for Curved {
            type Error = Infallible;

            fn sample_height(
                &self,
                sample: HeightSample,
            ) -> Result<HeightSampleValue, Self::Error> {
                let u = i64::from(sample.u_q16());
                Ok(HeightSampleValue::from_millimetres((u * u) / 32_768))
            }
        }

        let tile = TileKey::new(CubeFace::PositiveZ, 3, 3, 2).unwrap();
        let field =
            HeightfieldTile::bake(tile, HeightfieldSpec::new(9, 10).unwrap(), &Curved, LIMITS)
                .unwrap();
        let coarse = tile.coarser_neighbor(TileEdge::East).unwrap();
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
        let mut mesh = HeightfieldMesh::build(&field, [8.0, 8.0], LIMITS).unwrap();
        let base = mesh.clone();
        mesh.apply_lod_seam_plan(&plan, 16_384).unwrap();
        mesh.apply_lod_morph(&field, &plan, 0.0).unwrap();
        assert_eq!(mesh.positions(), base.positions());
        assert_eq!(mesh.normals(), base.normals());

        mesh.apply_lod_morph(&field, &plan, 1.0).unwrap();
        for morph in plan.transitions()[0].morph_vertices() {
            let source = mesh.positions()[index(9, morph.source[0], morph.source[1])][1];
            let target_a = height_metres(&field, morph.target_a[0], morph.target_a[1]);
            let target_b = height_metres(&field, morph.target_b[0], morph.target_b[1]);
            assert_eq!(source, 0.5 * (target_a + target_b));
        }
        for z in 0..9 {
            for x in 0..8 {
                if x != 8 {
                    assert_eq!(
                        mesh.positions()[index(9, x, z)],
                        base.positions()[index(9, x, z)]
                    );
                }
            }
        }
        mesh.apply_lod_morph(&field, &plan, 0.0).unwrap();
        assert_eq!(mesh.positions(), base.positions());
        assert_eq!(mesh.normals(), base.normals());
        assert_eq!(
            mesh.apply_lod_morph(&field, &plan, f32::NAN),
            Err(BakeError::InvalidGeometry)
        );
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
