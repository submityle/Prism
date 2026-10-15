//! Executable heightfield baking, atomic local rebakes, and indexed render geometry.
//!
//! Encoded heights are signed i16 values; `height_scale_mm` is millimetres per
//! encoded unit. Quantization rounds ties away from zero and rejects overflow.
//! Geometry is a local XZ patch in metres, not a cube-sphere projection. Consumers
//! place it using the world's high-precision transform boundary.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::{DirtyRegion, HeightSample, HeightSampler, HeightfieldSpec, TileKey};

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

    fn expanded(self, side: u32) -> Self {
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

    /// Owning spatial tile.
    pub const fn tile(&self) -> TileKey {
        self.tile
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
}
