//! Bounded SDF grid baking with atomic, three-dimensional local resampling.
//!
//! Samples are signed Q16.16 integers: `raw * distance_scale_mm / 65536`
//! millimetres. Negative values are inside, positive values outside, and zero
//! is the surface. Coordinates are integer XYZ grid indices; the source owns
//! their spatial embedding and must supply actual signed distances in the
//! declared scale. Baking stores distances, it does not compute a distance
//! transform or certify the source's Euclidean-distance property. In particular,
//! a vertical heightfield implicit function is not generally a Euclidean SDF.
//!
//! Storage is X-fastest, then Y, then Z, in packed little-endian i32 samples.
//! `SdfSpec::brick_size` remains streaming metadata, not storage padding.
//! [`quantize_sdf_mm`] can encode integer millimetre distances using round to
//! nearest, ties away from zero; sampling otherwise preserves raw bits exactly.

use alloc::vec::Vec;

use crate::{SdfSpec, VoxelComponentFormat, VoxelError, VoxelField, VoxelSample};

/// A source of signed Q16.16 distances with a stable grid and scale contract.
///
/// Implementations must keep their specification and spatial embedding stable
/// during a bake, and during local updates of the same cache. A changed spatial
/// embedding requires a full bake even if the specification is unchanged.
/// Deterministic output requires a deterministic source; calls occur in XYZ
/// row-major order with X fastest, exactly once per requested coordinate.
pub trait SdfField {
    /// Source-specific sampling error.
    type Error;

    /// Dimensions, streaming brick size, and millimetres per whole Q16.16 unit.
    fn spec(&self) -> SdfSpec;

    /// Returns one signed distance encoded in the specification's Q16.16 scale.
    ///
    /// A baker calls this only for coordinates inside `spec().dimensions()`.
    /// No clipping, floating-point quantization, or saturation is performed.
    fn sample_q16(&self, coordinate: [u32; 3]) -> Result<i32, Self::Error>;
}

/// Encodes integer millimetres in the specification's signed Q16.16 scale.
///
/// Computes `distance_mm * 65536 / distance_scale_mm` in i128, rounds to nearest
/// with ties away from zero, and rejects a rounded result outside signed i32.
/// Never saturates. This encodes an existing distance, not an implicit function
/// or density converted into a true distance field.
pub fn quantize_sdf_mm(distance_mm: i64, spec: SdfSpec) -> Result<i32, VolumeError> {
    // i64 distances times 2^16, including a half i64 scale adjustment, fit i128.
    let numerator = i128::from(distance_mm) * 65_536;
    let scale = i128::from(spec.distance_scale_mm());
    let rounded = if numerator < 0 {
        (numerator - scale / 2) / scale
    } else {
        (numerator + scale / 2) / scale
    };
    i32::try_from(rounded).map_err(|_| VolumeError::DistanceOutOfRange)
}

/// Hard per-operation limits, checked before allocating or sampling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VolumeBakeLimits {
    /// Maximum full-grid dimensions along X, Y, and Z, including local updates.
    pub max_dimensions: [u32; 3],
    /// Maximum source sampling calls in this operation, not the full cache size.
    pub max_samples: usize,
    /// Maximum live logical payload bytes owned by this baker.
    ///
    /// A full bake needs its output buffer. A local update needs the existing
    /// full buffer plus a packed temporary buffer for the updated region.
    /// Allocator metadata, capacity rounding, and source-owned data are excluded.
    pub max_bytes: usize,
}

/// Exact sampling and logical payload costs before an operation begins.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VolumeBakeCost {
    /// Exact number of source calls on success.
    pub sample_count: usize,
    /// Full grid payload size after success, even for a local update.
    pub output_bytes: usize,
    /// Additional temporary payload: zero for a full bake.
    pub scratch_bytes: usize,
    /// Checked sum of output and scratch payload bytes.
    pub peak_bytes: usize,
}

/// Validation, admission, allocation, or grid-access failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VolumeError {
    /// A sample count, byte count, index capacity, or peak calculation overflowed.
    SizeOverflow,
    /// A full-grid axis exceeded its dimension limit.
    DimensionLimit,
    /// The operation would make too many sampling calls.
    SampleLimit,
    /// The operation's peak logical payload would exceed the byte limit.
    ByteLimit,
    /// A fallible buffer reservation was rejected by the allocator.
    AllocationFailed,
    /// A rounded physical distance cannot be represented as signed Q16.16 i32.
    DistanceOutOfRange,
    /// A region is empty or has a reversed axis.
    InvalidRegion,
    /// A non-empty region extends beyond the grid.
    RegionOutOfBounds,
    /// A single sample coordinate extends beyond the grid.
    OutOfBounds {
        /// Requested grid coordinate.
        coordinate: [u32; 3],
    },
    /// Dimensions, brick size, or scale differ from the required contract.
    SpecMismatch,
    /// A voxel source does not declare signed Q8.8 distances.
    UnsupportedVoxelFormat,
    /// A local cache revision cannot be incremented.
    RevisionOverflow,
}

/// A bake failed either before publication or while sampling its temporary data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VolumeBakeError<E> {
    /// Contract, admission, allocation, or overflow failure.
    Volume(VolumeError),
    /// A source failed; existing grid bytes and revision remain unchanged.
    Sample(E),
}

impl<E> From<VolumeError> for VolumeBakeError<E> {
    fn from(value: VolumeError) -> Self {
        Self::Volume(value)
    }
}

/// A non-empty XYZ box of grid indices, with an exclusive upper endpoint.
///
/// Half-open indices avoid normalized-coordinate rounding and represent every
/// valid `SdfSpec` dimension, including singleton axes and `u32::MAX`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VolumeRegion {
    min: [u32; 3],
    end: [u32; 3],
}

impl VolumeRegion {
    /// Validates a non-empty box. Grid bounds are checked by the operation.
    pub fn new(min: [u32; 3], end: [u32; 3]) -> Result<Self, VolumeError> {
        if min.iter().zip(end).any(|(min, end)| *min >= end) {
            return Err(VolumeError::InvalidRegion);
        }
        Ok(Self { min, end })
    }

    /// Inclusive lower grid coordinate.
    pub const fn min(self) -> [u32; 3] {
        self.min
    }

    /// Exclusive upper grid coordinate.
    pub const fn end(self) -> [u32; 3] {
        self.end
    }

    /// Exact number of samples, or an explicit overflow error.
    pub fn sample_count(self) -> Result<usize, VolumeError> {
        sample_count([
            self.end[0] - self.min[0],
            self.end[1] - self.min[1],
            self.end[2] - self.min[2],
        ])
    }

    fn validate(self, dimensions: [u32; 3]) -> Result<(), VolumeError> {
        if self.end.iter().zip(dimensions).any(|(end, dim)| *end > dim) {
            return Err(VolumeError::RegionOutOfBounds);
        }
        Ok(())
    }
}

/// Successful local-update statistics; revisions are local to one grid instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfGridChange {
    /// Exact half-open box that was resampled.
    pub region: VolumeRegion,
    /// Cache revision before this transaction.
    pub previous_revision: u64,
    /// Cache revision after this transaction.
    pub revision: u64,
    /// Actual sampling and logical payload cost of this transaction.
    pub cost: VolumeBakeCost,
}

/// A bounded packed little-endian signed Q16.16 SDF grid.
///
/// The grid preserves the source specification exactly. Physical voxel spacing
/// is source-defined, not inferred from distance scale or streaming brick size.
#[derive(Debug, Eq, PartialEq)]
pub struct SdfGrid {
    spec: SdfSpec,
    revision: u64,
    bytes: Vec<u8>,
}

impl SdfGrid {
    /// Computes full-bake costs without allocating or sampling.
    pub fn bake_cost(spec: SdfSpec) -> Result<VolumeBakeCost, VolumeError> {
        let count = sample_count(spec.dimensions())?;
        let bytes = payload_bytes(count)?;
        Ok(VolumeBakeCost {
            sample_count: count,
            output_bytes: bytes,
            scratch_bytes: 0,
            peak_bytes: bytes,
        })
    }

    /// Bakes every grid sample after preflighting dimensions, bytes, and calls.
    ///
    /// Nothing is published on a contract, budget, allocation, or sample error.
    pub fn bake<F: SdfField + ?Sized>(
        source: &F,
        limits: VolumeBakeLimits,
    ) -> Result<Self, VolumeBakeError<F::Error>> {
        let spec = source.spec();
        let cost = Self::bake_cost(spec)?;
        admit(spec, cost, limits)?;
        let region = VolumeRegion::new([0; 3], spec.dimensions())?;
        let bytes = sample_region(source, region, cost.output_bytes)?;
        Ok(Self {
            spec,
            revision: 0,
            bytes,
        })
    }

    /// Original dimensions, streaming brick size, and signed-distance scale.
    pub const fn spec(&self) -> SdfSpec {
        self.spec
    }

    /// Monotonic local cache revision, incremented only by successful updates.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Exact X-fastest little-endian packed signed i32 sample bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Decodes one signed Q16.16 sample after validating all three axes.
    pub fn sample_q16(&self, coordinate: [u32; 3]) -> Result<i32, VolumeError> {
        let dimensions = self.spec.dimensions();
        if coordinate.iter().zip(dimensions).any(|(c, dim)| *c >= dim) {
            return Err(VolumeError::OutOfBounds { coordinate });
        }
        let offset = byte_offset(dimensions, coordinate);
        Ok(i32::from_le_bytes([
            self.bytes[offset],
            self.bytes[offset + 1],
            self.bytes[offset + 2],
            self.bytes[offset + 3],
        ]))
    }

    /// Computes exact local-update costs, including retained full-grid bytes.
    pub fn resample_cost(&self, region: VolumeRegion) -> Result<VolumeBakeCost, VolumeError> {
        region.validate(self.spec.dimensions())?;
        let sample_count = region.sample_count()?;
        let scratch_bytes = payload_bytes(sample_count)?;
        let output_bytes = self.bytes.len();
        let peak_bytes = output_bytes
            .checked_add(scratch_bytes)
            .ok_or(VolumeError::SizeOverflow)?;
        Ok(VolumeBakeCost {
            sample_count,
            output_bytes,
            scratch_bytes,
            peak_bytes,
        })
    }

    /// Atomically resamples exactly one half-open XYZ box.
    ///
    /// The source must preserve the original dimensions, brick size, scale, and
    /// spatial embedding. All fallible work, including the next revision and
    /// scratch reservation, finishes before any cache byte is overwritten.
    /// Outside samples are never read from the source or modified in the cache.
    /// Atomicity covers this cache, not side effects inside the source sampler.
    pub fn resample<F: SdfField + ?Sized>(
        &mut self,
        region: VolumeRegion,
        source: &F,
        limits: VolumeBakeLimits,
    ) -> Result<SdfGridChange, VolumeBakeError<F::Error>> {
        if source.spec() != self.spec {
            return Err(VolumeError::SpecMismatch.into());
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(VolumeError::RevisionOverflow)?;
        let cost = self.resample_cost(region)?;
        admit(self.spec, cost, limits)?;
        let staged = sample_region(source, region, cost.scratch_bytes)?;
        // These indices cannot overflow: the full payload was checked at bake,
        // the region is contained in it, and each staged row has the same width.
        // No fallible work remains after the first copy.
        let width_bytes = (region.end[0] - region.min[0]) as usize * 4;
        let mut staged_offset = 0;
        for z in region.min[2]..region.end[2] {
            for y in region.min[1]..region.end[1] {
                let offset = byte_offset(self.spec.dimensions(), [region.min[0], y, z]);
                self.bytes[offset..offset + width_bytes]
                    .copy_from_slice(&staged[staged_offset..staged_offset + width_bytes]);
                staged_offset += width_bytes;
            }
        }
        let change = SdfGridChange {
            region,
            previous_revision: self.revision,
            revision,
            cost,
        };
        self.revision = revision;
        Ok(change)
    }
}

impl SdfField for SdfGrid {
    type Error = VolumeError;

    fn spec(&self) -> SdfSpec {
        self.spec()
    }

    fn sample_q16(&self, coordinate: [u32; 3]) -> Result<i32, Self::Error> {
        self.sample_q16(coordinate)
    }
}

/// Exact Q8.8-to-Q16.16 signed-distance adapter for an existing `VoxelField`.
///
/// Accepts `VoxelBuffer` or any other voxel source. Both encodings use the
/// caller-supplied `SdfSpec::distance_scale_mm` per whole distance unit; the voxel
/// layout itself has no physical scale. This is an exact eight-bit fractional
/// expansion, not a metric reconstruction or a density-to-distance conversion.
pub struct VoxelSdfSource<'a, F: VoxelField + ?Sized> {
    field: &'a F,
    spec: SdfSpec,
}

impl<'a, F: VoxelField + ?Sized> VoxelSdfSource<'a, F> {
    /// Validates signed Q8.8 format and identical grid and streaming layout.
    pub fn new(field: &'a F, spec: SdfSpec) -> Result<Self, VolumeError> {
        let layout = field.layout();
        if layout.format() != VoxelComponentFormat::SignedDistanceQ8_8 {
            return Err(VolumeError::UnsupportedVoxelFormat);
        }
        if layout.dimensions() != spec.dimensions() || layout.brick_size() != spec.brick_size() {
            return Err(VolumeError::SpecMismatch);
        }
        Ok(Self { field, spec })
    }
}

impl<F: VoxelField + ?Sized> SdfField for VoxelSdfSource<'_, F> {
    type Error = VoxelError;

    fn spec(&self) -> SdfSpec {
        self.spec
    }

    fn sample_q16(&self, coordinate: [u32; 3]) -> Result<i32, Self::Error> {
        if coordinate
            .iter()
            .zip(self.spec.dimensions())
            .any(|(c, dim)| *c >= dim)
        {
            return Err(VoxelError::OutOfBounds { coordinate });
        }
        match self.field.sample(coordinate)? {
            VoxelSample::SignedDistance(value) => Ok(i32::from(value) * 256),
            _ => Err(VoxelError::FormatMismatch),
        }
    }
}

fn sample_count(dimensions: [u32; 3]) -> Result<usize, VolumeError> {
    dimensions.into_iter().try_fold(1_usize, |count, dim| {
        count
            .checked_mul(usize::try_from(dim).map_err(|_| VolumeError::SizeOverflow)?)
            .ok_or(VolumeError::SizeOverflow)
    })
}

fn payload_bytes(count: usize) -> Result<usize, VolumeError> {
    // Vec allocations cannot exceed isize::MAX even when usize multiplication
    // is representable. Report that as size overflow, not allocator exhaustion.
    count
        .checked_mul(4)
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or(VolumeError::SizeOverflow)
}

fn admit(spec: SdfSpec, cost: VolumeBakeCost, limits: VolumeBakeLimits) -> Result<(), VolumeError> {
    if spec
        .dimensions()
        .iter()
        .zip(limits.max_dimensions)
        .any(|(dim, limit)| *dim > limit)
    {
        return Err(VolumeError::DimensionLimit);
    }
    if cost.sample_count > limits.max_samples {
        return Err(VolumeError::SampleLimit);
    }
    if cost.peak_bytes > limits.max_bytes {
        return Err(VolumeError::ByteLimit);
    }
    Ok(())
}

fn sample_region<F: SdfField + ?Sized>(
    source: &F,
    region: VolumeRegion,
    bytes: usize,
) -> Result<Vec<u8>, VolumeBakeError<F::Error>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(bytes)
        .map_err(|_| VolumeError::AllocationFailed)?;
    for z in region.min[2]..region.end[2] {
        for y in region.min[1]..region.end[1] {
            for x in region.min[0]..region.end[0] {
                let value = source
                    .sample_q16([x, y, z])
                    .map_err(VolumeBakeError::Sample)?;
                result.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(result)
}

fn byte_offset(dimensions: [u32; 3], coordinate: [u32; 3]) -> usize {
    // Called only on contained coordinates of an already allocated, checked
    // full payload. Every product and sum is bounded by that payload length.
    ((coordinate[2] as usize * dimensions[1] as usize + coordinate[1] as usize)
        * dimensions[0] as usize
        + coordinate[0] as usize)
        * 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::{Cell, RefCell};
    use core::convert::Infallible;

    use crate::{VoxelBuffer, VoxelLayout};

    struct Plane {
        spec: SdfSpec,
        bias: i32,
        fail_at: Option<[u32; 3]>,
        visits: RefCell<Vec<[u32; 3]>>,
    }

    impl Plane {
        fn new(dimensions: [u32; 3]) -> Self {
            Self {
                spec: SdfSpec::new(dimensions, 2, 1000).unwrap(),
                bias: -65_536,
                fail_at: None,
                visits: RefCell::new(Vec::new()),
            }
        }

        fn value(&self, coordinate: [u32; 3]) -> i32 {
            // Euclidean distance to an axis-aligned plane; X spacing is one
            // metre and the SdfSpec has 1000 mm per whole Q16.16 unit.
            i32::try_from(coordinate[0]).unwrap() * 65_536 + self.bias
        }
    }

    impl SdfField for Plane {
        type Error = [u32; 3];

        fn spec(&self) -> SdfSpec {
            self.spec
        }

        fn sample_q16(&self, coordinate: [u32; 3]) -> Result<i32, Self::Error> {
            self.visits.borrow_mut().push(coordinate);
            if self.fail_at == Some(coordinate) {
                return Err(coordinate);
            }
            Ok(self.value(coordinate))
        }
    }

    fn limits(dimensions: [u32; 3]) -> VolumeBakeLimits {
        let count = sample_count(dimensions).unwrap();
        VolumeBakeLimits {
            max_dimensions: dimensions,
            max_samples: count,
            max_bytes: count * 8,
        }
    }

    #[test]
    fn integer_distance_quantization_rounds_ties_away_and_rejects_overflow() {
        let half = SdfSpec::new([1; 3], 1, 131_072).unwrap();
        for (distance, expected) in [(-3, -2), (-2, -1), (-1, -1), (0, 0), (1, 1), (2, 1), (3, 2)] {
            assert_eq!(quantize_sdf_mm(distance, half), Ok(expected));
        }
        let odd_scale = SdfSpec::new([1; 3], 1, 3).unwrap();
        for (distance, expected) in [(-2, -43691), (-1, -21845), (1, 21845), (2, 43691)] {
            assert_eq!(quantize_sdf_mm(distance, odd_scale), Ok(expected));
        }
        let exact = SdfSpec::new([1; 3], 1, 65_536).unwrap();
        for distance in [i32::MIN, -1, 0, 1, i32::MAX] {
            assert_eq!(quantize_sdf_mm(i64::from(distance), exact), Ok(distance));
        }
        for distance in [
            i64::MIN,
            i64::from(i32::MIN) - 1,
            i64::from(i32::MAX) + 1,
            i64::MAX,
        ] {
            assert_eq!(
                quantize_sdf_mm(distance, exact),
                Err(VolumeError::DistanceOutOfRange)
            );
        }
        let largest_scale = SdfSpec::new([1; 3], 1, i64::MAX).unwrap();
        assert_eq!(quantize_sdf_mm(i64::MIN, largest_scale), Ok(-65_536));
        assert_eq!(quantize_sdf_mm(i64::MAX, largest_scale), Ok(65_536));
        let smallest_scale = SdfSpec::new([1; 3], 1, 1).unwrap();
        assert_eq!(quantize_sdf_mm(-32768, smallest_scale), Ok(i32::MIN));
        assert_eq!(
            quantize_sdf_mm(32768, smallest_scale),
            Err(VolumeError::DistanceOutOfRange)
        );
    }

    #[test]
    fn plane_bakes_exact_little_endian_q16_distances_in_xyz_order() {
        let source = Plane::new([3, 2, 2]);
        let grid = SdfGrid::bake(&source, limits([3, 2, 2])).unwrap();
        assert_eq!(grid.spec(), source.spec);
        assert_eq!(grid.revision(), 0);
        assert_eq!(grid.bytes().len(), 3 * 2 * 2 * 4);
        let mut expected_bytes = Vec::new();
        let mut expected_visits = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..3 {
                    let coordinate = [x, y, z];
                    let value = source.value(coordinate);
                    assert_eq!(grid.sample_q16(coordinate).unwrap(), value);
                    expected_bytes.extend_from_slice(&value.to_le_bytes());
                    expected_visits.push(coordinate);
                }
            }
        }
        assert_eq!(grid.bytes(), expected_bytes);
        assert_eq!(*source.visits.borrow(), expected_visits);
        assert_eq!(&grid.bytes()[..4], &(-65_536_i32).to_le_bytes());
        assert_eq!(SdfGrid::bake(&source, limits([3, 2, 2])).unwrap(), grid);
        // The baked grid itself is a reusable, exact source.
        assert_eq!(SdfGrid::bake(&grid, limits([3, 2, 2])).unwrap(), grid);
    }

    #[test]
    fn singleton_axes_and_extreme_i32_encodings_are_preserved() {
        struct Constant(i32);
        impl SdfField for Constant {
            type Error = Infallible;

            fn spec(&self) -> SdfSpec {
                SdfSpec::new([1; 3], 1, i64::MAX).unwrap()
            }

            fn sample_q16(&self, _: [u32; 3]) -> Result<i32, Self::Error> {
                Ok(self.0)
            }
        }
        for value in [i32::MIN, -1, 0, 1, i32::MAX] {
            let grid = SdfGrid::bake(
                &Constant(value),
                VolumeBakeLimits {
                    max_dimensions: [1; 3],
                    max_samples: 1,
                    max_bytes: 4,
                },
            )
            .unwrap();
            assert_eq!(grid.bytes(), value.to_le_bytes());
            assert_eq!(grid.sample_q16([0; 3]), Ok(value));
            for coordinate in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [u32::MAX; 3]] {
                assert_eq!(
                    grid.sample_q16(coordinate),
                    Err(VolumeError::OutOfBounds { coordinate })
                );
            }
        }
    }

    #[test]
    fn exact_budgets_admit_and_each_insufficient_budget_prevents_sampling() {
        let source = Plane::new([3, 2, 2]);
        let exact = VolumeBakeLimits {
            max_dimensions: [3, 2, 2],
            max_samples: 12,
            max_bytes: 48,
        };
        let cost = SdfGrid::bake_cost(source.spec).unwrap();
        assert_eq!(cost.sample_count, 12);
        assert_eq!(cost.output_bytes, 48);
        assert_eq!(cost.scratch_bytes, 0);
        assert_eq!(cost.peak_bytes, 48);
        let mut cases = Vec::new();
        for axis in 0..3 {
            let mut smaller = exact;
            smaller.max_dimensions[axis] -= 1;
            cases.push((smaller, VolumeError::DimensionLimit));
        }
        cases.extend([
            (
                VolumeBakeLimits {
                    max_samples: 11,
                    ..exact
                },
                VolumeError::SampleLimit,
            ),
            (
                VolumeBakeLimits {
                    max_bytes: 47,
                    ..exact
                },
                VolumeError::ByteLimit,
            ),
            (
                VolumeBakeLimits {
                    max_samples: 0,
                    ..exact
                },
                VolumeError::SampleLimit,
            ),
            (
                VolumeBakeLimits {
                    max_bytes: 0,
                    ..exact
                },
                VolumeError::ByteLimit,
            ),
        ]);
        for (budget, error) in cases {
            assert_eq!(
                SdfGrid::bake(&source, budget),
                Err(VolumeBakeError::Volume(error))
            );
            assert!(source.visits.borrow().is_empty());
        }
        assert!(SdfGrid::bake(&source, exact).is_ok());
        assert_eq!(source.visits.borrow().len(), 12);
    }

    #[test]
    fn count_byte_and_capacity_overflows_fail_without_sampling() {
        for dimensions in [
            [u32::MAX; 3],
            [u32::MAX, u32::MAX, 1],
            [1_u32 << 30, 1_u32 << 31, 1],
        ] {
            let source = Plane::new(dimensions);
            assert_eq!(
                SdfGrid::bake_cost(source.spec),
                Err(VolumeError::SizeOverflow)
            );
            assert_eq!(
                SdfGrid::bake(
                    &source,
                    VolumeBakeLimits {
                        max_dimensions: [u32::MAX; 3],
                        max_samples: usize::MAX,
                        max_bytes: usize::MAX,
                    }
                ),
                Err(VolumeBakeError::Volume(VolumeError::SizeOverflow))
            );
            assert!(source.visits.borrow().is_empty());
        }
        let region = VolumeRegion::new([0; 3], [u32::MAX; 3]).unwrap();
        assert_eq!(region.sample_count(), Err(VolumeError::SizeOverflow));
        assert_eq!(payload_bytes(usize::MAX), Err(VolumeError::SizeOverflow));
    }

    #[test]
    fn local_resampling_is_bounded_atomic_and_matches_a_full_rebuild() {
        let mut source = Plane::new([4, 3, 2]);
        let mut grid = SdfGrid::bake(&source, limits([4, 3, 2])).unwrap();
        let old = grid.bytes().to_vec();
        source.visits.borrow_mut().clear();
        source.bias += 123;
        let region = VolumeRegion::new([1, 1, 0], [4, 3, 2]).unwrap();
        let cost = grid.resample_cost(region).unwrap();
        assert_eq!(cost.sample_count, 12);
        assert_eq!(cost.output_bytes, 96);
        assert_eq!(cost.scratch_bytes, 48);
        assert_eq!(cost.peak_bytes, 144);
        let change = grid
            .resample(
                region,
                &source,
                VolumeBakeLimits {
                    max_dimensions: [4, 3, 2],
                    max_samples: 12,
                    max_bytes: 144,
                },
            )
            .unwrap();
        assert_eq!(change.region, region);
        assert_eq!(change.previous_revision, 0);
        assert_eq!(change.revision, 1);
        assert_eq!(change.cost, cost);
        assert_eq!(grid.revision(), 1);
        assert_eq!(source.visits.borrow().len(), 12);
        for z in 0..2 {
            for y in 0..3 {
                for x in 0..4 {
                    let coordinate = [x, y, z];
                    if x >= 1 && y >= 1 {
                        assert_eq!(
                            grid.sample_q16(coordinate).unwrap(),
                            source.value(coordinate)
                        );
                    } else {
                        let offset = byte_offset([4, 3, 2], coordinate);
                        assert_eq!(&grid.bytes()[offset..offset + 4], &old[offset..offset + 4]);
                    }
                }
            }
        }
        struct Composite<'a> {
            source: &'a Plane,
        }
        impl SdfField for Composite<'_> {
            type Error = Infallible;

            fn spec(&self) -> SdfSpec {
                self.source.spec
            }

            fn sample_q16(&self, coordinate: [u32; 3]) -> Result<i32, Self::Error> {
                let value = self.source.value(coordinate);
                Ok(if coordinate[0] >= 1 && coordinate[1] >= 1 {
                    value
                } else {
                    value - 123
                })
            }
        }
        let full = SdfGrid::bake(&Composite { source: &source }, limits([4, 3, 2])).unwrap();
        assert_eq!(grid.bytes(), full.bytes());
    }

    #[test]
    fn local_sampling_failure_on_every_position_preserves_all_bytes_and_revision() {
        let mut source = Plane::new([4, 3, 2]);
        let mut grid = SdfGrid::bake(&source, limits([4, 3, 2])).unwrap();
        let old = grid.bytes().to_vec();
        source.bias += 999;
        let region = VolumeRegion::new([1, 1, 0], [4, 3, 2]).unwrap();
        let mut failed_calls = 0;
        for z in 0..2 {
            for y in 1..3 {
                for x in 1..4 {
                    source.visits.borrow_mut().clear();
                    source.fail_at = Some([x, y, z]);
                    failed_calls += 1;
                    assert_eq!(
                        grid.resample(region, &source, limits([4, 3, 2])),
                        Err(VolumeBakeError::Sample([x, y, z]))
                    );
                    assert_eq!(source.visits.borrow().len(), failed_calls);
                    assert_eq!(grid.bytes(), old);
                    assert_eq!(grid.revision(), 0);
                }
            }
        }
        source.fail_at = None;
        assert!(grid.resample(region, &source, limits([4, 3, 2])).is_ok());
        assert_eq!(grid.revision(), 1);
    }

    #[test]
    fn full_bake_sampling_failure_returns_no_grid_and_stops_immediately() {
        let mut source = Plane::new([3, 2, 2]);
        source.fail_at = Some([2, 1, 0]);
        assert_eq!(
            SdfGrid::bake(&source, limits([3, 2, 2])),
            Err(VolumeBakeError::Sample([2, 1, 0]))
        );
        assert_eq!(source.visits.borrow().len(), 6);
    }

    #[test]
    fn fallible_reservation_failure_precedes_source_calls() {
        let source = Plane::new([1; 3]);
        let region = VolumeRegion::new([0; 3], [1; 3]).unwrap();
        // Deterministic capacity failure in the reservation helper, without
        // relying on host memory exhaustion or a custom unsafe allocator.
        assert_eq!(
            sample_region(&source, region, usize::MAX),
            Err(VolumeBakeError::Volume(VolumeError::AllocationFailed))
        );
        assert!(source.visits.borrow().is_empty());
    }

    #[test]
    fn every_sub_box_copies_only_its_xyz_rows_in_deterministic_order() {
        let mut source = Plane::new([3, 2, 3]);
        for x0 in 0..3 {
            for x1 in x0 + 1..=3 {
                for y0 in 0..2 {
                    for y1 in y0 + 1..=2 {
                        for z0 in 0..3 {
                            for z1 in z0 + 1..=3 {
                                source.bias = 0;
                                let mut grid = SdfGrid::bake(&source, limits([3, 2, 3])).unwrap();
                                source.visits.borrow_mut().clear();
                                source.bias = 1;
                                let region = VolumeRegion::new([x0, y0, z0], [x1, y1, z1]).unwrap();
                                let cost = grid.resample_cost(region).unwrap();
                                let change = grid
                                    .resample(
                                        region,
                                        &source,
                                        VolumeBakeLimits {
                                            max_dimensions: [3, 2, 3],
                                            max_samples: cost.sample_count,
                                            max_bytes: cost.peak_bytes,
                                        },
                                    )
                                    .unwrap();
                                assert_eq!(change.cost, cost);
                                let mut expected_visits = Vec::new();
                                for z in 0..3 {
                                    for y in 0..2 {
                                        for x in 0..3 {
                                            let inside = (x0..x1).contains(&x)
                                                && (y0..y1).contains(&y)
                                                && (z0..z1).contains(&z);
                                            let value = (x as i32) * 65_536 + i32::from(inside);
                                            assert_eq!(grid.sample_q16([x, y, z]).unwrap(), value);
                                            if inside {
                                                expected_visits.push([x, y, z]);
                                            }
                                        }
                                    }
                                }
                                assert_eq!(*source.visits.borrow(), expected_visits);
                                assert_eq!(grid.revision(), 1);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_regions_contracts_budgets_and_revision_fail_before_sampling() {
        let mut source = Plane::new([3, 2, 2]);
        let mut grid = SdfGrid::bake(&source, limits([3, 2, 2])).unwrap();
        let old = grid.bytes().to_vec();
        source.visits.borrow_mut().clear();
        let region = VolumeRegion::new([2, 1, 1], [3, 2, 2]).unwrap();
        assert_eq!(region.min(), [2, 1, 1]);
        assert_eq!(region.end(), [3, 2, 2]);
        for axis in 0..3 {
            let mut end = [1; 3];
            end[axis] = 0;
            assert_eq!(
                VolumeRegion::new([0; 3], end),
                Err(VolumeError::InvalidRegion)
            );
            let mut min = [0; 3];
            min[axis] = 2;
            assert_eq!(
                VolumeRegion::new(min, [1; 3]),
                Err(VolumeError::InvalidRegion)
            );
            let mut end = [3, 2, 2];
            end[axis] += 1;
            let outside = VolumeRegion::new([0; 3], end).unwrap();
            assert_eq!(
                grid.resample_cost(outside),
                Err(VolumeError::RegionOutOfBounds)
            );
            assert_eq!(
                grid.resample(outside, &source, limits([3, 2, 2])),
                Err(VolumeBakeError::Volume(VolumeError::RegionOutOfBounds))
            );
        }
        let original_spec = source.spec;
        for spec in [
            SdfSpec::new([4, 2, 2], 2, 1000).unwrap(),
            SdfSpec::new([3, 3, 2], 2, 1000).unwrap(),
            SdfSpec::new([3, 2, 3], 2, 1000).unwrap(),
            SdfSpec::new([3, 2, 2], 4, 1000).unwrap(),
            SdfSpec::new([3, 2, 2], 2, 1).unwrap(),
        ] {
            source.spec = spec;
            assert_eq!(
                grid.resample(region, &source, limits([4, 3, 3])),
                Err(VolumeBakeError::Volume(VolumeError::SpecMismatch))
            );
        }
        source.spec = original_spec;
        for (budget, error) in [
            (
                VolumeBakeLimits {
                    max_dimensions: [3, 2, 1],
                    ..limits([3, 2, 2])
                },
                VolumeError::DimensionLimit,
            ),
            (
                VolumeBakeLimits {
                    max_samples: 0,
                    ..limits([3, 2, 2])
                },
                VolumeError::SampleLimit,
            ),
            (
                VolumeBakeLimits {
                    max_bytes: 51,
                    ..limits([3, 2, 2])
                },
                VolumeError::ByteLimit,
            ),
        ] {
            assert_eq!(
                grid.resample(region, &source, budget),
                Err(VolumeBakeError::Volume(error))
            );
        }
        assert_eq!(grid.bytes(), old);
        assert_eq!(grid.revision(), 0);
        grid.revision = u64::MAX;
        assert_eq!(
            grid.resample(region, &source, limits([3, 2, 2])),
            Err(VolumeBakeError::Volume(VolumeError::RevisionOverflow))
        );
        assert_eq!(grid.bytes(), old);
        assert_eq!(grid.revision(), u64::MAX);
        assert!(source.visits.borrow().is_empty());
    }

    #[test]
    fn local_singleton_update_charges_retained_grid_plus_scratch_exactly() {
        let mut source = Plane::new([1; 3]);
        let mut grid = SdfGrid::bake(&source, limits([1; 3])).unwrap();
        source.bias = 42;
        source.visits.borrow_mut().clear();
        let region = VolumeRegion::new([0; 3], [1; 3]).unwrap();
        assert_eq!(grid.resample_cost(region).unwrap().peak_bytes, 8);
        let change = grid.resample(region, &source, limits([1; 3])).unwrap();
        assert_eq!(change.cost.sample_count, 1);
        assert_eq!(change.cost.scratch_bytes, 4);
        assert_eq!(grid.sample_q16([0; 3]).unwrap(), 42);
        assert_eq!(*source.visits.borrow(), [[0; 3]]);
    }

    #[test]
    fn voxel_buffer_q8_distances_expand_exactly_without_scale_or_sign_loss() {
        let values = [i16::MIN, -256, -1, 0, 1, i16::MAX];
        let layout =
            VoxelLayout::new([3, 2, 1], 2, VoxelComponentFormat::SignedDistanceQ8_8).unwrap();
        let bytes = values.into_iter().flat_map(i16::to_le_bytes).collect();
        let buffer = VoxelBuffer::new(layout, bytes).unwrap();
        let spec = SdfSpec::new([3, 2, 1], 2, 250).unwrap();
        let source = VoxelSdfSource::new(&buffer, spec).unwrap();
        let grid = SdfGrid::bake(&source, limits([3, 2, 1])).unwrap();
        assert_eq!(grid.spec(), spec);
        assert_eq!(grid.bytes().len(), 24);
        for (index, value) in values.into_iter().enumerate() {
            let coordinate = [(index % 3) as u32, (index / 3) as u32, 0];
            let raw = grid.sample_q16(coordinate).unwrap();
            assert_eq!(raw, i32::from(value) * 256);
            // Physical millimetres are identical: Q8.8 / 256 = Q16.16 / 65536.
            assert_eq!(i128::from(raw) * 250, i128::from(value) * 250 * 256);
        }
        assert_eq!(
            source.sample_q16([3, 0, 0]),
            Err(VoxelError::OutOfBounds {
                coordinate: [3, 0, 0]
            })
        );
    }

    #[test]
    fn voxel_adapter_rejects_wrong_specs_formats_and_inconsistent_samples() {
        let spec = SdfSpec::new([1; 3], 1, 1).unwrap();
        for format in [
            VoxelComponentFormat::DensityU16,
            VoxelComponentFormat::MaterialIdU16,
        ] {
            let layout = VoxelLayout::new([1; 3], 1, format).unwrap();
            let buffer = VoxelBuffer::new(layout, alloc::vec![0; 2]).unwrap();
            assert!(matches!(
                VoxelSdfSource::new(&buffer, spec),
                Err(VolumeError::UnsupportedVoxelFormat)
            ));
        }
        let layout = VoxelLayout::new([1; 3], 1, VoxelComponentFormat::SignedDistanceQ8_8).unwrap();
        let buffer = VoxelBuffer::new(layout, alloc::vec![0; 2]).unwrap();
        for different in [
            SdfSpec::new([2, 1, 1], 1, 1).unwrap(),
            SdfSpec::new([1, 2, 1], 1, 1).unwrap(),
            SdfSpec::new([1, 1, 2], 1, 1).unwrap(),
            SdfSpec::new([1; 3], 2, 1).unwrap(),
        ] {
            assert!(matches!(
                VoxelSdfSource::new(&buffer, different),
                Err(VolumeError::SpecMismatch)
            ));
        }
        struct Inconsistent {
            layout: VoxelLayout,
            fail: Cell<bool>,
        }
        impl VoxelField for Inconsistent {
            fn layout(&self) -> VoxelLayout {
                self.layout
            }

            fn sample(&self, _: [u32; 3]) -> Result<VoxelSample, VoxelError> {
                if self.fail.get() {
                    Err(VoxelError::InvalidDimensions)
                } else {
                    Ok(VoxelSample::Density(1))
                }
            }
        }
        let field = Inconsistent {
            layout,
            fail: Cell::new(false),
        };
        let adapter = VoxelSdfSource::new(&field, spec).unwrap();
        assert_eq!(
            SdfGrid::bake(&adapter, limits([1; 3])),
            Err(VolumeBakeError::Sample(VoxelError::FormatMismatch))
        );
        field.fail.set(true);
        assert_eq!(
            SdfGrid::bake(&adapter, limits([1; 3])),
            Err(VolumeBakeError::Sample(VoxelError::InvalidDimensions))
        );
    }
}
