//! Engine-neutral terrain representation and conversion-selection contracts.

use crate::{TileKey, VoxelComponentFormat, VoxelLayout};

/// A representation family understood by the terrain data layer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RepresentationKind {
    /// A regular two-dimensional elevation sample grid.
    Heightfield,
    /// A regular three-dimensional field with an explicit voxel format.
    Voxel,
    /// A regular three-dimensional signed-distance field.
    Sdf,
}

/// A validated heightfield representation description.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldSpec {
    samples_per_edge: u32,
    height_scale_mm: i64,
}

impl HeightfieldSpec {
    /// Creates a heightfield with a square sample grid and a positive scale.
    pub const fn new(
        samples_per_edge: u32,
        height_scale_mm: i64,
    ) -> Result<Self, RepresentationError> {
        if samples_per_edge < 2 {
            return Err(RepresentationError::InvalidResolution);
        }
        if height_scale_mm <= 0 {
            return Err(RepresentationError::InvalidScale);
        }
        Ok(Self {
            samples_per_edge,
            height_scale_mm,
        })
    }

    /// Returns the number of samples along either grid axis.
    pub const fn samples_per_edge(self) -> u32 {
        self.samples_per_edge
    }

    /// Returns the signed millimetre height scale.
    pub const fn height_scale_mm(self) -> i64 {
        self.height_scale_mm
    }
}

/// A validated voxel representation description.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VoxelSpec {
    layout: VoxelLayout,
}

impl VoxelSpec {
    /// Creates a voxel description from the crate's validated storage layout.
    pub const fn from_layout(layout: VoxelLayout) -> Self {
        Self { layout }
    }

    /// Creates a voxel description after validating dimensions, brick size, and format.
    pub fn new(
        dimensions: [u32; 3],
        brick_size: u32,
        format: VoxelComponentFormat,
    ) -> Result<Self, RepresentationError> {
        Ok(Self::from_layout(
            VoxelLayout::new(dimensions, brick_size, format)
                .map_err(RepresentationError::InvalidVoxelLayout)?,
        ))
    }

    /// Returns the validated storage layout.
    pub const fn layout(self) -> VoxelLayout {
        self.layout
    }
}

/// A validated signed-distance-field representation description.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfSpec {
    dimensions: [u32; 3],
    brick_size: u32,
    distance_scale_mm: i64,
}

impl SdfSpec {
    /// Creates an SDF description using signed Q16.16 distance samples.
    pub fn new(
        dimensions: [u32; 3],
        brick_size: u32,
        distance_scale_mm: i64,
    ) -> Result<Self, RepresentationError> {
        if dimensions.contains(&0) {
            return Err(RepresentationError::InvalidResolution);
        }
        if brick_size == 0 || !brick_size.is_power_of_two() {
            return Err(RepresentationError::InvalidBrickSize);
        }
        if distance_scale_mm <= 0 {
            return Err(RepresentationError::InvalidScale);
        }
        Ok(Self {
            dimensions,
            brick_size,
            distance_scale_mm,
        })
    }

    /// Returns the voxel dimensions in X/Y/Z order.
    pub const fn dimensions(self) -> [u32; 3] {
        self.dimensions
    }

    /// Returns the streaming brick edge length.
    pub const fn brick_size(self) -> u32 {
        self.brick_size
    }

    /// Returns the signed millimetre distance scale.
    pub const fn distance_scale_mm(self) -> i64 {
        self.distance_scale_mm
    }
}

/// A validated representation description.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepresentationSpec {
    /// Heightfield description.
    Heightfield(HeightfieldSpec),
    /// Voxel description.
    Voxel(VoxelSpec),
    /// Signed-distance description.
    Sdf(SdfSpec),
}

impl RepresentationSpec {
    /// Returns the representation family.
    pub const fn kind(self) -> RepresentationKind {
        match self {
            Self::Heightfield(_) => RepresentationKind::Heightfield,
            Self::Voxel(_) => RepresentationKind::Voxel,
            Self::Sdf(_) => RepresentationKind::Sdf,
        }
    }

    /// Returns deterministic work and packed output bytes for an XY sample footprint.
    ///
    /// `sample_count` counts covered XY grid points, not volume samples. Voxel and
    /// SDF footprints rebuild complete Z columns; their depth is applied here to
    /// both work and bytes. Heightfield samples occupy two bytes, voxel samples
    /// use the layout's component width, and Q16.16 SDF samples occupy four bytes.
    /// There is no brick padding, header, compression, or GPU alignment in this
    /// packed data contract. One output sample costs `tile.level() + 1` abstract
    /// work units; this is a scheduling weight, not a wall-clock prediction.
    ///
    /// Returns [`RepresentationError::CostOverflow`] if any count, byte, or work
    /// multiplication exceeds `u64`; it never saturates a budget estimate.
    pub fn estimate_work(
        self,
        tile: TileKey,
        sample_count: u64,
    ) -> Result<RepresentationWork, RepresentationError> {
        let level_factor = u64::from(tile.level())
            .checked_add(1)
            .ok_or(RepresentationError::CostOverflow)?;
        let depth = match self {
            Self::Heightfield(_) => 1,
            Self::Voxel(spec) => u64::from(spec.layout().dimensions()[2]),
            Self::Sdf(spec) => u64::from(spec.dimensions()[2]),
        };
        let output_samples = sample_count
            .checked_mul(depth)
            .ok_or(RepresentationError::CostOverflow)?;
        let work_units = output_samples
            .checked_mul(level_factor)
            .ok_or(RepresentationError::CostOverflow)?;
        let bytes_per_sample = match self {
            Self::Heightfield(_) => 2,
            Self::Voxel(spec) => spec.layout().format().byte_width() as u64,
            Self::Sdf(_) => 4,
        };
        let bytes = output_samples
            .checked_mul(bytes_per_sample)
            .ok_or(RepresentationError::CostOverflow)?;
        Ok(RepresentationWork { work_units, bytes })
    }
}

/// Estimated deterministic work for one rebake task.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepresentationWork {
    /// Abstract work units used by a scheduler budget.
    pub work_units: u64,
    /// Estimated encoded bytes produced by the task.
    pub bytes: u64,
}

/// A source-to-target representation choice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepresentationChoice {
    /// Source representation already available for the tile.
    pub source: RepresentationSpec,
    /// Target representation requested by the consumer.
    pub target: RepresentationSpec,
    /// Whether the complete source and target specifications differ.
    ///
    /// Resolution, scale, component format, and brick-layout changes all require
    /// conversion, even within one family. This flag describes required work,
    /// not a guarantee that a particular source-to-target converter exists.
    pub conversion_required: bool,
}

impl RepresentationChoice {
    /// Creates a source-to-target choice from validated representation specs.
    pub const fn new(source: RepresentationSpec, target: RepresentationSpec) -> Self {
        Self {
            source,
            target,
            conversion_required: !same_spec(source, target),
        }
    }
}

const fn same_spec(left: RepresentationSpec, right: RepresentationSpec) -> bool {
    match (left, right) {
        (RepresentationSpec::Heightfield(left), RepresentationSpec::Heightfield(right)) => {
            left.samples_per_edge == right.samples_per_edge
                && left.height_scale_mm == right.height_scale_mm
        }
        (RepresentationSpec::Voxel(left), RepresentationSpec::Voxel(right)) => {
            let left = left.layout();
            let right = right.layout();
            same_dimensions(left.dimensions(), right.dimensions())
                && left.brick_size() == right.brick_size()
                && matches!(
                    (left.format(), right.format()),
                    (
                        VoxelComponentFormat::SignedDistanceQ8_8,
                        VoxelComponentFormat::SignedDistanceQ8_8
                    ) | (
                        VoxelComponentFormat::DensityU16,
                        VoxelComponentFormat::DensityU16
                    ) | (
                        VoxelComponentFormat::MaterialIdU16,
                        VoxelComponentFormat::MaterialIdU16
                    )
                )
        }
        (RepresentationSpec::Sdf(left), RepresentationSpec::Sdf(right)) => {
            same_dimensions(left.dimensions, right.dimensions)
                && left.brick_size == right.brick_size
                && left.distance_scale_mm == right.distance_scale_mm
        }
        _ => false,
    }
}

const fn same_dimensions(left: [u32; 3], right: [u32; 3]) -> bool {
    left[0] == right[0] && left[1] == right[1] && left[2] == right[2]
}

/// Deterministic selector for choosing among heightfield, voxel, and SDF data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerrainRepresentationSelector {
    heightfield: HeightfieldSpec,
    voxel: VoxelSpec,
    sdf: SdfSpec,
}

impl TerrainRepresentationSelector {
    /// Creates a selector with the three available representation contracts.
    pub const fn new(heightfield: HeightfieldSpec, voxel: VoxelSpec, sdf: SdfSpec) -> Self {
        Self {
            heightfield,
            voxel,
            sdf,
        }
    }

    /// Selects a representation deterministically from tile level and query requirements.
    ///
    /// Collision or volumetric queries choose SDF first, solid-volume queries choose voxel,
    /// and surface-only queries choose heightfield. A higher tile level never changes the
    /// family choice; it only affects the rebake cost estimate.
    pub const fn select(self, query: RepresentationQuery) -> RepresentationSpec {
        if query.requires_collision || query.requires_signed_distance {
            RepresentationSpec::Sdf(self.sdf)
        } else if query.requires_volume {
            RepresentationSpec::Voxel(self.voxel)
        } else {
            RepresentationSpec::Heightfield(self.heightfield)
        }
    }
}

/// Requirements used by [`TerrainRepresentationSelector`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RepresentationQuery {
    /// Whether collision or contact queries need a signed distance.
    pub requires_collision: bool,
    /// Whether the caller needs a signed-distance query directly.
    pub requires_signed_distance: bool,
    /// Whether the caller needs a full volumetric field.
    pub requires_volume: bool,
}

/// Errors returned by representation validation and cost estimation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepresentationError {
    /// A resolution or dimension is zero or otherwise too small.
    InvalidResolution,
    /// A scale is zero or negative.
    InvalidScale,
    /// A brick edge is not a non-zero power of two.
    InvalidBrickSize,
    /// The wrapped voxel layout is invalid.
    InvalidVoxelLayout(crate::VoxelError),
    /// An estimated cost overflowed `u64`.
    CostOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CubeFace, VoxelComponentFormat};

    fn specs() -> (HeightfieldSpec, VoxelSpec, SdfSpec) {
        (
            HeightfieldSpec::new(129, 10_000).unwrap(),
            VoxelSpec::new([32, 32, 16], 8, VoxelComponentFormat::DensityU16).unwrap(),
            SdfSpec::new([32, 32, 32], 8, 1).unwrap(),
        )
    }

    #[test]
    fn representation_contracts_validate_and_select_stably() {
        let (heightfield, voxel, sdf) = specs();
        let selector = TerrainRepresentationSelector::new(heightfield, voxel, sdf);
        assert_eq!(
            selector.select(RepresentationQuery::default()).kind(),
            RepresentationKind::Heightfield
        );
        assert_eq!(
            selector
                .select(RepresentationQuery {
                    requires_volume: true,
                    ..Default::default()
                })
                .kind(),
            RepresentationKind::Voxel
        );
        assert_eq!(
            selector
                .select(RepresentationQuery {
                    requires_collision: true,
                    ..Default::default()
                })
                .kind(),
            RepresentationKind::Sdf
        );
        assert!(
            RepresentationChoice::new(
                RepresentationSpec::Heightfield(heightfield),
                RepresentationSpec::Sdf(sdf)
            )
            .conversion_required
        );
        assert!(
            !RepresentationChoice::new(
                RepresentationSpec::Voxel(voxel),
                RepresentationSpec::Voxel(voxel)
            )
            .conversion_required
        );
    }

    #[test]
    fn representation_work_is_deterministic_and_level_sensitive() {
        let (heightfield, _, _) = specs();
        let low = TileKey::new(CubeFace::PositiveX, 0, 0, 0).unwrap();
        let high = TileKey::new(CubeFace::PositiveX, 4, 3, 7).unwrap();
        let spec = RepresentationSpec::Heightfield(heightfield);
        assert_eq!(
            spec.estimate_work(low, 64).unwrap(),
            spec.estimate_work(low, 64).unwrap()
        );
        assert!(
            spec.estimate_work(high, 64).unwrap().work_units
                > spec.estimate_work(low, 64).unwrap().work_units
        );
        assert_eq!(spec.estimate_work(low, 64).unwrap().bytes, 128);
        assert_eq!(spec.estimate_work(high, 64).unwrap().bytes, 128);
    }

    #[test]
    fn conversion_compares_every_field_not_only_the_family() {
        let heightfield = RepresentationSpec::Heightfield(HeightfieldSpec::new(33, 1).unwrap());
        for different in [
            RepresentationSpec::Heightfield(HeightfieldSpec::new(65, 1).unwrap()),
            RepresentationSpec::Heightfield(HeightfieldSpec::new(33, 2).unwrap()),
        ] {
            assert!(RepresentationChoice::new(heightfield, different).conversion_required);
        }

        let density = VoxelComponentFormat::DensityU16;
        let voxel = RepresentationSpec::Voxel(VoxelSpec::new([8, 9, 10], 2, density).unwrap());
        for (dimensions, brick, format) in [
            ([9, 9, 10], 2, density),
            ([8, 10, 10], 2, density),
            ([8, 9, 11], 2, density),
            ([8, 9, 10], 4, density),
            ([8, 9, 10], 2, VoxelComponentFormat::SignedDistanceQ8_8),
            ([8, 9, 10], 2, VoxelComponentFormat::MaterialIdU16),
        ] {
            let different =
                RepresentationSpec::Voxel(VoxelSpec::new(dimensions, brick, format).unwrap());
            assert!(RepresentationChoice::new(voxel, different).conversion_required);
        }

        let sdf = RepresentationSpec::Sdf(SdfSpec::new([8, 9, 10], 2, 1).unwrap());
        for (dimensions, brick, scale) in [
            ([9, 9, 10], 2, 1),
            ([8, 10, 10], 2, 1),
            ([8, 9, 11], 2, 1),
            ([8, 9, 10], 4, 1),
            ([8, 9, 10], 2, 2),
        ] {
            let different =
                RepresentationSpec::Sdf(SdfSpec::new(dimensions, brick, scale).unwrap());
            assert!(RepresentationChoice::new(sdf, different).conversion_required);
        }

        for source in [heightfield, voxel, sdf] {
            for target in [heightfield, voxel, sdf] {
                assert_eq!(
                    RepresentationChoice::new(source, target).conversion_required,
                    source != target
                );
            }
        }
        for format in [
            density,
            VoxelComponentFormat::SignedDistanceQ8_8,
            VoxelComponentFormat::MaterialIdU16,
        ] {
            let spec = RepresentationSpec::Voxel(VoxelSpec::new([8, 9, 10], 2, format).unwrap());
            assert!(!RepresentationChoice::new(spec, spec).conversion_required);
        }
    }

    #[test]
    fn full_volume_cost_includes_depth_exactly_once() {
        let tile = TileKey::root(CubeFace::NegativeZ);
        let dimensions = [7, 5, 3];
        let xy_count = u64::from(dimensions[0]) * u64::from(dimensions[1]);
        for format in [
            VoxelComponentFormat::DensityU16,
            VoxelComponentFormat::SignedDistanceQ8_8,
            VoxelComponentFormat::MaterialIdU16,
        ] {
            let spec = VoxelSpec::new(dimensions, 2, format).unwrap();
            let work = RepresentationSpec::Voxel(spec)
                .estimate_work(tile, xy_count)
                .unwrap();
            assert_eq!(work.work_units, 105);
            assert_eq!(
                work.bytes,
                spec.layout().expected_byte_len().unwrap() as u64
            );
        }
        let sdf = RepresentationSpec::Sdf(SdfSpec::new(dimensions, 2, 1).unwrap());
        assert_eq!(
            sdf.estimate_work(tile, xy_count).unwrap(),
            RepresentationWork {
                work_units: 105,
                bytes: 420
            }
        );
        assert_eq!(
            sdf.estimate_work(tile, 0).unwrap(),
            RepresentationWork {
                work_units: 0,
                bytes: 0
            }
        );
    }

    #[test]
    fn cost_overflow_is_explicit_for_samples_bytes_and_level_weight() {
        let root = TileKey::root(CubeFace::PositiveX);
        let leaf = TileKey::new(CubeFace::NegativeY, crate::MAX_TILE_LEVEL, 0, 0).unwrap();
        let volume = RepresentationSpec::Sdf(SdfSpec::new([1, 1, 3], 1, 1).unwrap());
        let scalar = RepresentationSpec::Sdf(SdfSpec::new([1, 1, 1], 1, 1).unwrap());
        let heightfield = RepresentationSpec::Heightfield(HeightfieldSpec::new(2, 1).unwrap());
        // Depth expansion overflows before either work weighting or encoding.
        assert_eq!(
            volume.estimate_work(root, u64::MAX / 3 + 1),
            Err(RepresentationError::CostOverflow)
        );
        // Sample count is representable; the packed output byte count is not.
        assert_eq!(
            scalar.estimate_work(root, u64::MAX / 4 + 1),
            Err(RepresentationError::CostOverflow)
        );
        assert_eq!(
            heightfield.estimate_work(root, u64::MAX / 2 + 1),
            Err(RepresentationError::CostOverflow)
        );
        // Bytes fit but the maximum-level scheduling weight does not.
        assert_eq!(
            scalar.estimate_work(leaf, u64::MAX / 31 + 1),
            Err(RepresentationError::CostOverflow)
        );
        let xy_count = u64::from(u32::MAX) * u64::from(u32::MAX);
        let huge = RepresentationSpec::Sdf(SdfSpec::new([u32::MAX; 3], 1, 1).unwrap());
        assert_eq!(
            huge.estimate_work(root, xy_count),
            Err(RepresentationError::CostOverflow)
        );
    }

    #[test]
    fn invalid_representation_inputs_are_rejected() {
        assert_eq!(
            HeightfieldSpec::new(1, 1),
            Err(RepresentationError::InvalidResolution)
        );
        assert_eq!(
            SdfSpec::new([1, 1, 1], 3, 1),
            Err(RepresentationError::InvalidBrickSize)
        );
        assert_eq!(
            SdfSpec::new([1, 1, 1], 2, 0),
            Err(RepresentationError::InvalidScale)
        );
    }
}
