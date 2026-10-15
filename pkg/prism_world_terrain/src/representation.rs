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

    /// Returns a deterministic work and byte estimate for a tile-local sample count.
    pub fn estimate_work(
        self,
        tile: TileKey,
        sample_count: u64,
    ) -> Result<RepresentationWork, RepresentationError> {
        let level_factor = u64::from(tile.level())
            .checked_add(1)
            .ok_or(RepresentationError::CostOverflow)?;
        let work_multiplier = match self {
            Self::Heightfield(spec) => u64::from(spec.samples_per_edge().max(2)),
            Self::Voxel(spec) => u64::from(spec.layout().dimensions()[2]),
            Self::Sdf(spec) => u64::from(spec.dimensions()[2]),
        };
        let work_units = sample_count
            .checked_mul(work_multiplier)
            .and_then(|value| value.checked_mul(level_factor))
            .ok_or(RepresentationError::CostOverflow)?;
        let bytes_per_sample = match self {
            Self::Heightfield(_) => 2,
            Self::Voxel(_) => 2,
            Self::Sdf(_) => 4,
        };
        let bytes = sample_count
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
    /// Whether the choice requires resampling or only a same-format rebake.
    pub conversion_required: bool,
}

impl RepresentationChoice {
    /// Validates and creates a source-to-target choice.
    pub const fn new(source: RepresentationSpec, target: RepresentationSpec) -> Self {
        Self {
            source,
            target,
            conversion_required: !same_kind(source.kind(), target.kind()),
        }
    }
}

const fn same_kind(left: RepresentationKind, right: RepresentationKind) -> bool {
    matches!(
        (left, right),
        (
            RepresentationKind::Heightfield,
            RepresentationKind::Heightfield
        ) | (RepresentationKind::Voxel, RepresentationKind::Voxel)
            | (RepresentationKind::Sdf, RepresentationKind::Sdf)
    )
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
        assert_eq!(
            RepresentationChoice::new(
                RepresentationSpec::Heightfield(heightfield),
                RepresentationSpec::Sdf(sdf)
            )
            .conversion_required,
            true
        );
        assert_eq!(
            RepresentationChoice::new(
                RepresentationSpec::Voxel(voxel),
                RepresentationSpec::Voxel(voxel)
            )
            .conversion_required,
            false
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
