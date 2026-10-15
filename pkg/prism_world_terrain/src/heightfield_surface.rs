//! Exact planar queries over an immutable, shared baked heightfield.
//!
//! The caller supplies the planar world embedding explicitly: a cube-sphere
//! tile key does not imply an XZ placement. Queries follow the same a-c-b /
//! b-c-d triangle topology as [`crate::HeightfieldMesh`], without constructing
//! a mesh, projecting scatter points, or converting world coordinates to floats.

use alloc::sync::Arc;
use core::fmt;

use crate::HeightfieldTile;

/// Revision of planar ownership, triangle selection, interpolation and normals.
pub const HEIGHTFIELD_SURFACE_VERSION: u32 = 1;

/// Axis-aligned planar placement of a square heightfield in world millimetres.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanarHeightfieldEmbedding {
    /// Position of sample `[0, 0]`; decoded sample heights add to its Y value.
    pub sample_origin_mm: [i64; 3],
    /// Positive spacing between adjacent samples on both the X and Z axes.
    pub sample_spacing_mm: u32,
}

/// Validation or ownership failure for a planar heightfield query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeightfieldQueryError {
    /// Sample spacing must be positive.
    InvalidSpacing,
    /// An exclusive XZ surface endpoint cannot be represented as `i64`.
    CoordinateOverflow,
    /// The queried XZ position is outside the half-open surface bounds.
    OutsideSurface,
}

impl fmt::Display for HeightfieldQueryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpacing => {
                formatter.write_str("heightfield sample spacing must be positive")
            }
            Self::CoordinateOverflow => {
                formatter.write_str("heightfield surface XZ endpoint exceeds i64 coordinates")
            }
            Self::OutsideSurface => {
                formatter.write_str("position is outside the half-open heightfield surface")
            }
        }
    }
}

impl core::error::Error for HeightfieldQueryError {}

/// Exact world height and geometric face normal at an integer XZ position.
///
/// World height is `height_numerator_mm() / height_denominator()` millimetres,
/// with no rounding or requirement that the height fit `i64`. The denominator
/// remains the sample spacing rather than being reduced by a common divisor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldSurfaceSample {
    height_numerator_mm: i128,
    height_denominator: u32,
    normal_mm: [i128; 3],
}

impl HeightfieldSurfaceSample {
    /// Numerator of the exact absolute world height in millimetres.
    pub const fn height_numerator_mm(self) -> i128 {
        self.height_numerator_mm
    }

    /// Positive height denominator, equal to the surface's sample spacing.
    pub const fn height_denominator(self) -> u32 {
        self.height_denominator
    }

    /// Upward, unnormalized geometric face normal `[-dx, spacing, -dz]`.
    ///
    /// `dx` and `dz` are decoded height differences along this triangle's X and
    /// Z edges. This is proportional to its cross-product normal, not the
    /// heightfield mesh's smooth vertex normal. Extreme components need not
    /// have squares representable in `u128`.
    pub const fn normal_mm(self) -> [i128; 3] {
        self.normal_mm
    }
}

/// Immutable baked heightfield snapshot with an explicit planar embedding.
///
/// Cloning shares the tile without copying its sample buffer. Equality compares
/// the embedding and complete tile contents, including samples and revision;
/// neither an Arc address nor tile key plus revision alone is content identity.
/// Retaining this surface also retains its snapshot's sample storage. Updating
/// terrain requires publishing a new snapshot, not mutating the shared tile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanarHeightfieldSurface {
    tile: Arc<HeightfieldTile>,
    embedding: PlanarHeightfieldEmbedding,
    max_xz_mm: [i64; 2],
}

impl PlanarHeightfieldSurface {
    /// Binds an existing snapshot without allocating or scanning its samples.
    ///
    /// Coverage is half-open on X and Z, from the sample origin to
    /// `origin + (samples_per_edge - 1) * spacing`. The exclusive endpoints must
    /// fit `i64`; decoded heights and world Y are deliberately not so limited.
    pub fn new(
        tile: Arc<HeightfieldTile>,
        embedding: PlanarHeightfieldEmbedding,
    ) -> Result<Self, HeightfieldQueryError> {
        if embedding.sample_spacing_mm == 0 {
            return Err(HeightfieldQueryError::InvalidSpacing);
        }
        let span = i128::from(tile.spec().samples_per_edge() - 1)
            * i128::from(embedding.sample_spacing_mm);
        let [origin_x, _, origin_z] = embedding.sample_origin_mm;
        let max_xz_mm = [
            i64::try_from(i128::from(origin_x) + span)
                .map_err(|_| HeightfieldQueryError::CoordinateOverflow)?,
            i64::try_from(i128::from(origin_z) + span)
                .map_err(|_| HeightfieldQueryError::CoordinateOverflow)?,
        ];
        Ok(Self {
            tile,
            embedding,
            max_xz_mm,
        })
    }

    /// Shared immutable source snapshot, retained by this surface.
    pub fn tile(&self) -> &Arc<HeightfieldTile> {
        &self.tile
    }

    /// Explicit world placement and integer sample spacing.
    pub const fn embedding(&self) -> PlanarHeightfieldEmbedding {
        self.embedding
    }

    /// Inclusive XZ coverage minimum in world millimetres.
    pub const fn min_xz_mm(&self) -> [i64; 2] {
        let [x, _, z] = self.embedding.sample_origin_mm;
        [x, z]
    }

    /// Exclusive XZ coverage maximum in world millimetres.
    pub const fn max_xz_mm(&self) -> [i64; 2] {
        self.max_xz_mm
    }

    /// Samples exact world height and the upward geometric triangle normal.
    ///
    /// Each successful query reads all four quad corners (four logical sample
    /// reads), uses only `i128` geometry arithmetic, and allocates nothing. At
    /// cell-local remainders `rx + rz <= spacing`, the a-c-b triangle wins,
    /// including its diagonal; otherwise the b-c-d triangle is selected. On an
    /// internal grid line the cell on the positive side owns the position.
    pub fn sample_xz_mm(
        &self,
        position_xz_mm: [i64; 2],
    ) -> Result<HeightfieldSurfaceSample, HeightfieldQueryError> {
        let min = self.min_xz_mm();
        if position_xz_mm
            .into_iter()
            .zip(min)
            .zip(self.max_xz_mm)
            .any(|((position, min), max)| position < min || position >= max)
        {
            return Err(HeightfieldQueryError::OutsideSurface);
        }

        let spacing = i128::from(self.embedding.sample_spacing_mm);
        let local_x = i128::from(position_xz_mm[0]) - i128::from(min[0]);
        let local_z = i128::from(position_xz_mm[1]) - i128::from(min[1]);
        // Bounds and positive spacing prove 0 <= cell < samples_per_edge - 1,
        // including origins at i64::MIN and differences larger than i64::MAX.
        let x = (local_x / spacing) as u32;
        let z = (local_z / spacing) as u32;
        let rx = local_x % spacing;
        let rz = local_z % spacing;
        let height = |coordinate| {
            self.tile
                .height_mm(coordinate)
                .ok_or(HeightfieldQueryError::OutsideSurface)
        };
        let a = height([x, z])?;
        let b = height([x + 1, z])?;
        let c = height([x, z + 1])?;
        let d = height([x + 1, z + 1])?;

        // Decoded i16 * positive i64 heights have magnitude <= 2^78. Each
        // triangle's nonnegative integer weights sum to spacing < 2^32, so
        // the height numerator has magnitude < 2^110. World Y * spacing has
        // magnitude < 2^95. Their sum and every normal component fit i128.
        let (local_height_numerator, dx, dz) = if rx + rz <= spacing {
            (a * (spacing - rx - rz) + b * rx + c * rz, b - a, c - a)
        } else {
            (
                b * (spacing - rz) + c * (spacing - rx) + d * (rx + rz - spacing),
                d - c,
                d - b,
            )
        };
        Ok(HeightfieldSurfaceSample {
            height_numerator_mm: i128::from(self.embedding.sample_origin_mm[1]) * spacing
                + local_height_numerator,
            height_denominator: self.embedding.sample_spacing_mm,
            normal_mm: [-dx, spacing, -dz],
        })
    }
}
