//! Deterministic placement on a cube-sphere heightfield tile.
//!
//! The kernel owns only candidate identity, face projection, height sampling,
//! and bounded output. It does not create renderer resources, persistence, or a
//! second scene truth. Cell centers are used so adjacent tiles have disjoint
//! ownership while sharing the same cube-sphere direction mapping.

use alloc::vec::Vec;
use core::fmt;

use crate::{CubeFace, HeightfieldTile, TileKey};

/// Revision of the planetary placement kernel.
pub const PLANET_PLACEMENT_VERSION: u32 = 1;

/// Deterministic recipe for one planetary placement request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanetPlacementConfig {
    /// Domain seed.
    pub seed: u64,
    /// Caller-owned recipe revision.
    pub version: u32,
    /// Stream namespace.
    pub salt: u64,
    /// Spherical reference radius in millimetres.
    pub radius_mm: i64,
    /// Integer origin of the planet in world millimetres.
    pub origin_mm: [i64; 3],
    /// Radial offset added to the sampled terrain height.
    pub vertical_offset_mm: i64,
    /// Inclusive UNORM16 acceptance threshold. `u16::MAX` accepts all cells.
    pub acceptance_threshold: u16,
}

impl PlanetPlacementConfig {
    fn validate(self) -> Result<(), PlanetPlacementError> {
        if !(1..=1_i64 << 52).contains(&self.radius_mm) {
            return Err(PlanetPlacementError::InvalidConfig);
        }
        Ok(())
    }
}

/// Half-open local cell ownership inside one heightfield tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanetPlacementRegion {
    /// Terrain tile providing the height samples.
    pub tile: TileKey,
    /// Inclusive local cell minimum `[u, v]`.
    pub min_cell: [u32; 2],
    /// Exclusive local cell maximum `[u, v]`.
    pub max_cell: [u32; 2],
}

impl PlanetPlacementRegion {
    /// Creates a validated half-open cell region.
    pub fn new(
        tile: TileKey,
        min_cell: [u32; 2],
        max_cell: [u32; 2],
    ) -> Result<Self, PlanetPlacementError> {
        let region = Self {
            tile,
            min_cell,
            max_cell,
        };
        region.validate(u32::MAX)?;
        Ok(region)
    }

    fn validate(self, cell_side: u32) -> Result<(), PlanetPlacementError> {
        if self.min_cell[0] > self.max_cell[0]
            || self.min_cell[1] > self.max_cell[1]
            || self.max_cell.iter().any(|&value| value > cell_side)
        {
            return Err(PlanetPlacementError::InvalidRegion);
        }
        Ok(())
    }

    fn count(self) -> Result<u64, PlanetPlacementError> {
        let width = u64::from(self.max_cell[0] - self.min_cell[0]);
        let height = u64::from(self.max_cell[1] - self.min_cell[1]);
        width
            .checked_mul(height)
            .ok_or(PlanetPlacementError::ArithmeticOverflow)
    }
}

/// Collision-free identity of a planetary placement candidate.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PlanetPlacementKey {
    /// Placement kernel revision.
    pub placement_version: u32,
    /// Candidate seed and namespace.
    pub seed: u64,
    /// Caller recipe revision.
    pub version: u32,
    /// Candidate namespace.
    pub salt: u64,
    /// Owning cube-sphere tile.
    pub tile: TileKey,
    /// Local cell coordinate.
    pub cell: [u32; 2],
}

/// One accepted planetary placement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanetPlacementPoint {
    /// Stable candidate identity.
    pub key: PlanetPlacementKey,
    /// Absolute world position in millimetres.
    pub position_mm: [i64; 3],
    /// Outward unit sphere normal.
    pub normal: [f32; 3],
    /// Exact bilinear terrain height numerator in millimetres.
    pub surface_height_numerator_mm: i128,
    /// Denominator for `surface_height_numerator_mm` (always four).
    pub surface_height_denominator: u8,
}

/// Bounded admission for one planetary placement request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanetPlacementBudget {
    /// Maximum candidate cells enumerated.
    pub max_candidate_evaluations: u64,
    /// Maximum accepted points.
    pub max_output_points: usize,
    /// Maximum reserved output payload bytes.
    pub max_output_bytes: usize,
}

/// Complete deterministic placement output.
#[derive(Debug, PartialEq)]
pub struct PlanetPlacementOutput {
    /// Accepted placements in local cell order.
    pub points: Vec<PlanetPlacementPoint>,
    /// Complete recipe and ownership inputs for cache validation.
    pub config: PlanetPlacementConfig,
    /// Owned region.
    pub region: PlanetPlacementRegion,
    /// Actual candidate evaluations.
    pub candidate_evaluations: u64,
}

/// Placement validation, arithmetic, allocation, or budget failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanetPlacementError {
    /// Invalid radius or unsupported numeric configuration.
    InvalidConfig,
    /// Inverted or out-of-range cell region.
    InvalidRegion,
    /// The heightfield belongs to another tile or has too few samples.
    TileMismatch,
    /// Checked arithmetic overflow.
    ArithmeticOverflow,
    /// A projected world coordinate cannot be represented by `i64` millimetres.
    CoordinateOverflow,
    /// Candidate work exceeds the supplied limit.
    CandidateBudgetExceeded {
        /// Required candidate evaluations.
        required: u64,
        /// Supplied candidate budget.
        limit: u64,
    },
    /// Accepted output count exceeds the supplied limit.
    OutputCountBudgetExceeded {
        /// Required accepted points.
        required: usize,
        /// Supplied output-count budget.
        limit: usize,
    },
    /// Reserved output payload exceeds the supplied limit.
    OutputBytesBudgetExceeded {
        /// Required reserved payload bytes.
        required: usize,
        /// Supplied output-byte budget.
        limit: usize,
    },
    /// A fallible output reservation failed.
    AllocationFailed,
    /// A non-positive radial distance was produced by height or offset.
    InvalidRadialDistance,
}

impl fmt::Display for PlanetPlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => f.write_str("invalid planetary placement configuration"),
            Self::InvalidRegion => f.write_str("invalid planetary placement cell region"),
            Self::TileMismatch => f.write_str("heightfield tile does not match placement region"),
            Self::ArithmeticOverflow => f.write_str("planetary placement arithmetic overflow"),
            Self::CoordinateOverflow => f.write_str("planetary placement coordinate overflow"),
            Self::CandidateBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "planetary placement requires {required} candidates, budget {limit}"
                )
            }
            Self::OutputCountBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "planetary placement requires {required} points, budget {limit}"
                )
            }
            Self::OutputBytesBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "planetary placement requires {required} bytes, budget {limit}"
                )
            }
            Self::AllocationFailed => f.write_str("planetary placement output allocation failed"),
            Self::InvalidRadialDistance => {
                f.write_str("planetary placement radial distance is non-positive")
            }
        }
    }
}

impl core::error::Error for PlanetPlacementError {}

/// Returns an upper bound for the number of local candidate cells.
pub fn estimate_planet_placement_work(
    region: PlanetPlacementRegion,
    tile: &HeightfieldTile,
) -> Result<u64, PlanetPlacementError> {
    if region.tile != tile.tile() {
        return Err(PlanetPlacementError::TileMismatch);
    }
    let side = tile.spec().samples_per_edge();
    let cell_side = side
        .checked_sub(1)
        .ok_or(PlanetPlacementError::TileMismatch)?;
    region.validate(cell_side)?;
    region.count()
}

/// Projects deterministic cell-center candidates onto a cube-sphere heightfield.
///
/// The request is preflighted before allocation or enumeration. Cube-face
/// mapping is evaluated in `f64`, height interpolation uses the four source
/// corners, and output coordinates are rounded to integer millimetres with ties
/// away from zero. No partial output is returned on any failure.
pub fn place_on_planet_heightfield(
    config: PlanetPlacementConfig,
    region: PlanetPlacementRegion,
    tile: &HeightfieldTile,
    budget: PlanetPlacementBudget,
) -> Result<PlanetPlacementOutput, PlanetPlacementError> {
    config.validate()?;
    let work = estimate_planet_placement_work(region, tile)?;
    if work > budget.max_candidate_evaluations {
        return Err(PlanetPlacementError::CandidateBudgetExceeded {
            required: work,
            limit: budget.max_candidate_evaluations,
        });
    }
    let capacity = usize::try_from(work)
        .map_err(|_| PlanetPlacementError::ArithmeticOverflow)?
        .min(budget.max_output_points);
    let bytes = capacity
        .checked_mul(core::mem::size_of::<PlanetPlacementPoint>())
        .ok_or(PlanetPlacementError::ArithmeticOverflow)?;
    if bytes > budget.max_output_bytes {
        return Err(PlanetPlacementError::OutputBytesBudgetExceeded {
            required: bytes,
            limit: budget.max_output_bytes,
        });
    }
    let mut points = Vec::new();
    points
        .try_reserve_exact(capacity)
        .map_err(|_| PlanetPlacementError::AllocationFailed)?;

    let mut candidate_evaluations = 0_u64;
    for v in region.min_cell[1]..region.max_cell[1] {
        for u in region.min_cell[0]..region.max_cell[0] {
            candidate_evaluations = candidate_evaluations
                .checked_add(1)
                .ok_or(PlanetPlacementError::ArithmeticOverflow)?;
            let key = PlanetPlacementKey {
                placement_version: PLANET_PLACEMENT_VERSION,
                seed: config.seed,
                version: config.version,
                salt: config.salt,
                tile: region.tile,
                cell: [u, v],
            };
            if rank(key) > config.acceptance_threshold {
                continue;
            }
            let point = project(config, key, tile)?;
            let required = points
                .len()
                .checked_add(1)
                .ok_or(PlanetPlacementError::ArithmeticOverflow)?;
            if required > budget.max_output_points {
                return Err(PlanetPlacementError::OutputCountBudgetExceeded {
                    required,
                    limit: budget.max_output_points,
                });
            }
            points.push(point);
        }
    }
    Ok(PlanetPlacementOutput {
        points,
        config,
        region,
        candidate_evaluations,
    })
}

fn project(
    config: PlanetPlacementConfig,
    key: PlanetPlacementKey,
    tile: &HeightfieldTile,
) -> Result<PlanetPlacementPoint, PlanetPlacementError> {
    let side = tile.spec().samples_per_edge();
    let cells = f64::from(side - 1);
    let face_tiles = f64::from(key.tile.side_len());
    let u = -1.0
        + 2.0 * (f64::from(key.tile.x()) + (f64::from(key.cell[0]) + 0.5) / cells) / face_tiles;
    let v = -1.0
        + 2.0 * (f64::from(key.tile.y()) + (f64::from(key.cell[1]) + 0.5) / cells) / face_tiles;
    let cube = match key.tile.face() {
        CubeFace::PositiveX => [1.0, v, -u],
        CubeFace::NegativeX => [-1.0, v, u],
        CubeFace::PositiveY => [u, 1.0, -v],
        CubeFace::NegativeY => [u, -1.0, v],
        CubeFace::PositiveZ => [u, v, 1.0],
        CubeFace::NegativeZ => [-u, v, -1.0],
    };
    let length = libm::sqrt(cube[0] * cube[0] + cube[1] * cube[1] + cube[2] * cube[2]);
    if !length.is_finite() || length == 0.0 {
        return Err(PlanetPlacementError::InvalidConfig);
    }
    let normal64 = [cube[0] / length, cube[1] / length, cube[2] / length];
    let x = key.cell[0];
    let y = key.cell[1];
    let a = tile
        .height_mm([x, y])
        .ok_or(PlanetPlacementError::TileMismatch)?;
    let b = tile
        .height_mm([x + 1, y])
        .ok_or(PlanetPlacementError::TileMismatch)?;
    let c = tile
        .height_mm([x, y + 1])
        .ok_or(PlanetPlacementError::TileMismatch)?;
    let d = tile
        .height_mm([x + 1, y + 1])
        .ok_or(PlanetPlacementError::TileMismatch)?;
    let height_numerator = a
        .checked_add(b)
        .and_then(|value| value.checked_add(c))
        .and_then(|value| value.checked_add(d))
        .ok_or(PlanetPlacementError::ArithmeticOverflow)?;
    let radial_numerator = i128::from(config.radius_mm)
        .checked_add(i128::from(config.vertical_offset_mm))
        .and_then(|value| value.checked_mul(4))
        .and_then(|value| value.checked_add(height_numerator))
        .ok_or(PlanetPlacementError::ArithmeticOverflow)?;
    if radial_numerator <= 0 {
        return Err(PlanetPlacementError::InvalidRadialDistance);
    }
    let radial = radial_numerator as f64 / 4.0;
    let mut position_mm = [0_i64; 3];
    for axis in 0..3 {
        let value = config.origin_mm[axis] as f64 + normal64[axis] * radial;
        position_mm[axis] = round_i64(value)?;
    }
    let normal = [normal64[0] as f32, normal64[1] as f32, normal64[2] as f32];
    if normal.iter().any(|value| !value.is_finite()) {
        return Err(PlanetPlacementError::InvalidConfig);
    }
    Ok(PlanetPlacementPoint {
        key,
        position_mm,
        normal,
        surface_height_numerator_mm: height_numerator,
        surface_height_denominator: 4,
    })
}

fn round_i64(value: f64) -> Result<i64, PlanetPlacementError> {
    if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(PlanetPlacementError::CoordinateOverflow);
    }
    let rounded = if value < 0.0 {
        libm::ceil(value - 0.5)
    } else {
        libm::floor(value + 0.5)
    };
    if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return Err(PlanetPlacementError::CoordinateOverflow);
    }
    Ok(rounded as i64)
}

fn rank(key: PlanetPlacementKey) -> u16 {
    let mut value = key.seed ^ key.salt;
    value ^= u64::from(key.placement_version) << 32 | u64::from(key.version);
    value = mix(value ^ u64::from(key.tile.face().index()));
    value = mix(value ^ u64::from(key.tile.level()));
    value = mix(value ^ u64::from(key.tile.x()));
    value = mix(value ^ u64::from(key.tile.y()));
    value = mix(value ^ u64::from(key.cell[0]));
    value = mix(value ^ u64::from(key.cell[1]));
    (value >> 48) as u16
}

fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::Infallible;

    use crate::{BakeLimits, HeightSample, HeightSampleValue, HeightSampler, HeightfieldSpec};

    struct Constant(i64);
    impl HeightSampler for Constant {
        type Error = Infallible;

        fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            Ok(HeightSampleValue::from_millimetres(self.0))
        }
    }

    fn tile(face: CubeFace, level: u8, x: u32, y: u32, height: i64) -> HeightfieldTile {
        HeightfieldTile::bake(
            TileKey::new(face, level, x, y).unwrap(),
            HeightfieldSpec::new(9, 1).unwrap(),
            &Constant(height),
            BakeLimits {
                max_samples: 81,
                max_bytes: 162,
            },
        )
        .unwrap()
    }

    fn config() -> PlanetPlacementConfig {
        PlanetPlacementConfig {
            seed: 7,
            version: 1,
            salt: 9,
            radius_mm: 6_371_000_000,
            origin_mm: [0, 0, 0],
            vertical_offset_mm: 0,
            acceptance_threshold: u16::MAX,
        }
    }

    #[test]
    fn all_faces_produce_outward_finite_positions() {
        for face in CubeFace::ALL {
            let source = tile(face, 0, 0, 0, 100);
            let region = PlanetPlacementRegion::new(source.tile(), [0, 0], [8, 8]).unwrap();
            let output = place_on_planet_heightfield(
                config(),
                region,
                &source,
                PlanetPlacementBudget {
                    max_candidate_evaluations: 64,
                    max_output_points: 64,
                    max_output_bytes: 64 * core::mem::size_of::<PlanetPlacementPoint>(),
                },
            )
            .unwrap();
            assert_eq!(output.points.len(), 64);
            for point in output.points {
                assert!(point.position_mm.iter().all(|value| *value != i64::MIN));
                let dot = point
                    .position_mm
                    .iter()
                    .zip(point.normal)
                    .map(|(position, normal)| *position as f32 * normal)
                    .sum::<f32>();
                assert!(dot > 0.0);
            }
        }
    }

    #[test]
    fn adjacent_tiles_have_disjoint_deterministic_ownership() {
        let a = tile(CubeFace::PositiveZ, 1, 0, 0, 0);
        let b = tile(CubeFace::PositiveZ, 1, 1, 0, 0);
        let ca = config();
        let ra = PlanetPlacementRegion::new(a.tile(), [0, 0], [8, 8]).unwrap();
        let rb = PlanetPlacementRegion::new(b.tile(), [0, 0], [8, 8]).unwrap();
        let budget = PlanetPlacementBudget {
            max_candidate_evaluations: 64,
            max_output_points: 64,
            max_output_bytes: 64 * core::mem::size_of::<PlanetPlacementPoint>(),
        };
        let first = place_on_planet_heightfield(ca, ra, &a, budget).unwrap();
        let second = place_on_planet_heightfield(ca, rb, &b, budget).unwrap();
        assert!(first.points.iter().all(|left| {
            second
                .points
                .iter()
                .all(|right| left.key != right.key && left.position_mm != right.position_mm)
        }));
        assert_eq!(
            first,
            place_on_planet_heightfield(ca, ra, &a, budget).unwrap()
        );
    }

    #[test]
    fn budget_and_overflow_fail_before_partial_output() {
        let source = tile(CubeFace::PositiveZ, 0, 0, 0, 0);
        let region = PlanetPlacementRegion::new(source.tile(), [0, 0], [8, 8]).unwrap();
        let mut limited = config();
        limited.acceptance_threshold = 0;
        assert_eq!(
            place_on_planet_heightfield(
                limited,
                region,
                &source,
                PlanetPlacementBudget {
                    max_candidate_evaluations: 63,
                    max_output_points: 64,
                    max_output_bytes: usize::MAX,
                }
            ),
            Err(PlanetPlacementError::CandidateBudgetExceeded {
                required: 64,
                limit: 63
            })
        );
        let mut invalid = config();
        invalid.vertical_offset_mm = -invalid.radius_mm;
        assert_eq!(
            place_on_planet_heightfield(
                invalid,
                PlanetPlacementRegion::new(source.tile(), [0, 0], [1, 1]).unwrap(),
                &source,
                PlanetPlacementBudget {
                    max_candidate_evaluations: 1,
                    max_output_points: 1,
                    max_output_bytes: usize::MAX,
                }
            ),
            Err(PlanetPlacementError::InvalidRadialDistance)
        );
    }
}
