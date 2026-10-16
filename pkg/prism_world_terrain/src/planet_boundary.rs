//! Exact shared-edge validation for cube-sphere heightfield tiles.

use core::fmt;

use crate::{HeightfieldTile, TileEdge};

/// Errors returned when validating one shared planetary tile edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanetHeightfieldBoundaryError {
    /// The right tile is not the declared neighbor of the left tile.
    TileMismatch,
    /// The two tiles do not have the same number of edge samples.
    ResolutionMismatch {
        /// Left edge sample count.
        left: u32,
        /// Right edge sample count.
        right: u32,
    },
    /// The complete edge exceeds the supplied comparison budget.
    ComparisonBudgetExceeded {
        /// Required edge comparisons.
        required: u64,
        /// Supplied comparison budget.
        limit: u64,
    },
    /// One decoded edge height differs from its counterpart.
    HeightMismatch {
        /// Edge sample index in the left tile direction.
        index: u32,
        /// Left decoded height in millimetres.
        left_mm: i128,
        /// Right decoded height in millimetres.
        right_mm: i128,
    },
}

impl fmt::Display for PlanetHeightfieldBoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TileMismatch => f.write_str("heightfield tiles are not adjacent"),
            Self::ResolutionMismatch { left, right } => {
                write!(f, "heightfield edge resolutions differ: {left} vs {right}")
            }
            Self::ComparisonBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "heightfield edge needs {required} comparisons, limit {limit}"
                )
            }
            Self::HeightMismatch {
                index,
                left_mm,
                right_mm,
            } => write!(
                f,
                "heightfield edge sample {index} differs: {left_mm} vs {right_mm} mm"
            ),
        }
    }
}

impl core::error::Error for PlanetHeightfieldBoundaryError {}

/// Validates exact decoded heights along one shared cube-sphere edge.
///
/// Cross-face orientation is resolved from [`TileEdge`] metadata rather than
/// floating-point direction comparisons. The whole edge is budgeted before
/// sampling, and the first mismatch fails closed without publishing state.
pub fn validate_planet_heightfield_edge(
    left: &HeightfieldTile,
    edge: TileEdge,
    right: &HeightfieldTile,
    max_comparisons: u64,
) -> Result<(), PlanetHeightfieldBoundaryError> {
    let (expected_tile, _, _) = left
        .tile()
        .neighbor_edge_index(edge, left.spec().samples_per_edge(), 0)
        .ok_or(PlanetHeightfieldBoundaryError::TileMismatch)?;
    if expected_tile != right.tile() {
        return Err(PlanetHeightfieldBoundaryError::TileMismatch);
    }
    let left_side = left.spec().samples_per_edge();
    let right_side = right.spec().samples_per_edge();
    if left_side != right_side {
        return Err(PlanetHeightfieldBoundaryError::ResolutionMismatch {
            left: left_side,
            right: right_side,
        });
    }
    let required = u64::from(left_side);
    if required > max_comparisons {
        return Err(PlanetHeightfieldBoundaryError::ComparisonBudgetExceeded {
            required,
            limit: max_comparisons,
        });
    }
    for index in 0..left_side {
        let left_coordinate = edge_coordinate(edge, left_side, index);
        let (_, right_edge, right_index) = left
            .tile()
            .neighbor_edge_index(edge, left_side, index)
            .ok_or(PlanetHeightfieldBoundaryError::TileMismatch)?;
        let right_coordinate = edge_coordinate(right_edge, right_side, right_index);
        let left_mm = left
            .height_mm(left_coordinate)
            .ok_or(PlanetHeightfieldBoundaryError::TileMismatch)?;
        let right_mm = right
            .height_mm(right_coordinate)
            .ok_or(PlanetHeightfieldBoundaryError::TileMismatch)?;
        if left_mm != right_mm {
            return Err(PlanetHeightfieldBoundaryError::HeightMismatch {
                index,
                left_mm,
                right_mm,
            });
        }
    }
    Ok(())
}

fn edge_coordinate(edge: TileEdge, side: u32, index: u32) -> [u32; 2] {
    match edge {
        TileEdge::West => [0, index],
        TileEdge::East => [side - 1, index],
        TileEdge::South => [index, 0],
        TileEdge::North => [index, side - 1],
    }
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use super::*;
    use crate::{
        BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler, HeightfieldSpec,
        TileKey,
    };

    struct Constant(i64);
    impl HeightSampler for Constant {
        type Error = Infallible;

        fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            Ok(HeightSampleValue::from_millimetres(self.0))
        }
    }

    fn tile(tile: TileKey, height: i64) -> HeightfieldTile {
        HeightfieldTile::bake(
            tile,
            HeightfieldSpec::new(5, 1).unwrap(),
            &Constant(height),
            BakeLimits {
                max_samples: 25,
                max_bytes: 50,
            },
        )
        .unwrap()
    }

    #[test]
    fn cross_face_and_reversed_edges_are_checked_exactly() {
        let left_key = TileKey::new(CubeFace::PositiveX, 1, 0, 0).unwrap();
        let right_key = left_key.neighbor(TileEdge::West);
        assert_eq!(
            left_key
                .neighbor_edge_index(TileEdge::West, 5, 0)
                .unwrap()
                .0,
            right_key
        );
        let left = tile(left_key, 17);
        let right = tile(right_key, 17);
        validate_planet_heightfield_edge(&left, TileEdge::West, &right, 5).unwrap();
    }

    #[test]
    fn mismatch_resolution_budget_and_height_fail_closed() {
        let left_key = TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap();
        let right_key = left_key.neighbor(TileEdge::East);
        let left = tile(left_key, 17);
        let right = tile(right_key, 18);
        assert_eq!(
            validate_planet_heightfield_edge(&left, TileEdge::East, &right, 5),
            Err(PlanetHeightfieldBoundaryError::HeightMismatch {
                index: 0,
                left_mm: 17,
                right_mm: 18,
            })
        );
        assert!(matches!(
            validate_planet_heightfield_edge(&left, TileEdge::East, &right, 4),
            Err(PlanetHeightfieldBoundaryError::ComparisonBudgetExceeded { .. })
        ));
        let wrong = tile(TileKey::root(CubeFace::PositiveZ), 17);
        assert_eq!(
            validate_planet_heightfield_edge(&left, TileEdge::East, &wrong, 5),
            Err(PlanetHeightfieldBoundaryError::TileMismatch)
        );
    }
}
