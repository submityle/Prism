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
    /// A 2:1 LOD edge uses a sample count that cannot map exact midpoints.
    InvalidLodResolution {
        /// Edge sample count on both tiles.
        samples: u32,
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
            Self::InvalidLodResolution { samples } => write!(
                f,
                "2:1 heightfield LOD edge requires an odd sample count, got {samples}"
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

/// Validates coincident decoded heights where a fine tile meets a 2:1 coarse tile.
///
/// Only the fine edge's even-indexed samples coincide with coarse samples and
/// are compared. Child-half and cube-face orientation are derived from integer
/// tile adjacency; no floating-point position comparisons are used. The full
/// comparison count is admitted before reading either edge.
pub fn validate_planet_heightfield_lod_edge(
    fine: &HeightfieldTile,
    edge: TileEdge,
    coarse: &HeightfieldTile,
    max_comparisons: u64,
) -> Result<(), PlanetHeightfieldBoundaryError> {
    let side = fine.spec().samples_per_edge();
    if side != coarse.spec().samples_per_edge() {
        return Err(PlanetHeightfieldBoundaryError::ResolutionMismatch {
            left: side,
            right: coarse.spec().samples_per_edge(),
        });
    }
    let segments = side.saturating_sub(1);
    if segments < 2 || !segments.is_multiple_of(2) {
        return Err(PlanetHeightfieldBoundaryError::InvalidLodResolution { samples: side });
    }
    let (fine_neighbor, _, _) = fine
        .tile()
        .neighbor_edge_index(edge, side, 0)
        .ok_or(PlanetHeightfieldBoundaryError::TileMismatch)?;
    let fine_parent = fine
        .tile()
        .parent()
        .ok_or(PlanetHeightfieldBoundaryError::TileMismatch)?;
    let (expected_coarse, coarse_edge) = fine_parent.neighbor_with_edge(edge);
    if expected_coarse != coarse.tile() {
        return Err(PlanetHeightfieldBoundaryError::TileMismatch);
    }
    let required = u64::from(segments / 2 + 1);
    if required > max_comparisons {
        return Err(PlanetHeightfieldBoundaryError::ComparisonBudgetExceeded {
            required,
            limit: max_comparisons,
        });
    }
    let child_half = match coarse_edge {
        TileEdge::West | TileEdge::East => fine_neighbor.y() % 2,
        TileEdge::South | TileEdge::North => fine_neighbor.x() % 2,
    };
    for fine_index in (0..=segments).step_by(2) {
        let (mapped_neighbor, mapped_edge, mapped_index) = fine
            .tile()
            .neighbor_edge_index(edge, side, fine_index)
            .ok_or(PlanetHeightfieldBoundaryError::TileMismatch)?;
        if mapped_neighbor != fine_neighbor || mapped_edge != coarse_edge {
            return Err(PlanetHeightfieldBoundaryError::TileMismatch);
        }
        let coarse_index = child_half * (segments / 2) + mapped_index / 2;
        let fine_mm = fine
            .height_mm(edge_coordinate(edge, side, fine_index))
            .ok_or(PlanetHeightfieldBoundaryError::TileMismatch)?;
        let coarse_mm = coarse
            .height_mm(edge_coordinate(coarse_edge, side, coarse_index))
            .ok_or(PlanetHeightfieldBoundaryError::TileMismatch)?;
        if fine_mm != coarse_mm {
            return Err(PlanetHeightfieldBoundaryError::HeightMismatch {
                index: fine_index,
                left_mm: fine_mm,
                right_mm: coarse_mm,
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
    use alloc::vec::Vec;
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

    struct LodEdgeOracle {
        coarse: TileKey,
        coarse_edge: TileEdge,
        fine: TileKey,
        fine_edge: TileEdge,
        side: u32,
    }

    impl HeightSampler for LodEdgeOracle {
        type Error = Infallible;

        fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            let node = |value: u16| {
                ((u64::from(value) * u64::from(self.side - 1) + 32_767) / 65_535) as u32
            };
            let edge_index = |edge: TileEdge| match edge {
                TileEdge::West | TileEdge::East => node(sample.v_q16()),
                TileEdge::South | TileEdge::North => node(sample.u_q16()),
            };
            let on_edge = |edge: TileEdge| match edge {
                TileEdge::West => sample.u_q16() == 0,
                TileEdge::East => sample.u_q16() == u16::MAX,
                TileEdge::South => sample.v_q16() == 0,
                TileEdge::North => sample.v_q16() == u16::MAX,
            };
            let coarse_value = |index: u32| 1_000 + i64::from(index) * 37;

            if sample.tile() == self.coarse && on_edge(self.coarse_edge) {
                return Ok(HeightSampleValue::from_millimetres(coarse_value(
                    edge_index(self.coarse_edge),
                )));
            }
            if sample.tile() == self.fine && on_edge(self.fine_edge) {
                let fine_index = edge_index(self.fine_edge);
                if fine_index % 2 == 0 {
                    let (same_level, coarse_edge, mapped_index) = self
                        .fine
                        .neighbor_edge_index(self.fine_edge, self.side, fine_index)
                        .expect("oracle boundary mapping");
                    let child_half = match coarse_edge {
                        TileEdge::West | TileEdge::East => same_level.y() % 2,
                        TileEdge::South | TileEdge::North => same_level.x() % 2,
                    };
                    return Ok(HeightSampleValue::from_millimetres(coarse_value(
                        child_half * ((self.side - 1) / 2) + mapped_index / 2,
                    )));
                }
            }
            Ok(HeightSampleValue::from_millimetres(
                i64::from(sample.u_q16() / 4_096) + i64::from(sample.v_q16() / 4_096),
            ))
        }
    }

    fn lod_edge_tiles(coarse: TileKey, edge: TileEdge) -> (TileKey, TileEdge, TileKey) {
        let (fine_parent, fine_edge) = coarse.neighbor_with_edge(edge);
        let fine = fine_parent
            .children()
            .unwrap()
            .into_iter()
            .find(|child| match fine_edge {
                TileEdge::West => child.x() % 2 == 0,
                TileEdge::East => child.x() % 2 == 1,
                TileEdge::South => child.y() % 2 == 0,
                TileEdge::North => child.y() % 2 == 1,
            })
            .unwrap();
        (fine, fine_edge, fine_parent)
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

    #[test]
    fn coarser_neighbor_is_crossed_from_owner_parent_not_same_level_neighbor_parent() {
        let owner = TileKey::new(CubeFace::PositiveZ, 4, 4, 4).unwrap();
        let edge = TileEdge::North;
        let same_level = owner.neighbor(edge);
        let incorrect = same_level.parent().unwrap();
        let correct = owner.coarser_neighbor(edge).unwrap();
        assert_ne!(incorrect, correct);

        let fine = tile(owner, 7);
        let wrong_coarse = tile(incorrect, 7);
        assert_eq!(
            validate_planet_heightfield_lod_edge(&fine, edge, &wrong_coarse, u64::MAX),
            Err(PlanetHeightfieldBoundaryError::TileMismatch)
        );
        let coarse = tile(correct, 7);
        validate_planet_heightfield_lod_edge(&fine, edge, &coarse, u64::MAX).unwrap();
    }

    #[test]
    fn lod_edges_validate_coincident_samples_across_faces_and_child_halves() {
        for face in CubeFace::ALL {
            for edge in TileEdge::ALL {
                let coarse = TileKey::new(face, 2, 1, 1).unwrap();
                let fine_parent = coarse.neighbor(edge);
                let fine_edge = coarse.neighbor_with_edge(edge).1;
                let children: Vec<_> = fine_parent
                    .children()
                    .unwrap()
                    .into_iter()
                    .filter(|child| match fine_edge {
                        TileEdge::West => child.x() % 2 == 0,
                        TileEdge::East => child.x() % 2 == 1,
                        TileEdge::South => child.y() % 2 == 0,
                        TileEdge::North => child.y() % 2 == 1,
                    })
                    .collect();
                assert_eq!(children.len(), 2, "{face:?} {edge:?}");
                let coarse_field = tile(coarse, 7);
                for child in children {
                    assert_eq!(child.coarser_neighbor(fine_edge), Some(coarse));
                    let fine_field = tile(child, 7);
                    validate_planet_heightfield_lod_edge(&fine_field, fine_edge, &coarse_field, 3)
                        .unwrap_or_else(|error| panic!("{face:?} {edge:?} {child:?}: {error:?}"));
                }
            }
        }
    }

    #[test]
    fn lod_edges_match_nonconstant_oracle_with_reversed_cross_face_mapping() {
        let mut checked_reversed_cross_face = false;
        for face in CubeFace::ALL {
            for edge in TileEdge::ALL {
                let (x, y) = match edge {
                    TileEdge::West => (0, 1),
                    TileEdge::East => (3, 1),
                    TileEdge::South => (1, 0),
                    TileEdge::North => (1, 3),
                };
                let coarse = TileKey::new(face, 2, x, y).unwrap();
                let (fine, fine_edge, fine_parent) = lod_edge_tiles(coarse, edge);
                let coarse_edge = fine_parent.neighbor_with_edge(fine_edge).1;
                let coarse_field = HeightfieldTile::bake(
                    coarse,
                    HeightfieldSpec::new(5, 1).unwrap(),
                    &LodEdgeOracle {
                        coarse,
                        coarse_edge,
                        fine,
                        fine_edge,
                        side: 5,
                    },
                    BakeLimits {
                        max_samples: 25,
                        max_bytes: 50,
                    },
                )
                .unwrap();
                let fine_field = HeightfieldTile::bake(
                    fine,
                    HeightfieldSpec::new(5, 1).unwrap(),
                    &LodEdgeOracle {
                        coarse,
                        coarse_edge,
                        fine,
                        fine_edge,
                        side: 5,
                    },
                    BakeLimits {
                        max_samples: 25,
                        max_bytes: 50,
                    },
                )
                .unwrap();
                validate_planet_heightfield_lod_edge(&fine_field, fine_edge, &coarse_field, 3)
                    .unwrap_or_else(|error| panic!("{face:?} {edge:?}: {error:?}"));
                if coarse.face() != fine.face()
                    && fine.neighbor_edge_index(fine_edge, 5, 0).unwrap().2 != 0
                {
                    checked_reversed_cross_face = true;
                }
            }
        }
        assert!(checked_reversed_cross_face);
    }

    #[test]
    fn lod_edge_mismatch_and_budget_fail_closed() {
        let coarse = TileKey::new(CubeFace::PositiveZ, 2, 1, 1).unwrap();
        let fine_parent = coarse.neighbor(TileEdge::East);
        let fine = fine_parent.children().unwrap()[0];
        let coarse_field = tile(coarse, 0);
        let fine_field = tile(fine, 1);
        assert_eq!(
            validate_planet_heightfield_lod_edge(&fine_field, TileEdge::West, &coarse_field, 2),
            Err(PlanetHeightfieldBoundaryError::ComparisonBudgetExceeded {
                required: 3,
                limit: 2,
            })
        );
        assert!(matches!(
            validate_planet_heightfield_lod_edge(&fine_field, TileEdge::West, &coarse_field, 3),
            Err(PlanetHeightfieldBoundaryError::HeightMismatch { .. })
        ));
        assert_eq!(
            validate_planet_heightfield_lod_edge(&fine_field, TileEdge::East, &coarse_field, 3),
            Err(PlanetHeightfieldBoundaryError::TileMismatch)
        );
    }
}
