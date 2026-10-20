//! Face-local quadtree tile keys and cube-face adjacency.

use crate::{CubeFace, FaceUv};

/// Maximum supported quadtree level. A level has `2^level` tiles per edge.
pub const MAX_TILE_LEVEL: u8 = 30;

/// Errors returned when constructing a tile key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TileError {
    /// The level cannot be represented by the supported tile coordinate type.
    LevelOutOfRange,
    /// A tile coordinate is outside the level's face extent.
    CoordinateOutOfRange,
    /// A face coordinate could not be converted because it was invalid.
    InvalidFaceCoordinate,
    /// The tile is already at the maximum supported level.
    NoChildren,
}

/// The four edges of a face-local tile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TileEdge {
    /// Decreasing local U.
    West,
    /// Increasing local U.
    East,
    /// Decreasing local V.
    South,
    /// Increasing local V.
    North,
}

impl TileEdge {
    /// All edges in stable order.
    pub const ALL: [Self; 4] = [Self::West, Self::East, Self::South, Self::North];

    const fn opposite(self) -> Self {
        match self {
            Self::West => Self::East,
            Self::East => Self::West,
            Self::South => Self::North,
            Self::North => Self::South,
        }
    }
}

/// A child quadrant in face-local coordinates.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TileQuadrant {
    /// Lower-left child.
    SouthWest,
    /// Lower-right child.
    SouthEast,
    /// Upper-left child.
    NorthWest,
    /// Upper-right child.
    NorthEast,
}

/// A face-local quadtree tile key.
///
/// Ordering is stable serialization order: face index, level, x, then y.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TileKey {
    face: CubeFace,
    level: u8,
    x: u32,
    y: u32,
}

impl TileKey {
    /// Creates a validated tile key.
    pub fn new(face: CubeFace, level: u8, x: u32, y: u32) -> Result<Self, TileError> {
        let side = side_len(level)?;
        if x >= side || y >= side {
            return Err(TileError::CoordinateOutOfRange);
        }
        Ok(Self { face, level, x, y })
    }

    /// Creates the level-zero root tile for a face.
    pub const fn root(face: CubeFace) -> Self {
        Self {
            face,
            level: 0,
            x: 0,
            y: 0,
        }
    }

    /// Converts a face-local coordinate into a tile key at `level`.
    pub fn from_face_uv(face_uv: FaceUv, level: u8) -> Result<Self, TileError> {
        let side = side_len(level)? as f32;
        let x = (((face_uv.u() + 1.0) * 0.5 * side) as u32).min(side as u32 - 1);
        let y = (((face_uv.v() + 1.0) * 0.5 * side) as u32).min(side as u32 - 1);
        Self::new(face_uv.face(), level, x, y)
    }

    /// Returns the face.
    pub const fn face(self) -> CubeFace {
        self.face
    }

    /// Returns the quadtree level.
    pub const fn level(self) -> u8 {
        self.level
    }

    /// Returns the horizontal tile coordinate.
    pub const fn x(self) -> u32 {
        self.x
    }

    /// Returns the vertical tile coordinate.
    pub const fn y(self) -> u32 {
        self.y
    }

    /// Returns the number of tiles along one face edge at this level.
    pub const fn side_len(self) -> u32 {
        1_u32 << self.level
    }

    /// Returns the parent, or `None` for a root tile.
    pub const fn parent(self) -> Option<Self> {
        if self.level == 0 {
            None
        } else {
            Some(Self {
                face: self.face,
                level: self.level - 1,
                x: self.x / 2,
                y: self.y / 2,
            })
        }
    }

    /// Returns children in southwest, southeast, northwest, northeast order.
    pub fn children(self) -> Result<[Self; 4], TileError> {
        if self.level >= MAX_TILE_LEVEL {
            return Err(TileError::NoChildren);
        }
        let level = self.level + 1;
        let x = self.x * 2;
        let y = self.y * 2;
        Ok([
            Self {
                face: self.face,
                level,
                x,
                y,
            },
            Self {
                face: self.face,
                level,
                x: x + 1,
                y,
            },
            Self {
                face: self.face,
                level,
                x,
                y: y + 1,
            },
            Self {
                face: self.face,
                level,
                x: x + 1,
                y: y + 1,
            },
        ])
    }

    /// Returns the tile sharing the requested edge, including across cube faces.
    pub fn neighbor(self, edge: TileEdge) -> Self {
        self.neighbor_with_edge(edge).0
    }

    /// Returns the neighboring tile and the edge on that tile that faces this tile.
    pub fn neighbor_with_edge(self, edge: TileEdge) -> (Self, TileEdge) {
        let side = self.side_len();
        match edge {
            TileEdge::West if self.x > 0 => (
                Self {
                    x: self.x - 1,
                    ..self
                },
                edge.opposite(),
            ),
            TileEdge::East if self.x + 1 < side => (
                Self {
                    x: self.x + 1,
                    ..self
                },
                edge.opposite(),
            ),
            TileEdge::South if self.y > 0 => (
                Self {
                    y: self.y - 1,
                    ..self
                },
                edge.opposite(),
            ),
            TileEdge::North if self.y + 1 < side => (
                Self {
                    y: self.y + 1,
                    ..self
                },
                edge.opposite(),
            ),
            _ => {
                let transform = edge_transform(self.face, edge);
                let index = match edge {
                    TileEdge::West | TileEdge::East => self.y,
                    TileEdge::South | TileEdge::North => self.x,
                };
                let index = if transform.reversed {
                    side - 1 - index
                } else {
                    index
                };
                let (x, y) = match transform.edge {
                    TileEdge::West => (0, index),
                    TileEdge::East => (side - 1, index),
                    TileEdge::South => (index, 0),
                    TileEdge::North => (index, side - 1),
                };
                (
                    Self {
                        face: transform.face,
                        level: self.level,
                        x,
                        y,
                    },
                    transform.edge,
                )
            }
        }
    }

    /// Returns the neighbor and corresponding edge index for one boundary
    /// sample. `sample_side` is the number of samples on the edge; it is
    /// separate from the number of quadtree tiles along a face.
    pub fn neighbor_edge_index(
        self,
        edge: TileEdge,
        sample_side: u32,
        index: u32,
    ) -> Option<(Self, TileEdge, u32)> {
        let tile_side = self.side_len();
        if sample_side == 0 || index >= sample_side {
            return None;
        }
        match edge {
            TileEdge::West if self.x > 0 => Some((
                Self {
                    x: self.x - 1,
                    ..self
                },
                TileEdge::East,
                index,
            )),
            TileEdge::East if self.x + 1 < tile_side => Some((
                Self {
                    x: self.x + 1,
                    ..self
                },
                TileEdge::West,
                index,
            )),
            TileEdge::South if self.y > 0 => Some((
                Self {
                    y: self.y - 1,
                    ..self
                },
                TileEdge::North,
                index,
            )),
            TileEdge::North if self.y + 1 < tile_side => Some((
                Self {
                    y: self.y + 1,
                    ..self
                },
                TileEdge::South,
                index,
            )),
            _ => {
                let transform = edge_transform(self.face, edge);
                let tile_index = match edge {
                    TileEdge::West | TileEdge::East => self.y,
                    TileEdge::South | TileEdge::North => self.x,
                };
                let neighbor_tile_index = if transform.reversed {
                    tile_side - 1 - tile_index
                } else {
                    tile_index
                };
                let sample_index = if transform.reversed {
                    sample_side - 1 - index
                } else {
                    index
                };
                let (x, y) = match transform.edge {
                    TileEdge::West => (0, neighbor_tile_index),
                    TileEdge::East => (tile_side - 1, neighbor_tile_index),
                    TileEdge::South => (neighbor_tile_index, 0),
                    TileEdge::North => (neighbor_tile_index, tile_side - 1),
                };
                Some((
                    Self {
                        face: transform.face,
                        level: self.level,
                        x,
                        y,
                    },
                    transform.edge,
                    sample_index,
                ))
            }
        }
    }

    /// Returns the inclusive face-UV bounds of this tile.
    pub fn bounds(self) -> (FaceUv, FaceUv) {
        let side = self.side_len() as f32;
        let u0 = self.x as f32 / side * 2.0 - 1.0;
        let v0 = self.y as f32 / side * 2.0 - 1.0;
        let u1 = (self.x + 1) as f32 / side * 2.0 - 1.0;
        let v1 = (self.y + 1) as f32 / side * 2.0 - 1.0;
        (
            FaceUv::new(self.face, u0, v0).expect("tile bounds are valid"),
            FaceUv::new(self.face, u1, v1).expect("tile bounds are valid"),
        )
    }
}

#[derive(Clone, Copy)]
struct EdgeTransform {
    face: CubeFace,
    edge: TileEdge,
    reversed: bool,
}

fn edge_transform(face: CubeFace, edge: TileEdge) -> EdgeTransform {
    use CubeFace::*;
    use TileEdge::*;
    match (face, edge) {
        (PositiveX, West) => EdgeTransform {
            face: PositiveZ,
            edge: East,
            reversed: false,
        },
        (PositiveX, East) => EdgeTransform {
            face: NegativeZ,
            edge: West,
            reversed: false,
        },
        (PositiveX, South) => EdgeTransform {
            face: NegativeY,
            edge: East,
            reversed: true,
        },
        (PositiveX, North) => EdgeTransform {
            face: PositiveY,
            edge: East,
            reversed: false,
        },
        (NegativeX, West) => EdgeTransform {
            face: NegativeZ,
            edge: East,
            reversed: false,
        },
        (NegativeX, East) => EdgeTransform {
            face: PositiveZ,
            edge: West,
            reversed: false,
        },
        (NegativeX, South) => EdgeTransform {
            face: NegativeY,
            edge: West,
            reversed: false,
        },
        (NegativeX, North) => EdgeTransform {
            face: PositiveY,
            edge: West,
            reversed: true,
        },
        (PositiveY, West) => EdgeTransform {
            face: NegativeX,
            edge: North,
            reversed: true,
        },
        (PositiveY, East) => EdgeTransform {
            face: PositiveX,
            edge: North,
            reversed: false,
        },
        (PositiveY, South) => EdgeTransform {
            face: PositiveZ,
            edge: North,
            reversed: false,
        },
        (PositiveY, North) => EdgeTransform {
            face: NegativeZ,
            edge: North,
            reversed: true,
        },
        (NegativeY, West) => EdgeTransform {
            face: NegativeX,
            edge: South,
            reversed: false,
        },
        (NegativeY, East) => EdgeTransform {
            face: PositiveX,
            edge: South,
            reversed: true,
        },
        (NegativeY, South) => EdgeTransform {
            face: NegativeZ,
            edge: South,
            reversed: true,
        },
        (NegativeY, North) => EdgeTransform {
            face: PositiveZ,
            edge: South,
            reversed: false,
        },
        (PositiveZ, West) => EdgeTransform {
            face: NegativeX,
            edge: East,
            reversed: false,
        },
        (PositiveZ, East) => EdgeTransform {
            face: PositiveX,
            edge: West,
            reversed: false,
        },
        (PositiveZ, South) => EdgeTransform {
            face: NegativeY,
            edge: North,
            reversed: false,
        },
        (PositiveZ, North) => EdgeTransform {
            face: PositiveY,
            edge: South,
            reversed: false,
        },
        (NegativeZ, West) => EdgeTransform {
            face: PositiveX,
            edge: East,
            reversed: false,
        },
        (NegativeZ, East) => EdgeTransform {
            face: NegativeX,
            edge: West,
            reversed: false,
        },
        (NegativeZ, South) => EdgeTransform {
            face: NegativeY,
            edge: South,
            reversed: true,
        },
        (NegativeZ, North) => EdgeTransform {
            face: PositiveY,
            edge: North,
            reversed: true,
        },
    }
}

fn side_len(level: u8) -> Result<u32, TileError> {
    if level > MAX_TILE_LEVEL {
        return Err(TileError::LevelOutOfRange);
    }
    Ok(1_u32 << level)
}
