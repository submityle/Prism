//! Large-world coordinate and spatial-cell contracts.

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct WorldCell {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WorldPosition {
    pub cell: WorldCell,
    pub local: [f64; 3],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WorldOrigin {
    pub cell: WorldCell,
    pub epoch: u64,
}
