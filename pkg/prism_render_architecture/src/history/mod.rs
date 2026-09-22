//! Shared history identity and invalidation.

use crate::abi::GenerationalHandle;

pub type ViewHistoryId = GenerationalHandle;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InvalidationMask(pub u32);

impl InvalidationMask {
    pub const CAMERA_CUT: Self = Self(1 << 0);
    pub const RESOLUTION: Self = Self(1 << 1);
    pub const EXPOSURE: Self = Self(1 << 2);
    pub const SCENE: Self = Self(1 << 3);
    pub const LIGHTING: Self = Self(1 << 4);
    pub const MATERIAL: Self = Self(1 << 5);
    pub const STREAMING_REVEAL: Self = Self(1 << 6);
    pub const SHADER_VERSION: Self = Self(1 << 7);

    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HistoryEpochs {
    pub scene: u64,
    pub lighting: u64,
    pub material: u64,
    pub origin: u64,
}
