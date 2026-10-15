//! Deterministic, engine-neutral height sampling contracts.

use crate::TileKey;

/// A quantized sample location within a cube-sphere tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightSample {
    tile: TileKey,
    u_q16: u16,
    v_q16: u16,
}

impl HeightSample {
    /// Creates a sample. `u_q16` and `v_q16` are normalized tile-local coordinates.
    pub const fn new(tile: TileKey, u_q16: u16, v_q16: u16) -> Self {
        Self { tile, u_q16, v_q16 }
    }

    /// Returns the owning tile.
    pub const fn tile(self) -> TileKey {
        self.tile
    }

    /// Returns the quantized local U coordinate.
    pub const fn u_q16(self) -> u16 {
        self.u_q16
    }

    /// Returns the quantized local V coordinate.
    pub const fn v_q16(self) -> u16 {
        self.v_q16
    }
}

/// A deterministic height result in integer millimetres.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightSampleValue {
    height_mm: i64,
}

impl HeightSampleValue {
    /// Creates a height result in millimetres.
    pub const fn from_millimetres(height_mm: i64) -> Self {
        Self { height_mm }
    }

    /// Returns the height in millimetres.
    pub const fn height_mm(self) -> i64 {
        self.height_mm
    }
}

/// A deterministic height sampler contract.
pub trait HeightSampler {
    /// Sampler-specific failure type.
    type Error;

    /// Samples a quantized cube-sphere location.
    fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error>;
}

/// A small integer hash sampler useful as a deterministic reference implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeterministicHeightSampler {
    seed: u64,
    amplitude_mm: i64,
}

impl DeterministicHeightSampler {
    /// Creates a sampler with a seed and a symmetric peak amplitude.
    pub const fn new(seed: u64, amplitude_mm: i64) -> Self {
        Self { seed, amplitude_mm }
    }
}

impl HeightSampler for DeterministicHeightSampler {
    type Error = core::convert::Infallible;

    fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        let tile = sample.tile();
        let mut state = self.seed ^ 0x9e37_79b9_7f4a_7c15;
        state = mix64(state ^ u64::from(tile.face().index()));
        state = mix64(state ^ u64::from(tile.level()));
        state = mix64(state ^ u64::from(tile.x()));
        state = mix64(state ^ u64::from(tile.y()));
        state = mix64(state ^ u64::from(sample.u_q16()));
        state = mix64(state ^ u64::from(sample.v_q16()));
        let centered = ((state >> 16) & 0xffff) as i64 - 32_768;
        let height = centered.saturating_mul(self.amplitude_mm) / 32_768;
        Ok(HeightSampleValue::from_millimetres(height))
    }
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
