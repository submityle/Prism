//! Ordered dithering and a `blue-noise` approximation for stochastic
//! transparency (design §16).
//!
//! Translucent particles resolved with a hard alpha test leave a crawling,
//! aliased edge. Stochastic (dithered) transparency instead keeps or discards
//! each fragment against a spatially varying threshold, then lets the temporal
//! accumulation pass (`TAA`) or `alpha-to-coverage` average the pattern back
//! into a smooth gradient. This module is the device-free, `CPU`-verifiable
//! contract for the threshold source those passes sample.
//!
//! Two families of thresholds live here:
//!
//! * **`Bayer` ordered dither.** [`bayer4_raw`] / [`bayer8_raw`] build the
//!   classic recursive `Bayer` matrices with the integer recurrence
//!   `M_{2n} = [[4M, 4M+2], [4M+3, 4M+1]]`, so every cell of a `4x4` (or `8x8`)
//!   tile holds a distinct rank. [`bayer4`] / [`bayer8`] normalize those ranks
//!   into `[0, 1)`.
//! * **`blue-noise` approximation.** [`blue_noise01`] hashes the pixel and
//!   frame with a pure integer avalanche, giving a per-pixel, per-frame value in
//!   `[0, 1)` whose spectrum is far flatter than a raw ordered tile.
//!
//! [`temporal_offset`] rotates the threshold every frame by a golden-ratio
//! integer step so a static pixel does not lock to one keep/discard decision,
//! which is what removes the shimmering the alpha test would otherwise show.
//!
//! Every value is produced with integer arithmetic plus `f32::floor` for the
//! wrap and a single widening cast for normalization — no transcendental
//! functions — so this reference matches a future `GPU` kernel bit for bit.
//! The `Bayer` tiles are also exported as flat `std430` `u32` lookup tables via
//! [`bayer4_lut`] / [`bayer8_lut`].

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE};

/// The `0x9E3779B9` golden-ratio integer step (`floor(2^32 / phi)`), used to
/// advance the dither threshold from one frame to the next.
const GOLDEN_U32: u32 = 0x9E37_79B9;

/// `2^24`, the largest power of two exactly representable in an `f32` mantissa;
/// used as the normalization divisor so every hashed threshold is exact.
const NORM_24: f32 = 16_777_216.0;

/// Widens a `u32` to `f32` for threshold normalization.
///
/// Callers only ever pass values below `2^24` (`Bayer` ranks, or a hash reduced
/// to 24 bits), which are exactly representable, so the widening loses nothing.
#[expect(
    clippy::cast_precision_loss,
    reason = "callers pass values below 2^24, which are exact in an f32 mantissa"
)]
fn u32_to_f32(v: u32) -> f32 {
    v as f32
}

/// Wraps a value into `[0, 1)` by subtracting its floor.
fn wrap01(v: f32) -> f32 {
    v - v.floor()
}

/// Integer avalanche hash (a `Bit-mix` finalizer) with good bit diffusion.
///
/// Pure `u32` wrapping arithmetic and shifts — no floating point — so the
/// `blue-noise` threshold is deterministic and portable to a `GPU` kernel.
fn hash_u32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

/// Recursive `Bayer` rank at `(x, y)` inside a `size x size` tile.
///
/// `size` must be a power of two and `x`, `y` are assumed already reduced to
/// `[0, size)`. Implements `M_{2n} = [[4M, 4M+2], [4M+3, 4M+1]]`, with rows
/// indexed by `y` and columns by `x`.
fn bayer_recursive(x: u32, y: u32, size: u32) -> u32 {
    if size <= 1 {
        return 0;
    }
    let half = size / 2;
    let sub = bayer_recursive(x % half, y % half, half);
    let base = match (y >= half, x >= half) {
        (false, false) => 0,
        (false, true) => 2,
        (true, false) => 3,
        (true, true) => 1,
    };
    4 * sub + base
}

/// The `4x4` `Bayer` rank in `0..16` at `(x, y)` (coordinates are taken modulo
/// `4`).
#[must_use]
pub fn bayer4_raw(x: u32, y: u32) -> u32 {
    bayer_recursive(x % 4, y % 4, 4)
}

/// The `8x8` `Bayer` rank in `0..64` at `(x, y)` (coordinates are taken modulo
/// `8`).
#[must_use]
pub fn bayer8_raw(x: u32, y: u32) -> u32 {
    bayer_recursive(x % 8, y % 8, 8)
}

/// The `4x4` `Bayer` threshold normalized into `[0, 1)`.
#[must_use]
pub fn bayer4(x: u32, y: u32) -> f32 {
    u32_to_f32(bayer4_raw(x, y)) / 16.0
}

/// The `8x8` `Bayer` threshold normalized into `[0, 1)`.
#[must_use]
pub fn bayer8(x: u32, y: u32) -> f32 {
    u32_to_f32(bayer8_raw(x, y)) / 64.0
}

/// A per-pixel, per-frame `blue-noise` approximation in `[0, 1)`.
///
/// The pixel coordinates and frame index are decorrelated by odd-prime
/// multipliers, mixed, and run through [`hash_u32`]; the top 24 bits are
/// normalized so the result is exact. Distinct inputs almost always yield
/// distinct outputs, and identical inputs always agree.
#[must_use]
pub fn blue_noise01(x: u32, y: u32, frame: u32) -> f32 {
    let a = x.wrapping_mul(0x9E37_79B1);
    let b = y.wrapping_mul(0x85EB_CA77);
    let c = frame.wrapping_mul(0xC2B2_AE3D);
    let h = hash_u32(a ^ b ^ c);
    u32_to_f32(h >> 8) / NORM_24
}

/// The integer threshold rotation applied at frame `frame`.
///
/// Multiplying by the odd constant [`GOLDEN_U32`] is a bijection modulo `2^32`,
/// so distinct frames always produce distinct offsets.
#[must_use]
pub fn temporal_offset(frame: u32) -> u32 {
    frame.wrapping_mul(GOLDEN_U32)
}

/// The frame's threshold rotation as a fraction in `[0, 1)`.
fn temporal_fraction(frame: u32) -> f32 {
    u32_to_f32(temporal_offset(frame) >> 8) / NORM_24
}

/// Which threshold source a dither pass samples.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DitherMode {
    /// `4x4` ordered `Bayer` tile.
    Bayer4,
    /// `8x8` ordered `Bayer` tile.
    Bayer8,
    /// Hashed `blue-noise` approximation.
    BlueNoise,
}

/// The dither threshold in `[0, 1)` for pixel `(x, y)` at `frame`.
///
/// `Bayer` modes rotate the tile by the frame's golden-ratio fraction and wrap
/// back into `[0, 1)`; the `blue-noise` mode already folds the frame into its
/// hash.
#[must_use]
pub fn dither_threshold(x: u32, y: u32, frame: u32, mode: DitherMode) -> f32 {
    match mode {
        DitherMode::Bayer4 => wrap01(bayer4(x, y) + temporal_fraction(frame)),
        DitherMode::Bayer8 => wrap01(bayer8(x, y) + temporal_fraction(frame)),
        DitherMode::BlueNoise => blue_noise01(x, y, frame),
    }
}

/// Whether a fragment of opacity `alpha` is discarded against `threshold`.
///
/// The test is strict (`alpha < threshold`), so a fully opaque fragment
/// (`alpha == 1.0`) is never discarded by any threshold in `[0, 1)`.
#[must_use]
pub fn should_discard(alpha: f32, threshold: f32) -> bool {
    alpha < threshold
}

/// A bundled dither configuration: the mode plus the sampling helpers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DitherConfig {
    /// The threshold source this configuration samples.
    pub mode: DitherMode,
}

impl DitherConfig {
    /// Builds a configuration for `mode`.
    #[must_use]
    pub fn new(mode: DitherMode) -> Self {
        Self { mode }
    }

    /// The dither threshold in `[0, 1)` for pixel `(x, y)` at `frame`.
    #[must_use]
    pub fn threshold(self, x: u32, y: u32, frame: u32) -> f32 {
        dither_threshold(x, y, frame, self.mode)
    }

    /// Whether a fragment of opacity `alpha` is discarded at pixel `(x, y)` on
    /// `frame` under this configuration.
    #[must_use]
    pub fn should_discard(self, alpha: f32, x: u32, y: u32, frame: u32) -> bool {
        should_discard(alpha, self.threshold(x, y, frame))
    }
}

impl Default for DitherConfig {
    fn default() -> Self {
        Self::new(DitherMode::Bayer4)
    }
}

/// The `4x4` `Bayer` tile as a flat, row-major `std430` `u32` lookup table
/// (16 entries).
#[must_use]
pub fn bayer4_lut() -> Vec<u32> {
    (0..16u32).map(|i| bayer4_raw(i % 4, i / 4)).collect()
}

/// The `8x8` `Bayer` tile as a flat, row-major `std430` `u32` lookup table
/// (64 entries).
#[must_use]
pub fn bayer8_lut() -> Vec<u32> {
    (0..64u32).map(|i| bayer8_raw(i % 8, i / 8)).collect()
}

/// The `std430` byte size of the [`bayer4_lut`] storage buffer.
#[must_use]
pub fn bayer4_lut_bytes() -> usize {
    storage_bytes(U32_STRIDE, 16)
}

/// The `std430` byte size of the [`bayer8_lut`] storage buffer.
#[must_use]
pub fn bayer8_lut_bytes() -> usize {
    storage_bytes(U32_STRIDE, 64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for `f32` equality assertions in this module's tests.
    const CMP_EPS: f32 = 1.0e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    #[test]
    fn bayer4_covers_every_rank_and_stays_in_unit_range() {
        let mut mask = 0u32;
        for y in 0..4u32 {
            for x in 0..4u32 {
                let raw = bayer4_raw(x, y);
                assert!(raw < 16);
                mask |= 1u32 << raw;
                let n = bayer4(x, y);
                assert!((0.0..1.0).contains(&n));
            }
        }
        // All sixteen ranks appear exactly once (a permutation of 0..16).
        assert_eq!(mask, 0xFFFF);
    }

    #[test]
    fn bayer4_coordinates_wrap_by_modulo() {
        assert_eq!(bayer4_raw(0, 0), bayer4_raw(4, 8));
        assert_eq!(bayer4_raw(1, 3), bayer4_raw(5, 7));
    }

    #[test]
    fn bayer8_recursion_matches_direct_composition_and_is_a_permutation() {
        let mut mask = 0u64;
        for y in 0..8u32 {
            for x in 0..8u32 {
                let base = match (y >= 4, x >= 4) {
                    (false, false) => 0,
                    (false, true) => 2,
                    (true, false) => 3,
                    (true, true) => 1,
                };
                let expected = 4 * bayer4_raw(x % 4, y % 4) + base;
                let raw = bayer8_raw(x, y);
                assert_eq!(raw, expected);
                assert!(raw < 64);
                mask |= 1u64 << raw;
            }
        }
        // All sixty-four ranks appear exactly once.
        assert_eq!(mask, u64::MAX);
    }

    #[test]
    fn blue_noise_is_deterministic_and_in_range() {
        let a = blue_noise01(3, 7, 2);
        let b = blue_noise01(3, 7, 2);
        assert!(approx(a, b));
        assert!((0.0..1.0).contains(&a));
    }

    #[test]
    fn blue_noise_varies_across_inputs() {
        let base = blue_noise01(0, 0, 0);
        // Varying x must move at least one sample.
        let mut moved_x = false;
        for k in 1..16u32 {
            if !approx(blue_noise01(k, 0, 0), base) {
                moved_x = true;
            }
        }
        assert!(moved_x);

        // Varying frame (temporal dimension) must also move a sample.
        let mut moved_frame = false;
        for f in 1..16u32 {
            if !approx(blue_noise01(0, 0, f), base) {
                moved_frame = true;
            }
        }
        assert!(moved_frame);
    }

    #[test]
    fn temporal_offset_is_distinct_per_frame() {
        assert_ne!(temporal_offset(0), temporal_offset(1));
        assert_ne!(temporal_offset(1), temporal_offset(2));
        assert_ne!(temporal_offset(100), temporal_offset(101));
        // Deterministic for a repeated frame.
        assert_eq!(temporal_offset(42), temporal_offset(42));
    }

    #[test]
    fn dither_threshold_stays_in_unit_range_for_all_modes() {
        for &mode in &[
            DitherMode::Bayer4,
            DitherMode::Bayer8,
            DitherMode::BlueNoise,
        ] {
            for frame in 0..4u32 {
                let t = dither_threshold(2, 5, frame, mode);
                assert!((0.0..1.0).contains(&t), "mode {mode:?} out of range");
            }
        }
    }

    #[test]
    fn dither_threshold_shifts_over_frames() {
        let base = dither_threshold(1, 1, 0, DitherMode::Bayer4);
        let mut shifted = false;
        for frame in 1..16u32 {
            if !approx(dither_threshold(1, 1, frame, DitherMode::Bayer4), base) {
                shifted = true;
            }
        }
        assert!(shifted);
    }

    #[test]
    fn should_discard_respects_the_strict_boundary() {
        assert!(should_discard(0.2, 0.5));
        // Equal alpha and threshold is kept, not discarded.
        assert!(!should_discard(0.5, 0.5));
        assert!(!should_discard(0.9, 0.5));
        // A fully opaque fragment survives any in-range threshold.
        assert!(!should_discard(1.0, 0.999));
    }

    #[test]
    fn config_delegates_to_free_functions() {
        let cfg = DitherConfig::new(DitherMode::Bayer8);
        assert!(approx(
            cfg.threshold(3, 4, 1),
            dither_threshold(3, 4, 1, DitherMode::Bayer8)
        ));
        assert_eq!(DitherConfig::default().mode, DitherMode::Bayer4);
        let t = cfg.threshold(3, 4, 1);
        assert_eq!(cfg.should_discard(0.0, 3, 4, 1), should_discard(0.0, t));
    }

    #[test]
    fn lut_tables_match_raw_ranks_and_std430_sizes() {
        let lut4 = bayer4_lut();
        assert_eq!(lut4.len(), 16);
        for (i, &v) in lut4.iter().enumerate() {
            let idx = u32::try_from(i).unwrap();
            assert_eq!(v, bayer4_raw(idx % 4, idx / 4));
        }

        let lut8 = bayer8_lut();
        assert_eq!(lut8.len(), 64);
        for (i, &v) in lut8.iter().enumerate() {
            let idx = u32::try_from(i).unwrap();
            assert_eq!(v, bayer8_raw(idx % 8, idx / 8));
        }

        // Four bytes per u32 element under std430.
        assert_eq!(bayer4_lut_bytes(), 64);
        assert_eq!(bayer8_lut_bytes(), 256);
    }
}
