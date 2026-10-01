//! A [`TexelSource`](crate::TexelSource) backed by in-memory block-compressed
//! mip data, tying the [`decode_bc*`](super::formats) decoders to the manual
//! sampler/filter stack.
//!
//! This closes the CPU-golden texture pipeline: a streaming virtual-texture
//! page (or any in-memory blob) holding BC blocks can now be sampled through
//! the exact same `address -> LOD -> anisotropy -> fetch -> filter` path the
//! GPU fixed-function sampler would take, with no AI/ML anywhere. It is a
//! correctness golden -- it decodes the whole 4x4 block on every `texel` read
//! rather than caching -- so a production GPU-resident source can be validated
//! against it bit-for-bit (within the formats' documented interpolation
//! tolerance).
//!
//! # Conventions
//! * Blocks are stored row-major in 4x4 tiles; `blocks_per_row =
//!   ceil(width / 4)` for each mip, matching the GPU memory layout.
//! * Mip `m` has dimensions `(max(w0 >> m, 1), max(h0 >> m, 1))`; the caller
//!   (bilinear/trilinear) has already wrapped `(x, y)` into that range, and
//!   `mip` is clamped to the available mip count.
//! * Decoded `u8` RGBA is converted to `f32` in `[0, 1]` by dividing by 255.
//!
//! # References
//! * Khronos Data Format Specification 1.3 (block memory layout).
//! * Vulkan `vkCmdCopyBufferToImage` tiling for compressed formats.

use alloc::vec::Vec;

use super::formats::{decode_bc1, decode_bc3, decode_bc4, decode_bc5};
use crate::TexelSource;

/// The subset of block-compressed formats this source can decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BcFormat {
    /// BC1 -- opaque / 1-bit-alpha albedo, 8-byte blocks.
    Bc1,
    /// BC3 -- albedo + smooth alpha, 16-byte blocks.
    Bc3,
    /// BC4 -- single channel (value in `R`), 8-byte blocks.
    Bc4,
    /// BC5 -- two channels (ch0 in `R`, ch1 in `G`), 16-byte blocks.
    Bc5,
}

impl BcFormat {
    /// Compressed block size in bytes for this format.
    #[inline]
    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            BcFormat::Bc1 | BcFormat::Bc4 => 8,
            BcFormat::Bc3 | BcFormat::Bc5 => 16,
        }
    }

    /// Decode one block at `bytes` (length already validated by the caller)
    /// into 16 `RGBA8` texels in row-major order.
    #[inline]
    fn decode(self, bytes: &[u8]) -> [[u8; 4]; 16] {
        match self {
            BcFormat::Bc1 => {
                let b: &[u8; 8] = bytes[..8].try_into().expect("bc1 block is 8 bytes");
                decode_bc1(b)
            }
            BcFormat::Bc4 => {
                let b: &[u8; 8] = bytes[..8].try_into().expect("bc4 block is 8 bytes");
                decode_bc4(b)
            }
            BcFormat::Bc3 => {
                let b: &[u8; 16] = bytes[..16].try_into().expect("bc3 block is 16 bytes");
                decode_bc3(b)
            }
            BcFormat::Bc5 => {
                let b: &[u8; 16] = bytes[..16].try_into().expect("bc5 block is 16 bytes");
                decode_bc5(b)
            }
        }
    }
}

/// Axis dimension (texels) at `mip`, floored to `>= 1`.
#[inline]
#[must_use]
fn dim_at(base: u32, mip: u32) -> u32 {
    (base >> mip.min(31)).max(1)
}

/// Number of 4x4 blocks spanning `dim` texels on one axis.
#[inline]
#[must_use]
fn blocks_along(dim: u32) -> u32 {
    dim.div_ceil(4)
}

/// Errors raised while validating a [`BcTexelSource`] mip chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BcSourceError {
    /// Base dimensions were zero (a texture must have at least one texel).
    EmptyDimensions,
    /// No mip levels were supplied.
    NoMips,
    /// Mip `index` held `found` bytes, but the format/dimensions require
    /// `expected`.
    MipByteCount {
        /// Mip level whose byte count was wrong.
        index: u32,
        /// Bytes the mip actually held.
        found: usize,
        /// Bytes the layout requires.
        expected: usize,
    },
}

/// An owned block-compressed texture exposed as a [`TexelSource`].
#[derive(Clone, Debug)]
pub struct BcTexelSource {
    format: BcFormat,
    width: u32,
    height: u32,
    mips: Vec<Vec<u8>>,
}

impl BcTexelSource {
    /// Build a source from `format`, base dimensions `(width, height)`, and one
    /// block blob per mip (mip 0 first). Each blob must hold exactly
    /// `blocks_per_row * blocks_per_col * block_bytes` bytes for that mip's
    /// dimensions, or [`BcSourceError::MipByteCount`] is returned.
    pub fn new(
        format: BcFormat,
        width: u32,
        height: u32,
        mips: Vec<Vec<u8>>,
    ) -> Result<Self, BcSourceError> {
        if width == 0 || height == 0 {
            return Err(BcSourceError::EmptyDimensions);
        }
        if mips.is_empty() {
            return Err(BcSourceError::NoMips);
        }
        let bb = format.block_bytes();
        for (i, blob) in mips.iter().enumerate() {
            let mip = i as u32;
            let w = dim_at(width, mip);
            let h = dim_at(height, mip);
            let expected = blocks_along(w) as usize * blocks_along(h) as usize * bb;
            if blob.len() != expected {
                return Err(BcSourceError::MipByteCount {
                    index: mip,
                    found: blob.len(),
                    expected,
                });
            }
        }
        Ok(Self {
            format,
            width,
            height,
            mips,
        })
    }

    /// Format this source decodes.
    #[inline]
    #[must_use]
    pub const fn format(&self) -> BcFormat {
        self.format
    }

    /// Number of stored mip levels (always `>= 1`).
    #[inline]
    #[must_use]
    pub fn mip_count(&self) -> u32 {
        self.mips.len() as u32
    }
}

impl TexelSource for BcTexelSource {
    #[inline]
    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn texel(&self, mip: u32, x: u32, y: u32) -> [f32; 4] {
        // Clamp mip into the stored range; dimensions follow the clamped mip.
        let last = self.mips.len() as u32 - 1;
        let mip = mip.min(last);
        let w = dim_at(self.width, mip);
        let h = dim_at(self.height, mip);
        // Caller pre-wraps, but defend against out-of-range inputs regardless.
        let x = x.min(w - 1);
        let y = y.min(h - 1);

        let blocks_per_row = blocks_along(w);
        let bx = x / 4;
        let by = y / 4;
        let block_index = (by * blocks_per_row + bx) as usize;
        let bb = self.format.block_bytes();
        let offset = block_index * bb;

        let blob = &self.mips[mip as usize];
        let Some(bytes) = blob.get(offset..offset + bb) else {
            // Layout was validated in `new`, so this is unreachable in practice;
            // return opaque black rather than panicking if ever reached.
            return [0.0, 0.0, 0.0, 1.0];
        };

        let texels = self.format.decode(bytes);
        let local = ((y % 4) * 4 + (x % 4)) as usize;
        let px = texels[local];
        [
            f32::from(px[0]) / 255.0,
            f32::from(px[1]) / 255.0,
            f32::from(px[2]) / 255.0,
            f32::from(px[3]) / 255.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// One opaque BC1 block whose colour0 index (index 0) selects endpoint 0.
    /// endpoint0 = RGB565 red (0xF800), endpoint1 = black; indices all 0.
    fn bc1_red_block() -> Vec<u8> {
        // c0 = 0xF800 (red) > c1 = 0x0000 -> 4-colour opaque mode; indices = 0.
        vec![0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    }

    #[test]
    fn new_rejects_wrong_byte_count() {
        let err = BcTexelSource::new(BcFormat::Bc1, 4, 4, vec![vec![0u8; 7]]).unwrap_err();
        match err {
            BcSourceError::MipByteCount { expected, found, .. } => {
                assert_eq!(expected, 8);
                assert_eq!(found, 7);
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn new_rejects_empty_dimensions_and_mips() {
        assert_eq!(
            BcTexelSource::new(BcFormat::Bc1, 0, 4, vec![vec![0u8; 8]]).unwrap_err(),
            BcSourceError::EmptyDimensions
        );
        assert_eq!(
            BcTexelSource::new(BcFormat::Bc1, 4, 4, vec![]).unwrap_err(),
            BcSourceError::NoMips
        );
    }

    #[test]
    fn single_block_decodes_red_everywhere() {
        let src = BcTexelSource::new(BcFormat::Bc1, 4, 4, vec![bc1_red_block()]).unwrap();
        assert_eq!(src.dimensions(), (4, 4));
        assert_eq!(src.mip_count(), 1);
        for y in 0..4 {
            for x in 0..4 {
                let t = src.texel(0, x, y);
                assert!(t[0] > 0.99, "red at ({x},{y}) = {t:?}");
                assert!(t[1] < 0.01 && t[2] < 0.01);
                assert!((t[3] - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn mip_is_clamped_and_coords_are_defended() {
        let src = BcTexelSource::new(BcFormat::Bc1, 4, 4, vec![bc1_red_block()]).unwrap();
        // mip far beyond range clamps to the only level; oversized coords clamp.
        let t = src.texel(99, 99, 99);
        assert!(t[0] > 0.99 && (t[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bc5_two_channel_block_fills_red_and_green_axes() {
        // ch0 (R): r0=255 r1=0, all indices 0 -> R = 255.
        // ch1 (G): r0=0  r1=255, all indices 0 -> G = 0.
        let mut blk = vec![0u8; 16];
        blk[0] = 255; // ch0 r0
        blk[1] = 0; // ch0 r1
        blk[8] = 0; // ch1 r0
        blk[9] = 255; // ch1 r1
        let src = BcTexelSource::new(BcFormat::Bc5, 4, 4, vec![blk]).unwrap();
        let t = src.texel(0, 0, 0);
        assert!(t[0] > 0.99, "R from ch0 = {t:?}");
        assert!(t[1] < 0.01, "G from ch1 index0 = {t:?}");
        assert!(t[2] < 0.01 && (t[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn multi_block_width_indexes_correct_tile() {
        // 8x4 BC1: two blocks across. Left block red, right block black.
        let mut blob = Vec::new();
        blob.extend_from_slice(&bc1_red_block()); // block (0,0)
        blob.extend_from_slice(&[0u8; 8]); // block (1,0): all-zero -> black
        let src = BcTexelSource::new(BcFormat::Bc1, 8, 4, vec![blob]).unwrap();
        assert!(src.texel(0, 0, 0)[0] > 0.99, "left tile red");
        assert!(src.texel(0, 4, 0)[0] < 0.01, "right tile black");
    }
}
