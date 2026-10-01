//! Block-compressed texture codecs (`BC1`/`BC3`/`BC4`/`BC5`) for the manual
//! texture sampler.
//!
//! AAA content ships textures in GPU block-compressed formats so they can be
//! sampled straight from VRAM. This module provides the classic
//! S3TC/RGTC decoders those formats require, letting a
//! [`TexelSource`](crate::TexelSource) over streaming virtual-texture pages
//! decode a compressed 4x4 block into sixteen `RGBA8` texels for the filter
//! stage. Decode is pure integer arithmetic -- no AI/ML path -- so a CPU
//! golden reproduces a GPU twin within each format's documented interpolation
//! tolerance.
//!
//! The decoders are split by concern:
//! * [`color_block`] -- the 8-byte `RGB565` + 2-bit-index colour block shared
//!   by BC1 and the colour half of BC3.
//! * [`alpha_block`] -- the 8-byte single-channel + 3-bit-index block shared by
//!   BC4, the alpha half of BC3, and both channels of BC5.
//! * [`formats`] -- the public per-format decoders that compose the two.
//!
//! BC6H/BC7 (multi-mode HDR / high-quality) are intentionally out of scope.
//!
//! # Conventions
//! * All blocks are little-endian; texel ordering is row-major with
//!   `t = y * 4 + x`, `t in [0, 16)`.
//! * Interpolated palette entries truncate; the low bit of an interpolated
//!   value is **implementation-defined within +/-1 LSB** across GPUs, so tests
//!   assert ordering/bounds rather than exact interpolated LSBs.
//!
//! # References
//! * Khronos Data Format Specification 1.3 (S3TC / RGTC block decode).
//! * Vulkan `VK_FORMAT_BC{1,3,4,5}_*` / D3D `DXGI_FORMAT_BC{1,3,4,5}_*`.

mod alpha_block;
mod color_block;
mod formats;

pub use color_block::rgb565_to_rgb888;
pub use formats::{decode_bc1, decode_bc3, decode_bc4, decode_bc5};
