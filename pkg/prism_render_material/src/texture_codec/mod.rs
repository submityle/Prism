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
//! * [`bc7`] -- the standalone BC7 single-subset (`RGBA`) block decoder.
//! * [`bc6h`] -- the BC6H HDR single-subset (`RGB` half-float) decoder.
//! * [`bitio`] -- the shared LSB-first bit cursor used by both BPTC decoders.
//!
//! BC7 **modes 4, 5, and 6** (single-subset `RGBA`, no partition table) are
//! decoded by [`bc7`], and BC6H **mode 11** (single-subset unsigned HDR) by
//! [`bc6h`]; the partitioned BC7 modes (0-3, 7), BC6H delta/partitioned
//! modes, and signed BC6H remain follow-ups (they need validated Khronos
//! partition/anchor and endpoint-transform tables).
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
mod bc6h;
mod bc7;
mod bitio;
mod color_block;
mod encode;
mod formats;
mod source;

pub use bc6h::{
    bc6h_mode_bits, decode_bc6h_mode11_signed, decode_bc6h_mode11_unsigned, decode_bc6h_signed,
    decode_bc6h_unsigned, half_bits_to_f32, Bc6hError,
};
pub use bc7::{
    bc7_mode, decode_bc7, decode_bc7_mode4, decode_bc7_mode5, decode_bc7_mode6, Bc7Error,
};
pub use color_block::rgb565_to_rgb888;
pub use encode::encode_bc1;
pub use formats::{decode_bc1, decode_bc2, decode_bc3, decode_bc4, decode_bc5};
pub use source::{BcFormat, BcSourceError, BcTexelSource};
