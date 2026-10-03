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
//! * [`snorm_block`] -- the signed BC4/BC5 (`SNORM`) single-channel block,
//!   decoding to `i8` for signed tangent-space data.
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
mod astc;
mod bc6h;
mod bc7;
mod bitio;
mod bptc_tables;
mod color_block;
mod eac;
mod encode;
mod etc2;
mod formats;
mod snorm_block;
mod source;

pub use astc::{
    decode_astc_4x4_hdr, decode_astc_4x4_ldr, decode_astc_4x4_weights, decode_astc_4x4_weights_ise,
    decode_astc_ldr, decode_astc_void_extent_hdr, decode_astc_void_extent_ldr, AstcError,
};
pub use bc6h::{
    bc6h_mode_bits, decode_bc6h_mode10_signed, decode_bc6h_mode10_unsigned,
    decode_bc6h_mode11_signed, decode_bc6h_mode11_unsigned, decode_bc6h_mode12_signed,
    decode_bc6h_mode12_unsigned, decode_bc6h_mode13_signed, decode_bc6h_mode13_unsigned,
    decode_bc6h_mode14_signed, decode_bc6h_mode14_unsigned, decode_bc6h_mode1_signed,
    decode_bc6h_mode1_unsigned, decode_bc6h_mode2_signed, decode_bc6h_mode2_unsigned,
    decode_bc6h_mode3_signed, decode_bc6h_mode3_unsigned, decode_bc6h_mode4_signed,
    decode_bc6h_mode4_unsigned, decode_bc6h_mode5_signed, decode_bc6h_mode5_unsigned,
    decode_bc6h_mode6_signed, decode_bc6h_mode6_unsigned, decode_bc6h_mode7_signed,
    decode_bc6h_mode7_unsigned, decode_bc6h_mode8_signed, decode_bc6h_mode8_unsigned,
    decode_bc6h_mode9_signed, decode_bc6h_mode9_unsigned, decode_bc6h_signed, decode_bc6h_unsigned,
    half_bits_to_f32, Bc6hError,
};
pub use bc7::{
    bc7_mode, decode_bc7, decode_bc7_mode0, decode_bc7_mode1, decode_bc7_mode2, decode_bc7_mode3,
    decode_bc7_mode4, decode_bc7_mode5, decode_bc7_mode6, decode_bc7_mode7, Bc7Error,
};
pub use color_block::rgb565_to_rgb888;
pub use eac::{
    decode_eac_r11_snorm, decode_eac_r11_unorm, decode_eac_rg11_snorm, decode_eac_rg11_unorm,
};
pub use encode::{
    encode_bc1, encode_bc2, encode_bc3, encode_bc4, encode_bc4_signed, encode_bc5,
    encode_bc5_signed, encode_bc6h_mode11_signed, encode_bc6h_mode11_unsigned, encode_bc7_mode4,
    encode_bc7_mode5, encode_bc7_mode6, encode_etc2_rgb8,
};
pub use etc2::{decode_etc2_rgb8, etc2_rgb8_mode, Etc2Error, Etc2Mode};
pub use formats::{
    decode_bc1, decode_bc2, decode_bc3, decode_bc4, decode_bc4_signed, decode_bc5,
    decode_bc5_signed,
};
pub use source::{BcFormat, BcSourceError, BcTexelSource};
