//! Optional `wgpu` compute backend for Prism's `GDeflate` asset-streaming
//! decompressor.
//!
//! Prism splits payloads into fixed-size raw-DEFLATE tiles and transposes
//! their words into a 32-lane layout: logical 32-bit word `r * 32 + lane` is
//! stored at lane-major position `lane * rounds + r`. This layout does not
//! make Huffman or LZ77 decoding warp-parallel. The complete, device-free codec
//! (container framing, the `RFC 1951` inflate core, and the reversible word
//! transpose) lives in
//! [`prism_render_architecture::compression`].
//!
//! The unified decoder uses a complete GPU raw-DEFLATE kernel: direct
//! lane-major word gather, stored/fixed/dynamic blocks, device-built canonical
//! Huffman trees, and LZ77 references across block boundaries. Host code parses
//! Prism's GDF1 framing and verifies tile checksums; no CPU inflate is used on
//! the successful GPU path. GPU rejection retries the CPU golden codec with
//! explicit fallback telemetry. Independent de-interleave, stored-plan, and
//! strict fixed-only algorithms are also available for direct composition.
//!
//! Tiles are bounded to 64 KiB and decoded serially by one invocation each.
//! Direct batch APIs submit independent tiles in one compute pass and one
//! submission, read back only status records, and publish owned GPU outputs
//! after all tiles validate. Synchronous byte-returning APIs also remain.
//! The lane-major GDF1 layout is Prism's format contract, not a claim of binary
//! interoperability with external DirectStorage/GDeflate implementations.
//!
//! # Exact parity
//!
//! Real-device tests compare GPU output directly against independently built
//! bytes and exercise dynamic trees, mixed blocks, malformed streams, and
//! the shared compressor. Successful unified decode must report full GPU
//! execution with no fallback; CPU correctness is not used as GPU evidence.
//!
//! Provenance: the public `GDeflate`/`DirectStorage` format description only.
//! Classical integer permutation and canonical Huffman/LZ77; no neural, learned, or
//! data-driven components. No Unreal Engine or NVIDIA `GDeflate` source or
//! derived code.

extern crate alloc;

pub mod buffer;
pub mod context;
pub mod decoder;
pub mod deflate;
pub mod deinterleave;
pub mod fixed;
pub mod stored;

pub use context::{GpuContext, GpuProbeError};
pub use decoder::{
    DecodePath, DecodeReport, DecodedGDeflate, DecoderError, GDeflateDecoder, GpuFallbackReason,
};
pub use deflate::{DeflateDecodeError, GpuDeflate};
pub use deinterleave::{
    reference_deinterleave, DeinterleaveError, GpuDeinterleave, GROUP, LANES, WORD,
};
pub use fixed::{
    DeflateBatchError, DeflateBatchLimits, DeflateTile, FixedDecodeError, GpuFixedGDeflate,
    ResidentDeflateTile,
};
pub use stored::{plan_stored_deflate, GpuStoredGDeflate, StoredBlockPlan, StoredPlanError};
