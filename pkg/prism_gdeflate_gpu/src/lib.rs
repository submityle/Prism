//! Optional `wgpu` compute backend for Prism's `GDeflate` asset-streaming
//! decompressor.
//!
//! `DirectStorage`-class streaming ships assets `GDeflate`-compressed: the
//! payload is split into fixed-size tiles, each tile is `DEFLATE`-compressed,
//! and the resulting bitstream is laid out so a 32-lane `GPU` warp can decode
//! it with coalesced access — logical 32-bit word `r * 32 + lane` is stored at
//! lane-major position `lane * rounds + r`. The complete, device-free codec
//! (container framing, the `RFC 1951` inflate core, and the reversible word
//! transpose) lives in
//! [`prism_render_architecture::compression`].
//!
//! This crate is the real-device half of the *embarrassingly-parallel* stage of
//! that pipeline: a compute kernel that reverses the warp-interleave transpose.
//! The Huffman inflate is inherently serial and variable-length, so it is
//! deliberately **not** moved to the `GPU`; instead the kernel performs the
//! per-output-word de-interleave gather (one invocation writes one linear word
//! by reading its lane-major source word) and the host runs the ordinary
//! `inflate` core on the recovered stream. This mirrors how the sparse
//! virtual-texture twin splits its parallel map onto the device and keeps the
//! serial dedup on the host.
//!
//! # Exact parity
//!
//! The de-interleave is a pure word permutation — every invocation computes
//! integer indices (`lane = logical % 32`, `r = logical / 32`,
//! `stored = lane * rounds + r`) and moves a whole 32-bit word, so no bit is
//! ever reinterpreted. There is no floating point anywhere, so the device
//! output equals the golden de-interleave bit-for-bit and the parity tests
//! assert exact equality rather than a tolerance. An end-to-end test feeds a
//! real [`gdeflate_compress`](prism_render_architecture::compression::gdeflate_compress)
//! tile payload through the device de-interleave and then the golden
//! [`inflate`](prism_render_architecture::compression::inflate), proving the
//! device stage composes with the real golden decode path.
//!
//! Provenance: the public `GDeflate`/`DirectStorage` format description only.
//! Classical, data-oblivious integer word permutation; no neural, learned, or
//! data-driven components. No Unreal Engine or NVIDIA `GDeflate` source or
//! derived code.

extern crate alloc;

pub mod buffer;
pub mod context;
pub mod deinterleave;

pub use context::GpuContext;
pub use deinterleave::{reference_deinterleave, GpuDeinterleave, GROUP, LANES, WORD};
