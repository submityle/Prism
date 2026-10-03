//! Asset-streaming decompression codecs.
//!
//! `DirectStorage`-class streaming pipelines ship data `GDeflate`-compressed and
//! decompress it on the way into resident memory. This module owns the
//! device-free, deterministic codecs those pipelines depend on:
//!
//! * [`deflate`] — a complete `RFC 1951` inflate core (stored, fixed-, and
//!   dynamic-Huffman blocks) usable both as a `CPU` fallback and as the golden
//!   reference a `GPU` decoder is validated against.
//! * [`deflate_encode`] — a minimal, standards-compliant `DEFLATE` encoder used
//!   to produce the inner per-tile payloads (so the container is round-trip
//!   verifiable without an external encoder).
//! * [`gdeflate`] — the 32-lane warp-interleaved `GDeflate` tile container built
//!   on top of the inner `DEFLATE` codec, with mutually consistent `CPU`
//!   encode/decode (see its module docs for the honest provenance and
//!   verification status).

pub mod deflate;
mod deflate_encode;
pub mod gdeflate;

pub use deflate::{inflate, InflateError};
pub use gdeflate::{
    gdeflate_compress, gdeflate_decompress, GDeflateError, GDeflateHeader, GDeflateTileDescriptor,
};
