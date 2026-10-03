//! Asset-streaming decompression codecs.
//!
//! `DirectStorage`-class streaming pipelines ship data `GDeflate`-compressed and
//! decompress it on the way into resident memory. This module owns the
//! device-free, deterministic decoders those pipelines depend on, starting with
//! the inner `DEFLATE` codec every `GDeflate` tile carries:
//!
//! * [`deflate`] — a complete `RFC 1951` inflate core (stored, fixed-, and
//!   dynamic-Huffman blocks) usable both as a `CPU` fallback and as the golden
//!   reference a `GPU` decoder is validated against.

pub mod deflate;

pub use deflate::{inflate, InflateError};
