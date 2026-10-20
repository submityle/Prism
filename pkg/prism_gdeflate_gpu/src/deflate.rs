//! Complete raw-DEFLATE tile decoding on the device, including dynamic tree
//! construction, stored blocks, and history across arbitrary block mixtures.

use alloc::vec::Vec;

use crate::{GpuContext, GpuFixedGDeflate};

pub use crate::fixed::DeflateDecodeError;

/// GPU decoder for all RFC 1951 block types in Prism's lane-major tile format.
///
/// The host validates only tile shape and resource limits. The kernel parses
/// block headers, constructs dynamic canonical trees, expands LZ77 references,
/// and validates the exact output size. It uses no CPU inflate or block plan.
/// One invocation decodes a tile, and this synchronous API submits one tile at
/// a time; it does not yet provide batched or GPU-resident output consumption.
pub struct GpuDeflate {
    kernel: GpuFixedGDeflate,
}

impl GpuDeflate {
    /// Compiles the complete DEFLATE kernel for this device.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            kernel: GpuFixedGDeflate::with_mode(ctx, false),
        }
    }

    /// Decodes a complete tile, returning bytes only after successful GPU
    /// status and exact output-length validation. The tile is at most 64 KiB;
    /// `interleaved` includes lane-group padding, `compressed_size` excludes it.
    /// Container checksums are the caller's responsibility.
    pub fn decompress(
        &self,
        ctx: &GpuContext,
        interleaved: &[u8],
        compressed_size: usize,
        expected_output_size: usize,
    ) -> Result<Vec<u8>, DeflateDecodeError> {
        self.kernel
            .decompress(ctx, interleaved, compressed_size, expected_output_size)
    }
}
