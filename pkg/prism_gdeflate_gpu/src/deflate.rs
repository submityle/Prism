//! Complete raw-DEFLATE tile decoding on the device, including dynamic tree
//! construction, stored blocks, and history across arbitrary block mixtures.

use alloc::vec::Vec;

use crate::{
    DeflateBatchError, DeflateBatchLimits, DeflateTile, GpuContext, GpuFixedGDeflate,
    ResidentDeflateTile,
};

pub use crate::fixed::DeflateDecodeError;

/// GPU decoder for all RFC 1951 block types in Prism's lane-major tile format.
///
/// The host validates only tile shape and resource limits. The kernel parses
/// block headers, constructs dynamic canonical trees, expands LZ77 references,
/// and validates the exact output size. It uses no CPU inflate or block plan.
/// One invocation decodes a tile. Independent tiles can share one compute pass
/// and submission while their validated outputs remain resident on the GPU.
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

    /// Decodes a bounded batch, returning owned GPU output buffers together
    /// only after all tile statuses and exact output lengths are valid.
    ///
    /// Delegates to the same batch transport and kernel as the fixed-only API,
    /// with stored, fixed, and dynamic blocks enabled. Only packed status
    /// records are read back; no payload staging buffer is allocated.
    pub fn decompress_batch_resident(
        &self,
        ctx: &GpuContext,
        requests: &[DeflateTile<'_>],
        admission: DeflateBatchLimits,
    ) -> Result<Vec<ResidentDeflateTile>, DeflateBatchError> {
        self.kernel
            .decompress_batch_resident(ctx, requests, admission)
    }
}
