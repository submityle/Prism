//! Readback of externally-produced GPU virtual-texture feedback words.
//!
//! This module only transfers and reads a feedback storage buffer. It does not
//! generate sampler feedback, perform raster sampling, or claim a complete
//! hardware feedback path.

use core::fmt;

use wgpu::{Buffer, BufferUsages, CommandEncoderDescriptor};

use crate::{
    buffer,
    context::{GpuContext, GpuError},
};

const WORD_BYTES: usize = size_of::<u32>();

#[derive(Clone, Debug, Eq, PartialEq)]
/// Errors raised while copying external feedback into staging.
pub enum FeedbackReadbackError {
    /// The staging reader was requested with zero word capacity.
    ZeroCapacity,
    /// A readback request exceeds the fixed staging capacity.
    CapacityExceeded {
        /// Number of words requested.
        requested: usize,
        /// Fixed staging capacity in words.
        capacity: usize,
    },
    /// The external source buffer is shorter than the requested copy.
    SourceTooSmall {
        /// Number of bytes requested from the source.
        requested_bytes: u64,
        /// Source buffer size in bytes.
        source_bytes: u64,
    },
    /// The source buffer does not permit GPU-to-GPU copies.
    SourceMissingCopySourceUsage,
    /// An intermediate byte-count conversion overflowed.
    ByteCountOverflow,
    /// The device or mapping operation failed while reading back the buffer.
    Gpu(GpuError),
}

impl fmt::Display for FeedbackReadbackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => {
                formatter.write_str("feedback readback capacity must be non-zero")
            }
            Self::CapacityExceeded {
                requested,
                capacity,
            } => write!(
                formatter,
                "feedback readback requested {requested} words but capacity is {capacity}"
            ),
            Self::SourceTooSmall {
                requested_bytes,
                source_bytes,
            } => write!(
                formatter,
                "feedback source has {source_bytes} bytes but {requested_bytes} are required"
            ),
            Self::SourceMissingCopySourceUsage => {
                formatter.write_str("feedback source buffer must include COPY_SRC usage")
            }
            Self::ByteCountOverflow => {
                formatter.write_str("feedback readback byte count overflowed")
            }
            Self::Gpu(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for FeedbackReadbackError {}

/// Reusable staging-backed reader for externally-produced feedback words.
pub struct GpuFeedbackReadback {
    staging: Buffer,
    capacity_words: usize,
}

impl GpuFeedbackReadback {
    /// Allocates a bounded staging buffer for feedback words.
    pub fn new(ctx: &GpuContext, capacity_words: usize) -> Result<Self, FeedbackReadbackError> {
        if capacity_words == 0 {
            return Err(FeedbackReadbackError::ZeroCapacity);
        }
        let capacity_bytes = capacity_words
            .checked_mul(WORD_BYTES)
            .ok_or(FeedbackReadbackError::ByteCountOverflow)?;
        let capacity_bytes =
            u64::try_from(capacity_bytes).map_err(|_| FeedbackReadbackError::ByteCountOverflow)?;
        Ok(Self {
            staging: buffer::staging(ctx.device(), "prism_vt_feedback_readback", capacity_bytes),
            capacity_words,
        })
    }

    /// Copies external GPU feedback words into staging and reads them back.
    ///
    /// The source must have `COPY_SRC` usage. This is a transfer/readback API
    /// only; feedback generation remains owned by the caller.
    /// Copies external GPU feedback words into staging and reads them back.
    pub fn read_from(
        &self,
        ctx: &GpuContext,
        source: &Buffer,
        word_count: usize,
    ) -> Result<Vec<u32>, FeedbackReadbackError> {
        if word_count > self.capacity_words {
            return Err(FeedbackReadbackError::CapacityExceeded {
                requested: word_count,
                capacity: self.capacity_words,
            });
        }
        let byte_count = word_count
            .checked_mul(WORD_BYTES)
            .ok_or(FeedbackReadbackError::ByteCountOverflow)?;
        let byte_count =
            u64::try_from(byte_count).map_err(|_| FeedbackReadbackError::ByteCountOverflow)?;
        if byte_count > source.size() {
            return Err(FeedbackReadbackError::SourceTooSmall {
                requested_bytes: byte_count,
                source_bytes: source.size(),
            });
        }
        if !source.usage().contains(BufferUsages::COPY_SRC) {
            return Err(FeedbackReadbackError::SourceMissingCopySourceUsage);
        }
        if word_count == 0 {
            return Ok(Vec::new());
        }
        let mut encoder = ctx
            .device()
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("prism_vt_feedback_readback_encoder"),
            });
        buffer::copy(&mut encoder, source, &self.staging, byte_count);
        ctx.queue().submit([encoder.finish()]);
        let mut words =
            buffer::try_read_back::<u32>(ctx, &self.staging).map_err(FeedbackReadbackError::Gpu)?;
        words.truncate(word_count);
        Ok(words)
    }

    #[must_use]
    /// Returns the fixed staging capacity in u32 words.
    pub const fn capacity_words(&self) -> usize {
        self.capacity_words
    }
}
