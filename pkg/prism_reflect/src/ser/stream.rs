//! Bounded, framed binary-stream decoding (design §24.8).
//!
//! The ordinary [`super::binary::from_binary`] API consumes one complete byte
//! slice.  A file/network reader cannot always provide that slice at once, so
//! this module adds a small framing and flow-control contract around it:
//!
//! ```text
//! frame := payload_len(u64 LE) + payload(from_binary)
//! ```
//!
//! [`BinaryStreamDecoder`] is the allocation-owning core.  It uses only
//! `core`/`alloc` primitives, accepts a complete input chunk or returns a
//! non-consuming [`StreamDecodeError::Backpressure`] result, and keeps all
//! accepted-but-not-yet-consumed bytes below a configured hard ceiling.  A
//! caller supplies monotonically increasing ticks to make timeout behaviour
//! deterministic on embedded/no-std targets.  [`StdStreamDecoder`] is the thin
//! `std` adapter that supplies monotonic `Instant` ticks.
//!
//! A frame is removed from the bounded queue only when
//! [`BinaryStreamDecoder::consume_next_at`] is called.  Decode failures are
//! associated with that frame and do not desynchronise later frames, because
//! the length prefix has already established the next frame boundary.

use crate::reflect::Reflect;
use crate::ser::binary::{from_binary_with_limits, to_binary};
use crate::ser::error::{DeserializeError, SerializeError};
use crate::ser::limits::DeserializeLimits;
use crate::type_info::TypeInfo;
use crate::TypeRegistry;
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::mem::size_of;

/// Number of bytes in a stream-frame length prefix.
pub const FRAME_PREFIX_BYTES: usize = size_of::<u64>();

/// A timeout policy expressed in caller-defined monotonic ticks.
///
/// The core never reads a clock.  The unit is deliberately left to the caller
/// (nanoseconds, microseconds, a platform tick, or a deterministic test
/// counter).  `None` disables the corresponding limit.  A timeout is reported
/// only when elapsed time is greater than the configured limit, so a limit of
/// zero still permits the first operation at its start tick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamTimeoutPolicy {
    /// Maximum time from the first accepted byte until the stream is finished.
    pub max_total_ticks: Option<u64>,
    /// Maximum time between two input chunks. Consumption does not reset this
    /// timer: backpressure may pause the producer without hiding a stalled
    /// source.
    pub max_idle_ticks: Option<u64>,
}

impl StreamTimeoutPolicy {
    /// A policy with both total and idle timeouts disabled.
    pub const DISABLED: Self = Self {
        max_total_ticks: None,
        max_idle_ticks: None,
    };

    /// Construct a policy from optional caller-defined tick limits.
    #[must_use]
    pub const fn new(max_total_ticks: Option<u64>, max_idle_ticks: Option<u64>) -> Self {
        Self {
            max_total_ticks,
            max_idle_ticks,
        }
    }

    /// Convert `std` durations to the nanosecond tick unit used by
    /// [`StdStreamDecoder`].
    #[cfg(feature = "std")]
    #[must_use]
    pub fn from_durations(
        max_total: Option<std::time::Duration>,
        max_idle: Option<std::time::Duration>,
    ) -> Self {
        fn nanos(duration: std::time::Duration) -> u64 {
            duration.as_nanos().min(u128::from(u64::MAX)) as u64
        }

        Self::new(max_total.map(nanos), max_idle.map(nanos))
    }
}

/// Resource bounds for one framed binary stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamDecodeLimits {
    /// Limits applied by the underlying reflected binary decoder.
    deserialize: DeserializeLimits,
    /// Maximum payload bytes declared by one frame.
    max_frame_bytes: usize,
    /// Maximum accepted wire bytes that have not yet been consumed. This
    /// includes the eight-byte prefix for every queued frame.
    max_buffered_bytes: usize,
    /// Total and idle timeout policy for the core tick API.
    timeout: StreamTimeoutPolicy,
}

impl StreamDecodeLimits {
    /// Conservative defaults for untrusted save/network streams.
    pub const DEFAULT: Self = Self {
        deserialize: DeserializeLimits::DEFAULT,
        max_frame_bytes: 64 * 1024 * 1024,
        max_buffered_bytes: 128 * 1024 * 1024,
        timeout: StreamTimeoutPolicy::DISABLED,
    };

    /// Validate and construct a bounded policy.
    ///
    /// The buffered budget must hold at least one complete maximum-sized frame
    /// and its prefix; otherwise a valid frame could never be admitted or
    /// consumed and the stream would deadlock by construction.
    pub const fn try_new(
        deserialize: DeserializeLimits,
        max_frame_bytes: usize,
        max_buffered_bytes: usize,
        timeout: StreamTimeoutPolicy,
    ) -> Result<Self, StreamLimitsError> {
        if max_frame_bytes == 0 {
            return Err(StreamLimitsError::ZeroFrameLimit);
        }
        if (max_frame_bytes as u128) > u64::MAX as u128 {
            return Err(StreamLimitsError::FrameSizeOverflow);
        }
        let Some(minimum_buffer) = max_frame_bytes.checked_add(FRAME_PREFIX_BYTES) else {
            return Err(StreamLimitsError::BufferSizeOverflow);
        };
        if max_buffered_bytes < minimum_buffer {
            return Err(StreamLimitsError::BufferBelowFrame {
                required: minimum_buffer,
                actual: max_buffered_bytes,
            });
        }
        Ok(Self {
            deserialize,
            max_frame_bytes,
            max_buffered_bytes,
            timeout,
        })
    }

    /// Construct a policy using the default binary-deserialization limits and
    /// disabled timeouts.
    pub const fn bounded(
        max_frame_bytes: usize,
        max_buffered_bytes: usize,
    ) -> Result<Self, StreamLimitsError> {
        Self::try_new(
            DeserializeLimits::DEFAULT,
            max_frame_bytes,
            max_buffered_bytes,
            StreamTimeoutPolicy::DISABLED,
        )
    }

    /// Return a copy with a different binary-deserialization policy.
    #[must_use]
    pub const fn with_deserialize_limits(mut self, deserialize: DeserializeLimits) -> Self {
        self.deserialize = deserialize;
        self
    }

    /// Return a copy with a different timeout policy.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: StreamTimeoutPolicy) -> Self {
        self.timeout = timeout;
        self
    }

    /// Binary-deserialization limits used when a frame is consumed.
    #[must_use]
    pub const fn deserialize_limits(self) -> DeserializeLimits {
        self.deserialize
    }

    /// Maximum payload bytes declared by one frame.
    #[must_use]
    pub const fn max_frame_bytes(self) -> usize {
        self.max_frame_bytes
    }

    /// Maximum accepted, not-yet-consumed wire bytes.
    #[must_use]
    pub const fn max_buffered_bytes(self) -> usize {
        self.max_buffered_bytes
    }

    /// Timeout policy captured by the decoder.
    #[must_use]
    pub const fn timeout(self) -> StreamTimeoutPolicy {
        self.timeout
    }
}

impl Default for StreamDecodeLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Configuration errors caught before a decoder can be constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamLimitsError {
    /// A zero-sized frame limit cannot admit any binary value.
    ZeroFrameLimit,
    /// The platform's `usize` frame limit cannot fit the wire `u64` length.
    FrameSizeOverflow,
    /// Adding the frame prefix to the payload limit overflowed `usize`.
    BufferSizeOverflow,
    /// The hard buffer budget cannot hold one maximum-sized frame.
    BufferBelowFrame {
        /// Minimum required buffer budget.
        required: usize,
        /// Supplied buffer budget.
        actual: usize,
    },
}

impl core::fmt::Display for StreamLimitsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroFrameLimit => f.write_str("stream max_frame_bytes must be non-zero"),
            Self::FrameSizeOverflow => {
                f.write_str("stream max_frame_bytes cannot fit the wire u64 length")
            }
            Self::BufferSizeOverflow => f.write_str("stream frame plus prefix overflows usize"),
            Self::BufferBelowFrame { required, actual } => write!(
                f,
                "stream buffer budget {actual} cannot hold one frame; required {required}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StreamLimitsError {}

/// Which timeout budget expired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamTimeoutKind {
    /// The total stream lifetime exceeded its budget.
    Total,
    /// The interval since the last accepted input chunk exceeded its budget.
    Idle,
}

/// A result of offering one input chunk to a bounded stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamFeed {
    /// Number of bytes accepted. A successful feed accepts the whole chunk.
    pub accepted_bytes: usize,
    /// Accepted wire bytes still held by the decoder.
    pub buffered_bytes: usize,
    /// Complete frames ready for consumption.
    pub ready_frames: usize,
}

/// An explicit admission result suitable for a producer loop.
///
/// `Deferred` is the stream equivalent of `prism_tasks::Admission::Deferred`:
/// no bytes were consumed and the producer should wait for the consumer to
/// drain frames before retrying the exact same chunk. `Rejected` carries a
/// chunk/protocol/timeout reason; the producer must either split the chunk or
/// stop the stream according to that reason.
#[derive(Debug, PartialEq, Eq)]
pub enum StreamOffer {
    /// The complete chunk was accepted.
    Admitted(StreamFeed),
    /// The hard byte budget is full for this chunk; no state changed.
    Deferred {
        /// Bytes the producer attempted to offer.
        requested: usize,
        /// Bytes that can be accepted immediately.
        available: usize,
    },
    /// The stream cannot continue accepting input.
    Rejected(StreamDecodeError),
}

/// Errors raised by the framing, bounded-buffer, timeout, or reflected decode
/// layers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamDecodeError {
    /// The whole chunk did not fit; no part of it was consumed.
    Backpressure {
        /// Bytes requested by the producer.
        requested: usize,
        /// Bytes available before the offer.
        available: usize,
    },
    /// The offered chunk is larger than the entire hard buffer and must be
    /// split before it can ever be admitted.
    ChunkTooLarge {
        /// Bytes offered by the producer.
        requested: usize,
        /// Maximum accepted bytes in one offer.
        limit: usize,
    },
    /// The length prefix declared a payload larger than the policy permits.
    FrameTooLarge {
        /// Declared payload length from the wire.
        declared: u64,
        /// Configured maximum payload length.
        limit: usize,
    },
    /// Input arrived after EOF or a terminal protocol/timeout failure.
    Closed,
    /// EOF arrived while a prefix or payload was incomplete.
    Incomplete {
        /// Wire bytes retained at EOF.
        buffered_bytes: usize,
    },
    /// The total or idle timeout was exceeded.
    TimedOut {
        /// Which budget expired.
        kind: StreamTimeoutKind,
        /// Elapsed ticks observed by the caller.
        elapsed_ticks: u64,
        /// Configured tick budget.
        limit_ticks: u64,
    },
    /// The frame boundary was valid, but its reflected binary payload was not.
    Deserialize(DeserializeError),
}

impl core::fmt::Display for StreamDecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Backpressure {
                requested,
                available,
            } => write!(
                f,
                "stream backpressure: requested {requested}, available {available}"
            ),
            Self::ChunkTooLarge { requested, limit } => write!(
                f,
                "stream input chunk has {requested} bytes, one offer is limited to {limit}"
            ),
            Self::FrameTooLarge { declared, limit } => {
                write!(
                    f,
                    "stream frame declares {declared} bytes, limit is {limit}"
                )
            }
            Self::Closed => f.write_str("stream is closed"),
            Self::Incomplete { buffered_bytes } => {
                write!(f, "stream ended with {buffered_bytes} buffered bytes")
            }
            Self::TimedOut {
                kind,
                elapsed_ticks,
                limit_ticks,
            } => write!(
                f,
                "stream {kind:?} timeout: elapsed {elapsed_ticks}, limit {limit_ticks}"
            ),
            Self::Deserialize(error) => write!(f, "stream frame decode failed: {error}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StreamDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Deserialize(error) => Some(error),
            _ => None,
        }
    }
}

/// The bounded core decoder for `u64 LE length + to_binary payload` frames.
pub struct BinaryStreamDecoder<'a> {
    registry: &'a TypeRegistry,
    target: &'a TypeInfo,
    limits: StreamDecodeLimits,
    prefix: [u8; FRAME_PREFIX_BYTES],
    prefix_len: usize,
    expected_payload: Option<usize>,
    current_payload: Vec<u8>,
    ready: VecDeque<ReadyFrame>,
    buffered_bytes: usize,
    started_at: Option<u64>,
    last_input_at: Option<u64>,
    sealed: bool,
    terminal: bool,
}

struct ReadyFrame {
    payload: Vec<u8>,
    wire_bytes: usize,
}

impl<'a> BinaryStreamDecoder<'a> {
    /// Create a decoder against a registered target schema.
    pub fn new(
        registry: &'a TypeRegistry,
        target: &'a TypeInfo,
        limits: StreamDecodeLimits,
    ) -> Self {
        Self {
            registry,
            target,
            limits,
            prefix: [0; FRAME_PREFIX_BYTES],
            prefix_len: 0,
            expected_payload: None,
            current_payload: Vec::new(),
            ready: VecDeque::new(),
            buffered_bytes: 0,
            started_at: None,
            last_input_at: None,
            sealed: false,
            terminal: false,
        }
    }

    /// Create a decoder and arm its timeout clock at `start_ticks`.
    ///
    /// Use this constructor when waiting for the first byte must be bounded as
    /// well as gaps between later chunks. [`Self::new`] starts lazily on the
    /// first non-empty feed, which is useful when the source is not opened
    /// until after the decoder is constructed.
    pub fn new_at(
        registry: &'a TypeRegistry,
        target: &'a TypeInfo,
        limits: StreamDecodeLimits,
        start_ticks: u64,
    ) -> Self {
        let mut decoder = Self::new(registry, target, limits);
        decoder.started_at = Some(start_ticks);
        decoder.last_input_at = Some(start_ticks);
        decoder
    }

    /// The policy captured by this decoder.
    #[must_use]
    pub const fn limits(&self) -> StreamDecodeLimits {
        self.limits
    }

    /// Accepted wire bytes that have not yet been consumed.
    #[must_use]
    pub const fn buffered_bytes(&self) -> usize {
        self.buffered_bytes
    }

    /// Remaining byte headroom before the hard backpressure limit.
    #[must_use]
    pub const fn available_capacity(&self) -> usize {
        self.limits
            .max_buffered_bytes
            .saturating_sub(self.buffered_bytes)
    }

    /// Number of complete frames waiting for [`Self::consume_next_at`].
    #[must_use]
    pub fn ready_frames(&self) -> usize {
        self.ready.len()
    }

    /// Whether no more input may be accepted.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.sealed || self.terminal
    }

    /// Offer a complete input chunk at caller-defined monotonic `now_ticks`.
    ///
    /// The operation is all-or-nothing with respect to the chunk. This makes
    /// retry after backpressure lossless: callers retain the same source slice
    /// and retry after one or more successful consumes.
    pub fn feed_at(
        &mut self,
        now_ticks: u64,
        chunk: &[u8],
    ) -> Result<StreamFeed, StreamDecodeError> {
        self.ensure_input_open()?;
        self.check_timeout(now_ticks)?;
        if chunk.len() > self.limits.max_buffered_bytes {
            return Err(StreamDecodeError::ChunkTooLarge {
                requested: chunk.len(),
                limit: self.limits.max_buffered_bytes,
            });
        }
        let available = self.available_capacity();
        if chunk.len() > available {
            return Err(StreamDecodeError::Backpressure {
                requested: chunk.len(),
                available,
            });
        }
        if chunk.is_empty() {
            return Ok(StreamFeed {
                accepted_bytes: 0,
                buffered_bytes: self.buffered_bytes,
                ready_frames: self.ready.len(),
            });
        }

        self.buffered_bytes += chunk.len();
        if self.started_at.is_none() {
            self.started_at = Some(now_ticks);
        }
        self.last_input_at = Some(now_ticks);
        if let Err(error) = self.ingest(chunk) {
            self.terminal = true;
            return Err(error);
        }
        Ok(StreamFeed {
            accepted_bytes: chunk.len(),
            buffered_bytes: self.buffered_bytes,
            ready_frames: self.ready.len(),
        })
    }

    /// Offer a chunk using explicit admitted/deferred/rejected flow control.
    pub fn offer_at(&mut self, now_ticks: u64, chunk: &[u8]) -> StreamOffer {
        if chunk.len() > self.limits.max_buffered_bytes && !self.is_closed() {
            return StreamOffer::Rejected(StreamDecodeError::ChunkTooLarge {
                requested: chunk.len(),
                limit: self.limits.max_buffered_bytes,
            });
        }
        if chunk.len() > self.available_capacity() && !self.is_closed() {
            return StreamOffer::Deferred {
                requested: chunk.len(),
                available: self.available_capacity(),
            };
        }
        match self.feed_at(now_ticks, chunk) {
            Ok(feed) => StreamOffer::Admitted(feed),
            Err(StreamDecodeError::Backpressure {
                requested,
                available,
            }) => StreamOffer::Deferred {
                requested,
                available,
            },
            Err(error) => StreamOffer::Rejected(error),
        }
    }

    /// Decode and consume the oldest complete frame, if one is ready.
    ///
    /// The frame's bytes leave the bounded queue before decoding, so a
    /// malformed frame cannot permanently block later length-delimited frames.
    pub fn consume_next_at(
        &mut self,
        now_ticks: u64,
    ) -> Result<Option<Box<dyn Reflect>>, StreamDecodeError> {
        if self.ready.is_empty() {
            if self.terminal {
                return Err(StreamDecodeError::Closed);
            }
            if !self.sealed {
                self.check_timeout(now_ticks)?;
            }
            return Ok(None);
        }
        let frame = self.ready.pop_front().expect("ready checked non-empty");
        self.buffered_bytes = self.buffered_bytes.saturating_sub(frame.wire_bytes);
        from_binary_with_limits(
            &frame.payload,
            self.registry,
            self.target,
            &self.limits.deserialize,
        )
        .map(Some)
        .map_err(StreamDecodeError::Deserialize)
    }

    /// Mark end-of-input at `now_ticks`.
    ///
    /// Complete frames already queued remain consumable after a successful
    /// finish. An incomplete prefix/payload closes the input and returns
    /// [`StreamDecodeError::Incomplete`].
    pub fn finish_at(&mut self, now_ticks: u64) -> Result<(), StreamDecodeError> {
        self.ensure_input_open()?;
        self.check_timeout(now_ticks)?;
        self.sealed = true;
        if self.prefix_len != 0 || self.expected_payload.is_some() {
            return Err(StreamDecodeError::Incomplete {
                buffered_bytes: self.buffered_bytes,
            });
        }
        Ok(())
    }

    fn ensure_input_open(&self) -> Result<(), StreamDecodeError> {
        if self.is_closed() {
            Err(StreamDecodeError::Closed)
        } else {
            Ok(())
        }
    }

    fn check_timeout(&mut self, now_ticks: u64) -> Result<(), StreamDecodeError> {
        if let (Some(start), Some(limit)) = (self.started_at, self.limits.timeout.max_total_ticks) {
            let elapsed = now_ticks.saturating_sub(start);
            if elapsed > limit {
                self.terminal = true;
                return Err(StreamDecodeError::TimedOut {
                    kind: StreamTimeoutKind::Total,
                    elapsed_ticks: elapsed,
                    limit_ticks: limit,
                });
            }
        }
        if let (Some(last_input), Some(limit)) =
            (self.last_input_at, self.limits.timeout.max_idle_ticks)
        {
            let elapsed = now_ticks.saturating_sub(last_input);
            if elapsed > limit {
                self.terminal = true;
                return Err(StreamDecodeError::TimedOut {
                    kind: StreamTimeoutKind::Idle,
                    elapsed_ticks: elapsed,
                    limit_ticks: limit,
                });
            }
        }
        Ok(())
    }

    fn ingest(&mut self, mut input: &[u8]) -> Result<(), StreamDecodeError> {
        while !input.is_empty() {
            if self.expected_payload.is_none() {
                let take = (FRAME_PREFIX_BYTES - self.prefix_len).min(input.len());
                self.prefix[self.prefix_len..self.prefix_len + take]
                    .copy_from_slice(&input[..take]);
                self.prefix_len += take;
                input = &input[take..];
                if self.prefix_len != FRAME_PREFIX_BYTES {
                    continue;
                }

                let declared = u64::from_le_bytes(self.prefix);
                if declared > self.limits.max_frame_bytes as u64 {
                    return Err(StreamDecodeError::FrameTooLarge {
                        declared,
                        limit: self.limits.max_frame_bytes,
                    });
                }
                self.expected_payload = Some(declared as usize);
                self.prefix_len = 0;
                self.current_payload.clear();
                if declared == 0 {
                    self.finish_current_frame();
                }
                continue;
            }

            let expected = self
                .expected_payload
                .expect("expected payload checked above");
            let remaining = expected.saturating_sub(self.current_payload.len());
            let take = remaining.min(input.len());
            let needed = self.current_payload.len() + take;
            if self.current_payload.capacity() < needed {
                self.current_payload
                    .reserve_exact(needed - self.current_payload.capacity());
            }
            self.current_payload.extend_from_slice(&input[..take]);
            input = &input[take..];
            if self.current_payload.len() == expected {
                self.finish_current_frame();
            }
        }
        Ok(())
    }

    fn finish_current_frame(&mut self) {
        let payload = core::mem::take(&mut self.current_payload);
        let wire_bytes = FRAME_PREFIX_BYTES + payload.len();
        self.ready.push_back(ReadyFrame {
            payload,
            wire_bytes,
        });
        self.expected_payload = None;
    }
}

/// Prefix a complete `from_binary` payload for use with
/// [`BinaryStreamDecoder`].
pub fn frame_binary(payload: &[u8]) -> Result<Vec<u8>, StreamEncodeError> {
    let capacity =
        FRAME_PREFIX_BYTES
            .checked_add(payload.len())
            .ok_or(StreamEncodeError::FrameTooLarge {
                bytes: payload.len(),
            })?;
    let declared = u64::try_from(payload.len()).map_err(|_| StreamEncodeError::FrameTooLarge {
        bytes: payload.len(),
    })?;
    let mut framed = Vec::with_capacity(capacity);
    framed.extend_from_slice(&declared.to_le_bytes());
    framed.extend_from_slice(payload);
    Ok(framed)
}

/// Serialize a reflected value and prefix it as one stream frame.
pub fn to_binary_frame(value: &dyn Reflect) -> Result<Vec<u8>, StreamEncodeError> {
    let payload = to_binary(value).map_err(StreamEncodeError::Serialize)?;
    frame_binary(&payload)
}

/// Errors raised while creating a stream frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamEncodeError {
    /// The reflected value could not be serialized.
    Serialize(SerializeError),
    /// The payload plus its prefix cannot be represented by the allocator or
    /// the fixed-width length field on the current target.
    FrameTooLarge {
        /// Payload bytes that could not be represented.
        bytes: usize,
    },
}

impl core::fmt::Display for StreamEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Serialize(error) => write!(f, "stream frame serialization failed: {error}"),
            Self::FrameTooLarge { bytes } => {
                write!(f, "stream frame payload is too large: {bytes} bytes")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StreamEncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialize(error) => Some(error),
            Self::FrameTooLarge { .. } => None,
        }
    }
}

/// `std` adapter that supplies monotonic nanosecond ticks to the no-std core.
#[cfg(feature = "std")]
pub struct StdStreamDecoder<'a> {
    inner: BinaryStreamDecoder<'a>,
    epoch: std::time::Instant,
}

#[cfg(feature = "std")]
impl<'a> StdStreamDecoder<'a> {
    /// Construct a stream decoder whose timeout policy is interpreted as
    /// nanoseconds. Use [`StreamTimeoutPolicy::from_durations`] to configure
    /// human-readable `Duration` values.
    pub fn new(
        registry: &'a TypeRegistry,
        target: &'a TypeInfo,
        limits: StreamDecodeLimits,
    ) -> Self {
        Self {
            inner: BinaryStreamDecoder::new_at(registry, target, limits, 0),
            epoch: std::time::Instant::now(),
        }
    }

    /// Offer a chunk using the adapter's monotonic clock.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<StreamFeed, StreamDecodeError> {
        self.inner.feed_at(self.now_ticks(), chunk)
    }

    /// Offer a chunk with explicit deferred/rejected admission.
    pub fn offer(&mut self, chunk: &[u8]) -> StreamOffer {
        self.inner.offer_at(self.now_ticks(), chunk)
    }

    /// Decode and consume one ready frame.
    pub fn consume_next(&mut self) -> Result<Option<Box<dyn Reflect>>, StreamDecodeError> {
        self.inner.consume_next_at(self.now_ticks())
    }

    /// Mark end-of-input.
    pub fn finish(&mut self) -> Result<(), StreamDecodeError> {
        self.inner.finish_at(self.now_ticks())
    }

    /// Access the underlying core decoder for metrics and integration.
    #[must_use]
    pub const fn core(&self) -> &BinaryStreamDecoder<'a> {
        &self.inner
    }

    /// Mutable access to the underlying core decoder.
    pub fn core_mut(&mut self) -> &mut BinaryStreamDecoder<'a> {
        &mut self.inner
    }

    fn now_ticks(&self) -> u64 {
        self.epoch.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FromReflect, Typed};
    use alloc::vec;

    fn registry() -> TypeRegistry {
        let mut registry = TypeRegistry::new();
        registry.register::<Vec<u32>>();
        registry.register::<u32>();
        registry
    }

    fn decoder<'a>(registry: &'a TypeRegistry) -> BinaryStreamDecoder<'a> {
        BinaryStreamDecoder::new(
            registry,
            <Vec<u32> as Typed>::type_info(),
            StreamDecodeLimits::bounded(1024, 2048).expect("valid limits"),
        )
    }

    #[test]
    fn fragmented_frame_is_decoded_only_after_complete_payload() {
        let registry = registry();
        let mut decoder = decoder(&registry);
        let frame = to_binary_frame(&vec![1_u32, 2, 3]).expect("frame");

        assert_eq!(decoder.feed_at(0, &frame[..3]).unwrap().ready_frames, 0);
        assert!(decoder.consume_next_at(0).unwrap().is_none());
        let middle = &frame[3..frame.len() - 1];
        assert_eq!(decoder.feed_at(1, middle).unwrap().ready_frames, 0);
        assert_eq!(
            decoder
                .feed_at(2, &frame[frame.len() - 1..])
                .unwrap()
                .ready_frames,
            1
        );
        let value = decoder.consume_next_at(2).unwrap().expect("decoded value");
        let concrete = Vec::<u32>::from_reflect(value.as_ref()).expect("typed value");
        assert_eq!(concrete, vec![1, 2, 3]);
        assert_eq!(decoder.buffered_bytes(), 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn std_adapter_drives_the_same_core_contract() {
        let registry = registry();
        let limits = StreamDecodeLimits::bounded(1024, 2048).expect("valid limits");
        let mut decoder =
            StdStreamDecoder::new(&registry, <Vec<u32> as Typed>::type_info(), limits);
        let frame = to_binary_frame(&vec![8_u32, 13]).expect("frame");
        decoder.feed(&frame).expect("accepted");
        let value = decoder.consume_next().unwrap().expect("decoded value");
        assert_eq!(
            Vec::<u32>::from_reflect(value.as_ref()).unwrap(),
            vec![8, 13]
        );
    }

    #[test]
    fn backpressure_is_all_or_nothing_and_releases_after_consume() {
        let registry = registry();
        let frame = to_binary_frame(&vec![9_u32; 4]).expect("frame");
        let payload_len = frame.len() - FRAME_PREFIX_BYTES;
        let limits = StreamDecodeLimits::bounded(payload_len, frame.len()).expect("valid limits");
        let mut decoder =
            BinaryStreamDecoder::new(&registry, <Vec<u32> as Typed>::type_info(), limits);
        let second = frame.clone();

        assert!(matches!(
            decoder.offer_at(0, &second),
            StreamOffer::Admitted(_)
        ));
        assert_eq!(decoder.buffered_bytes(), frame.len());
        assert!(matches!(
            decoder.offer_at(0, &second),
            StreamOffer::Deferred {
                requested,
                available: 0
            } if requested == second.len()
        ));
        decoder.consume_next_at(0).unwrap().expect("first frame");
        assert!(matches!(
            decoder.offer_at(0, &second),
            StreamOffer::Admitted(_)
        ));
    }

    #[test]
    fn oversized_declared_frame_is_terminal_before_payload_allocation() {
        let registry = registry();
        let mut decoder = decoder(&registry);
        let declared = 4096_u64.to_le_bytes();
        let error = decoder.feed_at(0, &declared).unwrap_err();
        assert_eq!(
            error,
            StreamDecodeError::FrameTooLarge {
                declared: 4096,
                limit: 1024
            }
        );
        assert!(matches!(
            decoder.feed_at(1, &[]),
            Err(StreamDecodeError::Closed)
        ));
    }

    #[test]
    fn an_offer_larger_than_the_hard_buffer_requires_producer_chunking() {
        let registry = registry();
        let limits = StreamDecodeLimits::bounded(16, 24).expect("valid limits");
        let mut decoder =
            BinaryStreamDecoder::new(&registry, <Vec<u32> as Typed>::type_info(), limits);
        let chunk = [0_u8; 25];
        assert_eq!(
            decoder.offer_at(0, &chunk),
            StreamOffer::Rejected(StreamDecodeError::ChunkTooLarge {
                requested: 25,
                limit: 24
            })
        );
        assert_eq!(decoder.buffered_bytes(), 0);
        assert!(!decoder.is_closed());
    }

    #[test]
    fn timeout_uses_total_and_idle_tick_budgets() {
        let registry = registry();
        let limits = StreamDecodeLimits::bounded(1024, 2048)
            .unwrap()
            .with_timeout(StreamTimeoutPolicy::new(Some(10), Some(3)));
        let mut decoder =
            BinaryStreamDecoder::new(&registry, <Vec<u32> as Typed>::type_info(), limits);
        let frame = to_binary_frame(&vec![1_u32]).unwrap();
        decoder.feed_at(100, &frame[..1]).unwrap();
        assert_eq!(
            decoder.feed_at(103, &frame[1..2]).unwrap().accepted_bytes,
            1
        );
        let error = decoder.feed_at(107, &frame[2..]).unwrap_err();
        assert_eq!(
            error,
            StreamDecodeError::TimedOut {
                kind: StreamTimeoutKind::Idle,
                elapsed_ticks: 4,
                limit_ticks: 3
            }
        );
    }

    #[test]
    fn armed_timeout_covers_waiting_for_the_first_chunk() {
        let registry = registry();
        let limits = StreamDecodeLimits::bounded(1024, 2048)
            .unwrap()
            .with_timeout(StreamTimeoutPolicy::new(None, Some(3)));
        let mut decoder =
            BinaryStreamDecoder::new_at(&registry, <Vec<u32> as Typed>::type_info(), limits, 10);
        let error = decoder.feed_at(14, &[0]).unwrap_err();
        assert_eq!(
            error,
            StreamDecodeError::TimedOut {
                kind: StreamTimeoutKind::Idle,
                elapsed_ticks: 4,
                limit_ticks: 3
            }
        );
    }

    #[test]
    fn finish_rejects_partial_frame_but_keeps_complete_frames_consumable() {
        let registry = registry();
        let mut decoder = decoder(&registry);
        let frame = to_binary_frame(&vec![7_u32]).unwrap();
        let mut bytes = frame.clone();
        bytes.extend_from_slice(&[1, 2, 3]);
        decoder.feed_at(0, &bytes).unwrap();
        assert_eq!(
            decoder.finish_at(0).unwrap_err(),
            StreamDecodeError::Incomplete {
                buffered_bytes: bytes.len()
            }
        );
        assert_eq!(decoder.ready_frames(), 1);
        decoder.consume_next_at(0).unwrap().expect("complete frame");
        assert_eq!(decoder.buffered_bytes(), 3);
        assert!(matches!(
            decoder.feed_at(1, &[]),
            Err(StreamDecodeError::Closed)
        ));
    }

    #[test]
    fn malformed_frame_does_not_desynchronise_following_frame() {
        let registry = registry();
        let mut decoder = decoder(&registry);
        let malformed = frame_binary(b"not a prism binary stream").unwrap();
        let valid = to_binary_frame(&vec![5_u32]).unwrap();
        let mut bytes = malformed;
        bytes.extend_from_slice(&valid);
        decoder.feed_at(0, &bytes).unwrap();
        assert!(matches!(
            decoder.consume_next_at(0),
            Err(StreamDecodeError::Deserialize(DeserializeError::BadMagic))
        ));
        let value = decoder.consume_next_at(0).unwrap().expect("second frame");
        assert_eq!(Vec::<u32>::from_reflect(value.as_ref()).unwrap(), vec![5]);
    }
}
