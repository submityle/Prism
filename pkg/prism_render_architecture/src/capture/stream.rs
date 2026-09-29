//! Ordered frame records and the append-only capture stream.
//!
//! A [`CaptureStream`] is the `CPU`-side recording of a render session: an
//! ordered list of [`FrameRecord`]s. Each record carries a monotonically
//! increasing `frame_index`, optional performance counters, and an opaque
//! command-summary blob. The command-summary bytes are a deterministic
//! `CPU`-side digest today; wiring in a real `GPU` command stream is a future
//! backend concern (see [`FrameRecord::command_digest`]).
//!
//! The stream is intentionally minimal and side-effect free so it can be
//! recorded, cloned, hashed, and replayed deterministically without a `GPU`.

use alloc::vec::Vec;

use crate::abi::AbiHash;

use super::hash_chain::FrameHashChain;

/// A single recorded frame in a capture stream.
///
/// `frame_index` is the authoritative ordering key checked during replay. The
/// two optional counters model per-frame telemetry that may or may not be
/// present depending on the capture mode, and `command_digest` holds an opaque
/// summary of the frame's submitted work.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FrameRecord {
    /// Monotonic frame ordinal; replay requires these to advance by exactly one.
    pub frame_index: u64,
    /// Optional `GPU` frame duration counter, in nanoseconds.
    ///
    /// Populated by the future `GPU` backend; `None` when timing was not
    /// captured for this frame.
    pub gpu_duration_ns: Option<u64>,
    /// Optional recorded draw-call count for the frame.
    pub draw_call_count: Option<u32>,
    /// Opaque `CPU`-side summary of the frame's submitted commands.
    ///
    /// Hashed verbatim into the capture digest so any change is observable. A
    /// real `GPU` command-stream encoder will populate this once the backend
    /// exists; until then callers supply their own deterministic summary.
    pub command_digest: Vec<u8>,
}

impl FrameRecord {
    /// Creates a bare record at `frame_index` with no counters or summary.
    #[must_use]
    pub fn new(frame_index: u64) -> Self {
        Self {
            frame_index,
            gpu_duration_ns: None,
            draw_call_count: None,
            command_digest: Vec::new(),
        }
    }

    /// Attaches a `GPU` duration counter (builder style).
    #[must_use]
    pub fn with_gpu_duration_ns(mut self, duration_ns: u64) -> Self {
        self.gpu_duration_ns = Some(duration_ns);
        self
    }

    /// Attaches a draw-call counter (builder style).
    #[must_use]
    pub fn with_draw_call_count(mut self, draw_calls: u32) -> Self {
        self.draw_call_count = Some(draw_calls);
        self
    }

    /// Attaches an opaque command-summary blob (builder style).
    #[must_use]
    pub fn with_command_digest(mut self, digest: impl Into<Vec<u8>>) -> Self {
        self.command_digest = digest.into();
        self
    }

    /// Serializes the record into a deterministic, self-delimiting byte blob.
    ///
    /// The encoding is stable across hosts: little-endian integers, a one-byte
    /// presence tag for each optional counter, and a length prefix for the
    /// command summary so its boundary is significant when hashed.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.frame_index.to_le_bytes());
        encode_opt_u64(&mut out, self.gpu_duration_ns);
        encode_opt_u32(&mut out, self.draw_call_count);
        out.extend_from_slice(&(self.command_digest.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.command_digest);
        out
    }
}

/// Encodes an optional `u64` as a presence tag followed by the value.
fn encode_opt_u64(out: &mut Vec<u8>, value: Option<u64>) {
    match value {
        Some(v) => {
            out.push(1);
            out.extend_from_slice(&v.to_le_bytes());
        }
        None => out.push(0),
    }
}

/// Encodes an optional `u32` as a presence tag followed by the value.
fn encode_opt_u32(out: &mut Vec<u8>, value: Option<u32>) {
    match value {
        Some(v) => {
            out.push(1);
            out.extend_from_slice(&v.to_le_bytes());
        }
        None => out.push(0),
    }
}

/// An append-only, order-preserving sequence of [`FrameRecord`]s.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CaptureStream {
    frames: Vec<FrameRecord>,
}

impl CaptureStream {
    /// Creates an empty stream.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty stream with room for `capacity` frames.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            frames: Vec::with_capacity(capacity),
        }
    }

    /// Appends a frame, preserving insertion order.
    pub fn push(&mut self, frame: FrameRecord) {
        self.frames.push(frame);
    }

    /// Returns `true` when no frames have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns the number of recorded frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns the recorded frames in order.
    #[must_use]
    pub fn frames(&self) -> &[FrameRecord] {
        &self.frames
    }

    /// Returns the frame at `position`, if any.
    #[must_use]
    pub fn get(&self, position: usize) -> Option<&FrameRecord> {
        self.frames.get(position)
    }

    /// Iterates the recorded frames in order.
    pub fn iter(&self) -> core::slice::Iter<'_, FrameRecord> {
        self.frames.iter()
    }

    /// Returns the first frame's index, or `None` for an empty stream.
    #[must_use]
    pub fn first_frame_index(&self) -> Option<u64> {
        self.frames.first().map(|frame| frame.frame_index)
    }

    /// Returns the last frame's index, or `None` for an empty stream.
    #[must_use]
    pub fn last_frame_index(&self) -> Option<u64> {
        self.frames.last().map(|frame| frame.frame_index)
    }

    /// Computes the rolling chain digest over all frames from `seed`.
    ///
    /// An empty stream returns the seed digest; the result is order- and
    /// byte-sensitive across the recorded frames.
    #[must_use]
    pub fn chain_hash(&self, seed: u64) -> AbiHash {
        let mut chain = FrameHashChain::seeded(seed);
        for frame in &self.frames {
            let encoded = frame.encode();
            chain.absorb(&encoded);
        }
        chain.chain_hash()
    }

    /// Computes the per-frame running digest after each frame, in order.
    ///
    /// The returned vector has one entry per frame; entry `i` is the chain
    /// digest after folding frames `0..=i`.
    #[must_use]
    pub fn frame_hashes(&self, seed: u64) -> Vec<AbiHash> {
        let mut chain = FrameHashChain::seeded(seed);
        let mut hashes = Vec::with_capacity(self.frames.len());
        for frame in &self.frames {
            let encoded = frame.encode();
            hashes.push(chain.absorb(&encoded));
        }
        hashes
    }
}

impl<'a> IntoIterator for &'a CaptureStream {
    type Item = &'a FrameRecord;
    type IntoIter = core::slice::Iter<'a, FrameRecord>;

    fn into_iter(self) -> Self::IntoIter {
        self.frames.iter()
    }
}

impl FromIterator<FrameRecord> for CaptureStream {
    fn from_iter<I: IntoIterator<Item = FrameRecord>>(iter: I) -> Self {
        Self {
            frames: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn new_stream_is_empty() {
        let stream = CaptureStream::new();
        assert!(stream.is_empty());
        assert_eq!(stream.len(), 0);
        assert_eq!(stream.first_frame_index(), None);
        assert_eq!(stream.last_frame_index(), None);
        assert_eq!(stream.frames(), &[] as &[FrameRecord]);
    }

    #[test]
    fn push_preserves_order_and_indices() {
        let mut stream = CaptureStream::new();
        stream.push(FrameRecord::new(0));
        stream.push(FrameRecord::new(1));
        stream.push(FrameRecord::new(2));
        assert_eq!(stream.len(), 3);
        assert!(!stream.is_empty());
        assert_eq!(stream.first_frame_index(), Some(0));
        assert_eq!(stream.last_frame_index(), Some(2));
        let collected: Vec<u64> = stream.iter().map(|frame| frame.frame_index).collect();
        assert_eq!(collected, vec![0, 1, 2]);
    }

    #[test]
    fn get_returns_expected_frame() {
        let mut stream = CaptureStream::new();
        stream.push(FrameRecord::new(10).with_draw_call_count(4));
        assert_eq!(stream.get(0).map(|f| f.frame_index), Some(10));
        assert_eq!(stream.get(0).and_then(|f| f.draw_call_count), Some(4));
        assert!(stream.get(1).is_none());
    }

    #[test]
    fn builders_populate_optional_fields() {
        let record = FrameRecord::new(3)
            .with_gpu_duration_ns(1_234)
            .with_draw_call_count(7)
            .with_command_digest(vec![9u8, 8, 7]);
        assert_eq!(record.gpu_duration_ns, Some(1_234));
        assert_eq!(record.draw_call_count, Some(7));
        assert_eq!(record.command_digest, vec![9u8, 8, 7]);
    }

    #[test]
    fn encoding_is_deterministic_and_field_sensitive() {
        let base = FrameRecord::new(1).with_command_digest(vec![1u8, 2, 3]);
        assert_eq!(base.encode(), base.clone().encode());

        // A present-but-zero counter differs from an absent counter.
        let absent = FrameRecord::new(1).with_command_digest(vec![1u8, 2, 3]);
        let present_zero = FrameRecord::new(1)
            .with_command_digest(vec![1u8, 2, 3])
            .with_draw_call_count(0);
        assert_ne!(absent.encode(), present_zero.encode());
    }

    #[test]
    fn command_digest_boundary_is_significant() {
        // Splitting the same bytes across index vs digest must change encoding.
        let a = FrameRecord::new(0x0102_0304).with_command_digest(vec![0u8]);
        let b = FrameRecord::new(0x0102_0304).with_command_digest(vec![0u8, 0]);
        assert_ne!(a.encode(), b.encode());
    }

    #[test]
    fn chain_hash_of_empty_is_seed_digest() {
        let stream = CaptureStream::new();
        assert_ne!(stream.chain_hash(42), AbiHash::ZERO);
        // Two empty streams with the same seed agree.
        assert_eq!(stream.chain_hash(42), CaptureStream::new().chain_hash(42));
    }

    #[test]
    fn chain_hash_is_order_sensitive() {
        let ordered: CaptureStream = [FrameRecord::new(0), FrameRecord::new(1)]
            .into_iter()
            .collect();
        let mut swapped = CaptureStream::new();
        swapped.push(FrameRecord::new(1));
        swapped.push(FrameRecord::new(0));
        assert_ne!(ordered.chain_hash(0), swapped.chain_hash(0));
    }

    #[test]
    fn frame_hashes_match_incremental_chain() {
        let mut stream = CaptureStream::new();
        stream.push(FrameRecord::new(0).with_command_digest(vec![1u8]));
        stream.push(FrameRecord::new(1).with_command_digest(vec![2u8]));
        let hashes = stream.frame_hashes(3);
        assert_eq!(hashes.len(), 2);
        // The last per-frame hash equals the summary chain hash.
        assert_eq!(*hashes.last().unwrap(), stream.chain_hash(3));
        assert_ne!(hashes[0], hashes[1]);
    }

    #[test]
    fn into_iter_and_from_iter_round_trip() {
        let source = [FrameRecord::new(5), FrameRecord::new(6)];
        let stream: CaptureStream = source.iter().cloned().collect();
        let echoed: Vec<u64> = (&stream).into_iter().map(|f| f.frame_index).collect();
        assert_eq!(echoed, vec![5, 6]);
    }
}
