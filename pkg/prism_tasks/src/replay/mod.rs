//! Record and replay of deterministic scheduling decisions (design §13 可回放,
//! §24.7).
//!
//! Rollback-netcode and record/replay debugging need a run to be reproducible:
//! *same input → same output*, independent of how many workers happened to be
//! free or in what order they stole work. The deterministic primitives in
//! [`crate::reduce`] already guarantee that by splitting through a
//! [`FixedPartition`](crate::FixedPartition), but the *split itself* is a
//! decision worth capturing — once recorded it can be replayed byte-for-byte on
//! another machine with a different core count.
//!
//! A [`DeterministicSession`] is the recorder/player:
//! * [`DeterministicSession::record`] starts a fresh log seeded with a `u64`.
//!   Each deterministic operation asks the session to [`plan`](DeterministicSession::plan)
//!   a split for a given length; the session derives a grain from the seed
//!   (reproducibly) and appends a [`SplitEvent`].
//! * [`DeterministicSession::finish`] hands back a [`ReplayRecord`], which
//!   serializes to compact text ([`ReplayRecord::to_text`]) or bytes
//!   ([`ReplayRecord::to_bytes`]).
//! * [`DeterministicSession::replay`] drives a run from a stored record: each
//!   `plan` pops the next recorded event, checks the length matches, and
//!   returns the recorded grain, so the operation re-executes the identical
//!   split.
//!
//! The seed makes the split a pure, reproducible function of
//! `(seed, length)` — different seeds explore different (each individually
//! deterministic) splits, and a replay reproduces whichever one was recorded.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use std::sync::Mutex;

use crate::partition::{DEFAULT_TARGET_CHUNKS, FixedPartition};

/// A single recorded split decision: an operation over `len` elements chose
/// chunks of `grain` elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitEvent {
    /// Number of elements the operation partitioned.
    pub len: usize,
    /// Fixed chunk size chosen for the split.
    pub grain: usize,
}

/// A serializable, replayable log of the split decisions a
/// [`DeterministicSession`] made, plus the seed that produced them.
///
/// Reproduce a run by feeding this back into
/// [`DeterministicSession::replay`]. The record round-trips through
/// [`ReplayRecord::to_text`]/[`ReplayRecord::from_text`] and
/// [`ReplayRecord::to_bytes`]/[`ReplayRecord::from_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayRecord {
    seed: u64,
    events: Vec<SplitEvent>,
}

/// Magic header distinguishing a serialized [`ReplayRecord`] and pinning the
/// format version.
const TEXT_MAGIC: &str = "prism-replay v1";
/// Binary magic bytes (`PRPL`) followed by a one-byte version.
const BYTES_MAGIC: [u8; 4] = *b"PRPL";
/// Binary format version.
const BYTES_VERSION: u8 = 1;

impl ReplayRecord {
    /// The seed the recording session used.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The recorded split events, in the order they were made.
    #[must_use]
    pub fn events(&self) -> &[SplitEvent] {
        &self.events
    }

    /// Serialize to a compact, human-readable text form:
    /// `prism-replay v1 seed=<hex>` followed by one `len:grain` line per event.
    #[must_use]
    pub fn to_text(&self) -> String {
        use core::fmt::Write as _;
        let mut out = String::new();
        // Writing to a `String` is infallible.
        let _ = writeln!(out, "{TEXT_MAGIC} seed={:016x}", self.seed);
        for event in &self.events {
            let _ = writeln!(out, "{}:{}", event.len, event.grain);
        }
        out
    }

    /// Parse the text form produced by [`ReplayRecord::to_text`].
    ///
    /// # Errors
    /// Returns [`ReplayError`] if the header, seed, or any event line is
    /// malformed.
    pub fn from_text(text: &str) -> Result<Self, ReplayError> {
        let mut lines = text.lines();
        let header = lines.next().ok_or(ReplayError::Truncated)?;
        let seed_str = header
            .strip_prefix(TEXT_MAGIC)
            .and_then(|rest| rest.trim().strip_prefix("seed="))
            .ok_or(ReplayError::BadHeader)?;
        let seed = u64::from_str_radix(seed_str.trim(), 16).map_err(|_| ReplayError::BadHeader)?;
        let mut events = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let (len, grain) = line.split_once(':').ok_or(ReplayError::BadEvent)?;
            let len = len.trim().parse().map_err(|_| ReplayError::BadEvent)?;
            let grain = grain.trim().parse().map_err(|_| ReplayError::BadEvent)?;
            events.push(SplitEvent { len, grain });
        }
        Ok(Self { seed, events })
    }

    /// Serialize to a compact little-endian binary form.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(13 + self.events.len() * 16);
        out.extend_from_slice(&BYTES_MAGIC);
        out.push(BYTES_VERSION);
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&(self.events.len() as u64).to_le_bytes());
        for event in &self.events {
            out.extend_from_slice(&(event.len as u64).to_le_bytes());
            out.extend_from_slice(&(event.grain as u64).to_le_bytes());
        }
        out
    }

    /// Parse the binary form produced by [`ReplayRecord::to_bytes`].
    ///
    /// # Errors
    /// Returns [`ReplayError`] if the magic, version, or length prefix is
    /// invalid or the buffer is truncated.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReplayError> {
        if bytes.len() < 13 {
            return Err(ReplayError::Truncated);
        }
        if bytes[..4] != BYTES_MAGIC {
            return Err(ReplayError::BadHeader);
        }
        if bytes[4] != BYTES_VERSION {
            return Err(ReplayError::BadVersion);
        }
        let mut cursor = 5;
        let seed = read_u64(bytes, &mut cursor)?;
        let count = read_u64(bytes, &mut cursor)? as usize;
        let mut events = Vec::with_capacity(count);
        for _ in 0..count {
            let len = read_u64(bytes, &mut cursor)? as usize;
            let grain = read_u64(bytes, &mut cursor)? as usize;
            events.push(SplitEvent { len, grain });
        }
        Ok(Self { seed, events })
    }
}

/// Read a little-endian `u64` at `*cursor`, advancing it, or fail if the slice
/// is too short.
fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, ReplayError> {
    let end = *cursor + 8;
    let slice = bytes.get(*cursor..end).ok_or(ReplayError::Truncated)?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(slice);
    *cursor = end;
    Ok(u64::from_le_bytes(buf))
}

/// Errors from parsing a [`ReplayRecord`] or replaying a mismatched run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayError {
    /// The serialized buffer ended before a full record was read.
    Truncated,
    /// The magic header or seed field was malformed.
    BadHeader,
    /// The binary version byte is not understood by this build.
    BadVersion,
    /// An event line/record was malformed.
    BadEvent,
    /// A replay step saw a different length than was recorded (the replayed
    /// program diverged from the recording), with `expected` and `found`.
    LengthMismatch {
        /// The length stored in the record for this step.
        expected: usize,
        /// The length the replayed operation actually requested.
        found: usize,
    },
    /// A replay asked for more splits than the record holds.
    Exhausted,
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("replay record is truncated"),
            Self::BadHeader => f.write_str("replay record has a bad header"),
            Self::BadVersion => f.write_str("replay record has an unsupported version"),
            Self::BadEvent => f.write_str("replay record has a malformed event"),
            Self::LengthMismatch { expected, found } => {
                write!(f, "replay length mismatch: expected {expected}, found {found}")
            }
            Self::Exhausted => f.write_str("replay record ran out of events"),
        }
    }
}

impl std::error::Error for ReplayError {}

/// Inner state of a [`DeterministicSession`].
enum Mode {
    /// Recording: derive splits from the seed and append them.
    Record(Mutex<Vec<SplitEvent>>),
    /// Replaying: step through a recorded log.
    Replay {
        /// The record being replayed.
        record: ReplayRecord,
        /// Index of the next event to consume.
        cursor: Mutex<usize>,
    },
}

/// A deterministic record-or-replay session threaded through deterministic
/// operations (see [`crate::TaskPool::deterministic_reduce_with`]).
///
/// In [`record`](DeterministicSession::record) mode the session *derives* each
/// split from its seed and logs it; in [`replay`](DeterministicSession::replay)
/// mode it *reproduces* the logged splits. Either way the split is independent
/// of worker count, so the computation it drives is bit-reproducible.
///
/// A session may be shared across threads (`&DeterministicSession`): its
/// internal cursor/log are mutex-guarded, and deterministic operations call
/// [`plan`](DeterministicSession::plan) once per operation (not per chunk), so
/// contention is negligible.
pub struct DeterministicSession {
    seed: u64,
    mode: Mode,
}

impl DeterministicSession {
    /// Start a fresh recording session seeded with `seed`.
    #[must_use]
    pub fn record(seed: u64) -> Self {
        Self {
            seed,
            mode: Mode::Record(Mutex::new(Vec::new())),
        }
    }

    /// Start a replay session that reproduces `record`.
    #[must_use]
    pub fn replay(record: ReplayRecord) -> Self {
        Self {
            seed: record.seed,
            mode: Mode::Replay {
                record,
                cursor: Mutex::new(0),
            },
        }
    }

    /// The session seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Whether this session is replaying a prior recording.
    #[must_use]
    pub fn is_replaying(&self) -> bool {
        matches!(self.mode, Mode::Replay { .. })
    }

    /// Plan the deterministic split for an operation over `len` elements.
    ///
    /// Recording derives a reproducible grain from `(seed, len)` and logs it;
    /// replaying returns the recorded grain after checking the length matches.
    ///
    /// # Errors
    /// In replay mode, returns [`ReplayError::Exhausted`] if the record has no
    /// more events or [`ReplayError::LengthMismatch`] if the replayed program
    /// diverged from the recording.
    pub fn plan(&self, len: usize) -> Result<FixedPartition, ReplayError> {
        match &self.mode {
            Mode::Record(events) => {
                let grain = derive_grain(self.seed, len);
                let partition = FixedPartition::with_grain(len, grain);
                events
                    .lock()
                    .unwrap()
                    .push(SplitEvent {
                        len,
                        grain: partition.grain(),
                    });
                Ok(partition)
            }
            Mode::Replay { record, cursor } => {
                let mut idx = cursor.lock().unwrap();
                let event = record.events.get(*idx).ok_or(ReplayError::Exhausted)?;
                if event.len != len {
                    return Err(ReplayError::LengthMismatch {
                        expected: event.len,
                        found: len,
                    });
                }
                *idx += 1;
                Ok(FixedPartition::with_grain(len, event.grain))
            }
        }
    }

    /// Finish recording and return the accumulated [`ReplayRecord`].
    ///
    /// For a replay session this returns the record being replayed (with its
    /// original seed and events), so round-tripping is lossless.
    #[must_use]
    pub fn finish(self) -> ReplayRecord {
        match self.mode {
            Mode::Record(events) => ReplayRecord {
                seed: self.seed,
                events: events.into_inner().unwrap(),
            },
            Mode::Replay { record, .. } => record,
        }
    }
}

/// Derive a reproducible grain from `(seed, len)`.
///
/// The seed perturbs the target chunk count within `[1, 2*DEFAULT]` via a
/// `splitmix64` step, so distinct seeds explore distinct (each individually
/// deterministic and worker-count-independent) splits. `len == 0` yields a
/// grain of `1` (an empty partition has no chunks regardless).
fn derive_grain(seed: u64, len: usize) -> usize {
    if len == 0 {
        return 1;
    }
    // splitmix64 finalizer mixing the seed with the length.
    let mut z = seed
        .wrapping_add(len as u64)
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    // Target chunk count in [1, 2*DEFAULT_TARGET_CHUNKS].
    let span = (DEFAULT_TARGET_CHUNKS * 2) as u64;
    let target = (z % span) as usize + 1;
    FixedPartition::with_target_chunks(len, target).grain()
}

#[cfg(test)]
mod tests {
    use super::{DeterministicSession, ReplayError, ReplayRecord, SplitEvent};

    #[test]
    fn record_then_replay_reproduces_splits() {
        let rec = DeterministicSession::record(0xdead_beef);
        let a = rec.plan(1000).unwrap();
        let b = rec.plan(37).unwrap();
        let record = rec.finish();

        let replay = DeterministicSession::replay(record);
        assert!(replay.is_replaying());
        assert_eq!(replay.plan(1000).unwrap(), a);
        assert_eq!(replay.plan(37).unwrap(), b);
    }

    #[test]
    fn text_round_trip() {
        let rec = DeterministicSession::record(0x0123_4567_89ab_cdef);
        let _ = rec.plan(500).unwrap();
        let _ = rec.plan(12).unwrap();
        let record = rec.finish();
        let text = record.to_text();
        let parsed = ReplayRecord::from_text(&text).unwrap();
        assert_eq!(record, parsed);
    }

    #[test]
    fn bytes_round_trip() {
        let rec = DeterministicSession::record(42);
        let _ = rec.plan(9999).unwrap();
        let _ = rec.plan(1).unwrap();
        let _ = rec.plan(0).unwrap();
        let record = rec.finish();
        let bytes = record.to_bytes();
        let parsed = ReplayRecord::from_bytes(&bytes).unwrap();
        assert_eq!(record, parsed);
    }

    #[test]
    fn seed_changes_the_split() {
        let a = DeterministicSession::record(1).plan(100_000).unwrap();
        let b = DeterministicSession::record(2).plan(100_000).unwrap();
        // Overwhelmingly likely to differ; both are internally deterministic.
        assert_ne!(a.grain(), b.grain());
    }

    #[test]
    fn replay_detects_divergence() {
        let rec = DeterministicSession::record(7);
        let _ = rec.plan(100).unwrap();
        let record = rec.finish();
        let replay = DeterministicSession::replay(record);
        // Replaying a different length than recorded is a divergence.
        assert_eq!(
            replay.plan(101),
            Err(ReplayError::LengthMismatch {
                expected: 100,
                found: 101
            })
        );
    }

    #[test]
    fn replay_exhaustion_is_reported() {
        let rec = DeterministicSession::record(7);
        let _ = rec.plan(10).unwrap();
        let record = rec.finish();
        let replay = DeterministicSession::replay(record);
        assert_eq!(replay.plan(10).unwrap().len(), 10);
        assert_eq!(replay.plan(10), Err(ReplayError::Exhausted));
    }

    #[test]
    fn malformed_text_is_rejected() {
        assert_eq!(ReplayRecord::from_text(""), Err(ReplayError::Truncated));
        assert_eq!(
            ReplayRecord::from_text("not-a-header\n"),
            Err(ReplayError::BadHeader)
        );
        assert_eq!(
            ReplayRecord::from_text("prism-replay v1 seed=zz\n"),
            Err(ReplayError::BadHeader)
        );
        assert_eq!(
            ReplayRecord::from_text("prism-replay v1 seed=01\nbroken\n"),
            Err(ReplayError::BadEvent)
        );
    }

    #[test]
    fn malformed_bytes_are_rejected() {
        assert_eq!(ReplayRecord::from_bytes(&[]), Err(ReplayError::Truncated));
        let mut good = ReplayRecord {
            seed: 1,
            events: alloc::vec![SplitEvent { len: 4, grain: 2 }],
        }
        .to_bytes();
        good[0] = b'X';
        assert_eq!(ReplayRecord::from_bytes(&good), Err(ReplayError::BadHeader));
    }
}
