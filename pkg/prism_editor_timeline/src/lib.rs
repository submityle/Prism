//! Deterministic timeline primitives shared by animation, sequencing, and VFX.
//!
//! Time is represented as integer ticks. Values are fixed-width numeric vectors
//! so domain adapters can decide how to interpret the sampled channels. Sampling
//! returns owned values, which keeps the timeline independent from temporary
//! storage owned by a caller.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::ops::RangeInclusive;

mod editor;
mod playback;
mod sequence;
mod sequence_document;
#[cfg(feature = "sequence-runtime")]
pub mod sequence_runtime;
mod sequence_storage;
mod storage;
mod wire;
pub use editor::{TimelineEdit, TimelineEditor, TimelineLimits, TimelineSnapshotToken};
pub use playback::{
    Playback, PlaybackConfig, PlaybackError, PlaybackEvent, PlaybackFrame, PlaybackMode,
    PlaybackRate, PlaybackStatus,
};
pub use sequence::{
    SequenceClip, SequenceClipId, SequenceError, SequenceEvent, SequenceFrame, SequenceId,
    SequenceInstance, SequenceLimits, SequencePlayer, SequenceProvider, SequenceRange,
    SequenceRegistry, SequenceSample,
};
pub use sequence_document::{
    SequenceDocumentError, SequenceDocumentLimits, SequenceProjectCommand, SequenceProjectDocument,
    SequenceProjectEdit, SequenceProjectEditor, SequenceProjectToken,
};
#[cfg(feature = "sequence-runtime")]
pub use sequence_runtime::*;
pub use sequence_storage::{
    SequenceProjectCodec, SequenceProjectLoadCandidate, SequenceProjectSaveReceipt,
    SequenceProjectStorageRequest, SequenceProjectStorageResult, SequenceProjectStorageService,
};
pub use storage::{
    DocumentLoadCandidate, DocumentSaveReceipt, DocumentStorageCodec, DocumentStorageRequest,
    DocumentStorageResult, DocumentStorageService, TimelineCancelOutcome, TimelineFileRevision,
    TimelineLoadCandidate, TimelineSaveDurability, TimelineSaveReceipt, TimelineStorageCodec,
    TimelineStorageError, TimelineStorageLimits, TimelineStorageRequest, TimelineStorageResult,
    TimelineStorageService, MAX_TIMELINE_STORAGE_BYTES, MAX_TIMELINE_STORAGE_REQUESTS,
};
pub use wire::{TimelineCommand, TimelineDocument};

/// Stable identity for a timeline track.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct TrackId(u128);

impl TrackId {
    /// Constructs a track ID from its serialized representation.
    #[must_use]
    pub const fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    /// Returns the serialized representation of this track ID.
    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

/// Stable identity for a keyframe within a timeline.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct KeyId(u128);

impl KeyId {
    /// Constructs a keyframe ID from its serialized representation.
    #[must_use]
    pub const fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    /// Returns the serialized representation of this keyframe ID.
    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

/// Interpolation applied from one keyframe to the next.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Interpolation {
    /// Holds the preceding keyframe value until the next keyframe.
    Step,
    /// Linearly interpolates each component between adjacent keyframes.
    Linear,
    /// Cubic Hermite interpolation using the adjacent keys' explicit slopes.
    Cubic,
}

/// A keyframe containing a fixed-dimensional numeric value.
#[derive(Clone, Debug, PartialEq)]
pub struct Keyframe {
    id: KeyId,
    tick: u64,
    values: Vec<f64>,
    interpolation: Interpolation,
    incoming: Vec<f64>,
    outgoing: Vec<f64>,
}

impl Keyframe {
    /// Creates a keyframe. Track insertion validates the value dimension.
    #[must_use]
    pub fn new(id: KeyId, tick: u64, values: Vec<f64>, interpolation: Interpolation) -> Self {
        let dimension = values.len();
        Self {
            id,
            tick,
            values,
            interpolation,
            incoming: alloc::vec![0.0; dimension],
            outgoing: alloc::vec![0.0; dimension],
        }
    }

    /// Returns the stable keyframe ID.
    #[must_use]
    pub const fn id(&self) -> KeyId {
        self.id
    }

    /// Returns the integer tick at which this keyframe occurs.
    #[must_use]
    pub const fn tick(&self) -> u64 {
        self.tick
    }

    /// Returns the keyframe components in declaration order.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Returns the interpolation used after this keyframe.
    #[must_use]
    pub const fn interpolation(&self) -> Interpolation {
        self.interpolation
    }

    /// Returns incoming slopes in value units per integer tick.
    #[must_use]
    pub fn incoming(&self) -> &[f64] {
        &self.incoming
    }

    /// Returns outgoing slopes in value units per integer tick.
    #[must_use]
    pub fn outgoing(&self) -> &[f64] {
        &self.outgoing
    }

    /// Changes both tangent vectors after validating dimensions and finite values.
    pub fn set_tangents(
        &mut self,
        incoming: Vec<f64>,
        outgoing: Vec<f64>,
    ) -> Result<(), TimelineError> {
        for values in [&incoming, &outgoing] {
            if values.len() != self.values.len() {
                return Err(TimelineError::DimensionMismatch {
                    expected: self.values.len(),
                    actual: values.len(),
                });
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(TimelineError::NonFiniteValue);
            }
        }
        self.incoming = incoming;
        self.outgoing = outgoing;
        Ok(())
    }
}

/// Errors returned by timeline and track mutations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TimelineError {
    /// The track ID is already present in the timeline.
    DuplicateTrack(TrackId),
    /// The requested track ID is not present in the timeline.
    MissingTrack(TrackId),
    /// The keyframe ID is already present in the track.
    DuplicateKey(KeyId),
    /// The requested keyframe ID is not present in the track.
    MissingKey(KeyId),
    /// A track or keyframe was given an empty value vector.
    EmptyValue,
    /// A keyframe tick is already occupied by another keyframe.
    DuplicateTick(u64),
    /// A keyframe vector has a different dimension from its track.
    DimensionMismatch {
        /// The dimension required by the track.
        expected: usize,
        /// The dimension supplied by the keyframe.
        actual: usize,
    },
    /// An event ID is already present in the timeline.
    DuplicateEvent(u128),
    /// The requested event does not exist.
    MissingEvent(u128),
    /// Values or tangents contain NaN or infinity.
    NonFiniteValue,
    /// A valid curve's mathematical sample exceeds finite f64 range.
    SampleOverflow,
    /// An editor resource limit would be exceeded.
    LimitExceeded(&'static str),
    /// A required label is empty or an edit transaction is empty.
    EmptyInput,
    /// The monotonic editor revision is exhausted.
    RevisionExhausted,
    /// Command bytes are malformed, truncated, unsupported or contain trailing data.
    InvalidEncoding,
}

impl fmt::Display for TimelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateTrack(id) => write!(formatter, "track {} already exists", id.raw()),
            Self::MissingTrack(id) => write!(formatter, "track {} does not exist", id.raw()),
            Self::DuplicateKey(id) => write!(formatter, "key {} already exists", id.raw()),
            Self::MissingKey(id) => write!(formatter, "key {} does not exist", id.raw()),
            Self::EmptyValue => formatter.write_str("keyframe values cannot be empty"),
            Self::DuplicateTick(tick) => write!(formatter, "tick {tick} already has a keyframe"),
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    formatter,
                    "keyframe dimension {actual} does not match {expected}"
                )
            }
            Self::DuplicateEvent(id) => write!(formatter, "event {} already exists", id),
            Self::MissingEvent(id) => write!(formatter, "event {id} does not exist"),
            Self::NonFiniteValue => formatter.write_str("timeline values must be finite"),
            Self::SampleOverflow => formatter.write_str("sample exceeds finite numeric range"),
            Self::LimitExceeded(resource) => {
                write!(formatter, "timeline {resource} limit exceeded")
            }
            Self::EmptyInput => formatter.write_str("timeline input cannot be empty"),
            Self::RevisionExhausted => formatter.write_str("timeline revision exhausted"),
            Self::InvalidEncoding => formatter.write_str("invalid timeline encoding"),
        }
    }
}

impl core::error::Error for TimelineError {}

/// A deterministic track of keyframes with one fixed value dimension.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    id: TrackId,
    name: String,
    dimension: usize,
    keys: BTreeMap<u64, Keyframe>,
    keys_by_id: BTreeMap<KeyId, u64>,
}

impl Track {
    /// Creates an empty track with the requested value dimension.
    ///
    /// A zero-dimensional track is rejected because it cannot produce a useful
    /// sample.
    pub fn new(
        id: TrackId,
        name: impl Into<String>,
        dimension: usize,
    ) -> Result<Self, TimelineError> {
        if dimension == 0 {
            return Err(TimelineError::EmptyValue);
        }
        Ok(Self {
            id,
            name: name.into(),
            dimension,
            keys: BTreeMap::new(),
            keys_by_id: BTreeMap::new(),
        })
    }

    /// Returns the stable track ID.
    #[must_use]
    pub const fn id(&self) -> TrackId {
        self.id
    }

    /// Returns the display name of the track.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the number of components in every keyframe value.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns keyframes in ascending tick order.
    #[must_use = "iterate over the track keyframes"]
    pub fn keyframes(&self) -> impl Iterator<Item = &Keyframe> {
        self.keys.values()
    }

    /// Key lookup by stable identity, independent of its current tick.
    #[must_use]
    pub fn key(&self, id: KeyId) -> Option<&Keyframe> {
        self.keys_by_id
            .get(&id)
            .and_then(|tick| self.keys.get(tick))
    }

    /// Queries only keys in the visible tick range, without scanning earlier keys.
    /// A reversed range is empty.
    pub fn keys_in(
        &self,
        ticks: RangeInclusive<u64>,
    ) -> impl DoubleEndedIterator<Item = &Keyframe> {
        let (start, end) = ticks.into_inner();
        self.keys
            .range(start..=end.max(start))
            .filter(move |_| start <= end)
            .map(|(_, key)| key)
    }

    /// Number of keys, without traversing the track.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Returns a currently unused ID above the greatest live key identity.
    pub fn next_key_id(&self) -> Result<KeyId, TimelineError> {
        self.keys_by_id
            .last_key_value()
            .map_or(Ok(KeyId::from_raw(1)), |(id, _)| {
                id.raw()
                    .checked_add(1)
                    .map(KeyId::from_raw)
                    .ok_or(TimelineError::LimitExceeded("key identity"))
            })
    }

    /// Replaces a key by stable ID, including moving its tick. Failure preserves
    /// both indexes and the original key. Returns the previous key.
    pub fn replace(&mut self, key: Keyframe) -> Result<Keyframe, TimelineError> {
        self.validate_key(&key)?;
        let old_tick = *self
            .keys_by_id
            .get(&key.id)
            .ok_or(TimelineError::MissingKey(key.id))?;
        if key.tick != old_tick && self.keys.contains_key(&key.tick) {
            return Err(TimelineError::DuplicateTick(key.tick));
        }
        let previous = self.keys.remove(&old_tick).expect("key indexes agree");
        self.keys_by_id.insert(key.id, key.tick);
        self.keys.insert(key.tick, key);
        Ok(previous)
    }

    fn validate_key(&self, key: &Keyframe) -> Result<(), TimelineError> {
        if key.values.is_empty() {
            return Err(TimelineError::EmptyValue);
        }
        for values in [&key.values, &key.incoming, &key.outgoing] {
            if values.len() != self.dimension {
                return Err(TimelineError::DimensionMismatch {
                    expected: self.dimension,
                    actual: values.len(),
                });
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(TimelineError::NonFiniteValue);
            }
        }
        Ok(())
    }

    /// Inserts a keyframe while preserving unique IDs and unique ticks.
    pub fn insert(&mut self, key: Keyframe) -> Result<(), TimelineError> {
        self.validate_key(&key)?;
        if self.keys_by_id.contains_key(&key.id) {
            return Err(TimelineError::DuplicateKey(key.id));
        }
        if self.keys.contains_key(&key.tick) {
            return Err(TimelineError::DuplicateTick(key.tick));
        }
        self.keys_by_id.insert(key.id, key.tick);
        self.keys.insert(key.tick, key);
        Ok(())
    }

    /// Removes and returns a keyframe by stable ID.
    pub fn remove(&mut self, id: KeyId) -> Result<Keyframe, TimelineError> {
        let tick = self
            .keys_by_id
            .remove(&id)
            .ok_or(TimelineError::MissingKey(id))?;
        self.keys.remove(&tick).ok_or(TimelineError::MissingKey(id))
    }

    /// Samples the track at a tick, or returns `None` before its first keyframe.
    pub fn sample(&self, tick: u64) -> Result<Option<Vec<f64>>, TimelineError> {
        let Some((_, left)) = self.keys.range(..=tick).next_back() else {
            return Ok(None);
        };
        let Some((_, right)) = self.keys.range(tick..).next() else {
            return Ok(Some(left.values.clone()));
        };
        if left.tick == right.tick || matches!(left.interpolation, Interpolation::Step) {
            return Ok(Some(left.values.clone()));
        }
        let span = right.tick - left.tick;
        let amount = (tick - left.tick) as f64 / span as f64;
        let values: Vec<_> = left
            .values
            .iter()
            .zip(&right.values)
            .enumerate()
            .map(|(i, (a, b))| {
                if left.interpolation == Interpolation::Linear {
                    a * (1.0 - amount) + b * amount
                } else {
                    let t2 = amount * amount;
                    let t3 = t2 * amount;
                    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                    let h10 = t3 - 2.0 * t2 + amount;
                    let h01 = -2.0 * t3 + 3.0 * t2;
                    let h11 = t3 - t2;
                    a * h00
                        + b * h01
                        + left.outgoing[i] * (h10 * span as f64)
                        + right.incoming[i] * (h11 * span as f64)
                }
            })
            .collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(TimelineError::SampleOverflow);
        }
        Ok(Some(values))
    }
}

/// An event emitted at a specific timeline tick.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimelineEvent {
    /// Stable event identity, unique within a timeline.
    pub id: u128,
    /// Integer tick at which the event occurs.
    pub tick: u64,
    /// Caller-defined event name.
    pub name: String,
}

/// A deterministic collection of tracks and point events.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Timeline {
    tracks: BTreeMap<TrackId, Track>,
    events: BTreeMap<u64, Vec<TimelineEvent>>,
    event_ids: BTreeMap<u128, u64>,
}

impl Timeline {
    /// Adds a track, rejecting a duplicate track ID.
    pub fn add_track(&mut self, track: Track) -> Result<(), TimelineError> {
        if self.tracks.contains_key(&track.id) {
            return Err(TimelineError::DuplicateTrack(track.id));
        }
        self.tracks.insert(track.id, track);
        Ok(())
    }

    /// Removes and returns a track by stable ID.
    pub fn remove_track(&mut self, id: TrackId) -> Result<Track, TimelineError> {
        self.tracks
            .remove(&id)
            .ok_or(TimelineError::MissingTrack(id))
    }

    /// Returns a track by stable ID.
    #[must_use]
    pub fn track(&self, id: TrackId) -> Option<&Track> {
        self.tracks.get(&id)
    }

    /// Returns a mutable track by stable ID.
    #[must_use]
    pub fn track_mut(&mut self, id: TrackId) -> Option<&mut Track> {
        self.tracks.get_mut(&id)
    }

    /// Returns tracks in ascending stable-ID order.
    #[must_use = "iterate over the timeline tracks"]
    pub fn tracks(&self) -> impl Iterator<Item = &Track> {
        self.tracks.values()
    }

    /// Adds an event and orders same-tick events by stable event ID.
    pub fn add_event(&mut self, event: TimelineEvent) -> Result<(), TimelineError> {
        if self.event_ids.contains_key(&event.id) {
            return Err(TimelineError::DuplicateEvent(event.id));
        }
        let tick = event.tick;
        self.event_ids.insert(event.id, tick);
        self.events.entry(tick).or_default().push(event);
        self.events
            .get_mut(&tick)
            .expect("event bucket exists")
            .sort_by_key(|entry| entry.id);
        Ok(())
    }

    /// Returns events at a tick in deterministic stable-ID order.
    #[must_use]
    pub fn events_at(&self, tick: u64) -> &[TimelineEvent] {
        self.events.get(&tick).map_or(&[], Vec::as_slice)
    }

    /// Removes an event by its stable ID and releases an empty tick bucket.
    pub fn remove_event(&mut self, id: u128) -> Result<TimelineEvent, TimelineError> {
        let tick = self
            .event_ids
            .remove(&id)
            .ok_or(TimelineError::MissingEvent(id))?;
        let bucket = self.events.get_mut(&tick).expect("event index agrees");
        let index = bucket
            .binary_search_by_key(&id, |event| event.id)
            .expect("event index agrees");
        let event = bucket.remove(index);
        if bucket.is_empty() {
            self.events.remove(&tick);
        }
        Ok(event)
    }

    /// Queries a caller-selected time range in tick/ID order; reversed is empty.
    pub fn events_in(&self, ticks: RangeInclusive<u64>) -> impl Iterator<Item = &TimelineEvent> {
        let (start, end) = ticks.into_inner();
        self.events
            .range(start..=end.max(start))
            .filter(move |_| start <= end)
            .flat_map(|(_, events)| events.iter())
    }

    /// Returns tracks starting at an ID without scanning preceding tracks.
    pub fn tracks_from(&self, start: TrackId) -> impl Iterator<Item = &Track> {
        self.tracks.range(start..).map(|(_, track)| track)
    }

    /// Tracks preceding an ID, permitting reverse window navigation.
    pub fn tracks_before(&self, end: TrackId) -> impl DoubleEndedIterator<Item = &Track> {
        self.tracks.range(..end).map(|(_, track)| track)
    }

    /// Number of tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Returns a currently unused ID above the greatest live track identity.
    pub fn next_track_id(&self) -> Result<TrackId, TimelineError> {
        self.tracks
            .last_key_value()
            .map_or(Ok(TrackId::from_raw(1)), |(id, _)| {
                id.raw()
                    .checked_add(1)
                    .map(TrackId::from_raw)
                    .ok_or(TimelineError::LimitExceeded("track identity"))
            })
    }

    /// Returns a currently unused ID above the greatest live event identity.
    pub fn next_event_id(&self) -> Result<u128, TimelineError> {
        self.event_ids.last_key_value().map_or(Ok(1), |(id, _)| {
            id.checked_add(1)
                .ok_or(TimelineError::LimitExceeded("event identity"))
        })
    }

    /// Number of point events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.event_ids.len()
    }

    /// Samples every track and returns results keyed by ascending track ID.
    pub fn sample(&self, tick: u64) -> Result<BTreeMap<TrackId, Vec<f64>>, TimelineError> {
        let mut samples = BTreeMap::new();
        for (id, track) in &self.tracks {
            if let Some(values) = track.sample(tick)? {
                samples.insert(*id, values);
            }
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_sampling_is_deterministic_and_dimension_checked() {
        let mut track = Track::new(TrackId::from_raw(1), "opacity", 1).expect("track");
        track
            .insert(Keyframe::new(
                KeyId::from_raw(1),
                0,
                vec![0.0],
                Interpolation::Linear,
            ))
            .expect("key");
        track
            .insert(Keyframe::new(
                KeyId::from_raw(2),
                10,
                vec![1.0],
                Interpolation::Linear,
            ))
            .expect("key");
        assert_eq!(track.sample(5), Ok(Some(vec![0.5])));
        assert_eq!(
            track.insert(Keyframe::new(
                KeyId::from_raw(3),
                20,
                vec![1.0, 2.0],
                Interpolation::Step
            )),
            Err(TimelineError::DimensionMismatch {
                expected: 1,
                actual: 2
            })
        );
    }

    #[test]
    fn events_are_sorted_by_stable_id() {
        let mut timeline = Timeline::default();
        timeline
            .add_event(TimelineEvent {
                id: 2,
                tick: 3,
                name: "b".into(),
            })
            .expect("event");
        timeline
            .add_event(TimelineEvent {
                id: 1,
                tick: 3,
                name: "a".into(),
            })
            .expect("event");
        assert_eq!(timeline.events_at(3)[0].id, 1);
        assert_eq!(timeline.events_at(3)[1].id, 2);
    }

    #[test]
    fn mutation_boundaries_preserve_indexes_and_sampling_edges() {
        assert_eq!(
            Track::new(TrackId::from_raw(1), "invalid", 0),
            Err(TimelineError::EmptyValue)
        );

        let mut track = Track::new(TrackId::from_raw(1), "value", 1).expect("track");
        let first = Keyframe::new(KeyId::from_raw(1), 10, vec![2.0], Interpolation::Step);
        track.insert(first.clone()).expect("first key");
        assert_eq!(track.sample(9), Ok(None));
        assert_eq!(track.sample(10), Ok(Some(vec![2.0])));

        assert_eq!(
            track.insert(Keyframe::new(
                KeyId::from_raw(2),
                10,
                vec![3.0],
                Interpolation::Linear,
            )),
            Err(TimelineError::DuplicateTick(10))
        );
        assert_eq!(
            track.insert(Keyframe::new(
                KeyId::from_raw(1),
                11,
                vec![3.0],
                Interpolation::Linear,
            )),
            Err(TimelineError::DuplicateKey(KeyId::from_raw(1)))
        );
        assert_eq!(
            track.insert(Keyframe::new(
                KeyId::from_raw(3),
                11,
                Vec::new(),
                Interpolation::Linear,
            )),
            Err(TimelineError::EmptyValue)
        );
        assert_eq!(track.remove(KeyId::from_raw(1)), Ok(first));
        assert_eq!(
            track.remove(KeyId::from_raw(1)),
            Err(TimelineError::MissingKey(KeyId::from_raw(1)))
        );
        assert_eq!(track.sample(10), Ok(None));

        let mut timeline = Timeline::default();
        let track = Track::new(TrackId::from_raw(4), "track", 1).expect("track");
        timeline.add_track(track.clone()).expect("track");
        assert_eq!(
            timeline.add_track(track),
            Err(TimelineError::DuplicateTrack(TrackId::from_raw(4)))
        );
        assert_eq!(
            timeline.remove_track(TrackId::from_raw(9)),
            Err(TimelineError::MissingTrack(TrackId::from_raw(9)))
        );
        timeline
            .add_event(TimelineEvent {
                id: 7,
                tick: 2,
                name: "event".into(),
            })
            .expect("event");
        assert_eq!(
            timeline.add_event(TimelineEvent {
                id: 7,
                tick: 3,
                name: "duplicate".into(),
            }),
            Err(TimelineError::DuplicateEvent(7))
        );
    }
}
