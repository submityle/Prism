//! Deterministic timeline primitives shared by animation, sequencing, and VFX.
//!
//! Time is represented as integer ticks. Values are fixed-width numeric vectors
//! so sampling is allocation-free and domain adapters decide how to interpret
//! the sampled channels.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct TrackId(u128);

impl TrackId {
    #[must_use]
    pub const fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct KeyId(u128);

impl KeyId {
    #[must_use]
    pub const fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Interpolation {
    Step,
    Linear,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Keyframe {
    id: KeyId,
    tick: u64,
    values: Vec<f64>,
    interpolation: Interpolation,
}

impl Keyframe {
    #[must_use]
    pub fn new(id: KeyId, tick: u64, values: Vec<f64>, interpolation: Interpolation) -> Self {
        Self {
            id,
            tick,
            values,
            interpolation,
        }
    }

    #[must_use]
    pub const fn id(&self) -> KeyId {
        self.id
    }

    #[must_use]
    pub const fn tick(&self) -> u64 {
        self.tick
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[must_use]
    pub const fn interpolation(&self) -> Interpolation {
        self.interpolation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TimelineError {
    DuplicateTrack(TrackId),
    MissingTrack(TrackId),
    DuplicateKey(KeyId),
    MissingKey(KeyId),
    EmptyValue,
    DimensionMismatch { expected: usize, actual: usize },
    DuplicateEvent(u128),
}

impl std::fmt::Display for TimelineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateTrack(id) => write!(formatter, "track {} already exists", id.raw()),
            Self::MissingTrack(id) => write!(formatter, "track {} does not exist", id.raw()),
            Self::DuplicateKey(id) => write!(formatter, "key {} already exists", id.raw()),
            Self::MissingKey(id) => write!(formatter, "key {} does not exist", id.raw()),
            Self::EmptyValue => formatter.write_str("keyframe values cannot be empty"),
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    formatter,
                    "keyframe dimension {actual} does not match {expected}"
                )
            }
            Self::DuplicateEvent(id) => write!(formatter, "event {} already exists", id),
        }
    }
}

impl std::error::Error for TimelineError {}

#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    id: TrackId,
    name: String,
    dimension: usize,
    keys: BTreeMap<u64, Keyframe>,
    keys_by_id: BTreeMap<KeyId, u64>,
}

impl Track {
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

    #[must_use]
    pub const fn id(&self) -> TrackId {
        self.id
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    #[must_use]
    pub fn keyframes(&self) -> impl Iterator<Item = &Keyframe> {
        self.keys.values()
    }

    pub fn insert(&mut self, key: Keyframe) -> Result<(), TimelineError> {
        if key.values.is_empty() {
            return Err(TimelineError::EmptyValue);
        }
        if key.values.len() != self.dimension {
            return Err(TimelineError::DimensionMismatch {
                expected: self.dimension,
                actual: key.values.len(),
            });
        }
        if self.keys_by_id.contains_key(&key.id) {
            return Err(TimelineError::DuplicateKey(key.id));
        }
        if self.keys.contains_key(&key.tick) {
            return Err(TimelineError::DuplicateKey(key.id));
        }
        self.keys_by_id.insert(key.id, key.tick);
        self.keys.insert(key.tick, key);
        Ok(())
    }

    pub fn remove(&mut self, id: KeyId) -> Result<Keyframe, TimelineError> {
        let tick = self
            .keys_by_id
            .remove(&id)
            .ok_or(TimelineError::MissingKey(id))?;
        self.keys.remove(&tick).ok_or(TimelineError::MissingKey(id))
    }

    #[must_use]
    pub fn sample(&self, tick: u64) -> Option<Vec<f64>> {
        let (_, left) = self.keys.range(..=tick).next_back()?;
        let Some((_, right)) = self.keys.range(tick..).next() else {
            return Some(left.values.clone());
        };
        if left.tick == right.tick || matches!(left.interpolation, Interpolation::Step) {
            return Some(left.values.clone());
        }
        let span = right.tick - left.tick;
        let amount = (tick - left.tick) as f64 / span as f64;
        Some(
            left.values
                .iter()
                .zip(&right.values)
                .map(|(a, b)| a + (b - a) * amount)
                .collect(),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimelineEvent {
    pub id: u128,
    pub tick: u64,
    pub name: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Timeline {
    tracks: BTreeMap<TrackId, Track>,
    events: BTreeMap<u64, Vec<TimelineEvent>>,
    event_ids: BTreeSet<u128>,
}

impl Timeline {
    pub fn add_track(&mut self, track: Track) -> Result<(), TimelineError> {
        if self.tracks.contains_key(&track.id) {
            return Err(TimelineError::DuplicateTrack(track.id));
        }
        self.tracks.insert(track.id, track);
        Ok(())
    }

    pub fn remove_track(&mut self, id: TrackId) -> Result<Track, TimelineError> {
        self.tracks
            .remove(&id)
            .ok_or(TimelineError::MissingTrack(id))
    }

    #[must_use]
    pub fn track(&self, id: TrackId) -> Option<&Track> {
        self.tracks.get(&id)
    }

    #[must_use]
    pub fn track_mut(&mut self, id: TrackId) -> Option<&mut Track> {
        self.tracks.get_mut(&id)
    }

    #[must_use]
    pub fn tracks(&self) -> impl Iterator<Item = &Track> {
        self.tracks.values()
    }

    pub fn add_event(&mut self, event: TimelineEvent) -> Result<(), TimelineError> {
        if !self.event_ids.insert(event.id) {
            return Err(TimelineError::DuplicateEvent(event.id));
        }
        let tick = event.tick;
        self.events.entry(tick).or_default().push(event);
        self.events
            .get_mut(&tick)
            .expect("event bucket exists")
            .sort_by_key(|entry| entry.id);
        Ok(())
    }

    #[must_use]
    pub fn events_at(&self, tick: u64) -> &[TimelineEvent] {
        self.events.get(&tick).map_or(&[], Vec::as_slice)
    }

    #[must_use]
    pub fn sample(&self, tick: u64) -> BTreeMap<TrackId, Vec<f64>> {
        self.tracks
            .iter()
            .filter_map(|(id, track)| track.sample(tick).map(|value| (*id, value)))
            .collect()
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
        assert_eq!(track.sample(5), Some(vec![0.5]));
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
}
