//! Reversible timeline command transactions with bounded inverse history.

use crate::{KeyId, Keyframe, Timeline, TimelineError, TimelineEvent, Track, TrackId};
use alloc::{collections::VecDeque, string::String, sync::Arc, vec::Vec};

/// Resource policy for one authoring document and its undo/redo history.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimelineLimits {
    /// Maximum tracks.
    pub max_tracks: usize,
    /// Maximum keys across all tracks.
    pub max_keys: usize,
    /// Maximum numeric channels per track.
    pub max_dimension: usize,
    /// Maximum point events.
    pub max_events: usize,
    /// Maximum UTF-8 bytes in a track, event or command label.
    pub max_text_bytes: usize,
    /// Maximum edits in one atomic command.
    pub max_edits: usize,
    /// Maximum retained commands across undo and redo.
    pub max_history_entries: usize,
    /// Maximum logical payload bytes reserved for history in either direction.
    pub max_history_bytes: usize,
}

impl Default for TimelineLimits {
    fn default() -> Self {
        Self {
            max_tracks: 4096,
            max_keys: 1_000_000,
            max_dimension: 64,
            max_events: 65_536,
            max_text_bytes: 4096,
            max_edits: 4096,
            max_history_entries: 256,
            max_history_bytes: 16 * 1024 * 1024,
        }
    }
}

/// Serializable command data. IDs and numeric channels carry no gameplay policy.
#[derive(Clone, Debug, PartialEq)]
pub enum TimelineEdit {
    /// Insert an entire track, rejecting existing identity.
    AddTrack(Track),
    /// Delete a track and retain its payload in the inverse.
    RemoveTrack(TrackId),
    /// Insert or replace a key by stable ID. Occupied other ticks are rejected.
    PutKey {
        /// Owning track.
        track: TrackId,
        /// Full key value, tick, interpolation and tangents.
        key: Keyframe,
    },
    /// Remove a key by stable ID.
    RemoveKey {
        /// Owning track.
        track: TrackId,
        /// Key identity.
        key: KeyId,
    },
    /// Insert a point event.
    AddEvent(TimelineEvent),
    /// Remove a point event.
    RemoveEvent(u128),
}

/// Exact snapshot identity. Numeric revisions alone cannot authorize a stale UI.
#[derive(Clone, Debug)]
pub struct TimelineSnapshotToken(Arc<()>);
impl PartialEq for TimelineSnapshotToken {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for TimelineSnapshotToken {}

struct Record {
    label: String,
    edits: Vec<TimelineEdit>,
    reserved_bytes: usize,
}

/// Owns the authoritative timeline. All mutations are atomic command data;
/// snapshots are borrowed and cannot bypass revision or undo bookkeeping.
pub struct TimelineEditor {
    timeline: Timeline,
    limits: TimelineLimits,
    keys: usize,
    revision: u64,
    token: TimelineSnapshotToken,
    undo: VecDeque<Record>,
    redo: Vec<Record>,
    history_bytes: usize,
}

impl TimelineEditor {
    /// Validates and adopts a timeline without manufacturing an undo entry.
    pub fn new(timeline: Timeline, limits: TimelineLimits) -> Result<Self, TimelineError> {
        if limits.max_edits == 0 || limits.max_history_entries == 0 || limits.max_history_bytes == 0
        {
            return Err(TimelineError::LimitExceeded("history configuration"));
        }
        let mut keys = 0usize;
        check(timeline.track_count(), limits.max_tracks, "tracks")?;
        check(timeline.event_count(), limits.max_events, "events")?;
        for track in timeline.tracks() {
            validate_track(track, limits)?;
            keys = keys
                .checked_add(track.key_count())
                .ok_or(TimelineError::LimitExceeded("keys"))?;
            check(keys, limits.max_keys, "keys")?;
        }
        for event in timeline.events_in(0..=u64::MAX) {
            text(&event.name, limits)?;
        }
        Ok(Self {
            timeline,
            limits,
            keys,
            revision: 0,
            token: TimelineSnapshotToken(Arc::new(())),
            undo: VecDeque::new(),
            redo: Vec::new(),
            history_bytes: 0,
        })
    }

    /// Authoritative document, read only.
    #[must_use]
    pub fn timeline(&self) -> &Timeline {
        &self.timeline
    }
    /// Limits for this document.
    #[must_use]
    pub const fn limits(&self) -> TimelineLimits {
        self.limits
    }
    /// Monotonic revision including undo and redo.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    /// Exact document snapshot identity for asynchronous views and adapters.
    #[must_use]
    pub fn snapshot_token(&self) -> TimelineSnapshotToken {
        self.token.clone()
    }
    /// Number of undoable commands.
    #[must_use]
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }
    /// Number of redoable commands.
    #[must_use]
    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }
    /// Logical reserved history bytes, including both directions of each command.
    #[must_use]
    pub const fn history_bytes(&self) -> usize {
        self.history_bytes
    }
    /// Label of the next undo command.
    #[must_use]
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.back().map(|entry| entry.label.as_str())
    }

    /// Atomically executes a transaction and records only its inverse changes.
    /// Failure restores data, indexes, history, revision and snapshot identity.
    /// Oldest undo entries are evicted only after a valid command fits the budget.
    pub fn execute(
        &mut self,
        label: impl Into<String>,
        edits: Vec<TimelineEdit>,
    ) -> Result<(), TimelineError> {
        let label = label.into();
        text(&label, self.limits)?;
        if edits.is_empty() {
            return Err(TimelineError::EmptyInput);
        }
        check(edits.len(), self.limits.max_edits, "transaction edits")?;
        if edits.iter().all(|edit| match edit {
            TimelineEdit::PutKey { track, key } => {
                self.timeline
                    .track(*track)
                    .and_then(|track| track.key(key.id()))
                    == Some(key)
            }
            _ => false,
        }) {
            return Ok(());
        }
        self.next_revision()?;
        let forward_bytes = edits_bytes(&edits).saturating_add(label.len());
        check(
            forward_bytes,
            self.limits.max_history_bytes,
            "history bytes",
        )?;
        let inverse = self.apply_atomic(edits)?;
        let reserved_bytes = forward_bytes.max(edits_bytes(&inverse).saturating_add(label.len()));
        if reserved_bytes > self.limits.max_history_bytes {
            self.apply_atomic(inverse)
                .expect("private inverse restores a successful transaction");
            return Err(TimelineError::LimitExceeded("history bytes"));
        }
        for record in self.redo.drain(..) {
            self.history_bytes -= record.reserved_bytes;
        }
        while self.undo.len() >= self.limits.max_history_entries
            || self.history_bytes > self.limits.max_history_bytes - reserved_bytes
        {
            let oldest = self
                .undo
                .pop_front()
                .expect("retained history accounts for reserved bytes");
            self.history_bytes -= oldest.reserved_bytes;
        }
        self.history_bytes += reserved_bytes;
        self.undo.push_back(Record {
            label,
            edits: inverse,
            reserved_bytes,
        });
        self.changed();
        Ok(())
    }

    /// Reverses a command, preserving its redo data and reservation.
    pub fn undo(&mut self) -> Result<bool, TimelineError> {
        if self.undo.is_empty() {
            return Ok(false);
        }
        self.next_revision()?;
        let record = self.undo.pop_back().expect("nonempty undo");
        let edits = self
            .apply_atomic(record.edits)
            .expect("private inverse remains valid");
        self.redo.push(Record { edits, ..record });
        self.changed();
        Ok(true)
    }

    /// Replays a previously undone command.
    pub fn redo(&mut self) -> Result<bool, TimelineError> {
        if self.redo.is_empty() {
            return Ok(false);
        }
        self.next_revision()?;
        let record = self.redo.pop().expect("nonempty redo");
        let edits = self
            .apply_atomic(record.edits)
            .expect("private redo remains valid");
        self.undo.push_back(Record { edits, ..record });
        self.changed();
        Ok(true)
    }

    fn next_revision(&self) -> Result<u64, TimelineError> {
        self.revision
            .checked_add(1)
            .ok_or(TimelineError::RevisionExhausted)
    }
    fn changed(&mut self) {
        self.revision += 1;
        self.token = TimelineSnapshotToken(Arc::new(()));
    }

    fn apply_atomic(
        &mut self,
        edits: Vec<TimelineEdit>,
    ) -> Result<Vec<TimelineEdit>, TimelineError> {
        let mut inverse = Vec::with_capacity(edits.len());
        for edit in edits {
            match self.apply_one(edit) {
                Ok(edit) => inverse.push(edit),
                Err(error) => {
                    for edit in inverse.into_iter().rev() {
                        self.apply_one(edit)
                            .expect("private rollback remains valid");
                    }
                    return Err(error);
                }
            }
        }
        inverse.reverse();
        Ok(inverse)
    }

    fn apply_one(&mut self, edit: TimelineEdit) -> Result<TimelineEdit, TimelineError> {
        Ok(match edit {
            TimelineEdit::AddTrack(track) => {
                validate_track(&track, self.limits)?;
                check(
                    self.timeline.track_count().saturating_add(1),
                    self.limits.max_tracks,
                    "tracks",
                )?;
                let keys = self
                    .keys
                    .checked_add(track.key_count())
                    .ok_or(TimelineError::LimitExceeded("keys"))?;
                check(keys, self.limits.max_keys, "keys")?;
                let id = track.id();
                self.timeline.add_track(track)?;
                self.keys = keys;
                TimelineEdit::RemoveTrack(id)
            }
            TimelineEdit::RemoveTrack(id) => {
                let track = self.timeline.remove_track(id)?;
                self.keys -= track.key_count();
                TimelineEdit::AddTrack(track)
            }
            TimelineEdit::PutKey { track, key } => {
                let target = self
                    .timeline
                    .track_mut(track)
                    .ok_or(TimelineError::MissingTrack(track))?;
                if target.key(key.id()).is_some() {
                    TimelineEdit::PutKey {
                        track,
                        key: target.replace(key)?,
                    }
                } else {
                    check(self.keys.saturating_add(1), self.limits.max_keys, "keys")?;
                    let id = key.id();
                    target.insert(key)?;
                    self.keys += 1;
                    TimelineEdit::RemoveKey { track, key: id }
                }
            }
            TimelineEdit::RemoveKey { track, key } => {
                let target = self
                    .timeline
                    .track_mut(track)
                    .ok_or(TimelineError::MissingTrack(track))?;
                let key = target.remove(key)?;
                self.keys -= 1;
                TimelineEdit::PutKey { track, key }
            }
            TimelineEdit::AddEvent(event) => {
                text(&event.name, self.limits)?;
                check(
                    self.timeline.event_count().saturating_add(1),
                    self.limits.max_events,
                    "events",
                )?;
                let id = event.id;
                self.timeline.add_event(event)?;
                TimelineEdit::RemoveEvent(id)
            }
            TimelineEdit::RemoveEvent(id) => {
                TimelineEdit::AddEvent(self.timeline.remove_event(id)?)
            }
        })
    }
}

fn check(actual: usize, max: usize, name: &'static str) -> Result<(), TimelineError> {
    if actual > max {
        Err(TimelineError::LimitExceeded(name))
    } else {
        Ok(())
    }
}
fn text(value: &str, limits: TimelineLimits) -> Result<(), TimelineError> {
    if value.trim().is_empty() {
        return Err(TimelineError::EmptyInput);
    }
    check(value.len(), limits.max_text_bytes, "text bytes")
}
fn validate_track(track: &Track, limits: TimelineLimits) -> Result<(), TimelineError> {
    text(track.name(), limits)?;
    check(track.dimension(), limits.max_dimension, "channels")?;
    check(track.key_count(), limits.max_keys, "keys")?;
    for key in track.keyframes() {
        track.validate_key(key)?;
    }
    Ok(())
}
fn edits_bytes(edits: &[TimelineEdit]) -> usize {
    edits.iter().fold(0usize, |bytes, edit| {
        let payload = match edit {
            TimelineEdit::AddTrack(track) => {
                track.keyframes().fold(track.name().len(), |size, key| {
                    size.saturating_add(key_bytes(key))
                })
            }
            TimelineEdit::PutKey { key, .. } => key_bytes(key),
            TimelineEdit::AddEvent(event) => event.name.len(),
            _ => 0,
        };
        bytes
            .saturating_add(size_of::<TimelineEdit>())
            .saturating_add(payload)
    })
}
fn key_bytes(key: &Keyframe) -> usize {
    size_of::<Keyframe>().saturating_add(key.values.len().saturating_mul(24))
}
