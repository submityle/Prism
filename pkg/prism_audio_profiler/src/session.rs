//! Bounded recording and deterministic playback of telemetry and events.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the recordable-session part of design section 26. A
//! [`ProfilerSession`] is a fixed-capacity ring of [`AudioTelemetry`] frames
//! paired with an [`EventTimeline`]; an observer thread pushes frames and
//! events as they arrive and a UI replays them without touching the RT thread.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

use crate::frame::AudioTelemetry;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Category of a recorded [`TimelineEvent`].
///
/// The profiler records engine-level occurrences alongside the per-block
/// telemetry so a replay can correlate a spike or dropout with the event that
/// caused it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum EventKind {
    /// A new voice began playing.
    VoiceStarted,
    /// A voice stopped (finished, stolen, or explicitly killed).
    VoiceStopped,
    /// A voice transitioned from physical to virtual (culled but retained).
    VoiceVirtualized,
    /// A voice transitioned from virtual back to physical.
    VoiceDevirtualized,
    /// A buffer underrun / dropout was detected.
    Dropout,
    /// The compiled graph was swapped for a new one.
    GraphSwapped,
    /// A parameter or RTPC value changed.
    ParameterChanged,
    /// A user-defined marker, carried by the event label.
    Marker,
}

/// A single time-stamped event on the [`EventTimeline`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TimelineEvent {
    /// Block index this event is associated with (shares the telemetry clock).
    pub block_index: u64,
    /// Playhead position in frames when the event occurred.
    pub playhead: u64,
    /// Category of the event.
    pub kind: EventKind,
    /// Human-readable label (voice name, parameter path, marker text).
    pub label: String,
}

impl TimelineEvent {
    /// Create a new timeline event.
    #[must_use]
    pub fn new(block_index: u64, playhead: u64, kind: EventKind, label: impl Into<String>) -> Self {
        Self {
            block_index,
            playhead,
            kind,
            label: label.into(),
        }
    }
}

/// A bounded, ordered log of [`TimelineEvent`] values.
///
/// Events are kept in insertion order; once the capacity is reached the oldest
/// event is dropped. A capacity of `0` disables recording entirely.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct EventTimeline {
    capacity: usize,
    events: VecDeque<TimelineEvent>,
}

impl EventTimeline {
    /// Create a timeline that retains at most `capacity` events.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            events: VecDeque::new(),
        }
    }

    /// Capacity (maximum retained events).
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of currently retained events.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// `true` when no events are retained.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Record an event, evicting the oldest if at capacity. A zero-capacity
    /// timeline silently discards the event.
    pub fn record(&mut self, event: TimelineEvent) {
        if self.capacity == 0 {
            return;
        }
        while self.events.len() >= self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    /// Iterate retained events oldest-first.
    pub fn iter(&self) -> impl Iterator<Item = &TimelineEvent> {
        self.events.iter()
    }

    /// Collect every event whose `block_index` lies within `[start, end]`
    /// (inclusive) into a fresh vector, oldest-first.
    #[must_use]
    pub fn in_block_range(&self, start: u64, end: u64) -> Vec<TimelineEvent> {
        self.events
            .iter()
            .filter(|event| event.block_index >= start && event.block_index <= end)
            .cloned()
            .collect()
    }

    /// Remove every retained event.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

/// A fixed-capacity recording of telemetry frames plus an event timeline,
/// with a cursor for deterministic playback.
///
/// Pushing more than `capacity` frames evicts the oldest, matching the
/// behaviour of the bounded RT telemetry ring. Playback walks the retained
/// frames in order via [`ProfilerSession::next_playback`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ProfilerSession {
    capacity: usize,
    frames: VecDeque<AudioTelemetry>,
    timeline: EventTimeline,
    playback_cursor: usize,
}

impl ProfilerSession {
    /// Create a session retaining at most `frame_capacity` telemetry frames and
    /// `event_capacity` timeline events.
    #[must_use]
    pub fn new(frame_capacity: usize, event_capacity: usize) -> Self {
        Self {
            capacity: frame_capacity,
            frames: VecDeque::new(),
            timeline: EventTimeline::new(event_capacity),
            playback_cursor: 0,
        }
    }

    /// Frame capacity (maximum retained telemetry frames).
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of currently retained telemetry frames.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// `true` when no telemetry frames are retained.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Shared access to the event timeline.
    #[must_use]
    #[inline]
    pub fn timeline(&self) -> &EventTimeline {
        &self.timeline
    }

    /// Mutable access to the event timeline for recording events.
    #[inline]
    pub fn timeline_mut(&mut self) -> &mut EventTimeline {
        &mut self.timeline
    }

    /// Record one telemetry frame, evicting the oldest when at capacity. A
    /// zero-capacity session silently discards the frame. Eviction shifts the
    /// playback cursor so it keeps pointing at the same logical frame when
    /// possible.
    pub fn record_frame(&mut self, frame: AudioTelemetry) {
        if self.capacity == 0 {
            return;
        }
        while self.frames.len() >= self.capacity {
            self.frames.pop_front();
            self.playback_cursor = self.playback_cursor.saturating_sub(1);
        }
        self.frames.push_back(frame);
    }

    /// Record a timeline event (convenience forwarder).
    pub fn record_event(&mut self, event: TimelineEvent) {
        self.timeline.record(event);
    }

    /// Borrow the retained frame at `index` (oldest-first), if present.
    #[must_use]
    pub fn frame(&self, index: usize) -> Option<&AudioTelemetry> {
        self.frames.get(index)
    }

    /// Iterate retained telemetry frames oldest-first.
    pub fn iter(&self) -> impl Iterator<Item = &AudioTelemetry> {
        self.frames.iter()
    }

    /// Current playback cursor position (index of the next frame to be returned
    /// by [`ProfilerSession::next_playback`]).
    #[must_use]
    #[inline]
    pub fn playback_cursor(&self) -> usize {
        self.playback_cursor
    }

    /// Reset the playback cursor to the oldest retained frame.
    #[inline]
    pub fn rewind(&mut self) {
        self.playback_cursor = 0;
    }

    /// Seek the playback cursor to `index`, clamped to the retained range.
    #[inline]
    pub fn seek(&mut self, index: usize) {
        self.playback_cursor = index.min(self.frames.len());
    }

    /// Advance playback by one frame, returning it, or `None` at the end.
    pub fn next_playback(&mut self) -> Option<AudioTelemetry> {
        let frame = self.frames.get(self.playback_cursor).copied();
        if frame.is_some() {
            self.playback_cursor += 1;
        }
        frame
    }

    /// Clear all recorded frames, events, and reset the cursor.
    pub fn clear(&mut self) {
        self.frames.clear();
        self.timeline.clear();
        self.playback_cursor = 0;
    }

    /// Mean CPU load across all retained frames, or `0.0` when empty.
    #[must_use]
    pub fn mean_cpu_load(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.frames.iter().map(|frame| frame.cpu_load).sum();
        sum / self.frames.len() as f32
    }

    /// Peak CPU load across all retained frames, or `0.0` when empty.
    #[must_use]
    pub fn peak_cpu_load(&self) -> f32 {
        self.frames
            .iter()
            .map(|frame| frame.cpu_load)
            .fold(0.0_f32, f32::max)
    }

    /// Count of retained frames whose CPU load indicates an overrun.
    #[must_use]
    pub fn overrun_count(&self) -> usize {
        self.frames.iter().filter(|frame| frame.is_overrun()).count()
    }

    /// Peak physical voice count observed across all retained frames.
    #[must_use]
    pub fn peak_physical_voices(&self) -> u32 {
        self.frames
            .iter()
            .map(|frame| frame.physical_voices)
            .max()
            .unwrap_or(0)
    }
}
