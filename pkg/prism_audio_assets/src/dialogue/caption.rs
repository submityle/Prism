//! Timecoded caption tracks driven by the playhead.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the caption-sync model of design section 35. Media carries a
//! frame-timecoded caption track; the playhead frame index drives which caption
//! event is active and, within it, which word is highlighted (per-sentence and
//! per-word accessibility highlighting). This is pure data surfaced off the
//! real-time thread for the telemetry ring and UI; it never touches the audio
//! callback.

#[cfg(not(feature = "std"))]
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A single highlightable word inside a caption line.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CaptionWord {
    /// Inclusive start frame of the word.
    pub start_frame: u64,
    /// Exclusive end frame of the word.
    pub end_frame: u64,
    /// The word text.
    pub text: String,
}

impl CaptionWord {
    /// Creates a caption word spanning `[start_frame, end_frame)`.
    #[must_use]
    pub fn new(start_frame: u64, end_frame: u64, text: &str) -> Self {
        Self {
            start_frame,
            end_frame,
            text: text.to_string(),
        }
    }

    /// Returns `true` when `frame` lies within this word.
    #[must_use]
    pub fn contains(&self, frame: u64) -> bool {
        frame >= self.start_frame && frame < self.end_frame
    }
}

/// A caption event: one line / sentence with optional per-word timings.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CaptionEvent {
    /// Inclusive start frame of the line.
    pub start_frame: u64,
    /// Exclusive end frame of the line.
    pub end_frame: u64,
    /// The full line text (sentence-level caption).
    pub text: String,
    /// Per-word timings for word-level highlighting, in order.
    pub words: Vec<CaptionWord>,
}

impl CaptionEvent {
    /// Creates a sentence-level caption with no word timings.
    #[must_use]
    pub fn new(start_frame: u64, end_frame: u64, text: &str) -> Self {
        Self {
            start_frame,
            end_frame,
            text: text.to_string(),
            words: Vec::new(),
        }
    }

    /// Adds a word timing and returns the event for chaining.
    #[must_use]
    pub fn with_word(mut self, word: CaptionWord) -> Self {
        self.words.push(word);
        self
    }

    /// Returns `true` when `frame` lies within this line.
    #[must_use]
    pub fn contains(&self, frame: u64) -> bool {
        frame >= self.start_frame && frame < self.end_frame
    }

    /// Returns the index of the highlighted word at `frame`, if any.
    #[must_use]
    pub fn highlighted_word(&self, frame: u64) -> Option<usize> {
        self.words.iter().position(|word| word.contains(frame))
    }
}

/// A frame-timecoded caption track for a single media asset.
///
/// Events are kept sorted by start frame; the playhead drives which event is
/// active. The track reports both newly entered and still-active events so a
/// consumer can drive UI without re-scanning.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CaptionTrack {
    /// Sample rate the frame timecodes are expressed in.
    pub sample_rate: u32,
    events: Vec<CaptionEvent>,
}

impl CaptionTrack {
    /// Creates an empty track at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            events: Vec::new(),
        }
    }

    /// Appends `event`, keeping events ordered by start frame.
    pub fn push(&mut self, event: CaptionEvent) {
        let position = self
            .events
            .iter()
            .position(|existing| existing.start_frame > event.start_frame)
            .unwrap_or(self.events.len());
        self.events.insert(position, event);
    }

    /// Returns the number of caption events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` when the track has no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns the caption event active at `frame`, if any.
    #[must_use]
    pub fn active_at(&self, frame: u64) -> Option<&CaptionEvent> {
        self.events.iter().find(|event| event.contains(frame))
    }

    /// Returns the index of the active event at `frame`, if any.
    #[must_use]
    pub fn active_index(&self, frame: u64) -> Option<usize> {
        self.events.iter().position(|event| event.contains(frame))
    }

    /// Converts a time in seconds to a frame index at this track's rate.
    #[must_use]
    pub fn frame_for_seconds(&self, seconds: f64) -> u64 {
        if seconds <= 0.0 {
            0
        } else {
            (seconds * f64::from(self.sample_rate)) as u64
        }
    }

    /// Returns the events whose span intersects `[from_frame, to_frame)`.
    ///
    /// This is the per-tick query a telemetry pump uses to surface captions
    /// entered since the previous playhead position.
    #[must_use]
    pub fn events_in_range(&self, from_frame: u64, to_frame: u64) -> Vec<&CaptionEvent> {
        self.events
            .iter()
            .filter(|event| event.start_frame < to_frame && event.end_frame > from_frame)
            .collect()
    }
}
