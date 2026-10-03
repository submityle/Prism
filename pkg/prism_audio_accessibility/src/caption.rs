//! Caption and audio-description metadata carried on an audio event.
//!
//! A [`CaptionMetadata`] record is attached by the authoring/runtime layer to
//! an event. When the event fires, a [`CaptionReport`] is assembled and handed
//! to the telemetry ring, which forwards it to the UI layer that renders the
//! on-screen caption. This module only produces the plain data structures and a
//! builder for the report; it performs no input/output of its own.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the caption half of design section 23. The report is the payload
//! placed on the telemetry ring of design section 26; this crate deliberately
//! stops at the data model so the ring stays the single transport.

use alloc::string::String;

/// Metadata describing a single caption or audio description line.
///
/// The record is self-contained: it carries the display `text`, an optional
/// `speaker` label, the intended on-screen `duration_ms`, a `BCP-47` language
/// `tag`, and a flag marking lines that describe non-speech audio (an audio
/// description) rather than spoken dialogue.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CaptionMetadata {
    /// Caption text shown to the player.
    text: String,
    /// Optional speaker label (for example a character name). Empty when absent.
    speaker: String,
    /// Suggested on-screen duration in milliseconds.
    duration_ms: u32,
    /// Language tag of `text` (a `BCP-47` tag such as `en-US`).
    language: String,
    /// Marks a description of non-speech audio rather than spoken dialogue.
    is_audio_description: bool,
}

impl CaptionMetadata {
    /// Creates a spoken-dialogue caption with the given `text` and `language`.
    ///
    /// `duration_ms` is the suggested on-screen time; the speaker label is
    /// empty and the audio-description flag is clear.
    #[must_use]
    pub fn new(text: String, language: String, duration_ms: u32) -> Self {
        Self {
            text,
            speaker: String::new(),
            duration_ms,
            language,
            is_audio_description: false,
        }
    }

    /// Sets the speaker label and returns the updated record.
    #[must_use]
    pub fn with_speaker(mut self, speaker: String) -> Self {
        self.speaker = speaker;
        self
    }

    /// Marks this record as an audio description (non-speech) line.
    #[must_use]
    pub fn as_audio_description(mut self) -> Self {
        self.is_audio_description = true;
        self
    }

    /// Overrides the suggested on-screen duration in milliseconds.
    #[must_use]
    pub fn with_duration_ms(mut self, duration_ms: u32) -> Self {
        self.duration_ms = duration_ms;
        self
    }

    /// Returns the caption text.
    #[inline]
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the speaker label, or an empty string when none is set.
    #[inline]
    #[must_use]
    pub fn speaker(&self) -> &str {
        &self.speaker
    }

    /// Returns `true` when a non-empty speaker label is present.
    #[inline]
    #[must_use]
    pub fn has_speaker(&self) -> bool {
        !self.speaker.is_empty()
    }

    /// Returns the suggested on-screen duration in milliseconds.
    #[inline]
    #[must_use]
    pub fn duration_ms(&self) -> u32 {
        self.duration_ms
    }

    /// Returns the language tag of the caption text.
    #[inline]
    #[must_use]
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Returns `true` when this record describes non-speech audio.
    #[inline]
    #[must_use]
    pub fn is_audio_description(&self) -> bool {
        self.is_audio_description
    }
}

/// A caption event ready to be forwarded through the telemetry ring.
///
/// The report pairs the [`CaptionMetadata`] with the identity of the event
/// that triggered it (`event_id`) and the playback position at which it fired
/// (`timestamp_samples`), so the UI can schedule the caption against the audio
/// timeline.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CaptionReport {
    /// The caption payload.
    caption: CaptionMetadata,
    /// Identifier of the triggering event.
    event_id: u64,
    /// Playback position (in samples) at which the event fired.
    timestamp_samples: u64,
}

impl CaptionReport {
    /// Returns the caption payload.
    #[inline]
    #[must_use]
    pub fn caption(&self) -> &CaptionMetadata {
        &self.caption
    }

    /// Returns the identifier of the triggering event.
    #[inline]
    #[must_use]
    pub fn event_id(&self) -> u64 {
        self.event_id
    }

    /// Returns the playback position (in samples) at which the event fired.
    #[inline]
    #[must_use]
    pub fn timestamp_samples(&self) -> u64 {
        self.timestamp_samples
    }
}

/// Builder that assembles a [`CaptionReport`] from a caption and event context.
///
/// The builder keeps the report construction explicit so a caller can fill in
/// the event identity and timeline position as that information becomes known,
/// then hand the finished report to the telemetry ring.
#[derive(Debug, Clone)]
pub struct CaptionReportBuilder {
    caption: CaptionMetadata,
    event_id: u64,
    timestamp_samples: u64,
}

impl CaptionReportBuilder {
    /// Starts a report for `caption` with a zeroed event id and timestamp.
    #[must_use]
    pub fn new(caption: CaptionMetadata) -> Self {
        Self {
            caption,
            event_id: 0,
            timestamp_samples: 0,
        }
    }

    /// Sets the identifier of the triggering event.
    #[must_use]
    pub fn event_id(mut self, event_id: u64) -> Self {
        self.event_id = event_id;
        self
    }

    /// Sets the playback position (in samples) at which the event fired.
    #[must_use]
    pub fn timestamp_samples(mut self, timestamp_samples: u64) -> Self {
        self.timestamp_samples = timestamp_samples;
        self
    }

    /// Consumes the builder and produces the finished [`CaptionReport`].
    #[must_use]
    pub fn build(self) -> CaptionReport {
        CaptionReport {
            caption: self.caption,
            event_id: self.event_id,
            timestamp_samples: self.timestamp_samples,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn dialogue_defaults_are_clear() {
        let cap = CaptionMetadata::new("Hello".to_string(), "en-US".to_string(), 1500);
        assert_eq!(cap.text(), "Hello");
        assert_eq!(cap.language(), "en-US");
        assert_eq!(cap.duration_ms(), 1500);
        assert!(!cap.has_speaker());
        assert!(!cap.is_audio_description());
    }

    #[test]
    fn builder_sets_speaker_and_description() {
        let cap = CaptionMetadata::new("door creaks".to_string(), "en-US".to_string(), 800)
            .with_speaker("Narrator".to_string())
            .as_audio_description()
            .with_duration_ms(900);
        assert!(cap.has_speaker());
        assert_eq!(cap.speaker(), "Narrator");
        assert!(cap.is_audio_description());
        assert_eq!(cap.duration_ms(), 900);
    }

    #[test]
    fn report_builder_carries_context() {
        let cap = CaptionMetadata::new("Fire!".to_string(), "en-US".to_string(), 500);
        let report = CaptionReportBuilder::new(cap.clone())
            .event_id(42)
            .timestamp_samples(96_000)
            .build();
        assert_eq!(report.event_id(), 42);
        assert_eq!(report.timestamp_samples(), 96_000);
        assert_eq!(report.caption(), &cap);
    }
}
