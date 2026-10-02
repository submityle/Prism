//! The [`AudioTelemetry`] resource: the latest audio-thread
//! [`TelemetryFrame`] plus a bounded history ring drained from the telemetry
//! ring every frame.
//!
//! The telemetry pump system calls
//! [`recv_telemetry`](prism_audio_rt::AudioRuntimeClient::recv_telemetry) in a
//! loop and feeds each frame to [`AudioTelemetry::push`]. Diagnostics overlays,
//! profilers, and tests read [`AudioTelemetry::latest`] or walk the history.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Fed by [`crate::systems::pump_telemetry`] from `prism_audio_rt`'s telemetry
//! ring.

use alloc::collections::VecDeque;

use bevy_ecs::resource::Resource;
use prism_audio_rt::TelemetryFrame;

/// Default number of recent frames retained in the history ring.
const DEFAULT_CAPACITY: usize = 128;

/// Most recent audio-thread telemetry, with a bounded history.
///
/// [`AudioTelemetry::latest`] is the newest frame observed; the history keeps
/// up to [`AudioTelemetry::capacity`] frames in arrival order, evicting the
/// oldest first.
#[derive(Resource, Debug, Clone)]
pub struct AudioTelemetry {
    /// Newest frame observed, or [`TelemetryFrame::empty`] before any arrive.
    latest: TelemetryFrame,
    /// Recent frames in arrival order (oldest front, newest back).
    history: VecDeque<TelemetryFrame>,
    /// Maximum number of frames retained in `history`.
    capacity: usize,
    /// Total number of frames ever pushed, across all evictions.
    received: u64,
}

impl AudioTelemetry {
    /// Builds a telemetry buffer retaining up to `capacity` recent frames.
    ///
    /// A `capacity` of zero keeps only [`AudioTelemetry::latest`] with no
    /// history.
    #[must_use]
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            latest: TelemetryFrame::empty(),
            history: VecDeque::with_capacity(capacity),
            capacity,
            received: 0,
        }
    }

    /// Records a newly arrived `frame`, updating [`AudioTelemetry::latest`] and
    /// appending to the bounded history.
    #[inline]
    pub fn push(&mut self, frame: TelemetryFrame) {
        self.latest = frame;
        self.received = self.received.saturating_add(1);
        if self.capacity == 0 {
            return;
        }
        if self.history.len() == self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(frame);
    }

    /// The newest frame observed.
    #[must_use]
    #[inline]
    pub fn latest(&self) -> TelemetryFrame {
        self.latest
    }

    /// The retained history, oldest front to newest back.
    #[must_use]
    #[inline]
    pub fn history(&self) -> &VecDeque<TelemetryFrame> {
        &self.history
    }

    /// Maximum number of frames the history retains.
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total number of frames ever pushed.
    #[must_use]
    #[inline]
    pub fn received(&self) -> u64 {
        self.received
    }

    /// Whether at least one frame has been observed.
    #[must_use]
    #[inline]
    pub fn has_data(&self) -> bool {
        self.received > 0
    }
}

impl Default for AudioTelemetry {
    #[inline]
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }
}
