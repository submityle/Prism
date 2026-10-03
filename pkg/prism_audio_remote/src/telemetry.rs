//! A read-only mirror of the engine telemetry ring for remote panels.
//!
//! The engine publishes a periodic snapshot of its health (active voice count,
//! per-bus levels, processing load, and a short event timeline). A remote
//! profiler panel cannot and must not touch the live ring, so it holds a
//! [`TelemetryMirror`] that stores the most recent [`TelemetrySnapshot`] and a
//! monotonically increasing generation counter. The mirror is strictly
//! read-only toward the engine: callers [`update`](TelemetryMirror::update) it
//! from decoded transport traffic and then render from
//! [`snapshot`](TelemetryMirror::snapshot).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the read-only telemetry side of design section 38 by mirroring
//! the profiler ring of design section 26. Bus and event identifiers are reused
//! from [`crate::command`] so a telemetry marker and the command that produced
//! it share one identifier space.

use alloc::vec::Vec;

use crate::command::{BusId, EventId};

/// A single bus level reading in decibels.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BusLevel {
    /// Which bus this reading belongs to.
    pub bus: BusId,
    /// Peak level over the reporting window, in decibels.
    pub peak_db: f32,
    /// Root-mean-square level over the reporting window, in decibels.
    pub rms_db: f32,
}

/// A single entry on the recent event timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EventMarker {
    /// Which event fired.
    pub event: EventId,
    /// The sample-clock time at which it fired.
    pub time_samples: u64,
}

/// An immutable, serializable snapshot of engine telemetry.
///
/// This is the unit the engine endpoint sends up the transport; the mirror
/// stores the latest one it decodes.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TelemetrySnapshot {
    /// The generation stamped by [`TelemetryMirror::update`]; zero when empty.
    pub generation: u64,
    /// Number of voices actively rendering.
    pub active_voices: u32,
    /// Normalized processing load in `[0, 1]`.
    pub cpu_load: f32,
    /// Per-bus level readings.
    pub bus_levels: Vec<BusLevel>,
    /// Recently fired events, oldest first.
    pub event_timeline: Vec<EventMarker>,
}

impl TelemetrySnapshot {
    /// Creates an empty snapshot with generation zero.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }
}

/// A read-only holder for the most recent [`TelemetrySnapshot`].
///
/// The mirror owns no connection; it is fed decoded snapshots and hands out
/// borrows for rendering. Each successful update bumps a generation counter so
/// a panel can cheaply detect staleness.
#[derive(Debug, Clone, Default)]
pub struct TelemetryMirror {
    current: TelemetrySnapshot,
    generation: u64,
}

impl TelemetryMirror {
    /// Creates an empty mirror at generation zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the stored snapshot, bumping and stamping the generation.
    ///
    /// The caller-supplied [`TelemetrySnapshot::generation`] field is ignored
    /// and overwritten with the mirror's own counter so the stored value is
    /// always authoritative.
    pub fn update(&mut self, mut snapshot: TelemetrySnapshot) {
        self.generation = self.generation.saturating_add(1);
        snapshot.generation = self.generation;
        self.current = snapshot;
    }

    /// Borrows the most recent snapshot.
    #[must_use]
    pub fn snapshot(&self) -> &TelemetrySnapshot {
        &self.current
    }

    /// Returns the current generation (zero before the first update).
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const EPS: f32 = 1.0e-6;

    #[test]
    fn fresh_mirror_is_empty() {
        let mirror = TelemetryMirror::new();
        assert_eq!(mirror.generation(), 0);
        assert_eq!(mirror.snapshot().active_voices, 0);
        assert!(mirror.snapshot().bus_levels.is_empty());
    }

    #[test]
    fn update_bumps_generation_and_stores() {
        let mut mirror = TelemetryMirror::new();
        let snap = TelemetrySnapshot {
            active_voices: 12,
            cpu_load: 0.25,
            bus_levels: vec![BusLevel {
                bus: BusId(1),
                peak_db: -6.0,
                rms_db: -12.0,
            }],
            event_timeline: vec![EventMarker {
                event: EventId(5),
                time_samples: 44_100,
            }],
            ..TelemetrySnapshot::empty()
        };
        mirror.update(snap);
        assert_eq!(mirror.generation(), 1);
        assert_eq!(mirror.snapshot().generation, 1);
        assert_eq!(mirror.snapshot().active_voices, 12);
        assert!((mirror.snapshot().cpu_load - 0.25).abs() < EPS);
        assert!((mirror.snapshot().bus_levels[0].peak_db + 6.0).abs() < EPS);
    }

    #[test]
    fn generation_is_monotonic() {
        let mut mirror = TelemetryMirror::new();
        mirror.update(TelemetrySnapshot::empty());
        mirror.update(TelemetrySnapshot::empty());
        mirror.update(TelemetrySnapshot::empty());
        assert_eq!(mirror.generation(), 3);
        assert_eq!(mirror.snapshot().generation, 3);
    }
}
