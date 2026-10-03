//! An Audio-Insights-style aggregate rollup over a profiling window.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the aggregate-report part of design section 26. An
//! [`InsightsReport`] folds a recorded [`ProfilerSession`] window, an optional
//! [`MeterSnapshot`], and a [`VoiceMonitor`] into a single read-only summary
//! for a dashboard. It performs no real-time work and allocates only the small
//! report structure; the heavy data stays in the borrowed sources.

use prism_audio_core::math::{linear_to_db, Sample};

use crate::meters::MeterSnapshot;
use crate::session::ProfilerSession;
use crate::voice_monitor::VoiceMonitor;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Aggregate voice counts for the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct VoiceRollup {
    /// Total voices tracked by the monitor.
    pub total: usize,
    /// Voices holding a physical slot in a running state.
    pub physical: usize,
    /// Voices that are retained but virtual.
    pub virtualized: usize,
    /// Voices audible at the configured floor.
    pub audible: usize,
    /// Voices that are silent for some diagnosed reason.
    pub silent: usize,
}

impl VoiceRollup {
    /// Build a rollup from a monitor at the given audibility floor.
    #[must_use]
    pub fn from_monitor(monitor: &VoiceMonitor, floor_db: Sample) -> Self {
        let total = monitor.len();
        let physical = monitor.physical_voices().count();
        let virtualized = monitor.virtual_voices().count();
        let audible = monitor.audible_count(floor_db);
        let silent = total.saturating_sub(audible);
        Self {
            total,
            physical,
            virtualized,
            audible,
            silent,
        }
    }
}

/// A single read-only summary of a profiling window.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct InsightsReport {
    /// Number of telemetry frames in the analysed window.
    pub frames_analyzed: usize,
    /// Mean CPU load across the window.
    pub mean_cpu_load: Sample,
    /// Peak CPU load across the window.
    pub peak_cpu_load: Sample,
    /// Count of frames that overran their real-time budget.
    pub overrun_count: usize,
    /// Peak physical voice count observed across the window.
    pub peak_physical_voices: u32,
    /// Number of recorded timeline events.
    pub event_count: usize,
    /// Master peak level in dBFS (from the meter snapshot when present, else
    /// from the most recent telemetry frame).
    pub master_peak_db: Sample,
    /// Master RMS level in dBFS (same source precedence as the peak).
    pub master_rms_db: Sample,
    /// Aggregate voice counts.
    pub voices: VoiceRollup,
}

impl InsightsReport {
    /// Fold a session window, an optional meter snapshot, and a voice monitor
    /// into one report.
    ///
    /// `floor_db` is the audibility floor passed through to the voice rollup.
    /// When `meter` is `None` the master levels are taken from the most recent
    /// telemetry frame, or the dB floor when the session is empty.
    #[must_use]
    pub fn from_parts(
        session: &ProfilerSession,
        meter: Option<&MeterSnapshot>,
        monitor: &VoiceMonitor,
        floor_db: Sample,
    ) -> Self {
        let (master_peak_db, master_rms_db) = match meter {
            Some(snapshot) => (snapshot.master_peak_db(), snapshot.master_rms_db()),
            None => {
                let latest = session.iter().last();
                match latest {
                    Some(frame) => (linear_to_db(frame.master_peak), linear_to_db(frame.master_rms)),
                    None => (Sample::NEG_INFINITY, Sample::NEG_INFINITY),
                }
            }
        };

        Self {
            frames_analyzed: session.len(),
            mean_cpu_load: session.mean_cpu_load(),
            peak_cpu_load: session.peak_cpu_load(),
            overrun_count: session.overrun_count(),
            peak_physical_voices: session.peak_physical_voices(),
            event_count: session.timeline().len(),
            master_peak_db,
            master_rms_db,
            voices: VoiceRollup::from_monitor(monitor, floor_db),
        }
    }

    /// `true` when no frame overran and the peak CPU load stayed within the
    /// real-time budget.
    #[must_use]
    #[inline]
    pub fn is_healthy(&self) -> bool {
        self.overrun_count == 0 && self.peak_cpu_load <= 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::AudioTelemetry;
    use crate::voice_monitor::VoiceStatus;

    fn frame(block: u64, cpu: Sample, peak: Sample, physical: u32) -> AudioTelemetry {
        AudioTelemetry {
            block_index: block,
            playhead: block * 128,
            frames: 128,
            physical_voices: physical,
            virtual_voices: 0,
            master_peak: peak,
            master_rms: peak * 0.5,
            cpu_load: cpu,
        }
    }

    #[test]
    fn report_summarizes_session_and_voices() {
        let mut session = ProfilerSession::new(16, 16);
        session.record_frame(frame(0, 0.4, 0.5, 3));
        session.record_frame(frame(1, 0.6, 0.9, 5));

        let mut monitor = VoiceMonitor::new();
        monitor.push(VoiceStatus::playing(1, "a", 1.0));
        let mut quiet = VoiceStatus::playing(2, "b", 0.00001);
        quiet.gain_linear = 0.00001;
        monitor.push(quiet);

        let report = InsightsReport::from_parts(&session, None, &monitor, -60.0);
        assert_eq!(report.frames_analyzed, 2);
        assert_eq!(report.peak_physical_voices, 5);
        assert_eq!(report.overrun_count, 0);
        assert!((report.mean_cpu_load - 0.5).abs() < 1e-4);
        assert!((report.peak_cpu_load - 0.6).abs() < 1e-4);
        assert_eq!(report.voices.total, 2);
        assert_eq!(report.voices.audible, 1);
        assert_eq!(report.voices.silent, 1);
        assert!(report.is_healthy());
        // Master levels fall back to the latest frame (peak 0.9 -> ~ -0.9 dB).
        assert!(report.master_peak_db < 0.0);
    }

    #[test]
    fn overrun_marks_report_unhealthy() {
        let mut session = ProfilerSession::new(8, 8);
        session.record_frame(frame(0, 1.3, 1.0, 2));
        let monitor = VoiceMonitor::new();
        let report = InsightsReport::from_parts(&session, None, &monitor, -60.0);
        assert_eq!(report.overrun_count, 1);
        assert!(!report.is_healthy());
    }

    #[test]
    fn empty_session_reports_floor_levels() {
        let session = ProfilerSession::new(8, 8);
        let monitor = VoiceMonitor::new();
        let report = InsightsReport::from_parts(&session, None, &monitor, -60.0);
        assert_eq!(report.frames_analyzed, 0);
        assert_eq!(report.master_peak_db, Sample::NEG_INFINITY);
        assert!(report.is_healthy());
        assert_eq!(report.voices.total, 0);
    }
}
