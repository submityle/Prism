//! End-to-end integration coverage for the profiler telemetry pipeline:
//! per-block `AudioTelemetry` accounting, the bounded `ProfilerSession` ring
//! buffer with playback cursor, the `EventTimeline` log with range queries,
//! voice-audibility diagnosis via `VoiceMonitor`/`VoiceStatus`, and the
//! `GoldenDiff` buffer comparator used for golden-master regression checks.
//!
//! The tests drive each public stage with deterministic data and verify the
//! observable contracts (eviction, cursor advance, diagnosis priority,
//! tolerance gating) as a connected whole rather than in isolation.
//!
//! # Provenance
//! Original work authored for Prism. It contains no source code or derived
//! code from Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Dolby,
//! MPEG, Google Resonance, Web Audio, or Microsoft Project Acoustics, and uses
//! no artificial-intelligence or machine-learning techniques. Only the ideas
//! of publicly documented concepts (dBFS metering, ULP distance, ring buffers)
//! are relied upon.
//!
//! # Relationship
//! Exercises the profiler/telemetry chapter of
//! `docs/prism_audio_engine_design_zh.md`. Depends only on the
//! `prism_audio_profiler` crate's public surface; it constructs telemetry and
//! voice records directly, so it needs no non-dev dependency.

use prism_audio_profiler::{
    AudioTelemetry, EventKind, EventTimeline, GoldenDiff, GoldenDiffConfig, ProfilerSession,
    SilenceReason, TimelineEvent, VoiceMonitor, VoiceState, VoiceStatus,
};

const FLOOR_DB: f32 = -60.0;

/// Branchless magnitude helper (the workspace lints forbid std float math in
/// these integration tests).
fn fabs(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Builds a telemetry frame from the fields the tests vary, zeroing the rest.
fn frame(block_index: u64, cpu_load: f32, physical_voices: u32) -> AudioTelemetry {
    AudioTelemetry {
        block_index,
        cpu_load,
        physical_voices,
        ..AudioTelemetry::empty()
    }
}

#[test]
fn golden_diff_detects_exact_match() {
    let reference = [0.0f32, 0.5, -0.5, 1.0];
    let candidate = [0.0f32, 0.5, -0.5, 1.0];
    let diff = GoldenDiff::compare(&reference, &candidate, GoldenDiffConfig::exact());
    assert_eq!(diff.compared, 4);
    assert!(!diff.length_mismatch);
    assert!(diff.first_divergence.is_none());
    assert!(diff.max_abs_error <= 0.0);
    assert_eq!(diff.max_ulp, 0);
    assert!(diff.is_match());
    assert!(diff.within(GoldenDiffConfig::exact()));
}

#[test]
fn golden_diff_reports_divergence_and_tolerance_gating() {
    let reference = [0.0f32, 0.5, -0.5, 1.0];
    let candidate = [0.0f32, 0.5, -0.4, 1.0];
    let diff = GoldenDiff::compare(&reference, &candidate, GoldenDiffConfig::exact());
    assert_eq!(diff.first_divergence, Some(2));
    assert!(fabs(diff.max_abs_error - 0.1) < 1.0e-5);
    assert!(!diff.is_match());
    // A tolerant budget wider than the error accepts the candidate.
    assert!(diff.within(GoldenDiffConfig::tolerant(0.2, 0.2)));
    // A budget tighter than the error rejects it.
    assert!(!diff.within(GoldenDiffConfig::tolerant(0.05, 0.05)));
}

#[test]
fn golden_diff_flags_length_mismatch() {
    let reference = [0.0f32, 0.5, -0.5, 1.0];
    let candidate = [0.0f32, 0.5, -0.5];
    let diff = GoldenDiff::compare(&reference, &candidate, GoldenDiffConfig::exact());
    assert!(diff.length_mismatch);
    assert_eq!(diff.compared, 3);
    assert!(!diff.within(GoldenDiffConfig::exact()));
}

#[test]
fn session_ring_buffer_evicts_oldest_and_aggregates() {
    let mut session = ProfilerSession::new(3, 4);
    let loads = [0.2f32, 0.4, 0.6, 1.5, 0.8];
    let voices = [1u32, 2, 3, 4, 5];
    for (block, (&load, &phys)) in loads.iter().zip(voices.iter()).enumerate() {
        session.record_frame(frame(block as u64, load, phys));
    }
    // Capacity 3 keeps the three most recent frames (blocks 2, 3, 4).
    assert_eq!(session.len(), 3);
    assert_eq!(session.frame(0).expect("oldest retained").block_index, 2);
    assert!(fabs(session.peak_cpu_load() - 1.5) < 1.0e-6);
    assert!(fabs(session.mean_cpu_load() - (0.6 + 1.5 + 0.8) / 3.0) < 1.0e-6);
    assert_eq!(session.overrun_count(), 1);
    assert_eq!(session.peak_physical_voices(), 5);
}

#[test]
fn session_playback_cursor_advances_and_terminates() {
    let mut session = ProfilerSession::new(4, 4);
    for block in 0..3u64 {
        session.record_frame(frame(block, 0.1, 1));
    }
    session.rewind();
    assert_eq!(session.playback_cursor(), 0);
    let first = session.next_playback().expect("first playback frame");
    assert_eq!(first.block_index, 0);
    assert_eq!(session.playback_cursor(), 1);

    session.seek(2);
    let third = session.next_playback().expect("third playback frame");
    assert_eq!(third.block_index, 2);
    // The cursor is now at the end; playback is exhausted.
    assert!(session.next_playback().is_none());
}

#[test]
fn event_timeline_records_range_and_evicts() {
    let mut timeline = EventTimeline::new(4);
    assert!(timeline.is_empty());
    timeline.record(TimelineEvent::new(1, 100, EventKind::VoiceStarted, "a"));
    timeline.record(TimelineEvent::new(2, 200, EventKind::Marker, "m"));
    timeline.record(TimelineEvent::new(3, 300, EventKind::VoiceStopped, "b"));
    timeline.record(TimelineEvent::new(5, 500, EventKind::Dropout, "d"));
    // Fifth event overflows capacity 4, evicting the oldest (block 1).
    timeline.record(TimelineEvent::new(7, 700, EventKind::GraphSwapped, "g"));
    assert_eq!(timeline.capacity(), 4);
    assert_eq!(timeline.len(), 4);
    assert_eq!(
        timeline.iter().next().expect("oldest retained").block_index,
        2
    );
    let windowed = timeline.in_block_range(3, 5);
    assert_eq!(windowed.len(), 2);
    assert_eq!(windowed[0].block_index, 3);
    assert_eq!(windowed[1].block_index, 5);
}

#[test]
fn voice_status_diagnoses_silence_in_priority_order() {
    let audible = VoiceStatus::playing(1, "lead", 1.0);
    assert!(audible.is_audible(FLOOR_DB));
    assert!(audible.silence_reason(FLOOR_DB).is_none());

    let mut muted = VoiceStatus::playing(2, "muted", 1.0);
    muted.muted = true;
    assert_eq!(muted.silence_reason(FLOOR_DB), Some(SilenceReason::Muted));

    let mut virtualized = VoiceStatus::playing(3, "virtual", 1.0);
    virtualized.physical = false;
    virtualized.state = VoiceState::Virtual;
    assert_eq!(
        virtualized.silence_reason(FLOOR_DB),
        Some(SilenceReason::Virtualized)
    );

    let mut unrouted = VoiceStatus::playing(4, "unrouted", 1.0);
    unrouted.routed = false;
    assert_eq!(
        unrouted.silence_reason(FLOOR_DB),
        Some(SilenceReason::Unrouted)
    );

    let quiet = VoiceStatus::playing(5, "quiet", 0.0001);
    assert_eq!(quiet.silence_reason(FLOOR_DB), Some(SilenceReason::BelowFloor));

    // Priority: a stopped and muted voice reports NotPlaying first.
    let mut stopped = VoiceStatus::playing(6, "stopped", 1.0);
    stopped.state = VoiceState::Stopped;
    stopped.muted = true;
    assert_eq!(
        stopped.silence_reason(FLOOR_DB),
        Some(SilenceReason::NotPlaying)
    );
}

#[test]
fn voice_monitor_partitions_and_counts() {
    let mut monitor = VoiceMonitor::new();
    monitor.push(VoiceStatus::playing(1, "lead", 1.0));

    let mut muted = VoiceStatus::playing(2, "muted", 1.0);
    muted.muted = true;
    monitor.push(muted);

    let mut virt = VoiceStatus::playing(3, "virtual", 1.0);
    virt.physical = false;
    virt.state = VoiceState::Virtual;
    monitor.push(virt);

    monitor.push(VoiceStatus::playing(4, "quiet", 0.0001));

    assert_eq!(monitor.len(), 4);
    assert_eq!(monitor.audible_count(FLOOR_DB), 1);
    assert_eq!(monitor.physical_voices().count(), 3);
    assert_eq!(monitor.virtual_voices().count(), 1);
    assert_eq!(monitor.find(3).expect("voice 3 present").id, 3);

    let silent = monitor.silent_voices(FLOOR_DB);
    assert_eq!(silent.len(), 3);

    monitor.clear();
    assert!(monitor.is_empty());
}

#[test]
fn telemetry_frame_derives_load_and_voice_metrics() {
    let overrun = AudioTelemetry {
        frames: 256,
        physical_voices: 4,
        virtual_voices: 6,
        master_peak: 0.5,
        master_rms: 0.25,
        cpu_load: 1.2,
        ..AudioTelemetry::empty()
    };
    assert_eq!(overrun.total_voices(), 10);
    assert!(overrun.is_overrun());
    assert!(overrun.cpu_headroom() <= 0.0);
    assert!(fabs(overrun.block_duration_s(48_000) - 256.0 / 48_000.0) < 1.0e-9);
    // 0.5 peak and 0.25 RMS are below unity, so both dB readings are negative.
    assert!(overrun.master_peak_db() < 0.0);
    assert!(overrun.master_rms_db() < overrun.master_peak_db());

    let healthy = AudioTelemetry {
        cpu_load: 0.25,
        ..AudioTelemetry::empty()
    };
    assert!(!healthy.is_overrun());
    assert!(fabs(healthy.cpu_headroom() - 0.75) < 1.0e-6);
}
