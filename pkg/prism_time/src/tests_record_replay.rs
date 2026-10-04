//! §24.1 tests: input+time record/replay — frame-exact replay, seed
//! reproduction, player scrubbing, and closed-loop determinism with an audit
//! trail driven by a deterministic clock + seeded RNG.

use crate::{AuditTrail, Duration, Player, RecordedFrame, Recorder, Recording, StateHasher, TickClock};
use alloc::vec::Vec;

/// A tiny deterministic input snapshot used by the tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Input {
    buttons: u16,
    axis: i16,
}

impl Input {
    fn new(buttons: u16, axis: i16) -> Self {
        Self { buttons, axis }
    }
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// `SplitMix64` — a deterministic, portable PRNG for reproduction tests.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A representative recorded session: varied steps, inputs and seeds.
fn sample_recording() -> Recording<Input> {
    let mut rec = Recorder::new();
    let frames = [
        (ms(16), Input::new(0b0001, 10), 1),
        (ms(17), Input::new(0b0011, -5), 2),
        (ms(16), Input::new(0b0000, 0), 3),
        (Duration::from_micros(16_666), Input::new(0b0100, 127), 4),
        (ms(33), Input::new(0b1000, -128), 5),
    ];
    for (dt, input, seed) in frames {
        rec.record_frame(dt, input, seed);
    }
    rec.finish()
}

#[test]
fn recorder_collects_frames_in_order() {
    let recording = sample_recording();
    assert_eq!(recording.len(), 5);
    assert!(!recording.is_empty());
    assert_eq!(recording.frame(0).unwrap().dt, ms(16));
    assert_eq!(recording.frame(1).unwrap().input, Input::new(0b0011, -5));
    assert_eq!(recording.frame(4).unwrap().seed, 5);
    // total_duration sums every dt exactly.
    let expected = ms(16) + ms(17) + ms(16) + Duration::from_micros(16_666) + ms(33);
    assert_eq!(recording.total_duration(), expected);
}

#[test]
fn player_replays_every_frame_identically() {
    let recording = sample_recording();
    let original: Vec<RecordedFrame<Input>> = recording.frames().to_vec();

    let mut player = Player::new(recording);
    assert_eq!(player.remaining(), 5);

    let mut replayed = Vec::new();
    while let Some(frame) = player.next_frame() {
        replayed.push(*frame);
    }
    assert_eq!(replayed, original);
    assert!(player.is_finished());
    assert_eq!(player.remaining(), 0);
    // Past the end yields None without panicking.
    assert!(player.next_frame().is_none());
}

#[test]
fn replay_drives_the_clock_frame_for_frame() {
    let recording = sample_recording();

    // "Live" run: feed the recorded deltas into a deterministic clock.
    let mut live = TickClock::from_hz(60);
    let mut live_ticks = Vec::new();
    for frame in recording.frames() {
        live.accumulate(frame.dt);
        live_ticks.push(live.expend_all());
    }

    // Replay run: identical clock, deltas fed from the player.
    let mut replay = TickClock::from_hz(60);
    let mut replay_ticks = Vec::new();
    let mut player = Player::new(recording);
    while let Some(frame) = player.next_frame() {
        replay.accumulate(frame.dt);
        replay_ticks.push(replay.expend_all());
    }

    assert_eq!(live_ticks, replay_ticks);
    assert_eq!(live.tick(), replay.tick());
    assert_eq!(live.overstep_subunits(), replay.overstep_subunits());
}

#[test]
fn same_seed_reproduces_identical_rng_stream() {
    let recording = sample_recording();

    // Run A: consume per-frame seeds through the PRNG.
    let mut run_a = Vec::new();
    for frame in recording.frames() {
        let mut state = frame.seed;
        run_a.push(splitmix64(&mut state));
        run_a.push(splitmix64(&mut state));
    }

    // Run B: replay the identical seeds via the player.
    let mut run_b = Vec::new();
    let mut player = Player::new(recording);
    while let Some(frame) = player.next_frame() {
        let mut state = frame.seed;
        run_b.push(splitmix64(&mut state));
        run_b.push(splitmix64(&mut state));
    }

    assert_eq!(run_a, run_b);
}

#[test]
fn closed_loop_record_then_replay_hashes_identically() {
    // Record a session while simulating; capture an audit trail of per-frame
    // state (clock tick + rng-derived "position"). Replaying the recording
    // must reproduce the identical trail — the CI determinism assertion.
    fn run(recording: &Recording<Input>) -> AuditTrail {
        let mut trail = AuditTrail::new();
        let mut clock = TickClock::from_hz(60);
        let mut position: u64 = 0;
        let mut player = Player::new(recording.clone());
        while let Some(frame) = player.next_frame() {
            clock.accumulate(frame.dt);
            let steps = clock.expend_all();
            // Deterministic "gameplay": advance a seeded RNG once per fixed step
            // and mix the input so state depends on dt + input + seed.
            let mut rng = frame.seed ^ u64::from(frame.input.buttons);
            for _ in 0..steps {
                position = position.wrapping_add(splitmix64(&mut rng));
            }
            let mut h = StateHasher::new();
            h.write_u64(clock.tick());
            h.write_u128(clock.overstep_subunits());
            h.write_u64(position);
            trail.record(h.finish());
        }
        trail
    }

    let recording = sample_recording();
    let trail_a = run(&recording);
    let trail_b = run(&recording);
    assert_eq!(trail_a.len(), recording.len());
    assert_eq!(crate::compare_trails(&trail_a, &trail_b), crate::AuditDiff::Identical);
}

#[test]
fn player_seek_and_reset_scrub_frames() {
    let recording = sample_recording();
    let mut player = Player::new(recording);

    player.seek(3);
    assert_eq!(player.cursor(), 3);
    assert_eq!(player.next_frame().unwrap().seed, 4);

    player.reset();
    assert_eq!(player.cursor(), 0);
    assert_eq!(player.peek().unwrap().seed, 1);

    // Seek clamps past the end rather than overrunning.
    player.seek(999);
    assert!(player.is_finished());
    assert_eq!(player.cursor(), player.len());
    assert!(player.next_frame().is_none());
}

#[test]
fn recorder_clear_and_defaults() {
    let mut rec: Recorder<Input> = Recorder::default();
    assert!(rec.is_empty());
    rec.record(RecordedFrame::new(ms(16), Input::new(1, 2), 7));
    assert_eq!(rec.len(), 1);
    assert_eq!(rec.recording().len(), 1);
    rec.clear();
    assert!(rec.is_empty());
}

#[test]
fn empty_recording_player_is_immediately_finished() {
    let recording: Recording<Input> = Recording::new();
    assert!(recording.is_empty());
    assert_eq!(recording.total_duration(), Duration::ZERO);
    let mut player = Player::new(recording);
    assert!(player.is_empty());
    assert!(player.is_finished());
    assert_eq!(player.remaining(), 0);
    assert!(player.next_frame().is_none());
}

#[test]
fn recording_round_trips_through_player_into_recording() {
    let recording = sample_recording();
    let player = Player::new(recording.clone());
    let back = player.into_recording();
    assert_eq!(back, recording);
}
