//! M6 integration tests: timeline/sequencer, frame-step, diagnostics, driver.

use crate::{
    DefaultSource, Duration, FrameBudget, FrameStats, FrameStepper, PlaybackMode, Sequencer,
    StepState, TimeDriver, Timeline, TimelineMarker,
};

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

// --- Timeline -------------------------------------------------------------

#[test]
fn timeline_once_advances_and_finishes() {
    let mut t = Timeline::new(ms(1000));
    assert!(t.is_playing());
    assert!(!t.is_finished());

    let tick = t.advance(ms(400));
    assert_eq!(tick.applied, ms(400));
    assert_eq!(tick.wrapped, 0);
    assert!(!tick.finished);
    assert_eq!(t.position(), ms(400));
    assert_eq!(t.previous_position(), Duration::ZERO);

    // Overshoot clamps to duration, finishes, stops playing.
    let tick = t.advance(ms(900));
    assert_eq!(tick.applied, ms(600));
    assert_eq!(tick.wrapped, 1);
    assert!(tick.finished);
    assert_eq!(t.position(), ms(1000));
    assert!(t.is_finished());
    assert!(!t.is_playing());

    // Further advances do nothing (already stopped at the end).
    let tick = t.advance(ms(100));
    assert_eq!(tick.applied, Duration::ZERO);
    assert!(tick.finished);
    assert_eq!(t.position(), ms(1000));
}

#[test]
fn timeline_progress_and_speed() {
    let mut t = Timeline::new(ms(2000));
    t.set_speed(2.0);
    assert_eq!(t.speed(), 2.0);
    t.advance(ms(500)); // 500ms real * 2 = 1000ms
    assert_eq!(t.position(), ms(1000));
    assert!((t.progress() - 0.5).abs() < 1e-9);

    // Negative speed clamps to zero; non-finite ignored.
    t.set_speed(-3.0);
    assert_eq!(t.speed(), 0.0);
    t.set_speed(f64::NAN);
    assert_eq!(t.speed(), 0.0);
    let tick = t.advance(ms(100));
    assert_eq!(tick.applied, Duration::ZERO);
}

#[test]
fn timeline_pause_play_stop() {
    let mut t = Timeline::new(ms(1000));
    t.advance(ms(200));
    t.pause();
    assert!(!t.is_playing());
    let tick = t.advance(ms(500));
    assert_eq!(tick.applied, Duration::ZERO);
    assert_eq!(t.position(), ms(200));

    t.play();
    t.advance(ms(100));
    assert_eq!(t.position(), ms(300));

    t.stop();
    assert!(!t.is_playing());
    assert_eq!(t.position(), Duration::ZERO);
}

#[test]
fn timeline_seek_clamps() {
    let mut t = Timeline::new(ms(1000));
    t.seek(ms(400));
    assert_eq!(t.position(), ms(400));
    t.seek(ms(5000));
    assert_eq!(t.position(), ms(1000));
    t.seek_progress(0.25);
    assert_eq!(t.position(), ms(250));
    t.seek_progress(2.0);
    assert_eq!(t.position(), ms(1000));
    t.seek_progress(-1.0);
    assert_eq!(t.position(), Duration::ZERO);
}

#[test]
fn timeline_loop_wraps() {
    let mut t = Timeline::with_mode(ms(1000), PlaybackMode::Loop);
    let tick = t.advance(ms(2500));
    // 2500 over 1000 => 2 wraps, lands at 500.
    assert_eq!(tick.wrapped, 2);
    assert_eq!(t.position(), ms(500));
    assert!(!tick.finished);
    assert!(!t.is_finished());
    assert!(t.is_playing());
}

#[test]
fn timeline_loop_exact_boundary_wraps_to_zero() {
    let mut t = Timeline::with_mode(ms(1000), PlaybackMode::Loop);
    let tick = t.advance(ms(1000));
    assert_eq!(tick.wrapped, 1);
    assert_eq!(t.position(), Duration::ZERO);
}

#[test]
fn timeline_ping_pong_reverses() {
    let mut t = Timeline::with_mode(ms(1000), PlaybackMode::PingPong);
    // Go forward to the end then bounce back 300ms into the reverse leg.
    let tick = t.advance(ms(1300));
    assert_eq!(tick.wrapped, 1);
    assert_eq!(t.position(), ms(700));
    // Continue reversing to the start then bounce forward 200ms.
    let tick = t.advance(ms(900));
    assert_eq!(tick.wrapped, 1);
    assert_eq!(t.position(), ms(200));
    assert!(t.is_playing());
}

#[test]
fn timeline_zero_duration_finishes_once() {
    let mut t = Timeline::new(Duration::ZERO);
    assert!((t.progress() - 1.0).abs() < 1e-9);
    let tick = t.advance(ms(10));
    assert!(tick.finished);
    assert!(!t.is_playing());

    let mut looped = Timeline::with_mode(Duration::ZERO, PlaybackMode::Loop);
    let tick = looped.advance(ms(10));
    assert!(!tick.finished);
    assert!(looped.is_playing());
}

// --- Sequencer ------------------------------------------------------------

#[test]
fn sequencer_markers_stay_sorted() {
    let mut seq: Sequencer<4> = Sequencer::new(Timeline::new(ms(1000)));
    seq.try_add_marker(2, ms(600)).unwrap();
    seq.try_add_marker(1, ms(200)).unwrap();
    seq.try_add_marker(3, ms(900)).unwrap();
    let ids = marker_ids(&seq);
    assert_eq!(ids, [1, 2, 3]);
    assert_eq!(seq.marker_count(), 3);
    assert!(!seq.is_full());
}

fn marker_ids<const N: usize>(seq: &Sequencer<N>) -> [u32; 3] {
    let m = seq.markers();
    [m[0].id, m[1].id, m[2].id]
}

#[test]
fn sequencer_full_rejects() {
    let mut seq: Sequencer<2> = Sequencer::new(Timeline::new(ms(1000)));
    seq.try_add_marker(1, ms(100)).unwrap();
    seq.try_add_marker(2, ms(200)).unwrap();
    assert!(seq.is_full());
    let err = seq.try_add_marker(3, ms(300));
    assert!(err.is_err());
}

#[test]
fn sequencer_fires_markers_in_order() {
    let mut seq: Sequencer<4> = Sequencer::new(Timeline::new(ms(1000)));
    seq.try_add_marker(10, ms(100)).unwrap();
    seq.try_add_marker(20, ms(300)).unwrap();
    seq.try_add_marker(30, ms(900)).unwrap();

    let mut fired: [u32; 4] = [0; 4];
    let mut n = 0;
    seq.advance_with(ms(400), |m: TimelineMarker| {
        fired[n] = m.id;
        n += 1;
    });
    // Crossed 100 and 300 but not 900.
    assert_eq!(n, 2);
    assert_eq!(&fired[..2], &[10, 20]);

    // No marker in (400, 500].
    let mut hit = false;
    seq.advance_with(ms(100), |_| hit = true);
    assert!(!hit);

    // Cross 900.
    let mut last = 0;
    seq.advance_with(ms(500), |m| last = m.id);
    assert_eq!(last, 30);
}

#[test]
fn sequencer_fires_across_single_loop_wrap() {
    let mut seq: Sequencer<4> = Sequencer::new(Timeline::with_mode(ms(1000), PlaybackMode::Loop));
    seq.try_add_marker(1, ms(50)).unwrap();
    seq.try_add_marker(2, ms(950)).unwrap();
    // Start near the end so a step wraps once: 900 -> (900,1000] fires 950,
    // then (0,100] fires 50.
    seq.timeline_mut().seek(ms(900));
    let mut fired: [u32; 4] = [0; 4];
    let mut n = 0;
    let tick = seq.advance_with(ms(200), |m| {
        fired[n] = m.id;
        n += 1;
    });
    assert_eq!(tick.wrapped, 1);
    assert_eq!(n, 2);
    assert_eq!(&fired[..2], &[2, 1]);
    assert_eq!(seq.timeline().position(), ms(100));
}

// --- FrameStepper ---------------------------------------------------------

#[test]
fn stepper_running_passes_delta_through() {
    let mut s = FrameStepper::new();
    assert!(s.is_running());
    assert_eq!(s.next_delta(ms(16)), ms(16));
}

#[test]
fn stepper_paused_freezes_until_stepped() {
    let mut s = FrameStepper::with_step_delta(ms(16));
    s.pause();
    assert_eq!(s.state(), StepState::Paused);
    // Frozen while no step is queued.
    assert_eq!(s.next_delta(ms(100)), Duration::ZERO);

    s.request_step();
    assert_eq!(s.pending_steps(), 1);
    assert!(s.would_advance());
    // Consumes one step and releases exactly one step delta.
    assert_eq!(s.next_delta(ms(100)), ms(16));
    assert_eq!(s.pending_steps(), 0);
    assert_eq!(s.next_delta(ms(100)), Duration::ZERO);
}

#[test]
fn stepper_request_many_and_resume_clears() {
    let mut s = FrameStepper::new();
    s.pause();
    s.request_steps(3);
    assert_eq!(s.pending_steps(), 3);
    s.resume();
    assert_eq!(s.pending_steps(), 0);
    assert!(s.is_running());
}

#[test]
fn stepper_toggle() {
    let mut s = FrameStepper::new();
    s.toggle();
    assert!(s.is_paused());
    s.toggle();
    assert!(s.is_running());
}

// --- FrameBudget / FrameStats --------------------------------------------

#[test]
fn frame_budget_over_under() {
    let b = FrameBudget::from_hz(60.0); // ~16.667ms
    assert!(b.is_over_budget(ms(20)));
    assert!(!b.is_over_budget(ms(10)));
    assert!(b.overrun(ms(10)).is_none());
    assert!(b.overrun(ms(20)).is_some());
    assert!(b.headroom(ms(10)).is_some());
    assert!(b.headroom(ms(20)).is_none());
    assert!(b.utilization(ms(10)) < 1.0);
    assert!(b.utilization(ms(20)) > 1.0);
}

#[test]
fn frame_budget_zero_target() {
    let b = FrameBudget::new(Duration::ZERO);
    assert_eq!(b.utilization(ms(5)), 0.0);
}

#[test]
fn frame_stats_ring_rolls_over() {
    let mut s: FrameStats<3> = FrameStats::new();
    assert!(s.is_empty());
    assert!(s.min().is_none());

    s.record(ms(10));
    s.record(ms(20));
    s.record(ms(30));
    assert!(s.is_full());
    assert_eq!(s.len(), 3);
    assert_eq!(s.latest(), Some(ms(30)));
    assert_eq!(s.min(), Some(ms(10)));
    assert_eq!(s.max(), Some(ms(30)));
    assert_eq!(s.mean(), Some(ms(20)));
    assert_eq!(s.jitter(), Some(ms(20)));

    // Evict the oldest (10) with a new sample.
    s.record(ms(40));
    assert_eq!(s.latest(), Some(ms(40)));
    assert_eq!(s.min(), Some(ms(20)));
    assert_eq!(s.max(), Some(ms(40)));
    assert_eq!(s.mean(), Some(ms(30)));
}

#[test]
fn frame_stats_stddev_zero_for_constant() {
    let mut s: FrameStats<4> = FrameStats::new();
    for _ in 0..4 {
        s.record(ms(16));
    }
    assert_eq!(s.stddev(), Some(Duration::ZERO));
    assert_eq!(s.jitter(), Some(Duration::ZERO));
}

#[test]
fn frame_stats_stddev_matches_known_spread() {
    // Samples 10,20,30,20 ms: mean = 20 ms, variance = (100+0+100+0)/4 = 50
    // ms^2 -> stddev = sqrt(50) ms ~= 7.0710678 ms. In ns^2 the integer sqrt
    // of 50_000_000^2-scaled variance yields floor(sqrt) ns.
    let mut s: FrameStats<4> = FrameStats::new();
    s.record(ms(10));
    s.record(ms(20));
    s.record(ms(30));
    s.record(ms(20));
    let sd = s.stddev().unwrap();
    // ~7.071 ms; allow +/- 1 ms for the integer-sqrt flooring.
    assert!(sd >= ms(6) && sd <= ms(8), "stddev was {sd:?}");
}

#[test]
fn frame_stats_zero_capacity_is_noop() {
    let mut s: FrameStats<0> = FrameStats::new();
    s.record(ms(16));
    assert!(s.is_empty());
    assert!(s.latest().is_none());
}

// --- TimeDriver -----------------------------------------------------------

#[test]
fn driver_advances_real_virtual_fixed() {
    let mut d: TimeDriver<8> = TimeDriver::from_hz(60.0);
    let report = d.advance(ms(16));
    assert_eq!(report.real_delta, ms(16));
    assert_eq!(report.virtual_delta, ms(16));
    assert!(!report.stepped);
    assert_eq!(d.clocks().source(), DefaultSource::Virtual);
    assert_eq!(d.clocks().real().delta(), ms(16));
    assert_eq!(d.clocks().virtual_time().delta(), ms(16));
    // Fixed step is 1/64s (~15.625ms); 16ms accumulates one step.
    assert!(d.expend_fixed());
    assert_eq!(d.clocks().source(), DefaultSource::Fixed);
    assert!(!d.expend_fixed());
    assert_eq!(d.clocks().source(), DefaultSource::Virtual);
}

#[test]
fn driver_pause_freezes_virtual_not_real() {
    let mut d: TimeDriver<8> = TimeDriver::from_hz(60.0);
    d.stepper_mut().pause();
    let report = d.advance(ms(16));
    // Real clock still advances; virtual is frozen.
    assert_eq!(report.real_delta, ms(16));
    assert_eq!(report.virtual_delta, Duration::ZERO);
    assert!(!report.stepped);
    assert_eq!(d.clocks().real().delta(), ms(16));
    assert_eq!(d.clocks().virtual_time().delta(), Duration::ZERO);
    assert!(!d.expend_fixed());
}

#[test]
fn driver_single_step_releases_one_frame() {
    let mut d: TimeDriver<8> = TimeDriver::from_hz(60.0);
    d.stepper_mut().pause();
    d.stepper_mut().set_step_delta(ms(16));
    d.stepper_mut().request_step();
    let report = d.advance(ms(100));
    assert!(report.stepped);
    assert_eq!(report.virtual_delta, ms(16));
    // Next frame with no queued step freezes again.
    let report = d.advance(ms(100));
    assert!(!report.stepped);
    assert_eq!(report.virtual_delta, Duration::ZERO);
}

#[test]
fn driver_over_budget_flagged_and_recorded() {
    let mut d: TimeDriver<4> = TimeDriver::from_hz(60.0);
    let report = d.advance(ms(50));
    assert!(report.over_budget);
    assert_eq!(d.stats().latest(), Some(ms(50)));
    assert_eq!(d.stats().len(), 1);
}

#[test]
fn driver_virtual_scale_applies() {
    let mut d: TimeDriver<8> = TimeDriver::from_hz(60.0);
    d.clocks_mut().virtual_time_mut().set_relative_speed(0.5);
    let report = d.advance(ms(20));
    assert_eq!(report.real_delta, ms(20));
    assert_eq!(report.virtual_delta, ms(10));
}
