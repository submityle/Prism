//! Runtime playback contracts against real timeline samples and event queries.

use std::time::Duration;

use prism_editor_timeline::{
    Interpolation, KeyId, Keyframe, Playback, PlaybackConfig, PlaybackError, PlaybackFrame,
    PlaybackMode, PlaybackRate, PlaybackStatus, Timeline, TimelineError, TimelineEvent, Track,
    TrackId,
};

fn config(start: u64, end: u64, mode: PlaybackMode, budget: usize) -> PlaybackConfig {
    PlaybackConfig {
        ticks_per_second: 10,
        start_tick: start,
        end_tick: end,
        mode,
        max_events: budget,
    }
}

fn event(timeline: &mut Timeline, id: u128, tick: u64) {
    timeline
        .add_event(TimelineEvent {
            id,
            tick,
            name: format!("event {id}"),
        })
        .unwrap();
}

fn occurrences(frame: &PlaybackFrame<'_>) -> Vec<(u128, u64, u128)> {
    frame
        .events
        .iter()
        .map(|item| (item.event.id, item.event.tick, item.iteration))
        .collect()
}

fn numeric_timeline() -> Timeline {
    let mut timeline = Timeline::default();
    let mut track = Track::new(TrackId::from_raw(10), "generic channels", 2).unwrap();
    track
        .insert(Keyframe::new(
            KeyId::from_raw(1),
            0,
            vec![0.0, 10.0],
            Interpolation::Linear,
        ))
        .unwrap();
    track
        .insert(Keyframe::new(
            KeyId::from_raw(2),
            10,
            vec![20.0, 30.0],
            Interpolation::Linear,
        ))
        .unwrap();
    timeline.add_track(track).unwrap();
    let mut late = Track::new(TrackId::from_raw(20), "later", 1).unwrap();
    late.insert(Keyframe::new(
        KeyId::from_raw(3),
        8,
        vec![42.0],
        Interpolation::Step,
    ))
    .unwrap();
    timeline.add_track(late).unwrap();
    timeline
}

#[test]
fn real_samples_and_borrowed_events_reach_the_consumer() {
    let mut timeline = numeric_timeline();
    event(&mut timeline, 4, 4);
    event(&mut timeline, 0, 0);
    let mut player = Playback::new(config(0, 10, PlaybackMode::Once, 8)).unwrap();
    assert!(player.play());
    let frame = player
        .advance(&timeline, Duration::from_millis(500))
        .unwrap();
    assert_eq!(frame.tick, 5);
    assert_eq!(frame.samples, timeline.sample(5).unwrap());
    assert_eq!(frame.samples[&TrackId::from_raw(10)], vec![10.0, 20.0]);
    assert!(!frame.samples.contains_key(&TrackId::from_raw(20)));
    assert_eq!(occurrences(&frame), vec![(0, 0, 0), (4, 4, 0)]);
    assert!(std::ptr::eq(
        frame.events[1].event,
        &timeline.events_at(4)[0]
    ));
    let held = player.advance(&timeline, Duration::from_secs(99)).unwrap();
    assert_eq!(held.samples[&TrackId::from_raw(20)], vec![42.0]);
    assert_eq!(held.status, PlaybackStatus::Finished);
}

#[test]
fn duration_and_rational_remainders_are_partition_invariant() {
    let timeline = Timeline::default();
    let mut cfg = config(0, u64::MAX, PlaybackMode::Loop, 0);
    cfg.ticks_per_second = 60;
    let mut split = Playback::new(cfg).unwrap();
    split.set_rate(PlaybackRate::new(3, 2).unwrap());
    split.play();
    let mut whole = split.clone();
    for _ in 0..60 {
        split
            .advance(&timeline, Duration::from_nanos(16_666_667))
            .unwrap();
    }
    whole
        .advance(&timeline, Duration::from_nanos(1_000_000_020))
        .unwrap();
    assert_eq!(split, whole);
    assert_eq!(split.tick(), 90);
    assert_eq!(split.fractional_tick(), (3600, 2_000_000_000));
}

#[test]
fn exact_time_conversion_matches_integer_reference_matrix() {
    let timeline = Timeline::default();
    for frequency in [1, 24, 48_000, u64::MAX] {
        for (numerator, denominator) in [(1, 1), (3, 2), (1, u64::MAX), (u64::MAX, u64::MAX - 1)] {
            let mut cfg = config(0, u64::MAX, PlaybackMode::Loop, 0);
            cfg.ticks_per_second = frequency;
            let mut player = Playback::new(cfg).unwrap();
            player.set_rate(PlaybackRate::new(numerator, denominator).unwrap());
            player.play();
            let (n, d) = player.rate().ratio();
            let divisor = 1_000_000_000 * d as u128;
            let mut expected_remainder = 0;
            let mut ticks = 0;
            for nanos in [0, 1, 7, 999, 12_345] {
                // These small cases fit a direct u128 reference product.
                if let Some(product) = (nanos as u128)
                    .checked_mul(frequency as u128)
                    .and_then(|value| value.checked_mul(n as u128))
                    .and_then(|value| value.checked_add(expected_remainder))
                {
                    ticks += product / divisor;
                    expected_remainder = product % divisor;
                    player
                        .advance(&timeline, Duration::from_nanos(nanos))
                        .unwrap();
                    assert_eq!(player.tick(), (ticks % (1_u128 << 64)) as u64);
                    assert_eq!(player.iteration(), ticks / (1_u128 << 64));
                    assert_eq!(player.fractional_tick(), (expected_remainder, divisor));
                }
            }
        }
    }
}

#[test]
fn paused_time_is_ignored_and_resuming_preserves_fraction_and_origin_cursor() {
    let mut timeline = Timeline::default();
    event(&mut timeline, 1, 0);
    let mut player = Playback::new(config(0, 20, PlaybackMode::Loop, 1)).unwrap();
    assert!(player
        .advance(&timeline, Duration::MAX)
        .unwrap()
        .events
        .is_empty());
    player.play();
    assert_eq!(
        player
            .advance(&timeline, Duration::from_millis(50))
            .unwrap()
            .events
            .len(),
        1
    );
    player.pause();
    let paused = player.clone();
    assert!(player
        .advance(&timeline, Duration::MAX)
        .unwrap()
        .events
        .is_empty());
    assert_eq!(player, paused);
    player.play();
    let frame = player
        .advance(&timeline, Duration::from_millis(50))
        .unwrap();
    assert_eq!(frame.tick, 1);
    assert!(frame.events.is_empty());
    assert_eq!(player.fractional_tick().0, 0);
}

#[test]
fn seek_samples_without_crossed_events_then_rearms_destination_once() {
    let mut timeline = numeric_timeline();
    event(&mut timeline, 2, 2);
    event(&mut timeline, 4, 4);
    let mut player = Playback::new(config(0, 10, PlaybackMode::Loop, 4)).unwrap();
    player.play();
    player
        .advance(&timeline, Duration::from_millis(125))
        .unwrap();
    let frame = player.seek(&timeline, 4).unwrap();
    assert_eq!(frame.samples, timeline.sample(4).unwrap());
    assert!(frame.events.is_empty());
    assert_eq!(frame.status, PlaybackStatus::Playing);
    assert_eq!(player.fractional_tick().0, 0);
    assert_eq!(
        occurrences(&player.advance(&timeline, Duration::ZERO).unwrap()),
        vec![(4, 4, 0)]
    );
    assert!(player
        .advance(&timeline, Duration::ZERO)
        .unwrap()
        .events
        .is_empty());
    let before = player.clone();
    assert_eq!(
        player.seek(&timeline, 11),
        Err(PlaybackError::SeekOutOfBounds)
    );
    assert_eq!(player, before);
    player.advance(&timeline, Duration::from_secs(2)).unwrap();
    assert!(player.iteration() > 0);
    player.seek(&timeline, 0).unwrap();
    assert_eq!(player.iteration(), 0);
}

#[test]
fn closed_loop_emits_end_then_start_in_tick_and_id_order() {
    let mut timeline = Timeline::default();
    for (id, tick) in [(99, 4), (8, 5), (2, 5), (6, 6), (7, 7), (100, 8)] {
        event(&mut timeline, id, tick);
    }
    let mut player = Playback::new(config(5, 7, PlaybackMode::Loop, 20)).unwrap();
    player.play();
    let frame = player
        .advance(&timeline, Duration::from_millis(700))
        .unwrap();
    assert_eq!((frame.tick, frame.iteration), (6, 2));
    assert_eq!(
        occurrences(&frame),
        vec![
            (2, 5, 0),
            (8, 5, 0),
            (6, 6, 0),
            (7, 7, 0),
            (2, 5, 1),
            (8, 5, 1),
            (6, 6, 1),
            (7, 7, 1),
            (2, 5, 2),
            (8, 5, 2),
            (6, 6, 2),
        ]
    );
    assert!(player
        .advance(&timeline, Duration::ZERO)
        .unwrap()
        .events
        .is_empty());
    assert_eq!(
        occurrences(
            &player
                .advance(&timeline, Duration::from_millis(200))
                .unwrap()
        ),
        vec![(7, 7, 2), (2, 5, 3), (8, 5, 3)]
    );
}

#[test]
fn loops_and_step_partitions_match_a_tick_by_tick_event_oracle() {
    for length in 1..=5_u64 {
        for seek in 9..9 + length {
            for steps in 0..=25_u64 {
                let mut timeline = Timeline::default();
                for tick in 9..9 + length {
                    event(&mut timeline, tick as u128, tick);
                }
                let cfg = config(9, 8 + length, PlaybackMode::Loop, 64);
                let mut big = Playback::new(cfg).unwrap();
                big.seek(&timeline, seek).unwrap();
                big.play();
                let mut small = big.clone();
                let actual = occurrences(
                    &big.advance(&timeline, Duration::from_millis(steps * 100))
                        .unwrap(),
                );
                let mut expected = vec![(seek as u128, seek, 0)];
                let mut tick = seek;
                let mut iteration = 0;
                for _ in 0..steps {
                    if tick == 8 + length {
                        tick = 9;
                        iteration += 1;
                    } else {
                        tick += 1;
                    }
                    expected.push((tick as u128, tick, iteration));
                }
                assert_eq!(actual, expected);
                let mut split = occurrences(&small.advance(&timeline, Duration::ZERO).unwrap());
                for _ in 0..steps {
                    split.extend(occurrences(
                        &small
                            .advance(&timeline, Duration::from_millis(100))
                            .unwrap(),
                    ));
                }
                assert_eq!(split, actual);
                assert_eq!(small, big);
            }
        }
    }
}

#[test]
fn budget_failure_preserves_full_state_and_smaller_retries_lose_no_events() {
    let mut timeline = Timeline::default();
    for tick in 0..3 {
        event(&mut timeline, tick as u128, tick);
    }
    let cfg = config(0, 2, PlaybackMode::Loop, 3);
    let mut player = Playback::new(cfg).unwrap();
    player.play();
    player
        .advance(&timeline, Duration::from_millis(25))
        .unwrap();
    let before = player.clone();
    assert_eq!(
        player.advance(&timeline, Duration::from_millis(500)),
        Err(PlaybackError::EventBudgetExceeded { limit: 3 })
    );
    assert_eq!(player, before);
    let mut output = occurrences(
        &player
            .advance(&timeline, Duration::from_millis(200))
            .unwrap(),
    );
    output.extend(occurrences(
        &player
            .advance(&timeline, Duration::from_millis(300))
            .unwrap(),
    ));
    assert_eq!(
        output,
        vec![(1, 1, 0), (2, 2, 0), (0, 0, 1), (1, 1, 1), (2, 2, 1)]
    );
    assert_eq!(player.fractional_tick().0, 250_000_000);
}

#[test]
fn origin_and_multiwrap_budget_failures_do_not_consume_pending_events() {
    let mut timeline = Timeline::default();
    event(&mut timeline, 1, 0);
    let mut player = Playback::new(config(0, 1, PlaybackMode::Loop, 0)).unwrap();
    player.play();
    let before = player.clone();
    assert_eq!(
        player.advance(&timeline, Duration::ZERO),
        Err(PlaybackError::EventBudgetExceeded { limit: 0 })
    );
    assert_eq!(player, before);
    let mut player = Playback::new(config(0, 1, PlaybackMode::Loop, 2)).unwrap();
    player.play();
    let before = player.clone();
    assert_eq!(
        player.advance(&timeline, Duration::from_secs(u64::MAX)),
        Err(PlaybackError::EventBudgetExceeded { limit: 2 })
    );
    assert_eq!(player, before);
    assert_eq!(
        occurrences(&player.advance(&timeline, Duration::ZERO).unwrap()),
        vec![(1, 0, 0)]
    );
}

#[test]
fn once_delivers_terminal_event_once_and_seek_restarts_finished_player() {
    let mut timeline = Timeline::default();
    event(&mut timeline, 0, 0);
    event(&mut timeline, 1, 1);
    let mut player = Playback::new(config(0, 1, PlaybackMode::Once, 2)).unwrap();
    player.play();
    let frame = player.advance(&timeline, Duration::MAX).unwrap();
    assert_eq!(occurrences(&frame), vec![(0, 0, 0), (1, 1, 0)]);
    assert_eq!(frame.status, PlaybackStatus::Finished);
    assert!(!player.play());
    player.pause();
    assert_eq!(player.status(), PlaybackStatus::Finished);
    assert!(player
        .advance(&timeline, Duration::MAX)
        .unwrap()
        .events
        .is_empty());
    assert_eq!(
        player.seek(&timeline, 1).unwrap().status,
        PlaybackStatus::Paused
    );
    player.play();
    assert_eq!(
        occurrences(&player.advance(&timeline, Duration::ZERO).unwrap()),
        vec![(1, 1, 0)]
    );
    assert_eq!(player.status(), PlaybackStatus::Finished);
}

#[test]
fn single_tick_loops_have_one_tick_period_and_single_tick_once_finishes() {
    let mut timeline = Timeline::default();
    event(&mut timeline, 5, 5);
    let mut player = Playback::new(config(5, 5, PlaybackMode::Loop, 4)).unwrap();
    player.play();
    assert_eq!(
        occurrences(
            &player
                .advance(&timeline, Duration::from_millis(300))
                .unwrap()
        ),
        vec![(5, 5, 0), (5, 5, 1), (5, 5, 2), (5, 5, 3)]
    );
    assert_eq!(player.iteration(), 3);
    let mut once = Playback::new(config(5, 5, PlaybackMode::Once, 1)).unwrap();
    once.play();
    let frame = once.advance(&timeline, Duration::ZERO).unwrap();
    assert_eq!(frame.status, PlaybackStatus::Finished);
    assert_eq!(frame.events.len(), 1);
}

#[test]
fn full_u64_closed_interval_and_maximum_endpoints_are_representable() {
    let mut timeline = Timeline::default();
    event(&mut timeline, 0, 0);
    event(&mut timeline, 1, u64::MAX);
    let mut cfg = config(0, u64::MAX, PlaybackMode::Loop, 3);
    cfg.ticks_per_second = 1;
    let mut player = Playback::new(cfg).unwrap();
    player.play();
    assert_eq!(
        occurrences(
            &player
                .advance(&timeline, Duration::from_secs(u64::MAX))
                .unwrap()
        ),
        vec![(0, 0, 0), (1, u64::MAX, 0)]
    );
    assert_eq!(
        occurrences(&player.advance(&timeline, Duration::from_secs(1)).unwrap()),
        vec![(0, 0, 1)]
    );
    assert_eq!((player.tick(), player.iteration()), (0, 1));
    let mut edge = Playback::new(config(u64::MAX, u64::MAX, PlaybackMode::Loop, 2)).unwrap();
    edge.play();
    assert_eq!(
        occurrences(&edge.advance(&timeline, Duration::from_millis(100)).unwrap()),
        vec![(1, u64::MAX, 0), (1, u64::MAX, 1)]
    );
}

#[test]
fn huge_empty_loops_skip_ticks_and_avoid_intermediate_nanosecond_overflow() {
    let timeline = Timeline::default();
    let mut cfg = config(7, 7, PlaybackMode::Loop, 0);
    cfg.ticks_per_second = u64::MAX;
    let mut player = Playback::new(cfg).unwrap();
    player.play();
    player
        .advance(&timeline, Duration::from_secs(u64::MAX))
        .unwrap();
    assert_eq!(player.iteration(), (u64::MAX as u128) * (u64::MAX as u128));
    assert_eq!(player.tick(), 7);
    let before = player.clone();
    assert_eq!(
        player.advance(&timeline, Duration::from_secs(3)),
        Err(PlaybackError::ArithmeticOverflow)
    );
    assert_eq!(player, before);
}

#[test]
fn wide_rate_product_preserves_exact_quotient_and_fraction() {
    let timeline = Timeline::default();
    let mut cfg = config(0, 0, PlaybackMode::Loop, 0);
    cfg.ticks_per_second = u64::MAX;
    let mut player = Playback::new(cfg).unwrap();
    player.set_rate(PlaybackRate::new(u64::MAX, u64::MAX - 1).unwrap());
    player.play();
    player
        .advance(&timeline, Duration::from_secs(u64::MAX))
        .unwrap();
    let t = u64::MAX as u128;
    // t^3 / (t - 1) = t^2 + t + 1, with remainder 1.
    assert_eq!(player.iteration(), t * t + t + 1);
    assert_eq!(
        player.fractional_tick(),
        (1_000_000_000, 1_000_000_000 * (t - 1))
    );
}

#[test]
fn true_time_quotient_overflow_rolls_back_before_origin_delivery() {
    let timeline = Timeline::default();
    let mut cfg = config(0, u64::MAX, PlaybackMode::Loop, 0);
    cfg.ticks_per_second = u64::MAX;
    let mut player = Playback::new(cfg).unwrap();
    player.set_rate(PlaybackRate::new(u64::MAX, 1).unwrap());
    player.play();
    let before = player.clone();
    assert_eq!(
        player.advance(&timeline, Duration::MAX),
        Err(PlaybackError::ArithmeticOverflow)
    );
    assert_eq!(player, before);
}

#[test]
fn sample_failure_rolls_back_time_events_status_and_seek() {
    let mut timeline = Timeline::default();
    let mut track = Track::new(TrackId::from_raw(1), "overflow", 1).unwrap();
    let mut left = Keyframe::new(KeyId::from_raw(1), 0, vec![0.0], Interpolation::Cubic);
    left.set_tangents(vec![0.0], vec![f64::MAX]).unwrap();
    track.insert(left).unwrap();
    track
        .insert(Keyframe::new(
            KeyId::from_raw(2),
            10,
            vec![1.0],
            Interpolation::Linear,
        ))
        .unwrap();
    timeline.add_track(track).unwrap();
    event(&mut timeline, 1, 0);
    let mut player = Playback::new(config(0, 10, PlaybackMode::Once, 1)).unwrap();
    player.play();
    let before = player.clone();
    assert_eq!(
        player.advance(&timeline, Duration::from_millis(325)),
        Err(PlaybackError::Sample(TimelineError::SampleOverflow))
    );
    assert_eq!(player, before);
    assert_eq!(
        player.seek(&timeline, 3),
        Err(PlaybackError::Sample(TimelineError::SampleOverflow))
    );
    assert_eq!(player, before);
    timeline.remove_track(TrackId::from_raw(1)).unwrap();
    assert_eq!(
        occurrences(
            &player
                .advance(&timeline, Duration::from_millis(325))
                .unwrap()
        ),
        vec![(1, 0, 0)]
    );
    assert_eq!(player.tick(), 3);
    assert_eq!(player.fractional_tick().0, 250_000_000);
}

#[test]
fn validation_and_rate_change_reset_contracts() {
    assert_eq!(
        Playback::new(config(2, 1, PlaybackMode::Loop, 0)),
        Err(PlaybackError::InvalidConfig)
    );
    let mut cfg = config(0, 9, PlaybackMode::Loop, 0);
    cfg.ticks_per_second = 0;
    assert_eq!(Playback::new(cfg), Err(PlaybackError::InvalidConfig));
    assert_eq!(PlaybackRate::new(0, 1), Err(PlaybackError::InvalidRate));
    assert_eq!(PlaybackRate::new(1, 0), Err(PlaybackError::InvalidRate));
    assert_eq!(PlaybackRate::new(6, 4).unwrap().ratio(), (3, 2));
    let mut player = Playback::new(config(0, 9, PlaybackMode::Loop, 0)).unwrap();
    assert_eq!(player.config().max_events, 0);
    player.play();
    let timeline = Timeline::default();
    player
        .advance(&timeline, Duration::from_millis(25))
        .unwrap();
    let before = player.clone();
    player.set_rate(PlaybackRate::new(2, 2).unwrap());
    assert_eq!(player, before);
    player.set_rate(PlaybackRate::new(2, 1).unwrap());
    assert_eq!(player.fractional_tick().0, 0);
    player
        .advance(&timeline, Duration::from_millis(50))
        .unwrap();
    assert_eq!(player.tick(), 1);
}
