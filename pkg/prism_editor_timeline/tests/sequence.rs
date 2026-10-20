//! Real document clocks, nested events, registry ownership and provider dispatch.
use prism_editor_timeline::*;
use std::time::Duration;

const ROOT: SequenceId = SequenceId(100);
const CHILD: SequenceId = SequenceId(200);
const LEAF: SequenceId = SequenceId(300);
fn span(start: u64, end: u64) -> SequenceRange {
    SequenceRange::new(start, end).unwrap()
}
fn document(hz: u64, events: &[u64]) -> TimelineDocument {
    let mut timeline = Timeline::default();
    let mut track = Track::new(TrackId::from_raw(7), "provider.value", 1).unwrap();
    track
        .insert(Keyframe::new(
            KeyId::from_raw(1),
            0,
            vec![0.0],
            Interpolation::Linear,
        ))
        .unwrap();
    track
        .insert(Keyframe::new(
            KeyId::from_raw(2),
            100,
            vec![100.0],
            Interpolation::Step,
        ))
        .unwrap();
    timeline.add_track(track).unwrap();
    for (id, tick) in events.iter().enumerate() {
        timeline
            .add_event(TimelineEvent {
                id: id as u128 + 1,
                tick: *tick,
                name: format!("provider cue {tick}"),
            })
            .unwrap();
    }
    TimelineDocument {
        ticks_per_second: hz,
        timeline,
    }
}
fn clip(
    id: u128,
    source: SequenceId,
    parent: (u64, u64),
    local: (u64, u64),
    repeats: u64,
    n: u64,
    d: u64,
) -> SequenceClip {
    SequenceClip {
        id: SequenceClipId(id),
        source,
        parent: span(parent.0, parent.1),
        source_range: span(local.0, local.1),
        repetitions: repeats,
        rate: PlaybackRate::new(n, d).unwrap(),
    }
}
fn registry<'a>(
    root: &'a TimelineDocument,
    child: &'a TimelineDocument,
    leaf: &'a TimelineDocument,
    limits: SequenceLimits,
) -> SequenceRegistry<'a> {
    let mut registry = SequenceRegistry::new(limits).unwrap();
    registry.insert(ROOT, root).unwrap();
    registry.insert(CHILD, child).unwrap();
    registry.insert(LEAF, leaf).unwrap();
    registry
}
fn local_ticks(frame: &SequenceFrame<'_>, id: SequenceId) -> Vec<(u64, u64, u64)> {
    frame
        .events
        .iter()
        .filter(|e| e.sequence == id)
        .map(|e| {
            (
                e.root_tick,
                e.event.tick,
                e.path.last().map_or(0, |p| p.iteration),
            )
        })
        .collect()
}

#[test]
fn nested_real_samples_use_frequency_ratios_offsets_and_stable_instance_paths() {
    let (root, child, leaf) = (document(2, &[]), document(6, &[]), document(3, &[]));
    let mut registry = registry(&root, &child, &leaf, SequenceLimits::default());
    registry
        .add_clip(ROOT, clip(20, CHILD, (2, 10), (10, 16), 3, 1, 1))
        .unwrap();
    registry
        .add_clip(CHILD, clip(30, LEAF, (10, 16), (20, 26), 1, 2, 1))
        .unwrap();
    registry
        .add_clip(ROOT, clip(10, LEAF, (0, 10), (50, 60), 1, 1, 1))
        .unwrap();
    assert!(std::ptr::eq(registry.document(CHILD).unwrap(), &child));
    assert_eq!(
        registry
            .clips(ROOT)
            .unwrap()
            .map(|c| c.id.0)
            .collect::<Vec<_>>(),
        [10, 20]
    );
    let mut player = SequencePlayer::new(&registry, ROOT, span(0, 12), PlaybackMode::Once).unwrap();
    let frame = player.seek(3).unwrap();
    assert!(frame.events.is_empty());
    assert_eq!(
        frame
            .samples
            .iter()
            .map(|s| (s.sequence, s.tick))
            .collect::<Vec<_>>(),
        [(ROOT, 3), (LEAF, 54), (CHILD, 13), (LEAF, 23)]
    );
    assert_eq!(
        frame.samples[3].path,
        [
            SequenceInstance {
                clip: SequenceClipId(20),
                iteration: 0
            },
            SequenceInstance {
                clip: SequenceClipId(30),
                iteration: 0
            }
        ]
    );
    assert_eq!(frame.samples[3].tracks[&TrackId::from_raw(7)], vec![23.0]);
    assert!(std::ptr::eq(frame.samples[3].document, &leaf));
    let frame = player.seek(4).unwrap();
    assert_eq!(
        frame
            .samples
            .iter()
            .find(|s| s.sequence == CHILD)
            .unwrap()
            .tick,
        10
    );
    assert_eq!(
        frame
            .samples
            .iter()
            .find(|s| s.sequence == CHILD)
            .unwrap()
            .path[0]
            .iteration,
        1
    );
    assert!(
        !player
            .seek(8)
            .unwrap()
            .samples
            .iter()
            .any(|s| s.sequence == CHILD),
        "finite repeat ends even inside parent range"
    );
    assert_eq!(
        player.seek(12).unwrap_err(),
        SequenceError::Playback(PlaybackError::SeekOutOfBounds)
    );
    assert_eq!(player.tick(), 8);
}

#[test]
fn fast_clock_dispatches_skipped_source_events_and_orders_source_ticks_before_ids() {
    let root = document(1, &[1]);
    let mut child = document(4, &[]);
    for (id, tick) in [
        (90, 0),
        (80, 1),
        (70, 2),
        (60, 3),
        (50, 4),
        (40, 5),
        (30, 6),
        (20, 7),
    ] {
        child
            .timeline
            .add_event(TimelineEvent {
                id,
                tick,
                name: format!("real {tick}"),
            })
            .unwrap();
    }
    let mut registry = registry(&root, &child, &root, SequenceLimits::default());
    registry
        .add_clip(ROOT, clip(1, CHILD, (0, 5), (0, 8), 2, 1, 1))
        .unwrap();
    let mut player = SequencePlayer::new(&registry, ROOT, span(0, 6), PlaybackMode::Once).unwrap();
    player.play();
    let first = player.advance(Duration::ZERO).unwrap();
    assert_eq!(local_ticks(&first, CHILD), [(0, 0, 0)]);
    let next = player.advance(Duration::from_secs(1)).unwrap();
    assert_eq!(
        local_ticks(&next, CHILD),
        [(1, 1, 0), (1, 2, 0), (1, 3, 0), (1, 4, 0)]
    );
    let next = player.advance(Duration::from_secs(2)).unwrap();
    assert_eq!(
        local_ticks(&next, CHILD),
        [
            (2, 5, 0),
            (2, 6, 0),
            (2, 7, 0),
            (2, 0, 1),
            (3, 1, 1),
            (3, 2, 1),
            (3, 3, 1),
            (3, 4, 1)
        ]
    );
    assert!(std::ptr::eq(
        next.events[0].event,
        &child.timeline.events_at(5)[0]
    ));
    assert!(player.advance(Duration::ZERO).unwrap().events.is_empty());
}

#[test]
fn slow_clock_does_not_duplicate_events_and_seek_rearms_only_exact_local_destination() {
    let (root, child) = (document(4, &[]), document(2, &[10, 11, 12, 13]));
    let mut registry = registry(&root, &child, &root, SequenceLimits::default());
    registry
        .add_clip(ROOT, clip(1, CHILD, (0, 20), (10, 14), 2, 1, 1))
        .unwrap();
    let mut player = SequencePlayer::new(&registry, ROOT, span(0, 20), PlaybackMode::Once).unwrap();
    player.play();
    assert_eq!(
        local_ticks(&player.advance(Duration::ZERO).unwrap(), CHILD),
        [(0, 10, 0)]
    );
    assert!(player
        .advance(Duration::from_millis(250))
        .unwrap()
        .events
        .is_empty());
    assert_eq!(
        local_ticks(&player.advance(Duration::from_millis(250)).unwrap(), CHILD),
        [(2, 11, 0)]
    );
    let seek = player.seek(7).unwrap();
    assert!(seek.events.is_empty());
    assert_eq!(seek.samples[1].tick, 13);
    assert_eq!(
        local_ticks(&player.advance(Duration::ZERO).unwrap(), CHILD),
        [(7, 13, 0)]
    );
    assert_eq!(
        local_ticks(&player.advance(Duration::from_millis(250)).unwrap(), CHILD),
        [(8, 10, 1)]
    );
    player.pause();
    assert!(player
        .advance(Duration::from_secs(100))
        .unwrap()
        .events
        .is_empty());
    assert_eq!(player.tick(), 8);
    player.play();
    assert!(player.advance(Duration::ZERO).unwrap().events.is_empty());
}

#[test]
fn nested_skipped_intervals_repeats_and_parent_clipping_dispatch_actual_events() {
    let (root, child, leaf) = (
        document(1, &[]),
        document(4, &[0, 4, 8]),
        document(8, &[10, 11, 12, 13]),
    );
    let mut registry = registry(&root, &child, &leaf, SequenceLimits::default());
    registry
        .add_clip(ROOT, clip(1, CHILD, (0, 3), (0, 9), 1, 1, 1))
        .unwrap();
    registry
        .add_clip(CHILD, clip(2, LEAF, (1, 4), (10, 14), 2, 1, 1))
        .unwrap();
    let mut player = SequencePlayer::new(&registry, ROOT, span(0, 5), PlaybackMode::Once).unwrap();
    player.play();
    let frame = player.advance(Duration::from_secs(4)).unwrap();
    assert_eq!(
        local_ticks(&frame, CHILD),
        [(0, 0, 0), (1, 4, 0), (2, 8, 0)]
    );
    assert_eq!(
        local_ticks(&frame, LEAF),
        [(1, 10, 0), (1, 11, 0), (1, 12, 0), (1, 13, 0), (1, 10, 1)]
    );
    assert_eq!(
        frame.samples.len(),
        1,
        "children inactive at destination, crossed cues still present"
    );
    assert!(frame
        .events
        .iter()
        .filter(|e| e.sequence == LEAF)
        .all(|e| e.path.len() == 2));
}

#[test]
fn split_and_whole_elapsed_have_identical_nested_occurrences_and_fractional_clocks() {
    let (root, child, leaf) = (
        document(3, &[0, 1, 4]),
        document(5, &[2, 3, 4, 5, 6]),
        document(7, &[20, 21, 22]),
    );
    let mut registry = registry(&root, &child, &leaf, SequenceLimits::default());
    registry
        .add_clip(ROOT, clip(1, CHILD, (1, 6), (2, 7), 3, 2, 3))
        .unwrap();
    registry
        .add_clip(CHILD, clip(2, LEAF, (2, 7), (20, 23), 2, 3, 2))
        .unwrap();
    let mut whole = SequencePlayer::new(&registry, ROOT, span(0, 7), PlaybackMode::Loop).unwrap();
    let mut split = SequencePlayer::new(&registry, ROOT, span(0, 7), PlaybackMode::Loop).unwrap();
    whole.set_rate(PlaybackRate::new(5, 3).unwrap());
    split.set_rate(PlaybackRate::new(5, 3).unwrap());
    whole.play();
    split.play();
    let result = whole.advance(Duration::from_millis(9100)).unwrap();
    let mut events = Vec::new();
    for millis in [1, 99, 700, 300, 2000, 3000, 3000] {
        events.extend(split.advance(Duration::from_millis(millis)).unwrap().events);
    }
    assert_eq!(events, result.events);
    assert_eq!(split.tick(), whole.tick());
    assert_eq!(split.fractional_tick(), whole.fractional_tick());
    assert_eq!(
        split.advance(Duration::ZERO).unwrap().samples,
        result.samples
    );
}

#[test]
fn root_loop_boundaries_and_one_tick_ranges_emit_once_per_instance() {
    let root = document(1, &[4]);
    let child = document(1, &[10]);
    let mut registry = registry(&root, &child, &root, SequenceLimits::default());
    registry
        .add_clip(ROOT, clip(1, CHILD, (4, 5), (10, 11), 1, 1, 1))
        .unwrap();
    let mut player = SequencePlayer::new(&registry, ROOT, span(4, 5), PlaybackMode::Loop).unwrap();
    player.play();
    let frame = player.advance(Duration::from_secs(3)).unwrap();
    assert_eq!(frame.iteration, 3);
    assert_eq!(frame.events.len(), 8);
    assert_eq!(
        frame
            .events
            .iter()
            .map(|e| e.root_iteration)
            .collect::<Vec<_>>(),
        [0, 0, 1, 1, 2, 2, 3, 3]
    );
    assert!(player.advance(Duration::ZERO).unwrap().events.is_empty());
    let mut once = SequencePlayer::new(&registry, ROOT, span(4, 5), PlaybackMode::Once).unwrap();
    once.play();
    assert_eq!(once.advance(Duration::ZERO).unwrap().events.len(), 2);
    assert_eq!(once.status(), PlaybackStatus::Finished);
    assert!(!once.play());
    once.seek(4).unwrap();
    assert_eq!(once.status(), PlaybackStatus::Paused);
    once.play();
    assert_eq!(once.advance(Duration::ZERO).unwrap().events.len(), 2);
}

#[test]
fn cycles_missing_references_global_ids_and_long_dag_depth_are_rejected_atomically() {
    let doc = document(1, &[]);
    let mut registry = registry(
        &doc,
        &doc,
        &doc,
        SequenceLimits {
            max_depth: 3,
            ..Default::default()
        },
    );
    registry
        .add_clip(ROOT, clip(1, CHILD, (0, 10), (0, 10), 1, 1, 1))
        .unwrap();
    registry
        .add_clip(CHILD, clip(2, LEAF, (0, 10), (0, 10), 1, 1, 1))
        .unwrap();
    assert_eq!(
        registry.add_clip(LEAF, clip(3, ROOT, (0, 10), (0, 10), 1, 1, 1)),
        Err(SequenceError::Cycle)
    );
    assert_eq!(registry.clips(LEAF).unwrap().count(), 0);
    assert_eq!(
        registry.add_clip(ROOT, clip(1, LEAF, (0, 10), (0, 10), 1, 1, 1)),
        Err(SequenceError::DuplicateClip(SequenceClipId(1)))
    );
    assert_eq!(
        registry.add_clip(ROOT, clip(3, SequenceId(404), (0, 10), (0, 10), 1, 1, 1)),
        Err(SequenceError::MissingSequence(SequenceId(404)))
    );
    registry.insert(SequenceId(400), &doc).unwrap();
    assert_eq!(
        registry.add_clip(LEAF, clip(3, SequenceId(400), (0, 10), (0, 10), 1, 1, 1)),
        Err(SequenceError::Limit("depth"))
    );
    assert_eq!(
        registry.insert(ROOT, &doc),
        Err(SequenceError::DuplicateSequence(ROOT))
    );
    registry
        .add_clip(ROOT, clip(3, LEAF, (0, 10), (0, 10), 1, 1, 1))
        .unwrap();
    assert_eq!(
        registry.clips(ROOT).unwrap().count(),
        2,
        "failed inserts release clip identities"
    );
}

#[test]
fn registry_and_codec_input_limits_reject_before_retaining_topology() {
    assert!(matches!(
        SequenceRegistry::new(SequenceLimits {
            max_depth: 65,
            ..Default::default()
        }),
        Err(SequenceError::InvalidLimits)
    ));
    assert!(matches!(
        SequenceRegistry::new(SequenceLimits {
            max_visits: 0,
            ..Default::default()
        }),
        Err(SequenceError::InvalidLimits)
    ));
    assert_eq!(SequenceRange::new(1, 1), Err(SequenceError::InvalidRange));
    assert_eq!(SequenceRange::new(2, 1), Err(SequenceError::InvalidRange));
    let doc = document(1, &[]);
    let invalid = document(0, &[]);
    let mut small = SequenceRegistry::new(SequenceLimits {
        max_sequences: 1,
        max_clips: 0,
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(
        small.insert(ROOT, &invalid),
        Err(SequenceError::Timeline(_))
    ));
    small.insert(ROOT, &doc).unwrap();
    assert_eq!(
        small.insert(CHILD, &doc),
        Err(SequenceError::Limit("sequences"))
    );
    assert_eq!(
        small.add_clip(ROOT, clip(1, ROOT, (0, 1), (0, 1), 1, 1, 1)),
        Err(SequenceError::Limit("clips"))
    );
    let mut bytes = SequenceRegistry::new(SequenceLimits {
        max_document_bytes: 1,
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(
        bytes.insert(ROOT, &doc),
        Err(SequenceError::Timeline(_))
    ));
    let mut registry = registry(&doc, &doc, &doc, SequenceLimits::default());
    assert_eq!(
        registry.add_clip(ROOT, clip(1, CHILD, (0, 1), (0, 1), 0, 1, 1)),
        Err(SequenceError::InvalidRepeat)
    );
}

#[test]
fn event_and_visit_limits_fail_atomically_without_consuming_elapsed_or_origin() {
    let (root, child) = (document(1, &[]), document(4, &[0, 1, 2, 3, 4, 5]));
    let mut registry = registry(
        &root,
        &child,
        &root,
        SequenceLimits {
            max_events: 3,
            ..Default::default()
        },
    );
    registry
        .add_clip(ROOT, clip(1, CHILD, (0, 10), (0, 6), 3, 1, 1))
        .unwrap();
    let mut player = SequencePlayer::new(&registry, ROOT, span(0, 10), PlaybackMode::Once).unwrap();
    player.play();
    assert_eq!(
        player.advance(Duration::from_secs(1)).unwrap_err(),
        SequenceError::Limit("events")
    );
    assert_eq!(player.tick(), 0);
    assert_eq!(player.fractional_tick(), (0, 1_000_000_000));
    assert_eq!(
        player.advance(Duration::ZERO).unwrap().events.len(),
        1,
        "failed advance retains origin"
    );
    assert!(player.advance(Duration::ZERO).unwrap().events.is_empty());
    let empty = document(1, &[]);
    let mut registry = self::registry(
        &empty,
        &empty,
        &empty,
        SequenceLimits {
            max_visits: 20,
            ..Default::default()
        },
    );
    registry
        .add_clip(
            ROOT,
            clip(1, CHILD, (0, u64::MAX), (0, 1), u64::MAX, u64::MAX, 1),
        )
        .unwrap();
    let mut player = SequencePlayer::new(&registry, ROOT, span(0, 10), PlaybackMode::Once).unwrap();
    player.play();
    assert_eq!(
        player.advance(Duration::from_secs(1)).unwrap_err(),
        SequenceError::Limit("visits")
    );
    assert_eq!(player.tick(), 0);
}

#[test]
fn sample_budget_and_real_cubic_failure_preserve_seek_and_play_state() {
    let doc = document(1, &[]);
    let mut registry = registry(
        &doc,
        &doc,
        &doc,
        SequenceLimits {
            max_samples: 1,
            ..Default::default()
        },
    );
    registry
        .add_clip(ROOT, clip(1, CHILD, (1, 3), (0, 2), 1, 1, 1))
        .unwrap();
    let mut player = SequencePlayer::new(&registry, ROOT, span(0, 4), PlaybackMode::Once).unwrap();
    assert_eq!(player.seek(1).unwrap_err(), SequenceError::Limit("samples"));
    assert_eq!(player.tick(), 0);
    player.play();
    assert_eq!(
        player.advance(Duration::from_secs(1)).unwrap_err(),
        SequenceError::Limit("samples")
    );
    assert_eq!(player.tick(), 0);
    let mut bad = TimelineDocument {
        ticks_per_second: 1,
        timeline: Timeline::default(),
    };
    let mut track = Track::new(TrackId::from_raw(1), "overflow", 1).unwrap();
    let mut left = Keyframe::new(KeyId::from_raw(1), 0, vec![0.0], Interpolation::Cubic);
    left.set_tangents(vec![0.0], vec![f64::MAX]).unwrap();
    let mut right = Keyframe::new(KeyId::from_raw(2), 100, vec![0.0], Interpolation::Cubic);
    right.set_tangents(vec![-f64::MAX], vec![0.0]).unwrap();
    track.insert(left).unwrap();
    track.insert(right).unwrap();
    bad.timeline.add_track(track).unwrap();
    let mut registry = self::registry(&doc, &bad, &doc, SequenceLimits::default());
    registry
        .add_clip(ROOT, clip(1, CHILD, (0, 101), (0, 101), 1, 1, 1))
        .unwrap();
    let mut player =
        SequencePlayer::new(&registry, ROOT, span(0, 101), PlaybackMode::Once).unwrap();
    assert_eq!(
        player.seek(50).unwrap_err(),
        SequenceError::Timeline(TimelineError::SampleOverflow)
    );
    assert_eq!(player.tick(), 0);
}

#[test]
fn exact_wide_rational_math_avoids_false_product_overflow_and_rejects_true_overflow() {
    let root = document(u64::MAX, &[]);
    let child = document(u64::MAX - 1, &[]);
    let mut registry = registry(&root, &child, &root, SequenceLimits::default());
    registry
        .add_clip(
            ROOT,
            clip(
                1,
                CHILD,
                (0, u64::MAX),
                (0, u64::MAX),
                1,
                u64::MAX - 2,
                u64::MAX - 3,
            ),
        )
        .unwrap();
    let mut player =
        SequencePlayer::new(&registry, ROOT, span(0, u64::MAX), PlaybackMode::Once).unwrap();
    let frame = player.seek(u64::MAX - 2).unwrap();
    // ((M-2)*(M-1)*(M-2))/(M*(M-3)) = M-2 plus a subunit fraction.
    assert_eq!(frame.samples[1].tick, u64::MAX - 2);
    let slow = document(1, &[]);
    let fast = document(u64::MAX, &[]);
    let mut registry = self::registry(&slow, &fast, &slow, SequenceLimits::default());
    assert_eq!(
        registry.add_clip(ROOT, clip(1, CHILD, (0, u64::MAX), (0, 1), 1, u64::MAX, 1)),
        Err(SequenceError::ArithmeticOverflow)
    );
    assert_eq!(registry.clips(ROOT).unwrap().count(), 0);
    let mut registry = SequenceRegistry::new(SequenceLimits::default()).unwrap();
    registry.insert(ROOT, &fast).unwrap();
    let mut player = SequencePlayer::new(&registry, ROOT, span(0, 10), PlaybackMode::Loop).unwrap();
    player.set_rate(PlaybackRate::new(u64::MAX, 1).unwrap());
    player.play();
    assert!(matches!(
        player.advance(Duration::MAX),
        Err(SequenceError::Playback(PlaybackError::ArithmeticOverflow))
    ));
    assert_eq!(player.tick(), 0);
}

#[derive(Default)]
struct Provider {
    samples: Vec<(SequenceId, u64, f64)>,
    events: Vec<(SequenceId, String, u64)>,
}
impl SequenceProvider for Provider {
    type Error = &'static str;
    fn sample(&mut self, sample: &SequenceSample<'_>) -> Result<(), Self::Error> {
        self.samples.push((
            sample.sequence,
            sample.tick,
            sample.tracks[&TrackId::from_raw(7)][0],
        ));
        Ok(())
    }
    fn event(&mut self, event: &SequenceEvent<'_>) -> Result<(), Self::Error> {
        self.events
            .push((event.sequence, event.event.name.clone(), event.root_tick));
        Ok(())
    }
}
#[test]
fn provider_receives_real_channels_and_authored_payloads_only_on_explicit_dispatch() {
    let (root, child) = (document(1, &[0]), document(2, &[0, 1, 2]));
    let mut registry = registry(&root, &child, &root, SequenceLimits::default());
    registry
        .add_clip(ROOT, clip(1, CHILD, (0, 4), (0, 4), 1, 1, 1))
        .unwrap();
    let mut player = SequencePlayer::new(&registry, ROOT, span(0, 4), PlaybackMode::Once).unwrap();
    player.play();
    let frame = player.advance(Duration::from_secs(1)).unwrap();
    let mut provider = Provider::default();
    assert!(provider.events.is_empty());
    frame.dispatch(&mut provider).unwrap();
    assert_eq!(provider.samples, [(ROOT, 1, 1.0), (CHILD, 2, 2.0)]);
    assert_eq!(
        provider.events,
        [
            (ROOT, "provider cue 0".into(), 0),
            (CHILD, "provider cue 0".into(), 0),
            (CHILD, "provider cue 1".into(), 1),
            (CHILD, "provider cue 2".into(), 1)
        ]
    );
    struct Reject;
    impl SequenceProvider for Reject {
        type Error = &'static str;
        fn sample(&mut self, _: &SequenceSample<'_>) -> Result<(), Self::Error> {
            Err("provider rejected")
        }
        fn event(&mut self, _: &SequenceEvent<'_>) -> Result<(), Self::Error> {
            panic!("dispatch must stop")
        }
    }
    assert_eq!(frame.dispatch(&mut Reject), Err("provider rejected"));
}

#[test]
fn nested_event_schedule_matches_independent_small_integer_occurrence_enumeration() {
    // This oracle enumerates each authored occurrence and projects it outward;
    // production traverses inbound ranges and never enumerates parent ticks.
    for root_hz in [1u64, 3, 5] {
        for child_hz in [1u64, 2, 7] {
            for (speed_n, speed_d) in [(1u64, 1u64), (3, 2), (2, 5)] {
                let root = document(root_hz, &[]);
                let child = document(child_hz, &[1, 2, 3, 4, 5, 6]);
                let leaf = document(5, &[10, 11, 12, 13]);
                let mut registry = registry(&root, &child, &leaf, SequenceLimits::default());
                registry
                    .add_clip(ROOT, clip(1, CHILD, (2, 16), (1, 7), 3, speed_n, speed_d))
                    .unwrap();
                registry
                    .add_clip(CHILD, clip(2, LEAF, (2, 6), (10, 14), 2, 2, 3))
                    .unwrap();
                let mut player =
                    SequencePlayer::new(&registry, ROOT, span(0, 18), PlaybackMode::Once).unwrap();
                player.play();
                let frame = player.advance(Duration::from_secs(18)).unwrap();
                let actual: Vec<_> = frame
                    .events
                    .iter()
                    .map(|e| {
                        (
                            e.root_tick,
                            e.path.clone(),
                            e.sequence,
                            e.event.tick,
                            e.event.id,
                        )
                    })
                    .collect();
                let root_projection = |tick: u64, repeat: u64| {
                    2 + ((repeat * 6 + tick - 1) * root_hz * speed_d).div_ceil(child_hz * speed_n)
                };
                let mut expected = Vec::new();
                for parent_repeat in 0..3 {
                    for event in child.timeline.events_in(1..=6) {
                        let tick = root_projection(event.tick, parent_repeat);
                        if tick < 16 {
                            expected.push((
                                tick,
                                vec![SequenceInstance {
                                    clip: SequenceClipId(1),
                                    iteration: parent_repeat,
                                }],
                                CHILD,
                                event.tick,
                                event.id,
                            ));
                        }
                    }
                    for leaf_repeat in 0..2 {
                        for event in leaf.timeline.events_in(10..=13) {
                            let parent_tick = 2
                                + ((leaf_repeat * 4 + event.tick - 10) * child_hz * 3)
                                    .div_ceil(5 * 2);
                            if parent_tick >= 6 {
                                continue;
                            }
                            let tick = root_projection(parent_tick, parent_repeat);
                            if tick < 16 {
                                expected.push((
                                    tick,
                                    vec![
                                        SequenceInstance {
                                            clip: SequenceClipId(1),
                                            iteration: parent_repeat,
                                        },
                                        SequenceInstance {
                                            clip: SequenceClipId(2),
                                            iteration: leaf_repeat,
                                        },
                                    ],
                                    LEAF,
                                    event.tick,
                                    event.id,
                                ));
                            }
                        }
                    }
                }
                expected.sort_by(|a, b| (&a.0, &a.1, &a.3, &a.4).cmp(&(&b.0, &b.1, &b.3, &b.4)));
                assert_eq!(
                    actual, expected,
                    "Hz {root_hz}/{child_hz}, speed {speed_n}/{speed_d}"
                );
            }
        }
    }
}
