//! Authoring transactions, bounded inverse history and numeric curve regressions.
use prism_editor_timeline::*;

fn key(id: u128, tick: u64, value: f64) -> Keyframe {
    Keyframe::new(
        KeyId::from_raw(id),
        tick,
        vec![value],
        Interpolation::Linear,
    )
}
fn track(id: u128, keys: &[(u128, u64, f64)]) -> Track {
    let mut track = Track::new(TrackId::from_raw(id), format!("Track {id}"), 1).unwrap();
    for &(id, tick, value) in keys {
        track.insert(key(id, tick, value)).unwrap();
    }
    track
}
fn make_editor(limits: TimelineLimits) -> TimelineEditor {
    let mut timeline = Timeline::default();
    timeline
        .add_track(track(1, &[(1, 0, 0.0), (2, 10, 1.0)]))
        .unwrap();
    TimelineEditor::new(timeline, limits).unwrap()
}
fn put(id: u128, tick: u64, value: f64) -> TimelineEdit {
    TimelineEdit::PutKey {
        track: TrackId::from_raw(1),
        key: key(id, tick, value),
    }
}

#[test]
fn finite_validation_and_failed_replacement_preserve_both_indexes() {
    let mut track = track(1, &[(1, 0, 0.0), (2, 10, 1.0)]);
    let before = track.clone();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            track.replace(key(1, 3, value)),
            Err(TimelineError::NonFiniteValue)
        );
        assert_eq!(
            track.insert(key(3, 5, value)),
            Err(TimelineError::NonFiniteValue)
        );
        assert_eq!(track, before);
    }
    assert_eq!(
        track.replace(key(1, 10, 2.0)),
        Err(TimelineError::DuplicateTick(10))
    );
    assert_eq!(track, before);
    let old = track.replace(key(1, u64::MAX, 3.0)).unwrap();
    assert_eq!(old.tick(), 0);
    assert_eq!(track.key(KeyId::from_raw(1)).unwrap().tick(), u64::MAX);
    assert_eq!(
        track.keyframes().map(Keyframe::tick).collect::<Vec<_>>(),
        [10, u64::MAX]
    );
}

#[test]
fn cubic_tangents_are_per_tick_and_overflow_is_explicit() {
    let mut track = Track::new(TrackId::from_raw(1), "curve", 1).unwrap();
    let mut left = Keyframe::new(KeyId::from_raw(1), 0, vec![0.0], Interpolation::Cubic);
    left.set_tangents(vec![0.0], vec![0.2]).unwrap();
    let mut right = key(2, 10, 1.0);
    right.set_tangents(vec![0.0], vec![0.0]).unwrap();
    track.insert(left).unwrap();
    track.insert(right).unwrap();
    assert_eq!(track.sample(0), Ok(Some(vec![0.0])));
    assert_eq!(track.sample(5), Ok(Some(vec![0.75])));
    assert_eq!(track.sample(10), Ok(Some(vec![1.0])));
    assert_eq!(track.sample(11), Ok(Some(vec![1.0])));
    let mut left = track.key(KeyId::from_raw(1)).unwrap().clone();
    let prior = left.clone();
    assert_eq!(
        left.set_tangents(vec![f64::NAN], vec![0.0]),
        Err(TimelineError::NonFiniteValue)
    );
    assert_eq!(left, prior);
    left.set_tangents(vec![0.0], vec![f64::MAX]).unwrap();
    track.replace(left).unwrap();
    assert_eq!(track.sample(3), Err(TimelineError::SampleOverflow));
    let mut extreme = Track::new(TrackId::from_raw(9), "finite linear", 1).unwrap();
    extreme.insert(key(1, 0, -f64::MAX)).unwrap();
    extreme.insert(key(2, 10, f64::MAX)).unwrap();
    assert_eq!(extreme.sample(5), Ok(Some(vec![0.0])));
}

#[test]
fn transaction_failure_restores_data_history_revision_and_exact_token() {
    let mut editor = make_editor(TimelineLimits::default());
    editor.execute("value", vec![put(1, 0, 2.0)]).unwrap();
    editor.undo().unwrap();
    let before = editor.timeline().clone();
    let token = editor.snapshot_token();
    let revision = editor.revision();
    let history_bytes = editor.history_bytes();
    assert_eq!(
        editor.execute("bad move", vec![put(1, 2, 4.0), put(3, 10, 9.0)]),
        Err(TimelineError::DuplicateTick(10))
    );
    assert_eq!(editor.timeline(), &before);
    assert_eq!(editor.snapshot_token(), token);
    assert_eq!(editor.revision(), revision);
    assert_eq!(editor.history_bytes(), history_bytes);
    assert_eq!(editor.redo_len(), 1);
    editor.redo().unwrap();
    assert_eq!(
        editor.timeline().sample(0).unwrap()[&TrackId::from_raw(1)],
        [2.0]
    );
}

#[test]
fn command_roundtrip_includes_track_and_event_payloads() {
    let mut editor = make_editor(TimelineLimits::default());
    let before = editor.timeline().clone();
    editor
        .execute(
            "replace tracks",
            vec![
                TimelineEdit::RemoveTrack(TrackId::from_raw(1)),
                TimelineEdit::AddTrack(track(2, &[(88, 90, 7.0)])),
                TimelineEdit::AddEvent(TimelineEvent {
                    id: 42,
                    tick: u64::MAX,
                    name: "cut".into(),
                }),
            ],
        )
        .unwrap();
    let after = editor.timeline().clone();
    for _ in 0..4 {
        assert!(editor.undo().unwrap());
        assert_eq!(editor.timeline(), &before);
        assert!(editor.redo().unwrap());
        assert_eq!(editor.timeline(), &after);
    }
    assert_eq!(editor.undo_len(), 1);
    assert_eq!(editor.undo_label(), Some("replace tracks"));
    let mut other = make_editor(TimelineLimits::default());
    assert_ne!(editor.snapshot_token(), other.snapshot_token());
    assert!(!other.undo().unwrap());
}

#[test]
fn limits_reject_partial_edits_and_history_evicts_oldest_only() {
    let limits = TimelineLimits {
        max_keys: 2,
        max_history_entries: 2,
        ..TimelineLimits::default()
    };
    let mut editor = make_editor(limits);
    let before = editor.timeline().clone();
    assert_eq!(
        editor.execute("too many", vec![put(1, 0, 5.0), put(3, 3, 0.5)]),
        Err(TimelineError::LimitExceeded("keys"))
    );
    assert_eq!(editor.timeline(), &before);
    for value in 1..=4 {
        editor
            .execute("value", vec![put(1, 0, f64::from(value))])
            .unwrap();
    }
    assert_eq!(editor.undo_len(), 2);
    editor.undo().unwrap();
    editor.undo().unwrap();
    assert!(!editor.undo().unwrap());
    assert_eq!(
        editor.timeline().sample(0).unwrap()[&TrackId::from_raw(1)],
        [2.0]
    );
    assert!(editor.history_bytes() <= limits.max_history_bytes);
    editor.execute("branch", vec![put(1, 0, 8.0)]).unwrap();
    assert_eq!(editor.redo_len(), 0);
}

#[test]
fn large_deleted_payload_must_fit_history_before_it_disappears() {
    let mut timeline = Timeline::default();
    let keys: Vec<_> = (0..40)
        .map(|index| (u128::from(index), index, index as f64))
        .collect();
    timeline.add_track(track(1, &keys)).unwrap();
    let mut editor = TimelineEditor::new(
        timeline.clone(),
        TimelineLimits {
            max_history_bytes: 512,
            ..TimelineLimits::default()
        },
    )
    .unwrap();
    let token = editor.snapshot_token();
    assert_eq!(
        editor.execute(
            "delete",
            vec![TimelineEdit::RemoveTrack(TrackId::from_raw(1))]
        ),
        Err(TimelineError::LimitExceeded("history bytes"))
    );
    assert_eq!(editor.timeline(), &timeline);
    assert_eq!(editor.snapshot_token(), token);
    assert_eq!(editor.undo_len(), 0);
}

#[test]
fn ordered_range_queries_do_not_panic_on_reversed_windows() {
    let track = track(7, &[(1, 0, 0.0), (2, u64::MAX, 1.0)]);
    let (end, start) = (0, 10);
    assert_eq!(track.keys_in(start..=end).count(), 0);
    assert_eq!(
        track.keys_in(u64::MAX..=u64::MAX).next().unwrap().id(),
        KeyId::from_raw(2)
    );
    let mut timeline = Timeline::default();
    timeline.add_track(track).unwrap();
    timeline
        .add_event(TimelineEvent {
            id: 1,
            tick: 4,
            name: "event".into(),
        })
        .unwrap();
    assert_eq!(timeline.events_in(start..=end).count(), 0);
    let event = timeline.remove_event(1).unwrap();
    assert_eq!(event.tick, 4);
    assert!(timeline.events_at(4).is_empty());
    timeline.add_event(event).unwrap();
    assert_eq!(timeline.events_in(4..=4).count(), 1);
}

#[test]
fn equal_key_writes_preserve_snapshot_and_redo() {
    let mut editor = make_editor(TimelineLimits::default());
    editor.execute("change", vec![put(1, 0, 3.0)]).unwrap();
    editor.undo().unwrap();
    let token = editor.snapshot_token();
    let revision = editor.revision();
    editor.execute("same", vec![put(1, 0, 0.0)]).unwrap();
    assert_eq!(editor.snapshot_token(), token);
    assert_eq!(editor.revision(), revision);
    assert_eq!(editor.redo_len(), 1);
}
