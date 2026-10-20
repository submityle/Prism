//! Exact command replay and adversarial byte packet validation.
use prism_editor_timeline::*;

fn fixture() -> TimelineCommand {
    let mut track = Track::new(TrackId::from_raw(u128::MAX), "相机", 2).unwrap();
    let mut key = Keyframe::new(
        KeyId::from_raw(u128::MAX),
        u64::MAX,
        vec![-0.0, f64::MAX],
        Interpolation::Cubic,
    );
    key.set_tangents(vec![0.1, -0.2], vec![0.3, -0.4]).unwrap();
    track.insert(key.clone()).unwrap();
    TimelineCommand {
        label: "Create camera".into(),
        edits: vec![
            TimelineEdit::AddTrack(track),
            TimelineEdit::PutKey {
                track: TrackId::from_raw(u128::MAX),
                key,
            },
            TimelineEdit::RemoveKey {
                track: TrackId::from_raw(u128::MAX),
                key: KeyId::from_raw(u128::MAX),
            },
            TimelineEdit::RemoveTrack(TrackId::from_raw(u128::MAX)),
            TimelineEdit::AddEvent(TimelineEvent {
                id: u128::MAX,
                tick: u64::MAX,
                name: "cut".into(),
            }),
            TimelineEdit::RemoveEvent(u128::MAX),
        ],
    }
}
#[test]
fn every_payload_roundtrips_exact_bytes_and_replays_atomically() {
    let limits = TimelineLimits::default();
    let original = fixture();
    let bytes = original.encode(limits).unwrap();
    let decoded = TimelineCommand::decode(&bytes, limits).unwrap();
    assert_eq!(original, decoded);
    assert_eq!(decoded.encode(limits).unwrap(), bytes);
    let mut editor = TimelineEditor::new(Timeline::default(), limits).unwrap();
    editor.execute(decoded.label, decoded.edits).unwrap();
    assert_eq!(editor.timeline(), &Timeline::default());
    editor.undo().unwrap();
    assert_eq!(editor.timeline(), &Timeline::default());
    let TimelineEdit::AddTrack(track) = &original.edits[0] else {
        panic!()
    };
    let single = TimelineCommand {
        label: "Add".into(),
        edits: vec![TimelineEdit::AddTrack(track.clone())],
    };
    let decoded = TimelineCommand::decode(&single.encode(limits).unwrap(), limits).unwrap();
    editor.execute(decoded.label, decoded.edits).unwrap();
    assert_eq!(
        editor
            .timeline()
            .track(TrackId::from_raw(u128::MAX))
            .unwrap()
            .key(KeyId::from_raw(u128::MAX))
            .unwrap()
            .values()[0]
            .to_bits(),
        (-0.0f64).to_bits()
    );
}
#[test]
fn every_truncation_unknown_version_and_extra_byte_are_rejected() {
    let limits = TimelineLimits::default();
    let bytes = fixture().encode(limits).unwrap();
    for end in 0..bytes.len() {
        assert!(
            TimelineCommand::decode(&bytes[..end], limits).is_err(),
            "truncation {end}"
        );
    }
    let mut invalid = bytes.clone();
    invalid[7] = b'9';
    assert_eq!(
        TimelineCommand::decode(&invalid, limits),
        Err(TimelineError::InvalidEncoding)
    );
    let mut invalid = bytes.clone();
    invalid.push(0);
    assert_eq!(
        TimelineCommand::decode(&invalid, limits),
        Err(TimelineError::InvalidEncoding)
    );
}
#[test]
fn corrupt_counts_nonfinite_channels_and_policy_limits_fail_before_allocation() {
    let limits = TimelineLimits::default();
    let mut invalid = Vec::from(&b"PRSTLN01"[..]);
    invalid.extend_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        TimelineCommand::decode(&invalid, limits),
        Err(TimelineError::LimitExceeded("text bytes"))
    );
    let mut invalid = Vec::from(&b"PRSTLN01"[..]);
    invalid.extend_from_slice(&1u32.to_le_bytes());
    invalid.push(b'x');
    invalid.extend_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        TimelineCommand::decode(&invalid, limits),
        Err(TimelineError::LimitExceeded("transaction edits"))
    );
    let command = TimelineCommand {
        label: "x".into(),
        edits: vec![TimelineEdit::PutKey {
            track: TrackId::from_raw(1),
            key: Keyframe::new(KeyId::from_raw(2), 0, vec![0.0], Interpolation::Step),
        }],
    };
    let mut bytes = command.encode(limits).unwrap();
    let values_offset = 8 + 4 + 1 + 4 + 1 + 16 + 16 + 8 + 1 + 4;
    bytes[values_offset..values_offset + 8].copy_from_slice(&f64::INFINITY.to_bits().to_le_bytes());
    assert_eq!(
        TimelineCommand::decode(&bytes, limits),
        Err(TimelineError::NonFiniteValue)
    );
    assert!(command
        .encode(TimelineLimits {
            max_history_bytes: 12,
            ..limits
        })
        .is_err());
    assert!(fixture()
        .encode(TimelineLimits {
            max_dimension: 1,
            ..limits
        })
        .is_err());
    let bytes = fixture().encode(limits).unwrap();
    assert!(TimelineCommand::decode(
        &bytes,
        TimelineLimits {
            max_keys: 1,
            ..limits
        }
    )
    .is_err());
}
#[test]
fn byte_mutations_never_panic_and_successful_packets_reencode() {
    let limits = TimelineLimits {
        max_history_bytes: 4096,
        max_keys: 128,
        ..TimelineLimits::default()
    };
    let original = fixture().encode(limits).unwrap();
    for index in 0..original.len() {
        for value in [0, 1, 127, 255] {
            let mut bytes = original.clone();
            bytes[index] = value;
            if let Ok(command) = TimelineCommand::decode(&bytes, limits) {
                assert_eq!(command.encode(limits).unwrap(), bytes);
            }
        }
    }
}
#[test]
fn identity_exhaustion_returns_errors_without_aliasing_live_objects() {
    let mut track = Track::new(TrackId::from_raw(u128::MAX), "id", 1).unwrap();
    track
        .insert(Keyframe::new(
            KeyId::from_raw(u128::MAX),
            0,
            vec![1.0],
            Interpolation::Linear,
        ))
        .unwrap();
    assert!(track.next_key_id().is_err());
    let mut timeline = Timeline::default();
    timeline.add_track(track).unwrap();
    timeline
        .add_event(TimelineEvent {
            id: u128::MAX,
            tick: 0,
            name: "event".into(),
        })
        .unwrap();
    assert!(timeline.next_track_id().is_err());
    assert!(timeline.next_event_id().is_err());
}
