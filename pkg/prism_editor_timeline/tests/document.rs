//! Complete asset snapshots have limits independent from individual commands.
use prism_editor_timeline::*;

fn fixture() -> TimelineDocument {
    let mut timeline = Timeline::default();
    let mut track = Track::new(TrackId::from_raw(u128::MAX), "相机 xyz", 2).unwrap();
    let mut key = Keyframe::new(
        KeyId::from_raw(u128::MAX),
        u64::MAX,
        vec![-0.0, f64::MAX],
        Interpolation::Cubic,
    );
    key.set_tangents(vec![0.25, -0.5], vec![-1.0, 2.0]).unwrap();
    track.insert(key).unwrap();
    timeline.add_track(track).unwrap();
    timeline
        .add_event(TimelineEvent {
            id: 42,
            tick: u64::MAX,
            name: "镜头切换".into(),
        })
        .unwrap();
    TimelineDocument {
        ticks_per_second: 48_000,
        timeline,
    }
}

#[test]
fn full_document_preserves_identities_times_rate_and_exact_numeric_bits() {
    let document = fixture();
    let limits = TimelineLimits::default();
    let bytes = document.encode(limits, 4096).unwrap();
    let decoded = TimelineDocument::decode(&bytes, limits, 4096).unwrap();
    assert_eq!(document, decoded);
    assert_eq!(bytes, decoded.encode(limits, 4096).unwrap());
    let track = decoded
        .timeline
        .track(TrackId::from_raw(u128::MAX))
        .unwrap();
    assert_eq!(
        track.key(KeyId::from_raw(u128::MAX)).unwrap().values()[0].to_bits(),
        (-0.0f64).to_bits()
    );
    let editor = TimelineEditor::new(decoded.timeline, limits).unwrap();
    assert_eq!(
        (editor.undo_len(), editor.redo_len(), editor.revision()),
        (0, 0, 0)
    );
}

#[test]
fn full_snapshot_is_not_restricted_by_history_capacity_or_command_edit_count() {
    let limits = TimelineLimits {
        max_edits: 1,
        max_history_bytes: 1,
        ..TimelineLimits::default()
    };
    let mut timeline = Timeline::default();
    for id in 0..5000u128 {
        timeline
            .add_event(TimelineEvent {
                id,
                tick: id as u64,
                name: "notify".into(),
            })
            .unwrap();
    }
    let document = TimelineDocument {
        ticks_per_second: 240,
        timeline,
    };
    let bytes = document.encode(limits, 256_000).unwrap();
    assert_eq!(
        TimelineDocument::decode(&bytes, limits, 256_000).unwrap(),
        document
    );
    assert!(document.encode(limits, bytes.len() - 1).is_err());
    assert!(TimelineDocument::decode(&bytes, limits, bytes.len() - 1).is_err());
    let lower = TimelineLimits {
        max_events: 4999,
        ..limits
    };
    assert!(document.encode(lower, 256_000).is_err());
    assert!(TimelineDocument::decode(&bytes, lower, 256_000).is_err());
}

#[test]
fn truncated_version_zero_rate_and_trailing_bytes_never_produce_a_candidate() {
    let limits = TimelineLimits::default();
    let bytes = fixture().encode(limits, 4096).unwrap();
    for end in 0..bytes.len() {
        assert!(
            TimelineDocument::decode(&bytes[..end], limits, 4096).is_err(),
            "prefix {end}"
        );
    }
    let mut bad = bytes.clone();
    bad[7] = b'9';
    assert_eq!(
        TimelineDocument::decode(&bad, limits, 4096),
        Err(TimelineError::InvalidEncoding)
    );
    let mut bad = bytes.clone();
    bad[8..16].fill(0);
    assert_eq!(
        TimelineDocument::decode(&bad, limits, 4096),
        Err(TimelineError::EmptyInput)
    );
    let mut bad = bytes.clone();
    bad.extend_from_slice(&[0]);
    assert_eq!(
        TimelineDocument::decode(&bad, limits, 4096),
        Err(TimelineError::InvalidEncoding)
    );
    let empty = TimelineDocument {
        ticks_per_second: 1,
        timeline: Timeline::default(),
    };
    assert_eq!(
        TimelineDocument::decode(&empty.encode(limits, 24).unwrap(), limits, 24).unwrap(),
        empty
    );
}

#[test]
fn malformed_counts_duplicate_ids_and_nonfinite_values_are_rejected() {
    let limits = TimelineLimits::default();
    let mut bytes = TimelineDocument {
        ticks_per_second: 1,
        timeline: Timeline::default(),
    }
    .encode(limits, 4096)
    .unwrap();
    bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        TimelineDocument::decode(&bytes, limits, 4096),
        Err(TimelineError::LimitExceeded("tracks"))
    );
    let mut document = fixture();
    document
        .timeline
        .add_event(TimelineEvent {
            id: 43,
            tick: u64::MAX,
            name: "镜头切换".into(),
        })
        .unwrap();
    let mut bytes = document.encode(limits, 4096).unwrap();
    let event_bytes = 16 + 8 + 4 + "镜头切换".len();
    let second_event = bytes.len() - event_bytes;
    bytes[second_event..second_event + 16].copy_from_slice(&42u128.to_le_bytes());
    assert_eq!(
        TimelineDocument::decode(&bytes, limits, 4096),
        Err(TimelineError::DuplicateEvent(42))
    );
    let mut bytes = fixture().encode(limits, 4096).unwrap();
    let value_offset = 20 + 16 + 4 + "相机 xyz".len() + 4 + 4 + 16 + 8 + 1 + 4;
    bytes[value_offset..value_offset + 8].copy_from_slice(&f64::NAN.to_le_bytes());
    assert_eq!(
        TimelineDocument::decode(&bytes, limits, 4096),
        Err(TimelineError::NonFiniteValue)
    );
    let mut invalid = fixture();
    invalid.ticks_per_second = 0;
    assert!(invalid.encode(limits, 4096).is_err());
}
