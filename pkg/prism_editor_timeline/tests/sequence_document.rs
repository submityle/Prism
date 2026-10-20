//! Real owned sequence project and authoring regressions.
use prism_editor_timeline::*;
use std::time::Duration;

fn limits() -> SequenceDocumentLimits {
    SequenceDocumentLimits::default()
}
fn doc(hz: u64) -> TimelineDocument {
    let mut timeline = Timeline::default();
    let mut track = Track::new(TrackId::from_raw(11), "position", 1).unwrap();
    let mut key = Keyframe::new(KeyId::from_raw(12), 0, vec![-0.0], Interpolation::Cubic);
    key.set_tangents(vec![0.125], vec![-0.25]).unwrap();
    track.insert(key).unwrap();
    track
        .insert(Keyframe::new(
            KeyId::from_raw(13),
            8,
            vec![9.25],
            Interpolation::Linear,
        ))
        .unwrap();
    timeline.add_track(track).unwrap();
    timeline
        .add_event(TimelineEvent {
            id: 99,
            tick: 2,
            name: "camera/audio:host".into(),
        })
        .unwrap();
    TimelineDocument {
        ticks_per_second: hz,
        timeline,
    }
}
fn clip(id: u128, source: u128) -> SequenceClip {
    SequenceClip {
        id: SequenceClipId(id),
        source: SequenceId(source),
        parent: SequenceRange::new(1, 9).unwrap(),
        source_range: SequenceRange::new(0, 4).unwrap(),
        rate: PlaybackRate::new(3, 2).unwrap(),
        repetitions: 3,
    }
}
fn command(edits: Vec<SequenceProjectEdit>) -> SequenceProjectCommand {
    SequenceProjectCommand {
        label: "真实编辑".into(),
        edits,
    }
}
fn editor() -> SequenceProjectEditor {
    let mut editor =
        SequenceProjectEditor::new(SequenceProjectDocument::default(), limits()).unwrap();
    editor
        .execute(command(vec![
            SequenceProjectEdit::PutDocument {
                id: SequenceId(7),
                document: doc(4),
            },
            SequenceProjectEdit::PutDocument {
                id: SequenceId(8),
                document: doc(8),
            },
            SequenceProjectEdit::PutClip {
                parent: SequenceId(7),
                clip: clip(20, 8),
            },
            SequenceProjectEdit::SetRoot(Some(SequenceId(7))),
        ]))
        .unwrap();
    editor
}

#[test]
fn complete_file_roundtrip_preserves_bits_topology_and_real_nested_dispatch() {
    let editor = editor();
    let original = editor.document();
    let bytes = original.encode(limits()).unwrap();
    let loaded = SequenceProjectDocument::decode(&bytes, limits()).unwrap();
    assert_eq!(&loaded, original);
    assert_eq!(loaded.encode(limits()).unwrap(), bytes);
    let value = loaded
        .document(SequenceId(8))
        .unwrap()
        .timeline
        .track(TrackId::from_raw(11))
        .unwrap()
        .key(KeyId::from_raw(12))
        .unwrap()
        .values()[0];
    assert_eq!(value.to_bits(), (-0.0f64).to_bits());
    let registry = loaded.registry(limits().sequence).unwrap();
    assert!(std::ptr::eq(
        registry.document(SequenceId(8)).unwrap(),
        loaded.document(SequenceId(8)).unwrap()
    ));
    let mut player = SequencePlayer::new(
        &registry,
        loaded.root().unwrap(),
        SequenceRange::new(0, 10).unwrap(),
        PlaybackMode::Once,
    )
    .unwrap();
    player.play();
    let frame = player.advance(Duration::from_secs(2)).unwrap();
    assert!(frame
        .events
        .iter()
        .any(|event| event.sequence == SequenceId(8)
            && event.event.id == 99
            && !event.path.is_empty()));
    assert!(!frame.samples[0].tracks.is_empty());
}

#[test]
fn inverses_restore_only_changed_payloads_and_preserve_high_water_across_reload() {
    let mut editor = editor();
    let before = editor.document().encode(limits()).unwrap();
    editor
        .execute(command(vec![
            SequenceProjectEdit::PutDocument {
                id: SequenceId(1000),
                document: doc(24),
            },
            SequenceProjectEdit::PutClip {
                parent: SequenceId(7),
                clip: clip(2000, 1000),
            },
        ]))
        .unwrap();
    editor.undo().unwrap();
    assert!(editor.document().document(SequenceId(1000)).is_none());
    assert_eq!(
        editor.document().next_sequence_id().unwrap(),
        SequenceId(1001)
    );
    assert_eq!(
        editor.document().next_clip_id().unwrap(),
        SequenceClipId(2001)
    );
    assert_ne!(editor.document().encode(limits()).unwrap(), before);
    let loaded =
        SequenceProjectDocument::decode(&editor.document().encode(limits()).unwrap(), limits())
            .unwrap();
    assert_eq!(loaded.next_clip_id().unwrap(), SequenceClipId(2001));
    editor.redo().unwrap();
    assert!(editor.document().document(SequenceId(1000)).is_some());
    editor
        .execute(command(vec![
            SequenceProjectEdit::RemoveClip(SequenceClipId(2000)),
            SequenceProjectEdit::RemoveDocument(SequenceId(1000)),
        ]))
        .unwrap();
    editor.undo().unwrap();
    assert!(editor.document().document(SequenceId(1000)).is_some());
}

#[test]
fn timeline_key_event_clock_and_root_edits_use_actual_inverses() {
    let mut editor = editor();
    let before = editor.document().clone();
    editor
        .execute(command(vec![
            SequenceProjectEdit::Timeline {
                id: SequenceId(8),
                edit: TimelineEdit::PutKey {
                    track: TrackId::from_raw(11),
                    key: Keyframe::new(KeyId::from_raw(12), 1, vec![77.0], Interpolation::Step),
                },
            },
            SequenceProjectEdit::Timeline {
                id: SequenceId(8),
                edit: TimelineEdit::RemoveEvent(99),
            },
            SequenceProjectEdit::SetFrequency {
                id: SequenceId(8),
                ticks_per_second: 12,
            },
            SequenceProjectEdit::SetRoot(Some(SequenceId(8))),
        ]))
        .unwrap();
    assert_eq!(
        editor
            .document()
            .document(SequenceId(8))
            .unwrap()
            .ticks_per_second,
        12
    );
    editor.undo().unwrap();
    assert_eq!(editor.document(), &before);
    editor.redo().unwrap();
    assert_eq!(editor.document().root(), Some(SequenceId(8)));
}

#[test]
fn invalid_final_graph_and_partial_operation_failures_restore_token_history_and_high_water() {
    let mut editor = editor();
    let before = editor.document().clone();
    let token = editor.snapshot_token();
    let history = editor.history_bytes();
    for edits in [
        vec![SequenceProjectEdit::PutClip {
            parent: SequenceId(8),
            clip: clip(9999, 7),
        }],
        vec![SequenceProjectEdit::RemoveDocument(SequenceId(8))],
        vec![SequenceProjectEdit::SetRoot(Some(SequenceId(404)))],
        vec![
            SequenceProjectEdit::PutDocument {
                id: SequenceId(9999),
                document: doc(4),
            },
            SequenceProjectEdit::RemoveClip(SequenceClipId(404)),
        ],
        vec![SequenceProjectEdit::SetFrequency {
            id: SequenceId(8),
            ticks_per_second: 0,
        }],
    ] {
        assert!(editor.execute(command(edits)).is_err());
        assert_eq!(editor.document(), &before);
        assert_eq!(editor.snapshot_token(), token);
        assert_eq!(editor.history_bytes(), history);
        assert_eq!(editor.revision(), 1);
    }
}

#[test]
fn valid_final_graph_can_delete_referenced_document_and_retarget_in_one_command() {
    let mut editor = editor();
    let before = editor.document().clone();
    editor
        .execute(command(vec![
            SequenceProjectEdit::RemoveDocument(SequenceId(8)),
            SequenceProjectEdit::PutDocument {
                id: SequenceId(9),
                document: doc(16),
            },
            SequenceProjectEdit::PutClip {
                parent: SequenceId(7),
                clip: clip(20, 9),
            },
        ]))
        .unwrap();
    editor.undo().unwrap();
    assert_eq!(
        editor.document().documents().collect::<Vec<_>>(),
        before.documents().collect::<Vec<_>>()
    );
    assert_eq!(
        editor.document().clips().collect::<Vec<_>>(),
        before.clips().collect::<Vec<_>>()
    );
    editor.redo().unwrap();
    assert!(editor.document().document(SequenceId(8)).is_none());
}

#[test]
fn malformed_truncation_trailing_count_and_noncanonical_order_are_rejected() {
    let editor = editor();
    let bytes = editor.document().encode(limits()).unwrap();
    for length in 0..bytes.len() {
        assert!(
            SequenceProjectDocument::decode(&bytes[..length], limits()).is_err(),
            "prefix {length}"
        );
    }
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(SequenceProjectDocument::decode(&extra, limits()).is_err());
    let mut bad = bytes.clone();
    bad[0] ^= 1;
    assert!(SequenceProjectDocument::decode(&bad, limits()).is_err());
    // Three present optional IDs occupy 51 bytes following magic. Then count.
    let mut huge = bytes.clone();
    huge[59..67].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(SequenceProjectDocument::decode(&huge, limits()).is_err());
    let mut high = bytes.clone();
    high[9..25].copy_from_slice(&0u128.to_le_bytes());
    assert_eq!(
        SequenceProjectDocument::decode(&high, limits()),
        Err(SequenceDocumentError::InvalidHighWater)
    );
    let mut duplicate = bytes.clone();
    let first_length = u64::from_le_bytes(bytes[83..91].try_into().unwrap()) as usize;
    let second_id = 91 + first_length;
    duplicate[second_id..second_id + 16].copy_from_slice(&7u128.to_le_bytes());
    assert_eq!(
        SequenceProjectDocument::decode(&duplicate, limits()),
        Err(SequenceDocumentError::InvalidEncoding)
    );
}

#[test]
fn decoder_rejects_cycles_missing_references_zero_repeats_and_noncanonical_rates() {
    let bytes = editor().document().encode(limits()).unwrap();
    let offset = bytes.len() - 104;
    for (position, payload) in [
        (offset + 32, 7u128.to_le_bytes().to_vec()),
        (offset + 32, 404u128.to_le_bytes().to_vec()),
        (offset + 96, 0u64.to_le_bytes().to_vec()),
        (offset + 80, 6u64.to_le_bytes().to_vec()),
    ] {
        let mut bad = bytes.clone();
        bad[position..position + payload.len()].copy_from_slice(&payload);
        if position == offset + 80 {
            bad[offset + 88..offset + 96].copy_from_slice(&4u64.to_le_bytes());
        }
        assert!(SequenceProjectDocument::decode(&bad, limits()).is_err());
    }
}

#[test]
fn exact_project_size_per_document_registry_and_history_caps_are_enforced() {
    let editor = editor();
    let bytes = editor.document().encode(limits()).unwrap();
    let exact = SequenceDocumentLimits {
        max_bytes: bytes.len(),
        ..limits()
    };
    assert_eq!(editor.document().encode(exact).unwrap(), bytes);
    assert!(editor
        .document()
        .encode(SequenceDocumentLimits {
            max_bytes: bytes.len() - 1,
            ..limits()
        })
        .is_err());
    assert!(SequenceProjectDocument::decode(
        &bytes,
        SequenceDocumentLimits {
            sequence: SequenceLimits {
                max_sequences: 1,
                ..limits().sequence
            },
            ..limits()
        }
    )
    .is_err());
    assert!(SequenceProjectDocument::decode(
        &bytes,
        SequenceDocumentLimits {
            sequence: SequenceLimits {
                max_document_bytes: 16,
                ..limits().sequence
            },
            ..limits()
        }
    )
    .is_err());
    let small = SequenceDocumentLimits {
        max_history_bytes: 1,
        ..limits()
    };
    let mut editor = SequenceProjectEditor::new(SequenceProjectDocument::default(), small).unwrap();
    let token = editor.snapshot_token();
    assert!(editor
        .execute(command(vec![SequenceProjectEdit::SetRoot(None)]))
        .is_err());
    assert_eq!(token, editor.snapshot_token());
}

#[test]
fn bounded_history_eviction_redo_branch_and_stale_save_token_are_explicit() {
    let mut editor = SequenceProjectEditor::new(
        SequenceProjectDocument::default(),
        SequenceDocumentLimits {
            max_history_entries: 2,
            ..limits()
        },
    )
    .unwrap();
    for id in 1..=3 {
        editor
            .execute(command(vec![SequenceProjectEdit::PutDocument {
                id: SequenceId(id),
                document: doc(24),
            }]))
            .unwrap();
    }
    assert_eq!(editor.undo_len(), 2);
    let saved = editor.snapshot_token();
    editor.undo().unwrap();
    assert_ne!(saved, editor.snapshot_token());
    assert_eq!(editor.redo_len(), 1);
    editor
        .execute(command(vec![SequenceProjectEdit::SetRoot(Some(
            SequenceId(1),
        ))]))
        .unwrap();
    assert_eq!(editor.redo_len(), 0);
    assert!(editor.history_bytes() <= editor.limits().max_history_bytes);
}

#[test]
fn exhausted_and_zero_ids_roundtrip_without_wrap_or_loss() {
    let mut editor =
        SequenceProjectEditor::new(SequenceProjectDocument::default(), limits()).unwrap();
    editor
        .execute(command(vec![
            SequenceProjectEdit::PutDocument {
                id: SequenceId(0),
                document: doc(1),
            },
            SequenceProjectEdit::PutDocument {
                id: SequenceId(u128::MAX),
                document: doc(1),
            },
            SequenceProjectEdit::PutClip {
                parent: SequenceId(0),
                clip: clip(u128::MAX, u128::MAX),
            },
        ]))
        .unwrap();
    editor.undo().unwrap();
    assert_eq!(
        editor.document().next_sequence_id(),
        Err(SequenceDocumentError::Exhausted)
    );
    assert_eq!(
        editor.document().next_clip_id(),
        Err(SequenceDocumentError::Exhausted)
    );
    let loaded =
        SequenceProjectDocument::decode(&editor.document().encode(limits()).unwrap(), limits())
            .unwrap();
    assert_eq!(
        loaded.next_sequence_id(),
        Err(SequenceDocumentError::Exhausted)
    );
}

#[test]
fn oversized_inverse_rolls_back_deletion_before_evicting_existing_history() {
    let source = editor();
    let small = SequenceDocumentLimits {
        max_history_bytes: 1024,
        ..limits()
    };
    let mut project = source.document().clone();
    // Prepare a document whose deletion inverse is larger than this history policy.
    let mut large = doc(4);
    for id in 1000..1100 {
        large
            .timeline
            .add_event(TimelineEvent {
                id,
                tick: id as u64,
                name: "large-event".into(),
            })
            .unwrap();
    }
    let mut prep = SequenceProjectEditor::new(project, limits()).unwrap();
    prep.execute(command(vec![SequenceProjectEdit::PutDocument {
        id: SequenceId(77),
        document: large,
    }]))
    .unwrap();
    project = prep.document().clone();
    let mut editor = SequenceProjectEditor::new(project.clone(), small).unwrap();
    editor
        .execute(command(vec![SequenceProjectEdit::SetRoot(Some(
            SequenceId(8),
        ))]))
        .unwrap();
    let token = editor.snapshot_token();
    let bytes = editor.history_bytes();
    let before = editor.document().clone();
    assert_eq!(
        editor.execute(command(vec![SequenceProjectEdit::RemoveDocument(
            SequenceId(77)
        )])),
        Err(SequenceDocumentError::Limit("history bytes"))
    );
    assert_eq!(editor.document(), &before);
    assert_eq!(editor.history_bytes(), bytes);
    assert_eq!(editor.snapshot_token(), token);
    assert_eq!(editor.undo_len(), 1);
    editor.undo().unwrap();
    assert_eq!(editor.document(), &project);
}

#[test]
fn track_add_remove_key_insert_remove_and_event_add_all_replay_original_payloads() {
    let mut editor = editor();
    let original = editor.document().clone();
    let track = Track::new(TrackId::from_raw(22), "new-track", 1).unwrap();
    let edits = vec![
        TimelineEdit::AddTrack(track),
        TimelineEdit::PutKey {
            track: TrackId::from_raw(22),
            key: Keyframe::new(KeyId::from_raw(23), 2, vec![5.0], Interpolation::Linear),
        },
        TimelineEdit::AddEvent(TimelineEvent {
            id: 33,
            tick: 2,
            name: "new-event".into(),
        }),
        TimelineEdit::RemoveKey {
            track: TrackId::from_raw(11),
            key: KeyId::from_raw(13),
        },
        TimelineEdit::RemoveTrack(TrackId::from_raw(11)),
    ];
    editor
        .execute(command(
            edits
                .into_iter()
                .map(|edit| SequenceProjectEdit::Timeline {
                    id: SequenceId(8),
                    edit,
                })
                .collect(),
        ))
        .unwrap();
    let changed = editor.document().clone();
    for _ in 0..3 {
        editor.undo().unwrap();
        assert_eq!(editor.document(), &original);
        editor.redo().unwrap();
        assert_eq!(editor.document(), &changed);
    }
}

#[test]
fn all_absent_metadata_slots_are_fixed_width_and_reject_nonzero_payload() {
    let empty = SequenceProjectDocument::default().encode(limits()).unwrap();
    assert_eq!(empty.len(), 75);
    let mut corrupt = empty;
    corrupt[9] = 1;
    assert_eq!(
        SequenceProjectDocument::decode(&corrupt, limits()),
        Err(SequenceDocumentError::InvalidEncoding)
    );
    for bad in [
        SequenceDocumentLimits {
            max_edits: 0,
            ..limits()
        },
        SequenceDocumentLimits {
            max_history_entries: 4097,
            ..limits()
        },
        SequenceDocumentLimits {
            max_bytes: 65 * 1024 * 1024,
            ..limits()
        },
    ] {
        assert_eq!(
            SequenceProjectDocument::default().encode(bad),
            Err(SequenceDocumentError::InvalidLimits)
        );
    }
}
