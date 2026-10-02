//! End-to-end integration tests for the interactive music planner.
//!
//! # Provenance
//!
//! This test suite contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code; pure classic logic,
//! no AI/ML.
//!
//! # Relationship
//!
//! Drives the public [`prism_audio_music`] API exactly as a game host would:
//! it builds an immutable [`MusicModel`], starts playback on a live
//! [`MusicSystem`], advances a transport forward in fixed blocks, requests
//! quantized transitions and stingers, drives vertical layering by intensity,
//! walks a clip graph, and asserts the resolved [`MusicAction`] stream is
//! sample-accurate and byte-identical across two identically seeded systems.

use prism_audio_core::time::TimeSignature;
use prism_audio_music::{
    Clip, ClipEdge, ClipGraph, Fade, Layer, LayerSet, MusicAction, MusicModel,
    MusicSystem, Playlist, PlaylistItem, PlaylistMode, Segment, Stinger, Transition,
    TransitionContext, TransitionType, TriggerCondition,
};
use prism_audio_music::{
    BranchId, ClipId, GraphId, LayerId, LayerSetId, PlaylistId, SegmentId, SoundId, StingerId,
};

const SR: u32 = 48_000;
// At 120 BPM @ 48 kHz: 24000 samples/beat, 96000 samples/bar (4/4).
const BEAT: u64 = 24_000;
const BAR: u64 = 96_000;

/// Builds a plain segment at 120 BPM, common time, with no margins.
fn seg(id: u32, body: u64) -> Segment {
    Segment::new(
        SegmentId::new(id),
        SoundId::new(id + 100),
        120.0,
        TimeSignature::default(),
        0,
        body,
        0,
    )
}

/// A model exercising every subsystem: two plain segments, a looping playlist,
/// a layer set, a stinger, and a two-clip graph.
fn authored_model() -> MusicModel {
    let mut m = MusicModel::new();
    m.add_segment(seg(1, BAR));
    m.add_segment(seg(2, BAR));
    m.add_segment(seg(3, BAR));

    m.add_playlist(
        Playlist::new(PlaylistId::new(1), PlaylistMode::Sequence)
            .with_item(PlaylistItem::once(SegmentId::new(1)))
            .with_item(PlaylistItem::once(SegmentId::new(2))),
    );

    m.add_layer_set(
        LayerSet::new(LayerSetId::new(1))
            .with_layer(Layer::new(LayerId::new(1), SoundId::new(201), 0.3, 0.2))
            .with_layer(Layer::new(LayerId::new(2), SoundId::new(202), 0.7, 0.6)),
    );

    m.add_stinger(Stinger::new(
        StingerId::new(1),
        SoundId::new(301),
        TransitionType::NextBar,
    ));

    m.add_graph(
        ClipGraph::new(GraphId::new(1), ClipId::new(1))
            .with_clip(Clip::new(ClipId::new(1), SegmentId::new(1)))
            .with_clip(Clip::new(ClipId::new(2), SegmentId::new(2)))
            .with_edge(ClipEdge::new(
                ClipId::new(1),
                ClipId::new(2),
                TriggerCondition::OnBranch(BranchId::new(7)),
                Transition::new(TransitionType::NextBar, Fade::cut()),
            )),
    );

    m
}

#[test]
fn playlist_drives_sample_accurate_action_stream_across_blocks() {
    let model = authored_model();
    let mut sys = MusicSystem::new(SR, 0xABCD);
    let mut out = Vec::new();

    assert!(sys.start_playlist(&model, PlaylistId::new(1), 0, 0, 0, &mut out));
    assert_eq!(sys.playing_segment(), Some(SegmentId::new(1)));

    // First action is the first playlist segment at sample zero.
    match out[0] {
        MusicAction::PlaySegment {
            segment, at_sample, ..
        } => {
            assert_eq!(segment, SegmentId::new(1));
            assert_eq!(at_sample, 0);
        }
        ref other => panic!("expected PlaySegment, got {other:?}"),
    }

    // Drive the transport forward one block at a time; the second segment is
    // handed off exactly at the first segment's exit cue (one bar in).
    out.clear();
    let block = BEAT; // quarter-bar blocks
    let mut now = 0;
    let mut handoff_at = None;
    for _ in 0..8 {
        now += block;
        sys.advance_to(&model, now, &mut out);
        if let Some(a) = out.iter().find(|a| {
            matches!(a, MusicAction::PlaySegment { segment, .. } if *segment == SegmentId::new(2))
        }) {
            handoff_at = Some(a.at_sample());
            break;
        }
    }
    assert_eq!(handoff_at, Some(BAR), "segment 2 must enter at the exit cue");
    assert_eq!(sys.playing_segment(), Some(SegmentId::new(2)));
}

#[test]
fn requested_transition_lands_on_next_bar() {
    let model = authored_model();
    let mut sys = MusicSystem::new(SR, 1);
    let mut out = Vec::new();

    // Play segment 1 anchored at sample zero.
    assert!(sys.play_segment(&model, SegmentId::new(1), 0, 0, 0, &mut out));
    out.clear();

    // Request a next-bar transition partway through the first bar.
    let now = BEAT + 5_000;
    let point = sys
        .request_transition(
            &model,
            SegmentId::new(3),
            Transition::new(TransitionType::NextBar, Fade::crossfade(1_000)),
            now,
            &mut out,
        )
        .expect("known target resolves");
    assert_eq!(point, BAR, "next bar after sample {now} is one bar in");

    // The stop of the old segment and the start of the new one both land on
    // the resolved bar boundary.
    let stop = out
        .iter()
        .find(|a| matches!(a, MusicAction::StopSegment { .. }))
        .expect("old segment stopped");
    let play = out
        .iter()
        .find(|a| matches!(a, MusicAction::PlaySegment { segment, .. } if *segment == SegmentId::new(3)))
        .expect("new segment started");
    assert_eq!(stop.at_sample(), BAR);
    assert_eq!(play.at_sample(), BAR);
    assert_eq!(sys.playing_segment(), Some(SegmentId::new(3)));
}

#[test]
fn stinger_aligns_to_bar_and_preserves_playback() {
    let model = authored_model();
    let mut sys = MusicSystem::new(SR, 2);
    let mut out = Vec::new();

    assert!(sys.play_segment(&model, SegmentId::new(1), 0, 0, 0, &mut out));
    out.clear();

    let now = 10_000;
    let point = sys
        .trigger_stinger(&model, StingerId::new(1), now, &mut out)
        .expect("known stinger resolves");
    assert_eq!(point, BAR);
    assert_eq!(out.len(), 1);
    match out[0] {
        MusicAction::PlayStinger {
            stinger, at_sample, ..
        } => {
            assert_eq!(stinger, StingerId::new(1));
            assert_eq!(at_sample, BAR);
        }
        ref other => panic!("expected PlayStinger, got {other:?}"),
    }
    // Playback is untouched by a stinger.
    assert_eq!(sys.playing_segment(), Some(SegmentId::new(1)));
}

#[test]
fn intensity_adds_and_removes_layers_with_hysteresis() {
    let model = authored_model();
    let mut sys = MusicSystem::new(SR, 3);
    let mut out = Vec::new();

    // Rising to 0.75 switches on both layers (enter 0.3 and 0.7).
    sys.set_intensity(&model, 0.75, 1_000, &mut out);
    assert!(sys.is_layer_active(LayerId::new(1)));
    assert!(sys.is_layer_active(LayerId::new(2)));
    assert_eq!(sys.active_layer_count(), 2);
    let starts = out
        .iter()
        .filter(|a| matches!(a, MusicAction::StartLayer { .. }))
        .count();
    assert_eq!(starts, 2);

    // Dropping to 0.65 stays within layer 2's hysteresis band (exit 0.6), so
    // nothing changes.
    out.clear();
    sys.set_intensity(&model, 0.65, 2_000, &mut out);
    assert!(out.is_empty());
    assert_eq!(sys.active_layer_count(), 2);

    // Dropping to 0.5 falls below layer 2's exit threshold; it stops.
    out.clear();
    sys.set_intensity(&model, 0.5, 3_000, &mut out);
    assert!(sys.is_layer_active(LayerId::new(1)));
    assert!(!sys.is_layer_active(LayerId::new(2)));
    assert_eq!(out.len(), 1);
    match out[0] {
        MusicAction::StopLayer {
            layer, at_sample, ..
        } => {
            assert_eq!(layer, LayerId::new(2));
            assert_eq!(at_sample, 3_000);
        }
        ref other => panic!("expected StopLayer, got {other:?}"),
    }
}

#[test]
fn clip_graph_takes_branch_edge_on_next_bar() {
    let model = authored_model();
    let mut sys = MusicSystem::new(SR, 4);
    let mut out = Vec::new();

    assert!(sys.clip_graph_start(&model, GraphId::new(1), 0, 0, &mut out));
    assert_eq!(sys.graph_clip(GraphId::new(1)), Some(ClipId::new(1)));
    out.clear();

    // A non-matching branch leaves the graph where it is.
    let settled = sys.clip_graph_event(
        &model,
        GraphId::new(1),
        TransitionContext::new(Some(BranchId::new(1)), 0.0),
        BEAT,
        &mut out,
    );
    assert_eq!(settled, Some(ClipId::new(1)));
    assert!(out.is_empty());

    // The authored branch (7) takes the next-bar edge to clip 2.
    let settled = sys
        .clip_graph_event(
            &model,
            GraphId::new(1),
            TransitionContext::new(Some(BranchId::new(7)), 0.0),
            BEAT,
            &mut out,
        )
        .expect("event resolves");
    assert_eq!(settled, ClipId::new(2));
    let play = out
        .iter()
        .find(|a| matches!(a, MusicAction::PlaySegment { segment, .. } if *segment == SegmentId::new(2)))
        .expect("clip 2 segment started");
    assert_eq!(play.at_sample(), BAR, "next-bar edge lands on the bar line");
    assert_eq!(sys.playing_segment(), Some(SegmentId::new(2)));
}

/// Replays a scripted session on a freshly seeded system and returns the full
/// resolved action stream.
fn replay(seed: u64) -> Vec<MusicAction> {
    let model = authored_model();
    let mut sys = MusicSystem::new(SR, seed);
    let mut out = Vec::new();

    sys.start_playlist(&model, PlaylistId::new(1), 0, 0, 0, &mut out);
    sys.set_intensity(&model, 0.8, 1_000, &mut out);
    sys.trigger_stinger(&model, StingerId::new(1), 10_000, &mut out);

    let mut now = 0;
    for _ in 0..6 {
        now += BEAT;
        sys.advance_to(&model, now, &mut out);
    }

    sys.request_transition(
        &model,
        SegmentId::new(3),
        Transition::new(TransitionType::NextBar, Fade::crossfade(2_000)),
        now + BEAT,
        &mut out,
    );
    sys.set_intensity(&model, 0.1, now + BAR, &mut out);
    out
}

#[test]
fn same_seed_and_requests_produce_identical_action_streams() {
    let a = replay(0x1234_5678);
    let b = replay(0x1234_5678);
    assert_eq!(a, b, "identical seed + model + requests must be deterministic");
    assert!(!a.is_empty());
}

#[test]
fn actions_are_emitted_in_non_decreasing_sample_order_per_request() {
    // Each individual request must schedule its stop before (or at) its
    // following start; verify the transition pair is well ordered.
    let model = authored_model();
    let mut sys = MusicSystem::new(SR, 9);
    let mut out = Vec::new();
    sys.play_segment(&model, SegmentId::new(1), 0, 0, 0, &mut out);
    out.clear();

    sys.request_transition(
        &model,
        SegmentId::new(2),
        Transition::new(TransitionType::SegmentEnd, Fade::cut()),
        BEAT,
        &mut out,
    );
    assert_eq!(out.len(), 2);
    assert!(out[0].at_sample() <= out[1].at_sample());
    assert_eq!(out[0].at_sample(), BAR);
    assert_eq!(out[1].at_sample(), BAR);
}
