#![cfg(feature = "sequence-runtime")]

use prism_app::App;
use prism_audio_assets::asset::AudioAssetHandle;
use prism_audio_assets::bank::{BankId, BankManifest, EntryId, MediaEntry, MediaFormat, Residency};
use prism_audio_assets::codec::CodecTag;
use prism_audio_core::buffer::AudioBuffer;
use prism_audio_core::buffer::ChannelLayout;
use prism_audio_core::voice::VoiceGroup;
use prism_audio_ecs::{AudioAssetStore, AudioPlayer, AudioRuntimeHost, AudioRuntimePlugin};
use prism_camera_ecs::{CameraBundle, CameraPlugin, CameraPoseComponent, PublishedViews};
use prism_editor_timeline::*;
use prism_math::DVec3;
use std::collections::BTreeMap;

fn source_document() -> TimelineDocument {
    TimelineDocument {
        ticks_per_second: 48_000,
        timeline: Timeline::default(),
    }
}

fn frame(document: &TimelineDocument, tracks: BTreeMap<TrackId, Vec<f64>>) -> SequenceFrame<'_> {
    SequenceFrame {
        tick: 0,
        iteration: 0,
        status: PlaybackStatus::Playing,
        samples: vec![SequenceSample {
            sequence: SequenceId(1),
            document,
            path: Vec::new(),
            tick: 0,
            tracks,
        }],
        events: Vec::new(),
    }
}

fn fence(world: &prism_ecs::world::World) -> (SequenceProjectToken, RuntimeFence) {
    let editor = SequenceProjectEditor::new(
        SequenceProjectDocument::default(),
        SequenceDocumentLimits::default(),
    )
    .unwrap();
    let token = editor.snapshot_token();
    let fence = RuntimeFence::new(token.clone(), editor.revision(), world);
    (token, fence)
}

#[test]
fn real_camera_components_publish_a_view_and_rollback_before_barrier() {
    let mut app = App::new();
    app.add_plugins(CameraPlugin::new([64, 64]));
    let entity = app.world_mut().spawn(CameraBundle::default());
    let document = source_document();
    let position = TrackId::from_raw(10);
    let rotation = TrackId::from_raw(11);
    let mut tracks = BTreeMap::new();
    tracks.insert(position, vec![3.0, 4.0, 5.0]);
    tracks.insert(rotation, vec![0.0, 0.0, 0.0, 1.0]);
    let frame = frame(&document, tracks);
    let mut adapter = SequenceRuntimeAdapter::new(SequenceRuntimeLimits::default()).unwrap();
    adapter
        .bind_camera(CameraTrackBinding {
            sequence: SequenceId(1),
            path: RuntimePath::new(&[]),
            entity,
            position,
            rotation,
            world_origin: None,
            selector: None,
            cut_value: 0,
            cut_group: 7,
        })
        .unwrap();
    let (token, fence) = fence(app.world());
    let mut prepared = adapter
        .prepare(&frame, RuntimeTransport::Playing, fence, app.world())
        .unwrap();
    adapter
        .apply(&mut prepared, app.world_mut(), &token, 0)
        .unwrap();
    assert_eq!(
        app.world()
            .get::<CameraPoseComponent>(entity)
            .unwrap()
            .0
            .position,
        DVec3::new(3.0, 4.0, 5.0)
    );
    adapter.rollback(&mut prepared, app.world_mut()).unwrap();
    assert_eq!(
        app.world().get::<CameraPoseComponent>(entity).unwrap().0,
        prism_camera::CameraPose::default()
    );
    adapter
        .apply(&mut prepared, app.world_mut(), &token, 0)
        .unwrap();
    adapter.publication_barrier(&mut prepared).unwrap();
    app.update();
    assert_eq!(app.world().resource::<PublishedViews>().len(), 1);
}

#[test]
fn resident_audio_route_receives_adapter_source_frame_and_real_dsp_output() {
    let mut app = App::new();
    app.add_plugins(AudioRuntimePlugin::default());
    let handle = AudioAssetHandle::new(BankId(9), EntryId(4));
    let blob = wav_mono(&[16_384, 16_384, 16_384]);
    let manifest =
        BankManifest::new(BankId(9), String::from("runtime"), 1).with_media(MediaEntry::new(
            EntryId(4),
            String::from("clip"),
            MediaFormat::new(CodecTag::Pcm, 1, 48_000, Some(3)),
            Residency::Memory,
            0,
            blob.len() as u64,
        ));
    app.world_mut()
        .resource_mut::<AudioAssetStore>()
        .load_bank(manifest, &blob)
        .unwrap();
    let entity = app
        .world_mut()
        .spawn(AudioPlayer::from_asset(handle, VoiceGroup(0)));
    let document = source_document();
    let time = TrackId::from_raw(20);
    let mut tracks = BTreeMap::new();
    tracks.insert(time, vec![1.0]);
    let frame = frame(&document, tracks);
    let mut adapter = SequenceRuntimeAdapter::new(SequenceRuntimeLimits::default()).unwrap();
    adapter
        .bind_audio(AudioTrackBinding {
            sequence: SequenceId(1),
            path: RuntimePath::new(&[]),
            entity,
            time: AudioTimeBinding {
                track: time,
                offset_frames: 0,
            },
            asset: handle,
        })
        .unwrap();
    let (token, fence) = fence(app.world());
    let mut prepared = adapter
        .prepare(&frame, RuntimeTransport::Playing, fence, app.world())
        .unwrap();
    adapter
        .apply(&mut prepared, app.world_mut(), &token, 0)
        .unwrap();
    assert_eq!(app.world().get::<AudioPlayer>(entity).unwrap().playhead, 1);
    adapter.publication_barrier(&mut prepared).unwrap();
    app.update();
    let mut output = AudioBuffer::new(ChannelLayout::Stereo, 1);
    app.world_mut()
        .get_non_send_mut::<AudioRuntimeHost>()
        .unwrap()
        .pump_block(&mut output);
    assert!(output.channel(0)[0] > 0.0);
}

fn wav_mono(samples: &[i16]) -> Vec<u8> {
    let mut data = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        data.extend_from_slice(&sample.to_le_bytes());
    }
    let mut wav = Vec::with_capacity(44 + data.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&48_000u32.to_le_bytes());
    wav.extend_from_slice(&96_000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
    wav.extend_from_slice(&data);
    wav
}
