//! End-to-end integration coverage for the section 49 immersive interchange and
//! loudspeaker-array output pipeline.
//!
//! The per-module unit tests inside `prism_audio_immersive` each validate one
//! concept in isolation (an `axml` round trip, a BW64 container round trip, a
//! single VBAP pan). This integration test instead drives the whole public
//! contract the way a production tool chain would: it starts from the engine's
//! runtime bed-plus-objects [`ObjectScene`], exports it to the declarative EIF
//! scene and re-imports it, assembles an ADM master document and ships it inside
//! a BW64 broadcast-WAV container, serialises a Serial-ADM live metadata stream,
//! and finally renders the same source-position truth onto a physical speaker
//! array. The point is to prove the modules compose: the ids, source taxonomy,
//! positions, and byte containers that one stage emits are exactly what the next
//! stage consumes, with no second acoustic truth drifting between them.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML. The
//! document only exercises the public standard semantics already implemented by
//! the crate (ADM ITU-R BS.2076, BW64 ITU-R BS.2088, Serial ADM ITU-R BS.2125,
//! EIF of MPEG-I Immersive Audio ISO/IEC 23090-4, and VBAP array panning).
//!
//! # Relationship
//!
//! Covers design section 49 across `prism_audio_immersive::{eif, adm, array}`
//! and their dependencies in `prism_audio_object` (bed/object scene model) and
//! `prism_audio_spatial` (listener geometry). It adds no new engine code; it is
//! a cross-module acceptance harness over the existing public API.

use bevy_math::{ops, Quat, Vec3};

use prism_audio_immersive::adm::chna::AudioId;
use prism_audio_immersive::adm::xml::{from_axml_bytes, to_axml_bytes};
use prism_audio_immersive::adm::{
    AdmDocument, AdmPosition, AudioBlockFormat, AudioChannelFormat, AudioContent,
    AudioObject as AdmObject, AudioPackFormat, AudioProgramme, AudioTrackUid, Bw64File, ChnaChunk,
    FrameType, SadmFrame, SadmFrameFormat, SadmObjectUpdate, SadmSequence, TypeDefinition,
    WaveFormat,
};
use prism_audio_immersive::array::decode::{ArrayRenderMode, ArrayRenderer, RenderDecision};
use prism_audio_immersive::array::layout::{ArrayLayout, ArraySpeaker};
use prism_audio_immersive::eif::export::export_object_scene;
use prism_audio_immersive::eif::import::import_scene;
use prism_audio_immersive::eif::source::EifSourceKind;

use prism_audio_object::bed::BedLayout;
use prism_audio_object::metadata::keyframe::Keyframe;
use prism_audio_object::metadata::stream::MetadataStream;
use prism_audio_object::metadata::ObjectMetadata;
use prism_audio_object::object::{AudioObject, ObjectId};
use prism_audio_object::scene::ObjectScene;

use prism_audio_spatial::geometry::Listener;

/// Absolute tolerance for position/gain comparisons that survive a lossy polar
/// or interpolation round trip.
const EPS: f32 = 1e-4;

/// Builds a representative 5.1.4 object scene: a full immersive bed plus two
/// freely-placed objects to the front-right and overhead-left.
fn authored_scene() -> ObjectScene {
    let mut scene = ObjectScene::new(BedLayout::Surround5_1_4);
    scene.add_object(AudioObject::new(ObjectId(1), Vec3::new(3.0, 0.0, -4.0), 0.9));
    scene.add_object(AudioObject::new(ObjectId(2), Vec3::new(-1.5, 2.5, -1.0), 0.6));
    scene
}

#[test]
fn object_scene_exports_and_reimports_through_eif() {
    let scene = authored_scene();
    let eif = export_object_scene(&scene);

    // A 5.1.4 bed has ten channels but the LFE carries no direction, so nine
    // directional channel sources survive export, plus the two objects.
    let channel_sources = eif
        .sources
        .iter()
        .filter(|s| matches!(s.kind, EifSourceKind::Channel { .. }))
        .count();
    let object_sources = eif
        .sources
        .iter()
        .filter(|s| matches!(s.kind, EifSourceKind::Object))
        .count();
    assert_eq!(channel_sources, 9, "5.1.4 keeps nine directional channels");
    assert_eq!(object_sources, 2, "both authored objects are exported");
    assert_eq!(eif.sources.len(), channel_sources + object_sources);

    // The exported scene must be self-consistent for a third-party loader.
    eif.validate().expect("exported EIF scene validates");

    // Re-importing against a listener reconstructs the same source population,
    // proving the exported ids/kinds are what the importer consumes.
    let listener = Listener::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ZERO);
    let imported = import_scene(&eif, &listener, 48_000);
    assert_eq!(imported.sources.len(), eif.sources.len());
    let imported_objects = imported
        .sources
        .iter()
        .filter(|s| matches!(s.kind, EifSourceKind::Object))
        .count();
    assert_eq!(imported_objects, object_sources);
}

/// Assembles a minimal but valid ADM object master mirroring one scene object.
fn adm_master() -> AdmDocument {
    let mut doc = AdmDocument::new();

    let mut channel = AudioChannelFormat::new("AC_00031001", "Object 1", TypeDefinition::Objects);
    channel.block_formats.push(AudioBlockFormat::new(
        "AB_00031001_00000001",
        AdmPosition::new(37.0, 0.0, 1.0),
    ));
    doc.channel_formats.push(channel);

    let mut pack = AudioPackFormat::new("AP_00031001", "Object 1", TypeDefinition::Objects);
    pack.channel_format_refs.push(String::from("AC_00031001"));
    doc.pack_formats.push(pack);

    doc.track_uids.push(AudioTrackUid::new(
        "ATU_00000001",
        1,
        "AC_00031001",
        "AP_00031001",
    ));

    let mut object = AdmObject::new("AO_1001", "Hero Object");
    object.pack_format_refs.push(String::from("AP_00031001"));
    object.track_uid_refs.push(String::from("ATU_00000001"));
    doc.objects.push(object);

    let mut content = AudioContent::new("ACO_1001", "Effects");
    content.object_refs.push(String::from("AO_1001"));
    doc.contents.push(content);

    let mut programme = AudioProgramme::new("APR_1001", "Main Mix");
    programme.content_refs.push(String::from("ACO_1001"));
    doc.programmes.push(programme);

    doc
}

#[test]
fn adm_master_ships_inside_bw64_and_round_trips() {
    let doc = adm_master();
    doc.validate().expect("authored ADM master validates");

    let axml = to_axml_bytes(&doc);

    // The chna table binds the track UID to its one audio track.
    let mut chna = ChnaChunk::new();
    chna.push(AudioId::new(1, "ATU_00000001", "AC_00031001", "AP_00031001"));

    // One mono 24-bit object track with a tiny deterministic payload.
    let format = WaveFormat::new(1, 1, 48_000, 24);
    let data: Vec<u8> = (0..30u8).collect();
    let file = Bw64File::new(format, data, chna, axml.clone());

    // The whole broadcast-WAV container must survive a byte-exact round trip.
    let bytes = file.to_bytes();
    let parsed = Bw64File::from_bytes(&bytes).expect("BW64 parses");
    assert_eq!(parsed, file);
    assert_eq!(parsed.to_bytes(), bytes);

    // And the carried axml must re-parse into the identical ADM graph.
    let recovered = from_axml_bytes(&parsed.axml).expect("carried axml parses");
    assert_eq!(recovered, doc);
}

#[test]
fn serial_adm_live_stream_round_trips_and_interpolates() {
    let mut sequence = SadmSequence::new(10.0);

    let mut first = SadmFrame::new(SadmFrameFormat::new(0, 0.0, 0.1, FrameType::Full));
    first.push(SadmObjectUpdate::new(1, AdmPosition::new(0.0, 0.0, 1.0), 1.0));
    sequence.push(first);

    let mut second = SadmFrame::new(SadmFrameFormat::new(1, 0.1, 0.1, FrameType::Divided));
    second.push(SadmObjectUpdate::new(1, AdmPosition::new(40.0, 0.0, 1.0), 0.5));
    sequence.push(second);

    // The live metadata stream survives a byte-exact serialisation round trip.
    let bytes = sequence.to_bytes();
    let parsed = SadmSequence::from_bytes(&bytes).expect("S-ADM parses");
    assert_eq!(parsed, sequence);

    // Sampling halfway between the two keyframes interpolates azimuth and gain.
    let midpoint = parsed.sample_at(1, 0.05).expect("object is live at 0.05 s");
    assert!((midpoint.position.azimuth - 20.0).abs() < EPS);
    assert!((midpoint.gain - 0.75).abs() < EPS);
}

/// A compact 3D octagon-plus-height array: eight horizontal speakers and four
/// elevated ones, enough directional feeds to render as a true physical array.
fn physical_array() -> ArrayLayout {
    let mut layout = ArrayLayout::new();
    let radius = 2.0;
    let ring = 8;
    for i in 0..ring {
        let angle = (i as f32) * core::f32::consts::TAU / (ring as f32);
        let dir = Vec3::new(ops::sin(angle), 0.0, -ops::cos(angle));
        layout.push(ArraySpeaker::from_direction(i, dir, radius, "ring"));
    }
    let heights = [
        Vec3::new(-0.7, 0.7, -0.7),
        Vec3::new(0.7, 0.7, -0.7),
        Vec3::new(-0.7, 0.7, 0.7),
        Vec3::new(0.7, 0.7, 0.7),
    ];
    for (i, dir) in heights.iter().enumerate() {
        layout.push(ArraySpeaker::from_direction(ring + i, *dir, radius, "height"));
    }
    layout
}

#[test]
fn object_scene_renders_onto_physical_array() {
    let layout = physical_array();
    let renderer = ArrayRenderer::new(&layout, ArrayRenderMode::Vbap);

    // Twelve directional speakers is plenty for a direct array render.
    assert!(matches!(renderer.decision(), RenderDecision::Array(_)));
    assert_eq!(renderer.output_channels(), layout.len());

    // Render a front-right object; VBAP spreads unit energy over a few
    // speakers, all gains finite and non-negative, with real energy present.
    let gains = renderer.render_object(Vec3::new(1.0, 0.0, -1.0));
    assert_eq!(gains.len(), renderer.output_channels());
    assert!(gains.iter().all(|g| g.is_finite() && *g >= 0.0));
    let energy: f32 = gains.iter().map(|g| g * g).sum();
    assert!(energy > 0.0, "a placed object must drive the array");
}

#[test]
fn empty_array_degrades_to_binaural_without_panicking() {
    // No physical speakers: the renderer must gracefully fall back rather than
    // fail, so a headphone listener still gets a valid two-channel feed.
    let renderer = ArrayRenderer::new(&ArrayLayout::new(), ArrayRenderMode::Wfs);
    assert_eq!(renderer.decision(), RenderDecision::FallbackBinaural);
    let gains = renderer.render_object(Vec3::new(0.0, 0.0, -1.0));
    assert_eq!(gains.len(), renderer.output_channels());
    assert!(gains.iter().all(|g| g.is_finite()));
}

#[test]
fn animated_object_metadata_feeds_the_export() {
    let mut scene = ObjectScene::new(BedLayout::Stereo);

    // An object that slides from the left to the right over 1 second.
    let mut stream = MetadataStream::new();
    stream.push(
        0.0,
        ObjectMetadata {
            position: Vec3::new(-2.0, 0.0, -1.0),
            gain: 0.5,
            spread: 0.0,
            priority: 1.0,
        },
    );
    stream.push(
        1.0,
        ObjectMetadata {
            position: Vec3::new(2.0, 0.0, -1.0),
            gain: 1.0,
            spread: 0.0,
            priority: 1.0,
        },
    );
    // Keyframe is part of the stream public API; assert the first entry mirrors
    // what we pushed so the authoring surface stays stable.
    let first = Keyframe::new(
        0.0,
        ObjectMetadata {
            position: Vec3::new(-2.0, 0.0, -1.0),
            gain: 0.5,
            spread: 0.0,
            priority: 1.0,
        },
    );
    assert_eq!(stream.keyframes()[0], first);

    scene.add_animated_object(AudioObject::new(ObjectId(5), Vec3::ZERO, 1.0), stream);

    // Advance to the midpoint: the object should be centred and the exported
    // EIF source must carry that live position, tying runtime metadata to the
    // interchange layer.
    scene.advance_to(0.5);
    let moved = scene.current_objects()[0];
    assert!(moved.position.x.abs() < EPS, "object is centred at t=0.5");

    let eif = export_object_scene(&scene);
    let object_source = eif
        .sources
        .iter()
        .find(|s| matches!(s.kind, EifSourceKind::Object))
        .expect("one exported object source");
    assert!((object_source.position.x - moved.position.x).abs() < EPS);
    assert!((object_source.position.z - moved.position.z).abs() < EPS);
}
