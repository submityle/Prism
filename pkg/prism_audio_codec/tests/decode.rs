//! Integration tests exercising the real Symphonia demux + decode path.
//!
//! # Provenance
//! Original work; contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code; no AI/ML. The Ogg
//! Vorbis fixture is a short sound effect already committed to the repository
//! under `assets/sounds/`, embedded at build time so the test needs no file
//! I/O and runs fully offline.

use prism_audio_assets::codec::decoder::{DecodeError, SourceDecoder};
use prism_audio_assets::codec::metadata::CodecTag;
use prism_audio_assets::codec::registry::DecoderRegistry;
use prism_audio_codec::{SymphoniaDecoder, register_symphonia_decoders};

/// A short, real Ogg Vorbis asset, embedded so the test exercises the genuine
/// container-probe plus Vorbis decode path rather than a synthetic stub.
const VORBIS_OGG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/sounds/breakout_collision.ogg"
));

/// The header of a real Vorbis stream must yield a plausible stream
/// description and the `Vorbis` codec tag.
#[test]
fn vorbis_header_is_parsed() {
    let decoder = SymphoniaDecoder::new(VORBIS_OGG).expect("probe real ogg vorbis");
    let info = decoder.info();
    assert!(info.channels >= 1, "channel count must be learned from the header");
    assert!(info.sample_rate >= 8000, "sample rate must be plausible");
    assert!(
        matches!(info.codec, CodecTag::Vorbis),
        "codec tag must be Vorbis, got {:?}",
        info.codec
    );
}

/// Decoding the whole asset must produce finite, normalised frames and leave
/// the decoder exhausted with its position tracking the frames emitted.
#[test]
fn vorbis_decodes_real_frames() {
    let mut decoder = SymphoniaDecoder::new(VORBIS_OGG).expect("probe");
    let channels = decoder.info().channels as usize;
    let pcm = decoder.decode_to_end().expect("decode whole stream");

    assert!(!pcm.is_empty(), "a real asset must decode to some samples");
    assert_eq!(pcm.len() % channels, 0, "output must be whole frames");
    assert!(decoder.is_exhausted(), "decoder must report exhaustion at end");
    for sample in &pcm {
        assert!(sample.is_finite(), "decoded sample must be finite");
        assert!((-1.5..=1.5).contains(sample), "decoded sample out of range: {sample}");
    }
    let frames = (pcm.len() / channels) as u64;
    assert_eq!(decoder.position(), frames, "position must track decoded frames");
}

/// Seeking back to the start must clear exhaustion and resume decoding real
/// frames. (Exact sample-count reproduction across a seek is intentionally
/// *not* asserted: gapless trimming, enabled for seamless looping per design
/// section 44.1, legitimately changes the trimmed sample count around a seek.)
#[test]
fn seek_to_start_resumes_decoding() {
    let mut decoder = SymphoniaDecoder::new(VORBIS_OGG).expect("probe");
    let channels = decoder.info().channels as usize;
    let mut scratch = vec![0.0_f32; channels * 512];
    let first = decoder.decode(&mut scratch).expect("partial decode");
    assert!(first > 0, "the opening decode must produce frames");

    decoder.seek(0).expect("seek to start");
    assert_eq!(decoder.position(), 0, "position must reset to zero after seek(0)");
    assert!(!decoder.is_exhausted(), "seeking back to the start must clear exhaustion");

    let after = decoder.decode_to_end().expect("decode after seek");
    assert!(!after.is_empty(), "decoding must resume after seeking to the start");
    assert_eq!(after.len() % channels, 0, "resumed output must be whole frames");
    for sample in &after {
        assert!(sample.is_finite(), "resumed sample must be finite");
    }
}

/// Seeking to a mid-stream frame must land at or before the requested frame
/// (accurate seek never overshoots) and let decoding resume.
#[test]
fn seek_to_middle_sets_position_and_resumes() {
    let mut decoder = SymphoniaDecoder::new(VORBIS_OGG).expect("probe");
    let Some(frames) = decoder.info().frame_count else {
        return;
    };
    if frames < 4 {
        return;
    }
    let target = frames / 2;
    decoder.seek(target).expect("seek to middle");
    assert!(
        decoder.position() <= target,
        "accurate seek must not overshoot: landed at {} for target {}",
        decoder.position(),
        target
    );

    let channels = decoder.info().channels as usize;
    let mut buffer = vec![0.0_f32; channels * 256];
    let produced = decoder.decode(&mut buffer).expect("decode after mid seek");
    assert!(produced > 0, "decoding must resume after a mid-stream seek");
}

/// Seeking past the known end is rejected; seeking exactly to the end is
/// allowed and leaves the decoder exhausted.
#[test]
fn seek_bounds_are_enforced() {
    let mut decoder = SymphoniaDecoder::new(VORBIS_OGG).expect("probe");
    if let Some(frames) = decoder.info().frame_count {
        assert_eq!(
            decoder.seek(frames + 1),
            Err(DecodeError::SeekOutOfRange),
            "seeking beyond the frame count must be rejected"
        );
        decoder.seek(frames).expect("seeking exactly to the end is allowed");
        assert!(decoder.is_exhausted(), "end-seek must leave the decoder exhausted");
    }
}

/// A buffer too small to hold one whole frame must be rejected rather than
/// silently decoding a partial frame.
#[test]
fn output_too_small_is_reported() {
    let mut decoder = SymphoniaDecoder::new(VORBIS_OGG).expect("probe");
    let mut empty: [f32; 0] = [];
    assert_eq!(decoder.decode(&mut empty), Err(DecodeError::OutputTooSmall));
}

/// Random and textual bytes must be rejected cleanly, never panicking.
#[test]
fn garbage_is_rejected_without_panicking() {
    let zeros = [0_u8; 128];
    assert!(SymphoniaDecoder::new(&zeros).is_err(), "all-zero bytes are not a container");
    let text = b"this is definitely not an audio container stream";
    assert!(SymphoniaDecoder::new(text).is_err(), "plain text is not a container");
}

/// Registration must advertise the compressed families it supports and must
/// keep Opus unsupported (the pinned Symphonia release ships no Opus decoder).
#[test]
fn registry_registers_compressed_families() {
    let mut registry = DecoderRegistry::with_native();
    register_symphonia_decoders(&mut registry);
    assert!(registry.supports_tag(&CodecTag::Vorbis), "Vorbis tag must be registered");
    assert!(registry.supports_tag(&CodecTag::Flac), "FLAC tag must be registered");
    assert!(
        !registry.supports_tag(&CodecTag::Opus),
        "Opus must stay unsupported rather than registering a stub"
    );
}

/// The registry must route an `OggS` container to the Symphonia decoder by
/// magic number and decode real frames through it.
#[test]
fn registry_decodes_ogg_by_container_magic() {
    let mut registry = DecoderRegistry::with_native();
    register_symphonia_decoders(&mut registry);

    let mut decoder = registry
        .decode_bytes(VORBIS_OGG)
        .expect("registry must route OggS bytes to the Symphonia decoder");
    assert!(matches!(decoder.info().codec, CodecTag::Vorbis));

    let channels = decoder.info().channels as usize;
    let mut buffer = vec![0.0_f32; channels * 128];
    let produced = decoder.decode(&mut buffer).expect("decode via registry");
    assert!(produced > 0, "registry-routed decode must produce frames");
}
