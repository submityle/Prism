//! Cross-module integration tests for `prism_audio_assets`.
//!
//! Exercises the full asset path end to end: synthesise an in-memory
//! RIFF/WAVE container, decode it through the native codec matrix and the
//! `DecoderRegistry`, stream it through the lock-free prefetch pipeline, pack
//! it into a bank blob and load it through the `BankRegistry`, then resolve a
//! `DialogueKey` down to a concrete `MediaRef` and read its decoded PCM back
//! out of the bank. Lossy ADPCM round-trips are asserted within tolerance.
//!
//! # Provenance
//! Original work. Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Dolby, MPEG, Google Resonance Audio, Web Audio, or Microsoft Project
//! Acoustics source or derived code, and no AI/ML. The RIFF/WAVE byte layout
//! synthesised here follows only the publicly documented container format.
//!
//! # Relationship
//! Covers design sections 20 and 44.1 (banks, streaming, codec matrix) and
//! section 35 (dialogue, localization) by driving the public APIs of the
//! `bank`, `codec`, `streaming`, and `dialogue` modules together.

use prism_audio_assets::bank::{
    BankId, BankManifest, BankRegistry, BankRegistryError, EntryId, LoadedBank, LoadedMedia,
    ManifestError, MediaEntry, MediaFormat, Residency,
};
use prism_audio_assets::codec::{
    AudioStreamInfo, CodecTag, DecodeError, DecoderRegistry, ImaAdpcmDecoder, ImaAdpcmEncoder,
    PcmDecoder, PcmSampleFormat, SourceDecoder, decode_wav,
};
use prism_audio_assets::dialogue::{
    DecisionNode, DialogueDecisionTree, DialogueKey, DialogueResolution, DialogueResolver,
    DialogueState, LanguageBankSet, LanguageId, MediaRef, NodeIndex, SwitchOutcome, TreeError,
};
use prism_audio_assets::streaming::PrefetchStream;

/// Absolute-value helper that avoids the std float method (clippy forbids
/// `f32::abs` in these integration tests) and keeps comparisons readable.
fn fabs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

/// Builds the raw little-endian S16 payload for interleaved `i16` samples.
fn s16_payload(samples: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    bytes
}

/// Assembles a minimal canonical RIFF/WAVE file for linear S16 PCM.
fn wav_s16le(samples: &[i16], channels: u16, sample_rate: u32) -> Vec<u8> {
    let data = s16_payload(samples);
    let bits_per_sample: u16 = 16;
    let block_align: u16 = channels * (bits_per_sample / 8);
    let byte_rate: u32 = sample_rate * u32::from(block_align);
    let data_len = data.len() as u32;
    // RIFF size = 4 ("WAVE") + (8 + 16) fmt chunk + (8 + data_len) data chunk.
    let riff_size: u32 = 4 + (8 + 16) + (8 + data_len);

    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // FORMAT_PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(&data);
    out
}

/// Deterministic interleaved stereo ramp used as the canonical test signal.
fn ramp_stereo(frames: usize) -> Vec<i16> {
    let mut v = Vec::with_capacity(frames * 2);
    for f in 0..frames {
        let left = ((f as i32 * 97) % 20_000 - 10_000) as i16;
        let right = ((f as i32 * 131) % 18_000 - 9_000) as i16;
        v.push(left);
        v.push(right);
    }
    v
}

#[test]
fn wav_parser_matches_direct_pcm_decoder() {
    let samples = ramp_stereo(256);
    let wav = wav_s16le(&samples, 2, 48_000);

    // Path A: parse the WAV container.
    let mut via_wav = decode_wav(&wav).expect("wav decodes");
    let info = via_wav.info();
    assert_eq!(info.channels, 2);
    assert_eq!(info.sample_rate, 48_000);
    assert_eq!(info.codec, CodecTag::Pcm);
    assert_eq!(info.frame_count, Some(256));
    let decoded_wav = via_wav.decode_to_end().expect("decode wav to end");

    // Path B: feed the identical raw payload straight to the PCM decoder.
    let raw = s16_payload(&samples);
    let mut direct = PcmDecoder::new(raw, PcmSampleFormat::S16Le, 2, 48_000).expect("pcm decoder");
    let decoded_direct = direct.decode_to_end().expect("decode direct to end");

    assert_eq!(decoded_wav.len(), samples.len());
    assert_eq!(decoded_wav, decoded_direct);
    // Spot-check the integer->float normalisation for the first frame.
    let expected_l = f32::from(samples[0]) / 32_768.0;
    assert!(fabs(decoded_wav[0] - expected_l) < 1e-7);
}

#[test]
fn native_registry_decodes_wav_and_rejects_unknown() {
    let samples = ramp_stereo(64);
    let wav = wav_s16le(&samples, 2, 44_100);
    let registry = DecoderRegistry::with_native();

    let mut decoder = registry.decode_bytes(&wav).expect("native registry decodes RIFF");
    assert_eq!(decoder.info().frame_count, Some(64));
    assert_eq!(decoder.decode_to_end().expect("decode").len(), samples.len());

    // Non-RIFF leading bytes have no registered container.
    match registry.decode_bytes(b"OggS not a wav at all") {
        Err(DecodeError::UnsupportedFormat) => {}
        Err(e) => panic!("expected UnsupportedFormat, got {e:?}"),
        Ok(_) => panic!("expected UnsupportedFormat, got a decoder"),
    }

    // Opus has no native decoder: the registry must honestly refuse rather than
    // return a stubbed/faked stream.
    assert!(!registry.supports_tag(&CodecTag::Opus));
    match registry.decode_tagged(&CodecTag::Opus, &wav) {
        Err(DecodeError::UnsupportedFormat) => {}
        Err(e) => panic!("expected UnsupportedFormat for Opus, got {e:?}"),
        Ok(_) => panic!("expected UnsupportedFormat for Opus, got a decoder"),
    }
}

#[test]
fn ima_adpcm_block_roundtrips_within_tolerance() {
    let samples_per_block = 505usize; // one standard 256-byte-ish mono block
    let encoder = ImaAdpcmEncoder::new(samples_per_block);
    let block_align = encoder.block_align();

    // A smooth mono ramp so quantisation error stays bounded and testable.
    // A continuous triangle wave (no instantaneous jumps) so the adaptive
    // step size can track the signal and quantisation error stays bounded.
    let mut mono: Vec<i16> = Vec::with_capacity(samples_per_block);
    let period = 200i32;
    let amplitude = 8_000i32;
    for i in 0..samples_per_block {
        let phase = i as i32 % period;
        let tri = if phase < period / 2 { phase } else { period - phase };
        let value = tri * (2 * amplitude) / (period / 2) - amplitude;
        mono.push(value as i16);
    }
    let encoded = encoder.encode_block(&mono);
    assert_eq!(encoded.len(), block_align);

    let mut decoder = ImaAdpcmDecoder::new(encoded, 1, 24_000, block_align, samples_per_block)
        .expect("adpcm decoder");
    let decoded = decoder.decode_to_end().expect("decode adpcm");
    assert_eq!(decoded.len(), samples_per_block);

    // The first frame is stored verbatim in the block header.
    let first_expected = f32::from(mono[0]) / 32_768.0;
    assert!(fabs(decoded[0] - first_expected) < 1e-6);

    // ADPCM is lossy: assert a bounded step-size reconstruction error, not
    // bit-exactness.
    let mut worst = 0.0f32;
    for (i, &target) in mono.iter().enumerate() {
        let want = f32::from(target) / 32_768.0;
        let err = fabs(decoded[i] - want);
        if err > worst {
            worst = err;
        }
    }
    assert!(worst < 0.05, "adpcm reconstruction error too large: {worst}");
}

#[test]
fn prefetch_stream_matches_full_decode() {
    let samples = ramp_stereo(300);
    let wav = wav_s16le(&samples, 2, 48_000);
    let registry = DecoderRegistry::with_native();

    // Ground truth: decode the whole asset in one shot.
    let full = registry
        .decode_bytes(&wav)
        .expect("decode")
        .decode_to_end()
        .expect("full decode");

    let decoder = registry.decode_bytes(&wav).expect("decode for stream");
    let prefetch_frames = 32usize;
    let mut stream = PrefetchStream::new(decoder, prefetch_frames, 128).expect("prefetch stream");
    assert_eq!(stream.channels(), 2);
    // The resident segment is the zero-latency head of the asset.
    assert_eq!(stream.resident_segment().len(), prefetch_frames * 2);
    assert_eq!(stream.resident_segment(), &full[..prefetch_frames * 2]);

    // Pump + drain until the decoder is exhausted and the ring is empty,
    // only ever reading frames the ring actually holds so no underrun occurs.
    let channels = stream.channels();
    let mut collected: Vec<f32> = Vec::new();
    let mut guard = 0;
    loop {
        stream.pump(4096).expect("pump");
        let avail = stream.buffered_frames();
        if avail > 0 {
            let mut out = vec![0.0f32; avail * channels];
            let result = stream.read(&mut out);
            assert_eq!(result.frames_from_data, avail);
            assert_eq!(result.frames_silenced, 0);
            collected.extend_from_slice(&out[..avail * channels]);
        }
        if stream.is_finished() {
            break;
        }
        guard += 1;
        assert!(guard < 10_000, "stream failed to drain");
    }

    assert_eq!(stream.underrun_frames(), 0);
    assert_eq!(collected.len(), full.len());
    assert_eq!(collected, full);
}

/// Builds a two-entry bank: a short memory-resident effect followed by a long
/// streamed asset, returning the manifest and the concatenated blob.
fn build_two_entry_bank(bank_id: BankId) -> (BankManifest, Vec<u8>, Vec<i16>, Vec<i16>) {
    let effect_samples = ramp_stereo(48);
    let music_samples = ramp_stereo(400);
    let effect_wav = wav_s16le(&effect_samples, 2, 48_000);
    let music_wav = wav_s16le(&music_samples, 2, 48_000);

    let mut blob = Vec::new();
    let effect_offset = 0u64;
    let effect_len = effect_wav.len() as u64;
    blob.extend_from_slice(&effect_wav);
    let music_offset = blob.len() as u64;
    let music_len = music_wav.len() as u64;
    blob.extend_from_slice(&music_wav);

    let effect = MediaEntry::new(
        EntryId(1),
        "sfx_hit".to_string(),
        MediaFormat::new(CodecTag::Pcm, 2, 48_000, Some(48)),
        Residency::Memory,
        effect_offset,
        effect_len,
    );
    let music = MediaEntry::new(
        EntryId(2),
        "music_loop".to_string(),
        MediaFormat::new(CodecTag::Pcm, 2, 48_000, Some(400)),
        Residency::Streaming { prefetch_frames: 64 },
        music_offset,
        music_len,
    );

    let manifest = BankManifest::new(bank_id, "test_bank".to_string(), 1)
        .with_media(effect)
        .with_media(music);
    (manifest, blob, effect_samples, music_samples)
}

#[test]
fn bank_loads_resident_and_streaming_media() {
    let bank_id = BankId(7);
    let (manifest, blob, effect_samples, _music) = build_two_entry_bank(bank_id);

    let registry = DecoderRegistry::with_native();
    let bank = LoadedBank::load(manifest, &blob, &registry).expect("bank loads");

    // The resident effect is fully decoded to interleaved f32 and matches a
    // direct decode of its own bytes.
    let effect = bank.media_by_name("sfx_hit").expect("effect present");
    let raw = s16_payload(&effect_samples);
    let direct = PcmDecoder::new(raw, PcmSampleFormat::S16Le, 2, 48_000)
        .expect("pcm")
        .decode_to_end()
        .expect("decode");
    match effect {
        LoadedMedia::Resident { pcm, info } => {
            assert_eq!(info.frame_count, Some(48));
            assert_eq!(pcm, &direct);
        }
        LoadedMedia::Streaming { .. } => panic!("effect should be memory-resident"),
    }
    assert!(!effect.is_streaming());

    // The music entry stays encoded and only materialises a prefetch stream on
    // demand.
    let music = bank.media_by_id(EntryId(2)).expect("music present");
    assert!(music.is_streaming());
    assert!(music.resident_pcm().is_none());
    let mut stream = music
        .open_stream(&registry, 256)
        .expect("streaming media yields a stream")
        .expect("stream constructs");
    assert_eq!(stream.channels(), 2);
    assert_eq!(stream.resident_segment().len(), 64 * 2);
    stream.pump(512).expect("pump music");
    assert!(stream.buffered_frames() > 0);

    // The resident effect carries PCM bytes; the streamed music carries encoded
    // bytes. Memory accounting reflects both.
    let usage = bank.memory_usage();
    assert!(usage.resident_pcm_bytes > 0);
    assert!(usage.encoded_bytes > 0);
    assert_eq!(usage.total(), usage.resident_pcm_bytes + usage.encoded_bytes + usage.metadata_bytes);
}

#[test]
fn bank_registry_load_unload_and_errors() {
    let bank_id = BankId(11);
    let (manifest, blob, _e, _m) = build_two_entry_bank(bank_id);
    let registry = DecoderRegistry::with_native();

    let mut banks = BankRegistry::new();
    assert!(banks.is_empty());
    let id = banks.load(manifest.clone(), &blob, &registry).expect("load");
    assert_eq!(id, bank_id);
    assert!(banks.is_loaded(bank_id));
    assert_eq!(banks.len(), 1);
    assert!(banks.media(bank_id, EntryId(1)).is_some());
    assert!(banks.media_by_name(bank_id, "music_loop").is_some());
    assert!(banks.total_memory_bytes() > 0);

    // Loading the same id twice is rejected.
    match banks.load(manifest, &blob, &registry) {
        Err(BankRegistryError::AlreadyLoaded(dup)) => assert_eq!(dup, bank_id),
        other => panic!("expected AlreadyLoaded, got {other:?}"),
    }

    banks.unload(bank_id).expect("unload");
    assert!(!banks.is_loaded(bank_id));
    match banks.unload(bank_id) {
        Err(BankRegistryError::NotLoaded(id)) => assert_eq!(id, bank_id),
        other => panic!("expected NotLoaded, got {other:?}"),
    }
}

#[test]
fn manifest_validation_rejects_malformed_banks() {
    // Duplicate media ids.
    let dup = BankManifest::new(BankId(1), "dup".to_string(), 1)
        .with_media(MediaEntry::new(
            EntryId(5),
            "a".to_string(),
            MediaFormat::new(CodecTag::Pcm, 1, 48_000, None),
            Residency::Memory,
            0,
            4,
        ))
        .with_media(MediaEntry::new(
            EntryId(5),
            "b".to_string(),
            MediaFormat::new(CodecTag::Pcm, 1, 48_000, None),
            Residency::Memory,
            4,
            4,
        ));
    match dup.validate(8) {
        Err(ManifestError::DuplicateMediaId(id)) => assert_eq!(id, EntryId(5)),
        other => panic!("expected DuplicateMediaId, got {other:?}"),
    }

    // Byte range outside the blob.
    let oob = BankManifest::new(BankId(2), "oob".to_string(), 1).with_media(MediaEntry::new(
        EntryId(1),
        "a".to_string(),
        MediaFormat::new(CodecTag::Pcm, 1, 48_000, None),
        Residency::Memory,
        0,
        64,
    ));
    match oob.validate(16) {
        Err(ManifestError::MediaRangeOutOfBounds(id)) => assert_eq!(id, EntryId(1)),
        other => panic!("expected MediaRangeOutOfBounds, got {other:?}"),
    }

    // A bank cannot depend on itself.
    let self_dep =
        BankManifest::new(BankId(3), "self".to_string(), 1).with_dependency(BankId(3));
    match self_dep.validate(0) {
        Err(ManifestError::SelfDependency(id)) => assert_eq!(id, BankId(3)),
        other => panic!("expected SelfDependency, got {other:?}"),
    }
}

#[test]
fn dialogue_resolver_exact_generic_and_silence() {
    let lang = LanguageId::new("en");
    let mut resolver = DialogueResolver::new();
    let exact = DialogueKey::new("hero", "angry", lang.clone(), 3);
    let generic = DialogueKey::new("hero", "angry", lang.clone(), 0);
    resolver.insert(exact.clone(), MediaRef::new(BankId(1), EntryId(10)));
    resolver.insert(generic, MediaRef::new(BankId(1), EntryId(11)));

    // Exact variant hit.
    match resolver.resolve(&exact) {
        DialogueResolution::Resolved(media) => {
            assert_eq!(media, MediaRef::new(BankId(1), EntryId(10)));
        }
        DialogueResolution::Silence { .. } => panic!("exact key must resolve"),
    }

    // Unknown variant falls back to the variant-0 generic take.
    let variant_miss = DialogueKey::new("hero", "angry", lang.clone(), 9);
    match resolver.resolve(&variant_miss) {
        DialogueResolution::Resolved(media) => {
            assert_eq!(media, MediaRef::new(BankId(1), EntryId(11)));
        }
        DialogueResolution::Silence { .. } => panic!("should fall back to generic"),
    }
    assert_eq!(resolver.miss_count(), 0);

    // No role match at all -> silence and a counted miss.
    let total_miss = DialogueKey::new("villain", "calm", lang, 0);
    assert!(resolver.resolve(&total_miss).is_silence());
    assert_eq!(resolver.miss_count(), 1);
}

#[test]
fn language_bank_set_switches_and_falls_back() {
    let en = LanguageId::new("en");
    let fr = LanguageId::new("fr");
    let jp = LanguageId::new("jp");
    let mut set = LanguageBankSet::new(en.clone(), BankId(100));
    set.register(fr.clone(), BankId(200));
    assert_eq!(set.active_language(), &en);
    assert_eq!(set.active_bank(), BankId(100));
    assert_eq!(set.language_count(), 2);

    match set.set_active(fr.clone()) {
        SwitchOutcome::Switched(lang) => assert_eq!(lang, fr),
        other => panic!("expected Switched, got {other:?}"),
    }
    assert_eq!(set.active_bank(), BankId(200));

    // Switching to an unregistered language falls back to the default.
    match set.set_active(jp.clone()) {
        SwitchOutcome::FellBackToDefault { requested, fallback } => {
            assert_eq!(requested, jp);
            assert_eq!(fallback, en);
        }
        other => panic!("expected fallback, got {other:?}"),
    }
    assert_eq!(set.active_bank(), BankId(100));
    assert_eq!(set.fallback_count(), 1);
}

#[test]
fn decision_tree_branches_and_selects_variant_deterministically() {
    let mut tree = DialogueDecisionTree::new();
    // Leaves first so the branch can reference their indices.
    let calm = tree.push(DecisionNode::Leaf {
        variants: vec![MediaRef::new(BankId(1), EntryId(1))],
    });
    let combat = tree.push(DecisionNode::Leaf {
        variants: vec![
            MediaRef::new(BankId(1), EntryId(2)),
            MediaRef::new(BankId(1), EntryId(3)),
        ],
    });
    // Root branch is pushed last; evaluation always starts at node 0, so make
    // the branch node 0 by building a fresh tree in root-first order instead.
    let mut rooted = DialogueDecisionTree::new();
    let root = rooted.push(DecisionNode::Branch {
        key: "intensity".to_string(),
        arms: vec![("combat".to_string(), NodeIndex(2))],
        fallback: Some(NodeIndex(1)),
    });
    assert_eq!(root, NodeIndex(0));
    rooted.push(DecisionNode::Leaf {
        variants: vec![MediaRef::new(BankId(1), EntryId(1))],
    });
    rooted.push(DecisionNode::Leaf {
        variants: vec![
            MediaRef::new(BankId(1), EntryId(2)),
            MediaRef::new(BankId(1), EntryId(3)),
        ],
    });
    rooted.validate().expect("tree is well formed");

    // Matching arm reaches the multi-variant leaf; selection is seed-stable.
    let combat_state = DialogueState::new().with("intensity", "combat");
    let a = rooted.evaluate(&combat_state, 42).expect("combat resolves");
    let b = rooted.evaluate(&combat_state, 42).expect("combat resolves");
    assert_eq!(a, b, "same seed must pick the same variant");
    assert!(a == MediaRef::new(BankId(1), EntryId(2)) || a == MediaRef::new(BankId(1), EntryId(3)));

    // Missing key takes the fallback leaf.
    let unknown = DialogueState::new();
    assert_eq!(
        rooted.evaluate(&unknown, 1),
        Some(MediaRef::new(BankId(1), EntryId(1)))
    );

    // Keep the standalone leaves meaningful so the earlier builder is exercised.
    assert_ne!(calm, combat);

    // A dangling index is caught by validation.
    let mut broken = DialogueDecisionTree::new();
    broken.push(DecisionNode::Branch {
        key: "k".to_string(),
        arms: vec![("v".to_string(), NodeIndex(99))],
        fallback: None,
    });
    match broken.validate() {
        Err(TreeError::DanglingIndex(idx)) => assert_eq!(idx, NodeIndex(99)),
        other => panic!("expected DanglingIndex, got {other:?}"),
    }
}

#[test]
fn end_to_end_dialogue_resolves_to_decoded_bank_media() {
    // Build and load a bank whose resident effect we will route dialogue to.
    let voice_bank = BankId(42);
    let (manifest, blob, effect_samples, _music) = build_two_entry_bank(voice_bank);
    let decoders = DecoderRegistry::with_native();
    let mut banks = BankRegistry::new();
    banks.load(manifest, &blob, &decoders).expect("bank loads");

    // Localization points the active language at this voice bank.
    let en = LanguageId::new("en");
    let languages = LanguageBankSet::new(en.clone(), voice_bank);
    assert_eq!(languages.active_bank(), voice_bank);

    // The resolver maps a dialogue line to the resident effect entry.
    let mut resolver = DialogueResolver::new();
    let key = DialogueKey::new("hero", "neutral", en, 0);
    resolver.insert(key.clone(), MediaRef::new(languages.active_bank(), EntryId(1)));

    // Resolve the line, then read the decoded PCM back out of the bank.
    let media = match resolver.resolve(&key) {
        DialogueResolution::Resolved(media) => media,
        DialogueResolution::Silence { .. } => panic!("dialogue line must resolve"),
    };
    let loaded = banks
        .media(media.bank, media.entry)
        .expect("resolved media is resident in the bank");
    let pcm = loaded.resident_pcm().expect("dialogue take is memory-resident");

    // The bytes read through the whole chain match a direct decode of the take.
    let direct = PcmDecoder::new(
        s16_payload(&effect_samples),
        PcmSampleFormat::S16Le,
        2,
        48_000,
    )
    .expect("pcm")
    .decode_to_end()
    .expect("decode");
    assert_eq!(pcm, &direct[..]);

    // Sanity: the stream description survived the round trip intact.
    let info: &AudioStreamInfo = loaded.info();
    assert_eq!(info.channels, 2);
    assert_eq!(info.sample_rate, 48_000);
    assert_eq!(info.codec, CodecTag::Pcm);
}
