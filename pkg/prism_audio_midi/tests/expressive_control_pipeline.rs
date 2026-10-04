//! End-to-end integration coverage for the `prism_audio_midi` expressive
//! control crate.
//!
//! These tests drive the crate's real public API along the whole Universal MIDI
//! Packet pipeline: from a raw big-endian device byte stream, through
//! [`UmpDecoder`] packet assembly into [`MidiMessage`]s, folding into
//! [`ChannelState`] and [`PerNoteExpression`], [`MpeAllocator`] member-channel
//! attribution, and finally [`ExpressionRouter`] emission of sample-offset
//! [`ModulationWrite`]s. Each test pins exactly one concern: the full
//! parse-to-schedule chain, bitwise determinism, every decoded channel-voice
//! variant, uniform-resolution up-scaling, utility and system messages,
//! byte/word stream equivalence with realignment, malformed and boundary
//! statuses, MPE allocation and zone layout, per-channel high-resolution and
//! `RPN` tracking, the per-note slot lifecycle, `Min-Center-Max` scaling
//! invariants, and router transform plus sample-offset ordering.
//!
//! # Provenance
//! Original work; contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Dolby, MPEG, Google Resonance Audio, Web Audio, or Project Acoustics
//! source or derived code; no AI/ML. The MIDI 1.0, MIDI 2.0 (UMP), and MPE
//! specifications are publicly published MIDI Association standards; only their
//! documented semantics are exercised, never any third-party source.
//!
//! # Relationship
//! Black-box integration test of design section 52 (expressive control / MIDI
//! 2.0 / MPE). It consumes only the published public surface of
//! `prism_audio_midi` and the [`prism_audio_core`] sample scalar, wiring the
//! crate's modules together the way the engine's section 8 scheduler and
//! section 12 modulation routing intend to.

use prism_audio_core::math::Sample;

use prism_audio_midi::expression::{
    DEFAULT_PITCH_BEND_RANGE, MAX_ACTIVE_NOTES, MAX_PER_NOTE_CONTROLLERS,
};
use prism_audio_midi::mapping::{normalize_u16, normalize_u32};
use prism_audio_midi::ump::{
    decode_midi1, decode_midi2, decode_system, decode_utility, PITCH_BEND_CENTER_32,
};
use prism_audio_midi::{
    scale_down, scale_up, ChannelState, ChannelVoice, Curve, ExpressionDimension, ExpressionRouter,
    MessageType, MidiMessage, ModulationTarget, ModulationWrite, MpeAllocator, MpeZone,
    NoteAttribute, PerNoteController, PerNoteExpression, SystemMessage, TargetMapping, UmpDecoder,
    UmpWord, UtilityMessage, VoiceExpression, ZoneKind,
};

// ------------------------------------------------------------------------
// Helpers (shared, no `std` float intrinsics, no `println!`).
// ------------------------------------------------------------------------

/// Branch-free absolute value avoiding `std` float intrinsics, so the helper is
/// equally valid under `no_std` builds of the crate.
fn fabs(x: Sample) -> Sample {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Approximate float equality using the local [`fabs`], with an explicit
/// epsilon per call site instead of a naked `==`.
fn close(a: Sample, b: Sample, eps: Sample) -> bool {
    fabs(a - b) < eps
}

/// Flattens a slice of 32-bit UMP words into the big-endian byte stream a real
/// device delivers.
fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &word in words {
        bytes.extend_from_slice(&word.to_be_bytes());
    }
    bytes
}

/// A representative MIDI 2.0 performance on group 0, channel 1, note 60:
/// note-on, channel pitch bend, per-note pitch bend, channel pressure, and a
/// full brightness (`CC74`) control change.
fn demo_stream() -> Vec<u8> {
    words_to_bytes(&[
        0x4091_3C00,
        0x9000_0000, // note on, velocity 0x9000, attribute none
        0x40E1_0000,
        0xC000_0000, // channel pitch bend, +0.5 of range
        0x4061_3C00,
        0xC000_0000, // per-note pitch bend, +0.5 of range
        0x40D1_0000,
        0x8000_0000, // channel pressure, ~0.5
        0x40B1_4A00,
        0xFFFF_FFFF, // control change 74 (brightness), full
    ])
}

/// The engine-side router: pitch bend to pitch, pressure to gain, timbre to
/// cutoff, and velocity to a custom slot, in that deterministic order.
fn router_full() -> ExpressionRouter {
    let mut router = ExpressionRouter::new();
    router.push(TargetMapping::linear(
        ExpressionDimension::PitchBend,
        ModulationTarget::Pitch,
    ));
    router.push(TargetMapping::linear(
        ExpressionDimension::Pressure,
        ModulationTarget::Gain,
    ));
    router.push(TargetMapping::linear(
        ExpressionDimension::Timbre,
        ModulationTarget::Cutoff,
    ));
    router.push(TargetMapping::linear(
        ExpressionDimension::Velocity,
        ModulationTarget::Custom { slot: 0 },
    ));
    router
}

/// Folds a decoded message into per-channel and per-note expression state,
/// mirroring how the engine routes the stream before snapshotting a voice.
fn apply_message(
    message: &MidiMessage,
    channels: &mut [ChannelState; 16],
    per_note: &mut PerNoteExpression,
) {
    let MidiMessage::ChannelVoice {
        channel, message, ..
    } = message
    else {
        return;
    };
    let ch = *channel;
    match message {
        ChannelVoice::NoteOn { note, velocity, .. } => {
            per_note.note_on(ch, *note, *velocity);
        }
        ChannelVoice::NoteOff { note, .. } => {
            per_note.note_off(ch, *note);
        }
        ChannelVoice::PerNotePitchBend { note, bend } => {
            per_note.set_pitch_bend(ch, *note, *bend);
        }
        ChannelVoice::PolyPressure { note, pressure } => {
            per_note.set_pressure(ch, *note, *pressure);
        }
        ChannelVoice::RegisteredPerNoteController { note, index, value }
        | ChannelVoice::AssignablePerNoteController { note, index, value } => {
            per_note.set_controller(ch, *note, *index, *value);
        }
        ChannelVoice::PerNoteManagement {
            note,
            detach,
            reset,
        } => {
            per_note.manage(ch, *note, *detach, *reset);
        }
        other => channels[ch as usize].apply(other),
    }
}

/// Snapshots a sounding voice from the folded channel and per-note state, the
/// way the engine assembles a [`VoiceExpression`] for routing.
fn build_voice(
    voice_id: u32,
    channel: u8,
    note: u8,
    channel_state: &ChannelState,
    per_note: &PerNoteExpression,
) -> VoiceExpression {
    let mut voice = VoiceExpression::new(voice_id, note);
    voice.channel_pitch_bend = channel_state.pitch_bend_raw();
    voice.pitch_bend_range_semitones = channel_state.pitch_bend_range();
    voice.channel_pressure = channel_state.pressure_raw();
    voice.timbre = channel_state.controller(PerNoteController::Brightness.index());
    if let Some(state) = per_note.get(channel, note) {
        voice.per_note_pitch_bend = state.pitch_bend;
        voice.per_note_pressure = state.pressure;
        voice.velocity = state.velocity;
        if let Some(value) = state.named_controller(PerNoteController::Brightness) {
            voice.set_controller(PerNoteController::Brightness.index(), value);
        }
    }
    voice
}

/// Drives a complete byte stream through decode, state folding, voice
/// snapshotting, and routing, returning the ordered modulation writes.
fn pipeline_writes(
    stream: &[u8],
    channel: u8,
    note: u8,
    voice_id: u32,
    router: &ExpressionRouter,
) -> Vec<ModulationWrite> {
    let mut decoder = UmpDecoder::new();
    let mut messages = Vec::new();
    decoder.push_bytes(stream, &mut messages);

    let mut channels: [ChannelState; 16] = core::array::from_fn(|_| ChannelState::new());
    let mut per_note = PerNoteExpression::new();
    for message in &messages {
        apply_message(message, &mut channels, &mut per_note);
    }

    let voice = build_voice(voice_id, channel, note, &channels[channel as usize], &per_note);
    let mut writes = Vec::new();
    router.route(&voice, 0, &mut writes);
    writes
}

// ------------------------------------------------------------------------
// 1. Full parse-to-schedule chain.
// ------------------------------------------------------------------------

#[test]
fn full_pipeline_bytes_to_modulation_writes() {
    let router = router_full();
    let writes = pipeline_writes(&demo_stream(), 1, 60, 1, &router);

    assert_eq!(writes.len(), 4);
    for write in &writes {
        assert_eq!(write.voice_id, 1);
        assert_eq!(write.sample_offset, 0);
    }

    // Channel (+0.5) and per-note (+0.5) bend combine to full scale at the
    // default range of 2 semitones.
    assert_eq!(writes[0].target, ModulationTarget::Pitch);
    assert!(close(writes[0].value, 2.0, 1.0e-3));

    // Channel pressure 0x8000_0000 normalises to about one half.
    assert_eq!(writes[1].target, ModulationTarget::Gain);
    assert!(close(writes[1].value, 0.5, 1.0e-4));

    // Full brightness control change maps to unit timbre.
    assert_eq!(writes[2].target, ModulationTarget::Cutoff);
    assert!(close(writes[2].value, normalize_u32(0xFFFF_FFFF), 1.0e-6));

    // Velocity 0x9000 passes through the velocity dimension.
    assert_eq!(writes[3].target, ModulationTarget::Custom { slot: 0 });
    assert!(close(writes[3].value, normalize_u16(0x9000), 1.0e-6));
}

// ------------------------------------------------------------------------
// 2. Determinism: identical input yields bitwise-identical output.
// ------------------------------------------------------------------------

#[test]
fn pipeline_is_bitwise_deterministic() {
    let router = router_full();
    let stream = demo_stream();
    let first = pipeline_writes(&stream, 1, 60, 7, &router);
    let second = pipeline_writes(&stream, 1, 60, 7, &router);

    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.sample_offset, b.sample_offset);
        assert_eq!(a.voice_id, b.voice_id);
        assert_eq!(a.target, b.target);
        // Exact bit equality, never a float `==`.
        assert_eq!(a.value.to_bits(), b.value.to_bits());
    }
}

// ------------------------------------------------------------------------
// 3. Every MIDI 2.0 channel-voice variant decodes to the right shape.
// ------------------------------------------------------------------------

#[test]
fn all_midi2_channel_voice_variants_decode() {
    let voice = |word0: u32, word1: u32| match decode_midi2(UmpWord::new(word0), word1) {
        Some(MidiMessage::ChannelVoice { message, .. }) => message,
        other => panic!("unexpected {other:?}"),
    };

    assert_eq!(
        voice(0x4000_3C05, 0x1111_2222),
        ChannelVoice::RegisteredPerNoteController {
            note: 60,
            index: 5,
            value: 0x1111_2222,
        }
    );
    assert_eq!(
        voice(0x4010_3C06, 0x0000_3333),
        ChannelVoice::AssignablePerNoteController {
            note: 60,
            index: 6,
            value: 0x0000_3333,
        }
    );
    assert_eq!(
        voice(0x4020_0102, 0x4000_0000),
        ChannelVoice::RegisteredController {
            bank: 1,
            index: 2,
            value: 0x4000_0000,
        }
    );
    assert_eq!(
        voice(0x4030_0304, 0x5000_0000),
        ChannelVoice::AssignableController {
            bank: 3,
            index: 4,
            value: 0x5000_0000,
        }
    );
    assert_eq!(
        voice(0x4040_0506, 0xFFFF_FFFF),
        ChannelVoice::RelativeRegisteredController {
            bank: 5,
            index: 6,
            value: -1,
        }
    );
    assert_eq!(
        voice(0x4050_0708, 0x0000_0002),
        ChannelVoice::RelativeAssignableController {
            bank: 7,
            index: 8,
            value: 2,
        }
    );
    assert_eq!(
        voice(0x4060_3C00, 0xC000_0000),
        ChannelVoice::PerNotePitchBend {
            note: 60,
            bend: 0xC000_0000,
        }
    );
    assert_eq!(
        voice(0x4080_3C00, 0x2000_0000),
        ChannelVoice::NoteOff {
            note: 60,
            velocity: 0x2000,
            attribute: NoteAttribute::None,
        }
    );
    assert_eq!(
        voice(0x4090_3C01, 0x7000_ABCD),
        ChannelVoice::NoteOn {
            note: 60,
            velocity: 0x7000,
            attribute: NoteAttribute::ManufacturerSpecific { data: 0xABCD },
        }
    );
    assert_eq!(
        voice(0x40A0_3C00, 0x5000_0000),
        ChannelVoice::PolyPressure {
            note: 60,
            pressure: 0x5000_0000,
        }
    );
    assert_eq!(
        voice(0x40B0_4A00, 0xDEAD_BEEF),
        ChannelVoice::ControlChange {
            index: 74,
            value: 0xDEAD_BEEF,
        }
    );
    assert_eq!(
        voice(0x40C0_0001, 0x0700_0203),
        ChannelVoice::ProgramChange {
            program: 7,
            bank: Some((2 << 7) | 3),
        }
    );
    assert_eq!(
        voice(0x40D0_0000, 0x6000_0000),
        ChannelVoice::ChannelPressure {
            pressure: 0x6000_0000,
        }
    );
    assert_eq!(
        voice(0x40E0_0000, 0xA000_0000),
        ChannelVoice::PitchBend { bend: 0xA000_0000 }
    );
    assert_eq!(
        voice(0x40F0_3C03, 0x0000_0000),
        ChannelVoice::PerNoteManagement {
            note: 60,
            detach: true,
            reset: true,
        }
    );

    // The group and channel fields are surfaced on the envelope.
    match decode_midi2(UmpWord::new(0x4295_3C00), 0x9000_0000) {
        Some(MidiMessage::ChannelVoice {
            group, channel, ..
        }) => {
            assert_eq!(group, 2);
            assert_eq!(channel, 5);
        }
        other => panic!("unexpected {other:?}"),
    }
}

// ------------------------------------------------------------------------
// 4. MIDI 1.0 up-scales into the uniform full-resolution representation.
// ------------------------------------------------------------------------

#[test]
fn midi1_upscales_to_uniform_resolution() {
    let voice = |word: u32| match decode_midi1(UmpWord::new(word)) {
        Some(MidiMessage::ChannelVoice { message, .. }) => message,
        other => panic!("unexpected {other:?}"),
    };

    // Attack velocity 127 fills the 16-bit range.
    assert_eq!(
        voice(0x2090_3C7F),
        ChannelVoice::NoteOn {
            note: 60,
            velocity: 0xFFFF,
            attribute: NoteAttribute::None,
        }
    );
    // A note-on with zero velocity folds into a note-off.
    assert_eq!(
        voice(0x2090_3C00),
        ChannelVoice::NoteOff {
            note: 60,
            velocity: 0,
            attribute: NoteAttribute::None,
        }
    );
    // Release velocity up-scales from 7 to 16 bits.
    assert_eq!(
        voice(0x2080_3C40),
        ChannelVoice::NoteOff {
            note: 60,
            velocity: scale_up(0x40, 7, 16) as u16,
            attribute: NoteAttribute::None,
        }
    );
    // Control change up-scales to 32 bits.
    assert_eq!(
        voice(0x20B0_0740),
        ChannelVoice::ControlChange {
            index: 7,
            value: scale_up(0x40, 7, 32),
        }
    );
    // A centred 14-bit pitch bend lands exactly on the 32-bit centre.
    assert_eq!(
        voice(0x20E0_0040),
        ChannelVoice::PitchBend {
            bend: PITCH_BEND_CENTER_32,
        }
    );
    // Program change carries no bank in MIDI 1.0.
    assert_eq!(
        voice(0x20C0_0500),
        ChannelVoice::ProgramChange {
            program: 5,
            bank: None,
        }
    );
    assert_eq!(
        voice(0x20D0_4000),
        ChannelVoice::ChannelPressure {
            pressure: scale_up(0x40, 7, 32),
        }
    );
    assert_eq!(
        voice(0x20A0_3C40),
        ChannelVoice::PolyPressure {
            note: 60,
            pressure: scale_up(0x40, 7, 32),
        }
    );
}

// ------------------------------------------------------------------------
// 5. Utility and system messages decode across every status.
// ------------------------------------------------------------------------

#[test]
fn utility_and_system_messages_decode() {
    let util = |word: u32| match decode_utility(UmpWord::new(word)) {
        Some(MidiMessage::Utility(message)) => message,
        other => panic!("unexpected {other:?}"),
    };

    assert_eq!(util(0x0000_0000), UtilityMessage::NoOp);
    assert_eq!(util(0x0010_1234), UtilityMessage::JrClock { clock: 0x1234 });
    assert_eq!(
        util(0x0020_5678),
        UtilityMessage::JrTimestamp { timestamp: 0x5678 }
    );
    assert_eq!(
        util(0x0030_0960),
        UtilityMessage::DeltaClockstampTicksPerQuarterNote { ticks: 0x0960 }
    );
    assert_eq!(
        util(0x004A_BCDE),
        UtilityMessage::DeltaClockstamp { ticks: 0x000A_BCDE }
    );

    let sys = |word: u32| match decode_system(UmpWord::new(word)) {
        Some(MidiMessage::System { group, message }) => (group, message),
        other => panic!("unexpected {other:?}"),
    };

    assert_eq!(
        sys(0x10F1_0045),
        (0, SystemMessage::MidiTimeCode { data: 0x45 })
    );
    assert_eq!(
        sys(0x10F2_1234),
        (0, SystemMessage::SongPositionPointer { position: 0x1A12 })
    );
    assert_eq!(sys(0x10F3_0012), (0, SystemMessage::SongSelect { song: 0x12 }));
    assert_eq!(sys(0x10F6_0000), (0, SystemMessage::TuneRequest));
    // Group field is extracted independently of the status byte.
    assert_eq!(sys(0x13F8_0000), (3, SystemMessage::TimingClock));
    assert_eq!(sys(0x10FA_0000), (0, SystemMessage::Start));
    assert_eq!(sys(0x10FB_0000), (0, SystemMessage::Continue));
    assert_eq!(sys(0x10FC_0000), (0, SystemMessage::Stop));
    assert_eq!(sys(0x10FE_0000), (0, SystemMessage::ActiveSensing));
    assert_eq!(sys(0x10FF_0000), (0, SystemMessage::Reset));
}

// ------------------------------------------------------------------------
// 6. Byte and word streams agree; the decoder never loses alignment.
// ------------------------------------------------------------------------

#[test]
fn decoder_byte_and_word_streams_agree_and_realign() {
    let words = [
        0x4091_3C00u32,
        0x9000_0000, // 2-word MIDI 2.0 note on
        0x20D0_4000, // 1-word MIDI 1.0 channel pressure
        0x2080_3C40, // 1-word MIDI 1.0 note off
    ];

    let mut from_words = Vec::new();
    let mut word_decoder = UmpDecoder::new();
    word_decoder.push_words(&words, &mut from_words);

    let mut from_bytes = Vec::new();
    let mut byte_decoder = UmpDecoder::new();
    byte_decoder.push_bytes(&words_to_bytes(&words), &mut from_bytes);

    assert_eq!(from_words, from_bytes);
    assert_eq!(from_words.len(), 3);

    // A 64-bit data packet (2 words) and a reserved 3-word packet are consumed
    // whole without surfacing, and the following note-on still decodes: proof
    // that alignment held across unsupported message types.
    let realign = [
        0x3000_0000u32,
        0x0000_0000, // Data64 (SysEx7), two words, not surfaced
        0xB000_0000,
        0x0000_0000,
        0x0000_0000, // reserved type 0xB, three words, not surfaced
        0x2090_3C7F, // MIDI 1.0 note on, must still decode
    ];
    let mut realigned = Vec::new();
    let mut decoder = UmpDecoder::new();
    decoder.push_words(&realign, &mut realigned);
    assert_eq!(realigned.len(), 1);
    assert!(matches!(
        realigned[0],
        MidiMessage::ChannelVoice {
            message: ChannelVoice::NoteOn { note: 60, .. },
            ..
        }
    ));

    // `reset` discards a partially assembled packet.
    let mut partial = UmpDecoder::new();
    assert!(partial.push_word(0x4091_3C00).is_none());
    partial.reset();
    assert!(partial.push_word(0x2090_3C7F).is_some());

    // A word is only emitted once all four of its bytes arrive.
    let mut by_byte = UmpDecoder::new();
    assert!(by_byte.push_byte(0x20).is_none());
    assert!(by_byte.push_byte(0x90).is_none());
    assert!(by_byte.push_byte(0x3C).is_none());
    assert!(by_byte.push_byte(0x7F).is_some());
}

// ------------------------------------------------------------------------
// 7. Boundary word-count table and malformed / unknown statuses.
// ------------------------------------------------------------------------

#[test]
fn message_type_boundaries_and_malformed_statuses() {
    // The message-type nibble round-trips across the full range.
    for n in 0u8..16 {
        assert_eq!(MessageType::from_nibble(n).nibble(), n);
    }
    // Word-count table, including the reserved-range boundaries.
    assert_eq!(MessageType::Utility.word_count(), 1);
    assert_eq!(MessageType::Midi1ChannelVoice.word_count(), 1);
    assert_eq!(MessageType::Data64.word_count(), 2);
    assert_eq!(MessageType::Midi2ChannelVoice.word_count(), 2);
    assert_eq!(MessageType::Data128.word_count(), 4);
    assert_eq!(MessageType::Reserved(0x6).word_count(), 1);
    assert_eq!(MessageType::Reserved(0xB).word_count(), 3);

    // Typed field extraction from a MIDI 2.0 note-on word.
    let word = UmpWord::new(0x4295_3C00);
    assert_eq!(word.message_type(), MessageType::Midi2ChannelVoice);
    assert_eq!(word.group(), 2);
    assert_eq!(word.status_nibble(), 0x9);
    assert_eq!(word.channel(), 5);
    assert_eq!(word.byte2(), 0x3C);
    assert_eq!(word.raw(), 0x4295_3C00);

    // Unrecognised statuses decode to `None` without panicking.
    assert!(decode_utility(UmpWord::new(0x0050_0000)).is_none());
    assert!(decode_system(UmpWord::new(0x10F0_0000)).is_none());
    assert!(decode_midi1(UmpWord::new(0x2000_0000)).is_none());
    assert!(decode_midi2(UmpWord::new(0x4070_0000), 0).is_none());
}

// ------------------------------------------------------------------------
// 8. MPE zone layout and deterministic member-channel allocation.
// ------------------------------------------------------------------------

#[test]
fn mpe_allocation_and_zone_layout() {
    let zone = MpeZone::lower(4);
    assert_eq!(zone.kind(), ZoneKind::Lower);
    assert_eq!(zone.manager_channel(), 0);
    assert!(zone.is_manager(0));
    assert_eq!(zone.member_channel(0), Some(1));
    assert_eq!(zone.member_channel(3), Some(4));
    assert_eq!(zone.member_channel(4), None);
    assert!(zone.is_member(1));
    assert!(zone.is_member(4));
    assert!(!zone.is_member(5));

    let mut alloc = MpeAllocator::new(zone);
    assert_eq!(alloc.zone().member_count(), 4);
    // Round-robin spreads consecutive notes across the member channels.
    assert_eq!(alloc.allocate(), Some(1));
    assert_eq!(alloc.allocate(), Some(2));
    assert_eq!(alloc.allocate(), Some(3));
    assert_eq!(alloc.allocate(), Some(4));
    assert_eq!(alloc.total_active(), 4);
    for channel in 1..=4u8 {
        assert_eq!(alloc.active_notes(channel), 1);
    }
    // Releasing frees a channel for reuse; releasing a non-member fails.
    assert!(alloc.release(2));
    assert_eq!(alloc.active_notes(2), 0);
    assert_eq!(alloc.allocate(), Some(2));
    assert!(!alloc.release(0));

    // When every member is loaded, stacking lands on the least-loaded channel,
    // ties broken by longest-idle then lowest ordinal.
    let mut full = MpeAllocator::new(MpeZone::lower(2));
    assert_eq!(full.allocate(), Some(1));
    assert_eq!(full.allocate(), Some(2));
    assert_eq!(full.allocate(), Some(1));
    assert_eq!(full.active_notes(1), 2);
    assert_eq!(full.allocate(), Some(2));
    assert_eq!(full.active_notes(2), 2);

    // The upper zone counts members downward from channel 14.
    let mut upper = MpeAllocator::new(MpeZone::upper(3));
    assert_eq!(upper.zone().manager_channel(), 15);
    assert_eq!(upper.allocate(), Some(14));
    assert_eq!(upper.allocate(), Some(13));
    assert_eq!(upper.allocate(), Some(12));

    // Reconfiguring clears all load.
    upper.reconfigure(MpeZone::lower(4));
    assert_eq!(upper.total_active(), 0);
    assert_eq!(upper.allocate(), Some(1));

    // Member counts clamp to the single-zone range.
    assert_eq!(MpeZone::lower(0).member_count(), 1);
    assert_eq!(MpeZone::upper(200).member_count(), 15);
}

// ------------------------------------------------------------------------
// 9. Per-channel high-resolution controllers and RPN tracking.
// ------------------------------------------------------------------------

#[test]
fn channel_state_tracks_highres_and_rpn() {
    let mut state = ChannelState::new();
    assert!(close(
        state.pitch_bend_range(),
        DEFAULT_PITCH_BEND_RANGE,
        1.0e-6
    ));
    // A full positive channel bend is +2 semitones at the default range.
    state.apply(&ChannelVoice::PitchBend { bend: 0xFFFF_FFFF });
    assert_eq!(state.pitch_bend_raw(), 0xFFFF_FFFF);
    assert!(close(state.pitch_bend_semitones(), 2.0, 1.0e-3));

    // The MIDI 1.0 RPN path (CC 101/100 select, CC 6 data entry) sets the
    // pitch-bend range to 12 semitones.
    state.apply(&ChannelVoice::ControlChange {
        index: 101,
        value: 0,
    });
    state.apply(&ChannelVoice::ControlChange {
        index: 100,
        value: 0,
    });
    state.apply(&ChannelVoice::ControlChange {
        index: 6,
        value: scale_up(12, 7, 32),
    });
    assert!(close(state.pitch_bend_range(), 12.0, 1.0e-4));

    // A MIDI 2.0 registered controller sets the same range directly.
    let mut direct = ChannelState::new();
    direct.apply(&ChannelVoice::RegisteredController {
        bank: 0,
        index: 0,
        value: scale_up(5 << 7, 14, 32),
    });
    assert!(close(direct.pitch_bend_range(), 5.0, 1.0e-4));

    // A coarse/fine controller pair recombines into one 14-bit value.
    let mut pair = ChannelState::new();
    pair.apply(&ChannelVoice::ControlChange {
        index: 1,
        value: scale_up(0x40, 7, 32),
    });
    pair.apply(&ChannelVoice::ControlChange {
        index: 33,
        value: scale_up(0x7F, 7, 32),
    });
    assert_eq!(
        pair.high_res_controller(1),
        scale_up((0x40 << 7) | 0x7F, 14, 32)
    );
    assert_eq!(pair.controller(1), scale_up(0x40, 7, 32));

    // Program and bank selection are tracked.
    let mut program = ChannelState::new();
    program.apply(&ChannelVoice::ProgramChange {
        program: 42,
        bank: Some(300),
    });
    assert_eq!(program.program(), 42);
    assert_eq!(program.bank(), 300);
}

// ------------------------------------------------------------------------
// 10. Per-note slot table lifecycle: update, free, detach, reset, steal.
// ------------------------------------------------------------------------

#[test]
fn per_note_table_lifecycle() {
    let mut table = PerNoteExpression::new();
    assert_eq!(table.active_count(), 0);
    table.note_on(1, 60, 0x8000);
    assert_eq!(table.active_count(), 1);
    assert!(table.set_pitch_bend(1, 60, 0xC000_0000));
    assert!(table.set_pressure(1, 60, 0x4000_0000));
    assert!(table.set_controller(1, 60, PerNoteController::Brightness.index(), 0x1234));
    let state = table.get(1, 60).expect("active");
    assert_eq!(state.pitch_bend, 0xC000_0000);
    assert_eq!(state.pressure, 0x4000_0000);
    assert_eq!(state.velocity, 0x8000);
    assert_eq!(
        state.named_controller(PerNoteController::Brightness),
        Some(0x1234)
    );

    // Updates miss an inactive note.
    assert!(!table.set_pitch_bend(2, 99, 0));

    // A plain note-off frees the slot.
    assert!(table.note_off(1, 60));
    assert_eq!(table.active_count(), 0);
    assert!(table.get(1, 60).is_none());

    // A detached note keeps its slot after note-off.
    table.note_on(0, 64, 0x4000);
    assert!(table.manage(0, 64, true, false));
    assert!(table.note_off(0, 64));
    assert!(table.set_controller(0, 64, 74, 0x10));

    // A reset restores defaults while keeping the note alive.
    table.note_on(0, 70, 0x2000);
    table.set_pitch_bend(0, 70, 0x1000_0000);
    table.set_pressure(0, 70, 0x2000_0000);
    assert!(table.manage(0, 70, false, true));
    let reset = table.get(0, 70).expect("active");
    assert_eq!(reset.pitch_bend, PITCH_BEND_CENTER_32);
    assert_eq!(reset.pressure, 0);

    // Controller capacity: distinct controllers fill the fixed table. The
    // table-level call reports a note hit, but a ninth distinct controller is
    // silently dropped and so never reads back, while updating an already
    // present controller still works at capacity.
    let mut cap = PerNoteExpression::new();
    cap.note_on(5, 80, 0x100);
    for index in 0..MAX_PER_NOTE_CONTROLLERS as u8 {
        assert!(cap.set_controller(5, 80, index, u32::from(index)));
    }
    assert!(cap.set_controller(5, 80, 99, 1));
    assert_eq!(cap.get(5, 80).and_then(|s| s.controller(99)), None);
    assert!(cap.set_controller(5, 80, 0, 777));
    assert_eq!(cap.get(5, 80).and_then(|s| s.controller(0)), Some(777));

    // A full table steals the oldest slot and stays full.
    let mut steal = PerNoteExpression::new();
    for note in 0..MAX_ACTIVE_NOTES {
        steal.note_on(0, note as u8, 0x100);
    }
    assert_eq!(steal.active_count(), MAX_ACTIVE_NOTES);
    steal.note_on(1, 100, 0x200);
    assert_eq!(steal.active_count(), MAX_ACTIVE_NOTES);
    assert!(steal.get(0, 0).is_none());
    assert!(steal.get(1, 100).is_some());
}

// ------------------------------------------------------------------------
// 11. Min-Center-Max resolution scaling invariants.
// ------------------------------------------------------------------------

#[test]
fn scaling_min_center_max_invariants() {
    // 7-bit endpoints and centre into 16 bits.
    assert_eq!(scale_up(0x00, 7, 16), 0x0000);
    assert_eq!(scale_up(0x40, 7, 16), 0x8000);
    assert_eq!(scale_up(0x7F, 7, 16), 0xFFFF);
    // 7-bit into 32 bits.
    assert_eq!(scale_up(0x00, 7, 32), 0x0000_0000);
    assert_eq!(scale_up(0x40, 7, 32), 0x8000_0000);
    assert_eq!(scale_up(0x7F, 7, 32), 0xFFFF_FFFF);
    // 14-bit into 32 bits.
    assert_eq!(scale_up(0x2000, 14, 32), 0x8000_0000);
    assert_eq!(scale_up(0x3FFF, 14, 32), 0xFFFF_FFFF);

    // Up-scaling is monotonic non-decreasing.
    let mut last = 0u32;
    for v in 0u32..=0x7F {
        let up = scale_up(v, 7, 32);
        assert!(up >= last);
        last = up;
    }

    // Down-scaling inverts the endpoints and is a plain truncation.
    assert_eq!(scale_down(scale_up(0x00, 7, 32), 32, 7), 0x00);
    assert_eq!(scale_down(scale_up(0x7F, 7, 32), 32, 7), 0x7F);
    assert_eq!(scale_down(0x8000, 16, 7), 0x40);

    // Degenerate widths stay total and never panic.
    assert_eq!(scale_up(5, 0, 32), 0);
    assert_eq!(scale_up(0x40, 7, 7), 0x40);
    assert_eq!(scale_down(0x40, 7, 16), 0x40);
}

// ------------------------------------------------------------------------
// 12. Router transform and sample-offset ordering invariants.
// ------------------------------------------------------------------------

#[test]
fn router_transform_and_sample_offsets_preserved() {
    let router = router_full();
    let mapping_count = router.mappings().len();
    assert_eq!(mapping_count, 4);

    let mut voice = VoiceExpression::new(9, 72);
    voice.pitch_bend_range_semitones = 2.0;
    voice.channel_pitch_bend = u32::MAX;
    voice.per_note_pressure = u32::MAX;
    voice.timbre = u32::MAX;
    voice.velocity = u16::MAX;

    let offsets = [0u32, 64, 128, 256];
    let mut writes = Vec::new();
    for &offset in &offsets {
        router.route(&voice, offset, &mut writes);
    }

    // One write per mapping per routed offset.
    assert_eq!(writes.len(), offsets.len() * mapping_count);

    // Sample offsets are preserved and emitted non-decreasing; the voice id is
    // attributed to every write.
    let mut prev = 0u32;
    for write in &writes {
        assert!(write.sample_offset >= prev);
        prev = write.sample_offset;
        assert_eq!(write.voice_id, 9);
    }

    // Each per-offset block preserves the mapping order deterministically.
    for (block, &offset) in offsets.iter().enumerate() {
        let base = block * mapping_count;
        assert_eq!(writes[base].target, ModulationTarget::Pitch);
        assert_eq!(writes[base + 1].target, ModulationTarget::Gain);
        assert_eq!(writes[base + 2].target, ModulationTarget::Cutoff);
        assert_eq!(writes[base + 3].target, ModulationTarget::Custom { slot: 0 });
        assert_eq!(writes[base].sample_offset, offset);
    }
    // Full channel bend with a centred per-note bend is about +2 semitones.
    assert!(close(writes[0].value, 2.0, 1.0e-3));

    // A shaped mapping applies curve, depth, and offset while preserving sign.
    let shaped = TargetMapping::new(
        ExpressionDimension::Pressure,
        ModulationTarget::Gain,
        0.5,
        0.25,
        Curve::Exponential { exponent: 2.0 },
    );
    // 0.5 -> 0.25 (squared) -> *0.5 + 0.25 = 0.375.
    assert!(close(shaped.apply(0.5), 0.375, 1.0e-6));
    assert!(close(shaped.curve.shape(-0.5), -0.25, 1.0e-6));
}
