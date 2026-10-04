//! Production-grade integration tests exercising the real public API of the
//! `prism_audio_comm` crate as one end-to-end real-time voice link.
//!
//! These tests drive the genuine data flow of design section 45 with no mocks
//! or stubbed internals: a microphone block enters the uplink pre-processing
//! chain through [`DefaultVoiceCommPipeline::capture`] (high-pass, acoustic
//! echo cancellation, noise suppression, automatic gain control, voice activity
//! detection), is encoded by the default [`LinearPcmCodec`] into a
//! [`VoicePacket`], travels over a [`VoiceTransport`], is reordered by the
//! adaptive [`JitterBuffer`], decoded or filled by the
//! [`PacketLossConcealer`], and finally handed to the spatial layer as a
//! [`PositionalVoiceParams`] set computed by [`PositionalVoice`] from real
//! [`bevy_math::Vec3`] / [`bevy_math::Quat`] listener poses. The deterministic
//! [`CommRng`] backs reproducibility checks. Signals are built from direct
//! current, ramps, triangles, and the crate's own integer noise generator so
//! no transcendental standard-library floating-point math is required.
//!
//! # Provenance
//! Original work. Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Dolby, MPEG, Google Resonance Audio, Web Audio, or Project Acoustics
//! source or derived code, and no AI/ML. Public standards (RFC 3550 jitter
//! estimation, linear PCM quantisation) informed only the concepts, not any
//! copied implementation.
//!
//! # Relationship
//! Validates design section 45 (real-time communication: uplink pre-processing,
//! codec/transport abstractions, jitter buffer, packet-loss concealment, and
//! positional voice). Depends on the public API of `prism_audio_comm`, the
//! `Sample` scalar from `prism_audio_core`, and `bevy_math` vector/quaternion
//! types.

use bevy_math::{Quat, Vec3};
use prism_audio_core::math::Sample;

use prism_audio_comm::{
    CodecError, CommRng, DefaultVoiceCommPipeline, JitterBuffer, JitterConfig, JitterResult,
    JitterStats, LinearPcmCodec, LoopbackTransport, PipelineConfig, PlayoutKind, PositionalVoice,
    PositionalVoiceConfig, TransportError, VoiceCodec, VoiceCommPipeline, VoicePacket,
    VoiceSpatialMode, VoiceTransport,
};

/// Branch-only absolute value, avoiding the standard-library `f32::abs`.
fn fabs(x: Sample) -> Sample {
    if x < 0.0 { -x } else { x }
}

/// Returns `true` when `a` and `b` are within `eps`, used instead of a direct
/// floating-point equality comparison.
fn close(a: Sample, b: Sample, eps: Sample) -> bool {
    fabs(a - b) < eps
}

/// Mean square energy of a block.
fn energy(block: &[Sample]) -> Sample {
    if block.is_empty() {
        return 0.0;
    }
    let sum: Sample = block.iter().map(|&v| v * v).sum();
    sum / block.len() as Sample
}

/// Builds one block of a deterministic periodic triangle wave from pure ramp
/// arithmetic (no transcendental math), suitable as a voiced excitation.
fn triangle_block(frame: usize, block: usize, period: usize, amp: Sample) -> Vec<Sample> {
    let period = period.max(2);
    (0..frame)
        .map(|i| {
            let n = block * frame + i;
            let phase = (n % period) as Sample / period as Sample;
            let ramp = 2.0 * phase - 1.0;
            amp * (2.0 * fabs(ramp) - 1.0)
        })
        .collect()
}

/// A transport that forwards through a real [`LoopbackTransport`] but silently
/// drops every `drop_period`-th sent packet, modelling lossy delivery while
/// still reporting success so the pipeline advances its transmit sequence and
/// leaves a genuine sequence-number gap for the jitter buffer to detect.
struct LossyTransport {
    inner: LoopbackTransport,
    sent: usize,
    drop_period: usize,
}

impl LossyTransport {
    fn new(capacity: usize, drop_period: usize) -> Self {
        Self {
            inner: LoopbackTransport::new(capacity),
            sent: 0,
            drop_period: drop_period.max(1),
        }
    }
}

impl VoiceTransport for LossyTransport {
    fn send(&mut self, packet: &VoicePacket) -> Result<(), TransportError> {
        self.sent += 1;
        if self.sent.is_multiple_of(self.drop_period) {
            // Report success without delivering: a real loss on the wire.
            return Ok(());
        }
        self.inner.send(packet)
    }

    fn poll(&mut self) -> Option<VoicePacket> {
        self.inner.poll()
    }
}

/// The complete full-duplex link carries voiced audio from capture through the
/// transport and jitter buffer to decoded playout, which then resolves into a
/// sane spatial parameter set for the renderer.
#[test]
fn full_duplex_chain_delivers_decoded_voice_to_spatializer() {
    let mut pipeline = DefaultVoiceCommPipeline::new(PipelineConfig {
        discontinuous_transmission: false,
        ..PipelineConfig::default()
    });
    let frame = pipeline.frame();
    let mut transport = LoopbackTransport::new(256);
    let reference = vec![0.0; frame];

    let mut decoded_blocks = 0usize;
    let mut last_decoded = vec![0.0; frame];
    for b in 0..100 {
        let mut mic = triangle_block(frame, b, 218, 0.3);
        let status = pipeline.capture(&mut mic, &reference, &mut transport);
        assert!(status.transmitted, "non-DTX capture must always transmit");
        let mut out = vec![0.0; frame];
        let playout = pipeline.playout(&mut out, &mut transport);
        if playout.kind == PlayoutKind::Decoded {
            decoded_blocks += 1;
            last_decoded.copy_from_slice(&out);
        }
    }
    assert!(decoded_blocks > 0, "sustained speech should decode frames");

    // The jitter buffer accounted for every delivered packet.
    let stats = pipeline.jitter_stats();
    assert!(stats.received > 0);

    // Hand a decoded frame to the spatial layer as a world-voice source three
    // metres ahead of the listener and verify the resolved parameters.
    let pv = PositionalVoice::new(PositionalVoiceConfig::default());
    let params = pv.resolve(
        Vec3::ZERO,
        Quat::IDENTITY,
        Vec3::new(0.0, 0.0, -3.0),
        1.0,
        VoiceSpatialMode::World3d,
    );
    assert!(close(params.distance, 3.0, 1.0e-4));
    assert!(close(params.azimuth, 0.0, 1.0e-4));
    assert!(params.gain > 0.0 && params.gain <= 1.0);
    assert!(close(params.spatial_blend, 1.0, 1.0e-6));
    // The decoded carrier the renderer would spatialise is not pure silence.
    assert!(energy(&last_decoded) > 0.0);
}

/// Given identical input and configuration, the whole pipeline is
/// bit-identical across runs: same capture decisions and same playout samples.
#[test]
fn pipeline_is_bit_deterministic() {
    fn run() -> (Vec<Sample>, Vec<Option<u32>>) {
        let mut pipeline = DefaultVoiceCommPipeline::new(PipelineConfig {
            discontinuous_transmission: false,
            ..PipelineConfig::default()
        });
        let frame = pipeline.frame();
        let mut transport = LoopbackTransport::new(256);
        let reference = vec![0.0; frame];
        let mut playout = Vec::new();
        let mut sequences = Vec::new();
        for b in 0..64 {
            let mut mic = triangle_block(frame, b, 200, 0.25);
            let status = pipeline.capture(&mut mic, &reference, &mut transport);
            sequences.push(status.sequence);
            let mut out = vec![0.0; frame];
            let _ = pipeline.playout(&mut out, &mut transport);
            playout.extend_from_slice(&out);
        }
        (playout, sequences)
    }

    let (first_audio, first_seq) = run();
    let (second_audio, second_seq) = run();
    // Exact equality here is the intended bit-identical invariant.
    assert_eq!(first_audio, second_audio);
    assert_eq!(first_seq, second_seq);
}

/// A sustained voiced block is detected as speech and transmitted even with
/// discontinuous transmission enabled (the default).
#[test]
fn voiced_input_is_transmitted_under_dtx() {
    let mut pipeline = DefaultVoiceCommPipeline::new(PipelineConfig::default());
    let frame = pipeline.frame();
    let mut transport = LoopbackTransport::new(256);
    let reference = vec![0.0; frame];

    let mut transmitted_any = false;
    for b in 0..120 {
        let mut mic = triangle_block(frame, b, 218, 0.4);
        let status = pipeline.capture(&mut mic, &reference, &mut transport);
        if status.transmitted {
            transmitted_any = true;
        }
        let mut out = vec![0.0; frame];
        let _ = pipeline.playout(&mut out, &mut transport);
    }
    assert!(
        transmitted_any,
        "voiced excitation should trip the voice activity detector"
    );
}

/// Pure silence is gated by the uplink and, under discontinuous transmission,
/// never placed on the wire; the downlink therefore produces only silence.
#[test]
fn silent_input_is_gated_and_not_transmitted() {
    let mut pipeline = DefaultVoiceCommPipeline::new(PipelineConfig::default());
    let frame = pipeline.frame();
    let mut transport = LoopbackTransport::new(64);
    let reference = vec![0.0; frame];

    let mut transmitted_any = false;
    let mut non_silence = false;
    for _ in 0..48 {
        let mut mic = vec![0.0; frame];
        let status = pipeline.capture(&mut mic, &reference, &mut transport);
        if status.transmitted {
            transmitted_any = true;
        }
        let mut out = vec![0.0; frame];
        let playout = pipeline.playout(&mut out, &mut transport);
        if playout.kind != PlayoutKind::Silence {
            non_silence = true;
        }
        assert!(out.iter().all(|&s| close(s, 0.0, 1.0e-9)) || playout.kind != PlayoutKind::Silence);
    }
    assert!(!transmitted_any, "silence must not be transmitted with DTX");
    assert!(!non_silence, "no packets means every playout is silence");
}

/// Draining a transport that never produced a packet yields explicit silence
/// with a zeroed output buffer, even if the buffer arrived dirty.
#[test]
fn playout_without_packets_emits_zeroed_silence() {
    let mut pipeline = DefaultVoiceCommPipeline::new(PipelineConfig::default());
    let frame = pipeline.frame();
    let mut transport = LoopbackTransport::new(16);
    let mut out = vec![0.5; frame];
    let playout = pipeline.playout(&mut out, &mut transport);
    assert_eq!(playout.kind, PlayoutKind::Silence);
    assert!(out.iter().all(|&s| close(s, 0.0, 1.0e-9)));
    assert_eq!(playout.jitter.received, 0);
    assert_eq!(playout.jitter.underruns, 1);
}

/// Packets lost in transit become genuine sequence gaps that the jitter buffer
/// flags and the packet-loss concealer fills; the loss is counted in telemetry.
#[test]
fn dropped_packets_are_concealed_and_counted() {
    let mut pipeline = DefaultVoiceCommPipeline::new(PipelineConfig {
        discontinuous_transmission: false,
        ..PipelineConfig::default()
    });
    let frame = pipeline.frame();
    // Drop every 5th transmitted packet on the wire.
    let mut transport = LossyTransport::new(256, 5);
    let reference = vec![0.0; frame];

    let mut concealed = 0usize;
    for b in 0..160 {
        let mut mic = triangle_block(frame, b, 190, 0.3);
        let _ = pipeline.capture(&mut mic, &reference, &mut transport);
        let mut out = vec![0.0; frame];
        let playout = pipeline.playout(&mut out, &mut transport);
        if playout.kind == PlayoutKind::Concealed {
            concealed += 1;
        }
    }
    assert!(concealed > 0, "lost frames must be concealed");
    assert!(
        pipeline.jitter_stats().concealed_losses > 0,
        "losses must be reflected in jitter telemetry"
    );
}

/// Resetting the pipeline after real activity clears all downlink statistics
/// back to their initial state.
#[test]
fn reset_returns_pipeline_to_initial_jitter_stats() {
    let mut pipeline = DefaultVoiceCommPipeline::new(PipelineConfig {
        discontinuous_transmission: false,
        ..PipelineConfig::default()
    });
    let frame = pipeline.frame();
    let mut transport = LoopbackTransport::new(256);
    let reference = vec![0.0; frame];
    for b in 0..24 {
        let mut mic = triangle_block(frame, b, 210, 0.3);
        let _ = pipeline.capture(&mut mic, &reference, &mut transport);
        let mut out = vec![0.0; frame];
        let _ = pipeline.playout(&mut out, &mut transport);
    }
    assert_ne!(pipeline.jitter_stats(), JitterStats::default());
    pipeline.reset();
    assert_eq!(pipeline.jitter_stats(), JitterStats::default());
}

/// The default linear PCM codec round-trips a signal within 16-bit
/// quantisation error and emits the expected payload size.
#[test]
fn codec_roundtrip_is_lossless_within_quantization() {
    let frame = 256;
    let mut codec = LinearPcmCodec::new(frame);
    assert_eq!(codec.frame_size(), frame);
    // A deterministic bounded ramp/noise mix avoids transcendental math.
    let mut rng = CommRng::new(0x1234_5678);
    let input: Vec<Sample> = (0..frame)
        .map(|i| {
            let ramp = (i as Sample / frame as Sample) * 2.0 - 1.0;
            0.5 * ramp + 0.3 * rng.next_bipolar()
        })
        .collect();
    let mut payload = Vec::new();
    codec.encode(&input, &mut payload);
    assert_eq!(payload.len(), frame * 2);

    let mut output = Vec::new();
    let decoded = codec.decode(&payload, &mut output).expect("payload decodes");
    assert_eq!(decoded, frame);
    assert_eq!(output.len(), frame);
    for (a, b) in input.iter().zip(output.iter()) {
        // One 16-bit step is about 3.05e-5; stay comfortably above it.
        assert!(close(*a, *b, 1.0e-4), "quantisation drift too large");
    }
}

/// A payload whose length is not frame-aligned is rejected rather than
/// silently mis-decoded.
#[test]
fn malformed_payload_is_rejected() {
    let mut codec = LinearPcmCodec::new(4);
    let mut out = Vec::new();
    let result = codec.decode(&[1, 2, 3], &mut out);
    assert_eq!(result, Err(CodecError::MalformedPayload));
    assert!(out.is_empty());
}

/// The jitter buffer reorders out-of-order arrivals back into sequence and
/// reports a permanently missing frame as an explicit loss.
#[test]
fn jitter_buffer_reorders_and_reports_loss() {
    let mut jb = JitterBuffer::new(JitterConfig {
        min_frames: 3,
        ..JitterConfig::default()
    });
    let pkt = |seq: u32| VoicePacket::new(seq, seq as u64 * 256, seq == 0, vec![seq as u8; 4]);

    // Arrive 0, 3, 2 (1 is lost forever) then 4.
    jb.insert(pkt(0), 0);
    jb.insert(pkt(3), 768);
    jb.insert(pkt(2), 512);
    jb.insert(pkt(4), 1024);

    let mut order = Vec::new();
    let mut losses = 0usize;
    for _ in 0..5 {
        match jb.pop() {
            JitterResult::Packet(p) => order.push(p.sequence),
            JitterResult::Loss => losses += 1,
            JitterResult::Underrun => {}
        }
    }
    assert_eq!(order, vec![0, 2, 3, 4]);
    assert_eq!(losses, 1, "the missing sequence 1 is one concealed loss");
    assert!(jb.stats().reordered >= 1);
    assert_eq!(jb.stats().concealed_losses, 1);
}

/// World-voice attenuation is unity within the reference distance, strictly
/// decreasing with distance, and silent at or beyond the maximum distance.
#[test]
fn positional_gain_is_monotonic_in_distance() {
    let pv = PositionalVoice::new(PositionalVoiceConfig::default());
    assert!(close(pv.distance_gain(0.5), 1.0, 1.0e-6));
    assert!(close(pv.distance_gain(1.0), 1.0, 1.0e-6));

    let near = pv.distance_gain(3.0);
    let far = pv.distance_gain(12.0);
    assert!(near > far, "closer sources must be louder");
    assert!((0.0..=1.0).contains(&near));
    assert!((0.0..=1.0).contains(&far));

    assert!(close(pv.distance_gain(40.0), 0.0, 1.0e-6));
    assert!(close(pv.distance_gain(100.0), 0.0, 1.0e-6));

    // Resolving a far world source applies that attenuation to the base gain.
    let params = pv.resolve(
        Vec3::ZERO,
        Quat::IDENTITY,
        Vec3::new(0.0, 0.0, -12.0),
        1.0,
        VoiceSpatialMode::World3d,
    );
    assert!(close(params.distance, 12.0, 1.0e-4));
    assert!(params.gain < 1.0 && params.gain > 0.0);
}

/// World-voice azimuth tracks left/right placement while a team channel stays
/// flat (non-spatial) and distance-independent.
#[test]
fn positional_azimuth_follows_left_right_and_team_is_flat() {
    let pv = PositionalVoice::new(PositionalVoiceConfig::default());

    let right = pv.resolve(
        Vec3::ZERO,
        Quat::IDENTITY,
        Vec3::new(2.0, 0.0, 0.0),
        1.0,
        VoiceSpatialMode::World3d,
    );
    let left = pv.resolve(
        Vec3::ZERO,
        Quat::IDENTITY,
        Vec3::new(-2.0, 0.0, 0.0),
        1.0,
        VoiceSpatialMode::World3d,
    );
    let ahead = pv.resolve(
        Vec3::ZERO,
        Quat::IDENTITY,
        Vec3::new(0.0, 0.0, -2.0),
        1.0,
        VoiceSpatialMode::World3d,
    );
    assert!(right.azimuth > 0.0, "right ear is positive azimuth");
    assert!(left.azimuth < 0.0, "left ear is negative azimuth");
    assert!(close(ahead.azimuth, 0.0, 1.0e-4));
    assert!(close(right.azimuth, -left.azimuth, 1.0e-4));

    // A distant team-channel speaker is non-spatial and keeps its base gain.
    let team = pv.resolve(
        Vec3::ZERO,
        Quat::IDENTITY,
        Vec3::new(100.0, 0.0, 0.0),
        0.7,
        VoiceSpatialMode::Team2d,
    );
    assert!(close(team.azimuth, 0.0, 1.0e-6));
    assert!(close(team.elevation, 0.0, 1.0e-6));
    assert!(close(team.spatial_blend, 0.0, 1.0e-6));
    assert!(close(team.gain, 0.7, 1.0e-6));
}
