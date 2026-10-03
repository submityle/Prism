//! End-to-end voice communication pipeline tying the uplink and downlink
//! together.
//!
//! Design section 45.1 describes two cooperating paths. The uplink captures the
//! microphone, runs the classic pre-processing chain (high-pass, echo
//! cancellation, noise suppression, automatic gain control, voice activity
//! detection), encodes speech frames, and hands them to the host transport. The
//! downlink pulls received packets off the transport, feeds them through the
//! adaptive jitter buffer, decodes the in-order frame or conceals a lost one,
//! and returns playable PCM that the engine then spatialises (see
//! [`crate::positional`]). Codec and network work never touch the audio
//! callback: the real-time thread only moves already-decoded PCM.
//!
//! This module defines the [`VoiceCommPipeline`] trait and a complete default
//! implementation, [`DefaultVoiceCommPipeline`], that assembles the pieces from
//! the other modules with a single fixed frame size and no per-call heap
//! allocation after warm-up.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the design section 45.1 overview. Composes [`crate::uplink`],
//! [`crate::transport`], and (downstream) [`crate::positional`] on top of
//! `prism_audio_core`.

use prism_audio_core::math::Sample;

use crate::transport::codec::{LinearPcmCodec, VoiceCodec};
use crate::transport::jitter_buffer::{JitterBuffer, JitterConfig, JitterResult, JitterStats};
use crate::transport::packet::{VoicePacket, VoiceTransport};
use crate::transport::plc::{PacketLossConcealer, PlcConfig};
use crate::uplink::{UplinkChain, UplinkConfig, UplinkStatus};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Outcome of a single [`VoiceCommPipeline::capture`] call.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CaptureStatus {
    /// The uplink chain's per-block status (voice activity, echo convergence).
    pub uplink: UplinkStatus,
    /// Whether an encoded packet was handed to the transport this block.
    pub transmitted: bool,
    /// The sequence number assigned to the transmitted packet, if any.
    pub sequence: Option<u32>,
}

/// How a downlink playout frame was produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum PlayoutKind {
    /// The in-order packet was available and decoded.
    Decoded,
    /// The frame was lost and filled by packet-loss concealment.
    Concealed,
    /// Nothing was ready (priming or starvation); silence was emitted.
    Silence,
}

/// Outcome of a single [`VoiceCommPipeline::playout`] call.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PlayoutStatus {
    /// How this frame was produced.
    pub kind: PlayoutKind,
    /// The jitter buffer statistics snapshot after this call.
    pub jitter: JitterStats,
}

/// A full-duplex voice communication pipeline.
///
/// `capture` drives the uplink for one microphone block; `playout` produces one
/// downlink block. Both operate on a fixed frame size reported by
/// [`VoiceCommPipeline::frame`].
pub trait VoiceCommPipeline {
    /// Returns the fixed frame size in samples shared by both directions.
    fn frame(&self) -> usize;

    /// Processes one microphone block and, when it carries speech, encodes and
    /// sends it over `transport`.
    ///
    /// `mic` is processed in place by the uplink chain; `reference` is the
    /// matching loudspeaker block for echo cancellation (pass zeros when there
    /// is no acoustic echo).
    fn capture<T: VoiceTransport>(
        &mut self,
        mic: &mut [Sample],
        reference: &[Sample],
        transport: &mut T,
    ) -> CaptureStatus;

    /// Produces one downlink playout block into `out`, draining `transport`
    /// into the jitter buffer and decoding or concealing as needed.
    fn playout<T: VoiceTransport>(&mut self, out: &mut [Sample], transport: &mut T)
    -> PlayoutStatus;

    /// Resets both directions to their initial state.
    fn reset(&mut self);
}

/// Configuration for [`DefaultVoiceCommPipeline`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PipelineConfig {
    /// Uplink pre-processing configuration (also fixes the frame size).
    pub uplink: UplinkConfig,
    /// Jitter buffer configuration; its `frame_samples` is overridden to match
    /// the uplink frame.
    pub jitter: JitterConfig,
    /// Packet-loss concealer configuration.
    pub plc: PlcConfig,
    /// When `true`, blocks the uplink gated as non-speech are not transmitted
    /// (discontinuous transmission).
    pub discontinuous_transmission: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            uplink: UplinkConfig::default(),
            jitter: JitterConfig::default(),
            plc: PlcConfig::default(),
            discontinuous_transmission: true,
        }
    }
}

/// A complete voice pipeline built from the crate's own modules.
///
/// The uplink frame size (rounded to a power of two by the noise suppressor)
/// drives the codec frame, the jitter buffer frame, and the concealer frame so
/// every stage stays aligned.
#[derive(Clone, Debug)]
pub struct DefaultVoiceCommPipeline {
    frame: usize,
    discontinuous_transmission: bool,
    uplink: UplinkChain,
    codec: LinearPcmCodec,
    jitter: JitterBuffer,
    plc: PacketLossConcealer,
    // Transmit-side clock and sequencing.
    capture_clock: u64,
    tx_sequence: u32,
    was_transmitting: bool,
    // Receive-side clock for arrival timestamping.
    playout_clock: u64,
    // Reused scratch buffers so steady-state processing does not allocate.
    encode_scratch: Vec<u8>,
    decode_scratch: Vec<Sample>,
}

impl DefaultVoiceCommPipeline {
    /// Builds the pipeline from a configuration.
    #[must_use]
    pub fn new(config: PipelineConfig) -> Self {
        let uplink = UplinkChain::new(config.uplink);
        let frame = uplink.frame();
        let codec = LinearPcmCodec::new(frame);
        let jitter = JitterBuffer::new(JitterConfig {
            frame_samples: frame as u32,
            ..config.jitter
        });
        let plc = PacketLossConcealer::new(frame, config.plc);
        Self {
            frame,
            discontinuous_transmission: config.discontinuous_transmission,
            uplink,
            codec,
            jitter,
            plc,
            capture_clock: 0,
            tx_sequence: 0,
            was_transmitting: false,
            playout_clock: 0,
            encode_scratch: Vec::new(),
            decode_scratch: Vec::new(),
        }
    }

    /// Borrows the uplink chain for telemetry or host inspection.
    #[must_use]
    pub fn uplink(&self) -> &UplinkChain {
        &self.uplink
    }

    /// Returns the current jitter buffer statistics.
    #[must_use]
    pub fn jitter_stats(&self) -> JitterStats {
        self.jitter.stats()
    }

    /// Drains every currently available packet from `transport` into the jitter
    /// buffer, timestamping arrivals with the downlink clock.
    fn drain_transport<T: VoiceTransport>(&mut self, transport: &mut T) {
        while let Some(packet) = transport.poll() {
            self.jitter.insert(packet, self.playout_clock);
        }
    }
}

impl VoiceCommPipeline for DefaultVoiceCommPipeline {
    fn frame(&self) -> usize {
        self.frame
    }

    fn capture<T: VoiceTransport>(
        &mut self,
        mic: &mut [Sample],
        reference: &[Sample],
        transport: &mut T,
    ) -> CaptureStatus {
        let status = self.uplink.process(mic, reference);

        // Decide whether this block should be transmitted. With discontinuous
        // transmission, only non-gated speech blocks go on the wire.
        let is_speech = status.vad.is_speech && !status.gated;
        let transmit = if self.discontinuous_transmission {
            is_speech
        } else {
            true
        };

        let mut sequence = None;
        let mut transmitted = false;
        if transmit && mic.len() == self.frame {
            let talkspurt_start = !self.was_transmitting;
            self.encode_scratch.clear();
            self.codec.encode(mic, &mut self.encode_scratch);
            let seq = self.tx_sequence;
            let packet = VoicePacket::new(
                seq,
                self.capture_clock,
                talkspurt_start,
                self.encode_scratch.clone(),
            );
            if transport.send(&packet).is_ok() {
                sequence = Some(seq);
                transmitted = true;
                self.tx_sequence = self.tx_sequence.wrapping_add(1);
            }
        }

        self.was_transmitting = transmitted;
        self.capture_clock = self.capture_clock.wrapping_add(self.frame as u64);

        CaptureStatus {
            uplink: status,
            transmitted,
            sequence,
        }
    }

    fn playout<T: VoiceTransport>(
        &mut self,
        out: &mut [Sample],
        transport: &mut T,
    ) -> PlayoutStatus {
        self.drain_transport(transport);

        let n = out.len().min(self.frame);
        let kind = match self.jitter.pop() {
            JitterResult::Packet(packet) => {
                self.decode_scratch.clear();
                match self.codec.decode(&packet.payload, &mut self.decode_scratch) {
                    Ok(_) => {
                        for (slot, &sample) in out.iter_mut().zip(self.decode_scratch.iter()) {
                            *slot = sample;
                        }
                        // Zero any tail the payload did not cover.
                        for slot in out.iter_mut().skip(self.decode_scratch.len()) {
                            *slot = 0.0;
                        }
                        // Feed the good frame through the concealer so it keeps
                        // history and cross-fades out of any prior loss.
                        self.plc.good_frame(&mut out[..n]);
                        PlayoutKind::Decoded
                    }
                    Err(_) => {
                        // A malformed payload is treated as a lost frame.
                        for slot in out.iter_mut() {
                            *slot = 0.0;
                        }
                        self.plc.conceal(&mut out[..n]);
                        PlayoutKind::Concealed
                    }
                }
            }
            JitterResult::Loss => {
                for slot in out.iter_mut() {
                    *slot = 0.0;
                }
                self.plc.conceal(&mut out[..n]);
                PlayoutKind::Concealed
            }
            JitterResult::Underrun => {
                for slot in out.iter_mut() {
                    *slot = 0.0;
                }
                PlayoutKind::Silence
            }
        };

        self.playout_clock = self.playout_clock.wrapping_add(self.frame as u64);

        PlayoutStatus {
            kind,
            jitter: self.jitter.stats(),
        }
    }

    fn reset(&mut self) {
        self.uplink.reset();
        self.codec.reset();
        self.jitter.reset();
        self.plc.reset();
        self.capture_clock = 0;
        self.tx_sequence = 0;
        self.was_transmitting = false;
        self.playout_clock = 0;
        self.encode_scratch.clear();
        self.decode_scratch.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::packet::LoopbackTransport;
    use bevy_math::ops;
    use core::f32::consts::PI;

    #[cfg(not(feature = "std"))]
    use alloc::vec;

    fn tone_block(frame: usize, block: usize, freq: Sample, sr: Sample) -> Vec<Sample> {
        (0..frame)
            .map(|i| {
                let n = (block * frame + i) as Sample;
                0.3 * ops::sin(2.0 * PI * freq * n / sr)
            })
            .collect()
    }

    #[test]
    fn captures_and_plays_back_through_loopback() {
        let mut pipeline = DefaultVoiceCommPipeline::new(PipelineConfig::default());
        let frame = pipeline.frame();
        let mut transport = LoopbackTransport::new(256);
        let reference = vec![0.0; frame];
        let sr = 48_000.0;

        let mut decoded_blocks = 0;
        for b in 0..80 {
            let mut mic = tone_block(frame, b, 220.0, sr);
            let _ = pipeline.capture(&mut mic, &reference, &mut transport);
            let mut out = vec![0.0; frame];
            let status = pipeline.playout(&mut out, &mut transport);
            if status.kind == PlayoutKind::Decoded {
                decoded_blocks += 1;
            }
        }
        // After priming, sustained speech should decode real frames.
        assert!(decoded_blocks > 0);
    }

    #[test]
    fn silent_input_is_not_transmitted_with_dtx() {
        let mut pipeline = DefaultVoiceCommPipeline::new(PipelineConfig::default());
        let frame = pipeline.frame();
        let mut transport = LoopbackTransport::new(256);
        let reference = vec![0.0; frame];

        let mut transmitted_any = false;
        for _ in 0..40 {
            let mut mic = vec![0.0; frame];
            let status = pipeline.capture(&mut mic, &reference, &mut transport);
            if status.transmitted {
                transmitted_any = true;
            }
            let mut out = vec![0.0; frame];
            let _ = pipeline.playout(&mut out, &mut transport);
        }
        assert!(!transmitted_any);
    }

    #[test]
    fn playout_without_packets_is_silence() {
        let mut pipeline = DefaultVoiceCommPipeline::new(PipelineConfig::default());
        let frame = pipeline.frame();
        let mut transport = LoopbackTransport::new(16);
        let mut out = vec![0.1; frame];
        let status = pipeline.playout(&mut out, &mut transport);
        assert_eq!(status.kind, PlayoutKind::Silence);
        assert!(out.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn dropped_packets_are_concealed() {
        let mut pipeline = DefaultVoiceCommPipeline::new(PipelineConfig {
            discontinuous_transmission: false,
            ..PipelineConfig::default()
        });
        let frame = pipeline.frame();
        let sr = 48_000.0;
        let mut transport = LoopbackTransport::new(256);
        let reference = vec![0.0; frame];

        let mut concealed = 0;
        for b in 0..120 {
            let mut mic = tone_block(frame, b, 180.0, sr);
            // Drop every 7th block's transmission to force loss.
            if b % 7 != 3 {
                let _ = pipeline.capture(&mut mic, &reference, &mut transport);
            } else {
                // Advance uplink and sequence without sending by polling the
                // uplink directly, then skipping transmission.
                let _ = pipeline.capture(&mut mic, &reference, &mut DiscardTransport);
            }
            let mut out = vec![0.0; frame];
            let status = pipeline.playout(&mut out, &mut transport);
            if status.kind == PlayoutKind::Concealed {
                concealed += 1;
            }
        }
        assert!(concealed > 0);
    }

    #[test]
    fn reset_clears_state() {
        let mut pipeline = DefaultVoiceCommPipeline::new(PipelineConfig::default());
        let frame = pipeline.frame();
        let mut transport = LoopbackTransport::new(64);
        let reference = vec![0.0; frame];
        let mut mic = vec![0.2; frame];
        let _ = pipeline.capture(&mut mic, &reference, &mut transport);
        pipeline.reset();
        assert_eq!(pipeline.jitter_stats(), JitterStats::default());
    }

    /// A transport that drops everything, used to simulate packet loss while
    /// still advancing the pipeline's transmit sequencing.
    struct DiscardTransport;

    impl VoiceTransport for DiscardTransport {
        fn send(&mut self, _packet: &VoicePacket) -> Result<(), crate::transport::TransportError> {
            Ok(())
        }

        fn poll(&mut self) -> Option<VoicePacket> {
            None
        }
    }
}
