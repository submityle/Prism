//! Adaptive jitter buffer for the downlink.
//!
//! Network packets arrive reordered and with varying delay. The jitter buffer
//! reorders them by sequence number, absorbs delay variation by holding a small
//! adaptive backlog, and presents frames to the decoder at a steady playout
//! rate. The target backlog tracks an RFC 3550-style inter-arrival jitter
//! estimate: more network jitter grows the buffer (more latency, fewer
//! glitches), less jitter shrinks it (lower latency). Missing frames surface as
//! an explicit loss so the packet-loss concealer can fill them, and overflow or
//! late arrivals are counted rather than silently mishandled. This is the
//! wall-clock low-latency path of design section 45.3, decoupled from any
//! musical clock.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the adaptive jitter buffer of design section 45.3. Consumes
//! [`crate::transport::packet::VoicePacket`]s from the host transport and
//! feeds [`crate::transport::codec`] / [`crate::transport::plc`] in
//! [`crate::pipeline`].

use alloc::collections::BTreeMap;
use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::transport::packet::VoicePacket;

/// Outcome of a single [`JitterBuffer::pop`] call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JitterResult {
    /// The in-order packet for this playout slot is available.
    Packet(VoicePacket),
    /// The in-order packet is permanently missing; conceal this frame.
    Loss,
    /// Nothing can be played yet (still priming or starved); emit silence or
    /// extend concealment and try again next slot.
    Underrun,
}

/// Running statistics for telemetry (design section 45.5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct JitterStats {
    /// Total packets accepted into the buffer.
    pub received: u64,
    /// Packets that arrived out of order (behind an already-buffered higher
    /// sequence).
    pub reordered: u64,
    /// Packets discarded because their slot had already played out.
    pub late_discarded: u64,
    /// Packets dropped because the buffer was at capacity.
    pub overflow_dropped: u64,
    /// Playout slots reported as [`JitterResult::Loss`].
    pub concealed_losses: u64,
    /// Playout slots reported as [`JitterResult::Underrun`].
    pub underruns: u64,
}

/// Configuration for [`JitterBuffer`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct JitterConfig {
    /// Samples per frame, used to convert the sample-domain jitter estimate
    /// into a backlog depth in frames.
    pub frame_samples: u32,
    /// Minimum backlog depth in frames.
    pub min_frames: usize,
    /// Maximum backlog depth in frames (also the hard storage capacity).
    pub max_frames: usize,
    /// Safety multiplier applied to the jitter estimate when sizing the
    /// backlog.
    pub jitter_scale: Sample,
}

impl Default for JitterConfig {
    fn default() -> Self {
        Self {
            frame_samples: 256,
            min_frames: 2,
            max_frames: 16,
            jitter_scale: 3.0,
        }
    }
}

/// An adaptive, reordering jitter buffer.
#[derive(Clone, Debug)]
pub struct JitterBuffer {
    config: JitterConfig,
    buffer: BTreeMap<u32, VoicePacket>,
    /// Sequence number of the next packet to play.
    next_seq: Option<u32>,
    /// Whether the buffer is still accumulating its initial backlog.
    priming: bool,
    /// Current adaptive target backlog in frames.
    target_frames: usize,
    /// RFC 3550-style jitter estimate in samples.
    jitter: Sample,
    /// Transit time (arrival minus timestamp) of the previous packet.
    last_transit: Option<i64>,
    stats: JitterStats,
}

impl JitterBuffer {
    /// Creates a jitter buffer from a configuration.
    #[must_use]
    pub fn new(config: JitterConfig) -> Self {
        let min_frames = config.min_frames.max(1);
        let max_frames = config.max_frames.max(min_frames);
        Self {
            config: JitterConfig {
                min_frames,
                max_frames,
                ..config
            },
            buffer: BTreeMap::new(),
            next_seq: None,
            priming: true,
            target_frames: min_frames,
            jitter: 0.0,
            last_transit: None,
            stats: JitterStats::default(),
        }
    }

    /// Returns the current statistics snapshot.
    #[must_use]
    pub fn stats(&self) -> JitterStats {
        self.stats
    }

    /// Returns the current adaptive target backlog in frames.
    #[must_use]
    pub fn target_frames(&self) -> usize {
        self.target_frames
    }

    /// Returns the current jitter estimate in samples.
    #[must_use]
    pub fn jitter_samples(&self) -> Sample {
        self.jitter
    }

    /// Returns the number of packets currently buffered.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.buffer.len()
    }

    /// Clears all state and statistics.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.next_seq = None;
        self.priming = true;
        self.target_frames = self.config.min_frames;
        self.jitter = 0.0;
        self.last_transit = None;
        self.stats = JitterStats::default();
    }

    /// Updates the jitter estimate from a packet's transit time.
    ///
    /// `arrival_samples` is the host wall-clock arrival time in samples; the
    /// transit is its difference from the packet timestamp, and the smoothed
    /// absolute deviation of successive transits is the jitter.
    fn update_jitter(&mut self, packet: &VoicePacket, arrival_samples: u64) {
        let transit = arrival_samples as i64 - packet.timestamp_samples as i64;
        if let Some(prev) = self.last_transit {
            let d = (transit - prev).unsigned_abs() as Sample;
            // Standard RFC 3550 gain of 1/16.
            self.jitter += (d - self.jitter) * (1.0 / 16.0);
        }
        self.last_transit = Some(transit);

        // Convert jitter (samples) to a backlog depth in frames.
        let frame = self.config.frame_samples.max(1) as Sample;
        let needed = ops::ceil(self.config.jitter_scale * self.jitter / frame) as i64 + 1;
        let clamped = needed.clamp(self.config.min_frames as i64, self.config.max_frames as i64);
        self.target_frames = clamped as usize;
    }

    /// Inserts a received packet, reordering by sequence and updating jitter.
    pub fn insert(&mut self, packet: VoicePacket, arrival_samples: u64) {
        self.update_jitter(&packet, arrival_samples);

        if self.next_seq.is_none() {
            self.next_seq = Some(packet.sequence);
        }
        let next = self.next_seq.unwrap_or(packet.sequence);

        // A packet whose slot already played out is too late to use.
        if seq_less_than(packet.sequence, next) {
            self.stats.late_discarded += 1;
            return;
        }

        // Reorder detection: a higher sequence is already buffered.
        if let Some((&max_seq, _)) = self.buffer.iter().next_back()
            && seq_less_than(packet.sequence, max_seq)
        {
            self.stats.reordered += 1;
        }

        self.buffer.insert(packet.sequence, packet);
        self.stats.received += 1;

        // Enforce capacity by dropping the oldest buffered packet.
        while self.buffer.len() > self.config.max_frames {
            if let Some((&oldest, _)) = self.buffer.iter().next() {
                self.buffer.remove(&oldest);
                self.stats.overflow_dropped += 1;
                // If we dropped the slot we were waiting for, skip past it.
                if Some(oldest) == self.next_seq {
                    self.next_seq = Some(oldest.wrapping_add(1));
                }
            } else {
                break;
            }
        }
    }

    /// Removes and returns the next playout result, advancing the playout
    /// pointer.
    pub fn pop(&mut self) -> JitterResult {
        // Prime until the backlog reaches the adaptive target.
        if self.priming {
            if self.buffer.len() < self.target_frames {
                self.stats.underruns += 1;
                return JitterResult::Underrun;
            }
            self.priming = false;
        }

        let Some(next) = self.next_seq else {
            self.stats.underruns += 1;
            return JitterResult::Underrun;
        };

        if let Some(packet) = self.buffer.remove(&next) {
            self.next_seq = Some(next.wrapping_add(1));
            return JitterResult::Packet(packet);
        }

        // The slot is empty. If later packets exist, the frame is lost; conceal
        // it and move on. Otherwise we are starved: re-prime and underrun.
        if self.buffer.is_empty() {
            self.priming = true;
            self.stats.underruns += 1;
            JitterResult::Underrun
        } else {
            self.next_seq = Some(next.wrapping_add(1));
            self.stats.concealed_losses += 1;
            JitterResult::Loss
        }
    }
}

/// Returns `true` when sequence `a` is strictly before `b` under 32-bit
/// wrap-around (RFC 1982 serial-number arithmetic).
#[inline]
fn seq_less_than(a: u32, b: u32) -> bool {
    a != b && b.wrapping_sub(a) < 0x8000_0000
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn pkt(seq: u32) -> VoicePacket {
        VoicePacket::new(seq, seq as u64 * 256, seq == 0, vec![seq as u8; 2])
    }

    #[test]
    fn in_order_playout() {
        let mut jb = JitterBuffer::new(JitterConfig {
            min_frames: 2,
            ..JitterConfig::default()
        });
        for seq in 0..6 {
            jb.insert(pkt(seq), seq as u64 * 256);
        }
        let mut got = Vec::new();
        for _ in 0..6 {
            if let JitterResult::Packet(p) = jb.pop() {
                got.push(p.sequence);
            }
        }
        assert_eq!(got, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn reorders_out_of_order_arrivals() {
        let mut jb = JitterBuffer::new(JitterConfig {
            min_frames: 3,
            ..JitterConfig::default()
        });
        // Arrive 0, 2, 1, 3 (2 and 1 swapped).
        jb.insert(pkt(0), 0);
        jb.insert(pkt(2), 512);
        jb.insert(pkt(1), 600);
        jb.insert(pkt(3), 768);
        let mut got = Vec::new();
        for _ in 0..4 {
            if let JitterResult::Packet(p) = jb.pop() {
                got.push(p.sequence);
            }
        }
        assert_eq!(got, vec![0, 1, 2, 3]);
        assert!(jb.stats().reordered >= 1);
    }

    #[test]
    fn missing_packet_reports_loss() {
        let mut jb = JitterBuffer::new(JitterConfig {
            min_frames: 2,
            ..JitterConfig::default()
        });
        // 1 is missing.
        jb.insert(pkt(0), 0);
        jb.insert(pkt(2), 512);
        jb.insert(pkt(3), 768);
        let mut results = Vec::new();
        for _ in 0..4 {
            results.push(jb.pop());
        }
        assert_eq!(results[0], JitterResult::Packet(pkt(0)));
        assert_eq!(results[1], JitterResult::Loss);
        assert_eq!(results[2], JitterResult::Packet(pkt(2)));
        assert_eq!(results[3], JitterResult::Packet(pkt(3)));
        assert_eq!(jb.stats().concealed_losses, 1);
    }

    #[test]
    fn capacity_overflow_is_counted() {
        let mut jb = JitterBuffer::new(JitterConfig {
            min_frames: 2,
            max_frames: 4,
            ..JitterConfig::default()
        });
        for seq in 0..8 {
            jb.insert(pkt(seq), seq as u64 * 256);
        }
        assert!(jb.depth() <= 4);
        assert!(jb.stats().overflow_dropped >= 1);
    }

    #[test]
    fn target_grows_with_jitter() {
        let mut jb = JitterBuffer::new(JitterConfig::default());
        let base = jb.target_frames();
        // Feed packets whose arrival time deviates increasingly from timestamp.
        for seq in 0..20u32 {
            let arrival = seq as u64 * 256 + (seq as u64 % 2) * 4096;
            jb.insert(pkt(seq), arrival);
        }
        assert!(jb.target_frames() >= base);
        assert!(jb.jitter_samples() > 0.0);
    }

    #[test]
    fn late_packet_discarded() {
        let mut jb = JitterBuffer::new(JitterConfig {
            min_frames: 1,
            ..JitterConfig::default()
        });
        jb.insert(pkt(0), 0);
        jb.insert(pkt(1), 256);
        let _ = jb.pop();
        let _ = jb.pop();
        // Sequence 0 already played; re-inserting it is "late".
        jb.insert(pkt(0), 0);
        assert!(jb.stats().late_discarded >= 1);
    }
}
