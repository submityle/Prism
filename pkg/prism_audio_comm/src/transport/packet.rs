//! Voice packet model and the host transport abstraction.
//!
//! The engine does not own a network stack: real delivery is provided by the
//! host or platform (WebRTC data channels, platform voice services, a game
//! netcode layer). This module defines the on-the-wire unit, [`VoicePacket`],
//! and the [`VoiceTransport`] trait the engine calls to hand encoded frames to
//! that layer and to pull received frames back. A fully in-memory
//! [`LoopbackTransport`] is included so the crate is self-contained and
//! testable without any external dependency.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the transport decoupling of design section 45.3. Encoded frames
//! come from [`crate::transport::codec`]; received frames feed
//! [`crate::transport::jitter_buffer`].

use alloc::collections::VecDeque;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Error returned by a [`VoiceTransport`] implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum TransportError {
    /// The transport's send capacity is full; the caller should retry later.
    Full,
    /// The underlying link is closed and will accept no further packets.
    Closed,
}

/// A single encoded voice frame as handed to or received from the host
/// transport.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VoicePacket {
    /// Monotonic frame sequence number, incremented once per encoded frame and
    /// wrapping on overflow; used by the jitter buffer to reorder and to detect
    /// loss.
    pub sequence: u32,
    /// Capture timestamp in samples at the stream's sample rate; the wall-clock
    /// playout reference, decoupled from any musical clock.
    pub timestamp_samples: u64,
    /// Set on the first frame of a talkspurt after a silence gap, so the jitter
    /// buffer can re-prime its target delay.
    pub talkspurt_start: bool,
    /// The codec payload bytes.
    pub payload: Vec<u8>,
}

impl VoicePacket {
    /// Creates a packet from its fields.
    #[must_use]
    pub fn new(sequence: u32, timestamp_samples: u64, talkspurt_start: bool, payload: Vec<u8>) -> Self {
        Self {
            sequence,
            timestamp_samples,
            talkspurt_start,
            payload,
        }
    }
}

/// The host network transport the engine sends to and receives from.
///
/// Implementations run off the audio thread. The engine never blocks on them:
/// [`VoiceTransport::poll`] is non-blocking and returns [`None`] when no packet
/// is currently available.
pub trait VoiceTransport {
    /// Hands one encoded packet to the host for delivery to the far end.
    fn send(&mut self, packet: &VoicePacket) -> Result<(), TransportError>;

    /// Returns the next received packet, or [`None`] if none is ready.
    fn poll(&mut self) -> Option<VoicePacket>;
}

/// An in-memory transport that delivers sent packets straight back to
/// [`VoiceTransport::poll`].
///
/// It models an ideal zero-loss, in-order link and is used for tests, offline
/// processing, and local echo diagnostics. Capacity bounds the queue so a
/// runaway producer surfaces [`TransportError::Full`] rather than growing
/// without limit.
#[derive(Clone, Debug)]
pub struct LoopbackTransport {
    queue: VecDeque<VoicePacket>,
    capacity: usize,
}

impl LoopbackTransport {
    /// Creates a loopback transport that holds at most `capacity` packets.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
        }
    }

    /// Returns the number of packets currently queued.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` if no packets are queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

impl VoiceTransport for LoopbackTransport {
    fn send(&mut self, packet: &VoicePacket) -> Result<(), TransportError> {
        if self.queue.len() >= self.capacity {
            return Err(TransportError::Full);
        }
        self.queue.push_back(packet.clone());
        Ok(())
    }

    fn poll(&mut self) -> Option<VoicePacket> {
        self.queue.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn loopback_round_trips_in_order() {
        let mut t = LoopbackTransport::new(8);
        for seq in 0..4 {
            let p = VoicePacket::new(seq, seq as u64 * 128, seq == 0, vec![seq as u8; 4]);
            assert_eq!(t.send(&p), Ok(()));
        }
        for seq in 0..4 {
            let p = t.poll().expect("packet available");
            assert_eq!(p.sequence, seq);
        }
        assert!(t.poll().is_none());
    }

    #[test]
    fn loopback_reports_full() {
        let mut t = LoopbackTransport::new(2);
        let p = VoicePacket::new(0, 0, true, vec![0; 2]);
        assert_eq!(t.send(&p), Ok(()));
        assert_eq!(t.send(&p), Ok(()));
        assert_eq!(t.send(&p), Err(TransportError::Full));
    }
}
