//! Reliable ordered delivery over an unreliable channel (design §6 "可靠 UDP").
//!
//! [`crate::transport`] is best-effort and unordered; [`crate::udp::UdpTransport`]
//! loses and reorders. This module adds the missing guarantee *above* the socket
//! and *below* the replication session: a [`ReliableSender`] numbers each message,
//! keeps unacknowledged ones, and retransmits after a timeout that adapts to the
//! measured round-trip time; a [`ReliableReceiver`] reassembles them in order
//! ([`DeliveryWindow`]) and acknowledges what it has. The pair is pure — no
//! sockets, no clocks — so it is driven deterministically by a tick counter and
//! tested against the lossy [`LoopbackLink`](crate::transport::LoopbackLink).
//!
//! This is a fixed-window stop-and-go ARQ (retransmit-on-timeout), not a sliding
//! window with selective repeat: the design's congestion / flow-control work
//! layers on top. It is deliberately simple and complete — a sent message is
//! acknowledged or retransmitted forever, so nothing is silently lost.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::sequence::{Delivery, DeliveryWindow};

/// The sender half of a reliable ordered channel.
#[derive(Clone, Debug)]
pub struct ReliableSender {
    next_seq: u32,
    min_rto: u32,
    rto: u32,
    srtt: Option<u32>,
    queued: VecDeque<(u32, Vec<u8>)>,
    in_flight: VecDeque<Outgoing>,
}

#[derive(Clone, Debug)]
struct Outgoing {
    seq: u32,
    payload: Vec<u8>,
    first_sent: u64,
    last_sent: u64,
    retransmitted: bool,
}

impl ReliableSender {
    /// A sender whose retransmit timeout never drops below `min_rto` ticks.
    #[must_use]
    pub fn new(min_rto: u32) -> Self {
        let min_rto = min_rto.max(1);
        Self {
            next_seq: 0,
            min_rto,
            rto: min_rto,
            srtt: None,
            queued: VecDeque::new(),
            in_flight: VecDeque::new(),
        }
    }

    /// Queue `payload` for reliable ordered delivery; returns its sequence number.
    pub fn send(&mut self, payload: Vec<u8>) -> u32 {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        self.queued.push_back((seq, payload));
        seq
    }

    /// The number of messages not yet acknowledged.
    #[must_use]
    pub fn unacked(&self) -> usize {
        self.queued.len() + self.in_flight.len()
    }

    /// The current retransmit timeout, in ticks.
    #[must_use]
    pub fn rto(&self) -> u32 {
        self.rto
    }

    /// The smoothed round-trip estimate, if any ACK has been sampled.
    #[must_use]
    pub fn srtt(&self) -> Option<u32> {
        self.srtt
    }

    /// Advance to `now`: append every packet to transmit to `out` — new messages
    /// once, then unacknowledged ones whose timeout has elapsed.
    pub fn tick(&mut self, now: u64, out: &mut Vec<(u32, Vec<u8>)>) {
        while let Some((seq, payload)) = self.queued.pop_front() {
            out.push((seq, payload.clone()));
            self.in_flight.push_back(Outgoing {
                seq,
                payload,
                first_sent: now,
                last_sent: now,
                retransmitted: false,
            });
        }
        let rto = u64::from(self.rto);
        let mut resend = Vec::new();
        for message in &mut self.in_flight {
            if now.saturating_sub(message.last_sent) >= rto {
                message.last_sent = now;
                message.retransmitted = true;
                resend.push((message.seq, message.payload.clone()));
            }
        }
        out.extend(resend);
    }

    /// Acknowledge `seq`, returning whether it was outstanding.
    ///
    /// The round-trip time feeds a smoothed estimate (`srtt`) and the timeout
    /// (`rto = 2 * srtt`, never below `min_rto`). A **retransmitted** message is
    /// never sampled (Karn's rule): its ACK could belong to either transmission.
    pub fn ack(&mut self, seq: u32, now: u64) -> bool {
        let Some(index) = self.in_flight.iter().position(|message| message.seq == seq) else {
            return false;
        };
        let message = self.in_flight.remove(index).expect("index from position");
        if !message.retransmitted {
            let sample = now.saturating_sub(message.first_sent).min(u64::from(u32::MAX)) as u32;
            let smoothed = match self.srtt {
                // `srtt = (7 * srtt + sample) / 8`.
                Some(previous) => (previous.saturating_mul(7).saturating_add(sample)) / 8,
                None => sample,
            };
            self.srtt = Some(smoothed);
            self.rto = smoothed.saturating_mul(2).max(self.min_rto);
        }
        true
    }
}

/// The receiver half of a reliable ordered channel.
#[derive(Clone, Debug)]
pub struct ReliableReceiver {
    window: DeliveryWindow<Vec<u8>>,
}

impl ReliableReceiver {
    /// A receiver buffering up to `capacity - 1` out-of-order messages.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            window: DeliveryWindow::new(capacity),
        }
    }

    /// The next sequence number that will be delivered.
    #[must_use]
    pub fn next(&self) -> u64 {
        self.window.next()
    }

    /// Receive one packet: the payloads newly deliverable (in order) and the
    /// sequence to acknowledge.
    ///
    /// A duplicate or already-delivered sequence is re-acknowledged (the original
    /// ACK was probably lost) so the sender stops retransmitting; a sequence past
    /// the window is not acknowledged, forcing an explicit resync rather than an
    /// unbounded buffer.
    pub fn receive(&mut self, seq: u32, payload: Vec<u8>) -> (Vec<Vec<u8>>, Option<u32>) {
        match self.window.insert(u64::from(seq), payload) {
            Delivery::Advanced(run) => {
                let delivered = run.into_iter().map(|(_, payload)| payload).collect();
                (delivered, Some(seq))
            }
            Delivery::Buffered | Delivery::Duplicate | Delivery::Stale => (Vec::new(), Some(seq)),
            Delivery::OutOfWindow => (Vec::new(), None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::LoopbackLink;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Packet codec for the integration test: `[kind: u8][seq: u32 LE][payload]`.
    fn data_packet(seq: u32, payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8];
        bytes.extend_from_slice(&seq.to_le_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    fn ack_packet(seq: u32) -> Vec<u8> {
        let mut bytes = vec![1u8];
        bytes.extend_from_slice(&seq.to_le_bytes());
        bytes
    }

    fn decode(bytes: &[u8]) -> (u8, u32, &[u8]) {
        let seq = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
        (bytes[0], seq, &bytes[5..])
    }

    #[test]
    fn in_order_messages_are_acknowledged_and_measured() {
        let mut sender = ReliableSender::new(2);
        let mut out = Vec::new();
        sender.send(b"a".to_vec());
        sender.send(b"b".to_vec());
        sender.tick(0, &mut out);
        assert_eq!(out.len(), 2, "both go out on the first tick");
        assert_eq!(sender.unacked(), 2);

        let mut receiver = ReliableReceiver::new(8);
        let (delivered, ack) = receiver.receive(out[0].0, out[0].1.clone());
        assert_eq!(delivered, vec![b"a".to_vec()]);
        assert_eq!(ack, Some(0));
        assert!(sender.ack(0, 4));
        assert_eq!(sender.unacked(), 1);
        assert_eq!(sender.srtt(), Some(4));
        assert_eq!(sender.rto(), 8, "rto = 2 * srtt");
        // A duplicate ACK is not outstanding and samples nothing.
        assert!(!sender.ack(0, 9));
        assert_eq!(sender.srtt(), Some(4));
    }

    #[test]
    fn an_unacknowledged_message_is_retransmitted_after_the_timeout() {
        let mut sender = ReliableSender::new(3);
        let mut out = Vec::new();
        sender.send(b"x".to_vec());
        sender.tick(0, &mut out);
        out.clear();
        sender.tick(2, &mut out);
        assert!(out.is_empty(), "not yet due");
        sender.tick(3, &mut out);
        assert_eq!(out.len(), 1, "retransmitted at the timeout");
        assert_eq!(sender.unacked(), 1, "still unacknowledged");
    }

    #[test]
    fn a_duplicate_is_re_acknowledged_but_not_redelivered() {
        let mut receiver = ReliableReceiver::new(8);
        let (first, _) = receiver.receive(0, b"a".to_vec());
        assert_eq!(first.len(), 1);
        let (again, ack) = receiver.receive(0, b"a".to_vec());
        assert!(again.is_empty(), "no re-delivery");
        assert_eq!(ack, Some(0), "re-acked so the sender stops retransmitting");
    }

    /// A sequence past the window is not acknowledged, so the sender cannot
    /// mistake a gap for delivery.
    #[test]
    fn a_far_ahead_packet_is_not_acknowledged() {
        let mut receiver = ReliableReceiver::new(2);
        let (delivered, ack) = receiver.receive(5, b"far".to_vec());
        assert!(delivered.is_empty());
        assert_eq!(ack, None);
    }

    /// End-to-end over a lossy, jittery link: every message is delivered exactly
    /// once and in order, and everything is acknowledged.
    #[test]
    fn the_arq_recovers_every_message_over_a_lossy_link() {
        const N: u32 = 10;
        let mut link = LoopbackLink::new(1, 2).with_drop_every(4);
        let mut sender = ReliableSender::new(3);
        let mut receiver = ReliableReceiver::new(32);

        for i in 0..N {
            sender.send(i.to_le_bytes().to_vec());
        }

        let mut delivered: Vec<u32> = Vec::new();
        let mut now = 0u64;
        // Keep pumping until everything is delivered *and* acknowledged (the last
        // ACKs may still be in flight or need a retransmit).
        while (delivered.len() < N as usize || sender.unacked() > 0) && now < 1_000 {
            now += 1;
            let mut out = Vec::new();
            sender.tick(now, &mut out);
            for (seq, payload) in out {
                link.send(true, data_packet(seq, &payload));
            }
            for packet in link.poll(false) {
                let (kind, seq, payload) = decode(&packet);
                if kind != 0 {
                    continue;
                }
                let (frames, ack) = receiver.receive(seq, payload.to_vec());
                for frame in frames {
                    delivered.push(u32::from_le_bytes(frame.try_into().expect("four bytes")));
                }
                if let Some(ack) = ack {
                    link.send(false, ack_packet(ack));
                }
            }
            for packet in link.poll(true) {
                let (kind, seq, _) = decode(&packet);
                if kind == 1 {
                    sender.ack(seq, now);
                }
            }
        }

        assert_eq!(
            delivered,
            (0..N).collect::<Vec<_>>(),
            "in order, exactly once"
        );
        assert_eq!(sender.unacked(), 0, "all acknowledged");
        assert!(link.dropped() > 0, "the link really dropped packets");
    }
}
